"""attnres_impl="fused" (moe_llm/layers/attnres_fused.py) against the running op (layers/attnres.py), an fp64
reference and itself.

CPU: the Triton kernels run in Triton's interpreter (TRITON_INTERPRET=1 is set below when there is no GPU; it must be
set before triton is imported), the same machinery also with the PyTorch backend. Op level: forward and every gradient
vs the current op and vs fp64 (the fused path must be no less accurate); whole tiny model: loss and all gradients vs
the compiled path, bit-identical under recompute (whole layers, FFN part, activation offloading) and reruns; training
steps, eval, state_dict compatibility, train.py --override with resume.
GPU (skipped without CUDA): main-size block accuracy vs torch.compile and fp64, determinism and no host syncs, a small
model under bf16 autocast, and the timing of the AttnRes part of a main_2b micro-batch (pytest -s to see it)."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from torch.utils.checkpoint import checkpoint  # noqa: E402

from moe_llm.config import ModelConfig, tiny  # noqa: E402
from moe_llm.layers import attnres_fused as af  # noqa: E402
from moe_llm.layers.attnres import DepthAttn, _depth_attn  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402
from moe_llm.optim import OptimizerBundle, wsd_lr  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
HAVE_TRITON = torch.cuda.is_available() or af._INTERPRET
BACKENDS = ["torch"] + (["triton"] if HAVE_TRITON else [])
# whole-model tests run on CPU: on a GPU the tiny model has atomics elsewhere (bmm MoE index_add_, flash backward),
# so bit-identity is only defined on CPU; the GPU checks of the model are the test_gpu_* tests
MCPU = "cpu"
CPU_BACKENDS = ["torch"] + (["triton"] if af._INTERPRET else [])
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_backend():
    merge = af._MERGE
    yield
    af.set_backend("auto")
    af.set_merge_reduce(merge)


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


# ----------------------------------------------------------------------------- helpers: one block of ops
def ref_op64(q, srcs, eps):
    """The op in float64 (reference)."""
    V = torch.stack([s.double() for s in srcs])
    r = torch.rsqrt(V.pow(2).mean(-1) + eps)
    a = torch.softmax((V * q.double()).sum(-1) * r, dim=0)
    return (a.unsqueeze(-1) * V).sum(0)


def make_block(n_c, slots=None, B=2, T=13, D=48, dtype=torch.float32, seed=0, device=DEV, w_std=0.5):
    """Completed sources, one DepthAttn per slot (slot 0 has no partial, like a block's first res_mix; block 0 has
    no such op), partials and upstream gradients."""
    slots = slots or tuple(range(0 if n_c > 1 else 1, 7))
    g = torch.Generator().manual_seed(seed)
    srcs = [(torch.randn(B, T, D, generator=g) * (0.5 + j)).to(device=device, dtype=dtype) for j in range(n_c)]
    mods = {}
    for s in slots:
        m = DepthAttn(D).to(device)
        with torch.no_grad():
            m.w.copy_(torch.randn(D, generator=g) * w_std)
            m.g.copy_(0.5 + torch.rand(D, generator=g))
        mods[s] = m
    parts = {s: (torch.randn(B, T, D, generator=g) * 2).to(device=device, dtype=dtype) for s in slots if s != 0}
    gouts = {s: torch.randn(B, T, D, generator=g).to(device=device, dtype=dtype) for s in slots}
    return srcs, mods, parts, gouts


def run_current(srcs, mods, parts, gouts, compiled=False):
    """The running code: one DepthAttn op per slot over the sources, autograd sums the source gradients."""
    from moe_llm.layers import attnres
    attnres.set_attnres_compile(compiled)
    try:
        S = [s.detach().clone().requires_grad_() for s in srcs]
        P = {k: p.detach().clone().requires_grad_() for k, p in parts.items()}
        M = {k: copy.deepcopy(m) for k, m in mods.items()}
        outs = {k: M[k](S + ([P[k]] if k in P else [])) for k in M}
        torch.autograd.backward([outs[k] for k in M], [gouts[k] for k in M])
    finally:
        attnres.set_attnres_compile(True)
    return {"out": {k: o.detach() for k, o in outs.items()}, "src": [s.grad for s in S],
            "part": {k: p.grad for k, p in P.items()}, "w": {k: m.w.grad for k, m in M.items()},
            "g": {k: m.g.grad for k, m in M.items()}}


def run_fused(srcs, mods, parts, gouts, n_slots=7, ckpt=False):
    """The fused path for the last block of len(srcs) blocks (earlier stack slots filled directly): every source
    gradient comes out of the block's shared backward (the earlier ones are read from the accumulator)."""
    S = [s.detach().clone().requires_grad_() for s in srcs]
    P = {k: p.detach().clone().requires_grad_() for k, p in parts.items()}
    M = {k: copy.deepcopy(m) for k, m in mods.items()}
    k_last = len(S) - 1
    st = af.FusedAttnRes(S[0], n_blocks=len(S), n_slots=n_slots)
    with torch.no_grad():
        for j in range(k_last):
            st.stack[j].copy_(S[j].reshape(st.N, st.D))
    blk, tok, qs = st.open_block(k_last, S[k_last], M)

    def op(k, p, q):
        return af.depth_attn(blk, tok, p, q, k, M[k].eps)

    outs = {}
    for k in M:
        if ckpt:
            outs[k] = checkpoint(op, k, P.get(k), qs[k], use_reentrant=False)
        else:
            outs[k] = op(k, P.get(k), qs[k])
    torch.autograd.backward([outs[k] for k in M], [gouts[k] for k in M])
    src_grads = [st.acc[j].view(S[j].shape).clone() for j in range(k_last)] + [S[k_last].grad] if k_last else \
        [S[0].grad]
    return {"out": {k: o.detach() for k, o in outs.items()}, "src": src_grads,
            "part": {k: p.grad for k, p in P.items()}, "w": {k: m.w.grad for k, m in M.items()},
            "g": {k: m.g.grad for k, m in M.items()}}


def run_ref64(srcs, mods, parts, gouts):
    S = [s.detach().double().requires_grad_() for s in srcs]
    P = {k: p.detach().double().requires_grad_() for k, p in parts.items()}
    W = {k: m.w.detach().double().requires_grad_() for k, m in mods.items()}
    G = {k: m.g.detach().double().requires_grad_() for k, m in mods.items()}
    outs = {k: ref_op64(G[k] * W[k], S + ([P[k]] if k in P else []), mods[k].eps) for k in mods}
    torch.autograd.backward([outs[k] for k in mods], [gouts[k].double() for k in mods])
    return {"out": outs, "src": [s.grad for s in S], "part": {k: p.grad for k, p in P.items()},
            "w": {k: w.grad for k, w in W.items()}, "g": {k: g.grad for k, g in G.items()}}


def errors(res, ref):
    """Worst rel. RMS error per quantity over slots / sources."""
    out = {}
    for key in ("out", "part", "w", "g"):
        out[key] = max(rel_rms(res[key][k], ref[key][k].detach()) for k in res[key]) if res[key] else 0.0
    out["src"] = max(rel_rms(a, b) for a, b in zip(res["src"], ref["src"]))
    return out


# ----------------------------------------------------------------------------- kernels vs the PyTorch backend
@pytest.mark.skipif(not HAVE_TRITON, reason="needs CUDA or the Triton interpreter")
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_c,has_p", [(1, True), (3, False), (3, True), (5, True)])
def test_kernels_match_torch_backend(n_c, has_p, dtype, merge):
    """Each of the three kernels against the plain-PyTorch version of it, at sizes that leave partial tiles
    (26 tokens, d = 48: not a power of two); with the per-row sums as separate reductions and merged into one (the
    GPU default)."""
    af.set_merge_reduce(merge)
    g = torch.Generator().manual_seed(n_c)
    N, D, eps = 26, 48, 1e-5
    stack = (torch.randn(n_c + 1, N, D, generator=g) * 2).to(DEV, dtype)
    p = (torch.randn(N, D, generator=g) * 2).to(DEV, dtype) if has_p else None
    q = (torch.randn(D, generator=g) * 0.7).to(DEV)
    dout = torch.randn(N, D, generator=g).to(DEV, dtype)
    fp32 = dtype == torch.float32
    res = {}
    for be in ("torch", "triton"):
        af.set_backend(be)
        out = af._fwd(stack, n_c, p, q, eps)
        coef = torch.full((3, 3, n_c, N), float("nan"), device=DEV)
        dp, dq = af._op_bwd(stack, n_c, p, dout, q, eps, coef[:, 1], coef.stride(0))
        coef[:, 0] = coef[:, 1] * 0.5     # a second op in slot 0 with other coefficients
        gs = {0: (dout * 0.3).to(dtype), 1: dout}
        qbuf = torch.stack([q * 2, q] + [torch.zeros_like(q)] * 6)
        acc = torch.randn(n_c + 1, N, D, generator=torch.Generator().manual_seed(9)).to(DEV, dtype)
        acc_first = acc.clone()
        dqb = af._blk_bwd(stack, n_c, coef, qbuf, gs, acc, first=False)
        dqb_first = af._blk_bwd(stack, n_c, coef, qbuf, gs, acc_first, first=True)
        res[be] = [out, coef[:, :2], acc[:n_c], acc_first[:n_c], dqb[:2]] + ([dp, dq] if has_p else [])
        assert dqb[2:].abs().sum() == 0
        assert torch.isfinite(coef[:, :2]).all()
    names = ["out", "coef", "acc", "acc_first", "dq_block", "dp", "dq_op"]
    for name, a, b in zip(names, res["triton"], res["torch"]):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        tol = 1e-5 if fp32 or a.dtype == torch.float32 else 4e-3
        assert rel_rms(a, b) < tol, (name, rel_rms(a, b))


# ----------------------------------------------------------------------------- op level: one block
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_c", [1, 4])
def test_block_fp32_matches_current_op(backend, n_c):
    af.set_backend(backend)
    srcs, mods, parts, gouts = make_block(n_c)
    cur = run_current(srcs, mods, parts, gouts)
    fus = run_fused(srcs, mods, parts, gouts)
    err = errors(fus, cur)
    assert all(v < 1e-5 for v in err.values()), err


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_c", [1, 4, 8])
def test_block_bf16_no_less_accurate_than_current_op(backend, n_c):
    """bf16 sources and partials (fp32 parameters), errors against fp64 on the same values: output and partial
    gradients within 5% of the current op's error (both round once), the completed sources' gradients more
    accurate (summed in fp32, one rounding; the current op rounds every op's gradient and adds in bf16), dw and dg
    at fp32 level."""
    af.set_backend(backend)
    T = 64 if backend == "torch" or DEV == "cuda" else 32  # the interpreter is slow
    srcs, mods, parts, gouts = make_block(n_c, dtype=torch.bfloat16, T=T, D=64, seed=n_c)
    ref = run_ref64(srcs, mods, parts, gouts)
    e_cur = errors(run_current(srcs, mods, parts, gouts), ref)
    e_fus = errors(run_fused(srcs, mods, parts, gouts), ref)
    print(f"\nn_c={n_c} {backend}: current {e_cur}\n          fused   {e_fus}")
    assert e_fus["out"] <= 1.05 * e_cur["out"] + 1e-6
    assert e_fus["part"] <= 1.05 * e_cur["part"] + 1e-6
    assert e_fus["src"] <= e_cur["src"] + 1e-6
    assert e_fus["w"] < 1e-5 and e_fus["g"] < 1e-5 and e_cur["w"] < 1e-5


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_recompute_and_rerun_bit_identical(backend):
    """torch.utils.checkpoint around every op and a plain rerun give exactly the same values and gradients."""
    af.set_backend(backend)
    srcs, mods, parts, gouts = make_block(3, dtype=torch.bfloat16)
    a = run_fused(srcs, mods, parts, gouts)
    for b in (run_fused(srcs, mods, parts, gouts), run_fused(srcs, mods, parts, gouts, ckpt=True)):
        for key in a:
            if isinstance(a[key], dict):
                assert all(torch.equal(a[key][k], b[key][k]) for k in a[key]), key
            else:
                assert all(torch.equal(x, y) for x, y in zip(a[key], b[key])), key


@pytest.mark.parametrize("backend", BACKENDS)
def test_chain_of_blocks_matches_current(backend):
    """Several blocks chained like the model (each block output = a function of the previous block's op outputs):
    the gradient of every block output is accumulated across blocks and handed back at its own block."""
    af.set_backend(backend)
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_attnres_fused as bench
    torch.manual_seed(0)
    mods, emb, gout = bench.make_workload(n_layers=7, lpb=2, B=2, T=9, d=24, device=DEV, dtype=torch.float32,
                                          w_std=0.5)
    cur_out, cur_g = bench.fwd_bwd("eager", mods, emb, gout, n_layers=7, lpb=2)
    fus_out, fus_g = bench.fwd_bwd("fused", mods, emb, gout, n_layers=7, lpb=2)
    assert rel_rms(fus_out, cur_out) < 1e-5
    for i, (a, b) in enumerate(zip(fus_g, cur_g)):
        assert a is not None and rel_rms(a, b) < 1e-4, (i, rel_rms(a, b))


# ----------------------------------------------------------------------------- whole model
def _batch(cfg, seed=3, T=33):
    x = torch.randint(0, cfg.vocab_size, (2, T), generator=torch.Generator().manual_seed(seed))
    return x[:, :-1], x[:, 1:]


def _model_run(cfg, base, impl, frac=0.0, ffn=0.0, off=0.0, autocast=False, x=None, y=None):
    m = MoELLM(cfg.replace(attnres_impl=impl))
    m.load_state_dict(base.state_dict())
    m.set_checkpointing(frac, ffn_frac=ffn, offload_frac=off)
    if m.offloader is not None:
        m.offloader.min_bytes = 0  # tiny model: offload everything
    with torch.autocast(MCPU, dtype=torch.bfloat16, enabled=autocast):
        loss, _ = m(x, y)
    loss.backward()
    if m.offloader is not None:
        assert m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0, m.offloader.stats
    return loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_model_matches_compiled_and_recompute_is_exact(backend):
    """Tiny model with Block AttnRes (fp32): fused vs the current path within fp32 rounding, every parameter gets a
    gradient; whole-layer recompute (some / all layers), FFN-only recompute and activation offloading are
    bit-identical to no recompute, and so is a rerun."""
    af.set_backend(backend)
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    x, y = _batch(cfg)
    l_c, g_c = _model_run(cfg, base, "compiled", x=x, y=y)
    l_f, g_f = _model_run(cfg, base, "fused", x=x, y=y)
    assert abs(l_f.item() - l_c.item()) < 1e-6 * abs(l_c.item())
    assert g_f.keys() == g_c.keys() == {n for n, _ in base.named_parameters()}
    for k in g_c:
        assert rel_rms(g_f[k], g_c[k]) < 1e-4, (k, rel_rms(g_f[k], g_c[k]))
    variants = [dict(), dict(frac=0.5), dict(frac=1.0), dict(frac=1.0, ffn=1.0), dict(frac=0.5, ffn=0.5),
                dict(frac=1.0, off=0.5), dict(frac=1.0, off=1.0)]
    if backend == "triton":  # the interpreter is slow: a rerun, mixed recompute, offloading
        variants = [dict(), dict(frac=0.5, ffn=0.5), dict(frac=1.0, off=0.5)]
    for kw in variants:
        l2, g2 = _model_run(cfg, base, "fused", x=x, y=y, **kw)
        assert torch.equal(l2, l_f), kw
        assert all(torch.equal(g2[k], g_f[k]) for k in g_f), kw


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_model_under_bf16_autocast(backend):
    """CPU bf16 autocast (fp32 residual stream, bf16 layer outputs: block outputs of mixed dtypes): close to the
    current path, and recompute stays bit-identical."""
    af.set_backend(backend)
    torch.manual_seed(1)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    x, y = _batch(cfg, seed=5)
    l_c, g_c = _model_run(cfg, base, "compiled", autocast=True, x=x, y=y)
    l_f, g_f = _model_run(cfg, base, "fused", autocast=True, x=x, y=y)
    l_r, g_r = _model_run(cfg, base, "fused", frac=0.5, ffn=0.5, autocast=True, x=x, y=y)
    assert abs(l_f.item() - l_c.item()) < 2e-3 * abs(l_c.item())
    for k in g_c:
        assert g_f[k].dtype == g_c[k].dtype and rel_rms(g_f[k], g_c[k]) < 5e-2, (k, rel_rms(g_f[k], g_c[k]))
    assert torch.equal(l_r, l_f) and all(torch.equal(g_r[k], g_f[k]) for k in g_f)


def test_model_training_steps_match():
    """A few optimizer steps (Muon + AdamW, gradient accumulation, router bias updates): the fused path tracks the
    current one, and with recompute it trains bit-identically to itself (PyTorch backend: the machinery; the
    kernels are compared above)."""
    af.set_backend("torch")
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    runs = {}
    for name, impl, frac in [("compiled", "compiled", 0.0), ("fused", "fused", 0.0), ("fused_ck", "fused", 0.5)]:
        m = MoELLM(cfg.replace(attnres_impl=impl))
        m.load_state_dict(base.state_dict())
        m.set_checkpointing(frac)
        opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16)
        ces = []
        for s in range(4):
            for k in range(2):
                x, y = _batch(cfg, seed=100 * s + k)
                loss, info = m(x, y)
                (loss / 2).backward()
                ces.append(float(info["ce"]))
            opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
            opt.zero_grad()
            m.update_router_bias()
        runs[name] = ces
    assert runs["fused"] == runs["fused_ck"]
    for a, b in zip(runs["fused"], runs["compiled"]):
        assert abs(a - b) < 1e-4 * abs(b), (runs["fused"], runs["compiled"])


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_model_eval_and_no_grad(backend):
    af.set_backend(backend)
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    m_c = MoELLM(cfg).eval()
    m_f = MoELLM(cfg.replace(attnres_impl="fused")).eval()
    m_f.load_state_dict(m_c.state_dict())
    x, _ = _batch(cfg)
    with torch.no_grad():
        a, b = m_c(x), m_f(x)
    assert rel_rms(b, a) < 1e-5
    assert torch.equal(m_f(x).detach(), b)  # the same forward kernels with gradients on


def test_state_dict_and_config_compatible():
    """Same parameters and state_dict keys as the compiled path (checkpoints of the running model load as they
    are); the option round-trips through the config and train.py's --override."""
    cfg = tiny(residual="block_attnres")
    m_c, m_f = MoELLM(cfg), MoELLM(cfg.replace(attnres_impl="fused"))
    assert m_c.state_dict().keys() == m_f.state_dict().keys()
    assert all(a.shape == b.shape for a, b in zip(m_c.state_dict().values(), m_f.state_dict().values()))
    m_f.load_state_dict(m_c.state_dict())
    assert ModelConfig.from_dict(cfg.replace(attnres_impl="fused").to_dict()).attnres_impl == "fused"
    assert ModelConfig.from_dict({k: v for k, v in cfg.to_dict().items() if k != "attnres_impl"}).attnres_impl \
        == "compiled"  # configs saved before the option existed
    with pytest.raises(NotImplementedError):
        MoELLM(cfg.replace(attnres_impl="bogus"))
    with pytest.raises(NotImplementedError):
        MoELLM(cfg.replace(attnres_impl="fused", attnres_layers_per_block=5, n_layers=10))


def test_fused_with_offload_master_trains_like_plain_bundle():
    """The fused path with persistent bf16 weight copies and fp32 gradient buffers (offload_master) under bf16
    autocast trains bit-identically to the plain optimizer bundle (as the compiled path does)."""
    from moe_llm.layers import common
    af.set_backend("torch")
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", attnres_impl="fused")
    base = MoELLM(cfg)
    base.set_checkpointing(0.5)
    runs = []
    try:
        for master in (True, False):
            m = copy.deepcopy(base)
            opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16, offload_master=master)
            opt.begin()
            ces = []
            for s in range(3):
                for k in range(2):
                    x, y = _batch(cfg, seed=1000 * s + k)
                    with torch.autocast(MCPU, dtype=torch.bfloat16):
                        loss, info = m(x, y)
                    (loss / 2).backward()
                    ces.append(float(info["ce"]))
                ces.append(float(opt.clip_grad_norm(0.05)))
                opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
                opt.zero_grad()
                m.update_router_bias()
            runs.append(ces)
    finally:
        common.clear_cached_copies()
    assert runs[0] == runs[1]


# ----------------------------------------------------------------------------- train.py: --override and resume
def _bins(root):
    import numpy as np
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


def _train(out, data, impl, steps=8):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", MOE_ATTNRES_FUSED_BACKEND="torch")
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--preset", "tiny",
           "--override", json.dumps({"engram_offload": True, "residual": "block_attnres", "attnres_impl": impl}),
           "--data", data, "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--out", out,
           "--seq-len", "32", "--micro-bsz", "2", "--global-tokens", "128", "--steps", str(steps), "--lr", "3e-3",
           "--warmup", "2", "--log-every", "1", "--eval-every", "0", "--ckpt-every", "4", "--grad-ckpt"]
    r = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"ce"' in l], r.stdout


def test_train_py_switch_to_fused_on_resume(tmp_path):
    """train.py --override '{"attnres_impl": "fused"}' resumes a checkpoint written by the compiled path: the
    switch is a restart, and the losses after it stay within rounding of the run that kept the compiled op."""
    data = str(tmp_path / "bins")
    _bins(data)
    full, _ = _train(str(tmp_path / "full"), data, "compiled")
    part = str(tmp_path / "part")
    _train(part, data, "compiled")
    os.remove(os.path.join(part, "ckpt_0000008.pt"))   # resume from step 4 with the fused op
    open(os.path.join(part, "log.jsonl"), "w").close()
    resumed, stdout = _train(part, data, "fused")
    assert "AttnRes op fused" in stdout, stdout
    assert [r["step"] for r in resumed] == list(range(4, 8))
    for a, b in zip(full[4:], resumed):
        assert abs(a["ce"] - b["ce"]) < 1e-3 * abs(a["ce"]), (a, b)


# ----------------------------------------------------------------------------- GPU, main size
def _bench():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_attnres_fused
    return bench_attnres_fused


@gpu
def test_gpu_block_main_size_accuracy():
    """Block 7 of main_2b: 8 completed sources [2, 4096, 1024] bf16, 7 ops (6 layers' ops + out_res), fp32
    parameters. Errors vs fp64: the fused path is no less accurate than the torch.compile'd op."""
    srcs, mods, parts, gouts = make_block(8, B=2, T=4096, D=1024, dtype=torch.bfloat16, w_std=0.05, device="cuda")
    ref = run_ref64(srcs, mods, parts, gouts)
    ref["out"] = {k: o.detach() for k, o in ref["out"].items()}
    torch.cuda.empty_cache()
    e_cur = errors(run_current(srcs, mods, parts, gouts, compiled=True), ref)
    e_fus = errors(run_fused(srcs, mods, parts, gouts), ref)
    print(f"\nmain-size block vs fp64: compiled {e_cur}\n                         fused    {e_fus}")
    assert e_fus["out"] <= 1.05 * e_cur["out"] + 1e-6
    assert e_fus["part"] <= 1.05 * e_cur["part"] + 1e-6
    assert e_fus["src"] <= e_cur["src"] + 1e-6
    assert e_fus["w"] < 1e-4 and e_fus["g"] < 1e-4


@gpu
def test_gpu_deterministic_and_no_host_sync():
    """The fused AttnRes part of a micro-batch at main size, forward + backward: a rerun and a run with every op
    under torch.utils.checkpoint are bit-identical, and nothing synchronizes with the host
    (torch.cuda.set_sync_debug_mode("error"); compile / autotune in a warm-up run first)."""
    bench = _bench()
    mods, emb, gout = bench.make_workload()
    ref_out, ref_g = bench.fwd_bwd("fused", mods, emb, gout)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out, g = bench.fwd_bwd("fused", mods, emb, gout)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(out, ref_out) and all(torch.equal(a, b) for a, b in zip(g, ref_g))
    srcs, mods2, parts, gouts = make_block(8, B=2, T=4096, D=1024, dtype=torch.bfloat16, w_std=0.05, device="cuda")
    a = run_fused(srcs, mods2, parts, gouts)
    b = run_fused(srcs, mods2, parts, gouts, ckpt=True)
    for key in ("out", "part", "w", "g"):
        assert all(torch.equal(a[key][k], b[key][k]) for k in a[key]), key
    assert all(torch.equal(x, y) for x, y in zip(a["src"], b["src"]))


@gpu
def test_gpu_small_model_matches_compiled():
    """Small model on the real GPU path (bf16 autocast, bf16 residual stream, fla KDA, partial recompute, fusion on):
    fused vs compiled. Same loss up to bf16 rounding, the whole gradient (all parameters together) no farther from an
    fp32 reference (same weights and inputs, autocast off, fp32 residual stream) than the compiled path's (1.1x), and
    every tensor within 5% of the compiled path. Prints the per-tensor table first.
    Dense FFNs in every layer: in a tiny MoE model (512 tokens) a last-bit difference anywhere flips near-tie top-k
    routing decisions, which moves expert and norm gradients by several percent (seen: 5.2% on blocks.3.ffn.w_up,
    then 5.5% vs 3.0% on blocks.3.norm2.weight), i.e. chaos, not rounding. AttnRes inside the MoE model is covered
    by the main-size accuracy tests here and by the 3-vs-3 drift test of the real model before any switch."""
    from moe_llm.layers import common
    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True, residual="block_attnres")
    cfg = cfg.replace(n_dense_layers=cfg.n_layers)  # no MoE routing: see the docstring
    base = MoELLM(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, 257), device="cuda")

    def run(impl, amp):
        m = MoELLM(cfg.replace(attnres_impl=impl, residual_bf16=amp)).cuda()
        m.load_state_dict(base.state_dict())
        m.set_checkpointing(0.5)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, _ = m(x[:, :-1], x[:, 1:])
        loss.backward()
        return float(loss), {n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None}

    common.set_fuse(True)
    try:
        l_c, g_c = run("compiled", True)
        l_f, g_f = run("fused", True)
        l_r, g_r = run("compiled", False)  # fp32 reference
    finally:
        common.set_fuse(False)
    assert g_c.keys() == g_f.keys() == g_r.keys()
    keys = sorted(g_c)
    rows = [(k, rel_rms(g_f[k], g_r[k]), rel_rms(g_c[k], g_r[k]), rel_rms(g_f[k], g_c[k])) for k in keys]
    flat = {name: torch.cat([g[k].flatten() for k in keys]) for name, g in (("c", g_c), ("f", g_f), ("r", g_r))}
    glob_f, glob_c = rel_rms(flat["f"], flat["r"]), rel_rms(flat["c"], flat["r"])
    better = sum(e_f <= e_c for _, e_f, e_c, _ in rows)
    print(f"\nloss compiled {l_c:.6f} fused {l_f:.6f} fp32 {l_r:.6f}")
    print(f"whole gradient vs fp32: fused {glob_f:.3e}  compiled {glob_c:.3e}; fused closer in {better}/{len(rows)} tensors")
    for k, e_f, e_c, d in sorted(rows, key=lambda r: r[1] - r[2], reverse=True):
        print(f"{k:40s} |fused-fp32| {e_f:.3e}  |compiled-fp32| {e_c:.3e}  |fused-compiled| {d:.3e}")
    # the meaningful check: the whole gradient no farther from fp32 than the compiled path's (per-tensor errors of a
    # fraction of a percent are noise: one of dozens can differ by 1.6x either way); per tensor only a sanity bound
    assert abs(l_f - l_c) < 2e-3 * l_c, (l_f, l_c)
    assert glob_f <= 1.1 * glob_c + 1e-4, (glob_f, glob_c)
    assert all(d < 0.05 for _, _, _, d in rows), [(k, d) for k, _, _, d in rows if d >= 0.05]


@gpu
def test_gpu_timing_microbatch():
    """Forward + backward of the AttnRes part of one main_2b micro-batch (24 layers, 8 blocks, [2, 4096, 1024] bf16),
    compiled vs fused: prints both and the estimated saving per optimizer step; the fused path must be faster."""
    bench = _bench()
    mods, emb, gout = bench.make_workload()
    res = {impl: bench.time_impl(impl, mods, emb, gout, reps=10) for impl in ("compiled", "fused")}
    for impl, (f, b, peak) in res.items():
        print(f"\n{impl:9s} forward {f:6.2f} ms  backward {b:6.2f} ms  total {f + b:6.2f} ms  peak +{peak / 2**30:.2f} GB")
    (fc, bc, _), (ff, bf, _) = res["compiled"], res["fused"]
    print(f"saving ~{((fc - ff) * 1.25 + (bc - bf)) * 64 / 1e3:.2f} s per optimizer step (64 micro-batches, "
          f"forward of 25% of the layers twice)")
    assert ff + bf < fc + bc
