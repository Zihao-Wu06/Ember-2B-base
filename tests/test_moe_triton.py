"""moe_impl="triton" (moe_llm/layers/moe_triton.py) against the plain-PyTorch reference and the bmm path.

CPU: the kernels run in Triton's interpreter at small sizes (TRITON_INTERPRET=1 is set below when there is no
GPU; it must be set before triton is imported). With a GPU the same small tests run on it, plus main-size
checks against bmm and an fp32 reference, bit-exactness of recompute / checkpoint / reruns, no CPU-GPU sync
(torch.cuda.set_sync_debug_mode("error")), every autotune candidate, memory kept for backward, and fwd+bwd
timing at routing max/mean 1.0, 1.3, 1.8, 3.0 (pytest -s to see the timings)."""
import os
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from torch.utils.checkpoint import checkpoint  # noqa: E402

from moe_llm.config import tiny  # noqa: E402
from moe_llm.layers import moe_triton as mt  # noqa: E402
from moe_llm.layers.moe import MoE  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NAMES = ["x", "w", "w_gate", "w_up", "w_down"]


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def routing(N, K, E, kind, gen):
    """Expert ids [N, K], distinct per token."""
    if kind == "collapsed":  # every token picks the same K experts: E - K experts get nothing
        return torch.arange(K).repeat(N, 1)
    logits = torch.zeros(N, E)
    if kind == "skewed":
        logits += torch.linspace(3.0, -1.0, E)
    elif kind == "empty":
        logits[:, 1::2] = -1e9  # only the even experts get tokens
    gumbel = -torch.log(-torch.log(torch.rand(N, E, generator=gen).clamp(1e-9, 1 - 1e-9)))
    return torch.topk(logits + gumbel, K, dim=-1).indices


def make_inputs(N=160, D=64, I=32, E=8, K=2, kind="balanced", dtype=torch.float32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=gen).to(dtype)
    idx = routing(N, K, E, kind, gen)
    w = torch.rand(N, K, generator=gen) + 0.1
    wg = (torch.randn(E, D, I, generator=gen) * D ** -0.5).to(dtype)
    wu = (torch.randn(E, D, I, generator=gen) * D ** -0.5).to(dtype)
    wd = (torch.randn(E, I, D, generator=gen) * I ** -0.5).to(dtype)
    leaves = [t.to(DEV).requires_grad_() for t in (x, w, wg, wu, wd)]
    return leaves, idx.to(DEV)


def fwd_bwd(fn, leaves, idx, dy=None, **kw):
    x, w, wg, wu, wd = leaves
    y = fn(x, idx, w, wg, wu, wd, **kw)
    if dy is None:
        dy = torch.randn(y.shape, generator=torch.Generator().manual_seed(7)).to(y.device, y.dtype)
    return [y.detach()] + list(torch.autograd.grad(y, leaves, dy)), dy


# ----------------------------------------------------------------------------- small sizes (interpreter on CPU)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["balanced", "skewed", "empty", "collapsed"])
def test_matches_reference(kind, dtype):
    leaves, idx = make_inputs(kind=kind, dtype=dtype)
    got, dy = fwd_bwd(mt.routed_experts, leaves, idx)
    ref, _ = fwd_bwd(mt.routed_experts_reference, leaves, idx, dy)
    fp32 = dtype == torch.float32
    assert got[0].dtype == dtype and got[0].shape == ref[0].shape
    assert rel_rms(got[0], ref[0]) < (1e-5 if fp32 else 4e-3)
    for name, a, b, leaf in zip(NAMES, got[1:], ref[1:], leaves):
        assert a.dtype == leaf.dtype and a.shape == leaf.shape, name
        assert torch.isfinite(a).all(), name
        assert rel_rms(a, b) < (1e-5 if fp32 else 1e-2), (name, rel_rms(a, b))
    if kind in ("empty", "collapsed"):  # experts without rows: exactly zero weight gradients
        used = torch.zeros(leaves[2].shape[0], dtype=torch.bool, device=idx.device)
        used[idx.flatten()] = True
        for g in got[3:]:
            assert (g[~used] == 0).all() and (g[used] != 0).any()


def test_x_in_fp32_with_bf16_weights():
    """x in fp32, weights in bf16 (the bmm path under autocast with a fp32 residual): computed in bf16, dx in
    fp32 with the k-sum kept in fp32."""
    leaves, idx = make_inputs(dtype=torch.bfloat16)
    leaves[0] = leaves[0].detach().float().requires_grad_()
    got, dy = fwd_bwd(mt.routed_experts, leaves, idx)
    ref, _ = fwd_bwd(mt.routed_experts_reference, leaves, idx, dy)
    assert got[0].dtype == torch.bfloat16 and got[1].dtype == torch.float32
    for a, b in zip(got, ref):
        assert rel_rms(a, b) < 1e-2


def test_recompute_and_checkpoint_are_bit_identical():
    """recompute=True and torch.utils.checkpoint (non-reentrant) give exactly the plain values and gradients;
    so does a second plain run (no atomics anywhere)."""
    leaves, idx = make_inputs(kind="skewed", dtype=torch.bfloat16)
    ref, dy = fwd_bwd(mt.routed_experts, leaves, idx)

    def ckpt(x, idx, *rest):
        return checkpoint(lambda *a: mt.routed_experts(a[0], idx, *a[1:]), x, *rest, use_reentrant=False)

    for fn, kw in [(mt.routed_experts, {}), (mt.routed_experts, {"recompute": True}), (ckpt, {})]:
        got, _ = fwd_bwd(fn, leaves, idx, dy, **kw)
        for name, a, b in zip(["y"] + NAMES, got, ref):
            assert torch.equal(a, b), (fn.__name__, kw, name)


def test_wgrad_copy_mode_bit_identical(monkeypatch):
    """Weight gradients from expert-sorted copies of x and dy (MOE_TRITON_WGRAD_COPY=1) equal the gathered ones."""
    leaves, idx = make_inputs(kind="skewed", dtype=torch.bfloat16)
    ref, dy = fwd_bwd(mt.routed_experts, leaves, idx)
    monkeypatch.setattr(mt, "_WGRAD_COPY", True)
    got, _ = fwd_bwd(mt.routed_experts, leaves, idx, dy)
    for name, a, b in zip(["y"] + NAMES, got, ref):
        assert torch.equal(a, b), name


def test_other_tile_sizes_agree():
    """Bigger tiles (fewer, fuller row blocks; a different k-chunking) give the same values up to fp32 rounding."""
    leaves, idx = make_inputs(N=200, kind="skewed")
    ref, dy = fwd_bwd(mt.routed_experts, leaves, idx)
    try:
        mt.set_configs(32, 32, 32) if DEV == "cpu" else mt.set_configs(64, 64, 32, 4, 3)
        got, _ = fwd_bwd(mt.routed_experts, leaves, idx, dy)
    finally:
        mt.reset_configs()
    for a, b in zip(got, ref):
        assert rel_rms(a, b) < 1e-5


def test_no_grad_forward_matches():
    leaves, idx = make_inputs(dtype=torch.bfloat16)
    with torch.no_grad():
        y = mt.routed_experts(leaves[0], idx, *leaves[1:])
    assert torch.equal(y, mt.routed_experts(leaves[0], idx, *leaves[1:]).detach())


@pytest.mark.parametrize("autocast", [False, True])
def test_moe_layer_matches_bmm(autocast):
    """The MoE layer with moe_impl="triton" vs "bmm": output and every parameter gradient (the router's through
    the combine weights), fp32 and under bf16 autocast."""
    cfg = tiny()
    torch.manual_seed(1)
    moe = MoE(cfg, 1).to(DEV)
    x = torch.randn(3, 40, cfg.d_model, device=DEV)
    res = {}
    for impl in ("bmm", "triton"):
        moe.impl = impl
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_()
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=autocast):
            y, aux = moe(xi)
        (y.float().square().mean() + aux).backward()
        res[impl] = [y.detach(), xi.grad] + [p.grad for p in moe.parameters()]
    tol = 2e-2 if autocast else 1e-5
    for i, (a, b) in enumerate(zip(res["triton"], res["bmm"])):
        assert a.dtype == b.dtype and rel_rms(a, b) < tol, (i, rel_rms(a, b))


def test_load_counter_matches_bincount():
    """The triton path counts expert loads with index_add_ (torch.bincount syncs with the host): same values."""
    cfg = tiny()
    torch.manual_seed(4)
    moe = MoE(cfg, 1).to(DEV).train()
    x = torch.randn(2, 40, cfg.d_model, device=DEV)
    loads = {}
    for impl in ("bmm", "triton"):
        moe.impl = impl
        moe.load.zero_()
        with torch.no_grad():
            for _ in range(3):
                moe(x)
        loads[impl] = moe.load.clone()
    assert torch.equal(loads["bmm"], loads["triton"])
    assert loads["triton"].sum().item() == 3 * 80 * cfg.top_k


def test_moe_layer_recompute_experts_bit_identical():
    cfg = tiny(moe_impl="triton")
    torch.manual_seed(2)
    moe = MoE(cfg, 1).to(DEV).train()
    x = torch.randn(2, 48, cfg.d_model, device=DEV)
    res = []
    for rec in (False, True):
        moe.recompute_experts = rec
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_()
        y, aux = moe(xi)
        (y.square().mean() + aux).backward()
        res.append([y.detach(), xi.grad] + [p.grad for p in moe.parameters()])
    for a, b in zip(*res):
        assert torch.equal(a, b)


@pytest.mark.parametrize("residual", ["standard", "block_attnres"])
def test_model_triton_matches_bmm_and_recompute_is_exact(residual):
    """Whole tiny model: triton vs bmm within fp32 rounding. With partial checkpointing (whole-layer recompute in
    some layers, built-in expert recompute in the others) and with activation offloading (moe_llm/offload.py,
    every saved tensor sent to host memory) the triton model is bit-identical to no recompute."""
    import copy
    torch.manual_seed(0)
    cfg = tiny(residual=residual)
    base = MoELLM(cfg).to(DEV)
    ids = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(3)).to(DEV)
    out = {}
    for name, impl, frac, off in [("bmm", "bmm", 0.0, 0.0), ("triton", "triton", 0.0, 0.0),
                                  ("triton_ckpt", "triton", 0.5, 0.0), ("triton_offload", "triton", 1.0, 0.5)]:
        m = copy.deepcopy(base)
        for moe in m.moe_layers():
            moe.impl = impl
        m.set_checkpointing(frac, offload_frac=off)
        if m.offloader is not None:
            m.offloader.min_bytes = 0  # tiny model: offload everything
        loss, _ = m(ids[:, :-1], ids[:, 1:])
        loss.backward()
        if m.offloader is not None:
            assert m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0, m.offloader.stats
        if name == "triton_ckpt" and residual == "standard":
            assert any(moe.recompute_experts for moe in m.moe_layers())
        out[name] = (loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None})
    l_b, g_b = out["bmm"]
    l_t, g_t = out["triton"]
    assert abs(l_t.item() - l_b.item()) < 1e-5 * abs(l_b.item())
    assert g_t.keys() == g_b.keys()
    for k in g_b:
        assert rel_rms(g_t[k], g_b[k]) < 1e-4, k
    for name in ("triton_ckpt", "triton_offload"):
        l_c, g_c = out[name]
        assert torch.equal(l_c, l_t) and g_c.keys() == g_t.keys(), name
        for k in g_t:
            assert torch.equal(g_c[k], g_t[k]), (name, k)


# ----------------------------------------------------------------------------- GPU, main sizes
def _bench():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import bench_moe_kernels
    return bench_moe_kernels


def _main_layer():
    from moe_llm.config import main_2b
    torch.manual_seed(0)
    cfg = main_2b()
    return cfg, MoE(cfg, 1).cuda().train()


def _experts_fwd_bwd(moe, x, idx, w, dy):
    moe.zero_grad(set_to_none=True)
    xi, wi = x.detach().requires_grad_(), w.detach().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = moe.experts(xi, idx, wi)
    y.backward(dy)
    return [y.detach(), xi.grad, wi.grad, moe.w_gate.grad, moe.w_up.grad, moe.w_down.grad]


@gpu
@pytest.mark.parametrize("skew", [1.0, 1.8, 3.0])
def test_gpu_main_size_vs_bmm_and_fp32_reference(skew):
    """N = 8192, d = 1024, E = 32, k = 4, I = 512 under bf16 autocast: triton vs bmm, and both vs an fp32
    reference on the same bf16 weights and inputs. triton must be as close to the fp32 values as bmm is
    (it rounds less), and recompute / a rerun must be bit-identical."""
    cfg, moe = _main_layer()
    N = 8192
    idx, w = _bench().synthetic_routing(N, cfg.top_k, cfg.n_experts, skew, seed=1)
    x = torch.randn(N, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(N, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    res = {}
    for impl, rec in [("bmm", False), ("triton", False), ("triton", True), ("triton2", False)]:
        moe.impl, moe.recompute_experts = impl.rstrip("2"), rec
        res[(impl, rec)] = _experts_fwd_bwd(moe, x, idx, w, dy)
    moe.recompute_experts = False
    # fp32 truth: the same function in fp32 on the bf16-valued inputs; w as the layer passes it (rounded to bf16)
    leaves = [t.detach().float().requires_grad_() for t in
              (x, w.to(torch.bfloat16), moe.w_gate.to(torch.bfloat16), moe.w_up.to(torch.bfloat16),
               moe.w_down.to(torch.bfloat16))]
    yr = mt.routed_experts_reference(leaves[0], idx, *leaves[1:])
    truth = [yr.detach()] + list(torch.autograd.grad(yr, leaves, dy.float()))
    names = ["y", "dx", "dw", "dW_gate", "dW_up", "dW_down"]
    b, t = res[("bmm", False)], res[("triton", False)]
    for n, gb, gt, gr in zip(names, b, t, truth):
        e_b, e_t, d = rel_rms(gb, gr), rel_rms(gt, gr), rel_rms(gt, gb)
        print(f"skew {skew} {n:8s} |bmm-fp32| {e_b:.2e}  |triton-fp32| {e_t:.2e}  |triton-bmm| {d:.2e}")
        assert d < (1e-2 if n == "y" else 3e-2), n
        assert e_t <= 1.1 * e_b + 1e-4, n
    for key in [("triton", True), ("triton2", False)]:
        for n, a, c in zip(names, res[key], t):
            assert torch.equal(a, c), (key, n)


@gpu
def test_gpu_no_host_sync():
    """The triton MoE layer (router, load counter, experts, shared expert; forward + backward; plain, with the
    expert recompute, and under torch.utils.checkpoint) makes no CPU-GPU synchronisation:
    torch.cuda.set_sync_debug_mode("error") raises on any. Compilation / autotuning (which sync, once per
    shape) happen in a warm-up step first. The bmm path is probed in "warn" mode and its sync sites printed."""
    import warnings
    cfg, moe = _main_layer()
    x0 = torch.randn(2, 4096, cfg.d_model, device="cuda", dtype=torch.bfloat16)

    def step(ckpt=False):
        moe.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y, aux = checkpoint(moe, x, use_reentrant=False) if ckpt else moe(x)
        (y.float().square().mean() + aux).backward()

    moe.impl = "triton"
    try:
        for rec, ckpt in [(False, False), (True, False), (False, True)]:
            moe.recompute_experts = rec
            step(ckpt)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                step(ckpt)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        moe.impl, moe.recompute_experts = "bmm", True
        step()
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                step()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        sites = [f"{os.path.basename(w.filename)}:{w.lineno}" for w in caught if "synchroniz" in str(w.message)]
        print(f"bmm path (expert recompute): {len(sites)} host syncs per layer fwd+bwd at {sorted(set(sites))}")
    finally:
        moe.impl, moe.recompute_experts = "bmm", False


@gpu
def test_gpu_every_config_matches():
    """Each autotune candidate, forced for all kernels, against the autotuned result (configs a kernel cannot
    fit in shared memory are skipped)."""
    leaves, idx = make_inputs(N=2048, D=256, I=128, E=8, K=2, kind="skewed", dtype=torch.bfloat16)
    ref, dy = fwd_bwd(mt.routed_experts, leaves, idx)
    cands = {tuple(c) for c in mt._SINGLE + mt._DUAL + mt._EPI}
    ran = 0
    try:
        for c in sorted(cands):
            mt.set_configs(*c)
            try:
                got, _ = fwd_bwd(mt.routed_experts, leaves, idx, dy)
            except Exception as e:  # OutOfResources, possibly re-raised through the autograd engine
                if "OutOfResources" in type(e).__name__ or "out of resource" in str(e).lower():
                    continue
                raise
            ran += 1
            for n, a, b in zip(["y"] + NAMES, got, ref):
                assert rel_rms(a, b) < 1e-2, (c, n, rel_rms(a, b))
    finally:
        mt.reset_configs()
    assert ran >= len(cands) // 2


@gpu
def test_gpu_kept_memory_independent_of_routing():
    """With recompute_experts the triton path keeps only w and the sort for backward (x is the caller's): the
    same fraction of a MB for balanced and skewed routing, far below the bmm path's kept buffers (gathered
    input and expert output). Persistent bf16 weight copies as in the main run (offload_master)."""
    from moe_llm.layers import common
    cfg, moe = _main_layer()
    copies = [p.detach().to(torch.bfloat16).requires_grad_() for p in (moe.w_gate, moe.w_up, moe.w_down)]
    for p, c in zip((moe.w_gate, moe.w_up, moe.w_down), copies):
        common.set_cached_copy(p, c)
    N = 8192
    x0 = torch.randn(N, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    kept = {}
    try:
        for impl in ("bmm", "triton"):
            for skew in (1.0, 3.0):
                idx, w = _bench().synthetic_routing(N, cfg.top_k, cfg.n_experts, skew, seed=2)
                moe.impl, moe.recompute_experts = impl, True
                moe.zero_grad(set_to_none=True)
                for c in copies:
                    c.grad = None
                x = x0.clone().requires_grad_()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = moe.experts(x, idx, w)
                kept[(impl, skew)] = torch.cuda.memory_allocated() - base - y.numel() * y.element_size()
                y.float().square().mean().backward()
    finally:
        common.clear_cached_copies()
        moe.recompute_experts = False
    mb = 2 ** 20
    print({k: f"{v / mb:.1f} MB" for k, v in kept.items()})
    assert abs(kept[("triton", 3.0)] - kept[("triton", 1.0)]) < 0.5 * mb
    assert kept[("triton", 1.0)] < 2 * mb < 64 * mb < kept[("bmm", 1.0)]


@gpu
def test_gpu_timing_vs_bmm():
    """Prints fwd+bwd time of the full experts() path (sort/dispatch, GEMMs, combine) for bmm and triton at
    routing max/mean 1.0, 1.3, 1.8, 3.0, with the expert recompute on (as in the main run) and off."""
    bench = _bench()
    cfg, moe = _main_layer()
    N = 8192
    x = torch.randn(N, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(N, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    for skew in (1.0, 1.3, 1.8, 3.0):
        idx, w = bench.synthetic_routing(N, cfg.top_k, cfg.n_experts, skew, seed=3)
        line = []
        for rec in (True, False):
            for impl in ("bmm", "triton"):
                moe.impl, moe.recompute_experts = impl, rec
                ms = bench.time_ms(lambda: _experts_fwd_bwd(moe, x, idx, w, dy), reps=10, warmup=3)
                line.append(f"{impl}{'+rec' if rec else ''} {ms:.2f} ms")
        print(f"max/mean {bench.max_over_mean(idx, cfg.n_experts):.2f}: " + ", ".join(line))
    moe.recompute_experts = False
