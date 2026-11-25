"""conv_impl "fla" / "fla_gluon" (moe_llm/layers/common.py fla_conv_silu, ShortConv): fla's own causal_conv1d kernels
for KDA's q/k/v short convs, against the running path (conv_silu: torch.compile'd shifted multiply-adds with fusion on,
eager conv1d otherwise) and an fp64 reference.

CPU: fla's Triton kernels run in Triton's interpreter (TRITON_INTERPRET=1 is set below when there is no GPU; it must be
set before triton is imported); fla's Gluon kernels do not run there (GPU tests only). Op level: forward and both
gradients vs the current op (eager and CPU-compiled) and vs fp64 - the fla path must be no less accurate - and the bf16
pre-activation fla's backward recomputes is bit-identical to the current op's. Whole tiny model (dense FFNs, bf16
autocast): loss and every gradient close to the current path, the whole gradient no farther from an fp32 reference;
recompute (whole layers, FFN part, activation offloading) and reruns bit-identical; eval / no_grad; optimizer steps with
persistent bf16 copies; CPU fallback; config, state_dict and train.py --override on resume.
GPU (skipped without CUDA): main-shape accuracy of both fla paths vs the compiled path and fp64, pre-activation bit
identity, determinism, no host syncs, saved and peak memory of a main_2b KDA layer, a small dense model vs an fp32
reference, and timing (pytest -s prints the numbers)."""
import copy
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from moe_llm.config import ModelConfig, main_2b, tiny  # noqa: E402
from moe_llm.layers import common  # noqa: E402
from moe_llm.layers.common import CONV_IMPLS, ShortConv, conv_silu, fla_conv_silu  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402
from moe_llm.optim import OptimizerBundle, wsd_lr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fla_loads() -> bool:
    try:
        common._fla_conv_entry("fla")
        return True
    except Exception:
        return False


# CPU tensors run fla's Triton kernels only under the interpreter (the GPU tests below cover CUDA)
FLA_CPU = not torch.cuda.is_available() and common._triton_interpret() and _fla_loads()
fla_cpu = pytest.mark.skipif(not FLA_CPU, reason="needs fla + Triton's CPU interpreter (TRITON_INTERPRET=1)")


@pytest.fixture(scope="module", autouse=True)
def _interpreter_rounds_to_nearest_even():
    """Triton's CPU interpreter truncates fp32 -> bf16 (see moe_llm/layers/attnres_fused.py _round); the GPU rounds to
    nearest even, which fla's kernels request (fp_downcast_rounding="rtne", and the default of implicit casts). While
    this module's tests run, the interpreter's fp32 -> bf16 conversion rounds to nearest even (NaN stays NaN, overflow
    goes to inf), so fla's interpreted kernels round as they do on the GPU. Restored afterwards."""
    if not FLA_CPU:
        yield
        return
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as interp
    orig = interp._convert_float

    def convert(input, input_dtype, output_dtype, rounding_mode):
        if input_dtype == tl.float32 and output_dtype == tl.bfloat16:
            b = np.frombuffer(input.tobytes(), dtype=np.uint32).astype(np.uint64)
            r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
            nan = (b & 0x7FFFFFFF) > 0x7F800000
            r[nan] = ((b[nan] >> 16) | 0x40).astype(np.uint16)
            return r.reshape(input.shape)
        return orig(input, input_dtype, output_dtype, rounding_mode)

    interp._convert_float = convert
    try:
        yield
    finally:
        interp._convert_float = orig


def test_interpreter_bf16_rounding_matches_torch():
    """The fixture above: an interpreted Triton store of fp32 values into bf16 gives torch's (round-to-nearest-even)
    bf16 values, ties included."""
    if not FLA_CPU:
        pytest.skip("needs Triton's CPU interpreter")
    import triton
    import triton.language as tl

    @triton.jit
    def store(x_ptr, y_ptr, z_ptr, N: tl.constexpr):
        o = tl.arange(0, N)
        v = tl.load(x_ptr + o)
        tl.store(y_ptr + o, v)
        tl.store(z_ptr + o, tl.cast(v, tl.bfloat16, fp_downcast_rounding="rtne"))

    x = torch.cat([torch.randn(1000) * 3, torch.tensor([1 + 2 ** -8, 1 + 3 * 2 ** -8, -(1 + 2 ** -8), 3e38, 0.0, -0.0]),
                   torch.zeros(18)])
    y, z = torch.empty(1024, dtype=torch.bfloat16), torch.empty(1024, dtype=torch.bfloat16)
    store[(1,)](x, y, z, N=1024)
    ref = x.to(torch.bfloat16)
    assert torch.equal(y.view(torch.int16), ref.view(torch.int16))
    assert torch.equal(z.view(torch.int16), ref.view(torch.int16))


@pytest.fixture
def fuse_cpu():
    common.set_fuse(True, cpu=True)
    yield common
    common.set_fuse(False)


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def rel_max(a, b):
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# ----------------------------------------------------------------------------- op level helpers
def make_case(B=2, T=150, D=48, k=4, w_std=0.3, seed=0, device="cpu"):
    """bf16 input and output gradient, an fp32 conv weight [D, 1, k] as the model holds it. T = 150 spans three
    64-token blocks of fla's kernels (the last one partial), D = 48 one and a half 32-channel blocks."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D, generator=g).to(device, torch.bfloat16)
    w = (torch.randn(D, 1, k, generator=g) * w_std).to(device)
    dy = torch.randn(B, T, D, generator=g).to(device, torch.bfloat16)
    return x, w, dy


def ref64(x, w2, dy):
    """fp64: y = silu(c), c[t] = sum_i w[i] x[t-k+1+i] (w2: [D, k]); dx, dw = the exact gradients for dy."""
    x, w, dy = x.double(), w2.double(), dy.double()
    T, k = x.shape[1], w.shape[-1]
    xp = F.pad(x, (0, 0, k - 1, 0))
    c = sum(xp[:, i:i + T] * w[:, i] for i in range(k))
    s = torch.sigmoid(c)
    dc = dy * s * (1 + c * (1 - s))
    dcp = F.pad(dc, (0, 0, 0, k - 1))
    dx = sum(dcp[:, k - 1 - i:k - 1 - i + T] * w[:, i] for i in range(k))
    dw = torch.stack([(dc * xp[:, i:i + T]).sum((0, 1)) for i in range(k)], -1)
    return c * s, dx, dw


def preact_bf16(x, w2):
    """The compiled path's bf16 pre-activation (_conv_silu_shifted before the SiLU): fp32 products of the bf16 values
    summed in the same order, rounded once. w2: [D, k] in the input dtype."""
    T, k = x.shape[1], w2.shape[-1]
    xp, wf = F.pad(x.float(), (0, 0, k - 1, 0)), w2.float()
    c = xp[:, 0:T] * wf[:, 0]
    for i in range(1, k):
        c = c + xp[:, i:i + T] * wf[:, i]
    return c.to(x.dtype)


def run_op(impl, x, w, dy):
    """(y, dx, dw [D, k] fp32 as accumulated into the parameter's .grad) of one ShortConv call. impl: "eager" /
    "compiled" (conv_silu; compiled when fusion is on) or "fla" / "fla_gluon" (fla_conv_silu)."""
    xi = x.detach().clone().requires_grad_()
    wi = w.detach().clone().requires_grad_()
    if impl in ("fla", "fla_gluon"):
        y = fla_conv_silu(xi, wi, impl)
    elif impl == "eager":
        y = conv_silu.eager(xi, wi)
    else:
        y = conv_silu(xi, wi)
    y.backward(dy)
    return y.detach(), xi.grad, wi.grad[:, 0]


def errors(out, ref):
    return {name: rel_rms(o, r) for name, o, r in zip(("y", "dx", "dw"), out, ref)}


# ----------------------------------------------------------------------------- CPU: plumbing without fla
def test_config_state_dict_and_validation():
    """Same parameters and state_dict keys for every conv_impl (checkpoints load as they are); the option round-trips
    through the config, configs saved before it existed load as "compiled", unknown values are refused, and the KDA
    convs carry it (the Engram conv has no such switch)."""
    cfg = tiny()
    m_c, m_f = MoELLM(cfg), MoELLM(cfg.replace(conv_impl="fla"))
    assert m_c.state_dict().keys() == m_f.state_dict().keys()
    assert all(a.shape == b.shape for a, b in zip(m_c.state_dict().values(), m_f.state_dict().values()))
    m_f.load_state_dict(m_c.state_dict())
    convs = [m for m in m_f.modules() if isinstance(m, ShortConv)]
    assert convs and all(c.impl == "fla" for c in convs)
    assert len(convs) == 3 * sum(t == "K" for t in cfg.layer_types())
    assert ModelConfig.from_dict(cfg.replace(conv_impl="fla_gluon").to_dict()).conv_impl == "fla_gluon"
    assert ModelConfig.from_dict({k: v for k, v in cfg.to_dict().items() if k != "conv_impl"}).conv_impl == "compiled"
    assert main_2b().conv_impl == "compiled"
    assert CONV_IMPLS == ("compiled", "fla", "fla_gluon")
    with pytest.raises(ValueError):
        MoELLM(cfg.replace(conv_impl="cuda"))
    with pytest.raises(ValueError):
        ShortConv(16, 4, impl="bogus")


def test_fuse_proj_warns_that_conv_impl_is_inactive():
    with pytest.warns(UserWarning, match="no effect with fuse_proj"):
        MoELLM(tiny(fuse_proj=True, conv_impl="fla"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MoELLM(tiny(fuse_proj=True))


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("impl", ["fla", "fla_gluon"])
def test_cpu_tensors_fall_back_to_conv_silu(monkeypatch, impl, fuse):
    """CPU tensors outside Triton's interpreter: ShortConv(impl="fla"/"fla_gluon") is exactly the current conv_silu
    (eager, or compiled with fusion on): values and gradients bit-identical, fla never loaded."""
    monkeypatch.setattr(common, "_triton_interpret", lambda: False)
    monkeypatch.setattr(common, "_fla_conv_entry", lambda impl: pytest.fail("fla must not be used on CPU tensors"))
    common.set_fuse(fuse, cpu=fuse)
    try:
        torch.manual_seed(0)
        cur, new = ShortConv(48), ShortConv(48, impl=impl)
        new.load_state_dict(cur.state_dict())
        x, _, dy = make_case()
        outs = []
        for m in (cur, new):
            xi = x.clone().requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = m(xi)
            y.backward(dy)
            outs.append((y, xi.grad, m.weight.grad))
        assert all(torch.equal(a, b) for a, b in zip(*outs))
    finally:
        common.set_fuse(False)
    assert not common._FAILED, common._FAILED


# ----------------------------------------------------------------------------- CPU: fla's Triton kernels (interpreter)
@fla_cpu
def test_op_no_less_accurate_than_current(fuse_cpu):
    """bf16 input and output gradient, fp32 weight; errors against fp64 on the same values (weight rounded to bf16, as
    every path rounds it). Forward: fla rounds once after an fp32 SiLU, the current op rounds the pre-activation and
    again after the SiLU -> fla more accurate. dx: fla keeps dy * silu' in fp32 -> more accurate. dw: both end with
    one bf16 rounding (the weight's dtype), equal within 5%. Against the fp32 weights (the fp32 model) all paths carry
    the same weight rounding. The weight gradient reaches the fp32 .grad as a bf16 value in every path."""
    rows = {}
    for seed, w_std in ((0, 0.3), (1, 1.0), (2, 0.05)):
        x, w, dy = make_case(seed=seed, w_std=w_std)
        w_b = w[:, 0].to(torch.bfloat16)
        ref_b, ref_f = ref64(x, w_b, dy), ref64(x, w[:, 0], dy)
        outs = {impl: run_op(impl, x, w, dy) for impl in ("eager", "compiled", "fla")}
        for impl, out in outs.items():
            assert out[0].dtype == out[1].dtype == torch.bfloat16 and out[2].dtype == torch.float32
            assert torch.equal(out[2], out[2].to(torch.bfloat16).float()), impl  # dw rounded to bf16 in every path
        e = {impl: errors(out, ref_b) for impl, out in outs.items()}
        e32 = {impl: errors(out, ref_f) for impl, out in outs.items()}
        same = (outs["fla"][0] == outs["compiled"][0]).float().mean().item()
        rows[(seed, w_std)] = (e, e32, same)
        print(f"\nw_std {w_std}: rel. RMS error vs fp64 (bf16 weight)  " + "  ".join(
            f"{impl}: y {v['y']:.3e} dx {v['dx']:.3e} dw {v['dw']:.3e}" for impl, v in e.items()))
        print(f"           vs fp64 (fp32 weight)  " + "  ".join(
            f"{impl}: y {v['y']:.3e} dx {v['dx']:.3e} dw {v['dw']:.3e}" for impl, v in e32.items())
            + f"   fla == compiled in {same:.1%} of y")
        assert e["fla"]["y"] < e["compiled"]["y"] and e["fla"]["y"] < e["eager"]["y"], e
        assert e["fla"]["dx"] < e["compiled"]["dx"] and e["fla"]["dx"] < e["eager"]["dx"], e
        assert e["fla"]["dw"] <= 1.05 * min(e["compiled"]["dw"], e["eager"]["dw"]) + 1e-7, e
        assert e32["fla"]["y"] <= 1.02 * e32["compiled"]["y"] and e32["fla"]["dx"] <= 1.02 * e32["compiled"]["dx"]
        for impl in ("compiled", "fla"):  # max errors: a bf16 rounding or two, nothing systematic
            assert rel_max(outs[impl][0], ref_b[0]) < 1e-2 and rel_max(outs[impl][1], ref_b[1]) < 2e-2, impl
        assert same > 0.5, same  # mostly the same bf16 values (~75%): only the double rounding differs
    assert not common._FAILED, common._FAILED


@fla_cpu
def test_preactivation_bit_identical_to_current():
    """fla's conv without activation (what its backward recomputes for the SiLU derivative, and the value its forward
    applies the SiLU to before rounding) is bit-identical to the current op's bf16 pre-activation: same products, same
    fp32 summation order, one rounding. Also at the left edge (zero padding) and for kernel sizes 2..4."""
    from fla.modules.causal_conv1d import causal_conv1d
    for k in (4, 3, 2):
        x, w, _ = make_case(k=k, seed=k)
        w_b = w[:, 0].to(torch.bfloat16).contiguous()
        y, _ = causal_conv1d(x=x, weight=w_b, activation=None, backend="triton")
        assert torch.equal(y, preact_bf16(x, w_b)), k


@fla_cpu
def test_saved_bytes_vs_compiled(fuse_cpu):
    """What each path keeps from the forward until the backward (saved-tensor hooks, by storage): the torch.compile'd
    conv_silu saves an fp32 padded copy of the input [B, T + k - 1, D] (AOTAutograd's partition: the pad is not
    recomputed in the backward) and the bf16 weight; fla saves the bf16 input itself and the bf16 weight - half the
    bytes. In KDA the conv input is a projection output nothing else keeps, so this is the activation memory saved."""
    x, w, _ = make_case(B=2, T=150, D=48)
    B, T, D, k = *x.shape, w.shape[-1]

    def saved(fn):
        st = {}

        def pack(t):
            s = t.untyped_storage()
            st[s.data_ptr()] = (s.nbytes(), t.dtype, tuple(t.shape))
            return t

        xi, wi = x.clone().requires_grad_(), w.clone().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            fn(xi, wi)
        return sorted(st.values(), key=lambda v: -v[0])

    comp = saved(conv_silu)
    fla = saved(lambda a, b: fla_conv_silu(a, b, "fla"))
    print(f"\nsaved for the backward: compiled {comp}\n                        fla      {fla}")
    assert comp[0] == (B * (T + k - 1) * D * 4, torch.float32, (B, T + k - 1, D)), comp
    assert fla[0] == (B * T * D * 2, torch.bfloat16, (B, T, D)), fla
    assert sum(v[0] for v in fla) <= sum(v[0] for v in comp) // 2 + D * k * 2
    assert not common._FAILED, common._FAILED


@fla_cpu
def test_op_recompute_rerun_eval_and_layouts():
    """A rerun, torch.utils.checkpoint around the call, and no_grad give the same bits; a non-contiguous output
    gradient gives the same gradients as a contiguous one; the input dtype passes through (fp32 input: fp32 weight)."""
    from torch.utils.checkpoint import checkpoint
    x, w, dy = make_case(seed=3)
    a = run_op("fla", x, w, dy)
    assert all(torch.equal(p, q) for p, q in zip(a, run_op("fla", x, w, dy)))
    xi, wi = x.clone().requires_grad_(), w.clone().requires_grad_()
    y = checkpoint(fla_conv_silu, xi, wi, "fla", use_reentrant=False)
    y.backward(dy)
    assert torch.equal(y, a[0]) and torch.equal(xi.grad, a[1]) and torch.equal(wi.grad[:, 0], a[2])
    with torch.no_grad():
        assert torch.equal(fla_conv_silu(x, w, "fla"), a[0])
    dy_t = dy.transpose(0, 1).contiguous().transpose(0, 1)  # same values, other strides
    b = run_op("fla", x, w, dy_t)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    x32 = x.float()
    y32, dx32, dw32 = run_op("fla", x32, w, dy.float())
    r = ref64(x32, w[:, 0], dy.float())
    assert y32.dtype == dx32.dtype == torch.float32
    assert rel_rms(y32, r[0]) < 1e-6 and rel_rms(dx32, r[1]) < 1e-6 and rel_rms(dw32, r[2]) < 1e-6


# ----------------------------------------------------------------------------- CPU: whole tiny model (interpreter)
def _cfg(**kw):
    cfg = tiny(**kw)
    return cfg.replace(n_dense_layers=cfg.n_layers)  # dense FFNs: no near-tie routing flips (see test_attnres_fused)


def _batch(cfg, seed=3, T=65):
    x = torch.randint(0, cfg.vocab_size, (2, T), generator=torch.Generator().manual_seed(seed))
    return x[:, :-1], x[:, 1:]


def _model_run(cfg, base, impl, amp=True, frac=0.0, ffn=0.0, off=0.0, x=None, y=None):
    m = MoELLM(cfg.replace(conv_impl=impl))
    m.load_state_dict(base.state_dict())
    m.set_checkpointing(frac, ffn_frac=ffn, offload_frac=off)
    if m.offloader is not None:
        m.offloader.min_bytes = 0  # tiny model: offload everything
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        loss, _ = m(x, y)
    loss.backward()
    if m.offloader is not None:
        assert m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0, m.offloader.stats
    return loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None}


@fla_cpu
def test_model_close_to_current_and_no_less_accurate():
    """Tiny model (KDA short convs, Engram, attention, dense FFNs) under bf16 autocast: fla vs the current path (eager
    conv1d) - same loss up to rounding, every gradient close, and the whole gradient (all parameters together) no
    farther from the fp32 model's than the current path's (1.1x; per-tensor differences are noise)."""
    torch.manual_seed(0)
    cfg = _cfg()
    base = MoELLM(cfg)
    x, y = _batch(cfg)
    l_c, g_c = _model_run(cfg, base, "compiled", x=x, y=y)
    l_f, g_f = _model_run(cfg, base, "fla", x=x, y=y)
    l_r, g_r = _model_run(cfg, base, "compiled", amp=False, x=x, y=y)  # fp32 reference
    assert g_c.keys() == g_f.keys() == g_r.keys() == {n for n, _ in base.named_parameters()}
    keys = sorted(g_c)
    flat = {name: torch.cat([g[k].flatten() for k in keys]) for name, g in (("c", g_c), ("f", g_f), ("r", g_r))}
    glob_f, glob_c = rel_rms(flat["f"], flat["r"]), rel_rms(flat["c"], flat["r"])
    conv = [k for k in keys if "_conv." in k]
    print(f"\nloss current {l_c.item():.6f} fla {l_f.item():.6f} fp32 {l_r.item():.6f}; whole gradient vs fp32: fla "
          f"{glob_f:.3e} current {glob_c:.3e}; conv weights: " + " ".join(
              f"{k.split('.')[1]}.{k.split('.')[3]} {rel_rms(g_f[k], g_r[k]):.2e}/{rel_rms(g_c[k], g_r[k]):.2e}"
              for k in conv))
    assert abs(l_f.item() - l_c.item()) < 2e-3 * abs(l_c.item())
    assert glob_f <= 1.1 * glob_c + 1e-4, (glob_f, glob_c)
    for k in keys:
        assert g_f[k].dtype == g_c[k].dtype and rel_rms(g_f[k], g_c[k]) < 5e-2, (k, rel_rms(g_f[k], g_c[k]))


@fla_cpu
def test_model_recompute_offload_and_rerun_bit_identical():
    """With conv_impl="fla": whole-layer recompute (some / all layers), FFN-only recompute and activation offloading
    give exactly the loss and gradients of no recompute, and so does a rerun (bf16 autocast)."""
    torch.manual_seed(1)
    cfg = _cfg()
    base = MoELLM(cfg)
    x, y = _batch(cfg, seed=5)
    l0, g0 = _model_run(cfg, base, "fla", x=x, y=y)
    for kw in (dict(), dict(frac=1.0), dict(frac=0.5, ffn=0.5), dict(frac=1.0, off=0.5)):
        l1, g1 = _model_run(cfg, base, "fla", x=x, y=y, **kw)
        assert torch.equal(l1, l0), kw
        assert g1.keys() == g0.keys() and all(torch.equal(g1[k], g0[k]) for k in g0), kw


@fla_cpu
def test_model_eval_and_no_grad():
    torch.manual_seed(0)
    cfg = _cfg()
    m_c = MoELLM(cfg).eval()
    m_f = MoELLM(cfg.replace(conv_impl="fla")).eval()
    m_f.load_state_dict(m_c.state_dict())
    x, _ = _batch(cfg)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        a, b = m_c(x), m_f(x)
    assert rel_rms(b, a) < 2e-2, rel_rms(b, a)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(m_f(x).detach(), b)  # the same forward kernels with gradients on


@fla_cpu
def test_training_steps_with_persistent_copies():
    """Optimizer steps (Muon + AdamW, 2 micro-batches each, clipping) under bf16 autocast: with conv_impl="fla" the
    persistent bf16 weight copies (offload_master) train bit-identically to the plain bundle (the conv weights are not
    among the copied weights: their gradients accumulate into .grad through cast_weight), and the main run's settings
    (offload_master + fused_grad_acc) track the current path within rounding."""
    torch.manual_seed(0)
    cfg = _cfg()
    base = MoELLM(cfg)
    runs = {}
    try:
        for name, impl, master, fused in (("fla_plain", "fla", False, False), ("fla_master", "fla", True, False),
                                          ("fla_main", "fla", True, True), ("cur_main", "compiled", True, True)):
            m = MoELLM(cfg.replace(conv_impl=impl))
            m.load_state_dict(base.state_dict())
            opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16, offload_master=master,
                                  fused_grad_acc=fused)
            opt.begin()
            ces = []
            for s in range(3):
                for k in range(2):
                    xb, yb = _batch(cfg, seed=1000 * s + k)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        loss, info = m(xb, yb)
                    (loss / 2).backward()
                    ces.append(float(info["ce"]))
                ces.append(float(opt.clip_grad_norm(0.05)))
                opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
                opt.zero_grad()
                m.update_router_bias()
            runs[name] = ces
            common.clear_cached_copies()
    finally:
        common.clear_cached_copies()
    assert runs["fla_plain"] == runs["fla_master"]
    for a, b in zip(runs["fla_main"], runs["cur_main"]):
        assert abs(a - b) < 5e-3 * abs(b), (runs["fla_main"], runs["cur_main"])


# ----------------------------------------------------------------------------- train.py: --override and resume
def _bins(root):
    import numpy as np
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


def _train(out, data, impl, steps=8):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", TRITON_INTERPRET="1")
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--preset", "tiny",
           "--override", json.dumps({"engram_offload": True, "conv_impl": impl}),
           "--data", data, "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--out", out,
           "--seq-len", "32", "--micro-bsz", "2", "--global-tokens", "128", "--steps", str(steps), "--lr", "3e-3",
           "--warmup", "2", "--log-every", "1", "--eval-every", "0", "--ckpt-every", "4", "--grad-ckpt"]
    r = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"ce"' in l], r.stdout


@fla_cpu
def test_train_py_switch_to_fla_on_resume(tmp_path):
    """train.py --override '{"conv_impl": "fla"}' resumes a checkpoint written by the compiled path (fp32 on CPU): the
    switch is a restart, and the losses after it stay within rounding of the run that kept the compiled op."""
    data = str(tmp_path / "bins")
    _bins(data)
    full, _ = _train(str(tmp_path / "full"), data, "compiled")
    part = str(tmp_path / "part")
    _train(part, data, "compiled")
    os.remove(os.path.join(part, "ckpt_0000008.pt"))  # resume from step 4 with fla's kernels
    open(os.path.join(part, "log.jsonl"), "w").close()
    resumed, stdout = _train(part, data, "fla")
    assert "KDA short convs: conv_impl fla" in stdout, stdout
    assert [r["step"] for r in resumed] == list(range(4, 8))
    for a, b in zip(full[4:], resumed):
        assert abs(a["ce"] - b["ce"]) < 1e-3 * abs(a["ce"]), (a, b)


# ----------------------------------------------------------------------------- GPU
_GPU_IMPLS: dict = {}  # impl -> None (works) or the error of a small CUDA forward + backward


def _gpu_probe(impl):
    if impl not in _GPU_IMPLS:
        try:
            x, w, dy = make_case(B=1, T=200, D=96, device="cuda")
            out = run_op(impl, x, w, dy)
            ref = run_op("eager", x, w, dy)
            assert all(rel_rms(o, r) < 2e-2 for o, r in zip(out, ref)), [rel_rms(o, r) for o, r in zip(out, ref)]
            _GPU_IMPLS[impl] = None
        except Exception as e:  # e.g. fla's Gluon kernels do not compile for this GPU / Triton
            _GPU_IMPLS[impl] = e
    return _GPU_IMPLS[impl]


def _gpu_impls():
    """The fla paths that run here ("fla_gluon" needs triton.experimental.gluon and fla's Gluon backend to compile for
    this GPU; test_gpu_fla_paths_run reports each)."""
    return [impl for impl in ("fla", "fla_gluon") if _gpu_probe(impl) is None]


@gpu
@pytest.mark.parametrize("impl", ["fla", "fla_gluon"])
def test_gpu_fla_paths_run(impl):
    """Each fla path runs a small forward + backward on this GPU and matches eager conv1d within bf16 rounding. "fla"
    (the Triton kernels) must; "fla_gluon" is skipped with the reason if fla's Gluon kernels do not run here (the
    other GPU tests then cover "fla" only)."""
    err = _gpu_probe(impl)
    if err is not None and impl == "fla_gluon":
        pytest.skip(f"fla_gluon does not run here: {err!r}")
    assert err is None, err


def _main_case(seed=0, w_std=0.3):
    return make_case(B=2, T=4096, D=1024, w_std=w_std, seed=seed, device="cuda")


@gpu
def test_gpu_main_shape_accuracy_and_preactivation():
    """KDA short conv at main size ([2, 4096, 1024] bf16, fp32 weight). Errors vs fp64: both fla paths no less
    accurate than the torch.compile'd op (the running path) and eager conv1d - forward and dx more accurate, dw within
    5% (one bf16 rounding at the end everywhere); the recomputed bf16 pre-activation is bit-identical to the compiled
    op's; dw arrives in .grad as a bf16 value."""
    impls = _gpu_impls()
    assert "fla" in impls, "fla's causal_conv1d kernels do not load"
    common.set_fuse(True)
    try:
        for seed, w_std in ((0, 0.3), (1, 0.02), (2, 1.0)):
            x, w, dy = _main_case(seed, w_std)
            w_b = w[:, 0].to(torch.bfloat16).contiguous()
            ref = ref64(x, w_b, dy)
            outs = {impl: run_op(impl, x, w, dy) for impl in ("eager", "compiled", *impls)}
            e = {impl: errors(out, ref) for impl, out in outs.items()}
            print(f"\nw_std {w_std}: rel. RMS error vs fp64  " + "  ".join(
                f"{i}: y {v['y']:.3e} dx {v['dx']:.3e} dw {v['dw']:.3e}" for i, v in e.items()))
            for impl in impls:
                assert torch.equal(outs[impl][2], outs[impl][2].to(torch.bfloat16).float()), impl
                assert e[impl]["y"] < e["compiled"]["y"] and e[impl]["dx"] < e["compiled"]["dx"], (impl, e)
                assert e[impl]["dw"] <= 1.05 * e["compiled"]["dw"] + 1e-7, (impl, e)
                assert rel_max(outs[impl][0], ref[0]) < 1e-2 and rel_max(outs[impl][1], ref[1]) < 2e-2, impl
            from fla.modules.causal_conv1d import causal_conv1d
            c, _ = causal_conv1d(x=x, weight=w_b, activation=None, backend="triton")
            assert torch.equal(c, preact_bf16(x, w_b))
    finally:
        common.set_fuse(False)
    assert not common._FAILED, common._FAILED


def _kda_layer(impl, seed=0):
    from moe_llm.layers.kda import KDA
    torch.manual_seed(seed)
    return KDA(main_2b().replace(conv_impl=impl), 1).cuda()


def _kda_step(layer, x0, dy):
    layer.zero_grad(set_to_none=True)
    x = x0.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x)
    y.backward(dy)
    return [y.detach(), x.grad] + [p.grad for p in layer.parameters()]


def _count_sync_warnings(fn):
    """fn() under torch.cuda.set_sync_debug_mode("warn"): (result, call sites of "synchronizing CUDA operation"
    warnings), as tests/test_fuse_proj.py counts them."""
    hits, prev = [], warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchronizing CUDA operation" in str(message):
            hits.append(f"{filename}:{lineno}")
        else:
            prev(message, category, filename, lineno, file, line)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = show
        torch.cuda.set_sync_debug_mode("warn")
        try:
            out = fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return out, hits


@gpu
def test_gpu_deterministic_and_no_host_sync():
    """The conv op at main size with each fla path: forward + backward and the no_grad forward make no CPU-GPU sync
    (torch.cuda.set_sync_debug_mode("error"); kernels compiled / autotuned in a warm-up call first) and a rerun is
    bit-identical. A main_2b KDA layer (fusion on, bf16 autocast) with each fla path: bit-identical on a rerun, and no
    more sync warnings than with the compiled conv (the layer's other parts are not under test here)."""
    impls = _gpu_impls()
    assert "fla" in impls
    old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    common.set_fuse(True)
    try:
        xc, wc, dyc = _main_case(seed=5)
        g = torch.Generator(device="cuda").manual_seed(0)
        x0 = torch.randn(2, 4096, 1024, device="cuda", generator=g).to(torch.bfloat16)
        dy = torch.randn(2, 4096, 1024, device="cuda", generator=g).to(torch.bfloat16)
        layer_c = _kda_layer("compiled")
        _kda_step(layer_c, x0, dy)
        torch.cuda.synchronize()
        _, hits_c = _count_sync_warnings(lambda: _kda_step(layer_c, x0, dy))
        for impl in impls:
            ref_op = run_op(impl, xc, wc, dyc)
            with torch.no_grad():
                fla_conv_silu(xc, wc, impl)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                op = run_op(impl, xc, wc, dyc)
                with torch.no_grad():
                    y_ng = fla_conv_silu(xc, wc, impl)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            assert all(torch.equal(a, b) for a, b in zip(ref_op, op)) and torch.equal(y_ng, ref_op[0]), impl
            layer = _kda_layer(impl)
            ref = _kda_step(layer, x0, dy)
            torch.cuda.synchronize()
            again, hits = _count_sync_warnings(lambda: _kda_step(layer, x0, dy))
            assert all(torch.equal(a, b) for a, b in zip(ref, again)), impl
            print(f"\n{impl}: KDA layer fwd+bwd sync warnings {len(hits)} (compiled conv: {len(hits_c)}) {hits[:4]}")
            assert len(hits) <= len(hits_c), (impl, hits, hits_c)
    finally:
        common.set_fuse(False)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old


@gpu
def test_gpu_kda_layer_memory():
    """A main_2b KDA layer under bf16 autocast with fusion on: memory held from the forward until the backward (saved
    activations + output) and the peak of forward + backward, per conv_impl. Neither may exceed the compiled path's by
    more than 1 MiB. Expected (CPU: test_saved_bytes_vs_compiled): the fla paths save the bf16 conv inputs (3 x 16 MiB)
    where the compiled graph saves fp32 padded copies (3 x 32 MiB), i.e. ~48 MiB less held per layer (~0.84 GiB over
    main_2b's 18 KDA layers). Prints the numbers and that projection."""
    impls = _gpu_impls()
    common.set_fuse(True)
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        x0 = torch.randn(2, 4096, 1024, device="cuda", generator=g).to(torch.bfloat16)
        dy = torch.randn(2, 4096, 1024, device="cuda", generator=g).to(torch.bfloat16)
        res = {}
        for impl in ("compiled", *impls):
            layer = _kda_layer(impl)
            _kda_step(layer, x0, dy)  # compile, autotune, allocate the gradients
            layer.zero_grad(set_to_none=False)
            x = x0.clone().requires_grad_()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = layer(x)
            torch.cuda.synchronize()
            held = torch.cuda.memory_allocated() - base
            y.backward(dy)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            res[impl] = (held, peak)
            del layer, x, y
            torch.cuda.empty_cache()
        mib = 2 ** 20
        for impl, (held, peak) in res.items():
            d_h, d_p = held - res["compiled"][0], peak - res["compiled"][1]
            print(f"\n{impl:9s} held until backward {held / mib:8.1f} MiB ({d_h / mib:+7.1f}; x 18 KDA layers "
                  f"{18 * d_h / 2 ** 30:+.3f} GiB)  peak fwd+bwd {peak / mib:8.1f} MiB ({d_p / mib:+7.1f})")
        for impl in impls:
            assert res[impl][0] <= res["compiled"][0] + mib, (impl, res)
            assert res[impl][1] <= res["compiled"][1] + mib, (impl, res)
    finally:
        common.set_fuse(False)


@gpu
def test_gpu_small_model_matches_compiled():
    """Small dense model on the real GPU path (bf16 autocast, bf16 residual, fla KDA, partial recompute, fusion on):
    each fla conv path vs the compiled one - same loss up to bf16 rounding, the whole gradient no farther from an fp32
    reference (same weights and inputs, autocast off) than the compiled path's (1.1x), every tensor within 5%."""
    impls = _gpu_impls()
    torch.manual_seed(0)
    cfg = _cfg(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True, residual="block_attnres")
    base = MoELLM(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, 257), device="cuda")

    def run(impl, amp):
        m = MoELLM(cfg.replace(conv_impl=impl, residual_bf16=amp)).cuda()
        m.load_state_dict(base.state_dict())
        m.set_checkpointing(0.5)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, _ = m(x[:, :-1], x[:, 1:])
        loss.backward()
        return float(loss), {n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None}

    common.set_fuse(True)
    try:
        l_c, g_c = run("compiled", True)
        l_r, g_r = run("compiled", False)  # fp32 reference
        keys = sorted(g_c)
        flat_c, flat_r = torch.cat([g_c[k].flatten() for k in keys]), torch.cat([g_r[k].flatten() for k in keys])
        glob_c = rel_rms(flat_c, flat_r)
        for impl in impls:
            l_f, g_f = run(impl, True)
            assert g_f.keys() == g_c.keys() == g_r.keys()
            glob_f = rel_rms(torch.cat([g_f[k].flatten() for k in keys]), flat_r)
            print(f"\n{impl}: loss {l_f:.6f} (compiled {l_c:.6f}, fp32 {l_r:.6f}); whole gradient vs fp32 {glob_f:.3e} "
                  f"(compiled {glob_c:.3e})")
            assert abs(l_f - l_c) < 2e-3 * l_c, (impl, l_f, l_c)
            assert glob_f <= 1.1 * glob_c + 1e-4, (impl, glob_f, glob_c)
            bad = [(k, rel_rms(g_f[k], g_c[k])) for k in keys if rel_rms(g_f[k], g_c[k]) >= 0.05]
            assert not bad, (impl, bad)
    finally:
        common.set_fuse(False)


@gpu
def test_gpu_timing():
    """Forward and forward + backward of one KDA short conv at main size and of a main_2b KDA layer, per conv_impl
    (CUDA events, after warm-up). Prints the numbers and the projection to one optimizer step (54 convs / 18 layers x 64
    micro-batches). Informational: isolated timings have misled before; scripts/ab_conv.py measures the real step."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_conv
    print()
    bench_conv.main(["--reps", "10", "--warmup-reps", "3"])
