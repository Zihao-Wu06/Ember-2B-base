"""Fused MoE fan-in (ModelConfig.moe_shared_fused; layers/moe.py MoE._forward_fan, layers/moe_triton.py MoEFan,
RoutedExpertsFan and the combine kernel's add / router flags).

Checked on CPU (the Triton kernels in the interpreter, as tests/test_moe_triton.py):
* the combine kernel: ROUND_FIRST equals the old separate bf16 / fp32 adds bit for bit; the fp32 fan-in sum with the
  router part is within one final rounding of the exact (fp64) sum and closer to it than the old composition;
* the MoE layer, bf16 and fp32 inputs, with and without a shared expert, expert recompute and checkpointing, plain and
  with persistent copies + fused gradient accumulation (offload_master / fused_grad_acc): "exact" is bit-identical in
  the output and every gradient; "router" / "router_lean" are bit-identical in the output and every weight gradient
  (except router_lean's router weight) and their input gradient is closer to the exact sum of the same parts;
  router_lean's router weight gradient is at fp32-GEMM accuracy;
* the [N, d] adds, the router's fp32 [N, d] dx GEMM output and its cast are gone (dispatch-mode op counts), and the
  tensors kept for the backward are the same (router_lean: minus the router's fp32 copy of the input);
* the whole tiny model (KDA, attention, Engram, AttnRes, dense FFN) with an offload_master + fused_grad_acc bundle over
  3 micro-batches, with the MoE inputs in bf16 as in the main run: "exact" bit-identical in losses and every
  gradient; the other modes bit-identical in the losses, rounding-level in the gradients; recompute and activation
  offloading bit-identical within each mode; train.py's loop (clipping, bf16 Muon) tracks the old path.
GPU tests at the end (skipped without CUDA): main sizes (N = 8192, d = 1024, E = 32, k = 4, I = 512, shared 768) with
persistent copies and fused accumulation against the old path, the exact fan-in sum and a fp32 layer; no host syncs;
memory kept for the backward and peak memory; per-layer timing (printed only: in-pipeline timing is
scripts/bench_shared_fused.py)."""
import copy
import os

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported (see tests/test_moe_triton.py)

from moe_llm.config import tiny  # noqa: E402
from moe_llm.layers import common  # noqa: E402
from moe_llm.layers import moe_triton as mt  # noqa: E402
from moe_llm.layers.moe import MoE  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402
from moe_llm.optim import OptimizerBundle, wsd_lr  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
BF16 = torch.bfloat16
MODES = ("exact", "router", "router_lean")


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


@pytest.fixture
def big_tiles():
    """Interpreter only: 32x32x32 expert GEMM tiles (faster on CPU; every compared run uses the same tiles)."""
    if DEV == "cpu":
        mt.set_configs(32, 32, 32, 4, 1)
    yield
    if DEV == "cpu":
        mt.reset_configs()


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def attach_copies(params, fused):
    """Persistent bf16 copies as OptimizerBundle(offload_master=True).begin() makes them (tests/test_grad_acc.py): the
    hook path, or a GradSink (fused_grad_acc). Returns the fp32 gradient buffers {id(param): buffer}."""
    store = {}
    for p in params:
        c = p.detach().to(BF16).requires_grad_(True)

        def hook(c, pid=id(p)):
            if c.grad is None:
                return
            if store.get(pid) is None:
                store[pid] = c.grad.float()
            else:
                store[pid].add_(c.grad)
            c.grad = None
        c.register_post_accumulate_grad_hook(hook)
        common.set_cached_copy(p, c, common.GradSink(store, id(p), p.shape, c.device) if fused else None)
    return store


def cast_params(moe):
    return [moe.w_gate, moe.w_up, moe.w_down] + ([lin.weight for lin in (moe.shared.gate, moe.shared.up,
                                                                           moe.shared.down)] if moe.shared else [])


def layer_run(moe, mode, x, dyt, copies=None, recompute=False, ckpt=False, n_micro=1):
    """n_micro forward + backward passes of the MoE layer under bf16 autocast in this fan mode: [y, dx, aux] of the
    last pass and every parameter's gradient (the fp32 buffer of a cast parameter with copies: None / "hook" /
    "fused")."""
    moe.fan, moe.recompute_experts = mode, recompute
    moe.zero_grad(set_to_none=True)
    store = attach_copies(cast_params(moe), copies == "fused") if copies else None
    for _ in range(n_micro):
        xi = x.detach().requires_grad_()
        with torch.autocast(DEV, dtype=BF16):
            y, aux = checkpoint(moe, xi, use_reentrant=False) if ckpt else moe(xi)
        (y.float().mul(dyt).sum() + aux).backward()
    grads = {}
    for n, p in moe.named_parameters():
        grads[n] = store.get(id(p)) if store is not None and any(p is q for q in cast_params(moe)) else p.grad
    common.clear_cached_copies()
    return [y.detach(), xi.grad, aux.detach()], grads


def make_layer(seed=0, n_shared=1, **kw):
    torch.manual_seed(seed)
    cfg = tiny(moe_impl="triton", n_shared=n_shared, **kw)
    moe = MoE(cfg, 1).to(DEV).train()
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(2, 64, cfg.d_model, generator=g)
    dyt = torch.randn(2, 64, cfg.d_model, generator=g).to(DEV)
    return cfg, moe, x, dyt


class FanParts:
    """Records the parts MoEFan's backward sums (the args of moe_triton._fan_in) and their exact (fp64) sum."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = mt._fan_in

        def rec(g_xr, g_xg, g_xu, g_xq, g_logits, w_r, dims, out_dtype, exact):
            self.calls.append([None if t is None else t.detach().clone() for t in (g_xr, g_xg, g_xu, g_xq, g_logits)]
                              + [None if w_r is None else w_r.detach().clone()])
            return real(g_xr, g_xg, g_xu, g_xq, g_logits, w_r, dims, out_dtype, exact)
        monkeypatch.setattr(mt, "_fan_in", rec)

    def exact_sum(self, i=-1):
        g_xr, g_xg, g_xu, g_xq, g_logits, w_r = self.calls[i]
        s = g_xr.double().sum(1)
        for t in (g_xg, g_xu, g_xq):
            if t is not None:
                s = s + t.double()
        if g_logits is not None:
            s = s + g_logits.double() @ w_r.double()
        return s


# ------------------------------------------------------------------------------------------------- the kernel
def _parts(N=96, D=64, K=4, E=8, dtype=BF16, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(N * K, D, generator=g).to(DEV, dtype)
    w = (torch.rand(N, K, generator=g) + 0.1).to(DEV)
    adds = [torch.randn(N, D, generator=g).to(DEV, dtype) * s for s in (1.0, 0.5, 0.25)]
    dl = (torch.randn(N, E, generator=g) * 0.3).to(DEV)
    wr = (torch.randn(E, D, generator=g) * 0.2).to(DEV)
    return src, w, adds, dl, wr


@pytest.mark.parametrize("dtype", [BF16, torch.float32])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n_add", [1, 2, 3, "a3_only"])
def test_combine_round_first_is_the_old_adds(dtype, weighted, n_add):
    """ROUND_FIRST: the plain combine's output, then autograd's adds in the old order ((a1 + a2) + k-sum) + a3, each
    an add of two tensors in their dtype: bit for bit."""
    src, w, (a1, a2, a3), _, _ = _parts(dtype=dtype)
    N, K = w.shape
    ww = w if weighted else None
    plain = mt._combine(src, ww, N, K, dtype)
    if n_add == "a3_only":
        got = mt._combine(src, ww, N, K, dtype, a3=a3, round_first=True)
        ref = plain + a3
    else:
        kw = dict(a1=a1, a2=a2 if n_add >= 2 else None, a3=a3 if n_add >= 3 else None)
        got = mt._combine(src, ww, N, K, dtype, round_first=True, **kw)
        ref = (a1 + a2 if n_add >= 2 else a1) + plain
        if n_add >= 3:
            ref = ref + a3
    assert got.dtype == dtype and torch.equal(got, ref)
    assert torch.equal(mt._combine(src, ww, N, K, dtype, round_first=True), plain)  # no adds: the plain combine


def test_combine_fp32_sum_with_router_part():
    """Without ROUND_FIRST: k-sum + a1 + a2 + dl @ wr in fp32, rounded once: within one bf16 rounding of the exact
    (fp64) sum (equal to it rounded for nearly every element) and closer to it than the old composition (each part
    rounded to bf16 and added in bf16, the router part in fp32 first)."""
    src, _, (a1, a2, _), dl, wr = _parts(N=256)
    N, K = 256, 4
    got = mt._combine(src, None, N, K, BF16, a1=a1, a2=a2, dl=dl, wr=wr)
    exact = src.double().view(N, K, -1).sum(1) + a1.double() + a2.double() + dl.double() @ wr.double()
    old = ((a1 + a2) + mt._combine(src, None, N, K, BF16)) + (dl @ wr).to(BF16)
    assert (got == exact.to(BF16)).float().mean() > 0.99
    ulp = torch.finfo(BF16).eps * exact.abs().clamp_min(1e-30)
    assert ((got.double() - exact).abs() <= ulp).all()
    e_new, e_old, e_round = rel_rms(got, exact), rel_rms(old, exact), rel_rms(exact.to(BF16), exact)
    assert e_new <= 1.02 * e_round and e_new < 0.75 * e_old, (e_new, e_old, e_round)


# ------------------------------------------------------------------------------------------------- the MoE layer
@pytest.mark.parametrize("dtype", [BF16, torch.float32])
@pytest.mark.parametrize("n_shared", [1, 0])
@pytest.mark.parametrize("copies", [None, "hook", "fused"])
def test_layer_exact_is_bit_identical(dtype, n_shared, copies, big_tiles):
    """moe_shared_fused="exact": output, aux, input gradient and every parameter gradient (fp32 buffers with persistent
    copies) bit-identical to the old triton path; also with the expert recompute and under checkpointing; 2 micro-batches
    when accumulating into buffers."""
    cfg, moe, x, dyt = make_layer(n_shared=n_shared)
    x = x.to(DEV, dtype)
    n = 2 if copies else 1
    for kw in ({}, {"recompute": True}, {"ckpt": True}):
        ref = layer_run(moe, "", x, dyt, copies, n_micro=n, **kw)
        got = layer_run(moe, "exact", x, dyt, copies, n_micro=n, **kw)
        for a, b in zip(ref[0], got[0]):
            assert a.dtype == b.dtype and torch.equal(a, b), kw
        assert ref[1].keys() == got[1].keys()
        for k in ref[1]:
            assert torch.equal(ref[1][k], got[1][k]), (kw, k)


@pytest.mark.parametrize("mode", ["router", "router_lean"])
@pytest.mark.parametrize("n_shared", [1, 0])
def test_layer_router_modes(mode, n_shared, monkeypatch, big_tiles):
    """bf16 input (the main run's): output, aux and every weight gradient bit-identical (router_lean: except the router
    weight); the input gradient within one bf16 rounding of the exact sum of the parts the fan node receives (the same
    parts the old path sums) and closer to it than the old input gradient; router_lean's router weight gradient at
    fp32-GEMM accuracy (CPU: the old one is a fp32 GEMM too; on the GPU it is TF32, see the GPU test)."""
    cfg, moe, x, dyt = make_layer(n_shared=n_shared)
    x = x.to(DEV, BF16)
    ref = layer_run(moe, "", x, dyt, "fused")
    parts = FanParts(monkeypatch)
    got = layer_run(moe, mode, x, dyt, "fused")
    assert len(parts.calls) == 1
    for a, b in ((ref[0][0], got[0][0]), (ref[0][2], got[0][2])):
        assert torch.equal(a, b)
    for k in ref[1]:
        if mode == "router_lean" and k == "router.weight":
            continue
        assert torch.equal(ref[1][k], got[1][k]), k
    exact = parts.exact_sum().view(got[0][1].shape)
    dx_new, dx_old = got[0][1], ref[0][1]
    e_new, e_old, e_round = rel_rms(dx_new, exact), rel_rms(dx_old, exact), rel_rms(exact.to(BF16), exact)
    print(f"{mode} shared={n_shared}: input gradient vs exact sum: fused {e_new:.3e}, old {e_old:.3e}, "
          f"exact rounded {e_round:.3e}")
    assert e_new <= 1.02 * e_round and e_new < 0.9 * e_old
    assert not torch.equal(dx_new, dx_old) and rel_rms(dx_new, dx_old) < 1e-2
    if mode == "router_lean":
        g_logits = parts.calls[0][4]
        truth = g_logits.double().t() @ x.reshape(-1, cfg.d_model).double()
        e_lean, e_old_w = rel_rms(got[1]["router.weight"], truth), rel_rms(ref[1]["router.weight"], truth)
        print(f"  router weight gradient vs exact: lean {e_lean:.3e}, old {e_old_w:.3e}")
        assert e_lean < 1e-6 and e_old_w < 1e-6


def test_layer_router_modes_fp32_input_and_recompute(big_tiles):
    """fp32 input (fp32 residual stream): lean has nothing to drop (the input is its own fp32 copy), so router_lean
    equals router bit for bit; the output and weight gradients equal the old path's; recompute and checkpointing are
    bit-identical within each mode."""
    cfg, moe, x, dyt = make_layer()
    x = x.to(DEV)
    ref = layer_run(moe, "", x, dyt, "fused")
    runs = {m: layer_run(moe, m, x, dyt, "fused") for m in ("router", "router_lean")}
    for m, got in runs.items():
        assert torch.equal(ref[0][0], got[0][0]) and rel_rms(got[0][1], ref[0][1]) < 1e-5
        for k in ref[1]:
            assert torch.equal(ref[1][k], got[1][k]), (m, k)
    for a, b in zip(runs["router"][0], runs["router_lean"][0]):
        assert torch.equal(a, b)
    for m in ("router", "router_lean"):
        base = layer_run(moe, m, x.to(BF16), dyt, "fused")
        for kw in ({"recompute": True}, {"ckpt": True}):
            got = layer_run(moe, m, x.to(BF16), dyt, "fused", **kw)
            for a, b in zip(base[0], got[0]):
                assert torch.equal(a, b), (m, kw)
            for k in base[1]:
                assert torch.equal(base[1][k], got[1][k]), (m, kw, k)


class OpLog(TorchDispatchMode):
    """(op name, output shape, output dtype, input dtypes) of every dispatched op with a tensor output."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.ops.append((func.overloadpacket.__name__, tuple(out.shape), out.dtype,
                             tuple(a.dtype for a in args if isinstance(a, torch.Tensor))))
        return out


def test_layer_fan_in_ops_removed():
    """[N, d]-sized work per layer and pass (bf16 input under autocast): the old path has 1 forward add, 3 backward adds
    (autograd's fan-in sum), the router's fp32 dx GEMM output and its cast to bf16; "exact" no adds (the router's GEMM
    and cast stay: they are what makes its part bit-identical); the router modes none of these at all."""
    cfg, moe, x, dyt = make_layer()
    x = x.to(DEV, BF16)
    N, D = x.shape[0] * x.shape[1], cfg.d_model
    want = {"": (1, 3, 1, 1), "exact": (0, 0, 1, 1), "router": (0, 0, 0, 0), "router_lean": (0, 0, 0, 0)}
    for mode, (f_add, b_add, b_mm32, b_cast) in want.items():
        moe.fan = mode
        xi = x.detach().requires_grad_()
        fwd, bwd = OpLog(), OpLog()
        with fwd, torch.autocast(DEV, dtype=BF16):
            y, aux = moe(xi)
        loss = y.float().mul(dyt).sum() + aux
        with bwd:
            loss.backward()

        def count(log, name, dtype, ins=None):
            return sum(1 for n, s, dt, i in log.ops if n == name and s == (N, D) and dt == dtype
                       and (ins is None or i == ins))
        got = (count(fwd, "add", BF16), count(bwd, "add", BF16) + count(bwd, "add_", BF16),
               count(bwd, "mm", torch.float32), count(bwd, "_to_copy", BF16, (torch.float32,)))
        assert got == (f_add, b_add, b_mm32, b_cast), (mode, got)


def _saved_bytes(moe, mode, x):
    """Bytes of the distinct non-parameter storages the layer's forward keeps for its backward (bf16 input)."""
    params = {p.untyped_storage().data_ptr() for p in moe.parameters()}
    seen = {}

    def pack(t):
        st = t.untyped_storage()
        if st.data_ptr() not in params:
            seen[st.data_ptr()] = st.nbytes()
        return t
    moe.fan = mode
    xi = x.detach().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), torch.autocast(DEV, dtype=BF16):
        y, aux = moe(xi)
    (y.float().mul(1.0).sum() + aux).backward()
    return sum(seen.values())


def test_saved_tensors_same_or_less():
    """The tensors kept for the backward: the same storages for "exact" and "router" as for the old path; router_lean
    drops exactly the router's fp32 copy of the input (N * d * 4 bytes)."""
    cfg, moe, x, _ = make_layer()
    x = x.to(DEV, BF16)
    N, D = x.shape[0] * x.shape[1], cfg.d_model
    old = _saved_bytes(moe, "", x)
    assert _saved_bytes(moe, "exact", x) == old
    assert _saved_bytes(moe, "router", x) == old
    assert _saved_bytes(moe, "router_lean", x) == old - N * D * 4


def test_no_grad_eval_and_other_impls():
    """Forward without gradients and in eval mode: the same output in every mode; a layer switched to another impl
    ignores the option (old path); the config is validated."""
    cfg, moe, x, _ = make_layer()
    x = x.to(DEV, BF16)
    for train in (True, False):
        moe.train(train)
        outs = []
        for mode in ("",) + MODES:
            moe.fan = mode
            with torch.no_grad(), torch.autocast(DEV, dtype=BF16):
                outs.append(moe(x)[0])
        assert all(torch.equal(outs[0], o) for o in outs[1:])
    moe.train()
    moe.impl = "bmm"
    res = []
    for mode in ("", "router"):
        moe.fan = mode
        with torch.no_grad(), torch.autocast(DEV, dtype=BF16):
            res.append(moe(x)[0])
    assert torch.equal(res[0], res[1])
    with pytest.raises(ValueError):
        MoE(tiny(moe_impl="triton", moe_shared_fused="yes"), 1)
    with pytest.raises(ValueError):
        MoE(tiny(moe_impl="bmm", moe_shared_fused="exact"), 1)


# ------------------------------------------------------------------------------------------------- whole model
def _batch(cfg, s, k):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(1000 * s + k))
    return x[:, :-1], x[:, 1:]


def _bf16_moe_inputs(m):
    """The main run's MoE inputs are bf16 (bf16 residual stream; CPU runs keep fp32): cast them at the layer."""
    return [moe.register_forward_pre_hook(lambda mod, args: (args[0].to(BF16),)) for moe in m.moe_layers()]


def _model(mode, base, frac=0.0, bf16_moe=True, **ck):
    m = copy.deepcopy(base)
    for moe in m.moe_layers():
        moe.fan = mode
    m.set_checkpointing(frac, **ck)
    if m.offloader is not None:
        m.offloader.min_bytes = 0
    if bf16_moe:
        _bf16_moe_inputs(m)
    return m


def bundle(model, fused=True):
    opt = OptimizerBundle(model, lr=3e-3, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=fused)
    opt.begin()
    return opt


def grads_of(opt, model):
    return {n: (opt._gbuf.get(id(p), p.grad) if id(p) in opt._cast else p.grad) for n, p in model.named_parameters()}


def _micro(m, batches):
    losses = []
    for x, y in batches:
        with torch.autocast(DEV, dtype=BF16):
            loss, _ = m(x.to(DEV), y.to(DEV))
        (loss / len(batches)).backward()
        losses.append(loss.detach())
    return losses


@pytest.mark.parametrize("residual,frac", [("standard", 0.0), ("block_attnres", 0.5)])
def test_model_modes(residual, frac, big_tiles):
    """Whole tiny model, offload_master + fused_grad_acc bundle, 3 micro-batches, bf16 MoE inputs: "exact" bit-identical
    in every loss and gradient; "router" / "router_lean" bit-identical in every loss (the forward is unchanged) and in
    the gradients of the last MoE layer's weights (they see the unchanged upstream gradient; router_lean: except its
    router), every other gradient within rounding level of the old path's."""
    torch.manual_seed(0)
    cfg = tiny(residual=residual, moe_impl="triton")
    base = MoELLM(cfg).to(DEV)
    batches = [_batch(cfg, 0, k) for k in range(3)]
    res = {}
    for mode in ("",) + MODES:
        m = _model(mode, base, frac)
        opt = bundle(m)
        res[mode] = (_micro(m, batches), grads_of(opt, m))
    l0, g0 = res[""]
    last = f"blocks.{cfg.n_layers - 1}.ffn."
    for mode in MODES:
        l1, g1 = res[mode]
        assert all(torch.equal(a, b) for a, b in zip(l0, l1)), mode
        assert g0.keys() == g1.keys()
        changed = 0
        for n in g0:
            if g0[n] is None:
                assert g1[n] is None, (mode, n)
                continue
            if mode == "exact" or (n.startswith(last) and not (mode == "router_lean" and n.endswith("router.weight"))):
                assert torch.equal(g0[n], g1[n]), (mode, n)
            else:
                assert rel_rms(g1[n], g0[n]) < 2e-2, (mode, n, rel_rms(g1[n], g0[n]))
                changed += not torch.equal(g0[n], g1[n])
        assert mode == "exact" or changed > 0, mode


def test_model_recompute_and_offload_bit_identical(big_tiles):
    """Within each mode: whole-layer recompute + expert recompute, FFN-only recompute and activation offloading (every
    saved tensor through host memory) give the same losses and gradients bit for bit as no recompute."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", moe_impl="triton")
    base = MoELLM(cfg).to(DEV)
    batches = [_batch(cfg, 1, k) for k in range(2)]
    for mode in MODES:
        out = []
        for frac, ck in [(0.0, {}), (0.5, {}), (1.0, {"ffn_frac": 0.5}), (1.0, {"offload_frac": 0.5})]:
            m = _model(mode, base, frac, **ck)
            opt = bundle(m)
            losses = _micro(m, batches)
            if m.offloader is not None:
                assert m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0
            out.append((losses, grads_of(opt, m)))
        l_ref, g_ref = out[0]
        for losses, g in out[1:]:
            assert all(torch.equal(a, b) for a, b in zip(losses, l_ref)), mode
            for n in g_ref:
                assert (g[n] is None and g_ref[n] is None) or torch.equal(g[n], g_ref[n]), (mode, n)


def _train(m, opt, cfg, steps, micro=2, clip=0.05):
    """train.py's loop under bf16 autocast (accumulate, clip, step, zero_grad, router bias): micro-batch CEs and the
    clipped norm of every step."""
    out = []
    for s in range(steps):
        for k in range(micro):
            x, y = _batch(cfg, s, k)
            with torch.autocast(DEV, dtype=BF16):
                loss, info = m(x.to(DEV), y.to(DEV))
            (loss / micro).backward()
            out.append(float(info["ce"]))
        out.append(float(opt.clip_grad_norm(clip)))
        opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
        opt.zero_grad()
        m.update_router_bias()
    return out


def test_training_steps_track_old_path(big_tiles):
    """3 training steps with an offload_master + fused_grad_acc bundle, bf16 Muon momentum, clipping active: "exact"
    gives the old path's losses and gradient norms bit for bit; the router modes the same first-step losses and later
    values within rounding-level drift."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", moe_impl="triton")
    base = MoELLM(cfg).to(DEV)
    runs = {}
    for mode in ("",) + MODES:
        m = _model(mode, base)
        runs[mode] = _train(m, bundle(m), cfg, 3)
    a = runs[""]
    assert runs["exact"] == a
    for mode in ("router", "router_lean"):
        b = runs[mode]
        assert b[:2] == a[:2] and b != a, mode
        for u, v in zip(a, b):
            assert abs(u - v) <= 2e-3 * abs(u), (mode, a, b)


# ------------------------------------------------------------------------------------------------- GPU, main sizes
def _main_layer(seed=0):
    from moe_llm.config import main_2b
    torch.manual_seed(seed)
    cfg = main_2b().replace(moe_impl="triton")
    return cfg, MoE(cfg, 1).cuda().train()


def _main_inputs(cfg, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(2, 4096, cfg.d_model, device="cuda", generator=g).to(BF16)
    dyt = torch.randn(2, 4096, cfg.d_model, device="cuda", generator=g)
    return x, dyt


@gpu
@pytest.mark.parametrize("tf32", [False, True])
def test_gpu_main_size_modes(tf32, monkeypatch):
    """Main sizes, persistent copies + fused accumulation: "exact" bit-identical in everything over 2 micro-batches; the
    router modes bit-identical in the output and every weight buffer / gradient (router_lean: except the router), their
    input gradient closer than the old one to the exact sum of the same parts. tf32=False: also closer to a fp32 layer
    (same bf16-valued weights and inputs; the router is fp32 in both, so the routing is the same); tf32=True (train.py's
    setting, the router's fp32 GEMMs in TF32): router_lean's router weight gradient closer to the exact value than the
    old TF32 one."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        cfg, moe = _main_layer()
        x, dyt = _main_inputs(cfg)
        ref = layer_run(moe, "", x, dyt, "fused", n_micro=2)
        got = layer_run(moe, "exact", x, dyt, "fused", n_micro=2)
        for a, b in zip(ref[0], got[0]):
            assert torch.equal(a, b)
        for k in ref[1]:
            assert torch.equal(ref[1][k], got[1][k]), ("exact", k)
        ref = layer_run(moe, "", x, dyt, "fused")  # one micro-batch for the accuracy comparisons
        parts = FanParts(monkeypatch)
        runs = {m: layer_run(moe, m, x, dyt, "fused") for m in ("router", "router_lean")}
        exact = parts.exact_sum(0).view(x.shape)
        x32_grad = None
        if not tf32:
            m32 = copy.deepcopy(moe)  # the plain fp32 PyTorch path (bmm) as an independent reference
            with torch.no_grad():
                for n, p in m32.named_parameters():
                    if n != "router.weight":
                        p.copy_(p.to(BF16).float())
            m32.fan, m32.impl = "", "bmm"
            x32 = x.float().requires_grad_()
            y32, aux32 = m32(x32)
            (y32.mul(dyt).sum() + aux32).backward()
            x32_grad = x32.grad
        for m, got in runs.items():
            assert torch.equal(ref[0][0], got[0][0]) and torch.equal(ref[0][2], got[0][2]), m
            for k in ref[1]:
                if not (m == "router_lean" and k == "router.weight"):
                    assert torch.equal(ref[1][k], got[1][k]), (m, k)
            e_new, e_old = rel_rms(got[0][1], exact), rel_rms(ref[0][1], exact)
            line = f"tf32={tf32} {m}: input gradient vs exact sum of the parts: fused {e_new:.3e}, old {e_old:.3e}"
            assert e_new < 0.8 * e_old, line
            if x32_grad is not None:
                f_new, f_old = rel_rms(got[0][1], x32_grad), rel_rms(ref[0][1], x32_grad)
                line += f"; vs the fp32 layer: fused {f_new:.5e}, old {f_old:.5e}"
                assert f_new <= f_old, line
            print(line)
        g_logits = parts.calls[1][4]
        truth = g_logits.double().t() @ x.reshape(-1, cfg.d_model).double()
        e_lean = rel_rms(runs["router_lean"][1]["router.weight"], truth)
        e_old = rel_rms(ref[1]["router.weight"], truth)
        print(f"tf32={tf32} router weight gradient vs exact: lean {e_lean:.3e}, old {e_old:.3e}")
        assert e_lean < 1e-5 and (e_lean < e_old if tf32 else e_lean < 4 * e_old + 1e-7)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@gpu
def test_gpu_no_host_sync():
    """Every mode, forward + backward, plain / expert recompute / torch.utils.checkpoint, with persistent copies and
    fused accumulation: no CPU-GPU synchronisation (torch.cuda.set_sync_debug_mode("error"); warm-up step first)."""
    cfg, moe = _main_layer()
    x, dyt = _main_inputs(cfg)
    for mode in MODES:
        for kw in ({}, {"recompute": True}, {"ckpt": True}):
            layer_run(moe, mode, x, dyt, "fused", **kw)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                layer_run(moe, mode, x, dyt, "fused", **kw)
            finally:
                torch.cuda.set_sync_debug_mode("default")


@gpu
def test_gpu_memory():
    """Main sizes with persistent copies + fused accumulation: memory kept from the forward to the backward (router_lean:
    32 MiB less, the router's fp32 input copy; the others: unchanged) and the peak of one forward + backward (printed).
    Peak: the fan node holds every part of the input gradient until the last one arrives, so the routed experts'
    per-assignment gradients [N, K, d] live through the shared expert's and the router's backward (the old path sums
    them into [N, d] first): exact and router at most 2 [N, d] bf16 tensors above the old path (+32 MiB measured at
    main sizes, a transient of one layer's backward, nothing kept); router_lean, which no longer keeps the router's
    fp32 input copy, at most 1 (it measured equal)."""
    cfg, moe = _main_layer()
    x, dyt = _main_inputs(cfg)
    mib = 2 ** 20
    kept, peak = {}, {}
    for mode in ("",) + MODES:
        layer_run(moe, mode, x, dyt, "fused")  # warm-up (autotuning, buffers)
        moe.fan = mode
        store = attach_copies(cast_params(moe), True)
        for p in cast_params(moe):  # the buffers exist already in a real step (after the first micro-batch)
            store[id(p)] = torch.zeros(p.shape, device="cuda")
        moe.zero_grad(set_to_none=True)
        xi = x.detach().requires_grad_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=BF16):
            y, aux = moe(xi)
        kept[mode] = (torch.cuda.memory_allocated() - base) / mib
        (y.float().mul(dyt).sum() + aux).backward()
        torch.cuda.synchronize()
        peak[mode] = (torch.cuda.max_memory_allocated() - base) / mib
        del y, aux, xi
        common.clear_cached_copies()
    print("kept for backward (MiB):", {k or "off": round(v, 1) for k, v in kept.items()})
    print("peak of fwd + bwd above the start (MiB):", {k or "off": round(v, 1) for k, v in peak.items()})
    N, D = x.shape[0] * x.shape[1], cfg.d_model
    assert abs(kept["exact"] - kept[""]) < 1 and abs(kept["router"] - kept[""]) < 1
    assert abs(kept[""] - kept["router_lean"] - N * D * 4 / mib) < 1
    for mode in MODES:
        extra = N * D * 2 * (1 if mode == "router_lean" else 2)
        assert peak[mode] <= peak[""] + extra / mib + 1, mode


@gpu
def test_gpu_layer_timing():
    """Prints forward + backward time of one MoE layer per mode (persistent copies + fused accumulation) at routing
    max/mean 1.0 and ~2. Isolated timings have misled before: decide with scripts/bench_shared_fused.py."""
    import time
    cfg, moe = _main_layer()
    x, dyt = _main_inputs(cfg)
    for skew in (0.0, 1.5):
        with torch.no_grad():
            moe.e_bias.copy_(torch.linspace(skew, -skew, cfg.n_experts, device="cuda"))
        line = []
        for mode in ("",) + MODES:
            for _ in range(3):
                layer_run(moe, mode, x, dyt, "fused")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                layer_run(moe, mode, x, dyt, "fused")
            torch.cuda.synchronize()
            line.append(f"{mode or 'off'} {(time.perf_counter() - t0) * 100:.3f} ms")
        print(f"bias skew {skew}: " + ", ".join(line))
    with torch.no_grad():
        moe.e_bias.zero_()
