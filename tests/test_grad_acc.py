"""Fused fp32 weight-gradient accumulation (OptimizerBundle(offload_master=True, fused_grad_acc=True), train.py
--fused-grad-acc; moe_llm/layers/common.py GradSink / _LinearAccumulate / add_mm_fp32_, the ACC weight-gradient
kernels of moe_llm/layers/moe_triton.py).

The weight-gradient GEMMs add their fp32 result straight into the parameters' fp32 gradient buffers instead of
writing a bf16 gradient that the copy's hook then adds. Checked on CPU (the expert kernels in Triton's interpreter):
* the forward and every activation gradient are bit-identical to the current path: only the gradients of the fused
  weights change (all other gradients, the losses and the hook-path bias stay bit-identical);
* those differ from the current path at the bf16 rounding level of one micro-batch gradient and are much closer to
  the exact (fp64) sum of the same per-micro-batch products: op level, MoE experts, whole model over 3 micro-batches;
* every micro-batch is accumulated exactly once: recompute (whole layer, FFN only, expert recompute) and activation
  offloading give bit-identical buffers; the copies never get a .grad (no second count through the hook);
* clipping, zero_grad, the non-finite-step skip, training steps and resume work as before; the flag is inert
  without offload_master and train.py refuses it there.
GPU tests at the end (skipped without CUDA): cuBLAS addmm(out_dtype=fp32) in place, CastLinear and the experts at
main sizes against an fp64 reference, the ACC kernels with forced and autotuned configs (restore_value), no host
syncs, and tiny-model training on the GPU."""
import copy
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported (see tests/test_moe_triton.py)

from moe_llm.config import tiny  # noqa: E402
from moe_llm.layers import common  # noqa: E402
from moe_llm.layers import moe_triton as mt  # noqa: E402
from moe_llm.layers.common import CastLinear  # noqa: E402
from moe_llm.layers.moe import MoE  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402
from moe_llm.optim import OptimizerBundle, wsd_lr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


@pytest.fixture
def big_tiles():
    """Interpreter only: 32x32x32 tiles for every expert GEMM instead of 16^3 (3-4x faster on CPU; both paths of a
    comparison use the same tiles, and the op-level tests keep the default ones)."""
    if DEV == "cpu":
        mt.set_configs(32, 32, 32, 4, 1)
    yield
    if DEV == "cpu":
        mt.reset_configs()


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


class Lin(nn.Module):
    """CastLinear layers as a 'model' for OptimizerBundle (it only needs cast_params(), as MoELLM has)."""

    def __init__(self, dims, bias=False):
        super().__init__()
        self.layers = nn.ModuleList(CastLinear(a, b, bias=bias) for a, b in zip(dims[:-1], dims[1:]))

    def cast_params(self):
        return [p for lin in self.layers for p in lin.parameters()]

    def forward(self, x):
        for lin in self.layers:
            x = lin(x)
        return x


def bundle(model, fused, offload_master=True):
    opt = OptimizerBundle(model, lr=3e-3, muon_state_dtype=BF16, offload_master=offload_master,
                          fused_grad_acc=fused)
    opt.begin()
    return opt


def grads_of(opt, model):
    """Each parameter's accumulated gradient as clip_grad_norm sees it: the fp32 buffer of a cast parameter (its
    .grad after a forward without autocast), else .grad."""
    return {n: (opt._gbuf.get(id(p), p.grad) if id(p) in opt._cast else p.grad) for n, p in model.named_parameters()}


def routing(N, K, E, kind, gen):
    """Expert ids [N, K], distinct per token (as tests/test_moe_triton.py)."""
    if kind == "collapsed":
        return torch.arange(K).repeat(N, 1)
    logits = torch.zeros(N, E)
    if kind == "skewed":
        logits += torch.linspace(3.0, -1.0, E)
    elif kind == "empty":
        logits[:, 1::2] = -1e9  # only the even experts get tokens
    gumbel = -torch.log(-torch.log(torch.rand(N, E, generator=gen).clamp(1e-9, 1 - 1e-9)))
    return torch.topk(logits + gumbel, K, dim=-1).indices


def attach_copies(params, fused):
    """Persistent bf16 copies as OptimizerBundle(offload_master=True).begin() makes them, without an optimizer:
    the same hook, and a GradSink when fused. Returns the fp32 buffer dict (id(param) -> buffer)."""
    store = {}
    for p in params:
        c = p.detach().to(BF16).requires_grad_(True)

        def hook(c, pid=id(p)):
            buf = store.get(pid)
            if c.grad is None:  # AccumulateGrad calls the hook even when no gradient arrived (fused paths)
                return
            if buf is None:
                store[pid] = c.grad.float()
            else:
                buf.add_(c.grad)
            c.grad = None
        c.register_post_accumulate_grad_hook(hook)
        common.set_cached_copy(p, c, common.GradSink(store, id(p), p.shape, c.device) if fused else None)
    return store


def wgrad_ref64(a, b, order, offs, K, P, Q, a_gather, b_gather, two):
    """What moe_triton._wgrad sums, exactly (fp64): per expert e, A_e^T B_e (and A_e^T B2_e) over its sorted rows."""
    tok = order // K
    A = (a.index_select(0, tok) if a_gather else a)[:, :P].double()
    B = (b.index_select(0, tok) if b_gather else b).double()
    E = offs.numel() - 1
    bounds = offs.tolist()
    outs = []
    for j in range(2 if two else 1):
        Bj = B[:, j * Q:(j + 1) * Q]
        o = torch.zeros(E, P, Q, dtype=torch.float64, device=a.device)
        for e in range(E):
            s, t = bounds[e], bounds[e + 1]
            if t > s:
                o[e] = A[s:t].T @ Bj[s:t]
        outs.append(o)
    return outs


class Truth:
    """The exact (fp64) sum of every per-micro-batch weight-gradient product that the fused paths add into a
    buffer, recorded from the operands they pass to add_mm_fp32_ and moe_triton._wgrad; keyed by the buffer.
    Since activations and their gradients are bit-identical between the paths, it is the reference for the
    current path's buffers too."""

    def __init__(self, monkeypatch):
        self.sums = {}  # id(buffer) -> fp64 sum
        real_mm, real_wgrad = common.add_mm_fp32_, mt._wgrad

        def add_mm(out, a, b):
            self._add(out, a.double() @ b.double())
            return real_mm(out, a, b)

        def wgrad(a, b, order, offs, perm, K, P, Q, a_gather, b_gather, two, out_dtype, acc=None):
            if acc is not None:
                for buf, t in zip(acc if two else (acc,), wgrad_ref64(a, b, order, offs, K, P, Q, a_gather, b_gather,
                                                                     two)):
                    self._add(buf, t)
            return real_wgrad(a, b, order, offs, perm, K, P, Q, a_gather, b_gather, two, out_dtype, acc)

        monkeypatch.setattr(common, "add_mm_fp32_", add_mm)
        monkeypatch.setattr(mt, "_wgrad", wgrad)

    def _add(self, buf, t):
        k = id(buf)
        self.sums[k] = t if k not in self.sums else self.sums[k] + t

    def of(self, buf):
        return self.sums[id(buf)]


def _batch(cfg, s, k):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(1000 * s + k))
    return x[:, :-1], x[:, 1:]


def _graph_names(t):
    names, seen, todo = [], set(), [t.grad_fn]
    while todo:
        fn = todo.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        names.append(fn.name())
        todo.extend(f for f, _ in fn.next_functions)
    return names


# ------------------------------------------------------------------------------------------------- op level
def test_add_mm_fp32_semantics():
    """CPU fallback of the fused GEMM (CUDA: tests at the end): the fp32 product of the bf16 values (exact), added
    once into the buffer in place."""
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(24, 40, generator=g).to(BF16), torch.randn(40, 16, generator=g).to(BF16)
    buf = torch.randn(24, 16, generator=g)
    prev, ptr = buf.clone(), buf.data_ptr()
    out = common.add_mm_fp32_(buf, a, b)
    assert out is buf and buf.data_ptr() == ptr and buf.dtype == torch.float32
    assert torch.equal(buf, prev + a.float() @ b.float())
    assert rel_rms(buf, prev.double() + a.double() @ b.double()) < 1e-6


def _linear_input(layout, g):
    if layout == "3d":
        return torch.randn(2, 5, 48, generator=g)
    if layout == "2d":
        return torch.randn(10, 48, generator=g)
    if layout == "3d_noncontig":  # reshape copies: x2 contiguous
        return torch.randn(2, 48, 5, generator=g).transpose(1, 2)
    if layout == "2d_colmajor":   # column-major x2: autograd's mm_mat1_backward takes its other branch
        return torch.randn(48, 10, generator=g).t()
    if layout == "3d_colmajor":   # [1, T, D] with transposed strides: reshape is a column-major view
        return torch.randn(1, 48, 10, generator=g).transpose(1, 2)
    raise ValueError(layout)


@pytest.mark.parametrize("layout", ["3d", "2d", "3d_noncontig", "2d_colmajor", "3d_colmajor"])
@pytest.mark.parametrize("in_dtype", [torch.float32, BF16])
def test_linear_fused_vs_current(layout, in_dtype):
    """Two CastLinear layers under bf16 autocast, persistent copies, 3 micro-batches: the outputs and the input
    gradients are bit-identical to the current path; the fp32 weight buffers differ by the bf16 rounding of the
    micro-batch gradients and are far closer to the exact sum of the same products."""
    torch.manual_seed(0)
    base = Lin([48, 40, 24])
    g = torch.Generator().manual_seed(1)
    xs = [_linear_input(layout, g).to(in_dtype) for _ in range(3)]
    dys = [torch.randn(*x.shape[:-1], 24, generator=g).to(BF16) for x in xs]
    res = {}
    for fused in (False, True):
        m = copy.deepcopy(base)
        opt = bundle(m, fused)
        ys, dxs, inner = [], [], []
        for x, dy in zip(xs, dys):
            xi = x.detach().requires_grad_()
            with torch.autocast("cpu", dtype=BF16):
                h = m.layers[0](xi)
                h.retain_grad()
                y = m.layers[1](h)
            assert ("_LinearAccumulateBackward" in _graph_names(y)) == fused
            y.backward(dy)
            ys.append(y.detach())
            dxs.append(xi.grad)
            inner.append((xi.detach(), h.detach(), h.grad))  # the products' operands, for the exact sums
        for _, c in opt._cast.values():
            assert c.grad is None  # fused: no gradient reaches the copy; current: the hook cleared it
        res[fused] = (ys, dxs, inner, [opt._gbuf[id(lin.weight)] for lin in m.layers])
    (y0, dx0, in0, b0), (y1, dx1, in1, b1) = res[False], res[True]
    for a, b in zip(y0 + dx0, y1 + dx1):
        assert a.dtype == b.dtype and torch.equal(a, b)
    # exact sums: layer 1: dy^T h, layer 0: dh^T x (x as the GEMM saw it, rounded to bf16)
    t1 = sum(dy.reshape(-1, 24).double().T @ h.reshape(-1, 40).double() for dy, (_, h, _) in zip(dys, in1))
    t0 = sum(dh.reshape(-1, 40).double().T @ x.to(BF16).reshape(-1, 48).double() for x, _, dh in in1)
    for cur, new, t in ((b0[0], b1[0], t0), (b0[1], b1[1], t1)):
        assert new.dtype == torch.float32 and new.shape == t.shape
        e_new, e_cur = rel_rms(new, t), rel_rms(cur, t)
        assert e_new < 1e-6 and e_cur > 1e-4 and e_new < 0.01 * e_cur, (e_new, e_cur)
        assert rel_rms(new, cur) < 1e-2


def test_linear_with_bias_keeps_hook_path():
    """A CastLinear with bias (KDA's g_up) stays on the current path under fused accumulation: identical values."""
    torch.manual_seed(0)
    base = Lin([32, 48], bias=True)
    xs = [torch.randn(2, 6, 32) for _ in range(2)]
    out = {}
    for fused in (False, True):
        m = copy.deepcopy(base)
        opt = bundle(m, fused)
        for x in xs:
            with torch.autocast("cpu", dtype=BF16):
                y = m(x)
            assert "_LinearAccumulateBackward" not in _graph_names(y)
            y.float().square().mean().backward()
        out[fused] = [opt._gbuf[id(p)] for p in m.cast_params()]
    assert all(torch.equal(a, b) for a, b in zip(out[False], out[True]))


def _wgrad_case(kind, two, gather, seed=0):
    """Operands of one moe_triton._wgrad call at small sizes: dW_down (A = w h sorted rows, B = dy gathered) or
    dW_gate/dW_up (A = x gathered, B = [dg | du] sorted rows); gather=False: expert-sorted copies instead."""
    g = torch.Generator().manual_seed(seed)
    N, D, I, E, K = 96, 48, 32, 8, 2
    idx = routing(N, K, E, kind, g).to(DEV)
    order, offs, perm = mt._sort_assignments(idx, E)
    M = N * K
    if two:
        a, b = torch.randn(N, D, generator=g), torch.randn(M, 2 * I, generator=g)
        P, Q, ag, bg = D, I, gather, False
        if not gather:
            a = a.index_select(0, (order // K).cpu())
    else:
        a, b = torch.randn(M, I, generator=g), torch.randn(N, D, generator=g)
        P, Q, ag, bg = I, D, False, gather
        if not gather:
            b = b.index_select(0, (order // K).cpu())
    a, b = a.to(DEV, BF16), b.to(DEV, BF16)
    return (a, b, order, offs, perm, K, P, Q, ag, bg, two), idx


@pytest.mark.parametrize("two", [False, True])
@pytest.mark.parametrize("gather", [True, False])
@pytest.mark.parametrize("kind", ["skewed", "empty"])
def test_wgrad_kernel_accumulates(two, gather, kind):
    """The ACC weight-gradient kernel adds exactly the fp32 sums that the plain kernel computes (bit for bit in the
    interpreter: same tile order, one fp32 add) into the buffer; the plain bf16 output is those sums rounded. Experts
    without rows leave their buffer slices unchanged."""
    args, idx = _wgrad_case(kind, two, gather)
    as_tuple = (lambda r: r) if two else (lambda r: (r,))
    sums = as_tuple(mt._wgrad(*args, torch.float32))
    rounded = as_tuple(mt._wgrad(*args, BF16))
    g = torch.Generator().manual_seed(5)
    prev = [torch.randn(s.shape, generator=g).to(DEV) * 4 for s in sums]
    bufs = [p.clone() for p in prev]
    assert mt._wgrad(*args, torch.float32, tuple(bufs) if two else bufs[0]) is None
    exact = wgrad_ref64(*args[:4], *args[5:])
    used = torch.zeros(sums[0].shape[0], dtype=torch.bool, device=idx.device)
    used[idx.flatten()] = True
    for buf, p, s, r, t in zip(bufs, prev, sums, rounded, exact):
        if DEV == "cpu":
            assert torch.equal(buf, p + s)
            assert torch.equal(r, s.to(BF16))
        else:
            assert rel_rms(buf - p, s) < 1e-5
        assert rel_rms(s, t) < 1e-6
        assert torch.equal(buf[~used], p[~used])
        if kind == "empty":
            assert (~used).any()


def _experts_run(moe, fused, mbs, recompute=False, monkeypatch=None):
    """3 micro-batches through MoE.experts (triton) with persistent copies: outputs, input gradients and the
    weight-gradient buffers; the Truth of the fused run when monkeypatch is given."""
    params = (moe.w_gate, moe.w_up, moe.w_down)
    store = attach_copies(params, fused)
    truth = Truth(monkeypatch) if monkeypatch is not None else None
    moe.impl, moe.recompute_experts = "triton", recompute
    outs = []
    for x, idx, w, dy in mbs:
        xi, wi = x.detach().requires_grad_(), w.detach().requires_grad_()
        with torch.autocast(DEV, dtype=BF16):
            y = moe.experts(xi, idx, wi)
        y.backward(dy)
        outs += [y.detach(), xi.grad, wi.grad]
    bufs = [store[id(p)] for p in params]
    common.clear_cached_copies()
    return outs, bufs, truth


def _expert_batches(cfg, N, n=3, device=DEV):
    g = torch.Generator().manual_seed(11)
    mbs = []
    for _ in range(n):
        x = torch.randn(N, cfg.d_model, generator=g).to(device, BF16)
        idx = routing(N, cfg.top_k, cfg.n_experts, "skewed", g).to(device)
        w = (torch.rand(N, cfg.top_k, generator=g) + 0.1).to(device)
        dy = torch.randn(N, cfg.d_model, generator=g).to(device, BF16)
        mbs.append((x, idx, w, dy))
    return mbs


def test_routed_experts_fused_vs_current(monkeypatch, big_tiles):
    """The MoE layer's triton experts, 3 micro-batches: y, dx and the combine-weight gradient bit-identical to the
    current path (plain and with the expert recompute); the weight buffers far closer to the exact sum; the expert
    recompute accumulates bit-identically to the plain fused run."""
    cfg = tiny(moe_impl="triton")
    torch.manual_seed(3)
    moe = MoE(cfg, 1).to(DEV).train()
    mbs = _expert_batches(cfg, 80)
    cur, b_cur, _ = _experts_run(moe, False, mbs)
    rec, b_rec, _ = _experts_run(moe, True, mbs, recompute=True)
    new, b_new, truth = _experts_run(moe, True, mbs, monkeypatch=monkeypatch)
    for a, b, c in zip(cur, new, rec):
        assert torch.equal(a, b) and torch.equal(a, c)
    for b_c, b_n, b_r in zip(b_cur, b_new, b_rec):
        assert torch.equal(b_n, b_r)
        t = truth.of(b_n)
        e_new, e_cur = rel_rms(b_n, t), rel_rms(b_c, t)
        assert e_new < 1e-6 and e_new < 0.01 * e_cur, (e_new, e_cur)
        assert rel_rms(b_n, b_c) < 1e-2


# ------------------------------------------------------------------------------------------------- whole model
def _micro(m, batches, autocast=True):
    losses = []
    for x, y in batches:
        with torch.autocast("cpu", dtype=BF16, enabled=autocast):
            loss, _ = m(x, y)
        (loss / len(batches)).backward()
        losses.append(loss.detach())
    return losses


def _kernels_run(m):
    """The triton expert kernels run for this model's tensors (else routed_experts falls back to the reference)."""
    return mt._INTERPRET or next(m.parameters()).is_cuda


def _fused_weights(m):
    """The weights whose gradients the fused paths accumulate: bias-free CastLinear weights, triton expert weights."""
    out = [mod.weight for mod in m.modules() if isinstance(mod, CastLinear) and mod.bias is None]
    for moe in m.moe_layers():
        if moe.impl == "triton" and _kernels_run(m):
            out += [moe.w_gate, moe.w_up, moe.w_down]
    return out


@pytest.mark.parametrize("residual,frac", [("standard", 0.0), ("block_attnres", 0.5)])
def test_model_grads_vs_current_and_exact(residual, frac, monkeypatch, big_tiles):
    """Whole tiny model (KDA, attention, Engram, dense FFN, triton MoE + shared expert), bf16 autocast, offload_master
    bundle, 3 accumulated micro-batches (frac 0.5: recomputed layers and the expert recompute in the others):
    losses and every gradient outside the fused weights bit-identical to the current path; each fused weight's
    buffer closer to the exact sum of its per-micro-batch products than the current buffer, and within the bf16
    rounding level of it; no copy ever holds a .grad."""
    torch.manual_seed(0)
    cfg = tiny(residual=residual, moe_impl="triton")
    base = MoELLM(cfg)
    batches = [_batch(cfg, 0, k) for k in range(3)]
    res = {}
    for fused in (False, True):
        m = copy.deepcopy(base)
        m.set_checkpointing(frac)
        opt = bundle(m, fused)
        truth = Truth(monkeypatch) if fused else None
        losses = _micro(m, batches)
        assert all(c.grad is None for _, c in opt._cast.values())
        res[fused] = (m, opt, losses, truth)
    (m0, o0, l0, _), (m1, o1, l1, truth) = res[False], res[True]
    assert all(torch.equal(a, b) for a, b in zip(l0, l1))
    g0, g1 = grads_of(o0, m0), grads_of(o1, m1)
    fused_names = {n for n, p in m1.named_parameters() if any(p is q for q in _fused_weights(m1))}
    assert len(fused_names) == len(_fused_weights(m1)) > 0
    for n in g0:
        if n not in fused_names:
            assert (g0[n] is None and g1[n] is None) or torch.equal(g0[n], g1[n]), n
    errs = []
    for n in sorted(fused_names):
        t = truth.of(g1[n])  # KeyError: this weight's gradient did not go through a fused GEMM
        e_new, e_cur = rel_rms(g1[n], t), rel_rms(g0[n], t)
        assert e_new <= e_cur and e_new < 1e-5, (n, e_new, e_cur)
        assert rel_rms(g1[n], g0[n]) < 2e-2, n
        errs.append((e_new, e_cur))
    assert len(truth.sums) == len(fused_names)
    mean_new = sum(e for e, _ in errs) / len(errs)
    mean_cur = sum(e for _, e in errs) / len(errs)
    print(f"{residual} frac {frac}: mean rel. RMS error vs exact sum: fused {mean_new:.2e}, current {mean_cur:.2e}")
    assert mean_new < 0.01 * mean_cur
    # against an fp32 model (no autocast, same weights and batches): both bf16 runs carry the same activation
    # rounding; the fused one lacks the extra rounding of every micro-batch's weight gradient, so it is no farther
    ref = copy.deepcopy(base)
    ref.set_checkpointing(frac)
    _micro(ref, batches, autocast=False)
    gr = dict(ref.named_parameters())
    sse_new = sum((g1[n].double() - gr[n].grad.double()).pow(2).sum() for n in fused_names)
    sse_cur = sum((g0[n].double() - gr[n].grad.double()).pow(2).sum() for n in fused_names)
    norm = sum(gr[n].grad.double().pow(2).sum() for n in fused_names).sqrt()
    print(f"  vs the fp32 model: rel. error fused {sse_new.sqrt() / norm:.4e}, current {sse_cur.sqrt() / norm:.4e}")
    assert sse_new <= sse_cur
    if frac == 0.0:  # every bias-free CastLinear and every MoE layer went through a fused autograd function
        m = copy.deepcopy(base)
        bundle(m, True)
        with torch.autocast("cpu", dtype=BF16):
            loss, _ = m(*batches[0])
        names = _graph_names(loss)
        n_lin = sum(isinstance(mod, CastLinear) and mod.bias is None for mod in m.modules())
        assert names.count("_LinearAccumulateBackward") == n_lin
        assert names.count("RoutedExpertsBackward") == (len(m.moe_layers()) if _kernels_run(m) else 0)


def test_fused_buffers_identical_under_recompute_and_offload(big_tiles):
    """Each micro-batch is accumulated exactly once: whole-layer recompute + expert recompute, FFN-only recompute
    and activation offloading (every saved tensor through host memory) give bit-identical fused buffers."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", moe_impl="triton")
    base = MoELLM(cfg)
    batches = [_batch(cfg, 1, k) for k in range(2)]
    out = []
    for frac, ffn, off in [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.5)]:
        m = copy.deepcopy(base)
        m.set_checkpointing(frac, ffn_frac=ffn, offload_frac=off)
        if m.offloader is not None:
            m.offloader.min_bytes = 0
        opt = bundle(m, True)
        losses = _micro(m, batches)
        if m.offloader is not None:
            assert m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0
        out.append((losses, grads_of(opt, m)))
    (l_ref, g_ref) = out[0]
    for losses, g in out[1:]:
        assert all(torch.equal(a, b) for a, b in zip(losses, l_ref))
        for n in g_ref:
            assert (g[n] is None and g_ref[n] is None) or torch.equal(g[n], g_ref[n]), n


def test_bmm_experts_and_fp32_forward_keep_hook_path():
    """moe_impl="bmm": the expert weights keep the hook path (bit-identical buffers), only the linear layers are
    fused. A forward without autocast uses the fp32 masters: identical to the current bundle."""
    torch.manual_seed(0)
    cfg = tiny()
    base = MoELLM(cfg)
    batches = [_batch(cfg, 2, k) for k in range(2)]
    for autocast in (True, False):
        res = {}
        for fused in (False, True):
            m = copy.deepcopy(base)
            opt = bundle(m, fused)
            losses = _micro(m, batches, autocast=autocast)
            res[fused] = (losses, grads_of(opt, m), m)
        (l0, g0, _), (l1, g1, m1) = res[False], res[True]
        assert all(torch.equal(a, b) for a, b in zip(l0, l1))
        lin = {n for n, p in m1.named_parameters() if any(p is q for q in _fused_weights(m1))}
        assert lin and not any(".w_gate" in n or ".w_up" in n or ".w_down" in n for n in lin)
        for n in g0:
            if autocast and n in lin:
                assert rel_rms(g1[n], g0[n]) < 2e-2 and not torch.equal(g1[n], g0[n]), n
            else:
                assert (g0[n] is None and g1[n] is None) or torch.equal(g0[n], g1[n]), n


# ------------------------------------------------------------------------------------------------- training
def _train(m, opt, cfg, steps, start=0, clip=0.05, micro=2, bad_step=None):
    """train.py's loop on CPU under bf16 autocast: accumulate, skip a non-finite step (zero_grad, no update),
    clip, step, zero_grad, router bias. Returns the CE of every micro-batch and the clipped norm of every step."""
    out = []
    for s in range(start, start + steps):
        loss_sum = torch.zeros(())
        for k in range(micro):
            x, y = _batch(cfg, s, k)
            with torch.autocast("cpu", dtype=BF16):
                loss, info = m(x, y)
            if s == bad_step:
                loss = loss * float("nan")
            (loss / micro).backward()
            loss_sum += loss.detach().float()
            out.append(float(info["ce"]))
        if not bool(torch.isfinite(loss_sum)):
            assert all(not torch.isfinite(g).all() for g in opt._gbuf.values())
            opt.zero_grad()
            assert not opt._gbuf and all(p.grad is None for p in m.parameters())
            continue
        out.append(float(opt.clip_grad_norm(clip)))
        opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
        opt.zero_grad()
        m.update_router_bias()
    return out


def test_training_steps_track_current_path(big_tiles):
    """3 training steps (clipping active, bf16 Muon momentum) with triton experts: the first step's losses are
    identical (same weights), the gradient norms and later losses differ only at the rounding level."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", moe_impl="triton")
    base = MoELLM(cfg)
    base.set_checkpointing(0.5)
    runs = {}
    for fused in (False, True):
        m = copy.deepcopy(base)
        runs[fused] = _train(m, bundle(m, fused), cfg, 3)
    a, b = runs[False], runs[True]
    assert a[:2] == b[:2]           # step 0's micro-batch CEs
    assert a != b                   # the gradients did change
    for x, y in zip(a, b):
        assert abs(x - y) <= 2e-3 * abs(x), (a, b)


def test_nonfinite_skip_resume_and_inert_without_offload_master():
    """(1) After a skipped non-finite step (train.py: zero_grad, no update) the next step's buffers are bit-identical
    to those of a run that never saw it; (2) a fused run resumed from a checkpoint continues bit-identically;
    (3) fused_grad_acc without offload_master changes nothing."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    # (1)
    m_bad, m_ref = copy.deepcopy(base), copy.deepcopy(base)
    o_bad, o_ref = bundle(m_bad, True), bundle(m_ref, True)
    _train(m_bad, o_bad, cfg, 1, start=0, bad_step=0)
    for m in (m_bad, m_ref):
        _micro(m, [_batch(cfg, 1, k) for k in range(2)])
    g_bad, g_ref = grads_of(o_bad, m_bad), grads_of(o_ref, m_ref)
    for n in g_ref:
        assert (g_bad[n] is None and g_ref[n] is None) or torch.equal(g_bad[n], g_ref[n]), n
    # (2)
    m = copy.deepcopy(base)
    opt = bundle(m, True)
    _train(m, opt, cfg, 2)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
    full = _train(m, opt, cfg, 2, start=2)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)
    m2 = MoELLM(cfg)
    m2.load_state_dict(sd["model"])
    o2 = OptimizerBundle(m2, lr=3e-3, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=True)
    o2.load_state_dict(sd["opt"])
    o2.begin()
    assert _train(m2, o2, cfg, 2, start=2) == full
    # (3)
    common.clear_cached_copies()
    runs = []
    for fused in (False, True):
        m = copy.deepcopy(base)
        opt = bundle(m, fused, offload_master=False)
        assert not opt._cast and not common._CACHE
        runs.append((_train(m, opt, cfg, 2), [p.detach().clone() for p in m.parameters()]))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))


def _bins(root):
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


def test_train_py_flag(tmp_path):
    """train.py --fused-grad-acc: refused without --offload-master; accepted with it (on CPU train.py runs without
    autocast, so the fp32 masters are used directly and nothing is fused: this only checks the plumbing)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    data = str(tmp_path / "bins")
    _bins(data)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny", "--data", data,
           "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--seq-len", "32", "--micro-bsz", "2",
           "--global-tokens", "128", "--steps", "2", "--warmup", "1", "--log-every", "1", "--eval-every", "0",
           "--ckpt-every", "0", "--fused-grad-acc"]
    r = subprocess.run(cmd + ["--out", str(tmp_path / "a")], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "--fused-grad-acc needs --offload-master" in r.stderr, r.stderr[-2000:]
    r = subprocess.run(cmd + ["--out", str(tmp_path / "b"), "--offload-master"], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "weight gradients: fp32, accumulated inside the weight-gradient GEMMs" in r.stdout
    assert "TRAIN_DONE" in r.stdout


# ------------------------------------------------------------------------------------------------- GPU
@gpu
def test_gpu_add_mm_fp32_in_place():
    """aten::addmm.dtype_out on CUDA with out aliasing self (what add_mm_fp32_ calls): runs in place (same tensor,
    same memory, no host sync), equals the buffer plus the fp32 GEMM result, and is far closer to the exact sum than
    the current path (bf16 GEMM output, then the fp32 add). Layout as in _LinearAccumulate.backward: dy^T x."""
    torch.manual_seed(0)
    N, dout, din = 8192, 1024, 3072
    dy = torch.randn(N, dout, device="cuda", dtype=BF16)
    x = torch.randn(N, din, device="cuda", dtype=BF16)
    buf = torch.randn(dout, din, device="cuda") * 200  # an earlier micro-batch's gradient sum
    prev, ptr = buf.clone(), buf.data_ptr()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = common.add_mm_fp32_(buf, dy.t(), x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert out is buf and buf.data_ptr() == ptr and buf.dtype == torch.float32
    sum32 = torch.mm(dy.t(), x, out_dtype=torch.float32)
    exact = prev.double() + dy.double().t() @ x.double()
    cur = prev + dy.t().mm(x).float()
    e_new, e_cur = rel_rms(buf, exact), rel_rms(cur, exact)
    print(f"addmm out_dtype in place: |fused - exact| {e_new:.2e}, |current - exact| {e_cur:.2e}, "
          f"bitwise == prev + mm(out_dtype=fp32): {torch.equal(buf, prev + sum32)}")
    assert rel_rms(buf, prev + sum32) < 1e-5  # (bitwise unless cuBLAS picks another algorithm for beta = 1)
    assert e_new < 0.05 * e_cur
    z = torch.zeros(dout, din, device="cuda")  # first micro-batch of a step: zeros + dW == dW
    common.add_mm_fp32_(z, dy.t(), x)
    assert rel_rms(z, sum32) < 1e-5


@gpu
def test_gpu_linear_fused_vs_current_main_sizes(monkeypatch):
    """CastLinear 1024 -> 3072 -> 1024 (main_2b's FFN shapes) under bf16 autocast with an offload_master bundle
    (fp32 masters in pinned CPU memory), 3 micro-batches of 2 x 4096 tokens: outputs and input gradients
    bit-identical to the current path; both weight buffers far closer to the exact sums."""
    torch.manual_seed(0)
    base = Lin([1024, 3072, 1024]).cuda()
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(2, 4096, 1024, generator=g).to("cuda", BF16) for _ in range(3)]
    dys = [torch.randn(2, 4096, 1024, generator=g).to("cuda", BF16) for _ in range(3)]
    res = {}
    for fused in (False, True):
        m = copy.deepcopy(base)
        opt = bundle(m, fused)
        assert all(p.device.type == "cpu" for p in m.cast_params())  # masters offloaded
        truth = Truth(monkeypatch) if fused else None
        outs = []
        for x, dy in zip(xs, dys):
            xi = x.detach().requires_grad_()
            with torch.autocast("cuda", dtype=BF16):
                y = m(xi)
            y.backward(dy)
            outs += [y.detach(), xi.grad]
        res[fused] = (outs, [opt._gbuf[id(lin.weight)] for lin in m.layers], truth)
    (o0, b0, _), (o1, b1, truth) = res[False], res[True]
    for a, b in zip(o0, o1):
        assert torch.equal(a, b)
    for cur, new in zip(b0, b1):
        t = truth.of(new)
        e_new, e_cur = rel_rms(new, t), rel_rms(cur, t)
        print(f"weight {tuple(new.shape)}: |fused - exact| {e_new:.2e}, |current - exact| {e_cur:.2e}")
        assert e_new < 0.05 * e_cur and rel_rms(new, cur) < 1e-2


@gpu
def test_gpu_expert_kernels_accumulate():
    """The ACC weight-gradient kernels at main sizes (N 8192, d 1024, E 32, k 4, I 512, max/mean 1.8): with one
    forced config the buffer gets exactly the plain kernel's fp32 sums added (up to fp32 rounding); autotuned from
    scratch (tuner and disk caches bypassed) it holds the previous value plus exactly ONE gradient: restore_value puts
    the buffers back after every timed run."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import bench_moe_kernels
    N, D, I, E, K = 8192, 1024, 512, 32, 4
    idx, _ = bench_moe_kernels.synthetic_routing(N, K, E, 1.8, seed=1)
    order, offs, perm = mt._sort_assignments(idx, E)
    g = torch.Generator().manual_seed(0)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.1).to("cuda", BF16)  # noqa: E731
    cases = {"down": (mk(N * K, I), mk(N, D), order, offs, perm, K, I, D, False, True, False),
             "gate_up": (mk(N, D), mk(N * K, 2 * I), order, offs, perm, K, D, I, True, False, True)}
    for name, args in cases.items():
        two = args[-1]
        as_tuple = (lambda r: r) if two else (lambda r: (r,))
        try:
            mt.set_configs(64, 64, 32, 4, 3)
            sums = as_tuple(mt._wgrad(*args, torch.float32))
            prev = [torch.randn_like(s) * s.std() * 5 for s in sums]
            bufs = [p.clone() for p in prev]
            mt._wgrad(*args, torch.float32, tuple(bufs) if two else bufs[0])
            for b, p, s in zip(bufs, prev, sums):
                print(f"{name} forced config: bitwise == prev + sums: {torch.equal(b, p + s)}")
                assert rel_rms(b, p + s) < 1e-5
        finally:
            mt.reset_configs()
        kern = mt._wgrad_dual_acc if two else mt._wgrad_single_acc
        saved = kern.cache_results
        kern.cache.clear()
        kern.cache_results = False
        try:
            bufs = [p.clone() for p in prev]
            mt._wgrad(*args, torch.float32, tuple(bufs) if two else bufs[0])
            torch.cuda.synchronize()
        finally:
            kern.cache_results = saved
        for b, p, s in zip(bufs, prev, sums):
            err = rel_rms(b - p, s)
            print(f"{name} autotuned over {len(kern.configs)} candidates ({kern.best_config}): rel. RMS of the "
                  f"added gradient {err:.2e}")
            assert err < 1e-5


@gpu
def test_gpu_routed_experts_fused_main_size(monkeypatch):
    """The routed experts at main sizes (main_2b layer, N 8192, skewed routing), persistent copies as with
    offload_master, 3 micro-batches: y, dx, dw bit-identical to the current path (plain and expert recompute),
    the recompute's buffers bit-identical to the plain fused run, every weight buffer far closer to the exact sum;
    no CPU-GPU sync in a fused forward + backward."""
    from moe_llm.config import main_2b
    cfg = main_2b()
    torch.manual_seed(0)
    moe = MoE(cfg, 1).cuda().train()
    mbs = _expert_batches(cfg, 8192, device="cuda")
    # no host sync, plain and with the expert recompute (after warm-ups that compile and autotune both)
    store = attach_copies((moe.w_gate, moe.w_up, moe.w_down), True)
    moe.impl = "triton"
    for check in (False, True):
        for rec in (False, True):
            moe.recompute_experts = rec
            x, idx, w, dy = mbs[0]
            xi = x.detach().requires_grad_()
            if check:
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
            try:
                with torch.autocast("cuda", dtype=BF16):
                    y = moe.experts(xi, idx, w)
                y.backward(dy)
            finally:
                torch.cuda.set_sync_debug_mode("default")
    assert len(store) == 3
    common.clear_cached_copies()
    cur, b_cur, _ = _experts_run(moe, False, mbs)
    rec, b_rec, _ = _experts_run(moe, True, mbs, recompute=True)
    new, b_new, truth = _experts_run(moe, True, mbs, monkeypatch=monkeypatch)
    for a, b, c in zip(cur, new, rec):
        assert torch.equal(a, b) and torch.equal(a, c)
    for name, b_c, b_n, b_r in zip(("W_gate", "W_up", "W_down"), b_cur, b_new, b_rec):
        assert torch.equal(b_n, b_r)
        t = truth.of(b_n)
        e_new, e_cur = rel_rms(b_n, t), rel_rms(b_c, t)
        print(f"{name}: |fused - exact| {e_new:.2e}, |current - exact| {e_cur:.2e}")
        assert e_new < 0.05 * e_cur and rel_rms(b_n, b_c) < 1e-2


@gpu
def test_gpu_model_training_fused_vs_current():
    """Tiny model on the GPU with the main run's exact paths (offload_master + offload_states, triton experts,
    recompute + expert recompute, block AttnRes, bf16 residual), 4 steps of 2 micro-batches: fused vs current losses
    track closely (GPU atomics elsewhere rule out bit-exactness); the fused copies never hold a .grad; peak memory
    is not higher."""
    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True, residual="block_attnres",
               moe_impl="triton")
    base = MoELLM(cfg).cuda()
    base.set_checkpointing(0.5)
    runs, peaks = {}, {}
    try:
        for fused in (False, True):
            m = copy.deepcopy(base)
            opt = OptimizerBundle(m, lr=1e-3, muon_state_dtype=BF16, offload_states=True, offload_master=True,
                                  fused_grad_acc=fused)
            opt.begin()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ces = []
            for s in range(4):
                for k in range(2):
                    xc = torch.randint(0, cfg.vocab_size, (2, 129), generator=torch.Generator().manual_seed(10 * s + k))
                    x = xc.cuda()
                    with torch.autocast("cuda", dtype=BF16):
                        loss, info = m(x[:, :-1], x[:, 1:])
                    (loss / 2).backward()
                    ces.append(float(info["ce"]))
                assert all(c.grad is None for _, c in opt._cast.values())
                opt.clip_grad_norm(1.0)
                opt.step(1e-3)
                opt.zero_grad()
                m.update_router_bias()
            torch.cuda.synchronize()
            peaks[fused] = torch.cuda.max_memory_allocated()
            runs[fused] = ces
            common.clear_cached_copies()
            del m, opt
    finally:
        common.clear_cached_copies()
    print(f"ces current {runs[False]}\nces fused   {runs[True]}\npeak MB {peaks[False] / 2**20:.1f} -> "
          f"{peaks[True] / 2**20:.1f}")
    for a, b in zip(runs[False], runs[True]):
        assert abs(a - b) < 2e-2 * abs(a)
    assert peaks[True] <= 1.02 * peaks[False] + 4 * 2 ** 20
