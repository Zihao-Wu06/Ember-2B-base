"""Fused projections (ModelConfig.fuse_proj; moe_llm/layers/common.py fused_linear / split_conv_silu / GroupGradSink,
OptimizerBundle's stacked persistent copies).

The bias-free projections that read the same input run as ONE GEMM over their stacked weights: KDA q/k/v/f_down/g_down/
b_proj (main_2b: N = 3336; the q/k/v short convs in one function) and attention q/k/v/gate (N = 2560). Checked on CPU:
* parameters, names, initial values, state_dict, optimizer groups and checkpoints are unchanged; a run written without
  the option resumes with it;
* forward: bit-identical to the separate GEMMs (CPU oneDNN sums each output element in the same order for the wider
  matrix), so layer outputs and whole-model losses are identical;
* dx: one GEMM with fp32 accumulation over all members and one bf16 rounding, instead of a bf16 GEMM per member plus
  autograd's adds of the partial dx: closer to the exact sum of the same products (op and layer level, vs fp64);
  whole-model gradients stay at the current path's rounding level and no farther from an fp32 model;
* dW: bf16 weight gradients (no fused accumulation) are bit-identical per row block; with fused accumulation the stack's
  one fp32 GEMM lands in one fp32 buffer whose row blocks are the members' buffers, equal to the separate fp32 GEMMs up
  to the fp32 summation order (CPU MKL sums in another order for another row count); fuse_proj_dw = "member" runs one
  GEMM per member instead, bit-identical to the separate layers;
* the merged q/k/v short conv is bit-identical to three ShortConvs (eager: one conv over all 3D channels; compiled:
  the per-block graph), and every parameter gradient other than the stacked weights' and the layer input's is
  bit-identical;
* persistent copies (offload_master): one stacked bf16 copy per group whose row blocks are the members' copies, refreshed
  in place; fused accumulation, clipping, zero_grad, the non-finite skip, recompute, activation offloading, eval /
  no_grad, fusion (torch.compile) and train.py's --override all work.
GPU tests at the end (skipped without CUDA): whether cuBLAS gives bit-identical forward blocks and dW row blocks at main
shapes (reported), dx accuracy, both layers at main size with fusion on (bit-identity report, no added host syncs) and
tiny-model training."""
import copy
import io
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_llm.config import main_2b, tiny
from moe_llm.layers import attention as attn_mod
from moe_llm.layers import common
from moe_llm.layers import kda as kda_mod
from moe_llm.layers.attention import Attention
from moe_llm.layers.common import CastLinear, GroupGradSink, conv_silu, fused_linear, split_conv_silu
from moe_llm.layers.kda import KDA
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, split_params, wsd_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
KDA_SIZES = (1024, 1024, 1024, 128, 128, 8)  # main_2b KDA: q, k, v, f_down, g_down, b_proj -> N = 3336
ATTN_SIZES = (1024, 256, 256, 1024)          # main_2b attention: q, k, v, gate -> N = 2560
MODES = ("plain", "hook", "fga")  # no persistent copies | --offload-master | --offload-master --fused-grad-acc
# + fuse_proj_dw = "member": the fp32 weight-gradient GEMM per member instead of per stack


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()
    common.set_fuse(False)


@pytest.fixture
def fuse_cpu():
    common.set_fuse(True, cpu=True)
    yield
    common.set_fuse(False)


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def bundle(model, mode, **kw):
    """OptimizerBundle as train.py builds it for the mode, begun. "fga_member": fused accumulation with
    fuse_proj_dw = "member" (read from model.cfg; the stand-in models below get a namespace)."""
    if mode == "fga_member" and not isinstance(model, MoELLM):
        model.cfg = types.SimpleNamespace(fuse_proj_dw="member")
    assert mode != "fga_member" or model.cfg.fuse_proj_dw == "member"
    opt = OptimizerBundle(model, lr=3e-3, muon_state_dtype=BF16, offload_master=mode != "plain",
                          fused_grad_acc=mode.startswith("fga"), **kw)
    opt.begin()
    return opt


def grads_of(opt, model):
    """Every parameter's accumulated gradient as clip_grad_norm sees it."""
    return {n: (opt._gbuf.get(id(p), p.grad) if id(p) in opt._cast else p.grad) for n, p in model.named_parameters()}


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


class Group(nn.Module):
    """Bias-free CastLinear layers that read the same x, as a 'model' for OptimizerBundle (cast_params and
    fused_proj_groups, as MoELLM has). fused: their outputs come from one fused_linear GEMM, else one layer each."""

    def __init__(self, d, sizes, fused):
        super().__init__()
        self.lins = nn.ModuleList(CastLinear(d, n, bias=False) for n in sizes)
        for lin in self.lins:
            nn.init.normal_(lin.weight, std=0.02)
        self.sizes, self.fused = tuple(sizes), fused

    def cast_params(self):
        return [lin.weight for lin in self.lins]

    def fused_proj_groups(self):
        return [self.cast_params()] if self.fused else []

    def forward(self, x):
        if self.fused:
            return fused_linear(x, self.cast_params()).split(self.sizes, dim=-1)
        return [lin(x) for lin in self.lins]


class Wrap(nn.Module):
    """One KDA / attention layer as a 'model' for OptimizerBundle."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def cast_params(self):
        return [p for m in self.layer.modules() if isinstance(m, CastLinear)
                for p in [m.weight] + ([m.bias] if m.bias is not None else [])]

    def fused_proj_groups(self):
        return [self.layer.proj_weights()] if self.layer.fuse_proj else []

    def forward(self, x):
        return self.layer(x)


# ------------------------------------------------------------------------------------------------- structure
def test_parameters_state_dict_and_optimizer_groups_unchanged():
    assert tiny().fuse_proj is False and main_2b().fuse_proj is False
    torch.manual_seed(0)
    a = MoELLM(tiny())
    torch.manual_seed(0)
    b = MoELLM(tiny(fuse_proj=True))
    assert [(n, p.shape) for n, p in a.named_parameters()] == [(n, p.shape) for n, p in b.named_parameters()]
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)  # same initial values too

    def groups(m):
        n = {id(p): k for k, p in m.named_parameters()}
        return {g: [n[id(p)] for p in ps] for g, ps in split_params(m).items()}

    assert groups(a) == groups(b)
    na, nb = ({id(p): k for k, p in m.named_parameters()} for m in (a, b))
    assert [na[id(p)] for p in a.cast_params()] == [nb[id(p)] for p in b.cast_params()]
    cfg = tiny()
    D, r, H = cfg.kda_heads * cfg.kda_head_dim, cfg.kda_lowrank, cfg.kda_heads
    assert not a.fused_proj_groups()
    rows = [sum(p.shape[0] for p in g) for g in b.fused_proj_groups()]
    assert rows == [3 * D + 2 * r + H] * 3 + [(2 * cfg.attn_heads + 2 * cfg.attn_kv_heads) * cfg.attn_head_dim]
    with torch.device("meta"):
        big = MoELLM(main_2b().replace(fuse_proj=True, engram_offload=False))
    rows = [sum(p.shape[0] for p in g) for g in big.fused_proj_groups()]
    assert len(rows) == 24 and rows.count(3336) == 18 and rows.count(2560) == 6
    assert main_2b().fuse_proj_dw == "stacked"
    with pytest.raises(ValueError, match="fuse_proj_dw"):
        MoELLM(tiny(fuse_proj=True, fuse_proj_dw="rows"))
    ced = MoELLM(tiny(ced_split=2, fuse_proj=True))  # CED decoder layers keep their own projections
    assert len(ced.fused_proj_groups()) == 2 and [b.kind for b in ced.blocks] == ["K", "K", "C", "C"]
    x, y = _batch(tiny(), 0, 0)
    with torch.autocast("cpu", dtype=BF16):
        loss, _ = ced(x, y)
    loss.backward()
    assert all(p.grad is not None for p in ced.blocks[0].mixer.proj_weights())


# ------------------------------------------------------------------------------------------------- op level
def _op_batches(d, sizes, n=3, tokens=96, seed=1):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(2, tokens // 2, d, generator=g).to(BF16) for _ in range(n)]
    dys = [[(torch.randn(2, tokens // 2, s, generator=g) * 0.1).to(BF16) for s in sizes] for _ in range(n)]
    return xs, dys


def _group_run(d, sizes, fused, mode, xs, dys):
    """3 micro-batches through the members (bf16 x as in the main run, bf16 autocast): blocks, dx, the model, bundle."""
    torch.manual_seed(0)
    m = Group(d, sizes, fused)
    opt = bundle(m, mode)
    outs, dxs = [], []
    for x, dy in zip(xs, dys):
        xi = x.detach().requires_grad_()
        with torch.autocast("cpu", dtype=BF16):
            ys = m(xi)
        torch.autograd.backward(list(ys), dy)
        outs.append([y.detach() for y in ys])
        dxs.append(xi.grad)
    return m, opt, outs, dxs


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sizes", [KDA_SIZES, ATTN_SIZES], ids=["kda", "attention"])
def test_fused_linear_vs_separate(sizes, mode):
    """main_2b widths (d 1024, N 3336 / 2560), 3 micro-batches: forward blocks bit-identical; dx closer to the exact
    sum (fp64 of the same bf16 products); dW bit-identical per row block in bf16 (plain, hook), and with fused
    accumulation one fp32 GEMM of the stack into one buffer whose row blocks are the members' buffers."""
    d, N = 1024, sum(sizes)
    xs, dys = _op_batches(d, sizes)
    m0, o0, out0, dx0 = _group_run(d, sizes, False, mode, xs, dys)
    m1, o1, out1, dx1 = _group_run(d, sizes, True, mode, xs, dys)
    for a, b in zip(out0, out1):
        for u, v in zip(a, b):
            assert u.dtype == v.dtype == BF16 and torch.equal(u, v)
    W = torch.cat([p.detach().to(BF16) for p in m1.cast_params()]).double()
    errs = []
    for dy, a, b in zip(dys, dx0, dx1):
        exact = torch.cat(dy, -1).double() @ W
        e_cur, e_new = rel_rms(a, exact), rel_rms(b, exact)
        assert b.dtype == BF16 and e_new < 3e-3 and e_new < 0.75 * e_cur, (e_new, e_cur)
        errs.append((e_new, e_cur))
    print(f"N {N} {mode}: dx rel. RMS error vs exact: fused {errs[0][0]:.2e}, separate {errs[0][1]:.2e}")
    if mode != "plain":
        assert len(o1._groups) == 1 and not o0._groups
        assert all(c.grad is None for _, c in o1._cast.values()) and o1._groups[0][1].grad is None
    g0 = [(o0._gbuf[id(p)] if mode != "plain" else p.grad) for p in m0.cast_params()]
    g1 = [(o1._gbuf[id(p)] if mode != "plain" else p.grad) for p in m1.cast_params()]
    if mode != "fga":
        for a, b in zip(g0, g1):
            assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b)
        return
    base = g1[0]._base
    assert base is not None and base.shape == (N, d) and all(g._base is base for g in g1)
    expect = torch.zeros(N, d)  # what add_mm_fp32_ does on the CPU for the stack's one GEMM, per micro-batch
    for x, dy in zip(xs, dys):
        expect.add_(torch.mm(torch.cat(dy, -1).reshape(-1, N).t().float(), x.reshape(-1, d).float()))
    assert torch.equal(base, expect)
    exact = sum(torch.cat(dy, -1).reshape(-1, N).double().t() @ x.reshape(-1, d).double() for x, dy in zip(xs, dys))
    o = 0
    for a, b in zip(g0, g1):
        t = exact[o:o + a.shape[0]]
        o += a.shape[0]
        assert rel_rms(b, a) < 1e-6 and rel_rms(b, t) < 1e-6 and rel_rms(a, t) < 1e-6
    print(f"  dW row blocks bit-identical to the separate fp32 GEMMs: {[torch.equal(a, b) for a, b in zip(g0, g1)]}")


@pytest.mark.parametrize("sizes", [KDA_SIZES, ATTN_SIZES], ids=["kda", "attention"])
def test_group_dw_member_mode(sizes):
    """fuse_proj_dw = "member": one fp32 GEMM per member into its row block of the stacked buffer, with the separate
    layers' operands read from the stacked output gradient: bit-identical to the separate layers' buffers on the CPU
    (the stacked GEMM is not: MKL sums the 1024-row blocks in another order); the forward and dx do not change."""
    d = 1024
    xs, dys = _op_batches(d, sizes, n=2)
    m0, o0, out0, dx0 = _group_run(d, sizes, False, "fga", xs, dys)
    m1, o1, out1, dx1 = _group_run(d, sizes, True, "fga", xs, dys)
    common.clear_cached_copies()
    m2, o2, out2, dx2 = _group_run(d, sizes, True, "fga_member", xs, dys)
    assert all(torch.equal(a, b) for a, b in zip(dx1, dx2))
    assert all(torch.equal(u, v) for a, b in zip(out1, out2) for u, v in zip(a, b))
    base = o2._gbuf[id(m2.cast_params()[0])]._base
    for p0, p1, p2 in zip(m0.cast_params(), m1.cast_params(), m2.cast_params()):
        g0, g1, g2 = o0._gbuf[id(p0)], o1._gbuf[id(p1)], o2._gbuf[id(p2)]
        assert g2._base is base and torch.equal(g2, g0) and rel_rms(g2, g1) < 1e-6


def test_group_grad_sink_buffers():
    """One stacked fp32 tensor whose row blocks are the members' store entries; created as zeros by the first write,
    found again by later writes, rebuilt (values kept) when an entry is no longer its row block, fresh after clear()."""
    store = {}
    s = GroupGradSink(store, ["a", "b", "c"], [(3, 4), (2, 4), (1, 4)], "cpu")
    pre = torch.randn(2, 4)
    store["b"] = pre.clone()  # a gradient that reached member b through another path earlier in this step
    buf = s.buffer()
    assert buf.shape == (6, 4) and buf.dtype == torch.float32 and all(store[k]._base is buf for k in "abc")
    assert torch.equal(buf[3:5], pre) and not buf[:3].any() and not buf[5:].any()
    assert s.buffer() is buf
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(6, 5, generator=g).to(BF16), torch.randn(5, 4, generator=g).to(BF16)
    prev = buf.clone()
    s.add_mm(a, b)
    assert torch.equal(buf, prev + torch.mm(a.float(), b.float()))
    s.per_member = True
    s.add_mm(a, b)  # per row block
    s.per_member = False
    assert rel_rms(buf, prev + 2 * a.double() @ b.double()) < 1e-6
    keep = buf.clone()
    store["c"] = torch.full((1, 4), 7.0)  # replaced by a standalone tensor: rebuilt, values carried over
    buf2 = s.buffer()
    assert buf2 is not buf and torch.equal(buf2[:5], keep[:5]) and torch.equal(buf2[5:], torch.full((1, 4), 7.0))
    store.clear()  # zero_grad
    buf3 = s.buffer()
    assert buf3 is not buf2 and not buf3.any() and len(store) == 3


# ------------------------------------------------------------------------------------------------- short convs
def _conv_case(D=48, r=16, H=4, B=2, T=40, dtype=BF16, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(B, T, 3 * D + 2 * r + H, generator=g).to(dtype)
    ws = [0.3 * torch.randn(D, 1, 4, generator=g) for _ in range(3)]
    douts = [torch.randn(B, T, s, generator=g).to(dtype) for s in (D, D, D, r, r, H)]
    return y, ws, douts, (D, D, D), (r, r, H)


def _conv_both(y0, ws0, douts, conv_sizes, sizes):
    """split_conv_silu vs one conv_silu per block on its contiguous slice (the current KDA code): outputs, dy, dw."""
    y, ws = y0.clone().requires_grad_(), [w.clone().requires_grad_() for w in ws0]
    outs = split_conv_silu(y, torch.cat(ws), conv_sizes, sizes)
    sum((o.float() * d.float()).sum() for o, d in zip(outs, douts)).backward()
    y2, ws2 = y0.clone().requires_grad_(), [w.clone().requires_grad_() for w in ws0]
    sep, a = [], 0
    for w, s in zip(ws2, conv_sizes):
        sep.append(conv_silu(y2[..., a:a + s].contiguous(), w))
        a += s
    for s in sizes:
        sep.append(y2[..., a:a + s])
        a += s
    sum((o.float() * d.float()).sum() for o, d in zip(sep, douts)).backward()
    return (outs, y.grad, [w.grad for w in ws]), (sep, y2.grad, [w.grad for w in ws2])


@pytest.mark.parametrize("dtype", [BF16, torch.float32])
def test_split_conv_silu_bit_identical_eager(dtype):
    """Eager: one depthwise conv over all 3D channels (per channel) == three ShortConvs, bit for bit, each conv block in
    the separate conv's memory layout (so the KDA recurrence downstream runs identically), the other blocks
    contiguous."""
    (o1, dy1, dw1), (o0, dy0, dw0) = _conv_both(*_conv_case(dtype=dtype))
    assert all(a.stride() == b.stride() for a, b in zip(o1[:3], o0[:3])) and all(o.is_contiguous() for o in o1[3:])
    assert all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(o1, o0))
    assert torch.equal(dy1, dy0) and all(torch.equal(a, b) for a, b in zip(dw1, dw0))


@pytest.mark.parametrize("dtype", [BF16, torch.float32])
def test_split_conv_silu_bit_identical_compiled(fuse_cpu, dtype):
    """torch.compile (CPU inductor): the per-block graph == three compiled ShortConvs, bit for bit (outputs, the
    stacked input's gradient, the conv weights' gradients); every output contiguous, as the compiled ShortConv's."""
    n0 = len(common._CLONES)
    (o1, dy1, dw1), (o0, dy0, dw0) = _conv_both(*_conv_case(dtype=dtype))
    assert not common._FAILED and len(common._CLONES) > n0
    assert all(o.is_contiguous() for o in o1) and all(o.is_contiguous() for o in o0[:3])
    assert all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(o1, o0))
    assert torch.equal(dy1, dy0) and all(torch.equal(a, b) for a, b in zip(dw1, dw0))


# ------------------------------------------------------------------------------------------------- layers
def _layer_run(layer, mode, xs, dys, monkeypatch, module):
    """2 micro-batches through one layer (bf16 x, bf16 autocast) with a bundle; captures the fused GEMM's output
    gradient (identical in both paths: the layer's forward and everything after the projections are)."""
    captured = []
    if layer.fuse_proj:
        real = common.fused_linear

        def capture(x, ws):
            y = real(x, ws)
            if y.requires_grad:
                y.register_hook(lambda g: captured.append(g.detach().clone()))
            return y
        monkeypatch.setattr(module, "fused_linear", capture)
    m = Wrap(layer)
    opt = bundle(m, mode)
    outs, dxs = [], []
    for x, dy in zip(xs, dys):
        xi = x.detach().requires_grad_()
        with torch.autocast("cpu", dtype=BF16):
            y = m(xi)
        y.backward(dy)
        outs.append(y.detach())
        dxs.append(xi.grad)
    return m, opt, outs, dxs, captured


def _check_layer(make, module, mode, monkeypatch):
    torch.manual_seed(0)
    ref = make(False)
    new = make(True)
    new.load_state_dict(ref.state_dict())
    g = torch.Generator().manual_seed(3)
    d = tiny().d_model
    xs = [torch.randn(2, 24, d, generator=g).to(BF16) for _ in range(2)]
    dys = [torch.randn(2, 24, d, generator=g).to(BF16) for _ in range(2)]
    m0, o0, out0, dx0, _ = _layer_run(ref, mode, xs, dys, monkeypatch, module)
    m1, o1, out1, dx1, cap = _layer_run(new, mode, xs, dys, monkeypatch, module)
    assert len(cap) == len(xs)
    for a, b in zip(out0, out1):
        assert torch.equal(a, b)
    W = torch.cat([p.detach().to(BF16) for p in new.proj_weights()]).double()
    for a, b, dy in zip(dx0, dx1, cap):
        exact = dy.double() @ W
        e_cur, e_new = rel_rms(a, exact), rel_rms(b, exact)
        assert e_new < e_cur and rel_rms(b, a) < 1e-2, (e_new, e_cur)
    g0, g1 = grads_of(o0, m0), grads_of(o1, m1)
    stacked = {n for n, p in m1.named_parameters() if any(p is q for q in new.proj_weights())}
    assert len(stacked) == len(new.proj_weights())
    for n in g0:
        if n in stacked and mode == "fga":  # one fp32 GEMM of the stack vs one per member: fp32 summation order
            assert rel_rms(g1[n], g0[n]) < 1e-6, n
        else:
            assert torch.equal(g1[n], g0[n]), n
    return o1


@pytest.mark.parametrize("mode", MODES + ("fga_member",))
def test_kda_layer_fused_vs_separate(mode, monkeypatch):
    """KDA (torch recurrence): output bit-identical; x's gradient closer to the exact sum of the same products; every
    parameter gradient bit-identical (the conv weights, f_up, g_up with its bias, o_proj, A_log, dt_bias, the norm),
    the stacked weights' too in bf16 and with per-member dW GEMMs, up to fp32 summation order with the stacked one."""
    cfg = tiny()
    o1 = _check_layer(lambda fp: KDA(cfg.replace(fuse_proj=fp), 0), kda_mod, mode, monkeypatch)
    assert len(o1._groups) == (0 if mode == "plain" else 1)


@pytest.mark.parametrize("mode", MODES + ("fga_member",))
@pytest.mark.parametrize("gate", [True, False])
def test_attention_layer_fused_vs_separate(mode, gate, monkeypatch):
    """Gated attention with q/k norm (and without the gate: q/k/v only)."""
    cfg = tiny(attn_out_gate=gate)
    _check_layer(lambda fp: Attention(cfg.replace(fuse_proj=fp), 0), attn_mod, mode, monkeypatch)


def test_kda_fused_graph_uses_one_gemm_node():
    """With fused accumulation the KDA layer has exactly one stacked-weight GEMM node for its six projections (the
    other bias-free layers keep theirs); the attention layer one for its four."""
    cfg = tiny(fuse_proj=True)
    for layer, n_lin in ((KDA(cfg, 0), 2), (Attention(cfg, 0), 1)):  # + f_up and o_proj / o_proj
        m = Wrap(layer)
        bundle(m, "fga")
        x = torch.randn(1, 8, cfg.d_model, requires_grad=True)
        with torch.autocast("cpu", dtype=BF16):
            names = _graph_names(m(x))
        assert names.count("_LinearAccumulateBackward") == 1 + n_lin
        common.clear_cached_copies()


# ------------------------------------------------------------------------------------------------- whole model
def _micro(m, batches, autocast=True):
    losses = []
    for x, y in batches:
        with torch.autocast("cpu", dtype=BF16, enabled=autocast):
            loss, _ = m(x, y)
        (loss / len(batches)).backward()
        losses.append(loss.detach())
    return losses


@pytest.mark.parametrize("residual,frac", [("standard", 0.0), ("block_attnres", 0.5)])
@pytest.mark.parametrize("mode", ["fga", "plain", "fga_member"])
def test_model_fused_vs_current(residual, frac, mode):
    """Whole tiny model, bf16 autocast, 3 accumulated micro-batches (frac 0.5: half the layers recomputed): losses
    bit-identical; gradients at the rounding level of the current path and, against an fp32 model with the same
    weights and batches, no farther in total."""
    torch.manual_seed(0)
    cfg = tiny(residual=residual, fuse_proj_dw="member" if mode == "fga_member" else "stacked")
    base = MoELLM(cfg)
    batches = [_batch(cfg, 0, k) for k in range(3)]
    res = {}
    for fp in (False, True):
        m = MoELLM(cfg.replace(fuse_proj=fp))
        m.load_state_dict(base.state_dict())
        m.set_checkpointing(frac)
        opt = bundle(m, mode)
        res[fp] = (_micro(m, batches), grads_of(opt, m), opt)
        if mode != "plain":
            assert all(c.grad is None for _, c in opt._cast.values())
        common.clear_cached_copies()
    (l0, g0, _), (l1, g1, o1) = res[False], res[True]
    assert all(torch.equal(a, b) for a, b in zip(l0, l1))
    assert (len(o1._groups) == 4) == (mode != "plain")
    names = [n for n in g0 if g0[n] is not None]
    assert all(g1[n] is not None for n in names)
    for n in names:
        assert rel_rms(g1[n], g0[n]) < 5e-2, (n, rel_rms(g1[n], g0[n]))
    flat0 = torch.cat([g0[n].double().flatten() for n in names])
    flat1 = torch.cat([g1[n].double().flatten() for n in names])
    assert rel_rms(flat1, flat0) < 1e-2
    ref = copy.deepcopy(base)
    ref.set_checkpointing(frac)
    _micro(ref, batches, autocast=False)
    gr = torch.cat([dict(ref.named_parameters())[n].grad.double().flatten() for n in names])
    e_new, e_cur = rel_rms(flat1, gr), rel_rms(flat0, gr)
    print(f"{residual} frac {frac} {mode}: all gradients vs the fp32 model: fused {e_new:.4e}, current {e_cur:.4e}; "
          f"fused vs current {rel_rms(flat1, flat0):.2e}")
    assert e_new <= e_cur * 1.01


def test_model_with_fusion_compiled(fuse_cpu):
    """Fusion on (torch.compile, CPU inductor): the compiled split conv and the q/k norms on the stacked output's
    views; losses bit-identical to the current path with fusion on, gradients at its rounding level."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    batches = [_batch(cfg, 5, k) for k in range(2)]
    res = {}
    for fp in (False, True):
        m = MoELLM(cfg.replace(fuse_proj=fp))
        m.load_state_dict(base.state_dict())
        opt = bundle(m, "fga")
        res[fp] = (_micro(m, batches), grads_of(opt, m))
        common.clear_cached_copies()
    assert not common._FAILED, common._FAILED
    (l0, g0), (l1, g1) = res[False], res[True]
    assert all(torch.equal(a, b) for a, b in zip(l0, l1))
    for n in g0:
        if g0[n] is not None:
            assert rel_rms(g1[n], g0[n]) < 5e-2, n


# ------------------------------------------------------------------------------------------------- training
def _train(m, opt, cfg, steps, start=0, clip=0.05, micro=2, bad_step=None):
    """train.py's loop on CPU under bf16 autocast (as tests/test_grad_acc.py): accumulate, skip a non-finite step,
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
            assert opt._gbuf and all(not torch.isfinite(g).all() for g in opt._gbuf.values())
            opt.zero_grad()
            assert not opt._gbuf and all(p.grad is None for p in m.parameters())
            continue
        out.append(float(opt.clip_grad_norm(clip)))
        opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
        opt.zero_grad()
        m.update_router_bias()
    return out


@pytest.mark.parametrize("mode", ["fga", "hook"])
def test_training_steps_track_current_path(mode):
    """3 steps (clipping active every step, bf16 Muon momentum, half the layers recomputed): the first step's losses
    are identical, gradient norms and later losses differ at the rounding level; the copies stay the bf16 rounding of
    the masters, refreshed in place in the stacked copies."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    runs = {}
    for fp in (False, True):
        m = MoELLM(cfg.replace(fuse_proj=fp))
        m.load_state_dict(base.state_dict())
        m.set_checkpointing(0.5)
        opt = bundle(m, mode)
        ptrs = [(c.data_ptr(), s.data_ptr()) for (_, s), c in
                zip(opt._groups, [opt._cast[id(g[0])][1] for g, _ in opt._groups])]
        runs[fp] = _train(m, opt, cfg, 3)
        for p, c in opt._cast.values():
            assert torch.equal(c, p.detach().to(BF16))
        for (params, s), (pc, ps) in zip(opt._groups, ptrs):
            assert s.data_ptr() == ps == pc  # refreshed in place, the first member's copy is the stack's first rows
            assert torch.equal(s, torch.cat([opt._cast[id(p)][1] for p in params]))
            a = 0
            for p in params:
                c = opt._cast[id(p)][1]
                assert c.data_ptr() == s.data_ptr() + a * s.stride(0) * s.element_size()
                a += p.shape[0]
        common.clear_cached_copies()
    a, b = runs[False], runs[True]
    assert a[:2] == b[:2] and a != b
    for x, y in zip(a, b):
        assert abs(x - y) <= 2e-3 * abs(x), (a, b)


def test_nonfinite_skip_and_buffers_reset():
    """A skipped non-finite step (zero_grad, no update) leaves nothing behind: the next step's stacked buffers are
    bit-identical to those of a run that never saw it."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", fuse_proj=True)
    base = MoELLM(cfg)
    m_bad, m_ref = copy.deepcopy(base), copy.deepcopy(base)
    o_bad = bundle(m_bad, "fga")
    _train(m_bad, o_bad, cfg, 1, bad_step=0)
    o_ref = bundle(m_ref, "fga")
    for m in (m_bad, m_ref):
        _micro(m, [_batch(cfg, 1, k) for k in range(2)])
    g_bad, g_ref = grads_of(o_bad, m_bad), grads_of(o_ref, m_ref)
    for n in g_ref:
        assert (g_bad[n] is None and g_ref[n] is None) or torch.equal(g_bad[n], g_ref[n]), n
    for params, _ in o_bad._groups:
        bufs = [o_bad._gbuf[id(p)] for p in params]
        assert all(b._base is bufs[0]._base for b in bufs)


def test_recompute_and_offload_identical_buffers():
    """Each micro-batch is accumulated exactly once into the stacked buffers: whole-layer recompute, FFN-only
    recompute and activation offloading (every saved tensor through host memory; the stacked copies are skipped as
    leaves) give bit-identical losses and buffers."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", fuse_proj=True)
    base = MoELLM(cfg)
    batches = [_batch(cfg, 2, k) for k in range(2)]
    out = []
    for frac, ffn, off in [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.5)]:
        m = copy.deepcopy(base)
        m.set_checkpointing(frac, ffn_frac=ffn, offload_frac=off)
        if m.offloader is not None:
            m.offloader.min_bytes = 0
        opt = bundle(m, "fga")
        losses = _micro(m, batches)
        if m.offloader is not None:
            assert m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0
        out.append((losses, grads_of(opt, m)))
        common.clear_cached_copies()
    l_ref, g_ref = out[0]
    for losses, g in out[1:]:
        assert all(torch.equal(a, b) for a, b in zip(losses, l_ref))
        for n in g_ref:
            assert (g[n] is None and g_ref[n] is None) or torch.equal(g[n], g_ref[n]), n


def test_resume_across_the_switch():
    """A checkpoint written without fuse_proj loads into a model and bundle with it (same keys, shapes, optimizer
    states) and training continues: the first loss after the resume is bit-identical to continuing without the
    option, the rest at the rounding level; and a fused run resumed from its own checkpoint continues exactly."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    m = MoELLM(cfg)
    opt = bundle(m, "fga")
    _train(m, opt, cfg, 2)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
    cont = _train(m, opt, cfg, 2, start=2)
    common.clear_cached_copies()
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)
    m2 = MoELLM(cfg.replace(fuse_proj=True))
    m2.load_state_dict(sd["model"])  # strict
    o2 = OptimizerBundle(m2, lr=3e-3, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=True)
    o2.load_state_dict(sd["opt"])
    o2.begin()
    assert len(o2._groups) == 4
    res = _train(m2, o2, cfg, 2, start=2)
    assert res[0] == cont[0] and res != cont
    for x, y in zip(cont, res):
        assert abs(x - y) <= 2e-3 * abs(x), (cont, res)
    buf2 = io.BytesIO()
    torch.save({"model": m2.state_dict(), "opt": o2.state_dict()}, buf2)
    assert list(m2.state_dict()) == list(sd["model"])
    full = _train(m2, o2, cfg, 1, start=4)
    common.clear_cached_copies()
    buf2.seek(0)
    sd2 = torch.load(buf2, weights_only=False)
    m3 = MoELLM(cfg.replace(fuse_proj=True))
    m3.load_state_dict(sd2["model"])
    o3 = OptimizerBundle(m3, lr=3e-3, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=True)
    o3.load_state_dict(sd2["opt"])
    o3.begin()
    assert _train(m3, o3, cfg, 1, start=4) == full


@pytest.mark.parametrize("mode", MODES)
def test_eval_and_no_grad(mode):
    """Eval / no_grad: the forward runs on the stacked copy (no gradient buffers are touched) and gives the same CE as
    the training forward and as the current path."""
    torch.manual_seed(0)
    cfg = tiny()
    base = MoELLM(cfg)
    x, y = _batch(cfg, 3, 0)
    ces = []
    for fp in (False, True):
        m = MoELLM(cfg.replace(fuse_proj=fp))
        m.load_state_dict(base.state_dict())
        opt = bundle(m, mode)
        with torch.autocast("cpu", dtype=BF16):
            _, info = m(x, y)
        ce_train = info["ce"].detach()
        m.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=BF16):
            _, info = m(x, y)
            logits = m(x)
        assert not opt._gbuf and torch.equal(info["ce"], ce_train) and logits.shape[-1] == cfg.vocab_size
        ces.append(info["ce"])
        common.clear_cached_copies()
    assert torch.equal(ces[0], ces[1])


def test_fp32_forward_without_autocast():
    """Without autocast (train.py on the CPU) the fp32 weights are stacked: losses within fp32 rounding of the
    current path, gradients in .grad as usual."""
    torch.manual_seed(0)
    cfg = tiny()
    base = MoELLM(cfg)
    batches = [_batch(cfg, 4, k) for k in range(2)]
    res = {}
    for fp in (False, True):
        m = MoELLM(cfg.replace(fuse_proj=fp))
        m.load_state_dict(base.state_dict())
        opt = bundle(m, "fga")
        res[fp] = (_micro(m, batches, autocast=False), {n: p.grad for n, p in m.named_parameters()})
        assert not opt._gbuf
        common.clear_cached_copies()
    (l0, g0), (l1, g1) = res[False], res[True]
    for a, b in zip(l0, l1):
        assert abs(float(a) - float(b)) < 1e-5
    for n in g0:
        if g0[n] is not None:
            assert rel_rms(g1[n], g0[n]) < 1e-4, n


def _bins(root):
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


def test_train_py_override(tmp_path):
    """train.py --override '{"fuse_proj": true}' with --offload-master --fused-grad-acc (on CPU train.py runs without
    autocast: this checks the plumbing, the config file and the log line)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    data = str(tmp_path / "bins")
    _bins(data)
    out = tmp_path / "run"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny", "--data", data, "--canon",
           "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--seq-len", "32", "--micro-bsz", "2",
           "--global-tokens", "128", "--steps", "2", "--warmup", "1", "--log-every", "1", "--eval-every", "0",
           "--ckpt-every", "0", "--offload-master", "--fused-grad-acc", "--override", json.dumps({"fuse_proj": True}),
           "--out", str(out)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "projections: fused, 4 groups" in r.stdout and "4 stacked persistent copies" in r.stdout, r.stdout[-2000:]
    assert "TRAIN_DONE" in r.stdout
    assert json.load(open(out / "config.json"))["fuse_proj"] is True


# ------------------------------------------------------------------------------------------------- GPU
def _count_sync_warnings(fn):
    """fn() under torch.cuda.set_sync_debug_mode("warn"): (result, list of "synchronizing CUDA operation" warnings), as
    tests/test_gpu.py counts them."""
    import warnings
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
@pytest.mark.parametrize("sizes", [KDA_SIZES, ATTN_SIZES], ids=["kda", "attention"])
def test_gpu_fused_gemm_bitwise_report(sizes):
    """main_2b shapes (2 x 4096 tokens, d 1024): reports whether cuBLAS gives bit-identical blocks for the stacked
    forward GEMM and bit-identical row blocks for the stacked fp32 dW GEMM (and for per-member dW GEMMs on the stacked
    gradient, fuse_proj_dw = "member"); asserts the rounding level, and that the one-GEMM dx is closer to the exact sum
    than the current per-member GEMMs + bf16 adds. No host sync in any of them."""
    torch.manual_seed(0)
    M, d, N = 8192, 1024, sum(sizes)
    x = torch.randn(M, d, device="cuda").to(BF16)
    ws = [(torch.randn(n, d, device="cuda") * 0.02).to(BF16) for n in sizes]
    W = torch.cat(ws)
    dy = (torch.randn(M, N, device="cuda") * 0.01).to(BF16)
    dys = [t.contiguous() for t in dy.split(sizes, -1)]  # the current path's separate output gradients
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        y = F.linear(x, W)
        sep = [F.linear(x, w) for w in ws]
        dx_new = dy.mm(W)
        dx_cur = None
        for g, w in zip(dys, ws):  # autograd adds the partial dx in bf16
            t = g.mm(w)
            dx_cur = t if dx_cur is None else dx_cur + t
        stack = torch.zeros(N, d, device="cuda")
        common.add_mm_fp32_(stack, dy.t(), x)
        bufs = [torch.zeros(n, d, device="cuda") for n in sizes]
        for b, g in zip(bufs, dys):
            common.add_mm_fp32_(b, g.t(), x)
        per = torch.zeros(N, d, device="cuda")
        o = 0
        for n in sizes:
            common.add_mm_fp32_(per[o:o + n], dy[:, o:o + n].t(), x)
            o += n
    finally:
        torch.cuda.set_sync_debug_mode("default")
    blocks = y.split(sizes, -1)
    fwd = [torch.equal(a, b) for a, b in zip(blocks, sep)]
    fwd_frac = [float((a == b).float().mean()) for a, b in zip(blocks, sep)]
    fwd_err = max(rel_rms(a, b) for a, b in zip(blocks, sep))
    exact = dy.double() @ W.double()
    e_new, e_cur = rel_rms(dx_new, exact), rel_rms(dx_cur, exact)
    dw = [torch.equal(a, b) for a, b in zip(stack.split(sizes, 0), bufs)]
    dw_per = [torch.equal(a, b) for a, b in zip(per.split(sizes, 0), bufs)]
    dw_err = max(rel_rms(a, b) for a, b in zip(stack.split(sizes, 0), bufs))
    print(f"\nN {N}: forward blocks bit-identical {fwd} (fraction of equal elements "
          f"{[round(f, 6) for f in fwd_frac]}, max rel. RMS diff {fwd_err:.2e}); dx rel. RMS error vs exact: one GEMM "
          f"{e_new:.3e}, separate + bf16 adds {e_cur:.3e}; dW row blocks bit-identical: one GEMM {dw}, per member "
          f"{dw_per} (max rel. RMS diff {dw_err:.2e})")
    assert fwd_err < 1e-2 and e_new < 0.75 * e_cur and dw_err < 1e-5


def _gpu_layer(kind, fp):
    cfg = main_2b().replace(fuse_proj=fp)
    torch.manual_seed(0)
    return (KDA(cfg, 0) if kind == "kda" else Attention(cfg, 3)).cuda()


@gpu
@pytest.mark.parametrize("kind", ["kda", "attention"])
def test_gpu_layer_main_size_fused_vs_separate(kind):
    """One main_2b layer (2 x 4096 tokens, bf16 x, fusion on, persistent copies + fused accumulation as in the main
    run), 2 micro-batches: reports whether the output and every non-stacked parameter gradient are bit-identical to
    the separate projections; asserts the rounding level, that x's gradient is closer to the exact sum of the products
    of each variant's own projection output gradients, and that the fused path adds no host syncs."""
    common.set_fuse(True)
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(2, 4096, 1024, generator=g).to("cuda", BF16) for _ in range(2)]
    dys = [(torch.randn(2, 4096, 1024, generator=g) * 0.01).to("cuda", BF16) for _ in range(2)]
    mod = kda_mod if kind == "kda" else attn_mod
    res = {}
    for fp in (False, True):
        layer = _gpu_layer(kind, fp)
        m = Wrap(layer)
        opt = bundle(m, "fga")
        members = {id(lin.weight): lin for lin in layer.modules() if isinstance(lin, CastLinear)}
        members = [members[id(w)] for w in layer.proj_weights()]
        grads = {}  # member index (or "all") -> output gradient of the current micro-batch
        real = mod.fused_linear

        def capture(x, ws):
            y = real(x, ws)
            if y.requires_grad:
                y.register_hook(lambda gr: grads.__setitem__("all", gr))
            return y
        def grad_hook(i):
            def hook(mod_, inp, out):  # forward hook of member i (separate layers): keep its output's gradient
                if out.requires_grad:
                    out.register_hook(lambda gr: grads.__setitem__(i, gr))
            return hook
        handles = [lin.register_forward_hook(grad_hook(i)) for i, lin in enumerate(members)]
        mod.fused_linear = capture
        try:
            for x, dy in zip(xs, dys):  # warm-up: compile, autotune
                xi = x.detach().requires_grad_()
                with torch.autocast("cuda", dtype=BF16):
                    m(xi).backward(dy)
            opt.zero_grad()
            torch.cuda.synchronize()
            outs, dxs, dYs, syncs = [], [], [], 0
            for x, dy in zip(xs, dys):
                xi = x.detach().requires_grad_()
                grads.clear()
                with torch.autocast("cuda", dtype=BF16):
                    y, hits = _count_sync_warnings(lambda: m(xi))
                _, hits2 = _count_sync_warnings(lambda: y.backward(dy))
                syncs += len(hits) + len(hits2)
                outs.append(y.detach())
                dxs.append(xi.grad)
                dYs.append(grads["all"] if fp else torch.cat([grads[i] for i in range(len(members))], -1))
            torch.cuda.synchronize()
        finally:
            mod.fused_linear = real
            for h in handles:
                h.remove()
        W = torch.cat([w.detach().to("cuda", BF16) for w in layer.proj_weights()]).double()  # masters are on the CPU
        errs = [rel_rms(dx.reshape(-1, W.shape[1]), dY.double().reshape(-1, W.shape[0]) @ W) for dx, dY in zip(dxs, dYs)]
        stacked = {n for n, p in m.named_parameters() if any(p is w for w in layer.proj_weights())}
        res[fp] = (outs, grads_of(opt, m), errs, syncs, stacked)
        common.clear_cached_copies()
        del m, opt, layer
    (o0, g0, e0, s0, _), (o1, g1, e1, s1, stacked) = res[False], res[True]
    same_out = [torch.equal(a, b) for a, b in zip(o0, o1)]
    same_grads = {n: torch.equal(g0[n], g1[n]) for n in g0 if n not in stacked and g0[n] is not None}
    print(f"\n{kind}: output bit-identical {same_out}; non-stacked gradients bit-identical "
          f"{sum(same_grads.values())}/{len(same_grads)} (differ: {[n for n, v in same_grads.items() if not v]}); "
          f"dx rel. RMS error vs the exact sum: fused {e1[0]:.3e}, separate {e0[0]:.3e}; host syncs in 2 micro-batches "
          f"fwd+bwd: separate {s0}, fused {s1}")
    # the compiled split conv ran (its eager fallback is correct but slower than the current compiled path)
    assert not [k for k in common._FAILED if k[0] == "split_conv_silu"], common._FAILED
    for a, b in zip(o0, o1):
        assert rel_rms(b, a) < 1e-2
    for n in g0:
        if g0[n] is not None:
            assert rel_rms(g1[n], g0[n]) < 5e-2, n
    assert all(a < 0.9 * b for a, b in zip(e1, e0)) and s1 <= s0


@gpu
def test_gpu_model_training_fused_vs_current():
    """Tiny model on the GPU with the main run's paths (offload_master + offload_states + fused accumulation, triton
    experts, block AttnRes, bf16 residual, fusion on), 4 steps of 2 micro-batches: losses track the current path
    (the forward matches; GPU atomics elsewhere rule out bit-exact runs); the copies never hold a .grad; the stacked
    copies stay the masters' bf16 rounding; peak memory not materially higher."""
    torch.manual_seed(0)
    common.set_fuse(True)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True, residual="block_attnres",
               moe_impl="triton")
    base = MoELLM(cfg).cuda()
    runs, peaks = {}, {}
    for fp in (False, True):
        m = MoELLM(cfg.replace(fuse_proj=fp)).cuda()
        m.load_state_dict(base.state_dict())
        opt = OptimizerBundle(m, lr=1e-3, muon_state_dtype=BF16, offload_states=True, offload_master=True,
                              fused_grad_acc=True)
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
        for params, s in opt._groups:
            assert torch.equal(s, torch.cat([p.detach().to("cuda", BF16) for p in params]))
        peaks[fp] = torch.cuda.max_memory_allocated()
        runs[fp] = ces
        common.clear_cached_copies()
        del m, opt
    print(f"\nces current {runs[False]}\nces fused   {runs[True]}\npeak MB {peaks[False] / 2**20:.1f} -> "
          f"{peaks[True] / 2**20:.1f}")
    assert abs(runs[False][0] - runs[True][0]) < 1e-3 * abs(runs[False][0])
    for a, b in zip(runs[False], runs[True]):
        assert abs(a - b) < 2e-2 * abs(a)
    assert peaks[True] <= 1.03 * peaks[False] + 8 * 2 ** 20
