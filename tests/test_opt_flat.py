"""Round 11: OptimizerBundle(flat=True) (train.py --opt-flat): masters, their gradient buffers, the bf16 copies and the
offloaded optimizer states in flat buffers laid out in update order, moved with one copy per buffer and update segment,
Muon's per-member elementwise ops as one op per chunk, the bf16 refresh as one copy.
CPU: the layout (alignment, runs), Muon's chunk ops against its member-by-member ops (bit-identical), the flat gradient
store with GradSink and the copies' hooks (the first gradient lands exactly as before, -0.0 included).
GPU (skipped without CUDA): a flat bundle, the live setting (offload + state_min_bytes) and a plain bundle fed the same
gradients stay bit-identical for 4 steps with the side stream stalled and fresh buffers NaN-filled (parameters, states,
bf16 copies, clipping norms), the flat path really runs (no fallback after the first step), and a checkpoint of the flat
bundle resumes into a new flat bundle that continues bit-identically."""
import copy
import math

import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.model import MoELLM
from moe_llm.optim import Muon, OptimizerBundle, _FlatGradStore, _FlatSpace, _GroupTensors, _Pinned

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


# ------------------------------------------------------------------------------------------------------------ CPU
def test_flat_space_layout():
    groups = [[("a", (3, 5))], [("b", (4, 64)), ("c", (4, 64)), ("d", (4, 64))], [("e", (7,))], []]
    sp = _FlatSpace(groups)
    off = {k: sp.slots[k][0] for k in "abcde"}
    assert off["a"] == 0 and off["b"] == 256 and off["c"] == 512 and off["d"] == 768 and off["e"] == 1024
    assert sp.numel % _FlatSpace.ALIGN == 0 and sp.numel >= 1031
    buf = torch.arange(sp.numel, dtype=torch.float32)
    run = sp.run(buf, ["b", "c", "d"])
    assert run is not None and run.numel() == 768 and run[0] == 256
    assert sp.run(buf, ["c", "b"]) is None and sp.run(buf, ["b", "d"]) is None and sp.run(buf, ["a", "b"]) is None
    assert sp.run(buf, ["zz"]) is None
    assert sp.span(["a", "e"]) == (0, 1031) and sp.span(["zz"]) is None
    v = sp.view(buf, "c")
    assert v.shape == (4, 64) and v[0, 0] == 512 and sp.is_view(v, buf, "c") and not sp.is_view(v, buf, "b")
    assert not sp.is_view(v.clone(), buf, "c")
    with pytest.raises(ValueError):
        _FlatSpace([[("a", (1,))], [("a", (2,))]])


def test_pinned_blocks_place_groups_inside_blocks_and_round_trip():
    """_Pinned with 1024-element blocks: a group that would straddle a block starts the next one, groups contiguous in
    the layout and a block are copied as one run, and copy_out / copy_in give every tensor back exactly."""
    # layout (256-element alignment): a [0, 100), b c [256, 456), d [512, 1412), e [1536, 1596)
    sp = _FlatSpace([[("a", (100,))], [("b", (2, 50)), ("c", (2, 50))], [("d", (900,))], [("e", (60,))]])
    pin = _Pinned(sp, torch.float32, pin=False, block_bytes=1024 * 4)
    for g, (start, end, _) in enumerate(sp.groups):
        lo = start + pin.delta[g]
        assert lo // pin.block_elems == (lo + end - start - 1) // pin.block_elems, g  # inside one block
    assert pin.delta[:2] == [0, 0] and pin.delta[2] == 512  # d would have straddled: it starts block 1
    runs = pin.runs()
    assert [(r[0], r[1], r[2]) for r in runs] == [(0, 456, 0), (512, 1412, 1), (1536, 1596, 2)]
    gpu = torch.randn(sp.numel)
    pin.copy_out(gpu, runs)
    for k in "abcde":
        assert torch.equal(pin.view(k), sp.view(gpu, k)) and pin.is_view(pin.view(k), k)
        assert not pin.is_view(pin.view(k).clone(), k)
    back = torch.full_like(gpu, -1.0)
    pin.copy_in(back, pin.runs(["b", "c", "e"]))  # only the groups of b, c and e
    for k in "bce":
        assert torch.equal(sp.view(back, k), sp.view(gpu, k))
    assert (sp.view(back, "a") == -1).all() and (sp.view(back, "d") == -1).all()
    assert pin.nbytes == (1024 + 1024 + 256) * 4
    with pytest.raises(ValueError):
        _Pinned(_FlatSpace([[("x", (2000,))]]), torch.float32, pin=False, block_bytes=1024 * 4)


def test_group_tensors_views_runs_and_pinned_group_copies():
    """_GroupTensors: one tensor per group; views, runs (only within one group, members back to back) and is_view match
    the layout; a group copied to / from its pinned slice round-trips exactly."""
    sp = _FlatSpace([[("a", (100,))], [("b", (2, 50)), ("c", (2, 50))], [("d", (900,))]])
    gt = _GroupTensors(sp, torch.float32, "cpu")
    assert [t.numel() for t in gt.t] == [100, 200, 900]
    for t in gt.t:
        t.copy_(torch.randn(t.numel()))
    assert gt.view("c").shape == (2, 50) and gt.is_view(gt.view("c"), "c") and not gt.is_view(gt.view("b"), "c")
    run = gt.run(["b", "c"])
    assert run is not None and run.numel() == 200 and run.data_ptr() == gt.t[1].data_ptr()
    assert gt.run(["c", "b"]) is None and gt.run(["a", "b"]) is None and gt.run(["zz"]) is None
    assert torch.equal(gt.run(["c"]), gt.view("c").reshape(-1))
    pin = _Pinned(sp, torch.float32, pin=False, block_bytes=1024 * 4)
    for g in range(3):
        pin.group_slice(g).copy_(gt.t[g])
    back = _GroupTensors(sp, torch.float32, "cpu")
    for g in range(3):
        back.t[g].copy_(pin.group_slice(g))
    for k in "abcd":
        assert torch.equal(back.view(k), gt.view(k)) and torch.equal(pin.view(k), gt.view(k))


def _muon_pair(nesterov):
    """Two Muon optimizers over identical parameters: a plain one and one whose parameters, gradients and momentum
    buffers are views of three flat buffers (chunk members back to back), plus its flat(ps) resolver."""
    torch.manual_seed(0)
    shapes = [(64, 96)] * 3 + [(96, 64)] * 2 + [(32, 32)] + [(48, 48)]  # two stacked chunks and two single ones
    ref = [torch.nn.Parameter(torch.randn(s)) for s in shapes]
    flat_p = [torch.nn.Parameter(p.detach().clone()) for p in ref]
    kw = dict(lr=0.02, weight_decay=0.1, state_dtype=BF16, impl="batched_fused", nesterov=nesterov)
    oa, ob = Muon(ref, **kw), Muon(flat_p, **kw)
    chunks = ob._chunks(0)
    assert sorted(len(c) for c in chunks) == [1, 1, 2, 3]
    sp = _FlatSpace([[(id(p), p.shape) for p in c] for c in chunks])
    P = torch.zeros(sp.numel)
    G = torch.zeros(sp.numel)
    M = torch.zeros(sp.numel, dtype=BF16)
    for p in flat_p:
        sp.view(P, id(p)).copy_(p.detach())
        p.data = sp.view(P, id(p))
        ob.state[p]["buf"] = sp.view(M, id(p))
    calls = []

    def flat(ps):
        keys = [id(p) for p in ps]
        runs = sp.run(G, keys), sp.run(M, keys), sp.run(P, keys)
        ok = all(r is not None for r in runs) and all(
            sp.is_view(p.grad, G, id(p)) and sp.is_view(p.data, P, id(p)) and sp.is_view(ob.state[p]["buf"], M, id(p))
            for p in ps)
        calls.append(ok)
        return runs if ok else None
    return ref, flat_p, oa, ob, sp, G, flat, calls


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_chunk_ops_equal_member_ops(nesterov):
    ref, flat_p, oa, ob, sp, G, flat, calls = _muon_pair(nesterov)
    for step in range(3):
        for a, b in zip(ref, flat_p):
            g = torch.randn(a.shape) * (step + 1)
            a.grad = g.clone()
            v = sp.view(G, id(b))
            v.copy_(g)
            b.grad = v
        oa.step()
        ob.step(flat=flat)
        for a, b in zip(ref, flat_p):
            assert torch.equal(a.detach(), b.detach()), step
            assert torch.equal(oa.state[a]["buf"], ob.state[b]["buf"]), step
    assert calls and all(calls)  # the chunk ops ran for both stacked chunks every step
    assert len(calls) == 3 * 2


def test_flat_grad_store_with_gradsink_and_first_copy():
    sp = _FlatSpace([[("w", (2, 3))], [("u", (4,))]])
    buf = torch.zeros(sp.numel)
    store = _FlatGradStore(sp, buf)
    sink = common.GradSink(store, "w", (2, 3), "cpu")
    b = sink.buffer()
    assert store["w"] is b and sp.is_view(b, buf, "w") and torch.equal(b, torch.zeros(2, 3))
    b.add_(torch.ones(2, 3))
    assert sink.buffer() is b and buf[:6].sum() == 6
    # a key outside the layout: a fresh tensor, as without the store
    other = common.GradSink(store, "x", (5,), "cpu").buffer()
    lo, hi = buf.data_ptr(), buf.data_ptr() + buf.numel() * buf.element_size()
    assert other.shape == (5,) and not lo <= other.data_ptr() < hi and store["x"] is other
    # the copies' hook path: the first gradient is copied into the zero view (-0.0 stays -0.0, as g.float() keeps it)
    g = torch.tensor([-0.0, 1.5, -2.0, 0.0], dtype=BF16)
    v = store.zero_view("u")
    v.copy_(g)
    assert torch.equal(v.view(torch.int32), g.float().view(torch.int32))
    store.clear()
    assert "w" not in store and store.zero_view("nope") is None


# ------------------------------------------------------------------------------------------------------------ GPU
def _gpu_cfg(**kw):
    return tiny(kda_heads=2, kda_head_dim=64, d_model=128, vocab_size=65536, residual_bf16=True,
                residual="block_attnres", moe_impl="triton", **kw)


def _batch(cfg, s, k, n=129):
    g = torch.Generator().manual_seed(1000 * s + k)
    x = torch.randint(0, cfg.vocab_size, (2, n), generator=g)
    return x


def _grads(opt, m):
    return [opt._gbuf.get(id(p), p.grad) if id(p) in opt._cast else p.grad for p in m.parameters()]


def _give(opt, m, grads):
    """The gradients into opt's buffers: the flat store's slots when it has them (as GradSink / the hooks fill them)."""
    for p, g in zip(m.parameters(), grads):
        if g is None:
            continue
        if id(p) in opt._cast:
            zero_view = getattr(opt._gbuf, "zero_view", None)
            v = zero_view(id(p)) if zero_view is not None else None
            if v is not None:
                v.copy_(g)
            else:
                opt._gbuf[id(p)] = g.clone()
        else:
            p.grad = g.clone()


def _nan_empty(real_empty):
    def empty(*args, **kw):
        t = real_empty(*args, **kw)
        if t.is_cuda and t.is_floating_point():
            t.fill_(float("nan"))
        return t
    return empty


def _equal(models, opts, where):
    torch.cuda.synchronize()
    for i, group in enumerate(zip(*[list(m.parameters()) for m in models])):
        ref = group[0].detach().cpu()
        assert torch.isfinite(ref).all(), (where, i)
        for other in group[1:]:
            assert torch.equal(ref, other.detach().cpu()), (where, i)
    sds = [o.state_dict()["opts"] for o in opts]
    for group in zip(*sds):
        for pi in group[0]["state"]:
            for key, v in group[0]["state"][pi].items():
                if torch.is_tensor(v):
                    for other in group[1:]:
                        assert torch.equal(v.cpu(), other["state"][pi][key].cpu()), (where, pi, key)
    for o in opts[1:]:
        for (_, ca), (_, cb) in zip(opts[0]._cast.values(), o._cast.values()):
            assert torch.equal(ca, cb), where


@gpu
def test_gpu_flat_bit_identical_and_resumes(monkeypatch):
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_layers=())
    base = MoELLM(cfg).cuda()
    base.set_checkpointing(0.5)
    ms = [copy.deepcopy(base) for _ in range(3)]
    kw = dict(lr=1e-3, muon_state_dtype=BF16, muon_impl="batched_fused")
    live = dict(offload_states=True, offload_master=True, fused_grad_acc=True, state_min_bytes=1 << 20)
    oa = OptimizerBundle(ms[0], **live, **kw)
    ob = OptimizerBundle(ms[1], **live, flat=True, **kw)
    oc = OptimizerBundle(ms[2], **kw)
    for o in (oa, ob, oc):
        o.begin()
    assert ob.flat and not oa.flat and isinstance(ob._gbuf, _FlatGradStore)
    runs, fallbacks = [], []
    real_runs, real_reload = ob._flat_runs, ob._reload_pipelined

    def count_runs(ps):
        r = real_runs(ps)
        runs.append(r is not None)
        return r

    def count_fallback():
        fallbacks.append(len(runs))
        return real_reload()
    monkeypatch.setattr(ob, "_flat_runs", count_runs)
    monkeypatch.setattr(ob, "_reload_pipelined", count_fallback)
    real_empty = torch.empty
    saved = None

    def one_step(step, models, opts):
        for k in range(2):
            x = _batch(cfg, step, k).cuda()
            with torch.autocast("cuda", dtype=BF16):
                loss, _ = models[0](x[:, :-1], x[:, 1:])
            (loss / 2).backward()
        g = [None if t is None else t.clone() for t in _grads(opts[0], models[0])]
        for o, m in zip(opts[1:], models[1:]):
            _give(o, m, g)
        norms = [o.clip_grad_norm(0.05) for o in opts]
        assert all(torch.equal(norms[0], n) for n in norms[1:]), step
        monkeypatch.setattr(torch, "empty", _nan_empty(real_empty))
        try:
            for o in opts:
                if o.offload_states:
                    with torch.cuda.stream(o._side_stream()):
                        torch.cuda._sleep(100_000_000)  # copies late: every reader must wait for its event
                o.step(1e-3)
        finally:
            monkeypatch.setattr(torch, "empty", real_empty)
        for o in opts:
            o.zero_grad()
        models[0].update_router_bias()  # the only model that runs forward passes (the others get its gradients)

    try:
        for step in range(4):
            one_step(step, ms, (oa, ob, oc))
            _equal(ms, (oa, ob, oc), step)
            if step == 1:  # a checkpoint of the flat bundle (as train.py saves it), to resume from below; the
                ob.sync()  # weights are equal in all models, the router biases are up to date in ms[0] only
                saved = (copy.deepcopy({k: v.detach().cpu() for k, v in ms[0].state_dict().items()}),
                         copy.deepcopy(ob.state_dict()))
        assert ob._sspaces and len(ob._segments) > 1
        assert ob._rspace  # the tiny model's Muon momentum is below state_min_bytes: the resident layout holds it
        assert not fallbacks, f"flat reload fell back to the per-tensor path after {fallbacks} chunk calls"
        assert runs and sum(runs) >= len(runs) // 2  # the chunk ops ran (step 1 has no flat states yet)
        # resume: the step-1 checkpoint of the flat bundle, into a new flat bundle and into the live setting, fed the
        # same gradients: bit-identical (GPU backward passes are not bit-reproducible, so the reference is resumed too)
        model_sd, opt_sd = saved
        common.clear_cached_copies()
        ref, m2 = copy.deepcopy(base), copy.deepcopy(base)
        ref.load_state_dict(model_sd)
        m2.load_state_dict(model_sd)
        ra = OptimizerBundle(ref, **live, **kw)
        o2 = OptimizerBundle(m2, **live, flat=True, **kw)
        ra.load_state_dict(copy.deepcopy(opt_sd))
        o2.load_state_dict(copy.deepcopy(opt_sd))
        assert not o2._pinned and not o2._moved  # flat requested: the loaded states wait on the GPU for begin()
        ra.begin()
        o2.begin()
        assert o2.flat and o2._sspaces  # the loaded states went into the layout at begin()
        assert not o2._pinned  # straight from the GPU: no per-tensor pinned buffers kept in PyTorch's cache (switch 19)
        _equal([ref, m2], (ra, o2), "resumed")
        for step in (2, 3):
            one_step(step, [ref, m2], (ra, o2))
            _equal([ref, m2], (ra, o2), f"resume {step}")
        print(f"flat: {len(ob._segments)} segments, {sum(runs)}/{len(runs)} chunk updates as one op per chunk, "
              f"layout {ob._fspace.numel} elements, {len(ob._sspaces)} state buffers")
    finally:
        monkeypatch.setattr(torch, "empty", real_empty)
