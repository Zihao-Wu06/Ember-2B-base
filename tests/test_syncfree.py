"""Host-sync-free hot path (switch 4): the MoE token counts (scatter-add instead of torch.bincount), the capacity
read-back (non-blocking copy + event, passed into the expert recompute as an int), the cached CPU copies of the
Engram hash buffers, and the compile-worker release on log steps. All values must be bit-identical to the code
before the change, which is kept here verbatim as the reference. The CUDA side (no implicit syncs, the read-back
waits for the value, pinned Engram copies) is in tests/test_gpu.py."""
import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from moe_llm.config import tiny
from moe_llm.layers.common import cast_weight, release_compile_workers, silu_mul
from moe_llm.layers.engram import EngramMemory
from moe_llm.layers.moe import MaxReadback, MoE, capacity_bucket, expert_counts


# ----------------------------------------------------------------------------- reference: the code before
def _old_bmm(moe, xs, e_sorted, counts):
    E = moe.E
    C = int(counts.max())
    C = max(64, (C + 63) // 64 * 64)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(xs.shape[0], device=xs.device) - starts[e_sorted]
    buf = xs.new_zeros(E, C, moe.d)
    buf.index_put_((e_sorted, pos), xs)
    dev = buf.device.type
    wdt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else buf.dtype
    h = silu_mul(torch.bmm(buf, cast_weight(moe.w_gate, wdt)), torch.bmm(buf, cast_weight(moe.w_up, wdt)))
    out = torch.bmm(h, cast_weight(moe.w_down, wdt))
    return out[e_sorted, pos]


def _old_experts(moe, x, idx, w):
    N, K = idx.shape
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    e_sorted = flat[order]
    tok = order // K
    counts = torch.bincount(flat, minlength=moe.E)
    xs = x.index_select(0, tok)
    if moe.recompute_experts and moe.training and torch.is_grad_enabled():
        ys = checkpoint(lambda a, b, c: _old_bmm(moe, a, b, c), xs, e_sorted, counts, use_reentrant=False,
                        preserve_rng_state=False)
    else:
        ys = _old_bmm(moe, xs, e_sorted, counts)
    ys = ys * w.reshape(-1)[order].unsqueeze(-1).to(ys.dtype)
    return torch.zeros(N, moe.d, device=x.device, dtype=ys.dtype).index_add_(0, tok, ys)


def _old_forward(moe, x):
    B, T, d = x.shape
    xf = x.reshape(-1, d)
    scores, idx, w = moe.route(xf)
    if moe.training:
        with torch.no_grad():
            moe.load += torch.bincount(idx.flatten(), minlength=moe.E).float()
    aux = (moe.seq_aux_loss(scores, idx, B, T) if moe.training and moe.seq_aux_coef > 0
           else scores.new_zeros(()))
    y = _old_experts(moe, xf, idx, w)
    if moe.shared is not None:
        y = y + moe.shared(xf).to(y.dtype)
    return y.view(B, T, d).to(x.dtype), aux


# ----------------------------------------------------------------------------- counts and capacity
@pytest.mark.parametrize("kind", ["random", "some_empty", "collapsed", "no_tokens"])
def test_expert_counts_equal_bincount(kind):
    E = 16
    g = torch.Generator().manual_seed(3)
    if kind == "random":
        flat = torch.randint(0, E, (4096,), generator=g)
    elif kind == "some_empty":  # experts 5..15 get nothing
        flat = torch.randint(0, 5, (999,), generator=g)
    elif kind == "collapsed":
        flat = torch.full((777,), 9, dtype=torch.long)
    else:
        flat = torch.zeros(0, dtype=torch.long)
    got, ref = expert_counts(flat, E), torch.bincount(flat, minlength=E)
    assert got.dtype == ref.dtype == torch.long and torch.equal(got, ref)


def test_capacity_readback_and_bucket_match_the_old_formula():
    for c in (0, 1, 63, 64, 65, 127, 128, 1000):
        assert capacity_bucket(c) == max(64, (c + 63) // 64 * 64)
    t = torch.tensor([3, 0, 900, 17])
    assert MaxReadback(t).get() == 900 and isinstance(MaxReadback(t).get(), int)


# ----------------------------------------------------------------------------- MoE layer, bit for bit
def _moe_pair(skew):
    torch.manual_seed(0)
    cfg = tiny()
    moe = MoE(cfg, 1).train()
    assert moe.impl in ("auto", "bmm")
    if skew:  # bias: every token picks experts 0 and 1, experts 2..7 stay empty
        moe.e_bias[:2] = 10.0
    return moe, copy.deepcopy(moe)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_moe_layer_bit_identical_to_the_old_code(recompute, amp, skew):
    new, old = _moe_pair(skew)
    new.recompute_experts = old.recompute_experts = recompute
    x0 = torch.randn(2, 40, new.d, generator=torch.Generator().manual_seed(7))
    outs = []
    for moe, fwd in ((new, lambda m, x: m(x)), (old, _old_forward)):
        x = x0.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            y, aux = fwd(moe, x)
        (y.float().square().mean() + aux).backward()
        grads = {n: p.grad for n, p in moe.named_parameters()}
        outs.append((y.detach(), aux.detach(), moe.load.clone(), x.grad, grads))
    (y1, a1, l1, gx1, g1), (y2, a2, l2, gx2, g2) = outs
    assert torch.equal(y1, y2) and torch.equal(a1, a2) and torch.equal(l1, l2) and torch.equal(gx1, gx2)
    assert g1.keys() == g2.keys()
    for n in g1:
        assert (g1[n] is None) == (g2[n] is None), n
        assert g1[n] is None or torch.equal(g1[n], g2[n]), n
    if skew:
        assert (l1[2:] == 0).all() and (l1[:2] > 0).all()  # the empty-expert path was exercised


def test_experts_without_counts_computes_them():
    moe, _ = _moe_pair(False)
    x = torch.randn(30, moe.d)
    _, idx, w = moe.route(x)
    with torch.no_grad():
        a = moe.experts(x, idx, w)
        b = moe.experts(x, idx, w, expert_counts(idx.reshape(-1), moe.E))
        c = _old_experts(moe, x, idx, w)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_bmm_direct_call_still_reads_capacity_itself():
    moe, _ = _moe_pair(True)
    xs = torch.randn(50, moe.d)
    e_sorted = torch.sort(torch.randint(0, moe.E, (50,), generator=torch.Generator().manual_seed(1))).values
    counts = torch.bincount(e_sorted, minlength=moe.E)
    with torch.no_grad():
        assert torch.equal(moe._bmm(xs, e_sorted, counts), _old_bmm(moe, xs, e_sorted, counts))
        assert torch.equal(moe._routed(xs, e_sorted, counts), _old_bmm(moe, xs, e_sorted, counts))


# ----------------------------------------------------------------------------- Engram prefetch buffers
def _engram():
    torch.manual_seed(0)
    return EngramMemory(tiny(engram_offload=True))


def _ids(seed=0):
    return torch.randint(0, 512, (2, 24), generator=torch.Generator().manual_seed(seed))


def test_engram_prefetch_matches_direct_lookup_and_reuses_cpu_buffers():
    mem = _engram()
    ids = _ids()
    with torch.no_grad():
        direct = mem(ids)
        mem.prefetch(ids)
        pre = mem(ids)
    assert direct.keys() == pre.keys() and all(torch.equal(direct[k], pre[k]) for k in direct)
    a = mem._cpu_buffers()
    assert mem._cpu_buffers() is a  # unchanged buffers: no new copies
    for buf, cpu in zip((mem.canon, mem.mult, mem.mods, mem.offs), a):
        assert torch.equal(buf, cpu)


def test_engram_cpu_buffers_refresh_after_set_canon_map_and_load_state_dict():
    mem = _engram()
    ids = _ids(1)
    a = mem._cpu_buffers()
    canon = torch.arange(512).flip(0)  # a different (valid) map
    mem.set_canon_map(canon.numpy())
    b = mem._cpu_buffers()
    assert b is not a and torch.equal(b[0], canon)
    with torch.no_grad():
        direct = mem(ids)
        mem.prefetch(ids)
        pre = mem(ids)
    assert all(torch.equal(direct[k], pre[k]) for k in direct)  # the prefetch used the new map
    other = _engram()
    other.set_canon_map((torch.arange(512) % 256).numpy())
    mem.load_state_dict(other.state_dict())
    c = mem._cpu_buffers()
    assert c is not b and torch.equal(c[0], other.canon)
    with torch.no_grad():
        mem.prefetch(ids)
        pre2 = mem(ids)
        direct2 = mem(ids)
    assert all(torch.equal(direct2[k], pre2[k]) for k in direct2)


# ----------------------------------------------------------------------------- compile workers
def test_release_compile_workers_is_safe_to_call():
    assert release_compile_workers() in (True, False)  # no pool yet: nothing to do, never raises
    assert release_compile_workers() in (True, False)


def test_compiled_code_keeps_working_after_releasing_workers():
    f = torch.compile(lambda x: torch.sin(x) * 2 + 1)
    x = torch.randn(256)
    try:
        y = f(x)
    except Exception as e:  # no C++ toolchain for inductor's CPU backend here
        pytest.skip(f"torch.compile unavailable on CPU: {type(e).__name__}")
    release_compile_workers()
    assert torch.equal(f(x), y)  # compiled code keeps running
    g = torch.compile(lambda x: torch.cos(x) * 3)  # a later compile starts again on demand
    assert torch.allclose(g(x), torch.cos(x) * 3, rtol=1e-5, atol=1e-6)  # vectorized cos: ~1 ulp from eager
    release_compile_workers()
