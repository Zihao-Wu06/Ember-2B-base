"""Exact speedups, second round (CPU; GPU checks in tests/test_gpu.py):
* OptimizerBundle(offload_master=True): persistent bf16 copies for the forward, fp32 gradient buffers, fp32
  masters (in pinned CPU memory on CUDA); gradient clipping through the bundle
* MoELLM.set_checkpointing(ffn_frac=...): recomputed layers that recompute only their FFN part
* OptimizerBundle(async_engram=True): the Engram table update on the Engram worker thread
All must train bit-identically to the plain code, including across a checkpoint save and load."""
import copy
import gc
import io

import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.layers.common import cast_weight
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, wsd_lr


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


def _batch(cfg, s, k):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(1000 * s + k))
    return x[:, :-1], x[:, 1:]


def _train(m, opt, cfg, steps, start=0, clip=0.05, autocast=True, micro=2):
    out = []
    for s in range(start, start + steps):
        for k in range(micro):
            x, y = _batch(cfg, s, k)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                loss, info = m(x, y)
            (loss / micro).backward()
            out.append(float(info["ce"]))
        out.append(float(opt.clip_grad_norm(clip)))  # clip=0.05: clipping is active every step
        opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
        opt.zero_grad()
        m.update_router_bias()
    return out


def _params(m):
    return {n: p.detach().clone() for n, p in m.state_dict().items()}


@pytest.mark.parametrize("residual", ["standard", "block_attnres"])
@pytest.mark.parametrize("autocast", [True, False])
def test_offload_master_trains_identically(residual, autocast):
    """With autocast the forward uses the bf16 copies; without it (CPU runs of train.py) the fp32 weights
    directly, whose gradients are then in .grad as usual: both must match the plain bundle."""
    torch.manual_seed(0)
    cfg = tiny(residual=residual)
    base = MoELLM(cfg)
    base.set_checkpointing(0.5)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    oa = OptimizerBundle(a, lr=3e-3, muon_state_dtype=torch.bfloat16, offload_master=True)
    ob = OptimizerBundle(b, lr=3e-3, muon_state_dtype=torch.bfloat16)
    oa.begin()
    ob.begin()  # no-op without offload_master
    assert len(oa._cast) == len(a.cast_params()) > 0 and not ob._cast
    p0 = a.cast_params()[0]
    assert cast_weight(p0, torch.bfloat16) is oa._cast[id(p0)][1]          # the forward uses the copy
    assert cast_weight(b.cast_params()[0], torch.bfloat16).dtype == torch.bfloat16
    ra, rb = _train(a, oa, cfg, 3, autocast=autocast), _train(b, ob, cfg, 3, autocast=autocast)
    assert ra == rb, (ra, rb)
    pa, pb = _params(a), _params(b)
    assert all(torch.equal(pa[k], pb[k]) for k in pa)
    for p, c in oa._cast.values():  # the copies are the bf16 rounding of the masters
        assert torch.equal(c, p.detach().to(torch.bfloat16))


def test_offload_master_gradients_live_in_buffers():
    torch.manual_seed(0)
    cfg = tiny()
    m = MoELLM(cfg)
    opt = OptimizerBundle(m, lr=3e-3, offload_master=True)
    opt.begin()
    x, y = _batch(cfg, 0, 0)
    for _ in range(2):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = m(x, y)
        loss.backward()
    cast = {id(p) for p in m.cast_params()}
    assert all(p.grad is None for p in m.cast_params())          # accumulated into the fp32 buffers instead
    assert set(opt._gbuf) == cast and all(g.dtype == torch.float32 for g in opt._gbuf.values())
    assert all(c.grad is None for _, c in opt._cast.values())
    assert all(p.grad is not None for p in m.parameters() if id(p) not in cast and p.requires_grad
               and p.dim() > 0 and "e_bias" not in str(p.shape))  # the rest as usual


def test_offload_master_resume_continues_exactly():
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    m = MoELLM(cfg)
    opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16, offload_master=True)
    opt.begin()
    _train(m, opt, cfg, 2)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
    full = _train(m, opt, cfg, 2, start=2)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)
    m2 = MoELLM(cfg)
    m2.load_state_dict(sd["model"])
    opt2 = OptimizerBundle(m2, lr=3e-3, muon_state_dtype=torch.bfloat16, offload_master=True)
    opt2.load_state_dict(sd["opt"])
    opt2.begin()
    assert _train(m2, opt2, cfg, 2, start=2) == full


def test_cached_copy_is_never_used_for_another_tensor():
    m = MoELLM(tiny())
    opt = OptimizerBundle(m, lr=3e-3, offload_master=True)
    opt.begin()
    ids = [id(p) for p in m.cast_params()]
    del m, opt
    gc.collect()
    m2 = MoELLM(tiny())  # new tensors may reuse the ids of the freed ones
    reused = [p for p in m2.cast_params() if id(p) in ids]
    for p in m2.cast_params():
        assert cast_weight(p, torch.bfloat16).grad_fn is not None or not p.requires_grad  # a fresh cast
    assert isinstance(reused, list)


@pytest.mark.parametrize("residual", ["standard", "block_attnres"])
@pytest.mark.parametrize("autocast", [False, True])
def test_ffn_only_recompute_gives_identical_grads(residual, autocast):
    torch.manual_seed(0)
    cfg = tiny(residual=residual)
    base = MoELLM(cfg)
    x, y = _batch(cfg, 0, 0)
    out = {}
    for frac, ffn in [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0), (1.0, 1.0)]:
        m = copy.deepcopy(base)
        m.set_checkpointing(frac, ffn_frac=ffn)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            loss, _ = m(x, y)
        loss.backward()
        out[(frac, ffn)] = (loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None})
    ref_loss, ref_g = out[(0.0, 0.0)]
    for key, (loss, g) in out.items():
        assert torch.equal(loss, ref_loss), key
        assert g.keys() == ref_g.keys() and all(torch.equal(g[k], ref_g[k]) for k in g), key


def test_ffn_only_layer_selection():
    m = MoELLM(tiny())  # 4 layers
    m.set_checkpointing(0.5, ffn_frac=0.5)  # recompute layers 0 and 2; layer 0 only its FFN part
    assert m.ckpt_layers == {0, 2} and m.ffn_only == {0}
    assert [b.ckpt_ffn for b in m.blocks] == [True, False, False, False]
    assert "FFN part only in layers [0]" in m.recompute_summary()
    m.set_checkpointing(1.0, ffn_frac=1.0)
    assert m.ffn_only == {0, 1, 2, 3} and all(b.ckpt_ffn for b in m.blocks)
    m.set_checkpointing(0.0, ffn_frac=1.0)
    assert not m.ffn_only and not any(b.ckpt_ffn for b in m.blocks)


def test_async_engram_update_trains_identically():
    torch.manual_seed(0)
    cfg = tiny(engram_offload=True)
    base = MoELLM(cfg)
    runs = []
    for async_engram in (False, True):
        m = copy.deepcopy(base)
        opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16, async_engram=async_engram)
        r = _train(m, opt, cfg, 3, autocast=False)
        tables = [{k: v.clone() for k, v in sd.items()} for sd in m.engram.offload_state_dict()]  # flushes first
        runs.append((r, tables))
    assert runs[0][0] == runs[1][0]
    for ta, tb in zip(runs[0][1], runs[1][1]):
        assert all(torch.equal(ta[k], tb[k]) for k in ta)


def test_router_stats_are_tensors_and_bias_update_unchanged():
    torch.manual_seed(0)
    cfg = tiny()
    m = MoELLM(cfg)
    x, y = _batch(cfg, 0, 0)
    loss, _ = m(x, y)
    moe = m.moe_layers()[0]
    load, bias = moe.load.clone(), moe.e_bias.clone()
    stats = m.update_router_bias()
    assert all(torch.is_tensor(v) and v.dim() == 0 for v in stats.values())
    assert torch.equal(moe.e_bias, bias + moe.gamma * torch.sign(load.mean() - load))
    assert float(moe.load.sum()) == 0
    m.update_router_bias()  # no tokens since: mean 0, sign(0) = 0, the bias stays
    assert torch.equal(moe.e_bias, bias + moe.gamma * torch.sign(load.mean() - load))
