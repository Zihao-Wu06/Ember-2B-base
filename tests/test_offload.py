"""Optimizer states across checkpoints and CPU offload (moe_llm/optim.py). Offload only moves the states
between optimizer steps (GPU checks in tests/test_gpu.py); a resumed run must keep the bf16 Muon momentum
and continue exactly like the uninterrupted one."""
import copy
import io

import torch

from moe_llm.config import tiny
from moe_llm.model import MoELLM
from moe_llm.optim import Muon, OptimizerBundle, wsd_lr


def test_muon_bf16_momentum_survives_reload():
    p = torch.nn.Parameter(torch.randn(32, 32))
    m = Muon([p], lr=1e-3, state_dtype=torch.bfloat16)
    p.grad = torch.randn(32, 32)
    m.step()
    buf = m.state[p]["buf"].clone()
    m2 = Muon([p], lr=1e-3, state_dtype=torch.bfloat16)
    m2.load_state_dict(m.state_dict())  # torch.optim alone would cast it to the fp32 parameter dtype
    assert m2.state[p]["buf"].dtype == torch.bfloat16 and torch.equal(m2.state[p]["buf"], buf)


def _batch(cfg, step):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(100 + step))
    return x[:, :-1], x[:, 1:]


def _run(m, opt, cfg, steps, start=0):
    out = []
    for s in range(start, start + steps):
        x, y = _batch(cfg, s)
        loss, info = m(x, y)
        loss.backward()
        opt.step(wsd_lr(s, 8, 3e-3, warmup=2))
        opt.zero_grad()
        m.update_router_bias()
        out.append(float(info["ce"]))
    return out


def _save(m, opt):
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
    buf.seek(0)
    return buf


def test_resume_with_bf16_muon_continues_exactly():
    torch.manual_seed(0)
    cfg = tiny()
    m = MoELLM(cfg)
    opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16)
    _run(m, opt, cfg, 4)
    ck = _save(m, opt)
    full = _run(m, opt, cfg, 4, start=4)
    sd = torch.load(ck, weights_only=False)
    m2 = MoELLM(cfg)
    m2.load_state_dict(sd["model"])
    opt2 = OptimizerBundle(m2, lr=3e-3, muon_state_dtype=torch.bfloat16)
    opt2.load_state_dict(sd["opt"])
    assert all(st["buf"].dtype == torch.bfloat16 for st in opt2.muon.state.values())
    resumed = _run(m2, opt2, cfg, 4, start=4)
    assert resumed == full, (full, resumed)


def test_offload_is_a_noop_without_cuda():
    torch.manual_seed(0)
    cfg = tiny()
    m0 = MoELLM(cfg)
    a, b = copy.deepcopy(m0), copy.deepcopy(m0)
    oa = OptimizerBundle(a, lr=3e-3, offload_states=True)
    ob = OptimizerBundle(b, lr=3e-3)
    assert _run(a, oa, cfg, 3) == _run(b, ob, cfg, 3)
    assert not oa._pinned and not oa._moved
