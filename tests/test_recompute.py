"""Partial activation recompute (MoELLM.set_checkpointing). Layers that keep their activations still
recompute their routed experts, so memory does not grow with routing imbalance (GPU memory check in
tests/test_gpu.py). Recompute only repeats the same ops: the loss, every gradient and a few training
steps must match exactly."""
import copy

import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers.moe import MoE
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, wsd_lr


def _flags(m):
    return {i: b.ffn.recompute_experts for i, b in enumerate(m.blocks) if isinstance(b.ffn, MoE)}


def test_expert_recompute_only_in_layers_that_keep_activations():
    m = MoELLM(tiny())                 # 4 layers, layer 0 dense
    m.set_checkpointing(0.5)           # recompute layers 0 and 2, keep 1 and 3
    assert m.ckpt_layers == {0, 2}
    assert _flags(m) == {1: True, 2: False, 3: True}
    assert "routed experts recomputed in 2" in m.recompute_summary()
    m.set_checkpointing(0.5, expert_recompute=False)
    assert not any(_flags(m).values())
    for frac in (0.0, 1.0):
        m.set_checkpointing(frac)
        assert not any(_flags(m).values())


def _batch(cfg, step):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(200 + step))
    return x[:, :-1], x[:, 1:]


@pytest.mark.parametrize("residual", ["standard", "block_attnres"])
def test_expert_recompute_gives_identical_loss_and_grads(residual):
    torch.manual_seed(0)
    cfg = tiny(residual=residual)
    base = MoELLM(cfg)
    x, y = _batch(cfg, 0)
    out = {}
    for name, frac, er in [("plain", 0.0, True), ("keep", 0.5, False), ("expert", 0.5, True), ("all", 1.0, True)]:
        m = copy.deepcopy(base)
        m.set_checkpointing(frac, expert_recompute=er)
        loss, _ = m(x, y)
        loss.backward()
        out[name] = (loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None})
    ref_loss, ref_g = out["plain"]
    for name, (loss, g) in out.items():
        assert torch.equal(loss, ref_loss), name
        assert g.keys() == ref_g.keys(), name
        for k in ref_g:
            assert torch.equal(g[k], ref_g[k]), (name, k)


def test_expert_recompute_training_matches_exactly():
    torch.manual_seed(0)
    cfg = tiny()
    base = MoELLM(cfg)
    ces = []
    for er in (False, True):
        m = copy.deepcopy(base)
        m.set_checkpointing(0.5, expert_recompute=er)
        opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16)
        run = []
        for s in range(4):
            x, y = _batch(cfg, s)
            loss, info = m(x, y)
            loss.backward()
            opt.step(wsd_lr(s, 8, 3e-3, warmup=2))
            opt.zero_grad()
            m.update_router_bias()
            run.append(float(info["ce"]))
        ces.append(run)
    assert ces[0] == ces[1], ces
