"""Activation offloading (moe_llm/offload.py, MoELLM.set_checkpointing(offload_frac=...)): layers that would be
recomputed instead keep their saved tensors in host memory between forward and backward. Only data movement:
losses, all gradients and training steps must be bit-identical to recomputing them. On CPU the offloader runs
the same logic with ordinary copies (the CUDA path with streams and pinned memory is in tests/test_gpu.py)."""
import copy

import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers.moe import MoE
from moe_llm.model import MoELLM
from moe_llm.offload import ActivationOffloader
from moe_llm.optim import OptimizerBundle, wsd_lr


def _batch(cfg, s, k=0):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(500 + 10 * s + k))
    return x[:, :-1], x[:, 1:]


def test_layer_selection():
    m = MoELLM(tiny())  # 4 layers
    m.set_checkpointing(1.0, offload_frac=0.5)  # recompute all but offload half of them
    assert m.offload_act == {0, 2} and m.ckpt_layers == {1, 3} and m.offloader is not None
    assert m._prefetch_at == {3: [0, 2]}  # prefetch distance 3, capped at the last layer
    assert "offloaded to CPU memory" in m.recompute_summary()
    moe = {i: b.ffn.recompute_experts for i, b in enumerate(m.blocks) if isinstance(b.ffn, MoE)}
    assert moe == {1: False, 2: True, 3: False}  # offloaded layers keep their activations, experts recomputed
    m.set_checkpointing(0.5, offload_frac=1.0, ffn_frac=1.0)
    assert m.offload_act == {0, 2} and m.ckpt_layers == set() and not m.ffn_only
    m.set_checkpointing(0.5)
    assert not m.offload_act and m.offloader is None


@pytest.mark.parametrize("residual", ["standard", "block_attnres"])
def test_offloaded_layers_give_identical_loss_and_grads(residual):
    torch.manual_seed(0)
    cfg = tiny(residual=residual)
    base = MoELLM(cfg)
    x, y = _batch(cfg, 0)
    out = {}
    for name, frac, off in [("plain", 0.0, 0.0), ("recompute", 1.0, 0.0), ("half", 1.0, 0.5), ("all", 1.0, 1.0),
                            ("mixed", 0.5, 0.5)]:
        m = copy.deepcopy(base)
        m.set_checkpointing(frac, offload_frac=off)
        if m.offloader is not None:
            m.offloader.min_bytes = 0  # tiny model: offload everything
        loss, _ = m(x, y)
        loss.backward()
        if m.offloader is not None:
            assert m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0, m.offloader.stats
        out[name] = (loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None})
    ref_loss, ref_g = out["plain"]
    for name, (loss, g) in out.items():
        assert torch.equal(loss, ref_loss), name
        assert g.keys() == ref_g.keys() and all(torch.equal(g[k], ref_g[k]) for k in g), name


def test_offloaded_training_matches_and_host_memory_is_reused():
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    runs = []
    for off in (0.0, 1.0):
        m = copy.deepcopy(base)
        m.set_checkpointing(1.0, offload_frac=off)
        if m.offloader is not None:
            m.offloader.min_bytes = 0
        opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16)
        ces, host = [], []
        for s in range(3):
            for k in range(2):
                x, y = _batch(cfg, s, k)
                loss, info = m(x, y)
                (loss / 2).backward()
                ces.append(float(info["ce"]))
                if m.offloader is not None:
                    host.append(m.offloader.stats["host_bytes"])
            opt.step(wsd_lr(s, 6, 3e-3, warmup=1))
            opt.zero_grad()
            m.update_router_bias()
        runs.append(ces)
        if host:
            assert host[0] > 0 and len(set(host)) == 1, host  # buffers allocated once, then reused
    assert runs[0] == runs[1]


def test_offloader_dedups_and_skips():
    off = ActivationOffloader(min_bytes=64)
    w = torch.nn.Parameter(torch.randn(16, 16))
    x = torch.randn(8, 16, requires_grad=True)
    small = torch.randn(2, 2, requires_grad=True)
    keep = torch.randn(8, 16, requires_grad=True)
    off.keep_on_gpu([keep])
    with off.layer(0):
        h = x * 2.0                      # h is saved by both matmuls below: one copy
        y = (h @ w).sum() + (h @ w.t()).sum() + (small * small).sum() + (keep * keep).sum()
    # saved: h (twice, deduplicated), w (parameter: skipped), small (below min_bytes), keep (kept on the device)
    assert off.stats["tensors"] == 1, off.stats
    off.prefetch(0)
    y.backward()
    assert off.stats["late"] == 0
    ref = (x * 2.0)
    assert torch.allclose(x.grad, (torch.ones(8, 16) @ w.t() + torch.ones(8, 16) @ w) * 2.0)
    assert ref.shape == h.shape
