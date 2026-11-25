"""Fused helpers (torch.compile, moe_llm/layers/common.py) must compute what the eager code computes:
every helper (values and gradients), the bf16 rounding of the fused conv, and a whole model trained
side by side with fusion off and on. On the GPU they run in bf16 (tests/test_gpu.py), here in fp32
with the CPU inductor backend, where any formula mistake would show up as a large difference."""
import copy
import math

import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.layers.common import (_rms_norm, add_rms_norm, conv_silu, gated_rms_norm, sigmoid_mul,
                                   silu_mul)
from moe_llm.layers.engram import _engram_gate
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, wsd_lr


@pytest.fixture
def fuse_cpu():
    common.set_fuse(True, cpu=True)
    yield common
    common.set_fuse(False)


def _leaf(t, grad=True):
    return t.detach().clone().requires_grad_(grad and t.is_floating_point())


def _check(helper, args, atol=1e-5):
    a_ref = [_leaf(a) if isinstance(a, torch.Tensor) else a for a in args]
    a_new = [_leaf(a) if isinstance(a, torch.Tensor) else a for a in args]
    ref, new = helper.eager(*a_ref), helper(*a_new)
    ref, new = (ref if isinstance(ref, tuple) else (ref,)), (new if isinstance(new, tuple) else (new,))
    for r, n in zip(ref, new):
        assert r.dtype == n.dtype and r.shape == n.shape, helper.__name__
        assert torch.allclose(n.float(), r.float(), atol=atol, rtol=1e-5), (helper.__name__, (n - r).abs().max())
    sum(r.float().square().sum() for r in ref).backward()
    sum(n.float().square().sum() for n in new).backward()
    for r, n in zip(a_ref, a_new):
        if isinstance(r, torch.Tensor) and r.grad is not None:
            assert torch.allclose(n.grad, r.grad, atol=10 * atol, rtol=1e-4), (helper.__name__, (n.grad - r.grad).abs().max())


def test_every_fused_helper_matches_eager(fuse_cpu):
    torch.manual_seed(0)
    r = torch.randn
    n_before = len(common._CLONES)
    _check(_rms_norm, [r(2, 16, 32), 1 + 0.1 * r(32), 1e-6])
    _check(_rms_norm, [r(2, 16, 4, 8), 1 + 0.1 * r(8), 1e-6])                  # per-head q/k norm
    _check(add_rms_norm, [r(2, 16, 32), r(2, 16, 32), 1 + 0.1 * r(32), 1e-6])
    _check(silu_mul, [r(2, 16, 48), r(2, 16, 48)])
    for c in (64, 128, 192):                                                    # MoE: tokens per expert vary
        _check(silu_mul, [r(4, c, 24), r(4, c, 24)])
    _check(sigmoid_mul, [r(2, 16, 4, 8), r(2, 16, 4, 8)])
    _check(gated_rms_norm, [r(2, 16, 4, 8), r(2, 16, 4, 8), 1 + 0.1 * r(8), 1e-5])
    _check(conv_silu, [r(2, 16, 32), 0.3 * r(32, 1, 4)])                        # KDA short conv
    _check(conv_silu, [r(2, 16, 32), 0.3 * r(32, 1, 4), 3, r(2, 16, 32), torch.float32])  # Engram (dilation 3)
    _check(_engram_gate, [r(2, 16, 32), r(2, 16, 32), r(2, 16, 32), math.sqrt(32), True])
    assert not common._FAILED, common._FAILED
    assert len(common._CLONES) > n_before                                       # compiled copies were used
    n_mid = len(common._CLONES)
    _check(silu_mul, [r(4, 256, 24), r(4, 256, 24)])                            # new size: no new copy
    assert len(common._CLONES) == n_mid


def test_fused_conv_keeps_bf16_rounding(fuse_cpu):
    """bf16 inputs: the fused conv rounds the weight and the conv output to bf16 like conv1d does."""
    torch.manual_seed(0)
    x = torch.randn(2, 64, 32).bfloat16()
    w = (0.5 * torch.randn(32, 1, 4)).requires_grad_()
    ref = conv_silu.eager(x, w, 1)
    new = conv_silu(x, w, 1)
    assert new.dtype == torch.bfloat16
    assert torch.allclose(new.float(), ref.float(), atol=2e-2, rtol=1e-2), (new.float() - ref.float()).abs().max()
    assert (new == ref).float().mean() > 0.9  # almost every element bit-identical


def _batch(cfg, seed=0, B=2, T=32):
    x = torch.randint(0, cfg.vocab_size, (B, T + 1), generator=torch.Generator().manual_seed(seed))
    return x[:, :-1], x[:, 1:]


def test_model_step_matches_with_fusion(fuse_cpu):
    """Whole tiny model (Engram, KDA short convs and gated norm, gated attention with q/k norm, MoE + dense
    SwiGLU, residual add + norm): one training step gives the same loss and gradients with fusion off/on."""
    torch.manual_seed(0)
    cfg = tiny()
    m_off = MoELLM(cfg)
    m_on = copy.deepcopy(m_off)
    x, y = _batch(cfg)
    common.set_fuse(False)
    l_off, _ = m_off(x, y)
    l_off.backward()
    common.set_fuse(True, cpu=True)
    l_on, _ = m_on(x, y)
    l_on.backward()
    assert abs(l_on.item() - l_off.item()) < 1e-5, (l_on.item(), l_off.item())
    for (n, p_off), p_on in zip(m_off.named_parameters(), m_on.parameters()):
        if p_off.grad is not None:
            assert torch.allclose(p_on.grad, p_off.grad, atol=1e-6, rtol=1e-4), n
    assert not common._FAILED


def _train(m, cfg, fuse, steps, make_opt, lr_at):
    opt = make_opt(m)
    out = []
    for step in range(steps):
        common.set_fuse(fuse, cpu=fuse)
        xb, yb = _batch(cfg, seed=step + 1)
        loss, info = m(xb, yb)
        loss.backward()
        opt.step(lr_at(step)) if isinstance(opt, OptimizerBundle) else opt.step()
        opt.zero_grad()
        m.update_router_bias()
        out.append(float(info["ce"]))
    return out


def test_model_trains_the_same_with_fusion_sgd(fuse_cpu):
    """Plain SGD does not amplify rounding noise, so 10 steps must give the same losses with fusion off/on."""
    torch.manual_seed(0)
    cfg = tiny()
    m0 = MoELLM(cfg)
    sgd = lambda m: torch.optim.SGD(m.parameters(), lr=0.1)  # noqa: E731
    off = _train(copy.deepcopy(m0), cfg, False, 10, sgd, None)
    on = _train(copy.deepcopy(m0), cfg, True, 10, sgd, None)
    assert max(abs(a - b) for a, b in zip(off, on)) < 2e-5, (off, on)
    assert on[-1] < on[0]
    assert not common._FAILED


def test_fusion_drift_is_within_rounding_noise(fuse_cpu):
    """With the real optimizer (Muon + AdamW): Muon's Newton-Schulz step and Adam's first steps magnify
    last-bit differences, so even two eager runs drift apart once one weight changes by one ulp. Fusion
    must not drift more than such a 1-ulp perturbation of the eager run does."""
    torch.manual_seed(0)
    cfg = tiny()
    m0 = MoELLM(cfg)
    bundle = lambda m: OptimizerBundle(m, lr=3e-3)  # noqa: E731
    lr_at = lambda s: wsd_lr(s, 8, 3e-3, warmup=2)  # noqa: E731
    ref = _train(copy.deepcopy(m0), cfg, False, 8, bundle, lr_at)
    floor = 0.0
    for seed in range(3):
        mp = copy.deepcopy(m0)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in mp.parameters():
                p.mul_(1 + 1.2e-7 * torch.randn(p.shape, generator=g))  # ~1 ulp of fp32
        pert = _train(mp, cfg, False, 8, bundle, lr_at)
        floor = max(floor, max(abs(a - b) for a, b in zip(ref, pert)))
    fused = _train(copy.deepcopy(m0), cfg, True, 8, bundle, lr_at)
    drift = max(abs(a - b) for a, b in zip(ref, fused))
    print(f"fused drift {drift:.2e}, 1-ulp perturbation drift {floor:.2e}")
    assert drift <= 3 * floor + 1e-5, (drift, floor)
    assert not common._FAILED


def test_fusion_off_never_compiles():
    common.set_fuse(False)
    before = len(common._CLONES)
    m = MoELLM(tiny())
    x, y = _batch(tiny())
    m(x, y)[0].backward()
    assert len(common._CLONES) == before


def test_param_groups_survive_deepcopy():
    """copy.deepcopy(model) drops parameter attributes such as _engram_table; the grouping must not depend
    on them (found while writing these tests; train.py never copies the model, so no run was affected)."""
    from moe_llm.optim import split_params
    m0 = MoELLM(tiny())
    m1 = copy.deepcopy(m0)

    def names(m):
        n = {id(p): k for k, p in m.named_parameters()}
        return {g: sorted(n[id(p)] for p in ps) for g, ps in split_params(m).items()}

    assert names(m0) == names(m1)
    assert names(m0)["engram"] == ["engram.tables.0"]
