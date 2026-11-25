"""Weight casts with fused gradient accumulation (moe_llm/layers/common.py: cast_weight, CastLinear).
Under bf16 autocast the gradient of every cast weight is added straight into the fp32 .grad instead of
going through an fp32 copy first. Same values: the losses, all gradients over several accumulated
micro-batches and whole training steps must be bit-identical with the fast path on and off. CPU autocast
runs the same casts (GPU check in tests/test_gpu.py)."""
import copy

import pytest
import torch
import torch.nn as nn

from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.layers.common import CastLinear, cast_weight
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, wsd_lr

AUTOCAST = dict(device_type="cpu", dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def _restore():
    yield
    common.set_fast_accum(True)


def test_cast_linear_matches_nn_linear():
    torch.manual_seed(0)
    ref = nn.Linear(48, 40, bias=True)
    new = CastLinear(48, 40, bias=True)
    new.load_state_dict(ref.state_dict())
    xs = [torch.randn(6, 48) for _ in range(3)]
    for m in (ref, new):
        for x in xs:  # three accumulated micro-batches: first assignment, then two in-place adds
            with torch.autocast(**AUTOCAST):
                y = m(x)
            assert y.dtype == torch.bfloat16
            y.float().square().mean().backward()
    assert torch.equal(ref.weight.grad, new.weight.grad) and torch.equal(ref.bias.grad, new.bias.grad)
    assert ref.weight.grad.dtype == torch.float32
    with torch.no_grad():  # no autocast: plain fp32 linear
        assert torch.equal(ref(xs[0]), new(xs[0]))


def test_cast_weight_falls_back_when_not_needed():
    w = nn.Parameter(torch.randn(4, 4))
    assert cast_weight(w, torch.float32) is w                  # same dtype: no cast, no extra node
    with torch.no_grad():
        assert cast_weight(w, torch.bfloat16).grad_fn is None  # no grad needed: plain cast
    assert cast_weight(w * 1.0, torch.bfloat16).grad_fn.name() != "_CastAccumulateBackward"  # not a leaf


def _grads(model, batches):
    model.zero_grad(set_to_none=True)
    losses = []
    for x, y in batches:
        with torch.autocast(**AUTOCAST):
            loss, _ = model(x, y)
        (loss / len(batches)).backward()
        losses.append(loss.detach())
    return losses, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _batches(cfg, n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(0, cfg.vocab_size, (2, 33), generator=g)
        out.append((x[:, :-1], x[:, 1:]))
    return out


@pytest.mark.parametrize("residual,frac", [("standard", 0.0), ("block_attnres", 0.5), ("block_attnres", 1.0)])
def test_model_grads_identical_with_fast_accumulation(residual, frac):
    """The whole model (KDA, attention, MoE experts, shared expert, dense FFN, Engram projections) with
    3 accumulated micro-batches, including recomputed layers and recomputed experts (frac 0.5)."""
    torch.manual_seed(0)
    cfg = tiny(residual=residual)
    base = MoELLM(cfg)
    batches = _batches(cfg, 3, 1)
    out = {}
    for fast in (False, True):
        common.set_fast_accum(fast)
        m = copy.deepcopy(base)
        m.set_checkpointing(frac)
        out[fast] = _grads(m, batches)
    (l0, g0), (l1, g1) = out[False], out[True]
    # the fast path really ran: every Linear weight and expert weight went through _CastAccumulate
    m = copy.deepcopy(base)
    m.set_checkpointing(frac)
    with torch.autocast(**AUTOCAST):
        loss, _ = m(*batches[0])
    names, seen, todo = [], set(), [loss.grad_fn]
    while todo:
        fn = todo.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        names.append(fn.name())
        todo.extend(f for f, _ in fn.next_functions)
    n_cast = sum(1 + (mod.bias is not None) for mod in m.modules() if isinstance(mod, CastLinear))
    n_cast += 3 * len(m.moe_layers())  # w_gate, w_up, w_down
    n_seen = sum("CastAccumulate" in n for n in names)
    if frac == 0.0:  # with recompute the checkpointed regions keep their own graphs
        assert n_seen == n_cast, (n_seen, n_cast)
    else:
        assert any("CastAccumulate" in n for n in names)
    assert all(torch.equal(a, b) for a, b in zip(l0, l1))
    assert g0.keys() == g1.keys() and len(g0) == sum(1 for _ in base.parameters())
    for k in g0:
        assert torch.equal(g0[k], g1[k]), k


def test_training_steps_identical_with_fast_accumulation():
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    runs = []
    for fast in (False, True):
        common.set_fast_accum(fast)
        m = copy.deepcopy(base)
        m.set_checkpointing(0.5)
        opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16)
        ces = []
        for s in range(3):
            for x, y in _batches(cfg, 2, 10 + s):
                with torch.autocast(**AUTOCAST):
                    loss, info = m(x, y)
                (loss / 2).backward()
                ces.append(float(info["ce"]))
            opt.step(wsd_lr(s, 6, 3e-3, warmup=1))
            opt.zero_grad()
            m.update_router_bias()
        runs.append((ces, [p.detach().clone() for p in m.parameters()]))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
