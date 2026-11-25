"""Round 11: the counting sort of the MoE assignments (ModelConfig.moe_sort = "counting", layers/moe_triton.py
sort_assignments) gives the same (order, offs, perm) integers as torch.sort + searchsorted + argsort, and a MoE layer
gives bit-identical outputs, gradients and load counts with it. CPU: the kernels run in Triton's interpreter
(TRITON_INTERPRET=1 is set below when there is no GPU, before triton is imported); with a GPU they run on it."""
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from moe_llm.config import tiny  # noqa: E402
from moe_llm.layers import moe_triton as mt  # noqa: E402
from moe_llm.layers.moe import MoE  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _idx(N, K, E, kind, seed=0):
    """Expert ids [N, K], distinct per token."""
    g = torch.Generator().manual_seed(seed)
    if kind == "collapsed":  # every token picks the same K experts: the other experts get nothing (tied counts)
        return torch.arange(K).repeat(N, 1)
    logits = torch.randn(N, E, generator=g)
    if kind == "skewed":
        logits[:, : max(1, E // 4)] += 3
    return logits.topk(K, dim=-1).indices


@pytest.mark.parametrize("N,K,E,kind", [(1, 1, 4, "uniform"), (37, 4, 32, "uniform"), (300, 4, 32, "skewed"),
                                        (513, 2, 7, "uniform"), (256, 4, 32, "collapsed"), (130, 3, 33, "skewed"),
                                        (2048, 4, 32, "uniform")])
def test_counting_sort_equals_torch_sort(N, K, E, kind):
    idx = _idx(N, K, E, kind).to(DEV)
    ref = mt._sort_assignments(idx, E)
    got = mt._sort_assignments_counting(idx, E)
    for a, b, name in zip(ref, got, ("order", "offs", "perm")):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), (name, a, b)
    assert all(torch.equal(a, b) for a, b in zip(ref, mt.sort_assignments(idx, E, "counting")))
    assert all(torch.equal(a, b) for a, b in zip(ref, mt.sort_assignments(idx, E, "torch")))
    with pytest.raises(ValueError):
        mt.sort_assignments(idx, E, "radix")


@pytest.mark.parametrize("fan", ["", "router_lean"])
def test_moe_layer_counting_sort_bit_identical(fan):
    """Two MoE layers with the same weights, moe_sort "torch" and "counting": the same output, aux loss, load counter,
    input gradient and parameter gradients, bit for bit (forward + backward, recompute of the experts on)."""
    torch.manual_seed(0)
    layers = []
    for sort in ("torch", "counting"):
        m = MoE(tiny(moe_impl="triton", moe_shared_fused=fan, moe_sort=sort), 1).to(DEV)
        if layers:
            m.load_state_dict(layers[0].state_dict())
        m.train()
        m.recompute_experts = True
        layers.append(m)
    x = torch.randn(2, 24, layers[0].d, device=DEV)
    res = []
    for m in layers:
        xi = x.clone().requires_grad_(True)
        y, aux = m(xi)
        (y.float().pow(2).mean() + aux).backward()
        res.append((y.detach(), aux.detach(), xi.grad, m.load.clone(), [p.grad for p in m.parameters()]))
    (ya, aa, ga, la, pa), (yb, ab, gb, lb, pb) = res
    assert torch.equal(ya, yb) and torch.equal(aa, ab) and torch.equal(ga, gb) and torch.equal(la, lb)
    assert la.sum() > 0
    for u, v in zip(pa, pb):
        assert (u is None) == (v is None) and (u is None or torch.equal(u, v))
    with pytest.raises(ValueError):
        MoE(tiny(moe_impl="triton", moe_sort="radix"), 1)
