import copy

import pytest
import torch
import torch.nn.functional as F

from moe_llm.config import tiny
from moe_llm.loss import linear_cross_entropy
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, sparse_row_adam_, split_params, wsd_lr, zeropower_via_newtonschulz5

VARIANTS = {
    "hybrid_engram": tiny(),
    "hybrid_engram_attnres": tiny(residual="block_attnres"),
    "fullattn_rope": tiny(layer_pattern="F", attn_rope=True, engram_layers=()),
    "hybrid_offload": tiny(engram_offload=True),
}


def batch(cfg, B=2, T=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (B, T + 1), generator=g)
    return x[:, :-1], x[:, 1:]


@pytest.mark.parametrize("name", list(VARIANTS))
def test_forward_backward_all_grads(name):
    torch.manual_seed(0)
    cfg = VARIANTS[name]
    m = MoELLM(cfg)
    x, y = batch(cfg)
    loss, info = m(x, y)
    assert torch.isfinite(loss) and abs(float(info["ce"]) - torch.log(torch.tensor(cfg.vocab_size)).item()) < 1.0
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("name", list(VARIANTS))
def test_model_is_causal(name):
    torch.manual_seed(0)
    cfg = VARIANTS[name]
    m = MoELLM(cfg).eval()
    x, _ = batch(cfg)
    x2 = x.clone()
    x2[:, 15] = (x2[:, 15] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5)
    assert not torch.allclose(a[:, 15:], b[:, 15:])


@pytest.mark.parametrize("name", ["hybrid_engram", "hybrid_engram_attnres", "hybrid_offload"])
def test_checkpointing_matches(name):
    torch.manual_seed(0)
    cfg = VARIANTS[name].replace(seq_aux_coef=0.0)
    m1 = MoELLM(cfg)
    m2 = copy.deepcopy(m1)
    m2.grad_checkpointing = True
    x, y = batch(cfg)
    l1, _ = m1(x, y)
    l2, _ = m2(x, y)
    l1.backward()
    l2.backward()
    assert torch.allclose(l1, l2, atol=1e-6)
    for (n, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), n


def test_offload_matches_gpu_tables_including_optimizer_step():
    torch.manual_seed(0)
    cfg = tiny()
    m_gpu = MoELLM(cfg)
    m_off = MoELLM(cfg.replace(engram_offload=True))
    m_off.load_state_dict(m_gpu.state_dict(), strict=False)
    for t_gpu, t_off in zip(m_gpu.engram.tables, m_off.engram.cpu_tables):
        t_off.weight.copy_(t_gpu.detach())
    x, y = batch(cfg)
    o_gpu, o_off = OptimizerBundle(m_gpu, lr=1e-2), OptimizerBundle(m_off, lr=1e-2)
    for _ in range(2):
        for m, o in ((m_gpu, o_gpu), (m_off, o_off)):
            loss, _ = m(x, y)
            loss.backward()
            o.step(1e-2)
            o.zero_grad()
            m.update_router_bias()
    l_gpu, _ = m_gpu(x, y)
    l_off, _ = m_off(x, y)
    assert torch.allclose(l_gpu, l_off, atol=1e-5)
    for t_gpu, t_off in zip(m_gpu.engram.tables, m_off.engram.cpu_tables):
        assert torch.allclose(t_gpu.detach(), t_off.weight, atol=1e-6)


def test_offload_prefetch_matches_sync():
    torch.manual_seed(0)
    cfg = tiny(engram_offload=True, seq_aux_coef=0.0)
    m1 = MoELLM(cfg)
    m2 = copy.deepcopy(m1)
    xs = [batch(cfg, seed=s) for s in range(3)]
    for x, _ in xs:
        m2.engram.prefetch(x)          # all three queued ahead, consumed in order
    for x, y in xs:
        l1, _ = m1(x, y)
        l2, _ = m2(x, y)
        l1.backward()
        l2.backward()
        assert torch.allclose(l1, l2, atol=1e-6)
    m2.engram.flush()
    for t1, t2 in zip(m1.engram.cpu_tables, m2.engram.cpu_tables):
        assert torch.allclose(t1.grad, t2.grad, atol=1e-6) and torch.equal(t1.touched, t2.touched)


def test_row_adam_cpu_matches_reference():
    from moe_llm.optim import row_adam_cpu
    torch.manual_seed(0)
    w, g = torch.randn(50, 4), torch.randn(50, 4)
    m, v = torch.randn(50, 4) * 0.1, torch.rand(50, 4) * 0.1
    ids = torch.tensor([0, 7, 9, 31])
    ref = [t.clone() for t in (w, g, m, v)]
    sparse_row_adam_(*ref, ids, 1e-2, (0.9, 0.95), 1e-8, 3)
    row_adam_cpu(w, g, m, v, ids, 1e-2, (0.9, 0.95), 1e-8, 3)
    for a, b in zip((w, m, v), (ref[0], ref[2], ref[3])):
        assert torch.allclose(a, b, atol=1e-6)
    assert torch.equal(g[ids], torch.zeros(4, 4)) and torch.equal(g[1], ref[1][1])


@pytest.mark.parametrize("name", ["hybrid_engram_attnres", "fullattn_rope"])
def test_overfits_one_batch(name):
    torch.manual_seed(0)
    cfg = VARIANTS[name]
    m = MoELLM(cfg)
    opt = OptimizerBundle(m, lr=3e-3)
    x, y = batch(cfg, B=4, T=32)
    first = None
    for step in range(80):
        loss, info = m(x, y)
        loss.backward()
        opt.step(wsd_lr(step, 80, 3e-3, warmup=5))
        opt.zero_grad()
        m.update_router_bias()
        first = first if first is not None else float(info["ce"])
    assert float(info["ce"]) < 0.4 * first, (first, float(info["ce"]))


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    m = MoELLM(cfg).eval()
    m.moe_layers()[0].e_bias.fill_(0.5)
    m2 = MoELLM(cfg).eval()
    m2.load_state_dict(m.state_dict())
    x, _ = batch(cfg)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x))
    assert torch.equal(m2.moe_layers()[0].e_bias, torch.full_like(m2.moe_layers()[0].e_bias, 0.5))


# ----------------------------------------------------------------------------- loss
def test_linear_cross_entropy_matches_reference():
    torch.manual_seed(0)
    N, d, V, z = 37, 16, 50, 1e-3
    h = torch.randn(N, d, dtype=torch.float64, requires_grad=True)
    W = torch.randn(V, d, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, V, (N,))
    t[[3, 10]] = -100
    loss, ce = linear_cross_entropy(h, W, t, z_coef=z, chunk=8)
    loss.backward()
    h2, W2 = h.detach().clone().requires_grad_(), W.detach().clone().requires_grad_()
    logits = h2 @ W2.t()
    valid = t != -100
    ref_ce = F.cross_entropy(logits, t, ignore_index=-100)
    ref = ref_ce + z * torch.logsumexp(logits[valid], -1).pow(2).mean()
    ref.backward()
    assert torch.allclose(loss, ref, atol=1e-10) and torch.allclose(ce, ref_ce, atol=1e-10)
    assert torch.allclose(h.grad, h2.grad, atol=1e-10) and torch.allclose(W.grad, W2.grad, atol=1e-10)


# ----------------------------------------------------------------------------- optim
def test_newton_schulz_is_near_orthogonal():
    G = torch.randn(3, 64, 32)
    X = zeropower_via_newtonschulz5(G)
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.3


def test_param_groups():
    m = MoELLM(tiny(residual="block_attnres"))
    g = split_params(m)
    ids = [id(p) for v in g.values() for p in v]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    names = {id(p): n for n, p in m.named_parameters()}
    muon = {names[id(p)] for p in g["muon"]}
    nodecay = {names[id(p)] for p in g["adamw_no_decay"]}
    assert any(n.endswith("ffn.w_gate") for n in muon) and any(n.endswith("mixer.q_proj.weight") for n in muon)
    assert any(n.endswith("A_log") for n in nodecay) and any(n.endswith("router.weight") for n in nodecay)
    assert any(n.endswith("res_mix.w") for n in nodecay)
    assert not any("embed" in n or "lm_head" in n for n in muon)
    assert len(g["engram"]) == len(m.engram.tables)


def test_sparse_row_adam_matches_adam_formula():
    w = torch.randn(6, 3)
    g = torch.randn(6, 3)
    m, v = torch.zeros(6, 3), torch.zeros(6, 3)
    ids = torch.tensor([1, 4])
    w0 = w.clone()
    sparse_row_adam_(w, g, m, v, ids, lr=0.1, betas=(0.9, 0.95), eps=1e-8, step=1)
    want = w0[ids] - 0.1 * (0.1 * g[ids] / 0.1) / ((0.05 * g[ids] ** 2 / 0.05).sqrt() + 1e-8)
    assert torch.allclose(w[ids], want, atol=1e-6)
    keep = torch.tensor([0, 2, 3, 5])
    assert torch.equal(w[keep], w0[keep])


def test_wsd_schedule():
    peak = 1.0
    lrs = [wsd_lr(s, 100, peak, warmup=10, decay_frac=0.2) for s in range(100)]
    assert lrs[0] == 0.1 and lrs[9] == 1.0 and lrs[50] == 1.0 and lrs[80] == 1.0
    assert lrs[99] < 0.25 and all(a >= b for a, b in zip(lrs[80:], lrs[81:]))


def test_partial_checkpointing_matches():
    torch.manual_seed(0)
    cfg = tiny(seq_aux_coef=0.0, n_layers=4)
    m1 = MoELLM(cfg)
    m2 = copy.deepcopy(m1)
    m2.set_checkpointing(0.5)
    assert m2.ckpt_layers == {0, 2}
    x, y = batch(cfg)
    l1, _ = m1(x, y)
    l2, _ = m2(x, y)
    l1.backward()
    l2.backward()
    assert torch.allclose(l1, l2, atol=1e-6)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)


def test_compiled_glue_matches_eager():
    from moe_llm.layers import common
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", seq_aux_coef=0.0)
    m = MoELLM(cfg)
    x, y = batch(cfg)
    l_eager, _ = m(x, y)
    common.set_compile(True)
    try:
        l_comp, _ = m(x, y)
        l_comp.backward()
    finally:
        common.set_compile(False)
    assert torch.allclose(l_eager, l_comp, atol=1e-5)
    assert all(p.grad is not None for p in m.parameters())
