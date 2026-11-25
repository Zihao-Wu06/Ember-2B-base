import math

import pytest
import torch
import torch.nn.functional as F

from moe_llm.config import tiny
from moe_llm.layers.attention import Attention
from moe_llm.layers.attnres import DepthAttn
from moe_llm.layers.engram import EngramBlock, EngramMemory
from moe_llm.layers.kda import KDA, kda_gate, kda_reference, l2norm
from moe_llm.layers.moe import MoE

torch.manual_seed(0)


# ----------------------------------------------------------------------------- KDA
def kda_matrix_form(q, k, v, g, beta):
    """Independent re-derivation: S_t = (I - b k k^T) Diag(exp g_t) S_{t-1} + b k v^T,  o_t = S_t^T q_t."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q = q * K ** -0.5
    S = torch.zeros(B, H, K, V, dtype=torch.float64)
    o = torch.zeros(B, T, H, V, dtype=torch.float64)
    eye = torch.eye(K, dtype=torch.float64)
    for t in range(T):
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        Dg = torch.diag_embed(g[:, t].exp())                                  # [B,H,K,K]
        P = eye - bt[..., None, None] * kt[..., :, None] * kt[..., None, :]   # I - b k k^T
        S = P @ Dg @ S + bt[..., None, None] * kt[..., :, None] * vt[..., None, :]
        o[:, t] = (S.transpose(-1, -2) @ q[:, t][..., None]).squeeze(-1)
    return o


def test_kda_reference_matches_matrix_form():
    B, T, H, K = 2, 13, 2, 8
    q, k, v = (torch.randn(B, T, H, K, dtype=torch.float64) for _ in range(3))
    g_raw = torch.randn(B, T, H, K, dtype=torch.float64)
    beta_logits = torch.randn(B, T, H, dtype=torch.float64)
    A_log = torch.log(torch.empty(H).uniform_(1, 16)).double()
    dt_bias = torch.randn(H * K).double() * 0.5
    ref = kda_reference(q, k, v, g_raw, beta_logits, A_log, dt_bias).double()
    g = kda_gate(g_raw, A_log, dt_bias).double()
    mat = kda_matrix_form(l2norm(q).double(), l2norm(k).double(), v, g, torch.sigmoid(beta_logits))
    assert torch.allclose(ref, mat, atol=1e-5, rtol=1e-4)


def test_kda_layer_causal_and_grads():
    cfg = tiny()
    layer = KDA(cfg, 0)
    x = torch.randn(2, 16, cfg.d_model, requires_grad=True)
    y = layer(x)
    x2 = x.detach().clone()
    x2[:, 9] += 1.0
    y2 = layer(x2)
    assert torch.allclose(y[:, :9], y2[:, :9], atol=1e-6)
    assert not torch.allclose(y[:, 9:], y2[:, 9:])
    y.square().mean().backward()
    for n, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


# ----------------------------------------------------------------------------- attention
def test_attention_causal_and_gqa():
    cfg = tiny(attn_rope=True)
    att = Attention(cfg, 0)
    x = torch.randn(2, 12, cfg.d_model)
    y = att(x)
    x2 = x.clone()
    x2[:, 7] += 1.0
    assert torch.allclose(y[:, :7], att(x2)[:, :7], atol=1e-6)
    assert y.shape == x.shape


# ----------------------------------------------------------------------------- MoE
def test_moe_impls_agree():
    cfg = tiny()
    torch.manual_seed(1)
    moe = MoE(cfg, 1)
    x = torch.randn(3, 10, cfg.d_model)
    outs, grads = {}, {}
    for impl in ("loop", "bmm"):
        moe.impl = impl
        moe.zero_grad()
        y, aux = moe(x)
        (y.square().mean() + aux).backward()
        outs[impl] = y.detach()
        grads[impl] = [p.grad.clone() for p in moe.parameters()]
    assert torch.allclose(outs["loop"], outs["bmm"], atol=1e-5)
    for a, b in zip(grads["loop"], grads["bmm"]):
        assert torch.allclose(a, b, atol=1e-5)


def test_moe_bias_selects_but_does_not_weigh():
    cfg = tiny(routed_scale=1.0)
    moe = MoE(cfg, 1)
    x = torch.randn(64, cfg.d_model)
    moe.e_bias.zero_()
    moe.e_bias[3] = 100.0
    scores, idx, w = moe.route(x)
    assert (idx == 3).any(-1).all()                              # expert 3 always selected
    assert torch.allclose(w.sum(-1), torch.ones(64), atol=1e-5)  # renormalised weights
    sel = scores.gather(1, idx)
    assert torch.allclose(w, sel / sel.sum(-1, keepdim=True), atol=1e-6)  # weights from unbiased scores


def test_moe_bias_update_direction():
    moe = MoE(tiny(), 1)
    moe.load.copy_(torch.tensor([10., 0, 0, 0, 0, 0, 0, 30.]))
    moe.update_bias()
    assert moe.e_bias[7] < 0 and moe.e_bias[1] > 0 and float(moe.load.sum()) == 0


def test_seq_aux_loss_formula():
    cfg = tiny(seq_aux_coef=1.0)
    moe = MoE(cfg, 1)
    B, T, E, K = 2, 6, cfg.n_experts, cfg.top_k
    scores = torch.rand(B * T, E)
    idx = torch.topk(scores, K, -1).indices
    got = moe.seq_aux_loss(scores, idx, B, T)
    want = 0.0
    for b in range(B):
        s = scores.view(B, T, E)[b]
        P = (s / s.sum(-1, keepdim=True)).mean(0)
        f = torch.zeros(E)
        for t in range(T):
            for e in idx.view(B, T, K)[b, t]:
                f[e] += E / (K * T)
        want += (f * P).sum() / B
    assert torch.allclose(got, torch.as_tensor(want), atol=1e-6)


# ----------------------------------------------------------------------------- Engram
def test_engram_hash_is_suffix_ngram_and_causal():
    cfg = tiny(engram_layers=(1,), engram_orders=(2, 3), engram_heads=2)
    mem = EngramMemory(cfg)
    a = torch.tensor([[5, 6, 7, 8, 9, 10]])
    b = torch.tensor([[1, 2, 7, 8, 9, 11]])
    ia, ib = mem.indices(a)[0], mem.indices(b)[0]  # [1, T, 2 orders * 2 heads]
    # position 4: last 3 tokens (7, 8, 9) equal -> both 2-gram and 3-gram rows equal
    assert torch.equal(ia[0, 4], ib[0, 4])
    # position 3: last 2 equal (7, 8) but 3-grams differ (6 vs 2)
    assert torch.equal(ia[0, 3, :2], ib[0, 3, :2]) and not torch.equal(ia[0, 3, 2:], ib[0, 3, 2:])
    # causality: changing the last token never changes earlier rows
    c = a.clone()
    c[0, 5] = 99
    assert torch.equal(mem.indices(c)[0][0, :5], ia[0, :5])
    # every head indexes inside its own table slice
    assert int(ia.max()) < mem.rows[0] and int(ia.min()) >= 0


def test_engram_per_order_rows():
    cfg = tiny(engram_layers=(1, 2), engram_heads=2, engram_rows=(50, 200))
    mem = EngramMemory(cfg)
    mods = mem.mods.flatten().tolist()
    assert len(set(mods)) == len(mods)                                  # every head: its own prime
    assert all(m < 200 for m in mem.mods[:, :2].flatten().tolist())     # 2-gram heads near 50
    assert all(m >= 200 for m in mem.mods[:, 2:].flatten().tolist())    # 3-gram heads >= 200
    idx = mem.indices(torch.randint(0, cfg.vocab_size, (2, 30)))
    assert all(int(i.max()) < r for i, r in zip(idx, mem.rows))


def test_engram_canon_map_merges_tokens():
    cfg = tiny(engram_layers=(1,))
    mem = EngramMemory(cfg)
    canon = torch.arange(cfg.vocab_size)
    canon[40] = 30  # pretend token 40 normalises to the same string as token 30
    mem.set_canon_map(canon)
    x = torch.tensor([[3, 4, 30], [3, 4, 40]])
    i = mem.indices(x)[0]
    assert torch.equal(i[0, 2], i[1, 2])


def test_engram_block_identity_at_init():
    cfg = tiny()
    blk = EngramBlock(cfg, 1)
    h, e = torch.randn(2, 9, cfg.d_model), torch.randn(2, 9, cfg.engram_dim)
    v = blk.w_v(e)
    s = (blk.norm_q(h) * blk.norm_k(blk.w_k(e))).sum(-1, keepdim=True) / math.sqrt(cfg.d_model)
    s = s.sign() * s.abs().clamp_min(1e-6).sqrt()
    assert torch.allclose(blk(h, e), torch.sigmoid(s) * v, atol=1e-6)  # zero conv => Y = gated value


# ----------------------------------------------------------------------------- AttnRes
def test_attnres_zero_init_is_uniform_average():
    op = DepthAttn(16)
    srcs = [torch.randn(2, 5, 16) * (i + 1) for i in range(4)]
    assert torch.allclose(op(srcs), torch.stack(srcs).mean(0), atol=1e-6)


def test_attnres_matches_stacked_formula():
    op = DepthAttn(16)
    with torch.no_grad():
        op.w.normal_()
        op.g.uniform_(0.5, 1.5)
    srcs = [torch.randn(2, 5, 16) for _ in range(3)]
    V = torch.stack(srcs)
    Kn = V * torch.rsqrt(V.pow(2).mean(-1, keepdim=True) + 1e-5)
    a = (Kn * (op.g * op.w)).sum(-1).softmax(0)
    assert torch.allclose(op(srcs), (a.unsqueeze(-1) * V).sum(0), atol=1e-5)


def test_attnres_cpu_calls_stay_eager():
    from moe_llm.layers import attnres
    attnres.set_attnres_compile(True)
    before = dict(attnres._BY_ARITY)
    op = DepthAttn(16)
    srcs = [torch.randn(2, 5, 16) for _ in range(3)]
    assert torch.equal(op(srcs), attnres._depth_attn(op.w, op.g, op.eps, *srcs))
    assert attnres._BY_ARITY == before        # torch.compile is only used for CUDA tensors


def test_attnres_compiled_copy_matches_eager():
    """The per-source-count torch.compile'd copy (used on CUDA) gives the same values and gradients."""
    from moe_llm.layers import attnres
    torch.manual_seed(0)
    for n in (2, 4):
        srcs = [torch.randn(2, 8, 16, requires_grad=True) for _ in range(n)]
        w = (0.5 * torch.randn(16)).requires_grad_()
        g = (1 + 0.1 * torch.randn(16)).requires_grad_()
        ref = attnres._depth_attn(w, g, 1e-5, *srcs)
        ref_grads = torch.autograd.grad(ref.square().sum(), [w, g, *srcs])
        try:
            out = attnres.compiled_depth_attn(n)(w, g, 1e-5, *srcs)
        except Exception as e:  # no C++ compiler etc.
            pytest.skip(f"torch.compile unavailable here: {e!r}")
        grads = torch.autograd.grad(out.square().sum(), [w, g, *srcs])
        assert torch.allclose(out, ref, atol=1e-5)
        for a, b in zip(grads, ref_grads):
            assert torch.allclose(a, b, atol=1e-4)
    assert attnres.compiled_depth_attn(2) is attnres.compiled_depth_attn(2)
    assert attnres.compiled_depth_attn(2) is not attnres.compiled_depth_attn(4)
