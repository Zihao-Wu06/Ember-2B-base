"""CED (causal encoder-decoder, DeepSeek-V4.1-Flash) decoder attention and model wiring: ablation G."""
import copy
import math

import torch

from moe_llm.config import ablation_ladder, tiny
from moe_llm.layers.attention import CEDAttention, ced_mask
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, split_params, wsd_lr


def _cfg(**kw):
    return tiny(engram_layers=(), ced_split=2, ced_window=4, **kw)


def _ref(layer, x, enc):
    """Per-query softmax over the allowed global (enc, s <= t) and local (x, t-W < s <= t) entries."""
    B, T, _ = x.shape
    Hq, Hkv, hd, W = layer.Hq, layer.Hkv, layer.hd, layer.window
    q = layer.q_norm(layer.q_proj(x).view(B, T, Hq, hd))
    kg = layer.kg_norm(layer.kg_proj(enc).view(B, T, Hkv, hd))
    vg = layer.vg_proj(enc).view(B, T, Hkv, hd)
    kl = layer.kl_norm(layer.kl_proj(x).view(B, T, Hkv, hd))
    vl = layer.vl_proj(x).view(B, T, Hkv, hd)
    rep = Hq // Hkv
    out = torch.zeros(B, T, Hq, hd, dtype=x.dtype)
    for t in range(T):
        loc = [s for s in range(T) if t - W < s <= t]
        keys = torch.cat([kg[:, :t + 1], kl[:, loc]], 1).repeat_interleave(rep, 2)  # [B, n, Hq, hd]
        vals = torch.cat([vg[:, :t + 1], vl[:, loc]], 1).repeat_interleave(rep, 2)
        s = torch.einsum("bhd,bnhd->bhn", q[:, t], keys) / math.sqrt(hd)
        out[:, t] = torch.einsum("bhn,bnhd->bhd", s.softmax(-1), vals)
    gate = torch.sigmoid(layer.gate_proj(x).view(B, T, Hq, hd))
    return layer.o_proj((out * gate).reshape(B, T, Hq * hd))


def _inputs(cfg, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(2, T, cfg.d_model, generator=g, dtype=torch.float64),
            torch.randn(2, T, cfg.d_model, generator=g, dtype=torch.float64))


def test_ced_mask():
    m = ced_mask(6, 2, "cpu")
    assert m.shape == (6, 12)
    assert m[3, :6].tolist() == [True] * 4 + [False] * 2                      # global: s <= 3
    assert m[3, 6:].tolist() == [False, False, True, True, False, False]      # local: s in {2, 3}
    assert m[0, 6:].tolist() == [True] + [False] * 5


def test_ced_attention_matches_reference():
    torch.manual_seed(0)
    cfg = _cfg()
    layer = CEDAttention(cfg, 2).double()
    x, enc = _inputs(cfg, 11)
    assert torch.allclose(layer(x, enc), _ref(layer, x, enc), atol=1e-10)


def test_ced_attention_locality():
    """Output at t depends on the encoder output at every s <= t, on the layer input only in the window."""
    torch.manual_seed(0)
    cfg = _cfg()
    layer = CEDAttention(cfg, 2).double()
    T, t, W = 12, 10, cfg.ced_window
    x, enc = _inputs(cfg, T)
    base = layer(x, enc)[:, t]

    def changed(which, s):
        x2, e2 = x.clone(), enc.clone()
        (x2 if which == "x" else e2)[:, s] += 1.0
        return not torch.allclose(layer(x2, e2)[:, t], base, atol=1e-12)

    assert changed("enc", 0) and changed("enc", t) and not changed("enc", t + 1)
    assert changed("x", t - W + 1) and changed("x", t)
    assert not changed("x", t - W) and not changed("x", 0) and not changed("x", t + 1)


def _batch(cfg, B=2, T=24, seed=0):
    x = torch.randint(0, cfg.vocab_size, (B, T + 1), generator=torch.Generator().manual_seed(seed))
    return x[:, :-1], x[:, 1:]


def test_ced_model_grads_causal_checkpointing():
    torch.manual_seed(0)
    cfg = _cfg(seq_aux_coef=0.0)
    m = MoELLM(cfg)
    assert [b.kind for b in m.blocks] == ["K", "K", "C", "C"]
    x, y = _batch(cfg)
    m2 = copy.deepcopy(m)
    m2.grad_checkpointing = True
    l1, _ = m(x, y)
    l2, _ = m2(x, y)
    l1.backward()
    l2.backward()
    assert not [n for n, p in m.named_parameters() if p.grad is None]
    assert torch.allclose(l1, l2, atol=1e-6)
    for (n, p1), p2 in zip(m.named_parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), n
    m.eval()
    x2 = x.clone()
    x2[:, 15] = (x2[:, 15] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5) and not torch.allclose(a[:, 15:], b[:, 15:])


def test_ced_param_groups_and_training():
    torch.manual_seed(0)
    cfg = _cfg()
    m = MoELLM(cfg)
    g = split_params(m)
    names = {id(p): n for n, p in m.named_parameters()}
    assert sum(len(v) for v in g.values()) == len(list(m.parameters()))
    assert "enc_norm.weight" in {names[id(p)] for p in g["adamw_no_decay"]}
    assert any(names[id(p)].endswith("mixer.kg_proj.weight") for p in g["muon"])
    opt = OptimizerBundle(m, lr=3e-3)
    x, y = _batch(cfg, B=4, T=32)
    first = None
    for step in range(80):
        loss, info = m(x, y)
        loss.backward()
        opt.step(wsd_lr(step, 80, 3e-3, warmup=5))
        opt.zero_grad()
        m.update_router_bias()
        first = first if first is not None else float(info["ce"])
    assert float(info["ce"]) < 0.4 * first, (first, float(info["ce"]))


def test_ablation_g_config_and_budget():
    from moe_llm.budget import report
    cfg = ablation_ladder()["G_kda_ced"]
    assert cfg.layer_types() == list("KKKFKK") + ["C"] * 6
    assert cfg.engram_layers == () and cfg.residual == "standard"
    assert "mixer_ced" in report("G", cfg)
