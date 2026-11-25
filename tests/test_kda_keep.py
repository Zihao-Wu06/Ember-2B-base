"""ModelConfig.kda_keep_layers: chosen KDA layers keep chunk_kda's intermediate states for the backward (fla
disable_recompute=True) instead of recomputing them. CPU: which layers, that nothing changes at 0, that the flag
reaches chunk_kda only when a backward can follow, and that recomputed / offloaded layers never keep. GPU: bit-identical
outputs and gradients (one main-size KDA layer, a small model, with and without activation checkpointing) and the
memory each keeping layer holds."""
import copy

import pytest
import torch

from moe_llm.config import ModelConfig, main_2b, tiny
from moe_llm.layers import kda as kda_mod
from moe_llm.layers.kda import KDA, kda_keep_bytes
from moe_llm.model import MoELLM

MAIN_KDA = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]   # KKKF x 6


# ----------------------------------------------------------------------------- CPU: selection
def test_keep_ids_are_evenly_spaced_over_the_kda_layers():
    cfg = main_2b()
    assert [i for i, t in enumerate(cfg.layer_types()) if t == "K"] == MAIN_KDA
    assert cfg.kda_keep_layers == 0 and cfg.kda_keep_ids() == set()
    assert cfg.replace(kda_keep_layers=6).kda_keep_ids() == {0, 4, 8, 12, 16, 20}    # first K of every KKKF group
    assert cfg.replace(kda_keep_layers=3).kda_keep_ids() == {0, 8, 16}
    assert cfg.replace(kda_keep_layers=1).kda_keep_ids() == {0}
    assert cfg.replace(kda_keep_layers=18).kda_keep_ids() == set(MAIN_KDA)
    assert cfg.replace(kda_keep_layers=99).kda_keep_ids() == set(MAIN_KDA)
    for n in range(19):     # the rule MoELLM.set_checkpointing uses for n of L layers: int(j * L / n)
        ids = cfg.replace(kda_keep_layers=n).kda_keep_ids()
        assert len(ids) == n and ids == {MAIN_KDA[int(j * 18 / n)] for j in range(n)}
    assert cfg.replace(layer_pattern="F", kda_keep_layers=4).kda_keep_ids() == set()
    with pytest.raises(ValueError):
        cfg.replace(kda_keep_layers=-1).kda_keep_ids()


def test_off_by_default_nothing_changes(monkeypatch):
    assert ModelConfig().kda_keep_layers == 0
    old = {k: v for k, v in tiny().to_dict().items() if k != "kda_keep_layers"}   # configs saved before the field
    assert ModelConfig.from_dict(old).kda_keep_layers == 0
    assert ModelConfig.from_dict(tiny(kda_keep_layers=3).to_dict()).kda_keep_layers == 3
    m = MoELLM(tiny())
    assert m.kda_keep_list() == [] and m.recompute_summary() == "recompute off"
    # the chunk_kda call is exactly today's: same keyword arguments, no disable_recompute
    calls = _fake_fla(monkeypatch)
    x = torch.randint(0, 512, (2, 17))
    loss, _ = m(x[:, :-1], x[:, 1:])
    loss.backward()
    assert len(calls) == 3 and all(sorted(c) == sorted(TODAY_KWARGS) for c in calls)


TODAY_KWARGS = ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "use_qk_l2norm_in_kernel", "use_gate_in_kernel",
                "use_beta_sigmoid_in_kernel"]


def _fake_fla(monkeypatch):
    """Route the CPU forward through the fla branch with a fake chunk_kda that records its keyword arguments and
    returns the reference output (so the model still trains)."""
    calls = []

    def fake_chunk_kda(q, k, v, g, beta, A_log, dt_bias, **kw):
        calls.append(dict(kw, q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias))
        return kda_mod.kda_reference(q, k, v, g, beta, A_log, dt_bias), None

    monkeypatch.setattr(KDA, "_use_fla", lambda self, x: True)
    monkeypatch.setattr(kda_mod, "fla_chunk_kda", lambda: fake_chunk_kda)
    return calls


def test_flag_reaches_chunk_kda_only_in_kept_layers_with_a_backward(monkeypatch):
    torch.manual_seed(0)
    m_off = MoELLM(tiny())
    m_keep = MoELLM(tiny(kda_keep_layers=2))
    m_keep.load_state_dict(m_off.state_dict())
    assert m_keep.kda_keep_list() == [0, 1]           # tiny: KKKF -> KDA layers 0, 1, 2; 2 of 3 -> 0, 1
    assert "KDA states kept for the backward in layers [0, 1]" in m_keep.recompute_summary()
    calls = _fake_fla(monkeypatch)
    x = torch.randint(0, 512, (2, 17))
    out = {}
    for name, m in (("off", m_off), ("keep", m_keep)):
        calls.clear()
        loss, _ = m(x[:, :-1], x[:, 1:])
        loss.backward()
        out[name] = (loss.detach(), [p.grad.clone() for p in m.parameters() if p.grad is not None])
        flags = [c.get("disable_recompute") for c in calls]
        assert flags == ([None, None, None] if name == "off" else [True, True, None])
    assert torch.equal(out["off"][0], out["keep"][0])     # the reference path ignores the flag
    assert all(torch.equal(a, b) for a, b in zip(out["off"][1], out["keep"][1]))
    calls.clear()
    with torch.no_grad():                                   # no backward can follow: plain call
        m_keep(x[:, :-1])
    assert [c.get("disable_recompute") for c in calls] == [None, None, None]


def test_recomputed_or_offloaded_layers_never_keep():
    m = MoELLM(tiny(n_layers=8, kda_keep_layers=99))       # KKKF KKKF: KDA layers 0, 1, 2, 4, 5, 6, all requested
    assert m.kda_keep_list() == [0, 1, 2, 4, 5, 6]
    m.set_checkpointing(0.5)                                # recomputes 0, 2, 4, 6
    assert m.kda_keep_list() == [1, 5]
    assert m.recompute_summary().endswith("KDA states kept for the backward in layers [1, 5]")
    m.set_checkpointing(0.5, ffn_frac=0.5)                  # 0 and 4 recompute only their FFN: their mixers keep
    assert m.kda_keep_list() == [0, 1, 4, 5]
    m.set_checkpointing(0.5, offload_frac=0.5)              # 0 and 4 offloaded, 2 and 6 recomputed
    assert sorted(m.offload_act) == [0, 4] and m.kda_keep_list() == [1, 5]
    m.set_checkpointing(1.0)
    assert m.kda_keep_list() == []
    m.set_checkpointing(0.0)
    assert m.kda_keep_list() == [0, 1, 2, 4, 5, 6]
    m2 = copy.deepcopy(m)
    assert m2.kda_keep_list() == m.kda_keep_list()


def test_kda_keep_bytes_formula_at_main_shape():
    b = kda_keep_bytes(2, 4096, 8, 128, 128)
    # 33.6 MB fp32 gate cumsum + 5 x 16.8 MB bf16 (w, qg, kg, u, v_new) + 33.6 MB bf16 h
    assert b == 33_554_432 + 5 * 16_777_216 + 33_554_432 == 150_994_944 and b / 2 ** 20 == 144.0


# ----------------------------------------------------------------------------- GPU
def _deterministic_cudnn():
    old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False   # ShortConv's conv1d backward
    return old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kda_layer_keep_states_bit_identical_and_memory_gpu():
    """One main-size KDA layer (d 1024, H 8, K=V 128, B=2, T=4096, bf16 autocast) on fla's chunk_kda: the layer
    output, the input gradient and every parameter gradient are bit-identical with and without keep_states, and
    keeping holds exactly the seven saved states (144 MiB) more until the backward."""
    assert kda_mod.fla_chunk_kda() is not None, "fla not importable"
    old = _deterministic_cudnn()
    try:
        torch.manual_seed(0)
        cfg = ModelConfig()
        layer = KDA(cfg, 0).cuda()
        B, T = 2, 4096
        x0 = torch.randn(B, T, cfg.d_model, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(B, T, cfg.d_model, device="cuda", dtype=torch.bfloat16)

        def run(keep):
            layer.keep_states = keep
            layer.zero_grad(set_to_none=True)
            x = x0.clone().requires_grad_()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = layer(x)
            torch.cuda.synchronize()
            held = torch.cuda.memory_allocated() - base
            y.backward(dy)
            torch.cuda.synchronize()
            return held, [y.detach(), x.grad] + [p.grad for p in layer.parameters()]

        held0, ref = run(False)
        _, again = run(False)
        assert all(torch.equal(a, b) for a, b in zip(ref, again)), "the KDA layer is not deterministic run to run"
        held1, kept = run(True)
        for i, (a, b) in enumerate(zip(ref, kept)):
            assert torch.equal(a, b), i
        extra, want = held1 - held0, kda_keep_bytes(B, T, cfg.kda_heads, cfg.kda_head_dim, cfg.kda_head_dim)
        print(f"KDA keep_states: +{extra / 2 ** 20:.1f} MiB held per layer until the backward (formula "
              f"{want / 2 ** 20:.1f} MiB) at B={B} T={T} H={cfg.kda_heads} K=V={cfg.kda_head_dim}")
        assert abs(extra - want) <= 0.01 * want
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_model_keep_states_bit_identical_gpu():
    """A small all-KDA model (dense FFN, no Engram / attention, whose kernels are deterministic) on the real GPU path
    (bf16 autocast, bf16 residual): loss and every gradient bit-identical with kda_keep_layers 0 vs 2 and vs all four
    with half the layers recomputed (the recomputed ones then keep nothing: memory grows by 2 layers, not 4)."""
    assert kda_mod.fla_chunk_kda() is not None, "fla not importable"
    old = _deterministic_cudnn()
    try:
        torch.manual_seed(0)
        cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, layer_pattern="K", n_experts=0, engram_layers=(),
                   residual_bf16=True)
        B, T = 2, 1024
        x = torch.randint(0, cfg.vocab_size, (B, T + 1), device="cuda")
        per_layer = kda_keep_bytes(B, T, cfg.kda_heads, cfg.kda_head_dim, cfg.kda_head_dim)
        m_off = MoELLM(cfg).cuda()

        def make(n, ckpt):
            m = MoELLM(cfg.replace(kda_keep_layers=n)).cuda()
            m.load_state_dict(m_off.state_dict())
            m.set_checkpointing(ckpt)
            return m

        def run(m):
            m.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = m(x[:, :-1], x[:, 1:])
            torch.cuda.synchronize()
            held = torch.cuda.memory_allocated() - base
            loss.backward()
            torch.cuda.synchronize()
            return held, {"loss": loss.detach(), **{n: p.grad.clone() for n, p in m.named_parameters()}}

        held0, ref = run(m_off)
        _, again = run(m_off)
        nondet = [n for n in ref if not torch.equal(ref[n], again[n])]
        assert not [n for n in nondet if n == "loss" or ".mixer." in n], f"KDA path not deterministic: {nondet}"
        if nondet:
            print(f"not deterministic run to run, compared to 1e-3 only: {nondet}")
        cases = [("keep 2", make(2, 0.0), [0, 2], 2), ("keep 4, recompute half", make(4, 0.5), [1, 3], 2)]
        ref_ck_held = None
        for label, m, kept, n_extra in cases:
            assert m.kda_keep_list() == kept, label
            held, out = run(m)
            for n in ref:
                if n in nondet:
                    assert torch.allclose(out[n].float(), ref[n].float(), rtol=1e-3, atol=1e-6), (label, n)
                else:
                    assert torch.equal(out[n], ref[n]), (label, n)
            if label.startswith("keep 4"):
                ref_ck_held = run(make(0, 0.5))[0]               # the same checkpointing without the option
                base_held = ref_ck_held
            else:
                base_held = held0
            extra = held - base_held
            print(f"{label}: +{extra / 2 ** 20:.2f} MiB held after the forward ({n_extra} keeping layers x "
                  f"{per_layer / 2 ** 20:.2f} MiB)")
            assert abs(extra - n_extra * per_layer) <= 0.05 * n_extra * per_layer, label
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old
