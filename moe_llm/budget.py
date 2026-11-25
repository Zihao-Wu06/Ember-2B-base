"""Static budget from the real module tree (built on the meta device, no memory used):
parameters by component, active parameters, FLOPs per token, and a GPU memory estimate.

    python -m moe_llm.budget                # main_2b + ablation ladder
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .config import ModelConfig, ablation_ladder, main_2b

GiB = 1 << 30


def build_meta(cfg: ModelConfig):
    from .model import MoELLM
    with torch.device("meta"):
        return MoELLM(cfg.replace(engram_offload=False))


def param_report(cfg: ModelConfig) -> dict:
    m = build_meta(cfg)
    cats = defaultdict(int)
    for i, b in enumerate(m.blocks):
        cats[{"K": "mixer_kda", "F": "mixer_full", "C": "mixer_ced"}[b.kind]] += sum(p.numel() for p in b.mixer.parameters())
    for name, p in m.named_parameters():
        n = p.numel()
        if ".mixer." in name:
            continue
        if name.startswith("embed."):
            cats["embedding"] += n
        elif name.startswith("lm_head."):
            cats["lm_head"] += n
        elif name.startswith("engram.tables"):
            cats["engram_tables"] += n
        elif ".engram." in name:
            cats["engram_dense"] += n
        elif ".ffn.w_" in name:
            cats["moe_routed"] += n
        elif ".ffn.shared." in name:
            cats["moe_shared"] += n
        elif ".ffn.router" in name:
            cats["router"] += n
        elif ".ffn.mlp." in name:
            cats["dense_ffn"] += n
        else:
            cats["norm_attnres_other"] += n
    total = sum(cats.values())
    frac = cfg.top_k / cfg.n_experts if cfg.n_experts else 0.0
    routed_active = int(cats["moe_routed"] * frac)
    # parameters touched by matmuls for one token (input embedding / table lookups are gathers)
    matmul_active = total - cats["embedding"] - cats["engram_tables"] - cats["moe_routed"] + routed_active
    return {"cats": dict(cats), "total": total, "gpu_backbone": total - cats["engram_tables"],
            "engram": cats["engram_tables"], "active_matmul": matmul_active,
            "active_non_embedding": matmul_active - cats["lm_head"]}


def flops_per_token(cfg: ModelConfig, seq_len: int, rep: dict | None = None) -> dict:
    rep = rep or param_report(cfg)
    types = cfg.layer_types()
    n_full, n_kda, n_ced = types.count("F"), types.count("K"), types.count("C")
    dense = 6 * rep["active_matmul"]
    attn = n_full * 6 * seq_len * cfg.attn_heads * cfg.attn_head_dim          # causal QK^T + AV, fwd+bwd
    # CED decoder: causal global part + local window (useful FLOPs; the dense-mask kernel does more)
    attn += n_ced * (6 * seq_len + 12 * cfg.ced_window) * cfg.attn_heads * cfg.attn_head_dim
    D, C, K = cfg.kda_heads * cfg.kda_head_dim, 64, cfg.kda_head_dim
    kda = n_kda * 3 * (4 * C * D + 6 * cfg.kda_heads * K * K)                 # chunked delta rule (approx.)
    return {"matmul": dense, "attention": attn, "kda": kda, "total": dense + attn + kda}


def memory_estimate(cfg: ModelConfig, rep: dict | None = None, tokens_per_micro: int = 8192,
                    muon_state_bytes: int = 2, offload_engram: bool = True) -> dict:
    """Rough GPU bytes for training with fp32 master params, fp32 grads, per-layer recompute."""
    rep = rep or param_report(cfg)
    c = rep["cats"]
    adam_params = c["embedding"] + c["lm_head"] + c["router"] + c["norm_attnres_other"]
    muon_params = rep["gpu_backbone"] - adam_params
    states = rep["gpu_backbone"] * 8 + muon_params * muon_state_bytes + adam_params * 8
    if not offload_engram:
        states += rep["engram"] * 16
    d = cfg.d_model
    n_ckpt = cfg.n_layers + (cfg.n_layers // max(1, cfg.attnres_layers_per_block) if cfg.residual == "block_attnres" else 0)
    act = tokens_per_micro * d * 2 * n_ckpt                     # saved layer inputs (bf16)
    act += tokens_per_micro * d * 4 * 40                        # one layer recomputed (rough upper bound)
    act += 2048 * cfg.vocab_size * 6                            # CE chunk buffer
    return {"states_GiB": states / GiB, "activations_GiB": act / GiB, "total_GiB": (states + act) / GiB + 1.0}


def report(name: str, cfg: ModelConfig, seq_len: int = 4096, tokens_per_micro: int = 8192) -> str:
    rep = param_report(cfg)
    f = flops_per_token(cfg, seq_len, rep)
    mem = memory_estimate(cfg, rep, tokens_per_micro, offload_engram=cfg.engram_offload or rep["engram"] > 2e8)
    c = rep["cats"]
    lines = [f"== {name}: total {rep['total'] / 1e9:.3f}B  (GPU backbone {rep['gpu_backbone'] / 1e9:.3f}B + "
             f"Engram {rep['engram'] / 1e9:.3f}B)  active {rep['active_matmul'] / 1e6:.0f}M "
             f"(non-emb {rep['active_non_embedding'] / 1e6:.0f}M)",
             "   " + "  ".join(f"{k}={v / 1e6:.1f}M" for k, v in sorted(c.items())),
             f"   FLOPs/token @ {seq_len}: {f['total'] / 1e9:.2f} G (matmul {f['matmul'] / 1e9:.2f}, "
             f"attn {f['attention'] / 1e9:.2f}, kda {f['kda'] / 1e9:.2f})",
             f"   GPU memory est. ({tokens_per_micro} tok/micro): states {mem['states_GiB']:.1f} GiB + "
             f"activations {mem['activations_GiB']:.1f} GiB + ctx ~1 GiB = {mem['total_GiB']:.1f} GiB"]
    for tf in (40e12, 55e12):
        tps = tf / f["total"]
        lines.append(f"   @ {tf / 1e12:.0f} TFLOP/s effective: {tps / 1e3:.1f}k tok/s = {tps * 86400 / 1e9:.2f}B tok/day")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report("main_2b", main_2b()))
    print(report("main_2b @32K", main_2b(), seq_len=32768, tokens_per_micro=32768))
    for n, c in ablation_ladder().items():
        print(report(n, c, tokens_per_micro=16384))
