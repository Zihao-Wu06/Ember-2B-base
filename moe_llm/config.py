"""Model configuration and presets.

Layer indices are 0-based everywhere in code. Papers usually count from 1: Engram's
"layers 2 and 12" of a 24-layer model are indices (1, 11) here.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # ---- width / depth / vocab
    vocab_size: int = 65536            # fits uint16 token storage
    d_model: int = 1024
    n_layers: int = 24
    norm_eps: float = 1e-6
    init_std: float = 0.02
    tie_embeddings: bool = False
    max_seq_len: int = 4096

    # ---- token mixers. Pattern is tiled over the depth: K = KDA, F = full attention.
    # "KKKF" x 6 puts full attention at 3, 7, ..., 23 (the last layer is always full attention,
    # as in Kimi Linear).
    layer_pattern: str = "KKKF"

    # KDA (Kimi Delta Attention), fla `KimiDeltaAttention` semantics
    kda_heads: int = 8
    kda_head_dim: int = 128
    kda_conv_size: int = 4
    kda_lowrank: int = 128             # rank of the f (decay) / g (output gate) bottlenecks
    kda_backend: str = "auto"          # auto | fla | torch  (torch = naive recurrence, tests only)
    # JSON of pinned launch configs for fla's KDA Triton kernels (scripts/sweep_kda.py, layers/kda_tuning.py); the env
    # var MOE_KDA_TUNING wins over it. "" = off: fla autotunes as before. Rounding-level differences only.
    kda_tuning: str = ""
    # Number of KDA layers (evenly spaced over the KDA layers, see kda_keep_ids) whose chunk_kda keeps its intermediate
    # states (gate cumsum, w, u, qg, kg, v_new, h) for the backward instead of recomputing them (fla's
    # disable_recompute=True): bit-identical values and gradients, less backward work, more memory held per layer
    # (~144 MiB at B=2, T=4096, H=8, K=V=128). Layers that MoELLM.set_checkpointing recomputes or offloads never keep.
    # 0 = off (today's behaviour); values above the number of KDA layers mean all of them.
    kda_keep_layers: int = 0
    # KDA's q/k/v short convs (ShortConv: causal depthwise conv, kernel kda_conv_size, then SiLU), layers/common.py:
    #   "compiled"  = conv_silu (today's path): with fusion on torch.compile'd shifted multiply-adds, else eager conv1d
    #   "fla"       = fla's Triton causal_conv1d kernels (fla.modules.causal_conv1d, activation="silu")
    #   "fla_gluon" = fla's Gluon kernels for the same op (fla.modules.causal_conv1d.backends.gluon; backward in one
    #                 kernel, fixed launch configs)
    # Same parameters and checkpoints, switchable on a restart. The fla paths use the same bf16-rounded weight and the
    # same fp32 sum (same order) of the pre-activation, then apply SiLU in fp32 before the one bf16 rounding (the
    # compiled path rounds the pre-activation first, then SiLU, then rounds again), and keep the SiLU gradient in fp32
    # (the compiled path rounds it to bf16): rounding-level, closer to the exact values. They save the bf16 conv input
    # for the backward instead of the compiled graph's fp32 padded copy (half the bytes). CUDA only (CPU tensors run
    # "compiled"); the Engram conv (dilation 3) and fuse_proj's merged convs (split_conv_silu) always run "compiled".
    conv_impl: str = "compiled"

    # Full attention (GQA). NoPE by default in the hybrid: KDA layers carry position info,
    # and no RoPE means no RoPE re-tuning when extending to 32K.
    attn_heads: int = 8
    attn_kv_heads: int = 2
    attn_head_dim: int = 128
    attn_rope: bool = False
    rope_theta: float = 10000.0
    attn_qk_norm: bool = True
    attn_out_gate: bool = True         # sigmoid gate on the attention output (elementwise)
    # log-n attention scaling for length extension (PLAN_LONGCTX.md): a query that sees n keys multiplies its logits by
    # max(1, ln n / ln attn_logn), so attention does not spread out over more keys than at the training length.
    # Identical to no scaling for n <= attn_logn. 0 = off. Measured without training on the 4K main run (step 45000,
    # runs/mac_eval/long_ctx.py --logn): 32K passkey 1/5 -> 5/5, loss beyond 8K closer to the 4K-window baseline.
    attn_logn: int = 0

    # ---- CED: causal encoder-decoder (DeepSeek-V4.1-Flash, dense variant). Ablation G only.
    # Layers >= ced_split become decoder layers ("C"): queries from the layer itself, global K/V
    # projected with layer-specific weights from the encoder output (the hidden state entering
    # layer ced_split), local K/V from the layer itself over the last ced_window tokens, one softmax.
    # No recurrence and no long-range KV of their own, so at inference prompt tokens could skip the
    # decoder. 0 = off.
    ced_split: int = 0
    ced_window: int = 128

    # ---- FFN / MoE (DeepSeek-V3 style: shared expert + fine-grained routed experts,
    # sigmoid router, aux-loss-free bias balancing + tiny sequence-wise aux loss)
    n_dense_layers: int = 1            # first k layers use a dense SwiGLU
    dense_inter: int = 2816
    n_experts: int = 32
    top_k: int = 4
    expert_inter: int = 512
    n_shared: int = 1
    shared_inter: int = 768
    routed_scale: float = 1.0
    router_bias_update: float = 1e-3   # gamma for the aux-loss-free bias
    seq_aux_coef: float = 1e-4         # alpha for the sequence-wise balance loss
    moe_impl: str = "bmm"              # bmm (fastest on the 4090) | grouped | loop | auto | triton (no padding)
    # MoE input fan-in/fan-out fused into the routed experts' Triton combine kernels (layers/moe.py, MoEFan in
    # layers/moe_triton.py; needs moe_impl="triton"). Off (""): the shared expert's output is added to the routed output
    # by a separate [N, d] add, and the MoE input's gradient is summed from four parts (shared gate, shared up, routed,
    # router) by three [N, d] bf16 adds, the router's part after a fp32 [N, d] dx GEMM output and its cast to bf16.
    #   "exact":  the shared expert runs first and the routed combine adds its output; every reader of the input gets
    #             its own view of it, and one combine kernel sums the four gradient parts with the old adds' roundings
    #             in the old order: forward and every gradient bit-identical, four [N, d] adds fewer per MoE layer and
    #             micro-batch (the router's fp32 dx GEMM and cast stay).
    #   "router": "exact", and the router's GEMM moves into the fan node: its input gradient dlogits @ W_r is computed in
    #             that kernel as fp32 products (no fp32 [N, d] GEMM output, no cast) and the whole input gradient is
    #             summed in fp32 and rounded once (the old path rounds four times and computes dlogits @ W_r in TF32).
    #             Forward, routing and every weight gradient bit-identical; only the MoE input's gradient changes,
    #             towards the exact sum.
    #   "router_lean": "router", and the router's weight gradient is computed from the bf16 input (exact bf16 products of
    #             a three-term bf16 split of dlogits, fp32 sums: fp32-GEMM accuracy) instead of from a fp32 copy of the
    #             input kept for the backward: 32 MiB less memory per MoE layer and micro-batch at d = 1024, N = 8192
    #             (736 MiB on main_2b); that gradient changes at the rounding level (the old GEMM runs in TF32 on the GPU).
    # Switchable on a restart (--override '{"moe_shared_fused": "router"}'); no parameter or checkpoint change.
    moe_shared_fused: str = ""
    # moe_impl="triton": how the token-expert assignments are sorted by expert (layers/moe_triton.py sort_assignments):
    # "torch" (torch.sort + searchsorted + argsort) or "counting" (three counting-sort kernels; the same integers,
    # and the per-expert counts for the load counter come from it). No parameter or checkpoint change.
    moe_sort: str = "torch"

    # ---- Engram (conditional memory). Empty tuple disables it.
    engram_layers: tuple = (1, 11)
    engram_orders: tuple = (2, 3)
    engram_heads: int = 8              # hash heads per n-gram order
    engram_dim: int = 640              # d_mem = len(orders) * heads * d_head  -> d_head 40
    engram_rows: int | tuple = 390_000  # base rows per head table, or one base per order (distinct primes >= base)
    engram_conv_size: int = 4          # depthwise causal conv, dilation = max order
    engram_signed_sqrt: bool = True    # official demo: sign(s)*sqrt(|s|) before the sigmoid gate
    engram_offload: bool = False       # True: tables live in CPU RAM with sparse (lazy) Adam
    engram_seed: int = 0

    # ---- residual stream
    residual: str = "standard"         # standard | block_attnres | full_attnres
    attnres_layers_per_block: int = 3  # block AttnRes: transformer layers per block (24/3 = 8 blocks)
    # block AttnRes implementation (same parameters and math, rounding-level differences): "compiled" =
    # layers/attnres.py (torch.compile per op); "fused" = layers/attnres_fused.py (Triton, one shared backward
    # per block, gradients of the block outputs accumulated in-kernel)
    attnres_impl: str = "compiled"

    # ---- loss
    z_loss: float = 1e-4
    # LM head + cross-entropy (moe_llm/loss.py): "chunked" = eager elementwise ops (the reference) | "fused" = one
    # Triton kernel per chunk, fp32 logits (more accurate) | "fused_bf16" = the same with bf16 logits as "chunked"
    ce_impl: str = "chunked"
    ce_chunk: int = 2048               # rows of the [rows, vocab] logits buffer per step of the loss loop
    ce_blas: str = ""                  # ce_impl "fused": "" = default cuBLAS, "cublaslt_logits" = logits GEMM via cuBLASLt (bit-identical, faster), "cublaslt" = all (grad_h not bit-identical)

    # ---- speed knobs (numerics-neutral or nearly so)
    residual_bf16: bool = False        # keep the residual stream in bf16 on GPU (Megatron default)
    # fused projections (layers/common.py fused_linear): the bias-free projections that read the same input run as
    # one GEMM over their stacked weights (KDA q/k/v/f_down/g_down/b_proj, N = 3336 on main_2b, with the q/k/v short
    # convs in one function; attention q/k/v/gate, N = 2560). Same parameters, names and checkpoints. Forward: the
    # same dot products; backward: dx in one GEMM (fp32 accumulation over all members, one bf16 rounding, instead of
    # bf16 partial dx added in bf16: rounding-level, closer to the exact sum); dW: the stack's GEMM, each member's row
    # block (same products). Switchable on a restart (--override '{"fuse_proj": true}').
    fuse_proj: bool = False
    # with fuse_proj and --offload-master --fused-grad-acc: the fp32 weight-gradient GEMM into the stacked buffer runs
    # "stacked" (one GEMM for the whole stack) or "member" (one per member: the separate layers' GEMMs, so their dW
    # values are unchanged wherever the GEMM library sums an element in another order for another row count)
    fuse_proj_dw: str = "stacked"

    # ------------------------------------------------------------------ helpers
    def layer_types(self) -> list[str]:
        pat = self.layer_pattern
        types = [pat[i % len(pat)] for i in range(self.n_layers)]
        if "F" in pat:
            types[-1] = "F"
        if self.ced_split:
            types[self.ced_split:] = ["C"] * (self.n_layers - self.ced_split)
        return types

    def kda_keep_ids(self) -> set[int]:
        """Layer ids of the kda_keep_layers KDA layers, spaced over the KDA layers the way MoELLM.set_checkpointing
        spaces recomputed layers (n of L: int(j * L / n)). main_2b, 6 of 18: the first KDA layer of every KKKF group."""
        if self.kda_keep_layers < 0:
            raise ValueError(f"kda_keep_layers={self.kda_keep_layers} must be >= 0")
        kda = [i for i, t in enumerate(self.layer_types()) if t == "K"]
        n = min(self.kda_keep_layers, len(kda))
        return {kda[int(j * len(kda) / n)] for j in range(n)} if n else set()

    def is_moe(self, i: int) -> bool:
        return i >= self.n_dense_layers and self.n_experts > 0

    @property
    def engram_d_head(self) -> int:
        n_heads = len(self.engram_orders) * self.engram_heads
        assert self.engram_dim % n_heads == 0, "engram_dim must divide by orders*heads"
        return self.engram_dim // n_heads

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**d)

    def replace(self, **kw) -> "ModelConfig":
        return dataclasses.replace(self, **kw)


# ---------------------------------------------------------------------- presets
def main_2b() -> ModelConfig:
    """The planned pretraining model: ~1.96B total (1.46B backbone on GPU + 0.50B Engram),
    ~0.38B active (incl. LM head). Engram tables live in CPU RAM (they would need ~8 GB of GPU
    memory with Adam states)."""
    # residual_bf16: +10-18% measured on the 4090 (bench 2026-11-09), loss within 1% of fp32 residual
    # block_attnres: ablation D -0.004 vs standard; only 8.1% slower on the 4090 with fusion on (2026-11-10),
    # within the agreed <= 10% rule (docs/DESIGN.md sections 6 and 7.5)
    return ModelConfig(engram_offload=True, residual_bf16=True, residual="block_attnres")


def ablation_small(**kw) -> ModelConfig:
    """Small model for 4090 ablations (~0.35B total, ~70M FLOP-active). Same building
    blocks and ratios as main_2b, half width and depth."""
    cfg = ModelConfig(
        d_model=512, n_layers=12,
        kda_heads=4, kda_head_dim=128, kda_lowrank=64,
        attn_heads=4, attn_kv_heads=1, attn_head_dim=128,
        dense_inter=1408, n_experts=32, top_k=4, expert_inter=256, shared_inter=384,
        engram_layers=(1, 6), engram_dim=320, engram_rows=200_000,
        attnres_layers_per_block=3,
    )
    return cfg.replace(**kw)


def tiny(**kw) -> ModelConfig:
    """Unit-test sized model (runs on CPU in milliseconds)."""
    cfg = ModelConfig(
        vocab_size=512, d_model=64, n_layers=4, max_seq_len=64,
        kda_heads=2, kda_head_dim=16, kda_lowrank=16, kda_conv_size=4,
        attn_heads=4, attn_kv_heads=2, attn_head_dim=16,
        n_dense_layers=1, dense_inter=128, n_experts=8, top_k=2, expert_inter=32,
        n_shared=1, shared_inter=32,
        engram_layers=(1,), engram_heads=2, engram_dim=32, engram_rows=97,
        attnres_layers_per_block=2,
    )
    return cfg.replace(**kw)


# Ablation ladder: each step adds one component on top of the previous one.
def ablation_ladder() -> dict[str, ModelConfig]:
    base = ablation_small()
    return {
        "A_moe_fullattn": base.replace(layer_pattern="F", attn_rope=True, engram_layers=()),
        "B_kda_hybrid": base.replace(engram_layers=()),
        "C_kda_engram": base,
        "D_kda_engram_attnres": base.replace(residual="block_attnres"),
        "E_fine_experts": base.replace(residual="block_attnres", n_experts=64, top_k=8,
                                       expert_inter=128, shared_inter=384),
        # real-data stats: 2-grams are covered ~87% by 390K rows/head, 3-grams only ~60%
        "F_engram_3gram_heavy": base.replace(residual="block_attnres", engram_rows=(120_000, 280_000)),
        # CED (DeepSeek-V4.1-Flash): B with layers 6-11 turned into decoder layers; compare with B
        "G_kda_ced": base.replace(engram_layers=(), ced_split=6),
    }


PRESETS = {"main_2b": main_2b, "ablation_small": ablation_small, "tiny": tiny}
