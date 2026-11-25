"""Kimi Delta Attention (KDA), following fla 0.6.0 `KimiDeltaAttention` exactly.

Recurrence (per head):  S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T,  o_t = S_t^T q_t
  a_t = exp(g_t),  g_t = -exp(A_log) * softplus(f_t + dt_bias)   (per channel, log space)
  b_t = sigmoid(W_b x_t),  q/k L2-normalised, q scaled by 1/sqrt(d_k)

GPU: fla's Triton `chunk_kda` with the gate, beta-sigmoid and q/k L2 norm fused in the kernel. In the layers picked by
ModelConfig.kda_keep_layers (keep_states, set by MoELLM) it keeps its intermediate states for the backward
(disable_recompute=True) instead of recomputing them: same values bit for bit, less work, more memory.
CPU / tests: the verbatim port of fla's `naive_recurrent_kda` (token loop, slow but exact).
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kda_tuning
from .common import CastLinear, ShortConv, fused_linear, gated_rms_norm, split_conv_silu

_CHUNK_KDA = None
_CHUNK_KDA_ERR = None


def fla_chunk_kda():
    """Return fla.ops.kda.chunk_kda or None (fla needs Triton, so never on macOS). Once fla is loaded, the pinned
    kernel configs requested by ModelConfig.kda_tuning / MOE_KDA_TUNING are applied (kda_tuning.py; default off)."""
    global _CHUNK_KDA, _CHUNK_KDA_ERR
    if _CHUNK_KDA is None and _CHUNK_KDA_ERR is None:
        try:
            from fla.ops.kda import chunk_kda
            _CHUNK_KDA = chunk_kda
        except Exception as e:  # pragma: no cover - depends on the machine
            _CHUNK_KDA_ERR = e
    if _CHUNK_KDA is not None:
        kda_tuning.sync()
    return _CHUNK_KDA


def kda_keep_bytes(B: int, T: int, H: int, K: int, V: int, chunk: int = 64) -> int:
    """GPU memory one keeping layer (keep_states, ModelConfig.kda_keep_layers) holds from its forward until its
    backward on top of the usual saved tensors: fla's fp32 gate cumsum, w / qg / kg [B,T,H,K] and u / v_new
    [B,T,H,V] in bf16, and the chunk states h [B, T/chunk, H, K, V] in bf16. main_2b at B=2, T=4096: 144 MiB."""
    return B * T * H * K * 4 + 3 * B * T * H * K * 2 + 2 * B * T * H * V * 2 + B * (-(-T // chunk)) * H * K * V * 2


# --------------------------------------------------------------------------- reference math
def kda_gate(g_raw: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """fla naive_kda_gate: g = -exp(A_log[h]) * softplus(g_raw + dt_bias), fp32. g_raw: [B,T,H,K]."""
    H, K = g_raw.shape[-2:]
    return -A_log.float().exp().view(H, 1) * F.softplus(g_raw.float() + dt_bias.float().view(H, K))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    """Verbatim port of fla/ops/kda/naive.py::naive_recurrent_kda (fla 0.6.0, MIT licence)."""
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def kda_reference(q, k, v, g_raw, beta_logits, A_log, dt_bias):
    """Same semantics as chunk_kda(..., use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True)."""
    g = kda_gate(g_raw, A_log, dt_bias)
    beta = torch.sigmoid(beta_logits.float())
    o, _ = naive_recurrent_kda(l2norm(q), l2norm(k), v.float(), g, beta)
    return o.to(v.dtype)


# --------------------------------------------------------------------------- layer
class KDA(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        d, H, K = cfg.d_model, cfg.kda_heads, cfg.kda_head_dim
        D, r, std = H * K, cfg.kda_lowrank, cfg.init_std
        out_std = std / math.sqrt(2 * cfg.n_layers)
        self.H, self.K, self.D, self.r = H, K, D, r
        self.backend = cfg.kda_backend
        self.o_eps = 1e-5  # fla o_norm eps
        # cfg.fuse_proj: q/k/v/f_down/g_down/b_proj (all read x) as one GEMM, the q/k/v short convs as one function
        # (same parameters; see forward)
        self.fuse_proj = bool(getattr(cfg, "fuse_proj", False))
        kda_tuning.configure(getattr(cfg, "kda_tuning", ""))  # pinned fla kernel configs, applied on first fla use
        # keep chunk_kda's states for the backward (cfg.kda_keep_layers); MoELLM.set_checkpointing turns it off in
        # layers it recomputes or offloads
        self.keep_states = layer_idx in cfg.kda_keep_ids()

        self.q_proj = CastLinear(d, D, bias=False)
        self.k_proj = CastLinear(d, D, bias=False)
        self.v_proj = CastLinear(d, D, bias=False)
        # cfg.conv_impl: "compiled" (conv_silu) or fla's kernels for the three short convs (layers/common.py ShortConv)
        conv_impl = getattr(cfg, "conv_impl", "compiled")
        if self.fuse_proj and conv_impl != "compiled":
            warnings.warn(f"conv_impl={conv_impl!r} has no effect with fuse_proj: the merged q/k/v convs run in "
                          "split_conv_silu (compiled)")
        self.q_conv = ShortConv(D, cfg.kda_conv_size, impl=conv_impl)
        self.k_conv = ShortConv(D, cfg.kda_conv_size, impl=conv_impl)
        self.v_conv = ShortConv(D, cfg.kda_conv_size, impl=conv_impl)
        # per-channel decay gate: low-rank d -> r -> H*K
        self.f_down = CastLinear(d, r, bias=False)
        self.f_up = CastLinear(r, D, bias=False)
        self.b_proj = CastLinear(d, H, bias=False)
        # A_log ~ log U(1, 16), dt_bias = inverse-softplus of log-uniform dt in [1e-3, 1e-1]  (fla init)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        dt = torch.exp(torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        # output gate: low-rank d -> r -> H*K (bias on the up projection, as in fla)
        self.g_down = CastLinear(d, r, bias=False)
        self.g_up = CastLinear(r, D, bias=True)
        self.o_norm_weight = nn.Parameter(torch.ones(K))
        self.o_proj = CastLinear(D, d, bias=False)

        for m in (self.q_proj, self.k_proj, self.v_proj, self.f_down, self.f_up, self.b_proj,
                  self.g_down, self.g_up):
            nn.init.normal_(m.weight, std=std)
        nn.init.zeros_(self.g_up.bias)
        nn.init.normal_(self.o_proj.weight, std=out_std)

    def _use_fla(self, x: torch.Tensor) -> bool:
        if self.backend == "torch":
            return False
        ok = x.is_cuda and fla_chunk_kda() is not None
        if self.backend == "fla" and not ok:
            raise RuntimeError(f"kda_backend='fla' but fla is unavailable: {_CHUNK_KDA_ERR}")
        return ok

    def proj_weights(self) -> list:
        """The weights of the projections that read x, in the order of the fused GEMM's output blocks (fuse_proj)."""
        return [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.f_down.weight, self.g_down.weight,
                self.b_proj.weight]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, K = self.H, self.K
        gd = None
        if self.fuse_proj:
            # one GEMM [.., 3D + 2r + H] = [q | k | v | f_down | g_down | b_proj] (main_2b: N = 3336), then the q/k/v
            # short convs over its first 3D channels in one function; the gradient of the GEMM output comes back as one
            # tensor, so dx is one GEMM too (layers/common.py fused_linear, split_conv_silu)
            D, r = self.D, self.r
            y = fused_linear(x, self.proj_weights())
            w = torch.cat([self.q_conv.weight, self.k_conv.weight, self.v_conv.weight])
            q, k, v, fd, gd, beta = split_conv_silu(y, w, (D, D, D), (r, r, H))
            q, k, v = q.view(B, T, H, K), k.view(B, T, H, K), v.view(B, T, H, K)
            g = self.f_up(fd).view(B, T, H, K)
        else:
            q = self.q_conv(self.q_proj(x)).view(B, T, H, K)
            k = self.k_conv(self.k_proj(x)).view(B, T, H, K)
            v = self.v_conv(self.v_proj(x)).view(B, T, H, K)
            g = self.f_up(self.f_down(x)).view(B, T, H, K)
            beta = self.b_proj(x)
        if self._use_fla(x):
            # only when a backward can follow: without it the saved states would just be extra forward work (qg)
            grad = torch.is_grad_enabled() and any(
                t.requires_grad for t in (q, k, v, g, beta, self.A_log, self.dt_bias))
            keep = {"disable_recompute": True} if self.keep_states and grad else {}
            o, _ = fla_chunk_kda()(
                q=q, k=k, v=v, g=g, beta=beta, A_log=self.A_log, dt_bias=self.dt_bias,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, **keep,
            )
        else:
            o = kda_reference(q, k, v, g, beta, self.A_log, self.dt_bias)
        gate = self.g_up(self.g_down(x) if gd is None else gd).view(B, T, H, K)
        o = gated_rms_norm(o, gate, self.o_norm_weight, self.o_eps)
        return self.o_proj(o.reshape(B, T, self.D))
