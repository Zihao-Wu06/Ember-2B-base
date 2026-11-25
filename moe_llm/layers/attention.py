"""Full causal attention (GQA) for the 1-in-4 global layers.

NoPE by default (Kimi Linear style), QK-norm for stability, and an elementwise sigmoid gate
on the attention output (Gated Attention, as in Qwen3-Next).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .common import CastLinear, RMSNorm, Rotary, fused_linear, sigmoid_mul


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        d, std = cfg.d_model, cfg.init_std
        self.Hq, self.Hkv, self.hd = cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
        assert self.Hq % self.Hkv == 0
        # cfg.fuse_proj: q/k/v/gate (all read x) as one GEMM (same parameters; layers/common.py fused_linear)
        self.fuse_proj = bool(getattr(cfg, "fuse_proj", False))
        self.q_proj = CastLinear(d, self.Hq * self.hd, bias=False)
        self.k_proj = CastLinear(d, self.Hkv * self.hd, bias=False)
        self.v_proj = CastLinear(d, self.Hkv * self.hd, bias=False)
        self.o_proj = CastLinear(self.Hq * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.attn_qk_norm else None
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.attn_qk_norm else None
        self.gate_proj = CastLinear(d, self.Hq * self.hd, bias=False) if cfg.attn_out_gate else None
        self.rope = Rotary(self.hd, cfg.rope_theta) if cfg.attn_rope else None
        self.logn = int(getattr(cfg, "attn_logn", 0))  # 0 = off; see logn_scale
        for m in (self.q_proj, self.k_proj, self.v_proj, self.gate_proj):
            if m is not None:
                nn.init.normal_(m.weight, std=std)
        nn.init.normal_(self.o_proj.weight, std=std / math.sqrt(2 * cfg.n_layers))

    def proj_weights(self) -> list:
        """The weights of the projections that read x, in the order of the fused GEMM's output blocks (fuse_proj)."""
        return [m.weight for m in (self.q_proj, self.k_proj, self.v_proj, self.gate_proj) if m is not None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        gate = None
        if self.fuse_proj:
            # one GEMM [.., (2 Hq + 2 Hkv) hd] = [q | k | v | gate] (main_2b: N = 2560); the blocks are views, and
            # split's backward hands their gradients back as one tensor (one cat), so dx is one GEMM too
            qd, kvd = self.Hq * self.hd, self.Hkv * self.hd
            sizes = (qd, kvd, kvd) + ((qd,) if self.gate_proj is not None else ())
            q, k, v, *rest = fused_linear(x, self.proj_weights()).split(sizes, dim=-1)
            gate = rest[0] if rest else None
            q, k, v = q.view(B, T, self.Hq, self.hd), k.view(B, T, self.Hkv, self.hd), v.view(B, T, self.Hkv, self.hd)
        else:
            q = self.q_proj(x).view(B, T, self.Hq, self.hd)
            k = self.k_proj(x).view(B, T, self.Hkv, self.hd)
            v = self.v_proj(x).view(B, T, self.Hkv, self.hd)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, hd]
        if self.logn and T > self.logn:
            q = q * logn_scale(T, T, self.logn, q.device, q.dtype).view(1, 1, T, 1)
        if self.Hq != self.Hkv:
            rep = self.Hq // self.Hkv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = causal_sdpa(q, k, v)
        o = o.transpose(1, 2)  # [B, T, Hq, hd]
        if self.gate_proj is not None:
            o = sigmoid_mul(o, (self.gate_proj(x) if gate is None else gate).view(B, T, self.Hq, self.hd))
        return self.o_proj(o.reshape(B, T, self.Hq * self.hd))


# The causal SDPA backend. MOE_SDPA=auto (default): cuDNN's fused attention first on compute capability >= 10
# (Blackwell; on a B300, causal forward + backward with 8 heads of 128: 204 ms vs flash's 399 ms at 1 x 128K tokens,
# 8.2 vs 27.0 ms at 32 x 4K, outputs equal up to bf16 rounding, 2026-11-22), PyTorch's own order elsewhere (the 4090
# keeps flash). MOE_SDPA=cudnn forces cuDNN first, MOE_SDPA=flash (or anything else) keeps PyTorch's order.
_SDPA_MODE = os.environ.get("MOE_SDPA", "auto")
_CUDNN_FIRST = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
_CUDNN_ON: dict = {}  # device index -> prefer cuDNN (decided on the first call: no CUDA query at import)


def causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    on = False
    if q.is_cuda and _SDPA_MODE in ("auto", "cudnn"):
        i = q.device.index if q.device.index is not None else torch.cuda.current_device()
        on = _CUDNN_ON.get(i)
        if on is None:
            on = _CUDNN_ON[i] = _SDPA_MODE == "cudnn" or torch.cuda.get_device_capability(i)[0] >= 10
    if on:
        with sdpa_kernel(_CUDNN_FIRST, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


_LOGN: dict = {}


def logn_scale(tq: int, tk: int, base: int, device, dtype) -> torch.Tensor:
    """[tq] query scales for the last tq of tk causal positions: max(1, ln n / ln base), n = keys the query sees
    (1..tk). Multiplying q by it multiplies the attention logits (QK-norm keeps |q| fixed otherwise)."""
    key = (tq, tk, base, device, dtype)
    s = _LOGN.get(key)
    if s is None:
        n = torch.arange(tk - tq + 1, tk + 1, dtype=torch.float64)
        s = torch.clamp(torch.log(n) / math.log(base), min=1.0).to(device=device, dtype=dtype)
        if len(_LOGN) > 64:
            _LOGN.clear()
        _LOGN[key] = s
    return s


_CED_MASKS: dict = {}


def ced_mask(T: int, window: int, device) -> torch.Tensor:
    """[T, 2T] bool (True = attend): keys 0..T-1 are the global entries (causal), keys T..2T-1 the
    layer's own entries, visible only for the last `window` positions (t - window < s <= t)."""
    key = (T, window, str(device))
    m = _CED_MASKS.get(key)
    if m is None:
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        m = _CED_MASKS[key] = torch.cat([j <= i, (j <= i) & (j > i - window)], dim=1)
    return m


class CEDAttention(nn.Module):
    """Decoder-layer attention of a causal encoder-decoder (DeepSeek-V4.1-Flash CED), dense variant.

    Each query sees, in one softmax, (a) global keys/values projected with this layer's own weights
    from the encoder output `enc` (the hidden state entering the first decoder layer) for every
    position <= t, and (b) local keys/values from this layer's input for the last `window` positions.
    CSA2's sparse top-k selection and FP4 KV are not used (dense global attention at 4K).
    """

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        d, std = cfg.d_model, cfg.init_std
        self.Hq, self.Hkv, self.hd = cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
        assert self.Hq % self.Hkv == 0
        self.window = cfg.ced_window
        self.q_proj = CastLinear(d, self.Hq * self.hd, bias=False)
        self.kg_proj = CastLinear(d, self.Hkv * self.hd, bias=False)   # global, from the encoder output
        self.vg_proj = CastLinear(d, self.Hkv * self.hd, bias=False)
        self.kl_proj = CastLinear(d, self.Hkv * self.hd, bias=False)   # local, from this layer
        self.vl_proj = CastLinear(d, self.Hkv * self.hd, bias=False)
        self.o_proj = CastLinear(self.Hq * self.hd, d, bias=False)
        qk = cfg.attn_qk_norm
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if qk else None
        self.kg_norm = RMSNorm(self.hd, cfg.norm_eps) if qk else None
        self.kl_norm = RMSNorm(self.hd, cfg.norm_eps) if qk else None
        self.gate_proj = CastLinear(d, self.Hq * self.hd, bias=False) if cfg.attn_out_gate else None
        self.rope = Rotary(self.hd, cfg.rope_theta) if cfg.attn_rope else None
        for m in (self.q_proj, self.kg_proj, self.vg_proj, self.kl_proj, self.vl_proj, self.gate_proj):
            if m is not None:
                nn.init.normal_(m.weight, std=std)
        nn.init.normal_(self.o_proj.weight, std=std / math.sqrt(2 * cfg.n_layers))

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.Hq, self.hd)
        kg = self.kg_proj(enc).view(B, T, self.Hkv, self.hd)
        vg = self.vg_proj(enc).view(B, T, self.Hkv, self.hd)
        kl = self.kl_proj(x).view(B, T, self.Hkv, self.hd)
        vl = self.vl_proj(x).view(B, T, self.Hkv, self.hd)
        if self.q_norm is not None:
            q, kg, kl = self.q_norm(q), self.kg_norm(kg), self.kl_norm(kl)
        if self.rope is not None:
            q_r, kg = self.rope(q, kg)
            _, kl = self.rope(q, kl)
            q = q_r
        k = torch.cat([kg, kl], dim=1).transpose(1, 2)  # [B, Hkv, 2T, hd]
        v = torch.cat([vg, vl], dim=1).transpose(1, 2)
        q = q.transpose(1, 2)
        if self.Hq != self.Hkv:
            rep = self.Hq // self.Hkv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=ced_mask(T, self.window, x.device))
        o = o.transpose(1, 2)  # [B, T, Hq, hd]
        if self.gate_proj is not None:
            o = sigmoid_mul(o, self.gate_proj(x).view(B, T, self.Hq, self.hd))
        return self.o_proj(o.reshape(B, T, self.Hq * self.hd))
