"""Attention Residuals (Kimi, arXiv 2603.15031), Block variant with Kimi-K3 semantics.

One op mixes a list of source tensors v_i (raw, unnormalised) per token:
    alpha_i = softmax_i( (g * w) . v_i / rms(v_i) ),   h = sum_i alpha_i v_i
w (pseudo-query) is ZERO-initialised, so every op starts as a uniform average; g (key-RMSNorm
gain) starts at 1. Computed in fp32. Sources are processed one by one instead of being
stacked, so the op never materialises a [S, B, T, d] fp32 tensor.

Speed: run eagerly, every source goes through about a dozen full-size fp32 passes, ~13x the memory
traffic of one fused read for the logits plus one for the weighted sum; that made the 12-layer
ablation model ~20% slower. On CUDA the op therefore runs through torch.compile, one compiled copy
per number of sources: a single shared copy would exceed dynamo's recompile limit (8) with up to
9 source counts x train/eval. MOE_ATTNRES_COMPILE=0 or set_attnres_compile(False) = eager.
"""
from __future__ import annotations

import os
import types
import warnings

import torch
import torch.nn as nn

_COMPILE = os.environ.get("MOE_ATTNRES_COMPILE", "1") == "1"
_BY_ARITY: dict = {}   # number of sources -> compiled function
_FAILED: set = set()   # source counts whose compile failed (they run eagerly)


def set_attnres_compile(enabled: bool) -> None:
    global _COMPILE
    _COMPILE = bool(enabled)


def attnres_compiled() -> bool:
    """Whether CUDA calls go through torch.compile."""
    return _COMPILE


class DepthAttn(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.zeros(d))
        self.g = nn.Parameter(torch.ones(d))
        self.w._attnres = True  # optimizer: AdamW, no weight decay (never Muon)
        self.g._attnres = True

    def forward(self, srcs: list[torch.Tensor]) -> torch.Tensor:
        if len(srcs) == 1:
            return srcs[0]
        return depth_attn(self.w, self.g, self.eps, *srcs)


def depth_attn(w, g, eps: float, *srcs):
    n = len(srcs)
    if _COMPILE and srcs[0].is_cuda and n not in _FAILED:
        try:
            return compiled_depth_attn(n)(w, g, eps, *srcs)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # a compiler problem must not stop a long run: fall back to eager
            _FAILED.add(n)
            warnings.warn(f"AttnRes: torch.compile failed for {n} sources, running eagerly: {e!r}")
    return _depth_attn(w, g, eps, *srcs)


def compiled_depth_attn(n: int):
    """torch.compile'd copy of _depth_attn for n sources, with its own code object (dynamo keeps one
    cache per code object, so every source count gets its own recompile budget)."""
    fn = _BY_ARITY.get(n)
    if fn is None:
        name = f"_depth_attn_{n}"
        code = _depth_attn.__code__
        code = code.replace(co_name=name, **({"co_qualname": name} if hasattr(code, "co_qualname") else {}))
        clone = types.FunctionType(code, _depth_attn.__globals__, name)
        fn = _BY_ARITY[n] = torch.compile(clone, dynamic=False, fullgraph=True)
    return fn


def _depth_attn(w, g, eps: float, *srcs):
    q = (g * w).float()
    logits = []
    for v in srcs:
        vf = v.float()
        r = torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)
        logits.append((vf * q).sum(-1, keepdim=True) * r)
    a = torch.softmax(torch.cat(logits, -1), dim=-1)  # [B, T, S]
    out = a[..., 0:1] * srcs[0].float()
    for i in range(1, len(srcs)):
        out = out + a[..., i:i + 1] * srcs[i].float()
    return out.to(srcs[0].dtype)
