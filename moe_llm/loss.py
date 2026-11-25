"""Memory-efficient LM head + cross-entropy (+ z-loss).

Logits for 8K tokens x 64K vocab would be 2 GiB in fp32 (plus the same for their gradient).
This computes the loss chunk by chunk and produces dL/dh and dL/dW inside the forward pass, so
no [N, V] tensor is ever kept: peak extra memory is one [chunk, V] fp32 buffer.

    loss = mean_t( lse_t - logit_t[y_t] ) + z_coef * mean_t( lse_t^2 )

linear_cross_entropy(impl="fused" | "fused_bf16") runs the same computation with the elementwise part in one Triton
kernel per chunk (moe_llm/loss_fused.py); the default "chunked" is _LinearCrossEntropy below, unchanged.
"""
from __future__ import annotations

import torch


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, targets, z_coef: float, chunk: int, ignore_index: int):
        with torch.autocast(device_type=h.device.type, enabled=False):
            N = h.shape[0]
            acc = torch.float64 if h.dtype == torch.float64 else torch.float32
            cdt = h.dtype if h.dtype in (torch.bfloat16, torch.float16) else acc
            if h.is_cuda and cdt == torch.float32:
                # the residual stream is fp32; run the three [N,d]x[d,V] GEMMs on bf16 tensor cores
                # (fp32 accumulation, softmax/lse in fp32). fp32 SIMT GEMMs were ~5% of step time.
                cdt = torch.bfloat16
            Wc = W.to(cdt)
            valid = targets != ignore_index
            n_valid = valid.sum().clamp_min(1).to(acc)
            grad_h = torch.empty_like(h)
            grad_W = torch.zeros(W.shape, dtype=acc, device=W.device)
            ce_sum = torch.zeros((), dtype=acc, device=h.device)
            z_sum = torch.zeros((), dtype=acc, device=h.device)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                hc = h[s:e].to(cdt)
                tgt = targets[s:e].clamp_min(0)
                vf = valid[s:e].to(acc)
                logits = (hc @ Wc.t()).to(acc)
                lse = torch.logsumexp(logits, dim=-1)
                tl = logits.gather(1, tgt[:, None]).squeeze(1)
                ce_sum += ((lse - tl) * vf).sum()
                z_sum += (lse.pow(2) * vf).sum()
                # dL/dlogits = (softmax * (1 + 2 z lse) - onehot(y)) * valid / n_valid   (in place)
                scale = vf / n_valid
                g = logits.sub_(lse[:, None]).exp_()
                g.mul_(((1.0 + 2.0 * z_coef * lse) * scale)[:, None])
                g[torch.arange(e - s, device=g.device), tgt] -= scale
                gc = g.to(cdt)
                grad_h[s:e] = (gc @ Wc).to(h.dtype)
                grad_W += (gc.t() @ hc).to(acc)
            ce = ce_sum / n_valid
            loss = ce + z_coef * (z_sum / n_valid)
        ctx.save_for_backward(grad_h, grad_W)
        ctx.w_dtype = W.dtype
        ce_out = ce.detach().clone()
        ctx.mark_non_differentiable(ce_out)
        return loss, ce_out

    @staticmethod
    def backward(ctx, grad_loss, grad_ce):
        grad_h, grad_W = ctx.saved_tensors
        return (grad_h * grad_loss.to(grad_h.dtype), (grad_W * grad_loss).to(ctx.w_dtype),
                None, None, None, None)


CE_IMPLS = ("chunked", "fused", "fused_bf16")
_FUSED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def linear_cross_entropy(h: torch.Tensor, W: torch.Tensor, targets: torch.Tensor, z_coef: float = 0.0,
                         chunk: int = 2048, ignore_index: int = -100, impl: str = "chunked"):
    """h [N, d], W [V, d], targets [N]. Returns (loss incl. z-loss, plain CE for logging).

    impl (ModelConfig.ce_impl): "chunked" = the code above (eager elementwise ops; the default and reference);
    "fused" = moe_llm/loss_fused.py: same chunking and GEMMs, the elementwise work in one Triton kernel per chunk,
    logits kept in fp32 (more accurate than "chunked"); "fused_bf16" = the same with the logits rounded to bf16 as
    "chunked" rounds them. The fused paths need CUDA tensors (or Triton's CPU interpreter) in fp32 / bf16 / fp16;
    anything else (CPU, fp64) runs "chunked"."""
    if impl != "chunked":
        if impl not in CE_IMPLS:
            raise ValueError(f"unknown ce_impl {impl!r}; expected one of {CE_IMPLS}")
        if h.dtype in _FUSED_DTYPES and W.dtype in _FUSED_DTYPES:
            from . import loss_fused  # imports triton: only when a fused path is requested
            if h.is_cuda or loss_fused._INTERPRET:
                return loss_fused.fused_linear_cross_entropy(h, W, targets, z_coef, chunk, ignore_index,
                                                             fp32_logits=impl == "fused")
    return _LinearCrossEntropy.apply(h, W, targets, float(z_coef), int(chunk), int(ignore_index))
