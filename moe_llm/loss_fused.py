"""ce_impl="fused" / "fused_bf16": the chunked LM head + cross-entropy (+ z-loss) of moe_llm/loss.py with all of
its elementwise work in one Triton kernel per chunk. The three GEMMs stay on cuBLAS.

    loss = mean_t( lse_t - logit_t[y_t] ) + z_coef * mean_t( lse_t^2 )        (over rows with y_t != ignore_index)

Per chunk of rows:
  logits = h_c @ W^T     bf16 operands, fp32 accumulation; "fused" keeps the fp32 result (torch.mm with
                         out_dtype=float32), "fused_bf16" rounds it to bf16 as the chunked path does
  _ce_rows_kernel        one program per row. Pass 1 reads the row once: per-lane running max and sum of exp
                         (online softmax, no barriers inside the loop) -> lse, and the target logit. Pass 2
                         (training only) re-reads the row, last column block first so it mostly comes from L2, and
                         writes g = dL/dlogits = (exp(l - lse) (1 + 2 z lse) - onehot(y)) * valid / n_valid in the
                         GEMM dtype (bf16), in place over the logits when they are bf16
  grad_h[c] = g @ W      cuBLAS with an fp32 output, rounded once to h's dtype (the chunked path asks cuBLAS for a bf16
                         output, where split-K partial sums over the 64K reduction may be bf16)
  grad_W   += g^T @ h_c  cuBLAS addmm with an fp32 output straight into the fp32 accumulator (out aliases it); the
                         first chunk writes it with mm (no zero fill)

Memory traffic of the elementwise part per [2048, 65536] chunk: ~0.75 GiB (fp32 logits read once from DRAM, g
written once) instead of ~7 GiB for the eager chain of the chunked path (upcast, logsumexp = amax + sub + exp + sum,
sub_, exp_, mul_, scatter, downcast), plus ~0.8 GiB less for grad_W (no bf16 partial, upcast and separate add).

Rounding differences against the chunked path (moe_llm/loss.py); each one either removes a rounding or only
reorders fp32 arithmetic:
  - "fused": the logits are not rounded to bf16 (the chunked path rounds every logit by up to 2^-9 relative before
    the softmax), so the loss, g, grad_h and grad_W are closer to the exact values. "fused_bf16": same logits.
  - grad_W: each chunk's g^T h is added into the fp32 accumulator without first rounding it to bf16.
  - grad_h: one rounding of the fp32 GEMM result (no bf16 split-K partial sums).
  - lse from the online rescaled sum (Triton's exp2-based exp / log) instead of torch.logsumexp: fp32 rounding
    differences (~1e-6 relative), far below the bf16 rounding of g that both paths apply.
  - The loss sums over all rows at once instead of per chunk (fp32 summation order).
Values do not depend on the run: no atomics, fixed tile sizes (set_config), so reruns are bit-identical.
No host syncs: n_valid stays on the GPU and enters through the per-row scale vector.
Gradients are produced in the forward and scaled by the incoming gradient in the backward, as in the chunked path;
under torch.no_grad (evaluation) only the logits GEMM and pass 1 run.
"""
from __future__ import annotations

import os
import warnings

import torch
import triton
import triton.language as tl
from triton import knobs

# Triton's CPU interpreter (TRITON_INTERPRET=1 before triton is imported): tests run the kernel on CPU tensors
_INTERPRET = bool(knobs.runtime.interpret)

# (BLOCK_V, num_warps) of the row kernel on the GPU. 32 warps (1024 threads) per program leave room for one program
# per SM on sm89 (1536 threads): ~128 rows in flight = 32 MiB of fp32 logits + 16 MiB of g, well inside the 72 MB L2,
# so pass 2 finds its row there. BLOCK_V / (32 * num_warps) = 8 columns per thread per block: 32 KiB of loads in
# flight per SM. scripts/bench_ce.py --sweep times the alternatives; MOE_CE_CFG="BLOCK_V,num_warps" overrides.
# The block size changes the fp32 summation order of lse (not the math): keep it fixed during a run.
DEFAULT_CFG = (8192, 32)
_INTERP_CFG = (128, 4)  # interpreter: small blocks, so tests cover several blocks and a masked tail
_cfg_override = None
# L2 eviction-priority hints on the row loads: evict_last in pass 1 (the row is read again), evict_first in pass 2.
# They change no value; MOE_CE_HINTS=0 / set_hints(False) drops them (scripts/bench_ce.py --sweep times both).
_HINTS = os.environ.get("MOE_CE_HINTS", "1") != "0"


def set_hints(enabled: bool) -> None:
    global _HINTS
    _HINTS = bool(enabled)


def set_config(block_v: int, num_warps: int) -> None:
    """Fixed (BLOCK_V, num_warps) for the row kernel (benchmarks, tests). reset_config() undoes."""
    global _cfg_override
    assert block_v & (block_v - 1) == 0, "BLOCK_V must be a power of two"
    _cfg_override = (int(block_v), int(num_warps))


def reset_config() -> None:
    global _cfg_override
    _cfg_override = None


def get_config() -> tuple[int, int]:
    if _cfg_override is not None:
        return _cfg_override
    if _INTERPRET:
        return _INTERP_CFG
    env = os.environ.get("MOE_CE_CFG", "")
    if env:
        b, w = (int(v) for v in env.split(","))
        return b, w
    return DEFAULT_CFG


# ---------------------------------------------------------------------------- kernel
@triton.jit
def _round(x, dt: tl.constexpr, INTERP: tl.constexpr):
    """fp32 x rounded to dt, to nearest even (the GPU's cvt.rn). The CPU interpreter truncates fp32 -> bf16, so
    there the bits are rounded first (finite values; tests only). Same helper as moe_triton._round."""
    if INTERP and dt == tl.bfloat16:
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return b.to(tl.float32, bitcast=True).to(dt)
    else:
        return x.to(dt)


@triton.jit
def _load_block(ptr, mask, POLICY: tl.constexpr):
    """Masked load of one block of a row. POLICY 1: evict_last, 2: evict_first (L1 + L2 cache hints), 0: none."""
    if POLICY == 1:
        return tl.load(ptr, mask=mask, other=0.0, eviction_policy="evict_last")
    elif POLICY == 2:
        return tl.load(ptr, mask=mask, other=0.0, eviction_policy="evict_first")
    else:
        return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def _ce_rows_kernel(x_ptr, g_ptr, tgt_ptr, scale_ptr, lse_ptr, xt_ptr, V, stride_x, stride_g, z2,
                    BLOCK_V: tl.constexpr, WRITE_G: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                    INTERP: tl.constexpr):
    """Row r of the logits x [rows, V] (fp32 or bf16): lse[r], xt[r] = x[r, tgt[r]] and, with WRITE_G,
    g[r, :] = (exp(x - lse) * (1 + z2 * lse) * scale[r]) - scale[r] at the target column, in g's dtype.
    g may alias x (same dtype and strides): every read of the row outside pass 2 (the target logit, pass 1) comes
    before pass 1's final reductions, whose barriers order it before any write of pass 2, and pass 2 reads each block
    (in the same thread layout) before writing it. P1 / P2: cache-hint policy of the pass 1 / pass 2 loads."""
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    offs = tl.arange(0, BLOCK_V)
    tgt = tl.load(tgt_ptr + row)
    xt = tl.load(x_row + tgt).to(tl.float32)  # before pass 1: see the docstring (g in place of x)
    # pass 1: per-lane running max m and sum s of exp(x - m) (online softmax). The update needs one exp:
    # exp(-|x - m|) is exp(x - m_new) or exp(m - m_new), whichever is not exp(0). Masked lanes keep m = -inf, s = 0.
    m = tl.full([BLOCK_V], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK_V], tl.float32)
    for start in range(0, V, BLOCK_V):
        cols = start + offs
        mask = cols < V
        x = _load_block(x_row + cols, mask, P1).to(tl.float32)
        e = tl.exp(-tl.abs(x - m))
        s = tl.where(mask, tl.where(x > m, s * e + 1.0, s + e), s)
        m = tl.where(mask, tl.maximum(m, x), m)
    mx = tl.max(m, 0)
    lse = mx + tl.log(tl.sum(s * tl.exp(m - mx), 0))
    tl.store(lse_ptr + row, lse)
    tl.store(xt_ptr + row, xt)
    if WRITE_G:
        scale = tl.load(scale_ptr + row)
        coef = (1.0 + z2 * lse) * scale
        g_row = g_ptr + row * stride_g
        n_blk = tl.cdiv(V, BLOCK_V)
        for i in range(0, n_blk):
            cols = (n_blk - 1 - i) * BLOCK_V + offs  # last block first: the most recently read, likeliest in L2
            mask = cols < V
            x = _load_block(x_row + cols, mask, P2).to(tl.float32)
            g = tl.exp(x - lse) * coef
            g = tl.where(cols == tgt, g - scale, g)
            tl.store(g_row + cols, _round(g, g_ptr.dtype.element_ty, INTERP), mask=mask)


def ce_rows(logits: torch.Tensor, tgt: torch.Tensor, scale: torch.Tensor, z_coef: float, g: torch.Tensor | None,
            lse: torch.Tensor, xt: torch.Tensor) -> None:
    """Launch the row kernel on logits [rows, V] (fp32 or bf16, unit column stride). tgt [rows] int in [0, V),
    scale [rows] fp32 (valid / n_valid), lse / xt [rows] fp32 outputs. g: [rows, V] output (may be `logits`
    itself when the dtypes match) or None for the loss only."""
    rows, V = logits.shape
    if rows == 0:
        return
    assert logits.stride(1) == 1 and (g is None or (g.shape == logits.shape and g.stride(1) == 1))
    block_v, warps = get_config()
    gg = logits if g is None else g
    _ce_rows_kernel[(rows,)](logits, gg, tgt, scale, lse, xt, V, logits.stride(0), gg.stride(0), 2.0 * float(z_coef),
                             BLOCK_V=block_v, WRITE_G=g is not None, P1=1 if _HINTS else 0, P2=2 if _HINTS else 0,
                             INTERP=_INTERPRET, num_warps=warps)


# ---------------------------------------------------------------------------- GEMM helpers
_BLAS_LT = False         # ModelConfig.ce_blas == "cublaslt": all GEMMs below through cuBLASLt
_BLAS_LT_LOGITS = False  # ce_blas == "cublaslt_logits": only the logits GEMM (bit-identical; the default)


BLAS_CHOICES = ("", "cublas", "cublaslt", "cublaslt_logits")


def set_blas(name: str) -> None:
    """ModelConfig.ce_blas: "" (default) = the default cuBLAS path; "cublaslt" = run this module's mm calls through
    cuBLASLt (torch.backends.cuda.preferred_blas_library, switched around each call and restored). On the 4090 the
    fp32-output logits GEMM [2048 x 1024] x [1024 x 65536] took 2.86 ms with the default cuBLAS kernel (an sm80
    1688 kernel) and 1.53 ms with cuBLASLt, bit-identical (runs/switch_check/switch9e.log; tests/test_ce_blas.py
    checks loss and both gradients bitwise). The in-place grad_W addmm is not affected (2-D self: plain cuBLAS).
    Switch 10 (tests/test_ce_blas.py): with "cublaslt" the loss is bit-identical but grad_h is not (the K = 65536
    reduction is summed in another order); "cublaslt_logits" routes only the logits GEMM, so loss, grad_h and grad_W
    stay bit-identical, and keeps most of the gain (~0.35 s per optimizer step)."""
    global _BLAS_LT, _BLAS_LT_LOGITS
    if name not in BLAS_CHOICES:
        raise ValueError(f"ce_blas={name!r}; expected one of {BLAS_CHOICES}")
    _BLAS_LT = name == "cublaslt"
    _BLAS_LT_LOGITS = name in ("cublaslt", "cublaslt_logits")


def _cuda_mm(a, b, out_dtype, out):
    return torch.mm(a, b, out_dtype=out_dtype) if out is None else torch.mm(a, b, out_dtype=out_dtype, out=out)


def _mm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype, out: torch.Tensor | None = None,
        logits: bool = False) -> torch.Tensor:
    """a @ b with fp32 accumulation, written in out_dtype (rounded once). CUDA: cuBLAS through torch.mm(out_dtype=)
    (bf16/fp16 operands -> fp32 output), or cuBLASLt (set_blas). CPU (tests): products of bf16/fp16 values are exact
    in fp32, so an fp32 GEMM of the upcast operands computes the same sum."""
    if a.dtype != out_dtype:
        if a.is_cuda:
            if not (_BLAS_LT_LOGITS if logits else _BLAS_LT):
                return _cuda_mm(a, b, out_dtype, out)
            prev = torch.backends.cuda.preferred_blas_library()
            torch.backends.cuda.preferred_blas_library("cublaslt")
            try:
                return _cuda_mm(a, b, out_dtype, out)
            finally:
                torch.backends.cuda.preferred_blas_library(prev)
        a, b = a.to(out_dtype), b.to(out_dtype)
    return torch.mm(a, b) if out is None else torch.mm(a, b, out=out)


_ALIAS_OK: dict = {}


def addmm_alias_ok(device: torch.device) -> bool:
    """Whether torch.addmm(c, a, b, out_dtype=float32, out=c) accumulates into c (out aliasing self, as
    c.addmm_(a, b) does for equal dtypes). Checked once per device on a small problem against the non-aliased form
    (one host sync, at the first fused call); a failure falls back to mm + add_ (same values, more memory traffic)."""
    key = (device.type, device.index)
    ok = _ALIAS_OK.get(key)
    if ok is None:
        gen = torch.Generator().manual_seed(0)
        a = torch.randn(96, 64, generator=gen).to(device, torch.bfloat16)
        b = torch.randn(64, 80, generator=gen).to(device, torch.bfloat16)
        c = torch.randn(96, 80, generator=gen).to(device)
        want = c + torch.mm(a, b, out_dtype=torch.float32)
        ptr = c.data_ptr()
        try:
            r = torch.addmm(c, a, b, out_dtype=torch.float32, out=c)
            ok = bool(r.data_ptr() == ptr and c.data_ptr() == ptr and torch.allclose(c, want, rtol=1e-5, atol=1e-5))
        except (RuntimeError, NotImplementedError, TypeError):
            ok = False
        if not ok:
            warnings.warn("torch.addmm(c, a, b, out_dtype=torch.float32, out=c) does not accumulate into c here; "
                          "ce_impl='fused' uses mm + add_ for the LM-head weight gradient instead (same values, slower)")
        _ALIAS_OK[key] = ok
    return ok


def _addmm_into(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """c += a @ b, c fp32, a / b in the GEMM dtype: the fp32 accumulator of the GEMM is added into c without being
    rounded to the GEMM dtype first."""
    if a.dtype == c.dtype:
        c.addmm_(a, b)
    elif not c.is_cuda:
        c.addmm_(a.to(c.dtype), b.to(c.dtype))
    elif addmm_alias_ok(c.device):
        torch.addmm(c, a, b, out_dtype=c.dtype, out=c)
    else:
        c.add_(torch.mm(a, b, out_dtype=c.dtype))


# ---------------------------------------------------------------------------- autograd
class FusedLinearCrossEntropy(torch.autograd.Function):
    """See the module docstring. need_h / need_w: whether grad_h / grad_W are needed (decided by the caller from
    torch.is_grad_enabled() and requires_grad: ctx.needs_input_grad ignores no_grad mode)."""

    @staticmethod
    def forward(ctx, h, W, targets, z_coef: float, chunk: int, ignore_index: int, fp32_logits: bool,
                need_h: bool, need_w: bool):
        with torch.autocast(device_type=h.device.type, enabled=False):
            N = h.shape[0]
            V = W.shape[0]
            acc = torch.float32
            # GEMM dtype exactly as in the chunked path
            cdt = h.dtype if h.dtype in (torch.bfloat16, torch.float16) else acc
            if h.is_cuda and cdt == torch.float32:
                cdt = torch.bfloat16
            ldt = acc if fp32_logits else cdt
            need_g = need_h or need_w
            Wc = W.to(cdt)
            valid = targets != ignore_index
            n_valid = valid.sum().clamp_min(1).to(acc)  # stays on the device
            vf = valid.to(acc)
            scale = vf / n_valid
            tgt = targets.clamp(0, V - 1)  # ignored rows: any column (their scale is 0)
            lse = torch.empty(N, dtype=acc, device=h.device)
            xt = torch.empty(N, dtype=acc, device=h.device)
            rows = min(chunk, N)
            lbuf = torch.empty(rows, V, dtype=ldt, device=h.device) if N else None
            gbuf = None
            if need_g and N:
                gbuf = lbuf if ldt == cdt else torch.empty(rows, V, dtype=cdt, device=h.device)
            grad_h = torch.empty_like(h) if need_h else None
            grad_W = (torch.empty if N else torch.zeros)(W.shape, dtype=acc, device=W.device) if need_w else None
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                hc = h[s:e].to(cdt)
                logits = _mm(hc, Wc.t(), ldt, out=lbuf[: e - s], logits=True)
                g = gbuf[: e - s] if need_g else None
                ce_rows(logits, tgt[s:e], scale[s:e], z_coef, g, lse[s:e], xt[s:e])
                if need_h:
                    if h.dtype == acc:
                        _mm(g, Wc, acc, out=grad_h[s:e])
                    else:  # fp32 result rounded once to h's dtype: no bf16 split-K partial sums over the V = 64K
                        grad_h[s:e] = _mm(g, Wc, acc)  # reduction (allow_bf16_reduced_precision_reduction is on)
                if need_w:
                    if s == 0:
                        _mm(g.t(), hc, acc, out=grad_W)
                    else:
                        _addmm_into(grad_W, g.t(), hc)
            ce = ((lse - xt) * vf).sum() / n_valid
            loss = ce + z_coef * ((lse.pow(2) * vf).sum() / n_valid)
        ctx.save_for_backward(grad_h, grad_W)
        ctx.w_dtype = W.dtype
        ce_out = ce.detach().clone()
        ctx.mark_non_differentiable(ce_out)
        return loss, ce_out

    @staticmethod
    def backward(ctx, grad_loss, grad_ce):
        grad_h, grad_W = ctx.saved_tensors
        gh = grad_h * grad_loss.to(grad_h.dtype) if grad_h is not None else None
        gw = (grad_W * grad_loss).to(ctx.w_dtype) if grad_W is not None else None
        return gh, gw, None, None, None, None, None, None, None


def fused_linear_cross_entropy(h, W, targets, z_coef: float, chunk: int, ignore_index: int, fp32_logits: bool):
    grad = torch.is_grad_enabled()
    return FusedLinearCrossEntropy.apply(h, W, targets, float(z_coef), int(chunk), int(ignore_index),
                                         bool(fp32_logits), grad and h.requires_grad, grad and W.requires_grad)
