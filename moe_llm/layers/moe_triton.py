"""Routed experts without padding (moe_impl="triton"): Triton grouped GEMMs over the expert-sorted
token-expert assignments.

    y[t] = sum_k w[t,k] * W_down[e] (silu(x[t] @ W_gate[e]) * (x[t] @ W_up[e])),   e = idx[t,k]

The N*K assignments are stably sorted by expert; row r of every [N*K, *] buffer below is sorted
assignment r, and the expert offsets stay on the GPU. Each GEMM program derives its (expert, row block)
from the offsets, so tiles never straddle two experts, nothing is padded beyond the last partial block of
an expert, and there is no host sync.

Forward (3 kernels):
  gate_up   rows of x read through the sorted token index (no gathered copy), the gate and up GEMMs on one
            A tile, epilogue h = silu(g) * u. g and u are kept for backward unless recompute=True.
  down      h @ W_down[e], each row written to its assignment slot t*K + k.
  combine   y[t] = sum_k w[t,k] * out[t*K + k] in fp32 with k in a fixed order (no index_add_, no atomics).
Backward (5 kernels):
  down_bwd  a = dy[t] @ W_down[e]^T; epilogue: dh = w a, the backward of silu(g) * u -> (dg, du), w h (for
            dW_down) and per-tile partial sums of <a, h> = <dy[t], expert output> (for dw).
  wgrad x2  dW_down[e] = (w h)_e^T dy_e and dW_gate[e], dW_up[e] = x_e^T dg_e, x_e^T du_e: fp32 sums over the
            expert's rows, written in the weight dtype; the experts with the most rows are scheduled first.
            With fused gradient accumulation (sinks: OptimizerBundle(fused_grad_acc=True)) each output tile is
            instead added to the parameter's fp32 gradient buffer (read, add, write by the one program that owns
            the tile: no atomics) and no weight gradient is returned.
  dx        dg @ W_gate[e]^T + du @ W_up[e]^T into the assignment slots, then the fixed-order k-sum.

Everything is deterministic (no atomics), so torch.utils.checkpoint reproduces the saved tensors bit for
bit. recompute=True does the same more cheaply: only x, w and the sort are kept, and backward reruns just
the gate/up GEMM (not the down GEMM and the combine as a checkpoint would).

Rounding points follow the bmm path: every GEMM accumulates in fp32 and rounds its output to the weight
dtype, silu(g) and h are rounded where eager rounds them, and so is every step of the SiLU-gate backward.
Where the kernels keep more precision than the bmm path: the k-sums of y and dx are fp32 (the bmm path adds
bf16 values with index_add_), dh = w (dy W_down^T) is rounded once (the bmm path rounds w dy first), and dw
is <dy W_down^T, h>, the same value as <dy, h W_down> without rounding the expert output first. dW_down sums
round(w h)^T dy where the bmm path sums h^T round(w dy): one rounding of a product with w either way.

Fused fan-in (ModelConfig.moe_shared_fused, layers/moe.py): the MoE input xf feeds the router, the routed experts and the
shared expert's gate and up projections, and the layer output is routed + shared. MoEFan is the fan-out node: it hands
each reader its own view of xf (the routed experts get an [N, K, D] view, stride 0 over k), so each reader's input
gradient arrives at its backward separately, and in modes "router" / "router_lean" it also computes the router logits
with the router's own F.linear call. RoutedExpertsFan returns the per-assignment input gradients instead of their
k-sum, and its forward combine adds the shared expert's output. MoEFan's backward writes the whole input gradient with
one combine kernel: ROUND_FIRST (mode "exact") rounds the k-sum as the plain combine does and then applies the old
path's autograd adds in their old order, each an fp32 add of two rounded values rounded again, which is what an add of
two bf16 tensors computes (the compiler may emit it as a correctly rounded bf16 add, the same value); without it every
part, and with ROUTER the router's part g_logits @ W_r as fp32 products, goes into one fp32 sum, rounded once.
"""
from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton import knobs

# Triton's CPU interpreter (TRITON_INTERPRET=1 before triton is imported): tests run the kernels on CPU tensors
_INTERPRET = bool(knobs.runtime.interpret)
# Weight-gradient GEMMs: read x and dy rows through the token index (default), or from expert-sorted copies made
# with index_select first (+2 x [N*K, d] of memory traffic). The in-loop index load limits Triton's software
# pipelining of the gathered version to one stage ahead; the copies allow num_stages. Same values either way.
_WGRAD_COPY = os.environ.get("MOE_TRITON_WGRAD_COPY", "0") == "1"


# ---------------------------------------------------------------------------- autotuning
def _cfg(bm, bn, bk, warps, stages):
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=warps, num_stages=stages)


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for sm89 (99 KB of shared memory per block). _pruner drops
# the ones a kernel cannot fit. "dual": two accumulators per tile (gate/up, dW_gate/dW_up); there BLOCK_N is
# the width of each of the two outputs.
_SINGLE = [(128, 128, 32, 4, 4), (128, 128, 32, 8, 4), (128, 128, 64, 8, 3), (128, 128, 64, 4, 3),
           (128, 128, 32, 8, 3), (128, 128, 32, 4, 3), (128, 64, 64, 4, 4), (128, 64, 32, 4, 4),
           (128, 64, 64, 8, 3), (64, 128, 64, 4, 4), (64, 128, 32, 4, 4), (64, 128, 64, 8, 3),
           (64, 64, 64, 4, 4), (64, 64, 32, 4, 3)]
_DUAL = [(128, 64, 32, 8, 4), (128, 64, 64, 8, 3), (128, 64, 32, 4, 4), (128, 128, 32, 8, 3),
         (128, 128, 32, 8, 4), (64, 128, 32, 4, 4), (64, 128, 32, 8, 4), (64, 64, 64, 4, 4),
         (64, 64, 32, 4, 4), (64, 64, 32, 4, 3)]
# down_bwd: its epilogue keeps ~5 fp32 tiles alive (a, g, u, silu, h): at most 32 elements per thread per tile
_EPI = [(128, 64, 32, 8, 4), (128, 64, 64, 8, 3), (128, 64, 32, 8, 3), (64, 128, 32, 8, 4), (64, 128, 64, 8, 3),
        (64, 64, 64, 4, 4), (64, 64, 32, 4, 4), (64, 64, 32, 4, 3)]
_INTERP_BLOCKS = (16, 16, 16)  # interpreter: one small config (no benchmarking on CPU)


def _configs(cands):
    if _INTERPRET:
        return [_cfg(*_INTERP_BLOCKS, 4, 1)]
    if os.environ.get("MOE_TRITON_TUNE", "") == "full":  # every combination of the ranges above
        cands = [(bm, bn, bk, w, s) for bm in (64, 128) for bn in (64, 128) for bk in (32, 64)
                 for w in (4, 8) for s in (3, 4)]
    return [_cfg(*c) for c in cands]


_SMEM = None


def _smem_limit() -> int:
    global _SMEM
    if _SMEM is None:
        try:
            _SMEM = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        except Exception:
            _SMEM = 101376  # sm89
    return _SMEM


def _pruner(operands):
    """operands(launch kwargs) -> (A tiles, B tiles per pipeline stage, fp32 accumulator tiles). Keeps the configs
    whose pipeline buffers (Triton allocates num_stages - 1 copies of the operand tiles) fit in shared memory and
    whose accumulators need at most 128 registers per thread."""
    def prune(configs, named_args, **kwargs):
        n_a, n_b, n_acc = operands(kwargs)
        esz = next(v for v in named_args.values() if isinstance(v, torch.Tensor)).element_size()

        def smem(c):
            tiles = (n_a * c.kwargs["BLOCK_M"] + n_b * c.kwargs["BLOCK_N"]) * c.kwargs["BLOCK_K"] * esz
            return max(1, c.num_stages - 1) * tiles

        keep = [c for c in configs if smem(c) <= _smem_limit()
                and n_acc * c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"] <= 128 * 32 * c.num_warps]
        return keep or [min(configs, key=smem)]
    return {"early_config_prune": prune}


def _even(dim_k, dim_n):
    return {"EVEN_K": lambda a: a[dim_k] % a["BLOCK_K"] == 0, "EVEN_N": lambda a: a[dim_n] % a["BLOCK_N"] == 0}


def _do_bench(fn, quantiles=(0.5, 0.2, 0.8)):
    """Autotuner timing: quantiles of 10 timed runs after 3 warm-up runs (the first compiles). Triton's default
    allocates a 256 MB buffer to flush L2 before every run; the backward kernels tune at the first backward,
    close to peak training memory. Results are cached on disk (cache_results), so this runs once per shape."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001  a config that fails to compile is skipped, not fatal
        if type(e).__name__ not in ("OutOfResources", "PTXASError", "CompileTimeAssertionFailure"):
            warnings.warn(f"moe_triton autotune: skipping a config: {type(e).__name__}: {e}")
        return [float("inf")] * len(quantiles)
    for _ in range(2):
        fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    t = torch.tensor([a.elapsed_time(b) for a, b in ev])
    return [torch.quantile(t, q).item() for q in quantiles]


def _autotune(cands, key, operands, **kw):
    return triton.autotune(configs=_configs(cands), key=key, prune_configs_by=_pruner(operands), do_bench=_do_bench,
                           cache_results=True, **kw)


# ---------------------------------------------------------------------------- kernels
@triton.jit
def _mma(a, b, acc, INTERP: tl.constexpr):
    """acc + a @ b, fp32 accumulation. The CPU interpreter multiplies bf16 bit patterns as integers, so it gets
    fp32 operands (exact for bf16 values); fp32 operands on the GPU use full fp32 products (no tf32)."""
    if INTERP:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
    elif a.dtype == tl.float32:
        return tl.dot(a, b, acc, input_precision="ieee")
    else:
        return tl.dot(a, b, acc)


@triton.jit
def _round(x, dt: tl.constexpr, INTERP: tl.constexpr):
    """fp32 x rounded to dt, to nearest even (the GPU's cvt.rn). The CPU interpreter truncates fp32 -> bf16, so
    there the bits are rounded first (finite values; tests only)."""
    if INTERP and dt == tl.bfloat16:
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return b.to(tl.float32, bitcast=True).to(dt)
    else:
        return x.to(dt)


@triton.jit
def _silu(g):
    """F.silu in fp32, the same formula as ATen: g / (1 + exp(-g))."""
    return g / (1.0 + tl.exp(-g))


@triton.jit
def _row_block(pid_m, offs_ptr, E: tl.constexpr, E_P2: tl.constexpr, BLOCK_M: tl.constexpr):
    """Row block pid_m of the expert-sorted rows -> (expert, first row, end of the expert's rows). Expert e owns
    ceil(count_e / BLOCK_M) consecutive blocks. expert >= E: past the last block (the grid is the upper bound
    cdiv(M, BLOCK_M) + E)."""
    ar = tl.arange(0, E_P2)
    lo = tl.load(offs_ptr + ar, mask=ar < E, other=0)
    hi = tl.load(offs_ptr + ar + 1, mask=ar < E, other=0)
    nblk = (hi - lo + BLOCK_M - 1) // BLOCK_M
    cum = tl.cumsum(nblk, 0)
    e = tl.sum((cum <= pid_m).to(tl.int32), 0)
    before = tl.sum(tl.where(ar < e, nblk, 0), 0)
    e_c = tl.minimum(e, E - 1)
    start = tl.load(offs_ptr + e_c)
    end = tl.load(offs_ptr + e_c + 1)
    return e, start + (pid_m - before) * BLOCK_M, end


@_autotune(_DUAL, ["M_BUCKET", "D", "I"], lambda kw: (1, 2, 2))
@triton.heuristics(_even("D", "I"))
@triton.jit
def _gate_up_kernel(x_ptr, wg_ptr, wu_ptr, h_ptr, gu_ptr, order_ptr, offs_ptr, M_BUCKET,
                    stride_xm, stride_we, stride_wk, stride_wn,
                    E: tl.constexpr, D: tl.constexpr, I: tl.constexpr, K_TOP: tl.constexpr, E_P2: tl.constexpr,
                    STORE_H: tl.constexpr, STORE_GU: tl.constexpr, INTERP: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    EVEN_K: tl.constexpr, EVEN_N: tl.constexpr):
    """Sorted row r (token order[r] // K, expert e): g = x[t] @ W_gate[e], u = x[t] @ W_up[e] from one A tile,
    rounded to the weight dtype; h = silu(g) * u with the SiLU output rounded first, as eager does.
    Writes h[r] (STORE_H) and gu[r] = [g | u] (STORE_GU)."""
    pid = tl.program_id(0)
    n_n = tl.cdiv(I, BLOCK_N)
    pid_m = pid // n_n
    pid_n = pid % n_n
    e, row0, end = _row_block(pid_m, offs_ptr, E, E_P2, BLOCK_M)
    if e >= E:
        return
    rows = row0 + tl.arange(0, BLOCK_M)
    mrow = rows < end
    tok = tl.load(order_ptr + rows, mask=mrow, other=0) // K_TOP  # rows past the expert's end read x[0], never stored
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mcol = cols < I
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = x_ptr + tok[:, None] * stride_xm + ks[None, :]
    b_off = e.to(tl.int64) * stride_we + ks[:, None] * stride_wk + cols[None, :] * stride_wn
    bg_ptrs = wg_ptr + b_off
    bu_ptrs = wu_ptr + b_off
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            if EVEN_N:
                bg = tl.load(bg_ptrs)
                bu = tl.load(bu_ptrs)
            else:
                bg = tl.load(bg_ptrs, mask=mcol[None, :], other=0.0)
                bu = tl.load(bu_ptrs, mask=mcol[None, :], other=0.0)
        else:
            mk = (k0 + ks) < D
            a = tl.load(a_ptrs, mask=mk[None, :], other=0.0)
            bg = tl.load(bg_ptrs, mask=mk[:, None] & mcol[None, :], other=0.0)
            bu = tl.load(bu_ptrs, mask=mk[:, None] & mcol[None, :], other=0.0)
        acc_g = _mma(a, bg, acc_g, INTERP)
        acc_u = _mma(a, bu, acc_u, INTERP)
        a_ptrs += BLOCK_K
        bg_ptrs += BLOCK_K * stride_wk
        bu_ptrs += BLOCK_K * stride_wk
    dt = x_ptr.dtype.element_ty
    g = _round(acc_g, dt, INTERP)
    u = _round(acc_u, dt, INTERP)
    m2 = mrow[:, None] & mcol[None, :]
    rr = rows[:, None]
    cc = cols[None, :]
    if STORE_H:
        s = _round(_silu(g.to(tl.float32)), dt, INTERP)
        h = _round(s.to(tl.float32) * u.to(tl.float32), dt, INTERP)
        tl.store(h_ptr + rr * I + cc, h, mask=m2)
    if STORE_GU:
        tl.store(gu_ptr + rr * (2 * I) + cc, g, mask=m2)
        tl.store(gu_ptr + rr * (2 * I) + I + cc, u, mask=m2)


@_autotune(_SINGLE, ["M_BUCKET", "KD", "NO", "TWO"], lambda kw: (2, 2, 1) if kw.get("TWO") else (1, 1, 1))
@triton.heuristics(_even("KD", "NO"))
@triton.jit
def _scatter_mm_kernel(a_ptr, b1_ptr, b2_ptr, out_ptr, order_ptr, offs_ptr, M, M_BUCKET,
                       stride_am, stride_be, stride_bk, stride_bn,
                       E: tl.constexpr, KD: tl.constexpr, NO: tl.constexpr, E_P2: tl.constexpr, TWO: tl.constexpr,
                       INTERP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       EVEN_K: tl.constexpr, EVEN_N: tl.constexpr):
    """out[order[r]] = A[r, :KD] @ B1[e] (+ A[r, KD:2KD] @ B2[e]) for the sorted rows r of expert e: the down
    projection (A = h) and the input gradient (A = [dg | du], B = W_gate^T, W_up^T as strided views). Each row
    lands in its assignment slot t*K + k for the fixed-order k-sum."""
    pid = tl.program_id(0)
    n_n = tl.cdiv(NO, BLOCK_N)
    pid_m = pid // n_n
    pid_n = pid % n_n
    e, row0, end = _row_block(pid_m, offs_ptr, E, E_P2, BLOCK_M)
    if e >= E:
        return
    rows = row0 + tl.arange(0, BLOCK_M)
    mrow = rows < end
    ra = tl.minimum(rows, M - 1)  # rows past the expert's end are computed from valid memory but not stored
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mcol = cols < NO
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + ra[:, None] * stride_am + ks[None, :]
    b_off = e.to(tl.int64) * stride_be + ks[:, None] * stride_bk + cols[None, :] * stride_bn
    b1_ptrs = b1_ptr + b_off
    b2_ptrs = b2_ptr + b_off
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, KD, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            if TWO:
                a2 = tl.load(a_ptrs + KD)
            if EVEN_N:
                b = tl.load(b1_ptrs)
                if TWO:
                    b2 = tl.load(b2_ptrs)
            else:
                b = tl.load(b1_ptrs, mask=mcol[None, :], other=0.0)
                if TWO:
                    b2 = tl.load(b2_ptrs, mask=mcol[None, :], other=0.0)
        else:
            mk = (k0 + ks) < KD
            a = tl.load(a_ptrs, mask=mk[None, :], other=0.0)
            b = tl.load(b1_ptrs, mask=mk[:, None] & mcol[None, :], other=0.0)
            if TWO:
                a2 = tl.load(a_ptrs + KD, mask=mk[None, :], other=0.0)
                b2 = tl.load(b2_ptrs, mask=mk[:, None] & mcol[None, :], other=0.0)
        acc = _mma(a, b, acc, INTERP)
        if TWO:
            acc = _mma(a2, b2, acc, INTERP)
        a_ptrs += BLOCK_K
        b1_ptrs += BLOCK_K * stride_bk
        b2_ptrs += BLOCK_K * stride_bk
    slot = tl.load(order_ptr + rows, mask=mrow, other=0)
    tl.store(out_ptr + slot[:, None] * NO + cols[None, :], _round(acc, out_ptr.dtype.element_ty, INTERP),
             mask=mrow[:, None] & mcol[None, :])


@_autotune(_EPI, ["M_BUCKET", "D", "I"], lambda kw: (1, 1, 4), reset_to_zero=["dwp_ptr"])
@triton.heuristics(_even("D", "I"))
@triton.jit
def _down_bwd_kernel(dy_ptr, wd_ptr, gu_ptr, w_ptr, dgu_ptr, hw_ptr, dwp_ptr, order_ptr, offs_ptr, M_BUCKET,
                     stride_dym, stride_we, stride_wk, stride_wn, NWP,
                     E: tl.constexpr, D: tl.constexpr, I: tl.constexpr, K_TOP: tl.constexpr, E_P2: tl.constexpr,
                     COMPUTE_DW: tl.constexpr, INTERP: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     EVEN_K: tl.constexpr, EVEN_N: tl.constexpr):
    """Sorted row r (assignment f = order[r] = t*K + k, expert e): a = dy[t] @ W_down[e]^T in fp32 (B is the
    strided view W_down^T). Epilogue with the saved g, u: h = silu(g) * u exactly as in the forward,
    dh = w_f a, the backward of silu(g) * u with eager's rounding points -> dgu[r] = [dg | du],
    hw[r] = w_f h for dW_down, and dwp[f, tile] = sum over this tile's columns of a * h (dw_f = sum_tiles)."""
    pid = tl.program_id(0)
    n_n = tl.cdiv(I, BLOCK_N)
    pid_m = pid // n_n
    pid_n = pid % n_n
    e, row0, end = _row_block(pid_m, offs_ptr, E, E_P2, BLOCK_M)
    if e >= E:
        return
    rows = row0 + tl.arange(0, BLOCK_M)
    mrow = rows < end
    flat = tl.load(order_ptr + rows, mask=mrow, other=0)
    tok = flat // K_TOP
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mcol = cols < I
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = dy_ptr + tok[:, None] * stride_dym + ks[None, :]
    b_ptrs = wd_ptr + e.to(tl.int64) * stride_we + ks[:, None] * stride_wk + cols[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mcol[None, :], other=0.0)
        else:
            mk = (k0 + ks) < D
            a = tl.load(a_ptrs, mask=mk[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mk[:, None] & mcol[None, :], other=0.0)
        acc = _mma(a, b, acc, INTERP)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_wk
    dt = dgu_ptr.dtype.element_ty
    m2 = mrow[:, None] & mcol[None, :]
    rr = rows[:, None]
    cc = cols[None, :]
    g = tl.load(gu_ptr + rr * (2 * I) + cc, mask=m2, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + rr * (2 * I) + I + cc, mask=m2, other=0.0).to(tl.float32)
    wf = tl.load(w_ptr + flat, mask=mrow, other=0.0).to(tl.float32)[:, None]
    s = _round(_silu(g), dt, INTERP).to(tl.float32)        # F.silu(g), rounded as in the forward
    h = _round(s * u, dt, INTERP).to(tl.float32)
    if COMPUTE_DW:
        tl.store(dwp_ptr + flat * NWP + pid_n, tl.sum(acc * h, axis=1), mask=mrow)
    dh = _round(acc * wf, dt, INTERP).to(tl.float32)
    ds = _round(dh * u, dt, INTERP).to(tl.float32)         # grad of F.silu(g) (mul backward)
    du = _round(dh * s, dt, INTERP)
    sig = 1.0 / (1.0 + tl.exp(-g))
    dg = _round((ds * sig) * (1.0 + g * (1.0 - sig)), dt, INTERP)  # ATen silu_backward
    tl.store(dgu_ptr + rr * (2 * I) + cc, dg, mask=m2)
    tl.store(dgu_ptr + rr * (2 * I) + I + cc, du, mask=m2)
    tl.store(hw_ptr + rr * I + cc, _round(h * wf, dt, INTERP), mask=m2)


@triton.heuristics({"EVEN_P": lambda a: a["P"] % a["BLOCK_M"] == 0, "EVEN_Q": lambda a: a["Q"] % a["BLOCK_N"] == 0})
@triton.jit
def _wgrad_kernel(a_ptr, b_ptr, out_ptr, out2_ptr, order_ptr, offs_ptr, perm_ptr, M_BUCKET,
                  stride_am, stride_bm,
                  P: tl.constexpr, Q: tl.constexpr, B2_OFF: tl.constexpr, K_TOP: tl.constexpr,
                  A_GATHER: tl.constexpr, B_GATHER: tl.constexpr, TWO: tl.constexpr, INTERP: tl.constexpr,
                  ACC: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  EVEN_P: tl.constexpr, EVEN_Q: tl.constexpr):
    """out[e] = A_e^T @ B_e (and out2[e] = A_e^T @ B2_e with B2 = B shifted by B2_OFF columns): fp32 sums over
    the sorted rows of expert e, [P, Q] per expert, zeros for an expert without rows. A and B rows are read
    directly (sorted buffers) or through the token index (x, dy). Programs take the experts in descending row
    count (perm): the longest reductions start first. ACC: out (and out2) are fp32 gradient buffers and get
    out += sum (the finished fp32 sum added once, like the hook's buf.add_(g) without rounding g to bf16 first);
    every tile is owned by exactly one program, so the read-add-write needs no atomics."""
    pid = tl.program_id(0)
    n_p = tl.cdiv(P, BLOCK_M)
    n_q = tl.cdiv(Q, BLOCK_N)
    e = tl.load(perm_ptr + pid // (n_p * n_q))
    t = pid % (n_p * n_q)
    pid_p = t // n_q
    pid_q = t % n_q
    start = tl.load(offs_ptr + e).to(tl.int32)
    end = tl.load(offs_ptr + e + 1).to(tl.int32)
    ps = pid_p * BLOCK_M + tl.arange(0, BLOCK_M)
    qs = pid_q * BLOCK_N + tl.arange(0, BLOCK_N)
    mp = ps < P
    mq = qs < Q
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for r0 in range(start, end, BLOCK_K):
        rows = r0 + rk
        mr = rows < end
        if A_GATHER:
            ar = tl.load(order_ptr + rows, mask=mr, other=0) // K_TOP
        else:
            ar = rows.to(tl.int64)
        if B_GATHER:
            br = tl.load(order_ptr + rows, mask=mr, other=0) // K_TOP
        else:
            br = rows.to(tl.int64)
        if EVEN_P:
            ma = mr[None, :]
        else:
            ma = mr[None, :] & mp[:, None]
        if EVEN_Q:
            mb = mr[:, None]
        else:
            mb = mr[:, None] & mq[None, :]
        # A^T loaded as it is consumed, [BLOCK_M, BLOCK_K] contiguous along dim 0 (a column-major operand): a
        # register transpose of a [BLOCK_K, BLOCK_M] tile would keep the pipeliner from streaming it into shared memory
        at = tl.load(a_ptr + ar[None, :] * stride_am + ps[:, None], mask=ma, other=0.0)
        b = tl.load(b_ptr + br[:, None] * stride_bm + qs[None, :], mask=mb, other=0.0)   # [BLOCK_K, BLOCK_N]
        acc = _mma(at, b, acc, INTERP)
        if TWO:
            b2 = tl.load(b_ptr + B2_OFF + br[:, None] * stride_bm + qs[None, :], mask=mb, other=0.0)
            acc2 = _mma(at, b2, acc2, INTERP)
    o_off = e.to(tl.int64) * (P * Q) + ps[:, None] * Q + qs[None, :]
    mo = mp[:, None] & mq[None, :]
    if ACC:
        acc += tl.load(out_ptr + o_off, mask=mo, other=0.0)
    tl.store(out_ptr + o_off, _round(acc, out_ptr.dtype.element_ty, INTERP), mask=mo)
    if TWO:
        if ACC:
            acc2 += tl.load(out2_ptr + o_off, mask=mo, other=0.0)
        tl.store(out2_ptr + o_off, _round(acc2, out2_ptr.dtype.element_ty, INTERP), mask=mo)


_wgrad_single = _autotune(_SINGLE, ["M_BUCKET", "P", "Q", "TWO"], lambda kw: (1, 1, 1))(_wgrad_kernel)
_wgrad_dual = _autotune(_DUAL, ["M_BUCKET", "P", "Q", "TWO"], lambda kw: (1, 2, 2))(_wgrad_kernel)
# ACC=True (fused gradient accumulation): the kernel adds into its outputs, so autotuning (several runs per config)
# must put the buffers back after every timed run (restore_value: a copy of each buffer while tuning, once per shape
# bucket; results are cached on disk). Separate tuners: the plain kernels' tuning stays as it was, and the fp32
# read-add-write epilogue gets configs of its own.
_wgrad_single_acc = _autotune(_SINGLE, ["M_BUCKET", "P", "Q", "TWO"], lambda kw: (1, 1, 1),
                              restore_value=["out_ptr"])(_wgrad_kernel)  # out2 is out (unused) when not TWO
_wgrad_dual_acc = _autotune(_DUAL, ["M_BUCKET", "P", "Q", "TWO"], lambda kw: (1, 2, 2),
                            restore_value=["out_ptr", "out2_ptr"])(_wgrad_kernel)


@triton.jit
def _combine_kernel(src_ptr, w_ptr, a1_ptr, a2_ptr, a3_ptr, dl_ptr, wr_ptr, out_ptr, N, D: tl.constexpr,
                    K_TOP: tl.constexpr, E_R: tl.constexpr, HAS_W: tl.constexpr, HAS_A1: tl.constexpr,
                    HAS_A2: tl.constexpr, HAS_A3: tl.constexpr, ROUND_FIRST: tl.constexpr, ROUTER: tl.constexpr,
                    INTERP: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    """out[t] = sum_k w[t, k] * src[t*K + k] (w = 1 without HAS_W): fp32 sum, k = 0..K-1 in order. Without the flags
    below this is the plain combine's code.
    a1, a2, a3: [N, D] tensors added to it. ROUND_FIRST reproduces separate adds of rounded tensors in the order the old
    path's autograd makes them (bit-identical): the k-sum rounded to out's dtype; a = round(a1 + a2); round(k-sum + a);
    + a3, rounded at the store (each add in fp32, then rounded: what an add of two bf16 tensors computes). Otherwise
    everything goes into the fp32 sum. ROUTER: + sum_e dl[t, e] * wr[e, :] (the router's input gradient), fp32 products
    added into the fp32 sum, e = 0..E_R-1 in order. One rounding at the store."""
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    ts = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    ds = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mt = ts < N
    m2 = mt[:, None] & (ds < D)[None, :]
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for k in tl.static_range(K_TOP):
        r = ts.to(tl.int64) * K_TOP + k
        v = tl.load(src_ptr + r[:, None] * D + ds[None, :], mask=m2, other=0.0).to(tl.float32)
        if HAS_W:
            v = v * tl.load(w_ptr + r, mask=mt, other=0.0).to(tl.float32)[:, None]
        acc += v
    o_off = ts.to(tl.int64)[:, None] * D + ds[None, :]
    if ROUND_FIRST:
        acc = _round(acc, out_ptr.dtype.element_ty, INTERP).to(tl.float32)  # the k-sum as the plain combine writes it
        if HAS_A1:
            a = tl.load(a1_ptr + o_off, mask=m2, other=0.0).to(tl.float32)
            if HAS_A2:
                a = a + tl.load(a2_ptr + o_off, mask=m2, other=0.0).to(tl.float32)
                a = _round(a, out_ptr.dtype.element_ty, INTERP).to(tl.float32)
            acc = acc + a
            if HAS_A3:
                acc = _round(acc, out_ptr.dtype.element_ty, INTERP).to(tl.float32)
        if HAS_A3:
            acc = acc + tl.load(a3_ptr + o_off, mask=m2, other=0.0).to(tl.float32)
    else:
        if HAS_A1:
            acc += tl.load(a1_ptr + o_off, mask=m2, other=0.0).to(tl.float32)
        if HAS_A2:
            acc += tl.load(a2_ptr + o_off, mask=m2, other=0.0).to(tl.float32)
        if HAS_A3:
            acc += tl.load(a3_ptr + o_off, mask=m2, other=0.0).to(tl.float32)
    if ROUTER:
        t64 = ts.to(tl.int64)
        md = ds < D
        for e in tl.static_range(E_R):
            dl = tl.load(dl_ptr + t64 * E_R + e, mask=mt, other=0.0).to(tl.float32)
            wr = tl.load(wr_ptr + e * D + ds, mask=md, other=0.0).to(tl.float32)
            acc += dl[:, None] * wr[None, :]
    tl.store(out_ptr + o_off, _round(acc, out_ptr.dtype.element_ty, INTERP), mask=m2)


_AUTOTUNED = (_gate_up_kernel, _scatter_mm_kernel, _down_bwd_kernel, _wgrad_single, _wgrad_dual, _wgrad_single_acc,
              _wgrad_dual_acc)
_DEFAULT_CONFIGS = [list(k.configs) for k in _AUTOTUNED]


def set_configs(block_m: int, block_n: int, block_k: int, num_warps: int = 4, num_stages: int = 3) -> None:
    """Use one fixed config for every GEMM kernel (tests: other tile sizes; debugging). reset_configs() undoes."""
    for k in _AUTOTUNED:
        k.configs = [_cfg(block_m, block_n, block_k, num_warps, num_stages)]
        k.cache.clear()


def reset_configs() -> None:
    for k, cfgs in zip(_AUTOTUNED, _DEFAULT_CONFIGS):
        k.configs = list(cfgs)
        k.cache.clear()


# ---------------------------------------------------------------------------- launchers
def _bucket(m: int) -> int:
    return triton.next_power_of_2(max(m, 1))  # autotune key: retune only when M changes by ~2x


def _min_block_n() -> int:
    return min(c.kwargs["BLOCK_N"] for c in _down_bwd_kernel.configs)


def _sort_assignments(idx: torch.Tensor, E: int):
    """Stable sort of the N*K assignments by expert, on the GPU: order (sorted position -> slot t*K + k),
    offs ([E + 1] first row of each expert, offs[E] = N*K) and perm (experts by descending row count)."""
    flat = idx.reshape(-1)
    e_sorted, order = torch.sort(flat, stable=True)
    offs = torch.searchsorted(e_sorted, torch.arange(E + 1, device=idx.device, dtype=e_sorted.dtype))
    perm = torch.argsort(offs[1:] - offs[:-1], descending=True, stable=True)
    return order, offs, perm


# Counting sort of the assignments (ModelConfig.moe_sort = "counting"): the same (order, offs, perm) as _sort_assignments
# with three small kernels instead of torch.sort (a radix sort over all 64 key bits), searchsorted and the argsort of the
# counts (~15 launches per MoE layer and micro-batch). Blocks of _SORT_BLOCK assignments: counts per block and expert;
# one program turns them into each expert's first row (offs), each (block, expert)'s first row (base) and the experts by
# descending count (perm, ties by index: torch.argsort(descending=True, stable=True)); then every assignment goes to
# base[its block, its expert] + the number of earlier assignments of its block with the same expert. Within an expert the
# rows keep the assignment order, which is what the stable sort gives: identical integers.
_SORT_BLOCK = 256


@triton.jit
def _sort_count_kernel(idx_ptr, cnt_ptr, M, E: tl.constexpr, E_P2: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(idx_ptr + offs, mask=offs < M, other=E_P2)  # masked lanes match no expert column
    cols = tl.arange(0, E_P2)
    hot = (e[:, None] == cols[None, :]).to(tl.int32)
    tl.store(cnt_ptr + pid * E + cols, tl.sum(hot, axis=0), mask=cols < E)


@triton.jit
def _sort_scan_kernel(cnt_ptr, base_ptr, offs_ptr, perm_ptr, NB, E: tl.constexpr, E_P2: tl.constexpr,
                      NB_P2: tl.constexpr):
    rows = tl.arange(0, NB_P2)
    cols = tl.arange(0, E_P2)
    m2 = (rows[:, None] < NB) & (cols[None, :] < E)
    c = tl.load(cnt_ptr + rows[:, None] * E + cols[None, :], mask=m2, other=0)
    tot = tl.sum(c, axis=0)                      # rows per expert
    start = tl.cumsum(tot, axis=0) - tot         # first row of each expert
    base = start[None, :] + (tl.cumsum(c, axis=0) - c)
    tl.store(base_ptr + rows[:, None] * E + cols[None, :], base, mask=m2)
    tl.store(offs_ptr + cols, start.to(tl.int64), mask=cols < E)
    tl.store(offs_ptr + E + cols, tl.sum(tot, axis=0).to(tl.int64) + cols * 0, mask=cols == 0)
    # perm: rank of expert i = experts with more rows + experts with as many rows and a smaller index
    t = tl.where(cols < E, tot, -1)
    before = (t[None, :] > t[:, None]) | ((t[None, :] == t[:, None]) & (cols[None, :] < cols[:, None]))
    rank = tl.sum(before.to(tl.int32), axis=1)
    tl.store(perm_ptr + rank, cols.to(tl.int64), mask=cols < E)


@triton.jit
def _sort_place_kernel(idx_ptr, base_ptr, order_ptr, M, E: tl.constexpr, E_P2: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < M
    e = tl.load(idx_ptr + offs, mask=valid, other=E_P2)
    cols = tl.arange(0, E_P2)
    hot = (e[:, None] == cols[None, :]).to(tl.int32)
    earlier = tl.sum((tl.cumsum(hot, axis=0) - hot) * hot, axis=1)  # earlier assignments of this block, same expert
    b = tl.load(base_ptr + pid * E + e, mask=valid, other=0)
    tl.store(order_ptr + b + earlier, offs.to(tl.int64), mask=valid)


def _sort_assignments_counting(idx: torch.Tensor, E: int):
    flat = idx.reshape(-1).contiguous()
    M, dev = flat.numel(), flat.device
    nb, e_p2 = triton.cdiv(M, _SORT_BLOCK), triton.next_power_of_2(E)
    cnt = torch.empty(nb, E, dtype=torch.int32, device=dev)
    base = torch.empty(nb, E, dtype=torch.int32, device=dev)
    order = torch.empty(M, dtype=torch.int64, device=dev)
    offs = torch.empty(E + 1, dtype=torch.int64, device=dev)
    perm = torch.empty(E, dtype=torch.int64, device=dev)
    _sort_count_kernel[(nb,)](flat, cnt, M, E=E, E_P2=e_p2, BLOCK=_SORT_BLOCK, num_warps=8)
    _sort_scan_kernel[(1,)](cnt, base, offs, perm, nb, E=E, E_P2=e_p2, NB_P2=triton.next_power_of_2(nb), num_warps=4)
    _sort_place_kernel[(nb,)](flat, base, order, M, E=E, E_P2=e_p2, BLOCK=_SORT_BLOCK, num_warps=8)
    return order, offs, perm


SORT_IMPLS = ("torch", "counting")


def sort_assignments(idx: torch.Tensor, E: int, impl: str = "torch"):
    """(order, offs, perm) of _sort_assignments; impl "counting" runs the counting-sort kernels (the same integers)
    where the Triton kernels run (kernels_run) and there are assignments."""
    if impl not in SORT_IMPLS:
        raise ValueError(f"moe_sort={impl!r}; expected one of {SORT_IMPLS}")
    if impl == "counting" and idx.numel() > 0 and kernels_run(idx):
        return _sort_assignments_counting(idx, E)
    return _sort_assignments(idx, E)


def _gate_up(xc, wg, wu, order, offs, K, store_h, store_gu):
    M = order.numel()
    E, D, I = wg.shape
    assert wu.stride() == wg.stride() and xc.stride(1) == 1
    h = xc.new_empty(M, I) if store_h else xc
    gu = xc.new_empty(M, 2 * I) if store_gu else xc
    grid = lambda meta: ((triton.cdiv(M, meta["BLOCK_M"]) + E) * triton.cdiv(I, meta["BLOCK_N"]),)  # noqa: E731
    _gate_up_kernel[grid](xc, wg, wu, h, gu, order, offs, _bucket(M),
                          xc.stride(0), wg.stride(0), wg.stride(1), wg.stride(2),
                          E=E, D=D, I=I, K_TOP=K, E_P2=triton.next_power_of_2(E),
                          STORE_H=store_h, STORE_GU=store_gu, INTERP=_INTERPRET)
    return (h if store_h else None), (gu if store_gu else None)


def _scatter_mm(a, b1, b2, order, offs, out_dtype):
    """b1, b2: [E, KD, NO] (possibly strided views); a: [M, KD] or [M, 2 KD] rows."""
    M = order.numel()
    E, KD, NO = b1.shape
    two = b2 is not None
    assert a.stride(1) == 1 and (not two or b2.stride() == b1.stride())
    out = torch.empty(M, NO, device=a.device, dtype=out_dtype)
    grid = lambda meta: ((triton.cdiv(M, meta["BLOCK_M"]) + E) * triton.cdiv(NO, meta["BLOCK_N"]),)  # noqa: E731
    _scatter_mm_kernel[grid](a, b1, b2 if two else b1, out, order, offs, M, _bucket(M),
                             a.stride(0), b1.stride(0), b1.stride(1), b1.stride(2),
                             E=E, KD=KD, NO=NO, E_P2=triton.next_power_of_2(E), TWO=two, INTERP=_INTERPRET)
    return out


def _down_bwd(dy, wd, gu, w, order, offs, K, compute_dw):
    M = order.numel()
    E, I, D = wd.shape
    wdt = wd.transpose(1, 2)  # B[k, n] = W_down[e, n, k]
    dgu = gu.new_empty(M, 2 * I)
    hw = gu.new_empty(M, I)
    nwp = triton.cdiv(I, _min_block_n())
    dwp = torch.zeros(M if compute_dw else 1, nwp, device=dy.device, dtype=torch.float32)
    grid = lambda meta: ((triton.cdiv(M, meta["BLOCK_M"]) + E) * triton.cdiv(I, meta["BLOCK_N"]),)  # noqa: E731
    _down_bwd_kernel[grid](dy, wdt, gu, w, dgu, hw, dwp, order, offs, _bucket(M),
                           dy.stride(0), wdt.stride(0), wdt.stride(1), wdt.stride(2), nwp,
                           E=E, D=D, I=I, K_TOP=K, E_P2=triton.next_power_of_2(E),
                           COMPUTE_DW=compute_dw, INTERP=_INTERPRET)
    return dgu, hw, (dwp if compute_dw else None)


def _wgrad(a, b, order, offs, perm, K, P, Q, a_gather, b_gather, two, out_dtype, acc=None):
    """[E, P, Q] weight gradient(s) in out_dtype; or, with acc (fp32 gradient buffer [E, P, Q], a pair when two),
    the gradients are added into acc and nothing is returned."""
    E = perm.numel()
    M = order.numel()
    if acc is None:
        out = torch.empty(E, P, Q, device=a.device, dtype=out_dtype)
        out2 = torch.empty(E, P, Q, device=a.device, dtype=out_dtype) if two else out
        kern = _wgrad_dual if two else _wgrad_single
    else:
        out, out2 = acc if two else (acc, acc)
        for t in (out, out2):
            assert (t.dtype == torch.float32 and tuple(t.shape) == (E, P, Q) and t.is_contiguous()
                    and t.device == a.device), "acc: contiguous fp32 [E, P, Q] buffers on the input device"
        kern = _wgrad_dual_acc if two else _wgrad_single_acc
    grid = lambda meta: (E * triton.cdiv(P, meta["BLOCK_M"]) * triton.cdiv(Q, meta["BLOCK_N"]),)  # noqa: E731
    kern[grid](a, b, out, out2, order, offs, perm, _bucket(M), a.stride(0), b.stride(0),
               P=P, Q=Q, B2_OFF=Q if two else 0, K_TOP=K, A_GATHER=a_gather, B_GATHER=b_gather, TWO=two,
               INTERP=_INTERPRET, ACC=acc is not None)
    if acc is None:
        return (out, out2) if two else out
    return None


def _combine(src, w, N, K, out_dtype, a1=None, a2=None, a3=None, round_first=False, dl=None, wr=None):
    """[N, D] k-sum of src [N*K, D] (weighted by w [N, K] if given) + a1 + a2 + a3 ([N, D] each; round_first: rounded
    as the old path's separate adds, see _combine_kernel) + dl [N, E] @ wr [E, D] in fp32."""
    D = src.shape[1]
    out = torch.empty(N, D, device=src.device, dtype=out_dtype)
    bt, bd = (16, 32) if _INTERPRET else (32, 128)
    bd = min(bd, triton.next_power_of_2(D))
    grid = (triton.cdiv(N, bt), triton.cdiv(D, bd))
    assert a2 is None or a1 is not None
    for a in (a1, a2, a3):
        assert a is None or (tuple(a.shape) == (N, D) and a.is_contiguous() and a.device == src.device)
    router = dl is not None
    if router:
        assert dl.shape[0] == N and tuple(wr.shape) == (dl.shape[1], D) and dl.is_contiguous() and wr.is_contiguous()
    _combine_kernel[grid](src, src if w is None else w, src if a1 is None else a1, src if a2 is None else a2,
                          src if a3 is None else a3, dl if router else src, wr if router else src, out, N, D=D,
                          K_TOP=K, E_R=dl.shape[1] if router else 1, HAS_W=w is not None, HAS_A1=a1 is not None,
                          HAS_A2=a2 is not None, HAS_A3=a3 is not None, ROUND_FIRST=bool(round_first), ROUTER=router,
                          INTERP=_INTERPRET, BLOCK_T=bt, BLOCK_D=bd, num_warps=4)
    return out


# ---------------------------------------------------------------------------- autograd
def _experts_forward(ctx, x, w, w_gate, w_up, w_down, idx, recompute: bool, train: bool, sinks, add=None, sort=None):
    """RoutedExperts' forward (see there); add: [N, D] in the weights' dtype added in the combine exactly as a separate
    add of the rounded routed output would add it (RoutedExpertsFan: the shared expert's output); sort: the (order, offs,
    perm) of idx if the caller already has them (sort_assignments), else they are computed here."""
    N, K = idx.shape
    E, D, I = w_gate.shape
    xc = x.to(w_gate.dtype).contiguous()
    w = w.contiguous()
    order, offs, perm = sort if sort is not None else _sort_assignments(idx, E)
    keep_gu = train and not recompute
    h, gu = _gate_up(xc, w_gate, w_up, order, offs, K, store_h=True, store_gu=keep_gu)
    out = _scatter_mm(h, w_down, None, order, offs, xc.dtype)
    del h
    y = _combine(out, w, N, K, xc.dtype, a1=add, round_first=add is not None)
    if train:
        ctx.save_for_backward(xc, w, w_gate, w_up, w_down, order, offs, perm, *([gu] if keep_gu else []))
        ctx.x_dtype, ctx.K, ctx.sinks = x.dtype, K, sinks
    return y


def _experts_backward(ctx, dy, combine: bool = True):
    """RoutedExperts' backward: (dx, dw, dW_gate, dW_up, dW_down), None where not needed. dx is the k-sum [N, D] in x's
    dtype (combine), or the per-assignment input gradients [N*K, D] in the weights' dtype, row t*K + k (the rows the
    k-sum adds, before it)."""
    xc, w, wg, wu, wd, order, offs, perm, *saved = ctx.saved_tensors
    N, K = xc.shape[0], ctx.K
    E, D, I = wg.shape
    need_x, need_w, need_g, need_u, need_d = ctx.needs_input_grad[:5]
    gu = saved[0] if saved else _gate_up(xc, wg, wu, order, offs, K, store_h=False, store_gu=True)[1]
    dy = dy.to(xc.dtype).contiguous()
    dgu, hw, dwp = _down_bwd(dy, wd, gu, w, order, offs, K, need_w)
    del gu, saved
    tok = order // K if _WGRAD_COPY else None
    sinks = ctx.sinks  # fused accumulation: add into the fp32 buffers, return no weight gradients
    dw_down = None
    if need_d:
        acc = sinks[2].buffer() if sinks is not None else None
        if _WGRAD_COPY:
            dw_down = _wgrad(hw, dy.index_select(0, tok), order, offs, perm, K, I, D, False, False, False, wd.dtype,
                             acc)
        else:
            dw_down = _wgrad(hw, dy, order, offs, perm, K, I, D, False, True, False, wd.dtype, acc)
    del hw
    dw_gate = dw_up = None
    if need_g or need_u:
        acc = (sinks[0].buffer(), sinks[1].buffer()) if sinks is not None else None
        if _WGRAD_COPY:
            res = _wgrad(xc.index_select(0, tok), dgu, order, offs, perm, K, D, I, False, False, True, wg.dtype,
                         acc)
        else:
            res = _wgrad(xc, dgu, order, offs, perm, K, D, I, True, False, True, wg.dtype, acc)
        if res is not None:
            dw_gate, dw_up = res
    dx = None
    if need_x:
        dxs = _scatter_mm(dgu, wg.transpose(1, 2), wu.transpose(1, 2), order, offs, xc.dtype)
        del dgu
        dx = _combine(dxs, None, N, K, ctx.x_dtype) if combine else dxs
    dw = dwp.sum(1).view(N, K).to(w.dtype) if need_w else None
    return dx, dw, dw_gate if need_g else None, dw_up if need_u else None, dw_down


class RoutedExperts(torch.autograd.Function):
    """See the module docstring. Inputs: x [N, D] (any float dtype; computed in the weights' dtype), w [N, K]
    combine weights (used at their own precision), weights [E, D, I] / [E, I, D], idx [N, K] expert ids.
    Output [N, D] in the weights' dtype; gradients in each input's dtype. sinks: None, or GradSinks (layers/
    common.py) of the gate, up and down weights: their gradients are then added into those fp32 buffers by the
    weight-gradient kernels and returned as None."""

    @staticmethod
    def forward(ctx, x, w, w_gate, w_up, w_down, idx, recompute: bool, train: bool, sinks=None, sort=None):
        return _experts_forward(ctx, x, w, w_gate, w_up, w_down, idx, recompute, train, sinks, sort=sort)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        return (*_experts_backward(ctx, dy), None, None, None, None, None)


class RoutedExpertsFan(torch.autograd.Function):
    """RoutedExperts inside the fused fan-in (MoEFan): x arrives as MoEFan's [N, K, D] view xr (stride 0 over k) and
    its gradient goes back as the per-assignment input gradients [N, K, D] (row t*K + k of the dx GEMM), which MoEFan
    sums together with the other parts of the input gradient; shared ([N, D] in the weights' dtype, or None) is added
    to the output in the combine, rounded as the separate add of the old path rounds it (bit-identical), and gets the
    output gradient as its gradient. Everything else as RoutedExperts (same kernels, same saved tensors)."""

    @staticmethod
    def forward(ctx, xr, w, w_gate, w_up, w_down, idx, shared, recompute: bool, train: bool, sinks=None, sort=None):
        return _experts_forward(ctx, xr[:, 0], w, w_gate, w_up, w_down, idx, recompute, train, sinks, add=shared,
                                sort=sort)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        dxs, dw, dw_gate, dw_up, dw_down = _experts_backward(ctx, dy, combine=False)
        g_xr = None
        if dxs is not None:
            g_xr = dxs.view(dxs.shape[0] // ctx.K, ctx.K, dxs.shape[1])
            g_xr = g_xr if g_xr.dtype == ctx.x_dtype else g_xr.to(ctx.x_dtype)
        g_shared = dy if ctx.needs_input_grad[6] else None
        return g_xr, dw, dw_gate, dw_up, dw_down, None, g_shared, None, None, None, None


def kernels_run(t: torch.Tensor) -> bool:
    """The Triton kernels run for this tensor: CUDA, or any tensor under Triton's CPU interpreter."""
    return t.is_cuda or _INTERPRET


def routed_experts(x, idx, w, w_gate, w_up, w_down, recompute: bool = False, sinks=None, sort=None):
    """y[t] = sum_k w[t,k] * W_down[e](silu(x[t] @ W_gate[e]) * (x[t] @ W_up[e])), e = idx[t,k].

    x: [N, D], idx: [N, K] int expert ids, w: [N, K], w_gate / w_up: [E, D, I], w_down: [E, I, D]. Computed in
    the weights' dtype, output in that dtype. recompute=True keeps only x, w and the sort for backward and
    reruns the gate/up GEMM there (bit-identical values). sinks: GradSinks of (w_gate, w_up, w_down) for fused
    gradient accumulation (see RoutedExperts). CPU tensors without the interpreter, and empty inputs, take the
    plain-PyTorch reference (its weight gradients reach the buffers through the copies' hooks instead)."""
    if idx.numel() == 0 or not (x.is_cuda or _INTERPRET):
        return routed_experts_reference(x, idx, w, w_gate, w_up, w_down)
    train = torch.is_grad_enabled() and any(t.requires_grad for t in (x, w, w_gate, w_up, w_down))
    return RoutedExperts.apply(x, w, w_gate, w_up, w_down, idx, bool(recompute) and train, train,
                               sinks if train else None, sort)


def routed_experts_fan(xr, idx, w, w_gate, w_up, w_down, shared=None, recompute: bool = False, sinks=None, sort=None):
    """routed_experts(xr[:, 0], ...) + shared inside the fused fan-in: xr is MoEFan's [N, K, D] view of the input, whose
    gradient is returned per assignment (see RoutedExpertsFan). Needs the kernels (kernels_run) and N > 0."""
    assert xr.dim() == 3 and xr.shape[:2] == idx.shape and idx.numel() > 0 and kernels_run(xr)
    train = torch.is_grad_enabled() and any(t is not None and t.requires_grad
                                            for t in (xr, w, w_gate, w_up, w_down, shared))
    return RoutedExpertsFan.apply(xr, w, w_gate, w_up, w_down, idx, shared, bool(recompute) and train, train,
                                  sinks if train else None, sort)


def _fan_in(g_xr, g_xg, g_xu, g_xq, g_logits, w_r, dims, out_dtype, exact: bool):
    """MoEFan's input gradient in one combine: sum_k g_xr[:, k] + g_xg + g_xu + g_xq + g_logits @ w_r (missing parts
    left out); exact: rounded as the old path's autograd adds them (see _combine_kernel)."""
    N, K, D = dims
    dev = next(t for t in (g_xr, g_xg, g_xu, g_xq, g_logits) if t is not None).device
    if g_xr is None:  # no gradient through the routed experts (not in training): the k-sum is zero
        src = torch.zeros(N * K, D, device=dev, dtype=out_dtype)
    else:
        src = g_xr.reshape(N * K, D).contiguous()
    shared = [t.contiguous() for t in (g_xg, g_xu) if t is not None]
    a1 = shared[0] if shared else None
    a2 = shared[1] if len(shared) > 1 else None
    a3 = g_xq.contiguous() if g_xq is not None else None
    dl = wr = None
    if g_logits is not None:
        dl, wr = g_logits.contiguous(), w_r.contiguous()
    return _combine(src, None, N, K, out_dtype, a1=a1, a2=a2, a3=a3, round_first=exact, dl=dl, wr=wr)


def _router_wgrad_lean(dl: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dl^T @ x for the fp32 logits gradient dl [N, E] and the bf16 input x [N, D], without a fp32 copy of x. dl is
    split into three bf16 terms, hi = bf16(dl), mid = bf16(dl - hi), lo = bf16(dl - hi - mid), whose sum is dl to fp32
    precision (|dl - hi - mid - lo| <= 2^-24 |dl|); one GEMM of [hi | mid | lo]^T with x: exact products of bf16
    values, fp32 sums and output (CUDA: cuBLAS bf16 GEMM with out_dtype fp32; elsewhere the same products in a fp32
    GEMM), then the three partial results added in fp32. Accuracy of a fp32 GEMM; the old path's fp32 GEMM runs in TF32
    on the GPU (train.py sets allow_tf32), which rounds dl to 2^-11."""
    N, E = dl.shape
    a = torch.empty(N, 3 * E, dtype=torch.bfloat16, device=dl.device)
    a[:, :E] = dl                     # hi: rounded to nearest even
    r = dl - a[:, :E]                 # fp32, exact
    a[:, E:2 * E] = r                 # mid
    a[:, 2 * E:] = r - a[:, E:2 * E]  # lo
    xb = x.to(torch.bfloat16)
    out = torch.mm(a.t(), xb, out_dtype=torch.float32) if x.is_cuda else a.t().float().mm(xb.float())
    return out.view(3, E, -1).sum(0)


class MoEFan(torch.autograd.Function):
    """The MoE layer's input fan-out / fan-in (ModelConfig.moe_shared_fused; see the module docstring).

    Forward: xf [N, D] -> views of xf for each reader, so that each one's input gradient arrives here separately:
      logits [N, E]   F.linear(xf.float(), w_r) in fp32, if w_r is given (the call MoE.route makes: the same logits);
      xr [N, K, D]    xf with stride 0 over k, for RoutedExpertsFan (its gradient: the per-assignment input gradients);
      xg, xu          xf for the shared expert's gate and up projections, if shared;
      xq              xf for the router, if exact (the router then stays outside: its input gradient arrives in bf16).
    Backward: one combine kernel writes dx = sum_k g_xr[:, k] + g_xg + g_xu + g_xq + g_logits @ w_r:
      exact=True:  rounded exactly as the old path's autograd sums them (the routed combine's output, then the adds
                   ((g_xg + g_xu) + routed) + g_xq, each rounded): bit-identical;
      exact=False: all parts summed in fp32 (g_logits @ w_r as fp32 products) and rounded once.
    The router weight's gradient is g_logits^T @ xf.float(), the call autograd makes for F.linear (bit-identical);
    lean: computed from the bf16 xf by _router_wgrad_lean, and the fp32 copy of xf is not kept for the backward."""

    @staticmethod
    def forward(ctx, xf, w_r, K: int, shared: bool, exact: bool, lean: bool):
        N, D = xf.shape
        router = w_r is not None
        assert router != bool(exact), "exact keeps the router outside the fan node; otherwise its GEMM is inside"
        outs, saved = [], []
        if router:
            with torch.autocast(xf.device.type, enabled=False):
                x32 = xf.float()
                outs.append(F.linear(x32, w_r.float()))
            lean = bool(lean) and x32 is not xf  # an fp32 input is its own fp32 copy
            saved = [xf if lean else x32, w_r]
        outs.append(xf.unsqueeze(1).expand(N, K, D))
        if shared:
            outs += [xf.view_as(xf), xf.view_as(xf)]
        if exact:
            outs.append(xf.view_as(xf))
        ctx.set_materialize_grads(False)
        ctx.router, ctx.shared, ctx.exact, ctx.lean = router, bool(shared), bool(exact), router and lean
        ctx.dims, ctx.x_dtype = (N, K, D), xf.dtype
        ctx.save_for_backward(*saved)
        return tuple(outs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grads):
        g = list(grads)
        g_logits = g.pop(0) if ctx.router else None
        g_xr = g.pop(0)
        g_xg, g_xu = (g.pop(0), g.pop(0)) if ctx.shared else (None, None)
        g_xq = g.pop(0) if ctx.exact else None
        x_in, w_r = ctx.saved_tensors if ctx.router else (None, None)
        dx = dw_r = None
        if ctx.needs_input_grad[0] and any(t is not None for t in (g_logits, g_xr, g_xg, g_xu, g_xq)):
            dx = _fan_in(g_xr, g_xg, g_xu, g_xq, g_logits, w_r, ctx.dims, ctx.x_dtype, ctx.exact)
        if ctx.router and ctx.needs_input_grad[1] and g_logits is not None:
            dw_r = _router_wgrad_lean(g_logits, x_in) if ctx.lean else g_logits.t().mm(x_in)
        return dx, dw_r, None, None, None, None


def moe_fan(xf, w_r, K: int, shared: bool, exact: bool, lean: bool = False):
    """MoEFan.apply -> ([logits,] xr[, xg, xu][, xq]); exact needs w_r None, the other modes w_r (see MoEFan)."""
    return MoEFan.apply(xf, w_r, K, shared, exact, lean)


def routed_experts_reference(x, idx, w, w_gate, w_up, w_down):
    """The same function in plain PyTorch, one expert at a time (host syncs; tests and the CPU fallback).
    Rounding points as in the kernels' forward: GEMM outputs, SiLU and h in the weights' dtype, the k-sum in
    fp32. Gradients come from autograd."""
    N, K = idx.shape
    E = w_gate.shape[0]
    xc = x.to(w_gate.dtype)
    flat = idx.reshape(-1)
    out = xc.new_zeros(N * K, w_down.shape[-1])
    for e in range(E):
        sel = torch.nonzero(flat == e).squeeze(1)
        if sel.numel() == 0:
            continue
        xe = xc.index_select(0, sel // K)
        h = F.silu(xe @ w_gate[e]) * (xe @ w_up[e])
        out = out.index_copy(0, sel, h @ w_down[e])
    y = (out.view(N, K, out.shape[-1]).float() * w.float().unsqueeze(-1)).sum(1)
    return y.to(xc.dtype)
