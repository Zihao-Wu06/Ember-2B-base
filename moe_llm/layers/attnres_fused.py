"""Block AttnRes with shared per-block backward (ModelConfig.attnres_impl="fused"; "compiled" = layers/attnres.py).

The op is the same as layers/attnres.py: for one op with pseudo-query q = g * w and sources v_i (per token, fp32)
    alpha_i = softmax_i( (q . v_i) * rsqrt(mean(v_i^2) + eps) ),   out = sum_i alpha_i v_i   (in the sources' dtype)
In Block AttnRes every op of block k reads the same completed block outputs b_0..b_k plus at most one partial sum of
its own (the running sum inside the block). What changes is how the work is organised:

* The completed block outputs live in one stacked buffer [n_blocks, B*T, d] (b_k is copied in when its block opens),
  so a kernel reads any number of them through one pointer.
* Forward, one kernel per op: an online softmax over the completed sources and the partial, one read of each
  source, fp32 accumulation, one rounding of the output. Saved for backward: the partial and q (references to
  tensors that exist anyway), no [tokens, d] intermediates.
* Backward of an op (when autograd reaches it): one kernel re-reads its sources and its upstream gradient and
  computes the gradient of its partial plus the partial's share of dq. For the completed sources it only writes
  three per-token scalars per source (alpha, ds*r, ds*s*r^2/d) and keeps a reference to the upstream gradient.
* Backward of a block (when autograd reaches the node that opened it, i.e. after every op of the block): one
  kernel reads every completed source and the upstream gradients of all ops of the block once and computes
      grad b_j += sum_c [alpha_cj g_c + (ds_cj r_j) q_c - (ds_cj s_cj r_j^2 / d) b_j]
  for all j at once, accumulated in place into a per-forward buffer (autograd's separate bf16 adds for the ~220
  extra uses of block outputs per micro-batch disappear), plus the completed sources' share of dq for every op.
  The gradient of b_k is complete after block k's own backward and is handed to autograd there.

Numerics: everything is computed in fp32 from the bf16 values; the output is rounded once (as before), the partial's
gradient once. A completed source's gradient from one block is summed over the block's ops in fp32 and added to the
running total with one rounding per block (before: every op's gradient rounded to bf16, then one bf16 add per use,
47 adds for the embedding; now 8 roundings in all). Summation orders differ from torch.compile's: rounding-level
differences, no less accurate against an fp64 reference (tests/test_attnres_fused.py).

Autograd structure, per block: _BlockStart(b_k, q_1..q_C) -> token; every op takes the token (no value: an edge
that makes autograd run the block's backward after all its ops). Ops get the completed sources through the block
state, not as autograd inputs, so the only gradient path into b_j is the _BlockStart of block j.
Works with torch.utils.checkpoint (whole layers or the FFN part: the op's forward is pure and deterministic, its
recompute saves the same tensors), the activation offloader (ops save only their partial and q), no_grad / eval,
and CPU: CUDA tensors and Triton's interpreter (TRITON_INTERPRET=1) run the Triton kernels, other CPU tensors a
plain-PyTorch version of the same three kernels (set_backend("torch"|"triton"|"auto")).
Deterministic: no atomics; per-program partial sums of dq are reduced in a fixed order. No host syncs (autotuning
syncs once per shape, on the first call; results are cached on disk).
"""
from __future__ import annotations

import os
import warnings

import torch

import triton
import triton.language as tl
from triton import knobs

_INTERPRET = bool(knobs.runtime.interpret)
_BACKEND = os.environ.get("MOE_ATTNRES_FUSED_BACKEND", "auto")  # auto | triton | torch
# per-row sums of a source in one multi-operand reduction (GPU); separate tl.sum calls in the CPU interpreter, where
# a custom-combine reduction is ~10x slower (same values up to fp32 summation order; tests run both)
_MERGE = os.environ.get("MOE_ATTNRES_MERGE_REDUCE", "0" if _INTERPRET else "1") == "1"
MAX_SLOTS = 9  # ops per block: 2 per layer + the final out_res -> attnres_layers_per_block <= 4


def set_merge_reduce(enabled: bool) -> None:
    global _MERGE
    _MERGE = bool(enabled)


def set_backend(name: str) -> None:
    """auto: Triton kernels for CUDA tensors (and CPU tensors under TRITON_INTERPRET=1), PyTorch otherwise."""
    global _BACKEND
    assert name in ("auto", "triton", "torch"), name
    _BACKEND = name


def describe(device_type: str) -> str:
    tri = _BACKEND != "torch" and (device_type == "cuda" or _INTERPRET)
    return "fused (" + ("Triton kernels" + (", interpreter" if device_type != "cuda" else "") if tri else
                        "PyTorch kernels") + ", one shared backward per block)"


def _use_triton(t: torch.Tensor) -> bool:
    if _BACKEND == "torch":
        return False
    ok = t.is_cuda or _INTERPRET
    if _BACKEND == "triton" and not ok:
        raise RuntimeError("attnres_fused: Triton backend needs CUDA tensors or TRITON_INTERPRET=1")
    return ok


# ---------------------------------------------------------------------------- autotuning
_FWD_CANDS = [(1, 4), (2, 4), (2, 8), (4, 4), (4, 8), (8, 8)]          # BLOCK_T, num_warps
_OP_CANDS = [(1, 4), (2, 4), (2, 8), (4, 8)]
_BLK_CANDS = [(4, 64, 4), (8, 64, 8), (4, 128, 4), (4, 128, 8), (8, 128, 8), (16, 64, 8),  # BLOCK_T, BLOCK_D, warps
              (4, 256, 8)]
_INTERP_T, _INTERP_D = 4, 32


def _row_cfgs(cands):
    if _INTERPRET:
        return [triton.Config({"BLOCK_T": _INTERP_T}, num_warps=4)]
    if os.environ.get("MOE_ATTNRES_TUNE", "") == "first":
        cands = cands[:1]
    return [triton.Config({"BLOCK_T": t}, num_warps=w) for t, w in cands]


def _blk_cfgs(cands):
    if _INTERPRET:
        return [triton.Config({"BLOCK_T": _INTERP_T, "BLOCK_D": _INTERP_D}, num_warps=4)]
    if os.environ.get("MOE_ATTNRES_TUNE", "") == "first":
        cands = cands[:1]
    return [triton.Config({"BLOCK_T": t, "BLOCK_D": d}, num_warps=w) for t, d, w in cands]


def _do_bench(fn, quantiles=(0.5, 0.2, 0.8)):
    """Autotuner timing: quantiles of 10 timed runs after 3 warm-up runs (the first compiles). Results are cached on
    disk (cache_results), so this runs once per shape."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001  a config that fails to compile is skipped, not fatal
        if type(e).__name__ not in ("OutOfResources", "PTXASError", "CompileTimeAssertionFailure"):
            warnings.warn(f"attnres_fused autotune: skipping a config: {type(e).__name__}: {e}")
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


_SM = None


def _programs(t: torch.Tensor) -> int:
    """Number of programs of the persistent kernels (fixed per device, so reductions keep their order)."""
    global _SM
    if not t.is_cuda:
        return 2
    if _SM is None:
        _SM = torch.cuda.get_device_properties(t.device).multi_processor_count
    return 4 * _SM


# ---------------------------------------------------------------------------- kernels
@triton.jit
def _round(x, dt: tl.constexpr, INTERP: tl.constexpr):
    """fp32 x rounded to dt, to nearest even (the GPU's cvt.rn). The CPU interpreter truncates fp32 -> bf16, so there
    the bits are rounded first (finite values; tests only)."""
    if INTERP and dt == tl.bfloat16:
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return b.to(tl.float32, bitcast=True).to(dt)
    else:
        return x.to(dt)


@triton.jit
def _add2(a0, b0, a1, b1):
    return a0 + a1, b0 + b1


@triton.jit
def _add3(a0, b0, c0, a1, b1, c1):
    return a0 + a1, b0 + b1, c0 + c1


@triton.jit
def _logit(v, q, eps, D: tl.constexpr, MERGE: tl.constexpr):
    """Rows of v [BLOCK_T, D_P2] (fp32, zero padded): logit (q . v) * r and r = rsqrt(mean(v^2) + eps). MERGE: both
    row sums in one reduction (one round of cross-warp exchange instead of two; slow in the CPU interpreter)."""
    if MERGE:
        ss, qv = tl.reduce((v * v, v * q[None, :]), 1, _add2)
    else:
        ss = tl.sum(v * v, axis=1)
        qv = tl.sum(v * q[None, :], axis=1)
    r = tl.rsqrt(ss / D + eps)
    return qv * r, r


@triton.jit
def _logit_dot(v, q, g, eps, D: tl.constexpr, MERGE: tl.constexpr):
    """_logit plus the row dot product with g."""
    if MERGE:
        ss, qv, gv = tl.reduce((v * v, v * q[None, :], v * g), 1, _add3)
    else:
        ss = tl.sum(v * v, axis=1)
        qv = tl.sum(v * q[None, :], axis=1)
        gv = tl.sum(v * g, axis=1)
    r = tl.rsqrt(ss / D + eps)
    return qv * r, r, gv


@triton.autotune(configs=_row_cfgs(_FWD_CANDS), key=["N", "D", "n_c", "HAS_P"], do_bench=_do_bench,
                 cache_results=True)
@triton.jit
def _fwd_kernel(src_ptr, p_ptr, q_ptr, out_ptr, N, n_c, stride_s, eps,
                D: tl.constexpr, D_P2: tl.constexpr, HAS_P: tl.constexpr, INTERP: tl.constexpr, MERGE: tl.constexpr,
                BLOCK_T: tl.constexpr):
    """out[t] = sum_i alpha_i v_i over the n_c stacked sources (src_ptr + j * stride_s) and the partial p (HAS_P):
    an online softmax, each source read once, fp32 accumulation, one rounding."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, D_P2)
    msk = (rows < N)[:, None] & (cols < D)[None, :]
    offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
    q = tl.load(q_ptr + cols, mask=cols < D, other=0.0)
    m = tl.full([BLOCK_T], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_T], tl.float32)
    acc = tl.zeros([BLOCK_T, D_P2], tl.float32)
    sp = src_ptr
    for j in range(n_c):
        v = tl.load(sp + offs, mask=msk, other=0.0).to(tl.float32)
        s, r = _logit(v, q, eps, D, MERGE)
        m_new = tl.maximum(m, s)
        a = tl.exp(m - m_new)
        e = tl.exp(s - m_new)
        acc = acc * a[:, None] + e[:, None] * v
        l = l * a + e
        m = m_new
        sp += stride_s
    if HAS_P:
        v = tl.load(p_ptr + offs, mask=msk, other=0.0).to(tl.float32)
        s, r = _logit(v, q, eps, D, MERGE)
        m_new = tl.maximum(m, s)
        a = tl.exp(m - m_new)
        e = tl.exp(s - m_new)
        acc = acc * a[:, None] + e[:, None] * v
        l = l * a + e
    out = acc / l[:, None]
    tl.store(out_ptr + offs, _round(out, out_ptr.dtype.element_ty, INTERP), mask=msk)


@triton.autotune(configs=_row_cfgs(_OP_CANDS), key=["N", "D", "n_c", "HAS_P"], do_bench=_do_bench,
                 cache_results=True)
@triton.jit
def _bwd_op_kernel(src_ptr, p_ptr, g_ptr, q_ptr, dp_ptr, coef_ptr, dq_ptr, N, n_c, stride_s, stride_cf, eps,
                   D: tl.constexpr, D_P2: tl.constexpr, K_P2: tl.constexpr, HAS_P: tl.constexpr,
                   INTERP: tl.constexpr, MERGE: tl.constexpr, BLOCK_T: tl.constexpr):
    """Backward of one op with upstream gradient g. Recomputes the logits and the softmax, then
    - for the partial p (HAS_P): dp = alpha_p g + ds_p r_p q - ds_p s_p r_p^2 / d p  and this program's share of
      dq = sum_t ds_p r_p p (dq_ptr row pid);
    - for every completed source j: alpha_j, U_j = ds_j r_j, Z_j = ds_j s_j r_j^2 / d into the coefficient planes
      (coef_ptr + {0, 1, 2} * stride_cf, each [n_c, N]), used by the block's backward.
    ds_i = alpha_i (g . v_i - sum_k alpha_k g . v_k). Persistent: program pid handles tiles pid, pid + nprog, ..."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_iter = tl.cdiv(tl.cdiv(N, BLOCK_T) - pid, nprog)
    cols = tl.arange(0, D_P2)
    cm = cols < D
    kk = tl.arange(0, K_P2)
    valid = kk < n_c
    q = tl.load(q_ptr + cols, mask=cm, other=0.0)
    dq = tl.zeros([D_P2], tl.float32)
    for it in range(n_iter):
        rows = (pid + it * nprog) * BLOCK_T + tl.arange(0, BLOCK_T)
        rm = rows < N
        msk = rm[:, None] & cm[None, :]
        offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
        g = tl.load(g_ptr + offs, mask=msk, other=0.0).to(tl.float32)
        S = tl.zeros([K_P2, BLOCK_T], tl.float32)
        R = tl.zeros([K_P2, BLOCK_T], tl.float32)
        DB = tl.zeros([K_P2, BLOCK_T], tl.float32)
        sp = src_ptr
        for j in range(n_c):
            v = tl.load(sp + offs, mask=msk, other=0.0).to(tl.float32)
            s, r, db = _logit_dot(v, q, g, eps, D, MERGE)
            sel = (kk == j)[:, None]
            S = tl.where(sel, s[None, :], S)
            R = tl.where(sel, r[None, :], R)
            DB = tl.where(sel, db[None, :], DB)
            sp += stride_s
        Sm = tl.where(valid[:, None], S, float("-inf"))
        mx = tl.max(Sm, axis=0)
        if HAS_P:
            pv = tl.load(p_ptr + offs, mask=msk, other=0.0).to(tl.float32)
            s_p, r_p, db_p = _logit_dot(pv, q, g, eps, D, MERGE)
            mx = tl.maximum(mx, s_p)
        E = tl.exp(Sm - mx[None, :])
        lsum = tl.sum(E, axis=0)
        if HAS_P:
            e_p = tl.exp(s_p - mx)
            lsum = lsum + e_p
        A = E / lsum[None, :]
        dsum = tl.sum(A * DB, axis=0)
        if HAS_P:
            a_p = e_p / lsum
            dsum = dsum + a_p * db_p
        U = A * (DB - dsum[None, :]) * R
        Z = U * S * R / D
        cmsk = valid[:, None] & rm[None, :]
        co = kk[:, None] * N + rows[None, :]
        tl.store(coef_ptr + co, A, mask=cmsk)
        tl.store(coef_ptr + stride_cf + co, U, mask=cmsk)
        tl.store(coef_ptr + 2 * stride_cf + co, Z, mask=cmsk)
        if HAS_P:
            u_p = a_p * (db_p - dsum) * r_p
            dpv = a_p[:, None] * g + u_p[:, None] * q[None, :] - (u_p * s_p * r_p / D)[:, None] * pv
            tl.store(dp_ptr + offs, _round(dpv, dp_ptr.dtype.element_ty, INTERP), mask=msk)
            dq += tl.sum(u_p[:, None] * pv, axis=0)
    if HAS_P:
        tl.store(dq_ptr + pid * D + cols, dq, mask=cm)


@triton.jit
def _slot(G, zs, dqc, cp, stride_cf, rows, rm, d, qc, v):
    """One op's share for a completed source v: G += alpha g + U q, zs += Z, dq += U v (dq kept per tile row)."""
    a = tl.load(cp + rows, mask=rm, other=0.0)
    u = tl.load(cp + stride_cf + rows, mask=rm, other=0.0)
    z = tl.load(cp + 2 * stride_cf + rows, mask=rm, other=0.0)
    G = G + a[:, None] * d + u[:, None] * qc[None, :]
    return G, zs + z, dqc + u[:, None] * v


@triton.autotune(configs=_blk_cfgs(_BLK_CANDS), key=["N", "D", "n_c", "HAS", "FIRST"], do_bench=_do_bench,
                 cache_results=True)  # acc is updated in place: _blk_bwd tunes a new shape on a scratch copy
@triton.jit
def _bwd_block_kernel(src_ptr, coef_ptr, q_ptr, acc_ptr, dq_ptr, g0, g1, g2, g3, g4, g5, g6, g7, g8,
                      N, n_c, stride_s, stride_cf,
                      D: tl.constexpr, C_P2: tl.constexpr, HAS: tl.constexpr, FIRST: tl.constexpr,
                      INTERP: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    """Backward of a block for its completed sources j < n_c: with the ops' coefficients (coef_ptr [3, slots, n_c, N])
    and upstream gradients g_c (slot c of the bit mask HAS),
        acc[j] (+)= sum_c alpha_cj g_c + sum_c U_cj q_c - (sum_c Z_cj) b_j     (FIRST: acc is written, not read)
    and per-program partial sums dq[pid, c] = sum_t sum_j U_cj b_j. Grid (token programs, d chunks); each program
    keeps its d chunk and walks tiles pid, pid + nprog, ...: every input byte is read once. The slots are unrolled
    (one variable per op: Triton has no arrays of tensors)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    pid_d = tl.program_id(1)
    n_iter = tl.cdiv(tl.cdiv(N, BLOCK_T) - pid, nprog)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    cm = cols < D
    slot_stride = n_c * N
    if (HAS >> 0) & 1:
        q0 = tl.load(q_ptr + 0 * D + cols, mask=cm, other=0.0)
        dq0 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 1) & 1:
        q1 = tl.load(q_ptr + 1 * D + cols, mask=cm, other=0.0)
        dq1 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 2) & 1:
        q2 = tl.load(q_ptr + 2 * D + cols, mask=cm, other=0.0)
        dq2 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 3) & 1:
        q3 = tl.load(q_ptr + 3 * D + cols, mask=cm, other=0.0)
        dq3 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 4) & 1:
        q4 = tl.load(q_ptr + 4 * D + cols, mask=cm, other=0.0)
        dq4 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 5) & 1:
        q5 = tl.load(q_ptr + 5 * D + cols, mask=cm, other=0.0)
        dq5 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 6) & 1:
        q6 = tl.load(q_ptr + 6 * D + cols, mask=cm, other=0.0)
        dq6 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 7) & 1:
        q7 = tl.load(q_ptr + 7 * D + cols, mask=cm, other=0.0)
        dq7 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    if (HAS >> 8) & 1:
        q8 = tl.load(q_ptr + 8 * D + cols, mask=cm, other=0.0)
        dq8 = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
    for it in range(n_iter):
        rows = (pid + it * nprog) * BLOCK_T + tl.arange(0, BLOCK_T)
        rm = rows < N
        msk = rm[:, None] & cm[None, :]
        offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
        if (HAS >> 0) & 1:
            d0 = tl.load(g0 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 1) & 1:
            d1 = tl.load(g1 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 2) & 1:
            d2 = tl.load(g2 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 3) & 1:
            d3 = tl.load(g3 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 4) & 1:
            d4 = tl.load(g4 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 5) & 1:
            d5 = tl.load(g5 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 6) & 1:
            d6 = tl.load(g6 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 7) & 1:
            d7 = tl.load(g7 + offs, mask=msk, other=0.0).to(tl.float32)
        if (HAS >> 8) & 1:
            d8 = tl.load(g8 + offs, mask=msk, other=0.0).to(tl.float32)
        sp = src_ptr
        ap = acc_ptr
        cp = coef_ptr
        for j in range(n_c):
            v = tl.load(sp + offs, mask=msk, other=0.0).to(tl.float32)
            G = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)
            zs = tl.zeros([BLOCK_T], tl.float32)
            if (HAS >> 0) & 1:
                G, zs, dq0 = _slot(G, zs, dq0, cp + 0 * slot_stride, stride_cf, rows, rm, d0, q0, v)
            if (HAS >> 1) & 1:
                G, zs, dq1 = _slot(G, zs, dq1, cp + 1 * slot_stride, stride_cf, rows, rm, d1, q1, v)
            if (HAS >> 2) & 1:
                G, zs, dq2 = _slot(G, zs, dq2, cp + 2 * slot_stride, stride_cf, rows, rm, d2, q2, v)
            if (HAS >> 3) & 1:
                G, zs, dq3 = _slot(G, zs, dq3, cp + 3 * slot_stride, stride_cf, rows, rm, d3, q3, v)
            if (HAS >> 4) & 1:
                G, zs, dq4 = _slot(G, zs, dq4, cp + 4 * slot_stride, stride_cf, rows, rm, d4, q4, v)
            if (HAS >> 5) & 1:
                G, zs, dq5 = _slot(G, zs, dq5, cp + 5 * slot_stride, stride_cf, rows, rm, d5, q5, v)
            if (HAS >> 6) & 1:
                G, zs, dq6 = _slot(G, zs, dq6, cp + 6 * slot_stride, stride_cf, rows, rm, d6, q6, v)
            if (HAS >> 7) & 1:
                G, zs, dq7 = _slot(G, zs, dq7, cp + 7 * slot_stride, stride_cf, rows, rm, d7, q7, v)
            if (HAS >> 8) & 1:
                G, zs, dq8 = _slot(G, zs, dq8, cp + 8 * slot_stride, stride_cf, rows, rm, d8, q8, v)
            G = G - zs[:, None] * v
            if FIRST:
                new = G
            else:
                new = tl.load(ap + offs, mask=msk, other=0.0).to(tl.float32) + G
            tl.store(ap + offs, _round(new, acc_ptr.dtype.element_ty, INTERP), mask=msk)
            sp += stride_s
            ap += stride_s
            cp += N
    if (HAS >> 0) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 0) * D + cols, tl.sum(dq0, axis=0), mask=cm)
    if (HAS >> 1) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 1) * D + cols, tl.sum(dq1, axis=0), mask=cm)
    if (HAS >> 2) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 2) * D + cols, tl.sum(dq2, axis=0), mask=cm)
    if (HAS >> 3) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 3) * D + cols, tl.sum(dq3, axis=0), mask=cm)
    if (HAS >> 4) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 4) * D + cols, tl.sum(dq4, axis=0), mask=cm)
    if (HAS >> 5) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 5) * D + cols, tl.sum(dq5, axis=0), mask=cm)
    if (HAS >> 6) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 6) * D + cols, tl.sum(dq6, axis=0), mask=cm)
    if (HAS >> 7) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 7) * D + cols, tl.sum(dq7, axis=0), mask=cm)
    if (HAS >> 8) & 1:
        tl.store(dq_ptr + (pid * C_P2 + 8) * D + cols, tl.sum(dq8, axis=0), mask=cm)


# ---------------------------------------------------------------------------- launchers (Triton or PyTorch)
def _fwd(stack, n_c, p, q, eps):
    """out [N, d] (stack dtype) of one op over stack[:n_c] and the partial p [N, d] (or None)."""
    N, D = stack.shape[1], stack.shape[2]
    out = torch.empty(N, D, device=stack.device, dtype=stack.dtype)
    if _use_triton(stack):
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_T"]),)  # noqa: E731
        _fwd_kernel[grid](stack, stack if p is None else p, q, out, N, n_c, stack.stride(0), eps,
                          D=D, D_P2=triton.next_power_of_2(D), HAS_P=p is not None, INTERP=_INTERPRET, MERGE=_MERGE)
    else:
        with torch.autocast(stack.device.type, enabled=False):
            srcs = [stack[j] for j in range(n_c)] + ([p] if p is not None else [])
            logits = []
            for v in srcs:
                vf = v.float()
                r = torch.rsqrt(vf.pow(2).sum(-1, keepdim=True) / D + eps)
                logits.append((vf * q).sum(-1, keepdim=True) * r)
            a = torch.softmax(torch.cat(logits, -1), dim=-1)
            o = a[:, 0:1] * srcs[0].float()
            for i in range(1, len(srcs)):
                o = o + a[:, i:i + 1] * srcs[i].float()
            out.copy_(o)
    return out


def _op_bwd(stack, n_c, p, g, q, eps, coef_slot, stride_cf):
    """Backward of one op: returns (dp [N, d] in p's dtype or None, dq [d] fp32 or None) and writes the op's
    coefficient planes (coef_slot: [3, n_c, N] view, planes stride_cf apart)."""
    N, D = stack.shape[1], stack.shape[2]
    dp = torch.empty_like(p) if p is not None else None
    if _use_triton(stack):
        P = min(N, _programs(stack))
        dqp = torch.empty(P, D, device=stack.device, dtype=torch.float32) if p is not None else None
        _bwd_op_kernel[(P,)](stack, stack if p is None else p, g, q, stack if p is None else dp, coef_slot,
                             stack if p is None else dqp, N, n_c, stack.stride(0), stride_cf, eps,
                             D=D, D_P2=triton.next_power_of_2(D), K_P2=max(2, triton.next_power_of_2(n_c)),
                             HAS_P=p is not None, INTERP=_INTERPRET, MERGE=_MERGE)
        return dp, (dqp.sum(0) if p is not None else None)
    with torch.autocast(stack.device.type, enabled=False):
        V = stack[:n_c].float()                                   # [n_c, N, d]
        gf = g.float()
        R = torch.rsqrt(V.pow(2).sum(-1) / D + eps)               # [n_c, N]
        S = (V * q).sum(-1) * R
        DB = (V * gf).sum(-1)
        if p is not None:
            pf = p.float()
            r_p = torch.rsqrt(pf.pow(2).sum(-1) / D + eps)
            s_p = (pf * q).sum(-1) * r_p
            S_all, DB_all = torch.cat([S, s_p[None]]), torch.cat([DB, (pf * gf).sum(-1)[None]])
        else:
            S_all, DB_all = S, DB
        A_all = torch.softmax(S_all, dim=0)
        dsum = (A_all * DB_all).sum(0)
        U_all = A_all * (DB_all - dsum)                           # ds
        U = U_all[:n_c] * R
        coef_slot[0].copy_(A_all[:n_c])
        coef_slot[1].copy_(U)
        coef_slot[2].copy_(U * S * R / D)
        if p is None:
            return None, None
        a_p, u_p = A_all[n_c], U_all[n_c] * r_p
        dp.copy_(a_p[:, None] * gf + u_p[:, None] * q - (u_p * s_p * r_p / D)[:, None] * pf)
        return dp, (u_p[:, None] * pf).sum(0)


_BLK_TUNED: set = set()  # shapes of _bwd_block_kernel calls the autotuner has seen in this process


def _blk_bwd(stack, n_c, coef, qbuf, gs, acc, first):
    """Backward of a block for its completed sources: acc[:n_c] (+)= the gradients from the ops with upstream
    gradients gs (dict slot -> [N, d]); returns dq [slots_p2, d] fp32 (zero rows for missing slots)."""
    N, D = stack.shape[1], stack.shape[2]
    C_P2 = qbuf.shape[0]
    if _use_triton(stack):
        P = min(N, max(1, _programs(stack) // 8))
        dqp = torch.zeros(P, C_P2, D, device=stack.device, dtype=torch.float32)
        ptrs = [gs.get(c, stack) for c in range(MAX_SLOTS)]
        has = sum(1 << c for c in gs)
        grid = lambda meta: (P, triton.cdiv(D, meta["BLOCK_D"]))  # noqa: E731

        def launch(a):
            _bwd_block_kernel[grid](stack, coef, qbuf, a, dqp, *ptrs, N, n_c, stack.stride(0), coef.stride(0),
                                    D=D, C_P2=C_P2, HAS=has, FIRST=bool(first), INTERP=_INTERPRET)

        key = (N, D, n_c, has, bool(first), stack.dtype, acc.dtype, coef.shape[1])
        if stack.is_cuda and not first and key not in _BLK_TUNED and len(_bwd_block_kernel.configs) > 1:
            launch(acc[:n_c].clone())  # first call of this shape: the autotuner reruns the kernel (in-place update)
        _BLK_TUNED.add(key)
        launch(acc)
        return dqp.sum(0)
    dq = torch.zeros(C_P2, D, device=stack.device, dtype=torch.float32)
    with torch.autocast(stack.device.type, enabled=False):
        for j in range(n_c):
            v = stack[j].float()
            G = torch.zeros(N, D, device=stack.device, dtype=torch.float32)
            zs = torch.zeros(N, device=stack.device, dtype=torch.float32)
            for c in sorted(gs):
                a, u, z = coef[0, c, j], coef[1, c, j], coef[2, c, j]
                G = G + a[:, None] * gs[c].float() + u[:, None] * qbuf[c]
                zs = zs + z
                dq[c] += (u[:, None] * v).sum(0)
            G = G - zs[:, None] * v
            acc[j].copy_(G if first else acc[j].float() + G)
    return dq


# ---------------------------------------------------------------------------- state and autograd
class FusedAttnRes:
    """One forward pass (one micro-batch): the stacked completed block outputs and, in backward, the accumulated
    gradients of the stacked block outputs (same dtype; one rounding per block)."""

    def __init__(self, first: torch.Tensor, n_blocks: int, n_slots: int):
        assert n_slots <= MAX_SLOTS, f"attnres_fused supports at most {MAX_SLOTS} ops per block"
        self.shape, self.D = first.shape, first.shape[-1]
        self.N = first.numel() // self.D
        self.n_slots = n_slots
        self.stack = torch.empty(n_blocks, self.N, self.D, device=first.device, dtype=first.dtype)
        self.acc = None
        self.acc_ready = [False] * n_blocks

    def views(self):
        return [self.stack[j] for j in range(self.stack.shape[0])]

    def open_block(self, k: int, b: torch.Tensor, ops: dict):
        """Block k starts with completed source b (= b_k). ops: slot -> DepthAttn module of every op that will run
        in this block. Returns (block state, token, {slot: q}); q = g * w (differentiable)."""
        if b.dtype != self.stack.dtype and torch.promote_types(b.dtype, self.stack.dtype) != self.stack.dtype:
            raise RuntimeError(f"attnres_fused: block output in {b.dtype} after {self.stack.dtype} ones (would round)")
        qs = {s: (m.g * m.w).float() for s, m in sorted(ops.items())}
        blk = _Block(self, k, {s: q.detach().contiguous() for s, q in qs.items()})
        train = torch.is_grad_enabled() and (b.requires_grad or any(q.requires_grad for q in qs.values()))
        if train:
            token = _BlockStart.apply(blk, b, *qs.values())
        else:
            with torch.no_grad():
                self.stack[k].copy_(b.reshape(self.N, self.D))
            token = None
        return blk, token, qs


class _Block:
    __slots__ = ("fs", "k", "n_c", "q", "gs", "coef")

    def __init__(self, fs: FusedAttnRes, k: int, q: dict):
        self.fs, self.k, self.n_c, self.q = fs, k, k + 1, q
        self.gs: dict = {}   # slot -> upstream gradient [N, d], stashed by the op's backward
        self.coef = None     # [3, n_slots, n_c, N] fp32 per-token coefficients of the completed sources


def _block_backward(blk: _Block) -> dict:
    fs = blk.fs
    gs, blk.gs = blk.gs, {}
    if not gs:
        return {}
    n_c = blk.n_c
    if fs.acc is None:
        fs.acc = torch.empty_like(fs.stack)
    ready = fs.acc_ready[:n_c]
    if any(ready) and not all(ready):
        for j in range(n_c):
            if not ready[j]:
                fs.acc[j].zero_()
        ready = [True] * n_c
    first = not any(ready)
    c_p2 = triton.next_power_of_2(fs.n_slots)
    zero = None
    qrows = []
    for c in range(c_p2):
        if c in gs:
            qrows.append(blk.q[c])
        else:
            zero = zero if zero is not None else torch.zeros(fs.D, device=fs.stack.device, dtype=torch.float32)
            qrows.append(zero)
    dq = _blk_bwd(fs.stack, n_c, blk.coef, torch.stack(qrows), gs, fs.acc, first)
    blk.coef = None
    for j in range(n_c):
        fs.acc_ready[j] = True
    return {c: dq[c] for c in gs}


class _BlockStart(torch.autograd.Function):
    """Copies b_k into the stack; returns an empty token the block's ops take as input. Backward (after all ops of
    the block): the block's shared backward kernel; returns the now complete gradient of b_k and the completed
    sources' share of dq for every op."""

    @staticmethod
    def forward(ctx, blk, b, *qs):
        fs = blk.fs
        fs.stack[blk.k].copy_(b.reshape(fs.N, fs.D))
        ctx.blk, ctx.b_meta = blk, (b.shape, b.dtype)
        return b.new_empty(0)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, _token_grad):
        blk = ctx.blk
        dqs = _block_backward(blk)
        fs = blk.fs
        gb = None
        if fs.acc is not None and fs.acc_ready[blk.k]:
            shape, dtype = ctx.b_meta
            gb = fs.acc[blk.k].view(shape)
            gb = gb if gb.dtype == dtype else gb.to(dtype)
        if blk.k == 0:  # the last block in backward order: nothing reads the buffers any more
            fs.acc, fs.stack = None, None
        return (None, gb, *[dqs.get(s) for s in blk.q])


class _Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token, p, q, blk, slot, eps):
        fs = blk.fs
        p2 = None if p is None else p.reshape(fs.N, fs.D).contiguous()
        out = _fwd(fs.stack, blk.n_c, p2, q, eps)
        ctx.blk, ctx.slot, ctx.eps = blk, slot, eps
        ctx.save_for_backward(p2, q)
        ctx.p_shape = None if p is None else p.shape
        return out.view(fs.shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        blk, slot = ctx.blk, ctx.slot
        p2, q = ctx.saved_tensors
        fs = blk.fs
        if fs.stack is None:
            raise RuntimeError("attnres_fused: backward through the same graph twice is not supported")
        g2 = g.reshape(fs.N, fs.D)
        if not g2.is_contiguous():
            g2 = g2.contiguous()
        if blk.coef is None:
            blk.coef = torch.empty(3, fs.n_slots, blk.n_c, fs.N, device=g2.device, dtype=torch.float32)
        dp, dq = _op_bwd(fs.stack, blk.n_c, p2, g2, q, ctx.eps, blk.coef[:, slot], blk.coef.stride(0))
        blk.gs[slot] = g2
        return None, (None if dp is None else dp.view(ctx.p_shape)), dq, None, None, None


def depth_attn(blk: _Block, token, p, q, slot: int, eps: float):
    """One AttnRes op of block blk over its completed sources and the partial p ([B, T, d] or None); q = g * w from
    FusedAttnRes.open_block, slot = its position in the block (2 * layer offset + {0: before the mixer, 1: before
    the FFN}, 2 * layers_per_block: the final op)."""
    if torch.is_grad_enabled() and ((token is not None and token.requires_grad) or q.requires_grad
                                    or (p is not None and p.requires_grad)):
        if token is None:
            raise RuntimeError("attnres_fused: block opened without gradients but its op needs them")
        return _Op.apply(token, p, q, blk, slot, eps)
    fs = blk.fs
    p2 = None if p is None else p.reshape(fs.N, fs.D).contiguous()
    return _fwd(fs.stack, blk.n_c, p2, q, eps).view(fs.shape)
