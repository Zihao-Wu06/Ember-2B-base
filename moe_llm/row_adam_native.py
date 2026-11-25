"""Lazy Adam for the offloaded Engram tables as a small C kernel called through ctypes.

Same arithmetic as optim.row_adam_cpu (the torch.compile'd _row_adam_fused), bit for bit: every scalar is rounded to
float32 first and each element goes through the same float32 operations in the same order
    m' = m * b1 + g * (1 - b1)            v' = v * b2 + (g * g) * (1 - b2)
    w' = w - ((m' / bc1) * lr) / (sqrt(v' / bc2) + eps)            g' = 0
(this is what inductor generates for _row_adam_fused; the tests compare the two kernels bitwise). The difference is
where it runs: inductor's kernel holds the Python GIL for most of its run time (~55% measured), which stalls the main
thread while it queues the GPU optimizer work; ctypes releases the GIL for the whole call, so the table update really
overlaps that work (OptimizerBundle engram_early). Rows are independent, so the touched rows are split over a few
threads; the result does not depend on the split.

Built once with the system C compiler into ~/.cache/moe_llm (no fast-math, no FMA contraction); available() is False
when that fails and the callers keep the torch.compile kernel.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

_SRC = r"""
#include <stdint.h>
#include <math.h>

void row_adam(float *restrict w, float *restrict g, float *restrict m, float *restrict v, const int64_t *ids,
              int64_t n, int64_t dim, float lr, float b1, float omb1, float b2, float omb2, float eps, float bc1,
              float bc2)
{
    for (int64_t i = 0; i < n; i++) {
        const int64_t r = ids[i] * dim;
        float *restrict wr = w + r, *restrict gr = g + r, *restrict mr = m + r, *restrict vr = v + r;
        for (int64_t j = 0; j < dim; j++) {
            const float gi = gr[j];
            const float mi = mr[j] * b1 + gi * omb1;
            const float vi = vr[j] * b2 + (gi * gi) * omb2;
            mr[j] = mi;
            vr[j] = vi;
            wr[j] = wr[j] - ((mi / bc1) * lr) / (sqrtf(vi / bc2) + eps);
            gr[j] = 0.0f;
        }
    }
}
"""
# IEEE semantics throughout: no -ffast-math, no contraction into FMA; sqrtf inlined (correctly rounded)
_FLAGS = ["-O3", "-march=native", "-fPIC", "-shared", "-ffp-contract=off", "-fno-math-errno", "-fno-trapping-math"]
_lock = threading.Lock()
_lib = None
_err = None
_pool = None


def _build():
    key = hashlib.sha256((_SRC + " ".join(_FLAGS)).encode()).hexdigest()[:16]
    d = os.path.join(os.path.expanduser("~"), ".cache", "moe_llm")
    so = os.path.join(d, f"row_adam_{key}.so")
    if not os.path.exists(so):
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, f"row_adam_{key}.c")
        with open(src, "w") as fh:
            fh.write(_SRC)
        tmp = f"{so}.{os.getpid()}.tmp"
        subprocess.run([os.environ.get("CC", "gcc"), *_FLAGS, "-o", tmp, src, "-lm"], check=True, capture_output=True,
                       timeout=120)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)   # CDLL (not PyDLL): the GIL is released while the function runs
    f = ctypes.c_float
    lib.row_adam.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64, ctypes.c_int64, f, f, f, f, f, f, f, f]
    lib.row_adam.restype = None
    return lib


def available() -> bool:
    global _lib, _err
    with _lock:
        if _lib is None and _err is None:
            try:
                _lib = _build()
            except Exception as e:  # no compiler, read-only home, ...
                _err = e
        return _lib is not None


def why_unavailable() -> str:
    return "" if _err is None else f"{type(_err).__name__}: {_err}"


def row_adam_native(weight, grad, m, v, ids, lr, betas, eps, step, threads: int | None = None) -> None:
    """In place on the rows `ids` (distinct, int64) of the CPU float32 tables; also zeroes grad[ids]. The values equal
    optim.row_adam_cpu's bit for bit."""
    global _pool
    n = ids.numel()
    if n == 0:
        return
    if not available():
        raise RuntimeError(f"native row Adam is not available ({why_unavailable()})")
    rows, dim = weight.shape
    for t in (weight, grad, m, v):
        if not (t.dtype == torch.float32 and t.device.type == "cpu" and t.is_contiguous() and t.shape == (rows, dim)):
            raise ValueError("row_adam_native needs contiguous float32 CPU tables of one shape")
    if not (ids.dtype == torch.int64 and ids.dim() == 1 and ids.is_contiguous() and ids.device.type == "cpu"):
        raise ValueError("row_adam_native needs a contiguous 1-D int64 CPU index")
    if int(ids.min()) < 0 or int(ids.max()) >= rows:
        raise IndexError("row_adam_native: row index out of range")
    b1, b2 = betas
    scal = (float(lr), b1, 1 - b1, b2, 1 - b2, eps, 1 - b1 ** step, 1 - b2 ** step)  # each rounded to float32 by ctypes
    ptrs = [t.data_ptr() for t in (weight, grad, m, v)]
    base = ids.data_ptr()
    k = max(1, min(threads or torch.get_num_threads(), n // 4096 or 1))
    bounds = [n * i // k for i in range(k + 1)]

    def part(i):
        _lib.row_adam(*ptrs, base + 8 * bounds[i], bounds[i + 1] - bounds[i], dim, *scal)
    if k == 1:
        part(0)
        return
    with _lock:
        if _pool is None or _pool._max_workers < k - 1:
            _pool = ThreadPoolExecutor(max_workers=max(k - 1, 15), thread_name_prefix="engram_adam")
        pool = _pool
    futs = [pool.submit(part, i) for i in range(1, k)]
    part(0)
    for f in futs:
        f.result()
