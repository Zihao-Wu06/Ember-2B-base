#!/usr/bin/env python
"""LM head + cross-entropy at main-run sizes on the GPU: ce_impl "chunked" (current) vs "fused" vs "fused_bf16".

  python scripts/bench_ce.py              # fwd+bwd time and peak memory per micro-batch, chunk 1024 .. 8192
  python scripts/bench_ce.py --sweep      # + the row kernel alone for each (BLOCK_V, num_warps) in SWEEP
  python scripts/bench_ce.py --profile    # + per-kernel CUDA time of one fwd+bwd per path (torch.profiler)
  python scripts/bench_ce.py --check      # + accuracy of each path against fp64 (fp64 GEMMs: ~10 s)

N = 8192 tokens per micro-batch (micro-bsz 2 x seq 4096), d 1024, V 65536, z-loss 1e-4, bf16 hidden states and the
fp32 head weight as in the main run, W.grad accumulating as in micro-batches 2..64. Both paths compute the gradients
in the forward; the backward only scales them. "per step" = x64 micro-batches (global batch 524288 tokens).
Ends with one JSON line (results) for the record. Needs an idle GPU: run it in a pause of the main run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm import loss_fused as lf  # noqa: E402
from moe_llm.loss import linear_cross_entropy  # noqa: E402

# (BLOCK_V, num_warps) candidates of the row kernel that compile without register spills on sm89
# (scripts/compile_ce_triton.py); 32768 x 32 spills.
SWEEP = [(2048, 4), (4096, 8), (4096, 16), (8192, 16), (8192, 32), (16384, 16), (16384, 32)]
MICRO_BATCHES = 64


def make_inputs(N=8192, d=1024, V=65536, seed=0, logit_std=3.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(N, d, device="cuda", generator=gen)
    h = (h / h.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
    W = torch.randn(V, d, device="cuda", generator=gen) * (logit_std / d ** 0.5)
    t = torch.randint(0, V, (N,), device="cuda", generator=gen)
    return h, W, t


def _events(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps


def time_loss(h, W, t, impl, chunk, reps=10, warmup=3, z=1e-4):
    """ms per forward + backward of the loss (W.grad accumulating, as in micro-batches 2..64)."""
    hh, WW = h.detach().clone().requires_grad_(), W.detach().clone().requires_grad_()
    WW.grad = torch.zeros_like(WW)

    def step():
        hh.grad = None
        loss, _ = linear_cross_entropy(hh, WW, t, z_coef=z, chunk=chunk, impl=impl)
        loss.backward()

    return _events(step, reps, warmup)


def peak_gib(h, W, t, impl, chunk, z=1e-4):
    """Peak memory of one loss fwd + bwd above what was allocated before (h, W, W.grad)."""
    hh, WW = h.detach().clone().requires_grad_(), W.detach().clone().requires_grad_()
    WW.grad = torch.zeros_like(WW)
    linear_cross_entropy(hh, WW, t, z_coef=z, chunk=chunk, impl=impl)[0].backward()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    linear_cross_entropy(hh, WW, t, z_coef=z, chunk=chunk, impl=impl)[0].backward()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def kernel_sweep(rows, V, configs, ldt, reps=20):
    """The row kernel alone on [rows, V] logits of dtype ldt (bf16 g; in place for bf16 logits): ms, and bandwidth
    counting one read of the logits + the g write (what it achieves if pass 2 hits L2) and two reads + the write."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(rows, V, device="cuda", generator=gen) * 3).to(ldt)
    tgt = torch.randint(0, V, (rows,), device="cuda", generator=gen)
    scale = torch.full((rows,), 1.0 / 8192, device="cuda")
    lse, xt = torch.empty(rows, device="cuda"), torch.empty(rows, device="cuda")
    # bf16: g overwrites the logits, so later repetitions run on gradient values (finite: same work, same time)
    g = x if ldt == torch.bfloat16 else torch.empty(rows, V, device="cuda", dtype=torch.bfloat16)
    one = rows * V * (x.element_size() + 2)
    two = rows * V * (2 * x.element_size() + 2)
    out = []
    try:
        for c in configs:
            lf.set_config(*c)
            ms = _events(lambda: lf.ce_rows(x, tgt, scale, 1e-4, g, lse, xt), reps, 3)
            out.append({"cfg": list(c), "ms": round(ms, 3), "GBps_1read": round(one / ms / 1e6),
                        "GBps_2reads": round(two / ms / 1e6)})
    finally:
        lf.reset_config()
    return out


def profile(h, W, t, impl, chunk, rows=12):
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as prof_ctx
    hh, WW = h.detach().clone().requires_grad_(), W.detach().clone().requires_grad_()
    WW.grad = torch.zeros_like(WW)
    for _ in range(2):
        linear_cross_entropy(hh, WW, t, z_coef=1e-4, chunk=chunk, impl=impl)[0].backward()
    torch.cuda.synchronize()
    with prof_ctx(activities=[ProfilerActivity.CUDA]) as p:
        linear_cross_entropy(hh, WW, t, z_coef=1e-4, chunk=chunk, impl=impl)[0].backward()
        torch.cuda.synchronize()
    ka = [e for e in p.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in ka)
    print(f"\n{impl} chunk {chunk}: {total / 1e3:.2f} ms CUDA time per micro-batch")
    for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:rows]:
        print(f"  {e.self_device_time_total / 1e3:7.3f} ms  x{e.count:<4d} {e.key[:100]}")


def reference_fp64(h, W, t, z, chunk=1024):
    h64, W64 = h.double(), W.to(torch.bfloat16).double()
    n = float(h.shape[0])
    grad_h, grad_W = torch.empty_like(h64), torch.zeros_like(W64)
    ce_sum = z_sum = 0.0
    for s in range(0, h.shape[0], chunk):
        e = min(s + chunk, h.shape[0])
        lg = h64[s:e] @ W64.t()
        lse = torch.logsumexp(lg, -1)
        ce_sum += (lse - lg.gather(1, t[s:e, None]).squeeze(1)).sum()
        z_sum += lse.pow(2).sum()
        g = torch.exp(lg - lse[:, None]) * ((1 + 2 * z * lse) / n)[:, None]
        g[torch.arange(e - s, device=g.device), t[s:e]] -= 1 / n
        del lg
        grad_h[s:e] = g @ W64
        grad_W += g.t() @ h64[s:e]
    return ce_sum / n + z * z_sum / n, grad_h, grad_W


def check(h, W, t, impls, chunk, z=1e-4):
    ref = reference_fp64(h, W, t, z)
    res = {}
    for impl in impls:
        hh, WW = h.detach().clone().requires_grad_(), W.detach().clone().requires_grad_()
        loss, _ = linear_cross_entropy(hh, WW, t, z_coef=z, chunk=chunk, impl=impl)
        loss.backward()
        rel = lambda a, b: ((a.double() - b).norm() / b.norm()).item()  # noqa: E731
        res[impl] = {"loss": abs(loss.item() - ref[0].item()), "grad_h": rel(hh.grad, ref[1]),
                     "grad_W": rel(WW.grad, ref[2])}
        print(f"{impl:11s} |loss-fp64| {res[impl]['loss']:.2e}  |grad_h-fp64| {res[impl]['grad_h']:.2e}  "
              f"|grad_W-fp64| {res[impl]['grad_W']:.2e}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--v", type=int, default=65536)
    ap.add_argument("--impls", default="chunked,fused,fused_bf16")
    ap.add_argument("--chunks", default="1024,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--sweep", action="store_true", help="time the row kernel alone for every config in SWEEP")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="accuracy against fp64 at chunk 2048")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs the GPU"
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | row kernel default (BLOCK_V, num_warps) "
          f"{lf.get_config()} | addmm out-aliasing ok: {lf.addmm_alias_ok(torch.device('cuda', 0))}", flush=True)
    h, W, t = make_inputs(a.n, a.d, a.v)
    impls, chunks = a.impls.split(","), [int(c) for c in a.chunks.split(",")]
    results = {"timing": [], "sweep": {}, "check": {}}
    base = None
    print(f"\n{'impl':11s} {'chunk':>5s} {'ms/micro-batch':>15s} {'s/step':>7s} {'saves s/step':>13s} {'peak GiB':>9s}")
    for impl in impls:
        for chunk in chunks:
            try:
                ms = time_loss(h, W, t, impl, chunk, reps=a.reps)
                pk = peak_gib(h, W, t, impl, chunk)
            except torch.OutOfMemoryError:
                print(f"{impl:11s} {chunk:5d}  out of memory")
                torch.cuda.empty_cache()
                continue
            if impl == "chunked" and chunk == 2048:
                base = ms
            row = {"impl": impl, "chunk": chunk, "ms": round(ms, 2), "s_per_step": round(ms * MICRO_BATCHES / 1e3, 3),
                   "peak_gib": round(pk, 2)}
            results["timing"].append(row)
            save = f"{(base - ms) * MICRO_BATCHES / 1e3:13.2f}" if base is not None else f"{'':13s}"
            print(f"{impl:11s} {chunk:5d} {ms:15.2f} {ms * MICRO_BATCHES / 1e3:7.2f} {save} {pk:9.2f}", flush=True)
            torch.cuda.empty_cache()
    if a.sweep:
        for ldt in (torch.float32, torch.bfloat16):
            name = "fp32 logits" if ldt == torch.float32 else "bf16 logits (in place)"
            res = kernel_sweep(2048, a.v, SWEEP, ldt)
            results["sweep"][name] = res
            print(f"\nrow kernel alone, [2048, {a.v}] {name}:")
            for r in res:
                print(f"  BLOCK_V {r['cfg'][0]:6d} warps {r['cfg'][1]:2d}: {r['ms']:6.3f} ms  "
                      f"{r['GBps_1read']:5d} GB/s if pass 2 hits L2 ({r['GBps_2reads']:5d} GB/s if it reads DRAM)")
            best = min(res, key=lambda r: r["ms"])
            print(f"  fastest: {tuple(best['cfg'])} (default {lf.DEFAULT_CFG}); MOE_CE_CFG="
                  f"{best['cfg'][0]},{best['cfg'][1]} selects it without a code change")
            lf.set_hints(False)
            try:
                nh = kernel_sweep(2048, a.v, [tuple(best["cfg"]), lf.DEFAULT_CFG], ldt)
            finally:
                lf.set_hints(True)
            results["sweep"][name + ", no cache hints"] = nh
            print("  without the L2 eviction hints (MOE_CE_HINTS=0): " +
                  ", ".join(f"{tuple(r['cfg'])} {r['ms']:.3f} ms" for r in nh))
            torch.cuda.empty_cache()
    if a.profile:
        for impl in impls:
            profile(h, W, t, impl, 2048)
    if a.check:
        print("\naccuracy against fp64 on the same bf16 operands (chunk 2048):")
        results["check"] = check(h, W, t, impls, 2048)
    print("\nRESULTS " + json.dumps(results))


if __name__ == "__main__":
    main()
