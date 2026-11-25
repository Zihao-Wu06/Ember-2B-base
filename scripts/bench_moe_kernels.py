#!/usr/bin/env python
"""Routed-expert implementations on the GPU at main sizes: bmm (padded), grouped (torch._grouped_mm) and triton.

  python scripts/bench_moe_kernels.py                       # N = 8192, routing max/mean 1.0, 1.3, 1.8, 3.0
  python scripts/bench_moe_kernels.py --skews 1.3 --profile  # + per-kernel CUDA time of one fwd+bwd

Times the full MoE.experts() path (sort / dispatch, the expert GEMMs, SiLU gate, weighted combine) under bf16
autocast, forward and forward+backward, with the expert recompute on (as in the main run: recompute_experts)
and off; reports the memory kept from forward to backward, the peak during fwd+bwd and the number of CPU-GPU
syncs per fwd+bwd (sync debug mode), and checks values and gradients against bmm. Routing is synthetic with a set busiest-expert load (max/mean). Weights are persistent
bf16 copies whose gradients a hook adds into fp32 buffers, as with offload_master in the main run
(--fp32-weights: plain fp32 parameters through cast_weight instead). The first triton call autotunes (cached on
disk afterwards); its time is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm.config import main_2b  # noqa: E402
from moe_llm.layers import common  # noqa: E402
from moe_llm.layers import moe_triton  # noqa: E402
from moe_llm.layers.moe import MoE  # noqa: E402

# "triton_copy": triton with the weight-gradient GEMMs reading expert-sorted copies of x and dy
# (moe_triton._WGRAD_COPY / MOE_TRITON_WGRAD_COPY=1) instead of gathering rows in the loop
IMPL = {"bmm": ("bmm", False), "grouped": ("grouped", False), "triton": ("triton", False),
        "triton_copy": ("triton", True)}


def synthetic_routing(N, K, E, max_over_mean, seed=0, device="cuda"):
    """Expert ids [N, K] (distinct per token) whose busiest expert gets ~max_over_mean x the mean load, and
    combine weights [N, K] (fp32, positive, summing to 1). max_over_mean <= 1: exactly balanced (token t takes
    experts tK .. tK+K-1 mod E, then experts relabelled and tokens shuffled). Otherwise the per-expert logits
    are b * z with fixed z ~ N(0, 1), b found by bisection; Gumbel noise picks each token's K experts."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.softmax(torch.randn(N, K, generator=gen), -1)
    if max_over_mean <= 1.0:
        idx = (torch.arange(N * K) % E).view(N, K)
        idx = torch.randperm(E, generator=gen)[idx][torch.randperm(N, generator=gen)]
        return idx.to(device), w.to(device)
    z = torch.randn(E, generator=gen)
    gumbel = -torch.log(-torch.log(torch.rand(N, E, generator=gen).clamp_(1e-9, 1 - 1e-9)))

    def pick(b):
        return torch.topk(b * z + gumbel, K, dim=-1).indices

    lo, hi = 0.0, 1.0
    while max_over_mean_of(pick(hi), E) < max_over_mean and hi < 1e4:
        hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if max_over_mean_of(pick(mid), E) < max_over_mean else (lo, mid)
    return pick(hi).to(device), w.to(device)


def max_over_mean_of(idx, E):
    counts = torch.bincount(idx.flatten().cpu(), minlength=E).float()
    return (counts.max() / counts.mean()).item()


max_over_mean = max_over_mean_of


def time_ms(fn, reps=20, warmup=5):
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


def persistent_copies(moe):
    """offload_master in the main run: persistent bf16 leaves; a hook adds each gradient into an fp32 buffer."""
    bufs = {}
    for name in ("w_gate", "w_up", "w_down"):
        p = getattr(moe, name)
        c = p.detach().to(torch.bfloat16).requires_grad_()

        def hook(c, name=name):
            if name in bufs:
                bufs[name].add_(c.grad)
            else:
                bufs[name] = c.grad.float()
            c.grad = None
        c.register_post_accumulate_grad_hook(hook)
        common.set_cached_copy(p, c)
    return bufs


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def count_syncs(fn):
    """CPU-GPU synchronisations made by fn() (torch.cuda.set_sync_debug_mode("warn") warnings)."""
    import warnings
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(w.message) for w in caught)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192, help="tokens per micro-batch (main run: 2 x 4096)")
    ap.add_argument("--skews", default="1.0,1.3,1.8,3.0", help="busiest expert load / mean load")
    ap.add_argument("--impls", default="bmm,grouped,triton,triton_copy", help=", ".join(IMPL))
    ap.add_argument("--recompute", default="both", choices=["both", "on", "off"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--fp32-weights", action="store_true", help="fp32 parameters through cast_weight instead of "
                    "persistent bf16 copies")
    ap.add_argument("--profile", action="store_true", help="print the top CUDA kernels of one fwd+bwd per case")
    ap.add_argument("--json", default="", help="append one JSON line per case to this file")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = main_2b()
    torch.manual_seed(0)
    moe = MoE(cfg, 1).cuda().train()
    bufs = {} if args.fp32_weights else persistent_copies(moe)
    N, d = args.tokens, cfg.d_model
    x = torch.randn(N, d, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(N, d, device="cuda", dtype=torch.bfloat16)
    recs = {"both": [True, False], "on": [True], "off": [False]}[args.recompute]
    impls = args.impls.split(",")
    print(f"{torch.cuda.get_device_name()}, N={N} d={d} E={cfg.n_experts} k={cfg.top_k} I={cfg.expert_inter}, "
          f"bf16 autocast, weights: {'fp32 params' if args.fp32_weights else 'persistent bf16 copies'}")

    def run(impl, rec, idx, w, backward=True):
        moe.impl, moe_triton._WGRAD_COPY = IMPL[impl]
        moe.recompute_experts = rec
        xi, wi = x.detach().requires_grad_(), w.detach().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe.experts(xi, idx, wi)
        if backward:
            y.backward(dy)
        return y, xi, wi

    def grads(impl, idx, w):
        moe.zero_grad(set_to_none=True)
        bufs.clear()
        y, xi, wi = run(impl, False, idx, w)
        ws = [bufs[n] for n in ("w_gate", "w_up", "w_down")] if bufs else [moe.w_gate.grad, moe.w_up.grad,
                                                                             moe.w_down.grad]
        return [y.detach(), xi.grad, wi.grad] + [g.clone() for g in ws]

    first = {}
    hdr = (f"{'max/mean':>8} {'impl':>8} {'recomp':>6} {'fwd ms':>8} {'fwd+bwd ms':>10} {'kept MB':>8} {'peak MB':>8} "
           f"{'syncs':>5}")
    for skew in [float(s) for s in args.skews.split(",")]:
        idx, w = synthetic_routing(N, cfg.top_k, cfg.n_experts, skew, seed=1)
        mom = max_over_mean_of(idx, cfg.n_experts)
        print(f"\nrouting max/mean {mom:.2f} (asked {skew})")
        ref = None
        for impl in impls:
            try:
                if impl not in first:  # first call: compile / autotune
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    run(impl, False, idx, w)
                    torch.cuda.synchronize()
                    first[impl] = time.perf_counter() - t0
                    print(f"  first {impl} call (compile/autotune): {first[impl]:.1f} s")
                g = grads(impl, idx, w)
            except Exception as e:  # e.g. grouped_mm unsupported
                print(f"  {impl}: failed: {e!r}")
                continue
            if ref is None:
                ref = g
            else:
                errs = [rel_rms(a, b) for a, b in zip(g, ref)]
                print(f"  {impl} vs {impls[0]} rel. RMS: " + ", ".join(
                    f"{n} {e:.1e}" for n, e in zip(["y", "dx", "dw", "dWg", "dWu", "dWd"], errs)))
        print(hdr)
        for impl in impls:
            if impl not in first:
                continue
            for rec in recs:
                try:
                    fwd = time_ms(lambda: run(impl, rec, idx, w, backward=False), args.reps, args.warmup)
                    full = time_ms(lambda: run(impl, rec, idx, w), args.reps, args.warmup)
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    y, _, _ = run(impl, rec, idx, w, backward=False)
                    kept = torch.cuda.memory_allocated() - base - y.numel() * y.element_size()
                    y.backward(dy)
                    del y
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() - base
                    syncs = count_syncs(lambda: run(impl, rec, idx, w))
                except Exception as e:
                    print(f"{mom:8.2f} {impl:>8} {str(rec):>6} failed: {e!r}")
                    continue
                mb = 2 ** 20
                print(f"{mom:8.2f} {impl:>8} {str(rec):>6} {fwd:8.2f} {full:10.2f} {kept / mb:8.1f} {peak / mb:8.1f} "
                      f"{syncs:5d}")
                if args.json:
                    with open(args.json, "a") as fh:
                        fh.write(json.dumps({"max_over_mean": mom, "impl": impl, "recompute": rec, "fwd_ms": fwd,
                                             "fwd_bwd_ms": full, "kept_mb": kept / mb, "peak_mb": peak / mb,
                                             "syncs": syncs, "tokens": N}) + "\n")
                if args.profile:
                    from torch.profiler import ProfilerActivity, profile
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        run(impl, rec, idx, w)
                        torch.cuda.synchronize()
                    ka = prof.key_averages()
                    for key in ("self_device_time_total", "self_cuda_time_total"):
                        try:
                            print(ka.table(sort_by=key, row_limit=12))
                            break
                        except Exception:
                            continue
        moe.zero_grad(set_to_none=True)
        bufs.clear()
    common.clear_cached_copies()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as in train.py
    main()
