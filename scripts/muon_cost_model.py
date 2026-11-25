#!/usr/bin/env python
"""Analytic cost of main_2b's optimizer phase per Muon impl (CPU only, measurement/analysis; nothing is trained).

Runs Muon.step on meta tensors with main_2b's 319 Muon parameter shapes (bf16 iteration and bf16 momentum, as in the
main run) under a dispatch counter and reports, per impl: kernel launches, GEMM FLOPs, bytes moved by the non-GEMM
kernels (inputs + outputs once each; baddbmm/addmm with beta != 0 also copy C into the output and read it again), split
into the Newton-Schulz part and the rest. Adds the other parts of the phase (reload / offload copies, AdamW, bf16
refresh) analytically and converts everything to milliseconds with the rates measured on this 4090
(docs/DESIGN.md: 182 TFLOPS bf16 at 8192^3, 919 GB/s copy, 23.8 GB/s pinned H2D) and assumed efficiencies (--gemm-tflops,
--gbs). Also times Muon.step on the CPU with the same parameter list scaled down (--cpu-scale), as a proxy for the
Python/dispatcher cost per step (the CUDA launch cost comes on top).
    python scripts/muon_cost_model.py [--gemm-tflops 140] [--gbs 800]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import moe_llm.optim as optim  # noqa: E402

GEMMS = {"mm", "bmm", "addmm", "baddbmm"}
NO_KERNEL = {"empty", "empty_strided", "empty_like", "detach", "lift_fresh", "alias"}
BF16 = torch.bfloat16


def main_2b_muon_shapes():
    from moe_llm.config import main_2b
    from moe_llm.model import MoELLM
    cfg = main_2b().replace(engram_offload=False, moe_impl="triton", ce_impl="fused", attnres_impl="fused")
    with torch.device("meta"):
        m = MoELLM(cfg)
    g = optim.split_params(m)
    other = {k: [tuple(p.shape) for p in v] for k, v in g.items() if k != "muon"}
    cast = [tuple(p.shape) for p in m.cast_params()]
    return [tuple(p.shape) for p in g["muon"]], other, cast


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.tag = "other"
        self.launch = Counter()
        self.flops = Counter()
        self.bytes = Counter()
        self.gemm_bytes = Counter()
        self.ops = Counter()
        self.gemm_shapes = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        if func.is_view or name in NO_KERNEL:
            return out
        tag = self.tag
        tens_in = [a for a in args if isinstance(a, torch.Tensor)]
        tens_in += [a for a in args if isinstance(a, (list, tuple)) for a in a if isinstance(a, torch.Tensor)]
        outs = out if isinstance(out, (list, tuple)) else [out]
        outs = [o for o in outs if isinstance(o, torch.Tensor)]

        def nb(t):
            return t.numel() * t.element_size()
        self.ops[(tag, name)] += 1
        if name in GEMMS:
            if name in ("mm", "addmm"):
                a, b = (args[0], args[1]) if name == "mm" else (args[1], args[2])
                batch, (m, k), n = 1, a.shape, b.shape[1]
            else:
                a, b = (args[0], args[1]) if name == "bmm" else (args[1], args[2])
                batch, m, k, n = a.shape[0], a.shape[1], a.shape[2], b.shape[2]
            self.flops[tag] += 2 * batch * m * n * k
            self.gemm_shapes[(tag, batch, m, n, k)] += 1
            self.gemm_bytes[tag] += nb(a) + nb(b) + sum(nb(o) for o in outs)
            self.launch[tag] += 1
            beta = kwargs.get("beta", 1)
            if name in ("addmm", "baddbmm") and beta != 0:
                c = args[0]
                self.bytes[tag] += 2 * nb(outs[0]) + nb(c)  # copy C -> out (read + write), epilogue reads it again
                self.launch[tag] += 1
            return out
        self.launch[tag] += 1
        self.bytes[tag] += sum(nb(t) for t in tens_in) + sum(nb(o) for o in outs)
        return out


def run_meta(shapes, impl, chunk_elems):
    """One Muon step on meta tensors with bf16 iteration and momentum, counted by Count."""
    params = []
    for s in shapes:
        p = torch.nn.Parameter(torch.empty(s, device="meta"))
        p.grad = torch.empty(s, device="meta")
        params.append(p)
    mu = optim.Muon(params, lr=2e-3, state_dtype=BF16, impl=impl, chunk_elems=chunk_elems)
    for p in params:
        mu.state[p]["buf"] = torch.empty(p.shape, dtype=BF16, device="meta")
    cnt = Count()
    orig = (optim._ns_dtype, optim.zeropower_via_newtonschulz5, optim.zeropower_via_newtonschulz5_fused,
            optim.newtonschulz5_stacked)

    def tagged(f, **fixed):
        def g(*a, **kw):
            cnt.tag = "ns"
            try:
                return f(*a, **{**kw, **fixed})
            finally:
                cnt.tag = "other"
        return g
    try:
        optim._ns_dtype = lambda t: BF16  # the CUDA iteration dtype on meta tensors
        optim.zeropower_via_newtonschulz5 = tagged(orig[1], dtype=BF16)
        optim.zeropower_via_newtonschulz5_fused = tagged(orig[2], dtype=BF16)
        optim.newtonschulz5_stacked = tagged(orig[3])
        with cnt:
            mu.step()
    finally:
        (optim._ns_dtype, optim.zeropower_via_newtonschulz5, optim.zeropower_via_newtonschulz5_fused,
         optim.newtonschulz5_stacked) = orig
    chunks = sum(len(mu._chunks(gi)) for gi in range(len(mu.param_groups))) if impl != "loop" else len(params)
    return cnt, chunks


def gemm_ms(cnt, tag, peak_tflops, sms=128):
    """GEMM time with a crude occupancy model: full rate from 2 waves of 128 x 128 output tiles up, proportionally
    less below (small GEMMs leave SMs idle; cuBLAS's split-K / smaller tiles recover part of it, ignored here)."""
    t = 0.0
    for (tg, batch, m, n, k), c in cnt.gemm_shapes.items():
        if tg != tag:
            continue
        tiles = batch * math.ceil(m / 128) * math.ceil(n / 128)
        eff = min(1.0, tiles / (2 * sms))
        t += c * 2 * batch * m * n * k / (peak_tflops * 1e12 * max(eff, 0.15))
    return t * 1e3


def cpu_dispatch_ms(shapes, impl, scale, reps=3):
    """Muon.step on real CPU tensors whose dims are divided by `scale` (same parameter count and op sequence)."""
    small = [tuple(max(16, d // scale) if i >= len(s) - 2 else d for i, d in enumerate(s)) for s in shapes]
    small = [tuple(min(d, 8) if (len(s) == 3 and i == 0) else d for i, d in enumerate(s)) for s in small]
    params = [torch.nn.Parameter(torch.randn(s)) for s in small]
    for p in params:
        p.grad = torch.randn(p.shape)
    mu = optim.Muon(params, lr=2e-3, state_dtype=BF16, impl=impl)
    mu.step()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        mu.step()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm-tflops", type=float, default=150.0, help="cuBLAS bf16 rate for well-filled GEMMs")
    ap.add_argument("--gbs", type=float, default=800.0, help="effective DRAM rate of the elementwise kernels, GB/s")
    ap.add_argument("--h2d-gbs", type=float, default=23.76)
    ap.add_argument("--min-kernel-us", type=float, default=2.5, help="floor per kernel (launch-bound small kernels)")
    ap.add_argument("--chunk-elems", type=int, default=optim.MUON_CHUNK_ELEMS)
    ap.add_argument("--cpu-scale", type=int, default=16)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    shapes, other, cast = main_2b_muon_shapes()
    n_muon = sum(math.prod(s) for s in shapes)
    n_adamw = sum(math.prod(s) for k in ("adamw_decay", "adamw_no_decay") for s in other[k])
    n_cast = sum(math.prod(s) for s in cast)
    by_shape = Counter(shapes)
    print(f"main_2b Muon: {len(shapes)} parameters, {n_muon / 1e9:.3f}B elements; shapes: "
          + ", ".join(f"{list(s)} x{c}" for s, c in by_shape.most_common()))
    # the rest of the phase, per step (main run: offload_master + offload_opt, bf16 Muon momentum)
    h2d = n_cast * 4 + n_muon * 2 + n_adamw * 8  # fp32 masters of the cast parameters, Muon momentum, AdamW moments
    rest_bytes = {"AdamW (fused, fp32 param/grad/2 moments, R+W)": n_adamw * 28,
                  "bf16 refresh of the cast parameters (R fp32, W bf16)": n_cast * 6}
    print(f"reload H2D {h2d / 1e9:.2f} GB = {h2d / args.h2d_gbs / 1e6:.0f} ms at {args.h2d_gbs} GB/s (side stream, "
          f"overlaps the compute); the same back D2H after the step (overlaps the next forward)")
    for k, v in rest_bytes.items():
        print(f"{k}: {v / 1e9:.2f} GB = {v / args.gbs / 1e6:.1f} ms")
    res = {}
    print(f"\n{'impl':14s} {'chunks':>6s} {'launches':>9s} {'NS launches':>11s} {'NS GEMM TF':>10s} {'NS GEMM ms':>10s} "
          f"{'NS elem GB':>10s} {'other GB':>9s} {'NS ms':>7s} {'other ms':>8s} {'Muon ms':>8s} {'CPU ms*':>8s}")
    for impl in optim.MUON_IMPLS:
        cnt, chunks = run_meta(shapes, impl, args.chunk_elems)
        ns_g = gemm_ms(cnt, "ns", args.gemm_tflops)
        ns_e = cnt.bytes["ns"] / args.gbs / 1e6
        ot_e = cnt.bytes["other"] / args.gbs / 1e6
        floor = args.min_kernel_us / 1e3
        ns_ms = max(ns_g + ns_e, cnt.launch["ns"] * floor)
        ot_ms = max(ot_e, cnt.launch["other"] * floor)
        cpu = cpu_dispatch_ms(shapes, impl, args.cpu_scale)
        res[impl] = {"chunks": chunks, "launches": sum(cnt.launch.values()), "ns_launches": cnt.launch["ns"],
                     "ns_gemm_tflop": cnt.flops["ns"] / 1e12, "ns_gemm_ms": ns_g, "ns_elem_gb": cnt.bytes["ns"] / 1e9,
                     "other_gb": cnt.bytes["other"] / 1e9, "ns_ms": ns_ms, "other_ms": ot_ms,
                     "muon_ms": ns_ms + ot_ms, "cpu_dispatch_ms": cpu,
                     "ops": {f"{t}:{n}": c for (t, n), c in sorted(cnt.ops.items())}}
        r = res[impl]
        print(f"{impl:14s} {chunks:6d} {r['launches']:9d} {r['ns_launches']:11d} {r['ns_gemm_tflop']:10.2f} "
              f"{ns_g:10.1f} {r['ns_elem_gb']:10.1f} {r['other_gb']:9.1f} {ns_ms:7.1f} {ot_ms:8.1f} "
              f"{ns_ms + ot_ms:8.1f} {cpu:8.1f}")
    print("* CPU ms: Muon.step on CPU tensors scaled down %dx per matrix dim (same parameter count and op sequence): "
          "Python + dispatcher cost per step; the CUDA launch cost comes on top" % args.cpu_scale)
    base = res["loop"]["muon_ms"]
    for impl in optim.MUON_IMPLS[1:]:
        d = base - res[impl]["muon_ms"]
        print(f"{impl}: Muon compute {d:+.0f} ms per step vs loop if the phase is compute-bound "
              f"({100 * d / 17100:+.2f}% of a 17.1 s step); the phase cannot drop below the reload copies "
              f"({h2d / args.h2d_gbs / 1e6:.0f} ms)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"shapes": {str(list(s)): c for s, c in by_shape.items()}, "h2d_gb": h2d / 1e9, "impls": res,
                       "args": vars(args)}, fh, indent=1)


if __name__ == "__main__":
    main()
