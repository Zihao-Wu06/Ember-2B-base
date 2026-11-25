#!/usr/bin/env python
"""GPU benchmark: Block AttnRes attnres_impl="compiled" (layers/attnres.py, the running code) vs "fused"
(layers/attnres_fused.py). Measurement only, nothing is trained or saved.

ops (default): the AttnRes part of one main_2b micro-batch in isolation: 24 layers in blocks of 3, every res_mix /
  res_ffn op and out_res with the real source counts (2..9), [2, 4096, 1024] bf16 sources, the real autograd
  structure (block outputs read by many later ops). The mixer and FFN are replaced by a per-channel scale, so the
  time is the AttnRes ops plus the residual adds and scales both paths share. Prints forward / backward ms per micro-batch, extra peak
  memory, and the estimated saving per optimizer step (64 micro-batches; the forward of the recomputed layers runs
  twice, --ckpt-frac).
e2e: scripts/train.py with the main-run settings from a fresh init for --steps steps per impl (same seed and data),
  median tok/s of the steps after --warmup and peak memory (needs the whole GPU: pause the main run first).

  python scripts/bench_attnres_fused.py ops [--reps 20]
  python scripts/bench_attnres_fused.py e2e [--steps 8] [--ckpt-frac 0.25]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# data/ lives in the project root; a staging copy of the code (staging_*/) has none of its own
PROJ = ROOT if os.path.isdir(os.path.join(ROOT, "data")) else os.path.dirname(ROOT)


# ---------------------------------------------------------------------------- the AttnRes part of a micro-batch
def make_workload(n_layers=24, lpb=3, B=2, T=4096, d=1024, device="cuda", dtype=None, seed=0, w_std=0.05):
    """Modules and inputs of one micro-batch: DepthAttn per (layer, mix|ffn) and out_res, the embedding output (a
    leaf), the upstream gradient of the final output. Pseudo-queries are not zero (trained-like logits)."""
    import torch
    from moe_llm.layers.attnres import DepthAttn
    dtype = dtype or torch.bfloat16
    g = torch.Generator(device="cpu").manual_seed(seed)
    mods = {}
    for i in range(n_layers):
        for kind in ("mix", "ffn"):
            if kind == "mix" and i == 0:
                continue
            m = DepthAttn(d).to(device)
            with torch.no_grad():
                m.w.copy_(torch.randn(d, generator=g) * w_std)
                m.g.copy_(1 + 0.1 * torch.randn(d, generator=g))
            mods[(i, kind)] = m
    m = DepthAttn(d).to(device)
    with torch.no_grad():
        m.w.copy_(torch.randn(d, generator=g) * w_std)
    mods["out"] = m
    for i in range(n_layers):  # stand-ins for the mixer and the FFN: a fixed per-channel scale (no saved activations)
        for name in ("mix_scale", "ffn_scale"):
            mods[(i, "ffn")].register_buffer(name, (0.25 + 0.5 * torch.rand(d, generator=g)).to(device, dtype))
    emb = torch.randn(B, T, d, generator=g).to(device=device, dtype=dtype)
    gout = torch.randn(B, T, d, generator=g).to(device=device, dtype=dtype)
    return mods, emb, gout


def attnres_forward(impl, mods, emb, n_layers=24, lpb=3):
    """The Block AttnRes data flow of MoELLM.hidden with the mixer / FFN replaced by a per-channel scale. impl:
    "compiled" | "eager" (layers/attnres.py, compiled on CUDA / eager) | "fused" (layers/attnres_fused.py)."""
    import torch
    from moe_llm.layers import attnres
    if impl == "fused":
        from moe_llm.layers.attnres_fused import FusedAttnRes, depth_attn
        st = FusedAttnRes(emb, n_blocks=(n_layers + lpb - 1) // lpb, n_slots=2 * lpb + 1)
        partial = emb
        for i in range(n_layers):
            o, boundary = 2 * (i % lpb), i % lpb == 0
            if boundary:
                ops = {}
                for j in range(i, min(i + lpb, n_layers)):
                    if (j, "mix") in mods:
                        ops[2 * (j % lpb)] = mods[(j, "mix")]
                    ops[2 * (j % lpb) + 1] = mods[(j, "ffn")]
                if i + lpb >= n_layers:
                    ops[2 * lpb] = mods["out"]
                blk, tok, qs = st.open_block(i // lpb, partial, ops)
            if i == 0:
                h = partial
            else:
                h = depth_attn(blk, tok, None if boundary else partial, qs[o], o, mods[(i, "mix")].eps)
            y = h * mods[(i, "ffn")].mix_scale
            p = y if boundary else partial + y
            f = depth_attn(blk, tok, p, qs[o + 1], o + 1, mods[(i, "ffn")].eps) * mods[(i, "ffn")].ffn_scale
            partial = p + f
        return depth_attn(blk, tok, partial, qs[2 * lpb], 2 * lpb, mods["out"].eps)
    attnres.set_attnres_compile(impl == "compiled")
    try:
        blocks, partial = [], emb
        for i in range(n_layers):
            boundary = i % lpb == 0
            if boundary:
                blocks.append(partial)
            srcs = list(blocks) if boundary else [*blocks, partial]
            h = srcs[0] if len(srcs) == 1 else mods[(i, "mix")](srcs)
            y = h * mods[(i, "ffn")].mix_scale
            p = y if boundary else partial + y
            f = mods[(i, "ffn")]([*blocks, p]) * mods[(i, "ffn")].ffn_scale
            partial = p + f
        return mods["out"](blocks + [partial])
    finally:
        attnres.set_attnres_compile(True)


def fwd_bwd(impl, mods, emb, gout, **kw):
    """One forward + backward; returns (output, grads of [emb, *params]) with params zeroed first."""
    import torch
    for m in mods.values():
        m.zero_grad(set_to_none=True)
    x = emb.detach().requires_grad_()
    out = attnres_forward(impl, mods, x, **kw)
    out.backward(gout)
    return out.detach(), [x.grad] + [p.grad for m in mods.values() for p in (m.w, m.g)]


def time_impl(impl, mods, emb, gout, reps=20, warmup=3, **kw):
    """Median forward and backward ms per micro-batch (CUDA events) and the extra peak memory of fwd+bwd."""
    import torch
    for _ in range(warmup):
        fwd_bwd(impl, mods, emb, gout, **kw)
    torch.cuda.synchronize()
    fw, bw = [], []
    for _ in range(reps):
        for m in mods.values():
            m.zero_grad(set_to_none=True)
        x = emb.detach().requires_grad_()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        out = attnres_forward(impl, mods, x, **kw)
        e1.record()
        out.backward(gout)
        e2.record()
        torch.cuda.synchronize()
        fw.append(e0.elapsed_time(e1))
        bw.append(e1.elapsed_time(e2))
        del out, x
    for m in mods.values():
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fwd_bwd(impl, mods, emb, gout, **kw)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return statistics.median(fw), statistics.median(bw), peak


def run_ops(args):
    import torch
    torch.manual_seed(0)
    mods, emb, gout = make_workload(n_layers=args.layers, lpb=args.lpb, T=args.seq)
    res = {}
    for impl in args.impls.split(","):
        f, b, peak = time_impl(impl, mods, emb, gout, reps=args.reps, n_layers=args.layers, lpb=args.lpb)
        res[impl] = {"fwd_ms": f, "bwd_ms": b, "total_ms": f + b, "peak_extra_gb": peak / 2 ** 30}
        print(f"{impl:9s} forward {f:7.2f} ms  backward {b:7.2f} ms  total {f + b:7.2f} ms  "
              f"peak +{peak / 2 ** 30:.2f} GB", flush=True)
    if "compiled" in res and "fused" in res:
        c, fz = res["compiled"], res["fused"]
        per_mb = (c["fwd_ms"] - fz["fwd_ms"]) * (1 + args.ckpt_frac) + (c["bwd_ms"] - fz["bwd_ms"])
        res["saving_s_per_step"] = per_mb * args.accum / 1e3
        res["speedup_attnres"] = c["total_ms"] / fz["total_ms"]
        print(f"fused vs compiled: {res['speedup_attnres']:.2f}x on the AttnRes part; estimated saving "
              f"{res['saving_s_per_step']:.2f} s per optimizer step ({args.accum} micro-batches, forward of "
              f"{args.ckpt_frac:.0%} of the layers twice)", flush=True)
    return res


# ---------------------------------------------------------------------------- end to end through train.py
def run_e2e(args):
    out_root = args.out
    res = {}
    for impl in args.impls.split(","):
        if impl not in ("compiled", "fused"):
            continue
        out = os.path.join(out_root, impl)
        os.makedirs(out, exist_ok=True)
        log = os.path.join(out, "log.jsonl")
        if os.path.exists(log):
            os.remove(log)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "main_2b",
               "--data", args.data, "--canon", args.canon, "--out", out, "--steps", str(args.steps),
               "--micro-bsz", "2", "--global-tokens", "524288", "--lr", "2e-3", "--warmup", "10", "--grad-ckpt",
               "--ckpt-frac", str(args.ckpt_frac), "--offload-master", "--offload-opt", "--fuse", "--muon-bf16",
               "--anneal-mix", "reasoning_anneal", "--log-every", "1", "--eval-every", "0", "--ckpt-every", "0",
               "--override", json.dumps({"moe_impl": "triton", "attnres_impl": impl})]
        t0 = time.time()
        with open(os.path.join(out, "stdout.log"), "w") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        recs = [json.loads(l) for l in open(log) if '"tok_s"' in l] if os.path.exists(log) else []
        meas = [r for r in recs if r["step"] >= args.warmup]
        res[impl] = {"rc": rc, "minutes": round((time.time() - t0) / 60, 1),
                     "tok_s": statistics.median(r["tok_s"] for r in meas) if meas else None,
                     "peak_gb": max(r["mem_gb"] for r in recs) if recs else None,
                     "ce": [r["ce"] for r in recs]}
        print(impl, json.dumps(res[impl]), flush=True)
    if all(res.get(k, {}).get("tok_s") for k in ("compiled", "fused")):
        res["speedup"] = res["fused"]["tok_s"] / res["compiled"]["tok_s"]
        print(f"fused vs compiled end to end: {res['speedup']:.3f}x tok/s, peak "
              f"{res['compiled']['peak_gb']:.2f} -> {res['fused']['peak_gb']:.2f} GB", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="ops", choices=["ops", "e2e"])
    ap.add_argument("--impls", default="compiled,fused", help="comma-separated: compiled, fused, eager (ops only)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--lpb", type=int, default=3)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--accum", type=int, default=64, help="micro-batches per optimizer step")
    ap.add_argument("--ckpt-frac", type=float, default=0.25, help="fraction of layers recomputed (forward runs twice)")
    ap.add_argument("--steps", type=int, default=8, help="e2e: steps per impl")
    ap.add_argument("--warmup", type=int, default=3, help="e2e: steps excluded from the median")
    ap.add_argument("--data", default=os.path.join(PROJ, "data", "bins_main"))
    ap.add_argument("--canon", default=os.path.join(PROJ, "data", "tokenizer", "canon_map.npy"))
    ap.add_argument("--out", default=os.path.join(PROJ, "runs", "bench_attnres_fused"))
    ap.add_argument("--json", default="", help="write the results to this file")
    ap.add_argument("--force", action="store_true", help="skip the GPU-busy check")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from supervise import Supervisor
    used = Supervisor.gpu_used_gb()
    if not args.force and used is not None and used > 3:
        sys.exit(f"GPU busy ({used:.1f} GB in use): not starting (pause the main run first, or --force)")
    res = run_ops(args) if args.mode == "ops" else run_e2e(args)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"mode": args.mode, "args": vars(args), "results": res}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
