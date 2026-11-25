#!/usr/bin/env python
"""Where the optimizer phase goes and how much of it could overlap the next step's first forward (GPU measurement only:
not training, nothing is saved, runs/main is never touched). Settings as in the main run: main_2b with triton MoE,
fused cross-entropy and AttnRes, fusion, offload_master + fused grad acc, optimizer states in pinned CPU memory, bf16
Muon momentum, no recompute.

Measures
  1. one micro-batch: forward up to the final norm (the most that could run while the previous optimizer step is still
     updating later parameters), final norm + fused LM head / cross-entropy (needs every layer and the LM head updated;
     computes the head's gradients in the forward), backward;
  2. the optimizer phase on the compute stream (CUDA events) in the default order and with adamw_last
     (train.py --opt-adamw-last: Muon first, AdamW last);
  3. its composition (torch.profiler, one step per order): kernel-busy time over the phase, idle time, wait before the
     first compute kernel, copies, top kernels (traces kept in --out);
  4. the Engram CPU update on the worker thread (the next micro-batch 0 forward cannot be queued before it is done);
  5. the overlap ceiling: opt.step on the compute stream while a second stream runs a no-grad forward of the next
     batch, with no dependency waits at all (so an upper bound for any scheme that overlaps the optimizer with the next
     forward): saving = T_opt + T_fwd_nograd - T_both.
  With --muon-impls loop,batched,batched_fused (train.py --muon-impl) the timed steps cycle over those Muon impls in
  the default order instead (one process, the same states), and each impl gets: the optimizer phase (CUDA events),
  step()'s CPU time, the peak memory of the phase next to the micro-batches' peak, and a profile whose GPU time is split
  by record_function ranges around the parts of the step (reload / offload copies, Newton-Schulz GEMMs, Newton-Schulz
  elementwise and norms, Muon momentum and update, AdamW, bf16 refresh) with the compute stream's idle time and the
  end of the last reload copy (copy-bound if it ends with the phase).
Run during a pause of the main run (needs the whole GPU; refuses to start if more than 3 GB are in use), from the
staging directory:
    python scripts/bench_opt_phase.py --ckpt /root/ckpt/main/ckpt_XXXXXXX.pt --data ../data/bins_main \\
        --canon ../data/tokenizer/canon_map.npy --out ../runs/profile/opt_phase
    python scripts/bench_opt_phase.py ... --muon-impls loop,batched,batched_fused --out ../runs/profile/opt_phase_muon
CPU smoke test (plumbing only): python scripts/bench_opt_phase.py --preset tiny --device cpu --seq-len 32 --out /tmp/x
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as train.py
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MAIN_OVERRIDE = {"moe_impl": "triton", "ce_impl": "fused", "attnres_impl": "fused"}


# ------------------------------------------------------------------------------------------------- trace analysis
def _nullctx():
    import contextlib
    return contextlib.nullcontext()


def union_us(intervals) -> float:
    """Length of the union of (start, end) intervals."""
    total, cur = 0.0, None
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if cur is None or s > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + (cur[1] - cur[0] if cur is not None else 0.0)


def category(name: str) -> str:
    n = name.lower()
    if "memcpy" in n:
        return "memcpy " + ("HtoD" if "htod" in n else "DtoH" if "dtoh" in n else "DtoD")
    if "memset" in n:
        return "memset"
    if any(k in n for k in ("gemm", "cutlass", "xmma", "s16816", "s1688", "tensorop", "gemv")):
        return "GEMM"
    if "adam" in n or "multi_tensor" in n:
        return "AdamW (multi-tensor apply)"
    if any(k in n for k in ("reduce", "norm")):
        return "reductions (incl. NS norms)"
    if any(k in n for k in ("elementwise", "vectorized", "unrolled", "copy")):
        return "elementwise / casts / copies"
    return "other"


def analyze_trace(events: list) -> dict:
    """GPU activities (kernels, memcpy, memset) of one profiled optimizer step, from a chrome trace. The phase runs
    from the first GPU activity to the end of the last kernel on the compute stream (the stream with the most kernel
    time); the offload copies after it are reported as the D2H tail."""
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    kern = [e for e in gpu if e["cat"] == "kernel"]
    if not kern:
        return {}

    def stream(e):
        return (e.get("args") or {}).get("stream", e.get("tid"))

    def span(e):
        return float(e["ts"]), float(e["ts"]) + float(e["dur"])

    per_stream: dict = {}
    for e in kern:
        per_stream[stream(e)] = per_stream.get(stream(e), 0.0) + float(e["dur"])
    compute = max(per_stream, key=per_stream.get)
    t0 = min(span(e)[0] for e in gpu)
    ck = sorted(span(e) for e in kern if stream(e) == compute)
    t_end = max(e for _, e in ck)
    t_all = max(span(e)[1] for e in gpu)

    def clipped(es):
        return [(max(s, t0), min(e, t_end)) for s, e in (span(x) for x in es)]

    busy = union_us(clipped(kern))
    cbusy = union_us(clipped([e for e in kern if stream(e) == compute]))
    gaps, prev = [], t0
    for s, e in ck:
        if s > prev:
            gaps.append((s - prev, prev - t0))
        prev = max(prev, e)
    cats: dict = {}
    tops: dict = {}
    copies: dict = {}
    for e in gpu:
        c = category(e["name"])
        cats[c] = cats.get(c, 0.0) + float(e["dur"]) / 1e3
        if e["cat"] == "kernel":
            t = tops.setdefault(e["name"][:110], [0.0, 0])
            t[0] += float(e["dur"]) / 1e3
            t[1] += 1
        elif e["cat"] == "gpu_memcpy":
            d = copies.setdefault(c, {"ms": 0.0, "gb": 0.0, "n": 0})
            d["ms"] += float(e["dur"]) / 1e3
            d["gb"] += float((e.get("args") or {}).get("bytes", 0)) / 1e9
            d["n"] += 1
    big = [g for g in gaps if g[0] >= 20.0]
    h2d = [span(e)[1] for e in gpu if e["cat"] == "gpu_memcpy" and "htod" in e["name"].lower()]
    return {
        "h2d_last_end_ms": (max(h2d) - t0) / 1e3 if h2d else 0.0,
        "phase_ms": (t_end - t0) / 1e3,
        "kernel_busy_ms": busy / 1e3,
        "idle_ms": (t_end - t0 - busy) / 1e3,
        "compute_stream_busy_ms": cbusy / 1e3,
        "first_compute_kernel_after_ms": (ck[0][0] - t0) / 1e3,
        "largest_compute_gap_ms": max((g[0] for g in gaps), default=0.0) / 1e3,
        "largest_compute_gap_at_ms": max(gaps, default=(0.0, 0.0))[1] / 1e3,
        "compute_gaps_over_20us": {"n": len(big), "ms": sum(g[0] for g in big) / 1e3},
        "d2h_tail_ms": (t_all - t_end) / 1e3,
        "n_kernels": len(kern),
        "streams_kernel_ms": {str(k): v / 1e3 for k, v in per_stream.items()},
        "copies": copies,
        "categories_ms": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "top_kernels": [{"ms": round(v[0], 3), "n": v[1], "name": k}
                        for k, v in sorted(tops.items(), key=lambda kv: -kv[1][0])[:15]],
    }


PARTS = {"opt/ns": "Newton-Schulz", "opt/muon": "Muon momentum / update", "Optimizer.step#Muon.step":
         "Muon momentum / update", "opt/adamw": "AdamW", "Optimizer.step#AdamW.step": "AdamW",
         "opt/refresh": "bf16 refresh", "opt/reload": "reload (non-copy kernels)", "opt/offload": "offload (non-copy kernels)"}


def breakdown(events: list) -> dict:
    """GPU time of one profiled optimizer step by part: each kernel goes to the innermost record_function range
    (profile_step's opt/* ranges, torch.optim's Optimizer.step#... ranges) around the CPU call that launched it (matched
    by correlation id); copies by direction. Also the CPU time spent in each range."""
    ann = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e.get("name") in PARTS]
    by_tid: dict = defaultdict(list)
    for a in ann:
        by_tid[a.get("tid")].append((float(a["ts"]), float(a["ts"]) + float(a["dur"]), a["name"]))
    launch = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launch[c] = (e.get("tid"), float(e["ts"]))

    def innermost(tid, ts):
        best = None
        for s0, s1, n in by_tid.get(tid, ()):
            if s0 <= ts <= s1 and (best is None or s1 - s0 < best[1] - best[0]):
                best = (s0, s1, n)
        return best[2] if best else None
    gpu_ms: dict = defaultdict(float)
    n: dict = defaultdict(int)
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = e["name"].lower()
        if e["cat"] == "gpu_memcpy":
            part = "copies H2D" if "htod" in name else "copies D2H" if "dtoh" in name else "copies D2D"
        else:
            where = launch.get((e.get("args") or {}).get("correlation"))
            rng = innermost(*where) if where else None
            part = PARTS.get(rng, "other")
            if part == "Newton-Schulz":
                part += " GEMM" if category(e["name"]) == "GEMM" else " elementwise / norms"
        gpu_ms[part] += float(e["dur"]) / 1e3
        n[part] += 1
    cpu_ms: dict = defaultdict(float)
    for a in ann:
        cpu_ms[a["name"]] += float(a["dur"]) / 1e3
    return {"gpu_ms": dict(sorted(gpu_ms.items(), key=lambda kv: -kv[1])), "kernels": dict(n), "cpu_range_ms": dict(cpu_ms)}


def ranges_installed(opt):
    """Context manager: record_function ranges around the parts of OptimizerBundle.step (profiling only)."""
    import contextlib

    from moe_llm import optim as om

    def ranged(name, f):
        def g(*a, **kw):
            with torch.profiler.record_function(name):
                return f(*a, **kw)
        return g

    @contextlib.contextmanager
    def cm():
        saved = []
        for owner, attr, name in ((om.OptimizerBundle, "_reload_pipelined", "opt/reload"),
                                  (om.OptimizerBundle, "_refresh_copies", "opt/refresh"),
                                  (om.OptimizerBundle, "_offload_masters", "opt/offload"),
                                  (om.OptimizerBundle, "offload", "opt/offload"), (om.Muon, "step", "opt/muon"),
                                  (om, "zeropower_via_newtonschulz5", "opt/ns"),
                                  (om, "zeropower_via_newtonschulz5_fused", "opt/ns"),
                                  (om, "newtonschulz5_stacked", "opt/ns")):
            if hasattr(owner, attr):
                saved.append((owner, attr, owner.__dict__[attr]))
                setattr(owner, attr, ranged(name, owner.__dict__[attr]))
        try:
            yield
        finally:
            for owner, attr, f in reversed(saved):
                setattr(owner, attr, f)
    return cm()


# ------------------------------------------------------------------------------------------------- measurement
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="main_2b")
    ap.add_argument("--override", default=None, help="JSON dict of ModelConfig fields (default for main_2b: the main "
                    f"run's {json.dumps(MAIN_OVERRIDE)}; none for other presets)")
    ap.add_argument("--ckpt", default="", help="checkpoint to start from (model, optimizer, Engram tables); else "
                    "random init (routing and Engram rows then differ from the main run)")
    ap.add_argument("--data", default="", help="token bins (as train.py --data); else random tokens")
    ap.add_argument("--mix", default="reasoning")
    ap.add_argument("--canon", default="")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per measured step (the optimizer phase does "
                    "not depend on it; micro-batch 0 of each step is left out of the micro-batch timings)")
    ap.add_argument("--main-accum", type=int, default=64, help="micro-batches per step of the main run, for the "
                    "percentages")
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4, help="timed steps, alternating default order / adamw_last (or "
                    "cycling over --muon-impls)")
    ap.add_argument("--muon-impls", default="", help="comma list of Muon impls to compare (loop,batched,batched_fused); "
                    "empty = the loop in both update orders as before")
    ap.add_argument("--muon-chunk-elems", type=int, default=0, help="Muon chunk size of the batched impls (0 = default)")
    ap.add_argument("--overlap-trials", type=int, default=2)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--ckpt-frac", type=float, default=0.0, help="recompute fraction (main run: 0.0)")
    ap.add_argument("--no-fuse", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="run even if the GPU is in use (do not: the main run can OOM)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "opt_phase"))
    args = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    cuda = device.type == "cuda"
    if cuda and not args.force:
        free, total = torch.cuda.mem_get_info()
        if total - free > 3 * 2 ** 30:
            sys.exit(f"{(total - free) / 2 ** 30:.1f} GB of GPU memory in use: pause the main run first (or --force)")
    if cuda:  # as train.py
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    from moe_llm.config import PRESETS
    from moe_llm.layers import common
    from moe_llm.model import MoELLM
    from moe_llm.optim import OptimizerBundle

    cfg = PRESETS[args.preset]()
    override = json.loads(args.override) if args.override is not None else (
        MAIN_OVERRIDE if args.preset == "main_2b" else {})
    cfg = cfg.replace(**override)
    log(f"building {args.preset} {override} on {device}")
    model = MoELLM(cfg).to(device)
    if model.engram is not None and args.canon and os.path.exists(args.canon):
        model.engram.set_canon_map(np.load(args.canon))
    model.set_checkpointing(args.ckpt_frac)
    if cuda and not args.no_fuse:
        common.set_fuse(True)
    from moe_llm.optim import MUON_CHUNK_ELEMS, MUON_IMPLS
    impls = [x for x in args.muon_impls.split(",") if x]
    if any(x not in MUON_IMPLS for x in impls):
        sys.exit(f"--muon-impls {args.muon_impls}: expected names from {MUON_IMPLS}")
    configs = [(x, False) for x in impls] if impls else [("loop", False), ("loop", True)]
    opt = OptimizerBundle(model, lr=args.lr, weight_decay=0.1, engram_lr_mult=5.0, muon_state_dtype=torch.bfloat16,
                          offload_states=cuda, offload_master=True, fused_grad_acc=True, muon_impl=configs[0][0],
                          muon_chunk_elems=args.muon_chunk_elems or MUON_CHUNK_ELEMS)

    def cname(cfg_):
        return cfg_[0] if impls else ("adamw_last" if cfg_[1] else "default")
    if args.ckpt:
        t = time.time()
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        if model.engram is not None:
            model.engram.load_offload_state_dict(sd["engram_offload"])
        del sd
        log(f"loaded {args.ckpt} ({time.time() - t:.1f}s)")
    opt.begin()
    model.train()
    amp = cuda
    offload = model.engram is not None and model.engram.offload

    if args.data and os.path.isdir(args.data):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from moe_llm.data import MixtureLoader
        from train import MIXES
        loader = MixtureLoader(args.data, MIXES[args.mix], args.seq_len, args.micro_bsz, seed=args.seed)
        source = loader.next_batch
        log(f"data: {args.data} mix {args.mix}")
    else:
        gen = torch.Generator().manual_seed(args.seed)

        def source():
            x = torch.randint(0, cfg.vocab_size, (args.micro_bsz, args.seq_len + 1), generator=gen)
            return x[:, :-1].contiguous(), x[:, 1:].contiguous()
        log("data: random tokens")

    def get():
        x, y = source()
        return (x.pin_memory(), y.pin_memory()) if cuda else (x, y)

    def mark():
        if cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def ms(a, b):
        return a.elapsed_time(b) if cuda else (b - a) * 1e3

    def sync():
        if cuda:
            torch.cuda.synchronize()

    head: list = []
    model.final_norm.register_forward_pre_hook(lambda mod, inp: head.append(mark()))
    rec: dict = {}

    def add(key, v):
        rec.setdefault(key, []).append(v)

    def micro_loop(timed):
        """The micro-batches of one step as train.py runs them (Engram prefetch one ahead, one host sync at the end)."""
        batches = [get() for _ in range(args.accum)]
        if offload:
            model.engram.prefetch(batches[0][0])
        loss_sum = torch.zeros((), device=device)
        marks = []
        for i in range(args.accum):
            if offload and i + 1 < args.accum:
                model.engram.prefetch(batches[i + 1][0])
            x, y = (t.to(device, non_blocking=True) for t in batches[i])
            head.clear()
            a = mark()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, _ = model(x, y)
            b = mark()
            (loss / args.accum).backward()
            c = mark()
            loss_sum = loss_sum + loss.detach().float()
            marks.append((a, head[0] if head else None, b, c))
        finite = bool(torch.isfinite(loss_sum))
        if timed:
            sync()
            for a, h, b, c in marks[1:]:  # micro-batch 0 waits for its Engram rows on the CPU first
                add("fwd_ms", ms(a, b))
                add("bwd_ms", ms(b, c))
                if h is not None:
                    add("fwd_to_final_norm_ms", ms(a, h))
                    add("final_norm_head_ce_ms", ms(h, b))
        return finite

    def finish_step():
        opt.zero_grad()
        model.update_router_bias()
        sync()

    def set_config(cfg_):
        opt.adamw_last = cfg_[1]
        if opt.muon is not None:
            opt.muon.impl = cfg_[0]

    def opt_step(cfg_, timed):
        opt.clip_grad_norm(1.0)
        set_config(cfg_)
        if cuda:
            mb_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        a = mark()
        t0 = time.perf_counter()
        opt.step(args.lr)
        t1 = time.perf_counter()
        b = mark()
        if cuda and timed:  # the allocator counts on the host: no sync needed
            add(f"mem_peak_opt_gb_{cname(cfg_)}", torch.cuda.max_memory_allocated() / 2 ** 30)
            add("mem_peak_before_opt_gb", mb_peak / 2 ** 30)
        eng = None
        if offload:  # the Engram CPU update was queued on the worker at the end of step(); FIFO: this runs after it
            model.engram._worker().submit(lambda: None).result()
            eng = (time.perf_counter() - t1) * 1e3
        finish_step()
        if timed:
            key = cname(cfg_)
            add(f"opt_ms_{key}", ms(a, b))
            add(f"opt_step_cpu_ms_{key}", (t1 - t0) * 1e3)
            if eng is not None:
                add("engram_update_after_step_ms", eng)

    def nograd_forward(xy, key=None):
        x, y = xy
        if offload:
            model.engram.prefetch(x)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.eval()
        try:
            head.clear()
            a = mark()
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                model(x, y)
            b = mark()
        finally:
            model.train()
        if key is not None:
            sync()
            add(key, ms(a, b))
            if head:
                add(key.replace("_ms", "_to_final_norm_ms"), ms(a, head[0]))

    def profile_step(cfg_):
        from torch.profiler import ProfilerActivity, profile
        opt.clip_grad_norm(1.0)
        set_config(cfg_)
        sync()
        acts = [ProfilerActivity.CUDA] + ([ProfilerActivity.CPU] if impls else [])
        with ranges_installed(opt) if impls else _nullctx():
            with profile(activities=acts) as prof:
                opt.step(args.lr)
                torch.cuda.synchronize()
        name = cname(cfg_)
        path = os.path.join(args.out, f"trace_opt_{name}.json")
        prof.export_chrome_trace(path)
        if offload:
            model.engram._worker().submit(lambda: None).result()
        finish_step()
        with open(path) as fh:
            ev = json.load(fh).get("traceEvents", [])
        res = analyze_trace(ev)
        if impls:
            res["breakdown"] = breakdown(ev)
        rec[f"profile_{name}"] = res
        return res

    s_fwd = torch.cuda.Stream() if cuda else None  # one stream: its allocator pool stays warm across trials

    def overlap_trial(timed):
        """opt.step on the compute stream exactly as in training (its ~9 GB of reload buffers come from the warm pool)
        while a no-grad forward of the next batch runs on s_fwd without waiting for any update. The forward reads
        weights while they are being written (values irrelevant here, all finite); its Engram prefetch is queued behind
        the Engram CPU update and it is queued after step() returns, as the next forward would be."""
        nb = get()
        opt.clip_grad_norm(1.0)
        set_config(configs[0])
        torch.cuda.synchronize()
        e0 = mark()
        s_fwd.wait_stream(torch.cuda.current_stream())
        opt.step(args.lr)
        e_opt = mark()
        with torch.cuda.stream(s_fwd):
            nograd_forward(nb)
            e_fwd = mark()
        torch.cuda.synchronize()
        if timed:
            add("overlap_opt_end_ms", e0.elapsed_time(e_opt))
            add("overlap_fwd_end_ms", e0.elapsed_time(e_fwd))
        if offload:
            model.engram._worker().submit(lambda: None).result()
        finish_step()

    log("warm-up (compile, autotune, pinned buffers)")
    n_warm = args.warmup_steps + 1 if not impls else max(args.warmup_steps + 1, len(configs))
    for i in range(n_warm):
        micro_loop(False)
        # without --muon-impls the last warm-up step runs the adamw_last order once; with it every impl runs once
        opt_step(configs[1] if not impls and i == n_warm - 1 else configs[i % len(configs)] if impls else configs[0],
                 False)
    nograd_forward(get())
    sync()
    log("timed steps")
    for s in range(args.steps):
        micro_loop(True)
        opt_step(configs[s % len(configs)], True)
    log("no-grad forward alone")
    for _ in range(3):
        nograd_forward(get(), "fwd_nograd_ms")
    if cuda and not args.no_profile:
        for cfg_ in configs:
            log(f"profiling one optimizer step ({cname(cfg_)})")
            micro_loop(False)
            try:
                profile_step(cfg_)
            except Exception as e:  # keep the other measurements
                log(f"profile failed: {e!r}")
                finish_step()
    if cuda and args.overlap_trials > 0:
        try:
            for i in range(args.overlap_trials + 1):
                log(f"overlap trial {i}/{args.overlap_trials}" + (" (warm-up, not timed)" if i == 0 else ""))
                micro_loop(False)
                overlap_trial(i > 0)
        except Exception as e:  # keep the other measurements
            log(f"overlap trial failed: {e!r}")

    med = {k: statistics.median(v) for k, v in rec.items() if isinstance(v, list) and v}
    if impls and f"opt_ms_{impls[0]}" in med:  # the report's reference is the first impl
        med["opt_ms_default"] = med[f"opt_ms_{impls[0]}"]
        med["opt_step_cpu_ms_default"] = med.get(f"opt_step_cpu_ms_{impls[0]}", float("nan"))
    s = {"settings": {"preset": args.preset, "override": override, "ckpt": args.ckpt, "data": args.data or "random",
                      "accum": args.accum, "main_accum": args.main_accum, "device": str(device), "muon_impls": impls},
         "medians_ms": med, "raw": {k: v for k, v in rec.items() if isinstance(v, list)},
         **{f"profile_{cname(c)}": rec.get(f"profile_{cname(c)}") for c in configs}}
    t_opt, t_fwd, t_bwd = med.get("opt_ms_default"), med.get("fwd_ms"), med.get("bwd_ms")
    lines = []
    if t_opt is not None and t_fwd is not None and t_bwd is not None:
        step_ms = args.main_accum * (t_fwd + t_bwd) + t_opt
        pct = lambda v: 100.0 * v / step_ms  # noqa: E731
        s["main_step_ms_est"] = step_ms
        lines.append(f"micro-batch: forward {t_fwd:.1f} ms (to final norm {med.get('fwd_to_final_norm_ms', 0):.1f}, "
                     f"final norm + LM head/CE {med.get('final_norm_head_ce_ms', 0):.1f}), backward {t_bwd:.1f} ms")
        if impls:
            for x in impls:
                d = t_opt - med.get(f"opt_ms_{x}", float("nan"))
                s[f"muon_{x}_saving_ms"], s[f"muon_{x}_saving_pct"] = d, pct(d)
                lines.append(f"optimizer phase with Muon {x}: {med.get(f'opt_ms_{x}', float('nan')):.1f} ms (GPU), "
                             f"step() on the CPU {med.get(f'opt_step_cpu_ms_{x}', float('nan')):.1f} ms, peak memory "
                             f"{med.get(f'mem_peak_opt_gb_{x}', float('nan')):.2f} GB (micro-batches "
                             f"{max(rec.get('mem_peak_before_opt_gb', [float('nan')])):.2f} GB); saves {d:.1f} ms "
                             f"= {pct(d):.2f}% vs {impls[0]}")
        else:
            lines.append(f"optimizer phase (compute stream): default {t_opt:.1f} ms, adamw_last "
                         f"{med.get('opt_ms_adamw_last', float('nan')):.1f} ms; step() on the CPU "
                         f"{med.get('opt_step_cpu_ms_default', float('nan')):.1f} ms")
        lines.append(f"main step estimate ({args.main_accum} micro-batches + optimizer): {step_ms / 1e3:.2f} s")
        pre = med.get("fwd_to_final_norm_ms")
        if pre is not None:
            s["overlap_structural_bound_ms"] = pre
            s["overlap_structural_bound_pct"] = pct(pre)
            lines.append(f"overlap, structural bound (forward up to the final norm): {pre:.1f} ms = {pct(pre):.2f}% "
                         "of a step")
        if "opt_ms_adamw_last" in med:
            d = t_opt - med["opt_ms_adamw_last"]
            s["adamw_last_saving_ms"], s["adamw_last_saving_pct"] = d, pct(d)
            lines.append(f"adamw_last saves {d:.1f} ms per step = {pct(d):.2f}%")
        if "engram_update_after_step_ms" in med:
            lines.append(f"Engram CPU update after step() returned: {med['engram_update_after_step_ms']:.1f} ms "
                         "(the next forward cannot be queued before it ends)")
        both = [max(a, b) for a, b in zip(rec.get("overlap_opt_end_ms", []), rec.get("overlap_fwd_end_ms", []))]
        if both and "fwd_nograd_ms" in med:
            t_both = statistics.median(both)
            ceil = t_opt + med["fwd_nograd_ms"] - t_both
            s["overlap_ceiling_saving_ms"], s["overlap_ceiling_saving_pct"] = ceil, pct(ceil)
            lines.append(f"overlap ceiling: optimizer alone {t_opt:.1f} + no-grad forward alone "
                         f"{med['fwd_nograd_ms']:.1f} ms, both at once {t_both:.1f} ms -> saves at most {ceil:.1f} ms "
                         f"= {pct(ceil):.2f}% of a step")
    for name in [cname(c) for c in configs]:
        p = rec.get(f"profile_{name}")
        if p:
            lines.append(f"profile ({name}): phase {p['phase_ms']:.1f} ms, kernels busy {p['kernel_busy_ms']:.1f} ms, "
                         f"idle {p['idle_ms']:.1f} ms, first compute kernel after {p['first_compute_kernel_after_ms']:.1f}"
                         f" ms, largest compute gap {p['largest_compute_gap_ms']:.1f} ms at "
                         f"{p['largest_compute_gap_at_ms']:.1f} ms, {p['n_kernels']} kernels, D2H tail "
                         f"{p['d2h_tail_ms']:.1f} ms")
            lines.append("  by category (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in p["categories_ms"].items()))
            lines.append(f"  last reload copy ends at {p.get('h2d_last_end_ms', 0):.1f} ms of {p['phase_ms']:.1f} ms; "
                         f"compute stream busy {p['compute_stream_busy_ms']:.1f} ms")
            bd = p.get("breakdown")
            if bd:
                lines.append("  GPU time by part (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in bd["gpu_ms"].items()))
                lines.append("  CPU time in ranges (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in bd["cpu_range_ms"].items()))
            for t in p["top_kernels"][:8]:
                lines.append(f"    {t['ms']:8.2f} ms x{t['n']:<5d} {t['name']}")
    s["report"] = lines
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print("\n".join(lines), flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
