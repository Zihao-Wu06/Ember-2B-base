#!/usr/bin/env python
"""KDA short convs on the GPU: conv_impl "compiled" (torch.compile'd conv_silu, the running path) vs fla's own kernels
("fla": Triton causal_conv1d, "fla_gluon": fla's Gluon kernels for the same op). Micro-benchmark; the real step is
measured by scripts/ab_conv.py (isolated timings have misled before).

  python scripts/bench_conv.py                  # op + one KDA layer: ~2 GB of GPU memory, < 1 min
  python scripts/bench_conv.py --profile        # + the kernels of one conv forward + backward per impl
  python scripts/bench_conv.py --model          # + main_2b micro-batches as in the main run, one process per mode
                                                #   (needs the whole GPU and ~16 GB of RAM: pause the main run)

Op level: one KDA short conv at main size (x [2, 4096, 1024] bf16, fp32 weight [1024, 1, 4], fusion on). Forward (the
graph recorded, as in training) and forward + backward with CUDA events (the weight gradient accumulating into .grad
as in training), bytes saved for the backward (saved-tensor hooks), error vs fp64 (y, dx, dw), and the projection to
one optimizer step (54 convs x --accum micro-batches; the 2 Engram convs stay compiled).
Layer level: one main_2b KDA layer with the main run's weight handling (persistent bf16 copies + fused fp32 gradient
accumulation), bf16 autocast, fusion on: forward, forward + backward, memory held for the backward, peak.
--model: main_2b with the main run's settings (moe_impl triton, ce_impl fused, ce_blas cublaslt_logits, attnres_impl
fused, kda_keep_layers 6, fusion on, no recompute, offload_master + fused_grad_acc + offload_opt, bf16 Muon momentum,
Engram tables in CPU memory), synthetic tokens (or --ckpt weights); per mode a fresh process runs --warmup + --micro
micro-batches (timed together, as train.py runs them) and one optimizer step. Prints s per micro-batch, tok/s, the
optimizer step and peak memory, and the step projection.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm.config import main_2b  # noqa: E402
from moe_llm.layers import common  # noqa: E402
from moe_llm.layers.common import CastLinear, conv_silu, fla_conv_silu  # noqa: E402

BF16 = torch.bfloat16
IMPLS = ("compiled", "fla", "fla_gluon")


def time_ms(fn, reps, warmup):
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


def available(impls):
    """The impls that run here: each fla path must load and run a small forward + backward (fla's Gluon kernels may not
    compile for every GPU / Triton)."""
    out = []
    for impl in impls:
        if impl == "compiled":
            out.append(impl)
            continue
        try:
            x = torch.randn(1, 200, 96, device="cuda", dtype=BF16, requires_grad=True)
            w = (torch.randn(96, 1, 4, device="cuda") * 0.3).requires_grad_()
            fla_conv_silu(x, w, impl).float().square().sum().backward()
            torch.cuda.synchronize()
            out.append(impl)
        except Exception as e:
            print(f"{impl}: not available ({e!r})")
    return out


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def ref64(x, w2, dy):
    """fp64 y = silu(conv(x)) and the exact dx, dw for dy (w2: [D, k], the bf16-rounded weight every path uses)."""
    x, w, dy = x.double(), w2.double(), dy.double()
    T, k = x.shape[1], w.shape[-1]
    xp = F.pad(x, (0, 0, k - 1, 0))
    c = sum(xp[:, i:i + T] * w[:, i] for i in range(k))
    s = torch.sigmoid(c)
    dc = dy * s * (1 + c * (1 - s))
    dcp = F.pad(dc, (0, 0, 0, k - 1))
    dx = sum(dcp[:, k - 1 - i:k - 1 - i + T] * w[:, i] for i in range(k))
    dw = torch.stack([(dc * xp[:, i:i + T]).sum((0, 1)) for i in range(k)], -1)
    return c * s, dx, dw


def conv_fn(impl):
    return conv_silu if impl == "compiled" else (lambda x, w: fla_conv_silu(x, w, impl))


def fla_configs():
    """The launch configs fla's autotuners picked for the Triton conv kernels (after a run), for the record."""
    try:
        from fla.modules.causal_conv1d import ops
        out = {}
        for name in ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"):
            at = getattr(ops, name).fn
            out[name] = {str(k): str(v) for k, v in at.cache.items()}
        return out
    except Exception as e:
        return {"error": repr(e)}


# ----------------------------------------------------------------------------- op level
def op_case(impl, x, w, dy, reps, warmup, profile):
    f = conv_fn(impl)
    xi = x.detach().clone().requires_grad_()
    wi = w.detach().clone().requires_grad_()

    def fwd():
        f(xi, wi)

    def step():
        xi.grad = None  # the conv input's gradient goes on to the projection (not accumulated); the weight's accumulates
        f(xi, wi).backward(dy)

    t0 = time.perf_counter()
    step()  # compile / autotune
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    t_f, t_fb = time_ms(fwd, reps, warmup), time_ms(step, reps, warmup)
    xi.grad, wi.grad = None, None
    saved = {}  # storage pointer -> bytes of every tensor saved for the backward (in the model the conv input is
    #             a projection output nothing else keeps, so saving it costs its bytes, like any other saved tensor)

    def pack(t):
        s = t.untyped_storage()
        saved[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = f(xi, wi)
    held = sum(saved.values())
    y.backward(dy)
    out = (y.detach(), xi.grad, wi.grad[:, 0])
    if profile:
        from torch.profiler import ProfilerActivity
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        print(f"\n--- {impl}: kernels of one conv forward + backward")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=12, max_name_column_width=90))
    return t_f, t_fb, first, held, out


# ----------------------------------------------------------------------------- layer level
class Wrap(nn.Module):
    """One layer as a 'model' for OptimizerBundle (cast_params / cfg, as MoELLM has)."""

    def __init__(self, layer, cfg):
        super().__init__()
        self.layer, self.cfg = layer, cfg

    def cast_params(self):
        return [p for m in self.layer.modules() if isinstance(m, CastLinear)
                for p in [m.weight] + ([m.bias] if m.bias is not None else [])]

    def forward(self, x):
        return self.layer(x)


def layer_case(impl, x, dy, reps, warmup):
    from moe_llm.layers.kda import KDA
    from moe_llm.optim import OptimizerBundle
    cfg = main_2b().replace(conv_impl=impl)
    torch.manual_seed(0)
    m = Wrap(KDA(cfg, 1).cuda(), cfg)
    opt = OptimizerBundle(m, lr=1e-4, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=True)
    opt.begin()
    xi = x.detach().requires_grad_()

    def fwd():
        with torch.autocast("cuda", dtype=BF16):
            m(xi)

    def step():
        xi.grad = None
        with torch.autocast("cuda", dtype=BF16):
            y = m(xi)
        y.backward(dy)

    step()
    torch.cuda.synchronize()
    t_f, t_fb = time_ms(fwd, reps, warmup), time_ms(step, reps, warmup)
    xi.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=BF16):
        y = m(xi)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base
    y.backward(dy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del opt, m, xi, y
    common.clear_cached_copies()
    gc.collect()
    torch.cuda.empty_cache()
    return t_f, t_fb, held, peak


# ----------------------------------------------------------------------------- whole model (one mode per process)
def model_one(args):
    """One --model mode in this process; prints one JSON line with the result."""
    from moe_llm.model import MoELLM
    from moe_llm.optim import OptimizerBundle
    torch.manual_seed(0)
    over = json.loads(args.override)
    cfg = main_2b().replace(**over, conv_impl=args.model_one)
    model = MoELLM(cfg).cuda()
    if args.ckpt:  # as train.py resumes: mmap, copied into place
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(sd["model"])
        if model.engram is not None and sd.get("engram_offload") is not None:
            model.engram.load_offload_state_dict(sd["engram_offload"])
        del sd
    model.set_checkpointing(args.ckpt_frac, expert_recompute=True)
    common.set_fuse(True)
    opt = OptimizerBundle(model, lr=1e-4, muon_state_dtype=BF16, offload_states=True, offload_master=True,
                          fused_grad_acc=True)
    opt.begin()
    offload = model.engram is not None and model.engram.offload
    g = torch.Generator().manual_seed(0)
    xs = [torch.randint(0, cfg.vocab_size, (args.micro_bsz, args.seq_len + 1), generator=g).pin_memory()
          for _ in range(args.warmup + args.micro)]
    n = len(xs)

    def micro(i):
        if offload and i + 1 < n:
            model.engram.prefetch(xs[i + 1][:, :-1])
        x = xs[i].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=BF16):
            loss, _ = model(x[:, :-1], x[:, 1:])
        (loss / n).backward()

    if offload:
        model.engram.prefetch(xs[0][:, :-1])
    for i in range(args.warmup):  # compile, autotune, allocate the gradient buffers
        micro(i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(args.warmup, n):
        micro(i)
    torch.cuda.synchronize()
    t_micro = (time.perf_counter() - t0) / args.micro
    t1 = time.perf_counter()
    opt.step(1e-4)
    torch.cuda.synchronize()
    t_opt = time.perf_counter() - t1
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    if offload:
        model.engram.flush()
    print("RESULT " + json.dumps({"mode": args.model_one, "t_micro": t_micro, "t_opt": t_opt, "peak_gb": peak,
                                  "failed_compiles": sorted({k[0] for k in common._FAILED})}), flush=True)


def model_modes(args):
    res = {}
    for mode in args.modes.split(","):
        print(f"\nmain_2b, conv_impl {mode} (own process) ...", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--model-one", mode, "--micro", str(args.micro),
               "--warmup", str(args.warmup), "--ckpt-frac", str(args.ckpt_frac), "--seq-len", str(args.seq_len),
               "--micro-bsz", str(args.micro_bsz), "--override", args.override] + (["--ckpt", args.ckpt] if args.ckpt
                                                                                      else [])
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if p.returncode != 0 or line is None:
            print(f"  FAILED (exit {p.returncode}): {(p.stdout + p.stderr).strip()[-1500:]}", flush=True)
            continue
        r = json.loads(line[len("RESULT "):])
        res[mode] = r
        tps = args.micro_bsz * args.seq_len / r["t_micro"]
        print(f"  {r['t_micro']:.4f} s per micro-batch ({tps:.0f} tok/s fwd+bwd), optimizer step {r['t_opt']:.2f} s, "
              f"peak {r['peak_gb']:.2f} GB" + (f", torch.compile FAILED for {r['failed_compiles']}"
                                               if r["failed_compiles"] else ""), flush=True)
    if "compiled" in res:
        a = res["compiled"]["t_micro"] * args.accum + res["compiled"]["t_opt"]
        for mode, r in res.items():
            if mode != "compiled":
                b = r["t_micro"] * args.accum + r["t_opt"]
                print(f"per optimizer step ({args.accum} micro-batches + step): compiled {a:.2f} s -> {mode} {b:.2f} s "
                      f"({(a / b - 1) * 100:+.2f}% speed); peak {res['compiled']['peak_gb']:.2f} -> {r['peak_gb']:.2f} GB")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default=",".join(IMPLS))
    ap.add_argument("--accum", type=int, default=64, help="micro-batches per optimizer step (main run: 64)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup-reps", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-layer", action="store_true", help="skip the KDA layer part")
    ap.add_argument("--model", action="store_true", help="also time main_2b micro-batches, one process per mode")
    ap.add_argument("--modes", default="compiled,fla", help="--model: conv_impl values, in this order")
    ap.add_argument("--micro", type=int, default=8, help="--model: timed micro-batches per mode")
    ap.add_argument("--warmup", type=int, default=3, help="--model: untimed micro-batches first")
    ap.add_argument("--ckpt-frac", type=float, default=0.0, help="--model: fraction of layers recomputed (main run: 0)")
    ap.add_argument("--ckpt", default="", help="--model: load model weights from this train.py checkpoint")
    ap.add_argument("--override", default=json.dumps({"moe_impl": "triton", "ce_impl": "fused", "attnres_impl": "fused",
                                                      "ce_blas": "cublaslt_logits", "kda_keep_layers": 6}),
                    help="--model: ModelConfig fields of the main run (conv_impl is set per mode)")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-bsz", type=int, default=2)
    ap.add_argument("--model-one", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("bench_conv.py needs CUDA")
    if args.model_one:
        return model_one(args)
    impls = available(args.impls.split(","))
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}; conv at main size [{args.micro_bsz}, "
          f"{args.seq_len}, 1024] bf16, kernel 4; {args.accum} micro-batches per step")
    common.set_fuse(True)
    try:
        g = torch.Generator().manual_seed(0)
        B, T, D = args.micro_bsz, args.seq_len, 1024
        x = torch.randn(B, T, D, generator=g).to("cuda", BF16)
        w = (torch.randn(D, 1, 4, generator=g) * 0.3).cuda()
        dy = torch.randn(B, T, D, generator=g).to("cuda", BF16)
        ref = ref64(x, w[:, 0].to(BF16), dy)
        n_conv = 3 * sum(t == "K" for t in main_2b().layer_types())
        print(f"\nop (one short conv), ms:  {'impl':>9} {'fwd':>7} {'bwd':>7} {'fwd+bwd':>8} {'saved MiB':>9} "
              f"{'first s':>7}   err vs fp64: y / dx / dw")
        res = {}
        for impl in impls:
            t_f, t_fb, first, held, out = op_case(impl, x, w, dy, args.reps, args.warmup_reps, args.profile)
            res[impl] = t_fb
            e = [rel_rms(o, r) for o, r in zip(out, ref)]
            print(f"{'':25s}{impl:>9} {t_f:7.3f} {t_fb - t_f:7.3f} {t_fb:8.3f} {held / 2 ** 20:9.1f} {first:7.1f}   "
                  f"{e[0]:.3e} / {e[1]:.3e} / {e[2]:.3e}", flush=True)
        for impl in impls:
            if impl != "compiled" and "compiled" in res:
                s = res["compiled"] - res[impl]
                print(f"  {impl}: saves {s:.3f} ms per conv -> {s * n_conv * args.accum / 1e3:.3f} s per step "
                      f"({n_conv} convs x {args.accum})")
        if "fla" in impls:
            print(f"  fla Triton launch configs: {fla_configs()}")
        if not args.no_layer:
            xl = torch.randn(B, T, D, generator=g).to("cuda", BF16)
            dyl = (torch.randn(B, T, D, generator=g) * 0.01).to("cuda", BF16)
            n_kda = n_conv // 3
            print(f"\nmain_2b KDA layer (persistent copies + fused fp32 grad accumulation), ms per layer and micro-batch:")
            print(f"{'impl':>11} {'fwd':>8} {'bwd':>8} {'fwd+bwd':>8} {'held MiB':>9} {'peak MiB':>9}")
            lres = {}
            for impl in impls:
                t_f, t_fb, held, peak = layer_case(impl, xl, dyl, args.reps, args.warmup_reps)
                lres[impl] = (t_fb, held, peak)
                print(f"{impl:>11} {t_f:8.3f} {t_fb - t_f:8.3f} {t_fb:8.3f} {held / 2 ** 20:9.1f} {peak / 2 ** 20:9.1f}",
                      flush=True)
            for impl in impls:
                if impl != "compiled" and "compiled" in lres:
                    s = lres["compiled"][0] - lres[impl][0]
                    dh = (lres[impl][1] - lres["compiled"][1]) * n_kda / 2 ** 30
                    print(f"  {impl}: saves {s:.3f} ms per layer -> {s * n_kda * args.accum / 1e3:.3f} s per step; "
                          f"held for the backward {dh:+.3f} GiB over {n_kda} KDA layers")
        if common._FAILED:
            print(f"WARNING: torch.compile failed (eager fallback) for {sorted({k[0] for k in common._FAILED})}")
    finally:
        common.set_fuse(False)
    if args.model:
        model_modes(args)


if __name__ == "__main__":
    main()
