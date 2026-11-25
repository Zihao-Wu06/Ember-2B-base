#!/usr/bin/env python
"""Sweep launch configs of fla's KDA Triton kernels at the main-run shape; write pins for layers/kda_tuning.py.

  python scripts/sweep_kda.py --out runs/kda_sweep     # GPU, nothing else on it; about 5-10 min (mostly compiling)
  python scripts/sweep_kda.py --check                  # CPU only: resolve the kernel table against the installed fla

Shape and flags exactly as layers/kda.py calls chunk_kda: B=2, T=4096, H=8, K=V=128, bf16 q/k/v/g/beta, fp32
A_log/dt_bias, chunk 64, use_qk_l2norm_in_kernel / use_gate_in_kernel / use_beta_sigmoid_in_kernel.

  1. one fwd+bwd with fla's own autotuning (what the main run uses: Triton's on-disk autotune cache is keyed on head
     sizes only, so those winners were tuned at micro-batch 1) while recording every KDA kernel launch
  2. per kernel and autotune key: time fla's config space plus an extended sm89 space on the recorded launch
     (triton.testing.do_bench, L2 flushed, median), re-time the top 3 and the default longer, keep the winner as a
     candidate only if it is at least --min-gain faster than the default
  3. accuracy: o and the 7 gradients vs fla's defaults and vs kda.py's fp32 reference (its token loop, checkpointed
     in segments), per candidate and all together; a candidate stays only if no error vs the reference grows by
     more than --acc-tol (relative)
  4. timing of the whole chunk_kda fwd+bwd (CUDA events, interleaved) and per kernel inside it (torch.profiler),
     defaults vs pins; then the written JSON is applied the way training applies it and checked to hit
  5. extra: disable_recompute=True (ModelConfig.kda_keep_layers; the backward reuses the forward's states instead
     of recomputing them): time saved and memory held per keeping layer, bitwise check, and how many layers fit
     under --cap-gib given --peak-gib
Writes OUT/kda_tuning.json (pins: MOE_KDA_TUNING=<it> or ModelConfig.kda_tuning), OUT/sweep_results.json, and
OUT/summary.txt. Partial results are saved after each stage.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from moe_llm.layers import kda_tuning  # noqa: E402  (no torch / fla import at module level)

OUT_NAMES = ["o", "dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias"]


def C(num_warps, num_stages, **kwargs):
    return {"kwargs": kwargs, "num_warps": num_warps, "num_stages": num_stages}


# (module, kernel, extension configs tried on top of fla's own space). fla's space on the 4090 (101376 B of shared
# memory: check_shared_mem() False, check_shared_mem('ada') True, check_shared_mem('ampere') False) is in the comments.
KERNELS = [
    # fla: BK, BV in {16, 32} x warps {2,4,8} x stages {2,3,4}; stages make no difference (loop not pipelined)
    ("fla.ops.kda.chunk_bwd", "chunk_kda_bwd_kernel_wy_dqkg_fused",
     [C(w, 2, BK=bk, BV=bv) for bk in (16, 32, 64) for bv in (64, 128) for w in (4, 8)]
     + [C(w, 2, BK=64, BV=32) for w in (4, 8)] + [C(8, 2, BK=128, BV=bv) for bv in (32, 64)]
     + [C(4, 1, BK=16, BV=32), C(2, 2, BK=16, BV=64)]),
    # fla: BV in {32, 64} x warps {2,4} x stages {1}: only N*H*V/BV = 32 or 64 CTAs on 128 SMs, no pipelining
    ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
     [C(w, s, BV=16) for w in (2, 4, 8) for s in (1, 2, 3)]
     + [C(w, s, BV=bv) for bv in (32, 64) for w in (2, 4, 8) for s in (2, 3)] + [C(8, 1, BV=bv) for bv in (32, 64)]),
    # fla: BV in {32, 64} x warps {2,4} x stages {1,2}
    ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
     [C(w, s, BV=16) for w in (2, 4, 8) for s in (1, 2, 3)]
     + [C(8, s, BV=bv) for bv in (32, 64) for s in (2, 3)] + [C(4, 3, BV=bv) for bv in (32, 64)]),
    # fla: warps {1,2,4,8} x stages {2,3,4}; BK=32 and BC=16 are fixed by fla's wrapper (not tunable here)
    ("fla.ops.kda.chunk_intra", "chunk_kda_bwd_kernel_intra", [C(w, 1) for w in (1, 2, 4)]),
    # fla: BK in {32, 64} x warps {1,2,4}, stages 3
    ("fla.ops.kda.chunk_intra", "chunk_kda_fwd_kernel_inter_solve_fused",
     [C(w, s, BK=32) for w in (1, 2) for s in (1, 2, 4)] + [C(1, s, BK=64) for s in (1, 2)]),
    # fla: BK {32,64} x BV {64,128} x warps {2,4,8} x stages {2,3,4}; flat landscape, near the DRAM roofline
    ("fla.ops.gla.chunk", "chunk_gla_fwd_kernel_o",
     [C(w, 2, BK=128, BV=bv) for bv in (64, 128) for w in (4, 8)]
     + [C(w, 2, BK=bk, BV=32) for bk in (32, 64) for w in (2, 4)]),
    # fla: BH {1,2,4,8} x warps {1,2,4,8}, stages 3
    ("fla.ops.kda.chunk_intra_token_parallel", "chunk_kda_fwd_kernel_intra_token_parallel",
     [C(w, s, BH=bh) for bh in (2, 4) for w in (2, 4) for s in (1, 2)]),
    # fla's own space only (memory-bound, at or near the DRAM roofline; re-timed at the real shape)
    ("fla.ops.kda.wy_fast", "recompute_w_u_fwd_kda_kernel", []),      # BK=BV=64 fixed by the wrapper
    ("fla.ops.kda.chunk_bwd", "chunk_kda_bwd_kernel_dAv", []),        # BK=BV=64 fixed by the wrapper
    ("fla.ops.kda.gate", "kda_gate_chunk_cumsum_vector_kernel", []),
    ("fla.ops.kda.gate", "kda_gate_bwd_kernel", []),
    ("fla.ops.utils.cumsum", "chunk_local_cumsum_vector_kernel", []),
    ("fla.modules.norm.l2norm", "l2norm_fwd_kernel", []),
    ("fla.modules.norm.l2norm", "l2norm_bwd_kernel", []),
]


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- autotuner helpers (CPU-testable)
def autotune_key(tuner, args, kwargs) -> tuple:
    """Triton 3.9 Autotuner.run's tuning key: the key args' values, then the dtype of every tensor argument."""
    all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
    named = {k: v for k, v in all_args.items() if k in tuner.arg_names}
    key = [named[k] for k in tuner.keys if k in named]
    key += [str(v.dtype) for v in named.values() if hasattr(v, "dtype")]
    return tuple(key)


def cfg_id(cfg) -> tuple:
    return tuple(sorted(cfg.kwargs.items())), int(cfg.num_warps), int(cfg.num_stages)


def key_str(key) -> str:
    """Compact autotune key: its non-dtype values, then the number of tensor arguments."""
    vals = [("T" if v else "F") if isinstance(v, bool) else str(v) for v in key if not isinstance(v, str)]
    return ",".join(vals) + f"|{sum(isinstance(v, str) for v in key)}t"


def cfg_str(cfg) -> str:
    kw = ",".join(f"{k}={v}" for k, v in sorted(cfg.kwargs.items()))
    return f"{kw + ' ' if kw else ''}w{cfg.num_warps} s{cfg.num_stages}"


def build_space(tuner, extension: list[dict]) -> list[tuple]:
    """[(config, in_fla_space)], fla's configs first, extension configs not already in fla's space after."""
    seen, out = set(), []
    for cfg in tuner.configs:
        if cfg_id(cfg) not in seen:
            seen.add(cfg_id(cfg))
            out.append((cfg, True))
    for spec in extension:
        cfg = kda_tuning.make_config(tuner, spec)
        if cfg_id(cfg) not in seen:
            seen.add(cfg_id(cfg))
            out.append((cfg, False))
    return out


class Recorder:
    """Wraps each autotuner's run() to record the first launch per autotune key (args kept alive for replays) and
    count launches per key. Replays bypass the wrapper (call_with)."""

    def __init__(self, tuners: dict):
        self.tuners = tuners          # kernel name -> autotuner
        self.calls: dict = {}         # (kernel, key) -> {"args", "kwargs", "count"}
        self.enabled = True

    def install(self):
        for name, tuner in self.tuners.items():
            tuner.run = self._wrap(name, tuner)
        return self

    def _wrap(self, name, tuner):
        orig = type(tuner).run

        def run(*args, **kwargs):
            if self.enabled:
                key = autotune_key(tuner, args, kwargs)
                rec = self.calls.get((name, key))
                if rec is None:
                    self.calls[(name, key)] = {"args": args, "kwargs": dict(kwargs), "count": 1}
                else:
                    rec["count"] += 1
            return orig(tuner, *args, **kwargs)
        return run

    def uninstall(self):
        for tuner in self.tuners.values():
            tuner.__dict__.pop("run", None)

    def reset(self):
        self.calls = {}


def call_with(tuner, rec, cfg):
    """Launch the recorded call once with config cfg (a one-config autotuner launches it without benchmarking)."""
    saved = tuner.configs
    tuner.configs = [cfg]
    try:
        type(tuner).run(tuner, *rec["args"], **rec["kwargs"])
    finally:
        tuner.configs = saved


def select_best(timings: list[dict], default_ms: float, min_gain: float):
    """timings: [{"cfg", "ms"}] (ms None = failed). Returns (best entry, gain vs default, is_candidate)."""
    ok = [t for t in timings if t["ms"] is not None and math.isfinite(t["ms"])]
    if not ok or default_ms is None:
        return None, 0.0, False
    best = min(ok, key=lambda t: t["ms"])
    gain = (default_ms - best["ms"]) / default_ms
    return best, gain, gain >= min_gain and not best.get("is_default", False)


# --------------------------------------------------------------------------- accuracy (CPU-testable)
def rel_rms(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def errors(outs, ref) -> dict:
    return {n: {"rel_rms": rel_rms(a, b), "max_abs": (a.float() - b.float()).abs().max().item()}
            for n, a, b in zip(OUT_NAMES, outs, ref)}


def worst_ratio(errs_by_seed: list[dict], base_by_seed: list[dict]) -> tuple[float, str]:
    """Largest ratio, over the 8 tensors, of the case's error vs the fp32 reference to fla's defaults' error, each
    error being the RMS over seeds of rel_rms (pooling seeds keeps 8-value dA_log from deciding by chance)."""
    worst, where = 0.0, ""
    for n in OUT_NAMES:
        e = math.sqrt(sum(x[n]["rel_rms"] ** 2 for x in errs_by_seed) / len(errs_by_seed))
        b = math.sqrt(sum(x[n]["rel_rms"] ** 2 for x in base_by_seed) / len(base_by_seed))
        r = e / max(b, 1e-30)
        if r > worst:
            worst, where = r, n
    return worst, where


def _ref_segment(q, k, v, g, b, S0, scale):
    from moe_llm.layers.kda import naive_recurrent_kda
    return naive_recurrent_kda(q, k, v, g, b, scale=scale, initial_state=S0, output_final_state=True)


def reference_fwd_bwd(inputs, do, seg: int = 256):
    """kda.py's fp32 reference (kda_reference: gate, sigmoid, l2norm, then naive_recurrent_kda's token loop), with
    the loop run in checkpointed segments chained through the state so the backward fits in memory. Returns
    [o, dq, dk, dv, dg, dbeta, dA_log, ddt_bias] in fp32."""
    import torch
    from torch.utils.checkpoint import checkpoint

    from moe_llm.layers.kda import kda_gate, l2norm
    leaves = [t.detach().float().requires_grad_() for t in inputs]
    q, k, v, g, beta, A_log, dt_bias = leaves
    gk, b, qn, kn = kda_gate(g, A_log, dt_bias), torch.sigmoid(beta), l2norm(q), l2norm(k)
    T, K = q.shape[1], q.shape[-1]
    S, outs = None, []
    for t0 in range(0, T, seg):
        sl = slice(t0, min(T, t0 + seg))
        o_s, S = checkpoint(_ref_segment, qn[:, sl], kn[:, sl], v[:, sl], gk[:, sl], b[:, sl], S, K ** -0.5,
                            use_reentrant=False)
        outs.append(o_s)
    o = torch.cat(outs, 1)
    grads = torch.autograd.grad(o, leaves, do.float())
    return [o.detach()] + [x.detach() for x in grads]


def make_inputs(B, T, H, K, seed, device):
    """bf16 q/k/v/g/beta and fp32 A_log/dt_bias like layers/kda.py (A_log and dt_bias drawn as at init)."""
    import torch
    gen = torch.Generator(device=device).manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=gen, device=device)
    q, k, v, g = (r(B, T, H, K).to(torch.bfloat16) for _ in range(4))
    beta = r(B, T, H).to(torch.bfloat16)
    A_log = torch.log(torch.empty(H, device=device).uniform_(1, 16, generator=gen))
    dt = torch.exp(torch.rand(H * K, generator=gen, device=device) * (math.log(0.1) - math.log(0.001))
                   + math.log(0.001)).clamp(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    do = r(B, T, H, K).to(torch.bfloat16)
    return [t.requires_grad_() for t in (q, k, v, g, beta, A_log, dt_bias)], do


def fla_fwd_bwd(chunk_kda, inputs, do, **kw):
    import torch
    q, k, v, g, beta, A_log, dt_bias = inputs
    o, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, use_qk_l2norm_in_kernel=True,
                     use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, **kw)
    grads = torch.autograd.grad(o, inputs, do)
    return [o.detach()] + [x.detach() for x in grads]


def build_pin(module, kernel, key, cfg, **info) -> dict:
    return {"module": module, "kernel": kernel, "key": kda_tuning.key_to_json(key),
            "config": kda_tuning.config_to_dict(cfg), **info}


# --------------------------------------------------------------------------- --check (CPU)
def resolve_kernels():
    import importlib
    out = {}
    for module, kernel, ext in KERNELS:
        tuner = kda_tuning.find_autotuner(getattr(importlib.import_module(module), kernel))
        out[kernel] = (module, tuner, ext)
    return out


def check_table() -> list[str]:
    lines = []
    for kernel, (module, tuner, ext) in resolve_kernels().items():
        space = build_space(tuner, ext)
        n_new = sum(1 for _, in_fla in space if not in_fla)
        lines.append(f"{kernel:52s} key={tuner.keys} fla configs={len(tuner.configs)} new={n_new}")
    return lines


# --------------------------------------------------------------------------- GPU sweep
class Sweep:
    def __init__(self, a):
        import torch
        self.a, self.torch = a, torch
        self.t0 = time.time()
        self.res: dict = {"args": vars(a), "started": datetime.datetime.now().isoformat(timespec="seconds")}
        self.cands: list = []
        os.makedirs(a.out, exist_ok=True)

    def elapsed_min(self):
        return (time.time() - self.t0) / 60

    def save(self):
        with open(os.path.join(self.a.out, "sweep_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.res, f, indent=1, default=str)

    # ---------------------------------------------------------------- setup
    def setup(self):
        torch, a = self.torch, self.a
        import fla
        import triton
        torch.backends.cuda.matmul.allow_tf32 = False      # the fp32 reference must be fp32
        torch.backends.cudnn.allow_tf32 = False
        free, total = torch.cuda.mem_get_info()
        log(f"GPU {torch.cuda.get_device_name()} sm{''.join(map(str, torch.cuda.get_device_capability()))}, "
            f"free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
        if free < a.min_free_gb * 2**30 and not a.force:
            raise SystemExit(f"only {free / 2**30:.1f} GiB free: is training paused? (--force to run anyway)")
        from fla.ops.kda import chunk_kda
        from fla.utils import check_shared_mem, get_all_max_shared_mem
        self.chunk_kda = chunk_kda
        self.env = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                    "fla": fla.__version__, "triton": triton.__version__, "torch": torch.__version__}
        self.res["env"] = dict(self.env, max_shared_mem=get_all_max_shared_mem()[0],
                               check_shared_mem={"default(102400)": check_shared_mem(),
                                                 "ada(101376)": check_shared_mem("ada"),
                                                 "ampere(166912)": check_shared_mem("ampere")})
        log(f"fla {fla.__version__}, triton {triton.__version__}, torch {torch.__version__}; "
            f"max_shared_mem {self.res['env']['max_shared_mem']} B, check_shared_mem {self.res['env']['check_shared_mem']}")
        self.kernels = resolve_kernels()
        self.tuners = {k: t for k, (_, t, _) in self.kernels.items()}
        self.res["shape"] = {"B": a.B, "T": a.T, "H": a.H, "K": a.K, "V": a.K, "dtype": "bfloat16", "chunk_size": 64}
        self.inputs = [make_inputs(a.B, a.T, a.H, a.K, seed, "cuda") for seed in range(a.seeds)]

    # ---------------------------------------------------------------- 1. default pass + recording
    def default_pass(self):
        torch = self.torch
        log("default pass (fla's own autotuning / Triton disk cache), recording launches")
        self.rec = Recorder(self.tuners).install()
        t = time.time()
        ins, do = self.inputs[0]
        fla_fwd_bwd(self.chunk_kda, ins, do)
        torch.cuda.synchronize()
        self.rec.enabled = False
        log(f"  done in {time.time() - t:.1f}s; {len(self.rec.calls)} (kernel, key) launches recorded")
        self.default_cfg = {}
        rows = []
        for (name, key), rc in self.rec.calls.items():
            tuner = self.tuners[name]
            cfg = tuner.configs[0] if len(tuner.configs) == 1 else tuner.cache.get(key)
            if cfg is None:
                raise RuntimeError(f"{name}: recorded key {key} not in the autotuner cache; autotune_key() no "
                                   "longer matches this Triton's Autotuner.run")
            self.default_cfg[(name, key)] = cfg
            rows.append({"kernel": name, "key": kda_tuning.key_to_json(key), "calls_per_fwd_bwd": rc["count"],
                         "default": cfg_str(cfg), "fla_space": len(self.tuners[name].configs)})
            log(f"  {name:52s} x{rc['count']}  default {cfg_str(cfg)}  (fla space {len(self.tuners[name].configs)})")
        self.res["default_pass"] = rows
        self.save()

    # ---------------------------------------------------------------- 2. per-kernel timing
    def bench(self, tuner, rc, cfg, rep_ms):
        import triton.testing
        call_with(tuner, rc, cfg)                       # compile + first launch (raises on OOR / compile errors)
        self.torch.cuda.synchronize()
        return triton.testing.do_bench(lambda: call_with(tuner, rc, cfg), warmup=self.a.warmup_ms, rep=rep_ms,
                                       return_mode="median")

    def time_kernels(self):
        a = self.a
        only = set(a.kernels.split(",")) if a.kernels else None
        self.kres = {}
        order = {k: i for i, (_, k, _) in enumerate(KERNELS)}     # most promising first (time budget)
        for (name, key), rc in sorted(self.rec.calls.items(), key=lambda kv: order[kv[0][0]]):
            if only and name not in only:
                continue
            module, tuner, ext = self.kernels[name]
            dflt = self.default_cfg[(name, key)]
            space = build_space(tuner, [] if a.no_extend else ext)
            if cfg_id(dflt) not in {cfg_id(c) for c, _ in space}:
                space.insert(0, (dflt, True))
            log(f"{name} key {key_str(key)}: {len(space)} configs ({sum(not f for _, f in space)} new)")
            timings = []
            for cfg, in_fla in space:
                entry = {"cfg": cfg, "config": cfg_str(cfg), "fla_space": in_fla, "is_default": cfg_id(cfg) == cfg_id(dflt),
                         "ms": None, "error": None}
                if self.elapsed_min() > a.budget_min and not entry["is_default"]:
                    entry["error"] = "skipped (time budget)"
                    timings.append(entry)
                    continue
                t = time.time()
                try:
                    entry["ms"] = self.bench(tuner, rc, cfg, a.rep_ms)
                except Exception as e:  # OutOfResources, CompilationError, PTXASError, ...
                    msg = str(e)
                    if "CUDA error" in msg or "illegal" in msg.lower():
                        raise                     # the CUDA context is gone: stop (partial results are saved)
                    entry["error"] = f"{type(e).__name__}: {msg.splitlines()[0][:160] if msg else ''}"
                entry["wall_s"] = round(time.time() - t, 2)
                timings.append(entry)
            ok = sorted((t for t in timings if t["ms"] is not None), key=lambda t: t["ms"])
            d = next(t for t in timings if t["is_default"])
            if d["ms"] is None:
                log(f"  default config failed to run ({d['error']}); skipping this kernel")
                continue
            refined = list({id(t): t for t in ok[:3] + [d]}.values())   # re-time the top 3 and the default longer
            for t in refined:
                t["ms_coarse"] = t["ms"]
                try:
                    t["ms"] = self.bench(tuner, rc, t["cfg"], a.refine_ms)
                except Exception as e:
                    log(f"  re-timing {t['config']} failed ({type(e).__name__}); keeping the first timing")
            best, gain, cand = select_best(refined, d["ms"], a.min_gain)
            n_fail = sum(1 for t in timings if t["error"] and not t["error"].startswith("skipped"))
            est = (d["ms"] - best["ms"]) * rc["count"] * a.calls_per_step / 1e3
            log(f"  default {d['config']} {d['ms'] * 1e3:.1f} us | best {best['config']} {best['ms'] * 1e3:.1f} us "
                f"({gain * 100:+.1f}%, {est:.3f} s/step){' -> candidate' if cand else ''} | {n_fail} failed")
            self.kres[(name, key)] = {"module": module, "timings": timings, "default": d, "best": best, "gain": gain,
                                      "candidate": cand, "calls": rc["count"]}
            self.res.setdefault("kernels", []).append({
                "kernel": name, "key": kda_tuning.key_to_json(key), "calls_per_fwd_bwd": rc["count"],
                "default": d["config"], "default_us": d["ms"] * 1e3, "best": best["config"], "best_us": best["ms"] * 1e3,
                "gain": gain, "est_s_per_step": est, "candidate": cand, "failed": n_fail,
                "timings": [{k: v for k, v in t.items() if k != "cfg"} for t in timings]})
            self.save()

    # ---------------------------------------------------------------- 3. accuracy
    def run_case(self, pins, **kw):
        if pins:
            kda_tuning.apply({"format": kda_tuning.FORMAT, "pins": pins}, strict=False, verbose=False)
        try:
            return [fla_fwd_bwd(self.chunk_kda, ins, do, **kw) for ins, do in self.inputs]
        finally:
            if pins:
                kda_tuning.restore(verbose=False)

    def accuracy(self):
        torch, a = self.torch, self.a
        log(f"fp32 reference (kda.py token loop, {a.seeds} seed(s), checkpoint segments of {a.ref_seg})")
        t = time.time()
        self.ref = [reference_fwd_bwd(ins, do, a.ref_seg) for ins, do in self.inputs]
        torch.cuda.synchronize()
        log(f"  done in {time.time() - t:.1f}s")
        self.out_default = self.run_case(None)
        again = self.run_case(None)
        deterministic = all(torch.equal(x, y) for s in range(a.seeds) for x, y in zip(self.out_default[s], again[s]))
        self.err_default = [errors(o, r) for o, r in zip(self.out_default, self.ref)]
        acc = {"deterministic_defaults": deterministic, "default_vs_ref": self.err_default, "cases": []}
        log(f"  defaults deterministic: {deterministic}; default vs fp32 ref (seed 0): "
            + ", ".join(f"{n} {e['rel_rms']:.2e}" for n, e in self.err_default[0].items()))

        def evaluate(label, pins):
            outs = self.run_case(pins)
            errs = [errors(o, r) for o, r in zip(outs, self.ref)]
            vs_def = [errors(o, d) for o, d in zip(outs, self.out_default)]
            ratio, where = worst_ratio(errs, self.err_default)
            # size of the change itself relative to the defaults' own error (<< 1: far below the error floor)
            change = max(v[n]["rel_rms"] / max(e[n]["rel_rms"], 1e-30)
                         for v, e in zip(vs_def, self.err_default) for n in OUT_NAMES)
            identical = all(torch.equal(x, y) for s in range(a.seeds) for x, y in zip(outs[s], self.out_default[s]))
            ok = ratio <= 1 + a.acc_tol
            acc["cases"].append({"case": label, "pins": len(pins), "worst_ratio_vs_default": ratio, "worst": where,
                                 "change_over_default_error": change, "pass": ok, "bitwise_equal_default": identical,
                                 "vs_ref": errs, "vs_default": vs_def})
            log(f"  {label}: worst error ratio vs defaults {ratio:.4f} ({where}), change/default error "
                f"{change:.3f}{' (bitwise identical)' if identical else ''} -> {'PASS' if ok else 'FAIL'}")
            return ok, outs

        self.cands = []
        for (name, key), kr in self.kres.items():
            if not kr["candidate"]:
                continue
            pin = build_pin(kr["module"], name, key, kr["best"]["cfg"])
            ok, _ = evaluate(f"{name} {kr['best']['config']}", [pin])
            if ok:
                self.cands.append(((name, key), pin, kr))
        self.final_outs = None
        while self.cands:
            ok, outs = evaluate(f"all {len(self.cands)} pins", [p for _, p, _ in self.cands])
            if ok:
                self.final_outs = outs
                break
            self.cands.sort(key=lambda c: c[2]["gain"])   # drop the smallest gain and retry
            dropped = self.cands.pop(0)
            log(f"  dropping {dropped[0][0]}")
        self.res["accuracy"] = acc
        self.save()

    # ---------------------------------------------------------------- 4. timing in the pipeline
    def e2e(self, pins, iters, **kw):
        torch = self.torch
        ins, do = self.inputs[0]
        if pins:
            kda_tuning.apply({"format": kda_tuning.FORMAT, "pins": pins}, strict=False, verbose=False)
        try:
            for _ in range(3):
                fla_fwd_bwd(self.chunk_kda, ins, do, **kw)
            torch.cuda.synchronize()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for s, e in ev:
                s.record()
                fla_fwd_bwd(self.chunk_kda, ins, do, **kw)
                e.record()
            torch.cuda.synchronize()
            return [s.elapsed_time(e) for s, e in ev]
        finally:
            if pins:
                kda_tuning.restore(verbose=False)

    def profile(self, pins, iters=5):
        torch = self.torch
        from torch.profiler import ProfilerActivity, profile
        ins, do = self.inputs[0]
        if pins:
            kda_tuning.apply({"format": kda_tuning.FORMAT, "pins": pins}, strict=False, verbose=False)
        try:
            fla_fwd_bwd(self.chunk_kda, ins, do)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(iters):
                    fla_fwd_bwd(self.chunk_kda, ins, do)
                torch.cuda.synchronize()
        finally:
            if pins:
                kda_tuning.restore(verbose=False)
        rows, total = {}, 0.0
        for e in prof.key_averages():
            t = e.self_device_time_total
            if t > 0:
                total += t
                rows[e.key[:60]] = {"us_per_fwd_bwd": t / iters, "launches": e.count // iters}
        return {"total_us": total / iters, "kernels": dict(sorted(rows.items(), key=lambda kv: -kv[1]["us_per_fwd_bwd"]))}

    def timing(self):
        a = self.a
        pins = [p for _, p, _ in self.cands]
        log(f"whole chunk_kda fwd+bwd, defaults vs {len(pins)} pin(s), {a.rounds} x {a.iters} interleaved iterations")
        td, tp = [], []
        for _ in range(a.rounds):
            td += self.e2e(None, a.iters)
            tp += self.e2e(pins, a.iters) if pins else []
        md = statistics.median(td)
        mp = statistics.median(tp) if tp else md
        per_step = (md - mp) * a.calls_per_step / 1e3
        log(f"  defaults {md:.3f} ms, pinned {mp:.3f} ms ({(md - mp) / md * 100:+.1f}%) -> {per_step:.3f} s per "
            f"training step ({a.calls_per_step} KDA calls)")
        prof_d, prof_p = self.profile(None), (self.profile(pins) if pins else None)
        self.res["timing"] = {"fwd_bwd_ms_default": md, "fwd_bwd_ms_pinned": mp, "saving_s_per_step": per_step,
                              "samples_default": td, "samples_pinned": tp,
                              "profile_default": prof_d, "profile_pinned": prof_p}
        self.save()

    # ---------------------------------------------------------------- write + verify the pins file
    def write_and_verify(self):
        torch, a = self.torch, self.a
        pins = [dict(p, default_config=cfg_str(kr["default"]["cfg"]), default_us=kr["default"]["ms"] * 1e3,
                     pinned_us=kr["best"]["ms"] * 1e3, gain=kr["gain"], calls_per_fwd_bwd=kr["calls"])
                for _, p, kr in self.cands]
        data = {"format": kda_tuning.FORMAT, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "env": {k: self.env[k] for k in ("gpu", "fla", "triton")}, "shape": self.res["shape"],
                "note": "pins for moe_llm/layers/kda_tuning.py; measured by scripts/sweep_kda.py",
                "fwd_bwd_ms": {"default": self.res["timing"]["fwd_bwd_ms_default"],
                               "pinned": self.res["timing"]["fwd_bwd_ms_pinned"]},
                "pins": pins}
        path = os.path.join(a.out, "kda_tuning.json")
        kda_tuning.save_tuning(path, data)
        log(f"wrote {path} ({len(pins)} pins)")
        if not pins:
            self.res["verify"] = {"pins": 0}
            return
        # apply from the file exactly as training does (strict: same fla / triton / GPU), check every pin is hit
        self.rec.reset()
        self.rec.enabled = True
        kda_tuning.apply(path)
        try:
            outs = [fla_fwd_bwd(self.chunk_kda, ins, do) for ins, do in self.inputs]
            torch.cuda.synchronize()
            st = kda_tuning.status()
        finally:
            kda_tuning.restore()
            self.rec.enabled = False
        hit = {(p["kernel"], tuple(kda_tuning.key_from_json(p["key"]))) for p in pins} <= set(self.rec.calls)
        same = self.final_outs is not None and all(
            torch.equal(x, y) for s in range(a.seeds) for x, y in zip(outs[s], self.final_outs[s]))
        self.res["verify"] = {"all_pinned_keys_launched": hit, "all_live": all(s["live"] for s in st),
                              "bitwise_equal_to_accuracy_run": same, "status": st}
        log(f"verify: every pinned key launched {hit}, pins live {all(s['live'] for s in st)}, "
            f"outputs identical to the accuracy run {same}")
        self.save()

    # ---------------------------------------------------------------- 5. extra: disable_recompute
    def extras(self):
        torch, a = self.torch, self.a
        ins, do = self.inputs[0]
        log("extra: disable_recompute=True (backward reuses the forward's w/u/kg/h), fla defaults")

        def held(flag):
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            q, k, v, g, beta, A_log, dt_bias = ins
            o, _ = self.chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                  use_beta_sigmoid_in_kernel=True, disable_recompute=flag)
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() - base
            del o
            return mem
        m0, m1 = held(False), held(True)
        out1 = fla_fwd_bwd(self.chunk_kda, ins, do, disable_recompute=True)
        out0 = fla_fwd_bwd(self.chunk_kda, ins, do)
        identical = all(torch.equal(x, y) for x, y in zip(out0, out1))
        td, tr = [], []
        for _ in range(a.rounds):
            td += self.e2e(None, a.iters)
            tr += self.e2e(None, a.iters, disable_recompute=True)
        md, mr = statistics.median(td), statistics.median(tr)
        from moe_llm.layers.kda import kda_keep_bytes
        n_layers, micro = a.kda_layers, a.calls_per_step // a.kda_layers   # KDA layers, KDA calls per layer per step
        extra = m1 - m0                                                    # bytes one keeping layer holds
        per_layer_s = (md - mr) * micro / 1e3                              # s per training step saved per layer
        room = (a.cap_gib - a.peak_gib) * 2**30
        n_max = n_layers if extra <= 0 else max(0, min(n_layers, int(room // extra)))
        table = [{"N": n, "saving_s_per_step": n * per_layer_s, "extra_GiB": n * extra / 2**30,
                  "peak_GiB": a.peak_gib + n * extra / 2**30} for n in range(1, n_layers + 1)]
        self.res["extra_disable_recompute"] = {
            "fwd_bwd_ms_recompute": md, "fwd_bwd_ms_saved": mr, "per_layer_saving_ms_per_call": md - mr,
            "per_layer_saving_s_per_step": per_layer_s, "per_layer_extra_MiB": extra / 2**20,
            "per_layer_extra_MiB_formula": kda_keep_bytes(a.B, a.T, a.H, a.K, a.K) / 2**20,
            "saving_s_per_step_all_layers": per_layer_s * n_layers, "extra_GiB_all_layers": extra * n_layers / 2**30,
            "peak_GiB": a.peak_gib, "cap_GiB": a.cap_gib, "max_layers_under_cap": n_max, "table": table,
            "held_MiB_per_call_recompute": m0 / 2**20, "held_MiB_per_call_saved": m1 / 2**20,
            "bitwise_identical": identical,
            "max_abs_diff": {n: (x.float() - y.float()).abs().max().item() for n, x, y in zip(OUT_NAMES, out0, out1)}}
        log(f"  {md:.3f} -> {mr:.3f} ms per fwd+bwd: per keeping layer -{per_layer_s:.4f} s/step and "
            f"+{extra / 2**20:.1f} MiB held until its backward (formula "
            f"{kda_keep_bytes(a.B, a.T, a.H, a.K, a.K) / 2**20:.1f}); bitwise identical: {identical}")
        log(f"  kda_keep_layers=N: N x {per_layer_s:.4f} s/step, peak {a.peak_gib:.2f} + N x {extra / 2**30:.3f} GiB; "
            f"N <= {n_max} stays under {a.cap_gib:.2f} GiB (all {n_layers}: -{per_layer_s * n_layers:.3f} s/step, "
            f"+{extra * n_layers / 2**30:.2f} GiB)")
        self.save()

    # ---------------------------------------------------------------- summary
    def summary(self):
        a, r = self.a, self.res
        env = r.get("env", {})
        L = [f"KDA kernel sweep {r['started']} on {env.get('gpu')} (fla {env.get('fla')}, triton {env.get('triton')})",
             f"shape B={a.B} T={a.T} H={a.H} K=V={a.K} bf16; max_shared_mem {env.get('max_shared_mem')} B; "
             f"check_shared_mem {env.get('check_shared_mem')}", "",
             f"{'kernel':44s} {'key':>24s} {'calls':>5s} {'default':>20s} {'us':>8s} {'best':>20s} {'us':>8s} "
             f"{'gain':>7s} {'s/step':>7s}  pin"]
        pinned = {k for k, _, _ in self.cands}
        for k in r.get("kernels", []):
            kk = (k["kernel"], kda_tuning.key_from_json(k["key"]))
            L.append(f"{k['kernel'][:44]:44s} {key_str(kk[1])[:24]:>24s} {k['calls_per_fwd_bwd']:5d} {k['default']:>20s} "
                     f"{k['default_us']:8.1f} {k['best']:>20s} {k['best_us']:8.1f} {k['gain'] * 100:+6.1f}% "
                     f"{k['est_s_per_step']:7.3f}  {'PIN' if kk in pinned else ('cand' if k['candidate'] else '')}")
        acc = r.get("accuracy", {})
        L += ["", f"defaults deterministic: {acc.get('deterministic_defaults')}"]
        for c in acc.get("cases", []):
            L.append(f"accuracy {c['case'][:70]:70s} worst ratio {c['worst_ratio_vs_default']:.4f} ({c['worst']}) "
                     f"{'PASS' if c['pass'] else 'FAIL'}{' bitwise=default' if c['bitwise_equal_default'] else ''}")
        if acc.get("default_vs_ref"):
            L.append("default vs fp32 reference, seed 0 (rel rms): "
                     + ", ".join(f"{n} {e['rel_rms']:.3e}" for n, e in acc["default_vs_ref"][0].items()))
        tm = r.get("timing")
        if tm:
            L += ["", f"chunk_kda fwd+bwd: default {tm['fwd_bwd_ms_default']:.3f} ms, pinned {tm['fwd_bwd_ms_pinned']:.3f}"
                  f" ms -> {tm['saving_s_per_step']:.3f} s per training step ({a.calls_per_step} calls)"]
            pp = tm.get("profile_pinned") or {}
            L.append(f"{'in-pipeline kernel (profiler)':60s} {'default us':>11s} {'pinned us':>10s}")
            for name, row in list(tm["profile_default"]["kernels"].items())[:16]:
                p = pp.get("kernels", {}).get(name, {}).get("us_per_fwd_bwd")
                L.append(f"{name:60s} {row['us_per_fwd_bwd']:11.1f} {p if p is None else round(p, 1)!s:>10s}")
        v = r.get("verify")
        if v:
            L += ["", f"verify (applied from kda_tuning.json like training): {v}"]
        ex = r.get("extra_disable_recompute")
        if ex:
            L += ["", f"extra disable_recompute=True (ModelConfig.kda_keep_layers): {ex['fwd_bwd_ms_recompute']:.3f} -> "
                  f"{ex['fwd_bwd_ms_saved']:.3f} ms per KDA fwd+bwd, bitwise identical: {ex['bitwise_identical']}",
                  f"  per keeping layer: -{ex['per_layer_saving_s_per_step']:.4f} s/step, +{ex['per_layer_extra_MiB']:.1f}"
                  f" MiB (formula {ex['per_layer_extra_MiB_formula']:.1f}); with peak {ex['peak_GiB']:.2f} GiB and cap "
                  f"{ex['cap_GiB']:.2f} GiB: N <= {ex['max_layers_under_cap']}",
                  "  " + "  ".join(f"N={row['N']}: -{row['saving_s_per_step']:.2f}s +{row['extra_GiB']:.2f}GiB"
                                   for row in ex["table"] if row["N"] in (1, 2, 4, 6, 8, 12, 18))]
        L += ["", f"wall time {self.elapsed_min():.1f} min"]
        text = "\n".join(L)
        with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text, flush=True)

    def run(self):
        try:
            self.setup()
            self.default_pass()
            self.time_kernels()
            self.accuracy()
            self.timing()
            self.write_and_verify()
            if not self.a.no_extras:
                self.extras()
        finally:                                 # keep whatever was measured, even after an error
            self.res["wall_min"] = self.elapsed_min()
            self.save()
            try:
                self.summary()
            except Exception as e:
                log(f"summary failed: {e!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "kda_sweep"))
    ap.add_argument("--check", action="store_true", help="CPU: resolve the kernel table against the installed fla")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--kernels", default="", help="comma-separated kernel names to sweep (default: all recorded)")
    ap.add_argument("--no-extend", action="store_true", help="only fla's own config spaces (re-timed at this shape)")
    ap.add_argument("--min-gain", type=float, default=0.02, help="pin only if this much faster than the default")
    ap.add_argument("--acc-tol", type=float, default=0.02,
                    help="max relative growth of any error vs the fp32 reference, compared with fla's defaults")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ref-seg", type=int, default=256, help="token-loop segment length (checkpointing) of the reference")
    ap.add_argument("--warmup-ms", type=int, default=10)
    ap.add_argument("--rep-ms", type=int, default=40)
    ap.add_argument("--refine-ms", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--calls-per-step", type=int, default=18 * 64, help="KDA calls per training step (layers x accum)")
    ap.add_argument("--budget-min", type=float, default=11.0, help="stop trying new configs after this many minutes")
    ap.add_argument("--min-free-gb", type=float, default=16.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-extras", action="store_true", help="skip the disable_recompute measurement")
    ap.add_argument("--kda-layers", type=int, default=18, help="KDA layers in the model (sizing kda_keep_layers)")
    ap.add_argument("--peak-gib", type=float, default=20.13, help="current peak allocated GiB (sizing kda_keep_layers)")
    ap.add_argument("--cap-gib", type=float, default=21.0, help="allowed peak allocated GiB (sizing kda_keep_layers)")
    a = ap.parse_args()
    if a.check:
        for line in check_table():
            print(line)
        return
    Sweep(a).run()


if __name__ == "__main__":
    main()
