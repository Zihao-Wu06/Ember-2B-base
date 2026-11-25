"""Opt-in pinned launch configs for fla's KDA Triton kernels. Default off: fla autotunes exactly as before.

Why: fla autotunes each KDA kernel over a config space it fixes at import time from the GPU's shared memory, with an
autotune key made of head sizes only (no B, no T), and Triton caches the winner on disk (cache_results=True). On the
4090 this means:
  - the main run reuses winners that were tuned once at micro-batch 1 (scripts/bench.py, 2026-11-09), because the key
    is the same at micro-batch 2;
  - some kernels never try tiles the 4090 can run: its 99 KB (101376 B) of shared memory per block is just under the
    100 KB (102400 B) that fla's check_shared_mem() asks for, so e.g. chunk_kda_bwd_kernel_wy_dqkg_fused only gets
    BK, BV in {16, 32}, and the recurrent state kernels get num_stages=1 (backward) and no BV=16.
scripts/sweep_kda.py times fla's space plus a wider sm89 space at the real shape, checks accuracy against kda.py's
fp32 reference, and writes a JSON of the winners. This module pins those configs into the autotuners' in-process
caches for exactly the autotune keys that were measured, so those launches skip autotuning and use the pinned config.
Every other key (another shape, another kernel) autotunes as before. The math is unchanged; a different tile size
or warp count can change the fp32 summation order inside a kernel (rounding level only).

Enable: ModelConfig.kda_tuning = "<path to JSON>" (applied when fla is first used, see layers/kda.py) or the env
var MOE_KDA_TUNING=<path>, which wins over the config ("0", "off" or "" turn it off). Relative paths are tried
against the working directory, then the project root. The autotuners are module-level objects in fla, so pinning is
process-wide: the last configure() wins, and restore() puts fla's own autotuning back. If the requested file cannot
be applied (missing, measured with another fla / Triton / GPU, bad pin), the model path prints one warning and fla
autotunes as before; apply() called directly raises instead.

JSON (written by scripts/sweep_kda.py; only "pins" is required):
  {"format": "moe_llm.kda_tuning/1",
   "env": {"fla": "0.6.0", "triton": "3.9.0", "gpu": "NVIDIA GeForce RTX 4090"},     # checked when strict
   "pins": [{"module": "fla.ops.kda.chunk_bwd", "kernel": "chunk_kda_bwd_kernel_wy_dqkg_fused",
             "key": [64, 8, false, 128, 128, "torch.bfloat16", ...],   # Triton autotune key; null = every key
             "config": {"kwargs": {"BK": 16, "BV": 64}, "num_warps": 4, "num_stages": 2}}, ...]}
"""
from __future__ import annotations

import importlib
import json
import os
import sys

ENV_VAR = "MOE_KDA_TUNING"
FORMAT = "moe_llm.kda_tuning/1"
_OFF = {"", "0", "off", "none", "false", "no"}
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MISSING = object()

_configured = False        # configure() was called in this process (only then does sync() touch anything)
_requested: str | None = None
_failed: str | None = None   # a request sync() could not apply (reported once, not retried until it changes)
_active: dict | None = None  # {"source", "saved": [...], "pins": [...]} while pins are applied


# --------------------------------------------------------------------------- request (model side)
def resolve(cfg_value: str | None = "") -> str | None:
    """The JSON path to use, or None (off). The env var MOE_KDA_TUNING, when set, wins over the config value."""
    env = os.environ.get(ENV_VAR)
    value = (env if env is not None else (cfg_value or "")).strip()
    return None if value.lower() in _OFF else value


def configure(cfg_value: str | None = "") -> None:
    """Remember the requested pins (KDA.__init__ calls this with ModelConfig.kda_tuning). Applied by sync()."""
    global _configured, _requested
    _configured = True
    _requested = resolve(cfg_value)


def requested() -> str | None:
    return _requested


def sync() -> None:
    """Make the applied pins match the request (layers/kda.py calls this once fla is importable). O(1) when nothing
    changed; does nothing unless configure() was called, so scripts that apply() by hand are left alone.
    A request that cannot be applied (missing file, other fla / Triton / GPU, bad pin) is reported once and fla keeps
    autotuning as before: a speed knob must not stop a long run."""
    global _failed
    if not _configured:
        return
    have = _active["source"] if _active is not None else None
    if _requested == have or (_failed is not None and _requested == _failed):
        return
    if _active is not None:
        restore()
    _failed = None
    if _requested is not None:
        try:
            apply(_requested)
        except Exception as e:
            _failed = _requested
            msg = f"kda_tuning: NOT applied ({type(e).__name__}: {e}); fla autotunes as before"
            print(msg, flush=True)
            print(msg, file=sys.stderr, flush=True)


def is_active() -> bool:
    return _active is not None


# --------------------------------------------------------------------------- JSON
def resolve_path(path: str | os.PathLike) -> str:
    p = os.fspath(path)
    if os.path.isabs(p) or os.path.exists(p):
        return p
    alt = os.path.join(_ROOT, p)
    return alt if os.path.exists(alt) else p


def key_to_json(key) -> list:
    """A Triton autotune key (tuple of ints, bools, strings) as a JSON list."""
    out = []
    for v in key:
        if isinstance(v, (list, tuple)):
            out.append(key_to_json(v))
        elif v is None or isinstance(v, (bool, int, float, str)):
            out.append(v)
        else:
            raise TypeError(f"kda_tuning: autotune key element {v!r} ({type(v).__name__}) is not JSON-serializable")
    return out


def key_from_json(key) -> tuple:
    return tuple(key_from_json(v) if isinstance(v, list) else v for v in key)


def config_to_dict(cfg) -> dict:
    return {"kwargs": dict(cfg.kwargs), "num_warps": int(cfg.num_warps), "num_stages": int(cfg.num_stages)}


def validate(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"kda_tuning: expected a JSON object, got {type(data).__name__}")
    if data.get("format", FORMAT) != FORMAT:
        raise ValueError(f"kda_tuning: unknown format {data.get('format')!r} (expected {FORMAT!r})")
    pins = data.get("pins")
    if not isinstance(pins, list):
        raise ValueError("kda_tuning: 'pins' must be a list")
    for i, pin in enumerate(pins):
        if not isinstance(pin, dict) or not all(k in pin for k in ("module", "kernel", "config")):
            raise ValueError(f"kda_tuning: pins[{i}] needs 'module', 'kernel' and 'config'")
        cfg = pin["config"]
        if not isinstance(cfg, dict) or not isinstance(cfg.get("kwargs", {}), dict) \
                or "num_warps" not in cfg or "num_stages" not in cfg:
            raise ValueError(f"kda_tuning: pins[{i}].config needs 'kwargs', 'num_warps' and 'num_stages'")
        if pin.get("key") is not None and not isinstance(pin["key"], list):
            raise ValueError(f"kda_tuning: pins[{i}].key must be a list or null")
    return data


def load_tuning(path: str | os.PathLike) -> dict:
    with open(resolve_path(path), encoding="utf-8") as f:
        return validate(json.load(f))


def save_tuning(path: str | os.PathLike, data: dict) -> None:
    validate(data)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.fspath(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- autotuner access
def find_autotuner(obj):
    """The Triton Autotuner inside a kernel object (through @triton.heuristics wrappers)."""
    o = obj
    for _ in range(8):
        if o is None:
            break
        if hasattr(o, "configs") and hasattr(o, "cache") and hasattr(o, "keys") and hasattr(o, "arg_names"):
            return o
        o = getattr(o, "fn", None)
    raise TypeError(f"kda_tuning: {getattr(obj, '__name__', obj)!r} is not an autotuned Triton kernel")


def kernel_name(tuner) -> str:
    base = getattr(tuner, "base_fn", None) or getattr(tuner, "fn", None)
    return getattr(base, "__name__", type(tuner).__name__)


def make_config(tuner, spec: dict):
    """A config of the kernel's own Config class (triton.Config) from {"kwargs", "num_warps", "num_stages"}. The
    kwarg names must be exactly the ones fla autotunes for that kernel (others are fixed by fla's wrapper)."""
    if not tuner.configs:
        raise ValueError(f"kda_tuning: {kernel_name(tuner)} has no configs to copy the Config class from")
    kwargs = dict(spec.get("kwargs", {}))
    expected = set(tuner.configs[0].kwargs)
    if set(kwargs) != expected:
        raise ValueError(f"kda_tuning: {kernel_name(tuner)} tunes {sorted(expected)}, the pin sets {sorted(kwargs)}")
    for k, v in kwargs.items():
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"kda_tuning: {kernel_name(tuner)}: {k}={v!r} must be a positive int")
    num_warps, num_stages = spec["num_warps"], spec["num_stages"]
    if num_warps not in (1, 2, 4, 8, 16, 32) or not isinstance(num_stages, int) or num_stages < 1:
        raise ValueError(f"kda_tuning: {kernel_name(tuner)}: bad num_warps={num_warps!r} / num_stages={num_stages!r}")
    return type(tuner.configs[0])(kwargs, num_warps=num_warps, num_stages=num_stages)


def _version(module: str) -> str:
    try:
        return getattr(importlib.import_module(module), "__version__", "?")
    except ImportError:
        return "not installed"


def _check_env(env: dict) -> None:
    problems = []
    for module in ("fla", "triton"):
        if env.get(module) and _version(module) != env[module]:
            problems.append(f"{module} {_version(module)} (tuned with {env[module]})")
    if env.get("gpu"):
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_name() != env["gpu"]:
            problems.append(f"GPU {torch.cuda.get_device_name()} (tuned on {env['gpu']})")
    if problems:
        raise RuntimeError("kda_tuning: pins were measured elsewhere: " + "; ".join(problems)
                           + ". Re-run scripts/sweep_kda.py here, or apply(..., strict=False).")


# --------------------------------------------------------------------------- apply / restore
def apply(spec, *, strict: bool = True, verbose: bool = True) -> list[dict]:
    """Pin the configs of `spec` (a JSON path or an already-loaded dict). Returns one record per pin.
    strict: refuse pins measured with another fla / Triton version or on another GPU (spec["env"])."""
    global _active
    if _active is not None:
        raise RuntimeError("kda_tuning: pins are already applied; restore() first")
    if isinstance(spec, dict):
        data, source = validate(spec), None
    else:
        data, source = load_tuning(spec), os.fspath(spec)
    if strict:
        _check_env(data.get("env") or {})
    saved, applied = [], []
    try:
        for pin in data["pins"]:
            tuner = find_autotuner(getattr(importlib.import_module(pin["module"]), pin["kernel"]))
            cfg = make_config(tuner, pin["config"])
            if pin.get("key") is None:      # every key: this config only
                saved.append(("configs", tuner, (tuner.configs, dict(tuner.cache))))
                tuner.configs = [cfg]
                tuner.cache.clear()
                key = None
            else:                           # one autotune key, as if the autotuner had picked cfg for it
                key = key_from_json(pin["key"])
                saved.append(("cache", tuner, (key, tuner.cache.get(key, _MISSING))))
                tuner.cache[key] = cfg
            applied.append({"module": pin["module"], "kernel": pin["kernel"], "key": key, "config": cfg,
                            "tuner": tuner})
    except Exception:
        _undo(saved)
        raise
    _active = {"source": source, "saved": saved, "pins": applied}
    if verbose:
        print(f"kda_tuning: pinned {len(applied)} fla KDA kernel config(s) from {source or 'a dict'}", flush=True)
    return applied


def _undo(saved) -> None:
    for kind, tuner, state in reversed(saved):
        if kind == "cache":
            key, old = state
            if old is _MISSING:
                tuner.cache.pop(key, None)
            else:
                tuner.cache[key] = old
        else:
            configs, cache = state
            tuner.configs = configs
            tuner.cache.clear()
            tuner.cache.update(cache)


def restore(verbose: bool = True) -> None:
    """Undo apply(): fla's own configs and autotune choices come back (a pinned key not tuned before is re-tuned,
    or read from Triton's disk cache, on its next launch)."""
    global _active
    if _active is None:
        return
    _undo(_active["saved"])
    _active = None
    if verbose:
        print("kda_tuning: restored fla's own autotuning", flush=True)


def status() -> list[dict]:
    """The applied pins and whether each is still in its autotuner (for logs and the sweep's final check)."""
    if _active is None:
        return []
    out = []
    for p in _active["pins"]:
        t, key, cfg = p["tuner"], p["key"], p["config"]
        live = (t.configs == [cfg]) if key is None else (t.cache.get(key) is cfg)
        out.append({"kernel": p["kernel"], "key": None if key is None else key_to_json(key), "config": str(cfg),
                    "live": live})
    return out
