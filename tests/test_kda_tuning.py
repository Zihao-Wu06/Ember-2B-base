"""Opt-in pinned fla KDA kernel configs (moe_llm/layers/kda_tuning.py) and the sweep's CPU-side logic
(scripts/sweep_kda.py). The mechanism is tested on fake autotuners everywhere, on real Triton autotuner objects when
Triton is importable, and against the installed fla's kernel table when fla is importable (all without a GPU); the
last test runs real fla kernels and needs CUDA."""
import json
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import sweep_kda  # noqa: E402
from moe_llm.config import ModelConfig, tiny  # noqa: E402
from moe_llm.layers import kda as kda_mod  # noqa: E402
from moe_llm.layers import kda_tuning  # noqa: E402

MOD = "fake_fla_kernels_for_test"


class FakeConfig:
    """The parts of triton.Config that kda_tuning and the sweep use."""

    def __init__(self, kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None):
        self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages

    def __eq__(self, other):
        return (self.kwargs, self.num_warps, self.num_stages) == (other.kwargs, other.num_warps, other.num_stages)

    def __hash__(self):
        return hash((tuple(sorted(self.kwargs.items())), self.num_warps, self.num_stages))


class FakeAutotuner:
    """Picks a config like Triton's Autotuner.run (one config: use it; else the cached one for the key) and records
    what each launch used."""

    def __init__(self, name, configs, keys, arg_names):
        self.configs, self.keys, self.arg_names, self.cache = configs, keys, arg_names, {}
        self.fn = self.base_fn = types.SimpleNamespace(__name__=name)
        self.launches = []

    def run(self, *args, **kwargs):
        if len(self.configs) == 1:
            cfg = self.configs[0]
        else:
            key = sweep_kda.autotune_key(self, args, kwargs)
            cfg = self.cache.setdefault(key, self.configs[0])
        self.launches.append(cfg)
        return cfg


class FakeHeuristics:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        return self.fn.run(*args, **kwargs)


class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv(kda_tuning.ENV_VAR, raising=False)
    yield
    kda_tuning.restore(verbose=False)
    kda_tuning._configured, kda_tuning._requested, kda_tuning._failed = False, None, None


@pytest.fixture
def fake(monkeypatch):
    """A fake fla module with two autotuned kernels: one behind @triton.heuristics, one bare."""
    mod = types.ModuleType(MOD)
    a = FakeAutotuner("kern_a", [FakeConfig({"BK": bk, "BV": bv}, w, s) for bk in (16, 32) for bv in (16, 32)
                                 for w in (2, 4) for s in (2, 3)], ["H", "K"], ["x", "y", "H", "K", "IS_VARLEN"])
    b = FakeAutotuner("kern_b", [FakeConfig({}, w, 2) for w in (1, 2, 4)], ["D"], ["x", "D"])
    mod.kern_a, mod.kern_b = FakeHeuristics(a), b
    mod.plain = lambda: None                # a function, not an autotuned kernel
    monkeypatch.setitem(sys.modules, MOD, mod)
    return types.SimpleNamespace(mod=mod, a=a, b=b)


KEY_A = (8, 128, "torch.bfloat16", "torch.float32")
KEY_A2 = (4, 128, "torch.bfloat16", "torch.float32")


def pins_doc(*pins):
    return {"format": kda_tuning.FORMAT, "pins": list(pins)}


def pin_a(key=KEY_A, BK=64, BV=64, w=8, s=1):
    return {"module": MOD, "kernel": "kern_a", "key": None if key is None else list(key),
            "config": {"kwargs": {"BK": BK, "BV": BV}, "num_warps": w, "num_stages": s}}


# ----------------------------------------------------------------------------- request / off by default
def test_resolve_env_wins_and_off_values(monkeypatch):
    assert kda_tuning.resolve("") is None and kda_tuning.resolve(None) is None
    assert kda_tuning.resolve("a.json") == "a.json"
    monkeypatch.setenv(kda_tuning.ENV_VAR, "b.json")
    assert kda_tuning.resolve("") == "b.json" and kda_tuning.resolve("a.json") == "b.json"
    for off in ("0", "off", "", " none "):
        monkeypatch.setenv(kda_tuning.ENV_VAR, off)
        assert kda_tuning.resolve("a.json") is None


def test_off_changes_nothing(fake):
    before = (list(fake.a.configs), dict(fake.a.cache), list(fake.b.configs), dict(fake.b.cache))
    kda_tuning.sync()                       # never configured: no-op
    kda_tuning.configure("")                # what KDA.__init__ does with the default ModelConfig
    kda_tuning.sync()
    assert not kda_tuning.is_active() and kda_tuning.requested() is None and kda_tuning.status() == []
    assert (fake.a.configs, fake.a.cache, fake.b.configs, fake.b.cache) == before
    assert ModelConfig().kda_tuning == "" and tiny().kda_tuning == ""


def test_kda_layer_and_fla_loader_wire_the_request(fake, tmp_path, monkeypatch):
    path = tmp_path / "pins.json"
    kda_tuning.save_tuning(path, pins_doc(pin_a()))
    kda_mod.KDA(tiny(kda_tuning=str(path)), 0)
    assert kda_tuning.requested() == str(path) and not kda_tuning.is_active()   # applied on first fla use only
    monkeypatch.setattr(kda_mod, "_CHUNK_KDA", object())                       # pretend fla is loaded
    kda_mod.fla_chunk_kda()
    assert kda_tuning.is_active() and fake.a.cache[KEY_A] == FakeConfig({"BK": 64, "BV": 64}, 8, 1)
    kda_mod.KDA(tiny(), 0)                  # a model without pins: the next fla use restores fla's autotuning
    kda_mod.fla_chunk_kda()
    assert not kda_tuning.is_active() and KEY_A not in fake.a.cache
    monkeypatch.setenv(kda_tuning.ENV_VAR, str(path))
    kda_mod.KDA(tiny(), 0)                  # the env var turns it on regardless of the config
    kda_mod.fla_chunk_kda()
    assert kda_tuning.is_active()


def test_unappliable_request_warns_once_and_keeps_fla_defaults(fake, tmp_path, capsys):
    kda_tuning.configure(str(tmp_path / "missing.json"))
    kda_tuning.sync()
    kda_tuning.sync()                       # not retried on every forward
    out = capsys.readouterr()
    assert out.out.count("kda_tuning: NOT applied") == 1 and "FileNotFoundError" in out.err
    assert not kda_tuning.is_active() and fake.a.cache == {}
    measured_elsewhere = dict(pins_doc(pin_a()), env={"fla": "0.0.0-not-this-one"})
    path = tmp_path / "other_fla.json"
    kda_tuning.save_tuning(path, measured_elsewhere)
    kda_tuning.configure(str(path))         # a new request is tried again
    kda_tuning.sync()
    assert "measured elsewhere" in capsys.readouterr().out and not kda_tuning.is_active()
    with pytest.raises(FileNotFoundError):  # a direct apply() still raises
        kda_tuning.apply(tmp_path / "missing.json")


def test_model_config_field_roundtrip():
    cfg = tiny(kda_tuning="runs/kda_sweep/kda_tuning.json")
    assert ModelConfig.from_dict(cfg.to_dict()).kda_tuning == "runs/kda_sweep/kda_tuning.json"
    old = {k: v for k, v in tiny().to_dict().items() if k != "kda_tuning"}   # configs saved before the field
    assert ModelConfig.from_dict(old).kda_tuning == ""


# ----------------------------------------------------------------------------- apply / restore
def test_key_pin_applies_and_restores(fake):
    orig = FakeConfig({"BK": 16, "BV": 32}, 4, 2)
    fake.a.cache[KEY_A] = orig              # fla had already tuned this key; KEY_A2 not yet
    configs_before = fake.a.configs
    applied = kda_tuning.apply(pins_doc(pin_a(KEY_A), pin_a(KEY_A2, BK=32, BV=16, w=2, s=3)), strict=False)
    assert len(applied) == 2 and kda_tuning.is_active()
    assert fake.a.cache[KEY_A] == FakeConfig({"BK": 64, "BV": 64}, 8, 1)
    assert fake.a.cache[KEY_A2] == FakeConfig({"BK": 32, "BV": 16}, 2, 3)
    assert type(fake.a.cache[KEY_A]) is FakeConfig and fake.a.configs is configs_before   # other keys still tune
    # a launch with the pinned key uses the pin; another key autotunes as before
    x = FakeTensor("torch.bfloat16")
    y = FakeTensor("torch.float32")
    assert fake.mod.kern_a.run(x, y, H=8, K=128, IS_VARLEN=False) == FakeConfig({"BK": 64, "BV": 64}, 8, 1)
    assert fake.mod.kern_a.run(x, y, H=2, K=128, IS_VARLEN=False) == fake.a.configs[0]
    assert all(s["live"] for s in kda_tuning.status())
    kda_tuning.restore(verbose=False)
    assert not kda_tuning.is_active()
    assert fake.a.cache[KEY_A] is orig and KEY_A2 not in fake.a.cache


def test_keyless_pin_replaces_configs_and_restores(fake):
    configs_before, cache_before = fake.b.configs, {("D", 128): fake.b.configs[2]}
    fake.b.cache.update(cache_before)
    kda_tuning.apply(pins_doc({"module": MOD, "kernel": "kern_b", "key": None,
                               "config": {"kwargs": {}, "num_warps": 8, "num_stages": 3}}), strict=False)
    assert fake.b.configs == [FakeConfig({}, 8, 3)] and fake.b.cache == {}
    assert fake.b.run(FakeTensor("torch.float32"), D=64) == FakeConfig({}, 8, 3)
    kda_tuning.restore(verbose=False)
    assert fake.b.configs is configs_before and fake.b.cache == cache_before


def test_bad_pins_are_rejected_atomically(fake):
    good = pin_a(KEY_A)
    for bad, exc in [
        (dict(pin_a(KEY_A2), config={"kwargs": {"BK": 64}, "num_warps": 4, "num_stages": 2}), ValueError),  # kwargs
        (dict(pin_a(KEY_A2), config={"kwargs": {"BK": 64, "BV": 64, "BT": 64}, "num_warps": 4, "num_stages": 2}),
         ValueError),
        (pin_a(KEY_A2, w=3), ValueError),
        (pin_a(KEY_A2, s=0), ValueError),
        (pin_a(KEY_A2, BK=0), ValueError),
        (dict(pin_a(KEY_A2), kernel="no_such_kernel"), AttributeError),
        (dict(pin_a(KEY_A2), module="no_such_module_xyz"), ImportError),
        (dict(pin_a(KEY_A2), kernel="plain"), TypeError),                # not an autotuned kernel
    ]:
        with pytest.raises(exc):
            kda_tuning.apply(pins_doc(good, bad), strict=False)
        assert not kda_tuning.is_active() and KEY_A not in fake.a.cache and KEY_A2 not in fake.a.cache
    for doc in [{"format": "other/1", "pins": []}, {"pins": {}}, {"pins": [{"module": MOD}]},
                {"pins": [dict(good, key="not a list")]}]:
        with pytest.raises(ValueError):
            kda_tuning.apply(doc, strict=False)
    kda_tuning.apply(pins_doc(good), strict=False)
    with pytest.raises(RuntimeError):
        kda_tuning.apply(pins_doc(good), strict=False)       # already applied


def test_strict_refuses_other_versions(fake):
    doc = dict(pins_doc(pin_a()), env={"fla": "0.0.0-not-this-one"})
    with pytest.raises(RuntimeError, match="measured elsewhere"):
        kda_tuning.apply(doc)
    assert not kda_tuning.is_active()
    kda_tuning.apply(doc, strict=False)
    assert kda_tuning.is_active()


# ----------------------------------------------------------------------------- JSON
def test_json_roundtrip_keeps_keys_exact(fake, tmp_path):
    key = (64, 8, False, 128, 128, True, "torch.bfloat16", "torch.float32")
    data = dict(pins_doc(pin_a(key)), env={}, shape={"B": 2, "T": 4096}, note="ünïcode ok")
    path = tmp_path / "sub" / "pins.json"
    kda_tuning.save_tuning(path, data)
    loaded = kda_tuning.load_tuning(path)
    assert loaded == json.loads(json.dumps(data)) and loaded["pins"][0]["key"] == list(key)
    back = kda_tuning.key_from_json(loaded["pins"][0]["key"])
    assert back == key and [type(v) for v in back] == [type(v) for v in key]     # bools stay bools
    kda_tuning.apply(path, strict=False)
    assert key in fake.a.cache and kda_tuning.status()[0]["key"] == list(key)
    with pytest.raises(TypeError):
        kda_tuning.key_to_json((1, object()))


def test_relative_paths_resolve_against_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert kda_tuning.resolve_path("moe_llm/config.py") == os.path.join(ROOT, "moe_llm/config.py")
    assert kda_tuning.resolve_path("not/there.json") == "not/there.json"


# ----------------------------------------------------------------------------- real Triton / fla objects (no GPU)
def test_real_triton_autotuner_pin_and_restore():
    triton = pytest.importorskip("triton")
    import triton.language as tl

    @triton.heuristics({"EVEN": lambda args: args["n"] % 2 == 0})
    @triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (32, 64) for w in (2, 4)],
                     key=["n"])
    @triton.jit
    def kern(x, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        pass

    mod = types.ModuleType("fake_triton_kernels_for_test")
    mod.kern = kern
    sys.modules[mod.__name__] = mod
    try:
        tuner = kda_tuning.find_autotuner(kern)
        assert isinstance(tuner, triton.runtime.autotuner.Autotuner)
        key = (1000, "torch.float32")
        kda_tuning.apply(pins_doc({"module": mod.__name__, "kernel": "kern", "key": list(key),
                                   "config": {"kwargs": {"BLOCK": 128}, "num_warps": 8, "num_stages": 2}}),
                         strict=False)
        cfg = tuner.cache[key]
        assert isinstance(cfg, triton.Config) and cfg.all_kwargs()["BLOCK"] == 128
        assert (cfg.num_warps, cfg.num_stages) == (8, 2) and len(tuner.configs) == 4
        kda_tuning.restore(verbose=False)
        assert key not in tuner.cache
    finally:
        del sys.modules[mod.__name__]


def test_sweep_kernel_table_matches_installed_fla():
    try:
        import fla.ops.kda  # noqa: F401
    except Exception as e:  # macOS has no Triton, so no fla
        pytest.skip(f"fla not importable: {e}")
    lines = sweep_kda.check_table()          # resolves every kernel and validates every extension config
    assert len(lines) == len(sweep_kda.KERNELS)
    for _, (_, tuner, ext) in sweep_kda.resolve_kernels().items():
        assert all(len(t) == 3 for t in [sweep_kda.cfg_id(c) for c, _ in sweep_kda.build_space(tuner, ext)])


# ----------------------------------------------------------------------------- sweep helpers (CPU)
def test_autotune_key_matches_triton_rule(fake):
    x, y = FakeTensor("torch.bfloat16"), FakeTensor("torch.float32")
    assert sweep_kda.autotune_key(fake.a, (x,), dict(y=y, H=8, K=128, IS_VARLEN=False, grid=(1,), warmup=False)) \
        == KEY_A
    assert sweep_kda.autotune_key(fake.a, (), dict(x=None, y=y, H=8, K=128)) == (8, 128, "torch.float32")


def test_recorder_records_first_launch_per_key_and_uninstalls(fake):
    x, y = FakeTensor("torch.bfloat16"), FakeTensor("torch.float32")
    rec = sweep_kda.Recorder({"kern_a": fake.a, "kern_b": fake.b}).install()
    fake.mod.kern_a.run(x, y, H=8, K=128, IS_VARLEN=False, grid=(4,), warmup=False)
    fake.mod.kern_a.run(x, y, H=8, K=128, IS_VARLEN=False, grid=(4,), warmup=False)
    fake.mod.kern_a.run(x, y, H=4, K=128, IS_VARLEN=False, grid=(4,), warmup=False)
    fake.b.run(y, D=64, grid=(2,), warmup=False)
    assert {k: v["count"] for k, v in rec.calls.items()} == {("kern_a", KEY_A): 2, ("kern_a", KEY_A2): 1,
                                                             ("kern_b", (64, "torch.float32")): 1}
    rc = rec.calls[("kern_a", KEY_A)]
    assert rc["args"] == (x, y) and rc["kwargs"]["grid"] == (4,)
    # replay with a chosen config: launched with it, the config list comes back, nothing recorded
    cfg = FakeConfig({"BK": 64, "BV": 64}, 8, 1)
    configs = fake.a.configs
    sweep_kda.call_with(fake.a, rc, cfg)
    assert fake.a.launches[-1] == cfg and fake.a.configs is configs and rec.calls[("kern_a", KEY_A)]["count"] == 2
    rec.uninstall()
    assert "run" not in fake.a.__dict__ and "run" not in fake.b.__dict__


def test_build_space_dedups_and_validates(fake):
    ext = [sweep_kda.C(4, 2, BK=16, BV=16), sweep_kda.C(8, 1, BK=64, BV=128), sweep_kda.C(8, 1, BK=64, BV=128)]
    space = sweep_kda.build_space(fake.a, ext)
    assert len(space) == len(fake.a.configs) + 1 and space[-1][1] is False
    with pytest.raises(ValueError):
        sweep_kda.build_space(fake.a, [sweep_kda.C(4, 2, BK=16)])


def test_select_best_and_worst_ratio():
    t = [{"ms": 1.0, "is_default": True}, {"ms": 0.95, "is_default": False}, {"ms": None, "is_default": False}]
    best, gain, cand = sweep_kda.select_best(t, 1.0, 0.02)
    assert best is t[1] and abs(gain - 0.05) < 1e-12 and cand
    assert not sweep_kda.select_best(t, 1.0, 0.10)[2]                 # gain below the threshold
    best, _, cand = sweep_kda.select_best([dict(t[0], ms=0.9), t[1]], 0.9, 0.0)
    assert best["is_default"] and not cand                             # the default is already the fastest
    e = {n: {"rel_rms": 1e-3, "max_abs": 1.0} for n in sweep_kda.OUT_NAMES}
    worse = {n: dict(v) for n, v in e.items()}
    worse["dg"]["rel_rms"] = 1.1e-3
    ratio, where = sweep_kda.worst_ratio([e, worse], [e, e])      # RMS over the two seeds
    assert abs(ratio - ((1 + 1.1 ** 2) / 2) ** 0.5) < 1e-9 and where == "dg"
    assert sweep_kda.worst_ratio([e, e], [e, e]) == (1.0, "o")


def test_segmented_reference_equals_kda_reference():
    """The sweep's checkpointed fp32 reference is kda.py's kda_reference: same outputs and gradients."""
    torch.manual_seed(0)
    B, T, H, K = 2, 37, 2, 8
    ins, do = sweep_kda.make_inputs(B, T, H, K, seed=3, device="cpu")
    got = sweep_kda.reference_fwd_bwd(ins, do, seg=8)
    leaves = [t.detach().float().requires_grad_() for t in ins]
    o = kda_mod.kda_reference(*leaves)
    want = [o.detach()] + [g.detach() for g in torch.autograd.grad(o, leaves, do.float())]
    for name, a, b in zip(sweep_kda.OUT_NAMES, got, want):
        assert a.dtype == torch.float32 and a.shape == b.shape, name
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), (name, (a - b).abs().max().item())


def test_pin_built_by_the_sweep_applies(fake, tmp_path):
    cfg = FakeConfig({"BK": 32, "BV": 64}, 4, 2)
    pin = sweep_kda.build_pin(MOD, "kern_a", KEY_A, cfg, gain=0.1)
    path = tmp_path / "kda_tuning.json"
    kda_tuning.save_tuning(path, pins_doc(pin))
    kda_tuning.apply(path, strict=False)
    assert fake.a.cache[KEY_A] == cfg


class OutOfResources(Exception):
    pass


def test_sweep_flow_end_to_end_with_fake_kernels(tmp_path, monkeypatch):
    """The whole Sweep pipeline on CPU with fake kernels: only do_bench, CUDA events, the profiler and CUDA memory
    queries are stubbed. Key 1's fastest config changes the output (must fail accuracy), key 2's fastest is
    rounding-neutral (must be pinned), one config is out of resources, kern_b has no faster config."""
    mod = types.ModuleType("fake_fla_flow_for_test")
    a = FakeAutotuner("kern_a", [FakeConfig({"BK": bk, "BV": bv}, w, s) for bk in (16, 32) for bv in (16, 32)
                                 for w in (2, 4) for s in (2, 3)], ["H", "K"], ["x", "y", "H", "K", "IS_VARLEN"])
    b = FakeAutotuner("kern_b", [FakeConfig({}, w, 2) for w in (1, 2, 4)], ["D"], ["x", "D"])
    mod.kern_a, mod.kern_b = FakeHeuristics(a), b
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(sweep_kda, "KERNELS", [
        (mod.__name__, "kern_a", [sweep_kda.C(8, 1, BK=64, BV=64), sweep_kda.C(8, 2, BK=128, BV=16)]),
        (mod.__name__, "kern_b", [])])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kw: None)
    mem = iter(range(0, 10 ** 9, 1000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kw: next(mem))
    B, T, H, K = 1, 16, 2, 8
    H1, H2 = H, H + 100                      # two autotune keys of kern_a

    def fake_chunk_kda(q, k, v, g, beta, A_log, dt_bias, **kw):
        c1 = mod.kern_a.run(x=q, y=A_log, H=H1, K=K, IS_VARLEN=False, grid=(1,), warmup=False)
        c2 = mod.kern_a.run(x=q, y=A_log, H=H2, K=K, IS_VARLEN=False, grid=(1,), warmup=False)
        mod.kern_b.run(x=q, D=K, grid=(1,), warmup=False)
        eps = {16: 0.0, 32: 0.0, 64: 1e-6, 128: 5e-2}
        o = kda_mod.kda_reference(q, k, v, g, beta, A_log, dt_bias)
        return o * (1 + eps[c1.kwargs["BK"]] + eps[c2.kwargs["BK"]]), None

    def fake_bench(tuner, rc, cfg, rep_ms):
        sweep_kda.call_with(tuner, rc, cfg)
        if tuner is b:
            return 1.0
        h = rc["kwargs"]["H"]
        if cfg.kwargs["BK"] == 128:
            if h == H2:
                raise OutOfResources("out of resource: shared memory, Required: 120000, Hardware limit: 101376")
            return 0.5                       # key 1: fastest, but changes the output
        if cfg.kwargs["BK"] == 64:
            return 0.8                       # key 2: fastest that works, rounding-neutral
        return 1.0 if cfg == tuner.configs[0] else 1.01

    args = sweep_kda.argparse.Namespace(
        out=str(tmp_path / "sweep"), B=B, T=T, H=H, K=K, kernels="", no_extend=False, min_gain=0.02, acc_tol=0.02,
        seeds=2, ref_seg=8, warmup_ms=1, rep_ms=1, refine_ms=1, rounds=1, iters=2, calls_per_step=1152,
        budget_min=100.0, min_free_gb=0.0, force=True, no_extras=False, check=False, kda_layers=18, peak_gib=20.13,
        cap_gib=21.0)
    sw = sweep_kda.Sweep(args)

    def fake_setup():
        sw.chunk_kda = fake_chunk_kda
        sw.env = {"gpu": None, "fla": None, "triton": None, "capability": [0, 0], "torch": torch.__version__}
        sw.res["env"] = dict(sw.env, max_shared_mem=0, check_shared_mem={})
        sw.kernels = sweep_kda.resolve_kernels()
        sw.tuners = {k: t for k, (_, t, _) in sw.kernels.items()}
        sw.res["shape"] = {"B": B, "T": T, "H": H, "K": K}
        sw.inputs = [sweep_kda.make_inputs(B, T, H, K, seed, "cpu") for seed in range(args.seeds)]

    sw.setup, sw.bench = fake_setup, fake_bench
    sw.e2e = lambda pins, iters, **kw: [0.9 if pins else 1.0] * iters
    sw.profile = lambda pins, iters=5: {"total_us": 1.0, "kernels": {"kern_a": {"us_per_fwd_bwd": 1.0, "launches": 2}}}
    sw.run()

    pins = kda_tuning.load_tuning(tmp_path / "sweep" / "kda_tuning.json")["pins"]
    assert [(p["kernel"], p["key"][0], p["config"]) for p in pins] == \
        [("kern_a", H2, {"kwargs": {"BK": 64, "BV": 64}, "num_warps": 8, "num_stages": 1})]
    res = json.load(open(tmp_path / "sweep" / "sweep_results.json"))
    cases = {c["case"]: c for c in res["accuracy"]["cases"]}
    assert res["accuracy"]["deterministic_defaults"]
    assert not cases["kern_a BK=128,BV=16 w8 s2"]["pass"] and cases["kern_a BK=64,BV=64 w8 s1"]["pass"]
    assert cases["all 1 pins"]["pass"] and cases["all 1 pins"]["bitwise_equal_default"]
    kres = {(k["kernel"], k["key"][0]): k for k in res["kernels"]}
    assert kres[("kern_a", H2)]["failed"] == 1 and not kres[("kern_b", K)]["candidate"]
    assert res["verify"]["all_pinned_keys_launched"] and res["verify"]["all_live"]
    assert res["verify"]["bitwise_equal_to_accuracy_run"]
    assert abs(res["timing"]["saving_s_per_step"] - 0.1 * 1152 / 1e3) < 1e-9
    ex = res["extra_disable_recompute"]
    assert ex["bitwise_identical"] and ex["max_layers_under_cap"] == 18 and len(ex["table"]) == 18
    assert ex["per_layer_extra_MiB_formula"] == kda_mod.kda_keep_bytes(B, T, H, K, K) / 2 ** 20
    assert (tmp_path / "sweep" / "summary.txt").read_text().count("PIN") == 1
    assert not kda_tuning.is_active()


# ----------------------------------------------------------------------------- real kernels (GPU)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pins_hit_real_fla_kernels_gpu(tmp_path):
    chunk_kda = kda_mod.fla_chunk_kda()
    assert chunk_kda is not None
    ins, do = sweep_kda.make_inputs(1, 256, 2, 64, seed=0, device="cuda")
    kernels = sweep_kda.resolve_kernels()
    tuners = {k: t for k, (_, t, _) in kernels.items()}
    rec = sweep_kda.Recorder(tuners).install()
    try:
        out0 = sweep_kda.fla_fwd_bwd(chunk_kda, ins, do)
        name = "chunk_kda_bwd_kernel_wy_dqkg_fused"
        (key,) = [k for (n, k) in rec.calls if n == name]
        tuner = tuners[name]
        default = tuner.cache[key]
        other = next(c for c in tuner.configs if sweep_kda.cfg_id(c) != sweep_kda.cfg_id(default))
        path = tmp_path / "pins.json"
        kda_tuning.save_tuning(path, pins_doc(sweep_kda.build_pin(kernels[name][0], name, key, other)))
        kda_tuning.apply(path, strict=False)
        out1 = sweep_kda.fla_fwd_bwd(chunk_kda, ins, do)
        assert sweep_kda.cfg_id(tuner.best_config) == sweep_kda.cfg_id(other)
        for n, a, b in zip(sweep_kda.OUT_NAMES, out1, out0):
            assert sweep_kda.rel_rms(a, b) < 1e-2, n
        kda_tuning.restore(verbose=False)
        assert tuner.cache[key] is default
        out2 = sweep_kda.fla_fwd_bwd(chunk_kda, ins, do)
        assert all(torch.equal(a, b) for a, b in zip(out2, out0))
    finally:
        rec.uninstall()
