"""moe_llm/row_adam_native.py (OptimizerBundle engram_native, train.py --engram-native): the offloaded Engram tables'
lazy Adam in a C kernel that releases the GIL. It must give the torch.compile kernel's values bit for bit:
kernel against kernel on wide-ranging inputs (zeros, negative zeros, denormals, large values; any thread split),
training with the bundle (with and without engram_early), the verify mode (passes, and catches a kernel that is one
ulp off), train.py end to end (same logs, bit-identical checkpoints), the fallback when the kernel cannot be built,
and the reason for it all: the main thread keeps running while the native kernel works."""
import copy
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from moe_llm import row_adam_native as rn
from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, row_adam_cpu, wsd_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train as T  # noqa: E402

needs_native = pytest.mark.skipif(not rn.available(), reason=f"no C compiler: {rn.why_unavailable()}")


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


def _tables(R, D, gen):
    def rnd(lo, hi):
        return torch.exp(torch.empty(R, D).uniform_(lo, hi, generator=gen)) * torch.randn(R, D, generator=gen)
    w, g, m, v = rnd(-12, 2), rnd(-30, 3), rnd(-30, 3), rnd(-60, 4).abs()
    g[::97] = 0.0
    g[1::97] = -0.0
    m[::89] = 0.0
    v[::83] = 0.0
    v[5::83] = 1e-45
    g[7::101] = 1e19
    m[3::103] = -1e-40
    return w, g, m, v


@needs_native
@pytest.mark.parametrize("R,D,frac,lr,step,threads", [(60_000, 40, 0.5, 1e-2, 1, None), (60_000, 40, 0.9, 1e-2, 7885, 3),
                                                     (20_000, 7, 1.0, 3.3e-3, 100, 8), (9_000, 1, 0.4, 5e-4, 47684, 1),
                                                     (50_000, 32, 0.02, 3.7e-3, 40000, 16)])
def test_native_kernel_bit_identical_to_compiled(R, D, frac, lr, step, threads):
    gen = torch.Generator().manual_seed(R + D)
    w, g, m, v = _tables(R, D, gen)
    ids = torch.randperm(R, generator=gen)[: max(1, int(R * frac))].sort().values
    a = [t.clone() for t in (w, g, m, v)]
    b = [t.clone() for t in (w, g, m, v)]
    row_adam_cpu(*a, ids, lr, (0.9, 0.95), 1e-8, step)
    rn.row_adam_native(*b, ids, lr, (0.9, 0.95), 1e-8, step, threads=threads)
    for name, x, y in zip(("weight", "grad", "m", "v"), a, b):
        assert torch.equal(x.view(torch.int32), y.view(torch.int32)), name   # bit patterns
    rest = torch.ones(R, dtype=torch.bool)
    rest[ids] = False
    for x, y in zip((w, g, m, v), b):
        assert torch.equal(x[rest].view(torch.int32), y[rest].view(torch.int32))   # other rows untouched
    assert not b[1][ids].any()


@needs_native
def test_native_kernel_rejects_bad_inputs_and_handles_empty():
    w, g, m, v = (torch.zeros(10, 4) for _ in range(4))
    rn.row_adam_native(w, g, m, v, torch.zeros(0, dtype=torch.int64), 1e-2, (0.9, 0.95), 1e-8, 1)
    with pytest.raises(IndexError):
        rn.row_adam_native(w, g, m, v, torch.tensor([3, 10]), 1e-2, (0.9, 0.95), 1e-8, 1)
    with pytest.raises(IndexError):
        rn.row_adam_native(w, g, m, v, torch.tensor([-1, 2]), 1e-2, (0.9, 0.95), 1e-8, 1)
    with pytest.raises(ValueError):
        rn.row_adam_native(w.double(), g, m, v, torch.tensor([1]), 1e-2, (0.9, 0.95), 1e-8, 1)
    with pytest.raises(ValueError):
        rn.row_adam_native(w, g, m, v, torch.tensor([1], dtype=torch.int32), 1e-2, (0.9, 0.95), 1e-8, 1)
    with pytest.raises(ValueError):
        rn.row_adam_native(w[:, ::2], g, m, v, torch.tensor([1]), 1e-2, (0.9, 0.95), 1e-8, 1)


def _batch(cfg, s, k):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(1000 * s + k))
    return x[:, :-1], x[:, 1:]


def _state(m, opt):
    sd = {f"model.{k}": v.detach().clone() for k, v in m.state_dict().items()}
    for i, o in enumerate(opt.state_dict()["opts"]):
        for pi, st in o["state"].items():
            for k, v in st.items():
                if torch.is_tensor(v):
                    sd[f"opt{i}.{pi}.{k}"] = v.detach().clone()
    for i, t in enumerate(m.engram.offload_state_dict()):
        for k, v in t.items():
            sd[f"engram{i}.{k}"] = v.clone()
    return sd


def _train(m, opt, cfg, steps):
    out = []
    for s in range(steps):
        for k in range(2):
            x, y = _batch(cfg, s, k)
            loss, info = m(x, y)
            (loss / 2).backward()
            out.append(float(info["ce"]))
        lr = wsd_lr(s, 8, 3e-3, warmup=1)
        opt.start_engram_update(lr)
        out.append(float(opt.clip_grad_norm(0.05)))
        opt.step(lr)
        opt.zero_grad()
        m.update_router_bias()
    return out


@needs_native
def test_bundle_engram_native_trains_bit_identically(capsys):
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", engram_offload=True)
    base = MoELLM(cfg)
    runs = []
    for native, early, verify in ((False, False, False), (True, False, False), (True, True, False), (True, True, True)):
        m = copy.deepcopy(base)
        opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=torch.bfloat16, engram_early=early, engram_native=native,
                              engram_native_verify=verify)
        assert opt.engram_native == native
        opt.begin()
        r = _train(m, opt, cfg, 4)
        runs.append((r, _state(m, opt)))
        common.clear_cached_copies()
    for r, st in runs[1:]:
        assert r == runs[0][0]
        assert st.keys() == runs[0][1].keys()
        for k in st:
            assert torch.equal(st[k], runs[0][1][k]), k
    assert capsys.readouterr().out.count("Engram native Adam verified") == 4 * len(base.engram.cpu_tables)


@needs_native
def test_verify_catches_a_kernel_that_is_one_ulp_off(monkeypatch):
    torch.manual_seed(0)
    cfg = tiny(engram_offload=True)
    m = MoELLM(cfg)
    opt = OptimizerBundle(m, lr=3e-3, engram_native=True, engram_native_verify=True)
    opt.begin()
    real = rn.row_adam_native

    def off_by_one_ulp(weight, grad, mm, v, ids, *a, **kw):
        real(weight, grad, mm, v, ids, *a, **kw)
        r = int(ids[0])
        weight[r, 0] = torch.nextafter(weight[r, 0], torch.tensor(float("inf")))
    monkeypatch.setattr(rn, "row_adam_native", off_by_one_ulp)
    x, y = _batch(cfg, 0, 0)
    m(x, y)[0].backward()
    opt.clip_grad_norm(1.0)
    opt.step(1e-3)
    with pytest.raises(RuntimeError, match="differs from the torch.compile kernel"):
        m.engram.flush()


def test_falls_back_to_the_compiled_kernel_when_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(rn, "available", lambda: False)
    monkeypatch.setattr(rn, "why_unavailable", lambda: "no compiler")
    torch.manual_seed(0)
    cfg = tiny(engram_offload=True)
    base = MoELLM(cfg)
    outs = []
    for native in (False, True):
        m = copy.deepcopy(base)
        opt = OptimizerBundle(m, lr=3e-3, engram_native=native)
        assert opt.engram_native is False
        opt.begin()
        outs.append((_train(m, opt, cfg, 2), _state(m, opt)))
        common.clear_cached_copies()
    assert "native Engram row Adam unavailable (no compiler)" in capsys.readouterr().out
    assert outs[0][0] == outs[1][0] and all(torch.equal(outs[0][1][k], outs[1][1][k]) for k in outs[0][1])


@needs_native
def test_native_kernel_leaves_the_main_thread_running():
    """While the native kernel works on a large table the main thread's Python loop keeps running: the time it is
    stalled is a small part of the update (the torch.compile kernel stalls it for about half of its run time)."""
    R, D = 600_000, 40
    w, g, m, v = (torch.randn(R, D) * 0.01 for _ in range(4))
    v.abs_()
    ids = torch.randperm(R)[: R // 2].sort().values
    done, dur = threading.Event(), []

    def work():
        for _ in range(6):
            gg = g.clone()
            t = time.perf_counter()
            rn.row_adam_native(w, gg, m, v, ids, 1e-2, (0.9, 0.95), 1e-8, 5)
            dur.append(time.perf_counter() - t)
        done.set()
    gaps, last = [], time.perf_counter()
    th = threading.Thread(target=work)
    th.start()
    while not done.is_set():
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    th.join()
    stalled = sum(x for x in gaps if x > 0.01)
    print(f"native update {sum(dur) / len(dur) * 1000:.0f} ms; main thread stalled {stalled * 1000:.0f} ms of "
          f"{sum(dur) * 1000:.0f} ms")
    assert stalled < 0.35 * sum(dur)


def _bins(root):
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


@needs_native
def test_train_py_engram_native_bit_identical(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    data = str(tmp_path / "bins")
    _bins(data)

    def cmd(out, extra=()):
        return [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny",
                "--override", json.dumps({"engram_offload": True, "residual": "block_attnres"}),
                "--data", data, "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--out", out,
                "--seq-len", "32", "--micro-bsz", "2", "--global-tokens", "128", "--steps", "5", "--lr", "3e-3",
                "--warmup", "1", "--log-every", "1", "--eval-every", "2", "--ckpt-every", "2", "--clip", "0.05",
                "--grad-ckpt", "--offload-master", "--fused-grad-acc", "--muon-bf16", "--offload-opt", *extra]
    outs = {}
    for name, extra in (("old", ()), ("new", ("--engram-early", "--state-min-bytes", "1048576", "--async-ckpt",
                                              "--engram-native", "--engram-native-verify"))):
        out = str(tmp_path / name)
        r = subprocess.run(cmd(out, extra), env=env, capture_output=True, text=True)
        assert r.returncode == 0 and "TRAIN_DONE" in r.stdout, r.stderr[-3000:]
        recs = [json.loads(line) for line in open(os.path.join(out, "log.jsonl"))]
        recs = [{k: v for k, v in rec.items() if k not in ("tok_s", "mfu", "mem_gb")} for rec in recs]
        cks = {}
        for step in (4, 5):
            ck = torch.load(os.path.join(out, f"ckpt_{step:07d}.pt"), map_location="cpu", weights_only=False)
            ck.pop("args")
            cks[step] = ck
        outs[name] = (recs, cks, r.stdout)
    assert outs["old"][0] == outs["new"][0] and len(outs["old"][0]) == 8
    for step in (4, 5):
        T._same_tensors(outs["old"][1][step], outs["new"][1][step])
    assert "Engram CPU update kernel: native C, GIL released" in outs["new"][2]
    assert outs["new"][2].count("Engram native Adam verified") == 5   # one table, 5 optimizer steps
    assert "Engram CPU update kernel" not in outs["old"][2] and "Engram native Adam" not in outs["old"][2]
