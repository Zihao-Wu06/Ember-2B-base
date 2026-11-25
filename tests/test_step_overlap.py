"""Round 10: three exact changes that take idle time out of the end of a step.

* engram_early (train.py --engram-early): the CPU Adam update of the offloaded Engram tables is queued right after the
  last backward (OptimizerBundle.start_engram_update(), before clip_grad_norm) instead of at the end of step().
* state_min_bytes (train.py --state-min-bytes, with offload_states): small optimizer states stay on the GPU.
* background checkpoints (train.py --async-ckpt): written on a thread during the next step's forward / backward,
  joined before the next optimizer step; --ckpt-verify checks every write.

CPU: engram_early trains bit-identically (with and without the early call, across a skipped non-finite step); the
snapshot copies everything that is not in the stable set and a changed by-reference tensor is caught; train.py end to
end with all three flags gives the same logs and bit-identical checkpoints, a checkpoint written in the background
resumes under the old settings to the same final checkpoint, and a stop request while a background save may be in
flight leaves the same checkpoint as without the flag.
GPU (skipped without CUDA): state_min_bytes keeps small states on the device and every parameter / state stays
bit-identical to offloading everything and to a plain bundle; engram_early gives bit-identical tables, parameters and
states from the same gradients; a checkpoint written in the background while the next forward / backward runs equals
the one written synchronously from the same state."""
import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, wsd_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train as T  # noqa: E402

BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


def _batch(cfg, s, k, n=33):
    x = torch.randint(0, cfg.vocab_size, (2, n), generator=torch.Generator().manual_seed(1000 * s + k))
    return x[:, :-1], x[:, 1:]


def _state(m, opt):
    sd = {f"model.{k}": v.detach().cpu().clone() for k, v in m.state_dict().items()}
    for i, o in enumerate(opt.state_dict()["opts"]):
        for pi, st in o["state"].items():
            for k, v in st.items():
                if torch.is_tensor(v):
                    sd[f"opt{i}.{pi}.{k}"] = v.detach().cpu().clone()
    if m.engram is not None and m.engram.offload:
        for i, t in enumerate(m.engram.offload_state_dict()):  # flushes the queued async update first
            for k, v in t.items():
                sd[f"engram{i}.{k}"] = v.clone()
    return sd


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k


def _train(m, opt, cfg, steps, early_call, bad_step=None, clip=0.05, micro=2):
    """train.py's step on CPU (staging16 order: finite check, start_engram_update, clip, step)."""
    out = []
    for s in range(steps):
        loss_sum = torch.zeros(())
        for k in range(micro):
            x, y = _batch(cfg, s, k)
            loss, info = m(x, y)
            if s == bad_step:
                loss = loss * float("nan")
            (loss / micro).backward()
            loss_sum += loss.detach().float()
            out.append(float(info["ce"]))
        if not bool(torch.isfinite(loss_sum)):
            opt.zero_grad()
            m.engram.flush()
            for t in m.engram.cpu_tables:
                t.grad.zero_()
                t.touched.zero_()
            out.append("skipped")
            continue
        lr = wsd_lr(s, 8, 3e-3, warmup=1)
        if early_call:
            opt.start_engram_update(lr)
        out.append(float(opt.clip_grad_norm(clip)))
        opt.step(lr)
        opt.zero_grad()
        m.update_router_bias()
    return out


# ------------------------------------------------------------------------------------------------------------ CPU
@pytest.mark.parametrize("bad_step", [None, 2])
def test_engram_early_trains_bit_identically(bad_step):
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", engram_offload=True)
    base = MoELLM(cfg)
    base.set_checkpointing(0.5)
    runs = []
    for early, call in ((False, False), (True, True), (True, False), (False, True)):
        m = copy.deepcopy(base)
        opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=BF16, engram_early=early)
        opt.begin()
        r = _train(m, opt, cfg, 5, early_call=call, bad_step=bad_step)
        assert not opt._engram_queued
        runs.append((r, _state(m, opt)))
        common.clear_cached_copies()
    for r, st in runs[1:]:   # early with the call, early without it (step() queues it first), the call without the flag
        assert r == runs[0][0]
        _assert_same(runs[0][1], st)
    assert ("skipped" in runs[0][0]) == (bad_step is not None)


def test_engram_early_update_is_queued_before_the_gpu_work(monkeypatch):
    """With the flag, the table update is queued by start_engram_update() (before clip / step) and step() does not
    queue a second one; without it, at the end of step()."""
    torch.manual_seed(0)
    cfg = tiny(engram_offload=True)
    for early in (False, True):
        m = MoELLM(cfg)
        opt = OptimizerBundle(m, lr=3e-3, engram_early=early)
        opt.begin()
        calls = []
        real = opt._engram_cpu_update
        monkeypatch.setattr(opt, "_engram_cpu_update", lambda lr, n, real=real: (calls.append(("update", n)), real(lr, n)))
        real_reload = opt._reload_pipelined
        monkeypatch.setattr(opt, "_reload_pipelined", lambda rr=real_reload: (calls.append(("gpu", 0)), rr())[1])
        for s in range(2):
            x, y = _batch(cfg, s, 0)
            m(x, y)[0].backward()
            opt.start_engram_update(1e-3)
            opt.clip_grad_norm(1.0)
            opt.step(1e-3)
            opt.zero_grad()
        m.engram.flush()
        want = [("update", 1), ("gpu", 0), ("update", 2), ("gpu", 0)] if early else \
            [("gpu", 0), ("update", 1), ("gpu", 0), ("update", 2)]
        assert calls == want, (early, calls)
        common.clear_cached_copies()


def test_snapshot_copies_everything_outside_the_stable_set():
    a, b, c = torch.arange(6.0), torch.ones(3), torch.zeros(2, dtype=torch.bool)
    obj = {"model": {"w": a, "v": b}, "list": [c, 3, "x"], "tup": (b,), "n": None}
    refs = []
    snap = T._snapshot(obj, {a.data_ptr()}, refs)
    assert snap["model"]["w"] is a and [n for n, _ in refs] == [".model.w"]
    assert snap["model"]["v"] is not b and snap["list"][0] is not c and snap["tup"][0] is not b
    assert type(snap["list"]) is list and type(snap["tup"]) is tuple and snap["list"][1:] == [3, "x"]
    b.add_(5)
    c.fill_(True)
    assert torch.equal(snap["model"]["v"], torch.ones(3)) and not snap["list"][0].any()
    n, nbytes = T._same_tensors(snap, {"model": {"w": a.clone(), "v": torch.ones(3)}, "list": [torch.zeros(2, dtype=torch.bool), 3, "x"],
                                       "tup": (torch.ones(3),), "n": None})
    assert (n, nbytes) == (4, 6 * 4 + 3 * 4 + 2 + 3 * 4)
    with pytest.raises(RuntimeError, match="differs"):
        T._same_tensors(snap, {"model": {"w": a + 1, "v": torch.ones(3)}, "list": snap["list"], "tup": snap["tup"], "n": None})
    for t in (a, b.to(BF16), c, torch.arange(5), torch.zeros(0)):
        s = T._checksum(t)
        assert s == T._checksum(t.clone())
        if t.numel():
            u = t.clone()
            u.view(-1)[0] = 1 if t.dtype == torch.bool and not bool(t.view(-1)[0]) else (0 if t.dtype == torch.bool else u.view(-1)[0] + 1)
            assert T._checksum(u) != s


def _tiny_run(steps=2):
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", engram_offload=True)
    m = MoELLM(cfg)
    opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=BF16, engram_early=True)
    opt.begin()
    _train(m, opt, cfg, steps, early_call=True)
    return cfg, m, opt


def test_background_save_equals_sync_save_and_catches_changes(tmp_path):
    cfg, m, opt = _tiny_run()
    args = argparse.Namespace(lr=3e-3, note="x")
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    os.makedirs(a)
    os.makedirs(b)
    pa = T.save_ckpt(a, 2, m, opt, {"step": 4}, cfg, args)
    h = T.BackgroundSave(b, 2, m, opt, {"step": 4}, cfg, args, verify=True)
    assert len(h.refs) == 3 * len(m.engram.cpu_tables)   # on the CPU only the Engram tables are written by reference
    x, y = _batch(cfg, 9, 0)
    m(x, y)[0].backward()     # the next step's forward / backward while the file is written
    waited, note = h.join()
    assert "verified" in note and h.write_s is not None and h.obj is None
    ca = torch.load(pa, map_location="cpu", weights_only=False)
    cb = torch.load(h.path, map_location="cpu", weights_only=False)
    assert T._same_tensors(ca, cb)[0] > 20
    # a by-reference tensor that changes during the write is caught
    h2 = T.BackgroundSave(b, 3, m, opt, {"step": 6}, cfg, args, verify=True)
    m.engram.cpu_tables[0].weight[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed while"):
        h2.join()
    # a failed write is reported by join()
    h3 = T.BackgroundSave(str(tmp_path / "missing_dir"), 4, m, opt, {"step": 8}, cfg, args)
    with pytest.raises(RuntimeError, match="failed"):
        h3.join()


def _bins(root):
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
NEW = ["--engram-early", "--state-min-bytes", "1048576", "--async-ckpt", "--ckpt-verify"]


def _cmd(out, data, steps, extra=()):
    return [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny",
            "--override", json.dumps({"engram_offload": True, "residual": "block_attnres"}),
            "--data", data, "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--out", out,
            "--seq-len", "32", "--micro-bsz", "2", "--global-tokens", "128", "--steps", str(steps), "--lr", "3e-3",
            "--warmup", "1", "--log-every", "1", "--eval-every", "2", "--ckpt-every", "1", "--clip", "0.05",
            "--grad-ckpt", "--offload-master", "--fused-grad-acc", "--muon-bf16", "--offload-opt", *extra]


def _recs(out):
    recs = [json.loads(line) for line in open(os.path.join(out, "log.jsonl"))]
    return [{k: v for k, v in r.items() if k not in ("tok_s", "mfu", "mem_gb")} for r in recs]


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck.pop("args")
    return ck


def test_train_py_three_flags_bit_identical_and_resume(tmp_path):
    data = str(tmp_path / "bins")
    _bins(data)
    old, new, res = (str(tmp_path / n) for n in ("old", "new", "res"))
    r = subprocess.run(_cmd(old, data, 5), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and "TRAIN_DONE" in r.stdout, r.stderr[-3000:]
    assert "in the background" not in r.stdout and r.stdout.count("saved ") == 5
    r = subprocess.run(_cmd(new, data, 5, NEW), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and "TRAIN_DONE" in r.stdout, r.stderr[-3000:]
    assert r.stdout.count("s in the background,") == 4 and r.stdout.count("verified:") == 4, r.stdout[-3000:]
    assert "periodic checkpoints: written in the background" in r.stdout
    assert "Engram CPU update: queued right after the last backward" in r.stdout
    assert _recs(old) == _recs(new) and len(_recs(old)) == 5 + 3   # 5 steps + evals after steps 2, 4 and the last
    for step in (4, 5):   # the last two checkpoints are kept: step 4 was written in the background, step 5 at the end
        T._same_tensors(_load(os.path.join(old, f"ckpt_{step:07d}.pt")), _load(os.path.join(new, f"ckpt_{step:07d}.pt")))
    # a checkpoint written in the background resumes under the old settings to the same final checkpoint
    os.makedirs(res)
    import shutil
    shutil.copy(os.path.join(new, "ckpt_0000004.pt"), res)
    r = subprocess.run(_cmd(res, data, 5), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and "resumed from" in r.stdout and "TRAIN_DONE" in r.stdout, r.stderr[-3000:]
    T._same_tensors(_load(os.path.join(old, "ckpt_0000005.pt")), _load(os.path.join(res, "ckpt_0000005.pt")))


def test_stop_request_with_background_checkpoints(tmp_path):
    """SAVE_AND_EXIT mid-run with a background checkpoint after every step: exit code 75, no half-written file, and
    resuming the stop checkpoint under the old settings ends in the same final checkpoint as an uninterrupted old run."""
    import time
    data = str(tmp_path / "bins")
    _bins(data)
    full, out = str(tmp_path / "full"), str(tmp_path / "part")
    r = subprocess.run(_cmd(full, data, 40), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and "TRAIN_DONE" in r.stdout, r.stderr[-3000:]
    p = subprocess.Popen(_cmd(out, data, 40, NEW), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def hb_step():
        try:
            with open(os.path.join(out, "heartbeat.json")) as fh:
                return json.load(fh).get("step", -1)
        except (OSError, ValueError):
            return -1
    while p.poll() is None and hb_step() < 3:
        time.sleep(0.005)
    open(os.path.join(out, "SAVE_AND_EXIT"), "w").close()
    text = p.communicate(timeout=300)[0]
    assert p.returncode == T.EXIT_RESTART, text[-3000:]
    assert "SAVE_AND_EXIT: checkpoint" in text and "in the background" in text
    cks = sorted(f for f in os.listdir(out) if f.startswith("ckpt_"))
    assert not [f for f in cks if f.endswith(".tmp")] and 1 <= len(cks) <= 2
    k = int(cks[-1][len("ckpt_"):-len(".pt")])
    assert 4 <= k < 40
    r = subprocess.run(_cmd(out, data, 40), env=ENV, capture_output=True, text=True)   # resume with the old settings
    assert r.returncode == 0 and f"at step {k}" in r.stdout and "TRAIN_DONE" in r.stdout, r.stderr[-3000:]
    T._same_tensors(_load(os.path.join(full, "ckpt_0000040.pt")), _load(os.path.join(out, "ckpt_0000040.pt")))


# ------------------------------------------------------------------------------------------------------------ GPU
def _grads(opt, m):
    return [opt._gbuf.get(id(p), p.grad) if id(p) in opt._cast else p.grad for p in m.parameters()]


def _give(opt, m, grads):
    for p, g in zip(m.parameters(), grads):
        if g is None:
            continue
        if id(p) in opt._cast:
            opt._gbuf[id(p)] = g.clone()
        else:
            p.grad = g.clone()


def _nan_empty(real_empty):
    def empty(*args, **kw):
        t = real_empty(*args, **kw)
        if t.is_cuda and t.is_floating_point():
            t.fill_(float("nan"))
        return t
    return empty


def _gpu_cfg(**kw):
    return tiny(kda_heads=2, kda_head_dim=64, d_model=128, vocab_size=65536, residual_bf16=True,
                residual="block_attnres", moe_impl="triton", **kw)


def _param_and_state_equal(models, opts, where):
    torch.cuda.synchronize()
    ps = [list(m.parameters()) for m in models]
    for i, group in enumerate(zip(*ps)):
        ref = group[0].detach().cpu()
        assert torch.isfinite(ref).all(), (where, i)
        for other in group[1:]:
            assert torch.equal(ref, other.detach().cpu()), (where, i)
    sds = [o.state_dict()["opts"] for o in opts]
    for group in zip(*sds):
        for pi in group[0]["state"]:
            for key, v in group[0]["state"][pi].items():
                if torch.is_tensor(v):
                    for other in group[1:]:
                        assert torch.equal(v.cpu(), other["state"][pi][key].cpu()), (where, pi, key)


@gpu
@pytest.mark.parametrize("muon_impl", ["loop", "batched_fused"])
def test_gpu_state_min_bytes_bit_identical(monkeypatch, muon_impl):
    """Three bundles fed the same gradients for 3 steps (clip active): offload every state, keep states below 1 MiB on
    the GPU, and a plain bundle. The side stream is stalled before every step and reload destinations start as NaN."""
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_layers=())
    base = MoELLM(cfg).cuda()
    base.set_checkpointing(0.5)
    ms = [copy.deepcopy(base) for _ in range(3)]
    kw = dict(lr=1e-3, muon_state_dtype=BF16, muon_impl=muon_impl)
    oa = OptimizerBundle(ms[0], offload_states=True, offload_master=True, fused_grad_acc=True, **kw)
    ob = OptimizerBundle(ms[1], offload_states=True, offload_master=True, fused_grad_acc=True, state_min_bytes=1 << 20, **kw)
    oc = OptimizerBundle(ms[2], **kw)
    for o in (oa, ob, oc):
        o.begin()
    real_empty = torch.empty
    try:
        for step in range(3):
            for k in range(2):
                x = _batch(cfg, step, k, n=129)
                x = torch.cat([x[0], x[1][:, -1:]], 1).cuda()
                with torch.autocast("cuda", dtype=BF16):
                    loss, _ = ms[0](x[:, :-1], x[:, 1:])
                (loss / 2).backward()
            g = [None if t is None else t.clone() for t in _grads(oa, ms[0])]
            _give(ob, ms[1], g)
            _give(oc, ms[2], g)
            norms = [o.clip_grad_norm(0.05) for o in (oa, ob, oc)]
            assert torch.equal(norms[0], norms[1]) and torch.equal(norms[0], norms[2])
            monkeypatch.setattr(torch, "empty", _nan_empty(real_empty))
            for o in (oa, ob, oc):
                if o.offload_states:
                    with torch.cuda.stream(o._side_stream()):
                        torch.cuda._sleep(100_000_000)
                o.step(1e-3)
            monkeypatch.setattr(torch, "empty", real_empty)
            for o in (oa, ob, oc):
                o.zero_grad()
            ms[0].update_router_bias()
            _param_and_state_equal(ms, (oa, ob, oc), step)
            for (_, ca), (_, cb) in zip(oa._cast.values(), ob._cast.values()):
                assert torch.equal(ca, cb)
        # where the states live between steps: all in pinned memory vs only the large ones
        def placement(o):
            on_gpu = on_cpu = small_cpu = big_gpu = 0
            for opt in o.optimizers():
                for st in opt.state.values():
                    for v in st.values():
                        if torch.is_tensor(v):
                            nb = v.numel() * v.element_size()
                            on_gpu += v.is_cuda
                            on_cpu += not v.is_cuda
                            small_cpu += (not v.is_cuda) and nb < (1 << 20)
                            big_gpu += v.is_cuda and nb >= (1 << 20)
            return on_gpu, on_cpu, small_cpu, big_gpu
        a, b = placement(oa), placement(ob)
        print(f"states on GPU / in pinned memory: offload all {a[0]} / {a[1]}, state_min_bytes 1 MiB {b[0]} / {b[1]}")
        assert a[0] == 0 and a[1] > 0
        assert b[0] > 0 and b[1] > 0 and b[2] == 0 and b[3] == 0
        assert len(ob.stable_host_ptrs()) < len(oa.stable_host_ptrs())
    finally:
        monkeypatch.setattr(torch, "empty", real_empty)
        common.clear_cached_copies()


@gpu
def test_gpu_engram_early_bit_identical():
    """Offloaded Engram tables on a GPU model, offload_states + offload_master: the default order and engram_early
    (queued before clip_grad_norm) get the same GPU gradients and the same table gradients for 3 steps."""
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_offload=True)
    base = MoELLM(cfg).cuda()
    base.set_checkpointing(0.5)
    ms = [copy.deepcopy(base) for _ in range(2)]
    kw = dict(lr=1e-3, muon_state_dtype=BF16, offload_states=True, offload_master=True, fused_grad_acc=True,
              muon_impl="batched_fused", state_min_bytes=1 << 20)
    oa, ob = OptimizerBundle(ms[0], **kw), OptimizerBundle(ms[1], engram_early=True, **kw)
    oa.begin()
    ob.begin()
    try:
        for step in range(3):
            for k in range(2):
                x = _batch(cfg, step, k, n=129)
                x = torch.cat([x[0], x[1][:, -1:]], 1).cuda()
                with torch.autocast("cuda", dtype=BF16):
                    loss, _ = ms[0](x[:, :-1], x[:, 1:])
                (loss / 2).backward()
            torch.cuda.synchronize()
            ms[0].engram.flush()   # the table gradients of model 0 have landed: give model 1 the same ones
            _give(ob, ms[1], [None if t is None else t.clone() for t in _grads(oa, ms[0])])
            touched = 0
            for ta, tb in zip(ms[0].engram.cpu_tables, ms[1].engram.cpu_tables):
                tb.grad.copy_(ta.grad)
                tb.touched.copy_(ta.touched)
                touched += int(ta.touched.sum())
            assert touched > 0
            na = oa.clip_grad_norm(0.05)
            oa.step(1e-3)
            ob.start_engram_update(1e-3)
            nb = ob.clip_grad_norm(0.05)
            ob.step(1e-3)
            assert torch.equal(na, nb)
            for o in (oa, ob):
                o.zero_grad()
            ms[0].update_router_bias()
            _param_and_state_equal(ms, (oa, ob), step)
            for ta, tb in zip(ms[0].engram.offload_state_dict(), ms[1].engram.offload_state_dict()):
                for key in ta:
                    assert torch.equal(ta[key], tb[key]), (step, key)
            for ta, tb in zip(ms[0].engram.cpu_tables, ms[1].engram.cpu_tables):
                assert not ta.touched.any() and not tb.touched.any() and not tb.grad.any()
    finally:
        common.clear_cached_copies()


@gpu
def test_gpu_background_save_matches_sync(tmp_path):
    """GPU model with offloaded masters, states (small ones resident) and Engram tables: a checkpoint written in the
    background while the next step's forward / backward passes run is verified and equals the synchronous one."""
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_offload=True)
    m = MoELLM(cfg).cuda()
    m.set_checkpointing(0.5)
    opt = OptimizerBundle(m, lr=1e-3, muon_state_dtype=BF16, offload_states=True, offload_master=True,
                          fused_grad_acc=True, engram_early=True, state_min_bytes=1 << 20, muon_impl="batched_fused")
    opt.begin()

    def fwd_bwd(step):
        for k in range(2):
            x = _batch(cfg, step, k, n=129)
            x = torch.cat([x[0], x[1][:, -1:]], 1).cuda()
            with torch.autocast("cuda", dtype=BF16):
                loss, _ = m(x[:, :-1], x[:, 1:])
            (loss / 2).backward()

    def opt_step():
        torch.cuda.synchronize()
        opt.start_engram_update(1e-3)
        opt.clip_grad_norm(0.05)
        opt.step(1e-3)
        opt.zero_grad()
        m.update_router_bias()

    args = argparse.Namespace(lr=1e-3)
    try:
        for step in range(2):
            fwd_bwd(step)
            opt_step()
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        os.makedirs(a)
        os.makedirs(b)
        pa = T.save_ckpt(a, 2, m, opt, {"step": 4}, cfg, args)
        h = T.BackgroundSave(b, 2, m, opt, {"step": 4}, cfg, args, verify=True)
        names = [n for n, _ in h.refs]
        assert any(n.startswith(".model.") for n in names) and any(n.startswith(".opt.") for n in names) \
            and any(n.startswith(".engram_offload.") for n in names), names[:5]
        fwd_bwd(2)                      # the next step's micro-batches while the file is written
        waited, note = h.join()
        assert "verified" in note
        opt_step()                      # only now are the masters, states and tables rewritten
        ca = torch.load(pa, map_location="cpu", weights_only=False)
        cb = torch.load(h.path, map_location="cpu", weights_only=False)
        n, nbytes = T._same_tensors(ca, cb)
        print(f"background vs synchronous checkpoint: {n} tensors, {nbytes / 2 ** 20:.1f} MiB identical; "
              f"{len(names)} tensors by reference")
        assert n > 50
    finally:
        common.clear_cached_copies()
