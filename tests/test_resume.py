"""End-to-end: train.py interrupted after a checkpoint and resumed must reproduce the uninterrupted
run exactly (CPU, tiny model, background data prefetch + Engram CPU offload enabled)."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def _bins(root):
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


def _cmd(out, data, steps=12, extra=()):
    return [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny",
            "--override", json.dumps({"engram_offload": True, "residual": "block_attnres"}),
            "--data", data, "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--out", out,
            "--seq-len", "32", "--micro-bsz", "2", "--global-tokens", "128", "--steps", str(steps), "--lr", "3e-3",
            "--warmup", "2", "--log-every", "1", "--eval-every", "0", "--ckpt-every", "6", "--grad-ckpt", *extra]


def _log(out):
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"ce"' in l]


def _train(out, data, steps=12, extra=()):
    subprocess.run(_cmd(out, data, steps, extra), check=True, env=ENV, capture_output=True, text=True)
    return _log(out)


def _heartbeat_step(out):
    try:
        with open(os.path.join(out, "heartbeat.json")) as fh:
            hb = json.load(fh)
        return hb.get("step", -1), hb.get("pid")
    except (OSError, ValueError):
        return -1, None


def _same(full, resumed):
    for a, b in zip(full, resumed):
        assert a["step"] == b["step"] and abs(a["ce"] - b["ce"]) < 1e-6, (a, b)


def test_resume_is_exact(tmp_path):
    data = str(tmp_path / "bins")
    _bins(data)
    full = _train(str(tmp_path / "full"), data)
    part = str(tmp_path / "part")
    _train(part, data)
    os.remove(os.path.join(part, "ckpt_0000012.pt"))          # pretend we crashed after step 6
    open(os.path.join(part, "log.jsonl"), "w").close()
    resumed = _train(part, data)
    assert [r["step"] for r in resumed] == list(range(6, 12))
    _same(full[6:], resumed)


def test_data_anneal_switches_at_decay_and_resumes_exactly(tmp_path):
    """--anneal-mix switches the data at the first LR-decay step (12 steps, decay_frac 0.2 -> step 10):
    earlier steps match a run without annealing, later ones differ, and a resume across it is exact."""
    data = str(tmp_path / "bins")
    _bins(data)
    ann = ["--anneal-mix", json.dumps({"a": 0.05, "b": 0.95})]
    plain = _train(str(tmp_path / "plain"), data)
    full_out = str(tmp_path / "full")
    r = subprocess.run(_cmd(full_out, data, extra=ann), check=True, env=ENV, capture_output=True, text=True)
    assert "from step 10 (LR decay)" in r.stdout and "data anneal: mix" in r.stdout, r.stdout
    full = _log(full_out)
    _same(plain[:10], full[:10])
    assert any(abs(a["ce"] - b["ce"]) > 1e-4 for a, b in zip(plain[10:], full[10:]))
    part = str(tmp_path / "part")
    _train(part, data, extra=ann)
    os.remove(os.path.join(part, "ckpt_0000012.pt"))          # resume from step 6, before the switch
    open(os.path.join(part, "log.jsonl"), "w").close()
    resumed = _train(part, data, extra=ann)
    assert [r["step"] for r in resumed] == list(range(6, 12))
    _same(full[6:], resumed)


def test_mid_mix_switches_and_resumes_exactly(tmp_path):
    """--mid-mix / --mid-from: steps before --mid-from match a run without it, later ones differ, and a resume
    after the switch is exact."""
    data = str(tmp_path / "bins")
    _bins(data)
    mid = ["--mid-mix", json.dumps({"a": 0.05, "b": 0.95}), "--mid-from", "4"]
    plain = _train(str(tmp_path / "plain"), data)
    full_out = str(tmp_path / "full")
    r = subprocess.run(_cmd(full_out, data, extra=mid), check=True, env=ENV, capture_output=True, text=True)
    assert "from step 4" in r.stdout and "data mid: mix" in r.stdout, r.stdout
    full = _log(full_out)
    _same(plain[:4], full[:4])
    assert any(abs(a["ce"] - b["ce"]) > 1e-4 for a, b in zip(plain[4:], full[4:]))
    part = str(tmp_path / "part")
    _train(part, data, extra=mid)
    os.remove(os.path.join(part, "ckpt_0000012.pt"))          # resume from step 6, after the switch
    open(os.path.join(part, "log.jsonl"), "w").close()
    resumed = _train(part, data, extra=mid)
    assert [r["step"] for r in resumed] == list(range(6, 12))
    _same(full[6:], resumed)


def test_switch_to_mid_mix_from_a_checkpoint(tmp_path):
    """The main-run switch: a run without --mid-mix checkpoints at step 6 and is resumed with --mid-from 6. It
    continues exactly like a run that had the mid mix scheduled from the start."""
    data = str(tmp_path / "bins")
    _bins(data)
    mid = ["--mid-mix", json.dumps({"a": 0.05, "b": 0.95}), "--mid-from", "6"]
    ref = _train(str(tmp_path / "ref"), data, extra=mid)
    part = str(tmp_path / "part")
    _train(part, data)                                         # the old command, no mid mix
    os.remove(os.path.join(part, "ckpt_0000012.pt"))
    open(os.path.join(part, "log.jsonl"), "w").close()
    r = subprocess.run(_cmd(part, data, extra=mid), check=True, env=ENV, capture_output=True, text=True)
    assert "from step 6 [active]" in r.stdout, r.stdout
    resumed = _log(part)
    assert [x["step"] for x in resumed] == list(range(6, 12))
    _same(ref[6:], resumed)


def test_save_and_exit_then_resume_from_backup(tmp_path):
    """SAVE_AND_EXIT mid-run -> checkpoint + exit code 75. With --ckpt-dir lost, the resume falls back
    to <out>/backup and still reproduces the uninterrupted run."""
    data = str(tmp_path / "bins")
    _bins(data)
    full = _train(str(tmp_path / "full"), data, steps=40)
    out, ck = str(tmp_path / "part"), str(tmp_path / "fast_disk")
    extra = ["--ckpt-dir", ck, "--ckpt-every", "1000", "--backup-every", "1"]
    p = subprocess.Popen(_cmd(out, data, 40, extra), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    while p.poll() is None and _heartbeat_step(out)[0] < 3:
        time.sleep(0.005)
    open(os.path.join(out, "SAVE_AND_EXIT"), "w").close()
    assert p.wait(120) == 75, p.stdout.read()
    saved = sorted(os.listdir(ck))
    assert len(saved) == 1 and sorted(os.listdir(os.path.join(out, "backup"))) == saved
    k = int(saved[0][len("ckpt_"):-len(".pt")])
    assert 4 <= k < 40
    shutil.rmtree(ck)
    open(os.path.join(out, "log.jsonl"), "w").close()
    resumed = _train(out, data, 40, extra)
    assert [r["step"] for r in resumed] == list(range(k, 40))
    _same(full[k:], resumed)
    assert "ckpt_0000040.pt" in os.listdir(os.path.join(out, "backup"))   # final model copied to --out


def test_supervisor_restarts_killed_run_exactly(tmp_path):
    """train.py SIGKILLed mid-run under scripts/supervise.py is restarted, resumes from its last
    checkpoint and ends exactly where the uninterrupted run ends."""
    data = str(tmp_path / "bins")
    _bins(data)
    full = _train(str(tmp_path / "full"), data, steps=30)
    out = str(tmp_path / "sup")
    sup = [sys.executable, os.path.join(ROOT, "scripts", "supervise.py"), "--out", out, "--poll-s", "0.05",
           "--cooldown-s", "0", "--gpu-free-gb", "0", "--"] + _cmd(out, data, 30, ["--ckpt-every", "5"])
    p = subprocess.Popen(sup, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while p.poll() is None:
        step, pid = _heartbeat_step(out)
        if step >= 12 and pid:
            os.kill(pid, signal.SIGKILL)
            break
        time.sleep(0.005)
    assert p.wait(300) == 0, p.stdout.read()
    log = open(os.path.join(out, "supervisor.log")).read()
    assert log.count("started train.py") == 2 and "exited with code -9" in log, log
    last = {r["step"]: r for r in _log(out)}           # steps after the last checkpoint ran twice
    assert sorted(last) == list(range(30))
    _same(full, [last[s] for s in range(30)])


