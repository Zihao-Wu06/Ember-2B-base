"""scripts/supervise.py against a fake trainer (fast, CPU only, no GPU check)."""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Behaviour per launch comes from a plan: crash, slow, hang, ok; anything else runs until asked to stop.
FAKE = r'''
import json, os, sys, time
out, plan = sys.argv[1], json.loads(sys.argv[2])
launches = os.path.join(out, "launches.txt")
n = len(open(launches).read().split()) if os.path.exists(launches) else 0
with open(launches, "a") as fh:
    fh.write(f"{time.time()}\n")
mode = plan[min(n, len(plan) - 1)]
flag = os.path.join(out, "SAVE_AND_EXIT")

def hb(step, dt):
    tmp = os.path.join(out, "heartbeat.json.tmp")
    with open(tmp, "w") as fh:
        json.dump({"t": time.time(), "step": step, "dt": dt, "phase": "train"}, fh)
    os.replace(tmp, os.path.join(out, "heartbeat.json"))

for step in range(100000):
    if mode == "crash" and step == 3:
        sys.exit(1)
    if mode == "ok" and step == 20:
        sys.exit(0)
    if mode == "hang" and step == 3:
        time.sleep(3600)                       # no more progress, ignores SAVE_AND_EXIT
    hb(step, 1.0 if mode == "slow" and step >= 60 else 0.01)
    time.sleep(0.01)
    if os.path.exists(flag):
        sys.exit(75)
'''


def _start(tmp_path, plan, *extra):
    out = tmp_path / "out"
    out.mkdir()
    fake = tmp_path / "fake_train.py"
    fake.write_text(FAKE)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "supervise.py"), "--out", str(out), "--poll-s", "0.02",
           "--cooldown-s", "0", "--gpu-free-gb", "0", "--warmup-steps", "2", "--min-samples", "5", *extra,
           "--", sys.executable, str(fake), str(out), json.dumps(plan)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), out


def _read(path):
    return path.read_text() if path.exists() else ""


def _launches(out):
    return len(_read(out / "launches.txt").split())


def _wait_for(cond, timeout=30):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout
        time.sleep(0.01)


def test_restarts_after_crash_and_slowdown(tmp_path):
    p, out = _start(tmp_path, ["crash", "slow", "ok"])
    assert p.wait(60) == 0, p.stdout.read()
    log = _read(out / "supervisor.log")
    assert _launches(out) == 3
    assert "exited with code 1" in log and "slow: 3 steps in a row" in log and "training finished" in log, log


def test_pause_checkpoints_and_waits(tmp_path):
    p, out = _start(tmp_path, ["run", "ok"])
    _wait_for(lambda: (out / "heartbeat.json").exists())
    (out / "PAUSE").touch()
    _wait_for(lambda: "paused:" in _read(out / "supervisor.log"))
    time.sleep(0.5)
    assert _launches(out) == 1
    (out / "PAUSE").unlink()
    assert p.wait(60) == 0, p.stdout.read()
    assert _launches(out) == 2 and "checkpointed and exited (paused)" in _read(out / "supervisor.log")


def test_stall_is_killed_and_restarted(tmp_path):
    p, out = _start(tmp_path, ["hang", "ok"], "--stall-min", "0.02", "--exit-timeout-min", "0.01",
                    "--kill-grace-s", "1")
    assert p.wait(60) == 0, p.stdout.read()
    log = _read(out / "supervisor.log")
    assert "stalled" in log and "killing it" in log and "training finished" in log, log
    assert _launches(out) == 2


def test_gives_up_on_crash_loop(tmp_path):
    p, out = _start(tmp_path, ["crash"], "--max-restarts", "2")
    assert p.wait(60) == 1, p.stdout.read()
    assert (out / "SUPERVISOR_STOPPED").exists() and _launches(out) == 3


def test_sigterm_checkpoints_and_stops(tmp_path):
    p, out = _start(tmp_path, ["run"])
    _wait_for(lambda: (out / "heartbeat.json").exists())
    p.terminate()
    assert p.wait(60) == 0, p.stdout.read()
    log = _read(out / "supervisor.log")
    assert "supervisor received SIGTERM" in log and "stopping (SIGTERM)" in log and _launches(out) == 1, log
