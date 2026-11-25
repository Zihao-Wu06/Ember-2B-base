#!/usr/bin/env python
"""Keep a long training run alive on a desktop GPU box (WSL2 + Windows WDDM driver).

    python scripts/supervise.py --out runs/main -- python scripts/train.py --out runs/main ...

* Restarts train.py when it exits with an error; train.py resumes from its newest checkpoint.
  Before every start it waits until the GPU memory of the previous process has been released.
* Watches <out>/heartbeat.json, which train.py rewrites after every step. If training stalls, or
  steps take much longer than usual several times in a row (when VRAM runs out, Windows pages it to
  system RAM instead of raising OOM and training gets ~20x slower), it creates <out>/SAVE_AND_EXIT:
  train.py checkpoints at the end of the current step, exits with code 75 and is started again.
  A process that does not exit within --exit-timeout-min is killed.
* `touch <out>/PAUSE` checkpoints and stops training (frees the GPU) until the file is removed.
* SIGTERM / Ctrl-C: checkpoint, stop train.py, exit.
* Gives up after more than --max-restarts unplanned restarts within --window-h hours and writes
  <out>/SUPERVISOR_STOPPED.
Log: <out>/supervisor.log (train.py output is appended to <out>/stdout.log).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import statistics
import subprocess
import sys
import time

EXIT_RESTART = 75  # train.py: checkpoint saved on request, start again to continue


class Supervisor:
    def __init__(self, args, cmd):
        self.a = args
        self.cmd = cmd
        self.out = args.out
        self.hb_path = os.path.join(self.out, "heartbeat.json")
        self.flag = os.path.join(self.out, "SAVE_AND_EXIT")
        self.pause = os.path.join(self.out, "PAUSE")
        self.dts = collections.deque(maxlen=200)  # recent normal step times, kept across restarts
        self.unplanned = collections.deque()      # times of restarts after crashes, stalls, slowdowns
        self.stop_sig = None

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(os.path.join(self.out, "supervisor.log"), "a") as fh:
            fh.write(line + "\n")

    def on_signal(self, signum, frame):
        self.stop_sig = signal.Signals(signum).name

    @staticmethod
    def gpu_used_gb():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=60)
            return float(r.stdout.split()[0]) / 1024
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None  # no GPU visible: nothing to wait for

    def heartbeat(self, since):
        try:
            with open(self.hb_path) as fh:
                hb = json.load(fh)
        except (OSError, ValueError):
            return None
        return hb if hb.get("t", 0) >= since else None  # ignore one left by an earlier process

    def wait_until_runnable(self):
        said = False
        while os.path.exists(self.pause) and self.stop_sig is None:
            if not said:
                self.log(f"paused: remove {self.pause} to resume")
                said = True
            time.sleep(self.a.poll_s)
        said = False
        while self.a.gpu_free_gb > 0 and self.stop_sig is None and not os.path.exists(self.pause):
            used = self.gpu_used_gb()
            if used is None or used < self.a.gpu_free_gb:
                return
            if not said:
                self.log(f"waiting for GPU memory to be released ({used:.1f} GB in use)")
                said = True
            time.sleep(max(self.a.poll_s, 15.0))

    def kill(self, proc):
        for sig, wait in ((signal.SIGTERM, self.a.kill_grace_s), (signal.SIGKILL, 60)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(wait)
                return
            except subprocess.TimeoutExpired:
                pass

    def run_once(self):
        """Start train.py and watch it until it exits. Returns (exit code, why an exit was requested)."""
        if os.path.exists(self.flag):
            os.remove(self.flag)
        start = time.time()
        with open(os.path.join(self.out, "stdout.log"), "a") as so:
            proc = subprocess.Popen(self.cmd, stdout=so, stderr=subprocess.STDOUT, start_new_session=True)
        self.log(f"started train.py (pid {proc.pid})")
        first = last = None
        slow = 0
        reason = asked = None
        while proc.poll() is None:
            time.sleep(self.a.poll_s)
            now = time.time()
            if reason is not None:
                if now - asked > self.a.exit_timeout_min * 60:
                    self.log(f"train.py did not exit within {self.a.exit_timeout_min:g} min: killing it")
                    self.kill(proc)
                continue
            hb = self.heartbeat(start)
            quiet = now - (hb["t"] if hb else start)
            if os.path.exists(self.pause):
                reason = "PAUSE file"
            elif self.stop_sig is not None:
                reason = f"supervisor received {self.stop_sig}"
            elif quiet > self.a.stall_min * 60:
                reason = f"stalled: no progress for {quiet / 60:.0f} min"
            elif hb and hb.get("phase") == "train" and hb.get("step") != last:
                last = hb["step"]
                first = last if first is None else first
                if last - first >= self.a.warmup_steps:  # first steps after a start include autotuning
                    typical = statistics.median(self.dts) if len(self.dts) >= self.a.min_samples else None
                    if typical and hb["dt"] > self.a.slow_factor * typical:
                        slow += 1
                        if slow >= self.a.slow_steps:
                            reason = f"slow: {slow} steps in a row, last {hb['dt']:.1f}s (typical {typical:.1f}s)"
                    else:
                        slow = 0
                        self.dts.append(hb["dt"])
            if reason is not None:
                self.log(f"asking train.py to checkpoint and exit: {reason}")
                open(self.flag, "w").close()
                asked = now
        return proc.returncode, reason

    def run(self):
        os.makedirs(self.out, exist_ok=True)
        stopped = os.path.join(self.out, "SUPERVISOR_STOPPED")
        if os.path.exists(stopped):
            os.remove(stopped)
        self.log(f"supervising: {' '.join(self.cmd)}")
        while True:
            self.wait_until_runnable()
            if self.stop_sig is not None:
                self.log(f"stopping ({self.stop_sig})")
                return 0
            if os.path.exists(self.pause):
                continue
            rc, reason = self.run_once()
            if os.path.exists(self.flag):
                os.remove(self.flag)
            if rc == 0:
                self.log("training finished")
                return 0
            if self.stop_sig is not None:
                self.log(f"train.py exited with code {rc}; stopping ({self.stop_sig})")
                return 0
            if rc == EXIT_RESTART and reason == "PAUSE file":
                self.log("train.py checkpointed and exited (paused)")
                continue
            self.log(f"train.py exited with code {rc}" + (f" ({reason})" if reason else ""))
            now = time.time()
            self.unplanned.append(now)
            while now - self.unplanned[0] > self.a.window_h * 3600:
                self.unplanned.popleft()
            if len(self.unplanned) > self.a.max_restarts:
                msg = f"{len(self.unplanned)} restarts within {self.a.window_h:g} h: giving up"
                self.log(msg)
                with open(stopped, "w") as fh:
                    fh.write(msg + "\n")
                return 1
            time.sleep(self.a.cooldown_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="train.py --out (heartbeat, flags and logs live here)")
    ap.add_argument("--stall-min", type=float, default=30.0, help="no heartbeat for this long = stalled")
    ap.add_argument("--slow-factor", type=float, default=2.5, help="slow step = this many times the typical")
    ap.add_argument("--slow-steps", type=int, default=3, help="consecutive slow steps before restarting")
    ap.add_argument("--warmup-steps", type=int, default=10, help="steps after each start that are not judged")
    ap.add_argument("--min-samples", type=int, default=20, help="normal steps seen before judging speed")
    ap.add_argument("--exit-timeout-min", type=float, default=60.0)
    ap.add_argument("--kill-grace-s", type=float, default=120.0)
    ap.add_argument("--gpu-free-gb", type=float, default=3.0,
                    help="start only when GPU memory in use is below this (0 = do not check)")
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--window-h", type=float, default=6.0)
    ap.add_argument("--cooldown-s", type=float, default=30.0, help="pause between an exit and the restart")
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- train.py command")
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("missing the training command after --")
    sup = Supervisor(args, cmd)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, sup.on_signal)
    return sup.run()


if __name__ == "__main__":
    sys.exit(main())
