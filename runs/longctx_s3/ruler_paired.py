"""(.venv-mac/bin/python runs/longctx_s3/ruler_paired.py A.json B.json [tasks]) RULER-lite at two lengths on the same
questions. ruler_lite.py draws every random choice (keys, numbers, haystack start, relative needle depths) from
Random(0) independently of the length, so two single-length runs with the same --n and --tasks ask the same questions
and only the haystack length differs. Pairs the samples, checks the answers match, and gives paired bootstrap 95%
intervals for each task's difference and for the ratio of the task means (B / A)."""
import json
import sys

import numpy as np

a, b = (json.load(open(p)) for p in sys.argv[1:3])
tasks = sys.argv[3].split(",") if len(sys.argv) > 3 else ["niah_single", "niah_multikey", "niah_multivalue"]
rng = np.random.default_rng(0)
R = 10000
la, lb = a["samples"][0]["length"], b["samples"][0]["length"]
print(f"A = {la // 1024}K, B = {lb // 1024}K (same questions, paired bootstrap, {R} resamples)")
boot_a, boot_b = np.zeros(R), np.zeros(R)
for t in tasks:
    sa = [s for s in a["samples"] if s["task"] == t]
    sb = [s for s in b["samples"] if s["task"] == t]
    n = min(len(sa), len(sb))
    assert n > 0 and all(x["want"] == y["want"] for x, y in zip(sa[:n], sb[:n])), f"{t}: questions differ"
    x = 100 * np.array([s["score"] for s in sa[:n]])
    y = 100 * np.array([s["score"] for s in sb[:n]])
    idx = rng.integers(0, n, (R, n))
    d = (y[idx] - x[idx]).mean(1)
    boot_a += x[idx].mean(1) / len(tasks)
    boot_b += y[idx].mean(1) / len(tasks)
    lo, hi = np.percentile(d, [2.5, 97.5])
    both = int(((x > 0) & (y > 0)).sum())
    print(f"  {t:16s} n={n:3d}  A {x.mean():5.1f}  B {y.mean():5.1f}  B-A {y.mean() - x.mean():+5.1f} "
          f"[{lo:+.1f}, {hi:+.1f}]  (A only {int(((x > y)).sum())}, B only {int((y > x).sum())})")
ratio = boot_b / boot_a
lo, hi = np.percentile(ratio, [2.5, 97.5])
ma = np.mean([100 * np.mean([s["score"] for s in a["samples"] if s["task"] == t]) for t in tasks])
mb = np.mean([100 * np.mean([s["score"] for s in b["samples"] if s["task"] == t]) for t in tasks])
print(f"  mean of {len(tasks)} tasks: A {ma:.1f}  B {mb:.1f}  B/A {mb / ma:.3f} [{lo:.3f}, {hi:.3f}]")
