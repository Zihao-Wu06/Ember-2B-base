#!/usr/bin/env python
"""Summarise runs: final per-source validation loss, throughput and memory.

  python scripts/compare_runs.py runs/ablation/*
"""
import json
import sys


def summarise(run_dir):
    evals, last = [], {}
    try:
        for line in open(f"{run_dir}/log.jsonl"):
            r = json.loads(line)
            if "eval" in r:
                evals.append(r)
            else:
                last = r
    except FileNotFoundError:
        return None
    return {"run": run_dir.rstrip("/").split("/")[-1], "step": last.get("step"), "tokens": last.get("tokens"),
            "tok_s": last.get("tok_s"), "mem_gb": last.get("mem_gb"), "eval": evals[-1]["eval"] if evals else {}}


def main():
    rows = [s for s in (summarise(d) for d in sys.argv[1:]) if s]
    srcs = sorted({k for r in rows for k in r["eval"] if k != "weighted"})
    head = ["run", "tokens(B)", "tok/s", "mem GB", "weighted"] + srcs
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        e = r["eval"]
        vals = [r["run"], f"{(r['tokens'] or 0) / 1e9:.2f}", f"{r['tok_s'] or 0:.0f}", f"{r['mem_gb'] or 0:.1f}",
                f"{e.get('weighted', float('nan')):.4f}"] + [f"{e.get(s, float('nan')):.4f}" for s in srcs]
        print("| " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
