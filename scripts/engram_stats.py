#!/usr/bin/env python
"""Check Engram table sizing on real tokens.

- distinct canonical 2-/3-grams in a sample, and how much of all occurrences the most frequent
  `rows_per_head` n-grams cover (the rest share rows through hashing; multi-head hashing makes a
  collision on all heads at once very unlikely)
- unique rows touched per micro-batch and per optimizer step -> real cost of the lazy Adam step

  python scripts/engram_stats.py --sources code math en_web --tokens 2e7
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm.config import main_2b  # noqa: E402
from moe_llm.layers.engram import EngramMemory  # noqa: E402
from moe_llm.optim import sparse_row_adam_  # noqa: E402


def load_tokens(src, n):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "bins", src, "*.bin")))
    out, got = [], 0
    for f in files:
        a = np.memmap(f, dtype=np.uint16, mode="r")
        take = min(len(a), n - got)
        out.append(np.asarray(a[:take], dtype=np.int64))
        got += take
        if got >= n:
            break
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["code", "math", "en_web"])
    ap.add_argument("--tokens", type=float, default=2e7)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--accum", type=int, default=64)
    args = ap.parse_args()
    cfg = main_2b().replace(engram_offload=False)
    canon = np.load(os.path.join(ROOT, "data", "tokenizer", "canon_map.npy")).astype(np.int64)
    Vc = int(canon.max()) + 2
    with torch.device("meta"):
        pass
    mem = EngramMemory(cfg.replace(engram_rows=cfg.engram_rows))  # buffers only; tables are big -> use indices()
    mem.tables = None
    mem.set_canon_map(canon)
    rows_per_head = int(mem.mods[0].float().mean())
    print(f"rows per head ~{rows_per_head}, heads/layer {mem.mods.shape[1]}, rows/layer {mem.rows[0]}")
    for src in args.sources:
        x = load_tokens(src, int(args.tokens))
        c = canon[x]
        for n in (2, 3):
            key = c[n - 1:].copy()
            for i in range(1, n):
                key = key + c[n - 1 - i:len(c) - i] * (Vc ** i)
            _, cnt = np.unique(key, return_counts=True)
            cnt = np.sort(cnt)[::-1]
            cover = cnt[:rows_per_head].sum() / cnt.sum()
            print(f"{src:8s} {n}-grams in {len(x) / 1e6:.0f}M tokens: {len(cnt) / 1e6:.2f}M distinct, "
                  f"top {rows_per_head} cover {cover:.1%} of occurrences")
        # rows actually touched (all 16 heads of layer 0) per micro-batch and per optimizer step
        ids = torch.from_numpy(x[: args.micro_tokens * args.accum].reshape(-1, args.micro_tokens))
        idx = mem.indices(ids)[0]  # [accum, micro, heads]
        per_micro = np.mean([torch.unique(idx[i]).numel() for i in range(min(8, idx.shape[0]))])
        per_step = torch.unique(idx).numel()
        print(f"{src:8s} unique rows/layer: {per_micro:.0f} per micro-batch, {per_step} per step "
              f"({per_step / mem.rows[0]:.1%} of the table)")
    # time the lazy Adam step at that size (one layer, fp32 on CPU)
    R, D = mem.rows[0], cfg.engram_d_head
    w, g, m, v = (torch.zeros(R, D) for _ in range(4))
    ids = torch.randperm(R)[:per_step].sort().values
    t0 = time.time()
    sparse_row_adam_(w, g, m, v, ids, 1e-2, step=1)
    print(f"lazy Adam on {per_step} rows x {D}: {time.time() - t0:.2f} s per layer per optimizer step")


if __name__ == "__main__":
    main()
