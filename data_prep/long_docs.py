#!/usr/bin/env python
"""Index the long documents of each source for context extension (docs/PLAN_LONGCTX.md).

The bins are uint16 token streams with EOS (id 0) after every document. For each source this writes
<bins>/<source>/long_docs.npz with
    docs   [N, 2] int64: (first token, length) of every document of at least --min-len tokens, in the coordinates of
           the source's concatenated stream (moe_llm.data.TokenStream: its .bin files in sorted order)
    total  the stream's token count (the loader refuses an index built for another stream)
moe_llm.data.MixtureLoader reads it for the mixture entries "long:<source>". Only reads the bins (sequentially, in
chunks); prints how many tokens and windows each source offers at 32K / 64K / 128K, outside the held-out tail.

    python data_prep/long_docs.py --bins data/bins_main            # all sources
    python data_prep/long_docs.py --bins data/bins_main --only code,math
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm.data import LONG_INDEX, TokenStream, long_window_starts  # noqa: E402

EOS = 0
CHUNK = 1 << 26  # 64M tokens = 128 MB per read


def eos_positions(ts: TokenStream) -> np.ndarray:
    out = []
    for base, m in zip(ts.cum[:-1].tolist(), ts.maps):
        for a in range(0, len(m), CHUNK):
            part = np.asarray(m[a:a + CHUNK])
            out.append(np.flatnonzero(part == EOS).astype(np.int64) + base + a)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def index_source(src_dir: str, min_len: int, val_tokens: int) -> dict:
    t0 = time.time()
    ts = TokenStream(src_dir)
    eos = eos_positions(ts)
    prev = np.concatenate([[-1], eos[:-1]])  # the stream's first document has no EOS before it
    n = eos - prev - 1
    keep = n >= min_len
    docs = np.stack([prev[keep] + 1, n[keep]], axis=1).astype(np.int64)
    np.savez(os.path.join(src_dir, LONG_INDEX), docs=docs, total=np.int64(ts.total), min_len=np.int64(min_len))
    train_end = ts.total - min(val_tokens, ts.total // 10)
    row = {"total": ts.total, "docs": len(eos), "secs": time.time() - t0}
    for L in (32768, 65536, 131072):
        sel = docs[docs[:, 1] > L // 2]
        row[f"tok>{L // 2048}K"] = int(sel[:, 1].sum())
        row[f"win{L // 1024}K"] = len(long_window_starts(sel, train_end, L))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", default=os.path.join(ROOT, "data", "bins_main"))
    ap.add_argument("--only", default="", help="comma-separated sources (default: every directory with .bin files)")
    ap.add_argument("--min-len", type=int, default=16384)
    ap.add_argument("--val-tokens", type=int, default=2_000_000, help="the loader's held-out tail (same default)")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else sorted(
        d for d in os.listdir(args.bins)
        if os.path.isdir(os.path.join(args.bins, d)) and any(f.endswith(".bin") for f in os.listdir(os.path.join(args.bins, d))))
    print("source: stream tokens, documents | tokens in documents > 16K / 32K / 64K | windows at 32K / 64K / 128K "
          "(outside the held-out tail)", flush=True)
    for name in names:
        r = index_source(os.path.join(args.bins, name), args.min_len, args.val_tokens)
        print(f"  {name}: {r['total'] / 1e9:.2f}B, {r['docs']} docs | {r['tok>16K'] / 1e9:.3f}B / "
              f"{r['tok>32K'] / 1e9:.3f}B / {r['tok>64K'] / 1e9:.3f}B | {r['win32K']} / {r['win64K']} / "
              f"{r['win128K']} [{r['secs']:.0f}s]", flush=True)
    print("LONG_DOCS_DONE", flush=True)


if __name__ == "__main__":
    main()
