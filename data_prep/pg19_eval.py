#!/usr/bin/env python
"""Tokenize the PG19 evaluation books (source pg19_eval in sources.py: test 100 books, validation 50) for
runs/mac_eval/long_ctx.py --books: data/eval/pg19_<split>.npz with
    ids      uint16, every book's tokens one after another (no EOS; the same encoding as tokenize_to_bins.py)
    offsets  int64 [n_books + 1], book i is ids[offsets[i]:offsets[i + 1]]
    titles   str [n_books]
Never part of the training bins. Needs pyarrow (run in WSL).

    python data_prep/pg19_eval.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from textio import ROOT  # noqa: E402
from tok_lib import strip_specials  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw", "pg19_eval", "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer", "tokenizer.json"))
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    os.makedirs(a.out, exist_ok=True)
    for split in ("test", "validation"):
        (path,) = glob.glob(os.path.join(a.raw, f"{split}-*.parquet"))
        t = pq.read_table(path, columns=["short_book_title", "text"]).to_pydict()
        parts, offsets = [], [0]
        for text in t["text"]:
            ids = tok.encode(strip_specials(text)).ids
            parts.append(np.asarray(ids, dtype=np.uint32))
            offsets.append(offsets[-1] + len(ids))
        ids = np.concatenate(parts)
        assert ids.max() < 65536
        out = os.path.join(a.out, f"pg19_{split}.npz")
        np.savez(out, ids=ids.astype(np.uint16), offsets=np.asarray(offsets, dtype=np.int64),
                 titles=np.asarray(t["short_book_title"]))
        n = np.diff(offsets)
        print(f"{split}: {len(n)} books, {ids.size / 1e6:.1f}M tokens, median {int(np.median(n))}, "
              f">=64K {int((n >= 65536).sum())}, >=128K {int((n >= 131072).sum())} -> {out}", flush=True)
    print("PG19_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
