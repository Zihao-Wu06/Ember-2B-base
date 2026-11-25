"""Engram tokenizer-compression map: raw id -> canonical id (data/tokenizer/canon_map.npy).

    python data_prep/make_canon_map.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from textio import ROOT  # noqa: E402
from tok_lib import build_canon_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer", "tokenizer.json"))
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    canon, meta = build_canon_map(tok)
    out = os.path.dirname(args.tokenizer)
    np.save(os.path.join(out, "canon_map.npy"), canon)
    with open(os.path.join(out, "canon_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"vocab {meta['vocab_size']} -> {meta['num_canonical']} canonical ids "
          f"({meta['reduction']:.1%} reduction)")
    for g in meta["largest_groups"][:5]:
        print(g["size"], [repr(x) for x in g["examples"]])


if __name__ == "__main__":
    main()
