"""Train the 65536-token byte-level BPE on the sample and report compression on held-out docs.

    python data_prep/train_tokenizer.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
import unicodedata

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sources import BY_NAME  # noqa: E402
from textio import ROOT  # noqa: E402
from tok_lib import VOCAB_SIZE, check_tokenizer, new_tokenizer, new_trainer  # noqa: E402


def iter_sample(paths):
    for p in paths:
        for b in pq.ParquetFile(p).iter_batches(columns=["text"], batch_size=2048):
            yield from b.column("text").to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.path.join(ROOT, "data", "tokenizer", "sample"))
    ap.add_argument("--heldout", default=os.path.join(ROOT, "data", "tokenizer", "heldout.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "tokenizer"))
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.sample, "*.parquet")))
    n_docs = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    print(f"training on {len(paths)} files, {n_docs} docs, vocab {args.vocab_size}", flush=True)
    t0 = time.time()
    tok = new_tokenizer()
    tok.train_from_iterator(iter_sample(paths), new_trainer(args.vocab_size), length=n_docs)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30 if sys.platform == "darwin" else 1 << 20)
    print(f"trained in {time.time() - t0:.0f}s, peak RSS {rss:.1f} GB", flush=True)
    check_tokenizer(tok, args.vocab_size)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "tokenizer.json")
    tok.save(out_path)

    # held-out compression + round trip
    held = pq.read_table(args.heldout).to_pydict()
    per = {}
    bad = 0
    for text, src in zip(held["text"], held["source"]):
        ids = tok.encode(text).ids
        if tok.decode(ids) != unicodedata.normalize("NFC", text):
            bad += 1
        d = per.setdefault(src, {"bytes": 0, "chars": 0, "tokens": 0, "docs": 0})
        d["bytes"] += len(text.encode("utf-8"))
        d["chars"] += len(text)
        d["tokens"] += len(ids)
        d["docs"] += 1
    lines = ["# Tokenizer report", "", f"vocab {tok.get_vocab_size()}, round-trip failures {bad}/{len(held['text'])}",
             "", "| source | kind | bytes/token | chars/token |", "|---|---|---|---|"]
    for src, d in per.items():
        lines.append(f"| {src} | {BY_NAME[src].kind} | {d['bytes'] / d['tokens']:.2f} | {d['chars'] / d['tokens']:.2f} |")
    report = "\n".join(lines)
    print(report, flush=True)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"per_source": per, "roundtrip_failures": bad, "train_seconds": time.time() - t0}, f, indent=1)


if __name__ == "__main__":
    main()
