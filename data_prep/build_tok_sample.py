"""Build the tokenizer-training sample (reasoning-focused mix) + a held-out set.

    python data_prep/build_tok_sample.py --total-bytes 1e9
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sources import BY_NAME  # noqa: E402
from textio import ROOT, iter_source_docs, source_files  # noqa: E402

# Goal = reasoning, English only (user decision): English + code + math.
KIND_SHARE = {"en": 0.55, "code": 0.25, "math": 0.20}
SOURCE_SHARE = {
    "en": {"en_web": 0.7, "textbook": 0.2, "wiki_en": 0.1},
    "code": {"code_py": 0.5, "code": 0.35, "code_web": 0.15},
    "math": {"math": 0.7, "math_infi": 0.3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-bytes", type=float, default=1e9)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "tokenizer", "sample"))
    ap.add_argument("--heldout-docs", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    targets = {}
    for kind, kshare in KIND_SHARE.items():
        avail = {n: s for n, s in SOURCE_SHARE[kind].items() if source_files(BY_NAME[n])}
        missing = set(SOURCE_SHARE[kind]) - set(avail)
        if missing:
            print(f"WARNING: no files yet for {sorted(missing)}; redistributing within '{kind}'", flush=True)
        z = sum(avail.values())
        for n, s in avail.items():
            targets[n] = args.total_bytes * kshare * s / z

    stats, held = {}, {"text": [], "source": []}
    for name, target in targets.items():
        t0, got, docs, buf = time.time(), 0, 0, []
        it = iter_source_docs(BY_NAME[name], max_chars=100_000)
        for doc in it:
            buf.append(doc)
            got += len(doc.encode("utf-8"))
            docs += 1
            if got >= target:
                break
        for _, doc in zip(range(args.heldout_docs), it):
            held["text"].append(doc)
            held["source"].append(name)
        pq.write_table(pa.table({"text": buf}), os.path.join(args.out, f"{name}.parquet"), compression="zstd")
        stats[name] = {"target_bytes": int(target), "bytes": got, "docs": docs}
        flag = "" if got >= target * 0.98 else "  (SHORT: source exhausted)"
        print(f"{name:10s} {got / 1e6:8.1f} MB / {target / 1e6:8.1f} MB  {docs:8d} docs  "
              f"{time.time() - t0:5.0f}s{flag}", flush=True)

    pq.write_table(pa.table(held), os.path.join(os.path.dirname(args.out), "heldout.parquet"), compression="zstd")
    with open(os.path.join(args.out, "sample_stats.json"), "w") as f:
        json.dump({"kind_share": KIND_SHARE, "source_share": SOURCE_SHARE, "sources": stats}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
