#!/usr/bin/env python
"""Compact copy of the training bins for a rented GPU (context extension S3 / S4, runs/longctx_s3/cloud/): per source
only what the given stages read, ~12 GB instead of the 73 GB of data/bins_main.

For every source the stages' mixtures use:
  <out>/<source>/000.bin       (if the plain source has weight) EOS-separated whole documents from blocks spread
                               evenly over the training region, enough for the plain windows the stages draw
                               (x --margin), then the source's held-out tail (its last val_tokens tokens) unchanged,
                               so the per-source validation loss stays comparable with the earlier runs.
  <out>/<source>.long/000.bin  (if "long:<source>" has weight) EOS, every document of more than half the shortest
                               long window (exact tokens, so "long:<source>" offers the same windows as before),
                               a pad of short documents (the last long document's windows must not run into the
                               held-out part), the source's tail as the held-out part; plus its long_docs.npz.
moe_llm.data.MixtureLoader reads "long:<source>" from <source>.long when that directory exists. Keeping the long
documents out of the plain sample keeps the plain windows' mix of short and long documents as in the full data.
Checks at the end: each long stream's window counts at the stages' lengths match the full data (+-2 for a document
cut by the held-out tail) and each plain stream's validation tail is bit-identical; writes manifest.json (tokens and
SHA-256 of every file). Reads the bins once, sequentially per block; run in WSL with --out on the ext4 disk.

    python data_prep/make_bundle.py --out /root/bundle/bins \\
        --stages longctx_s3:131072:1.5e9,longctx_s4:262144:1.5e9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from moe_llm.data import LONG, LONG_DIR, LONG_INDEX, TokenStream, long_window_starts  # noqa: E402

EOS = 0
COPY = 1 << 22  # tokens per read while copying


def find_eos(ts: TokenStream, pos: int, end: int) -> int:
    """Position of the first EOS in [pos, end), or -1."""
    while pos < end:
        part = ts.read(pos, min(1 << 20, end - pos))
        hit = np.flatnonzero(part == EOS)
        if len(hit):
            return pos + int(hit[0])
        pos += len(part)
    return -1


def plain_blocks(ts: TokenStream, train_end: int, need: int, n_blocks: int) -> list:
    """[start, end) intervals of whole documents (each ends with its EOS), about `need` tokens in n_blocks blocks
    spread evenly over [0, train_end); the whole region if it holds less than `need`."""
    if need >= train_end:
        last = train_end - 1
        while last >= 0 and int(ts.read(last, 1)[0]) != EOS:
            last -= 1
        return [(0, last + 1)]
    blen = max(1, need // n_blocks)
    out, prev = [], 0
    for i in range(n_blocks):
        p = max(prev, i * train_end // n_blocks)
        if p == 0:
            a = 0
        else:
            e = find_eos(ts, p - 1, train_end)  # a block starts right after an EOS
            if e < 0:
                break
            a = e + 1
        b = find_eos(ts, min(a + blen, train_end), train_end)
        if b < 0:
            break
        out.append((a, b + 1))
        prev = b + 1
    return out


def write_stream(path: str, ts: TokenStream, intervals: list, lead_eos: bool) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path + ".tmp", "wb") as f:
        if lead_eos:
            np.zeros(1, dtype=np.uint16).tofile(f)
            n += 1
        for a, b in intervals:
            for c in range(a, b, COPY):
                part = ts.read(c, min(COPY, b - c)).astype(np.uint16)
                part.tofile(f)
                n += len(part)
    os.replace(path + ".tmp", path)
    return n


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(stages: list, mixes: dict, margin: float) -> dict:
    """source -> {"plain": tokens to sample, "long_min": keep documents longer than this, "lens": long window
    lengths}, from stages [(mix name, seq_len, tokens)]."""
    out = {}
    for mix, seq_len, tokens in stages:
        for name, w in mixes[mix].items():
            if w <= 0:
                continue
            base = name[len(LONG):] if name.startswith(LONG) else name
            p = out.setdefault(base, {"plain": 0, "long_min": None, "lens": []})
            if name.startswith(LONG):
                p["long_min"] = min(p["long_min"] or seq_len // 2, seq_len // 2)
                p["lens"] = sorted(set(p["lens"] + [seq_len]))
            else:
                p["plain"] += int(w * tokens * margin)
    return out


def build(bins: str, out: str, stages: list, mixes: dict, margin: float = 1.3, val_tokens: int = 2_000_000,
          block_tokens: int = 2_000_000, log=print, only=(), exclude=(), manifest_name="manifest.json") -> dict:
    from long_docs import index_source
    manifest = {"stages": [list(s) for s in stages], "margin": margin, "val_tokens": val_tokens, "sources": {}}
    for base, p in sorted(plan(stages, mixes, margin).items()):
        if (only and base not in only) or base in exclude:
            continue
        t0 = time.time()
        ts = TokenStream(os.path.join(bins, base))
        vt = min(val_tokens, ts.total // 10)
        train_end = ts.total - vt
        tail = (train_end, ts.total)
        row = {}
        if p["plain"]:
            blocks = plain_blocks(ts, train_end, p["plain"], max(16, p["plain"] // block_tokens))
            path = os.path.join(out, base, "000.bin")
            n = write_stream(path, ts, blocks + [tail], lead_eos=False)
            assert min(val_tokens, n // 10) == vt, f"{base}: plain sample too small to keep the validation tail"
            new = TokenStream(os.path.join(out, base))
            assert np.array_equal(new.read(n - vt, vt), ts.read(train_end, vt)), f"{base}: validation tail differs"
            row["plain"] = {"tokens": n, "sampled": int(sum(b - a for a, b in blocks)), "blocks": len(blocks),
                            "sha256": sha256(path)}
        if p["long_min"]:
            z = np.load(os.path.join(bins, base, LONG_INDEX))
            assert int(z["total"]) == ts.total, f"{base}: stale {LONG_INDEX}"
            docs = z["docs"].astype(np.int64)
            inside = docs[docs[:, 0] + docs[:, 1] < train_end]  # the document and its EOS before the tail
            keep = inside[inside[:, 1] > p["long_min"]]
            short = inside[inside[:, 1] <= p["long_min"]]
            pad, got, need_pad = [], 0, max(p["lens"]) + 1
            for first, n_ in short.tolist():
                if got >= need_pad:
                    break
                pad.append((first, first + n_ + 1))
                got += n_ + 1
            if got < need_pad:  # not enough indexed short documents: plain blocks of whole documents
                for a, b in plain_blocks(ts, train_end, need_pad - got, 4):
                    pad.append((a, b))
            ldir = os.path.join(out, base + LONG_DIR)
            path = os.path.join(ldir, "000.bin")
            n = write_stream(path, ts, [(f, f + n_ + 1) for f, n_ in keep.tolist()] + pad + [tail], lead_eos=True)
            index_source(ldir, min_len=int(z["min_len"]), val_tokens=val_tokens)  # as the full data's index
            z2 = np.load(os.path.join(ldir, LONG_INDEX))
            vt2 = min(val_tokens, n // 10)
            counts = {}
            for L in p["lens"]:
                # every kept document gives all its windows (ceil((n - L/2) / L)): the pad and the tail follow it. The
                # full data can differ slightly: its first document has no EOS before it (no windows) and the last
                # windows before the held-out tail are cut.
                want = int(sum(-(-(n_ - L // 2) // L) for n_ in keep[:, 1].tolist() if n_ > L // 2))
                before = len(long_window_starts(docs, train_end, L))
                after = len(long_window_starts(z2["docs"].astype(np.int64), n - vt2, L))
                assert after == want and abs(after - before) <= max(2, before // 1000), \
                    f"{base}: {after} long windows at {L}, expected {want} (full data {before})"
                counts[str(L)] = [before, after]
            row["long"] = {"tokens": n, "docs": len(keep), "doc_tokens": int(keep[:, 1].sum()), "windows": counts,
                           "sha256": sha256(path), "index_sha256": sha256(os.path.join(ldir, LONG_INDEX))}
        manifest["sources"][base] = row
        log(f"  {base}: " + ", ".join(
            f"{k} {v['tokens'] / 1e6:.0f}M tokens" + (f" (windows {v['windows']})" if "windows" in v else "")
            for k, v in row.items()) + f" [{time.time() - t0:.0f}s]")
    manifest["tokens"] = sum(v["tokens"] for r in manifest["sources"].values() for v in r.values())
    with open(os.path.join(out, manifest_name), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", default=os.path.join(ROOT, "data", "bins_main"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="longctx_s3:131072:1.5e9,longctx_s4:262144:1.5e9",
                    help="mix:seq_len:tokens per stage (scripts/train.py MIXES)")
    ap.add_argument("--margin", type=float, default=1.3, help="plain sample = expected plain tokens x margin")
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--only", default="", help="comma-separated sources to build (default: all the stages use)")
    ap.add_argument("--exclude", default="", help="comma-separated sources to leave out")
    ap.add_argument("--manifest", default="manifest.json", help="file name of the manifest in --out")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train
    stages = [(m, int(L), int(float(t))) for m, L, t in (s.split(":") for s in a.stages.split(","))]
    t0 = time.time()
    split = lambda x: tuple(t for t in x.split(",") if t)  # noqa: E731
    m = build(a.bins, a.out, stages, train.MIXES, a.margin, a.val_tokens, log=lambda s: print(s, flush=True),
              only=split(a.only), exclude=split(a.exclude), manifest_name=a.manifest)
    print(f"{m['tokens'] / 1e9:.2f}B tokens ({2 * m['tokens'] / 1e9:.1f} GB) in {len(m['sources'])} sources -> "
          f"{a.out} [{time.time() - t0:.0f}s]", flush=True)
    print("BUNDLE_DONE", flush=True)


if __name__ == "__main__":
    main()
