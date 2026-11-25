"""Tokenize every downloaded file into data/bins/<source>/<stem>.bin (uint16, EOS after each doc).

Resumable: finished .bin files are skipped (written as .tmp, renamed when complete).

    python data_prep/tokenize_to_bins.py                 # all sources
    python data_prep/tokenize_to_bins.py --only math code_py --workers 5
    python data_prep/tokenize_to_bins.py --only pg19 --out data/bins_main --workers 1 --batch-docs 8 --read-rows 64
        # books (~400K characters each): small batches keep a worker near 1 GB (WSL memory while training)
    python data_prep/tokenize_to_bins.py --only edu_python opc_algo --out data/bins_main --decontam
        # --decontam drops documents sharing a 13-gram with HumanEval / MBPP or containing a short MBPP
        # statement verbatim (data_prep/decontam.py);
        # up to 30 dropped documents per file are kept in <out>/decontam/<source>/<file>.jsonl for review
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sources import PHASE2_EXTRA, with_phase  # noqa: E402
from textio import ROOT, iter_file_docs, source_files  # noqa: E402
from tok_lib import EOS_ID, strip_specials  # noqa: E402
from decontam import BENCH_DIR  # noqa: E402

_TOK = None
_GRAMS = None
_SHORT = None


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".parquet", ".json.gz", ".jsonl.gz", ".jsonl", ".json"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def split_at_words(text: str, n: int, search: int = 100_000) -> list:
    """Pieces of at most ~n characters, cut only inside "<letter> <letter>" (before the space). The pre-tokenizer starts
    a new piece at " word" anyway (tokenizer.json: Split regex "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+", NFC is local), so
    encoding the pieces one by one gives exactly the ids of the whole text (data_prep tests check it). Used for books of
    tens of millions of characters, whose single encode() call needs several GB. A stretch of `search` characters
    without such a spot is cut where it is (never seen in PG19)."""
    out, a = [], 0
    while len(text) - a > n:
        b = a + n
        lo = max(a + 1, b - search)
        while b > lo and not (text[b] == " " and text[b - 1].isalpha() and text[b + 1].isalpha()):
            b -= 1
        if b == lo:
            b = a + n
        out.append(text[a:b])
        a = b
    out.append(text[a:])
    return out


def _work(src, path: str, out_path: str, tok_path: str, threads: int, limit_docs: int = 0,
          min_cpt: float = 2.0, batch_docs: int = 512, decontam_dir: str = "", read_rows: int = 1024,
          stream: bool = False, max_encode_chars: int = 0):
    global _TOK, _GRAMS, _SHORT
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from tokenizers import Tokenizer
    if _TOK is None:
        _TOK = Tokenizer.from_file(tok_path)
    contaminated = None
    if decontam_dir:
        from decontam import benchmark_ngrams, benchmark_short_statements, contaminated
        if _GRAMS is None:
            _GRAMS = benchmark_ngrams(decontam_dir)
            _SHORT = benchmark_short_statements(decontam_dir)
    name = src.name
    tmp = out_path + ".tmp"
    n_tok = n_doc = n_skip = n_contam = 0
    dropped = []
    # digits are tokenized one by one, so number-heavy math pages legitimately sit at 1.5-2 chars/token
    thr = min(min_cpt, 1.3) if src.kind == "math" else min_cpt
    t0 = time.time()
    with open(tmp, "wb") as fh:
        batch = []

        def write(text, ids):
            nonlocal n_tok, n_doc, n_skip
            # base64 / hex / binary blobs compress to ~1.3-1.6 chars per token (prose and code: 3.5-4.5)
            if thr and len(ids) > 64 and len(text) / len(ids) < thr:
                n_skip += 1
                return
            ids.append(EOS_ID)
            arr = np.asarray(ids, dtype=np.uint32)
            assert arr.max() < 65536
            fh.write(arr.astype(np.uint16).tobytes())
            n_tok += len(ids)
            n_doc += 1

        def flush():
            for text, enc in zip(batch, _TOK.encode_batch(batch)):
                write(text, enc.ids)
            batch.clear()

        for i, doc in enumerate(iter_file_docs(path, src, batch_size=read_rows, stream=stream)):
            if limit_docs and i >= limit_docs:
                break
            doc = strip_specials(doc)
            if contaminated is not None and contaminated(doc, _GRAMS, _SHORT):
                n_contam += 1
                if len(dropped) < 30:
                    dropped.append(doc)
                continue
            if max_encode_chars and len(doc) > max_encode_chars:  # one book of up to ~28M characters: in pieces
                flush()
                ids = []
                for piece in split_at_words(doc, max_encode_chars):
                    ids += _TOK.encode(piece).ids
                write(doc, ids)
                continue
            batch.append(doc)
            if len(batch) >= batch_docs:
                flush()
        if batch:
            flush()
    if dropped:
        ddir = os.path.join(os.path.dirname(os.path.dirname(out_path)), "decontam", name)
        os.makedirs(ddir, exist_ok=True)
        with open(os.path.join(ddir, os.path.basename(out_path)[:-4] + ".jsonl"), "w", encoding="utf-8") as fh:
            for d in dropped:
                fh.write(json.dumps({"text": d}) + "\n")
    os.replace(tmp, out_path)
    return name, os.path.basename(out_path), n_tok, n_doc, time.time() - t0, n_skip, n_contam


def main():
    # User decision: heavy CPU work runs on the Windows box (WSL), not on the Mac.
    if sys.platform == "darwin" and os.path.exists(os.path.join(HERE, ".no_tokenize_on_mac")):
        print("tokenization disabled on the Mac (data_prep/.no_tokenize_on_mac); run it in WSL", flush=True)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer", "tokenizer.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "bins"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2, help="tokenizer threads per worker")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--extra-only", action="store_true", help="with --phase 2: only the PHASE2_EXTRA files")
    ap.add_argument("--limit-docs", type=int, default=0, help="testing: only the first N docs per file")
    ap.add_argument("--min-chars-per-token", type=float, default=2.0, help="drop blob-like docs (0 = keep all)")
    ap.add_argument("--decontam", action="store_true", help="drop docs sharing a 13-gram with HumanEval / MBPP")
    ap.add_argument("--batch-docs", type=int, default=512, help="documents per encode_batch call (memory: books ~8)")
    ap.add_argument("--read-rows", type=int, default=1024, help="rows per parquet read (memory: books ~64)")
    ap.add_argument("--stream-parquet", action="store_true", help="read parquet without pre-buffering (less memory)")
    ap.add_argument("--max-encode-chars", type=int, default=0,
                    help="encode longer documents in pieces cut between words (same ids, bounded memory; books ~1e6)")
    args = ap.parse_args()

    SOURCES = with_phase(args.phase)
    by_name = {s.name: s for s in SOURCES}
    tok_sha = hashlib.sha256(open(args.tokenizer, "rb").read()).hexdigest()
    jobs = []
    for src in SOURCES:
        if (args.only and src.name not in args.only) or (not args.only and not src.use):
            continue
        os.makedirs(os.path.join(args.out, src.name), exist_ok=True)
        files = source_files(src)
        if args.extra_only:
            extra = {os.path.basename(f) for f in PHASE2_EXTRA.get(src.name, ())}
            files = [f for f in files if os.path.basename(f) in extra]
        for p in files:
            out_path = os.path.join(args.out, src.name, _stem(p) + ".bin")
            if os.path.exists(out_path) and not args.force:
                continue
            jobs.append((src.name, p, out_path))
    print(f"{len(jobs)} files to tokenize with {args.workers} workers", flush=True)

    t0, total = time.time(), 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_work, by_name[n], p, o, args.tokenizer, args.threads, args.limit_docs, args.min_chars_per_token,
                          args.batch_docs, BENCH_DIR if args.decontam else "", args.read_rows, args.stream_parquet,
                          args.max_encode_chars)
                for n, p, o in jobs]
        for f in as_completed(futs):
            name, fname, n_tok, n_doc, dt, n_skip, n_contam = f.result()
            total += n_tok
            extra = f", {n_contam} contaminated docs dropped" if args.decontam else ""
            print(f"OK {name}/{fname}: {n_tok / 1e6:.1f}M tokens, {n_doc} docs ({n_skip} blob docs dropped{extra}), {dt:.0f}s "
                  f"(total {total / 1e9:.2f}B, {total / max(1e-9, time.time() - t0) / 1e6:.2f}M tok/s)", flush=True)

    # per-source metadata (recomputed from the files on disk)
    summary = {}
    for src in SOURCES:
        if args.only and src.name not in args.only:
            continue          # leave the metadata of other sources in this folder untouched
        d = os.path.join(args.out, src.name)
        if not os.path.isdir(d):
            continue
        files = {f: os.path.getsize(os.path.join(d, f)) // 2 for f in sorted(os.listdir(d)) if f.endswith(".bin")}
        if not files:
            continue
        meta = {"source": src.name, "kind": src.kind, "tokens": sum(files.values()), "files": files,
                "eos_id": EOS_ID, "dtype": "uint16", "tokenizer_sha256": tok_sha}
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        summary[src.name] = meta["tokens"]
    for k, v in summary.items():
        print(f"{k:10s} {v / 1e9:7.3f}B tokens")
    print(f"TOTAL {sum(summary.values()) / 1e9:.3f}B tokens", flush=True)


if __name__ == "__main__":
    main()
