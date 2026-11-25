"""Download files listed in sources.py with curl: stdlib only (no huggingface_hub needed), resumable.

Files land where the rest of data_prep expects them: data/raw/<source>/<file>, benchmarks in
data/raw/bench/<name>/<file>. Sizes are checked against the Hub's file listing. Safe to re-run.
--segments N splits each file into N byte ranges fetched over parallel connections (one connection is often
much slower than the line); a partly downloaded <file>.part keeps its bytes and only the rest is split.
--verify checks every file against the Hub's SHA-256 (LFS oid) and deletes a file that does not match.

    python3 data_prep/download_curl.py --only edu_python edu_code opc_algo opc_snippet --bench
    python3 data_prep/download_curl.py --only math2 edu_python2 --workers 6 --segments 4 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sources import BENCH, BY_NAME  # noqa: E402

RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
TREE = "https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def remote_files(repo: str) -> dict[str, dict]:
    """path -> {"size", "sha256"} from the Hub's listing (sha256 = the LFS oid; None for non-LFS files)."""
    with urllib.request.urlopen(TREE.format(repo=repo), timeout=120) as r:
        return {x["path"]: {"size": x.get("size", 0), "sha256": (x.get("lfs") or {}).get("oid")}
                for x in json.load(r) if x.get("type") == "file"}


def remote_sizes(repo: str) -> dict[str, int]:
    return {p: m["size"] for p, m in remote_files(repo).items()}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _get_range(url: str, path: str, a: int, b: int):
    """Bytes [a, b) of url into path; a partial segment file is resumed. No curl --retry here: a retried range
    would be written twice to the open file, so retries resume from the bytes on disk instead."""
    want = b - a
    for attempt in range(10):
        got = os.path.getsize(path) if os.path.exists(path) else 0
        if got == want:
            return
        if got > want:
            raise RuntimeError(f"{path}: {got} bytes for a {want}-byte range (range not honoured?)")
        with open(path, "ab") as fh:
            r = subprocess.run(["curl", "-sS", "-L", "--fail", "--connect-timeout", "30", "--speed-limit", "20000",
                                "--speed-time", "60", "-r", f"{a + got}-{b - 1}", url],
                               stdout=fh, stderr=subprocess.PIPE, text=True)
        got = os.path.getsize(path)
        if got == want:
            return
        if got > want:
            raise RuntimeError(f"{path}: {got} bytes for a {want}-byte range (range not honoured?)")
        wait = min(120, 10 * 2 ** attempt)
        log(f"RETRY range {a}-{b - 1} of {url.rsplit('/', 1)[-1]}: curl rc {r.returncode}, {got}/{want} bytes, "
            f"{r.stderr.strip()[:160]} (sleep {wait}s)")
        time.sleep(wait)
    raise RuntimeError(f"giving up on range {a}-{b - 1} of {url}")


def fetch_segmented(repo: str, fname: str, dest: str, size: int, segments: int, min_seg: int = 32 << 20):
    """Like fetch(), with the missing bytes split into up to `segments` ranges downloaded in parallel and then
    appended to <dest>.part in order. Falls back to fetch() for small remainders or if a range is not honoured."""
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest, size, 0.0, True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    rest = size - have
    n = max(1, min(segments, rest // min_seg))
    if n <= 1:
        return fetch(repo, fname, dest, size)
    t0 = time.time()
    url = RESOLVE.format(repo=repo, file=fname)
    bounds = [have + rest * i // n for i in range(n + 1)]
    segs = [(f"{part}.seg{i}", bounds[i], bounds[i + 1]) for i in range(n)]
    try:
        with ThreadPoolExecutor(n) as ex:
            for f in [ex.submit(_get_range, url, p, a, b) for p, a, b in segs]:
                f.result()
    except RuntimeError as e:
        if "not honoured" not in str(e):
            raise
        log(f"ranges not honoured for {repo}/{fname}: single connection")
        for p, _, _ in segs:
            if os.path.exists(p):
                os.remove(p)
        return fetch(repo, fname, dest, size)
    with open(part, "ab") as out:
        for p, _, _ in segs:
            with open(p, "rb") as fh:
                shutil.copyfileobj(fh, out, 8 << 20)
    if os.path.getsize(part) != size:
        raise RuntimeError(f"{repo}/{fname}: assembled {os.path.getsize(part)} of {size} bytes")
    for p, _, _ in segs:
        os.remove(p)
    os.replace(part, dest)
    return dest, size, time.time() - t0, False


def fetch(repo: str, fname: str, dest: str, size: int):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest, size, 0.0, True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    t0 = time.time()
    for attempt in range(8):
        if os.path.exists(part) and os.path.getsize(part) == size:   # finished before an interruption
            break
        r = subprocess.run(["curl", "-sS", "-L", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "10",
                            "--connect-timeout", "30", "-o", part, RESOLVE.format(repo=repo, file=fname)],
                           capture_output=True, text=True)
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if r.returncode == 0 and got == size:
            break
        wait = min(300, 15 * 2 ** attempt)
        log(f"RETRY {repo}/{fname}: curl rc {r.returncode}, {got}/{size} bytes, {r.stderr.strip()[:200]} (sleep {wait}s)")
        time.sleep(wait)
    else:
        raise RuntimeError(f"giving up on {repo}/{fname}")
    os.replace(part, dest)
    return dest, size, time.time() - t0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[], help="source names from sources.py")
    ap.add_argument("--bench", action="store_true", help="also fetch the decontamination benchmarks")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--segments", type=int, default=1, help="parallel byte ranges per file (1 = one connection)")
    ap.add_argument("--verify", action="store_true", help="check each file's SHA-256 against the Hub (LFS oid)")
    ap.add_argument("--manifest", default="MANIFEST_code.json")
    a = ap.parse_args()
    os.nice(10)
    raw = os.path.join(ROOT, "data", "raw")
    jobs = []   # (repo, file, dest)
    for name in a.only:
        s = BY_NAME[name]
        jobs += [(s.repo, f, os.path.join(raw, s.name, f)) for f in s.files]
    if a.bench:
        for name, (repo, files) in BENCH.items():
            jobs += [(repo, f, os.path.join(raw, "bench", name, f)) for f in files]
    meta = {}
    for repo in sorted({j[0] for j in jobs}):
        meta[repo] = remote_files(repo)
    missing = [f"{r}/{f}" for r, f, _ in jobs if f not in meta[r]]
    if missing:
        raise SystemExit(f"not on the Hub: {missing}")
    total = sum(meta[r][f]["size"] for r, f, _ in jobs)
    log(f"{len(jobs)} files, {total / 1e9:.2f} GB -> {raw} ({a.workers} files at a time, {a.segments} connection(s) "
        f"each{', SHA-256 checked' if a.verify else ''})")

    def job(r, f, d):
        size, sha = meta[r][f]["size"], meta[r][f]["sha256"]
        res = fetch_segmented(r, f, d, size, a.segments) if a.segments > 1 else fetch(r, f, d, size)
        if a.verify and sha:
            got = sha256_of(d)
            if got != sha:
                os.remove(d)
                raise RuntimeError(f"SHA-256 mismatch ({got[:12]} != {sha[:12]}): file deleted, re-run to fetch it again")
        return res, bool(a.verify and sha)

    manifest, failed, done, fresh = [], [], 0, 0
    t_start = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, r, f, d): (r, f) for r, f, d in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r, f = futs[fut]
            try:
                (path, size, dt, cached), checked = fut.result()
            except Exception as e:
                failed.append(f"{r}/{f}")
                log(f"FAIL {r}/{f}: {e}")
                continue
            done += size
            fresh += 0 if cached else size
            manifest.append({"repo": r, "file": f, "path": os.path.relpath(path, ROOT), "bytes": size})
            log(f"OK {i}/{len(jobs)} {r}/{f} {size / 1e6:.0f} MB {'(already here)' if cached else f'in {dt:.0f}s'}"
                f"{', SHA-256 ok' if checked else ''} (total {done / 1e9:.2f}/{total / 1e9:.2f} GB, this run "
                f"{fresh / 1e6 / max(1, time.time() - t_start):.1f} MB/s)")
    with open(os.path.join(raw, a.manifest), "w", encoding="utf-8") as fh:
        json.dump({"files": manifest, "failed": failed, "bytes": done}, fh, indent=1)
    log(f"ALL DONE: {len(manifest)} ok, {len(failed)} failed, {done / 1e9:.2f} GB")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
