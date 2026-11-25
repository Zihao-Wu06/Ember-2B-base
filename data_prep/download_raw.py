"""Download the raw parquet files listed in sources.py.

Needs only the stdlib + huggingface_hub, so it can run with an existing Python without
installing anything. Safe to re-run: finished files are skipped, partial ones resume.

    python data_prep/download_raw.py --low-priority
    python data_prep/download_raw.py --only en_web math
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sources import SOURCES, with_phase  # noqa: E402


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def set_low_priority():
    if os.name == "nt":
        import ctypes
        BELOW_NORMAL = 0x00004000
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL)
    else:
        os.nice(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--only", nargs="*", help="source names to fetch (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--phase", type=int, default=1, help="2 = also fetch the main-run files (PHASE2_EXTRA)")
    args = ap.parse_args()

    if args.low_priority:
        set_low_priority()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    # Everything stays on the project drive: force (not setdefault) every HF cache,
    # incl. the xet chunk cache, next to the data even if HF_HOME is set globally.
    hf_home = os.path.join(os.path.dirname(out), "hf_home")
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_XET_CACHE"] = os.path.join(hf_home, "xet")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:  # python-build-standalone (uv) has no system CA store on macOS
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from huggingface_hub import hf_hub_download

    jobs = [(s, f) for s in with_phase(args.phase) if (not args.only or s.name in args.only) and s.use
            for f in s.files]
    log(f"{len(jobs)} files -> {out} (HF_HOME={os.environ['HF_HOME']})")

    def fetch(src, fname):
        dest = os.path.join(out, src.name)
        for attempt in range(6):
            try:
                t0 = time.time()
                path = hf_hub_download(src.repo, fname, repo_type="dataset", local_dir=dest)
                return path, os.path.getsize(path), time.time() - t0
            except Exception as e:  # network hiccups: back off and retry, partial files resume
                wait = min(300, 10 * 2 ** attempt)
                log(f"RETRY {src.name}/{fname} attempt {attempt + 1}: {type(e).__name__}: {e} (sleep {wait}s)")
                time.sleep(wait)
        raise RuntimeError(f"giving up on {src.name}/{fname}")

    manifest, failed, total = [], [], 0
    t_start = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, s, f): (s, f) for s, f in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            s, f = futs[fut]
            try:
                path, size, dt = fut.result()
            except Exception as e:
                failed.append(f"{s.name}/{f}")
                log(f"FAIL {s.name}/{f}: {e}")
                continue
            total += size
            manifest.append({"source": s.name, "repo": s.repo, "file": f, "path": path, "bytes": size})
            log(f"OK {i}/{len(jobs)} {s.name}/{f} {size / 1e6:.0f} MB in {dt:.0f}s "
                f"(total {total / 1e9:.2f} GB, {total / 1e6 / max(1, time.time() - t_start):.1f} MB/s)")

    with open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump({"files": manifest, "failed": failed, "bytes": total}, fh, indent=1)
    log(f"ALL DONE: {len(manifest)} ok, {len(failed)} failed, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
