"""13-gram decontamination against code benchmarks (HumanEval, MBPP).

Same rule as FineMath / Qwen2.5-Math use for math: a document is dropped if it shares any 13-gram of words
with a benchmark problem. Words are runs of [a-z0-9_] after lowercasing, so punctuation and spacing do not
matter and identifiers, keywords and numbers all count. Each problem contributes its statement, its statement
followed by its reference solution, the solution alone and its test asserts, so a document that copies the
problem text, the solution or the tests is caught, also when they appear next to each other. MBPP statements
of 8-12 words (about a quarter of them) have no 13-gram of their own, so a document containing one of them
verbatim is dropped too (OpenCoder's algorithmic corpus restates MBPP problems in other languages).

    python data_prep/decontam.py            # stats about the benchmark n-grams (needs pyarrow)
"""
from __future__ import annotations

import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(os.path.dirname(HERE), "data", "raw", "bench")
N = 13
_WORD = re.compile(r"[a-z0-9_]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(ws: list[str], n: int = N):
    return zip(*(ws[i:] for i in range(n)))


def benchmark_texts(bench_dir: str = BENCH_DIR) -> dict[str, list[str]]:
    """Text units per benchmark (read from the parquet files that download_curl.py fetched)."""
    import pyarrow.parquet as pq
    out = {}
    he = []
    for p in sorted(glob.glob(os.path.join(bench_dir, "humaneval", "**", "*.parquet"), recursive=True)):
        for r in pq.read_table(p).to_pylist():
            he += [r["prompt"], r["prompt"] + r["canonical_solution"], r["canonical_solution"], r["test"]]
    out["humaneval"] = he
    mb = []
    for p in sorted(glob.glob(os.path.join(bench_dir, "mbpp", "**", "*.parquet"), recursive=True)):
        for r in pq.read_table(p).to_pylist():
            stmt = r.get("text") or r.get("prompt") or ""
            tests = "\n".join(r.get("test_list") or [])
            mb += [stmt, stmt + "\n" + r["code"], r["code"], tests, r["code"] + "\n" + tests]
    out["mbpp"] = mb
    return out


def benchmark_short_statements(bench_dir: str = BENCH_DIR, lo: int = 8) -> dict[int, frozenset]:
    """MBPP problem statements of lo..N-1 words, as word tuples grouped by length."""
    by_len: dict[int, set] = {}
    for unit in benchmark_texts(bench_dir)["mbpp"][0::5]:   # every 5th unit is a statement
        ws = tuple(words(unit))
        if lo <= len(ws) < N:
            by_len.setdefault(len(ws), set()).add(ws)
    return {k: frozenset(v) for k, v in by_len.items()}


def benchmark_ngrams(bench_dir: str = BENCH_DIR, n: int = N) -> frozenset:
    grams = set()
    for texts in benchmark_texts(bench_dir).values():
        for t in texts:
            grams.update(ngrams(words(t), n))
    if not grams:
        raise FileNotFoundError(f"no benchmark n-grams under {bench_dir}")
    return frozenset(grams)


def contaminated(text: str, grams: frozenset, short: dict | None = None, n: int = N) -> bool:
    ws = words(text)
    if len(ws) >= n and not grams.isdisjoint(ngrams(ws, n)):
        return True
    return bool(short) and any(len(ws) >= k and not s.isdisjoint(ngrams(ws, k)) for k, s in short.items())


if __name__ == "__main__":
    texts = benchmark_texts()
    for k, v in texts.items():
        print(k, len(v), "text units")
    print(len(benchmark_ngrams()), f"distinct {N}-grams;",
          sum(len(v) for v in benchmark_short_statements().values()), "short MBPP statements")
