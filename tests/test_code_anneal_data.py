"""Anneal-phase code data: jsonl / json.gz reading with Stack-Edu's score and language filters, 13-gram
decontamination against HumanEval / MBPP, and the tokenizer worker dropping contaminated documents.
The benchmark problems here are made up (same columns as the real parquet files)."""
import gzip
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import decontam
from sources import Source
from textio import ROOT, iter_file_docs

STMT = "Write a python function to count how many widgets in a crate are painted a blue colour shade."
CODE = ("def count_blue_widgets(crate):\n    total = 0\n    for widget in crate:\n"
        "        if widget.colour == 'blue':\n            total += 1\n    return total\n")
TESTS = ["assert count_blue_widgets([W('blue'), W('red')]) == 1",
         "assert count_blue_widgets([]) == 0",
         "assert count_blue_widgets([W('blue'), W('blue'), W('green')]) == 2"]
HE_PROMPT = ('def frobnicate_widgets(widgets: list, factor: int) -> list:\n'
             '    """ Return every widget scaled by factor, dropping widgets whose weight is zero.\n'
             '    >>> frobnicate_widgets([1, 0, 2], 3)\n    [3, 6]\n    """\n')
HE_SOL = "    return [w * factor for w in widgets if w != 0]\n"


SHORT_STMT = "Write a function to find the volume of a hexagonal prism."   # 11 words: no 13-gram


def make_bench(root):
    he_dir = os.path.join(root, "humaneval", "openai_humaneval")
    mb_dir = os.path.join(root, "mbpp", "full")
    os.makedirs(he_dir)
    os.makedirs(mb_dir)
    pq.write_table(pa.table({"task_id": ["X/0"], "prompt": [HE_PROMPT], "canonical_solution": [HE_SOL],
                             "test": ["def check(candidate):\n    assert candidate([1, 0, 2], 3) == [3, 6]\n"],
                             "entry_point": ["frobnicate_widgets"]}),
                   os.path.join(he_dir, "test-00000-of-00001.parquet"))
    pq.write_table(pa.table({"task_id": [1, 2], "text": [STMT, SHORT_STMT],
                             "code": [CODE, "def prism_volume(a, h):\n    return 2.598 * a * a * h\n"],
                             "test_list": [TESTS, ["assert prism_volume(1, 1) == 2.598"]],
                             "test_setup_code": ["", ""], "challenge_test_list": [[], []]}),
                   os.path.join(mb_dir, "test-00000-of-00001.parquet"))


def test_jsonl_and_json_gz_with_score_and_language_filters(tmp_path):
    body = "def f(x):\n    return x * 2  # doubles the input value\n"
    recs = [{"text": body + "# keep", "int_score": 3, "metadata": {"language": "Python"}},
            {"text": body + "# low score", "int_score": 2, "metadata": {"language": "Python"}},
            {"text": body + "# other language", "int_score": 4, "metadata": {"language": "Go"}},
            {"text": "short", "int_score": 5, "metadata": {"language": "Python"}}]
    src = Source("t", "repo", (), kind="code", min_score=3, keep_langs=("Python",))
    plain = tmp_path / "a.jsonl"
    plain.write_text("".join(json.dumps(r) + "\n" for r in recs) + "\n")
    gz = tmp_path / "b.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in recs)
    assert list(iter_file_docs(str(plain), src)) == [recs[0]["text"]]
    assert list(iter_file_docs(str(gz), src)) == [recs[0]["text"]]
    # OpenCoder-style records: no score, "lang" column, no filters -> everything long enough is kept
    opc = Source("o", "repo", (), kind="code")
    assert len(list(iter_file_docs(str(plain), opc))) == 3


def test_decontam_catches_copies_not_generic_code(tmp_path):
    make_bench(str(tmp_path))
    grams = decontam.benchmark_ngrams(str(tmp_path))
    hit = lambda t: decontam.contaminated(t, grams)   # noqa: E731
    assert hit("Exercise 12.\n" + STMT + "\n\nSolution:\n" + CODE)          # problem + solution
    assert hit("# " + STMT.upper())                                          # statement alone, case changed
    assert hit(CODE.replace("    ", "\t").replace(" == ", "=="))             # solution, reformatted
    assert hit("\n".join(TESTS))                                             # the tests
    assert hit("Some notes.\n" + HE_PROMPT + HE_SOL)                         # HumanEval prompt + solution
    assert not hit("def count_red_balls(box):\n    return sum(1 for b in box if b == 'red')\n" * 3)
    assert not hit("Our widgets come in a blue colour shade and we count them by hand every week.")
    assert not hit("def frobnicate_widgets(widgets, factor):\n    pass\n")    # same name, not the problem


def test_tokenize_worker_drops_contaminated(tmp_path):
    from tokenize_to_bins import _work
    from tok_lib import EOS_ID
    bench = tmp_path / "bench"
    make_bench(str(bench))
    clean = ["def add(a, b):\n    # add two numbers and return the sum\n    return a + b\n",
             "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n"]
    raw = tmp_path / "1.jsonl"
    raw.write_text("".join(json.dumps({"text": t, "lang": "python"}) + "\n" for t in [clean[0], STMT + "\n" + CODE, clean[1]]))
    out = tmp_path / "bins" / "opc_t" / "1.bin"
    out.parent.mkdir(parents=True)
    src = Source("opc_t", "repo", (), kind="code")
    tok = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")
    name, fname, n_tok, n_doc, dt, n_skip, n_contam = _work(src, str(raw), str(out), tok, threads=1, min_cpt=0,
                                                            decontam_dir=str(bench))
    assert (n_doc, n_contam) == (2, 1)
    ids = np.fromfile(out, dtype=np.uint16)
    assert int((ids == EOS_ID).sum()) == 2 and len(ids) == n_tok
    kept = [json.loads(line)["text"] for line in open(tmp_path / "bins" / "decontam" / "opc_t" / "1.jsonl")]
    assert kept == [STMT + "\n" + CODE]


def test_non_latin_filter(tmp_path):
    code = "def area(r):\n    return 3.14159 * r * r\n" * 5
    recs = [{"text": "# 计算圆的面积，返回浮点数结果\n" + code},          # ~10% Chinese: dropped
            {"text": "# café, naïve, π ≈ 3.14 → résumé\n" + code},      # accents, Greek pi, arrows: kept
            {"text": "# 面积\n" + code * 4}]                            # a few Chinese characters in a long file: kept
    p = tmp_path / "x.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    src = Source("t", "repo", (), kind="code", max_non_latin=0.02)
    assert list(iter_file_docs(str(p), src)) == [recs[1]["text"], recs[2]["text"]]
    assert len(list(iter_file_docs(str(p), Source("t", "repo", (), kind="code")))) == 3


def test_decontam_short_statement_verbatim(tmp_path):
    make_bench(str(tmp_path))
    grams = decontam.benchmark_ngrams(str(tmp_path))
    short = decontam.benchmark_short_statements(str(tmp_path))
    assert {k: len(v) for k, v in short.items()} == {11: 1}
    restated = SHORT_STMT.replace("function", "Function") + "\nfunction prismVolume(a, h) { return 2.598*a*a*h; }"
    assert not decontam.contaminated(restated, grams)          # the 13-gram rule alone misses it
    assert decontam.contaminated(restated, grams, short)       # the statement rule catches it
    assert not decontam.contaminated("Write a function to find the volume of a cube.", grams, short)
