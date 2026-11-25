"""Dataset registry.

Every source is downloaded to data/raw/<name>/ and tokenized into its own folder
data/bins/<name>/. The pretraining mix is therefore only a set of sampling weights in
the training config: it can be changed (or ablated) without re-tokenizing anything.

Only stdlib here, so the download script can run on any Python that has huggingface_hub.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str          # local folder name
    repo: str          # HuggingFace dataset repo
    files: tuple       # files to fetch for the current phase (ablation + tokenizer)
    text_field: str = "text"
    kind: str = "en"   # en / zh / code / math -- used for the tokenizer sample mix and reports
    note: str = ""
    keep_langs: tuple = ()   # optional allowlist on a "language" column (github-code-clean)
    use: bool = True         # False = downloaded but not used (Chinese: goal is reasoning, not multilingual)
    min_score: int = 0       # keep only records with int_score >= this (Stack-Edu's educational score)
    max_non_latin: float = 0.0   # drop documents with more than this share of non-Latin-script characters (0 = off)


def _range(fmt, n, start=0):
    return tuple(fmt.format(i) for i in range(start, start + n))


# Phase 1 (now): tokenizer training + small-scale ablations, ~11 GB of parquet.
# Phase 2 (later, needs separate OK): ~25B tokens for the main run -- extend `files`.
SOURCES = (
    Source("en_web", "HuggingFaceFW/fineweb-edu",
           ("sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"),
           kind="en", note="FineWeb-Edu, ~0.7B tokens per file"),
    Source("zh_web", "opencsg/Fineweb-Edu-Chinese-V2.1",
           _range("4_5/{:06d}.parquet", 200),
           kind="zh", note="Chinese FineWeb-Edu 4-5 (unused: English-only model)", use=False),
    Source("code", "codeparrot/github-code-clean",
           _range("data/train-{:05d}-of-00880.parquet", 4),
           text_field="code", kind="code", note="GitHub files; markup (HTML/CSS) and PHP dropped",
           keep_langs=("Python", "C", "C++", "Java", "JavaScript", "TypeScript", "C#", "GO", "Rust",
                       "Shell", "SQL", "Scala", "Haskell", "Julia", "TeX", "Markdown", "Lua", "Ruby",
                       "FORTRAN", "Makefile", "CMake", "Dockerfile", "Perl", "Assembly")),
    Source("code_py", "codeparrot/codeparrot-clean",
           _range("file-{:012d}.json.gz", 3, start=1),
           text_field="content", kind="code", note="Python-only GitHub files (reasoning focus)"),
    Source("code_web", "OpenCoder-LLM/opc-fineweb-code-corpus",
           ("data/train-00000-of-00510.parquet",),
           kind="code", note="code-related web pages"),
    Source("math", "HuggingFaceTB/finemath",
           _range("finemath-4plus/train-{:05d}-of-00064.parquet", 6),
           kind="math", note="FineMath 4+"),
    Source("math_infi", "HuggingFaceTB/finemath",
           _range("infiwebmath-4plus/train-{:05d}-of-00032.parquet", 2),
           kind="math", note="InfiWebMath 4+"),
    Source("wiki_en", "wikimedia/wikipedia",
           _range("20231101.en/train-{:05d}-of-00041.parquet", 2),
           kind="en"),
    Source("wiki_zh", "wikimedia/wikipedia",
           _range("20231101.zh/train-{:05d}-of-00006.parquet", 2),
           kind="zh", use=False),
    Source("textbook", "HuggingFaceTB/smollm-corpus",
           ("cosmopedia-v2/train-00000-of-00104.parquet",),
           kind="en", note="Cosmopedia v2 synthetic textbooks"),
)

# Anneal-phase code data (2026-11-12, user approved): replaces code_web (prose, its code blocks were lost in
# FineWeb's text extraction) and github-code-clean in the LR-decay mix. Only the anneal mix uses them, so the
# stable phase of the main run is unchanged.
# - Stack-Edu, openly licensed release of the Common Pile: GitHub files with SmolLM2's educational-code score.
#   Shards are sorted by language (probed 11-11): C++ 0020-0027, Java 0038-0049, JavaScript 0050-0064,
#   Python 0073-0084, TypeScript 0091-0094. The release keeps int_score >= 2; we keep >= 3 like Stack-Edu.
# - OpenCoder's annealing corpus: algorithm-focused code from The Stack v2 and snippets rewritten from it
#   (8 languages, about 1/8 each). Its synthetic QA part (about 13% non-English) is not used.
# The model is English-only: documents whose comments are mostly in other scripts (3-4% of Stack-Edu, e.g.
# Chinese comments) are dropped with max_non_latin.
CODE_ANNEAL = (
    Source("edu_python", "common-pile/stackv2_edu_filtered",
           ("stack-edu-0073.json.gz", "stack-edu-0077.json.gz", "stack-edu-0081.json.gz"),
           kind="code", note="Stack-Edu (openly licensed) Python, int_score >= 3", min_score=3, keep_langs=("Python",),
           max_non_latin=0.02),
    Source("edu_code", "common-pile/stackv2_edu_filtered",
           ("stack-edu-0020.json.gz", "stack-edu-0038.json.gz", "stack-edu-0050.json.gz", "stack-edu-0091.json.gz"),
           kind="code", note="Stack-Edu (openly licensed) C++ / Java / JavaScript / TypeScript, int_score >= 3",
           min_score=3, keep_langs=("C++", "Java", "JavaScript", "TypeScript"), max_non_latin=0.02),
    Source("opc_algo", "OpenCoder-LLM/opc-annealing-corpus", _range("algorithmic_corpus/{}.jsonl", 54, start=1),
           kind="code", note="OpenCoder annealing corpus: algorithmic code", max_non_latin=0.02),
    Source("opc_snippet", "OpenCoder-LLM/opc-annealing-corpus", _range("synthetic_code_snippet/{}.jsonl", 31, start=1),
           kind="code", note="OpenCoder annealing corpus: synthetic code snippets", max_non_latin=0.02),
)
SOURCES = SOURCES + CODE_ANNEAL

# Stage-2 data (2026-11-12, user approved): more of the same math and code for the reasoning-heavy stable mix
# that the main run switches to at step ~12k (MIXES["reasoning_s2"] in scripts/train.py). They are separate
# sources on purpose: appending files to a source the run already trains on would move its held-out validation
# tail (the last tokens of the stream) and reshuffle its window order.
STAGE2 = (
    Source("math2", "HuggingFaceTB/finemath", _range("finemath-4plus/train-{:05d}-of-00064.parquet", 15, start=25),
           kind="math", note="FineMath 4+, shards 25-39 (0-24 are in math)"),
    Source("math_infi2", "HuggingFaceTB/finemath", _range("infiwebmath-4plus/train-{:05d}-of-00032.parquet", 6, start=6),
           kind="math", note="InfiWebMath 4+, shards 6-11 (0-5 are in math_infi)"),
    Source("edu_python2", "common-pile/stackv2_edu_filtered",
           tuple(f"stack-edu-{i:04d}.json.gz" for i in (74, 75, 78, 79, 82, 83)),
           kind="code", note="Stack-Edu (openly licensed) Python, more shards, int_score >= 3", min_score=3,
           keep_langs=("Python",), max_non_latin=0.02),
    Source("edu_code2", "common-pile/stackv2_edu_filtered",
           tuple(f"stack-edu-{i:04d}.json.gz" for i in (24, 44, 57)),
           kind="code", note="Stack-Edu (openly licensed) C++ / Java / JavaScript, more shards, int_score >= 3",
           min_score=3, keep_langs=("C++", "Java", "JavaScript", "TypeScript"), max_non_latin=0.02),
)
SOURCES = SOURCES + STAGE2

# Context extension (docs/PLAN_LONGCTX.md, user 2026-11-20 "全部下载 7 GB"): PG19, DeepMind's Project Gutenberg books
# published before 1919 (28,602 train / 50 validation / 100 test books, ~400K characters each), from the parquet mirror
# emozilla/pg19 (the deepmind/pg19 repo only has a loading script). pg19 (train) is a source of documents longer than
# 64K tokens for S3; pg19_eval (test + validation) is read by runs/mac_eval/long_ctx.py for 64K / 128K evaluation and
# never tokenized into training bins (use=False).
PG19_TRAIN = (
    "data/train-00000-of-00023-5263dc4323e881d2.parquet",
    "data/train-00001-of-00023-2b13c68dbeba63b5.parquet",
    "data/train-00002-of-00023-8e4dd6b70d51dfcf.parquet",
    "data/train-00003-of-00023-e7f0ba0d3c2eb6c0.parquet",
    "data/train-00004-of-00023-3f9e7e096fa07f4b.parquet",
    "data/train-00005-of-00023-87dece6a0ab708e8.parquet",
    "data/train-00006-of-00023-c50fea56d0518a87.parquet",
    "data/train-00007-of-00023-e9fea5c158e172e2.parquet",
    "data/train-00008-of-00023-275017526d706c9d.parquet",
    "data/train-00009-of-00023-f07a29af90b71085.parquet",
    "data/train-00010-of-00023-dd5d8583a9f6a549.parquet",
    "data/train-00011-of-00023-6c0bc91e6df2e661.parquet",
    "data/train-00012-of-00023-78d820574bddc6bb.parquet",
    "data/train-00013-of-00023-5b853732c713dded.parquet",
    "data/train-00014-of-00023-54b567998cd5eb4b.parquet",
    "data/train-00015-of-00023-7a78b1e66cfe52de.parquet",
    "data/train-00016-of-00023-8269d10108995a28.parquet",
    "data/train-00017-of-00023-e240b232c0daddca.parquet",
    "data/train-00018-of-00023-f59d477d80180c78.parquet",
    "data/train-00019-of-00023-198af082bd917d87.parquet",
    "data/train-00020-of-00023-f9b058c5e9338514.parquet",
    "data/train-00021-of-00023-e221306b7359a366.parquet",
    "data/train-00022-of-00023-5a956eb2a5d6cab5.parquet",
)
PG19_EVAL = (
    "data/test-00000-of-00001-29a571947c0b5ccc.parquet",
    "data/validation-00000-of-00001-0f92e2337f79aeac.parquet",
)
LONGCTX = (
    Source("pg19", "emozilla/pg19", PG19_TRAIN, kind="en", note="PG19 train: books before 1919, ~2.7B tokens"),
    Source("pg19_eval", "emozilla/pg19", PG19_EVAL, kind="en", note="PG19 test + validation (evaluation only)",
           use=False),
)
SOURCES = SOURCES + LONGCTX

# Benchmarks the anneal code data is decontaminated against (13-gram overlap), data/raw/bench/<name>/<file>.
BENCH = {
    "humaneval": ("openai/openai_humaneval", ("openai_humaneval/test-00000-of-00001.parquet",)),
    "mbpp": ("google-research-datasets/mbpp",
             tuple(f"{c}/{s}-00000-of-00001.parquet" for c in ("full", "sanitized")
                   for s in ("prompt", "test", "train", "validation"))),
}

BY_NAME = {s.name: s for s in SOURCES}

# Phase 2 (main run, ~25B tokens with the "reasoning" mix): extra files per source, ~67 GB parquet.
# Tokens per file measured with our tokenizer: fineweb-edu 0.74B, cosmopedia-v2 ~0.27B,
# finemath-4plus 0.15B, codeparrot-clean 0.24B, github-code-clean 0.19B, opc-fineweb-code 0.08B.
PHASE2_EXTRA = {
    "en_web": _range("sample/10BT/{:03d}_00000.parquet", 12, start=2),          # -> ~10.4B tokens
    "textbook": _range("cosmopedia-v2/train-{:05d}-of-00104.parquet", 11, start=1),  # -> ~3.2B
    "wiki_en": _range("20231101.en/train-{:05d}-of-00041.parquet", 10, start=2),    # -> ~1.5B
    "math": _range("finemath-4plus/train-{:05d}-of-00064.parquet", 19, start=6),    # -> ~3.8B
    "math_infi": _range("infiwebmath-4plus/train-{:05d}-of-00032.parquet", 4, start=2),
    "code_py": _range("file-{:012d}.json.gz", 8, start=4),                          # -> ~2.6B
    "code": _range("data/train-{:05d}-of-00880.parquet", 10, start=4),              # -> ~2.6B
    "code_web": _range("data/train-{:05d}-of-00510.parquet", 11, start=1),          # -> ~1.0B
}


def with_phase(phase: int):
    """Sources with the file lists of the given phase (2 = phase 1 + PHASE2_EXTRA)."""
    if phase < 2:
        return SOURCES
    from dataclasses import replace
    return tuple(replace(s, files=s.files + PHASE2_EXTRA.get(s.name, ())) for s in SOURCES)
ACTIVE = tuple(s for s in SOURCES if s.use)
