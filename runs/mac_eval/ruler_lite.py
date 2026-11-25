#!/usr/bin/env python
"""A small RULER-style long-context test (Hsieh et al. 2024) for base models: synthetic needles in natural text.

Haystack: held-out English (the last 2M tokens of wiki_en and en_web, which training never reads), decoded and split
into sentences. Tasks, with RULER's base-model templates:
  niah_single    one "special magic number" for a key word; asked for it
  niah_multikey  four needles with different key words; asked for one of them
  niah_multivalue four numbers for the same key word; asked for all (score = share found)
  vt             variable tracking: VAR A = 12345, then VAR B = VAR A ... (4 hops) spread over the text; asked which
                 variables hold 12345 (score = share found)
Needles go at random depths between sentences; the prompt is the longest that fits --lengths tokens. Greedy decoding
through decode.py (uses the model's attn_logn if its config has one; --logn forces the scaling on other models).

    .venv-mac/bin/python runs/mac_eval/ruler_lite.py --export runs/mac_eval/ckpt_0047684 --lengths 4096,8192,16384,32768
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mac_lm  # noqa: E402
import decode  # noqa: E402
from moe_llm.data import TokenStream  # noqa: E402

HEAD = ("Some special magic {kind} are hidden within the following text. Make sure to memorize it. I will quiz you "
        "about the {kind} afterwards.\n")
NEEDLE = "One of the special magic numbers for {key} is: {val}."
Q_ONE = ("\nWhat is the special magic number for {key} mentioned in the provided text? The special magic number for "
         "{key} mentioned in the provided text is")
Q_ALL = ("\nWhat are all the special magic numbers for {key} mentioned in the provided text? The special magic "
         "numbers for {key} mentioned in the provided text are")
VT_HEAD = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n")
VT_Q = ("\nQuestion: Find all variables that are assigned the value {val} in the text above. Answer: According to the "
        "chain(s) of variable assignment in the text above, {n} variables are assigned the value {val}, they are:")
WORDS = ("apple river candle orbit maple violin harbor lantern meadow pepper falcon marble tunnel cactus velvet anchor "
         "breeze cobalt dagger ember glacier hazel ivory jungle kettle lemon mirror nectar oyster pebble quartz saddle "
         "thunder umbrella walnut zephyr").split()


def haystack_sentences(tok, n_docs=400):
    root = os.path.join(mac_lm.ROOT, "data", "bins_main")
    sents = []
    for src in ("wiki_en", "en_web"):
        ts = TokenStream(os.path.join(root, src))
        vt = min(2_000_000, ts.total // 10)
        arr = ts.read(ts.total - vt, vt)
        eos = np.flatnonzero(arr == mac_lm.EOS)
        for a, b in list(zip(eos[:-1], eos[1:]))[:n_docs]:
            text = tok.decode(arr[a + 1:b].tolist())
            sents += [s.strip() + "." for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(s) > 20]
    return sents


def build(tok, sents, n_tokens, head, needles, question, rng, lazy=False):
    """Longest prompt <= n_tokens: head + sentences (needles inserted at random depths) + question. lazy: draw the
    random choices only and return them; build_spec() does the (slow, tokenizer-bound) search later."""
    start = rng.randrange(len(sents) // 2)
    depths = sorted(rng.random() for _ in needles)
    spec = (n_tokens, head, needles, question, start, depths)
    return spec if lazy else build_spec(tok, sents, spec)


def build_spec(tok, sents, spec):
    n_tokens, head, needles, question, start, depths = spec

    def make(k):
        body = sents[start:start + k] if start + k <= len(sents) else (sents * 2)[start:start + k]
        body = list(body)
        for d, nd in sorted(zip(depths, needles), reverse=True):
            body.insert(int(d * len(body)), nd)
        return mac_lm.encode(tok, head + " ".join(body) + question)

    lo, hi = 1, 4
    while len(make(hi)) < n_tokens:
        lo, hi = hi, hi * 2
    while hi - lo > 1:  # binary search on the number of sentences
        mid = (lo + hi) // 2
        if len(make(mid)) <= n_tokens:
            lo = mid
        else:
            hi = mid
    return make(lo)


def sample(task, tok, sents, n_tokens, rng, lazy=False):
    num = lambda: str(rng.randint(1_000_000, 9_999_999))  # noqa: E731
    if task == "niah_single":
        key, val = rng.choice(WORDS), num()
        ids = build(tok, sents, n_tokens, HEAD.format(kind="numbers"), [NEEDLE.format(key=key, val=val)],
                    Q_ONE.format(key=key), rng, lazy)
        return ids, [val], 16
    if task == "niah_multikey":
        keys = rng.sample(WORDS, 4)
        vals = [num() for _ in keys]
        i = rng.randrange(4)
        ids = build(tok, sents, n_tokens, HEAD.format(kind="numbers"),
                    [NEEDLE.format(key=k, val=v) for k, v in zip(keys, vals)], Q_ONE.format(key=keys[i]), rng, lazy)
        return ids, [vals[i]], 16
    if task == "niah_multivalue":
        key = rng.choice(WORDS)
        vals = [num() for _ in range(4)]
        ids = build(tok, sents, n_tokens, HEAD.format(kind="numbers"), [NEEDLE.format(key=key, val=v) for v in vals],
                    Q_ALL.format(key=key), rng, lazy)
        return ids, vals, 48
    if task == "vt":
        names = ["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(5)]
        val = str(rng.randint(10000, 99999))
        chain = [f"VAR {names[0]} = {val}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, 5)]
        ids = build(tok, sents, n_tokens, VT_HEAD, chain, VT_Q.format(val=val, n=5), rng, lazy)
        return ids, names, 40
    raise ValueError(task)


@torch.inference_mode()
def answer(model, tok, ids, max_new):
    cache = decode.prefill(model, ids[:-1])
    return tok.decode(decode.generate_ids(model, ids[-1:], max_new, init=cache))


@torch.inference_mode()
def answer_batch(model, tok, items):
    """answer() for several (ids, want, max_new) samples decoded together. A row stops once every wanted string is in
    its output: the score (share of wanted strings found) cannot change after that, only the stored text is shorter."""
    caches = [decode.prefill(model, ids[:-1]) for ids, _, _ in items]
    wants = [w for _, w, _ in items]
    outs = decode.generate_batch(model, caches, [ids[-1] for ids, _, _ in items], max(m for _, _, m in items),
                                 done=lambda b, o: all(w in tok.decode(o) for w in wants[b]))
    return [tok.decode(o) for o in outs]


_TOK = _SENTS = None  # set before the worker pool forks


def _realize(spec):
    return build_spec(_TOK, _SENTS, spec)


def main():
    global _TOK, _SENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--tasks", default="niah_single,niah_multikey,niah_multivalue,vt")
    ap.add_argument("--n", type=int, default=5, help="samples per task and length")
    ap.add_argument("--batch", type=int, default=1,
                    help="samples decoded together at 32K, more for shorter prompts (decode.batch_rows); 10 on the GPU")
    ap.add_argument("--workers", type=int, default=8, help="processes building prompts ahead of the model (0: inline)")
    ap.add_argument("--logn", type=int, default=0, help="force log-n scaling with this base on a model without it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # every random choice drawn up front, in the order the one-at-a-time loop drew them (the rng is only used in
    # sample()); the tokenizer-bound prompt search runs in worker processes, forked before CUDA starts
    from tokenizers import Tokenizer
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _TOK = Tokenizer.from_file(os.path.join(mac_lm.ROOT, "data", "tokenizer", "tokenizer.json"))
    _SENTS = haystack_sentences(_TOK)
    rng = random.Random(0)
    lengths, tasks = [int(x) for x in args.lengths.split(",")], args.tasks.split(",")
    cells = [(n_tok, task, [sample(task, _TOK, _SENTS, n_tok, rng, lazy=True) for _ in range(args.n)])
             for n_tok in lengths for task in tasks]
    specs = [s for _, _, items in cells for s, _, _ in items]
    pool = None
    if args.workers:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(args.workers)
        built = pool.imap(_realize, specs, chunksize=2)  # in order, running ahead of the GPU
    else:
        built = map(_realize, specs)
    model, tok, meta = mac_lm.load(args.export, args.device)
    if args.logn:
        for m in model.modules():
            if hasattr(m, "logn"):
                m.logn = args.logn
    res = {"step": meta["step"], "logn": max(args.logn, int(meta["cfg"].get("attn_logn", 0))), "scores": {},
           "samples": []}
    for ci, (n_tok, task, lazy_items) in enumerate(cells):
        t0, scores = time.time(), []
        items = [(next(built), want, max_new) for _, want, max_new in lazy_items]
        rows = decode.batch_rows(n_tok, args.batch) if args.batch > 1 else 1
        for a in range(0, len(items), rows):
            part = items[a:a + rows]
            outs = ([answer(model, tok, part[0][0], part[0][2])] if rows == 1
                    else answer_batch(model, tok, part))
            for (ids, want, _), out in zip(part, outs):
                got = sum(w in out for w in want) / len(want)
                scores.append(got)
                res["samples"].append({"length": n_tok, "tokens": len(ids), "task": task, "want": want,
                                       "output": out, "score": got})
        res["scores"][f"{task}@{n_tok}"] = round(100 * sum(scores) / len(scores), 1)
        print(f"  {task} {n_tok}: {res['scores'][f'{task}@{n_tok}']:.1f} [{time.time() - t0:.0f}s]", flush=True)
        if ci + 1 == len(cells) or cells[ci + 1][0] != n_tok:
            avg = [v for k, v in res["scores"].items() if k.endswith(f"@{n_tok}")]
            print(f"ruler-lite {n_tok}: average {sum(avg) / len(avg):.1f}", flush=True)
    if pool is not None:
        pool.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("RULER_LITE_DONE", flush=True)


if __name__ == "__main__":
    main()
