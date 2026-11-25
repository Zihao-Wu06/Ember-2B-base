#!/usr/bin/env python
"""Does the model use context beyond its 4096-token training length? Two tests on an exported checkpoint (Mac CPU).

1. Loss by position on long held-out documents. Documents of >= 16K tokens come from the held-out tail of every
   source (the last min(2M, total/10) tokens of each stream, which training never reads; the partial documents at
   both ends of the tail are skipped). Each document (EOS + its first L tokens, L = 32768, or 16384 for documents
   shorter than 32K) is scored twice:
     full    one forward pass over all L tokens
     window  sliding 4096-token windows with stride 2048: every token is scored with 2048-4095 tokens of context
             (the first window scores all of its tokens), i.e. what the model can do within its training length
   Below position 4096 both see the same context (a check). Beyond it, full < window means the extra context helps;
   full > window means the model breaks past its training length. Reported per position bucket and per category
   (code / math / English) with per-document paired differences.
2. Passkey retrieval (Mohtashami & Jaggi 2023): a 5-digit key hidden at depth 10/30/50/70/90% of repeated filler
   text 2K-32K tokens long, greedy decoding after "What is the pass key? The pass key is".

--logn: a training-free length extension to try if full > window: the 6 NoPE full-attention layers scale the
attention logits of a query that sees n keys by max(1, ln n / ln 4096), so attention does not spread out over more
keys than in training. With --baseline (the JSON of a plain run) the window pass is reused, not recomputed.

    .venv-mac/bin/python runs/mac_eval/long_ctx.py --export runs/mac_eval/ckpt_0045000 \\
        --out runs/mac_eval/longctx_0045000.json"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mac_lm  # noqa: E402
import decode  # noqa: E402
import moe_llm.layers.attention as attn_mod  # noqa: E402
from moe_llm.data import TokenStream  # noqa: E402

TRAIN_LEN = 4096
EOS = mac_lm.EOS
CATS = {"code": ("code", "code_py", "edu_code", "edu_code2", "edu_python", "edu_python2"),
        "math": ("math", "math2", "math_infi", "math_infi2"),
        "english": ("en_web", "wiki_en", "textbook")}
BUCKETS = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536),
           (65536, 131072), (131072, 262144), (262144, 524288))
LENS = (16384, 32768, 65536, 131072, 262144, 524288)  # a document is scored over the longest of these that fits
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
HEAD = ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. "
        "I will quiz you about the important information there.\n\n")
KEY = "The pass key is {k}. Remember it. {k} is the pass key. "
TAIL = "\n\nWhat is the pass key? The pass key is"


class _LogNF:
    """torch.nn.functional with scaled_dot_product_attention scaling each query row by max(1, ln n / ln 4096),
    n = the number of keys it sees (causal)."""

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, **kw):
        tq, tk = q.shape[-2], k.shape[-2]
        n = torch.arange(tk - tq + 1, tk + 1, dtype=torch.float32, device=q.device)
        s = torch.clamp(torch.log(n) / math.log(TRAIN_LEN), min=1.0).to(q.dtype)
        return F.scaled_dot_product_attention(q * s.view(1, 1, -1, 1), k, v, attn_mask=attn_mask,
                                              is_causal=is_causal, **kw)


def heldout_docs(min_len=16384, val_tokens=2_000_000):
    root = os.path.join(mac_lm.ROOT, "data", "bins_main")
    out = []
    for cat, srcs in CATS.items():
        for s in srcs:
            ts = TokenStream(os.path.join(root, s))
            vt = min(val_tokens, ts.total // 10)
            base = ts.total - vt
            arr = ts.read(base, vt)
            eos = np.flatnonzero(arr == EOS)
            for a, b in zip(eos[:-1], eos[1:]):
                if b - a - 1 >= min_len:
                    out.append({"cat": cat, "source": s, "offset": int(base + a + 1), "len": int(b - a - 1),
                                "ids": arr[a + 1:b]})
    return out


def pg19_books(n, max_len, split="test", min_len=65536, windows=False):
    """n PG19 books (data/eval/pg19_<split>.npz, data_prep/pg19_eval.py; never in the training bins) of at least
    min_len tokens, a fixed random choice, each scored like pick_docs() documents (category "books")."""
    if n <= 0:
        return []
    z = np.load(os.path.join(mac_lm.ROOT, "data", "eval", f"pg19_{split}.npz"))
    ids, off = z["ids"], z["offsets"]
    idx = [i for i in range(len(off) - 1) if off[i + 1] - off[i] >= min_len]
    random.Random(2).shuffle(idx)
    out = []
    if windows:  # every book of >= min_len, cut into back-to-back windows of max_len (the tail is dropped)
        for i in sorted(idx):
            for w in range(int(off[i + 1] - off[i]) // max_len):
                a = int(off[i]) + w * max_len
                out.append({"id": f"pg19_{split}@{i}w{w}", "cat": "books", "source": f"pg19_{split}", "len": max_len,
                            "offset": i, "ids": ids[a:a + max_len], "L": max_len})
        return out
    for i in sorted(idx[:n]):
        n_tok = int(off[i + 1] - off[i])
        out.append({"id": f"pg19_{split}@{i}", "cat": "books", "source": f"pg19_{split}", "len": n_tok, "offset": i,
                    "ids": ids[off[i]:off[i + 1]],
                    "L": max(m for m in LENS if m <= max(16384, min(n_tok, max_len)))})
    return out


def pick_docs(docs, per_cat, max_len=32768):
    chosen = []
    for cat in CATS:
        c = [d for d in docs if d["cat"] == cat]
        long_ = [d for d in c if d["len"] >= 32768]
        short = [d for d in c if d["len"] < 32768]
        random.Random(0).shuffle(long_)
        random.Random(1).shuffle(short)
        chosen += (long_ + short)[:per_cat]
    for d in chosen:
        d["L"] = max(n for n in LENS if n <= max(16384, min(d["len"], max_len)))
        d["id"] = f"{d['source']}@{d['offset']}"
    return chosen


@torch.inference_mode()
def token_nll(model, ids, chunk=2048, fwd_chunk=65536):
    """NLL of ids[1:] given the preceding ids: one forward pass, or for more than fwd_chunk tokens a decode.session
    fed fwd_chunk tokens per call (the same up to rounding, with activation memory for fwd_chunk tokens)."""
    dev = next(model.parameters()).device
    x = torch.as_tensor(np.asarray(ids[:-1]), dtype=torch.long).view(1, -1).to(dev)
    y = torch.as_tensor(np.asarray(ids[1:]), dtype=torch.long).to(dev)
    out = []

    def score(h, off):
        for a in range(0, h.shape[1], chunk):
            hs = h[0, a:a + chunk]
            out.append(F.cross_entropy(model.lm_head(hs).float(), y[off + a:off + a + hs.shape[0]], reduction="none"))

    if x.shape[1] > 262144:  # 512K on 24 GB: the cache takes the room, 32K chunks as in decode.prefill (measured 23.3 GB)
        fwd_chunk = min(fwd_chunk, 32768)
    if x.shape[1] <= fwd_chunk:
        score(model.hidden(x)[0], 0)
    else:
        with decode.session():
            for a in range(0, x.shape[1], fwd_chunk):
                score(model.hidden(x[:, a:a + fwd_chunk])[0], a)
    return torch.cat(out).cpu().numpy()


def window_nll(model, ids, win=TRAIN_LEN, stride=TRAIN_LEN // 2):
    T = len(ids) - 1
    res = np.empty(T)
    s = 0
    while True:
        e = min(s + win, T)
        part = token_nll(model, ids[s:e + 1])
        lo = 0 if s == 0 else win - stride
        res[s + lo:e] = part[lo:]
        if e == T:
            return res
        s += stride


def bucket_means(nll):
    return [float(nll[a:b].mean()) if len(nll) >= b else None for a, b in BUCKETS]


def passkey_ids(tok, n_tokens, depth, key):
    """The longest prompt of at most n_tokens tokens (repeated filler tokenizes shorter than one filler alone)."""
    fixed = len(mac_lm.encode(tok, HEAD + KEY.format(k=key) + TAIL))

    def build(reps):
        before = round(depth * reps)
        return mac_lm.encode(tok, HEAD + FILLER * before + KEY.format(k=key) + FILLER * (reps - before) + TAIL)

    reps = max(1, (n_tokens - fixed) // len(tok.encode(FILLER).ids))
    ids = build(reps)
    per = (len(ids) - fixed) / reps
    reps = max(1, int((n_tokens - fixed) / per))
    ids = build(reps)
    while len(ids) > n_tokens and reps > 1:
        reps -= 1
        ids = build(reps)
    return ids


@torch.inference_mode()
def passkey_answer(model, tok, ids, max_new=8):
    cache = decode.prefill(model, ids[:-1])
    out = decode.generate_ids(model, ids[-1:], max_new, init=cache)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--tests", default="loss,passkey")
    ap.add_argument("--per-cat", type=int, default=6, help="documents per category (code / math / English)")
    ap.add_argument("--lengths", default="2048,4096,8192,16384,32768")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--keys", type=int, default=2, help="keys per (length, depth)")
    ap.add_argument("--logn", action="store_true")
    ap.add_argument("--baseline", default=None, help="JSON of a plain run: reuse its window pass (with --logn)")
    ap.add_argument("--max-len", type=int, default=32768, help="score at most this many tokens per document")
    ap.add_argument("--books", type=int, default=0,
                    help="also score this many PG19 test books of >= 64K tokens (category books, not in 'all')")
    ap.add_argument("--books-min-len", type=int, default=65536, help="only PG19 books of at least this many tokens")
    ap.add_argument("--books-windows", default="",
                    help="comma list of PG19 splits (test,validation): score every book of >= --books-min-len tokens, "
                         "cut into windows of --max-len (more 256K samples than whole books; replaces --books)")
    ap.add_argument("--scan", action="store_true", help="only count the long held-out documents")
    ap.add_argument("--batch", type=int, default=10,
                    help="passkey prompts decoded together at 32K, more for shorter ones (decode.batch_rows); 1: one by one")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    docs = heldout_docs()
    print(f"held-out documents >= 16K tokens: {len(docs)} ({time.time() - t0:.0f}s)", flush=True)
    for cat in CATS:
        c = [d for d in docs if d["cat"] == cat]
        by_src = {}
        for d in c:
            by_src[d["source"]] = by_src.get(d["source"], 0) + 1
        print(f"  {cat}: {len(c)} (>= 32K: {sum(d['len'] >= 32768 for d in c)}) {by_src}", flush=True)
    if args.scan:
        return

    model, tok, meta = mac_lm.load(args.export, args.device)
    if args.logn:
        attn_mod.F = _LogNF()
        decode.F = _LogNF()
        decode.FORCE_LOGN = TRAIN_LEN  # batched passkey rows: the same scaling with each row's own key count
    res = {"step": meta["step"], "logn": args.logn, "buckets": BUCKETS}
    tests = args.tests.split(",")

    if "loss" in tests:
        chosen = pick_docs(docs, args.per_cat, args.max_len) + pg19_books(args.books, args.max_len, min_len=args.books_min_len)
        for sp in filter(None, args.books_windows.split(",")):
            chosen += pg19_books(1, args.max_len, split=sp, min_len=args.books_min_len, windows=True)
        base = {}
        if args.baseline:
            base = {d["id"]: d for d in json.load(open(args.baseline))["loss"]["docs"]}
        rows = []
        for i, d in enumerate(chosen):
            t1 = time.time()
            ids = np.concatenate([[EOS], d["ids"][:d["L"]]])
            full = bucket_means(token_nll(model, ids))
            win = base[d["id"]]["window"] if d["id"] in base else bucket_means(window_nll(model, ids))
            rows.append({k: d[k] for k in ("id", "cat", "source", "len", "L")} | {"full": full, "window": win})
            diff = " ".join("   -  " if f is None else f"{f - w:+.3f}" for f, w in zip(full, win))
            print(f"  doc {i + 1}/{len(chosen)} {d['id']} ({d['cat']}, L={d['L']}): full - window by bucket: {diff} "
                  f"[{time.time() - t1:.0f}s]", flush=True)
        summ = {}
        cats = list(CATS) + (["books"] if args.books or args.books_windows else [])
        for cat in cats + ["all"]:  # "all": the held-out documents of the training sources, as before
            rs = [r for r in rows if r["cat"] == cat or (cat == "all" and r["cat"] in CATS)]
            per = []
            for j, (a, b) in enumerate(BUCKETS):
                pairs = [(r["full"][j], r["window"][j]) for r in rs if r["full"][j] is not None]
                if not pairs:
                    per.append(None)
                    continue
                f_, w_ = np.array(pairs).T
                per.append({"bucket": f"{a}-{b}", "docs": len(pairs), "full": round(float(f_.mean()), 4),
                            "window": round(float(w_.mean()), 4), "diff": round(float((f_ - w_).mean()), 4),
                            "docs_full_better": int((f_ < w_).sum())})
            summ[cat] = per
        res["loss"] = {"docs": rows, "summary": summ}
        print("loss by position (mean over documents): full / window / full - window / docs where full is better",
              flush=True)
        for cat, per in summ.items():
            for p in per:
                if p:
                    print(f"  {cat:8s} {p['bucket']:>11s}: {p['full']:.4f} / {p['window']:.4f} / {p['diff']:+.4f} / "
                          f"{p['docs_full_better']}/{p['docs']}", flush=True)

    if "passkey" in tests:
        lengths = [int(x) for x in args.lengths.split(",")]
        depths = [float(x) for x in args.depths.split(",")]
        rng = random.Random(0)
        trials = []
        for n in lengths:
            if args.logn and n <= TRAIN_LEN:
                continue
            hit = 0
            t1 = time.time()
            todo = []
            for dep in depths:
                for _ in range(args.keys):
                    key = rng.randint(10000, 99999)
                    todo.append((dep, key, passkey_ids(tok, n, dep, key)))
            rows = decode.batch_rows(n, args.batch) if args.batch > 1 else 1
            for a in range(0, len(todo), rows):
                part = todo[a:a + rows]
                if rows == 1:
                    answers = [passkey_answer(model, tok, part[0][2])]
                else:  # a row stops at its first 5-digit number: the only thing scored, and later tokens cannot move it
                    caches = [decode.prefill(model, ids[:-1]) for _, _, ids in part]
                    outs = decode.generate_batch(model, caches, [ids[-1] for _, _, ids in part], 8,
                                                 done=lambda b, o: re.search(r"\d{5}", tok.decode(o)) is not None)
                    answers = [tok.decode(o) for o in outs]
                for (dep, key, ids), ans in zip(part, answers):
                    m = re.search(r"\d{5}", ans)
                    ok = m is not None and int(m.group(0)) == key
                    hit += ok
                    trials.append({"length": n, "tokens": len(ids), "depth": dep, "key": key, "answer": ans,
                                   "ok": ok})
            k = len(depths) * args.keys
            print(f"  passkey {n} tokens: {hit}/{k} [{time.time() - t1:.0f}s]", flush=True)
        res["passkey"] = trials

    res["seconds"] = round(time.time() - t0)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("LONGCTX_DONE", flush=True)


if __name__ == "__main__":
    main()
