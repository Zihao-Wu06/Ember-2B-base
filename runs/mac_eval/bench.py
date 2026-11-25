#!/usr/bin/env python
"""Multiple-choice benchmarks of an exported checkpoint on the Mac, scored by log-likelihood like lm-eval-harness.

    .venv-mac/bin/python runs/mac_eval/bench.py --tasks hellaswag,arc_easy --limit 200
    .venv-mac/bin/python runs/mac_eval/bench.py --tasks all --limit 500 --mmlu-per-subject 10 --out runs/mac_eval/x.json

Formats follow lm-eval-harness: hellaswag (its preprocessing; acc_norm = log-likelihood per character), arc_easy /
arc_challenge and piqa ("Question: ...\\nAnswer:" + " choice"), winogrande (partial scoring: the rest of the sentence
after each filled-in option), mmlu (5-shot from the dev split, next-token probabilities of " A".." D"). Data:
data/raw/bench/*/ (JSON lines written by bench_to_jsonl.py). --limit takes the first N items of each task (fixed,
so runs on different checkpoints are comparable)."""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mac_lm  # noqa: E402  (sets TORCH_COMPILE_DISABLE before torch is imported)
import torch  # noqa: E402

BENCH = os.path.join(mac_lm.ROOT, "data", "raw", "bench")
CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "piqa": 0.5, "winogrande": 0.5, "mmlu": 0.25}


def read_jsonl(*parts):
    with open(os.path.join(BENCH, *parts), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Scorer:
    """Batched log-likelihood of continuations; the LM head runs only on the positions that are scored."""

    def __init__(self, model, tok, batch_tokens=6144, max_batch=32):
        self.model, self.tok = model, tok
        self.dev = next(model.parameters()).device
        self.batch_tokens, self.max_batch = batch_tokens, max_batch
        self.n_tokens = 0
        self.t = 0.0

    def ids(self, text, bos=True):
        return mac_lm.encode(self.tok, text, bos)

    def _batches(self, lengths):
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        b = []
        for i in order:
            if b and (len(b) + 1) * lengths[b[0]] > self.batch_tokens or len(b) >= self.max_batch:
                yield b
                b = []
            b.append(i)
        if b:
            yield b

    @torch.inference_mode()
    def _run(self, seqs, positions):
        """seqs: token lists; positions[i]: positions of seq i whose output is needed -> list of [n_i, V] log-probs."""
        out = [None] * len(seqs)
        for b in self._batches([len(s) for s in seqs]):
            t0 = time.time()
            T = max(len(seqs[i]) for i in b)
            x = torch.full((len(b), T), 1, dtype=torch.long)
            for r, i in enumerate(b):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i])
            h, _ = self.model.hidden(x.to(self.dev))
            rows = torch.cat([h[r, torch.tensor(positions[i], device=self.dev)] for r, i in enumerate(b)])
            lp = torch.log_softmax(self.model.lm_head(rows).float(), dim=-1).cpu()
            k = 0
            for i in b:
                n = len(positions[i])
                out[i] = lp[k:k + n]
                k += n
            self.n_tokens += sum(len(seqs[i]) for i in b)
            self.t += time.time() - t0
        return out

    def loglik(self, requests):
        """requests: (context ids, continuation ids) -> sum of log p(continuation | context) each."""
        seqs = [c + d for c, d in requests]
        pos = [list(range(len(c) - 1, len(c) - 1 + len(d))) for c, d in requests]
        lps = self._run(seqs, pos)
        return [float(lp[torch.arange(len(d)), torch.tensor(d)].sum()) for lp, (_, d) in zip(lps, requests)]

    def next_token(self, contexts, cand_ids):
        """log p(candidate token | context) for every context and candidate token id."""
        lps = self._run(contexts, [[len(c) - 1] for c in contexts])
        return [[float(lp[0, t]) for t in cand_ids] for lp in lps]


def multiple_choice(sc, items, name):
    """items: (context text, [continuation texts], gold index). acc and acc_norm (per-character normalisation)."""
    reqs, spans = [], []
    for ctx, conts, _ in items:
        c = sc.ids(ctx)
        spans.append((len(reqs), len(conts)))
        reqs += [(c, sc.ids(x, bos=False)) for x in conts]
    ll = sc.loglik(reqs)
    acc = acc_n = 0
    for (s, n), (_, conts, gold) in zip(spans, items):
        v = ll[s:s + n]
        acc += max(range(n), key=lambda j: v[j]) == gold
        acc_n += max(range(n), key=lambda j: v[j] / max(1, len(conts[j]))) == gold
    return {"task": name, "n": len(items), "acc": acc / len(items), "acc_norm": acc_n / len(items)}


def hellaswag(limit):
    def pre(t):
        t = t.strip().replace(" [title]", ". ")
        t = re.sub("\\[.*?\\]", "", t)
        return t.replace("  ", " ")
    items = []
    for d in read_jsonl("hellaswag", "validation.jsonl")[:limit]:
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        items.append((pre(d["activity_label"] + ": " + ctx), [" " + pre(e) for e in d["endings"]], int(d["label"])))
    return items


def arc(limit, which):
    items = []
    for d in read_jsonl("arc", f"ARC-{which}-test.jsonl")[:limit]:
        labels, texts = d["choices"]["label"], d["choices"]["text"]
        items.append((f"Question: {d['question']}\nAnswer:", [" " + t for t in texts], labels.index(d["answerKey"])))
    return items


def piqa(limit):
    return [(f"Question: {d['goal']}\nAnswer:", [" " + d["sol1"], " " + d["sol2"]], int(d["label"]))
            for d in read_jsonl("piqa", "validation.jsonl")[:limit]]


def winogrande(sc, limit):
    data = read_jsonl("winogrande", "winogrande_xl-validation.jsonl")[:limit]
    reqs = []
    for d in data:
        i = d["sentence"].index("_")
        rest = sc.ids(" " + d["sentence"][i + 1:].strip(), bos=False)
        reqs += [(sc.ids(d["sentence"][:i] + d[o]), rest) for o in ("option1", "option2")]
    ll = sc.loglik(reqs)
    acc = sum((0 if ll[2 * k] > ll[2 * k + 1] else 1) == int(d["answer"]) - 1 for k, d in enumerate(data))
    return {"task": "winogrande", "n": len(data), "acc": acc / len(data)}


def mmlu(sc, per_subject):
    dev, test = read_jsonl("mmlu", "dev.jsonl"), read_jsonl("mmlu", "test.jsonl")
    shots, taken, items = {}, {}, []
    for d in dev:
        shots.setdefault(d["subject"], []).append(d)

    def fmt(d, ans=True):
        s = d["question"].strip() + "\n" + "".join(f"{L}. {c}\n" for L, c in zip("ABCD", d["choices"])) + "Answer:"
        return s + (f" {'ABCD'[d['answer']]}\n\n" if ans else "")
    for d in test:
        s = d["subject"]
        if taken.get(s, 0) >= per_subject:
            continue
        taken[s] = taken.get(s, 0) + 1
        head = f"The following are multiple choice questions (with answers) about {s.replace('_', ' ')}.\n\n"
        items.append((head + "".join(fmt(x) for x in shots[s][:5]) + fmt(d, False), d["answer"], s))
    cand = [sc.tok.encode(f" {L}").ids for L in "ABCD"]
    assert all(len(c) == 1 for c in cand), cand
    lps = sc.next_token([sc.ids(p) for p, _, _ in items], [c[0] for c in cand])
    hits = [max(range(4), key=lambda j: v[j]) == gold for v, (_, gold, _) in zip(lps, items)]
    return {"task": "mmlu", "n": len(items), "acc": sum(hits) / len(items), "subjects": len(taken), "shots": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--mmlu-per-subject", type=int, default=10)
    ap.add_argument("--batch-tokens", type=int, default=6144)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = list(CHANCE) if args.tasks == "all" else args.tasks.split(",")
    model, tok, meta = mac_lm.load(args.export, args.device)
    sc = Scorer(model, tok, args.batch_tokens)
    results = []
    for t in tasks:
        t0, n0 = time.time(), sc.n_tokens
        if t == "hellaswag":
            r = multiple_choice(sc, hellaswag(args.limit), t)
        elif t in ("arc_easy", "arc_challenge"):
            r = multiple_choice(sc, arc(args.limit, "Easy" if t == "arc_easy" else "Challenge"), t)
        elif t == "piqa":
            r = multiple_choice(sc, piqa(args.limit), t)
        elif t == "winogrande":
            r = winogrande(sc, args.limit)
        elif t == "mmlu":
            r = mmlu(sc, args.mmlu_per_subject)
        else:
            raise ValueError(t)
        r.update(chance=CHANCE[t], seconds=round(time.time() - t0), tokens=sc.n_tokens - n0)
        results.append(r)
        print(json.dumps(r), flush=True)
    summary = {"step": meta["step"], "tokens": meta["tokens"], "device": args.device, "limit": args.limit,
               "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
