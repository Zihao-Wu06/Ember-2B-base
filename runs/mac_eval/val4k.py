#!/usr/bin/env python
"""4K-window validation loss of exported checkpoints on the held-out windows train.py's evaluate() uses for the main
run (micro-bsz 2 x 8 batches = the first 16 windows of 4096 tokens of each source's held-out tail). Compares models on
the Mac with the same code (fp32), e.g. v1 final against a context-extension checkpoint: does short context degrade?

    .venv-mac/bin/python runs/mac_eval/val4k.py --exports runs/mac_eval/ckpt_0047684,runs/mac_eval/s1_0000400
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mac_lm  # noqa: E402
from moe_llm.data import MixtureLoader  # noqa: E402

SOURCES = "en_web,wiki_en,textbook,math,math2,code_py,edu_python2"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sources = args.sources.split(",")
    loader = MixtureLoader(os.path.join(mac_lm.ROOT, "data", "bins_main"), {s: 1.0 for s in sources}, 4096, 2)
    res = {}
    for exp in args.exports.split(","):
        model, tok, meta = mac_lm.load(exp, "cpu")
        name = os.path.basename(exp.rstrip("/"))
        res[name] = {}
        for s in sources:
            t0, tot, n = time.time(), 0.0, 0
            for x, y in loader.val_batches(s, args.windows // 2):
                lg = mac_lm.logits(model, x)
                tot += float(torch.nn.functional.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="sum"))
                n += y.numel()
            res[name][s] = tot / n
            print(f"  {name} {s}: {tot / n:.4f} on {n} tokens [{time.time() - t0:.0f}s]", flush=True)
        del model
    names = list(res)
    if len(names) > 1:
        print("source: " + " | ".join(names) + " | diff (last - first)")
        for s in sources:
            print(f"  {s}: " + " | ".join(f"{res[m][s]:.4f}" for m in names) + f" | {res[names[-1]][s] - res[names[0]][s]:+.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    print("VAL4K_DONE", flush=True)


if __name__ == "__main__":
    main()
