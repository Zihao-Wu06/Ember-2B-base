"""Per-layer MoE router-bias summary from training checkpoints (CPU only, memory-mapped, cheap).

The aux-free bias moves by +-gamma every step: b_i += gamma * sign(mean_load - load_i). So
(b_i - median) / gamma is roughly how many more steps expert i was under-loaded than over-loaded.
A large positive outlier = an expert the router keeps starving.

    python scripts/router_stats.py runs/ablation/C_kda_engram/ckpt_0000800.pt runs/ablation/D_*/ckpt_*.pt
"""
import argparse
import re

import torch


def summarize(path: str, gamma: float, top: int) -> None:
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    step, model = sd.get("step"), sd["model"]
    print(f"== {path}  (step {step})")
    print(f"   {'layer':>5} {'min':>7} {'median':>7} {'max':>7} {'spread/gamma':>12} {'router_row_norm min/med/max':>28}"
          f"   most-starved experts (bias above median, in steps)")
    for name in sorted((n for n in model if n.endswith("ffn.e_bias")), key=lambda n: int(re.findall(r"\d+", n)[0])):
        layer = int(re.findall(r"\d+", name)[0])
        b = model[name].float()
        med = b.median()
        w = model.get(name.replace("e_bias", "router.weight"))
        norms = "" if w is None else "{:.3f}/{:.3f}/{:.3f}".format(*(w.float().norm(dim=1).quantile(
            torch.tensor([0.0, 0.5, 1.0])).tolist()))
        order = torch.argsort(b, descending=True)[:top].tolist()
        worst = " ".join(f"e{e}:{(b[e] - med) / gamma:+.0f}" for e in order)
        print(f"   {layer:>5} {b.min():7.3f} {med:7.3f} {b.max():7.3f} {(b.max() - b.min()) / gamma:12.0f} {norms:>28}   {worst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--gamma", type=float, default=1e-3, help="router_bias_update of the run")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    for p in args.ckpts:
        summarize(p, args.gamma, args.top)


if __name__ == "__main__":
    main()
