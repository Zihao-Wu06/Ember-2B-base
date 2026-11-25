#!/usr/bin/env python
"""Accuracy of Muon's Newton-Schulz variants against the exact (fp64) iteration, and the Gram Newton-Schulz analysis
(measurement only: nothing here is used by training).

Variants (moe_llm/optim.py): "loop" = zeropower_via_newtonschulz5 (the training code), "batched" =
newtonschulz5_stacked (same ops on a stack of matrices), "batched_fused" = newtonschulz5_stacked(fused=True) (the
scalar terms in the GEMM epilogues), plus two variants that are only analysed here:
  "diagfold"  B' = b*A + c*A^2 (one GEMM), then a*I added to B''s diagonal and X = B' @ X as a plain GEMM (no X pass
              at all, but a + B_ii is rounded to the iteration dtype);
  "gram"      Gram Newton-Schulz (Tri Dao, 2026): iterate on the m x m Gram matrix R = X X^T and apply the accumulated
              polynomial Q to X once at the end; "gram_r2" restarts after two iterations (X = Q2 X, R = X X^T, Q = I).
Derivation for our constant coefficients (a, b, c) = (3.4445, -4.7750, 2.0315), T = 5, m <= n:
  standard  X_t = h(R_{t-1}) X_{t-1},  R = X X^T,  h(R) = a I + b R + c R^2
  Gram      Z_t = h(R_{t-1}),  Q_t = Q_{t-1} Z_t,  R_t = Z_t R_{t-1} Z_t,  X_T = Q_T X_0   (Q_0 = I, R_0 = X_0 X_0^T)
  Proof: by induction X_t = Q_t X_0 with Q_t a polynomial in R_0 (symmetric, commutes with R_0), so
  R_t = X_t X_t^T = Z_t X_{t-1} X_{t-1}^T Z_t = Z_t R_{t-1} Z_t and X_t = Z_t X_{t-1} = Z_t Q_{t-1} X_0.
  FLOPs with dense GEMMs: standard T(4 m^2 n + 2 m^3) = 20 m^2 n + 10 m^3; Gram 4 m^2 n + (8T - 6) m^3 = 4 m^2 n + 34 m^3
  (R_0 and Q_T X_0: 2 m^2 n each; R^2 every step; Q_1 = Z_1 is free; Q_t = Q_{t-1} Z_t for t >= 2; R_t = Z (R Z) for
  t < T). For the experts (m = 512, n = 1024): 42 vs 50 m^3 (-16%); square 1024 x 1024: 38 vs 30 (+27%).
    python scripts/muon_ns_accuracy.py [--sizes 256x512,512x1024] [--trials 3]
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm.optim import (newtonschulz5_stacked, zeropower_via_newtonschulz5,  # noqa: E402
                           zeropower_via_newtonschulz5_fused)

A_, B_, C_ = 3.4445, -4.7750, 2.0315
F64 = torch.float64


# ------------------------------------------------------------------------------------------------ variants
def ns_diagfold(G: torch.Tensor, steps: int = 5, dtype=torch.bfloat16) -> torch.Tensor:
    X = G.to(dtype)
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    mm = torch.baddbmm if X.dim() == 3 else torch.addmm
    for _ in range(steps):
        A = X @ X.mT
        B = mm(A, A, A, beta=B_, alpha=C_)
        B.diagonal(dim1=-2, dim2=-1).add_(A_)
        X = B @ X
    return X.mT if tr else X


def gram_ns5(G: torch.Tensor, steps: int = 5, dtype=torch.bfloat16, restart_after=None) -> torch.Tensor:
    """Gram Newton-Schulz with the a*I term applied as separate products (the blog's stabilised form):
    Z' = b R + c R^2;  Q <- Q Z' + a Q;  RZ = R Z' + a R;  R <- Z' RZ + a RZ;  every '+ a ...' in a GEMM epilogue.
    restart_after=k: before step k + 1, X = Q X, R = X X^T, Q = I."""
    X = G.to(dtype)
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    batched = X.dim() == 3
    mm = torch.baddbmm if batched else torch.addmm
    m = X.size(-2)
    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    if batched:
        eye = eye.expand(X.size(0), m, m)
    R, Q = X @ X.mT, None
    for t in range(1, steps + 1):
        if restart_after is not None and t == restart_after + 1 and Q is not None:
            X = Q @ X
            R, Q = X @ X.mT, None
        Zp = mm(R, R, R, beta=B_, alpha=C_)                    # b R + c R^2
        Q = mm(eye, eye, Zp, beta=A_) if Q is None else mm(Q, Q, Zp, beta=A_)  # Q Z = Q Z' + a Q (Q_0 = I)
        last = t == steps or (restart_after is not None and t == restart_after)
        if not last:
            RZ = mm(R, R, Zp, beta=A_)                         # R Z
            R = mm(RZ, Zp, RZ, beta=A_)                         # Z R Z = Z' (R Z) + a (R Z)
    X = Q @ X
    return X.mT if tr else X


def run_variant(name: str, G: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """G: [k, m, n] or [m, n] in fp32. Returns the variant's output in fp64."""
    if name == "loop":
        out = zeropower_via_newtonschulz5(G, 5, dtype=dtype)
    elif name == "loop_fused":
        out = zeropower_via_newtonschulz5_fused(G, 5, dtype=dtype)
    elif name in ("batched", "batched_fused"):
        S = G.to(dtype)
        if S.dim() == 2:
            S = S.unsqueeze(0)
        out = newtonschulz5_stacked(S, 5, fused=name == "batched_fused")
        if G.dim() == 2:
            out = out[0]
    elif name == "diagfold":
        out = ns_diagfold(G, 5, dtype)
    elif name == "gram":
        out = gram_ns5(G, 5, dtype)
    elif name == "gram_r2":
        out = gram_ns5(G, 5, dtype, restart_after=2)
    else:
        raise ValueError(name)
    return out.to(F64)


def reference(G: torch.Tensor, rounded_to=None) -> torch.Tensor:
    """The exact iteration: zeropower_via_newtonschulz5 in fp64 on the fp32 input, or (rounded_to=dtype) on the input
    rounded to the iteration dtype first, as every variant rounds it (isolates the iteration's own rounding errors)."""
    if rounded_to is not None:
        G = G.to(rounded_to)
    return zeropower_via_newtonschulz5(G.to(F64), 5, dtype=F64)


# ------------------------------------------------------------------------------------------------ inputs
def make_inputs(kind: str, k: int, m: int, n: int, seed: int) -> torch.Tensor:
    """[k, m, n] fp32 test matrices. gauss: i.i.d. normal; decay1 / decay2: singular values i^-1 / i^-2 with random
    singular vectors (momentum-like spectra); lowrank: rank 8 plus 1e-3 noise (badly conditioned); outlier: gauss with
    a few large entries (heavy rows/columns)."""
    g = torch.Generator().manual_seed(seed)
    if kind == "gauss":
        return torch.randn(k, m, n, generator=g)
    if kind in ("decay1", "decay2"):
        r = min(m, n)
        U = torch.linalg.qr(torch.randn(k, m, r, generator=g, dtype=F64))[0]
        V = torch.linalg.qr(torch.randn(k, n, r, generator=g, dtype=F64))[0]
        s = torch.arange(1, r + 1, dtype=F64) ** (-1.0 if kind == "decay1" else -2.0)
        return ((U * s) @ V.mT).float()
    if kind == "lowrank":
        L = torch.randn(k, m, 8, generator=g) @ torch.randn(k, 8, n, generator=g) / math.sqrt(8)
        return L + 1e-3 * torch.randn(k, m, n, generator=g)
    if kind == "outlier":
        X = torch.randn(k, m, n, generator=g)
        idx = torch.randint(0, m * n, (k, 8), generator=g)
        X.view(k, -1).scatter_(1, idx, 50.0)
        return X
    raise ValueError(kind)


def rel_err(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-matrix relative Frobenius error, [k]."""
    return (x - ref).flatten(-2).norm(dim=-1) / ref.flatten(-2).norm(dim=-1)


def compare(variants, kinds, sizes, k=4, trials=2, dtype=torch.bfloat16, seed=0, rounded_ref=False):
    """{(kind, size): {variant: [per-matrix relative errors]}}"""
    res = {}
    for kind in kinds:
        for (m, n) in sizes:
            errs = {v: [] for v in variants}
            for t in range(trials):
                G = make_inputs(kind, k, m, n, seed + 1000 * t + 7 * m + n)
                ref = reference(G, dtype if rounded_ref else None)
                for v in variants:
                    errs[v] += rel_err(run_variant(v, G, dtype), ref).tolist()
            res[(kind, f"{m}x{n}")] = errs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256x512,512x1024,512x512,128x1024")
    ap.add_argument("--kinds", default="gauss,decay1,decay2,lowrank,outlier")
    ap.add_argument("--variants", default="loop,batched,batched_fused,diagfold,gram,gram_r2")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    torch.manual_seed(0)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    variants = args.variants.split(",")
    dtype = getattr(torch, args.dtype)
    # fp64 equivalence of every variant with the training code's iteration
    G = make_inputs("decay1", 2, 64, 128, 0)
    ref = reference(G)
    print("fp64 equivalence (max relative error vs zeropower_via_newtonschulz5 in fp64):")
    for v in variants:
        print(f"  {v:14s} {float(rel_err(run_variant(v, G, F64), ref).max()):.2e}")
    for rounded in (False, True):
        res = compare(variants, args.kinds.split(","), sizes, args.k, args.trials, dtype, rounded_ref=rounded)
        print(f"\n{args.dtype} relative error vs the fp64 iteration on the "
              f"{'input rounded to ' + args.dtype if rounded else 'fp32 input'}: mean (max) over "
              f"{args.k * args.trials} matrices")
        print(f"{'input':22s}" + "".join(f"{v:>22s}" for v in variants))
        ratio = {v: [] for v in variants}
        for key, errs in res.items():
            base = statistics.mean(errs["loop"]) if "loop" in errs else None
            print(f"{key[0] + ' ' + key[1]:22s}" + "".join(
                f"{statistics.mean(e):12.3e} ({max(e):.2e})" for e in errs.values()))
            if base:
                for v, e in errs.items():
                    ratio[v].append(statistics.mean(e) / base)
        if "loop" in variants:
            print("mean error relative to loop, geometric mean over inputs (max):")
            for v, r in ratio.items():
                print(f"  {v:14s} {math.exp(statistics.mean(math.log(x) for x in r)):.3f} ({max(r):.3f})")


if __name__ == "__main__":
    main()
