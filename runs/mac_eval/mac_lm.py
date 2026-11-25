#!/usr/bin/env python
"""Run an exported main-run checkpoint on the Mac (CPU by default; --device mps for the Apple GPU).

    .venv-mac/bin/python runs/mac_eval/mac_lm.py --check            # load + validation loss vs the training eval
    .venv-mac/bin/python runs/mac_eval/mac_lm.py --samples          # greedy continuations of a few prompts
    .venv-mac/bin/python runs/mac_eval/mac_lm.py --prompt "def fib(n):" --max-new 64

Weights come from runs/mac_eval/ckpt_<step>/ (written in WSL by export_weights.py; the newest one unless --export).
The model is built with the same config as the training run, except the CUDA-only implementations: the Triton MoE,
cross-entropy and AttnRes kernels and fla's KDA / conv kernels are replaced by the reference PyTorch paths (same
parameters and math; KDA runs fla's naive recurrence), torch.compile is off, Engram tables live in the model, and the
weights stay in fp32 (training ran the forward in bf16 autocast, so losses differ at the rounding level)."""
import argparse
import glob
import json
import math
import os
import sys
import time

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # the "compiled" ops run eagerly: same math, no inductor

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import moe_llm.layers.common as _common  # noqa: E402
from moe_llm.config import ModelConfig  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402

EOS = 0  # <|endoftext|>: ends every training document, so a prompt starts after one
_conv1d_causal = _common.causal_depthwise_conv


def _causal_depthwise_conv_cpu(x, weight, dilation: int = 1):
    """causal_depthwise_conv as k shifted multiply-adds (the formula of common._conv_silu_shifted). On the CPU a
    depthwise F.conv1d runs one slow conv per channel (~64k calls per decode step, 70% of its time)."""
    if x.is_cuda:
        return _conv1d_causal(x, weight, dilation)
    T, k = x.shape[1], weight.shape[-1]
    xp = torch.nn.functional.pad(x, (0, 0, (k - 1) * dilation, 0))
    w = weight.to(x.dtype)[:, 0, :]
    y = xp[:, 0:T] * w[:, 0]
    for i in range(1, k):
        y = y + xp[:, i * dilation:i * dilation + T] * w[:, i]
    return y


_common.causal_depthwise_conv = _causal_depthwise_conv_cpu
# moe_impl "loop": only the experts that got tokens (bmm pads every expert to the busiest one: 8x the weight reads
# per decoded token)
CPU_OVERRIDES = dict(engram_offload=False, moe_impl="loop", moe_shared_fused="", moe_sort="torch",
                     attnres_impl="compiled", conv_impl="compiled", ce_impl="chunked", ce_blas="",
                     residual_bf16=False, kda_backend="torch", kda_keep_layers=0)


def latest_export() -> str:
    metas = sorted(glob.glob(os.path.join(ROOT, "runs", "mac_eval", "ckpt_*", "meta.json")))
    if not metas:
        raise FileNotFoundError("no export in runs/mac_eval/ (run export_weights.py in WSL first)")
    return os.path.dirname(metas[-1])


def load(export: str | None = None, device: str = "cpu"):
    """(model, tokenizer, meta) for the export directory (newest by default)."""
    from tokenizers import Tokenizer
    d = export or latest_export()
    meta = json.load(open(os.path.join(d, "meta.json")))
    cfg = ModelConfig.from_dict(meta["cfg"]).replace(**CPU_OVERRIDES)
    if os.environ.get("MAC_LM_KDA"):  # e.g. MAC_LM_KDA=auto on the GPU: fla's chunked kernel (fast, TF32 rounding)
        cfg = cfg.replace(kda_backend=os.environ["MAC_LM_KDA"])
    if str(device).startswith("cuda"):  # fp32 everywhere, as on the CPU (fla's own kernels aside)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # past the card's memory WSL silently spills into shared system memory and everything crawls: fail instead
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MAC_LM_MEM_FRAC", "0.94")))
    t0 = time.time()
    torch.manual_seed(0)
    model = MoELLM(cfg)
    want = set(model.state_dict().keys())
    got = set()
    for name in meta["shards"]:
        sd = torch.load(os.path.join(d, name), map_location="cpu", weights_only=True)
        res = model.load_state_dict(sd, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"{name}: unexpected keys {res.unexpected_keys[:5]}")
        got.update(sd.keys())
        del sd
    missing = sorted(want - got)
    if missing:
        raise RuntimeError(f"keys not in the export: {missing[:8]} ({len(missing)})")
    model.eval().to(device)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer", "tokenizer.json"))
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {os.path.basename(d)}: step {meta['step']}, {meta['tokens'] / 1e9:.2f}B tokens, {n / 1e9:.3f}B "
          f"parameters on {device} ({time.time() - t0:.0f}s)", flush=True)
    return model, tok, meta


def encode(tok, text: str, bos: bool = True) -> list:
    return ([EOS] if bos else []) + tok.encode(text).ids


@torch.inference_mode()
def logits(model, ids: torch.Tensor) -> torch.Tensor:
    """[B, T] token ids -> [B, T, V] fp32 logits."""
    return model(ids.to(next(model.parameters()).device)).float()


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new: int = 64, temperature: float = 0.0, top_k: int = 0,
             stop: tuple = ("\n\n\n",), seed: int = 0) -> str:
    """Greedy (temperature 0) or sampled continuation. No cache: every new token reruns the whole sequence."""
    g = torch.Generator().manual_seed(seed)
    ids = encode(tok, prompt)
    start = len(ids)
    for _ in range(max_new):
        lg = logits(model, torch.tensor([ids]))[0, -1].cpu()
        if temperature <= 0:
            nxt = int(lg.argmax())
        else:
            lg = lg / temperature
            if top_k:
                v, _ = lg.topk(top_k)
                lg[lg < v[-1]] = -float("inf")
            nxt = int(torch.multinomial(lg.softmax(-1), 1, generator=g))
        if nxt == EOS:
            break
        ids.append(nxt)
        text = tok.decode(ids[start:])
        if any(s in text for s in stop):
            break
    return tok.decode(ids[start:])


@torch.inference_mode()
def continuation_logprobs(model, ctx_ids: list, conts: list, pad: int = 1) -> list:
    """Sum of log p(continuation | context) for each continuation (lists of token ids), one batched forward."""
    seqs = [ctx_ids + c for c in conts]
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    lp = torch.log_softmax(logits(model, x), dim=-1).cpu()
    out = []
    for i, c in enumerate(conts):
        n0 = len(ctx_ids)
        pos = torch.arange(n0 - 1, n0 - 1 + len(c))
        out.append(float(lp[i, pos, torch.tensor(c)].sum()))
    return out


@torch.inference_mode()
def check(model, meta, sources=("wiki_en", "code_py", "math", "edu_python"), windows: int = 2, seq: int = 4096):
    """Validation loss on the first windows of the training eval split of some sources (the same windows
    train.py's evaluate() starts with), next to the latest training eval in runs/main/log.jsonl."""
    from moe_llm.data import MixtureLoader
    ref, ref_step = {}, None
    try:
        for line in open(os.path.join(ROOT, "runs", "main", "log.jsonl")):
            if '"eval"' in line:
                r = json.loads(line)
                if r["step"] < meta["step"]:
                    ref, ref_step = r["eval"], r["step"]
    except OSError:
        pass
    loader = MixtureLoader(os.path.join(ROOT, "data", "bins_main"), {s: 1.0 for s in sources}, seq, windows)
    dev = next(model.parameters()).device
    for s in sources:
        t0 = time.time()
        x, y = next(loader.val_batches(s, 1))
        lg = logits(model, x)
        ce = float(torch.nn.functional.cross_entropy(lg.flatten(0, 1), y.to(dev).flatten()))
        r = f"(training eval at step {ref_step}: {ref[s]:.4f} over 16 windows)" if s in ref else ""
        print(f"  {s}: {ce:.4f} on {x.numel()} tokens {r} [{time.time() - t0:.0f}s]", flush=True)


SAMPLES = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "Question: Tom has 3 boxes with 12 apples each. He gives away 7 apples. How many apples does he have left?\nAnswer:",
    "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is a prime number.\"\"\"\n",
    "In 1969, Neil Armstrong",
    "Theorem. The sum of the first n odd numbers is n^2.\nProof.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model, tok, meta = load(args.export, args.device)
    if args.check:
        check(model, meta)
    prompts = SAMPLES if args.samples else ([args.prompt] if args.prompt else [])
    for p in prompts:
        t0 = time.time()
        out = generate(model, tok, p, args.max_new, args.temperature)
        print(f"\n### {p}\n{out}\n[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
