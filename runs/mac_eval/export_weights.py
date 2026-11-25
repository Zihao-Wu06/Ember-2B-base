#!/usr/bin/env python
"""Export the inference weights of a main-run checkpoint (runs in WSL next to the training, CPU only).

    CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c 19 python runs/mac_eval/export_weights.py [--ckpt PATH] [--out DIR]

Reads the newest /root/ckpt/main/ckpt_*.pt (or --ckpt) memory-mapped, through a hard link so the checkpoint rotation
cannot delete it mid-read, and writes runs/mac_eval/ckpt_<step>/: the model state dict plus the two offloaded Engram
tables (as engram.tables.<i>, the names a model built with engram_offload=False uses), fp32 as stored, in shards of
<= 512 MB (so this process never holds more than one shard), and meta.json (step, tokens, model config, shard index).
Optimizer states are not exported. Nothing is loaded onto a GPU and the checkpoint is only read."""
import argparse
import glob
import json
import os
import re
import time

import torch

SHARD_BYTES = 512 * 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = args.ckpt or sorted(glob.glob("/root/ckpt/main/ckpt_*.pt"))[-1]
    step = int(re.search(r"ckpt_(\d+)\.pt$", ck).group(1))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = args.out or os.path.join(root, "runs", "mac_eval", f"ckpt_{step:07d}")
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(os.path.dirname(os.path.abspath(ck)), ".export_tmp")  # same file system: the hard link works
    os.makedirs(tmp, exist_ok=True)
    link = os.path.join(tmp, os.path.basename(ck))
    if os.path.exists(link):
        os.remove(link)
    os.link(ck, link)
    t0 = time.time()
    try:
        sd = torch.load(link, map_location="cpu", mmap=True, weights_only=False)
        items = [(k, v) for k, v in sd["model"].items() if torch.is_tensor(v)]
        if sd.get("engram_offload") is not None:
            items += [(f"engram.tables.{i}", t["weight"]) for i, t in enumerate(sd["engram_offload"])]
        shards, index, cur, size = [], {}, {}, 0

        def flush():
            nonlocal cur, size
            name = f"model-{len(shards):04d}.pt"
            torch.save(cur, os.path.join(out, name + ".tmp"))
            os.replace(os.path.join(out, name + ".tmp"), os.path.join(out, name))
            shards.append(name)
            index.update({k: name for k in cur})
            print(f"  {name}: {len(cur)} tensors, {size / 2 ** 20:.0f} MiB ({time.time() - t0:.0f}s)", flush=True)
            cur, size = {}, 0

        n_el = 0
        for k, v in items:
            t = v.detach().contiguous().clone()  # its own storage: a shard never drags a shared buffer along
            nb = t.numel() * t.element_size()
            if cur and size + nb > SHARD_BYTES:
                flush()
            cur[k] = t
            size += nb
            if t.is_floating_point():
                n_el += t.numel()
        if cur:
            flush()
        a = sd.get("args") or {}
        meta = {"step": sd["step"], "tokens": sd["step"] * int(a.get("global_tokens", 524288)), "source": ck,
                "cfg": sd["cfg"], "float_params": n_el, "shards": shards, "index": index,
                "train_args": {k: a.get(k) for k in ("preset", "lr", "total_tokens", "mix", "mid_mix", "mid_from",
                                                     "anneal_mix", "override")},
                "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"EXPORT_DONE {out}: step {sd['step']}, {len(index)} tensors, {n_el / 1e9:.3f}B float parameters, "
              f"{len(shards)} shards, {time.time() - t0:.0f}s", flush=True)
    finally:
        os.remove(link)


if __name__ == "__main__":
    main()
