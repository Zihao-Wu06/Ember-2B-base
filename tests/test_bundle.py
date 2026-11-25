"""data_prep/make_bundle.py (the compact data for a rented GPU) and the loader's <source>.long streams."""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from make_bundle import build, plan  # noqa: E402
from long_docs import index_source  # noqa: E402
from moe_llm.data import MixtureLoader, TokenStream  # noqa: E402

L = 64


def _bins(root, name, lengths, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for n in lengths:
        parts += [rng.integers(1, 500, n, dtype=np.uint16), np.zeros(1, dtype=np.uint16)]
    arr = np.concatenate(parts)
    os.makedirs(os.path.join(root, name), exist_ok=True)
    half = len(arr) // 2
    arr[:half].tofile(os.path.join(root, name, "000.bin"))
    arr[half:].tofile(os.path.join(root, name, "001.bin"))
    index_source(os.path.join(root, name), min_len=16, val_tokens=300)
    return arr


def _docs(arr):
    cut = np.flatnonzero(arr == 0)
    return [tuple(arr[a + 1:b].tolist()) for a, b in zip(np.concatenate([[-1], cut[:-1]]), cut)]


def test_plan():
    mixes = {"s3": {"a": 0.5, "long:a": 0.3, "long:b": 0.2}, "s4": {"a": 0.6, "long:b": 0.4}}
    p = plan([("s3", 128, 1000), ("s4", 256, 2000)], mixes, margin=1.0)
    assert p["a"] == {"plain": 500 + 1200, "long_min": 64, "lens": [128]}
    assert p["b"] == {"plain": 0, "long_min": 64, "lens": [128, 256]}


def test_bundle_keeps_long_windows_and_validation(tmp_path):
    full, out = str(tmp_path / "full"), str(tmp_path / "bundle")
    # many short documents, some long ones (> L/2), short documents at the end (the held-out tail)
    a = _bins(full, "a", ([10, 25, 7, 100, 12, 40, 18, 70, 9, 15, 33] * 60) + [8] * 80, seed=0)
    b = _bins(full, "b", ([20, 90, 15, 130, 24, 40] * 30) + [6] * 80, seed=1)
    mixes = {"m": {"a": 0.5, "long:a": 0.3, "long:b": 0.2}}
    man = build(full, out, [("m", L, 6000)], mixes, margin=1.0, val_tokens=300, block_tokens=500, log=lambda s: None)
    assert set(os.listdir(out)) == {"a", "a.long", "b.long", "manifest.json"}  # b has no plain weight
    assert json.load(open(os.path.join(out, "manifest.json")))["tokens"] == man["tokens"]
    for src in ("a", "b"):
        before, after = man["sources"][src]["long"]["windows"][str(L)]
        assert before == after > 0
    # the plain sample: whole documents of the training region, about the planned 3000 tokens, then the same tail
    pa = np.fromfile(os.path.join(out, "a", "000.bin"), dtype=np.uint16)
    assert set(_docs(pa[:-300])) <= set(_docs(a))
    assert 3000 <= man["sources"]["a"]["plain"]["sampled"] < 3000 + 16 * 110
    assert np.array_equal(pa[-300:], a[-300:])

    w = {"a": 0.5, "long:a": 0.3, "long:b": 0.2}
    old = MixtureLoader(full, w, seq_len=L, batch_size=4, seed=0, val_tokens=300)
    new = MixtureLoader(out, w, seq_len=L, batch_size=4, seed=0, val_tokens=300)
    assert new.streams["long:a"] is not new.streams["a"]  # a.long, not the plain sample
    for src in ("long:a", "long:b"):
        # every window starts at the EOS before a long document and covers more than half of it
        heads = lambda ld: sorted(tuple(ld.streams[src].read(int(s), L // 2 + 1).tolist()) for s in ld.starts[src])
        assert heads(old) == heads(new)
    va = [x for x, _ in old.val_batches("a", 10)]
    vb = [x for x, _ in new.val_batches("a", 10)]
    assert len(va) == len(vb) > 0 and all(torch.equal(x, y) for x, y in zip(va, vb))
    for _ in range(5):  # the bundle trains
        x, y = new.next_batch()
        assert x.shape == (4, L) and torch.equal(x[:, 1:], y[:, :-1])


def test_loader_without_long_dir_is_unchanged(tmp_path):
    full = str(tmp_path / "full")
    _bins(full, "a", [10, 100, 12, 70] * 40 + [8] * 80, seed=0)
    ld = MixtureLoader(full, {"a": 0.5, "long:a": 0.5}, seq_len=L, batch_size=2, val_tokens=300)
    assert ld.streams["long:a"] is ld.streams["a"]
    assert isinstance(ld.streams["a"], TokenStream)


def test_offloaded_table_loads_weights_only():
    from moe_llm.layers.engram import OffloadedTable
    t = OffloadedTable(50, 4, 0.02, seed=0)
    t.m.fill_(1.0)
    t.v.fill_(2.0)
    w = torch.randn(50, 4)
    t.load_state_dict({"weight": w.to(torch.bfloat16)})  # slim_init.py: bf16 weights, no Adam moments
    assert torch.equal(t.weight, w.to(torch.bfloat16).float()) and t.weight.dtype == torch.float32
    assert (t.m == 1).all() and (t.v == 2).all()  # untouched (zeros in a fresh table)
    full = {"weight": w, "m": torch.ones(50, 4) * 3, "v": torch.ones(50, 4) * 4}
    t.load_state_dict(full)
    assert torch.equal(t.weight, w) and (t.m == 3).all() and (t.v == 4).all()


def test_bundle_only_and_manifest_name(tmp_path):
    full, out = str(tmp_path / "full"), str(tmp_path / "bundle")
    _bins(full, "a", ([10, 25, 7, 100, 12, 40, 18, 70, 9, 15, 33] * 60) + [8] * 80, seed=0)
    _bins(full, "b", ([20, 90, 15, 130, 24, 40] * 30) + [6] * 80, seed=1)
    mixes = {"m": {"a": 0.5, "long:a": 0.3, "long:b": 0.2}}
    m = build(full, out, [("m", L, 6000)], mixes, margin=1.0, val_tokens=300, block_tokens=500, log=lambda s: None,
              only=("b",), manifest_name="manifest_b.json")
    assert set(os.listdir(out)) == {"b.long", "manifest_b.json"} and list(m["sources"]) == ["b"]
