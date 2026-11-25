"""Batched Muon (OptimizerBundle(muon_impl=...), train.py --muon-impl; moe_llm/optim.py newtonschulz5_stacked):
* "batched": the Muon parameters of one shape are stacked into chunks and orthogonalised by one Newton-Schulz per chunk;
  per element the same ops as the per-parameter loop (each member's norm over its own slice, the same elementwise ops,
  the GEMMs over all matrices of the chunk at once), plus two exact shortcuts (the Nesterov input rounded straight into
  the stack; the bf16 output added to the fp32 weight without an fp32 copy).
* "batched_fused": batched, with b*A and a*X in the epilogues of the two GEMMs of each iteration (one rounding instead of
  four and three): rounding-level change, closer to the exact iteration.
Checked on CPU:
* fp64: every variant (and the Gram Newton-Schulz derivation, analysis only) computes the same function;
* bf16 (the CUDA iteration dtype, emulated on CPU): stacked == per-matrix bit for bit, batched_fused's error against the
  fp64 iteration is below the loop's for every input class (mean and max); Gram NS is not (documents why it is not
  offered);
* the chunk plan, the update / reload order, Muon steps and whole-model training (offload_master, fused gradient
  accumulation, bf16 autocast) bit-identical to the loop in CUDA emulation (and within fp32 rounding with CPU's fp32
  iteration, where MKL's batched GEMM sums in another order), checkpoints interchangeable between impls, train.py flag.
GPU tests at the end (skipped without CUDA): main_2b shapes against fp64 (bit-identity reported), bundles with stalled
reload copies and NaN-filled destinations, no host sync, optimizer-step peak memory."""
import copy
import functools
import io
import json
import math
import os
import statistics
import subprocess
import sys

import numpy as np
import pytest
import torch

from moe_llm import optim
from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.model import MoELLM
from moe_llm.optim import (Muon, OptimizerBundle, newtonschulz5_stacked, wsd_lr, zeropower_via_newtonschulz5,
                           zeropower_via_newtonschulz5_fused)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import muon_ns_accuracy as acc  # noqa: E402

BF16, F64 = torch.bfloat16, torch.float64
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
IMPLS = ("loop", "batched", "batched_fused")


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


@pytest.fixture
def cuda_numerics(monkeypatch):
    """Run every Newton-Schulz in bf16 on CPU, as on CUDA (loop, chunks of one and stacks)."""
    monkeypatch.setattr(optim, "_ns_dtype", lambda t: BF16)
    monkeypatch.setattr(optim, "zeropower_via_newtonschulz5",
                        functools.partial(zeropower_via_newtonschulz5, dtype=BF16))
    monkeypatch.setattr(optim, "zeropower_via_newtonschulz5_fused",
                        functools.partial(zeropower_via_newtonschulz5_fused, dtype=BF16))


SHAPES = [(64, 128), (128, 64), (96, 96), (8, 32, 64), (8, 64, 32)]


# ------------------------------------------------------------------------------------------------ Newton-Schulz
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fused", [False, True])
def test_stacked_ns_bit_identical_per_member_bf16(shape, fused):
    """newtonschulz5_stacked on a stack == the per-tensor function on each member, bit for bit, in bf16 (wide, tall
    and square 2-D members, 3-D expert-like members in both orientations)."""
    g = torch.Generator().manual_seed(0)
    G = torch.randn(3, *shape, generator=g)
    O = newtonschulz5_stacked(G.to(BF16), 5, fused)
    f = zeropower_via_newtonschulz5_fused if fused else zeropower_via_newtonschulz5
    for i in range(3):
        ref = f(G[i], 5, dtype=BF16)
        assert O[i].dtype == BF16 and O[i].shape == ref.shape and torch.equal(O[i], ref), i


@pytest.mark.parametrize("shape", SHAPES)
def test_stacked_ns_fp32_within_rounding(shape):
    """CPU's fp32 iteration: MKL's batched GEMM may sum in another order than the per-matrix GEMM; the results agree to
    fp32 rounding (the bf16 test above checks the stacking itself bit for bit)."""
    G = torch.randn(3, *shape, generator=torch.Generator().manual_seed(1))
    O = newtonschulz5_stacked(G.float(), 5)
    for i in range(3):
        torch.testing.assert_close(O[i], zeropower_via_newtonschulz5(G[i], 5), rtol=0, atol=5e-6)


def test_fp64_all_variants_same_function():
    """In fp64 every variant computes zeropower_via_newtonschulz5's iteration (to fp64 rounding; Gram NS's
    accumulated polynomial too: the derivation in scripts/muon_ns_accuracy.py)."""
    for kind in ("gauss", "decay1"):
        for (m, n) in ((32, 64), (64, 32), (48, 48)):
            G = acc.make_inputs(kind, 2, m, n, seed=3)
            ref = acc.reference(G)
            for v in ("loop", "loop_fused", "batched", "batched_fused", "gram", "gram_r2"):
                err = float(acc.rel_err(acc.run_variant(v, G, F64), ref).max())
                assert err < 1e-11, (kind, m, n, v, err)


@pytest.mark.parametrize("rounded_ref", [False, True])
def test_bf16_accuracy_fused_no_worse_than_loop(rounded_ref):
    """Relative error against the fp64 iteration (on the fp32 input, or on the input rounded to bf16 as every variant
    rounds it): batched == loop exactly; batched_fused below the loop in mean and in max for every input class
    (well and badly conditioned, wide / tall / square)."""
    res = acc.compare(["loop", "batched", "batched_fused"], ["gauss", "decay1", "decay2", "lowrank", "outlier"],
                      [(64, 128), (128, 64), (96, 96)], k=3, trials=1, dtype=BF16, rounded_ref=rounded_ref)
    ratios = []
    for key, e in res.items():
        assert e["batched"] == e["loop"], key
        assert statistics.mean(e["batched_fused"]) < statistics.mean(e["loop"]), (key, e)
        assert max(e["batched_fused"]) < max(e["loop"]), (key, e)
        ratios.append(statistics.mean(e["batched_fused"]) / statistics.mean(e["loop"]))
    print(f"batched_fused / loop mean error: geometric mean {math.exp(statistics.mean(map(math.log, ratios))):.3f}, "
          f"max {max(ratios):.3f}")


def test_gram_ns_not_offered_bf16_accuracy():
    """Documents the analysis: Gram Newton-Schulz is the same function (fp64 test above) but in bf16 it is less
    accurate than the loop on some inputs, naive by orders of magnitude on badly conditioned ones (the m x m polynomial
    Q grows up to 3.4445^5 in the directions of small singular values and its rounding leaks into the others), with the
    restart after two iterations still above the loop on square inputs: it fails "no worse than today"."""
    res = acc.compare(["loop", "gram", "gram_r2"], ["gauss", "decay1"], [(64, 128), (96, 96)], k=2, trials=1,
                      dtype=BF16, rounded_ref=True)
    worst = {v: max(statistics.mean(e[v]) / statistics.mean(e["loop"]) for e in res.values()) for v in ("gram", "gram_r2")}
    assert worst["gram"] > 3 and worst["gram_r2"] > 1.0, worst


# ------------------------------------------------------------------------------------------------ chunk plan
def _params(shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g) * 0.02) for s in shapes]


def test_chunk_plan_and_update_order():
    shapes = [(64, 128), (8, 64, 32), (64, 128), (128, 64), (8, 64, 32), (64, 128), (5, 7), (5, 7), (96, 96)]
    ps = _params(shapes)
    for budget, sizes in ((4 * 64 * 128, [3, 2, 1, 1, 1, 1]), (2 * 64 * 128, [2, 1, 1, 1, 1, 1, 1, 1])):
        m = Muon(ps, impl="batched", chunk_elems=budget)
        chunks = m._chunks(0)
        flat = [p for c in chunks for p in c]
        assert sorted(map(id, flat)) == sorted(map(id, ps)) and len(flat) == len(ps)
        for c in chunks:
            assert len({tuple(p.shape) for p in c}) == 1
            assert len(c) == 1 or sum(p.numel() for p in c) <= budget
        # groups in order of their first parameter, members in parameter order, each group split evenly; (5, 7) is not
        # 512-byte aligned: chunks of one; (128, 64) and (96, 96) are alone
        assert [len(c) for c in chunks] == sizes
        order = [tuple(c[0].shape) for c in chunks]
        assert sorted(set(order), key=order.index) == [(64, 128), (8, 64, 32), (128, 64), (5, 7), (96, 96)]
        assert chunks[0] == ([ps[0], ps[2], ps[5]] if sizes[0] == 3 else [ps[0], ps[2]])
        assert m.update_params() == flat and m._chunks(0) is chunks  # cached
    loop = Muon(ps, impl="loop")
    assert loop.update_params() == ps
    with pytest.raises(ValueError):
        Muon(ps, impl="gram")


def test_main_2b_plan_on_meta():
    """main_2b's 319 Muon parameters: 47 chunks with the default chunk size, every expert chunk 2 tensors."""
    from moe_llm.config import main_2b
    with torch.device("meta"):
        m = MoELLM(main_2b().replace(engram_offload=False, moe_impl="triton"))
    ps = optim.split_params(m)["muon"]
    mu = Muon(ps, impl="batched_fused")
    chunks = mu._chunks(0)
    assert len(ps) == 319 and len(chunks) == 47
    by = {}
    for c in chunks:
        by.setdefault(tuple(c[0].shape), []).append(len(c))
    assert by[(32, 1024, 512)] == [2] * 23 and by[(32, 512, 1024)] == [2] * 11 + [1]
    assert by[(1024, 1024)] == [30, 30, 30] and by[(128, 1024)] == [36] and by[(1024, 2816)] == [1]
    assert max(sum(p.numel() for p in c) for c in chunks) <= optim.MUON_CHUNK_ELEMS


def test_bundle_reload_order_follows_chunks():
    m = MoELLM(tiny())
    for impl in IMPLS:
        opt = OptimizerBundle(m, lr=3e-3, muon_impl=impl)
        order = [p for o in opt.update_order() for p in opt._update_params(o)]
        mu = opt.muon
        expect = [p for g in opt.adamw.param_groups for p in g["params"]] + \
                 ([p for g in opt.engram_gpu.param_groups for p in g["params"]] if opt.engram_gpu else []) + \
                 (mu.update_params())
        assert order == expect
        if impl == "loop":
            assert mu.update_params() == [p for g in mu.param_groups for p in g["params"]]
        else:
            assert mu.update_params() == [p for c in mu._chunks(0) for p in c]


# ------------------------------------------------------------------------------------------------ Muon steps
def _muon_run(impl, shapes, steps=3, state_dtype=BF16, nesterov=True, param_dtype=torch.float32, wd=0.1, seed=0):
    ps = _params(shapes, seed)
    for p in ps:
        p.data = p.data.to(param_dtype)
    m = Muon(ps, lr=0.02, weight_decay=wd, nesterov=nesterov, state_dtype=state_dtype, impl=impl, chunk_elems=1 << 14)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g).to(param_dtype)
        m.step()
    return [p.detach().clone() for p in ps], [m.state[p]["buf"].clone() for p in ps]


MIXED = [(64, 128), (8, 64, 32), (64, 128), (128, 64), (8, 64, 32), (64, 128), (5, 7), (128, 64), (8, 32, 64)]


@pytest.mark.parametrize("variant", ["bf16_state", "fp32_state", "no_nesterov", "bf16_params", "no_wd"])
def test_muon_batched_bit_identical_cuda_numerics(cuda_numerics, variant):
    kw = {"bf16_state": {}, "fp32_state": {"state_dtype": torch.float32}, "no_nesterov": {"nesterov": False},
          "bf16_params": {"param_dtype": BF16}, "no_wd": {"wd": 0.0}}[variant]
    pa, sa = _muon_run("loop", MIXED, **kw)
    pb, sb = _muon_run("batched", MIXED, **kw)
    for i, (a, b) in enumerate(zip(pa + sa, pb + sb)):
        assert a.dtype == b.dtype and torch.equal(a, b), (variant, i)


def test_muon_batched_fp32_iteration_close():
    pa, sa = _muon_run("loop", MIXED)
    pb, sb = _muon_run("batched", MIXED)
    for a, b in zip(pa, pb):
        torch.testing.assert_close(a, b, rtol=0, atol=2e-6)
    for a, b in zip(sa, sb):  # the momentum never sees Newton-Schulz: bit-identical
        assert torch.equal(a, b)


def test_muon_batched_fused_tracks_loop(cuda_numerics):
    """Rounding-level: the update differs from the loop's by bf16 rounding of the Newton-Schulz output only."""
    p0 = [p.detach().clone() for p in _params(MIXED)]
    pa, sa = _muon_run("loop", MIXED, steps=1)
    pf, sf = _muon_run("batched_fused", MIXED, steps=1)
    for x0, a, f in zip(p0, pa, pf):
        ua, uf = (a - x0 * (1 - 0.02 * 0.1)), (f - x0 * (1 - 0.02 * 0.1))
        assert float((ua - uf).norm() / ua.norm()) < 0.05
    for a, f in zip(sa, sf):
        assert torch.equal(a, f)


# ------------------------------------------------------------------------------------------------ training
def _batch(cfg, s, k):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(1000 * s + k))
    return x[:, :-1], x[:, 1:]


def _train(m, opt, cfg, steps, start=0, clip=0.05, micro=2, autocast=True):
    out = []
    for s in range(start, start + steps):
        for k in range(micro):
            x, y = _batch(cfg, s, k)
            with torch.autocast("cpu", dtype=BF16, enabled=autocast):
                loss, info = m(x, y)
            (loss / micro).backward()
            out.append(float(info["ce"]))
        out.append(float(opt.clip_grad_norm(clip)))
        opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
        opt.zero_grad()
        m.update_router_bias()
    return out


def _state(m, opt):
    sd = {f"model.{k}": v.detach().clone() for k, v in m.state_dict().items()}
    for i, o in enumerate(opt.state_dict()["opts"]):
        for pi, st in o["state"].items():
            for k, v in st.items():
                if torch.is_tensor(v):
                    sd[f"opt{i}.{pi}.{k}"] = v.detach().clone()
    return sd


def _bundle(m, impl, chunk_elems=1 << 12):
    opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=True, muon_impl=impl,
                          muon_chunk_elems=chunk_elems)
    opt.begin()
    return opt


def test_training_batched_bit_identical_cuda_numerics(cuda_numerics):
    """Tiny model (block AttnRes, recompute, MoE), offload_master + fused gradient accumulation under bf16 autocast,
    4 steps with clipping: loop and batched give bit-identical losses, norms, weights, router biases and states;
    batched_fused tracks them at the rounding level."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    base.set_checkpointing(0.5)
    runs = {}
    for impl in IMPLS:
        m = copy.deepcopy(base)
        opt = _bundle(m, impl)
        assert impl == "loop" or len(opt.muon._chunks(0)) < len(opt.muon.param_groups[0]["params"])
        runs[impl] = (_train(m, opt, cfg, 4), _state(m, opt))
        common.clear_cached_copies()
    (ra, sa), (rb, sb), (rf, sf) = runs["loop"], runs["batched"], runs["batched_fused"]
    assert ra == rb, (ra, rb)
    assert sa.keys() == sb.keys() == sf.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]), k
    assert max(abs(x - y) for x, y in zip(ra, rf)) < 2e-3, (ra, rf)


def test_training_batched_fp32_iteration_close():
    torch.manual_seed(0)
    cfg = tiny()
    base = MoELLM(cfg)
    m1, m2 = copy.deepcopy(base), copy.deepcopy(base)
    r1 = _train(m1, _bundle(m1, "loop"), cfg, 3)
    common.clear_cached_copies()
    r2 = _train(m2, _bundle(m2, "batched"), cfg, 3)
    assert max(abs(x - y) for x, y in zip(r1, r2)) < 1e-4


def test_checkpoints_interchangeable_between_impls(cuda_numerics):
    """A checkpoint written under one impl resumes under another exactly like the uninterrupted run (CUDA numerics:
    loop and batched are bit-identical); the optimizer-state layout is the same for all impls."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    layouts = []
    for first, second in (("loop", "batched"), ("batched", "loop"), ("batched_fused", "batched_fused")):
        m = copy.deepcopy(base)
        opt = _bundle(m, first)
        _train(m, opt, cfg, 2)
        buf = io.BytesIO()
        torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
        sd0 = opt.state_dict()
        layouts.append({(i, pi, k): (tuple(v.shape), v.dtype) for i, o in enumerate(sd0["opts"])
                        for pi, st in o["state"].items() for k, v in st.items() if torch.is_tensor(v)})
        full = _train(m, opt, cfg, 2, start=2)
        full_state = _state(m, opt)
        common.clear_cached_copies()
        buf.seek(0)
        sd = torch.load(buf, weights_only=False)
        m2 = MoELLM(cfg)
        m2.load_state_dict(sd["model"])
        o2 = OptimizerBundle(m2, lr=3e-3, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=True,
                             muon_impl=second, muon_chunk_elems=1 << 12)
        o2.load_state_dict(sd["opt"])
        o2.begin()
        assert all(st["buf"].dtype == BF16 for st in o2.muon.state.values())
        assert _train(m2, o2, cfg, 2, start=2) == full, (first, second)
        s2 = _state(m2, o2)
        for k in full_state:
            assert torch.equal(s2[k], full_state[k]), (first, second, k)
        common.clear_cached_copies()
    assert layouts[0] == layouts[1] == layouts[2]


def _bins(root):
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


def test_train_py_muon_impl_flag(tmp_path):
    """train.py --muon-impl end to end on CPU (offload_master + fused grad acc + bf16 momentum): batched trains and
    logs like loop (CPU iterates in fp32: equal within fp32 rounding), prints its chunk line, and resumes a loop
    checkpoint (and batched_fused runs)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    data = str(tmp_path / "bins")
    _bins(data)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny", "--data", data,
           "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--seq-len", "32", "--micro-bsz", "2",
           "--global-tokens", "128", "--warmup", "1", "--log-every", "1", "--eval-every", "0",
           "--clip", "0.05", "--offload-master", "--fused-grad-acc", "--muon-bf16"]
    recs = {}
    for impl in IMPLS:
        out = tmp_path / impl
        r = subprocess.run(cmd + ["--steps", "3", "--ckpt-every", "2", "--out", str(out), "--muon-impl", impl], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        assert "TRAIN_DONE" in r.stdout
        assert ("Muon Newton-Schulz: " in r.stdout) == (impl != "loop")
        recs[impl] = [json.loads(line) for line in open(out / "log.jsonl")]
        ck = torch.load(out / "ckpt_0000003.pt", weights_only=False)
        assert ck["args"]["muon_impl"] == impl
    for impl in ("batched", "batched_fused"):
        assert len(recs[impl]) == 3
        for a, b in zip(recs["loop"], recs[impl]):
            assert abs(a["ce"] - b["ce"]) < 1e-3 and abs(a["gnorm"] - b["gnorm"]) < 1e-2 * a["gnorm"], (impl, a, b)
    # resume the loop run's step-2 checkpoint under batched for one more step
    res = tmp_path / "resume"
    os.makedirs(res)
    os.link(tmp_path / "loop" / "ckpt_0000002.pt", res / "ckpt_0000002.pt")
    r = subprocess.run(cmd + ["--steps", "3", "--ckpt-every", "0", "--out", str(res), "--muon-impl", "batched"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "resumed from" in r.stdout and "TRAIN_DONE" in r.stdout
    rec = [json.loads(line) for line in open(res / "log.jsonl")]
    assert len(rec) == 1 and rec[0]["step"] == 2 and abs(rec[0]["ce"] - recs["loop"][2]["ce"]) < 1e-3


# ------------------------------------------------------------------------------------------------ scripts
def test_cost_model_counts():
    """scripts/muon_cost_model.py's dispatch counter on a few main_2b shapes: same GEMM FLOPs for all impls, fewer
    launches when batched, fewer non-GEMM bytes when fused."""
    import muon_cost_model as cm
    shapes = [(32, 1024, 512)] * 2 + [(1024, 1024)] * 4 + [(128, 1024)] * 3
    r = {impl: cm.run_meta(shapes, impl, optim.MUON_CHUNK_ELEMS)[0] for impl in IMPLS}
    flops = {impl: r[impl].flops["ns"] for impl in IMPLS}
    assert flops["loop"] == flops["batched"] == flops["batched_fused"] > 0
    expect = 5 * (2 * (4 * 512 * 512 * 1024 + 2 * 512 ** 3) * 32 + 4 * 6 * 1024 ** 3 + 3 * (4 * 128 ** 2 * 1024 + 2 * 128 ** 3))
    assert flops["loop"] == expect
    launches = {impl: sum(r[impl].launch.values()) for impl in IMPLS}
    assert launches["batched_fused"] < launches["batched"] < launches["loop"]
    assert r["batched_fused"].bytes["ns"] < 0.6 * r["batched"].bytes["ns"]
    assert r["batched"].bytes["other"] < r["loop"].bytes["other"]


def test_bench_opt_phase_cpu_smoke(tmp_path):
    """scripts/bench_opt_phase.py on CPU with several Muon impls (plumbing; the profile needs CUDA)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    out = tmp_path / "bench"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "bench_opt_phase.py"), "--preset", "tiny",
                        "--device", "cpu", "--seq-len", "32", "--accum", "2", "--warmup-steps", "1", "--steps", "3",
                        "--muon-impls", "loop,batched,batched_fused", "--overlap-trials", "0", "--out", str(out)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "BENCH_DONE" in r.stdout
    s = json.load(open(out / "summary.json"))
    for impl in IMPLS:
        assert s["medians_ms"][f"opt_ms_{impl}"] >= 0
    assert s["settings"]["muon_impls"] == list(IMPLS)


# ------------------------------------------------------------------------------------------------ GPU
MAIN_SHAPES = [(32, 1024, 512), (32, 512, 1024), (1024, 1024), (768, 1024), (1024, 768), (128, 1024), (1024, 128),
               (256, 1024), (1024, 640), (2816, 1024)]


def _momentum_like(k, shape, seed, device):
    """[k, *shape] fp32 with decaying spectra (singular values i^-1) and a gaussian part."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    *lead, m, n = shape
    r = min(m, n)
    out = []
    for _ in range(k * math.prod(lead)):
        U = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=F64))[0]
        V = torch.linalg.qr(torch.randn(n, r, generator=g, dtype=F64))[0]
        s = torch.arange(1, r + 1, dtype=F64) ** -1.0
        out.append(((U * s) @ V.T + 1e-3 * torch.randn(m, n, generator=g, dtype=F64)).float())
    return torch.stack(out).view(k, *shape).to(device)


@gpu
@pytest.mark.parametrize("shape", MAIN_SHAPES)
def test_gpu_main_shapes_vs_fp64(shape):
    """Each main_2b Muon shape, a stack of 2 members as the batched step builds it: batched against the loop
    (bit-identity reported: cuBLAS may pick another kernel for the larger batch), both and batched_fused against the
    fp64 iteration of the same bf16 input. Rounding parity for batched (mean error within 2% of the loop's), fused below
    the loop in mean and max."""
    torch.manual_seed(0)
    G = _momentum_like(2, shape, seed=hash(shape) % 1000, device="cuda")
    S = G.to(BF16)
    loop = [zeropower_via_newtonschulz5(G[i], 5) for i in range(2)]
    bat = newtonschulz5_stacked(S, 5)
    fus = newtonschulz5_stacked(S, 5, fused=True)
    ident = all(torch.equal(bat[i], loop[i]) for i in range(2))
    m, n = shape[-2:]
    ref = torch.stack([zeropower_via_newtonschulz5(S[i].to(F64), 5, dtype=F64) for i in range(2)]).reshape(-1, m, n)
    e = {name: acc.rel_err(torch.stack([o[i] for i in range(2)]).to(F64).reshape(-1, m, n), ref)
         for name, o in (("loop", loop), ("batched", bat), ("fused", fus))}
    print(f"{shape}: batched bit-identical to loop: {ident}; mean rel error loop {float(e['loop'].mean()):.3e} "
          f"batched {float(e['batched'].mean()):.3e} fused {float(e['fused'].mean()):.3e}")
    assert float(e["batched"].mean()) <= 1.02 * float(e["loop"].mean())
    assert float(e["fused"].mean()) < float(e["loop"].mean()) and float(e["fused"].max()) < float(e["loop"].max())


def _nan_empty(real_empty):
    def empty(*args, **kw):
        t = real_empty(*args, **kw)
        if t.is_cuda and t.is_floating_point():
            t.fill_(float("nan"))
        return t
    return empty


def _gpu_cfg(**kw):
    return tiny(kda_heads=2, kda_head_dim=64, d_model=128, vocab_size=65536, residual_bf16=True,
                residual="block_attnres", moe_impl="triton", **kw)


def _grads(opt, m):
    return [opt._gbuf.get(id(p), p.grad) if id(p) in opt._cast else p.grad for p in m.parameters()]


def _give(opt, m, grads):
    for p, g in zip(m.parameters(), grads):
        if g is None:
            continue
        if id(p) in opt._cast:
            opt._gbuf[id(p)] = g.clone()
        else:
            p.grad = g.clone()


@gpu
def test_gpu_bundle_impls_with_stalled_copies(monkeypatch):
    """Four bundles fed the same gradients for 3 steps (clip active): loop / batched / batched_fused with
    offload_states + offload_master + fused_grad_acc and small chunks (several stacked chunks per shape), and a plain
    loop bundle. The offloading bundles' side stream sleeps ~50 ms before every step and every CUDA tensor allocated in
    step() starts as NaN, so a chunk that read a member before its copy landed would turn NaN. Everything finite;
    batched vs loop bit-identity reported (per tensor), updates within bf16 rounding; AdamW tensors bit-identical."""
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_layers=())
    base = MoELLM(cfg).cuda()
    base.set_checkpointing(0.5)
    impls = ["loop", "batched", "batched_fused", "plain"]
    ms = [copy.deepcopy(base) for _ in impls]
    kw = dict(lr=1e-3, muon_state_dtype=BF16)
    opts = [OptimizerBundle(m, offload_states=True, offload_master=True, fused_grad_acc=True, muon_impl=impl,
                            muon_chunk_elems=1 << 16, **kw) for m, impl in zip(ms[:3], impls[:3])]
    opts.append(OptimizerBundle(ms[3], **kw))
    for o in opts:
        o.begin()
    muon_ids = {id(p) for p in opts[0].muon.param_groups[0]["params"]}
    real_empty = torch.empty
    try:
        for step in range(3):
            for k in range(2):
                xc = torch.randint(0, cfg.vocab_size, (2, 129), generator=torch.Generator().manual_seed(10 * step + k))
                x = xc.cuda()
                with torch.autocast("cuda", dtype=BF16):
                    loss, _ = ms[0](x[:, :-1], x[:, 1:])
                (loss / 2).backward()
            g = [None if t is None else t.clone() for t in _grads(opts[0], ms[0])]
            for o, m in zip(opts[1:], ms[1:]):
                _give(o, m, g)
            norms = [o.clip_grad_norm(0.05) for o in opts]
            assert all(torch.equal(norms[0], n) for n in norms)
            before = [[p.detach().clone() for p in m.parameters()] for m in ms]  # each model's own update below
            monkeypatch.setattr(torch, "empty", _nan_empty(real_empty))
            for o in opts:
                if o.offload_states:
                    with torch.cuda.stream(o._side_stream()):
                        torch.cuda._sleep(100_000_000)
                o.step(1e-3)
            monkeypatch.setattr(torch, "empty", real_empty)
            for o in opts:
                o.zero_grad()
            ms[0].update_router_bias()
            torch.cuda.synchronize()
            ident, worst = 0, 0.0
            for i, ps in enumerate(zip(*(m.parameters() for m in ms))):
                a, b, f, c = (p.detach() for p in ps)
                assert all(torch.isfinite(t).all() for t in (a, b, f, c)), (step, i)
                # offloading loop == plain loop (offload_master keeps the offloading bundles' fp32 masters on the CPU)
                assert torch.equal(a.to(c.device), c), (step, i)
                if id(ps[0]) in muon_ids:
                    ident += int(torch.equal(a, b))
                    ua = (a - before[0][i]).double()  # same gradients and momentum in all bundles: compare the updates
                    for j, t in ((1, b), (2, f)):
                        ut = (t - before[j][i]).double()
                        worst = max(worst, float((ut - ua).norm() / ua.norm().clamp_min(1e-30)))
                else:
                    assert torch.equal(a, b) and torch.equal(a, f), (step, i)
            print(f"step {step}: batched bit-identical to loop on {ident}/{len(muon_ids)} Muon tensors, worst relative "
                  f"update difference (batched / fused vs loop) {worst:.2e}")
            assert worst < 0.1
            for (_, ca), (_, cb) in zip(opts[0]._cast.values(), opts[1]._cast.values()):
                assert torch.isfinite(cb.float()).all()
    finally:
        monkeypatch.setattr(torch, "empty", real_empty)
        common.clear_cached_copies()


@gpu
@pytest.mark.parametrize("impl", ["batched", "batched_fused"])
def test_gpu_batched_step_has_no_host_sync(impl):
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_layers=())
    m = MoELLM(cfg).cuda()
    opt = OptimizerBundle(m, lr=1e-3, muon_state_dtype=BF16, offload_states=True, offload_master=True,
                          fused_grad_acc=True, muon_impl=impl, muon_chunk_elems=1 << 16)
    opt.begin()
    x = torch.randint(0, cfg.vocab_size, (2, 129), device="cuda")
    try:
        for check in (False, True):
            with torch.autocast("cuda", dtype=BF16):
                loss, _ = m(x[:, :-1], x[:, 1:])
            loss.backward()
            opt.clip_grad_norm(1.0)
            torch.cuda.synchronize()
            if check:
                torch.cuda.set_sync_debug_mode("error")
            try:
                opt.step(1e-3)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            opt.zero_grad()
            torch.cuda.synchronize()
    finally:
        common.clear_cached_copies()


@gpu
def test_gpu_muon_step_peak_memory():
    """Peak memory of one Muon step over main_2b-shaped parameters (4 expert tensors, 30 of 1024 x 1024, 6 of
    768 x 1024, 12 of 128 x 1024; bf16 momentum) per impl, above what the parameters, gradients and states hold.
    Reported; bounded by the stack and iteration temporaries of one chunk."""
    shapes = [(32, 1024, 512)] * 2 + [(32, 512, 1024)] * 2 + [(1024, 1024)] * 30 + [(768, 1024)] * 6 + [(128, 1024)] * 12
    out = {}
    for impl in IMPLS:
        torch.cuda.empty_cache()
        g = torch.Generator(device="cuda").manual_seed(0)
        ps = [torch.nn.Parameter(torch.randn(s, device="cuda", generator=g) * 0.02) for s in shapes]
        for p in ps:
            p.grad = torch.randn(p.shape, device="cuda", generator=g)
        mu = Muon(ps, lr=1e-3, state_dtype=BF16, impl=impl)
        mu.step()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        mu.step()
        torch.cuda.synchronize()
        out[impl] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        del ps, mu
    print("Muon step peak above its tensors (MiB): " + ", ".join(f"{k} {v:.0f}" for k, v in out.items()))
    assert out["batched"] <= out["loop"] + 512 and out["batched_fused"] <= out["loop"] + 384
