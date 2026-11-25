"""OptimizerBundle(adamw_last=True) / train.py --opt-adamw-last: the optimizer step copies back and updates the Muon
parameters first and the AdamW (and GPU Engram table) parameters last. Only the order of independent per-parameter
updates changes, so training must stay bit-identical to the default order:
* CPU: bit-identical training over several steps (plain bundle with GPU-resident Engram tables; offload_master with
  fused gradient accumulation under bf16 autocast; offloaded Engram tables with the async update), checkpoints are
  interchangeable between the orders, a skipped non-finite step behaves the same, and train.py end to end (logs and
  checkpoint tensors) is bit-identical with and without the flag; optimizers() / the checkpoint layout is unchanged.
* GPU (skipped without CUDA): with offload_states + offload_master and the same gradients, both orders and a plain
  bundle give bit-identical parameters, states and bf16 copies, while the side stream is stalled before every step and
  every reload destination starts as NaN (a missed wait would read NaN); step() has no host sync in either order."""
import copy
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers import common
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle, wsd_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BF16 = torch.bfloat16
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _clean():
    yield
    common.clear_cached_copies()


def _batch(cfg, s, k):
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(1000 * s + k))
    return x[:, :-1], x[:, 1:]


def _train(m, opt, cfg, steps, start=0, clip=0.05, micro=2, autocast=True, bad_step=None):
    """train.py's step on CPU: accumulate, skip a non-finite step, clip (active at 0.05), step, zero_grad, router bias.
    Returns the CE of every micro-batch and the clipped norm of every step."""
    out = []
    for s in range(start, start + steps):
        loss_sum = torch.zeros(())
        for k in range(micro):
            x, y = _batch(cfg, s, k)
            with torch.autocast("cpu", dtype=BF16, enabled=autocast):
                loss, info = m(x, y)
            if s == bad_step:
                loss = loss * float("nan")
            (loss / micro).backward()
            loss_sum += loss.detach().float()
            out.append(float(info["ce"]))
        if not bool(torch.isfinite(loss_sum)):
            opt.zero_grad()
            if m.engram is not None and m.engram.offload:
                m.engram.flush()
                for t in m.engram.cpu_tables:
                    t.grad.zero_()
                    t.touched.zero_()
            out.append("skipped")
            continue
        out.append(float(opt.clip_grad_norm(clip)))
        opt.step(wsd_lr(s, 8, 3e-3, warmup=1))
        opt.zero_grad()
        m.update_router_bias()
    return out


def _state(m, opt):
    """Every tensor that training produces: model state (weights, router biases), optimizer states (checkpoint
    layout), offloaded Engram tables and their Adam moments."""
    sd = {f"model.{k}": v.detach().clone() for k, v in m.state_dict().items()}
    for i, o in enumerate(opt.state_dict()["opts"]):
        for pi, st in o["state"].items():
            for k, v in st.items():
                if torch.is_tensor(v):
                    sd[f"opt{i}.{pi}.{k}"] = v.detach().clone()
    if m.engram is not None and m.engram.offload:
        for i, t in enumerate(m.engram.offload_state_dict()):  # flushes the queued async update first
            for k, v in t.items():
                sd[f"engram{i}.{k}"] = v.clone()
    return sd


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k


def _bundle(m, adamw_last, offload_master=False, fused=False):
    opt = OptimizerBundle(m, lr=3e-3, muon_state_dtype=BF16, offload_master=offload_master, fused_grad_acc=fused,
                          adamw_last=adamw_last)
    opt.begin()
    return opt


# ------------------------------------------------------------------------------------------------------------ CPU
def test_update_order_and_checkpoint_layout():
    m = MoELLM(tiny())  # GPU-resident Engram tables: all three optimizers exist
    a, b = OptimizerBundle(m, lr=3e-3), OptimizerBundle(m, lr=3e-3, adamw_last=True)
    assert a.muon is not None and a.adamw is not None and a.engram_gpu is not None
    assert a.update_order() == [a.adamw, a.engram_gpu, a.muon]
    assert b.update_order() == [b.muon, b.adamw, b.engram_gpu]
    assert [type(o) for o in a.optimizers()] == [type(o) for o in b.optimizers()]  # state_dict order unchanged
    m2 = MoELLM(tiny(engram_offload=True))
    c = OptimizerBundle(m2, lr=3e-3, adamw_last=True)
    assert c.engram_gpu is None and c.update_order() == [c.muon, c.adamw]


@pytest.mark.parametrize("case", ["plain_gpu_tables", "offload_master_fused", "engram_offload_async"])
def test_adamw_last_trains_bit_identically(case):
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres", engram_offload=case == "engram_offload_async")
    base = MoELLM(cfg)
    base.set_checkpointing(0.5)
    om = case == "offload_master_fused"
    runs = []
    for adamw_last in (False, True):
        m = copy.deepcopy(base)
        opt = _bundle(m, adamw_last, offload_master=om, fused=om)
        r = _train(m, opt, cfg, 4, autocast=om)
        runs.append((r, _state(m, opt)))
        common.clear_cached_copies()
    assert runs[0][0] == runs[1][0], runs
    _assert_same(runs[0][1], runs[1][1])


def test_adamw_last_checkpoints_interchangeable_and_nonfinite_skip():
    """A checkpoint written by the default order resumes under adamw_last (and the other way round) exactly like the
    uninterrupted run; a skipped non-finite step leaves both orders bit-identical."""
    torch.manual_seed(0)
    cfg = tiny(residual="block_attnres")
    base = MoELLM(cfg)
    for first, second in ((False, True), (True, False)):
        m = copy.deepcopy(base)
        opt = _bundle(m, first, offload_master=True, fused=True)
        _train(m, opt, cfg, 2)
        buf = io.BytesIO()
        torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
        full = _train(m, opt, cfg, 2, start=2)
        full_state = _state(m, opt)
        common.clear_cached_copies()
        buf.seek(0)
        sd = torch.load(buf, weights_only=False)
        m2 = MoELLM(cfg)
        m2.load_state_dict(sd["model"])
        o2 = OptimizerBundle(m2, lr=3e-3, muon_state_dtype=BF16, offload_master=True, fused_grad_acc=True,
                             adamw_last=second)
        o2.load_state_dict(sd["opt"])
        o2.begin()
        assert _train(m2, o2, cfg, 2, start=2) == full
        _assert_same(_state(m2, o2), full_state)
        common.clear_cached_copies()
    runs = []
    for adamw_last in (False, True):
        m = copy.deepcopy(base)
        opt = _bundle(m, adamw_last, offload_master=True, fused=True)
        r = _train(m, opt, cfg, 3, bad_step=1)
        assert r.count("skipped") == 1
        runs.append((r, _state(m, opt)))
        common.clear_cached_copies()
    assert runs[0][0] == runs[1][0]
    _assert_same(runs[0][1], runs[1][1])


def _bins(root):
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        os.makedirs(os.path.join(root, s), exist_ok=True)
        np.tile(rng.integers(1, 500, 101, dtype=np.uint16), 400).tofile(os.path.join(root, s, "000.bin"))


def test_train_py_flag_bit_identical(tmp_path):
    """train.py --opt-adamw-last end to end on CPU (offload_master + fused grad acc; on CPU train.py runs without
    autocast): the same logged CE / gradient norms / router stats and bit-identical checkpoint tensors."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    data = str(tmp_path / "bins")
    _bins(data)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny", "--data", data,
           "--canon", "none", "--mix", json.dumps({"a": 0.6, "b": 0.4}), "--seq-len", "32", "--micro-bsz", "2",
           "--global-tokens", "128", "--steps", "3", "--warmup", "1", "--log-every", "1", "--eval-every", "0",
           "--ckpt-every", "3", "--clip", "0.05", "--offload-master", "--fused-grad-acc", "--muon-bf16"]
    outs = {}
    for flag in (False, True):
        out = tmp_path / ("b" if flag else "a")
        r = subprocess.run(cmd + ["--out", str(out)] + (["--opt-adamw-last"] if flag else []), env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        assert "TRAIN_DONE" in r.stdout
        assert ("optimizer step order: Muon first, AdamW last" in r.stdout) == flag
        recs = [json.loads(line) for line in open(out / "log.jsonl")]
        recs = [{k: v for k, v in rec.items() if k not in ("tok_s", "mfu", "mem_gb")} for rec in recs]
        ck = torch.load(out / "ckpt_0000003.pt", weights_only=False)
        outs[flag] = (recs, ck)
    (ra, ca), (rb, cb) = outs[False], outs[True]
    assert len(ra) == 3 and ra == rb
    _assert_same({k: v for k, v in ca["model"].items()}, {k: v for k, v in cb["model"].items()})
    for oa, ob in zip(ca["opt"]["opts"], cb["opt"]["opts"]):
        assert oa["state"].keys() == ob["state"].keys()
        for pi in oa["state"]:
            for k, v in oa["state"][pi].items():
                w = ob["state"][pi][k]
                assert (torch.equal(v, w) and v.dtype == w.dtype) if torch.is_tensor(v) else v == w, (pi, k)
    assert ca["args"]["opt_adamw_last"] is False and cb["args"]["opt_adamw_last"] is True


def test_bench_script_cpu_smoke(tmp_path):
    """scripts/bench_opt_phase.py runs end to end on CPU (plumbing: micro-batch / optimizer timings, both orders, the
    no-grad forward; the profiler and overlap trials need CUDA) and writes its summary."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    out = tmp_path / "bench"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "bench_opt_phase.py"), "--preset", "tiny",
                        "--device", "cpu", "--seq-len", "32", "--accum", "3", "--warmup-steps", "1", "--steps", "2",
                        "--out", str(out)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "BENCH_DONE" in r.stdout
    s = json.load(open(out / "summary.json"))
    med = s["medians_ms"]
    for k in ("fwd_ms", "bwd_ms", "fwd_to_final_norm_ms", "final_norm_head_ce_ms", "opt_ms_default",
              "opt_ms_adamw_last", "fwd_nograd_ms", "fwd_nograd_to_final_norm_ms"):
        assert k in med and med[k] >= 0, k
    assert len(s["raw"]["fwd_ms"]) == 2 * 2  # micro-batch 0 of each timed step is left out
    assert "overlap_structural_bound_pct" in s and "adamw_last_saving_ms" in s


def test_trace_analysis_on_synthetic_trace():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import bench_opt_phase as b

    def k(name, ts, dur, stream):
        return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"stream": stream}}

    def c(name, ts, dur, stream, nbytes):
        return {"ph": "X", "cat": "gpu_memcpy", "name": name, "ts": ts, "dur": dur,
                "args": {"stream": stream, "bytes": nbytes}}
    ev = [c("Memcpy HtoD (Pinned -> Device)", 0, 50, 20, 1e9),   # side-stream reload copy: the phase starts here
          k("ampere_bf16_s1688gemm_bf16_128x128_ldg8", 50, 100, 7),  # compute stream: first kernel after 50 us
          k("vectorized_elementwise_kernel", 150, 20, 7),
          k("vectorized_elementwise_kernel", 200, 30, 7),          # 30 us gap before it
          k("multi_tensor_apply_kernel adam", 210, 10, 20),        # another stream, overlapping
          c("Memcpy DtoH (Device -> Pinned)", 230, 40, 20, 5e8),   # offload tail after the compute stream's end
          {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 999}]
    r = b.analyze_trace(ev)
    assert r["phase_ms"] == pytest.approx(0.230)
    assert r["kernel_busy_ms"] == pytest.approx(0.150)  # [50, 170] + [200, 230]
    assert r["compute_stream_busy_ms"] == pytest.approx(0.150)
    assert r["idle_ms"] == pytest.approx(0.080)
    assert r["first_compute_kernel_after_ms"] == pytest.approx(0.050)
    assert r["largest_compute_gap_ms"] == pytest.approx(0.050) and r["largest_compute_gap_at_ms"] == 0
    assert r["compute_gaps_over_20us"] == {"n": 2, "ms": pytest.approx(0.080)}
    assert r["d2h_tail_ms"] == pytest.approx(0.040)
    assert r["copies"]["memcpy HtoD"]["gb"] == pytest.approx(1.0) and r["copies"]["memcpy DtoH"]["n"] == 1
    assert r["categories_ms"]["GEMM"] == pytest.approx(0.1)
    assert r["categories_ms"]["AdamW (multi-tensor apply)"] == pytest.approx(0.01)
    assert b.analyze_trace(ev[-1:]) == {}
    assert b.union_us([(0, 2), (1, 3), (5, 6), (4, 4)]) == 4


# ------------------------------------------------------------------------------------------------------------ GPU
def _grads(opt, m):
    """Each parameter's accumulated gradient as the optimizer will read it (fp32 buffer of a cast parameter)."""
    return [opt._gbuf.get(id(p), p.grad) if id(p) in opt._cast else p.grad for p in m.parameters()]


def _give(opt, m, grads):
    for p, g in zip(m.parameters(), grads):
        if g is None:
            continue
        if id(p) in opt._cast:
            opt._gbuf[id(p)] = g.clone()
        else:
            p.grad = g.clone()


def _nan_empty(real_empty):
    def empty(*args, **kw):
        t = real_empty(*args, **kw)
        if t.is_cuda and t.is_floating_point():
            t.fill_(float("nan"))  # on the compute stream, before _reload_pipelined lets the side stream start copying
        return t
    return empty


def _gpu_cfg(**kw):
    # large vocabulary: the AdamW moments of the embedding and LM head (4 x 65536 x 128 fp32 = 134 MB) take a few ms
    # to copy, as on main_2b where they are the reason for adamw_last; triton experts as in the main run
    return tiny(kda_heads=2, kda_head_dim=64, d_model=128, vocab_size=65536, residual_bf16=True,
                residual="block_attnres", moe_impl="triton", **kw)


@gpu
def test_gpu_orders_bit_identical_with_stalled_copies(monkeypatch):
    """Tiny model on the GPU, three bundles fed the same gradients for 3 steps (clip active): default order and
    adamw_last, both with offload_states + offload_master + fused_grad_acc, and a plain bundle. Before every step the
    offloading bundles' side stream sleeps ~50 ms (all reload copies land late) and every CUDA tensor allocated inside
    step() starts as NaN, so an update that read its master or state before the copy landed would turn NaN. Masters,
    states and bf16 copies must be bit-identical across all three."""
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_layers=())
    base = MoELLM(cfg).cuda()
    base.set_checkpointing(0.5)
    ms = [copy.deepcopy(base) for _ in range(3)]
    kw = dict(lr=1e-3, muon_state_dtype=BF16)
    oa = OptimizerBundle(ms[0], offload_states=True, offload_master=True, fused_grad_acc=True, **kw)
    ob = OptimizerBundle(ms[1], offload_states=True, offload_master=True, fused_grad_acc=True, adamw_last=True, **kw)
    oc = OptimizerBundle(ms[2], **kw)
    for o in (oa, ob, oc):
        o.begin()
    real_empty = torch.empty
    try:
        for step in range(3):
            for k in range(2):
                xc = torch.randint(0, cfg.vocab_size, (2, 129), generator=torch.Generator().manual_seed(10 * step + k))
                x = xc.cuda()
                with torch.autocast("cuda", dtype=BF16):
                    loss, _ = ms[0](x[:, :-1], x[:, 1:])
                (loss / 2).backward()
            g = [None if t is None else t.clone() for t in _grads(oa, ms[0])]
            _give(ob, ms[1], g)
            _give(oc, ms[2], g)
            norms = [o.clip_grad_norm(0.05) for o in (oa, ob, oc)]
            assert torch.equal(norms[0], norms[1]) and torch.equal(norms[0], norms[2])
            monkeypatch.setattr(torch, "empty", _nan_empty(real_empty))
            for o in (oa, ob, oc):
                if o.offload_states:
                    with torch.cuda.stream(o._side_stream()):
                        torch.cuda._sleep(100_000_000)  # ~50 ms at ~2 GHz
                o.step(1e-3)
            monkeypatch.setattr(torch, "empty", real_empty)
            for o in (oa, ob, oc):
                o.zero_grad()
            ms[0].update_router_bias()
            torch.cuda.synchronize()
            pa, pb, pc = (list(m.parameters()) for m in ms)
            for i, (a, b, c) in enumerate(zip(pa, pb, pc)):
                a, b, c = a.detach().cpu(), b.detach().cpu(), c.detach().cpu()
                assert torch.isfinite(a).all() and torch.equal(a, b) and torch.equal(a, c), (step, i)
            for (_, ca), (_, cb) in zip(oa._cast.values(), ob._cast.values()):
                assert torch.equal(ca, cb)
            sa, sb, sc = (o.state_dict()["opts"] for o in (oa, ob, oc))
            for xa, xb, xc_ in zip(sa, sb, sc):
                for pi in xa["state"]:
                    for key, v in xa["state"][pi].items():
                        if torch.is_tensor(v):
                            assert torch.equal(v.cpu(), xb["state"][pi][key].cpu()), (step, pi, key)
                            assert torch.equal(v.cpu(), xc_["state"][pi][key].cpu()), (step, pi, key)
    finally:
        monkeypatch.setattr(torch, "empty", real_empty)
        common.clear_cached_copies()


@gpu
@pytest.mark.parametrize("adamw_last", [False, True])
def test_gpu_step_has_no_host_sync(adamw_last):
    """opt.step() with offload_states + offload_master queues everything without a CPU-GPU sync in either order (after
    a warm-up step that allocates the pinned buffers). No Engram (its GPU-table optimizer reads nonzero())."""
    torch.manual_seed(0)
    cfg = _gpu_cfg(engram_layers=())
    m = MoELLM(cfg).cuda()
    opt = OptimizerBundle(m, lr=1e-3, muon_state_dtype=BF16, offload_states=True, offload_master=True,
                          fused_grad_acc=True, adamw_last=adamw_last)
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
