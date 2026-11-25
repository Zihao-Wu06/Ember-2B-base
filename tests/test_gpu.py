"""GPU-only checks (skipped on machines without CUDA): fla KDA kernel vs the reference,
grouped-GEMM MoE vs bmm, and a bf16 train step of the real code path."""
import pytest
import torch

from moe_llm.config import tiny
from moe_llm.layers.kda import fla_chunk_kda, kda_reference
from moe_llm.layers.moe import MoE
from moe_llm.model import MoELLM
from moe_llm.optim import OptimizerBundle
from torch.utils.checkpoint import checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12)).item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fla_chunk_kda_matches_reference(dtype):
    chunk_kda = fla_chunk_kda()
    assert chunk_kda is not None, "fla not importable"
    torch.manual_seed(0)
    B, T, H, K = 2, 300, 4, 128
    dev = "cuda"
    mk = lambda *s: torch.randn(*s, device=dev, dtype=dtype).requires_grad_()  # noqa: E731
    q, k, v, g = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    beta = mk(B, T, H)
    A_log = torch.log(torch.empty(H, device=dev).uniform_(1, 16)).requires_grad_()
    dt_bias = (torch.randn(H * K, device=dev) * 0.5).requires_grad_()
    ins = [q, k, v, g, beta, A_log, dt_bias]
    o = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, use_qk_l2norm_in_kernel=True,
                  use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)[0]
    do = torch.randn_like(o)
    g_tri = torch.autograd.grad(o, ins, do)
    o_ref = kda_reference(q.float(), k.float(), v.float(), g.float(), beta.float(), A_log, dt_bias)
    g_ref = torch.autograd.grad(o_ref, ins, do.float())
    scale = 1 if dtype == torch.float32 else 3  # bf16 inputs are rounded before the kernel
    assert rel_rms(o, o_ref) < 0.005 * scale
    for name, a, b, tol in zip(["dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias"], g_tri, g_ref,
                               [0.008, 0.008, 0.008, 0.02, 0.02, 0.02, 0.02]):
        assert rel_rms(a, b) < tol * scale, (name, rel_rms(a, b))


def test_moe_grouped_matches_bmm():
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm missing")
    torch.manual_seed(0)
    cfg = tiny(d_model=128, expert_inter=128, n_experts=8, top_k=2)
    moe = MoE(cfg, 1).cuda()
    x = torch.randn(2, 64, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        moe.impl = "bmm"
        a, _ = moe(x)
        moe.impl = "grouped"
        try:
            b, _ = moe(x)
        except Exception as e:
            pytest.skip(f"grouped_mm unsupported here: {e}")
    assert rel_rms(b, a) < 1e-2


def test_offload_async_pipeline_matches_gpu_tables():
    """CUDA path of the offload pipeline (prefetch on the worker thread + async D2H gradient
    scatter) vs GPU-resident tables. Gradients are compared tightly; after optimizer steps only
    the loss is compared, because Adam's first step is ~lr*sign(g) and turns 1e-10 atomic-add
    noise on near-zero gradients into lr-sized parameter differences."""
    torch.manual_seed(0)
    cfg = tiny(seq_aux_coef=0.0)
    m_gpu = MoELLM(cfg).cuda()
    m_off = MoELLM(cfg.replace(engram_offload=True)).cuda()
    m_off.load_state_dict(m_gpu.state_dict(), strict=False)
    for t_gpu, t_off in zip(m_gpu.engram.tables, m_off.engram.cpu_tables):
        t_off.weight.copy_(t_gpu.detach().cpu())
    o_gpu, o_off = OptimizerBundle(m_gpu, lr=1e-3), OptimizerBundle(m_off, lr=1e-3)
    xs = [torch.randint(0, cfg.vocab_size, (2, 33)) for _ in range(6)]
    for step in range(3):
        mbs = xs[2 * step: 2 * step + 2]
        m_off.engram.prefetch(mbs[0][:, :-1])
        for i, x in enumerate(mbs):
            if i + 1 < len(mbs):
                m_off.engram.prefetch(mbs[i + 1][:, :-1])
            for m in (m_gpu, m_off):
                xc = x.cuda()
                loss, _ = m(xc[:, :-1], xc[:, 1:])
                (loss / len(mbs)).backward()
        if step == 0:
            m_off.engram.flush()
            for t_gpu, t_off in zip(m_gpu.engram.tables, m_off.engram.cpu_tables):
                g = t_gpu.grad.cpu()
                assert (g - t_off.grad).abs().max() <= 1e-6 * g.abs().max()
                assert torch.equal(g.ne(0).any(-1), t_off.touched)
        for m, o in ((m_gpu, o_gpu), (m_off, o_off)):
            o.step(1e-3)
            o.zero_grad()
    with torch.no_grad():
        x = xs[0].cuda()
        l_gpu, _ = m_gpu(x[:, :-1], x[:, 1:])
        l_off, _ = m_off(x[:, :-1], x[:, 1:])
    assert abs(float(l_gpu) - float(l_off)) < 1e-3 * float(l_gpu)


@pytest.mark.parametrize("offload", [False, True])
def test_bf16_train_step_matches_fp32_reference(offload):
    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, engram_offload=offload, residual="block_attnres")
    m = MoELLM(cfg).cuda()
    m.grad_checkpointing = True
    ref = MoELLM(cfg.replace(kda_backend="torch")).cuda()
    ref.load_state_dict(m.state_dict(), strict=False)
    if offload:
        for a, b in zip(ref.engram.cpu_tables, m.engram.cpu_tables):
            a.weight.copy_(b.weight)
    x = torch.randint(0, cfg.vocab_size, (2, 65), device="cuda")
    with torch.no_grad():
        l_ref, _ = ref(x[:, :-1], x[:, 1:])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = m(x[:, :-1], x[:, 1:])
    assert abs(float(loss) - float(l_ref)) < 2e-2 * float(l_ref)
    loss.backward()
    opt = OptimizerBundle(m, lr=1e-3)
    opt.step(1e-3)
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_bf16_residual_close_to_fp32_residual():
    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual="block_attnres")
    m = MoELLM(cfg).cuda()
    m16 = MoELLM(cfg.replace(residual_bf16=True)).cuda()
    m16.load_state_dict(m.state_dict())
    x = torch.randint(0, cfg.vocab_size, (2, 129), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l32, _ = m(x[:, :-1], x[:, 1:])
        l16, _ = m16(x[:, :-1], x[:, 1:])
    l16.backward()
    assert abs(float(l16) - float(l32)) < 1e-2 * float(l32)


def test_attnres_compiled_matches_eager_and_is_faster():
    """Every source count of the main model (2..9), train and eval graphs: the compiled AttnRes op matches
    the eager one, never falls back, and is clearly faster (a silent fallback would time like eager)."""
    import time

    from moe_llm.layers import attnres
    torch.manual_seed(0)
    attnres.set_attnres_compile(True)
    d = 1024
    w = (0.02 * torch.randn(d, device="cuda")).requires_grad_()
    g = (1 + 0.1 * torch.randn(d, device="cuda")).requires_grad_()
    for n in range(2, 10):
        srcs = [torch.randn(2, 512, d, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(n)]
        ref = attnres._depth_attn(w, g, 1e-5, *srcs)
        ref_grads = torch.autograd.grad(ref.float().square().sum(), [w, g, *srcs])
        out = attnres.depth_attn(w, g, 1e-5, *srcs)
        grads = torch.autograd.grad(out.float().square().sum(), [w, g, *srcs])
        assert rel_rms(out, ref) < 1e-2, n
        for a, b in zip(grads, ref_grads):
            assert rel_rms(a, b) < 2e-2, n
        with torch.no_grad():
            assert rel_rms(attnres.depth_attn(w, g, 1e-5, *srcs), ref) < 1e-2, n
    assert not attnres._FAILED

    srcs = [torch.randn(2, 4096, d, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(9)]

    def fwd_bwd_ms(fn, reps=10):
        for i in range(reps + 2):
            if i == 2:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            fn(w, g, 1e-5, *srcs).float().sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    t_eager, t_comp = fwd_bwd_ms(attnres._depth_attn), fwd_bwd_ms(attnres.depth_attn)
    print(f"AttnRes 9 sources [2, 4096, {d}] fwd+bwd: eager {t_eager:.2f} ms, compiled {t_comp:.2f} ms")
    assert t_comp < 0.7 * t_eager, (t_eager, t_comp)


def test_fused_helpers_match_eager_gpu():
    """Every fused helper at the main model's sizes in bf16: same values (bit-identical for almost every
    element) and gradients as eager; prints fwd+bwd time eager vs fused."""
    import math
    import time

    from moe_llm.layers import common
    from moe_llm.layers.engram import _engram_gate
    torch.manual_seed(0)
    bf, dev = torch.bfloat16, "cuda"

    def r(*s, scale=1.0, dtype=bf):
        return (scale * torch.randn(*s, device=dev)).to(dtype)

    cases = [
        ("rms_norm", common._rms_norm, lambda: [r(2, 4096, 1024), 1 + r(1024, scale=0.1, dtype=torch.float32), 1e-6]),
        ("add_rms_norm", common.add_rms_norm,
         lambda: [r(2, 4096, 1024), r(2, 4096, 1024), 1 + r(1024, scale=0.1, dtype=torch.float32), 1e-6]),
        ("q_norm", common._rms_norm, lambda: [r(2, 4096, 8, 128), 1 + r(128, scale=0.1, dtype=torch.float32), 1e-6]),
        ("gated_rms_norm", common.gated_rms_norm,
         lambda: [r(2, 4096, 8, 128), r(2, 4096, 8, 128), 1 + r(128, scale=0.1, dtype=torch.float32), 1e-5]),
        ("sigmoid_mul", common.sigmoid_mul, lambda: [r(2, 4096, 8, 128), r(2, 4096, 8, 128)]),
        ("silu_mul shared", common.silu_mul, lambda: [r(8192, 768), r(8192, 768)]),
        ("silu_mul experts", common.silu_mul, lambda: [r(32, 1536, 512), r(32, 1536, 512)]),
        ("conv_silu kda", common.conv_silu, lambda: [r(2, 4096, 1024), r(1024, 1, 4, scale=0.3, dtype=torch.float32)]),
        ("conv_silu engram", common.conv_silu,
         lambda: [r(2, 4096, 1024), r(1024, 1, 4, scale=0.3, dtype=torch.float32), 3, r(2, 4096, 1024), bf]),
        ("engram_gate", _engram_gate, lambda: [r(2, 4096, 1024), r(2, 4096, 1024), r(2, 4096, 1024), math.sqrt(1024), True]),
    ]

    def run(fn, args, reps=0):
        args = [a.detach().clone().requires_grad_(a.is_floating_point()) if isinstance(a, torch.Tensor) else a
                for a in args]
        for i in range(reps + 1):
            if i == 1:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            out = fn(*args)
            outs = out if isinstance(out, tuple) else (out,)
            sum(o.float().square().mean() for o in outs).backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1e3 if reps else 0.0
        return outs, [a.grad for a in args if isinstance(a, torch.Tensor) and a.grad is not None], ms

    common.set_fuse(True)
    try:
        for name, fn, make in cases:
            args = make()
            ref, ref_g, t_eager = run(fn.eager, args, reps=10)
            new, new_g, t_fused = run(fn, args, reps=10)
            for a, b in zip(new, ref):
                assert a.dtype == b.dtype and rel_rms(a, b) < 1e-2, name
                assert (a == b).float().mean() > 0.95, (name, (a == b).float().mean())
            for a, b in zip(new_g, ref_g):
                assert rel_rms(a, b) < 2e-2, name
            print(f"{name:18s} fwd+bwd eager {t_eager:6.2f} ms  fused {t_fused:6.2f} ms")
        assert not common._FAILED
    finally:
        common.set_fuse(False)


def test_fusion_train_step_matches_eager_gpu():
    """One training step of a small model on the real GPU path (bf16 autocast, bf16 residual stream, fla
    KDA, per-layer recompute): same loss and gradients with fusion off and on, up to bf16 rounding."""
    from moe_llm.layers import common
    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True)
    m_off = MoELLM(cfg).cuda()
    m_on = MoELLM(cfg).cuda()
    m_on.load_state_dict(m_off.state_dict())
    x = torch.randint(0, cfg.vocab_size, (2, 257), device="cuda")

    def step(m, on):
        common.set_fuse(on)
        m.grad_checkpointing = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = m(x[:, :-1], x[:, 1:])
        loss.backward()
        return float(loss)

    try:
        l_off, l_on = step(m_off, False), step(m_on, True)
    finally:
        common.set_fuse(False)
    assert abs(l_on - l_off) < 2e-3 * l_off, (l_on, l_off)
    for (n, a), b in zip(m_off.named_parameters(), m_on.parameters()):
        if a.grad is not None:
            assert rel_rms(b.grad, a.grad) < 5e-2, n
    assert not common._FAILED


def _gpu_states(opt):
    return {(id(o), id(p), k): v for o in opt.optimizers() for p, st in o.state.items() for k, v in st.items()
            if torch.is_tensor(v)}


def test_optimizer_offload_is_exact_and_frees_memory():
    """Offloading the optimizer states (pinned CPU between steps) frees their GPU memory, round-trips them
    bit for bit, gives bit-identical parameter updates (same gradients, deterministic optimizer kernels)
    and survives a checkpoint reload with the bf16 Muon momentum intact."""
    import copy
    import io

    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True)
    m_a = MoELLM(cfg).cuda()
    m_b = copy.deepcopy(m_a)
    o_a = OptimizerBundle(m_a, lr=1e-3, muon_state_dtype=torch.bfloat16, offload_states=True)
    o_b = OptimizerBundle(m_b, lr=1e-3, muon_state_dtype=torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (2, 129), device="cuda")
    for step in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = m_a(x[:, :-1], x[:, 1:])
        loss.backward()
        for pa, pb in zip(m_a.parameters(), m_b.parameters()):  # identical gradients for both optimizers
            pb.grad = None if pa.grad is None else pa.grad.clone()
        o_a.step(1e-3)
        o_b.step(1e-3)
        o_a.zero_grad()
        o_b.zero_grad()
        for (n, pa), pb in zip(m_a.named_parameters(), m_b.parameters()):
            assert torch.equal(pa, pb), (step, n)
    states = _gpu_states(o_a)
    assert states and not any(v.is_cuda for v in states.values())          # all off the GPU between steps
    before = torch.cuda.memory_allocated()
    snap = {k: v.clone() for k, v in states.items()}
    o_a.reload()
    moved = sum(v.numel() * v.element_size() for v in _gpu_states(o_a).values())
    assert torch.cuda.memory_allocated() - before >= 0.9 * moved           # reload really re-occupies the GPU
    assert all(torch.equal(v.cpu(), snap[k]) for k, v in _gpu_states(o_a).items())
    o_a.offload()
    torch.cuda.synchronize()
    assert all(torch.equal(v, snap[k]) for k, v in _gpu_states(o_a).items())
    buf = io.BytesIO()
    torch.save(o_a.state_dict(), buf)
    buf.seek(0)
    o_c = OptimizerBundle(copy.deepcopy(m_a), lr=1e-3, muon_state_dtype=torch.bfloat16, offload_states=True)
    o_c.load_state_dict(torch.load(buf, weights_only=False))
    torch.cuda.synchronize()
    loaded = list(_gpu_states(o_c).values())
    assert loaded and not any(v.is_cuda for v in loaded)
    assert all(st["buf"].dtype == torch.bfloat16 for st in o_c.muon.state.values())
    assert sorted(torch.cat([v.float().flatten() for v in loaded]).tolist()) == \
        sorted(torch.cat([v.float().flatten() for v in snap.values()]).tolist())


@pytest.mark.parametrize("fuse", [False, True])
def test_expert_recompute_exact_and_bounded(fuse):
    """Main-size MoE layer (bf16 autocast). Recomputing the routed experts in backward gives bit-identical
    values and gradients for the recomputed part, and the memory kept for backward no longer grows when the
    routing collapses onto k experts (bmm pads every expert to the busiest one). The rest of the layer
    (routing, gather, index_add_ combine) is the same code either way."""
    from moe_llm.config import main_2b
    from moe_llm.layers import common
    cfg = main_2b()
    torch.manual_seed(0)
    moe = MoE(cfg, 1).cuda()
    x0 = torch.randn(2, 4096, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    common.set_fuse(fuse)
    try:
        # 1) the recomputed part on its own: bit-identical with and without checkpointing
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            idx = moe.route(x0.view(-1, cfg.d_model))[1]
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        e_sorted, counts = flat[order], torch.bincount(flat, minlength=moe.E)
        xs0 = x0.view(-1, cfg.d_model).index_select(0, order // moe.k)
        dy = torch.randn(xs0.shape, device="cuda", dtype=torch.bfloat16)
        outs = []
        for ck in (False, True):
            moe.zero_grad(set_to_none=True)
            xs = xs0.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ys = (checkpoint(moe._routed, xs, e_sorted, counts, use_reentrant=False, preserve_rng_state=False) if ck
                      else moe._routed(xs, e_sorted, counts))
            ys.backward(dy)
            outs.append([ys.detach(), xs.grad, moe.w_gate.grad, moe.w_up.grad, moe.w_down.grad])
        for a, b in zip(*outs):
            assert torch.equal(a, b)

        # 2) the whole layer: memory kept for backward, balanced vs collapsed routing
        def kept(recompute, collapse):
            moe.recompute_experts = recompute
            moe.zero_grad(set_to_none=True)
            with torch.no_grad():
                moe.e_bias.zero_()
                if collapse:
                    moe.e_bias[:cfg.top_k] = 10.0  # every token picks the same k experts: padded to all tokens
            x = x0.clone().requires_grad_()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = moe(x)
            held = torch.cuda.memory_allocated() - base
            (y.float().square().mean() + aux).backward()
            assert torch.isfinite(x.grad).all()
            return held

        gb = 2 ** 30
        k_bal, k_col = kept(False, False), kept(False, True)
        r_bal, r_col = kept(True, False), kept(True, True)
        print(f"kept for backward (fuse {fuse}): balanced {k_bal / gb:.2f} -> {r_bal / gb:.2f} GB with expert "
              f"recompute, collapsed {k_col / gb:.2f} -> {r_col / gb:.2f} GB")
        assert k_col > k_bal + 0.5 * gb            # without: grows with the imbalance
        assert abs(r_col - r_bal) < 0.02 * gb      # with: independent of the routing
        assert r_bal < k_bal
    finally:
        moe.recompute_experts = False
        common.set_fuse(False)


def test_fast_weight_grad_accumulation_exact_gpu():
    """CastLinear layers (with bias) and the MoE expert matmuls at main sizes under bf16 autocast: gradients
    accumulated over 3 micro-batches are bit-identical with the fused accumulation on and off. Only
    deterministic ops here (the full model has atomics elsewhere; it is compared on CPU in
    tests/test_fast_accum.py). Prints the time of both paths."""
    import time

    from moe_llm.config import main_2b
    from moe_llm.layers import common
    from moe_llm.layers.common import CastLinear
    torch.manual_seed(0)
    cfg = main_2b()
    lin = torch.nn.Sequential(CastLinear(1024, 3072), CastLinear(3072, 1024)).cuda()
    moe = MoE(cfg, 1).cuda()
    xs = [torch.randn(8192, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        idx = moe.route(xs[0])[1]
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    e_sorted, counts, tok = flat[order], torch.bincount(flat, minlength=moe.E), order // moe.k
    params = list(lin.parameters()) + [moe.w_gate, moe.w_up, moe.w_down]

    def run(fast):
        common.set_fast_accum(fast)
        for p in params:
            p.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for x in xs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = lin(x)
                ys = moe._routed(x.index_select(0, tok), e_sorted, counts)
            (y.float().square().mean() + ys.float().square().mean()).backward()
        torch.cuda.synchronize()
        return [p.grad.clone() for p in params], time.perf_counter() - t0

    try:
        run(True)
        run(False)  # warm-up
        g_fast, t_fast = run(True)
        g_ref, t_ref = run(False)
    finally:
        common.set_fast_accum(True)
    for a, b in zip(g_fast, g_ref):
        assert a.dtype == torch.float32 and torch.equal(a, b)
    print(f"3 micro-batches fwd+bwd: reference {t_ref * 1e3:.1f} ms, fused accumulation {t_fast * 1e3:.1f} ms")


def test_offload_master_exact_and_frees_memory_gpu():
    """fp32 masters in pinned CPU memory between steps, persistent bf16 copies on the GPU: with the same
    gradients the clipped norm and every parameter update are bit-identical to the plain bundle; the masters
    really leave the GPU (memory drops by their fp32 size minus the bf16 copies); the copies stay the bf16
    rounding of the masters; a checkpoint round trip gives the same weights."""
    import copy
    import io

    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True)
    m_a = MoELLM(cfg).cuda()
    m_b = copy.deepcopy(m_a)
    o_a = OptimizerBundle(m_a, lr=1e-3, muon_state_dtype=torch.bfloat16, offload_states=True, offload_master=True)
    o_b = OptimizerBundle(m_b, lr=1e-3, muon_state_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    o_a.begin()
    torch.cuda.synchronize()
    cast = m_a.cast_params()
    assert cast and all(p.device.type == "cpu" for p in cast)
    saved = sum(p.numel() for p in cast) * (4 - 2)
    assert before - torch.cuda.memory_allocated() >= 0.9 * saved
    x = torch.randint(0, cfg.vocab_size, (2, 129), device="cuda")
    try:
        for step in range(3):
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = m_a(x[:, :-1], x[:, 1:])
                (loss / 2).backward()
            for pa, pb in zip(m_a.parameters(), m_b.parameters()):  # the same gradients for the plain bundle
                g = o_a._gbuf.get(id(pa)) if id(pa) in o_a._cast else pa.grad
                pb.grad = None if g is None else g.clone()
            ga = o_a.clip_grad_norm(0.05)
            gb = torch.nn.utils.clip_grad_norm_([p for p in m_b.parameters() if p.grad is not None], 0.05)
            assert torch.equal(ga, gb)
            o_a.step(1e-3)
            o_b.step(1e-3)
            o_a.zero_grad()
            o_b.zero_grad()
            torch.cuda.synchronize()
            for (n, pa), pb in zip(m_a.named_parameters(), m_b.parameters()):
                assert torch.equal(pa.detach().cpu(), pb.detach().cpu()), (step, n)
            for p, c in o_a._cast.values():
                assert p.device.type == "cpu" and torch.equal(c.cpu(), p.detach().to(torch.bfloat16))
        o_a.sync()
        buf = io.BytesIO()
        torch.save({"model": m_a.state_dict(), "opt": o_a.state_dict()}, buf)
        buf.seek(0)
        sd = torch.load(buf, weights_only=False)
        m_c = MoELLM(cfg).cuda()
        m_c.load_state_dict(sd["model"])
        o_c = OptimizerBundle(m_c, lr=1e-3, muon_state_dtype=torch.bfloat16, offload_states=True, offload_master=True)
        o_c.load_state_dict(sd["opt"])
        o_c.begin()
        torch.cuda.synchronize()
        for (n, pa), pc in zip(m_a.named_parameters(), m_c.parameters()):
            assert torch.equal(pa.detach().cpu(), pc.detach().cpu()), n
    finally:
        from moe_llm.layers import common
        common.clear_cached_copies()


def test_new_exact_paths_train_together_gpu():
    """fp32 masters + optimizer states offloaded, FFN-only recompute and the threaded Engram update over
    offloaded tables, together on the GPU (tiny model, bf16, prefetch as in train.py): the losses track the
    plain configuration. GPU atomics rule out bit-exactness here; the exact comparisons are on CPU in
    tests/test_speedups2.py."""
    import copy

    from moe_llm.layers import common
    torch.manual_seed(0)
    cfg = tiny(kda_heads=2, kda_head_dim=64, d_model=128, residual_bf16=True, residual="block_attnres",
               engram_offload=True)
    base = MoELLM(cfg).cuda()
    runs = []
    try:
        for new in (False, True):
            m = copy.deepcopy(base)
            if new:  # FFN-only recompute and activation offloading together
                m.set_checkpointing(1.0, ffn_frac=0.5, offload_frac=0.5)
                m.offloader.min_bytes = 0
            else:
                m.set_checkpointing(0.5)
            opt = OptimizerBundle(m, lr=1e-3, muon_state_dtype=torch.bfloat16, offload_states=True,
                                  offload_master=new, async_engram=new)
            opt.begin()
            ces = []
            for s in range(4):
                for k in range(2):
                    xc = torch.randint(0, cfg.vocab_size, (2, 129), generator=torch.Generator().manual_seed(10 * s + k))
                    m.engram.prefetch(xc[:, :-1])
                    x = xc.cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, info = m(x[:, :-1], x[:, 1:])
                    (loss / 2).backward()
                    ces.append(float(info["ce"]))
                opt.clip_grad_norm(1.0)
                opt.step(1e-3)
                opt.zero_grad()
                m.update_router_bias()
            m.engram.flush()
            runs.append(ces)
    finally:
        common.clear_cached_copies()
    for a, b in zip(*runs):
        assert abs(a - b) < 2e-2 * abs(a), runs
    assert m.offloader is not None and m.offloader.stats["tensors"] > 0 and m.offloader.stats["late"] == 0


def test_moe_grouped_mm_backward_and_speed_gpu():
    """The no-padding path (torch._grouped_mm, expert offsets stay on the GPU) against the padded bmm path at
    main sizes: same forward and gradients up to bf16 rounding; prints fwd+bwd time of both with balanced and
    with skewed routing (bmm pads every expert to the busiest one)."""
    import time

    from moe_llm.config import main_2b
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm missing")
    torch.manual_seed(0)
    cfg = main_2b()
    moe = MoE(cfg, 1).cuda()
    x0 = torch.randn(2, 4096, cfg.d_model, device="cuda", dtype=torch.bfloat16)

    def run(impl, reps=0):
        moe.impl = impl
        for i in range(reps + 1):
            if i == 1:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            moe.zero_grad(set_to_none=True)
            x = x0.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = moe(x)
            (y.float().square().mean() + aux).backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1e3 if reps else 0.0
        return y.detach(), [x.grad, moe.w_gate.grad, moe.w_up.grad, moe.w_down.grad, moe.router.weight.grad], ms

    try:
        for skew in (0.0, 0.3):
            with torch.no_grad():
                moe.e_bias.zero_()
                moe.e_bias[:4] += skew  # skewed: a few experts get far more tokens
            ya, ga, ta = run("bmm", reps=10)
            try:
                yb, gb, tb = run("grouped", reps=10)
            except Exception as e:
                pytest.skip(f"grouped_mm unsupported here: {e}")
            assert rel_rms(yb, ya) < 1e-2
            for a, b in zip(ga, gb):
                assert rel_rms(b, a) < 3e-2
            print(f"MoE layer fwd+bwd, routing skew {skew}: bmm {ta:.2f} ms, grouped_mm {tb:.2f} ms")
    finally:
        moe.impl = cfg.moe_impl


def test_activation_offload_exact_and_frees_memory_gpu():
    """CUDA path of moe_llm/offload.py on deterministic layers (norm, linear, SwiGLU under bf16 autocast): the
    saved tensors leave the GPU during the forward (memory held drops by about what was offloaded), come back on
    the side stream after prefetch() without late fetches, the gradients are bit-identical to keeping them, and
    the pinned host buffers are reused across passes."""
    from moe_llm.layers.common import CastLinear, RMSNorm, SwiGLU
    from moe_llm.offload import ActivationOffloader
    torch.manual_seed(0)
    layers = torch.nn.ModuleList([torch.nn.Sequential(RMSNorm(1024), CastLinear(1024, 1024),
                                                      SwiGLU(1024, 2048, 0.02, 0.02)) for _ in range(4)]).cuda()
    x0 = torch.randn(8192, 1024, device="cuda", dtype=torch.bfloat16)

    def run(off):
        for p in layers.parameters():
            p.grad = None
        x = x0.clone().requires_grad_()
        h = x
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i, layer in enumerate(layers):
                if off is not None and i < 3:
                    with off.layer(i):
                        h = layer(h)
                else:
                    h = layer(h)
        torch.cuda.synchronize()
        torch.empty(1, device="cuda")  # let the allocator retire blocks whose copies have finished
        held = torch.cuda.memory_allocated() - m0
        if off is not None:
            for i in (2, 1, 0):
                off.prefetch(i)
        h.float().square().mean().backward()
        torch.cuda.synchronize()
        return held, [x.grad.clone()] + [p.grad.clone() for p in layers.parameters()]

    held_ref, g_ref = run(None)
    off = ActivationOffloader()
    held_off, g_off = run(off)
    host, per_pass = off.stats["host_bytes"], off.stats["bytes"]
    _, g_off2 = run(off)
    assert off.stats["late"] == 0 and off.stats["host_bytes"] == host and per_pass > 0
    for a, b in zip(g_ref, g_off):
        assert torch.equal(a, b)
    for a, b in zip(g_ref, g_off2):
        assert torch.equal(a, b)
    print(f"held after the forward: {held_ref / 2 ** 30:.2f} GB kept vs {held_off / 2 ** 30:.2f} GB offloaded "
          f"({per_pass / 2 ** 30:.2f} GB moved per pass, host buffers {host / 2 ** 30:.2f} GB)")
    assert held_off < held_ref - 0.5 * per_pass


# ----------------------------------------------------------------------------- host-sync-free hot path (switch 4)
def _count_sync_warnings(fn):
    """Run fn() under torch.cuda.set_sync_debug_mode("warn") and count "synchronizing CUDA operation" warnings
    (from any thread: the autograd engine replays its warnings, the expert recompute runs Python in its thread)."""
    import warnings
    hits, prev = [], warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchronizing CUDA operation" in str(message):
            hits.append(f"{filename}:{lineno}")
        else:
            prev(message, category, filename, lineno, file, line)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = show
        torch.cuda.set_sync_debug_mode("warn")
        try:
            out = fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return out, hits


def test_moe_has_no_implicit_host_syncs_gpu():
    """MoE forward + backward on CUDA, expert recompute on and off: the backward never waits for the host, the
    forward only for the capacity read-back (an explicit event wait). The old code synced 5 times per forward
    (twice per torch.bincount, once for int(counts.max())) and once more in the expert recompute."""
    torch.manual_seed(0)
    moe = MoE(tiny(), 1).cuda().train()
    x = torch.randn(2, 256, moe.d, device="cuda", requires_grad=True)
    ev = torch.cuda.Event()
    ev.record()
    _, ev_hits = _count_sync_warnings(ev.synchronize)  # does this torch flag explicit event waits? (0 or 1)
    for recompute in (True, False):
        moe.recompute_experts = recompute
        for _ in range(2):  # warm-up (pinned scalar blocks, kernels)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = moe(x)
            (y.float().square().mean() + aux).backward()
        torch.cuda.synchronize()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (y, aux), fwd = _count_sync_warnings(lambda: moe(x))
        loss = y.float().square().mean() + aux
        _, bwd = _count_sync_warnings(loss.backward)
        torch.cuda.synchronize()
        print(f"recompute={recompute}: syncs forward {len(fwd)}, backward {len(bwd)} "
              f"(explicit event wait flagged: {bool(ev_hits)})")
        assert not bwd, bwd
        assert len(fwd) == len(ev_hits), fwd  # only the capacity read-back, and only if event waits are flagged


def test_capacity_readback_waits_for_the_value_gpu():
    """The read-back must wait for its own copy: with the GPU busy (a large matmul before the counts), an
    unsynchronized read of the pinned scalar would see a stale or uninitialized value."""
    from moe_llm.layers.moe import MaxReadback
    a = torch.randn(4096, 4096, device="cuda")
    for i in range(20):
        b = a @ a
        t = (b[0, :64].abs() * 1000).long() + i
        assert MaxReadback(t).get() == int(t.max())


def test_engram_rows_and_inverse_are_pinned_gpu():
    """Both host tensors that lookup() copies to the GPU are pinned, so the copies are asynchronous."""
    from moe_llm.layers.engram import EngramMemory
    mem = EngramMemory(tiny(engram_offload=True)).cuda()  # hash buffers on the GPU, tables stay on the CPU
    idx = torch.randint(0, mem.rows[0], (2, 24, 4))
    uniq, inv, rows = mem.cpu_tables[0].gather(idx)
    assert rows.is_pinned() and inv.is_pinned()
    ids = torch.randint(0, 512, (2, 24))
    with torch.no_grad():
        direct = {k: v.clone() for k, v in mem(ids.cuda()).items()}
        mem.prefetch(ids)
        pre = mem(ids.cuda())
    assert all(torch.equal(direct[k], pre[k]) for k in direct)
