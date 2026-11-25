"""ce_impl="fused" / "fused_bf16" (moe_llm/loss_fused.py) against ce_impl="chunked" (moe_llm/loss.py, the current
code) and an fp64 reference computed on the same GEMM operands (h and W cast to the GEMM dtype).

CPU: the Triton row kernel runs in the interpreter (TRITON_INTERPRET=1 is set below when there is no GPU; it must be
set before the kernel module is imported) at small sizes. torch.mm has no out_dtype on CPU, so the fused path's
"bf16 operands, fp32 output" GEMMs are fp32 GEMMs of the bf16 values there: the same sums cuBLAS computes.
With a GPU the same tests run on it, plus main-size checks (N 8192, d 1024, V 65536): accuracy against an fp64
reference, bit-identical reruns, no CPU-GPU sync, the addmm out-aliasing, peak memory, every kernel config, a model
step under bf16 autocast, and timings (pytest -s to see the tables)."""
import copy
import os

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from moe_llm import loss_fused as lf  # noqa: E402
from moe_llm.config import ModelConfig, tiny  # noqa: E402
from moe_llm.loss import linear_cross_entropy  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
IMPLS = ("chunked", "fused", "fused_bf16")


# ----------------------------------------------------------------------------- helpers
def gemm_dtype(h):
    """The GEMM dtype the loss uses for h (both paths): bf16/fp16 as is; fp32 -> bf16 on CUDA, fp32 on CPU."""
    if h.dtype in (torch.bfloat16, torch.float16):
        return h.dtype
    return torch.bfloat16 if h.is_cuda else torch.float32


def make_case(N=96, d=48, V=1000, seed=0, logit_std=4.0, n_ignore=5, h_dtype=torch.bfloat16, device=DEV):
    """RMS-normalised hidden states (like final_norm's output), a head whose logits have std ~logit_std (a trained
    model's scale, where bf16 rounding of the logits matters), targets sampled from the model's own softmax."""
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(N, d, generator=gen)
    h = (h / h.pow(2).mean(-1, keepdim=True).sqrt()).to(h_dtype)
    W = torch.randn(V, d, generator=gen) * (logit_std / d ** 0.5)
    p = torch.softmax(h.float() @ W.t(), -1)
    t = torch.multinomial(p, 1, generator=gen).squeeze(1)
    if n_ignore:
        t[torch.randperm(N, generator=gen)[:n_ignore]] = -100
    return h.to(device), W.to(device), t.to(device)


def reference(h, W, t, z, ignore_index=-100):
    """fp64 loss, CE and gradients of the same function on the GEMM operands (h, W cast to the GEMM dtype)."""
    cdt = gemm_dtype(h)
    h64 = h.detach().to(cdt).double().requires_grad_()
    W64 = W.detach().to(cdt).double().requires_grad_()
    logits = h64 @ W64.t()
    valid = t != ignore_index
    ce = F.cross_entropy(logits, t, ignore_index=ignore_index)
    lse = torch.logsumexp(logits, -1)
    loss = ce + z * (lse.pow(2) * valid).sum() / valid.sum().clamp_min(1)
    loss.backward()
    return loss.detach(), ce.detach(), h64.grad, W64.grad


def run(impl, h, W, t, z=1e-4, chunk=32, ignore_index=-100):
    hh, WW = h.detach().clone().requires_grad_(), W.detach().clone().requires_grad_()
    loss, ce = linear_cross_entropy(hh, WW, t, z_coef=z, chunk=chunk, ignore_index=ignore_index, impl=impl)
    loss.backward()
    return loss.detach(), ce, hh.grad, WW.grad


def rel_err(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def errors(impl, h, W, t, z, chunk, ref):
    loss, ce, gh, gw = run(impl, h, W, t, z, chunk)
    l_ref, ce_ref, gh_ref, gw_ref = ref
    return {"loss": abs(loss.item() - l_ref.item()), "ce": abs(ce.item() - ce_ref.item()),
            "grad_h": rel_err(gh, gh_ref), "grad_W": rel_err(gw, gw_ref)}


def bf16_ulp_diff(a, b):
    """Elementwise distance in bf16 ulps (same-sign finite values; 0 = identical)."""
    ia = a.to(torch.bfloat16).view(torch.int16).int()
    ib = b.to(torch.bfloat16).view(torch.int16).int()
    return (ia - ib).abs()


# ----------------------------------------------------------------------------- the row kernel
@pytest.mark.parametrize("ldt", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gdt", [torch.bfloat16, torch.float32])
def test_row_kernel_matches_formula(ldt, gdt):
    """lse, target logit and g = (softmax (1 + 2 z lse) - onehot) * scale against the chunked path's formula in torch
    on the same logits. V = 1000 is not a multiple of the block (masked tail); rows with one dominant logit, constant
    rows, targets at the first and last column, zero scale (ignored rows)."""
    if gdt == torch.float32 and ldt == torch.bfloat16:
        pytest.skip("fp32 g comes only with fp32 logits")
    gen = torch.Generator().manual_seed(1)
    R, V, z = 40, 1000, 1e-2
    x = (torch.randn(R, V, generator=gen) * 5).to(ldt)
    x[1, 17] = 30.0       # softmax ~ one-hot
    x[2] = 3.0            # constant row: p = 1/V
    x[3, :] = -30.0       # all very negative
    x[3, 999] = -29.0
    tgt = torch.randint(0, V, (R,), generator=gen)
    tgt[0], tgt[1], tgt[2], tgt[3] = 0, 17, V - 1, 999
    scale = torch.full((R,), 1.0 / 37)
    scale[[5, 9]] = 0.0
    x, tgt, scale = x.to(DEV), tgt.to(DEV), scale.to(DEV)
    lse, xt = torch.empty(R, device=DEV), torch.empty(R, device=DEV)
    g = torch.empty(R, V, dtype=gdt, device=DEV)
    lf.ce_rows(x, tgt, scale, z, g, lse, xt)

    xf = x.float()
    lse_ref = torch.logsumexp(xf.double(), -1)
    assert (lse.double() - lse_ref).abs().max().item() < 2e-6 * max(1.0, lse_ref.abs().max().item())
    assert torch.equal(xt, xf.gather(1, tgt[:, None]).squeeze(1))
    # the chunked path's formula in torch (fp32), on the same logits
    lse32 = torch.logsumexp(xf, -1)
    ref = torch.exp(xf - lse32[:, None]) * ((1.0 + 2.0 * z * lse32) * scale)[:, None]
    ref[torch.arange(R, device=DEV), tgt] -= scale
    tiny_ = ref.abs() < 1e-30  # fp32 denormal range: the GPU's exp2 flushes these to zero, torch keeps them
    assert (g.float()[tiny_].abs() < 1e-30).all()
    if gdt == torch.bfloat16:
        d = bf16_ulp_diff(g.float(), ref)[~tiny_]
        assert d.max().item() <= 1 and (d == 0).float().mean().item() > 0.99
    else:  # a few fp32 ulps (lse rounded differently); relative to the row's scale where the target term cancels
        assert ((g - ref).abs() <= 4e-6 * (ref.abs() + scale[:, None]) + 1e-30).all()
    assert (g[[5, 9]] == 0).all()                          # zero scale: exactly zero gradient rows
    assert abs(g[2, 0].item() - (1 + 2 * z * lse_ref[2].item()) / (V * 37)) < 1e-6  # constant row: p = 1/V
    assert g[0, 0].item() < 0 and g[4, tgt[4]].item() < 0  # the target's p (1 + 2 z lse) < 1 there

    # loss only (WRITE_G=False) gives the same lse / target logits; bf16 logits: g in place gives the same g
    lse2, xt2 = torch.empty_like(lse), torch.empty_like(xt)
    lf.ce_rows(x, tgt, scale, z, None, lse2, xt2)
    assert torch.equal(lse2, lse) and torch.equal(xt2, xt)
    if ldt == gdt:
        xi = x.clone()
        lf.ce_rows(xi, tgt, scale, z, xi, lse2, xt2)
        assert torch.equal(xi, g) and torch.equal(lse2, lse)


def test_block_size_changes_only_rounding():
    """Other (BLOCK_V, num_warps): the same lse up to fp32 rounding, g up to one bf16 ulp."""
    gen = torch.Generator().manual_seed(2)
    R, V = 24, (1000 if DEV == "cpu" else 20000)  # several blocks per row for every config
    x = (torch.randn(R, V, generator=gen) * 6).to(DEV)
    tgt = torch.randint(0, V, (R,), generator=gen).to(DEV)
    scale = torch.full((R,), 0.04, device=DEV)
    res = []
    cfgs = [(64, 4), (256, 4), (1024, 4)] if DEV == "cpu" else [(1024, 4), (4096, 16), (8192, 32)]
    try:
        for c in cfgs + [None]:
            if c is None:  # the default config without the cache hints: identical values
                lf.reset_config()
                lf.set_hints(False)
            else:
                lf.set_config(*c)
            lse, xt = torch.empty(R, device=DEV), torch.empty(R, device=DEV)
            g = torch.empty(R, V, dtype=torch.bfloat16, device=DEV)
            lf.ce_rows(x, tgt, scale, 1e-4, g, lse, xt)
            res.append((lse, g))
        lf.set_hints(True)
        lse, xt = torch.empty(R, device=DEV), torch.empty(R, device=DEV)
        g = torch.empty(R, V, dtype=torch.bfloat16, device=DEV)
        lf.ce_rows(x, tgt, scale, 1e-4, g, lse, xt)
        assert torch.equal(lse, res[-1][0]) and torch.equal(g, res[-1][1])
    finally:
        lf.reset_config()
        lf.set_hints(True)
    for lse, g in res[1:]:
        assert (lse - res[0][0]).abs().max().item() < 2e-6 * res[0][0].abs().max().item()
        assert bf16_ulp_diff(g.float(), res[0][1].float()).max().item() <= 1


# ----------------------------------------------------------------------------- the whole loss vs fp64
@pytest.mark.parametrize("z", [0.0, 1e-4, 1e-2])
@pytest.mark.parametrize("chunk", [32, 96])
def test_no_less_accurate_than_chunked(z, chunk):
    """Loss, CE, grad_h, grad_W of each path against fp64 (bf16 hidden states, fp32 head weight, ignored rows,
    z-loss). "fused" keeps the logits in fp32: much closer for the loss, closer for both gradients. "fused_bf16" has
    the chunked path's logits: the same accuracy, and grad_W at least as close (fp32 accumulation)."""
    h, W, t = make_case(seed=3)
    ref = reference(h, W, t, z)
    e = {impl: errors(impl, h, W, t, z, chunk, ref) for impl in IMPLS}
    print({k: {q: f"{v:.2e}" for q, v in d.items()} for k, d in e.items()})
    c, f, b = e["chunked"], e["fused"], e["fused_bf16"]
    for q in ("loss", "ce"):
        assert f[q] <= max(c[q] / 10, 1e-5), (q, e)
        assert b[q] <= 1.1 * c[q] + 1e-5, (q, e)
    for q in ("grad_h", "grad_W"):
        assert f[q] <= 0.9 * c[q], (q, e)
        assert b[q] <= 1.05 * c[q] + 1e-6, (q, e)
    assert b["grad_W"] <= c["grad_W"] + 1e-6, e


def test_fp32_hidden_states_match_chunked_closely():
    """fp32 h on CPU: both paths use fp32 GEMMs and fp32 g (nothing rounded to bf16): the same values up to fp32
    rounding, both at fp32 accuracy against fp64."""
    if DEV == "cuda":
        pytest.skip("on CUDA fp32 h goes through bf16 GEMMs (covered by the bf16 tests)")
    h, W, t = make_case(seed=4, h_dtype=torch.float32)
    ref = reference(h, W, t, 1e-4)
    for impl in ("fused", "fused_bf16"):
        e = errors(impl, h, W, t, 1e-4, 32, ref)
        assert e["loss"] < 1e-5 and e["grad_h"] < 1e-5 and e["grad_W"] < 1e-5, (impl, e)
    a, b = run("chunked", h, W, t), run("fused", h, W, t)
    assert abs(a[0].item() - b[0].item()) < 1e-5 and rel_rms(b[2], a[2]) < 1e-5 and rel_rms(b[3], a[3]) < 1e-5


@pytest.mark.parametrize("impl", ["fused", "fused_bf16"])
def test_ignore_index(impl):
    """Ignored rows: exactly zero grad_h rows; the loss averages over the other rows only; a non-negative
    ignore_index (a real class id) works; all rows ignored gives loss 0 and zero gradients, without NaN."""
    h, W, t = make_case(seed=5, n_ignore=9)
    valid = t != -100
    loss, ce, gh, gw = run(impl, h, W, t)
    assert (gh[~valid] == 0).all() and (gh[valid] != 0).any()
    l_ref, ce_ref, _, _ = reference(h, W, t, 1e-4)
    assert abs(ce.item() - ce_ref.item()) < 2e-3 * ce_ref.item()

    t7 = t.clone()
    t7[~valid] = 7
    t7[0] = 7
    loss7, ce7, gh7, gw7 = run(impl, h, W, t7, ignore_index=7)
    r7 = reference(h, W, t7, 1e-4, ignore_index=7)
    assert (gh7[t7 == 7] == 0).all() and abs(ce7.item() - r7[1].item()) < 2e-3 * r7[1].item()

    t_all = torch.full_like(t, -100)
    loss0, ce0, gh0, gw0 = run(impl, h, W, t_all)
    assert loss0.item() == 0.0 and ce0.item() == 0.0
    assert (gh0 == 0).all() and (gw0 == 0).all()


def test_z_loss_term():
    """loss - CE = z * mean(lse^2) over the valid rows; z = 0 gives loss == CE exactly."""
    h, W, t = make_case(seed=6)
    valid = t != -100
    lse = torch.logsumexp(h.float() @ W.to(h.dtype).float().t(), -1)
    want = (lse.pow(2) * valid).sum() / valid.sum()
    loss, ce, _, _ = run("fused", h, W, t, z=1e-2)
    assert abs((loss - ce).item() - 1e-2 * want.item()) < 1e-5 * max(1.0, 1e-2 * want.item())
    loss, ce, _, _ = run("fused", h, W, t, z=0.0)
    assert torch.equal(loss, ce)


def test_chunk_size_changes_only_rounding():
    """Chunk sizes that do and do not divide N: the same values up to fp32 rounding (grad_W summed in a different
    grouping) and rare bf16 flips in grad_h."""
    h, W, t = make_case(seed=7)
    base = run("fused", h, W, t, chunk=96)
    # GPU: cuBLAS picks other kernels for other row counts (another fp32 summation order of the logits), so a few
    # elements of g round the other way
    tol_w = 1e-5 if DEV == "cpu" else 5e-4
    for chunk in (7, 32, 50, 1000):
        got = run("fused", h, W, t, chunk=chunk)
        assert abs(got[0].item() - base[0].item()) < 1e-6 * abs(base[0].item()), chunk
        assert rel_rms(got[2], base[2]) < 5e-3 and rel_rms(got[3], base[3]) < tol_w, chunk


def test_no_grad_and_partial_requires_grad(monkeypatch):
    """Under torch.no_grad only the logits GEMM and pass 1 run (the eval path), with the same loss bits as in
    training. With W frozen only grad_h is computed (equal to the full run's)."""
    h, W, t = make_case(seed=8)
    calls = {"mm": 0, "addmm": 0}
    real_mm, real_addmm = lf._mm, lf._addmm_into
    monkeypatch.setattr(lf, "_mm", lambda *a, **k: (calls.__setitem__("mm", calls["mm"] + 1), real_mm(*a, **k))[1])
    monkeypatch.setattr(lf, "_addmm_into",
                        lambda *a, **k: (calls.__setitem__("addmm", calls["addmm"] + 1), real_addmm(*a, **k))[1])
    full = run("fused", h, W, t, chunk=32)  # 3 chunks: logits, grad_h, grad_W (mm, then addmm)
    assert calls == {"mm": 3 * 2 + 1, "addmm": 2}
    calls.update(mm=0, addmm=0)
    with torch.no_grad():
        loss, ce = linear_cross_entropy(h, W.requires_grad_(), t, z_coef=1e-4, chunk=32, impl="fused")
    assert calls == {"mm": 3, "addmm": 0}
    if DEV == "cpu":
        assert torch.equal(loss, full[0]) and torch.equal(ce, full[1])
    else:  # the loss-only kernel is a separate compilation of the same pass 1
        assert torch.allclose(loss, full[0], rtol=1e-6, atol=0) and torch.allclose(ce, full[1], rtol=1e-6, atol=0)
    calls.update(mm=0, addmm=0)
    hh = h.detach().clone().requires_grad_()
    loss, _ = linear_cross_entropy(hh, W.detach(), t, z_coef=1e-4, chunk=32, impl="fused")
    loss.backward()
    assert calls == {"mm": 6, "addmm": 0} and torch.equal(hh.grad, full[2])


def test_falls_back_to_chunked_without_triton_device(monkeypatch):
    """CPU tensors without the interpreter, and fp64 inputs, run the chunked path (bit-identical to it)."""
    h, W, t = make_case(seed=9)
    want = run("chunked", h, W, t)
    monkeypatch.setattr(lf, "_INTERPRET", False)
    if DEV == "cpu":
        got = run("fused", h, W, t)
        for a, b in zip(got, want):
            assert torch.equal(a, b)
    h64, W64 = h.double(), W.double()
    got, want = run("fused", h64, W64, t), run("chunked", h64, W64, t)
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    with pytest.raises(ValueError):
        linear_cross_entropy(h, W, t, impl="fast")


# ----------------------------------------------------------------------------- model
def test_config_plumbing(monkeypatch):
    """ce_impl / ce_chunk are ModelConfig fields (so --override can switch them); the defaults are the current path;
    old configs without them load with the defaults; an unknown ce_impl fails at construction."""
    assert ModelConfig().ce_impl == "chunked" and ModelConfig().ce_chunk == 2048
    old = {k: v for k, v in tiny().to_dict().items() if k not in ("ce_impl", "ce_chunk")}
    assert ModelConfig.from_dict(old).ce_impl == "chunked"
    m = MoELLM(tiny(ce_impl="fused", ce_chunk=20)).to(DEV)
    assert m.ce_chunk == 20
    seen = []
    real = lf.fused_linear_cross_entropy
    monkeypatch.setattr(lf, "fused_linear_cross_entropy", lambda *a, **k: (seen.append((a[4], k)), real(*a, **k))[1])
    x = torch.randint(0, 512, (2, 25)).to(DEV)
    m(x[:, :-1], x[:, 1:])
    assert seen == [(20, {"fp32_logits": True})]
    with pytest.raises(ValueError):
        MoELLM(tiny(ce_impl="nope"))


@pytest.mark.parametrize("residual", ["standard", "block_attnres"])
def test_model_training_step_close_to_chunked(residual):
    """Whole tiny model (fp32 on CPU), one optimizer step and the next forward: ce_impl="fused" vs "chunked". Same loss
    and gradients up to fp32 rounding (the head computes in fp32 either way here), and the loss after the step."""
    from moe_llm.optim import OptimizerBundle
    torch.manual_seed(0)
    cfg = tiny(residual=residual, seq_aux_coef=0.0)
    base = MoELLM(cfg).to(DEV)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=g).to(DEV)
    x2 = torch.randint(0, cfg.vocab_size, (2, 33), generator=g).to(DEV)
    out = {}
    for impl in ("chunked", "fused"):
        m = copy.deepcopy(base)
        m.cfg = m.cfg.replace(ce_impl=impl)
        opt = OptimizerBundle(m, lr=1e-3)
        loss, info = m(x[:, :-1], x[:, 1:])
        loss.backward()
        grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        opt.step(1e-3)
        opt.zero_grad()
        m.update_router_bias()
        loss2, _ = m(x2[:, :-1], x2[:, 1:])
        out[impl] = (loss.detach(), info["ce"], grads, loss2.detach())
    (l_c, ce_c, g_c, l2_c), (l_f, ce_f, g_f, l2_f) = out["chunked"], out["fused"]
    tol = 1e-5 if DEV == "cpu" else 2e-2
    assert abs(l_f.item() - l_c.item()) < tol * abs(l_c.item()) and abs(ce_f.item() - ce_c.item()) < tol * ce_c.item()
    assert g_f.keys() == g_c.keys()
    for k in g_c:
        assert rel_rms(g_f[k], g_c[k]) < (1e-4 if DEV == "cpu" else 5e-2), k
    assert abs(l2_f.item() - l2_c.item()) < (1e-4 if DEV == "cpu" else 2e-2) * abs(l2_c.item())


def test_bf16_model_close_to_chunked():
    """The tiny model in bf16 (bf16 hidden states and head weight, as the GEMMs see them in the main run): loss and
    every gradient within bf16 noise of the chunked path; the head's gradient no farther from fp64 than chunked's."""
    torch.manual_seed(0)
    cfg = tiny(seq_aux_coef=0.0)
    base = MoELLM(cfg).to(DEV).to(torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(2)).to(DEV)
    out, hidden = {}, {}
    for impl in ("chunked", "fused"):
        m = copy.deepcopy(base)
        m.cfg = m.cfg.replace(ce_impl=impl)
        hook = m.final_norm.register_forward_hook(lambda mod, inp, o, impl=impl: hidden.__setitem__(impl, o.detach()))
        loss, info = m(x[:, :-1], x[:, 1:])
        loss.backward()
        hook.remove()
        out[impl] = (loss.detach().float(), {n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
    (l_c, g_c), (l_f, g_f) = out["chunked"], out["fused"]
    if DEV == "cpu":
        assert torch.equal(hidden["chunked"], hidden["fused"])  # identical up to the head (deterministic on CPU)
    assert abs(l_f.item() - l_c.item()) < 1e-2 * abs(l_c.item())
    for k in g_c:
        assert rel_rms(g_f[k], g_c[k]) < 5e-2, k
    # the head's weight gradient against fp64 on the same hidden states and bf16 weight
    e = {}
    for impl, grads in (("chunked", g_c), ("fused", g_f)):
        h = hidden[impl].flatten(0, 1)
        _, _, _, gw_ref = reference(h, base.lm_head.weight.detach(), x[:, 1:].flatten(), cfg.z_loss)
        e[impl] = rel_err(grads["lm_head.weight"], gw_ref)
    assert e["fused"] <= e["chunked"] + 1e-3, e


# ----------------------------------------------------------------------------- GPU, main sizes
MAIN = dict(N=8192, d=1024, V=65536)


def _main_case(seed=0, logit_std=3.0):
    """Main-run shapes: bf16 hidden states (RMS 1), fp32 head weight with logits of std ~logit_std, targets sampled
    from the softmax with the Gumbel-max trick, chunk by chunk."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    N, d, V = MAIN["N"], MAIN["d"], MAIN["V"]
    h = torch.randn(N, d, device="cuda", generator=gen)
    h = (h / h.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
    W = torch.randn(V, d, device="cuda", generator=gen) * (logit_std / d ** 0.5)
    t = torch.empty(N, dtype=torch.long, device="cuda")
    Wb = W.to(torch.bfloat16)
    for s in range(0, N, 1024):
        u = torch.rand(min(1024, N - s), V, device="cuda", generator=gen).clamp_(1e-12, 1 - 1e-7)
        t[s:s + 1024] = (torch.mm(h[s:s + 1024], Wb.t(), out_dtype=torch.float32) - torch.log(-torch.log(u))).argmax(-1)
    return h, W, t


def _reference_fp64_chunked(h, W, t, z, chunk=1024, ignore_index=-100):
    """fp64 loss and gradients on the GPU, chunk by chunk (fp64 logits would be 4 GiB at once)."""
    h64, W64 = h.double(), W.to(torch.bfloat16).double()
    valid = t != ignore_index
    n = valid.sum().clamp_min(1).double()
    grad_h = torch.empty_like(h64)
    grad_W = torch.zeros_like(W64)
    ce_sum = z_sum = 0.0
    for s in range(0, h.shape[0], chunk):
        e = min(s + chunk, h.shape[0])
        lg = h64[s:e] @ W64.t()
        lse = torch.logsumexp(lg, -1)
        v = valid[s:e].double()
        tt = t[s:e].clamp_min(0)
        ce_sum += ((lse - lg.gather(1, tt[:, None]).squeeze(1)) * v).sum()
        z_sum += (lse.pow(2) * v).sum()
        g = torch.exp(lg - lse[:, None]) * ((1 + 2 * z * lse) * v / n)[:, None]
        g[torch.arange(e - s, device=g.device), tt] -= v / n
        del lg
        grad_h[s:e] = g @ W64
        grad_W += g.t() @ h64[s:e]
    ce = ce_sum / n
    return ce + z * z_sum / n, ce, grad_h, grad_W


@gpu
def test_gpu_main_size_no_less_accurate():
    """N 8192, d 1024, V 65536, chunk 2048, z 1e-4: every path against fp64 on the same bf16 operands. fused must be
    closer than chunked for the loss and both gradients; fused_bf16 as close as chunked (grad_W at least as close)."""
    h, W, t = _main_case()
    t[::97] = -100
    z = 1e-4
    ref = _reference_fp64_chunked(h, W, t, z)
    e = {}
    for impl in IMPLS:
        loss, ce, gh, gw = run(impl, h, W, t, z, 2048)
        e[impl] = {"loss": abs(loss.item() - ref[0].item()), "ce": abs(ce.item() - ref[1].item()),
                   "grad_h": rel_err(gh, ref[2]), "grad_W": rel_err(gw, ref[3])}
        del gh, gw
    for impl, d in e.items():
        print(f"{impl:11s} " + "  ".join(f"|{q}-fp64| {v:.2e}" for q, v in d.items()))
    c, f, b = e["chunked"], e["fused"], e["fused_bf16"]
    for q in ("loss", "ce"):
        assert f[q] <= max(c[q] / 10, 1e-5), (q, e)
        assert b[q] <= 1.1 * c[q] + 1e-5, (q, e)
    for q in ("grad_h", "grad_W"):
        assert f[q] <= c[q], (q, e)
        assert b[q] <= 1.05 * c[q] + 1e-6, (q, e)
    assert b["grad_W"] <= c["grad_W"] + 1e-6


@gpu
@pytest.mark.parametrize("impl", ["fused", "fused_bf16"])
def test_gpu_reruns_bit_identical(impl):
    h, W, t = _main_case(seed=1)
    a = run(impl, h, W, t, 1e-4, 2048)
    b = run(impl, h, W, t, 1e-4, 2048)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@gpu
def test_gpu_no_host_sync():
    """The fused paths (training forward + backward, and the no_grad eval forward) make no CPU-GPU synchronisation:
    torch.cuda.set_sync_debug_mode("error") raises on any. The first call compiles the kernel and runs the one-time
    addmm aliasing check (one sync), so it is made before."""
    h, W, t = _main_case(seed=2)
    for impl in ("fused", "fused_bf16"):
        run(impl, h, W, t, 1e-4, 2048)
        with torch.no_grad():
            linear_cross_entropy(h, W, t, z_coef=1e-4, chunk=2048, impl=impl)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            run(impl, h, W, t, 1e-4, 2048)
            with torch.no_grad():
                linear_cross_entropy(h, W, t, z_coef=1e-4, chunk=2048, impl=impl)
        finally:
            torch.cuda.set_sync_debug_mode("default")


@gpu
def test_gpu_addmm_out_aliasing():
    """torch.addmm(C, A, B, out_dtype=float32, out=C) accumulates into C at the loss's shapes (A = g^T, a transposed
    view of [2048, 65536] bf16; C the [65536, 1024] fp32 accumulator), the same as C + mm(A, B, out_dtype=float32)
    up to the epilogue's rounding; the one-time check in loss_fused agrees."""
    assert lf.addmm_alias_ok(torch.device("cuda", torch.cuda.current_device()))
    gen = torch.Generator(device="cuda").manual_seed(3)
    g = (torch.randn(2048, 65536, device="cuda", generator=gen) * 1e-4).to(torch.bfloat16)
    hc = torch.randn(2048, 1024, device="cuda", generator=gen).to(torch.bfloat16)
    C = torch.randn(65536, 1024, device="cuda", generator=gen) * 1e-3
    want = C + torch.mm(g.t(), hc, out_dtype=torch.float32)
    ptr = C.data_ptr()
    r = torch.addmm(C, g.t(), hc, out_dtype=torch.float32, out=C)
    assert r.data_ptr() == ptr == C.data_ptr()
    assert ((C - want).abs().max() / want.abs().max()).item() < 1e-6


@gpu
def test_gpu_peak_memory():
    """Peak memory of one loss forward + backward above what is allocated before (h, W, W.grad): fused at chunk
    2048 must stay below the chunked path at 2048 (its eager chain holds up to ~3.5 [chunk, V] fp32 buffers)."""
    h, W, t = _main_case(seed=4)
    peaks = {}
    for impl, chunk in [("chunked", 2048), ("fused", 2048), ("fused", 4096), ("fused", 8192), ("fused_bf16", 2048),
                        ("fused_bf16", 4096)]:
        hh, WW = h.detach().clone().requires_grad_(), W.detach().clone().requires_grad_()
        WW.grad = torch.zeros_like(WW)
        run_once = lambda: linear_cross_entropy(hh, WW, t, z_coef=1e-4, chunk=chunk, impl=impl)[0].backward()  # noqa
        run_once()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        run_once()
        torch.cuda.synchronize()
        peaks[(impl, chunk)] = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        del hh, WW
    print({f"{k[0]}@{k[1]}": f"{v:.2f} GiB" for k, v in peaks.items()})
    assert peaks[("fused", 2048)] < peaks[("chunked", 2048)]
    assert peaks[("fused_bf16", 2048)] < peaks[("fused", 2048)]


@gpu
def test_gpu_every_config_matches():
    """Each (BLOCK_V, num_warps) of scripts/bench_ce.py's sweep against the default: the same loss up to fp32 rounding
    and gradients up to bf16 flips."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import bench_ce
    h, W, t = _main_case(seed=5)
    h, t = h[:2048], t[:2048]
    ref = run("fused", h, W, t, 1e-4, 2048)
    try:
        for c in bench_ce.SWEEP:
            lf.set_config(*c)
            got = run("fused", h, W, t, 1e-4, 2048)
            assert abs(got[0].item() - ref[0].item()) < 1e-6 * ref[0].item(), c
            assert rel_rms(got[2], ref[2]) < 1e-2 and rel_rms(got[3], ref[3]) < 1e-3, c
    finally:
        lf.reset_config()


@gpu
def test_gpu_model_step_close_to_chunked():
    """Tiny model on the GPU under bf16 autocast with the bf16 residual stream (as in the main run): the fused paths'
    loss and every gradient within bf16 noise of the chunked path."""
    torch.manual_seed(0)
    cfg = tiny(seq_aux_coef=0.0, residual_bf16=True, residual="block_attnres")
    base = MoELLM(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (4, 65), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    out = {}
    for impl in IMPLS:
        m = copy.deepcopy(base)
        m.cfg = m.cfg.replace(ce_impl=impl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, info = m(x[:, :-1], x[:, 1:])
        loss.backward()
        out[impl] = (loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None})
    l_c, g_c = out["chunked"]
    for impl in ("fused", "fused_bf16"):
        l_f, g_f = out[impl]
        assert abs(l_f.item() - l_c.item()) < 1e-2 * abs(l_c.item()), impl
        assert g_f.keys() == g_c.keys()
        for k in g_c:
            assert rel_rms(g_f[k], g_c[k]) < 5e-2, (impl, k)


@gpu
def test_gpu_timing():
    """Prints forward + backward time of the loss at main sizes per micro-batch and per 64-micro-batch step."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import bench_ce
    h, W, t = _main_case(seed=6)
    base = None
    for impl, chunk in [("chunked", 2048), ("fused", 2048), ("fused", 4096), ("fused_bf16", 2048)]:
        ms = bench_ce.time_loss(h, W, t, impl, chunk, reps=5, warmup=2)
        base = base or ms
        print(f"{impl:11s} chunk {chunk:5d}: {ms:7.2f} ms per micro-batch, {ms * 64 / 1e3:5.2f} s per step "
              f"(saves {(base - ms) * 64 / 1e3:5.2f} s per step vs chunked@2048)")
