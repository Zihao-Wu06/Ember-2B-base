"""ModelConfig.ce_blas (moe_llm/loss_fused.py set_blas): the fused cross-entropy's logits GEMM through cuBLASLt
instead of the default cuBLAS kernel ("cublaslt_logits"). Must be bit-identical (loss, grad_h, grad_W): only the logits
GEMM changes library, and it gives the same bits. "cublaslt" (all GEMMs) is reported only: switch 10 found grad_h
not bit-identical with it (the K = 65536 reduction summed in another order)."""
import pytest
import torch

from moe_llm.config import tiny
from moe_llm.model import MoELLM

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_set_blas_validates_and_switches():
    from moe_llm import loss_fused
    try:
        loss_fused.set_blas("cublaslt")
        assert loss_fused._BLAS_LT and loss_fused._BLAS_LT_LOGITS
        loss_fused.set_blas("cublaslt_logits")
        assert loss_fused._BLAS_LT_LOGITS and not loss_fused._BLAS_LT
        loss_fused.set_blas("")
        assert not loss_fused._BLAS_LT and not loss_fused._BLAS_LT_LOGITS
        loss_fused.set_blas("cublas")
        assert not loss_fused._BLAS_LT
        with pytest.raises(ValueError):
            loss_fused.set_blas("magma")
    finally:
        loss_fused.set_blas("")


def test_model_sets_and_resets_the_process_wide_choice():
    from moe_llm import loss_fused
    try:
        MoELLM(tiny(ce_impl="fused", ce_blas="cublaslt_logits"))
        assert loss_fused._BLAS_LT_LOGITS
        MoELLM(tiny(ce_impl="fused"))  # a later model never inherits the earlier choice
        assert not loss_fused._BLAS_LT_LOGITS and not loss_fused._BLAS_LT
        with pytest.raises(ValueError):
            MoELLM(tiny(ce_blas="nope"))
    finally:
        loss_fused.set_blas("")


def test_cpu_forward_unchanged_with_the_option():
    torch.manual_seed(0)
    from moe_llm import loss_fused
    cfg = tiny(ce_impl="fused")
    m0 = MoELLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 33))
    try:
        l0, _ = m0(x[:, :-1], x[:, 1:])
        m1 = MoELLM(cfg.replace(ce_blas="cublaslt_logits"))
        m1.load_state_dict(m0.state_dict())
        l1, _ = m1(x[:, :-1], x[:, 1:])
    finally:
        loss_fused.set_blas("")
    assert torch.equal(l0, l1)  # on CPU the option changes nothing


@gpu
def test_gpu_cublaslt_bit_identical_and_timing():
    """Main-run shapes: 8192 tokens, d 1024, vocab 65536, chunk 2048, z-loss on, bf16 h, fp32 W."""
    from moe_llm import loss_fused
    torch.manual_seed(0)
    N, d, V = 8192, 1024, 65536
    h0 = (torch.randn(N, d, device="cuda") * 0.5).to(torch.bfloat16)
    W0 = torch.randn(V, d, device="cuda") * 0.02
    tgt = torch.randint(0, V, (N,), device="cuda")
    tgt[::97] = -100  # some ignored rows

    def run():
        h, W = h0.clone().requires_grad_(), W0.clone().requires_grad_()
        loss, ce = loss_fused.fused_linear_cross_entropy(h, W, tgt, 1e-4, 2048, -100, fp32_logits=True)
        loss.backward()
        return loss.detach(), ce, h.grad, W.grad

    out, ms = {}, {}
    try:
        for name in ("", "cublaslt_logits", "cublaslt"):
            loss_fused.set_blas(name)
            run()  # warm-up
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(5):
                r = run()
            ev1.record()
            torch.cuda.synchronize()
            out[name], ms[name] = r, ev0.elapsed_time(ev1) / 5
    finally:
        loss_fused.set_blas("")
    names = ("loss", "ce", "grad_h", "grad_W")
    same_all = [torch.equal(a, b) for a, b in zip(out[""], out["cublaslt"])]
    print(f"\nfused CE fwd+bwd per micro-batch: cuBLAS {ms['']:.2f} ms, cuBLASLt logits only "
          f"{ms['cublaslt_logits']:.2f} ms (saves ~{(ms[''] - ms['cublaslt_logits']) * 64 / 1e3:.2f} s per optimizer "
          f"step), cuBLASLt all {ms['cublaslt']:.2f} ms (bit-identical: {dict(zip(names, same_all))})")
    for a, b, name in zip(out[""], out["cublaslt_logits"], names):
        assert torch.equal(a, b), name
