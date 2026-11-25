"""Small shared building blocks.

Elementwise "glue" (norms, gates, SwiGLU, short convs, residual add + norm) is memory-bound: run
eagerly, every step is a separate full-size pass and the norms keep fp32 copies for the backward.
With fusion on (env MOE_FUSE=1 or set_fuse(True); MOE_COMPILE / set_compile are old names) these
helpers run through torch.compile on CUDA: same formulas, same fp32 math inside, and inductor is
told to round intermediates to bf16 exactly where eager does (emulate_precision_casts). Each call
signature gets its own compiled copy with its own code object, so dynamo's recompile limit (8 per
code object) is never hit. A failed compile falls back to eager with a warning. Default: eager
(fusion is switched on only after the GPU equivalence checks pass).
"""
from __future__ import annotations

import os
import types
import warnings
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

_FUSE = False
_FUSE_CPU = False      # tests: compile for CPU tensors as well
_CLONES: dict = {}     # (helper, call signature) -> compiled copy
_FAILED: set = set()   # signatures whose compile failed (they run eagerly)


def set_fuse(enabled: bool, cpu: bool = False) -> None:
    """Fusion on/off for the helpers below; cpu=True also compiles for CPU tensors (tests)."""
    global _FUSE, _FUSE_CPU
    _FUSE, _FUSE_CPU = bool(enabled), bool(cpu)
    if _FUSE:
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True  # bf16 rounding of intermediates as in eager


set_compile = set_fuse  # old name, used by train.py / bench.py --compile


def release_compile_workers() -> bool:
    """Shut down torch.compile's idle compile-worker processes. Once compilation is done, PyTorch 2.15 quiesces
    its workers but keeps a sidecar process with ~0.4 GB of private RAM. shutdown_compile_workers() stops the
    pools and clears AsyncCompile.process_pool's cache, so a later compile (new shapes, e.g. evaluation) starts a
    pool again on demand; with no pool it does nothing. Compiled code keeps running unchanged. Call it from the
    main thread while nothing is compiling (train.py: on log steps, after the step). Returns False if the call is
    unavailable or failed."""
    try:
        from torch._inductor.async_compile import shutdown_compile_workers
        shutdown_compile_workers()
        return True
    except Exception:
        return False


def fusion_on() -> bool:
    return _FUSE


def fuse_active(t: torch.Tensor) -> bool:
    return _FUSE and (t.is_cuda or _FUSE_CPU)


def _signature(args, dynamic: bool) -> tuple:
    sig = [torch.is_grad_enabled()]
    for a in args:
        if isinstance(a, torch.Tensor):
            sig.append((a.ndim if dynamic else tuple(a.shape), a.dtype, a.requires_grad, a.device.type))
        else:
            sig.append(a)
    return tuple(sig)


def _clone(fn, tag: int):
    name = f"{fn.__name__}__{tag}"
    code = fn.__code__
    code = code.replace(co_name=name, **({"co_qualname": name} if hasattr(code, "co_qualname") else {}))
    return types.FunctionType(code, fn.__globals__, name, fn.__defaults__, fn.__closure__)


def fused(fn=None, *, dynamic: bool = False, compiled_fn=None):
    """Decorator: eager, or (fusion on, tensors on CUDA) a torch.compile'd copy of `compiled_fn or fn`
    per call signature. dynamic=True: one copy per tensor rank with symbolic sizes (MoE token counts
    per expert change every step). `compiled_fn` must compute the same values as `fn`."""
    def deco(f):
        target = compiled_fn or f

        def wrapper(*args):
            t = next((a for a in args if isinstance(a, torch.Tensor)), None)
            if t is None or not fuse_active(t):
                return f(*args)
            key = (f.__name__, _signature(args, dynamic))
            if key not in _FAILED:
                try:
                    c = _CLONES.get(key)
                    if c is None:
                        c = _CLONES[key] = torch.compile(_clone(target, len(_CLONES)), dynamic=dynamic,
                                                         fullgraph=True)
                    return c(*args)
                except torch.OutOfMemoryError:
                    raise
                except Exception as e:  # a compiler problem must not stop a long run
                    _FAILED.add(key)
                    warnings.warn(f"fused {f.__name__}: torch.compile failed, running eagerly: {e!r}")
            return f(*args)

        wrapper.__name__ = f.__name__
        wrapper.eager = f
        wrapper.fused_target = target
        return wrapper

    return deco(fn) if fn is not None else deco


_FUSE_ENV = os.environ.get("MOE_FUSE", os.environ.get("MOE_COMPILE", "0")) == "1"
if _FUSE_ENV:
    set_fuse(True)


@fused
def _rms_norm(x, w, eps: float):
    return F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps).to(x.dtype)


@fused
def add_rms_norm(h, delta, w, eps: float):
    """(h + delta, rms_norm(h + delta)): the residual add and the next norm in one pass."""
    s = h + delta
    return s, F.rms_norm(s.float(), (s.shape[-1],), w.float(), eps).to(s.dtype)


@fused(dynamic=True)
def silu_mul(g, u):
    return F.silu(g) * u


@fused
def sigmoid_mul(o, g):
    """Attention output gate: o * sigmoid(g)."""
    return o * torch.sigmoid(g).to(o.dtype)


# ------------------------------------------------------------------ weight casts with fused grad accumulation
# Under bf16 autocast every fp32 weight is cast to bf16 for its matmul. Autograd then turns the bf16 weight
# gradient into a new fp32 tensor (ToCopyBackward) and adds that into .grad (AccumulateGrad): 18 bytes of
# memory traffic per parameter per micro-batch, ~1.7 TB per main-run step. cast_weight() does the same cast
# but adds the bf16 gradient straight into the fp32 .grad in one pass (10 bytes). The values are identical:
# bf16 -> fp32 is exact and the add is the same fp32 add, in the same order.
_FAST_ACCUM = os.environ.get("MOE_FAST_ACCUM", "1") != "0"


def set_fast_accum(enabled: bool) -> None:
    global _FAST_ACCUM
    _FAST_ACCUM = bool(enabled)


class _CastAccumulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, dtype):
        ctx.w = w  # the leaf parameter itself (not a saved tensor: nothing to recompute under checkpointing)
        return w.to(dtype)

    @staticmethod
    def backward(ctx, g):
        w = ctx.w
        if w.grad is None:
            w.grad = g.to(w.dtype)
        else:
            w.grad.add_(g)  # fp32 += bf16 in one kernel: same rounding as adding the fp32 copy
        return None, None


# Persistent bf16 copies (OptimizerBundle(offload_master=True)): the fp32 masters live in pinned CPU memory
# between optimizer steps and the forward uses these copies, refreshed once per step instead of cast in every
# micro-batch. Their gradients are accumulated into fp32 buffers by a hook (same values as above).
#
# Fused accumulation (OptimizerBundle(offload_master=True, fused_grad_acc=True)): a copy can carry a GradSink, its
# parameter's fp32 gradient buffer. The weight-gradient GEMMs of CastLinear (without bias) and of the Triton routed
# experts then add their fp32 result straight into that buffer (fp32 GEMM output, beta = 1) and return no gradient
# for the copy (AccumulateGrad still calls its hook, which finds no gradient and does nothing: nothing is counted
# twice). Per micro-batch and weight this drops the bf16 rounding of the gradient and the separate add kernel (bf16
# write + bf16 read + fp32 read + fp32 write -> fp32 read + fp32 write in the GEMM epilogue). The forward and every
# activation gradient are computed exactly as before; only the weight gradients change, by that rounding, towards
# the exact sum.
_CACHE: dict = {}  # id(parameter) -> (weak reference to it, bf16 leaf tensor, GradSink or None)


class GradSink:
    """The fp32 gradient buffer of one parameter, store[key] (OptimizerBundle._gbuf[id(param)], which clipping,
    the optimizer step and zero_grad use). The first backward of a step that writes it creates it as zeros (adding
    to zeros is exact, so the first micro-batch's gradient lands unrounded)."""
    __slots__ = ("store", "key", "shape", "device")

    def __init__(self, store: dict, key, shape, device):
        self.store, self.key, self.shape, self.device = store, key, tuple(shape), torch.device(device)

    def buffer(self) -> torch.Tensor:
        b = self.store.get(self.key)
        if b is None:
            # OptimizerBundle(flat=True): the store hands out the zeroed view of its persistent flat buffer instead
            zero_view = getattr(self.store, "zero_view", None)
            b = zero_view(self.key) if zero_view is not None else None
            if b is None:
                b = self.store[self.key] = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        return b

    def add_mm(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """buffer += a @ b (fp32, one GEMM: add_mm_fp32_)."""
        add_mm_fp32_(self.buffer(), a, b)


# Fused projections (ModelConfig.fuse_proj): bias-free linear layers that read the same input run as ONE GEMM over
# their weights stacked row-wise, y = x @ cat(W_i)^T, each member's output a column block of y (fused_linear below).
# With persistent copies the stacked bf16 copy is one tensor whose row blocks ARE the members' copies (views, so the
# per-step refresh of the copies writes it in place); with fused accumulation the group's fp32 gradient buffer is one
# tensor too, whose row blocks are the members' buffers (GroupGradSink), so one weight-gradient GEMM fills them all.
_GROUPS: dict = {}  # tuple of id(parameter) -> (weak references, stacked bf16 leaf, GroupGradSink or None)


class GroupGradSink:
    """The fp32 gradient buffers of a group of parameters used as one stacked matrix (OptimizerBundle._gbuf[id(p)]):
    ONE [sum of rows, cols] tensor, and each member's store entry is its row block (a view). Clipping, the optimizer
    step, zero_grad and the non-finite skip keep seeing one buffer per parameter. The first backward of a step that
    writes it creates it as zeros (as GradSink); a gradient a member already received in this step through another
    path (its copy's hook or its own GradSink) is copied into its row block first. Nothing is cached here: whether
    the store entries are this group's row blocks is checked on every call (so zero_grad's clear() just works).
    per_member (ModelConfig.fuse_proj_dw = "member"): the weight-gradient GEMM runs once per member on its row block
    (the separate layers' GEMMs) instead of once for the whole stack. Both sum the same products in fp32; they differ
    only where a GEMM library sums an output element in another order for another row count (CPU MKL does for the
    1024-row blocks; cuBLAS: tests/test_fuse_proj.py reports it)."""
    __slots__ = ("store", "keys", "rows", "cols", "device", "per_member")

    def __init__(self, store: dict, keys, shapes, device, per_member: bool = False):
        shapes = [tuple(s) for s in shapes]
        if any(len(s) != 2 or s[1] != shapes[0][1] for s in shapes):
            raise ValueError(f"GroupGradSink needs 2-D shapes with equal columns, got {shapes}")
        self.store, self.keys = store, tuple(keys)
        self.rows, self.cols = tuple(s[0] for s in shapes), shapes[0][1]
        self.device = torch.device(device)
        self.per_member = bool(per_member)

    def _current(self):
        es = [self.store.get(k) for k in self.keys]
        base = es[0]._base if es[0] is not None else None
        if base is None or base.shape != (sum(self.rows), self.cols) or base.dtype != torch.float32:
            return es, None
        a = base.storage_offset()
        for e, r in zip(es, self.rows):
            if e is None or e._base is not base or e.storage_offset() != a or e.shape != (r, self.cols):
                return es, None
            a += r * self.cols
        return es, base

    def buffer(self) -> torch.Tensor:
        es, base = self._current()
        if base is not None:
            return base
        base = torch.zeros(sum(self.rows), self.cols, dtype=torch.float32, device=self.device)
        a = 0
        for k, e, r in zip(self.keys, es, self.rows):
            v = base[a:a + r]
            if e is not None:
                v.copy_(e)  # exact
            self.store[k] = v
            a += r
        return base

    def add_mm(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """buffer += a @ b for a = the stacked weights' output gradient^T [sum of rows, K], b = the input [K, cols]: one
        fp32 GEMM for the whole stack (or one per member's row block, per_member)."""
        buf = self.buffer()
        if not self.per_member:
            add_mm_fp32_(buf, a, b)
            return
        o = 0
        for r in self.rows:
            add_mm_fp32_(buf[o:o + r], a[o:o + r], b)
            o += r


def set_cached_copy(w: torch.Tensor, copy: torch.Tensor, sink: GradSink | None = None) -> None:
    _CACHE[id(w)] = (weakref.ref(w), copy, sink)


def set_cached_group(params, stacked: torch.Tensor, sink: GroupGradSink | None = None) -> None:
    """The persistent stacked copy of a fused group (OptimizerBundle.begin): stacked[rows of member i] is member i's
    copy; sink: the group's fp32 gradient buffers (fused accumulation) or None (the stacked copy's hook hands each
    member its row block of the bf16 gradient)."""
    _GROUPS[tuple(id(p) for p in params)] = (tuple(weakref.ref(p) for p in params), stacked, sink)


def clear_cached_copies() -> None:
    _CACHE.clear()
    _GROUPS.clear()


def cast_weight(w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """w.to(dtype). For a full-precision leaf parameter that needs a gradient, the gradient is accumulated
    straight into w.grad (see above). Each weight must go through here at most once per forward (as with
    autocast's own cast cache), and only through here."""
    e = _CACHE.get(id(w))
    if e is not None and e[0]() is w and e[1].dtype == dtype:
        return e[1]
    if (not _FAST_ACCUM or w.dtype == dtype or not w.requires_grad or w.grad_fn is not None
            or not torch.is_grad_enabled()):
        return w.to(dtype)
    return _CastAccumulate.apply(w, dtype)


def fused_copy(w: torch.Tensor, dtype: torch.dtype):
    """(persistent copy, GradSink) if w's gradient goes straight from its weight-gradient GEMM into the fp32
    buffer (see above), else None. Use instead of cast_weight, at most once per weight and forward; only while
    gradients are recorded (otherwise the plain copy from cast_weight is all a forward needs)."""
    e = _CACHE.get(id(w))
    if e is None or e[2] is None or e[0]() is not w or e[1].dtype != dtype or not torch.is_grad_enabled():
        return None
    return e[1], e[2]


def group_copy(params, dtype: torch.dtype):
    """(persistent stacked copy, GroupGradSink or None) of a fused group, or None if the group has none in dtype.
    The sink only while gradients are recorded."""
    e = _GROUPS.get(tuple(id(p) for p in params))
    if e is None or e[1].dtype != dtype or any(r() is not p for r, p in zip(e[0], params)):
        return None
    return e[1], (e[2] if torch.is_grad_enabled() else None)


def add_mm_fp32_(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """out += a @ b for bf16 a, b and an fp32 out. CUDA: one cuBLAS GEMM with fp32 output and beta = 1
    (aten::addmm.dtype_out with out aliasing self, which ATen runs in place: cublasGemmEx with C = D, no copy).
    The products are summed in fp32 and the sum is added to out once. Elsewhere (CPU tests; addmm.dtype has no CPU
    kernel) the same: the fp32 product of the bf16 values (exact), then one fp32 add."""
    if out.is_cuda:
        return torch.addmm(out, a, b, out_dtype=torch.float32, out=out)
    return out.add_(torch.mm(a.float(), b.float()))


class _LinearAccumulate(torch.autograd.Function):
    """F.linear(x, w) without bias for a persistent copy w whose gradient dy^T x is added into its parameter's fp32
    buffer by add_mm_fp32_ (sink: a GradSink; or a GroupGradSink for the stacked copy of a fused group, whose buffer
    is the members' buffers stacked); no gradient is returned for w. Forward and dx are exactly what autograd computes
    for F.linear (matmul folds x to 2-D, then mm; dx as aten's mm_mat1_backward)."""

    @staticmethod
    def forward(ctx, x, w, sink):
        with torch.autocast(x.device.type, enabled=False):  # x and w are already in the autocast dtype
            y = F.linear(x, w)
        ctx.save_for_backward(x, w)  # w is a leaf: activation offloading leaves it alone
        ctx.sink = sink
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])  # the 2-D input of matmul's mm, same layout as in the forward
        dx = None
        if ctx.needs_input_grad[0]:
            if x2.stride(0) == 1 and x2.stride(1) == x2.shape[0]:
                dx2 = w.t().mm(dy2.t()).t()  # column-major input: mm_mat1_backward keeps that layout
            else:
                dx2 = dy2.mm(w)
            dx = dx2.reshape(x.shape)
        ctx.sink.add_mm(dy2.t(), x2)  # add_mm_fp32_ with the same operands as autograd's dW = dy^T x
        return dx, None, None


class CastLinear(nn.Linear):
    """nn.Linear whose weight (and bias) cast under autocast goes through cast_weight: same math as
    nn.Linear under autocast, less memory traffic in the backward. Same parameters and state_dict keys.
    With fused accumulation (see above) the weight of a layer without bias goes through _LinearAccumulate
    (the only biased layer, KDA's g_up, keeps the hook path: same values as before)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device.type
        if not torch.is_autocast_enabled(dev):
            return F.linear(x, self.weight, self.bias)
        dt = torch.get_autocast_dtype(dev)
        if self.bias is None:
            f = fused_copy(self.weight, dt)
            if f is not None:  # autocast's own cast of x (fp32 / fp16 -> dt, recorded by autograd), then the GEMM
                cast = x.is_floating_point() and x.dtype not in (dt, torch.float64)
                return _LinearAccumulate.apply(x.to(dt) if cast else x, f[0], f[1])
        b = cast_weight(self.bias, dt) if self.bias is not None else None
        return F.linear(x, cast_weight(self.weight, dt), b)


def fused_linear(x: torch.Tensor, weights) -> torch.Tensor:
    """The bias-free linear layers with these weights ([n_i, d] each), all applied to the same x, as ONE GEMM:
    x @ cat(weights)^T, i.e. the members' outputs side by side ([..., sum n_i]; split it with Tensor.split, whose
    backward is one cat, or a function that consumes all blocks such as split_conv_silu).
    Each output element is the same dot product as in the member's own GEMM: bit-identical where the GEMM library
    sums an element in the same order for a wider matrix (CPU oneDNN does; cuBLAS: tests/test_fuse_proj.py reports).
    Backward: dx = dy @ cat(W) as one GEMM (fp32 accumulation over all members inside it, one bf16 rounding) instead
    of one bf16 GEMM per member plus autograd's bf16 adds of the partial dx: a rounding-level change towards the
    exact sum. dW: one GEMM for the stack; every member's gradient is its row block (same products).
    Under autocast the weights come from the group's persistent stacked copy (OptimizerBundle(offload_master=True)):
    with fused accumulation through _LinearAccumulate into the group's fp32 buffers, else through the stacked copy's
    hook; without persistent copies the members' cast_weight casts are stacked (their gradients flow back through
    cast_weight as before). Without autocast: the fp32 weights stacked."""
    dev = x.device.type
    if not torch.is_autocast_enabled(dev):
        return F.linear(x, torch.cat(list(weights)))
    dt = torch.get_autocast_dtype(dev)
    g = group_copy(weights, dt)
    if g is None:
        return F.linear(x, torch.cat([cast_weight(w, dt) for w in weights]))
    w, sink = g
    if sink is None:
        return F.linear(x, w)
    cast = x.is_floating_point() and x.dtype not in (dt, torch.float64)  # autocast's own cast of x, as CastLinear
    return _LinearAccumulate.apply(x.to(dt) if cast else x, w, sink)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in fp32, return the input dtype
        return _rms_norm(x, self.weight, self.eps)


@fused
def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fla FusedRMSNormGated(activation='sigmoid'): rmsnorm(x) * w * sigmoid(gate), over the last dim."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return (y * torch.sigmoid(gate.float())).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, inter: int, std: float, out_std: float):
        super().__init__()
        self.gate = CastLinear(d, inter, bias=False)
        self.up = CastLinear(d, inter, bias=False)
        self.down = CastLinear(inter, d, bias=False)
        nn.init.normal_(self.gate.weight, std=std)
        nn.init.normal_(self.up.weight, std=std)
        nn.init.normal_(self.down.weight, std=out_std)

    def forward(self, x: torch.Tensor, x_up: torch.Tensor | None = None) -> torch.Tensor:
        """x_up: the up projection's input if it is a different tensor object holding the same values (the MoE's fused
        fan-in hands gate and up separate views of its input, so their input gradients arrive separately); same ops."""
        return self.down(silu_mul(self.gate(x), self.up(x if x_up is None else x_up)))


def causal_depthwise_conv(x: torch.Tensor, weight: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    """x: [B, T, D], weight: [D, 1, k]. Output at t only sees x[t - j*dilation], j = 0..k-1."""
    B, T, D = x.shape
    k = weight.shape[-1]
    pad = (k - 1) * dilation
    y = F.conv1d(F.pad(x.transpose(1, 2), (pad, 0)), weight.to(x.dtype), groups=D, dilation=dilation)
    return y.transpose(1, 2)


def _conv_silu_shifted(x, weight, dilation: int = 1, residual=None, out_dtype=None):
    """conv_silu written as k shifted multiply-adds, which inductor turns into one kernel (no transposes,
    no padded copy, no cuDNN). Same values: conv1d output at t = sum_i w[i] * x[t - (k-1-i)*dilation]
    with the weight rounded to x.dtype, accumulated in fp32 and rounded to x.dtype before the SiLU."""
    T, k = x.shape[1], weight.shape[-1]
    xp = F.pad(x.float(), (0, 0, (k - 1) * dilation, 0))
    w = weight.to(x.dtype).float()[:, 0, :]  # [D, k]
    c = xp[:, 0:T] * w[:, 0]
    for i in range(1, k):
        c = c + xp[:, i * dilation:i * dilation + T] * w[:, i]
    y = F.silu(c.to(x.dtype))
    if residual is not None:
        y = y + residual
    return y if out_dtype is None else y.to(out_dtype)


@fused(compiled_fn=_conv_silu_shifted)
def conv_silu(x, weight, dilation: int = 1, residual=None, out_dtype=None):
    """SiLU(causal depthwise conv(x)) [+ residual] [-> out_dtype]. Pass arguments positionally."""
    y = F.silu(causal_depthwise_conv(x, weight, dilation))
    if residual is not None:
        y = y + residual
    return y if out_dtype is None else y.to(out_dtype)


def _split_conv_silu_shifted(y, weight, conv_sizes: tuple, sizes: tuple):
    """split_conv_silu for torch.compile: conv_silu's shifted form per conv block. Per block it is exactly the graph
    of one compiled ShortConv (same shapes, same reductions for the weight gradient, contiguous output as before);
    each block's output is written directly (one conv over all conv channels then sliced would make inductor store
    the whole conv output and copy the blocks out of it, since it has several readers). AOTAutograd returns y's
    gradient as one tensor, written by the backward kernels (no separate cat)."""
    outs, a = [], 0
    for s in conv_sizes:
        outs.append(_conv_silu_shifted(y[..., a:a + s], weight[a:a + s]).contiguous())
        a += s
    for s in sizes:
        outs.append(y[..., a:a + s].contiguous())
        a += s
    return tuple(outs)


@fused(compiled_fn=_split_conv_silu_shifted)
def split_conv_silu(y, weight, conv_sizes: tuple, sizes: tuple):
    """The column blocks of y (a fused projection's output, [B, T, sum(conv_sizes) + sum(sizes)]): the first
    sum(conv_sizes) channels go through one causal depthwise conv + SiLU over all of them (weight = the blocks' conv
    weights concatenated; per channel, so the same values as one ShortConv per block) and come back per block, the
    remaining blocks come back as they are. Each output has the memory layout the separate layer gives (here: eager
    conv1d's channels-first layout per block; contiguous after torch.compile and for the plain blocks), so everything
    downstream runs exactly as before (fla's kernels need contiguous inputs and copy anything else). The gradient of y
    arrives as one tensor, which is what the fused projection's dx GEMM needs. Pass arguments positionally."""
    n = sum(conv_sizes)
    parts = y.split((n, *sizes), dim=-1)
    c = conv_silu.eager(parts[0], weight)
    convs = (b.transpose(1, 2).contiguous().transpose(1, 2) for b in c.split(conv_sizes, dim=-1))
    return (*convs, *(t.contiguous() for t in parts[1:]))


# ------------------------------------------------------------------ short convs through fla's kernels
# ModelConfig.conv_impl "fla" / "fla_gluon": ShortConv runs fla's own causal_conv1d kernels (fla 0.6.0,
# fla/modules/causal_conv1d: "fla" = the Triton kernels behind fla.modules.causal_conv1d.causal_conv1d, "fla_gluon" =
# the Gluon kernels of fla's Gluon backend for the same op) instead of conv_silu. Against the compiled path
# (_conv_silu_shifted; inductor with emulate_precision_casts):
#   weight    rounded to the input dtype (cast_weight; there weight.to(x.dtype)); its gradient arrives in that dtype
#             (bf16) and is added into the fp32 .grad exactly as there (exact upcast, one fp32 add)
#   forward   pre-activation c[t] = sum_i w[i] * x[t - k + 1 + i] summed in fp32 in the same order (i = 0, 1, ...; a
#             bf16 x bf16 product is exact in fp32, so FMA contraction changes nothing): the same fp32 value (the
#             kernels start from +0.0, so an all-zero sum can come out +0.0 instead of -0.0). Then SiLU on that fp32 value
#             and ONE rounding to bf16; the compiled path rounds c to bf16, applies SiLU and rounds again
#   backward  SiLU's derivative at the bf16-rounded c (as the compiled path), times dy, kept in fp32 (the compiled path
#             rounds it to bf16 first); dx = one bf16 rounding of the fp32 sum of the k products; dw = fp32 partial sums
#             per 64-token block, summed in fp32, one rounding to the weight's dtype (bf16, as the compiled path's
#             weight gradient). No atomics: deterministic. fla's Triton kernels are autotuned (Triton caches the choice
#             on disk; x, y and dx do not depend on it, dw's fp32 partial sums can); the Gluon kernels have fixed configs
#   memory    saved for the backward: the bf16 input and the bf16 weight; the compiled graph saves an fp32 padded copy
#             of the input instead (twice the bytes). The backward recomputes c ("fla": a separate conv kernel writing a
#             transient bf16 tensor; "fla_gluon": in registers, inside the one backward kernel)
# CUDA tensors only; other tensors take conv_silu, except CPU tensors with conv_impl "fla" under Triton's CPU
# interpreter (TRITON_INTERPRET=1 set before triton is imported: tests), which run fla's Triton kernels there with one
# fixed launch config (the interpreter cannot time the candidates; Gluon kernels do not run in the interpreter).
# "fla" goes through fla's own dispatcher, so fla's env switches still apply to it: FLA_GLUON=1 / FLA_CONV_GLUON=1 would
# route it to the Gluon kernels too (neither is set in the main run's environment); "fla_gluon" calls them directly.
CONV_IMPLS = ("compiled", "fla", "fla_gluon")
_FLA_CONV: dict = {}  # impl -> fla entry point(s), or the exception importing them raised
_TRITON = None        # the triton module (False: not importable), for _triton_interpret


def _triton_interpret() -> bool:
    global _TRITON
    if _TRITON is None:
        try:
            import triton
            _TRITON = triton
        except Exception:
            _TRITON = False
    return bool(_TRITON and _TRITON.knobs.runtime.interpret)


def _fla_conv_entry(impl: str):
    """"fla": fla.modules.causal_conv1d.causal_conv1d; "fla_gluon": (forward, backward) of fla's Gluon backend."""
    f = _FLA_CONV.get(impl)
    if f is None:
        try:
            if impl == "fla":
                from fla.modules.causal_conv1d import causal_conv1d as f
                if _triton_interpret():
                    import triton
                    from fla.modules.causal_conv1d import ops
                    for kernel in (ops.causal_conv1d_fwd_kernel, ops.causal_conv1d_bwd_kernel):
                        kernel.fn.configs = [triton.Config({"BD": 32}, num_warps=4)]  # Heuristics -> Autotuner
            elif impl == "fla_gluon":
                from fla.modules.causal_conv1d.backends.gluon import causal_conv1d_bwd_gluon, causal_conv1d_fwd_gluon
                f = (causal_conv1d_fwd_gluon, causal_conv1d_bwd_gluon)
            else:
                raise ValueError(f"conv_impl={impl!r}; expected one of {CONV_IMPLS}")
        except Exception as e:  # depends on the machine (fla needs Triton)
            f = e
        _FLA_CONV[impl] = f
    if isinstance(f, Exception):
        raise RuntimeError(f"conv_impl={impl!r} needs fla's causal_conv1d kernels, which could not be loaded: {f!r}")
    return f


class _FlaGluonConvSilu(torch.autograd.Function):
    """SiLU(causal depthwise conv(x)) through fla's Gluon kernels, called as fla's dispatcher calls them when its Gluon
    backend is enabled (env FLA_CONV_GLUON=1); here selected by conv_impl instead of a process-wide env var.
    x: [B, T, D] with unit channel stride, w: [D, k] contiguous, 2 <= k <= 4 (the backend's own conditions)."""

    @staticmethod
    def forward(ctx, x, w):
        fwd, _ = _fla_conv_entry("fla_gluon")
        y, _ = fwd(x, w, activation="silu")
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        _, bwd = _fla_conv_entry("fla_gluon")
        dx, dw, _, _, _ = bwd(x, dy, None, weight=w, activation="silu")
        return dx, dw


def fla_conv_silu(x: torch.Tensor, weight: torch.Tensor, impl: str = "fla") -> torch.Tensor:
    """conv_silu(x, weight) (dilation 1, no residual) through fla's kernels (see above). x: [B, T, D], weight: [D, 1, k]
    (an fp32 parameter: rounded to x.dtype by cast_weight, as conv_silu rounds it)."""
    D, k = weight.shape[0], weight.shape[-1]
    w = cast_weight(weight, x.dtype).view(D, k)
    if impl == "fla_gluon" and x.dim() == 3 and x.stride(-1) == 1 and 2 <= k <= 4:
        return _FlaGluonConvSilu.apply(x, w)
    y, _ = _fla_conv_entry("fla")(x=x, weight=w, activation="silu", backend="triton")
    return y


class ShortConv(nn.Module):
    """fla ShortConvolution: depthwise causal conv (kernel 4, no bias) followed by SiLU. impl = ModelConfig.conv_impl:
    "compiled" runs conv_silu; "fla" / "fla_gluon" run fla's kernels on CUDA tensors (fla_conv_silu), conv_silu on
    others (see above). Same parameter either way."""

    def __init__(self, dim: int, kernel_size: int = 4, std: float = 0.02, impl: str = "compiled"):
        super().__init__()
        if impl not in CONV_IMPLS:
            raise ValueError(f"conv_impl={impl!r}; expected one of {CONV_IMPLS}")
        self.impl = impl
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size))
        nn.init.normal_(self.weight, std=std)  # fla _init_weights re-inits every Conv1d to N(0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl != "compiled" and (x.is_cuda or (self.impl == "fla" and _triton_interpret())):
            return fla_conv_silu(x, self.weight, self.impl)
        return conv_silu(x, self.weight)


class Rotary(nn.Module):
    """Standard RoPE (rotate-half). Only used by the pure full-attention baseline."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):  # [B, T, H, hd]
        T = q.shape[1]
        t = torch.arange(T, device=q.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(q.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

        def rot(x):
            x1, x2 = x.float().chunk(2, dim=-1)
            xr = torch.cat([-x2, x1], dim=-1)
            return (x.float() * cos + xr * sin).to(x.dtype)

        return rot(q), rot(k)
