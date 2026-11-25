"""DeepSeek-V3 style MoE: 1 shared expert + fine-grained routed experts.

Routing: sigmoid affinities s = sigmoid(W_r x) computed in fp32. Experts are SELECTED by
top-k of (s + b), where b is a non-learned per-expert bias (aux-loss-free balancing); the
combine weights use the UNBIASED s, renormalised over the selected k (x routed_scale).
After every optimizer step: b_i += gamma * sign(mean_load - load_i).
A tiny sequence-wise balance loss (alpha = 1e-4) guards against extreme imbalance inside
one sequence.

Note on activation checkpointing: the recomputed forward routes identically, so the load
counters for a checkpointed layer are exactly doubled for every expert. The bias update
only uses sign(mean - load), which is invariant to that uniform scaling.

recompute_experts (set by MoELLM.set_checkpointing for layers that keep their activations):
the bmm path pads every expert to the busiest one, so the buffers it keeps for backward grow
with routing imbalance (up to 8x at E/k = 8). With this flag only the gathered input is kept
and the expert part is recomputed in backward: the same ops on the same inputs, so values and
gradients are unchanged, and memory no longer depends on the routing.

Host syncs: every time the CPU reads a GPU value it waits for the GPU's whole queue, and the GPU then idles
until the CPU has queued new work. The per-expert token counts use a scatter-add instead of torch.bincount
(on CUDA bincount reads the input's min and max back to size its output: two syncs per call). The bmm path
needs the padded capacity on the CPU: it is read back with a non-blocking copy started as soon as the counts
exist, the independent dispatch work is queued before waiting for it, and the expert recompute in backward
gets the same capacity as a plain int instead of reading it again. All values are unchanged.

moe_impl="triton" (layers/moe_triton.py): the whole expert part (dispatch, the three GEMMs, SiLU gate,
weighted combine) as Triton kernels over the expert-sorted assignments: no padding, no host sync,
deterministic. With recompute_experts it keeps only x, w and the sort for backward and reruns the
gate/up GEMM there (bit-identical values). The load counter above is already sync-free for every impl.
With fused gradient accumulation (OptimizerBundle(fused_grad_acc=True)) its weight-gradient kernels add into
the fp32 gradient buffers directly; the other impls keep the copies' hooks.

Fused fan-in (cfg.moe_shared_fused, moe_impl="triton"; layers/moe_triton.py MoEFan / RoutedExpertsFan): the layer
input xf feeds the router, the routed experts and the shared expert's gate and up projections, so its gradient has
four parts, which autograd sums with three [N, d] bf16 adds (the router's part after a fp32 [N, d] GEMM output and a
cast to bf16); the output routed + shared is one more [N, d] add. With the option the fan node MoEFan hands each reader
its own view of xf, so each part of the gradient arrives there separately and one combine kernel sums them; the shared
expert runs before the routed experts, whose combine kernel adds its output. "exact" reproduces every rounding of the
old adds in their old order (bit-identical; four [N, d] adds fewer per layer and micro-batch); "router" also moves the
router's GEMM into the fan node and adds its input gradient dlogits @ W_r there as fp32 products, the whole input
gradient rounded once (closer to the exact sum; no fp32 [N, d] dx, no cast); "router_lean" also stops keeping the
router's fp32 copy of the input for the backward. The routing, the forward values and every weight gradient are
bit-identical in all modes, except "router_lean"'s router weight gradient (fp32-GEMM accuracy instead of TF32). The
order of the backward work changes (the shared expert's backward now runs after the routed experts'); every sum is
fixed by the data flow (the fan node runs once all parts have arrived), not by that order.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import SwiGLU, cast_weight, fused_copy, silu_mul

FAN_MODES = ("", "exact", "router", "router_lean")  # ModelConfig.moe_shared_fused


def expert_counts(flat: torch.Tensor, E: int) -> torch.Tensor:
    """Tokens per expert (int64): the same values as torch.bincount(flat, minlength=E), without its host syncs."""
    return torch.zeros(E, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))


def capacity_bucket(c: int) -> int:
    return max(64, (c + 63) // 64 * 64)  # bucket the capacity to limit shape churn


class MaxReadback:
    """t.max() as a Python int without draining the GPU queue: on CUDA the value goes to pinned host memory with
    a non-blocking copy and an event, started right away; get() waits for that copy only, so GPU work queued in
    between keeps the GPU busy while the CPU waits."""
    __slots__ = ("val", "ev")

    def __init__(self, t: torch.Tensor):
        m = t.max()
        self.ev = None
        if m.is_cuda:
            self.val = torch.empty((), dtype=m.dtype, pin_memory=True)
            self.val.copy_(m, non_blocking=True)
            self.ev = torch.cuda.Event()
            self.ev.record()
        else:
            self.val = m

    def get(self) -> int:
        if self.ev is not None:
            self.ev.synchronize()
        return int(self.val)


class MoE(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        d, E, I, std = cfg.d_model, cfg.n_experts, cfg.expert_inter, cfg.init_std
        self.E, self.k, self.d, self.I = E, cfg.top_k, d, I
        self.routed_scale = cfg.routed_scale
        self.gamma = cfg.router_bias_update
        self.seq_aux_coef = cfg.seq_aux_coef
        self.impl = cfg.moe_impl
        self._grouped_ok = None
        self.recompute_experts = False
        self.fan = cfg.moe_shared_fused
        self.sort_impl = getattr(cfg, "moe_sort", "torch")
        if self.fan not in FAN_MODES:
            raise ValueError(f"moe_shared_fused={self.fan!r}; expected one of {FAN_MODES}")
        if self.sort_impl not in ("torch", "counting"):  # moe_triton.SORT_IMPLS (imported lazily, like Triton)
            raise ValueError(f"moe_sort={self.sort_impl!r}; expected 'torch' or 'counting'")
        if self.fan and self.impl != "triton":
            raise ValueError(f"moe_shared_fused={self.fan!r} needs moe_impl='triton' (got {self.impl!r})")

        self.router = nn.Linear(d, E, bias=False)
        nn.init.normal_(self.router.weight, std=std)
        self.register_buffer("e_bias", torch.zeros(E), persistent=True)
        self.register_buffer("load", torch.zeros(E), persistent=False)

        self.w_gate = nn.Parameter(torch.empty(E, d, I))
        self.w_up = nn.Parameter(torch.empty(E, d, I))
        self.w_down = nn.Parameter(torch.empty(E, I, d))
        nn.init.normal_(self.w_gate, std=std)
        nn.init.normal_(self.w_up, std=std)
        nn.init.normal_(self.w_down, std=std / math.sqrt(2 * cfg.n_layers))
        self.shared = (SwiGLU(d, cfg.n_shared * cfg.shared_inter, std, std / math.sqrt(2 * cfg.n_layers))
                       if cfg.n_shared > 0 else None)

    # ------------------------------------------------------------------ routing
    def route(self, xf: torch.Tensor):
        with torch.autocast(device_type=xf.device.type, enabled=False):
            scores = torch.sigmoid(F.linear(xf.float(), self.router.weight.float()))  # [N, E]
        return self._select(scores)

    def _select(self, scores: torch.Tensor):
        idx = torch.topk(scores + self.e_bias, self.k, dim=-1).indices                 # select with bias
        w = scores.gather(1, idx)                                                       # weigh without bias
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.routed_scale
        return scores, idx, w

    def seq_aux_loss(self, scores: torch.Tensor, idx: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """DeepSeek-V3 eq. 17-20: alpha * sum_i f_i * P_i, per sequence, averaged over the batch."""
        E, K = self.E, self.k
        s = scores.view(B, T, E)
        P = (s / s.sum(-1, keepdim=True)).mean(1)                                       # [B, E]
        counts = torch.zeros(B, E, device=scores.device).scatter_add_(
            1, idx.view(B, T * K), torch.ones(B, T * K, device=scores.device))
        f = counts * (E / (K * T))
        return self.seq_aux_coef * (f * P).sum(-1).mean()

    @torch.no_grad()
    def update_bias(self) -> dict:
        """Call once per optimizer step. Returns load stats for logging as 0-dim tensors: no host sync (a layer
        that saw no tokens has mean 0 and sign(0) = 0, so its bias stays unchanged, as before)."""
        load = self.load
        mean = load.mean()
        self.e_bias.add_(self.gamma * torch.sign(mean - load))
        stats = {"max_over_mean": load.max() / mean, "min_over_mean": load.min() / mean,
                 "dead": (load == 0).sum()}
        self.load.zero_()
        return stats

    # ------------------------------------------------------------------ experts
    def _loop(self, xs, counts):
        dev = xs.device.type
        wdt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else xs.dtype
        wg, wu, wd = (cast_weight(w, wdt) for w in (self.w_gate, self.w_up, self.w_down))
        outs, start = [], 0
        for e, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            seg = xs[start:start + c]
            start += c
            h = silu_mul(seg @ wg[e], seg @ wu[e])
            outs.append(h @ wd[e])
        return torch.cat(outs, 0) if outs else xs.new_zeros(0, self.d)

    def _bmm(self, xs, e_sorted, counts, C=None, pos=None):
        E = self.E
        if C is None:  # direct calls; experts() reads the capacity back without draining the queue and passes it
            C = capacity_bucket(int(counts.max()))
        if pos is None:
            starts = torch.cumsum(counts, 0) - counts
            pos = torch.arange(xs.shape[0], device=xs.device) - starts[e_sorted]
        buf = xs.new_zeros(E, C, self.d)
        buf.index_put_((e_sorted, pos), xs)  # in place: the fresh buffer has no other users (saves a full copy)
        # weights in the matmul dtype (autocast's, else the activations'): the same casts autocast would do,
        # through cast_weight so their gradients are accumulated in one pass
        dev = buf.device.type
        wdt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else buf.dtype
        h = silu_mul(torch.bmm(buf, cast_weight(self.w_gate, wdt)), torch.bmm(buf, cast_weight(self.w_up, wdt)))
        out = torch.bmm(h, cast_weight(self.w_down, wdt))
        return out[e_sorted, pos]

    def _grouped(self, xs, counts):
        offs = torch.cumsum(counts, 0).to(torch.int32)
        dt = torch.bfloat16
        xb = xs.to(dt)
        h = F.silu(torch._grouped_mm(xb, cast_weight(self.w_gate, dt), offs=offs)) * \
            torch._grouped_mm(xb, cast_weight(self.w_up, dt), offs=offs)
        return torch._grouped_mm(h, cast_weight(self.w_down, dt), offs=offs).to(xs.dtype)

    def _routed(self, xs, e_sorted, counts, C=None, pos=None):
        impl = self.impl
        if impl == "auto":  # measured on the 4090 (sm89): bmm beats torch._grouped_mm by 5-30%
            impl = "bmm"
        if impl == "grouped":
            try:
                ys = self._grouped(xs, counts)
                self._grouped_ok = True
                return ys
            except Exception:
                if self.impl == "grouped":
                    raise
                self._grouped_ok = False
                return self._bmm(xs, e_sorted, counts, C, pos)
        if impl == "bmm":
            return self._bmm(xs, e_sorted, counts, C, pos)
        return self._loop(xs, counts)

    @staticmethod
    def _wdt(x: torch.Tensor) -> torch.dtype:
        dev = x.device.type
        return torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype

    def _expert_weights(self, wdt):
        """(w_gate, w_up, w_down in wdt, sinks): the expert weights as the Triton kernels take them."""
        params = (self.w_gate, self.w_up, self.w_down)
        fused = [fused_copy(p, wdt) for p in params]  # OptimizerBundle(fused_grad_acc=True): gradients go straight
        if all(f is not None for f in fused):         # from the weight-gradient kernels into the fp32 buffers
            (wg, sg), (wu, su), (wd, sd) = fused
            return wg, wu, wd, (sg, su, sd)
        wg, wu, wd = (cast_weight(p, wdt) for p in params)
        return wg, wu, wd, None

    def _triton(self, x, idx, w):
        from .moe_triton import routed_experts
        wdt = self._wdt(x)
        wg, wu, wd, sinks = self._expert_weights(wdt)
        recompute = self.recompute_experts and self.training and torch.is_grad_enabled()
        sort = self._sort(idx)
        # w in the GEMM dtype, as the bmm path multiplies it (ys * w.to(ys.dtype)); output in that dtype too
        return routed_experts(x, idx, w.to(wdt), wg, wu, wd, recompute=recompute, sinks=sinks, sort=sort)

    def _sort(self, idx: torch.Tensor):
        """moe_sort="counting": the expert sort of the assignments (sort_assignments), else None (the experts sort)."""
        if self.sort_impl != "counting" or self.impl != "triton":
            return None
        from .moe_triton import sort_assignments
        return sort_assignments(idx, self.E, "counting")

    def experts(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, counts=None) -> torch.Tensor:
        if self.impl == "triton":
            return self._triton(x, idx, w)
        N, K = idx.shape
        flat = idx.reshape(-1)
        if counts is None:
            counts = expert_counts(flat, self.E)
        # the bmm path needs the capacity on the CPU: start reading it back now and queue the dispatch work
        # below (which does not depend on it) before waiting for it
        cap = MaxReadback(counts) if self.impl in ("auto", "bmm") else None
        order = torch.argsort(flat, stable=True)
        e_sorted = flat[order]
        tok = order // K
        xs = x.index_select(0, tok)
        C = pos = None
        if cap is not None:
            starts = torch.cumsum(counts, 0) - counts
            pos = torch.arange(xs.shape[0], device=xs.device) - starts[e_sorted]
            C = capacity_bucket(cap.get())
        if self.recompute_experts and self.training and torch.is_grad_enabled():
            # backward keeps xs only; the padded expert buffers are rebuilt from it (no randomness inside). The
            # capacity goes in as a plain int, so the recompute does not read it back again
            ys = checkpoint(self._routed, xs, e_sorted, counts, C, pos, use_reentrant=False,
                            preserve_rng_state=False)
        else:
            ys = self._routed(xs, e_sorted, counts, C, pos)
        ys = ys * w.reshape(-1)[order].unsqueeze(-1).to(ys.dtype)
        return torch.zeros(N, self.d, device=x.device, dtype=ys.dtype).index_add_(0, tok, ys)

    # ------------------------------------------------------------------ forward
    def fan_mode(self, xf: torch.Tensor) -> str:
        """The fused fan-in mode for this input (cfg.moe_shared_fused), "" where it does not apply: other impls, inputs
        the Triton kernels do not run on (CPU tensors without Triton's interpreter) and no tokens."""
        mode = self.fan
        if not mode or self.impl != "triton" or xf.shape[0] == 0:
            return ""
        from .moe_triton import kernels_run
        return mode if kernels_run(xf) else ""

    def _forward_fan(self, xf: torch.Tensor, B: int, T: int, mode: str):
        """forward() with the fused fan-in (see the module docstring): the same routing, values and weight gradients.
        Every reader of xf gets its own view of it from the fan node, so each input gradient arrives there separately."""
        from .moe_triton import moe_fan, routed_experts_fan
        sh = self.shared
        if mode == "exact":  # the router reads its own view: its bf16 input gradient joins the fan-in sum last, as before
            xr, *views = moe_fan(xf, None, self.k, sh is not None, exact=True)
            scores, idx, w = self.route(views[-1])
        else:                # the router's F.linear inside the fan node (the same call: the same logits)
            logits, xr, *views = moe_fan(xf, self.router.weight, self.k, sh is not None, exact=False,
                                         lean=mode == "router_lean")
            with torch.autocast(device_type=xf.device.type, enabled=False):
                scores = torch.sigmoid(logits)
            scores, idx, w = self._select(scores)
        sort = self._sort(idx)  # moe_sort="counting": sorted once here, its offsets give the counts
        counts = sort[1][1:] - sort[1][:-1] if sort is not None else expert_counts(idx.reshape(-1), self.E)
        if self.training:
            with torch.no_grad():
                self.load += counts.float()
        aux = (self.seq_aux_loss(scores, idx, B, T) if self.training and self.seq_aux_coef > 0
               else scores.new_zeros(()))
        wdt = self._wdt(xf)
        # the shared expert first (the routed combine adds its output), gate and up on their own views
        shared = sh(views[0], views[1]).to(wdt) if sh is not None else None
        wg, wu, wd, sinks = self._expert_weights(wdt)
        recompute = self.recompute_experts and self.training and torch.is_grad_enabled()
        y = routed_experts_fan(xr, idx, w.to(wdt), wg, wu, wd, shared, recompute=recompute, sinks=sinks, sort=sort)
        return y, aux

    def forward(self, x: torch.Tensor):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        mode = self.fan_mode(xf)
        if mode:
            y, aux = self._forward_fan(xf, B, T, mode)
            return y.view(B, T, d).to(x.dtype), aux
        scores, idx, w = self.route(xf)
        counts = expert_counts(idx.reshape(-1), self.E)  # one sync-free count for the load counter and the experts
        if self.training:
            with torch.no_grad():
                self.load += counts.float()
        aux = (self.seq_aux_loss(scores, idx, B, T) if self.training and self.seq_aux_coef > 0
               else scores.new_zeros(()))
        y = self.experts(xf, idx, w, counts)
        if self.shared is not None:
            y = y + self.shared(xf).to(y.dtype)
        return y.view(B, T, d).to(x.dtype), aux


class DenseFFN(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        std = cfg.init_std
        self.mlp = SwiGLU(cfg.d_model, cfg.dense_inter, std, std / math.sqrt(2 * cfg.n_layers))

    def forward(self, x):
        return self.mlp(x), x.new_zeros((), dtype=torch.float32)
