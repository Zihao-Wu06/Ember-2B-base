"""Optimizers: Muon for matrices, AdamW for the rest, lazy Adam for Engram tables; WSD schedule.

Muon (Moonlight recipe): Nesterov momentum 0.95, 5 Newton-Schulz steps, update scaled by
0.2 * sqrt(max(A, B)) so its RMS matches AdamW and both share one learning rate.
Expert weights [E, A, B] are orthogonalised per expert (batched NS).
"""
from __future__ import annotations

import math
from contextlib import nullcontext as _nullcontext

import torch


# ------------------------------------------------------------------------- Muon
@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, dtype=None) -> torch.Tensor:
    """dtype: the iteration's dtype (default: bf16 on CUDA, fp32 elsewhere; tests pass bf16 to emulate CUDA on CPU)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype) if dtype is not None else (G.to(torch.bfloat16) if G.is_cuda else G.float())
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


MUON_IMPLS = ("loop", "batched", "batched_fused")
MUON_CHUNK_ELEMS = 1 << 25  # elements per stacked Newton-Schulz batch (main_2b: 2 expert tensors, 30 of 1024 x 1024)


def _ns_dtype(t: torch.Tensor) -> torch.dtype:
    """The iteration dtype zeropower_via_newtonschulz5 uses for t (bf16 on CUDA, fp32 elsewhere)."""
    return torch.bfloat16 if t.is_cuda else torch.float32


def _ns_iterations(X: torch.Tensor, steps: int, fused: bool) -> torch.Tensor:
    """The quintic iteration on an oriented (rows <= cols), normalised 2-D or 3-D X. fused=False: exactly the
    expressions of zeropower_via_newtonschulz5. fused=True: B = b*A + c*(A @ A) is one GEMM with b*A in its epilogue
    (fp32 accumulate, one rounding to the iteration dtype instead of four) and X = a*X + B @ X is one GEMM with a*X in
    its epilogue (one rounding instead of three): the same polynomial with fewer roundings, and the separate
    elementwise passes over A and X are gone."""
    a, b, c = 3.4445, -4.7750, 2.0315
    if not fused:
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X
    mm = torch.baddbmm if X.dim() == 3 else torch.addmm
    for _ in range(steps):
        A = X @ X.mT
        B = mm(A, A, A, beta=b, alpha=c)
        X = mm(X, B, X, beta=a)
    return X


@torch.no_grad()
def zeropower_via_newtonschulz5_fused(G: torch.Tensor, steps: int = 5, dtype=None) -> torch.Tensor:
    """zeropower_via_newtonschulz5 with the scalar terms in the GEMM epilogues (_ns_iterations fused=True)."""
    X = G.to(dtype) if dtype is not None else (G.to(torch.bfloat16) if G.is_cuda else G.float())
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    lead = X.shape[:-2]
    if X.dim() > 3:
        X = X.flatten(0, -3)
    X = _ns_iterations(X, steps, fused=True)
    if len(lead) > 1:
        X = X.unflatten(0, lead)
    if transposed:
        X = X.mT
    return X


@torch.no_grad()
def newtonschulz5_stacked(S: torch.Tensor, steps: int = 5, fused: bool = False) -> torch.Tensor:
    """Newton-Schulz of every member S[i] of a stack S = [k, *shape] (already in the iteration dtype) as
    zeropower_via_newtonschulz5 computes it for that member alone: each member's Frobenius norms are taken over its own
    slice (the same reduction over the same layout), the division and the iteration's elementwise ops act on every
    element alone, and the GEMMs run once over all k * prod(batch dims) matrices instead of once per member.
    fused: the epilogue iteration of _ns_iterations. Returns [k, *shape] (a transposed view for wide-first shapes)."""
    X = S
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    norms = torch.stack([x.norm(dim=(-2, -1), keepdim=True) for x in X.unbind(0)])
    X = X / (norms + 1e-7)
    lead = X.shape[:-2]
    if X.dim() > 3:
        X = X.flatten(0, -3)
    X = _ns_iterations(X, steps, fused)
    if len(lead) > 1:
        X = X.unflatten(0, lead)
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True, weight_decay=0.1, ns_steps=5,
                 state_dtype=torch.float32, impl: str = "loop", chunk_elems: int = MUON_CHUNK_ELEMS):
        """impl: "loop" (one Newton-Schulz per parameter, the original code), "batched" (parameters of the same shape
        stacked into chunks of at most chunk_elems elements, one Newton-Schulz per chunk; per element the same ops as
        "loop") or "batched_fused" (batched, with the scalar terms of each iteration in the GEMM epilogues). The state
        (one momentum buffer per parameter) and the checkpoint layout are the same for all three."""
        if impl not in MUON_IMPLS:
            raise ValueError(f"Muon impl {impl!r}: expected one of {MUON_IMPLS}")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))
        self.state_dtype = state_dtype
        self.impl = impl
        self.chunk_elems = int(chunk_elems)
        self._plans: dict = {}  # param group index -> (param ids, chunks)

    def update_params(self) -> list:
        """The parameters in the order step() updates them (OptimizerBundle copies their tensors back in this order):
        the param groups' order for "loop", chunk by chunk for the batched impls."""
        if self.impl == "loop":
            return [p for grp in self.param_groups for p in grp["params"]]
        return [p for gi in range(len(self.param_groups)) for chunk in self._chunks(gi) for p in chunk]

    def _chunks(self, gi: int) -> list:
        """Chunks of param group gi: parameters with the same shape, dtype and device (groups ordered by their first
        parameter, members in parameter order), each group split evenly into the fewest chunks of at most chunk_elems
        elements. A parameter is stacked only if its slice keeps the alignment of a separate tensor (a multiple of 512
        bytes in bf16, so in fp32 too; the plan does not depend on where the parameters are when it is made: with
        offload_master they are still in pinned CPU memory at the first reload), else it is a chunk of its own."""
        params = self.param_groups[gi]["params"]
        key = tuple(id(p) for p in params)
        plan = self._plans.get(gi)
        if plan is not None and plan[0] == key:
            return plan[1]
        groups: dict = {}
        for p in params:
            groups.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
        chunks = []
        for (shape, _, _), ps in groups.items():
            numel = math.prod(shape)
            if len(shape) < 2 or (numel * 2) % 512 or len(ps) == 1:
                chunks.extend([p] for p in ps)
                continue
            n = max(1, math.ceil(len(ps) * numel / self.chunk_elems))
            n = min(n, len(ps))
            size = math.ceil(len(ps) / n)
            chunks.extend(ps[i:i + size] for i in range(0, len(ps), size))
        self._plans[gi] = (key, chunks)
        return chunks

    @torch.no_grad()
    def step(self, wait=None, flat=None):
        """wait(p): called before p is used (OptimizerBundle: its tensors may still be arriving from CPU).
        flat(ps) (batched impls; OptimizerBundle(flat=True)): (gradients, momentum buffers, parameters) of the chunk
        members ps as three 1-D runs that hold exactly the members' tensors back to back in member order, or None; with
        the runs, the per-member elementwise ops become one op per chunk (the same op on every element)."""
        if self.impl != "loop":
            return self._step_batched(wait, fused=self.impl == "batched_fused", flat=flat)
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wait is not None:
                    wait(p)
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p, dtype=self.state_dtype)
                buf = st["buf"]
                buf.mul_(mu).add_(g.to(buf.dtype))
                u = g.add(buf.to(g.dtype), alpha=mu) if group["nesterov"] else buf.to(g.dtype)
                o = zeropower_via_newtonschulz5(u, group["ns_steps"]).to(p.dtype)
                scale = 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(o, alpha=-lr * scale)

    def _buf(self, p):
        st = self.state[p]
        if "buf" not in st:
            st["buf"] = torch.zeros_like(p, dtype=self.state_dtype)
        return st["buf"]

    def _step_batched(self, wait, fused: bool, flat=None) -> None:
        """Chunk by chunk (see _chunks): every member's momentum update with the loop's ops (after wait(p)), its
        Nesterov input written straight into a stack in the Newton-Schulz dtype, one Newton-Schulz for the stack
        (newtonschulz5_stacked), then every member's weight decay and update with the loop's ops. A chunk of one
        parameter runs the loop's code for it (fused: with zeropower_via_newtonschulz5_fused). flat: see step()."""
        ns_one = zeropower_via_newtonschulz5_fused if fused else zeropower_via_newtonschulz5
        for gi, group in enumerate(self.param_groups):
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            nesterov, steps = group["nesterov"], group["ns_steps"]
            for chunk in self._chunks(gi):
                ps = [p for p in chunk if p.grad is not None]
                if not ps:
                    continue
                p0 = ps[0]
                scale = 0.2 * math.sqrt(max(p0.shape[-2], p0.shape[-1]))
                if len(ps) == 1:
                    if wait is not None:
                        wait(p0)
                    g, buf = p0.grad, self._buf(p0)
                    buf.mul_(mu).add_(g.to(buf.dtype))
                    u = g.add(buf.to(g.dtype), alpha=mu) if nesterov else buf.to(g.dtype)
                    o = ns_one(u, steps).to(p0.dtype)
                    if wd:
                        p0.mul_(1 - lr * wd)
                    p0.add_(o, alpha=-lr * scale)
                    continue
                dt = _ns_dtype(p0)
                runs = flat(ps) if flat is not None else None
                if runs is not None:
                    # the loop below, member by member, as one op per chunk: every element gets the same ops
                    if wait is not None:
                        for p in ps:
                            wait(p)
                    G, M, P = runs
                    M.mul_(mu).add_(G.to(M.dtype))
                    S = torch.empty((len(ps), *p0.shape), dtype=dt, device=p0.device)
                    if nesterov:
                        torch.add(G, M, alpha=mu, out=S.view(-1))
                    else:
                        S.view(-1).copy_(M)
                    O = newtonschulz5_stacked(S, steps, fused)
                    del S
                    if wd:
                        P.mul_(1 - lr * wd)
                    P.view(len(ps), *p0.shape).add_(O, alpha=-lr * scale)
                    continue
                S = torch.empty((len(ps), *p0.shape), dtype=dt, device=p0.device)
                for i, p in enumerate(ps):
                    if wait is not None:
                        wait(p)
                    g, buf = p.grad, self._buf(p)
                    buf.mul_(mu).add_(g.to(buf.dtype))
                    if g.dtype == torch.float32:
                        # g + mu * buf in fp32 (the bf16 momentum upcast exactly inside the op), rounded once into the
                        # stack: the values of g.add(buf.to(g.dtype), alpha=mu).to(dt) without its two fp32 temporaries
                        if nesterov:
                            torch.add(g, buf, alpha=mu, out=S[i])
                        else:
                            S[i].copy_(buf)
                    else:
                        S[i].copy_(g.add(buf.to(g.dtype), alpha=mu) if nesterov else buf.to(g.dtype))
                O = newtonschulz5_stacked(S, steps, fused)
                del S
                for i, p in enumerate(ps):
                    if wd:
                        p.mul_(1 - lr * wd)
                    # fp32 parameter: the output is upcast exactly inside the add (the loop adds its fp32 copy)
                    p.add_(O[i] if p.dtype == torch.float32 else O[i].to(p.dtype), alpha=-lr * scale)

    def load_state_dict(self, state_dict) -> None:
        # torch.optim casts floating-point state to the parameter dtype (fp32) when loading, which turned the
        # bf16 momentum of --muon-bf16 into fp32 after every resume (+2.65 GB on main_2b, different math).
        # The saved values are bf16, so casting back is exact.
        super().load_state_dict(state_dict)
        for st in self.state.values():
            if "buf" in st and st["buf"].dtype != self.state_dtype:
                st["buf"] = st["buf"].to(self.state_dtype)


# ------------------------------------------------------------------------- lazy Adam (Engram)
@torch.no_grad()
def sparse_row_adam_(weight, grad, m, v, ids, lr, betas=(0.9, 0.95), eps=1e-8, step=1, wd=0.0):
    """Adam on the given rows only ("lazy Adam"); untouched rows keep their state frozen."""
    if ids.numel() == 0:
        return
    b1, b2 = betas
    g = grad.index_select(0, ids).float()
    mi = m.index_select(0, ids).mul_(b1).add_(g, alpha=1 - b1)
    vi = v.index_select(0, ids).mul_(b2).addcmul_(g, g, value=1 - b2)
    m.index_copy_(0, ids, mi)
    v.index_copy_(0, ids, vi)
    upd = (mi / (1 - b1 ** step)) / (vi.div(1 - b2 ** step).sqrt_().add_(eps))
    w = weight.index_select(0, ids)
    if wd:
        w.mul_(1 - lr * wd)
    weight.index_copy_(0, ids, w.add_(upd.to(w.dtype), alpha=-lr))


def _row_adam_fused(weight, grad, m, v, ids, lr: float, b1: float, b2: float, eps: float, bc1: float, bc2: float):
    gi = grad[ids]
    mi = m[ids] * b1 + gi * (1 - b1)
    vi = v[ids] * b2 + gi * gi * (1 - b2)
    m[ids] = mi
    v[ids] = vi
    weight[ids] = weight[ids] - lr * (mi / bc1) / ((vi / bc2).sqrt() + eps)
    grad[ids] = 0.0


_ROW_ADAM = None


def row_adam_cpu(weight, grad, m, v, ids, lr, betas, eps, step):
    """Lazy Adam for CPU (offloaded) tables; torch.compile fuses it into one pass (~2.3x faster
    on 2M rows). Same math as sparse_row_adam_; also zeroes grad[ids]."""
    global _ROW_ADAM
    if ids.numel() == 0:
        return
    b1, b2 = betas
    args = (weight, grad, m, v, ids, float(lr), b1, b2, eps, 1 - b1 ** step, 1 - b2 ** step)
    if _ROW_ADAM is None:
        try:
            fn = torch.compile(_row_adam_fused, dynamic=True)
            fn(*args)
            _ROW_ADAM = fn
            return
        except Exception as e:  # no C++ compiler etc. -> eager fallback
            print(f"[optim] torch.compile of the Engram row-Adam failed ({type(e).__name__}); using eager")
            _ROW_ADAM = _row_adam_fused
    _ROW_ADAM(*args)


class EngramTableAdam(torch.optim.Optimizer):
    """Lazy Adam for GPU-resident Engram tables (dense .grad; rows with all-zero grad are untouched)."""

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["step"] += 1
                ids = p.grad.ne(0).any(-1).nonzero().squeeze(1)
                sparse_row_adam_(p, p.grad, st["m"], st["v"], ids, group["lr"], group["betas"],
                                 group["eps"], st["step"])


# ------------------------------------------------------------------------- grouping
def split_params(model) -> dict[str, list]:
    groups = {"muon": [], "adamw_decay": [], "adamw_no_decay": [], "engram": []}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        # the name check also covers copies: copy.deepcopy(model) drops attributes like _engram_table
        if getattr(p, "_engram_table", False) or name.startswith("engram.tables."):
            groups["engram"].append(p)
        elif getattr(p, "_attnres", False) or getattr(p, "_no_weight_decay", False) or p.ndim < 2:
            groups["adamw_no_decay"].append(p)
        elif name.startswith(("embed.", "lm_head.")) or name.endswith("router.weight"):
            (groups["adamw_no_decay"] if "router" in name else groups["adamw_decay"]).append(p)
        elif min(p.shape[-2], p.shape[-1]) >= 16 and not name.endswith(("conv.weight", "conv_w")):
            groups["muon"].append(p)
        else:
            groups["adamw_decay"].append(p)
    return groups


class _GradHolder:
    """Stands in for a parameter in clip_grad_norm_ (which only reads .grad)."""
    __slots__ = ("grad",)

    def __init__(self, grad):
        self.grad = grad


class _FlatSpace:
    """Tensors laid out back to back in one 1-D buffer: key -> (offset, shape). The keys come in groups; every group
    starts at a multiple of ALIGN elements (512 bytes in bf16, as a separate allocation would be aligned) and its
    members follow each other without gaps (a Muon chunk's gradients, momentum buffers and masters are each one run).
    The same layout serves buffers of any dtype (offsets are in elements)."""
    ALIGN = 256

    def __init__(self, groups):
        self.slots, off = {}, 0
        self.groups = []    # [(start, end, [keys])] in layout order
        self.group_of = {}  # key -> index into groups
        for grp in groups:
            if not grp:
                continue
            off = -(-off // self.ALIGN) * self.ALIGN
            start = off
            for key, shape in grp:
                if key in self.slots:
                    raise ValueError(f"_FlatSpace: key {key!r} twice")
                self.slots[key] = (off, tuple(shape))
                self.group_of[key] = len(self.groups)
                off += math.prod(shape)
            self.groups.append((start, off, [k for k, _ in grp]))
        self.numel = -(-off // self.ALIGN) * self.ALIGN

    def __contains__(self, key) -> bool:
        return key in self.slots

    def view(self, buf: torch.Tensor, key) -> torch.Tensor:
        off, shape = self.slots[key]
        return buf[off:off + math.prod(shape)].view(shape)

    def span(self, keys) -> tuple | None:
        """(lo, hi): the smallest range of the buffer holding these keys (gaps between groups included), None if none
        of them is in the layout."""
        rng = [(self.slots[k][0], self.slots[k][0] + math.prod(self.slots[k][1])) for k in keys if k in self.slots]
        if not rng:
            return None
        return min(r[0] for r in rng), max(r[1] for r in rng)

    def run(self, buf: torch.Tensor, keys):
        """The 1-D slice of buf that holds exactly these keys back to back in this order, or None."""
        if not keys or any(k not in self.slots for k in keys):
            return None
        lo = off = self.slots[keys[0]][0]
        for k in keys:
            o, shape = self.slots[k]
            if o != off:
                return None
            off += math.prod(shape)
        return buf[lo:off]

    def is_view(self, t: torch.Tensor, buf: torch.Tensor, key) -> bool:
        """t is the view of key in buf (same memory, same shape)."""
        if buf is None or key not in self.slots or t.device != buf.device or t.dtype != buf.dtype:
            return False
        off, shape = self.slots[key]
        return tuple(t.shape) == shape and t.data_ptr() == buf.data_ptr() + off * buf.element_size()


class _Pinned:
    """A _FlatSpace's tensors in pinned CPU memory, in blocks of at most BLOCK bytes instead of one buffer: the pinned
    allocator rounds every allocation up to a power of two (one 5.3 GB buffer would take 8 GB) and Windows / WSL caps
    pinned memory at about half the RAM, so a 2.5 GB-larger total did not fit (switch 19, 11-13 00:57). Every group keeps
    its layout's relative offsets and lies inside one block; a run of groups that is contiguous in both the layout and
    the blocks is one copy (runs()). pin=False: plain CPU memory (tests without CUDA)."""
    BLOCK = 1 << 28  # bytes (256 MiB: a 65536 x 1024 fp32 tensor fills one exactly; a Muon chunk is at most 128 MiB)

    def __init__(self, space: _FlatSpace, dtype: torch.dtype, pin: bool = True, block_bytes: int | None = None):
        self.space, self.dtype = space, dtype
        esz = torch.tensor([], dtype=dtype).element_size()
        be = (block_bytes or self.BLOCK) // esz  # elements per block (a multiple of ALIGN)
        self.delta, cur = [], 0                  # per group: pinned offset - layout offset
        for start, end, keys in space.groups:
            size = end - start
            if size > be:
                raise ValueError(f"_Pinned: a group of {size * esz} bytes does not fit a {be * esz}-byte block")
            lo = -(-cur // space.ALIGN) * space.ALIGN
            if lo // be != (lo + size - 1) // be:  # would straddle a block boundary: start the next block
                lo = -(-lo // be) * be
            self.delta.append(lo - start)
            cur = lo + size
        self.block_elems = be
        nb = max(1, -(-cur // be))
        sizes = [be] * (nb - 1) + [max(space.ALIGN, cur - be * (nb - 1))]
        self.blocks = [torch.empty(n, dtype=dtype, pin_memory=pin) for n in sizes]
        self.nbytes = sum(b.numel() for b in self.blocks) * esz

    def _where(self, key):
        off, shape = self.space.slots[key]
        p = off + self.delta[self.space.group_of[key]]
        return p // self.block_elems, p % self.block_elems, shape

    def view(self, key) -> torch.Tensor:
        b, lo, shape = self._where(key)
        return self.blocks[b][lo:lo + math.prod(shape)].view(shape)

    def is_view(self, t: torch.Tensor, key) -> bool:
        if key not in self.space.slots or t.device.type != "cpu" or t.dtype != self.dtype:
            return False
        b, lo, shape = self._where(key)
        blk = self.blocks[b]
        return tuple(t.shape) == shape and t.data_ptr() == blk.data_ptr() + lo * blk.element_size()

    def runs(self, keys=None) -> list:
        """[(layout lo, layout hi, block, block offset)]: the copies that move these keys' groups (all groups if None),
        consecutive groups merged where they are contiguous in the layout and in one block."""
        gs = range(len(self.space.groups)) if keys is None else sorted({self.space.group_of[k] for k in keys
                                                                        if k in self.space.group_of})
        out = []
        for g in gs:
            start, end, _ = self.space.groups[g]
            b, lo = (start + self.delta[g]) // self.block_elems, (start + self.delta[g]) % self.block_elems
            if out and out[-1][2] == b and out[-1][1] <= start and out[-1][3] + (start - out[-1][0]) == lo:
                out[-1] = (out[-1][0], end, b, out[-1][3])  # the gap between them is the same in both
            else:
                out.append((start, end, b, lo))
        return out

    def group_slice(self, g: int) -> torch.Tensor:
        """Group g's pinned memory (one contiguous slice of one block, the layout's gaps inside the group included)."""
        start, end, _ = self.space.groups[g]
        p = start + self.delta[g]
        b, lo = p // self.block_elems, p % self.block_elems
        return self.blocks[b][lo:lo + end - start]

    def copy_in(self, gpu: torch.Tensor, runs) -> None:
        """gpu[layout ranges] <- the pinned blocks (non-blocking, on the current stream)."""
        for lo, hi, b, blo in runs:
            gpu[lo:hi].copy_(self.blocks[b][blo:blo + hi - lo], non_blocking=True)

    def copy_out(self, gpu: torch.Tensor, runs) -> None:
        """The pinned blocks <- gpu[layout ranges] (non-blocking, on the current stream)."""
        for lo, hi, b, blo in runs:
            self.blocks[b][blo:blo + hi - lo].copy_(gpu[lo:hi], non_blocking=True)


class _GroupTensors:
    """A layout's tensors on the GPU during step(): one tensor per group (a group is one Muon chunk or one tensor, at most
    256 MiB) instead of one buffer for the whole layout, which WSL2 could not map in one piece in the middle of training
    (switch 19, 11-13 01:13: "Failed to create GPU mapping" for 5.3 GB; the per-tensor allocations of the old path never
    exceed 256 MiB). Allocated on the current stream."""

    def __init__(self, space: _FlatSpace, dtype: torch.dtype, device):
        self.space = space
        self.t = [torch.empty(end - start, dtype=dtype, device=device) for start, end, _ in space.groups]

    def _local(self, key):
        g = self.space.group_of[key]
        off, shape = self.space.slots[key]
        return g, off - self.space.groups[g][0], shape

    def view(self, key) -> torch.Tensor:
        g, lo, shape = self._local(key)
        return self.t[g][lo:lo + math.prod(shape)].view(shape)

    def is_view(self, t: torch.Tensor, key) -> bool:
        if key not in self.space.slots:
            return False
        g, lo, shape = self._local(key)
        base = self.t[g]
        return (t.device == base.device and t.dtype == base.dtype and tuple(t.shape) == shape
                and t.data_ptr() == base.data_ptr() + lo * base.element_size())

    def run(self, keys):
        """The 1-D slice holding exactly these keys back to back (all in one group), or None."""
        if not keys or any(k not in self.space.slots for k in keys):
            return None
        g = self.space.group_of[keys[0]]
        if any(self.space.group_of[k] != g for k in keys):
            return None
        start = self.space.groups[g][0]
        lo = off = self.space.slots[keys[0]][0]
        for k in keys:
            o, shape = self.space.slots[k]
            if o != off:
                return None
            off += math.prod(shape)
        return self.t[g][lo - start:off - start]

    def record_stream(self, stream) -> None:
        for t in self.t:
            t.record_stream(stream)


class _FlatGradStore(dict):
    """OptimizerBundle._gbuf with flat=True: an entry that does not exist yet is created as the view of the persistent
    fp32 gradient buffer (zeroed by zero_grad) instead of a fresh tensor; GradSink.buffer() uses zero_view() for the
    zeros it would create, the copies' hooks copy the first gradient into it (what g.float() gives)."""

    def __init__(self, space: _FlatSpace, buf: torch.Tensor):
        super().__init__()
        self.space, self.buf = space, buf

    def zero_view(self, key):
        """The (zero) view of key, registered as its buffer; None if key is not in the layout."""
        if key not in self.space:
            return None
        v = self[key] = self.space.view(self.buf, key)
        return v


class _StateSpace:
    """The offloaded optimizer states of one dtype (OptimizerBundle(flat=True)): their layout, the pinned buffer that
    holds them between steps, the GPU buffer during step(), and per entry (key, id(param), state dict, state key)."""
    __slots__ = ("space", "dtype", "pin", "gpu", "entries", "by_pid")

    def __init__(self, space: _FlatSpace, dtype: torch.dtype, entries: list):
        self.space, self.dtype, self.entries = space, dtype, entries
        self.pin = _Pinned(space, dtype)  # pinned blocks (see _Pinned)
        self.gpu = None
        self.by_pid: dict = {}
        for key, pid, _, _ in entries:
            self.by_pid.setdefault(pid, []).append(key)


class OptimizerBundle:
    """One object that steps every optimizer with a shared, scheduled learning rate."""

    def __init__(self, model, lr=2e-3, weight_decay=0.1, betas=(0.9, 0.95), engram_lr_mult=5.0,
                 muon_state_dtype=torch.float32, offload_states: bool = False, offload_master: bool = False,
                 async_engram: bool = True, fused_grad_acc: bool = False, adamw_last: bool = False,
                 muon_impl: str = "loop", muon_chunk_elems: int = MUON_CHUNK_ELEMS, engram_early: bool = False,
                 state_min_bytes: int = 0, engram_native: bool = False, engram_native_verify: bool = False,
                 flat: bool = False):
        self.model = model
        # flat (train.py --opt-flat; with offload_master, offload_states and fused_grad_acc): the fp32 masters, their fp32
        # gradient buffers and their persistent bf16 copies each live in ONE buffer laid out in update order (_FlatSpace:
        # the AdamW tensors first, then Muon chunk by chunk, a chunk's members back to back), and so do the offloaded
        # optimizer states, one buffer per dtype. Each step then moves them with one copy per buffer and update segment
        # (the non-Muon optimizers' tensors, then each Muon chunk) and one event per segment instead of a copy and an
        # event per tensor, writes them back with one copy per buffer, refreshes the bf16 copies with one copy, and
        # Muon's per-member momentum update, Nesterov input, weight decay and update become one op per chunk. Every
        # element gets the same ops on the same values: bit-identical results; on main_2b ~1300 copies, ~1000 event
        # records and waits and ~2000 small kernels fewer per step (host time, ~0.1 ms each under WSL2). The checkpoint
        # holds the same tensors (views of the flat buffers; torch.save stores each buffer once). begin() switches it on.
        self.flat_requested = flat
        self.flat = False
        self._fspace = None    # _FlatSpace of the cast parameters: masters, gradient buffers and bf16 copies
        self._fgrad = None     # persistent fp32 gradient buffer (GPU)
        self._fcopy = None     # persistent bf16 copies (GPU)
        self._fpin = None      # fp32 masters between steps (_Pinned blocks)
        self._fgpu = None      # fp32 masters during step() (GPU), freed after the write-back
        self._sspaces = None   # offloaded optimizer states: [_StateSpace] (one per dtype), built at the first offload
        self._rspace = None    # Muon momentum that stays on the GPU (state_min_bytes): _FlatSpace, False if none
        self._rbuf = None      # its persistent GPU buffer
        self._segments = None  # [[id(param), ...]] update segments, in update order
        self._waited = set()   # events this step's compute stream already waits on
        self._flat_xfer = False  # flat transfers on (off again if the pinned blocks cannot be allocated)
        # engram_early (train.py --engram-early): the CPU Adam update of the offloaded Engram tables is queued at the
        # start of the optimizer phase (start_engram_update(), which train.py calls before clip_grad_norm; step()
        # queues it first thing otherwise) instead of at the end of step(). It reads only the tables' own gradient,
        # moments, the learning rate and the step count, none of which the GPU updates touch, so the values are the
        # same; it now runs while the host queues the GPU optimizer work (~0.8 s on main_2b under WSL2) instead of
        # after it, when the next forward had to wait for it (~0.4 s of idle GPU per step).
        self.engram_early = engram_early
        self._engram_queued = False
        # state_min_bytes (train.py --state-min-bytes, with offload_states): optimizer state tensors smaller than this
        # stay on the GPU instead of travelling to pinned memory and back every step (main_2b: ~660 small AdamW
        # moments; each copy costs a fixed ~0.1 ms of host time under WSL2 whatever its size). Same values, same
        # updates, same checkpoint layout; 0 = offload every state (the original behaviour).
        self.state_min_bytes = int(state_min_bytes)
        # muon_impl (train.py --muon-impl): "loop" = one Newton-Schulz per Muon parameter (the original code);
        # "batched" = the parameters of one shape stacked into chunks (Muon.chunk_elems elements), one Newton-Schulz
        # per chunk, per element the same ops as the loop; "batched_fused" = batched with the scalar terms of each
        # Newton-Schulz iteration in the GEMM epilogues (fewer roundings). With the batched impls the reload below
        # copies the Muon parameters back chunk by chunk (Muon.update_params()), so each chunk can start as soon as
        # its own members have arrived. Same states and checkpoint layout for all impls.
        if muon_impl not in MUON_IMPLS:
            raise ValueError(f"muon_impl {muon_impl!r}: expected one of {MUON_IMPLS}")
        self.muon_impl = muon_impl
        # adamw_last: step() copies back and updates the Muon parameters first and the AdamW / Engram-table ones last
        # (default: AdamW first). With offload_states the AdamW moments of the embedding and LM head (~1.08 GB on
        # main_2b) were the first copies, and the compute stream idled until all of them had arrived (~45 ms at
        # ~24 GB/s) before its first kernel; Muon's first parameter needs ~6 MB. Every update reads only its own
        # parameter, gradient and state: same ops on the same values, only the order changes. optimizers() (and so
        # the checkpoint layout) is unchanged.
        self.adamw_last = adamw_last
        # offload_states: keep the GPU optimizer states (Muon momentum, AdamW moments; ~3.75 GB on main_2b) in
        # pinned CPU memory between optimizer steps. They are only used inside step(), so the forward/backward
        # passes get that memory back; values are copied exactly, the math does not change.
        self.offload_states = offload_states
        self._pinned: dict = {}  # (optimizer, param, key) -> reusable pinned CPU buffer
        self._moved: list = []   # (id(param), state dict, key, device) currently offloaded
        self._stream = None      # side stream for the device->host copies (CUDA only)
        # offload_master: the fp32 weights of every parameter that goes through cast_weight (linear layers and
        # expert weights: ~1.33B of main_2b's 1.47B GPU parameters) stay in pinned CPU memory between steps;
        # the forward uses persistent bf16 copies, refreshed once per step instead of cast in every micro-batch,
        # and their gradients are accumulated into fp32 buffers. Same values (exact casts, same fp32 adds and
        # updates); ~2.7 GB less GPU memory. begin() switches it on (after any resume).
        self.offload_master = offload_master
        # fused_grad_acc (with offload_master): the weight-gradient GEMMs of the bias-free linear layers and of the
        # Triton routed experts write their fp32 result straight into these gradient buffers (GEMM epilogue
        # out += dW, layers/common.py GradSink) instead of a bf16 gradient that the hook then adds. Each micro-batch's
        # gradient is no longer rounded to bf16 before the fp32 add (rounding-level change, closer to the exact sum;
        # activations and their gradients are unchanged) and the separate add kernels go away. Other weights (the
        # bias, experts with moe_impl != "triton") keep the hook. Without offload_master it does nothing.
        self.fused_grad_acc = fused_grad_acc
        # async_engram: the CPU Adam update of the offloaded Engram tables runs on the Engram worker thread,
        # queued before the next prefetches and gradient scatters (FIFO), overlapping the GPU optimizer step
        self.async_engram = async_engram
        self._cast: dict = {}     # id(param) -> (param, bf16 copy)
        self._groups: list = []   # (params, stacked bf16 copy) of the fused projection groups (cfg.fuse_proj)
        self._gbuf: dict = {}     # id(param) -> fp32 gradient accumulated over the micro-batches
        self._mpinned: dict = {}  # id(param) -> reusable pinned CPU buffer for the fp32 master
        self._mmoved: list = []   # (param, device) whose fp32 master is in pinned CPU memory
        self.base_lr = lr
        self.engram_lr_mult = engram_lr_mult
        self.betas = betas
        g = split_params(model)
        self.groups = {k: len(v) for k, v in g.items()}
        self.muon = Muon(g["muon"], lr=lr, weight_decay=weight_decay, state_dtype=muon_state_dtype, impl=muon_impl,
                         chunk_elems=muon_chunk_elems) if g["muon"] else None
        adam_groups = [{"params": g["adamw_decay"], "weight_decay": weight_decay},
                       {"params": g["adamw_no_decay"], "weight_decay": 0.0}]
        adam_groups = [x for x in adam_groups if x["params"]]
        fused = all(p.is_cuda for x in adam_groups for p in x["params"])
        self.adamw = torch.optim.AdamW(adam_groups, lr=lr, betas=betas, eps=1e-8, fused=fused) if adam_groups else None
        self.engram_gpu = EngramTableAdam(g["engram"], lr=lr * engram_lr_mult, betas=betas) if g["engram"] else None
        eng = getattr(model, "engram", None)
        self.engram_cpu = eng.cpu_tables if (eng is not None and eng.offload) else None
        self.step_count = 0
        # engram_native (train.py --engram-native): the offloaded tables' lazy Adam runs in row_adam_native's C kernel
        # instead of the torch.compile one. Bit-identical values; the kernel releases the GIL, which the torch.compile
        # kernel holds for most of its run time, so with engram_early the update really overlaps the host's queuing of
        # the GPU optimizer work. Falls back to the torch.compile kernel when the C kernel cannot be built.
        # engram_native_verify (tests): every update also runs the torch.compile kernel on copies of the touched rows
        # and compares the results bit for bit.
        self.engram_native, self.engram_native_verify = False, engram_native_verify
        if engram_native and self.engram_cpu is not None:
            from . import row_adam_native as rn
            self.engram_native = rn.available()
            if not self.engram_native:
                print(f"[optim] native Engram row Adam unavailable ({rn.why_unavailable()}); using the torch.compile "
                      "kernel", flush=True)

    def optimizers(self):
        return [o for o in (self.muon, self.adamw, self.engram_gpu) if o is not None]

    def update_order(self) -> list:
        """The optimizers in the order step() updates them and _reload_pipelined copies their tensors back."""
        rest = [o for o in (self.adamw, self.engram_gpu) if o is not None]
        muon = [self.muon] if self.muon is not None else []
        return muon + rest if self.adamw_last else rest + muon

    @staticmethod
    def _update_params(o) -> list:
        """o's parameters in the order its step() uses them (Muon: Muon.update_params(), chunk order when batched)."""
        if isinstance(o, Muon):
            return o.update_params()
        return [p for grp in o.param_groups for p in grp["params"]]

    def set_lr(self, lr: float) -> None:
        for o in (self.muon, self.adamw):
            if o is not None:
                for grp in o.param_groups:
                    grp["lr"] = lr
        if self.engram_gpu is not None:
            for grp in self.engram_gpu.param_groups:
                grp["lr"] = lr * self.engram_lr_mult
        self.lr = lr

    def offload(self) -> None:
        """Copy every GPU optimizer state tensor into pinned CPU memory and drop the GPU copy (no-op unless
        offload_states). The copies run on a side stream after the optimizer step, so the next step's compute
        does not wait for them; record_stream keeps the freed GPU memory from being reused before its copy
        has finished, and reload() waits for the copies before reading the buffers back."""
        if not self.offload_states:
            return
        if self.flat:
            self._flat_resident()  # on the compute stream, before the side stream takes over
        stream = None
        if torch.cuda.is_available() and any(torch.is_tensor(v) and v.is_cuda for o in self.optimizers()
                                             for st in o.state.values() for v in st.values()):
            stream = self._side_stream()
            stream.wait_stream(torch.cuda.current_stream())  # the states are final once the step is done
        with torch.cuda.stream(stream) if stream is not None else _nullcontext():
            if self.flat and self._flat_xfer and stream is not None:
                try:
                    self._flat_offload_states(stream)
                except RuntimeError as e:  # pinned memory exhausted while laying out (before anything moved)
                    self._flat_xfer, self._sspaces = False, None
                    print(f"[optim] flat transfers off, pinned states not allocated ({str(e).splitlines()[0][:120]}); "
                          "per-tensor copies instead", flush=True)
            self._offload_all(stream)  # flat: only what is not in the layouts (still on the GPU)

    def _offload_all(self, stream) -> None:
        for o in self.optimizers():
            for p, st in o.state.items():
                for k, v in list(st.items()):
                    if not (torch.is_tensor(v) and v.is_cuda):
                        continue
                    if v.numel() * v.element_size() < self.state_min_bytes:
                        continue  # small state: stays on the GPU (state_min_bytes)
                    key = (id(o), id(p), k)
                    buf = self._pinned.get(key)
                    if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                        buf = self._pinned[key] = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                    buf.copy_(v, non_blocking=True)
                    if stream is not None:
                        v.record_stream(stream)
                    st[k] = buf
                    self._moved.append((id(p), st, k, v.device))

    def _side_stream(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    # ---------------------------------------------------------------- fp32 masters in CPU memory
    def begin(self) -> None:
        """Call once before the first training step, after any resume (offload_master): create the bf16
        copies (with fused_grad_acc each one also gets the GradSink of its fp32 gradient buffer) and move the fp32
        masters to pinned CPU memory (CUDA; on CPU only the copies are used)."""
        if not self.offload_master or self._cast:
            return
        from .layers import common
        # fused projection groups (cfg.fuse_proj): one stacked bf16 copy per group whose row blocks are the members'
        # copies (leaves that alias it, so _refresh_copies writes it in place), and with fused_grad_acc one stacked
        # fp32 gradient buffer per group whose row blocks are the members' _gbuf entries (common.GroupGradSink)
        groups = self._stackable_groups()
        if self.flat_requested:
            self._flat_init(groups)  # flat=True: the copies and gradient buffers become views of two buffers
        block = {}
        for params, stacked in groups:
            a = 0
            for p in params:
                block[id(p)] = stacked[a:a + p.shape[0]]
                a += p.shape[0]
        for p in self.model.cast_params():
            v = block.get(id(p))
            if v is None and self.flat:
                c = self._fspace.view(self._fcopy, id(p)).detach()  # a leaf that shares the flat copy buffer
                c.copy_(p.detach())  # the same bf16 rounding as .to(torch.bfloat16)
            elif v is None:
                c = p.detach().to(torch.bfloat16)
            else:
                c = v.detach()  # a leaf that shares the stacked copy's memory (and version counter)
                c.copy_(p.detach())  # the same bf16 rounding as .to(torch.bfloat16)
            c.requires_grad_(True)
            c.register_post_accumulate_grad_hook(self._accumulate_hook(id(p)))
            # the copy stays a leaf that requires grad (activation offloading skips it); with a sink, the fused
            # paths return no gradient for it, so its hook only fires for the uses that are not fused
            sink = common.GradSink(self._gbuf, id(p), p.shape, c.device) if self.fused_grad_acc else None
            common.set_cached_copy(p, c, sink)
            self._cast[id(p)] = (p, c)
        per_member = getattr(getattr(self.model, "cfg", None), "fuse_proj_dw", "stacked") == "member"
        for params, stacked in groups:
            # a leaf that requires grad (so the fused GEMM's backward always runs; activation offloading skips it);
            # without fused_grad_acc its bf16 gradient goes to the members' buffers row block by row block
            stacked.requires_grad_(True)
            stacked.register_post_accumulate_grad_hook(self._group_hook(params))
            sink = (common.GroupGradSink(self._gbuf, [id(p) for p in params], [p.shape for p in params], stacked.device,
                                         per_member=per_member) if self.fused_grad_acc else None)
            common.set_cached_group(params, stacked, sink)
        self._groups = groups
        self._offload_masters()
        if self.flat_requested and self.offload_states and torch.cuda.is_available():
            # load_state_dict left the loaded states on the GPU (flat requested): into the layout now (or tensor by
            # tensor when the layout is off). States already offloaded tensor by tensor go back to the GPU first.
            if self.flat and self._moved:
                self.reload()
                self._pinned.clear()
            self.offload()
            if self.flat:
                self._release_caches()

    def _release_caches(self) -> None:
        """End of begin() with the flat layout: wait for the copies, then return the freed GPU memory (the loaded
        per-tensor masters and states) and the cached pinned blocks to the driver, so that training starts from the
        memory it needs. WSL2 reports running out of GPU or pinned memory as "device not ready" (the driver cannot make
        an allocation resident), which PyTorch does not treat as out-of-memory: it neither frees its cache nor retries."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()

    # ---------------------------------------------------------------- flat layout (flat=True)
    def _flat_init(self, groups) -> None:
        """begin() with flat=True: the layout of the cast parameters in update order (_flat_segments: every tensor of the
        non-Muon optimizers an aligned group of its own, each Muon chunk one group with its members back to back) and,
        in it, the persistent fp32 gradient buffer (zeroed; _gbuf becomes a _FlatGradStore over it) and bf16 copies.
        The masters move into the layout at _offload_masters, the states at the first offload. Sets self.flat."""
        params = self.model.cast_params()
        why = None
        if not (self.offload_states and self.fused_grad_acc):
            why = "needs offload_states and fused_grad_acc"
        elif not torch.cuda.is_available() or not params or any(not p.is_cuda for p in params):
            why = "needs the cast parameters on a CUDA device"
        elif groups:
            why = "fused projection groups (cfg.fuse_proj) are not supported"
        elif self.muon is not None and self.muon.impl == "loop":
            why = "needs a batched Muon impl"
        elif len({p.device for p in params}) != 1 or len({id(p) for p in params}) != len(params):
            why = "cast parameters on several devices or listed twice"
        if why is not None:
            print(f"[optim] flat layout off: {why}", flush=True)
            return
        segs = self._flat_segments()
        by_id = {id(p): p for p in params}
        layout, seen = [], set()
        for pids, chunk in segs:
            members = [pid for pid in pids if pid in by_id and pid not in seen]
            seen.update(members)
            if chunk:
                layout.append([(pid, by_id[pid].shape) for pid in members])
            else:
                layout.extend([(pid, by_id[pid].shape)] for pid in members)
        layout.extend([(id(p), p.shape)] for p in params if id(p) not in seen)
        sp = _FlatSpace(layout)
        dev = params[0].device
        self._fspace, self._segments = sp, segs
        self._fgrad = torch.zeros(sp.numel, dtype=torch.float32, device=dev)
        self._fcopy = torch.empty(sp.numel, dtype=torch.bfloat16, device=dev)
        self._gbuf = _FlatGradStore(sp, self._fgrad)
        self.flat = True
        self._flat_xfer = True

    def _flat_segments(self) -> list:
        """[(param ids, is_chunk)] in update order: each Muon chunk is a segment (is_chunk: more than one member), every
        other optimizer's parameters one segment."""
        segs = []
        for o in self.update_order():
            if isinstance(o, Muon) and o.impl != "loop":
                for gi in range(len(o.param_groups)):
                    for chunk in o._chunks(gi):
                        segs.append(([id(p) for p in chunk], len(chunk) > 1))
            else:
                segs.append(([id(p) for p in self._update_params(o)], False))
        return segs

    def _flat_state_spaces(self) -> list:
        """One _StateSpace per dtype for the GPU states that offload moves (state_min_bytes and up), in update order:
        each Muon chunk's states one group per dtype (its momentum buffers back to back), every other state its own."""
        owner = {id(p): (o, p) for o in self.update_order() for p in self._update_params(o)}
        layouts: dict = {}  # dtype -> ([groups], [entries])
        for pids, chunk in self._segments:
            per_dtype: dict = {}
            for pid in pids:
                o, p = owner[pid]
                st = o.state.get(p)
                if not st:
                    continue
                for k, v in st.items():
                    if not (torch.is_tensor(v) and v.is_cuda) or v.numel() * v.element_size() < self.state_min_bytes:
                        continue
                    key = (id(o), pid, k)
                    groups, entries = layouts.setdefault(v.dtype, ([], []))
                    entries.append((key, pid, st, k))
                    if chunk:
                        per_dtype.setdefault(v.dtype, []).append((key, v.shape))
                    else:
                        groups.append([(key, v.shape)])
            for dt, grp in per_dtype.items():
                layouts[dt][0].append(grp)
        return [_StateSpace(_FlatSpace(groups), dt, entries) for dt, (groups, entries) in layouts.items()]

    def _flat_resident(self) -> None:
        """flat: the Muon momentum buffers that stay on the GPU (below state_min_bytes; a chunk's members have one shape,
        so a chunk is all in or all out) go into one persistent GPU buffer, each chunk's members back to back, so Muon's
        chunk ops apply to these chunks too. Built once, when the momentum exists; values copied exactly."""
        if self._rspace is not None or self.muon is None or not self.muon.state:
            return
        groups, members = [], []
        for gi in range(len(self.muon.param_groups)):
            for chunk in self.muon._chunks(gi):
                bufs = [self.muon.state.get(p, {}).get("buf") for p in chunk]
                if (len(chunk) < 2 or any(not torch.is_tensor(b) or not b.is_cuda for b in bufs)
                        or len({b.dtype for b in bufs}) != 1
                        or bufs[0].numel() * bufs[0].element_size() >= self.state_min_bytes):
                    continue  # offloaded chunks are laid out by _flat_state_spaces
                groups.append([((id(self.muon), id(p), "buf"), b.shape) for p, b in zip(chunk, bufs)])
                members += list(zip(chunk, bufs))
        dtypes = {b.dtype for _, b in members}
        if not groups or len(dtypes) != 1:
            self._rspace = False
            return
        sp = _FlatSpace(groups)
        buf = torch.empty(sp.numel, dtype=dtypes.pop(), device=members[0][1].device)
        for p, b in members:
            v = sp.view(buf, (id(self.muon), id(p), "buf"))
            v.copy_(b)
            self.muon.state[p]["buf"] = v
        self._rspace, self._rbuf = sp, buf

    def _flat_offload_masters(self) -> None:
        """_offload_masters for the flat layout: one copy of the whole GPU master buffer into the pinned one when the
        masters are its views (every step), else a copy per tensor into its slot (begin(), after a resume)."""
        sp = self._fspace
        params = [p for p, _ in self._cast.values() if p.is_cuda]
        if not params:
            return
        if self._fpin is None:
            self._fpin = _Pinned(sp, torch.float32)
        gpu = self._fgpu
        whole = gpu is not None and all(gpu.is_view(p.data, id(p)) for p in params)
        stream = self._side_stream()
        stream.wait_stream(torch.cuda.current_stream())  # after the update and the bf16 refresh
        with torch.cuda.stream(stream):
            if whole:
                for g in range(len(sp.groups)):
                    self._fpin.group_slice(g).copy_(gpu.t[g], non_blocking=True)
                gpu.record_stream(stream)  # the GPU memory is reused only after these copies
            else:
                for p in params:
                    self._fpin.view(id(p)).copy_(p.data, non_blocking=True)
                    p.data.record_stream(stream)
        for p in params:
            self._mmoved.append((p, p.device))
            p.data = self._fpin.view(id(p))
        self._fgpu = None

    def _flat_offload_states(self, stream) -> None:
        """The flat part of offload(): states in a _StateSpace go into its pinned buffer, with one copy when they are the
        views of its GPU buffer (every step), else a copy each (the first offload, after a resume). States that are not in
        the layout (created later) are left to _offload_all."""
        if self._sspaces is None:
            self._sspaces = self._flat_state_spaces()
        for ss in self._sspaces:
            live = [(key, pid, st, k) for key, pid, st, k in ss.entries if torch.is_tensor(st.get(k)) and st[k].is_cuda]
            if not live:
                continue
            whole = ss.gpu is not None and len(live) == len(ss.entries) and all(
                ss.gpu.is_view(st[k], key) for key, _, st, k in live)
            if whole:
                for g in range(len(ss.space.groups)):
                    ss.pin.group_slice(g).copy_(ss.gpu.t[g], non_blocking=True)
                ss.gpu.record_stream(stream)
            else:
                for key, _, st, k in live:
                    ss.pin.view(key).copy_(st[k], non_blocking=True)
                    st[k].record_stream(stream)
            for key, pid, st, k in live:
                self._moved.append((pid, st, k, st[k].device))
                st[k] = ss.pin.view(key)
            ss.gpu = None

    def _flat_reload(self) -> dict:
        """_reload_pipelined for the flat layout: the GPU buffers are allocated on the compute stream (they share its
        memory), then, on the side stream in update order, one copy per buffer and segment and one event per segment.
        Falls back to _reload_pipelined when something is not in the layout. Returns {id(param): event}."""
        if not (self._mmoved or self._moved):
            return {}
        sp, spaces = self._fspace, self._sspaces or []
        # steady state: every master and every state of the layouts is offloaded into its slot, and nothing else is
        moved = {(pid, k) for pid, _, k, _ in self._moved}
        masters_ok = (self._fpin is not None and {id(p) for p, _ in self._mmoved} == set(sp.slots)
                      and all(self._fpin.is_view(p.data, id(p)) for p, _ in self._mmoved))
        states_ok = (len(moved) == len(self._moved)
                     and moved == {(pid, k) for ss in spaces for _, pid, _, k in ss.entries}
                     and all(ss.pin.is_view(st[k], key) for ss in spaces for key, _, st, k in ss.entries))
        if not (masters_ok and states_ok):
            self._sspaces = None  # rebuilt at the next offload, with whatever states exist then
            return self._reload_pipelined()
        dev = self._mmoved[0][1] if self._mmoved else self._moved[0][3]
        bufs = []  # (layout, pinned, GPU, keys of a parameter)
        if self._mmoved:
            self._fgpu = _GroupTensors(sp, torch.float32, dev)
            bufs.append((sp, self._fpin, self._fgpu, lambda pid: (pid,) if pid in sp else ()))
        for ss in spaces:
            ss.gpu = _GroupTensors(ss.space, ss.dtype, dev)
            bufs.append((ss.space, ss.pin, ss.gpu, lambda pid, ss=ss: ss.by_pid.get(pid, ())))
        cur, side = torch.cuda.current_stream(), self._side_stream()
        side.wait_stream(cur)  # the destinations are free in stream order (earlier write-backs: same stream)
        events = {}
        with torch.cuda.stream(side):
            for pids, _ in self._segments:
                copied = False
                for space, pin, gpu, keys_of in bufs:
                    for g in sorted({space.group_of[key] for pid in pids for key in keys_of(pid)}):
                        gpu.t[g].copy_(pin.group_slice(g), non_blocking=True)  # one copy per group
                        copied = True
                if copied:
                    ev = torch.cuda.Event()
                    ev.record(side)
                    for pid in pids:
                        events[pid] = ev
        for p, _ in self._mmoved:
            p.data = self._fgpu.view(id(p))
        for ss in spaces:
            for key, _, st, k in ss.entries:
                st[k] = ss.gpu.view(key)
        self._mmoved, self._moved = [], []
        return events

    def _flat_runs(self, ps):
        """Muon's flat(ps): the chunk members' gradient, momentum and master runs, if all three are the members' current
        tensors back to back in member order; else None (the member-by-member code runs)."""
        sp, gpu = self._fspace, self._fgpu
        if gpu is None:
            return None
        keys = [id(p) for p in ps]
        G, P = sp.run(self._fgrad, keys), gpu.run(keys)
        if G is None or P is None:
            return None
        # the momentum: offloaded (a _StateSpace's GPU tensors during the step) or resident (_flat_resident)
        mkeys = [(id(self.muon), id(p), "buf") for p in ps]
        if self._rspace and all(k in self._rspace for k in mkeys):
            M = self._rspace.run(self._rbuf, mkeys)
            m_is = lambda t, k: self._rspace.is_view(t, self._rbuf, k)  # noqa: E731
        else:
            ss = next((s for s in self._sspaces or () if s.gpu is not None and all(k in s.space for k in mkeys)), None)
            if ss is None:
                return None
            M = ss.gpu.run(mkeys)
            m_is = ss.gpu.is_view
        if M is None:
            return None
        for p, mk in zip(ps, mkeys):
            st = self.muon.state.get(p)
            if not (p.grad is not None and sp.is_view(p.grad, self._fgrad, id(p)) and gpu.is_view(p.data, id(p))
                    and st is not None and torch.is_tensor(st.get("buf")) and m_is(st["buf"], mk)):
                return None
        return G, M, P

    def _stackable_groups(self) -> list:
        """[(params, empty stacked bf16 tensor)] for the model's fused projection groups whose members are all cast
        parameters with the same input width, device and dtype (any other group keeps per-member copies, and
        fused_linear stacks their casts in every forward)."""
        fn = getattr(self.model, "fused_proj_groups", None)
        if fn is None:
            return []
        cast, used, out = {id(p) for p in self.model.cast_params()}, set(), []
        for params in fn():
            p0 = params[0]
            ids = [id(p) for p in params]
            if (len(params) < 2 or len(set(ids)) != len(ids) or any(i in used or i not in cast for i in ids)
                    or any(p.dim() != 2 or p.shape[1] != p0.shape[1] or p.device != p0.device or p.dtype != p0.dtype
                           for p in params)):
                continue
            used.update(ids)
            out.append((list(params), torch.empty(sum(p.shape[0] for p in params), p0.shape[1], dtype=torch.bfloat16,
                                                  device=p0.device)))
        return out

    def _group_hook(self, params):
        """Hook of a stacked copy (without fused_grad_acc): each member's row block of the bf16 gradient goes into the
        member's fp32 buffer exactly as the member copy's own hook would add it."""
        members, a = [], 0
        for p in params:
            members.append((id(p), a, a + p.shape[0]))
            a += p.shape[0]

        def hook(s):
            g = s.grad
            if g is None:  # fused accumulation: the GEMM added into the buffers itself and returned no gradient
                return
            for pid, lo, hi in members:
                gb, buf = g[lo:hi], self._gbuf.get(pid)
                if buf is None:
                    self._gbuf[pid] = gb.float()
                else:
                    buf.add_(gb)
            s.grad = None
        return hook

    def _accumulate_hook(self, pid):
        def hook(c):
            g, buf = c.grad, self._gbuf.get(pid)
            if g is None:  # no gradient arrived: AccumulateGrad calls this hook even then (torch 2.15), e.g. after a
                return     # fused GEMM returned None for the copy
            if buf is None:
                zero_view = getattr(self._gbuf, "zero_view", None)  # flat: the slot in the flat gradient buffer
                v = zero_view(pid) if zero_view is not None else None
                if v is not None:
                    v.copy_(g)               # the exact fp32 values g.float() gives
                else:
                    self._gbuf[pid] = g.float()  # what AccumulateGrad stores for the fp32 cast of the first gradient
            else:
                buf.add_(g)                  # fp32 += bf16 in one pass: same values as adding the fp32 copy
            c.grad = None
        return hook

    def _offload_masters(self) -> None:
        if self.flat and self._flat_xfer:
            try:
                return self._flat_offload_masters()
            except RuntimeError as e:  # pinned memory exhausted: the per-tensor path below (the gradient buffers and
                self._flat_xfer = False  # bf16 copies stay flat; nothing was moved before the allocation failed)
                self._fpin = None
                print(f"[optim] flat transfers off, pinned masters not allocated ({str(e).splitlines()[0][:120]}); "
                      "per-tensor copies instead", flush=True)
        cuda = [(pid, p) for pid, (p, _) in self._cast.items() if p.is_cuda]
        if not cuda:
            return
        stream = self._side_stream()
        stream.wait_stream(torch.cuda.current_stream())  # after the update and the bf16 refresh
        with torch.cuda.stream(stream):
            for pid, p in cuda:
                buf = self._mpinned.get(pid)
                if buf is None:
                    buf = self._mpinned[pid] = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
                buf.copy_(p.data, non_blocking=True)
                p.data.record_stream(stream)  # the GPU memory is reused only after this copy
                self._mmoved.append((p, p.device))
                p.data = buf

    def _reload_masters(self) -> None:
        if not self._mmoved:
            return
        torch.cuda.current_stream().wait_stream(self._side_stream())  # the offload copies have landed
        for p, dev in self._mmoved:
            p.data = p.data.to(dev, non_blocking=True)
        self._mmoved = []

    def _refresh_copies(self) -> None:
        with torch.no_grad():
            sp, gpu = self._fspace, self._fgpu
            if self.flat and gpu is not None and all(gpu.is_view(p.data, pid) and sp.is_view(c, self._fcopy, pid)
                                                     for pid, (p, c) in self._cast.items()):
                for g, (start, end, _) in enumerate(sp.groups):  # a group's copies at once: the same bf16
                    self._fcopy[start:end].copy_(gpu.t[g])        # rounding of every element
                return
            for p, c in self._cast.values():
                c.copy_(p)  # bf16 rounding of the updated master: what casting it in the next forward gives

    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """torch.nn.utils.clip_grad_norm_ over all gradients in parameter order (bit-identical to calling it on
        model.parameters()); with offload_master the cast parameters' gradients are still in their buffers."""
        params = self.model.parameters()
        if not self._cast:
            return torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], max_norm)
        # a forward without autocast uses the fp32 weights directly: their gradients are then in .grad as usual
        grads = [self._gbuf.get(id(p), p.grad) if id(p) in self._cast else p.grad for p in params]
        return torch.nn.utils.clip_grad_norm_([_GradHolder(g) for g in grads if g is not None], max_norm)

    def stable_host_ptrs(self) -> set:
        """data_ptr() of the pinned CPU buffers that currently hold fp32 masters and optimizer states. After sync()
        nothing writes them until the next step() (which reads them back and, at its end, copies the new values into
        the same buffers), so train.py's background checkpoint may write them to disk by reference meanwhile."""
        ptrs = {p.data.data_ptr() for p, _ in self._mmoved}
        ptrs.update(st[k].data_ptr() for _, st, k, _ in self._moved)
        return ptrs

    def sync(self) -> None:
        """Wait for the copies into pinned memory (before reading offloaded tensors on the CPU)."""
        if (self._moved or self._mmoved) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def reload(self) -> None:
        if self._moved and self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)  # the offload copies have landed
        for _, st, k, dev in self._moved:
            st[k] = st[k].to(dev, non_blocking=True)
        self._moved = []

    def _reload_pipelined(self) -> dict:
        """Masters and optimizer states back to the GPU for step(): destinations are allocated on the compute
        stream (they share its memory), the copies run on the side stream in update order (update_order(): AdamW's
        parameters first, then Muon's; with adamw_last the reverse) with one event per parameter, so the updates
        can start on the first parameters while the later ones are still copying. Returns {id(param): event}."""
        if not (self._mmoved or self._moved):
            return {}
        if not torch.cuda.is_available():
            self._reload_masters()
            self.reload()
            return {}
        cur, side = torch.cuda.current_stream(), self._side_stream()
        jobs: dict = {}  # id(param) -> [(install, gpu destination, pinned source)]
        for p, dev in self._mmoved:
            jobs.setdefault(id(p), []).append((("data", p), torch.empty(p.shape, dtype=p.dtype, device=dev), p.data))
        for pid, st, k, dev in self._moved:
            src = st[k]
            jobs.setdefault(pid, []).append((("state", st, k), torch.empty(src.shape, dtype=src.dtype, device=dev), src))
        order = [id(p) for o in self.update_order() for p in self._update_params(o)]
        seen = set(order)
        order += [pid for pid in jobs if pid not in seen]
        side.wait_stream(cur)  # the destinations are free in stream order (earlier offload copies: same stream)
        events = {}
        with torch.cuda.stream(side):
            for pid in order:
                if pid not in jobs:
                    continue
                for _, dst, src in jobs[pid]:
                    dst.copy_(src, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(side)
                events[pid] = ev
        for items in jobs.values():
            for inst, dst, _ in items:
                if inst[0] == "data":
                    inst[1].data = dst
                else:
                    inst[1][inst[2]] = dst
        self._mmoved, self._moved = [], []
        return events

    def _engram_cpu_update(self, lr: float, n: int) -> None:
        """Lazy Adam on the offloaded Engram tables for optimizer step n (async_engram: on the Engram worker thread,
        queued behind the gradient scatters and before the next prefetches)."""
        if self.engram_cpu is None:
            return
        self.model.engram.flush()  # all async gradient scatters have landed
        tables, lr_e = self.engram_cpu, lr * self.engram_lr_mult

        def update():
            for i, t in enumerate(tables):
                ids = t.touched.nonzero().squeeze(1)
                if self.engram_native:
                    self._native_row_adam(i, t, ids, lr_e, n)
                else:
                    row_adam_cpu(t.weight, t.grad, t.m, t.v, ids, lr_e, self.betas, 1e-8, n)
                t.touched.index_fill_(0, ids, False)
        if self.async_engram:
            tables[0]._pending.append(self.model.engram._worker().submit(update))  # flush() waits for it
        else:
            update()

    def _native_row_adam(self, i: int, t, ids, lr_e: float, n: int) -> None:
        from .row_adam_native import row_adam_native
        if not self.engram_native_verify or ids.numel() == 0:
            row_adam_native(t.weight, t.grad, t.m, t.v, ids, lr_e, self.betas, 1e-8, n)
            return
        # verify: the torch.compile kernel on compact copies of the touched rows must give the same bits
        ref = [x.index_select(0, ids) for x in (t.weight, t.grad, t.m, t.v)]
        row_adam_native(t.weight, t.grad, t.m, t.v, ids, lr_e, self.betas, 1e-8, n)
        row_adam_cpu(*ref, torch.arange(ids.numel()), lr_e, self.betas, 1e-8, n)
        for name, x, r in zip(("weight", "grad", "m", "v"), (t.weight, t.grad, t.m, t.v), ref):
            if not torch.equal(x.index_select(0, ids).view(torch.int32), r.view(torch.int32)):
                raise RuntimeError(f"Engram native Adam: {name} of table {i} differs from the torch.compile kernel "
                                   f"at optimizer step {n}")
        print(f"Engram native Adam verified: optimizer step {n}, table {i}, {ids.numel()} rows x {t.weight.shape[1]} "
              "bit-identical to the torch.compile kernel", flush=True)

    def start_engram_update(self, lr: float) -> None:
        """engram_early: queue this step's Engram CPU update now. Call after the last backward of the step (the
        step is known to be finite) and before step(); step() then does not queue it again. No-op otherwise."""
        if self.engram_early and self.engram_cpu is not None and not self._engram_queued:
            self._engram_queued = True
            self._engram_cpu_update(lr, self.step_count + 1)

    def step(self, lr: float) -> None:
        self.set_lr(lr)
        self.step_count += 1
        if self.engram_early and not self._engram_queued:
            self._engram_cpu_update(lr, self.step_count)
        # values arrive asynchronously: wait on a parameter's event before use
        events = self._flat_reload() if (self.flat and self._flat_xfer) else self._reload_pipelined()
        for pid, (p, _) in self._cast.items():
            g = self._gbuf.pop(pid, None)
            if g is not None:  # else the forward ran without autocast and .grad already holds it
                p.grad = g
        waited = set()  # flat: one event per segment, waited on once

        def wait(p):
            ev = events.get(id(p))
            if ev is not None and ev not in waited:
                torch.cuda.current_stream().wait_event(ev)
                waited.add(ev)
        # AdamW (fused over all its tensors) waits for all of its copies, Muon parameter by parameter for each one, so
        # the remaining copies overlap its Newton-Schulz work; the updates are independent, so the order (AdamW first,
        # or last with adamw_last) changes nothing
        for o in self.update_order():
            if o is self.muon:
                o.step(wait=wait if events else None, flat=self._flat_runs if self.flat else None)
                continue
            if events:
                for grp in o.param_groups:
                    for p in grp["params"]:
                        wait(p)
            o.step()
        if self._cast:
            self._refresh_copies()
            for p, _ in self._cast.values():
                p.grad = None
            self._offload_masters()
        self.offload()
        if not self.engram_early:
            self._engram_cpu_update(lr, self.step_count)
        self._engram_queued = False

    def zero_grad(self) -> None:
        for o in self.optimizers():
            o.zero_grad(set_to_none=True)
        self._gbuf.clear()  # in place: the GradSinks of fused_grad_acc hold this dict
        if self.flat:
            self._fgrad.zero_()  # the next step's buffers start as zeros (GradSink) or get the first gradient copied

    def state_dict(self) -> dict:
        self.sync()  # offloaded states: the copies into pinned memory must have landed
        return {"step_count": self.step_count, "opts": [o.state_dict() for o in self.optimizers()]}

    def load_state_dict(self, sd: dict) -> None:
        self._reload_masters()
        self.reload()
        self._sspaces = None  # flat: the loaded states are new tensors in new state dicts; laid out again at offload()
        self._rspace = self._rbuf = None
        self.step_count = sd["step_count"]
        for o, s in zip(self.optimizers(), sd["opts"]):
            o.load_state_dict(s)  # torch.optim puts the states on the parameters' device
        if self._cast:
            self._refresh_copies()  # the model weights may have been loaded too
            self._offload_masters()
        elif self.flat_requested and self.offload_master and self.offload_states and torch.cuda.is_available():
            # before begin() with the flat layout requested: the states stay on the GPU and begin() copies them into
            # the layout. Offloading them here would put them in per-tensor pinned buffers that PyTorch keeps cached
            # after begin() drops them (~3.5 GB of extra pinned memory on main_2b: the crash of switch 19).
            return
        self.offload()


# ------------------------------------------------------------------------- schedule
def wsd_decay_start(total: int, decay_frac: float = 0.2) -> int:
    """First step of the WSD decay phase (train.py also switches to the anneal data mix here)."""
    return total - max(1, int(total * decay_frac))


def wsd_lr(step: int, total: int, peak: float, warmup: int, decay_frac: float = 0.2,
           final_frac: float = 0.0, shape: str = "1-sqrt") -> float:
    """Warmup-Stable-Decay. step is 0-based; returns the LR to use for this step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    start = wsd_decay_start(total, decay_frac)
    decay_steps = total - start
    if step < start:
        return peak
    t = min(1.0, (step - start) / decay_steps)
    f = 1 - math.sqrt(t) if shape == "1-sqrt" else 1 - t
    return peak * (final_frac + (1 - final_frac) * f)
