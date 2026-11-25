"""Engram conditional memory (DeepSeek, arXiv 2601.07372), single residual stream (no mHC).

Per Engram layer l, for every position t (Sec. 2.2-2.3 of the paper, official demo for details):
  x'      = canon[x]                       tokenizer compression (NFKC/lower/strip map, see data_prep)
  mix_n   = XOR_{i<n} (x'_{t-i} * m_{l,i})  multiplicative-XOR hash of the suffix n-gram, n in {2, 3}
  z_{n,k} = mix_n mod p_{l,n,k}             K heads per order, each with its own distinct prime table size
  e_t     = concat_{n,k} E_{l,n,k}[z_{n,k}]  d_mem = len(orders) * K * d_head
  k, v    = W_K e_t, W_V e_t
  alpha   = sigmoid( signed_sqrt( RMSNorm(h_t) . RMSNorm(k_t) / sqrt(d) ) )   one scalar per token
  vt      = alpha * v
  Y       = SiLU( DWConv_causal(RMSNorm(vt); kernel 4, dilation max_order) ) + vt   (conv zero-init)
  h      <- h + Y                           at the start of the transformer layer

The table lookups depend only on token ids, so they run ONCE per forward at the model level
(`EngramMemory`), outside activation checkpointing. That is also what makes host-memory offload
work: with `engram_offload=True` the tables live in CPU RAM; each micro-batch gathers only the
unique rows it needs, and a gradient hook scatters the row gradients back into a CPU buffer,
updated by a lazy (touched-rows-only) Adam at each optimizer step (see optim.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CastLinear, RMSNorm, conv_silu, fused


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def primes_from(start: int, count: int) -> list[int]:
    out, n = [], max(2, start)
    while len(out) < count:
        if _is_prime(n):
            out.append(n)
        n += 1
    return out


class OffloadedTable:
    """One Engram layer's flat table in CPU RAM + lazy Adam state (fp32).

    Pipelining (CUDA): `gather` (unique + row gather into pinned memory) can run on a worker
    thread for the NEXT micro-batch while the GPU computes the current one; gradient rows are
    copied back asynchronously and scattered into `grad` on the same worker. `flush()` must be
    called before the optimizer reads `grad` (OptimizerBundle.step does it).
    """

    def __init__(self, rows: int, dim: int, std: float, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(rows, dim, generator=g) * std
        self.grad = torch.zeros(rows, dim)
        self.m = torch.zeros(rows, dim)
        self.v = torch.zeros(rows, dim)
        self.touched = torch.zeros(rows, dtype=torch.bool)
        self.pool = None          # shared single-worker ThreadPoolExecutor (set by EngramMemory)
        self._pending = []

    def gather(self, idx_cpu: torch.Tensor):
        uniq, inv = torch.unique(idx_cpu, return_inverse=True)
        rows = self.weight.index_select(0, uniq)
        if torch.cuda.is_available():  # pinned: lookup's copies to the GPU are asynchronous (no host sync)
            rows = rows.pin_memory()
            inv = inv.pin_memory()
        return uniq, inv, rows

    def lookup(self, idx: torch.Tensor | None, device: torch.device, pre=None) -> torch.Tensor:
        uniq, inv, rows = pre if pre is not None else self.gather(idx.detach().to("cpu"))
        rows = rows.to(device, non_blocking=True) if device.type == "cuda" else rows.clone()
        if torch.is_grad_enabled():
            rows.requires_grad_(True)
            rows.register_hook(self._hook(uniq))
        return rows[inv.to(device, non_blocking=True)]

    def _hook(self, uniq):
        def scatter(g_cpu):
            self.grad.index_add_(0, uniq, g_cpu)
            self.touched[uniq] = True

        def hook(g):
            if g.is_cuda and self.pool is not None:
                buf = torch.empty(g.shape, dtype=torch.float32, pin_memory=True)
                buf.copy_(g.detach(), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()

                def work():
                    ev.synchronize()   # the async device->host copy has landed
                    scatter(buf)
                self._pending.append(self.pool.submit(work))
            else:
                scatter(g.detach().float().cpu())
        return hook

    def flush(self) -> None:
        for f in self._pending:
            f.result()
        self._pending.clear()

    def state_dict(self) -> dict:
        return {"weight": self.weight, "m": self.m, "v": self.v}

    def load_state_dict(self, sd: dict) -> None:
        self.weight.copy_(sd["weight"])
        if "m" in sd:  # a weights-only state (train.py --init-from a slim checkpoint) leaves the moments at zero
            self.m.copy_(sd["m"])
            self.v.copy_(sd["v"])


class EngramMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = tuple(cfg.engram_layers)
        self.orders = tuple(sorted(cfg.engram_orders))
        assert self.orders[0] >= 2, "unigrams are already covered by the token embedding"
        self.N = max(self.orders)
        self.K = cfg.engram_heads
        self.d_head = cfg.engram_d_head
        self.d_mem = cfg.engram_dim
        self.offload = cfg.engram_offload
        self.vocab_size = V = cfg.vocab_size
        self.pad_id = V  # canonical ids live in [0, V); V is reserved for "before sequence start"

        self.register_buffer("canon", torch.arange(V, dtype=torch.long), persistent=True)
        n_heads = len(self.orders) * self.K
        L = len(self.layers)
        bound = ((2 ** 63 - 1) // (V + 1)) // 2            # x' * m can never overflow int64
        if isinstance(cfg.engram_rows, (tuple, list)):     # per-order base sizes, e.g. more rows for 3-grams
            assert len(cfg.engram_rows) == len(self.orders)
            taken, per_order = set(), []
            for base in cfg.engram_rows:
                ps, n = [], int(base)
                while len(ps) < L * self.K:
                    if _is_prime(n) and n not in taken:
                        ps.append(n)
                        taken.add(n)
                    n += 1
                per_order.append(ps)
            allp = [per_order[oi][li * self.K + k] for li in range(L)
                    for oi in range(len(self.orders)) for k in range(self.K)]
        else:
            allp = primes_from(cfg.engram_rows, L * n_heads)  # distinct consecutive primes for every head
        mult, mods, offs, rows = [], [], [], []
        for li, layer in enumerate(self.layers):
            rng = np.random.default_rng(cfg.engram_seed + 10007 * layer)
            mult.append(2 * rng.integers(0, bound, size=self.N, dtype=np.int64) + 1)
            p = np.array(allp[li * n_heads:(li + 1) * n_heads], dtype=np.int64)
            mods.append(p)
            offs.append(np.concatenate([[0], np.cumsum(p)[:-1]]))
            rows.append(int(p.sum()))
        self.register_buffer("mult", torch.tensor(np.stack(mult)), persistent=True)   # [L, N]
        self.register_buffer("mods", torch.tensor(np.stack(mods)), persistent=True)   # [L, n_heads]
        self.register_buffer("offs", torch.tensor(np.stack(offs)), persistent=True)   # [L, n_heads]
        self.rows = rows

        std = cfg.init_std
        self._prefetched = []   # FIFO of per-micro-batch futures (offload mode)
        self._pool = None
        self._cpu_bufs = None   # CPU copies of canon/mult/mods/offs for the prefetch worker (see _cpu_buffers)
        if self.offload:
            self.tables = None
            self.cpu_tables = [OffloadedTable(r, self.d_head, std, cfg.engram_seed + li) for li, r in enumerate(rows)]
        else:
            self.tables = nn.ParameterList()
            for r in rows:
                t = nn.Parameter(torch.randn(r, self.d_head) * std)
                t._engram_table = True
                self.tables.append(t)
            self.cpu_tables = None

    @property
    def num_table_params(self) -> int:
        return sum(self.rows) * self.d_head

    @torch.no_grad()
    def set_canon_map(self, canon) -> None:
        canon = torch.as_tensor(np.asarray(canon), dtype=torch.long)
        assert canon.shape == self.canon.shape and int(canon.max()) < self.vocab_size and int(canon.min()) >= 0
        self.canon.copy_(canon.to(self.canon.device))

    def _cpu_buffers(self):
        """canon/mult/mods/offs on the CPU for the prefetch worker. Cached: copying them from the GPU every
        micro-batch cost four host syncs. The cache is keyed on each buffer's identity, address and version
        counter, so anything that writes or moves them (set_canon_map, load_state_dict, .to()) refreshes it; the
        source buffers are held so their identities and addresses cannot be reused while cached."""
        bufs = (self.canon, self.mult, self.mods, self.offs)
        key = tuple((id(b), b.data_ptr(), b._version) for b in bufs)
        if self._cpu_bufs is None or self._cpu_bufs[0] != key:
            self._cpu_bufs = (key, bufs, tuple(b.cpu() for b in bufs))
        return self._cpu_bufs[2]

    def _worker(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram")
            for t in self.cpu_tables:
                t.pool = self._pool
        return self._pool

    def prefetch(self, ids_cpu: torch.Tensor) -> None:
        """Offload mode: hash + gather the rows of an upcoming micro-batch on the worker thread.
        Forward passes consume prefetched batches in FIFO order (one prefetch per forward)."""
        if not self.offload:
            return
        pool = self._worker()
        bufs = self._cpu_buffers()

        def job():
            idxs = self.indices(ids_cpu, bufs)
            return [t.gather(i) for t, i in zip(self.cpu_tables, idxs)]
        self._prefetched.append((tuple(ids_cpu.shape), pool.submit(job)))

    def flush(self) -> None:
        """Wait for all async gradient scatters. Prefetches still queued now would hold rows from
        before the coming optimizer update, so they are dropped (the forward recomputes them)."""
        if self.cpu_tables is not None:
            for t in self.cpu_tables:
                t.flush()
        for _, fut in self._prefetched:
            fut.result()
        self._prefetched.clear()

    def indices(self, ids: torch.Tensor, bufs=None) -> list[torch.Tensor]:
        """ids [B, T] -> per Engram layer a [B, T, len(orders)*K] tensor of flat table rows."""
        canon, mult_all, mods_all, offs_all = bufs if bufs is not None else (self.canon, self.mult, self.mods, self.offs)
        x = canon[ids]
        B, T = x.shape
        shifted = [x]
        for i in range(1, self.N):
            pad = torch.full((B, min(i, T)), self.pad_id, dtype=x.dtype, device=x.device)
            shifted.append(torch.cat([pad, x[:, :max(T - i, 0)]], dim=1))
        out = []
        for li in range(len(self.layers)):
            m = mult_all[li]
            mix = shifted[0] * m[0]
            cols, h = [], 0
            for n in range(2, self.N + 1):
                mix = mix ^ (shifted[n - 1] * m[n - 1])
                if n in self.orders:
                    for _ in range(self.K):
                        cols.append(mix % mods_all[li, h] + offs_all[li, h])
                        h += 1
            out.append(torch.stack(cols, -1))
        return out

    def forward(self, ids: torch.Tensor) -> dict[int, torch.Tensor]:
        out = {}
        if self.offload and self._prefetched:
            shape, fut = self._prefetched.pop(0)
            assert shape == tuple(ids.shape), "prefetch order does not match the forward passes"
            pres = fut.result()
            for li, layer in enumerate(self.layers):
                out[layer] = self.cpu_tables[li].lookup(None, ids.device, pre=pres[li]).flatten(-2)
            return out
        if self.offload:  # direct lookups (eval, no prefetch): queued table work (an optimizer update) first
            for t in self.cpu_tables:
                t.flush()
        for li, (layer, idx) in enumerate(zip(self.layers, self.indices(ids))):
            if self.offload:
                e = self.cpu_tables[li].lookup(idx, ids.device)
            else:
                e = F.embedding(idx, self.tables[li])
            out[layer] = e.flatten(-2)  # [B, T, d_mem]
        return out

    # offloaded tables are not nn.Parameters: checkpoint them explicitly
    def offload_state_dict(self) -> list | None:
        if self.cpu_tables is None:
            return None
        self.flush()  # a queued optimizer update (OptimizerBundle async_engram) must have finished
        return [t.state_dict() for t in self.cpu_tables]

    def load_offload_state_dict(self, sd) -> None:
        if sd is not None:
            for t, s in zip(self.cpu_tables, sd):
                t.load_state_dict(s)


class EngramBlock(nn.Module):
    """Gating + short conv for one layer; consumes the e_t retrieved by EngramMemory."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        d, dm, std = cfg.d_model, cfg.engram_dim, cfg.init_std
        self.d = d
        self.w_k = CastLinear(dm, d, bias=False)
        self.w_v = CastLinear(dm, d, bias=False)
        nn.init.normal_(self.w_k.weight, std=std)
        nn.init.normal_(self.w_v.weight, std=std)
        self.norm_q = RMSNorm(d, cfg.norm_eps)
        self.norm_k = RMSNorm(d, cfg.norm_eps)
        self.norm_c = RMSNorm(d, 1e-5)
        self.conv_w = nn.Parameter(torch.zeros(d, 1, cfg.engram_conv_size))  # zero init => Y = vt at start
        self.dilation = max(cfg.engram_orders)
        self.signed_sqrt = cfg.engram_signed_sqrt

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        k = self.w_k(e)
        v = self.w_v(e)
        vt = _engram_gate(self.norm_q(h), self.norm_k(k), v, math.sqrt(self.d), self.signed_sqrt)
        return conv_silu(self.norm_c(vt), self.conv_w, self.dilation, vt, h.dtype)  # SiLU(conv) + vt


@fused
def _engram_gate(hq, kk, v, sqrt_d: float, signed_sqrt: bool):
    s = (hq.float() * kk.float()).sum(-1, keepdim=True) / sqrt_d
    if signed_sqrt:
        s = s.sign() * s.abs().clamp_min(1e-6).sqrt()
    return torch.sigmoid(s).to(v.dtype) * v
