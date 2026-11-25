"""Activation offloading (exact): instead of recomputing a layer in the backward, the tensors it saves for its
backward go to pinned CPU memory during the forward and come back shortly before its backward (prefetched when
the backward reaches a later layer). Only data movement: values are restored bit for bit.

On CUDA the copies run on a side stream (the 4090 under WSL2 moves ~24 GB/s over PCIe, both directions
together). Skipped: small tensors (< min_bytes), leaves that require grad (parameters, persistent bf16 weight
copies) and tensors registered with keep_on_gpu() (e.g. the AttnRes block outputs that later layers still
read, so offloading them would free nothing). A tensor saved by several ops of one layer is copied once.
Host buffers are pooled: after the first micro-batch the pinned memory stays constant (one micro-batch of
offloaded tensors). On CPU the same logic runs with ordinary copies (tests).
"""
from __future__ import annotations

import contextlib
from collections import defaultdict

import torch


class _Rec:
    __slots__ = ("host", "gpu", "ready", "shape", "stride", "dtype", "device", "uses")


def _dense(t) -> bool:
    """Non-overlapping and dense (a permutation of a contiguous layout): restorable with the same strides."""
    try:
        from torch._prims_common import is_non_overlapping_and_dense
        return bool(is_non_overlapping_and_dense(t))
    except Exception:
        return t.is_contiguous()


class ActivationOffloader:
    def __init__(self, min_bytes: int = 1 << 20):
        self.min_bytes = min_bytes
        self.stream = None
        self.pool: dict = defaultdict(list)     # (numel, dtype, pinned) -> free host buffers
        self.pending: dict = defaultdict(list)  # layer -> records not fetched back yet
        self.keep: set = set()                  # data_ptrs of tensors that stay on the GPU anyway
        self.stats = {"bytes": 0, "tensors": 0, "late": 0, "host_bytes": 0}
        self._layer = None
        self._dedup: dict = {}

    # ------------------------------------------------------------------ forward side
    @contextlib.contextmanager
    def layer(self, i):
        """Context for the forward of offloaded layer i."""
        self.pending.pop(i, None)  # records of an earlier micro-batch were all fetched by now
        self._layer, self._dedup = i, {}
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            self._layer, self._dedup = None, {}

    def keep_on_gpu(self, tensors) -> None:
        self.keep = {t.data_ptr() for t in tensors}

    def _side(self):
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        return self.stream

    def _host(self, shape, stride, dtype, pinned):
        free = self.pool[(shape, stride, dtype, pinned)]
        if free:
            return free.pop()
        buf = torch.empty_strided(shape, stride, dtype=dtype, pin_memory=pinned)
        self.stats["host_bytes"] += buf.numel() * buf.element_size()
        return buf

    def _pack(self, t):
        base = t._base if t._base is not None else t
        if (t.layout != torch.strided or (base.requires_grad and base.is_leaf) or t.data_ptr() in self.keep
                or t.numel() * t.element_size() < self.min_bytes):
            return t  # small, a parameter / persistent weight copy (or a view of one), or kept on the device
        if not _dense(t):
            return t  # broadcast / gapped views: their layout could not be restored exactly
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype,
               t._version)
        hit = self._dedup.get(key)
        if hit is not None:
            hit[0].uses += 1
            return hit[0]
        rec = _Rec()
        rec.shape, rec.stride, rec.dtype, rec.device = tuple(t.shape), tuple(t.stride()), t.dtype, t.device
        rec.uses, rec.gpu, rec.ready = 1, None, None
        # same strides on the host and back on the device: kernels in the backward see the identical layout
        rec.host = self._host(rec.shape, rec.stride, t.dtype, t.is_cuda)
        if t.is_cuda:
            side = self._side()
            side.wait_stream(torch.cuda.current_stream())  # t has been written
            with torch.cuda.stream(side):
                rec.host.copy_(t, non_blocking=True)
            t.record_stream(side)  # its GPU memory is reused only after this copy
        else:
            rec.host.copy_(t)
        self.pending[self._layer].append(rec)
        # the reference keeps t alive until the layer's forward ends, so its address cannot be reused by
        # another tensor of this layer and mistaken for it
        self._dedup[key] = (rec, t)
        self.stats["bytes"] += t.numel() * t.element_size()
        self.stats["tensors"] += 1
        return rec

    # ------------------------------------------------------------------ backward side
    def _fetch(self, rec):
        if rec.device.type == "cuda":
            cur, side = torch.cuda.current_stream(), self._side()
            # compute-stream memory pool, original strides
            rec.gpu = torch.empty_strided(rec.shape, rec.stride, dtype=rec.dtype, device=rec.device)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                rec.gpu.copy_(rec.host, non_blocking=True)
            rec.ready = torch.cuda.Event()
            rec.ready.record(side)
        else:
            rec.gpu = torch.empty_strided(rec.shape, rec.stride, dtype=rec.dtype).copy_(rec.host)
        # later copies into this buffer are queued on the same side stream, i.e. after this one
        self.pool[(rec.shape, rec.stride, rec.dtype, rec.device.type == "cuda")].append(rec.host)
        rec.host = None

    def prefetch(self, layer) -> None:
        for rec in self.pending.pop(layer, []):
            if rec.gpu is None and rec.host is not None:
                self._fetch(rec)

    def prefetch_hook(self, layer):
        """Tensor hook: prefetch `layer` when the backward reaches the hooked (later) tensor."""
        def hook(grad):
            self.prefetch(layer)
        return hook

    def _unpack(self, rec):
        if not isinstance(rec, _Rec):
            return rec
        if rec.gpu is None:
            self.stats["late"] += 1  # not prefetched: fetched now, the compute stream waits for it
            self._fetch(rec)
        if rec.ready is not None:
            torch.cuda.current_stream().wait_event(rec.ready)
        t = rec.gpu
        rec.uses -= 1
        if rec.uses == 0:
            rec.gpu = None
        return t
