#!/usr/bin/env python
"""Incremental decoding for MoELLM on the Mac (no change to moe_llm/): while a session is active, the time-mixing
parts keep their state between calls and model.hidden() runs on the new tokens only.

  KDA          recurrent state S [B, H, K, V] (fla's naive recurrence with initial_state) and the last 3 inputs of
               its q / k / v short convs
  Attention    the keys and values so far (after QK-norm; NoPE, so no positions are needed; with cfg.attn_logn the
               queries are scaled by the number of keys they see, from the cache length)
  Engram       the last N-1 token ids for the n-gram hashes, and the last (kernel-1)*dilation inputs of its conv
Everything else (embedding, norms, MoE, Block AttnRes, LM head) works per token. The prefill runs through the same
code with empty caches, so a session reproduces the full forward up to floating-point rounding (checked by
check_against_full). Batch rows must have equal lengths (no padding support): generation runs one sequence at a time.

    with decode.session():
        h, _ = model.hidden(prompt_ids)      # prefill
        h, _ = model.hidden(next_ids)        # one or more new tokens per call
"""
import contextlib
import math
import os

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import CausalBias, CausalVariant

import moe_llm.layers.engram as eng
import moe_llm.layers.kda as kda
from moe_llm.layers.attention import Attention
from moe_llm.layers.common import ShortConv, conv_silu

_CACHE = None  # dict while a session is active
GQA_NATIVE_FROM = int(os.environ.get("DECODE_GQA_NATIVE_FROM", 262144))  # cache length in keys
_SDPA = F.scaled_dot_product_attention  # the real one: long_ctx.py --logn swaps decode.F for its log n wrapper
FORCE_LOGN = 0  # long_ctx.py --logn: log n base for padded batch rows of a model without cfg.attn_logn


def causal_lower_right(T, Tk):
    """torch.nn.attention.bias.causal_lower_right(T, Tk) without its stray allocation: CausalBias subclasses Tensor
    without a __new__, so constructing one runs the legacy torch.Tensor(variant, T, Tk) constructor, an uninitialized
    float32 [2, T, Tk] CPU tensor (68.7 GB of address space for T = 32K, Tk = 256K; at T = 64K, Tk = 192K the 103 GB
    pass the overcommit limit and the 256K book losses failed, 2026-11-21). The bias is only (variant, T, Tk)."""
    b = torch.Tensor._make_subclass(CausalBias, torch.empty(0, dtype=torch.bool), False)
    b.variant, b.seq_len_q, b.seq_len_kv = CausalVariant.LOWER_RIGHT, T, Tk
    return b


@contextlib.contextmanager
def session(init=None):
    """init: a cache dict yielded by an earlier session (e.g. after a shared few-shot prefix). Cached tensors are
    never written in place (every update makes new tensors), so a shallow copy is an independent snapshot."""
    global _CACHE
    _patch()
    prev, _CACHE = _CACHE, dict(init or {})
    try:
        yield _CACHE
    finally:
        _CACHE = prev


def _window(key, x, keep):
    """Cached inputs + x along time; the last `keep` positions become the cache."""
    prev = _CACHE.get(key)
    win = x if prev is None else torch.cat([prev, x], 1)
    # a copy, not a view: a view would keep the whole chunk's input alive in the cache (1.7 GB per 32K prompt)
    _CACHE[key] = win[:, -keep:].clone() if keep else win[:, :0].clone()
    return win


_ORIG = {}


def _patch():
    if _ORIG:
        return
    _ORIG.update(sc=ShortConv.forward, kda=kda.KDA.forward, attn=Attention.forward,
                 em=eng.EngramMemory.forward, eb=eng.EngramBlock.forward)

    def sc_forward(self, x):
        if _CACHE is None:
            return _ORIG["sc"](self, x)
        win = _window((id(self), "x"), x, self.weight.shape[-1] - 1)
        return conv_silu(win, self.weight)[:, -x.shape[1]:]

    def kda_forward(self, x):
        if _CACHE is None:
            return _ORIG["kda"](self, x)
        if self.fuse_proj:
            raise NotImplementedError("decode: fuse_proj")
        B, T, _ = x.shape
        H, K = self.H, self.K
        q = self.q_conv(self.q_proj(x)).view(B, T, H, K)
        k = self.k_conv(self.k_proj(x)).view(B, T, H, K)
        v = self.v_conv(self.v_proj(x)).view(B, T, H, K)
        g = self.f_up(self.f_down(x)).view(B, T, H, K)
        beta = self.b_proj(x)
        S0 = _CACHE.get((id(self), "S"))
        if T > 1 and self._use_fla(x):  # GPU prefill with MAC_LM_KDA=auto: fla's chunked kernel, same state layout
            o, S = kda.fla_chunk_kda()(q=q, k=k, v=v, g=g, beta=beta, A_log=self.A_log, dt_bias=self.dt_bias,
                                       initial_state=S0, output_final_state=True, use_qk_l2norm_in_kernel=True,
                                       use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)
            S = S.float()
        else:
            # kda_reference with a carried state
            gk = kda.kda_gate(g, self.A_log, self.dt_bias)
            o, S = kda.naive_recurrent_kda(kda.l2norm(q), kda.l2norm(k), v.float(), gk, torch.sigmoid(beta.float()),
                                           initial_state=S0, output_final_state=True)
        _CACHE[(id(self), "S")] = S
        o = o.to(v.dtype)
        gate = self.g_up(self.g_down(x)).view(B, T, H, K)
        o = kda.gated_rms_norm(o, gate, self.o_norm_weight, self.o_eps)
        return self.o_proj(o.reshape(B, T, self.D))

    def attn_forward(self, x):
        if _CACHE is None:
            return _ORIG["attn"](self, x)
        if self.fuse_proj or self.rope is not None:
            raise NotImplementedError("decode: fuse_proj / RoPE")
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.Hq, self.hd)
        k = self.k_proj(x).view(B, T, self.Hkv, self.hd)
        v = self.v_proj(x).view(B, T, self.Hkv, self.hd)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        pk, pv = _CACHE.get((id(self), "kv"), (None, None))
        if pk is not None:
            k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
        _CACHE[(id(self), "kv")] = (k, v)
        gqa = {}
        if self.Hq != self.Hkv:
            if k.shape[2] > GQA_NATIVE_FROM:  # past 256K: SDPA's own GQA, no 4x copy of the cache (OOM at 512K on 24 GB)
                gqa = {"enable_gqa": True}
            else:  # up to 256K: the path every earlier result used
                rep = self.Hq // self.Hkv
                k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        Tk = k.shape[2]
        pad = _CACHE.get("pad")  # merge_caches(): rows of different prompt lengths, keys left-padded
        if pad is not None:
            if T != 1:
                raise NotImplementedError("decode: padded rows take one token per call")
            base = getattr(self, "logn", 0) or FORCE_LOGN
            if base:  # logn_scale(1, keys this row sees), per row
                n = (Tk - pad).to(torch.float64)
                q = q * torch.clamp(torch.log(n) / math.log(base), min=1.0).to(q.dtype).view(B, 1, 1, 1)
            mask = torch.arange(Tk, device=x.device).view(1, 1, 1, Tk) >= pad.view(B, 1, 1, 1)
            o = _SDPA(q, k, v, attn_mask=mask, **gqa)
        elif getattr(self, "logn", 0) and Tk > self.logn:  # cfg.attn_logn (moe_llm/layers/attention.py logn_scale)
            from moe_llm.layers.attention import logn_scale
            q = q * logn_scale(T, Tk, self.logn, q.device, q.dtype).view(1, 1, T, 1)
        if pad is not None:
            pass
        elif T == Tk:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, **gqa)
        elif T == 1:
            o = F.scaled_dot_product_attention(q, k, v, **gqa)
        else:  # a later prefill chunk: query i sees the cache and its chunk up to i, with no [T, Tk] mask in memory
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(T, Tk), **gqa)
        o = o.transpose(1, 2)
        if self.gate_proj is not None:
            o = eng_sigmoid_mul(o, self.gate_proj(x).view(B, T, self.Hq, self.hd))
        return self.o_proj(o.reshape(B, T, self.Hq * self.hd))

    def em_forward(self, ids):
        if _CACHE is None or self.offload:
            return _ORIG["em"](self, ids)
        win = _window((id(self), "ids"), ids, self.N - 1)
        T = ids.shape[1]
        return {layer: F.embedding(idx[:, -T:], self.tables[li]).flatten(-2)
                for li, (layer, idx) in enumerate(zip(self.layers, self.indices(win)))}

    def eb_forward(self, h, e):
        if _CACHE is None:
            return _ORIG["eb"](self, h, e)
        k = self.w_k(e)
        v = self.w_v(e)
        vt = eng._engram_gate(self.norm_q(h), self.norm_k(k), v, math.sqrt(self.d), self.signed_sqrt)
        win = _window((id(self), "x"), self.norm_c(vt), (self.conv_w.shape[-1] - 1) * self.dilation)
        y = conv_silu(win, self.conv_w, self.dilation)[:, -h.shape[1]:]
        return (y + vt).to(h.dtype)

    ShortConv.forward = sc_forward
    kda.KDA.forward = kda_forward
    Attention.forward = attn_forward
    eng.EngramMemory.forward = em_forward
    eng.EngramBlock.forward = eb_forward


def eng_sigmoid_mul(o, g):
    from moe_llm.layers.common import sigmoid_mul
    return sigmoid_mul(o, g)


@torch.inference_mode()
def check_against_full(model, ids, split=(8, 1, 1, 5, 1)):
    """Max |logit difference| between one full forward and a session fed the same ids in chunks of `split`."""
    full = model(ids)
    outs, a = [], 0
    with session():
        for n in list(split) + [ids.shape[1] - sum(split)]:
            if n <= 0:
                continue
            h, _ = model.hidden(ids[:, a:a + n])
            outs.append(model.lm_head(h))
            a += n
    inc = torch.cat(outs, 1)
    return float((inc - full).abs().max()), float(full.abs().max())


@torch.inference_mode()
def prefill(model, ids, chunk=32768):
    """A cache holding the state after `ids` (for generate_ids(init=...)), fed `chunk` tokens per call (0: all at
    once): the same state up to rounding (check_against_full; fla's TF32 kernel moves 32K logits by ~1e-2), with
    activation memory for `chunk` tokens. On the 4090 in fp32 a 32K call peaks at 11.4 GB, 8K chunks at 10.4 GB."""
    dev = next(model.parameters()).device
    with session() as c:
        for a in range(0, len(ids), chunk or len(ids)):
            model.hidden(torch.tensor([ids[a:a + chunk] if chunk else ids], device=dev))
        return dict(c)


def merge_caches(caches):
    """One batched cache from single-row prefill() caches (emptied as they are merged). Attention keys and values are
    left-padded to the longest row; "pad" (per row) masks the padding and gives each row its own key count for
    attn_logn. Everything else carried is fixed-size per row (KDA state, conv / n-gram windows)."""
    out, pad = {}, None
    for key in list(caches[0]):
        vals = [c.pop(key) for c in caches]
        if isinstance(key, tuple) and key[1] == "kv":
            tk = [kv[0].shape[2] for kv in vals]
            m = max(tk)
            out[key] = tuple(torch.cat([F.pad(kv[i], (0, 0, m - t, 0)) for kv, t in zip(vals, tk)], 0)
                             for i in (0, 1))
            if pad is None:
                pad = torch.tensor([m - t for t in tk], device=vals[0][0].device)
        else:
            out[key] = torch.cat(vals, 0)
    if pad is not None:
        out["pad"] = pad
    return out


@torch.inference_mode()
def batch_rows(n_tokens, rows_at_32k):
    """Rows per generate_batch() call for prompts of n_tokens: the attention cache grows with rows x tokens, so
    shorter prompts take proportionally more rows than `rows_at_32k` and longer ones fewer (decode steps cost about
    the same either way)."""
    return max(1, rows_at_32k * 32768 // max(1, n_tokens))


@torch.inference_mode()
def generate_batch(model, caches, last_ids, max_new, done=None, eos=0):
    """Greedy continuations of several prompts at once (per-token decoding is kernel-launch bound on the GPU, so rows
    are nearly free). caches: prefill() caches of each prompt minus its last token; last_ids: those last tokens.
    done(row, new_ids) -> True: that row needs no more tokens. Each row's ids match generate_ids() up to rounding
    (stopped earlier where done() says so); the call ends when every row has finished."""
    dev = next(model.parameters()).device
    outs, fin = [[] for _ in last_ids], [False] * len(last_ids)
    with session(merge_caches(caches)):
        h, _ = model.hidden(torch.tensor([[t] for t in last_ids], device=dev))
        for _ in range(max_new):
            nxt = model.lm_head(h[:, -1]).float().argmax(-1).tolist()
            for b, t in enumerate(nxt):
                if fin[b]:
                    continue
                if t == eos:
                    fin[b] = True
                    continue
                outs[b].append(t)
                if done is not None and done(b, outs[b]):
                    fin[b] = True
            if all(fin):
                break
            h, _ = model.hidden(torch.tensor([[t] for t in nxt], device=dev))
    return outs


@torch.inference_mode()
def generate_ids(model, prompt_ids, max_new, stop=None, temperature=0.0, eos=0, seed=0, init=None):
    """Greedy (or sampled) continuation token ids; stop(list of new ids) -> True ends early. init: a prefill() cache
    the prompt continues from."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    out = []
    with session(init):
        h, _ = model.hidden(torch.tensor([prompt_ids], device=dev))
        for _ in range(max_new):
            lg = model.lm_head(h[0, -1]).float().cpu()
            if temperature > 0:
                nxt = int(torch.multinomial((lg / temperature).softmax(-1), 1, generator=g))
            else:
                nxt = int(lg.argmax())
            if nxt == eos:
                break
            out.append(nxt)
            if stop is not None and stop(out):
                break
            h, _ = model.hidden(torch.tensor([[nxt]], device=dev))
    return out
