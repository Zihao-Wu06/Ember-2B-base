"""Full model: embedding -> 24 x [Engram?] + KDA/full-attention + dense/MoE FFN -> norm -> LM head."""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .layers.attention import Attention, CEDAttention
from .layers.attnres import DepthAttn
from .layers.common import CONV_IMPLS, CastLinear, RMSNorm, add_rms_norm
from .layers.engram import EngramBlock, EngramMemory
from .layers.kda import KDA
from .layers.moe import DenseFFN, MoE
from .loss import CE_IMPLS, linear_cross_entropy
from .offload import ActivationOffloader


def depth_attn_fused(*args):
    from .layers.attnres_fused import depth_attn  # Triton is imported only when the fused path runs
    return depth_attn(*args)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int, kind: str):
        super().__init__()
        d = cfg.d_model
        self.kind = kind
        self.norm1 = RMSNorm(d, cfg.norm_eps)
        self.mixer = {"K": KDA, "F": Attention, "C": CEDAttention}[kind](cfg, i)
        self.norm2 = RMSNorm(d, cfg.norm_eps)
        self.ffn = MoE(cfg, i) if cfg.is_moe(i) else DenseFFN(cfg, i)
        self.engram = EngramBlock(cfg, i) if i in cfg.engram_layers else None
        if cfg.residual == "block_attnres":
            # layer 0's mixer only ever sees the embedding (a single source), so it gets no op
            self.res_mix = DepthAttn(d) if i > 0 else None
            self.res_ffn = DepthAttn(d)
            self.attnres_slot = 2 * (i % cfg.attnres_layers_per_block)  # attnres_impl="fused": op slots in the block
        self.ckpt_ffn = False  # MoELLM.set_checkpointing: recompute only the FFN part, keep the mixer's activations

    def _ffn(self, fn, *args):
        if self.ckpt_ffn and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _ffn_attnres(self, p, *blocks):
        return self.ffn(self.norm2(self.res_ffn([*blocks, p])))

    def forward_standard(self, h, e, enc=None):
        # residual add + the following norm in one helper (fused when fusion is on; same math)
        if self.engram is not None:
            h, x = add_rms_norm(h, self.engram(h, e), self.norm1.weight, self.norm1.eps)
        else:
            x = self.norm1(h)
        y = self.mixer(x, enc) if self.kind == "C" else self.mixer(x)
        h, x = add_rms_norm(h, y, self.norm2.weight, self.norm2.eps)
        f, aux = self._ffn(self.ffn, x)
        return h + f, aux

    def forward_block_attnres(self, partial, boundary: bool, e, *blocks):
        srcs = list(blocks) if boundary else [*blocks, partial]
        h = srcs[0] if len(srcs) == 1 else self.res_mix(srcs)
        c = None
        if self.engram is not None:
            c = self.engram(h, e)
            h = h + c
        y = self.mixer(self.norm1(h))
        if c is not None:
            y = y + c                       # Engram output joins the block's partial sum
        p = y if boundary else partial + y  # a new block starts with this layer's output
        f, aux = self._ffn(self._ffn_attnres, p, *blocks)
        return p + f, aux

    # attnres_impl="fused": the same layer; the completed block outputs come from the block state `blk`
    # (layers/attnres_fused.py), x0 is the running partial sum (None at a block start; layer 0: the embedding)
    def forward_block_attnres_fused(self, x0, boundary: bool, e, blk, token, q_mix, q_ffn):
        if self.res_mix is None:
            h = x0  # layer 0: the embedding is the only source
        else:
            h = depth_attn_fused(blk, token, None if boundary else x0, q_mix, self.attnres_slot, self.res_mix.eps)
        c = None
        if self.engram is not None:
            c = self.engram(h, e)
            h = h + c
        y = self.mixer(self.norm1(h))
        if c is not None:
            y = y + c
        p = y if boundary else x0 + y
        f, aux = self._ffn(self._ffn_attnres_fused, p, blk, token, q_ffn)
        return p + f, aux

    def _ffn_attnres_fused(self, p, blk, token, q_ffn):
        return self.ffn(self.norm2(depth_attn_fused(blk, token, p, q_ffn, self.attnres_slot + 1, self.res_ffn.eps)))


class MoELLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.residual not in ("standard", "block_attnres"):
            raise NotImplementedError(f"residual={cfg.residual}")
        if cfg.ced_split and (cfg.residual != "standard" or not 0 < cfg.ced_split < cfg.n_layers):
            raise NotImplementedError("CED needs residual='standard' and 0 < ced_split < n_layers")
        if cfg.ce_impl not in CE_IMPLS:
            raise ValueError(f"ce_impl={cfg.ce_impl!r}; expected one of {CE_IMPLS}")
        if cfg.ce_blas not in ("", "cublas", "cublaslt", "cublaslt_logits"):
            raise ValueError(f"ce_blas={cfg.ce_blas!r}; expected '', 'cublas', 'cublaslt' or 'cublaslt_logits'")
        # ce_blas is process-wide (loss_fused.set_blas): a fused-CE model sets it when it is given, and resets it when
        # loss_fused is already loaded (so a later model never inherits an earlier one's choice; triton is only
        # imported for it when needed)
        if cfg.ce_impl != "chunked" and (cfg.ce_blas or (__package__ + ".loss_fused") in sys.modules):
            from . import loss_fused
            loss_fused.set_blas(cfg.ce_blas)
        if cfg.attnres_impl not in ("compiled", "fused"):
            raise NotImplementedError(f"attnres_impl={cfg.attnres_impl}")
        if cfg.fuse_proj_dw not in ("stacked", "member"):
            raise ValueError(f"fuse_proj_dw={cfg.fuse_proj_dw!r}; expected 'stacked' or 'member'")
        if cfg.conv_impl not in CONV_IMPLS:
            raise ValueError(f"conv_impl={cfg.conv_impl!r}; expected one of {CONV_IMPLS}")
        self.attnres_fused = cfg.residual == "block_attnres" and cfg.attnres_impl == "fused"
        if self.attnres_fused and 2 * cfg.attnres_layers_per_block + 1 > 9:
            raise NotImplementedError("attnres_impl='fused' supports attnres_layers_per_block <= 4")
        self.cfg = cfg
        d, V = cfg.d_model, cfg.vocab_size
        self.embed = nn.Embedding(V, d)
        nn.init.normal_(self.embed.weight, std=cfg.init_std)
        self.enc_norm = RMSNorm(d, cfg.norm_eps) if cfg.ced_split else None  # CED encoder output
        self.engram = EngramMemory(cfg) if cfg.engram_layers else None
        self.blocks = nn.ModuleList([Block(cfg, i, k) for i, k in enumerate(cfg.layer_types())])
        self.out_res = DepthAttn(d) if cfg.residual == "block_attnres" else None
        self.final_norm = RMSNorm(d, cfg.norm_eps)
        self.lm_head = nn.Linear(d, V, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=cfg.init_std)
        self.grad_checkpointing = False
        self.ckpt_layers = None  # None = every layer (when grad_checkpointing); else a set of layer ids
        self.ffn_only: set = set()  # recomputed layers that recompute only their FFN part
        self.offload_act: set = set()  # layers whose saved activations go to CPU memory instead of being recomputed
        self.offloader = None
        self.prefetch_distance = 3  # an offloaded layer's tensors start coming back this many layers above it
        self._prefetch_at: dict = {}
        self.ce_chunk = cfg.ce_chunk
        self._update_kda_keep()

    def set_checkpointing(self, frac: float, expert_recompute: bool = True, ffn_frac: float = 0.0,
                          offload_frac: float = 0.0) -> None:
        """Recompute activations for `frac` of the layers (evenly spaced); 0 = off, 1 = all.
        Layers that are not checkpointed keep their activations: faster, more memory. With
        0 < frac < 1 their routed experts are still recomputed (expert_recompute): the expert buffers
        are padded to the busiest expert, so keeping them makes memory grow with routing imbalance
        (main_2b at 0.75: 18.7 -> 21.5 GB over the first 21 steps, MoE layer max/mean load up to 7.5).
        ffn_frac: that fraction of the recomputed layers (evenly spaced) recompute only their FFN part (depth
        attention, norm, MoE / dense FFN) and keep the activations of their mixer part: a finer step between
        recomputing a whole layer and keeping it. Recompute repeats the same ops: values do not change.
        offload_frac: that fraction of the would-be recomputed layers (evenly spaced) is not recomputed; the
        tensors it saves go to pinned CPU memory in the forward and come back before its backward (moe_llm/
        offload.py; only data movement, values unchanged). ffn_frac then applies to the remaining ones."""
        L = len(self.blocks)
        n = round(frac * L)
        self.grad_checkpointing = n > 0
        self.ckpt_layers = None if n >= L else {int(i * L / n) for i in range(n)} if n else set()
        rec = list(range(L)) if self.ckpt_layers is None else sorted(self.ckpt_layers)
        no = round(offload_frac * len(rec)) if self.grad_checkpointing else 0
        self.offload_act = {rec[int(i * len(rec) / no)] for i in range(no)} if no else set()
        if self.offload_act:
            rec = [i for i in rec if i not in self.offload_act]
            self.ckpt_layers = set(rec)
        self.offloader = ActivationOffloader() if self.offload_act else None
        self._prefetch_at = {}
        for o in sorted(self.offload_act):
            self._prefetch_at.setdefault(min(o + self.prefetch_distance, L - 1), []).append(o)
        nf = round(ffn_frac * len(rec)) if self.grad_checkpointing else 0
        self.ffn_only = {rec[int(i * len(rec) / nf)] for i in range(nf)} if nf else set()
        partial = self.grad_checkpointing and self.ckpt_layers is not None
        for i, blk in enumerate(self.blocks):
            blk.ckpt_ffn = i in self.ffn_only
            if isinstance(blk.ffn, MoE):
                blk.ffn.recompute_experts = bool(expert_recompute and partial and i not in self.ckpt_layers)
        self._update_kda_keep()

    def _recomputes_mixer(self, i: int) -> bool:
        """Training recomputes layer i's mixer in the backward (same condition as _call)."""
        return self.grad_checkpointing and (self.ckpt_layers is None or i in self.ckpt_layers) \
            and i not in self.ffn_only

    def _update_kda_keep(self) -> None:
        """KDA layers of cfg.kda_keep_layers keep chunk_kda's states for the backward only where the layer keeps its
        activations anyway (not recomputed, not offloaded; FFN-only recompute keeps the mixer's). In a recomputed
        layer the first forward's copies would be dropped by the checkpoint and the states only held during its
        recompute, and an offloaded layer would copy them to CPU memory, so there it stays off: those layers run
        exactly as without the option."""
        want = self.cfg.kda_keep_ids()
        for i, blk in enumerate(self.blocks):
            if isinstance(blk.mixer, KDA):
                blk.mixer.keep_states = i in want and not self._recomputes_mixer(i) and i not in self.offload_act

    def kda_keep_list(self) -> list[int]:
        """Layers whose KDA currently keeps chunk_kda's states for the backward."""
        return [i for i, b in enumerate(self.blocks) if isinstance(b.mixer, KDA) and b.mixer.keep_states]

    def cast_params(self) -> list:
        """Parameters that the forward only uses through cast_weight (linear layers, expert weights)."""
        out = []
        for m in self.modules():
            if isinstance(m, CastLinear):
                out += [m.weight] + ([m.bias] if m.bias is not None else [])
            elif isinstance(m, MoE):
                out += [m.w_gate, m.w_up, m.w_down]
        return out

    def fused_proj_groups(self) -> list:
        """cfg.fuse_proj: per KDA / attention layer, the weights its fused projection GEMM stacks, in block order
        (OptimizerBundle.begin gives each group one stacked persistent copy and one stacked fp32 gradient buffer)."""
        return [m.proj_weights() for m in self.modules() if isinstance(m, (KDA, Attention)) and m.fuse_proj]

    def recompute_summary(self) -> str:
        L = len(self.blocks)
        kept = self.kda_keep_list()
        kda = f"; KDA states kept for the backward in layers {kept}" if kept else ""
        if not self.grad_checkpointing:
            return "recompute off" + kda
        if self.ckpt_layers is None:
            return (f"recompute all {L} layers" + (f" (FFN part only in {sorted(self.ffn_only)})" if self.ffn_only else "")
                    + kda)
        keep = [i for i in range(L) if i not in self.ckpt_layers]
        exp = [i for i in keep if isinstance(self.blocks[i].ffn, MoE) and self.blocks[i].ffn.recompute_experts]
        return (f"recompute {L - len(keep)}/{L} layers; activations kept in layers {keep}"
                + (f" (routed experts recomputed in {len(exp)} of them)" if exp else "")
                + (f"; FFN part only in layers {sorted(self.ffn_only)}" if self.ffn_only else "")
                + (f"; activations of layers {sorted(self.offload_act)} offloaded to CPU memory"
                   if self.offload_act else "") + kda)

    # ---------------------------------------------------------------- MoE helpers
    def moe_layers(self) -> list[MoE]:
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    @torch.no_grad()
    def update_router_bias(self) -> dict:
        """Bias update of every MoE layer; load stats as 0-dim tensors (no host sync until they are logged)."""
        stats = [m.update_bias() for m in self.moe_layers()]
        if not stats:
            return {}
        return {"moe_max_over_mean": torch.stack([s["max_over_mean"] for s in stats]).max(),
                "moe_min_over_mean": torch.stack([s["min_over_mean"] for s in stats]).min(),
                "moe_dead_experts": torch.stack([s["dead"] for s in stats]).sum()}

    # ---------------------------------------------------------------- forward
    def _call(self, i, fn, *args):
        train = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        if train and self.offloader is not None and i in self.offload_act:
            with self.offloader.layer(i):
                return fn(*args)
        if train and (self.ckpt_layers is None or i in self.ckpt_layers) and i not in self.ffn_only:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _prefetch_hooks(self, i, out):
        """After layer i: when the backward reaches its output, start bringing back offloaded layers below."""
        if self.offloader is not None and out.requires_grad:
            for o in self._prefetch_at.get(i, ()):
                out.register_hook(self.offloader.prefetch_hook(o))

    def hidden(self, ids: torch.Tensor):
        h = self.embed(ids)
        if self.cfg.residual_bf16 and h.is_cuda:
            h = h.to(torch.bfloat16)
        es = self.engram(ids) if self.engram is not None else {}
        aux = torch.zeros((), dtype=torch.float32, device=ids.device)
        if self.cfg.residual == "standard":
            enc = None
            for i, blk in enumerate(self.blocks):
                if self.enc_norm is not None and i == self.cfg.ced_split:
                    enc = self.enc_norm(h)  # global K/V source for every decoder layer
                h, a = self._call(i, blk.forward_standard, h, es.get(i), enc)
                self._prefetch_hooks(i, h)
                aux = aux + a
        elif self.attnres_fused:
            h, aux = self._hidden_attnres_fused(h, es, aux)
        else:
            lpb = self.cfg.attnres_layers_per_block
            blocks, partial = [], h
            for i, blk in enumerate(self.blocks):
                boundary = i % lpb == 0
                if boundary:
                    blocks.append(partial)  # at i = 0 the embedding becomes b_0
                if self.offloader is not None and i in self.offload_act:
                    self.offloader.keep_on_gpu(blocks)  # later layers still read them: offloading frees nothing
                partial, a = self._call(i, blk.forward_block_attnres, partial, boundary, es.get(i), *blocks)
                self._prefetch_hooks(i, partial)
                aux = aux + a
            h = self.out_res(blocks + [partial])
        return self.final_norm(h), aux

    def _hidden_attnres_fused(self, h, es, aux):
        """Block AttnRes through layers/attnres_fused.py: same layers and values as the loop above (rounding-level).
        Block k opens at its first layer with b_k = the running sum so far (b_0 = the embedding); its ops are the
        res_mix / res_ffn of its layers (slot 2 * offset + 0 / 1) and, in the last block, out_res (slot 2 * lpb)."""
        from .layers.attnres_fused import FusedAttnRes, depth_attn
        lpb, L = self.cfg.attnres_layers_per_block, len(self.blocks)
        st = FusedAttnRes(h, n_blocks=(L + lpb - 1) // lpb, n_slots=2 * lpb + 1)
        partial, blk, token, qs = h, None, None, {}
        for i, layer in enumerate(self.blocks):
            boundary = i % lpb == 0
            if boundary:
                ops = {}
                for j in range(i, min(i + lpb, L)):
                    if self.blocks[j].res_mix is not None:
                        ops[self.blocks[j].attnres_slot] = self.blocks[j].res_mix
                    ops[self.blocks[j].attnres_slot + 1] = self.blocks[j].res_ffn
                if i + lpb >= L:
                    ops[2 * lpb] = self.out_res
                blk, token, qs = st.open_block(i // lpb, partial, ops)
            if self.offloader is not None and i in self.offload_act:
                self.offloader.keep_on_gpu(st.views())
            x0 = None if boundary and i > 0 else partial  # b_k itself is read from the block state
            o = layer.attnres_slot
            partial, a = self._call(i, layer.forward_block_attnres_fused, x0, boundary, es.get(i), blk, token,
                                    qs.get(o), qs.get(o + 1))
            self._prefetch_hooks(i, partial)
            aux = aux + a
        return depth_attn(blk, token, partial, qs[2 * lpb], 2 * lpb, self.out_res.eps), aux

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        h, aux = self.hidden(ids)
        if targets is None:
            return self.lm_head(h)
        loss, ce = linear_cross_entropy(h.flatten(0, 1), self.lm_head.weight, targets.flatten(),
                                        z_coef=self.cfg.z_loss, chunk=self.ce_chunk, impl=self.cfg.ce_impl)
        return loss + aux, {"ce": ce, "aux": aux.detach()}
