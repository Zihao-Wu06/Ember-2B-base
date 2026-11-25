#!/usr/bin/env python
"""Draw the main_2b architecture: docs/architecture.svg, plus docs/architecture.png via rsvg-convert.

Layer layout, widths and expert counts come from moe_llm/config.py (plain Python, no torch needed).
Parameter counts come from moe_llm.budget.param_report when torch is installed, otherwise from the
numbers it reported for main_2b on 2026-11-10 (FALLBACK_CATS).
    python scripts/plot_architecture.py [--out docs/architecture] [--zoom 2]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm.config import main_2b  # noqa: E402

W, H = 1600, 1000
FONT = "Helvetica Neue, Helvetica, Arial, DejaVu Sans, sans-serif"
TEXT, MUTED, LINE = "#1F2328", "#57606A", "#6E7781"
COL = {  # fill, stroke
    "kda": ("#DCEBFA", "#3B7BC8"),
    "attn": ("#FCE3CF", "#D9772B"),
    "moe": ("#DDF2E3", "#3E9B5F"),
    "dense": ("#EEF1E4", "#86995A"),
    "engram": ("#EDE3F7", "#8A5CC2"),
    "attnres": ("#FFF3C4", "#B8901A"),
    "io": ("#F1F3F5", "#7A828C"),
    "panel": ("#FFFFFF", "#C8CDD6"),
    "block": ("#F6F8FA", "#C8CDD6"),
}
# moe_llm.budget.param_report(main_2b())["cats"] on 2026-11-10 (Block AttnRes default), used without torch
FALLBACK_CATS = {"mixer_kda": 85342608, "mixer_full": 22021632, "embedding": 67108864, "engram_tables": 499464240,
                 "norm_attnres_other": 148480, "dense_ffn": 8650752, "moe_routed": 1157627904, "router": 753664,
                 "moe_shared": 54263808, "engram_dense": 2635776, "lm_head": 67108864}


def param_counts(cfg) -> dict:
    try:
        from moe_llm.budget import param_report  # needs torch (meta-device model, no memory)
        return dict(param_report(cfg)["cats"])
    except ImportError:
        return dict(FALLBACK_CATS)


def sci(x: float) -> str:
    """1e-4 instead of 0.0001."""
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def text_width(s: str, size: float) -> float:
    """Rough rendered width, enough to catch labels that would overflow their boxes."""
    w = 0.0
    for c in s:
        o = ord(c)
        if o >= 0x2E80 or c in "→":
            w += size
        elif c in "il.,:;'|!()[]{} ·":
            w += 0.32 * size
        elif c.isupper() or c in "mwMW@%×−√–":
            w += 0.68 * size
        else:
            w += 0.56 * size
    return w


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.items, self.overflow = w, h, [], []

    def rect(self, x, y, w, h, key="io", rx=7, sw=1.3, dash=None, fill=None, stroke=None):
        f, s = COL[key]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill or f}" stroke="{stroke or s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13.0, weight="normal", anchor="middle", color=TEXT, max_w=None, rotate=None):
        """y is the vertical centre of the text."""
        if max_w is not None and text_width(s, size) > max_w:
            self.overflow.append(f"{s!r}: {text_width(s, size):.0f} > {max_w:.0f}")
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.items.append(f'<text x="{x:.1f}" y="{y + 0.36 * size:.1f}" font-size="{size}" font-weight="{weight}" '
                          f'text-anchor="{anchor}" fill="{color}"{tr}>{escape(s)}</text>')

    def path(self, pts, color=LINE, sw=1.5, arrow="gray", dash=None):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        m = f' marker-end="url(#arr_{arrow})"' if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{m}/>')

    def dot(self, x, y, r=3.2, color=LINE):
        self.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')

    def boxed(self, x, y, w, h, label, key="io", size=13.0, weight="normal", dash=None, sub=None):
        self.rect(x, y, w, h, key, dash=dash)
        if sub:
            self.text(x + w / 2, y + h / 2 - 8.5, label, size, weight, max_w=w - 14)
            self.text(x + w / 2, y + h / 2 + 9.5, sub, size - 1, color=MUTED, max_w=w - 14)
        else:
            self.text(x + w / 2, y + h / 2, label, size, weight, max_w=w - 14)

    def render(self) -> str:
        markers = "".join(
            f'<marker id="arr_{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for n, c in (("gray", LINE), ("purple", COL["engram"][1]), ("gold", COL["attnres"][1])))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">\n<defs>{markers}</defs>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>\n' + "\n".join(self.items) + "\n</svg>\n")


def panel(s: Svg, x, y, w, title, key, lines, size=13.0, lh=19.0):
    h = 33 + lh * len(lines) + 8
    s.rect(x, y, w, h, "panel")
    s.rect(x, y, w, 30, key, rx=7)
    s.text(x + 12, y + 15, title, 15, "bold", anchor="start", max_w=w - 24)
    for i, ln in enumerate(lines):
        s.text(x + 14, y + 33 + lh * (i + 0.5) + 1, "· " + ln, size, anchor="start", max_w=w - 26)
    return y + h


def draw(cfg) -> Svg:
    assert cfg.residual == "block_attnres", "the figure shows Block AttnRes"
    types = cfg.layer_types()
    L, lpb = cfg.n_layers, cfg.attnres_layers_per_block
    nblk = L // lpb
    full = [i for i, t in enumerate(types) if t == "F"]
    moe = [i for i in range(L) if cfg.is_moe(i)]
    c = param_counts(cfg)
    total, engram = sum(c.values()), c["engram_tables"]
    routed_active = c["moe_routed"] * cfg.top_k / cfg.n_experts
    active = total - c["embedding"] - engram - c["moe_routed"] + routed_active
    s = Svg(W, H)

    # ---------------------------------------------------------------- title + legend
    s.text(40, 34, "Ember-2B architecture (main_2b)", 26, "bold", anchor="start")
    s.text(40, 68, f"{total / 1e9:.3f}B total ({(total - engram) / 1e9:.3f}B on GPU + {engram / 1e9:.3f}B Engram) "
           f"· {active / 1e6:.0f}M active per token · vocab {cfg.vocab_size} · d = {cfg.d_model} · {L} layers · pretrained at 4K",
           14, anchor="start", color=MUTED, max_w=1000)
    lx = 905
    for key, label in (("kda", "KDA"), ("attn", "Full attention"), ("moe", "MoE"), ("dense", "Dense FFN"),
                       ("engram", "Engram"), ("attnres", "Block AttnRes")):
        s.rect(lx, 25, 16, 16, key, rx=3)
        s.text(lx + 23, 33, label, 13, anchor="start")
        lx += 30 + text_width(label, 13) + 20

    # ---------------------------------------------------------------- left: the stack
    cx0, cx1, mix_w = 250, 610, 236
    cxc = (cx0 + cx1) / 2
    cell_h, cell_gap, pad, blk_gap = 20, 3, 5, 8
    pitch = lpb * cell_h + (lpb - 1) * cell_gap + 2 * pad + blk_gap
    col_bottom = 878
    gx0, gx1, bus_x = 238, 700, 742

    def cell_top(i):
        b, j = divmod(i, lpb)
        return col_bottom - b * pitch - (j + 1) * cell_h - j * cell_gap

    for b in range(nblk):
        top, bot = cell_top(b * lpb + lpb - 1) - pad, cell_top(b * lpb) + cell_h + pad
        s.rect(gx0, top, gx1 - gx0, bot - top, "block", rx=8)
        s.text(655, (top + bot) / 2, f"Block {b + 1}", 12.5, color=MUTED)
    for i, t in enumerate(types):
        y = cell_top(i)
        mk = "kda" if t == "K" else "attn"
        s.rect(cx0, y, mix_w, cell_h, mk, rx=4, sw=1.1)
        s.text(cx0 + 9, y + cell_h / 2, f"L{i}", 12, "bold", anchor="start")
        s.text(cx0 + 44, y + cell_h / 2, "KDA" if t == "K" else "Full attention", 12.5, anchor="start")
        fk = "moe" if cfg.is_moe(i) else "dense"
        s.rect(cx0 + mix_w, y, cx1 - cx0 - mix_w, cell_h, fk, rx=4, sw=1.1)
        s.text(cx0 + mix_w + (cx1 - cx0 - mix_w) / 2, y + cell_h / 2, "MoE" if fk == "moe" else "Dense FFN", 12.5)

    # input -> embedding -> stack -> final -> loss
    s.boxed(cx0, 950, cx1 - cx0, 32, f"Input tokens ({cfg.max_seq_len} per sequence)", "io")
    s.boxed(cx0, 898, cx1 - cx0, 32, f"Embedding ({cfg.vocab_size} × {cfg.d_model})", "io")
    s.path([(cxc, 950), (cxc, 931)])
    s.path([(cxc, 898), (cxc, col_bottom + pad + 1)])
    top_stack = cell_top(L - 1) - pad
    s.boxed(cx0, 150, cx1 - cx0, 48, f"Depth attention over b0–b{nblk - 1} + block {nblk} → RMSNorm", "attnres",
            sub=f"LM head ({cfg.d_model} → {cfg.vocab_size})")
    s.path([(cxc, top_stack), (cxc, 199)])
    s.boxed(cx0, 98, cx1 - cx0, 30, f"Chunked cross-entropy + z-loss ({sci(cfg.z_loss)})", "io")
    s.path([(cxc, 150), (cxc, 129)])

    # Engram: looked up from the token n-grams, added in layers engram_layers
    ec, es = COL["engram"]
    ey0, ey1 = 640, 760
    s.rect(40, ey0, 160, ey1 - ey0, "engram")
    s.text(120, ey0 + 24, "Engram tables", 14, "bold")
    s.text(120, ey0 + 50, "CPU memory · 0.50B", 12.5, color=MUTED)
    s.text(120, ey0 + 72, f"{'/'.join(str(o) for o in cfg.engram_orders)}-gram hashed lookup", 12.5, color=MUTED)
    s.text(120, ey0 + 94, "Used at " + ", ".join(f"L{i}" for i in cfg.engram_layers), 12.5, color=MUTED)
    s.path([(cx0, 966), (120, 966), (120, ey1 + 1)], color=es, arrow="purple")
    for k, i in enumerate(sorted(cfg.engram_layers, reverse=True)):
        yo = ey0 + 40 + k * 40
        yc = cell_top(i) + cell_h / 2
        s.path([(200, yo), (224, yo), (224, yc), (cx0 - 1, yc)], color=es, arrow="purple")

    # Block AttnRes: block outputs b0 (embedding) .. b7 go on a bus that every later layer reads
    gc, gs = COL["attnres"]
    s.path([(bus_x, 914), (bus_x, 174), (cx1 + 1, 174)], color=gs, sw=2.2, arrow="gold")
    s.path([(cx1, 914), (bus_x, 914)], color=gs, sw=1.6, arrow=None)
    s.dot(bus_x, 914, color=gs)
    s.text(bus_x - 12, 904, "b0", 12, "bold", anchor="end", color=gs)
    for b in range(nblk - 1):
        y = cell_top(b * lpb + lpb - 1) + cell_h / 2
        s.path([(gx1, y), (bus_x, y)], color=gs, sw=1.6, arrow=None)
        s.dot(bus_x, y, color=gs)
        s.text(bus_x - 12, y - 10, f"b{b + 1}", 12, "bold", anchor="end", color=gs)
    s.text(bus_x + 18, 545, "Block outputs b0–b7, read by depth attention in every later layer (Block AttnRes)", 13, color=gs, rotate=-90)

    # ---------------------------------------------------------------- panel A: inside one layer
    ax, ay, aw, ah = 790, 95, 380, 695
    s.rect(ax, ay, aw, ah, "panel")
    s.rect(ax, ay, aw, 30, "io", rx=7)
    s.text(ax + 12, ay + 15, "Inside one layer (Block AttnRes)", 15, "bold", anchor="start")
    steps = [  # bottom to top
        ("Input: earlier block outputs + partial sum", "io", None),
        ("Depth attention res_mix (fp32 softmax)", "attnres", None),
        ("+ Engram memory (only " + ", ".join(f"L{i}" for i in cfg.engram_layers) + ")", "engram", "5,4"),
        ("RMSNorm", "io", None),
        (f"Token mixer: KDA ({L - len(full)} layers) or full attention ({len(full)})", "kda", None),
        ("Partial sum += mixer output", "io", None),
        ("Depth attention res_ffn (blocks + partial sum)", "attnres", None),
        ("RMSNorm", "io", None),
        (f"FFN: MoE (L{moe[0]}–L{moe[-1]}) or dense (L0)", "moe", None),
        ("Output: partial sum += FFN output → next layer", "io", None),
    ]
    bx, bw, bh = ax + 20, aw - 40, 40
    top, bottom = ay + 46, ay + ah - 62
    gap = (bottom - top - len(steps) * bh) / (len(steps) - 1)
    for k, (label, key, dash) in enumerate(steps):
        y = bottom - bh - k * (bh + gap)
        s.boxed(bx, y, bw, bh, label, key, dash=dash)
        if k:
            s.path([(bx + bw / 2, y + bh + gap), (bx + bw / 2, y + bh + 1)])
    s.text(ax + aw / 2, ay + ah - 38, "First layer of a block: the previous partial sum becomes", 12, color=MUTED, max_w=aw - 20)
    s.text(ax + aw / 2, ay + ah - 18, "a new block output and the partial sum restarts here", 12, color=MUTED, max_w=aw - 20)

    # ---------------------------------------------------------------- panels B-E: components
    px, pw = 1190, 380
    y = panel(s, px, 95, pw, f"KDA ({L - len(full)} layers)", "kda", [
        f"{cfg.kda_heads} heads × {cfg.kda_head_dim}, short conv (kernel {cfg.kda_conv_size})",
        "Delta rule with per-channel forget gate",
        "Fixed-size state, compute linear in length",
        f"Forget / output gates via rank-{cfg.kda_lowrank} bottleneck",
        "fla chunk_kda kernel on GPU",
    ])
    y = panel(s, px, y + 10, pw, f"Full attention ({len(full)} layers: L{full[0]}, L{full[1]} … L{full[-1]})", "attn", [
        f"GQA: {cfg.attn_heads} Q heads, {cfg.attn_kv_heads} KV heads, head dim {cfg.attn_head_dim}",
        "No positional encoding (NoPE); KDA supplies position",
        "QK-norm, elementwise sigmoid output gate",
        "The last layer is always full attention",
    ])
    y = panel(s, px, y + 10, pw, f"MoE ({len(moe)} layers: L{moe[0]}–L{moe[-1]})", "moe", [
        f"Top-{cfg.top_k} of {cfg.n_experts} routed experts (width {cfg.expert_inter})",
        f"+ {cfg.n_shared} shared expert (width {cfg.shared_inter})",
        "Router: fp32 sigmoid scores, bias used for selection",
        f"Balancing: bias step ±{cfg.router_bias_update:g} + {sci(cfg.seq_aux_coef)} sequence aux loss",
        f"L0: dense SwiGLU (width {cfg.dense_inter})",
    ])
    rows = cfg.engram_rows if isinstance(cfg.engram_rows, int) else max(cfg.engram_rows)
    y = panel(s, px, y + 10, pw, "Engram (" + ", ".join(f"L{i}" for i in cfg.engram_layers) + ")", "engram", [
        f"{' and '.join(f'{o}-gram' for o in cfg.engram_orders)}, {cfg.engram_heads} hash heads per order",
        f"~{rows / 1e3:.0f}K rows per head, d_mem {cfg.engram_dim}",
        "Canonicalized vocab: 65536 → 42993 ids",
        f"Gate (signed sqrt) + causal conv (kernel {cfg.engram_conv_size})",
        "Tables in CPU memory, sparse Adam updates",
    ])
    rest = c["dense_ffn"] + c["router"] + c["engram_dense"] + c["norm_attnres_other"]
    y = panel(s, px, y + 10, pw, f"Parameters ({total / 1e9:.3f}B total)", "io", [
        f"Routed experts {c['moe_routed'] / 1e9:.3f}B, {cfg.top_k}/{cfg.n_experts} active"
        f" ({routed_active / 1e9:.3f}B)",
        f"Engram tables {engram / 1e9:.3f}B (CPU, not counted as active)",
        f"KDA {c['mixer_kda'] / 1e9:.3f}B · full attn {c['mixer_full'] / 1e9:.3f}B · shared {c['moe_shared'] / 1e9:.3f}B",
        f"Embedding {c['embedding'] / 1e9:.3f}B · LM head {c['lm_head'] / 1e9:.3f}B · other {rest / 1e9:.3f}B",
    ])
    assert y <= ay + ah + 1, y

    # ---------------------------------------------------------------- training settings
    tx, ty, tw_, th = 790, 806, 780, 176
    s.rect(tx, ty, tw_, th, "panel")
    s.rect(tx, ty, tw_, 30, "io", rx=7)
    s.text(tx + 12, ty + 15, "Pretraining setup (single RTX 4090)", 15, "bold", anchor="start")
    for k, ln in enumerate([
        "25B tokens at 4K context; 524,288 tokens per step (2 × 4096 × 64 accumulation), 47,684 steps",
        "Muon (matrices, bf16 momentum) + AdamW (embedding, LM head, norms, router)",
        "WSD schedule, peak LR 2e-3; last 20% decays as 1−√t with a math/code-heavier annealing mix",
        "Fused kernels · partial recomputation · optimizer state offloaded to CPU",
        "Then extended to 32K → 64K → 128K context (5B more tokens)",
    ]):
        s.text(tx + 14, ty + 34 + 27 * (k + 0.5) + 2, "· " + ln, 13, anchor="start", max_w=tw_ - 26)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "architecture"), help="path without extension")
    ap.add_argument("--zoom", type=float, default=2.0, help="PNG scale factor")
    args = ap.parse_args(argv)
    s = draw(main_2b())
    for o in s.overflow:
        print("label may overflow:", o, file=sys.stderr)
    with open(args.out + ".svg", "w", encoding="utf-8") as fh:
        fh.write(s.render())
    print("wrote", args.out + ".svg")
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-z", str(args.zoom), "-o", args.out + ".png", args.out + ".svg"], check=True)
        print("wrote", args.out + ".png")
    else:
        print("rsvg-convert not found: only the SVG was written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
