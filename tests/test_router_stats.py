"""scripts/router_stats.py reads a training checkpoint and reports per-layer router-bias outliers."""
import sys
from pathlib import Path

import torch

from moe_llm.config import tiny
from moe_llm.model import MoELLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import router_stats  # noqa: E402


def test_router_stats_finds_starved_expert(tmp_path, capsys):
    torch.manual_seed(0)
    cfg = tiny(engram_layers=())
    m = MoELLM(cfg)
    moe = m.moe_layers()
    assert moe, "tiny config should have MoE layers"
    moe[0].e_bias.add_(0.01)
    moe[0].e_bias[3] += 0.05                       # expert 3: 50 more under-loaded steps than the rest
    path = tmp_path / "ckpt_0000007.pt"
    torch.save({"step": 7, "model": m.state_dict()}, path)
    router_stats.summarize(str(path), gamma=1e-3, top=1)
    out = capsys.readouterr().out
    assert "(step 7)" in out
    rows = [ln.split() for ln in out.splitlines() if ln.strip()[:1].isdigit()]
    assert len(rows) == len(moe)
    first = rows[0]
    assert first[-1] == "e3:+50" and float(first[4]) == 50
