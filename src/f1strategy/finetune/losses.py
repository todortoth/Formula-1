"""losses.py - masked supervised losses of the fine-tuning head (every term averages over its own valid (window, car) pairs)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from f1strategy.finetune.model import HeadConfig


def _mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def ft_loss(
    out: dict, y: dict, hc: HeadConfig, weights: dict[str, float] | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """``out``: head logits (``FtModel.head_forward``); ``y``: ``valid`` [B, D] bool, ``laps`` [B, D] (NaN = no next stop),
    ``cmp`` [B, D] (-1 = none), ``nomore`` [B, D] bool.  Returns (total, per-task floats)."""
    w = {"pit": 1.0, "laps": 1.0, "cmp": 1.0, "nomore": 0.5} | (weights or {})
    v = y["valid"]
    laps = y["laps"].float()
    has = v & torch.isfinite(laps)
    terms: dict[str, torch.Tensor] = {}
    if "pit" in hc.tasks:
        h = torch.tensor(hc.horizons, device=laps.device, dtype=laps.dtype)
        tgt = (has[..., None] & (torch.nan_to_num(laps, nan=1e9)[..., None] <= h)).to(out["pit"].dtype)
        loss = Fn.binary_cross_entropy_with_logits(out["pit"].float(), tgt.float(), reduction="none").mean(-1)
        terms["pit"] = _mean(loss, v)
    if "laps" in hc.tasks:
        t = torch.log1p(torch.nan_to_num(laps, nan=0.0).clamp(0.0, hc.laps_cap))
        loss = Fn.smooth_l1_loss(out["laps"].float(), t, beta=0.3, reduction="none")
        terms["laps"] = _mean(loss, has)
    if "cmp" in hc.tasks:
        m = has & (y["cmp"] >= 0)
        loss = Fn.cross_entropy(
            out["cmp"].float().flatten(0, 1), y["cmp"].clamp(min=0).flatten(), reduction="none"
        ).view_as(laps)
        terms["cmp"] = _mean(loss, m)
    if "nomore" in hc.tasks:
        loss = Fn.binary_cross_entropy_with_logits(out["nomore"].float(), y["nomore"].float(), reduction="none")
        terms["nomore"] = _mean(loss, v)
    if not terms:
        raise ValueError("no task selected")
    den = sum(w[k] for k in terms)
    total = sum(w[k] * t for k, t in terms.items()) / den
    return total, {k: float(t.detach()) for k, t in terms.items()}


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
