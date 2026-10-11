"""Shared test helpers."""

from __future__ import annotations

import numpy as np


def true_gap(truth: dict, behind: str, ahead: str, t: float) -> float:
    """Reference time gap: how long ago was `ahead` where `behind` is now (from the synthetic ground truth)."""
    tt = np.arange(0.0, truth["t_total"], 0.005)
    s_ahead = truth["S"](ahead, tt)
    return t - float(np.interp(float(truth["S"](behind, t)), s_ahead, tt))


def clone_session(session, **overrides):
    """Shallow copy of a (synthetic) session with some attributes replaced."""
    from types import SimpleNamespace

    return SimpleNamespace(**{**vars(session), **overrides})
