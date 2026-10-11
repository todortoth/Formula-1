"""
utils.py - small shared helpers
=========================================================
Unit conversions, cleaning of FastF1 position streams and a few array utilities that are used by
several sub-packages. Kept free of any other f1strategy import so it can be imported from anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POS_UNIT_TO_M = 0.1  # FastF1 X/Y/Z are in 1/10 m


def td_seconds(series: Any) -> np.ndarray:
    """Timedelta-like column -> float seconds (NaT -> NaN)."""
    return pd.to_timedelta(series).dt.total_seconds().to_numpy(dtype=float)


def wrap_angle(a: Any) -> Any:
    """Wrap angle(s) in radians to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def pos_frame(pos: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cleaned position stream: t [s], x/y/z [m], on (OnTrack flag).

    Sorted by time and de-duplicated. The all-zero "no signal" samples FastF1 sometimes contains are dropped
    (a car is never really at x = y = z = 0)."""
    t = td_seconds(pos["SessionTime"])
    x = pos["X"].to_numpy(float) * POS_UNIT_TO_M
    y = pos["Y"].to_numpy(float) * POS_UNIT_TO_M
    z = pos["Z"].to_numpy(float) * POS_UNIT_TO_M
    on = (pos["Status"].astype(str).to_numpy() != "OffTrack") if "Status" in pos else np.ones(len(pos), bool)
    ok = np.isfinite(t) & ~((x == 0) & (y == 0) & (z == 0))
    t, x, y, z, on = t[ok], x[ok], y[ok], z[ok], on[ok]
    t, idx = np.unique(t, return_index=True)
    return {"t": t, "x": x[idx], "y": y[idx], "z": z[idx], "on": on[idx]}


def pos_arrays(pos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(t, x, y, z)`` of :func:`pos_frame`."""
    f = pos_frame(pos)
    return f["t"], f["x"], f["y"], f["z"]


def run_length(run: tuple[int, int], n: int) -> int:
    """Number of elements of a circular (start, end) run in an array of length ``n``."""
    return (run[1] - run[0]) % n + 1


def circular_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of the True-runs in a circular boolean array (a run may wrap around the end)."""
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n - 1)]
    off = int(np.flatnonzero(~mask)[0])
    rolled = np.roll(mask, -off)
    runs, i = [], 0
    while i < n:
        if rolled[i]:
            j = i
            while j + 1 < n and rolled[j + 1]:
                j += 1
            runs.append(((i + off) % n, (j + off) % n))
            i = j + 1
        else:
            i += 1
    return runs


def format_time_mmssms(seconds: Any) -> str:
    """Seconds or Timedelta -> ``M:SS:mmm`` (``"N/A"`` for NaN/NaT)."""
    if pd.isna(seconds):
        return "N/A"
    if isinstance(seconds, (pd.Timedelta, np.timedelta64)):
        total = pd.Timedelta(seconds).total_seconds()
    else:
        total = float(seconds)
    minutes = int(total // 60)
    secs = int(total % 60)
    millisecs = int(round((total % 1) * 1000))
    if millisecs == 1000:  # 59.9996 s must not render as "0:59:1000"
        millisecs, secs = 0, secs + 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
    return f"{minutes:01d}:{secs:02d}:{millisecs:03d}"
