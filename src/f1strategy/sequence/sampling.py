"""
sampling.py · Formula 1 Strategic Decisions
=========================================================
Choosing *when* to take snapshots. The graph is a function of session time, so "continuous" data is simply that
function sampled on a dense, regular time grid.

Choosing the step: the raw position stream arrives about every 0.22 s and the car data about every 0.24 s (~4.5 Hz
each; merging both clocks gives ~8-9 samples per second). A 10 Hz grid (``hz=10``) therefore samples a bit
more than twice as often as each stream and interpolates in between; the per-driver ``pos_age_s`` / ``car_age_s``
features say how old the newest real sample behind each value is. 0.5-1 s is a good range for most models, 2-5 s for
quick experiments.
"""

from __future__ import annotations

import numpy as np

# Frames per chunk when streaming. Peak memory is roughly proportional to it: about 0.1 MB per frame for a 20 car grid
# (measured on the synthetic race), i.e. ~200 MB at the default.
DEFAULT_CHUNK_FRAMES = 2000


def resolve_step(step_s: float | None = None, hz: float | None = None, default_s: float = 1.0) -> float:
    """The frame step in seconds from either ``step_s`` or a rate ``hz`` (10 Hz = 0.1 s); both is an error."""
    if step_s is not None and hz is not None:
        raise ValueError("pass either step_s or hz, not both")
    if hz is not None:
        if not hz > 0:
            raise ValueError(f"hz must be positive, got {hz}")
        return 1.0 / hz
    return default_s if step_s is None else step_s


def time_grid(start_s: float, end_s: float, step_s: float | None = None, hz: float | None = None) -> np.ndarray:
    """Regular session-time grid ``start, start + step, ...`` up to ``end`` (inclusive when it falls on the grid).

    Give the step as ``step_s`` seconds or as a rate ``hz``; the default is one frame per second."""
    step_s = resolve_step(step_s, hz)
    if not step_s > 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    if end_s < start_s:
        raise ValueError(f"end_s ({end_s}) is before start_s ({start_s})")
    n = int(np.floor((end_s - start_s) / step_s + 1e-9)) + 1
    return start_s + np.arange(n) * step_s
