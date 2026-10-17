"""
config.py - the three sequence scales and the model presets of the temporal (sequence) BERT.

A *sequence* is a window of consecutive race graphs, one token per graph (frame). Three scales, all defined in
physical time so that the same code serves every circuit:

    short   10 s             20 tokens  (0.5 s per token - the cache rate, no sub-sampling)
    medium  1 lap            96 tokens  (lap = median representative lap time of the race, ~1 s per token)
    long    10 laps         192 tokens  (~4.5 s per token)

The number of tokens is fixed per scale (so batches stack); the time step between two tokens is derived from the
race's median lap time and rounded to a whole number of cache frames (cache = 2 Hz, i.e. 0.5 s). Token counts are a
trade-off between temporal resolution and memory: raise/lower ``n_tokens`` with ``--n-tokens`` once you know what your
GPU can carry (see ``bench.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scale:
    name: str
    n_tokens: int
    span_s: float | None = None  # fixed duration in seconds ...
    span_laps: float | None = None  # ... or a number of laps (x the race's median lap time)
    scale_id: int = 0

    def span_seconds(self, lap_s: float) -> float:
        return self.span_s if self.span_s is not None else float(self.span_laps) * lap_s

    def step_frames(self, lap_s: float, base_step_s: float = 0.5) -> int:
        """Cache frames between two tokens (>= 1)."""
        return max(1, round(self.span_seconds(lap_s) / self.n_tokens / base_step_s))


SCALES = {
    "short": Scale("short", 20, span_s=10.0, scale_id=0),
    "medium": Scale("medium", 96, span_laps=1.0, scale_id=1),
    "long": Scale("long", 192, span_laps=10.0, scale_id=2),
}

PRESETS = {
    "tiny": {"hidden": 128, "layers": 2, "heads": 4, "ffn": 512, "gnn_dim": 64},  # smoke tests / CPU
    "small": {"hidden": 256, "layers": 6, "heads": 8, "ffn": 1024, "gnn_dim": 128},
    "base": {"hidden": 384, "layers": 8, "heads": 8, "ffn": 1536, "gnn_dim": 128},
}
