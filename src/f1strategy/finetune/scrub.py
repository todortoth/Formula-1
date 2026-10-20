"""
scrub.py - leak control for the *input* of the fine-tuning tasks.

The labels are about the next stop; the window may still contain an earlier stop, and a stop leaves obvious fingerprints:
the in/out laps are 20+ s slower, the car is in the pit lane, drives at the limiter off the racing line.  A model that
reads those from the window learns "a stop is in the window" instead of reading tyres, traffic and race state.
Features are therefore removed from the *whole* window (all cars, all frames) before the model sees it.

Levels (``--scrub``):

* ``none``    nothing removed (ablation: how much do the fingerprints help?).
* ``basic``   default: ``in_pit`` and ``last_lap_time_s`` (the lap time of the in/out lap).  ``pace_last3_s`` stays: by
              definition it excludes in/out laps and laps under SC / VSC / red.
* ``strict``  ``basic`` + the pit-lane geometry (``lateral_offset_m``, ``heading_rel_track_*``, ``on_track``) + the stop
              counters (``stint``, ``pit_stops``); tyre age and compound stay, they *are* the strategy state.

What no scrub can remove: the stop still shows in ``position`` / the gaps (a car loses ~20 s) and in a reset ``tyre_age``.
That is real information, a strategist sees it too.  Removed columns are overwritten with a constant (``zero``, default,
safe for any encoder) or NaN (``nan``: the cache's own "missing").
"""

from __future__ import annotations

LEVELS = {
    "none": (),
    "basic": ("in_pit", "last_lap_time_s"),
    "strict": (
        "in_pit",
        "last_lap_time_s",
        "lateral_offset_m",
        "heading_rel_track_sin",
        "heading_rel_track_cos",
        "on_track",
        "stint",
        "pit_stops",
    ),  # fmt: skip
}


def scrub_columns(driver_names: list[str], level: str) -> list[int]:
    """Column indices (into the 42 training-schema driver features) that a level removes."""
    if level not in LEVELS:
        raise ValueError(f"unknown scrub level {level!r}; choose from {sorted(LEVELS)}")
    names = list(driver_names)
    missing = [n for n in LEVELS[level] if n not in names]
    if missing:
        raise ValueError(f"driver features not in the schema: {missing}")
    return [names.index(n) for n in LEVELS[level]]


def scrub_raw(raw: dict, cols: list[int], fill: str = "zero") -> dict:
    """Overwrite ``raw['drv'][..., cols]`` (a collated window batch, normalised values) in place; returns ``raw``."""
    if not cols:
        return raw
    if fill not in ("zero", "nan"):
        raise ValueError("fill must be 'zero' or 'nan'")
    raw["drv"][..., cols] = 0.0 if fill == "zero" else float("nan")
    return raw
