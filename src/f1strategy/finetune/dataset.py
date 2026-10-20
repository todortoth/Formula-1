"""
dataset.py - labelled windows for fine-tuning.

A sample is a window of T graphs (``seqbert`` scale ``medium`` = 1 lap or ``long`` = 10 laps) *plus one label set per car
slot*, taken at the LAST frame of the window ("at the end of the sequence, what happens next to car d?").  Windows end on
the ``GRID`` of ``labels.py`` (every 10 s of race time by default hop 30 s / 60 s) so the tabular baselines can use the
same samples.  Reuses ``seqbert.data.WindowIndex`` / ``WindowDataset`` for the heavy lifting (windows are read exactly as in
pre-training); only the list of windows and the labels are new.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from f1strategy.finetune.labels import GRID, LABEL_FILE, load_race_labels
from f1strategy.finetune.scrub import scrub_raw
from f1strategy.pretrain.data import RaceCache, open_races
from f1strategy.seqbert.data import WindowDataset, WindowIndex

DEFAULT_HOP_S = {"short": 10.0, "medium": 30.0, "long": 60.0}


def race_year(r: RaceCache) -> int:
    return int(r.name.split("_")[0])


def parse_years(s: str) -> set[int]:
    """``2019-2024,2026`` -> {2019, ..., 2024, 2026}; empty -> empty set."""
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in s.split(","))):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def open_labelled_races(cache: str | Path, years: set[int]) -> tuple[list[RaceCache], dict[str, dict]]:
    """The races of ``years`` that have a ``ft_labels.npz`` (others are reported and skipped)."""
    races, labels = [], {}
    for r in open_races(str(cache)):
        if race_year(r) not in years:
            continue
        if not (Path(cache) / r.name / LABEL_FILE).exists():
            warnings.warn(f"{r.name}: no {LABEL_FILE} (run python -m f1strategy.finetune.labels), skipped", stacklevel=2)
            continue
        races.append(r)
        labels[r.name] = load_race_labels(Path(cache) / r.name)
    return races, labels


def finetune_index(races: list[RaceCache], scale, hop_s: float | None = None, min_valid: float = 0.9) -> WindowIndex:
    """A ``WindowIndex`` whose windows END on the label grid, one every ``hop_s`` seconds of race time."""
    ix = WindowIndex(races, scale, hop_frac=1.0, min_valid=min_valid)  # used for span_frames / valid tables only
    hop_s = DEFAULT_HOP_S.get(getattr(scale, "name", ""), 30.0) if hop_s is None else hop_s
    T = scale.n_tokens
    items: list[tuple[int, int, int]] = []
    for ri, r in enumerate(races):
        span = ix.span_frames[ri]
        step = (span - 1) // (T - 1)
        hop = GRID * max(1, round(hop_s / (GRID * float(r.meta["step_s"]))))
        cs = np.concatenate([[0], np.cumsum(ix.valid[ri])])
        e = GRID * (-(-(span - 1) // GRID))  # first grid frame that leaves room for a whole window
        while e <= r.n_frames - 1:
            s = e - span + 1
            if (cs[s + span] - cs[s]) / span >= min_valid:
                items.append((ri, s, step))
            e += hop
    ix.items = items
    return ix


def end_frame(ix: WindowIndex, k: int) -> int:
    ri, s, _ = ix.items[k]
    return s + ix.span_frames[ri] - 1


def window_targets(ix: WindowIndex, labels: dict[str, dict], ids: list[int] | None = None) -> dict[str, np.ndarray]:
    """Labels (and baseline features) of the windows ``ids`` in window order, all ``[N, D(, K)]`` numpy arrays.
    The same function feeds the model and the baselines, so every method is scored on identical (window, car) pairs."""
    ids = list(range(len(ix))) if ids is None else list(ids)
    n, D = len(ids), 22
    out = {
        "valid": np.zeros((n, D), bool), "laps": np.full((n, D), np.nan, np.float32), "cmp": np.full((n, D), -1, np.int64),
        "nomore": np.zeros((n, D), bool), "race": np.zeros(n, np.int32), "frame": np.zeros(n, np.int64),
    }  # fmt: skip
    tab = None
    for j, k in enumerate(ids):
        ri, _, _ = ix.items[k]
        r = ix.races[ri]
        lab = labels[r.name]
        fe = end_frame(ix, k)
        a = r.a
        ok = bool(np.isfinite(a["t"][fe]) and np.asarray(a["present"][fe]).sum() >= 2)
        pres = np.asarray(a["present"][fe], dtype=bool)
        out["valid"][j] = lab["valid"][fe] & pres & ok
        out["laps"][j] = lab["laps_to_pit"][fe]
        out["cmp"][j] = lab["next_cmp"][fe]
        out["nomore"][j] = lab["no_more"][fe]
        out["race"][j], out["frame"][j] = ri, fe
        if tab is None:
            tab = np.full((n, D, lab["tab"].shape[-1]), np.nan, np.float32)
        tab[j] = lab["tab"][fe // GRID]
    out["tab"] = tab if tab is not None else np.zeros((0, D, 0), np.float32)
    return out


class FtDataset(WindowDataset):
    """``{"x": raw window (as in pre-training), "y": labels of the last frame}``."""

    def __init__(self, index: WindowIndex, labels: dict[str, dict], ids: list[int] | None = None):
        super().__init__(index, ids)
        self.labels = labels

    def __getitem__(self, k: int) -> dict:
        raw = super().__getitem__(k)
        ix = self.index
        gid = self.ids[k]
        r = ix.races[ix.items[gid][0]]
        lab = self.labels[r.name]
        fe = end_frame(ix, gid)
        ok_last = bool(raw["ok"][-1])
        pres = raw["present"][-1].numpy().astype(bool)
        valid = lab["valid"][fe] & pres & ok_last
        y = {
            "valid": torch.from_numpy(valid),
            "laps": torch.from_numpy(np.asarray(lab["laps_to_pit"][fe], dtype=np.float32)),
            "cmp": torch.from_numpy(np.asarray(lab["next_cmp"][fe], dtype=np.int64)),
            "nomore": torch.from_numpy(np.asarray(lab["no_more"][fe], dtype=bool)),
        }
        return {"x": raw, "y": y}


def targets_to_torch(t: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(t[k])) for k in ("valid", "laps", "cmp", "nomore")}


def to_masked_batch(x: dict, masker, device, cols: list[int], fill: str = "zero"):
    """Raw window batch -> scrubbed -> everything visible (``SeqMasker.plain``), on ``device``."""
    raw = {k: v.to(device) for k, v in x.items()}
    scrub_raw(raw, cols, fill)
    return masker.plain(raw, token_dropout=0.0)
