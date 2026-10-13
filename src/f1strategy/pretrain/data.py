"""
data.py - datasets over the race caches written by ``cache.py``.

* ``PretrainDataset`` yields one snapshot (frame) per item as raw tensors (NaN = missing). Masking happens later, on
  the GPU, in ``masking.Masker``, so the same frame can be masked differently each time it is drawn.
* ``split_races`` keeps validation honest. Neighbouring frames are near duplicates, so a random frame split would leak:
  whole races are held out when there are enough, otherwise interleaved time blocks of every race (with an embargo gap).
* ``frame_weights`` oversamples frames with a flag on track (yellow / SC / VSC / red), which are rare
  (about 1 % of frames in the 2019 Australian GP) but are the entire signal of pretext task C.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from f1strategy.pretrain import schema as S


class RaceCache:
    """Lazy handle on one race cache (memmaps are opened on first access, so DataLoader workers each get their own)."""

    ARRAYS = ("drv", "drv_tok", "present", "glb", "ysec", "dsec", "edge", "eexists", "t")

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.n_frames = int(self.meta["n_frames"])
        self.name = self.path.name
        self._a: dict[str, np.ndarray] | None = None
        self.track = np.load(self.path / "track.npy")
        self.track_marshal = np.load(self.path / "track_marshal.npy").astype(np.int64)

    @property
    def a(self) -> dict[str, np.ndarray]:
        if self._a is None:
            self._a = {k: np.load(self.path / f"{k}.npy", mmap_mode="r") for k in self.ARRAYS}
        return self._a

    def flagged_frames(self) -> np.ndarray:
        """Boolean [T]: any flag active anywhere on track (read from the global features)."""
        g = np.load(self.path / "glb.npy", mmap_mode="r")
        cols = [S.GI[c] for c in S.FLAGGED_FRAME_COLS]
        return np.nan_to_num(np.asarray(g[:, cols])).max(1) > 0.5

    def valid_frames(self) -> np.ndarray:
        """Frames with at least two cars present and a global row."""
        pres = np.load(self.path / "present.npy", mmap_mode="r")
        t = np.load(self.path / "t.npy", mmap_mode="r")
        return (np.asarray(pres).sum(1) >= 2) & np.isfinite(np.asarray(t))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        a = self.a
        return {
            "drv": torch.from_numpy(np.array(a["drv"][i])),
            "tok": torch.from_numpy(np.array(a["drv_tok"][i]).astype(np.int64)),
            "present": torch.from_numpy(np.array(a["present"][i])),
            "glb": torch.from_numpy(np.array(a["glb"][i])),
            "ysec": torch.from_numpy(np.array(a["ysec"][i])),
            "dsec": torch.from_numpy(np.array(a["dsec"][i])),
            "edge": torch.from_numpy(np.array(a["edge"][i])),
            "eexists": torch.from_numpy(np.array(a["eexists"][i])),
            "trk": torch.from_numpy(self.track),
            "tmar": torch.from_numpy(self.track_marshal),
        }


@dataclass
class Split:
    train: list[tuple[int, int]]  # (race index, frame index)
    val: list[tuple[int, int]]
    train_races: list[str]
    val_races: list[str]


def split_races(
    races: list[RaceCache],
    val_frac: float = 0.1,
    min_races_for_race_split: int = 5,
    embargo_s: float = 60.0,
    block_s: float = 180.0,
    seed: int = 0,
) -> Split:
    """Validation split that never puts neighbouring (near duplicate) frames on both sides.

    * With >= ``min_races_for_race_split`` races, whole races are held out.
    * With fewer, every ``round(1/val_frac)``-th block of ``block_s`` seconds of each race is held out and ``embargo_s``
      seconds on both sides of it are dropped from training. Blocks are interleaved over the whole race rather than
      cut from its end: the last laps have progress / lap number / tyre age values the model never trained on, which
      would turn validation into an extrapolation test.
    """
    valid = [r.valid_frames() for r in races]
    tr: list[tuple[int, int]] = []
    va: list[tuple[int, int]] = []
    if len(races) >= min_races_for_race_split:
        rng = np.random.default_rng(seed)
        n_val = max(1, round(val_frac * len(races)))
        val_ids = set(rng.choice(len(races), n_val, replace=False).tolist())
        for ri, v in enumerate(valid):
            (va if ri in val_ids else tr).extend((ri, int(i)) for i in np.flatnonzero(v))
        return Split(tr, va, [r.name for i, r in enumerate(races) if i not in val_ids], [races[i].name for i in sorted(val_ids)])
    k = max(2, round(1 / val_frac))
    for ri, (r, v) in enumerate(zip(races, valid)):
        block = max(1, int(round(block_s / r.meta["step_s"])))
        gap = int(round(embargo_s / r.meta["step_s"]))
        for i in np.flatnonzero(v):
            b, pos = divmod(int(i), block)
            if b % k == k - 1:
                va.append((ri, int(i)))
            elif (b + 1) % k == k - 1 and block - pos <= gap:  # just before a validation block
                continue
            elif (b - 1) % k == k - 1 and pos < gap:  # just after one
                continue
            else:
                tr.append((ri, int(i)))
    names = [r.name for r in races]
    return Split(tr, va, names, [f"{n} (every {k}th {block_s:g}s block)" for n in names])


def frame_weights(races: list[RaceCache], items: list[tuple[int, int]], flag_oversample: float = 5.0) -> np.ndarray:
    """Sampling weight per item: ``flag_oversample`` for frames with any flag, 1 otherwise."""
    flagged = [r.flagged_frames() for r in races]
    return np.array([flag_oversample if flagged[ri][fi] else 1.0 for ri, fi in items], dtype=np.float64)


class PretrainDataset(Dataset):
    def __init__(self, races: list[RaceCache], items: list[tuple[int, int]]):
        self.races, self.items = races, items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        ri, fi = self.items[k]
        return self.races[ri][fi]


def open_races(cache_dir: str | Path) -> list[RaceCache]:
    paths = sorted(p.parent for p in Path(cache_dir).rglob("meta.json"))
    if not paths:
        raise FileNotFoundError(f"no race caches under {cache_dir}; run: python -m f1strategy.pretrain.cache --src ... --dst {cache_dir}")
    races = [RaceCache(p) for p in paths]
    ntok = {r.meta["n_track_tokens"] for r in races}
    if len(ntok) != 1:
        raise ValueError(f"caches were built with different track token counts: {ntok}")
    return races
