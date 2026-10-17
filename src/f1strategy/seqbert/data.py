"""
data.py - windows (sequences of consecutive graphs) over the race caches written by ``pretrain.cache``.

* ``WindowIndex`` enumerates the windows of one scale: (race, first frame, frame step).
* ``split_windows`` keeps validation honest: whole races when >= 5 are available, otherwise interleaved time blocks;
  a window is validation only if it lies completely inside a held-out block and training only if it touches none.
* ``circuit_ids`` recognises the same circuit in different years from the pooled curvature profile, and
  ``PairBatchSampler`` uses it to build contrastive batches (same circuit + same race phase = positive pair).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from f1strategy.pretrain import schema as S
from f1strategy.pretrain.data import RaceCache, open_races  # noqa: F401  (re-exported)
from f1strategy.seqbert.config import Scale


class WindowIndex:
    """All windows of ``scale`` over ``races``: ``items[k] = (race, start_frame, step)``."""

    def __init__(self, races: list[RaceCache], scale: Scale, hop_frac: float = 0.25, min_valid: float = 0.9):
        self.races, self.scale = races, scale
        self.items: list[tuple[int, int, int]] = []
        self.valid = [r.valid_frames() for r in races]
        self.flag = [np.asarray(r.flagged_frames()) for r in races]
        self.span_frames: list[int] = []
        T = scale.n_tokens
        for ri, r in enumerate(races):
            step = scale.step_frames(r.meta["norm"]["pace_ref_s"], r.meta["step_s"])
            span = (T - 1) * step + 1
            self.span_frames.append(span)
            n = r.n_frames
            if n < span:
                continue
            hop = max(1, int(span * hop_frac))
            cs = np.concatenate([[0], np.cumsum(self.valid[ri])])
            for s in range(0, n - span + 1, hop):
                idx_valid = cs[s + span] - cs[s]
                # fraction of the *tokens* that are valid (approx. by the share of valid frames in the span)
                if idx_valid / span >= min_valid:
                    self.items.append((ri, s, step))

    def __len__(self) -> int:
        return len(self.items)

    def span(self, k: int) -> int:
        ri, s, step = self.items[k]
        return self.span_frames[ri]

    def has_flag(self, k: int) -> bool:
        ri, s, step = self.items[k]
        return bool(self.flag[ri][s : s + self.span_frames[ri]].any())


class WindowDataset(Dataset):
    def __init__(self, index: WindowIndex, ids: list[int] | None = None):
        self.index = index
        self.ids = list(range(len(index))) if ids is None else list(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        ix = self.index
        ri, s, step = ix.items[self.ids[k]]
        r = ix.races[ri]
        T = ix.scale.n_tokens
        fr = s + step * np.arange(T)
        a = r.a
        t = np.array(a["t"][fr])
        present = np.array(a["present"][fr])
        out = {
            "drv": torch.from_numpy(np.array(a["drv"][fr])),
            "tok": torch.from_numpy(np.array(a["drv_tok"][fr]).astype(np.int64)),
            "present": torch.from_numpy(present),
            "glb": torch.from_numpy(np.array(a["glb"][fr])),
            "ysec": torch.from_numpy(np.array(a["ysec"][fr])),
            "dsec": torch.from_numpy(np.array(a["dsec"][fr])),
            "edge": torch.from_numpy(np.array(a["edge"][fr])),
            "eexists": torch.from_numpy(np.array(a["eexists"][fr])),
            "trk": torch.from_numpy(r.track),
            "tmar": torch.from_numpy(r.track_marshal),
            "ok": torch.from_numpy(np.isfinite(t) & (present.sum(1) >= 2)),
            "t": torch.from_numpy(np.nan_to_num(t)),
        }
        return out


# ------------------------------------------------------------------------------------------------ split
def split_windows(
    ix: WindowIndex, val_frac: float = 0.1, min_races_for_race_split: int = 5, block_s: float = 180.0, seed: int = 0
) -> tuple[list[int], list[int], list[str]]:
    """-> (train ids, val ids, description of the validation part)."""
    races = ix.races
    tr: list[int] = []
    va: list[int] = []
    if len(races) >= min_races_for_race_split:
        rng = np.random.default_rng(seed)
        n_val = max(1, round(val_frac * len(races)))
        val_races = set(rng.choice(len(races), n_val, replace=False).tolist())
        for k, (ri, _, _) in enumerate(ix.items):
            (va if ri in val_races else tr).append(k)
        return tr, va, [races[i].name for i in sorted(val_races)]
    k_blk = max(2, round(1 / val_frac))
    for k, (ri, s, _step) in enumerate(ix.items):
        r = races[ri]
        span = ix.span_frames[ri]
        blk = max(int(round(block_s / r.meta["step_s"])), 2 * span)
        n_blocks = max(1, -(-r.n_frames // blk))
        kk = max(2, min(k_blk, n_blocks))
        b0, b1 = s // blk, (s + span - 1) // blk
        is_val = lambda b, kk=kk: b % kk == kk - 1  # noqa: E731
        if b0 == b1 and is_val(b0):
            va.append(k)
        elif not is_val(b0) and not is_val(b1):
            tr.append(k)  # windows touching a validation block are dropped (embargo = the whole window)
    return tr, va, [f"{r.name} (interleaved blocks)" for r in races]


def window_weights(ix: WindowIndex, ids: list[int], flag_oversample: float = 3.0) -> np.ndarray:
    return np.array([flag_oversample if ix.has_flag(k) else 1.0 for k in ids], dtype=np.float64)


# ------------------------------------------------------------------------------------------------ contrastive groups
def circuit_ids(races: list[RaceCache], corr_min: float = 0.9, len_tol: float = 0.03) -> list[int]:
    """Same circuit in different years -> same id (curvature profile correlation + lap length)."""
    ci = S.TI["curvature"]
    sig, length = [], []
    for r in races:
        c = np.asarray(r.track[:, ci], dtype=np.float64)
        sig.append(c)
        length.append(float(r.meta["norm"]["track_length_m"]))
    ids: list[int] = []
    reps: list[int] = []
    for i in range(len(races)):
        found = -1
        for cid, j in enumerate(reps):
            if abs(length[i] / length[j] - 1) > len_tol:
                continue
            a, b = sig[i], sig[j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                if a.std() < 1e-9 and b.std() < 1e-9 and np.allclose(a, b):  # degenerate (e.g. synthetic) tracks
                    found = cid
                    break
                continue
            if np.corrcoef(a, b)[0, 1] >= corr_min:
                found = cid
                break
        if found < 0:
            reps.append(i)
            found = len(reps) - 1
        ids.append(found)
    return ids


def window_groups(ix: WindowIndex, ids: list[int], phase_bins: int = 10) -> dict[tuple[int, int], list[int]]:
    """key = (circuit id, race-phase bin) -> positions (into ``ids``) of the windows in that group."""
    circ = circuit_ids(ix.races)
    groups: dict[tuple[int, int], list[int]] = {}
    for pos, k in enumerate(ids):
        ri, s, _ = ix.items[k]
        centre = s + ix.span_frames[ri] / 2
        b = min(phase_bins - 1, int(phase_bins * centre / ix.races[ri].n_frames))
        groups.setdefault((circ[ri], b), []).append(pos)
    return groups


class PairBatchSampler(Sampler):
    """Yields lists of 2P dataset positions: the first P and the last P are positive pairs (same circuit + phase),
    drawn from P different groups (different circuits first, so slow and fast tracks are the negatives)."""

    def __init__(self, groups: dict, ix: WindowIndex, ids: list[int], pairs: int, n_batches: int, seed: int = 0):
        self.keys = list(groups)
        self.groups, self.ix, self.ids = groups, ix, ids
        self.pairs, self.n_batches, self.seed = pairs, n_batches, seed

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        P = min(self.pairs, len(self.keys))
        for _ in range(self.n_batches):
            order = rng.permutation(len(self.keys))
            chosen, circuits = [], set()
            for j in order:  # prefer distinct circuits
                if self.keys[j][0] not in circuits:
                    chosen.append(j)
                    circuits.add(self.keys[j][0])
                if len(chosen) == P:
                    break
            for j in order:
                if len(chosen) == P:
                    break
                if j not in chosen:
                    chosen.append(j)
            a, b = [], []
            for j in chosen:
                members = self.groups[self.keys[j]]
                x = int(rng.choice(members))
                rx = self.ix.items[self.ids[x]][0]
                other = [m for m in members if m != x and self.ix.items[self.ids[m]][0] != rx] or [
                    m for m in members if m != x
                ]
                y = int(rng.choice(other)) if other else x
                a.append(x)
                b.append(y)
            yield a + b
