"""
labels.py - ground truth for the fine-tuning tasks, built from the Parquet datasets (numpy + polars only, no torch).

For every cache frame ``f`` and car slot ``d`` three questions are answered about the *next tyre-change stop that starts
after f*:

* ``laps_to_pit``  how many lap ends until the car turns into the pit lane. Laps are counted the way a strategist does:
  a car that has completed ``c = lap_number - 1`` laps and stops at the end of lap ``L`` has ``L - c`` laps to go
  (end of lap 10, stop in lap 22 -> 12; end of lap 17 -> 5).  NaN if there is no further stop.
* ``next_cmp``     the compound it receives (0 soft, 1 medium, 2 hard, 3 intermediate, 4 wet, -1 unknown / none).
* ``no_more``      True if the car makes no further tyre-change stop (this is a real "0" for the 5-lap question).

A *tyre-change stop* is a pit-lane entry (``pit_stops`` increases) that is followed by a new stint (``stint`` increases)
before the next entry: drive-through / stop-go penalties and aborted entries do not count.

``valid`` says whether (f, d) may be used as a training / test sample: the car exists, runs lap >= 1, is not in the pit
lane, is not finished or retired, and it is not a retiring car without a next stop (its "no more stops" would be an
artefact of the retirement, not a strategy).

The raw (de-normalised) values of the strategy features at every ``GRID``-th cache frame are stored too (``tab``), for the
tabular baselines; they contain no pit-lane footprint (no ``in_pit``, no lap times).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABEL_FILE = "ft_labels.npz"
GRID = 20  # cache frames between the frames that carry tabular features (10 s at 2 Hz); window ends lie on this grid
N_CMP = 5
CMP_NAMES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
MAX_STOP_LAG = 3000  # parquet frames (5 min at 10 Hz) between pit entry and the new stint (red flags can take long)

DRV_COLS = [
    "frame", "t", "driver", "slot", "lap_number", "progress_m", "pit_stops", "in_pit", "stint", "tyre_age",
    "fresh_tyre", "position", "gap_to_leader_s", "gap_ahead_s", "gap_behind_s", "pace_last3_s",
    "finished_or_retired", "compound_soft", "compound_medium", "compound_hard", "compound_intermediate",
    "compound_wet", "compound_unknown",
]  # fmt: skip
GLB_COLS = [
    "frame", "leader_lap", "laps_remaining", "race_fraction", "is_sc", "is_vsc", "is_yellow", "is_red",
    "track_temp", "air_temp", "rainfall", "humidity",
]  # fmt: skip
TAB_NAMES = [
    "tyre_age", "fresh_tyre", "stint", "pit_stops", "cmp_soft", "cmp_medium", "cmp_hard", "cmp_inter", "cmp_wet",
    "position", "gap_to_leader_s", "gap_ahead_s", "gap_behind_s", "pace_last3_s", "lap_number", "leader_lap",
    "laps_remaining", "race_fraction", "is_sc", "is_vsc", "is_yellow", "is_red", "track_temp", "air_temp",
    "rainfall", "humidity", "track_length_m", "n_cars", "laps_total",
]  # fmt: skip
# the strategy state a strategist has: nothing here reveals a pit stop *in the window* beyond what tyre age / compound say


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward fill NaN along axis 0 of a 1-D array (leading NaN stay NaN)."""
    idx = np.where(np.isfinite(a), np.arange(len(a)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = a[np.maximum(idx, 0)]
    out[idx < 0] = np.nan
    return out


def compound_index(soft, medium, hard, inter, wet) -> np.ndarray:
    """One-hots -> class index (0..4), -1 where none is set."""
    c = np.full(np.shape(soft), -1, dtype=np.int8)
    for k, v in enumerate((soft, medium, hard, inter, wet)):
        c[np.nan_to_num(v) > 0.5] = k
    return c


def tyre_change_events(pit_stops, stint, cmp, lap, max_lag: int = MAX_STOP_LAG) -> list[tuple[int, int, int]]:
    """One car's tyre-change stops -> ``[(entry_frame, in_lap, compound_after), ...]`` in time order.

    ``pit_stops`` / ``stint`` / ``cmp`` / ``lap`` are 1-D arrays over the (parquet) frames of the car."""
    ps = _ffill(np.asarray(pit_stops, dtype=np.float64))
    st = _ffill(np.asarray(stint, dtype=np.float64))
    n = len(ps)
    ok = np.isfinite(ps)
    d = np.zeros(n)
    d[1:] = np.where(ok[1:] & ok[:-1], ps[1:] - ps[:-1], 0.0)
    entries = np.flatnonzero(d > 0)
    out = []
    for k, e in enumerate(entries):
        end = min(n, e + max_lag, entries[k + 1] if k + 1 < len(entries) else n)
        s0 = st[e - 1] if e > 0 and np.isfinite(st[e - 1]) else st[e]
        if not np.isfinite(s0):
            continue
        later = np.flatnonzero(st[e:end] > s0)
        if later.size == 0:
            continue  # penalty / aborted entry: no new set
        g = e + int(later[0])
        seg = np.asarray(cmp[g : g + 50])
        seg = seg[seg >= 0]
        c = int(np.bincount(seg).argmax()) if seg.size else -1
        lp = lap[e]
        if not np.isfinite(lp):
            continue
        out.append((int(e), int(lp), c))
    return out


def car_labels(lap, in_pit, fin, events, retired: bool):
    """Per-frame labels of one car: ``(laps_to_pit f32, next_cmp i8, no_more bool, valid bool)`` over its frames."""
    n = len(lap)
    lap = np.asarray(lap, dtype=np.float64)
    ev_f = np.array([e[0] for e in events], dtype=np.int64)
    ev_l = np.array([e[1] for e in events], dtype=np.float64)
    ev_c = np.array([e[2] for e in events], dtype=np.int8)
    nxt = np.searchsorted(ev_f, np.arange(n), side="right")  # first event strictly after f
    has = nxt < len(ev_f)
    nx = np.minimum(nxt, max(len(ev_f) - 1, 0))
    laps = np.full(n, np.nan, dtype=np.float32)
    cmp = np.full(n, -1, dtype=np.int8)
    if len(ev_f):
        laps[has] = (ev_l[nx[has]] - (lap[has] - 1)).astype(np.float32)
        cmp[has] = ev_c[nx[has]]
    no_more = ~has
    valid = np.isfinite(lap) & (lap >= 1) & (np.nan_to_num(in_pit) < 0.5) & (np.nan_to_num(fin) < 0.5)
    valid &= ~(no_more & retired)
    valid &= ~(has & (laps < 1))  # lap counter glitch: stop "in the past"
    return laps, cmp, no_more, valid


def _dense(df, col: str, F: int, D: int, dtype=np.float32):
    a = np.full((F, D), np.nan, dtype=dtype)
    a[df["frame"].to_numpy(), df["slot"].to_numpy()] = df[col].to_numpy().astype(dtype)
    return a


def find_dataset(parquet_root: Path, race: str) -> Path | None:
    base = Path(parquet_root) / f"{race}.ds"
    cands = sorted(base.glob("sequence_lap*/manifest.json")) or sorted(base.glob("manifest.json"))
    return cands[0].parent if cands else None


def build_race_labels(ds_dir: Path, cache_dir: Path) -> dict[str, np.ndarray]:
    """Labels of one race, on the frames of its cache (``cache_dir/t.npy``)."""
    import polars as pl

    ds_dir, cache_dir = Path(ds_dir), Path(cache_dir)
    man = json.loads((ds_dir / "manifest.json").read_text())
    drv = pl.concat([pl.read_parquet(f, columns=DRV_COLS) for f in sorted((ds_dir / "drivers").glob("*.parquet"))])
    glb = pl.concat([pl.read_parquet(f, columns=GLB_COLS + ["t"]) for f in sorted((ds_dir / "global").glob("*.parquet"))])
    glb = glb.sort("frame")
    F = int(glb["frame"].max()) + 1
    D = max(22, int(drv["slot"].max()) + 1)
    pq_t = np.full(F, np.nan)
    pq_t[glb["frame"].to_numpy()] = glb["t"].to_numpy()

    d = {c: _dense(drv, c, F, D) for c in DRV_COLS if c not in ("frame", "t", "driver", "slot")}
    cmp = compound_index(d["compound_soft"], d["compound_medium"], d["compound_hard"], d["compound_intermediate"], d["compound_wet"])
    g = {c: np.full(F, np.nan, dtype=np.float32) for c in GLB_COLS if c != "frame"}
    for c in g:
        g[c][glb["frame"].to_numpy()] = glb[c].to_numpy().astype(np.float32)

    # --- who retires: the finish flag of a car well before the others
    tf = np.full(D, np.nan)
    for k in range(D):
        hit = np.flatnonzero(np.nan_to_num(d["finished_or_retired"][:, k]) > 0.5)
        if hit.size:
            tf[k] = pq_t[hit[0]]
    done = tf[np.isfinite(tf)]
    ref_end = np.quantile(done, 0.75) if done.size >= 3 else np.inf
    retired = np.isfinite(tf) & (tf < ref_end - 120.0)

    laps = np.full((F, D), np.nan, dtype=np.float32)
    ncmp = np.full((F, D), -1, dtype=np.int8)
    nomore = np.zeros((F, D), dtype=bool)
    valid = np.zeros((F, D), dtype=bool)
    n_events = np.zeros(D, dtype=np.int16)
    for k in range(D):
        if not np.isfinite(d["lap_number"][:, k]).any():
            continue
        ev = tyre_change_events(d["pit_stops"][:, k], d["stint"][:, k], cmp[:, k], d["lap_number"][:, k])
        n_events[k] = len(ev)
        laps[:, k], ncmp[:, k], nomore[:, k], valid[:, k] = car_labels(
            d["lap_number"][:, k], d["in_pit"][:, k], d["finished_or_retired"][:, k], ev, bool(retired[k])
        )

    # --- align to the cache frames by session time
    t_cache = np.load(cache_dir / "t.npy", mmap_mode="r")
    t_cache = np.asarray(t_cache, dtype=np.float64)
    n_c = len(t_cache)
    ok_t = np.isfinite(t_cache)
    ix = np.searchsorted(pq_t, np.nan_to_num(t_cache))  # pq_t is increasing
    ix = np.clip(ix, 1, F - 1)
    ix = np.where(np.abs(pq_t[ix - 1] - t_cache) <= np.abs(pq_t[ix] - t_cache), ix - 1, ix)
    ok_t &= np.abs(pq_t[ix] - t_cache) < 0.5

    def at(a, fill):
        out = a[ix]
        out[~ok_t] = fill
        return out

    out = {
        "laps_to_pit": at(laps, np.nan).astype(np.float32),
        "next_cmp": at(ncmp, -1).astype(np.int8),
        "no_more": at(nomore, False),
        "valid": at(valid, False),
        "retired": retired,
        "n_events": n_events,
    }
    # --- tabular features on the grid
    gi = np.arange(0, n_c, GRID)
    px = ix[gi]
    K = len(TAB_NAMES)
    tab = np.full((len(gi), D, K), np.nan, dtype=np.float32)
    names = {n: j for j, n in enumerate(TAB_NAMES)}
    simple = {"tyre_age": "tyre_age", "fresh_tyre": "fresh_tyre", "stint": "stint", "pit_stops": "pit_stops",
              "cmp_soft": "compound_soft", "cmp_medium": "compound_medium", "cmp_hard": "compound_hard",
              "cmp_inter": "compound_intermediate", "cmp_wet": "compound_wet", "position": "position",
              "gap_to_leader_s": "gap_to_leader_s", "gap_ahead_s": "gap_ahead_s", "gap_behind_s": "gap_behind_s",
              "pace_last3_s": "pace_last3_s", "lap_number": "lap_number"}  # fmt: skip
    for n, c in simple.items():
        tab[:, :, names[n]] = d[c][px]
    for n in ("leader_lap", "laps_remaining", "race_fraction", "is_sc", "is_vsc", "is_yellow", "is_red", "track_temp",
              "air_temp", "rainfall", "humidity"):  # fmt: skip
        tab[:, :, names[n]] = g[n][px][:, None]
    tab[:, :, names["track_length_m"]] = float(man.get("track_length_m", np.nan))
    tab[:, :, names["n_cars"]] = np.isfinite(d["lap_number"][px]).sum(1)[:, None]
    tab[:, :, names["laps_total"]] = (g["leader_lap"][px] + g["laps_remaining"][px])[:, None]
    out["tab"] = tab
    out["tab_names"] = np.array(TAB_NAMES)
    out["driver_names"] = np.array(man["driver_names"])
    out["driver_codes"] = np.array(man["driver_codes"])
    return out


def save_race_labels(lab: dict, cache_dir: Path) -> Path:
    p = Path(cache_dir) / LABEL_FILE
    np.savez_compressed(p, **lab)
    return p


def load_race_labels(cache_dir: Path) -> dict[str, np.ndarray]:
    with np.load(Path(cache_dir) / LABEL_FILE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the fine-tuning labels (ft_labels.npz) next to every race cache.")
    ap.add_argument("--parquet", default="parquet_datasets")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--only", default="", help="only races whose name starts with this (e.g. 2026)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    done = skipped = 0
    for cd in sorted(Path(a.cache).iterdir()):
        if not (cd / "meta.json").exists() or not cd.name.startswith(a.only):
            continue
        if (cd / LABEL_FILE).exists() and not a.overwrite:
            skipped += 1
            continue
        ds = find_dataset(Path(a.parquet), cd.name)
        if ds is None:
            print(f"SKIPPED {cd.name}: no parquet dataset in {a.parquet}")
            continue
        try:
            lab = build_race_labels(ds, cd)
        except Exception as e:  # keep the bulk run going
            print(f"SKIPPED {cd.name}: {type(e).__name__}: {e}")
            continue
        save_race_labels(lab, cd)
        v = lab["valid"]
        print(f"{cd.name}: {int(lab['n_events'].sum())} stops, {int(v.sum())} valid (frame, car) pairs")
        done += 1
    print(f"done: {done} written, {skipped} already present")


if __name__ == "__main__":
    main()
