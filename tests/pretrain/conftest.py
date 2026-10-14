"""Synthetic Parquet dataset with the same layout and schema as the real f1strategy output (small and fast)."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from f1strategy.pretrain import schema as S

N_NODES, N_FRAMES, N_DRV, N_TOK = 200, 120, 6, 32
YELLOW_FRAMES = range(40, 70)  # marshal sector 3 is yellow, 5 double yellow; SC for frames 50-60


def make_synthetic_dataset(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    ds = root / "syn.ds" / "sequence_lap1-3"
    for sub in ("drivers", "global", "edges"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    ang = np.linspace(0, 2 * np.pi, N_NODES, endpoint=False)
    track = {"node": np.arange(N_NODES)}
    for n in S.TRACK_STATIC:
        track[n] = np.zeros(N_NODES)
    track.update(s_m=np.arange(N_NODES) * 10.0, x_m=500 * np.cos(ang), y_m=300 * np.sin(ang), z_m=np.sin(ang) * 5,
                 heading_sin=np.cos(ang), heading_cos=-np.sin(ang), marshal_sector=(np.arange(N_NODES) * 8 // N_NODES) + 1.0,
                 sector_1=(np.arange(N_NODES) < 70).astype(float), is_corner=(np.arange(N_NODES) % 50 == 0).astype(float))  # fmt: skip
    pl.DataFrame(track).write_parquet(ds / "track.parquet")

    rows = []
    for f in range(N_FRAMES):
        for slot in range(N_DRV):
            r = {
                "frame": f,
                "t": 100 + 0.1 * f,
                "driver": f"D{slot:02d}",
                "slot": slot,
                "track_node": int((f * 3 + slot * 25) % N_NODES),
            }
            for n in S.DRIVER_FEATURES:
                r[n] = float(rng.uniform(0, 1))
            for n in S.BIN_DRIVER:
                r[n] = float(rng.random() < 0.3)
            comp = int(rng.integers(0, 3))
            for i, c in enumerate(S.COMPOUND_COLS):
                r[c] = float(i == comp)
            r["gap_ahead_s"] = None if slot == 0 else float(rng.uniform(0, 5))
            r["speed_kmh"] = float(rng.uniform(50, 330))
            rows.append(r)
    pl.DataFrame(rows).cast(dict.fromkeys(S.DRIVER_FEATURES, pl.Float32)).write_parquet(
        ds / "drivers" / "part-00000.parquet"
    )

    g = []
    for f in range(N_FRAMES):
        r = {"frame": f, "t": 100 + 0.1 * f, **{n: float(rng.uniform(0, 1)) for n in S.GLOBAL_FEATURES}}
        yel = f in YELLOW_FRAMES
        sc = 50 <= f < 60
        r.update(is_green=float(not yel), is_yellow=float(yel), is_sc=float(sc), is_vsc=0.0, is_vsc_ending=0.0, is_red=0.0,
                 sc_in_this_lap=float(sc), yellow_unlocalised=0.0, n_yellow_sectors=2.0 if yel else 0.0, drs_enabled=float(not sc),
                 yellow_sectors=[3] if yel else [], double_yellow_sectors=[5] if yel else [])  # fmt: skip
        g.append(r)
    pl.DataFrame(
        g, schema_overrides={"yellow_sectors": pl.List(pl.Int16), "double_yellow_sectors": pl.List(pl.Int16)}
    ).write_parquet(ds / "global" / "part-00000.parquet")

    e = []
    for f in range(N_FRAMES):
        for a in range(N_DRV):
            for b in range(N_DRV):
                if a != b and abs(a - b) <= 2:
                    e.append({"frame": f, "t": 100 + 0.1 * f, "src": f"D{a:02d}", "dst": f"D{b:02d}", "src_slot": a, "dst_slot": b,
                              **{n: float(rng.normal()) for n in S.EDGE_FEATURES}})  # fmt: skip
    pl.DataFrame(e).write_parquet(ds / "edges" / "part-00000.parquet")

    manifest = {"format": "f1strategy-parquet", "format_version": 1, "driver_codes": [f"D{i:02d}" for i in range(N_DRV)],
                "track_static_names": S.TRACK_STATIC, "track_names": S.TRACK_STATIC + S.FLAG_BITS + ["n_drivers"],
                "driver_names": S.DRIVER_FEATURES, "global_names": S.GLOBAL_FEATURES, "edge_names": S.EDGE_FEATURES,
                "flag_bits": S.FLAG_BITS, "track_length_m": 2000.0, "track_spacing_m": 10.0, "n_track": N_NODES,
                "race_distance_m": 6000.0, "step_s": 0.1, "n_frames": N_FRAMES, "t_start": 100.0, "t_stop": 112.0,
                "chunks": [{"index": 0, "frame_start": 0, "frame_stop": N_FRAMES}]}  # fmt: skip
    (ds / "manifest.json").write_text(json.dumps(manifest))
    return ds


@pytest.fixture(scope="session")
def synthetic_ds(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def cache_dir(synthetic_ds, tmp_path_factory):
    from f1strategy.pretrain.cache import build_race_cache

    out = tmp_path_factory.mktemp("cache")
    build_race_cache(synthetic_ds, out / "syn", stride=2, n_track_tokens=N_TOK)
    return out
