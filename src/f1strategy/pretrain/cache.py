"""
cache.py - turn f1strategy Parquet datasets into compact, memory-mappable training caches.

One cache directory per race::

    meta.json         race constants, names, normalisation constants, stride
    drv.npy           [T, D, 45]   f32  normalised driver features (NaN = missing)
    drv_tok.npy       [T, D]       i16  track token of the car (-1 unknown)
    present.npy       [T, D]       bool car exists in the frame
    glb.npy           [T, 23]      f32  normalised global features (NaN = missing)
    ysec.npy/dsec.npy [T, S]       bool yellow / double yellow per marshal sector (index = sector id - 1)
    edge.npy          [T, D, D, 7] f16  normalised proximity-edge features (NaN = missing / no edge)
    eexists.npy       [T, D, D]    bool edge exists
    t.npy             [T]          f64  session time of the frame
    track.npy         [M, 18]      f32  track static features pooled to M tokens (normalised)
    track_marshal.npy [M]          i16  marshal sector id of each track token (0 = unknown)

The data is decimated by ``stride`` (default 5 -> 2 Hz): at 10 Hz neighbouring frames are near duplicates (and the
real telemetry only arrives at ~4.5 Hz), so stride 1 mostly multiplies disk and epoch time. Frames are subsampled
by ``(frame - first_frame) % stride == 0``.

    python -m f1strategy.pretrain.cache --src parquet_datasets --dst cache --stride 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

from f1strategy.pretrain import schema as S

CACHE_VERSION = 1
_POOL_MAX = {"is_corner", "corner_number", "in_drs_zone", "is_drs_start", "is_drs_end", "is_drs_detection"}


def find_datasets(root: str | Path) -> list[Path]:
    """Every directory below ``root`` that holds a ``manifest.json`` (the CLI nests it in ``sequence_lap*``)."""
    root = Path(root)
    return sorted(p.parent for p in root.rglob("manifest.json") if (p.parent / "track.parquet").exists())


def _pool_track(track: pl.DataFrame, n_tokens: int, rn: S.RaceNorm):
    """Pool the 10 m ring into ``n_tokens`` tokens: max for markers, mean for geometry, majority for marshal sector."""
    n = track.height
    if n < n_tokens:
        raise ValueError(f"track has {n} nodes, fewer than {n_tokens} tokens")
    node = track["node"].to_numpy()
    raw = track.select(S.TRACK_STATIC).to_numpy().astype(np.float64)
    tok = np.minimum((node * n_tokens) // n, n_tokens - 1)
    pooled = np.zeros((n_tokens, S.F_TRACK_STATIC), np.float64)
    marshal = np.zeros(n_tokens, np.int16)
    mcol = S.TI["marshal_sector"]
    for m in range(n_tokens):
        rows = raw[tok == m]
        pooled[m] = rows.mean(0)
        for name in _POOL_MAX:
            pooled[m, S.TI[name]] = rows[:, S.TI[name]].max()
        ids, counts = np.unique(rows[:, mcol], return_counts=True)
        marshal[m] = int(ids[counts.argmax()]) if ids.size and ids[counts.argmax()] > 0 else 0
        pooled[m, mcol] = marshal[m]
    return S.normalise_track(pooled.astype(np.float32), rn), marshal


def build_race_cache(
    ds_dir: str | Path,
    out_dir: str | Path,
    stride: int = 5,
    n_track_tokens: int = S.N_TRACK_TOKENS,
    max_drivers: int = S.MAX_DRIVERS,
    overwrite: bool = False,
) -> Path:
    ds_dir, out_dir = Path(ds_dir), Path(out_dir)
    if (out_dir / "meta.json").exists() and not overwrite:
        return out_dir
    manifest = json.loads((ds_dir / "manifest.json").read_text())
    S.check_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    f0 = int(manifest["chunks"][0]["frame_start"])
    n_frames = int(manifest["n_frames"])
    T = math.ceil(n_frames / stride)
    D, F, G, FE = max_drivers, S.F_DRIVER, S.F_GLOBAL, S.F_EDGE
    if len(manifest["driver_codes"]) > D:
        raise ValueError(f"{len(manifest['driver_codes'])} drivers > max_drivers={D}")

    track = pl.read_parquet(ds_dir / "track.parquet").sort("node")
    pace = pl.scan_parquet(ds_dir / "drivers" / "part-*.parquet").select(pl.col("pace_last3_s").median()).collect().item()
    rn = S.RaceNorm.from_track(
        track["x_m"].to_numpy(), track["y_m"].to_numpy(), track["z_m"].to_numpy(),
        manifest["track_length_m"], manifest["race_distance_m"], pace if pace is not None else 90.0,
    )  # fmt: skip
    track_tok, track_marshal = _pool_track(track, n_track_tokens, rn)
    n_nodes = track.height
    if int(track_marshal.max()) > S.MAX_MARSHAL:
        raise ValueError(f"marshal sector id {int(track_marshal.max())} > MAX_MARSHAL={S.MAX_MARSHAL}")

    mm = lambda name, shape, dtype, fill: _memmap(out_dir / name, shape, dtype, fill)  # noqa: E731
    drv = mm("drv.npy", (T, D, F), np.float32, np.nan)
    drv_tok = mm("drv_tok.npy", (T, D), np.int16, -1)
    present = mm("present.npy", (T, D), bool, False)
    glb = mm("glb.npy", (T, G), np.float32, np.nan)
    ysec = mm("ysec.npy", (T, S.MAX_MARSHAL), bool, False)
    dsec = mm("dsec.npy", (T, S.MAX_MARSHAL), bool, False)
    edge = mm("edge.npy", (T, D, D, FE), np.float16, np.nan)
    eexists = mm("eexists.npy", (T, D, D), bool, False)
    times = mm("t.npy", (T,), np.float64, np.nan)

    def decimated(part: Path) -> pl.DataFrame:
        df = pl.read_parquet(part)
        return df.filter(((pl.col("frame") - f0) % stride) == 0).with_columns(fi=((pl.col("frame") - f0) // stride))

    dspec, gspec, espec = S.driver_spec(rn), S.global_spec(rn), S.edge_spec(rn)
    for part in sorted((ds_dir / "drivers").glob("part-*.parquet")):
        df = decimated(part)
        if df.is_empty():
            continue
        fi, slot = df["fi"].to_numpy(), df["slot"].to_numpy().astype(int)
        if slot.max() >= D:
            raise ValueError(f"slot {slot.max()} >= max_drivers")
        drv[fi, slot] = S._apply(df.select(S.DRIVER_FEATURES).to_numpy(), dspec)
        present[fi, slot] = True
        node = df["track_node"].to_numpy().astype(np.int64)
        drv_tok[fi, slot] = np.where(node >= 0, np.minimum(node * n_track_tokens // n_nodes, n_track_tokens - 1), -1)
    for part in sorted((ds_dir / "global").glob("part-*.parquet")):
        df = decimated(part)
        if df.is_empty():
            continue
        fi = df["fi"].to_numpy()
        glb[fi] = S._apply(df.select(S.GLOBAL_FEATURES).to_numpy(), gspec)
        times[fi] = df["t"].to_numpy()
        for col, dst in (("yellow_sectors", ysec), ("double_yellow_sectors", dsec)):
            ex = df.select("fi", col).filter(pl.col(col).list.len() > 0).explode(col).drop_nulls()
            if not ex.is_empty():
                sec = ex[col].to_numpy().astype(int) - 1
                ok = (sec >= 0) & (sec < S.MAX_MARSHAL)
                dst[ex["fi"].to_numpy()[ok], sec[ok]] = True
    for part in sorted((ds_dir / "edges").glob("part-*.parquet")):
        df = decimated(part)
        if df.is_empty():
            continue
        fi, a, b = df["fi"].to_numpy(), df["src_slot"].to_numpy().astype(int), df["dst_slot"].to_numpy().astype(int)
        edge[fi, a, b] = S._apply(df.select(S.EDGE_FEATURES).to_numpy(), espec).astype(np.float16)
        eexists[fi, a, b] = True

    for arr in (drv, drv_tok, present, glb, ysec, dsec, edge, eexists, times):
        arr.flush()
    np.save(out_dir / "track.npy", track_tok.astype(np.float32))
    np.save(out_dir / "track_marshal.npy", track_marshal)
    meta = {
        "cache_version": CACHE_VERSION, "source": str(ds_dir), "name": ds_dir.parent.name if ds_dir.name.startswith("sequence") else ds_dir.name,
        "stride": stride, "n_frames_src": n_frames, "n_frames": T, "first_frame": f0, "step_s": manifest["step_s"] * stride,
        "n_track_tokens": n_track_tokens, "max_drivers": D, "driver_codes": manifest["driver_codes"],
        "norm": rn.to_dict(), "t_start": manifest["t_start"], "t_stop": manifest["t_stop"],
    }  # fmt: skip
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return out_dir


def _memmap(path: Path, shape, dtype, fill):
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[...] = fill
    return arr


def build_all(src: str | Path, dst: str | Path, stride: int = 5, overwrite: bool = False, **kw) -> list[Path]:
    """Build a cache for every dataset under ``src``; races that fail are reported and skipped."""
    out = []
    for ds in find_datasets(src):
        name = ds.parent.name if ds.name.startswith("sequence") else ds.name
        t0 = time.time()
        try:
            out.append(build_race_cache(ds, Path(dst) / name.replace(".ds", ""), stride, overwrite=overwrite, **kw))
            print(f"[cache] {name}: ok ({time.time() - t0:.0f}s)")
        except Exception as e:  # noqa: BLE001 - one broken race must not stop a bulk build
            print(f"[cache] {name}: SKIPPED ({type(e).__name__}: {e})")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with the .ds dataset directories (searched recursively)")
    ap.add_argument("--dst", required=True, help="cache output folder (one sub-folder per race)")
    ap.add_argument("--stride", type=int, default=5, help="keep every n-th frame (10 Hz source -> 2 Hz at 5)")
    ap.add_argument("--tokens", type=int, default=S.N_TRACK_TOKENS, help="track tokens per snapshot")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    caches = build_all(a.src, a.dst, a.stride, a.overwrite, n_track_tokens=a.tokens)
    print(f"{len(caches)} race caches in {a.dst}")


if __name__ == "__main__":
    main()
