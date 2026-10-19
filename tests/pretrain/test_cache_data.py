"""Cache building and the race-cache readers (``pretrain/cache.py`` and ``pretrain/data.py``)."""

import json
import shutil

import numpy as np
import polars as pl
import pytest

from f1strategy.pretrain import cache as C
from f1strategy.pretrain import schema as S
from f1strategy.pretrain.data import PretrainDataset, frame_weights, open_races, split_races

DI = S.DI


@pytest.fixture(scope="module")
def races(cache_dir):
    return open_races(cache_dir)


# ------------------------------------------------------------------------------------------------ cache content
def test_cache_matches_source(synthetic_ds, races):
    r = races[0]
    assert r.meta["stride"] == 2 and r.n_frames == 60 and r.track.shape == (32, S.F_TRACK_STATIC)
    rn = S.RaceNorm(**r.meta["norm"])
    src = pl.read_parquet(synthetic_ds / "drivers" / "part-00000.parquet").filter(pl.col("frame") == 10).sort("slot")
    ref = S.normalise_driver(src.select(S.DRIVER_FEATURES).to_numpy(), rn)
    np.testing.assert_allclose(np.asarray(r.a["drv"][5][: len(src)]), ref, equal_nan=True, atol=1e-6)
    assert r.a["present"][5].sum() == 6 and not r.a["present"][5][6:].any()
    assert np.isnan(r.a["drv"][5][0, DI["gap_ahead_s"]])  # a null stays NaN, never 0
    assert r.a["ysec"][25][2] and r.a["dsec"][25][4] and not r.a["ysec"][5].any()  # marshal sectors 3 and 5 flagged
    assert r.track_marshal.max() == 8 and r.track_marshal.min() >= 1
    node = (10 * 3 + 2 * 25) % 200  # a driver on node n sits on token n * 32 // 200
    assert r.a["drv_tok"][5][2] == node * 32 // 200


def test_edges_cached(races):
    a = races[0].a
    assert a["eexists"][3].sum() == 6 * 4 - 2 * 2 - 2  # |a-b|<=2 pairs among 6 cars
    assert np.isnan(np.asarray(a["edge"][3][~a["eexists"][3]], dtype=np.float32)).all()


def test_race_readers(races):
    r = races[0]
    v = r.valid_frames()
    assert v.dtype == bool and len(v) == r.n_frames and v.any()
    assert r.flagged_frames().any()
    item = r[3]
    assert item["drv"].shape[-1] == S.F_DRIVER and item["present"].shape[0] == r.a["present"].shape[1]
    ds = PretrainDataset(races, [(0, 1), (0, 2)])
    assert len(ds) == 2 and ds[1]["drv"].shape == ds[0]["drv"].shape


# ------------------------------------------------------------------------------------------------ splits and weights
def test_split_blocks_embargo_and_no_overlap(races):
    step = races[0].meta["step_s"]  # 0.2 s per cached frame, 60 frames
    sp = split_races(races, val_frac=0.25, embargo_s=2 * step, block_s=5 * step)
    tr, va = np.array(sorted(f for _, f in sp.train)), np.array(sorted(f for _, f in sp.val))
    assert len(tr) and len(va) and not set(tr) & set(va)
    assert {i // 5 for i in va} == {3, 7, 11}  # every 4th block, interleaved over the whole race
    assert min(abs(t - v) for t in tr for v in va) > 2  # embargo around every validation block
    assert tr.min() < va.min() and tr.max() > va.min()  # not cut from the end of the race


def test_split_whole_races(cache_dir, tmp_path):
    for i in range(6):
        shutil.copytree(cache_dir / "syn", tmp_path / f"r{i}")
    rs = open_races(tmp_path)
    sp = split_races(rs, val_frac=0.34)
    assert len(sp.val_races) == 2 and len(sp.train_races) == 4
    assert {ri for ri, _ in sp.val}.isdisjoint({ri for ri, _ in sp.train})


def test_flagged_frames_are_oversampled(races):
    items = [(0, i) for i in range(races[0].n_frames)]
    w = frame_weights(races, items, 5.0)
    fl = races[0].flagged_frames()
    assert fl.sum() > 0 and np.all(w[fl] == 5.0) and np.all(w[~fl] == 1.0)


# ------------------------------------------------------------------------------------------------ cache builder
def test_build_is_idempotent_and_overwritable(synthetic_ds, tmp_path):
    out = C.build_race_cache(synthetic_ds, tmp_path / "x", stride=2, n_track_tokens=32)
    stamp = (out / "meta.json").stat().st_mtime_ns
    assert C.build_race_cache(synthetic_ds, tmp_path / "x", stride=2, n_track_tokens=32) == out
    assert (out / "meta.json").stat().st_mtime_ns == stamp  # skipped
    C.build_race_cache(synthetic_ds, tmp_path / "x", stride=2, n_track_tokens=32, overwrite=True)
    assert (out / "meta.json").stat().st_mtime_ns != stamp


def test_build_rejects_bad_sizes(synthetic_ds, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="fewer than"):
        C.build_race_cache(synthetic_ds, tmp_path / "a", stride=2, n_track_tokens=500)
    with pytest.raises(ValueError, match="drivers"):
        C.build_race_cache(synthetic_ds, tmp_path / "b", stride=2, n_track_tokens=32, max_drivers=3)
    monkeypatch.setattr(S, "MAX_MARSHAL", 2)
    with pytest.raises(ValueError, match="marshal"):
        C.build_race_cache(synthetic_ds, tmp_path / "c", stride=2, n_track_tokens=32)


def test_find_datasets_build_all_and_cli(synthetic_ds, tmp_path, capsys):
    src = tmp_path / "src"
    shutil.copytree(synthetic_ds, src / "2020_round_01.ds" / "sequence_lap1-3")  # the nesting the scraper writes
    shutil.copytree(synthetic_ds, src / "2020_round_02.ds")
    broken = src / "2020_round_03.ds"
    shutil.copytree(synthetic_ds, broken)
    m = json.loads((broken / "manifest.json").read_text())
    m["format"] = "something-else"
    (broken / "manifest.json").write_text(json.dumps(m))
    assert len(C.find_datasets(src)) == 3
    out = C.build_all(src, tmp_path / "dst", stride=2, n_track_tokens=32)
    assert sorted(p.name for p in out) == ["2020_round_01", "2020_round_02"]
    assert "SKIPPED" in capsys.readouterr().out
    C.main(["--src", str(src), "--dst", str(tmp_path / "dst2"), "--stride", "2", "--tokens", "32"])
    assert "2 race caches" in capsys.readouterr().out
