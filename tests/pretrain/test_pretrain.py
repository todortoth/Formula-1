import copy
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from f1strategy.pretrain import schema as S
from f1strategy.pretrain.data import PretrainDataset, frame_weights, open_races, split_races
from f1strategy.pretrain.losses import compute_losses
from f1strategy.pretrain.masking import MaskConfig, Masker
from f1strategy.pretrain.model import F1Bert, F1BertConfig, load_encoder

A, B, C, D = range(4)
DI, GI, EI = S.DI, S.GI, S.EI


@pytest.fixture(scope="module")
def races(cache_dir):
    return open_races(cache_dir)


@pytest.fixture(scope="module")
def raw(races):
    items = [(0, i) for i in range(races[0].n_frames)]
    return next(iter(DataLoader(PretrainDataset(races, items), batch_size=len(items))))


def mask(raw, task, seed=0, **cfg):
    g = torch.Generator().manual_seed(seed)
    return Masker(MaskConfig(**cfg))(raw, task=task, generator=g)


# ------------------------------------------------------------------------------------------------ cache
def test_cache_matches_source(cache_dir, synthetic_ds, races):
    import polars as pl

    r = races[0]
    assert r.meta["stride"] == 2 and r.n_frames == 60 and r.track.shape == (32, S.F_TRACK_STATIC)
    rn = S.RaceNorm(**r.meta["norm"])
    src = pl.read_parquet(synthetic_ds / "drivers" / "part-00000.parquet").filter(pl.col("frame") == 10).sort("slot")
    ref = S.normalise_driver(src.select(S.DRIVER_FEATURES).to_numpy(), rn)
    np.testing.assert_allclose(np.asarray(r.a["drv"][5][: len(src)]), ref, equal_nan=True, atol=1e-6)
    assert r.a["present"][5].sum() == 6 and not r.a["present"][5][6:].any()
    assert np.isnan(r.a["drv"][5][0, DI["gap_ahead_s"]])  # a null stays NaN, never 0
    # flags: marshal sectors 3 (yellow) and 5 (double yellow) while frames 40..69
    assert r.a["ysec"][25][2] and r.a["dsec"][25][4] and not r.a["ysec"][5].any()
    assert r.track_marshal.max() == 8 and r.track_marshal.min() >= 1
    # driver on node n sits on token n * 32 // 200
    node = (10 * 3 + 2 * 25) % 200
    assert r.a["drv_tok"][5][2] == node * 32 // 200


def test_edges_cached(races):
    a = races[0].a
    assert a["eexists"][3].sum() == 6 * 4 - 2 * 2 - 2  # |a-b|<=2 pairs among 6 cars
    assert np.isnan(np.asarray(a["edge"][3][~a["eexists"][3]], dtype=np.float32)).all()


# ------------------------------------------------------------------------------------------------ split / sampling
def test_split_blocks_embargo_and_no_overlap(races):
    step = races[0].meta["step_s"]  # 0.2 s per cached frame, 60 frames
    sp = split_races(races, val_frac=0.25, embargo_s=2 * step, block_s=5 * step)
    tr, va = np.array(sorted(f for _, f in sp.train)), np.array(sorted(f for _, f in sp.val))
    assert len(tr) and len(va) and not set(tr) & set(va)
    assert {i // 5 for i in va} == {3, 7, 11}  # every 4th block, interleaved over the whole race
    assert (
        min(abs(t - v) for t in tr for v in va) > 2
    )  # embargo: no training frame within 2 frames of a validation frame
    assert tr.min() < va.min() and tr.max() > va.min()  # not cut from the end of the race


def test_flagged_frames_are_oversampled(races):
    items = [(0, i) for i in range(races[0].n_frames)]
    w = frame_weights(races, items, 5.0)
    fl = races[0].flagged_frames()
    assert fl.sum() > 0 and np.all(w[fl] == 5.0) and np.all(w[~fl] == 1.0)


# ------------------------------------------------------------------------------------------------ task A
def test_task_A_hides_leaking_channels(raw):
    mb = mask(raw, A, p_cell=0.15)
    tgt = mb.drv_tgt
    tele = [DI[n] for n in S.TELEMETRY_TARGETS]
    assert tgt[..., tele].any() and not tgt[..., [i for i in range(S.F_DRIVER) if i not in tele]].any()
    sp = mb.drv_tgt[..., DI["speed_kmh"]]
    assert sp.any()
    for n in ("speed_kmh", "vx_mps", "vy_mps"):  # speed masked -> velocity components hidden with it
        assert not mb.drv_vis[..., DI[n]][sp].any()
    # edges touching a car with hidden speed lose rel_speed_kmh, in both directions
    inc = sp[:, :, None] | sp[:, None, :]
    assert not mb.edge_vis[..., EI["rel_speed_kmh"]][inc].any()
    assert mb.edge_vis[..., EI["gap_s"]][inc & mb.edge_exists_true & ~torch.isnan(raw["edge"][..., EI["gap_s"]])].any()
    rate = tgt[..., tele].sum().item() / (mb.present.sum().item() * len(tele))
    assert 0.10 < rate < 0.20


# ------------------------------------------------------------------------------------------------ task B
@pytest.mark.parametrize("mode", ["relational", "topology", "none"])
def test_ghost_car_is_fully_hidden(raw, mode):
    mb = mask(raw, B, ghost_edge_mode=mode)
    assert (mb.ghost.sum(1) == 1).all() and (mb.ghost & ~mb.present).sum() == 0
    g = mb.ghost
    assert not mb.drv_vis[g].any()  # every feature of the ghost is hidden
    assert mb.drv_tgt[g][:, [i for i in range(S.F_DRIVER) if i not in [DI[c] for c in S.COMPOUND_COLS]]].any()
    others = (~g & mb.present)[..., None].expand_as(mb.drv_vis)
    assert mb.drv_vis[others].float().mean() > 0.8  # the other cars stay visible
    inc = g[:, :, None] | g[:, None, :]
    if mode == "relational":
        assert not mb.edge_vis[..., EI["rel_speed_kmh"]][inc].any()
        assert mb.edge_vis[..., EI["gap_s"]][inc & mb.edge_exists_true].any() and mb.tok_vis[g].all()
    else:
        assert not mb.edge_vis[inc].any() and not mb.tok_vis[g].any()
        assert (mb.edge_exists[inc] == mb.edge_exists_true[inc]).all() == (mode == "topology")


# ------------------------------------------------------------------------------------------------ task C
def test_flag_global_mode_hides_every_flag_signal(raw):
    mb = mask(raw, C, c_global_prob=1.0)
    assert not mb.flag_vis.any()  # all six flag bits on every track token
    gcols = [GI[n] for n in S.GLOBAL_FLAG_FEATURES]
    assert not mb.glb_vis[:, gcols].any()
    rest = [i for i in range(S.F_GLOBAL) if i not in gcols]
    assert mb.glb_vis[:, rest].all() and not mb.glb_tgt[:, rest].any()
    yellow_frames = mb.flag_x[..., 0].sum(1) > 0
    assert yellow_frames.any() and mb.flag_tgt[..., 0].any()  # yellow sectors exist and are targets
    # the truth that the head is trained on really is the sector state from the data
    np.testing.assert_array_equal(
        mb.flag_x[0, :, 0].bool().numpy(),
        (raw["ysec"][0].gather(0, (raw["tmar"][0] - 1).clamp(min=0)) & (raw["tmar"][0] > 0)).numpy(),
    )


def test_flag_local_mode_hides_only_chosen_sectors(raw):
    mb = mask(raw, C, c_global_prob=0.0, c_sector_prob=0.2, c_yellow_bias=1.0)
    hid = ~mb.flag_vis[..., 0]  # tokens whose yellow bit is hidden
    assert hid.any() and not hid.all()
    assert mb.flag_vis[..., 2:].all()  # SC / VSC / red stay visible in local mode
    assert (
        mb.glb_vis[:, GI["is_sc"]].all() and not mb.glb_vis[:, GI["is_yellow"]].any()
    )  # summary that would reveal it is hidden
    assert not mb.glb_vis[:, GI["n_yellow_sectors"]].any()
    # all tokens of a masked marshal sector are hidden together
    tm = mb.flag_vis[..., 0].numpy()
    marsh = raw["tmar"].numpy()
    for b in range(tm.shape[0]):
        for s in np.unique(marsh[b]):
            assert len(set(tm[b][marsh[b] == s])) == 1
    # a flagged frame has its yellow sector masked (c_yellow_bias=1)
    assert (mb.flag_tgt[..., 0] & (mb.flag_x[..., 0] > 0.5)).any()


# ------------------------------------------------------------------------------------------------ task D
def test_strategy_groups_and_edge_leaks(raw):
    mb = mask(raw, D, p_group=0.3)
    for g, cols in S.STRATEGY_GROUPS.items():
        idx = [DI[c] for c in cols]
        hidden = ~mb.drv_vis[..., idx[0]] & mb.present  # first column of the group
        assert hidden.any()
    gaps = [DI[c] for c in S.STRATEGY_GROUPS["gaps"]]
    hid_g = (~mb.drv_vis[..., gaps]).all(-1) & mb.present & ~torch.isnan(raw["drv"][..., gaps]).any(-1)
    inc = hid_g[:, :, None] | hid_g[:, None, :]
    for n in S.EDGE_LEAKS["gaps"]:
        assert not mb.edge_vis[..., EI[n]][inc].any()
    assert mb.comp_tgt.any() and (mb.comp_idx[mb.comp_tgt] < 3).all()
    assert not mb.drv_tgt[..., [DI[c] for c in S.COMPOUND_COLS]].any()  # compound is a CE target, not a cell target


def test_each_sample_has_targets_of_its_task_only(raw):
    mb = Masker(MaskConfig())(raw, generator=torch.Generator().manual_seed(3))
    assert set(mb.task.tolist()) == {0, 1, 2, 3}
    has_drv = mb.drv_tgt.flatten(1).any(1) | mb.comp_tgt.any(1)
    has_c = mb.flag_tgt.flatten(1).any(1) | mb.glb_tgt.any(1)
    assert (has_c == (mb.task == C)).all() and not (has_drv & (mb.task == C)).any()
    assert not (mb.drv_tgt.flatten(1).any(1) & (mb.task == C)).any()


def test_masks_are_reproducible_with_a_generator(raw):
    a, b = mask(raw, A, seed=7), mask(raw, A, seed=7)
    assert torch.equal(a.drv_tgt, b.drv_tgt) and not torch.equal(a.drv_tgt, mask(raw, A, seed=8).drv_tgt)


# ------------------------------------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def tiny():
    return F1Bert(F1BertConfig.preset("tiny", n_track_tokens=32, dropout=0.0)).eval()


@pytest.mark.parametrize("task", [A, B, C, D])
def test_forward_backward_all_tasks(tiny, raw, task):
    small = {k: v[:8] for k, v in raw.items()}
    mb = mask(small, task, ghost_edge_mode="topology")
    out = tiny(mb)
    assert (
        out["driver"].shape == (8, 22, S.F_DRIVER)
        and out["flags"].shape == (8, 32, 2)
        and out["global"].shape == (8, S.F_GLOBAL)
    )
    loss, st = compute_losses(out, mb)
    assert torch.isfinite(loss) and f"loss_{'ABCD'[task]}" in st
    tiny.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in tiny.parameters() if p.grad is not None)


def test_output_independent_of_hidden_values(tiny, raw):
    """Overwrite every hidden cell with garbage: the encoder must not notice (all task modes)."""
    small = {k: v[:6] for k, v in raw.items()}
    for task in (A, B, C, D):
        for mode in ("relational", "topology", "none"):
            mb = mask(small, task, ghost_edge_mode=mode)
            mb2 = copy.copy(mb)
            junk = lambda t: torch.randn_like(t) * 50  # noqa: E731
            mb2.drv_x = torch.where(mb.drv_vis, mb.drv_x, junk(mb.drv_x))
            mb2.glb_x = torch.where(mb.glb_vis, mb.glb_x, junk(mb.glb_x))
            mb2.edge_x = torch.where(mb.edge_vis, mb.edge_x, junk(mb.edge_x))
            mb2.flag_x = torch.where(mb.flag_vis, mb.flag_x, junk(mb.flag_x))
            mb2.tok = torch.where(mb.tok_vis, mb.tok, torch.randint_like(mb.tok, 0, 32))
            with torch.no_grad():
                o1, o2 = tiny(mb), tiny(mb2)
            for k in ("driver", "flags", "global"):
                torch.testing.assert_close(
                    o1[k][mb.present] if k == "driver" else o1[k],
                    o2[k][mb.present] if k == "driver" else o2[k],
                    atol=1e-5,
                    rtol=1e-4,
                )


def test_driver_slots_are_permutation_equivariant(tiny, raw):
    small = {k: v[:4].clone() for k, v in raw.items()}
    mb = mask(small, A)
    perm = torch.randperm(22, generator=torch.Generator().manual_seed(0))
    p = copy.copy(mb)
    for k in ("present", "drv_x", "drv_vis", "drv_tgt", "ghost", "tok", "tok_vis", "comp_tgt", "comp_idx"):
        setattr(p, k, getattr(mb, k)[:, perm])
    for k in ("edge_x", "edge_vis", "edge_exists", "edge_exists_true"):
        setattr(p, k, getattr(mb, k)[:, perm][:, :, perm])
    with torch.no_grad():
        o, op = tiny(mb), tiny(p)
    torch.testing.assert_close(o["driver"][:, perm], op["driver"], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(o["flags"], op["flags"], atol=1e-4, rtol=1e-3)


def test_absent_drivers_do_not_influence_others(tiny, raw):
    small = {k: v[:4].clone() for k, v in raw.items()}
    mb = mask(small, A)
    a = tiny.eval()(mb)["flags"]
    p = copy.copy(mb)
    p.drv_x = mb.drv_x.clone()
    p.drv_x[:, 10:] = 99.0  # slots 10.. are padding (only 6 cars)
    p.drv_vis = mb.drv_vis.clone()
    p.drv_vis[:, 10:] = True
    with torch.no_grad():
        b = tiny(p)["flags"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------------------------------ training
def test_loss_goes_down_on_a_fixed_batch(raw):
    torch.manual_seed(0)
    small = {k: v[:16] for k, v in raw.items()}
    model = F1Bert(F1BertConfig.preset("tiny", n_track_tokens=32, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    masker = Masker()
    first = last = None
    for i in range(60):
        mb = masker(small, generator=torch.Generator().manual_seed(i % 4))  # a few fixed masks -> it can memorise
        loss, _ = compute_losses(model(mb), mb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < 0.7 * first


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_train_cli_and_checkpoint_roundtrip(cache_dir, tmp_path):
    from f1strategy.pretrain.train import main

    main(["--cache", str(cache_dir), "--out", str(tmp_path), "--preset", "tiny", "--steps", "6", "--batch-size", "4", "--warmup", "2",
          "--eval-every", "6", "--eval-frames", "16", "--save-every", "6", "--log-every", "3", "--device", "cpu", "--val-frac", "0.25", "--embargo-s", "0.4", "--block-s", "1"])  # fmt: skip
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    enc = load_encoder(str(tmp_path / "best.pt"))
    assert enc.cfg.n_track_tokens == 32
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any("val/loss" in r for r in rows)
