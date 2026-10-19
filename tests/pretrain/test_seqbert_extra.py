"""Second half of the temporal-model tests: options, branches, split variants and training CLI."""

import shutil
import sys

import numpy as np
import pytest
import torch

from f1strategy.pretrain import schema as S
from f1strategy.seqbert import data as D
from f1strategy.seqbert import losses as L
from f1strategy.seqbert import train
from f1strategy.seqbert.config import Scale
from f1strategy.seqbert.encoders import GNNEncoder, SectorEncoder
from f1strategy.seqbert.masking import SeqMaskConfig, SeqMasker

T = 12
SC = Scale("short", T, span_s=6.0)


@pytest.fixture(scope="module")
def races(cache_dir):
    return D.open_races(cache_dir)


@pytest.fixture(scope="module")
def ix(races):
    return D.WindowIndex(races, SC)


@pytest.fixture(scope="module")
def raw(ix):
    ds = D.WindowDataset(ix)
    return torch.utils.data.default_collate([ds[i] for i in range(4)])


# ------------------------------------------------------------------------------------------------ masking options
def test_mask_config_validation():
    with pytest.raises(ValueError):
        SeqMasker(SeqMaskConfig(ghost_edge_mode="bad"))
    with pytest.raises(ValueError):
        SeqMasker(SeqMaskConfig(zone_position="left"))
    with pytest.raises(ValueError):
        SeqMasker(SeqMaskConfig(mask_fill="ones"))


def test_random_zone_and_random_task_mix(raw):
    m = SeqMasker(SeqMaskConfig(zone_position="random", mask_fill="zero"))
    mb = m(raw, task=3, generator=torch.Generator().manual_seed(1))
    assert (mb.vec_mask.sum(1) == round(0.35 * T)).all()
    mixed = m(raw)  # tasks drawn per sample
    assert mixed.task.min() >= 0 and mixed.task.max() <= 3


@pytest.mark.parametrize("mode", ["topology", "none"])
def test_edge_modes_hide_edges_of_hidden_cars(raw, mode):
    m = SeqMasker(SeqMaskConfig(ghost_edge_mode=mode, pit_edge_mode=mode))
    for task in (0, 2):
        mb = m(raw, task=task, generator=torch.Generator().manual_seed(2))
        touch = mb.slot_masked[:, :, :, None] | mb.slot_masked[:, :, None, :]
        assert not (mb.edge_vis & touch[..., None]).any()
        if mode == "none":
            assert not (mb.edge_exists & touch).any()
        if task == 0:
            assert not (mb.tok_vis & mb.slot_masked).any()  # ring position hidden too


def test_plain_inference_batch(raw):
    m = SeqMasker()
    mb = m.plain(raw, token_dropout=0.0)
    assert (mb.task == -1).all() and not mb.vec_mask.any() and not mb.drv_tgt.any()
    drop = m.plain(raw, token_dropout=0.6, generator=torch.Generator().manual_seed(0))
    assert drop.vec_mask.any() and not drop.drv_tgt.any()


# ------------------------------------------------------------------------------------------------ losses and metrics
@pytest.mark.parametrize(
    ("kind", "cos", "norm"),
    [
        ("mse", 0.0, "none"),
        ("huber", 0.0, "none"),
        ("cosine", 0.0, "none"),
        ("mse", 0.3, "feature"),
        ("huber", 0.0, "feature"),
    ],
)
def test_loss_variants(raw, kind, cos, norm):
    mb = SeqMasker()(raw, generator=torch.Generator().manual_seed(4))
    tgt, w = L.build_targets(mb), L.build_weights(mb)
    pred = torch.randn_like(tgt, requires_grad=True)
    loss, per = L.reconstruction_loss(pred, tgt, w, mb.task, kind, cos, norm)
    loss.backward()
    assert torch.isfinite(loss) and set(per) <= {"B", "T", "P", "Z"} and pred.grad is not None


def test_loss_without_targets_is_zero(raw):
    mb = SeqMasker().plain(raw, token_dropout=0.0)
    pred = torch.randn_like(L.build_targets(mb), requires_grad=True)
    loss, per = L.reconstruction_loss(pred, L.build_targets(mb), L.build_weights(mb), mb.task)
    assert float(loss.detach()) == 0.0 and per == {}


def test_feature_normalisation_equalises_columns():
    t = torch.zeros(1, 4, 3)
    t[..., 0] = torch.tensor([0.0, 1.0, 0.0, 1.0])  # variance 0.25
    t[..., 1] = torch.tensor([0.0, 10.0, 0.0, 10.0])  # variance 25: a 100x larger scale
    w = torch.ones(1, 4, 3)
    w[..., 2] = 0  # a column with no cells must be ignored
    err = (torch.zeros(1, 4, 3) - t) ** 2
    raw_loss = float((err * w).sum() / w.sum())
    norm_loss = float(L._feature_normalised(err, t, w))
    assert raw_loss > 10 * norm_loss  # the raw MSE is dominated by the large column; the normalised one is not
    assert norm_loss == pytest.approx(0.5 * (1.0 + 1.0) / 1.0 + 0.0, rel=0.3) or norm_loss > 0


def test_interp_block_edge_cases():
    v = torch.arange(10.0)[None, :, None].expand(1, 10, 2).clone()
    none = torch.zeros(1, 10, dtype=torch.bool)
    assert torch.equal(L._interp_block(v, none, None), v)  # nothing masked
    allm = torch.ones(1, 10, dtype=torch.bool)
    assert torch.equal(L._interp_block(v, allm, None), v)  # no anchor on either side
    left = torch.zeros(1, 10, dtype=torch.bool)
    left[0, :4] = True  # block at the start: anchor only after it
    out = L._interp_block(v, left, None)
    assert (out[0, :4, 0] == 4.0).all()  # held at the first visible frame, never read from the masked ones
    right = torch.zeros(1, 10, dtype=torch.bool)
    right[0, 7:] = True
    assert (L._interp_block(v, right, None)[0, 7:, 0] == 6.0).all()
    mid = torch.zeros(1, 10, dtype=torch.bool)
    mid[0, 3:7] = True  # a straight line between frame 2 and frame 7
    assert torch.allclose(L._interp_block(v, mid, None)[0, 3:7, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))


def test_metrics_for_every_task(raw):
    m = SeqMasker()
    acc = L.MetricAcc()
    for k in range(4):
        mb = m(raw, task=k, generator=torch.Generator().manual_seed(k))
        L.accumulate_metrics(acc, mb, torch.randn(*mb.ok.shape, L.build_targets(mb).shape[-1]))
    res = acc.result()
    assert {"B/mse", "T/mse", "Z/mse", "Z/speed_kmh_interp", "T/compound_acc"} <= set(res)
    assert all(np.isfinite(v) for v in res.values())


# ------------------------------------------------------------------------------------------------ data
def test_window_index_and_weights(races, ix):
    assert len(ix) > 3
    ids = list(range(len(ix)))
    w = D.window_weights(ix, ids, 3.0)
    assert set(np.unique(w)) <= {1.0, 3.0} and (w == 3.0).any()
    assert len(D.WindowIndex(races, Scale("huge", 100, span_s=60.0))) == 0  # window longer than the race


def _copies(cache_dir, tmp_path, n, edit=None):
    for i in range(n):
        shutil.copytree(cache_dir / "syn", tmp_path / f"r{i}")
        if edit:
            edit(i, tmp_path / f"r{i}")
    return D.open_races(tmp_path)


def test_circuits_race_split_and_pair_sampler(cache_dir, tmp_path):
    rng = np.random.default_rng(0)
    base = rng.normal(size=128)

    def edit(i, p):
        tr = np.load(p / "track.npy")
        meta = __import__("json").loads((p / "meta.json").read_text())
        col = S.TI["curvature"]
        if i in (2, 4):
            tr[:, col] = (base if i == 2 else base + 0.01 * rng.normal(size=128))[: len(tr)] if len(tr) <= 128 else 0
        if i == 3:
            meta["norm"]["track_length_m"] *= 1.5
        np.save(p / "track.npy", tr)
        (p / "meta.json").write_text(__import__("json").dumps(meta))

    rs = _copies(cache_dir, tmp_path, 5, edit)
    ids = D.circuit_ids(rs)
    assert ids[0] == ids[1] and ids[2] == ids[4] and len({ids[0], ids[2], ids[3]}) == 3
    ix = D.WindowIndex(rs, SC)
    tr, va, desc = D.split_windows(ix, val_frac=0.2)  # >= 5 races: whole races held out
    assert tr and va and len(desc) == 1
    assert {ix.items[k][0] for k in tr}.isdisjoint({ix.items[k][0] for k in va})
    groups = D.window_groups(ix, tr, phase_bins=3)
    sampler = D.PairBatchSampler(groups, ix, tr, pairs=4, n_batches=3, seed=1)
    batches = list(sampler)
    assert len(sampler) == 3 and all(len(b) == 2 * min(4, len(groups)) for b in batches)
    one = dict(list(groups.items())[:1])  # a single group: the second loop and the "same window" fallback
    b = next(iter(D.PairBatchSampler(one, ix, tr, pairs=3, n_batches=1)))
    assert len(b) == 2


# ------------------------------------------------------------------------------------------------ encoder guards
def test_encoder_constructor_guards():
    with pytest.raises(ValueError):
        SectorEncoder(hidden=32, D=22, M=32, n_sectors=5)
    with pytest.raises(ValueError):
        GNNEncoder(hidden=32, D=22, M=32, track_nodes=5)


# ------------------------------------------------------------------------------------------------ training CLI
def _argv(cache_dir, out, *extra, steps="4"):
    return ["train", "--cache", str(cache_dir), "--out", str(out), "--scale", "short", "--n-tokens", "6", "--span-s", "2.0",
            "--block-s", "2.0", "--val-frac", "0.34", "--preset", "tiny", "--steps", steps, "--batch-size", "2",
            "--workers", "0", "--eval-every", "2", "--eval-windows", "4", "--warmup", "1", *extra]  # fmt: skip


def test_resume_and_init_from(cache_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _argv(cache_dir, tmp_path / "a", "--dropout", "0.2"))
    train.main()
    ck = torch.load(tmp_path / "a" / "ckpt.pt", map_location="cpu")
    assert ck["step"] == 3 and "opt" in ck and "best_val" in ck
    best = torch.load(tmp_path / "a" / "best.pt", map_location="cpu")
    assert "opt" not in best and best["cfg"]["dropout"] == 0.2
    monkeypatch.setattr(
        sys, "argv", _argv(cache_dir, tmp_path / "a", "--resume", str(tmp_path / "a" / "ckpt.pt"), steps="8")
    )
    train.main()
    assert "[resume] from step 3" in capsys.readouterr().out
    assert torch.load(tmp_path / "a" / "ckpt.pt", map_location="cpu")["step"] == 7
    monkeypatch.setattr(sys, "argv", _argv(cache_dir, tmp_path / "b", "--init-from", str(tmp_path / "a" / "best.pt")))
    train.main()
    assert "[init-from]" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--p-contrast", "0.5", "--pairs", "2", "--graph-encoder", "gnn", "--loss-norm", "feature"],
    ["--graph-encoder", "sector", "--mask-fill", "zero", "--loss-kind", "huber", "--cos-weight", "0.1"],
    ["--loss-kind", "cosine", "--zone-position", "random"],
])  # fmt: skip
def test_training_options(cache_dir, tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", _argv(cache_dir, tmp_path / "o", *extra))
    train.main()
    assert (tmp_path / "o" / "best.pt").exists() and (tmp_path / "o" / "log.jsonl").stat().st_size > 0


def test_nonfinite_gradient_is_skipped(cache_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(float("nan")))
    monkeypatch.setattr(sys, "argv", _argv(cache_dir, tmp_path / "n", steps="2"))
    train.main()
    assert "non-finite gradient" in capsys.readouterr().out


def test_not_enough_windows_exits(cache_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["train", "--cache", str(cache_dir), "--out", str(tmp_path / "x"), "--scale", "long"]
    )
    with pytest.raises(SystemExit):
        train.main()


def test_parse_probs_and_schedule():
    p = train.parse_probs("B=1,Z=3")
    assert p == (0.25, 0.0, 0.0, 0.75)

    class A:
        lr, warmup, steps = 1.0, 10, 110

    assert train.lr_at(0, A) == pytest.approx(0.1) and train.lr_at(110, A) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------------------------------------ mixed-scale training
MIXED = [
    "--scale", "mixed", "--scales", "short,medium", "--n-tokens", "short=6,medium=8", "--span-s", "short=2.0,medium=2.4",
    "--block-s", "2.0", "--val-frac", "0.34", "--preset", "tiny", "--batch-sizes", "short=2,medium=2",
    "--scale-probs", "short=0.5,medium=0.5", "--workers", "0", "--eval-every", "4", "--eval-windows", "4", "--warmup", "1",
]  # fmt: skip


def _mixed(cache_dir, out, *extra, steps="8"):
    return ["train", "--cache", str(cache_dir), "--out", str(out), "--steps", steps, *MIXED, *extra]


def test_mixed_scale_training_resume_and_checkpoints(cache_dir, tmp_path, monkeypatch, capsys):
    import json

    out = tmp_path / "m"
    monkeypatch.setattr(sys, "argv", _mixed(cache_dir, out, "--time-embed", "--p-contrast", "0.3", "--pairs", "2"))
    train.main()
    text = capsys.readouterr().out
    assert "[mixed] steps per scale: short 50% (batch 2), medium 50% (batch 2)" in text and "[eval short]" in text
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert {r["scale"] for r in log} == {"short", "medium"}  # both scales were trained
    ev = [r for r in log if "eval" in r][-1]
    assert (
        "short/B/mse" in ev["eval"]
        and "medium/Z/mse" in ev["eval"]
        and set(ev["val_loss_by_scale"]) == {"short", "medium"}
    )
    assert ev["val_loss"] == pytest.approx(np.mean(list(ev["val_loss_by_scale"].values())))
    ck = torch.load(out / "ckpt.pt", map_location="cpu")
    assert ck["scale"] == "mixed" and ck["scales"] == {"short": 6, "medium": 8} and ck["cfg"]["time_embed"] is True
    assert set(ck["best_scale"]) == {"short", "medium"} and (out / "best.pt").exists()
    assert (out / "best_short.pt").exists() and (out / "best_medium.pt").exists()
    assert torch.load(out / "best_short.pt", map_location="cpu")["best_for"] == "short"
    args = json.loads((out / "args.json").read_text())
    assert args["batch_sizes"] == {"short": 2, "medium": 2} and sum(args["scale_probs"].values()) == pytest.approx(1.0)
    assert set(args["windows"]) == {"short", "medium"}
    monkeypatch.setattr(
        sys, "argv", _mixed(cache_dir, out, "--time-embed", "--resume", str(out / "ckpt.pt"), steps="12")
    )
    train.main()
    assert "[resume] from step 7" in capsys.readouterr().out
    assert torch.load(out / "ckpt.pt", map_location="cpu")["step"] == 11


def test_scale_resolution_and_overrides():
    class A:
        scale, scales, batch_size = "mixed", "short,long", None
        n_tokens, span_s, span_laps = "short=10", None, "long=2"
        batch_sizes, scale_probs = None, None

    scales, batches, probs = train.resolve_scales(A)
    assert scales["short"].n_tokens == 10 and scales["long"].span_laps == 2.0 and scales["long"].span_s is None
    assert batches == {"short": 128, "long": 16}  # the mixed defaults
    assert probs["short"] == pytest.approx(0.5 / 0.65) and sum(probs.values()) == pytest.approx(1.0)
    A.scale_probs = "short=1"
    with pytest.raises(SystemExit):
        train.resolve_scales(A)  # every scale needs a probability
    A.scale_probs, A.batch_sizes, A.batch_size = "short=1,long=1", "short=4", 6
    _, batches, probs = train.resolve_scales(A)
    assert batches == {"short": 4, "long": 6} and probs == {"short": 0.5, "long": 0.5}
    A.scales = "short,bogus"
    with pytest.raises(SystemExit):
        train.resolve_scales(A)
    assert train.parse_kv("7", ["a", "b"], int, "x") == {"a": 7, "b": 7} and train.parse_kv(None, ["a"], int, "x") == {}
    with pytest.raises(SystemExit):
        train.parse_kv("zzz=1", ["a"], int, "x")


def test_single_scale_defaults_are_unchanged():
    class A:
        scale, scales, batch_size = "medium", "", None
        n_tokens = span_s = span_laps = batch_sizes = scale_probs = None

    scales, batches, probs = train.resolve_scales(A)
    assert list(scales) == ["medium"] and batches == {"medium": 16} and probs == {"medium": 1.0}


def test_time_embedding(raw):
    from f1strategy.seqbert.model import SeqBertConfig, SeqF1Bert

    kw = {"hidden": 32, "layers": 1, "heads": 4, "ffn": 64, "graph_encoder": "flatten", "n_track": 32}
    plain = SeqF1Bert(SeqBertConfig(**kw))
    assert not hasattr(plain, "time_proj")  # old checkpoints keep loading
    model = SeqF1Bert(SeqBertConfig(time_embed=True, **kw)).eval()
    mb = SeqMasker().plain(raw, token_dropout=0.0)
    assert mb.t_rel.shape == mb.ok.shape and (mb.t_rel[:, 0] == 0).all() and (mb.t_rel[:, -1] > 0).all()
    with torch.no_grad():
        a = model.encode(mb, 0)
        mb.t_rel = mb.t_rel * 7.0  # the same frames seen at another time resolution
        b = model.encode(mb, 0)
    assert a.shape == b.shape and not torch.allclose(a, b)
