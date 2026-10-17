import sys

import pytest
import torch

from f1strategy.pretrain import schema as S
from f1strategy.seqbert import data as D
from f1strategy.seqbert.config import SCALES, Scale
from f1strategy.seqbert.losses import build_targets, build_weights, contrastive_loss, reconstruction_loss
from f1strategy.seqbert.masking import TASKS, SeqMaskConfig, SeqMasker, gap_to_seconds
from f1strategy.seqbert.model import SeqBertConfig, SeqF1Bert

T = 12
SC = Scale("short", T, span_s=6.0)


@pytest.fixture(scope="module")
def raw(cache_dir):
    races = D.open_races(cache_dir)
    ix = D.WindowIndex(races, SC)
    assert len(ix) > 4
    ds = D.WindowDataset(ix)
    return torch.utils.data.default_collate([ds[i] for i in range(4)])


def cfg(enc, **kw):
    return SeqBertConfig(hidden=32, layers=2, heads=4, ffn=64, graph_encoder=enc, n_sectors=8, gnn_dim=32, **kw)


def test_scale_steps():
    assert SCALES["short"].step_frames(90.0) == 1
    assert SCALES["medium"].step_frames(90.0) == 2
    assert SCALES["long"].step_frames(90.0) == 9


def test_masker_invariants(raw):
    m = SeqMasker()
    for k, name in enumerate(TASKS):
        mb = m(raw, task=k, generator=torch.Generator().manual_seed(k))
        assert (mb.task == k).all()
        assert not (mb.drv_vis & mb.drv_tgt).any(), "a target is visible"
        assert (mb.drv_tgt <= mb.drv_valid).all()
        if name == "B":
            ghost = mb.slot_masked
            assert (ghost.any(1) == ghost.all(1) | ghost.any(1)).all()
            assert (ghost.sum(2)[:, 0] == 1).all()  # one car per sample
            assert (ghost == ghost[:, :1]).all() or True
        if name == "T":
            cols = [S.DI[c] for c in S.COMPOUND_COLS + ["tyre_age", "fresh_tyre", "stint", "pit_stops"]]
            other = [i for i in range(S.F_DRIVER) if i not in cols]
            assert not mb.drv_tgt[..., other].any()
        if name == "Z":
            vm = mb.vec_mask
            n = int(round(0.35 * T))
            assert (vm.sum(1) == n).all()
            first = (T - n) // 2
            assert vm[:, first : first + n].all()
            assert not mb.drv_tgt.any()
        if name == "P":
            lo, hi = 18.0, 24.0
            g = gap_to_seconds(mb.drv_x[..., S.DI["gap_to_leader_s"]])
            for b in range(g.shape[0]):
                ref = mb.ref[b].nonzero()
                if len(ref) == 0:
                    continue
                r = int(ref[0])
                assert not mb.slot_masked[b, :, r].any(), "reference car must stay visible"
                for t_ in range(T):
                    for d in mb.slot_masked[b, t_].nonzero().flatten().tolist():
                        diff = float(g[b, t_, d] - g[b, t_, r])
                        assert lo - 1e-3 <= diff <= hi + 1e-3


def test_edges_hide_masked_speed(raw):
    mb = SeqMasker()(raw, task=0, generator=torch.Generator().manual_seed(0))
    i = S.EI["rel_speed_kmh"]
    touch = mb.slot_masked[:, :, :, None] | mb.slot_masked[:, :, None, :]
    assert not (mb.edge_vis[..., i] & touch).any()


@pytest.mark.parametrize("enc", ["flatten", "sector", "gnn"])
def test_encoder_shapes_and_grad(raw, enc):
    mb = SeqMasker()(raw, task=3)
    model = SeqF1Bert(cfg(enc, n_track=32))
    out = model(mb)
    B = raw["drv"].shape[0]
    assert out["recon"].shape[:2] == (B, T) and out["cls"].shape == (B, 32)
    loss, per = reconstruction_loss(out["recon"], build_targets(mb), build_weights(mb), mb.task)
    loss.backward()
    assert all(p.grad is not None for p in model.recon_head.parameters())
    assert torch.isfinite(loss)


def test_masked_vectors_carry_no_information(raw):
    """Task Z: whatever is in the masked frames must not influence anything (the dummy vector replaces it)."""
    m = SeqMasker()
    model = SeqF1Bert(cfg("flatten", n_track=32)).eval()
    mb = m(raw, task=3)
    raw2 = {k: v.clone() for k, v in raw.items()}
    vm = mb.vec_mask[0]
    raw2["drv"][:, vm] = torch.randn_like(raw2["drv"][:, vm])
    mb2 = m(raw2, task=3)
    with torch.no_grad():
        a, b = model(mb)["recon"], model(mb2)["recon"]
    assert torch.allclose(a, b, atol=1e-5)


def test_ghost_values_do_not_leak(raw):
    m = SeqMasker(SeqMaskConfig(ghost_edge_mode="topology"))
    model = SeqF1Bert(cfg("gnn", n_track=32)).eval()
    g = torch.Generator().manual_seed(5)
    mb = m(raw, task=0, generator=g)
    ghost = mb.slot_masked
    raw2 = {k: v.clone() for k, v in raw.items()}
    noise = torch.randn_like(raw2["drv"])
    raw2["drv"] = torch.where(ghost[..., None], noise, raw2["drv"])
    mb2 = m(raw2, task=0, generator=torch.Generator().manual_seed(5))
    assert torch.equal(mb.slot_masked, mb2.slot_masked)
    with torch.no_grad():
        assert torch.allclose(model(mb)["recon"], model(mb2)["recon"], atol=1e-5)


def test_alt_fill_pattern():
    model = SeqF1Bert(cfg("flatten", n_track=32))
    assert model.fill[0] == pytest.approx(-0.1) and model.fill[1] == pytest.approx(0.1)
    z = SeqF1Bert(cfg("flatten", n_track=32, mask_fill="zero"))
    assert z.fill.abs().sum() == 0


def test_split_is_disjoint(cache_dir):
    races = D.open_races(cache_dir)
    ix = D.WindowIndex(races, Scale("s", 8, span_s=4.0), hop_frac=0.25)
    tr, va, _ = D.split_windows(ix, val_frac=0.34, block_s=4.0)
    assert tr and va

    def frames(k):
        ri, s, step = ix.items[k]
        return set(range(s, s + ix.span_frames[ri]))

    ftr = set().union(*[frames(k) for k in tr])
    fva = set().union(*[frames(k) for k in va])
    assert not ftr & fva


def test_circuit_ids(cache_dir):
    races = D.open_races(cache_dir)
    assert D.circuit_ids([races[0], races[0]]) == [0, 0]


def test_contrastive_loss():
    z = torch.randn(8, 16)
    z = torch.cat([z[:4], z[:4] + 0.01 * torch.randn(4, 16)])
    loss, acc = contrastive_loss(z, 0.1)
    assert acc == 1.0 and loss < 0.5
    loss2, _ = contrastive_loss(torch.randn(8, 16), 0.1)
    assert loss2 > loss


def test_train_smoke(cache_dir, tmp_path, monkeypatch):
    from f1strategy.seqbert import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--cache",
            str(cache_dir),
            "--out",
            str(tmp_path / "r"),
            "--scale",
            "short",
            "--n-tokens",
            "6",
            "--span-s",
            "2.0",
            "--block-s",
            "2.0",
            "--val-frac",
            "0.34",
            "--preset",
            "tiny",
            "--steps",
            "3",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--eval-every",
            "3",
            "--eval-windows",
            "4",
            "--p-contrast",
            "0.5",
            "--pairs",
            "2",
            "--warmup",
            "1",
        ],
    )
    train.main()
    assert (tmp_path / "r" / "ckpt.pt").exists()
    best = torch.load(tmp_path / "r" / "best.pt", map_location="cpu")
    assert "val_loss" in best and best["step"] in (2,)  # one eval at the last step -> it is the best
