"""FtModel on a tiny random backbone: shapes, pooling, freeze modes, loss.  Needs torch and the repo's seqbert package."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
S = pytest.importorskip("f1strategy.pretrain.schema")
from f1strategy.finetune.losses import ft_loss  # noqa: E402
from f1strategy.finetune.model import FtModel, HeadConfig  # noqa: E402
from f1strategy.seqbert.model import SeqBertConfig, SeqF1Bert  # noqa: E402

B, T, D, M = 2, 6, S.MAX_DRIVERS, S.N_TRACK_TOKENS


def fake_batch(ok_last=True):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    ok = torch.ones(B, T, dtype=torch.bool)
    ok[:, -1] = ok_last
    return SimpleNamespace(
        ok=ok, present=torch.ones(B, T, D, dtype=torch.bool), drv_vis=torch.ones(B, T, D, S.F_DRIVER, dtype=torch.bool),
        tok_vis=torch.ones(B, T, D, dtype=torch.bool), tok=torch.randint(0, M, (B, T, D)), drv_x=r(B, T, D, S.F_DRIVER),
        edge_x=r(B, T, D, D, S.F_EDGE), edge_vis=torch.ones(B, T, D, D, S.F_EDGE, dtype=torch.bool),
        edge_exists=torch.zeros(B, T, D, D, dtype=torch.bool), glb_x=r(B, T, S.F_GLOBAL),
        glb_valid=torch.ones(B, T, S.F_GLOBAL, dtype=torch.bool), flag_x=torch.zeros(B, T, M, S.N_FLAGS, dtype=torch.bool),
        trk_static=r(B, M, S.F_TRACK_STATIC), vec_mask=torch.zeros(B, T, dtype=torch.bool),
    )  # fmt: skip


def make(mode="full", **hk):
    cfg = SeqBertConfig(hidden=32, layers=3, heads=2, ffn=64, graph_encoder="flatten", dropout=0.1)
    bb = SeqF1Bert(cfg)
    del bb.recon_head, bb.proj_head
    m = FtModel(bb, HeadConfig(**hk))
    m.set_mode(mode, unfreeze_last=1)
    return m


def targets():
    laps = torch.full((B, D), float("nan"))
    laps[:, :10] = torch.arange(1, 11).float()
    return {"valid": torch.ones(B, D, dtype=torch.bool), "laps": laps, "cmp": torch.where(torch.isfinite(laps), 1, -1),
            "nomore": ~torch.isfinite(laps)}  # fmt: skip


def test_shapes_and_decode():
    m = make(pool="cls_last_mean", horizons=(3, 5))
    out = m(fake_batch(), 0)
    assert out["pit"].shape == (B, D, 2) and out["laps"].shape == (B, D) and out["cmp"].shape == (B, D, 5)
    d = m.decode(out)
    assert (d["laps"] >= 1).all() and torch.allclose(d["cmp_prob"].sum(-1), torch.ones(B, D), atol=1e-5)


def test_last_token_is_the_last_usable_frame():
    m = make(pool="last").eval()
    f_ok, f_bad = m.features(fake_batch(True), 0), m.features(fake_batch(False), 0)
    assert not torch.allclose(f_ok, f_bad)  # the unusable last frame is skipped, the previous token is read instead


@pytest.mark.parametrize(
    "mode,expect_backbone", [("frozen", False), ("partial", True), ("full", True), ("scratch", True)]
)
def test_freeze_modes(mode, expect_backbone):
    m = make(mode)
    loss, _ = ft_loss(m(fake_batch(), 0), targets(), m.hc)
    loss.backward()
    assert all(p.grad is not None for p in m.head.parameters())
    bb_grad = [p.grad is not None and p.grad.abs().sum() > 0 for p in m.backbone.parameters() if p.requires_grad]
    assert bool(bb_grad) == expect_backbone
    if mode == "partial":
        assert not any(p.requires_grad for p in m.backbone.graph_enc.parameters())
        assert all(p.requires_grad for p in m.backbone.encoder.layers[-1].parameters())
        assert not any(p.requires_grad for p in m.backbone.encoder.layers[0].parameters())


def test_frozen_backbone_is_deterministic_even_in_train_mode():
    m = make("frozen")
    m.train()
    b = fake_batch()
    assert torch.equal(m.features(b, 0), m.features(b, 0))  # no dropout in a frozen backbone


def test_mlp_head_and_task_subset():
    m = make(kind="mlp", layers=2, hidden=16, tasks=("laps",))
    loss, parts = ft_loss(m(fake_batch(), 0), targets(), m.hc)
    assert set(parts) == {"laps"} and torch.isfinite(loss)
