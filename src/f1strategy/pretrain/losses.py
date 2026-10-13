"""
losses.py - per-task pre-training losses and validation metrics.

Loss per task (averaged over that task's target cells, then combined with ``task_weights``):
  A, B, D   Huber on continuous driver features, BCE on binary ones, cross-entropy for the tyre compound (6 classes).
  C         BCE on yellow / double yellow per masked track token (positives up-weighted) + BCE/Huber on the masked
            global status features.
Metrics are reported in physical units (km/h, %, rpm, laps, s, positions) next to a trivial baseline (predict the
batch mean), because a normalised loss on its own says little.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from f1strategy.pretrain import schema as S
from f1strategy.pretrain.masking import TASKS, MaskedBatch

_BIN_D = torch.zeros(S.F_DRIVER, dtype=torch.bool)
_BIN_D[[S.DI[n] for n in S.BIN_DRIVER]] = True
_BIN_G = torch.zeros(S.F_GLOBAL, dtype=torch.bool)
_BIN_G[[S.GI[n] for n in S.BIN_GLOBAL]] = True
_COMP = torch.tensor([S.DI[c] for c in S.COMPOUND_COLS])


def _cell_loss(pred: torch.Tensor, target: torch.Tensor, is_bin: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred, target.clamp(0, 1), reduction="none")
    hub = F.huber_loss(pred, target, reduction="none", delta=1.0)
    return torch.where(is_bin.to(pred.device), bce, hub)


def compute_losses(
    out: dict[str, torch.Tensor],
    mb: MaskedBatch,
    task_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    yellow_pos_weight: float = 20.0,
    global_pos_weight: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    dev = out["driver"].device
    pred = out["driver"].float()
    total = pred.new_zeros(())
    stats: dict[str, float] = {}

    cell = _cell_loss(pred, mb.drv_x, _BIN_D)  # [B, D, F]
    ce = F.cross_entropy(pred[..., _COMP.to(dev)].reshape(-1, len(_COMP)), mb.comp_idx.reshape(-1), reduction="none").view(mb.comp_idx.shape)

    # global-token loss (task C): pos_weight on the binary flag features
    gp = out["global"].float()
    g_bce = F.binary_cross_entropy_with_logits(gp, mb.glb_x.clamp(0, 1), reduction="none", pos_weight=gp.new_tensor(global_pos_weight))
    g_cell = torch.where(_BIN_G.to(dev), g_bce, F.huber_loss(gp, mb.glb_x, reduction="none", delta=1.0))
    fl = F.binary_cross_entropy_with_logits(out["flags"].float(), mb.flag_x[..., :2], reduction="none", pos_weight=gp.new_tensor(yellow_pos_weight))

    for k, name in enumerate(TASKS):
        sel = (mb.task == k)
        if not sel.any():
            continue
        s3 = sel[:, None, None]
        w = (mb.drv_tgt & s3).float()
        n = w.sum()
        parts, count = [], 0.0
        if n > 0:
            parts.append((cell * w).sum())
            count += n.item()
        if k in (1, 3):  # B and D carry compound targets
            wc = (mb.comp_tgt & sel[:, None]).float()
            if wc.sum() > 0:
                parts.append((ce * wc).sum())
                count += wc.sum().item()
        if k == 2:
            wf = (mb.flag_tgt & sel[:, None, None]).float()
            wg = (mb.glb_tgt & sel[:, None]).float()
            if wf.sum() > 0:
                parts.append((fl * wf).sum())
                count += wf.sum().item()
            if wg.sum() > 0:
                parts.append((g_cell * wg).sum())
                count += wg.sum().item()
        if not parts:
            continue
        loss = torch.stack(parts).sum() / max(count, 1.0)
        stats[f"loss_{name}"] = loss.item()
        total = total + task_weights[k] * loss
    stats["loss"] = total.item()
    return total, stats


class Metrics:
    """Accumulates sums over batches; ``result()`` gives physical-unit metrics per task."""

    def __init__(self):
        self.s: dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        self.s[key] = self.s.get(key, 0.0) + float(value)

    @torch.no_grad()
    def update(self, out: dict[str, torch.Tensor], mb: MaskedBatch) -> None:
        pred = out["driver"].float()
        x = mb.drv_x
        dev = pred.device

        def feat(name: str, mask: torch.Tensor, key: str, scale: float = 1.0, baseline: bool = True):
            i = S.DI[name]
            m = mask[..., i]
            n = m.sum().item()
            if n == 0:
                return
            err = (pred[..., i] - x[..., i]).abs()[m]
            self.add(f"{key}_n", n)
            self.add(f"{key}_abs", err.sum().item() * scale)
            if baseline:
                tgt = x[..., i][m]
                self.add(f"{key}_base_abs", (tgt - tgt.mean()).abs().sum().item() * scale)

        a = mb.drv_tgt & (mb.task == 0)[:, None, None]
        for nm in ("speed_kmh", "throttle", "rpm"):
            feat(nm, a, f"A_{nm}", S.PHYS_SCALE[nm])
        i = S.DI["brake"]
        mb_a = a[..., i]
        if mb_a.any():
            self.add("A_brake_n", mb_a.sum().item())
            self.add("A_brake_correct", ((pred[..., i] > 0) == (x[..., i] > 0.5))[mb_a].sum().item())
            self.add("A_brake_pos", (x[..., i] > 0.5)[mb_a].sum().item())

        b = mb.drv_tgt & (mb.task == 1)[:, None, None]
        for nm in ("speed_kmh", "progress_m", "position", "tyre_age"):
            feat(nm, b, f"B_{nm}", S.PHYS_SCALE.get(nm, 1.0), baseline=True)
        cont = ~_BIN_D.to(dev)
        cont[_COMP.to(dev)] = False
        bc = b & cont
        if bc.any():
            self.add("B_cont_n", bc.sum().item())
            self.add("B_cont_abs", (pred - x).abs()[bc].sum().item())
        self._compound(pred, mb, 1, "B")

        d = mb.drv_tgt & (mb.task == 3)[:, None, None]
        feat("tyre_age", d, "D_tyre_age", S.PHYS_SCALE["tyre_age"])
        feat("pace_last3_s", d, "D_pace", S.PHYS_SCALE["pace_last3_s"])
        feat("position", d, "D_position", S.PHYS_SCALE["position"])
        feat("gap_ahead_s", d, "D_gap_ahead_norm", 1.0)
        self._compound(pred, mb, 3, "D")

        c = mb.task == 2
        if c.any():
            y_pred = out["flags"][..., 0].float() > 0
            y_true = mb.flag_x[..., 0] > 0.5
            m = mb.flag_tgt[..., 0] & c[:, None]
            self.add("C_yellow_tp", (y_pred & y_true & m).sum().item())
            self.add("C_yellow_pred", (y_pred & m).sum().item())
            self.add("C_yellow_true", (y_true & m).sum().item())
            self.add("C_yellow_n", m.sum().item())
            gm = mb.glb_tgt & c[:, None]
            for nm in ("is_yellow", "is_sc", "is_vsc", "is_red"):
                j = S.GI[nm]
                mj = gm[:, j]
                if mj.any():
                    pj = out["global"][:, j].float() > 0
                    tj = mb.glb_x[:, j] > 0.5
                    self.add(f"C_{nm}_n", mj.sum().item())
                    self.add(f"C_{nm}_correct", ((pj == tj) & mj).sum().item())
                    self.add(f"C_{nm}_pos", (tj & mj).sum().item())
                    self.add(f"C_{nm}_tp", (pj & tj & mj).sum().item())

    def _compound(self, pred, mb, k, tag):
        m = mb.comp_tgt & (mb.task == k)[:, None]
        if m.any():
            logits = pred[..., _COMP.to(pred.device)]
            self.add(f"{tag}_compound_n", m.sum().item())
            self.add(f"{tag}_compound_correct", (logits.argmax(-1) == mb.comp_idx)[m].sum().item())

    def result(self) -> dict[str, float]:
        s, r = self.s, {}
        for key in {k[:-4] for k in s if k.endswith("_abs") and not k.endswith("_base_abs") and k != "B_cont_abs"}:
            if s.get(f"{key}_n"):
                r[f"{key}_mae"] = s[f"{key}_abs"] / s[f"{key}_n"]
                if f"{key}_base_abs" in s:
                    r[f"{key}_mae_baseline"] = s[f"{key}_base_abs"] / s[f"{key}_n"]
        if s.get("B_cont_n"):
            r["B_all_cont_mae_norm"] = s["B_cont_abs"] / s["B_cont_n"]
        for tag in ("A_brake", "B_compound", "D_compound"):
            if s.get(f"{tag}_n"):
                r[f"{tag}_acc"] = s[f"{tag}_correct"] / s[f"{tag}_n"]
        if s.get("A_brake_n"):
            r["A_brake_base_acc"] = max(s["A_brake_pos"], s["A_brake_n"] - s["A_brake_pos"]) / s["A_brake_n"]
        if s.get("C_yellow_n"):
            r["C_yellow_true_cells"] = s["C_yellow_true"]
            r["C_yellow_recall"] = s["C_yellow_tp"] / s["C_yellow_true"] if s["C_yellow_true"] else float("nan")
            r["C_yellow_precision"] = s["C_yellow_tp"] / s["C_yellow_pred"] if s["C_yellow_pred"] else float("nan")
        for nm in ("is_yellow", "is_sc", "is_vsc", "is_red"):
            if s.get(f"C_{nm}_n"):
                r[f"C_{nm}_acc"] = s[f"C_{nm}_correct"] / s[f"C_{nm}_n"]
                r[f"C_{nm}_positives"] = s[f"C_{nm}_pos"]
                if s[f"C_{nm}_pos"]:
                    r[f"C_{nm}_recall"] = s[f"C_{nm}_tp"] / s[f"C_{nm}_pos"]
        return r
