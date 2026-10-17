"""
losses.py - reconstruction targets, losses, metrics in physical units, baselines and the contrastive loss.

The 1-layer head outputs, for every time step, the *flattened frame*:
    [ drivers (D*F) | global (G) | track-dynamic (2*40 marshal-sector flags + cars per ring token) ]
Each task selects the cells it is responsible for (a per-cell weight mask); the loss is MSE (default), Huber, and/or
1 - cosine similarity over the selected cells (``--loss-kind``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fn

from f1strategy.pretrain import schema as S
from f1strategy.seqbert.masking import TASKS, SeqBatch, gap_to_seconds

D_, F_, G_ = S.MAX_DRIVERS, S.F_DRIVER, S.F_GLOBAL


def build_targets(mb: SeqBatch) -> torch.Tensor:
    B, T = mb.ok.shape
    trk = torch.cat([mb.ysec.float(), mb.dsec.float(), mb.n_cars_tok / 5.0], -1)
    return torch.cat([mb.drv_x.flatten(2), mb.glb_x, trk], -1)


def build_weights(mb: SeqBatch) -> torch.Tensor:
    """[B, T, O] 1.0 where the sample's task asks for a reconstruction."""
    B, T, D, F = mb.drv_x.shape
    z = (mb.task == TASKS.index("Z"))[:, None] & mb.vec_mask  # [B, T] Z frames
    w_drv = mb.drv_tgt | (mb.drv_valid & z[..., None, None])
    w_glb = mb.glb_valid & z[..., None]
    n_trk = 2 * S.MAX_MARSHAL + mb.n_cars_tok.shape[-1]
    w_trk = z[..., None].expand(B, T, n_trk)
    return torch.cat([w_drv.flatten(2), w_glb, w_trk], -1).float()


def reconstruction_loss(pred, tgt, w, task, kind: str = "mse", cos_weight: float = 0.0):
    """-> (total, {task: loss}). Per task: mean over its cells; tasks are averaged afterwards."""
    pred, tgt = pred.float(), tgt.float()
    per, terms = {}, []
    for k, name in enumerate(TASKS):
        sel = task == k
        wk = w[sel]
        if not sel.any() or wk.sum() == 0:
            continue
        p, t = pred[sel], tgt[sel]
        err = (p - t) ** 2 if kind != "huber" else Fn.huber_loss(p, t, reduction="none", delta=1.0)
        loss = (err * wk).sum() / wk.sum()
        if kind == "cosine" or cos_weight > 0:
            pw, tw = (p * wk).flatten(0, 1), (t * wk).flatten(0, 1)
            has = wk.flatten(0, 1).sum(-1) > 0
            cos = (1 - Fn.cosine_similarity(pw[has], tw[has], dim=-1, eps=1e-6)).mean()
            loss = cos if kind == "cosine" else loss + cos_weight * cos
        per[name] = loss
        terms.append(loss)
    if not terms:
        return pred.sum() * 0.0, per
    return torch.stack(terms).mean(), per


def contrastive_loss(z: torch.Tensor, temperature: float = 0.1):
    """NT-Xent on 2P embeddings, partner of i is (i + P) mod 2P. -> (loss, top1 retrieval accuracy)."""
    z = Fn.normalize(z.float(), dim=-1)
    n = z.shape[0]
    P = n // 2
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(n, dtype=torch.bool, device=z.device), float("-inf"))
    pos = (torch.arange(n, device=z.device) + P) % n
    loss = Fn.cross_entropy(sim, pos)
    acc = (sim.argmax(1) == pos).float().mean()
    return loss, acc


# ------------------------------------------------------------------------------------------------ metrics
def _split(recon):
    B, T, _ = recon.shape
    return recon[..., : D_ * F_].view(B, T, D_, F_)


class MetricAcc:
    """Accumulates (sum of abs error, count) per metric across eval batches."""

    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key, err_sum, n):
        self.s[key] = self.s.get(key, 0.0) + float(err_sum)
        self.n[key] = self.n.get(key, 0) + int(n)

    def result(self):
        return {k: self.s[k] / self.n[k] for k in self.s if self.n[k] > 0}


def _phys(x, name):
    if name == "gap_to_leader_s":
        return gap_to_seconds(x)
    return x * S.PHYS_SCALE.get(name, 1.0)


FEATS = {
    "B": ["position", "speed_kmh", "gap_to_leader_s", "tyre_age"],
    "T": ["tyre_age", "stint", "pit_stops"],
    "P": ["gap_to_leader_s", "position", "speed_kmh", "tyre_age"],
    "Z": ["speed_kmh", "throttle", "rpm", "position", "tyre_age"],
}
PHYS_UNIT = {"gap_to_leader_s": 1.0, "stint": 4.0, "pit_stops": 4.0}


@torch.no_grad()
def accumulate_metrics(acc: MetricAcc, mb: SeqBatch, recon: torch.Tensor):
    """MAE in physical units for the task's cells, the batch-mean baseline ('_mean') and, for task Z, linear
    interpolation across the masked block ('_interp')."""
    pred = _split(recon.float())
    x = mb.drv_x
    comp = [S.DI[c] for c in S.COMPOUND_COLS]
    for k, name in enumerate(TASKS):
        sel = mb.task == k
        if not sel.any():
            continue
        z = mb.vec_mask[sel][..., None, None] if name == "Z" else None
        cell = (mb.drv_valid[sel] & z) if name == "Z" else mb.drv_tgt[sel]
        p, t = pred[sel], x[sel]
        for f in FEATS[name]:
            i = S.DI[f]
            m = cell[..., i]
            if m.sum() == 0:
                continue
            u = PHYS_UNIT.get(f, 1.0)
            pp = _phys(p[..., i], f) if f not in PHYS_UNIT or f == "gap_to_leader_s" else p[..., i] * u
            tt = _phys(t[..., i], f) if f not in PHYS_UNIT or f == "gap_to_leader_s" else t[..., i] * u
            acc.add(f"{name}/{f}_mae", ((pp - tt).abs() * m).sum(), m.sum())
            mean = (tt * m).sum() / m.sum()
            acc.add(f"{name}/{f}_mean", ((tt - mean).abs() * m).sum(), m.sum())
            if name == "Z":
                ip = _interp_block(t[..., i], mb.vec_mask[sel], mb.drv_valid[sel][..., i])
                ii = _phys(ip, f)
                acc.add(f"Z/{f}_interp", ((ii - tt).abs() * m).sum(), m.sum())
        cm = (
            (mb.comp_tgt[sel] & (z[..., 0] if z is not None else True))
            if name != "Z"
            else (mb.drv_valid[sel][..., comp].all(-1) & mb.vec_mask[sel][..., None] & mb.present[sel])
        )
        if cm.sum() > 0 and name in ("B", "T", "P", "Z"):
            hit = (p[..., comp].argmax(-1) == mb.comp_idx[sel]) & cm
            acc.add(f"{name}/compound_acc", hit.sum(), cm.sum())
            maj = torch.mode(mb.comp_idx[sel][cm]).values
            acc.add(f"{name}/compound_majority_acc", ((mb.comp_idx[sel] == maj) & cm).sum(), cm.sum())
    # overall MSE per task
    t_all, w = build_targets(mb), build_weights(mb)
    for k, name in enumerate(TASKS):
        sel = mb.task == k
        if sel.any() and w[sel].sum() > 0:
            acc.add(f"{name}/mse", (((recon.float()[sel] - t_all[sel]) ** 2) * w[sel]).sum(), w[sel].sum())


def _interp_block(v: torch.Tensor, vec_mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """v [b, T, D]: replace the masked block by a straight line between the frames just before and after it."""
    out = v.clone()
    T = v.shape[1]
    for b in range(v.shape[0]):
        idx = vec_mask[b].nonzero().flatten()
        if len(idx) == 0:
            continue
        a, e = int(idx[0]) - 1, int(idx[-1]) + 1
        if a < 0 and e >= T:
            continue
        a, e = max(a, 0), min(e, T - 1)
        lam = ((idx.float() - a) / max(e - a, 1))[:, None]
        out[b, idx] = v[b, a][None] * (1 - lam) + v[b, e][None] * lam
    return out
