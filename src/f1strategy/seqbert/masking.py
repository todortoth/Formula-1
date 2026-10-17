"""
masking.py - the pretext tasks of the sequence model, as batched masking over windows [B, T, ...].

Every sample of a batch gets exactly one task (so each loss stays clean and its leakage is controlled on its own):

  B  ghost car      ALL features of one car are hidden in ALL T frames (the car is a "ghost" for the whole window).
  T  tyre set       compound, tyre age, fresh/used, stint and pit-stop count of one car are hidden in all frames:
                    "which tyres is it on and how long ago were they fitted".
  P  pit-return     the cars that sit 18-24 s behind a reference car (the traffic it would rejoin into after a pit
     window         stop, ~ the pit-lane time loss) are hidden, frame by frame (the window moves with the cars).
  Z  zone           a contiguous block of whole graph *vectors* (default: the middle 35 % of the window) is replaced
                    by a constant dummy vector (zeros, or -0.1/+0.1 alternating) AFTER the graph encoder.

Tasks A (telemetry cells) and C (flags) of the single-snapshot model are gone. A masked value is never silently
zeroed: the encoders get ``[value * visible, visible]``. Whatever else restates a hidden value is hidden with it
(leak maps in ``pretrain/schema.py``):

  * a hidden car's speed        -> its vx / vy and ``rel_speed_kmh`` on every edge touching it
  * hidden gaps / position      -> ``gap_s``, ``order_neighbor``, ``drs_window`` on edges touching it
  * task P additionally hides ``gap_ahead_s`` / ``gap_behind_s`` of ALL cars in frames with a hidden car, because the
    gap of a visible neighbour to the hidden car IS the hidden car's gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from f1strategy.pretrain import schema as S

TASKS = ("B", "T", "P", "Z")
TYRE_COLS = ["tyre_age", "fresh_tyre", "stint", "pit_stops"]  # + the compound one-hot
GAP_RAW_A, GAP_RAW_SCALE = 10.0, 2.0  # gap features are stored as asinh(x / 10) / 2


def gap_to_seconds(x: torch.Tensor) -> torch.Tensor:
    """Inverse of the cache normalisation of the gap features."""
    return GAP_RAW_A * torch.sinh(GAP_RAW_SCALE * x.clamp(-10, 10))


@dataclass
class SeqMaskConfig:
    task_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)  # B, T, P, Z
    zone_frac: float = 0.35  # task Z: share of the window that is masked
    zone_position: str = "middle"  # "middle" (as specified) or "random"
    mask_fill: str = "alt"  # dummy vector: "alt" (-0.1/+0.1 alternating) or "zero"
    pit_window_s: tuple[float, float] = (18.0, 24.0)  # task P: gap behind the reference car
    ghost_edge_mode: str = "relational"  # task B: relational | topology | none (see pretrain/masking.py)
    pit_edge_mode: str = "topology"  # task P: same options, default harder
    extra: dict = field(default_factory=dict)


@dataclass
class SeqBatch:
    task: torch.Tensor  # [B] long (index into TASKS, -1 = none)
    ok: torch.Tensor  # [B, T] bool, frame usable (else it is a padding key)
    present: torch.Tensor  # [B, T, D]
    drv_x: torch.Tensor  # [B, T, D, F] clean normalised values
    drv_vis: torch.Tensor  # [B, T, D, F] bool, what the encoder may see
    drv_valid: torch.Tensor  # [B, T, D, F] bool, value exists
    drv_tgt: torch.Tensor  # [B, T, D, F] bool, reconstruction targets (tasks B, T, P)
    slot_masked: torch.Tensor  # [B, T, D] bool, car has hidden features in this frame
    comp_tgt: torch.Tensor  # [B, T, D] bool (metrics: compound accuracy)
    comp_idx: torch.Tensor  # [B, T, D] long
    tok: torch.Tensor  # [B, T, D] long ring token (-1 unknown)
    tok_vis: torch.Tensor  # [B, T, D] bool
    glb_x: torch.Tensor  # [B, T, G]
    glb_valid: torch.Tensor  # [B, T, G] bool
    trk_static: torch.Tensor  # [B, M, 18]
    flag_x: torch.Tensor  # [B, T, M, 6] bool (yellow, double yellow, sc, vsc, vsc_ending, red) - never masked
    ysec: torch.Tensor  # [B, T, 40] bool
    dsec: torch.Tensor  # [B, T, 40] bool
    n_cars_tok: torch.Tensor  # [B, T, M] float, cars per track token (true)
    edge_x: torch.Tensor  # [B, T, D, D, Fe]
    edge_vis: torch.Tensor  # [B, T, D, D, Fe] bool
    edge_exists: torch.Tensor  # [B, T, D, D] bool (visible existence)
    vec_mask: torch.Tensor  # [B, T] bool, whole graph vector replaced by the dummy vector (task Z / token dropout)
    ref: torch.Tensor  # [B, D] bool, reference car of task P (visible), else False

    def to(self, device) -> SeqBatch:
        return SeqBatch(**{k: v.to(device) for k, v in self.__dict__.items()})


def _idx(names, table):
    return torch.tensor([table[n] for n in names], dtype=torch.long)


class SeqMasker:
    def __init__(self, cfg: SeqMaskConfig | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg or SeqMaskConfig()
        for m in (self.cfg.ghost_edge_mode, self.cfg.pit_edge_mode):
            if m not in ("relational", "topology", "none"):
                raise ValueError(f"unknown edge mode {m!r}")
        if self.cfg.zone_position not in ("middle", "random") or self.cfg.mask_fill not in ("alt", "zero"):
            raise ValueError("zone_position must be middle|random, mask_fill alt|zero")
        self.device = torch.device(device)
        d = self.device
        self.tyre_cols = _idx(S.COMPOUND_COLS + TYRE_COLS, S.DI).to(d)
        self.comp_cols = _idx(S.COMPOUND_COLS, S.DI).to(d)
        self.speed_cols = _idx(["speed_kmh", "vx_mps", "vy_mps"], S.DI).to(d)
        self.neigh_gap_cols = _idx(["gap_ahead_s", "gap_behind_s"], S.DI).to(d)
        self.e_speed = _idx(S.EDGE_LEAKS["speed"], S.EI).to(d)
        self.e_gaps = _idx(S.EDGE_LEAKS["gaps"], S.EI).to(d)
        self.g_sc = _idx(["is_sc", "is_vsc", "is_vsc_ending", "is_red"], S.GI).to(d)
        self.probs = torch.tensor(self.cfg.task_probs, dtype=torch.float32, device=d)

    # ---------------------------------------------------------------------------------------------------------
    def _pick_car(self, score: torch.Tensor, gate: torch.Tensor, rnd) -> torch.Tensor:
        """One car per sample ([B, D] one-hot) from ``score`` [B, D] (> 0 = eligible), only where ``gate`` [B]."""
        s = score * (0.25 + rnd(*score.shape))
        pick = torch.zeros_like(score, dtype=torch.bool)
        pick[torch.arange(score.shape[0], device=score.device), s.argmax(1)] = True
        return pick & (score > 0) & gate[:, None]

    def __call__(
        self, raw: dict[str, torch.Tensor], task: int | None = None, generator: torch.Generator | None = None,
        token_dropout: float = 0.0,
    ) -> SeqBatch:  # fmt: skip
        """``raw``: collated windows. ``task`` forces one task (evaluation); ``task=-1`` = nothing masked (inference /
        contrastive views, optionally with ``token_dropout`` random vector masking)."""
        dev, cfg = self.device, self.cfg
        raw = {k: v.to(dev) for k, v in raw.items()}
        present, drv, glb = raw["present"], raw["drv"], raw["glb"]
        B, T, D, F = drv.shape
        M = raw["trk"].shape[1]
        Fe = raw["edge"].shape[-1]
        rnd = lambda *shape: torch.rand(*shape, device=dev, generator=generator)  # noqa: E731
        ok = raw["ok"]

        valid = ~torch.isnan(drv) & present[..., None]
        drv_x = torch.nan_to_num(drv)
        glb_valid = ~torch.isnan(glb)
        glb_x = torch.nan_to_num(glb)
        edge = raw["edge"].float()
        eexists = raw["eexists"]
        e_valid = ~torch.isnan(edge) & eexists[..., None]
        edge_x = torch.nan_to_num(edge)

        if task is None:
            task_id = torch.multinomial(self.probs, B, replacement=True, generator=generator)
        else:
            task_id = torch.full((B,), int(task), dtype=torch.long, device=dev)
        is_b, is_t, is_p, is_z = (task_id == k for k in range(4))

        hide = torch.zeros(B, T, D, F, dtype=torch.bool, device=dev)
        hide_pos = torch.zeros(B, T, D, dtype=torch.bool, device=dev)  # car location hidden
        full = torch.zeros(B, T, D, dtype=torch.bool, device=dev)  # car completely hidden (B, P)
        comp_tgt = torch.zeros_like(full)
        ref = torch.zeros(B, D, dtype=torch.bool, device=dev)

        frac_present = (present & ok[..., None]).float().mean(1)  # [B, D]
        elig = torch.where(frac_present > 0.5, frac_present, torch.zeros_like(frac_present))

        # ---- B: ghost car (all frames) ----------------------------------------------------------------------
        gb = self._pick_car(elig, is_b, rnd)
        ghost = gb[:, None, :] & present
        full |= ghost
        if cfg.ghost_edge_mode != "relational":
            hide_pos |= ghost

        # ---- T: tyre set (all frames) -----------------------------------------------------------------------
        gt = self._pick_car(elig, is_t, rnd)
        tcar = gt[:, None, :] & present
        hide[..., self.tyre_cols] |= tcar[..., None]
        comp_tgt |= tcar

        # ---- P: pit-return window -----------------------------------------------------------------------------
        gi = S.DI["gap_to_leader_s"]
        gsec = gap_to_seconds(drv_x[..., gi])  # [B, T, D] seconds behind the leader
        gok = valid[..., gi]
        lo, hi = cfg.pit_window_s
        # behind[b,t,x,j] = how far car j is behind candidate reference x
        behind = gsec[:, :, None, :] - gsec[:, :, :, None]
        sel = (behind >= lo) & (behind <= hi) & gok[:, :, None, :] & gok[:, :, :, None] & ok[:, :, None, None]
        eye = torch.eye(D, dtype=torch.bool, device=dev)
        sel &= ~eye
        cnt = sel.any(-1).float().sum(1)  # [B, D] frames in which candidate x has a window
        gp = self._pick_car(cnt, is_p, rnd)
        win = (sel & gp[:, None, :, None]).any(2)  # [B, T, D]
        full |= win
        ref |= gp
        if cfg.pit_edge_mode != "relational":
            hide_pos |= win
        pit_frames = win.any(-1)  # [B, T] frames that have a hidden car
        leak = torch.zeros_like(hide)  # hidden only to stop a leak, not a reconstruction target
        leak[..., self.neigh_gap_cols] |= (pit_frames[..., None] & present)[..., None]

        # ---- assemble the driver masks --------------------------------------------------------------------------
        hide |= full[..., None]
        tgt = hide & valid
        hide = hide | leak
        comp_valid = valid[..., self.comp_cols].all(-1)
        comp_tgt = (comp_tgt | full) & comp_valid
        comp_idx = drv_x[..., self.comp_cols].argmax(-1)
        tok_vis = present & (raw["tok"] >= 0) & ~hide_pos
        slot_masked = tgt.any(-1) & present

        # ---- edges: hide what restates a hidden driver feature -----------------------------------------------
        inc = lambda m: m[:, :, :, None] | m[:, :, None, :]  # noqa: E731 - edges touching a masked car
        eh = torch.zeros(B, T, D, D, Fe, dtype=torch.bool, device=dev)
        eh_exists = torch.zeros(B, T, D, D, dtype=torch.bool, device=dev)
        speed_hidden = hide[..., self.speed_cols[0]] | full
        gaps_hidden = hide[..., S.DI["position"]] | full
        e_spd, e_gp = inc(speed_hidden), inc(gaps_hidden)
        for col in self.e_speed:
            eh[..., col] |= e_spd
        for col in self.e_gaps:
            eh[..., col] |= e_gp
        for mode, who in ((cfg.ghost_edge_mode, ghost), (cfg.pit_edge_mode, win)):
            if mode in ("topology", "none"):
                eh |= inc(who)[..., None]
            if mode == "none":
                eh_exists |= inc(who)
        edge_vis = e_valid & ~eh
        edge_exists = eexists & ~eh_exists

        # ---- flags / cars per token (never masked, used by the encoders) ----------------------------------------
        tmar = raw["tmar"]
        sec_of_tok = (tmar - 1).clamp(min=0)  # [B, M]
        tok_known = tmar > 0
        ysec, dsec = raw["ysec"], raw["dsec"]
        gather = lambda x: torch.gather(x, 2, sec_of_tok[:, None, :].expand(B, T, M))  # noqa: E731
        yel = gather(ysec) & tok_known[:, None]
        dbl = gather(dsec) & tok_known[:, None]
        glb_sc = glb_x[..., self.g_sc] > 0.5  # [B, T, 4]
        flag_x = torch.cat([yel[..., None], dbl[..., None], glb_sc[:, :, None, :].expand(B, T, M, 4)], -1)
        seen = (present & (raw["tok"] >= 0)).float()
        n_cars = torch.zeros(B, T, M, device=dev).scatter_add_(2, raw["tok"].clamp(0, M - 1), seen)

        # ---- Z: vector masking --------------------------------------------------------------------------------
        span = max(1, int(round(cfg.zone_frac * T)))
        ar = torch.arange(T, device=dev)
        if cfg.zone_position == "middle":
            start = torch.full((B,), (T - span) // 2, device=dev)
        else:
            start = (rnd(B) * (T - span + 1)).floor().long()
        vec_mask = (ar[None] >= start[:, None]) & (ar[None] < start[:, None] + span) & is_z[:, None]
        if token_dropout > 0:
            vec_mask |= rnd(B, T) < token_dropout
        if task is not None and task < 0:
            tgt = torch.zeros_like(tgt)

        return SeqBatch(
            task=task_id if (task is None or task >= 0) else torch.full_like(task_id, -1),
            ok=ok, present=present, drv_x=drv_x, drv_vis=valid & ~hide, drv_valid=valid, drv_tgt=tgt,
            slot_masked=slot_masked, comp_tgt=comp_tgt, comp_idx=comp_idx, tok=raw["tok"], tok_vis=tok_vis,
            glb_x=glb_x, glb_valid=glb_valid, trk_static=torch.nan_to_num(raw["trk"]), flag_x=flag_x,
            ysec=ysec, dsec=dsec, n_cars_tok=n_cars, edge_x=edge_x, edge_vis=edge_vis, edge_exists=edge_exists,
            vec_mask=vec_mask, ref=ref,
        )  # fmt: skip

    def plain(self, raw, generator=None, token_dropout: float = 0.15) -> SeqBatch:
        """Everything visible (inference, contrastive views); ``token_dropout`` hides random graph vectors."""
        return self(raw, task=-1, generator=generator, token_dropout=token_dropout)
