"""
masking.py - the four self-supervised pretext tasks, as batched GPU masking.

Every sample in a batch gets exactly one task, so the loss of each task stays clean and every task's leakage is
controlled separately:

  A  Telemetry value regression   15 % of the (driver, channel) cells of speed / throttle / brake / rpm are hidden.
  B  Ghost car                    ALL features of one random car are hidden; the other cars and the proximity edges stay.
  C  Flag / marshal-sector state  hide the flag state of the track and the global status flags and predict them back
                                  ("global" mode), or hide yellow / double yellow on a few marshal sectors ("local").
  D  Strategy state               15 % of the (driver, group) cells of tyre compound, tyre/stint, pace and gaps/position.

A masked value is never just zeroed in the input: the model receives ``[value * visible, visible]``. Whatever else in
the input restates a hidden value is hidden with it (leak maps in ``schema.py``):

  * speed         -> vx, vy (|v| = 0.96 * speed), and ``rel_speed_kmh`` on every edge touching the car
  * gap/position  -> ``gap_s``, ``order_neighbor``, ``drs_window`` on every edge touching the car
  * flags         -> the global status flags (is_sc ... drs_enabled) and, per sector, the global yellow summary
  * ghost car     -> see ``ghost_edge_mode``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from f1strategy.pretrain import schema as S

TASKS = ("A", "B", "C", "D")  # telemetry, ghost car, flags, strategy


@dataclass
class MaskConfig:
    task_probs: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
    p_cell: float = 0.15  # task A: probability that a (driver, channel) cell is masked
    p_group: float = 0.15  # task D: probability that a (driver, group) cell is masked
    # What the ghost car's proximity edges reveal (the car's own features are always fully hidden):
    #   "relational": edges and their gaps / lap_diff / order stay visible, only rel_speed_kmh (the car's speed) is hidden,
    #                 and the car's ring position stays known. This is the "keep the other 19 drivers and proximity
    #                 edges visible" setting. The car's progress / position are then largely derivable from the gaps.
    #   "topology":   edge existence stays, all edge values and the ring position of the car are hidden (harder).
    #   "none":       the car's edges vanish altogether (hardest).
    ghost_edge_mode: str = "relational"
    c_global_prob: float = 0.5  # task C: share of "global" mode (else "local" sector mode)
    c_sector_prob: float = 0.3  # local mode: chance that a marshal sector is masked
    c_yellow_bias: float = 0.7  # local mode: chance that a flagged sector is masked (so the task is not mostly empty)
    extra: dict = field(default_factory=dict)


@dataclass
class MaskedBatch:
    """Everything the model and the loss need. ``*_x`` are the clean normalised values (NaN already replaced by 0),
    ``*_vis`` what the model may see, ``*_tgt`` which cells are reconstruction targets."""

    task: torch.Tensor  # [B] long, index into TASKS
    present: torch.Tensor  # [B, D] bool
    # drivers
    drv_x: torch.Tensor  # [B, D, F]
    drv_vis: torch.Tensor  # [B, D, F] bool
    drv_tgt: torch.Tensor  # [B, D, F] bool
    ghost: torch.Tensor  # [B, D] bool
    tok: torch.Tensor  # [B, D] long, ring token (-1 unknown)
    tok_vis: torch.Tensor  # [B, D] bool, may the model see where the car is
    comp_tgt: torch.Tensor  # [B, D] bool, compound is a target (cross-entropy over the 6 classes)
    comp_idx: torch.Tensor  # [B, D] long
    # global
    glb_x: torch.Tensor  # [B, G]
    glb_vis: torch.Tensor  # [B, G] bool
    glb_tgt: torch.Tensor  # [B, G] bool
    # track tokens
    trk_static: torch.Tensor  # [B, M, 18]
    flag_x: torch.Tensor  # [B, M, 6] true flag bits (yellow, double yellow, sc, vsc, vsc_ending, red)
    flag_vis: torch.Tensor  # [B, M, 6] bool
    flag_tgt: torch.Tensor  # [B, M, 2] bool, targets: yellow / double yellow per token
    # edges
    edge_x: torch.Tensor  # [B, D, D, Fe]
    edge_vis: torch.Tensor  # [B, D, D, Fe] bool
    edge_exists: torch.Tensor  # [B, D, D] bool, existence the model may see
    edge_exists_true: torch.Tensor  # [B, D, D] bool

    def to(self, device) -> MaskedBatch:
        return MaskedBatch(**{k: v.to(device) for k, v in self.__dict__.items()})


def _idx(names, table):
    return torch.tensor([table[n] for n in names], dtype=torch.long)


class Masker:
    def __init__(self, cfg: MaskConfig | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg or MaskConfig()
        if self.cfg.ghost_edge_mode not in ("relational", "topology", "none"):
            raise ValueError(f"unknown ghost_edge_mode {self.cfg.ghost_edge_mode!r}")
        self.device = torch.device(device)
        d = self.device
        self.tele_hide = [_idx(S.TELEMETRY_HIDES[n], S.DI).to(d) for n in S.TELEMETRY_TARGETS]
        self.tele_tgt = _idx(S.TELEMETRY_TARGETS, S.DI).to(d)
        self.groups = list(S.STRATEGY_GROUPS)
        self.group_cols = [_idx(S.STRATEGY_GROUPS[g], S.DI).to(d) for g in self.groups]
        self.comp_cols = _idx(S.COMPOUND_COLS, S.DI).to(d)
        self.e_speed = _idx(S.EDGE_LEAKS["speed"], S.EI).to(d)
        self.e_gaps = _idx(S.EDGE_LEAKS["gaps"], S.EI).to(d)
        self.g_flags = _idx(S.GLOBAL_FLAG_FEATURES, S.GI).to(d)
        self.g_local = _idx(S.GLOBAL_LOCAL_FLAG_FEATURES, S.GI).to(d)
        self.g_sc = _idx(["is_sc", "is_vsc", "is_vsc_ending", "is_red"], S.GI).to(d)
        self.probs = torch.tensor(self.cfg.task_probs, dtype=torch.float32, device=d)

    # ------------------------------------------------------------------------------------------------------------
    def __call__(self, raw: dict[str, torch.Tensor], task: int | None = None, generator: torch.Generator | None = None) -> MaskedBatch:
        """``raw``: a collated batch from ``PretrainDataset`` (any device). ``task`` forces one task for all samples
        (evaluation); ``generator`` makes the masks reproducible."""
        dev, cfg = self.device, self.cfg
        raw = {k: v.to(dev) for k, v in raw.items()}
        present, drv, glb = raw["present"], raw["drv"], raw["glb"]
        B, D, F = drv.shape
        Fe = raw["edge"].shape[-1]
        rnd = lambda *shape: torch.rand(*shape, device=dev, generator=generator)  # noqa: E731

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
        is_a, is_b, is_c, is_d = (task_id == k for k in range(4))

        hide = torch.zeros(B, D, F, dtype=torch.bool, device=dev)
        tgt = torch.zeros_like(hide)
        hide_speed = torch.zeros(B, D, dtype=torch.bool, device=dev)
        hide_gaps = torch.zeros_like(hide_speed)
        comp_tgt = torch.zeros_like(hide_speed)

        # ---- A: telemetry cells ------------------------------------------------------------------------------
        sel_a = (rnd(B, D, len(S.TELEMETRY_TARGETS)) < cfg.p_cell) & present[..., None] & is_a[:, None, None]
        sel_a &= valid[..., self.tele_tgt]
        for k, name in enumerate(S.TELEMETRY_TARGETS):
            tgt[..., self.tele_tgt[k]] |= sel_a[..., k]
            for col in self.tele_hide[k]:
                hide[..., col] |= sel_a[..., k]
            if name == "speed_kmh":
                hide_speed |= sel_a[..., k]

        # ---- D: strategy groups ------------------------------------------------------------------------------
        sel_d = (rnd(B, D, len(self.groups)) < cfg.p_group) & present[..., None] & is_d[:, None, None]
        for k, g in enumerate(self.groups):
            for col in self.group_cols[k]:
                hide[..., col] |= sel_d[..., k]
                tgt[..., col] |= sel_d[..., k]
            if g == "gaps":
                hide_gaps |= sel_d[..., k]
            if g == "compound":
                comp_tgt |= sel_d[..., k]

        # ---- B: ghost car ------------------------------------------------------------------------------------
        score = torch.where(present, rnd(B, D), torch.full((B, D), -1.0, device=dev))
        ghost = torch.zeros(B, D, dtype=torch.bool, device=dev)
        ghost[torch.arange(B, device=dev), score.argmax(1)] = True
        ghost &= is_b[:, None] & present
        hide |= ghost[..., None]
        tgt |= ghost[..., None]
        comp_tgt |= ghost

        tgt &= valid
        comp_valid = valid[..., self.comp_cols].all(-1)
        comp_tgt &= comp_valid
        comp_idx = drv_x[..., self.comp_cols].argmax(-1)
        # the compound one-hot is predicted with cross-entropy, not per-column
        tgt[..., self.comp_cols] = False

        tok_vis = present & (raw["tok"] >= 0)
        if cfg.ghost_edge_mode != "relational":
            tok_vis &= ~ghost

        # ---- edges: hide what restates a masked driver feature ----------------------------------------------
        inc = lambda m: (m[:, :, None] | m[:, None, :])  # noqa: E731 - edges touching a masked car (either end)
        eh = torch.zeros(B, D, D, Fe, dtype=torch.bool, device=dev)
        eh_exists = torch.zeros(B, D, D, dtype=torch.bool, device=dev)
        spd = inc(hide_speed | ghost)
        gp = inc(hide_gaps)
        for col in self.e_speed:
            eh[..., col] |= spd
        for col in self.e_gaps:
            eh[..., col] |= gp
        if cfg.ghost_edge_mode in ("topology", "none"):
            eh |= inc(ghost)[..., None]
        if cfg.ghost_edge_mode == "none":
            eh_exists |= inc(ghost)
        edge_vis = e_valid & ~eh
        edge_exists = eexists & ~eh_exists

        # ---- C: flags ----------------------------------------------------------------------------------------
        M = raw["trk"].shape[1]
        tmar = raw["tmar"]  # [B, M] marshal sector id (0 unknown)
        sec_of_tok = (tmar - 1).clamp(min=0)
        tok_known = tmar > 0
        ysec, dsec = raw["ysec"], raw["dsec"]
        yel_tok = torch.gather(ysec, 1, sec_of_tok) & tok_known
        dbl_tok = torch.gather(dsec, 1, sec_of_tok) & tok_known
        glb_sc = glb_x[:, self.g_sc] > 0.5  # [B, 4]
        flag_x = torch.cat([yel_tok[..., None], dbl_tok[..., None], glb_sc[:, None, :].expand(B, M, 4)], -1).float()

        mode_global = is_c & (rnd(B) < cfg.c_global_prob)
        mode_local = is_c & ~mode_global
        S_ = ysec.shape[1]
        exists = torch.zeros(B, S_, device=dev).scatter_add_(1, sec_of_tok, tok_known.float()) > 0
        pick = (rnd(B, S_) < cfg.c_sector_prob) | ((ysec | dsec) & (rnd(B, S_) < cfg.c_yellow_bias))
        pick &= exists
        fallback = (rnd(B, S_) * exists).argmax(1)
        none_picked = ~pick.any(1)
        pick[torch.arange(B, device=dev), fallback] |= none_picked & exists.any(1)
        pick &= mode_local[:, None]
        tok_pick = torch.gather(pick, 1, sec_of_tok) & tok_known

        flag_hide = mode_global[:, None, None].expand(B, M, S.N_FLAGS).clone()
        flag_hide[..., 0] |= tok_pick
        flag_hide[..., 1] |= tok_pick
        flag_vis = ~flag_hide
        flag_tgt = flag_hide[..., :2] & tok_known[..., None]

        gh = torch.zeros_like(glb_valid)
        gh[:, self.g_flags] |= mode_global[:, None]
        gh[:, self.g_local] |= mode_local[:, None]
        glb_vis = glb_valid & ~gh
        glb_tgt = gh & glb_valid

        return MaskedBatch(
            task=task_id, present=present, drv_x=drv_x, drv_vis=valid & ~hide, drv_tgt=tgt, ghost=ghost,
            tok=raw["tok"], tok_vis=tok_vis, comp_tgt=comp_tgt, comp_idx=comp_idx,
            glb_x=glb_x, glb_vis=glb_vis, glb_tgt=glb_tgt,
            trk_static=torch.nan_to_num(raw["trk"]), flag_x=flag_x, flag_vis=flag_vis, flag_tgt=flag_tgt,
            edge_x=edge_x, edge_vis=edge_vis, edge_exists=edge_exists, edge_exists_true=eexists,
        )  # fmt: skip
