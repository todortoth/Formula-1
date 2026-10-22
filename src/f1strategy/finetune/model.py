"""
model.py - the pre-trained temporal BERT with the pre-training heads removed and a NEW supervised head on top.

    window of T graphs -> graph encoder -> T vectors -> temporal transformer -> [CLS], token_1 ... token_T
        -> pooling (CLS | last token | mean) -> NEW head -> per car slot: pit-within-H logits, laps-to-pit, next compound, no-more-stops

Freeze modes (``set_mode``):

* ``frozen``   the whole backbone is fixed (also no dropout): the embedding is exactly the pre-trained one; only the head learns.
               The pooled features can be computed once, so the head can be tried in seconds (``linear`` -> ``mlp`` if too weak).
* ``partial``  graph encoder frozen, only the last ``unfreeze_last`` transformer layers (+ final norm) and the head learn.
* ``full``     everything learns, with a much smaller learning rate for the backbone (see ``train.param_groups``); an optional
               head-only warm-up first ("linear probe, then fine-tune") keeps the random head from distorting good features.
* ``scratch``  same architecture, random weights, everything learns: the control that shows what pre-training is worth.

The head reads one car slot per output block (like the pre-training head did): the vector keeps cars in their slots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from f1strategy.finetune.labels import N_CMP
from f1strategy.pretrain import schema as S
from f1strategy.seqbert.model import SeqBertConfig, SeqF1Bert

TASKS = ("pit", "laps", "cmp", "nomore")
POOLS = {"cls": 1, "last": 1, "mean": 1, "cls_last": 2, "cls_last_mean": 3}
MODES = ("frozen", "partial", "full", "scratch")


@dataclass
class HeadConfig:
    horizons: tuple[int, ...] = (5,)  # "pits within the next H laps" (one logit per horizon)
    pool: str = "cls_last"
    kind: str = (
        "linear"  # linear = ONE layer (like pre-training); mlp = hidden layers; car = shared per-car MLP, see CarHead
    )
    hidden: int = 512
    layers: int = 1  # hidden layers of the mlp head
    dropout: float = 0.1
    n_slots: int = 22
    tasks: tuple[str, ...] = TASKS  # which losses are trained (the outputs always exist)
    laps_cap: float = 60.0  # laps-to-pit is regressed as log1p(min(laps, cap))
    ctx_dim: int = 64  # car head: width of the projected embedding context (0 = no embedding at all: tabular control)
    slot_branch: bool = True  # car head: also add the per-slot linear read-out of the embedding

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> HeadConfig:
        d = dict(d)
        d["horizons"], d["tasks"] = tuple(d["horizons"]), tuple(d["tasks"])
        return HeadConfig(**d)

    @property
    def n_out(self) -> int:  # outputs per car slot
        return len(self.horizons) + 1 + N_CMP + 1


def load_backbone(path: str, pretrained: bool = True, dropout: float | None = None) -> tuple[SeqF1Bert, dict]:
    """Rebuild a pre-trained ``SeqF1Bert`` from ``best.pt`` and delete its pre-training heads (reconstruction + contrastive)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    if dropout is not None:
        cfg["dropout"] = dropout
    model = SeqF1Bert(SeqBertConfig(**cfg))
    if pretrained:
        model.load_state_dict(ck["model"])  # strict: includes the old heads, which are dropped next
    del model.recon_head, model.proj_head
    return model, ck


class CarHead(nn.Module):
    """One MLP shared by all car slots.

    Input per car: its own state at the END of the window (``feat`` extras: last usable frame, change since the first usable frame,
    visibility), the race-level global features and a projection of the pooled embedding; optionally the per-slot linear
    read-out of the embedding is added to the logits.  Because the weights are shared, "tyre age 25 on mediums" means the
    same thing for every slot and every race, which a per-slot read-out of a pooled vector has to learn 22 times."""

    def __init__(self, hc: HeadConfig, P: int):
        super().__init__()
        self.P, self.D, self.K, self.Fd = P, hc.n_slots, hc.n_out, S.F_DRIVER
        use_emb = hc.ctx_dim > 0
        self.ctx = nn.Sequential(nn.LayerNorm(P), nn.Linear(P, hc.ctx_dim), nn.GELU()) if use_emb else None
        self.slot = nn.Linear(P, hc.n_slots * hc.n_out) if (use_emb and hc.slot_branch) else None
        d = 3 * S.F_DRIVER + S.F_GLOBAL + (hc.ctx_dim if use_emb else 0)
        layers: list[nn.Module] = [nn.LayerNorm(d)]
        for _ in range(max(1, hc.layers)):
            layers += [nn.Linear(d, hc.hidden), nn.GELU(), nn.Dropout(hc.dropout)]
            d = hc.hidden
        layers.append(nn.Linear(d, hc.n_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, n_car = feat.shape[0], self.D * 3 * self.Fd
        pooled = feat[:, : self.P]
        car = feat[:, self.P : self.P + n_car].view(B, self.D, 3 * self.Fd)
        parts = [car, feat[:, self.P + n_car :][:, None].expand(-1, self.D, -1)]
        if self.ctx is not None:
            parts.append(self.ctx(pooled)[:, None].expand(-1, self.D, -1))
        out = self.mlp(torch.cat(parts, -1))  # [B, D, K]
        if self.slot is not None:
            out = out + self.slot(pooled).view(B, self.D, self.K)
        return out.flatten(1)


class FtModel(nn.Module):
    def __init__(self, backbone: SeqF1Bert, hc: HeadConfig):
        super().__init__()
        self.backbone, self.hc = backbone, hc
        H = backbone.cfg.hidden
        if hc.pool not in POOLS:
            raise ValueError(f"pool must be one of {sorted(POOLS)}")
        P = POOLS[hc.pool] * H
        self.pool_dim = P
        self.feat_dim = P + (hc.n_slots * 3 * S.F_DRIVER + S.F_GLOBAL if hc.kind == "car" else 0)
        out = hc.n_slots * hc.n_out
        if hc.kind == "car":
            self.head = CarHead(hc, P)
        elif hc.kind == "linear":
            self.head = nn.Linear(P, out)
        elif hc.kind == "mlp":
            layers: list[nn.Module] = []
            d = P
            for _ in range(hc.layers):
                layers += [nn.Linear(d, hc.hidden), nn.GELU(), nn.Dropout(hc.dropout)]
                d = hc.hidden
            layers.append(nn.Linear(d, out))
            self.head = nn.Sequential(*layers)
        else:
            raise ValueError("head kind must be 'linear', 'mlp' or 'car'")
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.mode = "full"
        self._frozen: list[nn.Module] = []

    # ---------------------------------------------------------------- freezing
    def set_mode(self, mode: str, unfreeze_last: int = 2) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        bb = self.backbone
        for p in bb.parameters():
            p.requires_grad = mode in ("full", "scratch")
        self._frozen = []
        if mode == "frozen":
            self._frozen = [bb]
        elif mode == "partial":
            n = max(0, min(unfreeze_last, len(bb.encoder.layers)))
            self._frozen = [bb.graph_enc, bb.emb_norm]
            self._frozen += list(bb.encoder.layers[: len(bb.encoder.layers) - n])
            for layer in bb.encoder.layers[len(bb.encoder.layers) - n :]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in bb.encoder.norm.parameters():
                p.requires_grad = True
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self._frozen:  # fixed parts behave deterministically: no dropout
            m.eval()
        return self

    def head_params(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_params(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    # ---------------------------------------------------------------- forward
    @staticmethod
    def _edge_idx(ok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Index of the last / first usable time step of every window."""
        T = ok.shape[1]
        ar = torch.arange(T, device=ok.device).expand_as(ok)
        last = torch.where(ok, ar, torch.full_like(ar, -1)).max(1).values.clamp(min=0)
        first = torch.where(ok, ar, torch.full_like(ar, T)).min(1).values.clamp(max=T - 1)
        return last, first

    def features(self, mb, scale_id: int) -> torch.Tensor:
        """Pooled contextualised embedding of a window -> [B, feat_dim] (``car`` head: + per-car end-of-window state)."""
        h = self.backbone.encode(mb, scale_id)  # [B, 1+T, H]
        tok = h[:, 1:]
        last, first = self._edge_idx(mb.ok)
        b = torch.arange(len(tok), device=tok.device)
        parts = []
        for name in self.hc.pool.split("_"):
            if name == "cls":
                parts.append(h[:, 0])
            elif name == "last":  # the last usable time step = the end of the sequence
                parts.append(tok[b, last])
            elif name == "mean":
                w = mb.ok[..., None].to(tok.dtype)
                parts.append((tok * w).sum(1) / w.sum(1).clamp(min=1.0))
        if self.hc.kind == "car":
            vis = mb.drv_vis.float() * mb.present[..., None].float()  # [B, T, D, F]
            x = mb.drv_x.float() * vis
            xl, xf, vl, vf = x[b, last], x[b, first], vis[b, last], vis[b, first]
            car = torch.cat([xl, (xl - xf) * vl * vf, vl], -1).flatten(1)  # [B, D * 3F]
            glb = (mb.glb_x.float() * mb.glb_valid.float())[b, last]  # [B, G]
            parts += [car, glb]  # float32: no bf16 rounding of the raw state
        return torch.cat(parts, -1)

    def head_forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        hc = self.hc
        o = self.head(feat).view(len(feat), hc.n_slots, hc.n_out)
        nh = len(hc.horizons)
        return {
            "pit": o[..., :nh], "laps": o[..., nh], "cmp": o[..., nh + 1 : nh + 1 + N_CMP], "nomore": o[..., nh + 1 + N_CMP],
        }  # fmt: skip

    def forward(self, mb, scale_id: int) -> dict[str, torch.Tensor]:
        if self.mode == "frozen":
            with torch.no_grad():
                feat = self.features(mb, scale_id)
        else:
            feat = self.features(mb, scale_id)
        return self.head_forward(feat)

    # ---------------------------------------------------------------- decoding
    def decode(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Logits -> probabilities / laps / class."""
        cmp_p = out["cmp"].softmax(-1)
        return {
            "pit": out["pit"].sigmoid(),
            "laps": torch.expm1(out["laps"]).clamp(1.0, self.hc.laps_cap),
            "cmp_prob": cmp_p,
            "cmp": cmp_p.argmax(-1),
            "nomore": out["nomore"].sigmoid(),
        }


def load_finetuned(path: str, device="cpu") -> tuple[FtModel, dict]:
    """Rebuild a model saved by ``train.py`` (backbone + head weights, no pre-training checkpoint needed)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bb = SeqF1Bert(SeqBertConfig(**ck["backbone_cfg"]))
    del bb.recon_head, bb.proj_head
    model = FtModel(bb, HeadConfig.from_dict(ck["head"]))
    model.load_state_dict(ck["model"])
    model.set_mode("frozen")  # inference only
    return model.to(device).eval(), ck
