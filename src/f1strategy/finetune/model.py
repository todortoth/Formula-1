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
from f1strategy.seqbert.model import SeqBertConfig, SeqF1Bert

TASKS = ("pit", "laps", "cmp", "nomore")
POOLS = {"cls": 1, "last": 1, "mean": 1, "cls_last": 2, "cls_last_mean": 3}
MODES = ("frozen", "partial", "full", "scratch")


@dataclass
class HeadConfig:
    horizons: tuple[int, ...] = (5,)  # "pits within the next H laps" (one logit per horizon)
    pool: str = "cls_last"
    kind: str = "linear"  # linear = ONE layer (like pre-training); mlp = hidden layers
    hidden: int = 512
    layers: int = 1  # hidden layers of the mlp head
    dropout: float = 0.1
    n_slots: int = 22
    tasks: tuple[str, ...] = TASKS  # which losses are trained (the outputs always exist)
    laps_cap: float = 60.0  # laps-to-pit is regressed as log1p(min(laps, cap))

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


class FtModel(nn.Module):
    def __init__(self, backbone: SeqF1Bert, hc: HeadConfig):
        super().__init__()
        self.backbone, self.hc = backbone, hc
        H = backbone.cfg.hidden
        if hc.pool not in POOLS:
            raise ValueError(f"pool must be one of {sorted(POOLS)}")
        P = POOLS[hc.pool] * H
        self.feat_dim = P
        out = hc.n_slots * hc.n_out
        if hc.kind == "linear":
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
            raise ValueError("head kind must be 'linear' or 'mlp'")
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

    def backbone_params(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    # ---------------------------------------------------------------- forward
    def features(self, mb, scale_id: int) -> torch.Tensor:
        """Pooled contextualised embedding of a window -> [B, feat_dim]."""
        h = self.backbone.encode(mb, scale_id)  # [B, 1+T, H]
        tok = h[:, 1:]
        ok = mb.ok
        T = tok.shape[1]
        parts = []
        for name in self.hc.pool.split("_"):
            if name == "cls":
                parts.append(h[:, 0])
            elif name == "last":  # the last usable time step = the end of the sequence
                idx = (
                    torch.where(ok, torch.arange(T, device=tok.device), torch.full_like(ok, -1, dtype=torch.long))
                    .max(1)
                    .values
                )
                parts.append(tok[torch.arange(len(tok)), idx.clamp(min=0)])
            elif name == "mean":
                w = ok[..., None].to(tok.dtype)
                parts.append((tok * w).sum(1) / w.sum(1).clamp(min=1.0))
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
