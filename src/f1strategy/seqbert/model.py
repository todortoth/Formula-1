"""
model.py - the temporal BERT.

    frames (T graphs) --graph encoder--> T vectors --(task Z: middle block -> dummy vector)--> [CLS] v_1 ... v_T
        --> + learned time positions (+ scale embedding) --> torch ``nn.TransformerEncoder`` (BERT-style, bidirectional)
        --> contextualised vectors --> 1-layer linear head --> flattened frame data (drivers + global + track)
                                  \\--> [CLS] --> projection head --> contrastive loss

The encoder is a standard BERT encoder stack (post-embedding LayerNorm, GELU, multi-head self-attention, learned
absolute positions, a [CLS] summary token); the *tokens are time steps*, not nodes: attention now runs over time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from f1strategy.pretrain import schema as S
from f1strategy.seqbert.encoders import ENCODERS
from f1strategy.seqbert.masking import SeqBatch


@dataclass
class SeqBertConfig:
    hidden: int = 256
    layers: int = 6
    heads: int = 8
    ffn: int = 1024
    dropout: float = 0.1
    graph_encoder: str = "flatten"
    n_drivers: int = S.MAX_DRIVERS
    n_track: int = S.N_TRACK_TOKENS
    max_len: int = 256
    n_sectors: int = 16
    gnn_dim: int = 128
    gnn_layers: int = 2
    mask_fill: str = "alt"
    proj_dim: int = 64
    time_embed: bool = False  # add an embedding of the elapsed seconds, so one model can read all three scales

    def as_dict(self) -> dict:
        return asdict(self)


N_TIME_PERIODS = 12


def target_dim(D: int = S.MAX_DRIVERS, M: int = S.N_TRACK_TOKENS) -> int:
    return D * S.F_DRIVER + S.F_GLOBAL + 2 * S.MAX_MARSHAL + M


class SeqF1Bert(nn.Module):
    def __init__(self, cfg: SeqBertConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        self.graph_enc = ENCODERS[cfg.graph_encoder](
            hidden=H, D=cfg.n_drivers, M=cfg.n_track, n_sectors=cfg.n_sectors, gnn_dim=cfg.gnn_dim,
            gnn_layers=cfg.gnn_layers,
        )  # fmt: skip
        self.cls = nn.Parameter(torch.zeros(1, 1, H))
        self.pos = nn.Embedding(cfg.max_len + 1, H)
        self.scale = nn.Embedding(3, H)
        if cfg.time_embed:  # sin / cos of the elapsed seconds at 12 log-spaced periods (2 s ... 4000 s)
            self.time_proj = nn.Linear(2 * N_TIME_PERIODS, H)
            self.register_buffer(
                "periods", torch.logspace(math.log10(2.0), math.log10(4000.0), N_TIME_PERIODS), persistent=False
            )
        self.emb_norm = nn.LayerNorm(H)
        self.emb_drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            H, cfg.heads, cfg.ffn, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, norm=nn.LayerNorm(H), enable_nested_tensor=False)
        self.recon_head = nn.Linear(H, target_dim(cfg.n_drivers, cfg.n_track))  # THE 1-layer head
        self.proj_head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, cfg.proj_dim))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.scale.weight, std=0.02)
        fill = torch.full((H,), 0.1)
        fill[::2] = -0.1
        self.register_buffer("fill", fill if cfg.mask_fill == "alt" else torch.zeros(H), persistent=False)

    def graph_vectors(self, mb: SeqBatch) -> torch.Tensor:
        return self.graph_enc(mb)  # [B, T, H]

    def encode(self, mb: SeqBatch, scale_id: int = 0) -> torch.Tensor:
        """-> [B, 1+T, H]; index 0 is the [CLS] summary."""
        v = self.graph_vectors(mb)
        B, T, H = v.shape
        v = torch.where(mb.vec_mask[..., None], self.fill.to(v.dtype).expand(B, T, H), v)  # dummy vectors (task Z)
        x = torch.cat([self.cls.expand(B, 1, H), v], 1)
        x = x + self.pos(torch.arange(T + 1, device=x.device))[None] + self.scale.weight[scale_id]
        if self.cfg.time_embed:
            ang = 2 * math.pi * mb.t_rel[..., None] / self.periods
            te = self.time_proj(torch.cat([ang.sin(), ang.cos()], -1).to(x.dtype))
            x = x + torch.cat([torch.zeros_like(te[:, :1]), te], 1)  # [CLS] has no time
        x = self.emb_drop(self.emb_norm(x))
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mb.ok], 1)
        return self.encoder(x, src_key_padding_mask=pad)

    def forward(self, mb: SeqBatch, scale_id: int = 0) -> dict[str, torch.Tensor]:
        h = self.encode(mb, scale_id)
        return {"cls": h[:, 0], "tokens": h[:, 1:], "recon": self.recon_head(h[:, 1:])}

    def embed(self, mb: SeqBatch, scale_id: int = 0) -> torch.Tensor:
        """Contrastive embedding (projection of [CLS])."""
        return self.proj_head(self.encode(mb, scale_id)[:, 0])
