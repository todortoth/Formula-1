"""
model.py - a BERT-style bidirectional encoder over one race snapshot, plus the pre-training heads.

Tokens (fixed layout, T = 1 + MAX_DRIVERS + N_TRACK_TOKENS):

    [GLOBAL] [driver slot 0 .. D-1] [track token 0 .. M-1]

Every token type has its own input projection (the feature widths differ), a token-type embedding and a shared
*ring position* embedding: a driver token gets the embedding of the track token its car is on, so a car and the piece of
track under it share a position code, as in BERT's absolute positions but on the circular lap. A learnable "unknown"
position is used when the car's location is hidden (ghost car in the harder modes). Driver slots carry NO order or identity
(slot = alphabetical code in that race), so the encoder is permutation-equivariant over cars.

The graph structure enters as a learned additive attention bias per layer and head (the encoder is otherwise plain BERT):
  * ring distance bucket between any two ring-positioned tokens (T5-style, circular), plus 'global' / 'unknown' buckets
  * the proximity-edge features of every driver pair, through a small shared MLP and one linear map per layer
Absent drivers are masked as attention keys. Pre-LN residual blocks (more stable than BERT's post-LN when training from
scratch), GELU, learned LayerNorm-ed embeddings.

Heads (used only for pre-training and thrown away afterwards; ``F1Bert.encode`` returns the hidden states):
  driver head -> all 45 driver features (A, B, D; compound logits are the 6 compound columns)
  flag head   -> yellow / double-yellow logits per track token (C)
  global head -> all 23 global features (C)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from f1strategy.pretrain import schema as S
from f1strategy.pretrain.masking import MaskedBatch

PRESETS = {
    "tiny": {"hidden": 128, "layers": 2, "heads": 4, "ffn": 512},  # smoke tests / CPU
    "small": {"hidden": 256, "layers": 6, "heads": 8, "ffn": 1024},  # ~5 M parameters
    "base": {"hidden": 384, "layers": 8, "heads": 8, "ffn": 1536},  # ~15 M parameters
}
N_DIST_BUCKETS = 24


@dataclass
class F1BertConfig:
    hidden: int = 256
    layers: int = 6
    heads: int = 8
    ffn: int = 1024
    dropout: float = 0.1
    n_track_tokens: int = S.N_TRACK_TOKENS
    max_drivers: int = S.MAX_DRIVERS
    edge_hidden: int = 32

    @staticmethod
    def preset(name: str, **kw) -> F1BertConfig:
        return F1BertConfig(**{**PRESETS[name], **kw})

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, h: int, heads: int, ffn: int, p: float):
        super().__init__()
        self.heads, self.p = heads, p
        self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.qkv, self.out = nn.Linear(h, 3 * h), nn.Linear(h, h)
        self.ff = nn.Sequential(nn.Linear(h, ffn), nn.GELU(), nn.Linear(ffn, h))
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype), dropout_p=self.p if self.training else 0.0
        )
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, T, H)))
        return x + self.drop(self.ff(self.ln2(x)))


def _head(h: int, n_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, n_out))  # BERT MLM-style transform


class F1Bert(nn.Module):
    def __init__(self, cfg: F1BertConfig):
        super().__init__()
        self.cfg = cfg
        h, M, D = cfg.hidden, cfg.n_track_tokens, cfg.max_drivers
        self.M, self.D = M, D
        self.proj_g = nn.Linear(2 * S.F_GLOBAL, h)
        self.proj_d = nn.Linear(2 * S.F_DRIVER, h)
        self.proj_t = nn.Linear(S.F_TRACK_STATIC + 2 * S.N_FLAGS + 1, h)  # static, flags*vis, flag vis, n_drivers
        self.type_emb = nn.Embedding(3, h)
        self.ring_emb = nn.Embedding(M + 1, h)  # index M = unknown position
        self.ghost_emb = nn.Parameter(torch.zeros(h))
        self.emb_ln, self.emb_drop = nn.LayerNorm(h), nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(h, cfg.heads, cfg.ffn, cfg.dropout) for _ in range(cfg.layers))
        self.final_ln = nn.LayerNorm(h)
        # attention biases
        self.dist_bias = nn.Parameter(torch.zeros(cfg.layers, N_DIST_BUCKETS + 2, cfg.heads))
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * S.F_EDGE + 1, cfg.edge_hidden), nn.GELU(), nn.Linear(cfg.edge_hidden, cfg.edge_hidden)
        )
        self.edge_to_bias = nn.ModuleList(nn.Linear(cfg.edge_hidden, cfg.heads) for _ in range(cfg.layers))
        # pre-training heads
        self.driver_head = _head(h, S.F_DRIVER)
        self.flag_head = _head(h, 2)
        self.global_head = _head(h, S.F_GLOBAL)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------------------------------------------------
    def _tokens(self, mb: MaskedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        B, D, M = mb.drv_x.shape[0], self.D, self.M
        dev = mb.drv_x.device
        g = self.proj_g(torch.cat([mb.glb_x * mb.glb_vis, mb.glb_vis.float()], -1))[:, None]
        d = self.proj_d(torch.cat([mb.drv_x * mb.drv_vis, mb.drv_vis.float()], -1))
        d = d + self.ghost_emb * mb.ghost[..., None]
        # cars per track token, counting only cars whose location the model may see
        seen = (mb.present & mb.tok_vis).float()
        ndrv = torch.zeros(B, M, device=dev).scatter_add_(1, mb.tok.clamp(0, M - 1), seen)
        flags = torch.cat([mb.flag_x * mb.flag_vis, mb.flag_vis.float(), (ndrv / 5.0)[..., None]], -1)
        t = self.proj_t(torch.cat([mb.trk_static, flags], -1))
        ring_d = torch.where(mb.tok_vis, mb.tok, torch.full_like(mb.tok, M))
        ring_t = torch.arange(M, device=dev).expand(B, M)
        x = torch.cat([g, d, t], 1)
        kinds = torch.cat([torch.zeros(B, 1), torch.ones(B, D), torch.full((B, M), 2.0)], 1).long().to(dev)
        ring = torch.cat([torch.full((B, 1), M, device=dev), ring_d, ring_t], 1)
        x = x + self.type_emb(kinds) + self.ring_emb(ring)
        return self.emb_drop(self.emb_ln(x)), ring

    def _bias(self, mb: MaskedBatch, ring: torch.Tensor) -> list[torch.Tensor]:
        """Additive attention bias per layer: [B, heads, T, T]."""
        B, T = ring.shape
        _, M = self.D, self.M
        dev = ring.device
        unknown = ring == M  # global token and hidden-position cars
        diff = (ring[:, :, None] - ring[:, None, :]).abs()
        dist = torch.minimum(diff, M - diff).float()
        bucket = (
            torch.where(dist < 16, dist, 16 + torch.log2(dist.clamp(min=16) / 16) * 2)
            .floor()
            .clamp(max=N_DIST_BUCKETS - 1)
            .long()
        )
        is_global = torch.zeros(B, T, dtype=torch.bool, device=dev)
        is_global[:, 0] = True
        pair_global = is_global[:, :, None] | is_global[:, None, :]
        pair_unknown = (unknown[:, :, None] | unknown[:, None, :]) & ~pair_global
        bucket = torch.where(pair_global, torch.full_like(bucket, N_DIST_BUCKETS), bucket)
        bucket = torch.where(pair_unknown, torch.full_like(bucket, N_DIST_BUCKETS + 1), bucket)
        # keys: absent drivers cannot be attended to
        key_ok = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.bool, device=dev),
                mb.present,
                torch.ones(B, M, dtype=torch.bool, device=dev),
            ],
            1,
        )
        pad = torch.zeros(B, 1, 1, T, device=dev).masked_fill(~key_ok[:, None, None, :], -1e9)
        # driver-driver edges
        e_in = torch.cat([mb.edge_x * mb.edge_vis, mb.edge_vis.float(), mb.edge_exists[..., None].float()], -1)
        e_h = self.edge_mlp(e_in)  # [B, D, D, eh]
        out = []
        for layer_idx in range(self.cfg.layers):
            b = self.dist_bias[layer_idx][bucket].permute(0, 3, 1, 2)  # [B, H, T, T]
            eb = self.edge_to_bias[layer_idx](e_h).permute(0, 3, 1, 2)  # [B, H, D, D]
            eb = F.pad(eb, (1, M, 1, M))  # place on the driver block
            out.append(b + eb + pad)
        return out

    def encode(self, mb: MaskedBatch) -> torch.Tensor:
        """Hidden states [B, T, hidden] (the part kept for fine-tuning later)."""
        x, ring = self._tokens(mb)
        for blk, bias in zip(self.blocks, self._bias(mb, ring), strict=False):
            x = blk(x, bias)
        return self.final_ln(x)

    def forward(self, mb: MaskedBatch) -> dict[str, torch.Tensor]:
        h = self.encode(mb)
        D = self.D
        return {
            "hidden": h,
            "driver": self.driver_head(h[:, 1 : 1 + D]),  # [B, D, F]
            "flags": self.flag_head(h[:, 1 + D :]),  # [B, M, 2]
            "global": self.global_head(h[:, 0]),  # [B, G]
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_encoder(path: str, device: str = "cpu") -> F1Bert:
    """Rebuild a model from a checkpoint written by ``train.py`` (for fine-tuning later)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    m = F1Bert(F1BertConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.to(device)
