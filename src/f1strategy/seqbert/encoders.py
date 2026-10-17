"""
encoders.py - step I of the pipeline: ONE graph (one frame) -> ONE vector.

Three interchangeable implementations of the "Graph to vector" box:

  flatten  every visible number of the frame is concatenated and projected linearly (drivers, edges, flags, global;
           the circuit, constant over the window, enters once per sample).      cheapest, keeps every slot exactly.
  sector   the lap ring is cut into S arcs (default 16); per arc the track features, flags, the number of cars and the
           mean of their features are pooled; arcs are processed by a shared MLP and concatenated.
           most abstract (a car is only seen through its arc), good at "where is the traffic".
  gnn      a graph-attention network on the real graph: nodes = global + cars + coarse track nodes, edges = track
           ring, car-on-track-node and the proximity edges (with their features as attention bias).
           Cars are read out slot by slot, so the vector stays slot-aware.      most expressive, most expensive.

All three take the masked ``SeqBatch`` (hidden values enter as ``[value * visible, visible]``, never as plain zeros)
and return [B, T, H]. The slot-aware read-out (a car keeps its place in the vector) is what lets the 1-layer head
reconstruct "slot 7's features" from the contextualised vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from f1strategy.pretrain import schema as S
from f1strategy.seqbert.masking import SeqBatch

F_DRV, F_GLB, F_EDGE, F_TRK, N_FLAG = S.F_DRIVER, S.F_GLOBAL, S.F_EDGE, S.F_TRACK_STATIC, S.N_FLAGS


def driver_inputs(mb: SeqBatch, M: int) -> torch.Tensor:
    """[B, T, D, 2F+2]: values*visible, visible, ring position*visible, position visible. Absent cars are all zero."""
    pres = mb.present[..., None].float()
    vis = mb.drv_vis.float() * pres
    tv = mb.tok_vis[..., None].float()
    pos = (mb.tok.clamp(min=0).float() / M)[..., None] * tv
    return torch.cat([mb.drv_x * vis, vis, pos, tv], -1)


def edge_inputs(mb: SeqBatch) -> torch.Tensor:
    ev = mb.edge_vis.float()
    return torch.cat([mb.edge_x * ev, ev, mb.edge_exists[..., None].float()], -1)  # [B, T, D, D, 2Fe+1]


def glob_inputs(mb: SeqBatch) -> torch.Tensor:
    v = mb.glb_valid.float()
    return torch.cat([mb.glb_x * v, v], -1)


class FlattenEncoder(nn.Module):
    def __init__(self, hidden: int, D: int, M: int, **_):
        super().__init__()
        self.M = M
        self.drv = nn.Linear(D * (2 * F_DRV + 2), hidden)
        self.edge = nn.Linear(D * D * (2 * F_EDGE + 1), hidden, bias=False)
        self.glb = nn.Linear(2 * F_GLB, hidden, bias=False)
        self.flag = nn.Linear(M * N_FLAG, hidden, bias=False)
        self.trk = nn.Linear(M * F_TRK, hidden, bias=False)  # the circuit: once per sample, added to every frame
        self.norm = nn.LayerNorm(hidden)

    def forward(self, mb: SeqBatch) -> torch.Tensor:
        B, T = mb.ok.shape
        h = self.drv(driver_inputs(mb, self.M).flatten(2))
        h = h + self.edge(edge_inputs(mb).flatten(2))
        h = h + self.glb(glob_inputs(mb)) + self.flag(mb.flag_x.flatten(2).float())
        h = h + self.trk(mb.trk_static.flatten(1))[:, None]
        return self.norm(h)


class SectorEncoder(nn.Module):
    def __init__(self, hidden: int, D: int, M: int, n_sectors: int = 16, gnn_dim: int = 128, **_):
        super().__init__()
        if M % n_sectors:
            raise ValueError("track tokens must be divisible by the number of sectors")
        self.M, self.S, self.D = M, n_sectors, D
        g = gnn_dim
        self.sector_mlp = nn.Sequential(nn.Linear(F_TRK + N_FLAG + 1 + 2 * F_DRV, g), nn.GELU(), nn.Linear(g, g))
        self.sector_emb = nn.Parameter(torch.zeros(n_sectors, g))
        nn.init.normal_(self.sector_emb, std=0.02)
        self.out = nn.Linear(n_sectors * g, hidden)
        self.glb = nn.Linear(2 * F_GLB + D, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, mb: SeqBatch) -> torch.Tensor:
        B, T, D = mb.present.shape
        S_, M = self.S, self.M
        per = M // S_
        trk = mb.trk_static.view(B, S_, per, F_TRK).mean(2)[:, None].expand(B, T, S_, F_TRK)
        flags = mb.flag_x.view(B, T, S_, per, N_FLAG).any(3).float()
        sec = (mb.tok.clamp(min=0) // per).clamp(max=S_ - 1)
        located = (mb.present & mb.tok_vis).float()
        w = Fn.one_hot(sec, S_).float() * located[..., None]  # [B, T, D, S]
        vis = mb.drv_vis.float() * mb.present[..., None].float()
        num = torch.einsum("btds,btdf->btsf", w, mb.drv_x * vis)
        den = torch.einsum("btds,btdf->btsf", w, vis)
        count = w.sum(2)  # [B, T, S]
        mean = num / den.clamp(min=1.0)
        frac = den / count[..., None].clamp(min=1.0)
        x = torch.cat([trk, flags, count[..., None] / 5.0, mean, frac], -1)
        z = self.sector_mlp(x) + self.sector_emb
        h = self.out(z.flatten(2)) + self.glb(torch.cat([glob_inputs(mb), mb.slot_masked.float()], -1))
        return self.norm(h)


class _Block(nn.Module):
    """Pre-LN graph-attention block; ``bias`` [B, H, N, N] is added to the logits (-inf outside the graph)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, bias):
        B, N, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
        return x + self.ff(self.n2(x))


class GNNEncoder(nn.Module):
    def __init__(self, hidden: int, D: int, M: int, gnn_dim: int = 128, gnn_layers: int = 2, gnn_heads: int = 4,
                 track_nodes: int = 32, **_):  # fmt: skip
        super().__init__()
        pf = max(1, M // track_nodes)
        if M % pf:
            raise ValueError("track tokens must be divisible by the pooling factor")
        self.M, self.D, self.pf, self.Mg, self.g, self.heads = M, D, pf, M // pf, gnn_dim, gnn_heads
        g = gnn_dim
        self.n_drv = nn.Linear(2 * F_DRV + 2, g)
        self.n_glb = nn.Linear(2 * F_GLB, g)
        self.n_trk = nn.Linear(F_TRK + N_FLAG + 1, g)
        self.type_emb = nn.Parameter(torch.zeros(3, g))
        self.ring_emb = nn.Parameter(torch.zeros(self.Mg + 1, g))  # last row = "location unknown"
        nn.init.normal_(self.type_emb, std=0.02)
        nn.init.normal_(self.ring_emb, std=0.02)
        self.edge_bias = nn.Sequential(nn.Linear(2 * F_EDGE + 1, 32), nn.GELU(), nn.Linear(32, gnn_heads))
        self.blocks = nn.ModuleList(_Block(g, gnn_heads) for _ in range(gnn_layers))
        self.read = nn.Sequential(nn.LayerNorm((D + 2) * g), nn.Linear((D + 2) * g, hidden))
        ring = torch.zeros(self.Mg, self.Mg, dtype=torch.bool)
        j = torch.arange(self.Mg)
        for d in (-1, 0, 1):
            ring[j, (j + d) % self.Mg] = True
        self.register_buffer("ring", ring, persistent=False)

    def forward(self, mb: SeqBatch) -> torch.Tensor:
        B, T, D = mb.present.shape
        N, Mg, _, pf = 1 + D + self.Mg, self.Mg, self.g, self.pf
        BT = B * T
        # --- nodes
        d_in = driver_inputs(mb, self.M).flatten(0, 1)  # [BT, D, 2F+2]
        coarse = (mb.tok.clamp(min=0) // pf).clamp(max=Mg - 1)
        coarse = torch.where(mb.tok_vis, coarse, torch.full_like(coarse, Mg)).flatten(0, 1)  # [BT, D]
        pres = mb.present.flatten(0, 1)
        drv = (self.n_drv(d_in) + self.ring_emb[coarse] + self.type_emb[1]) * pres[..., None]
        glb = self.n_glb(glob_inputs(mb).flatten(0, 1)) + self.type_emb[0]
        trk_s = mb.trk_static.view(B, Mg, pf, F_TRK).mean(2)[:, None].expand(B, T, Mg, F_TRK)
        flags = mb.flag_x.view(B, T, Mg, pf, N_FLAG).any(3).float()
        ncar = mb.n_cars_tok.view(B, T, Mg, pf).sum(3, keepdim=True) / 5.0
        trk = self.n_trk(torch.cat([trk_s, flags, ncar], -1).flatten(0, 1)) + self.ring_emb[:Mg] + self.type_emb[2]
        x = torch.cat([glb[:, None], drv, trk], 1)  # [BT, N, g]
        # --- graph (adjacency) and edge bias
        adj = torch.zeros(BT, N, N, dtype=torch.bool, device=x.device)
        adj[:, 0, :] = True  # global node sees everything ...
        adj[:, :, 0] = True  # ... and everything sees it
        adj[:, 1 + D :, 1 + D :] = self.ring
        on = Fn.one_hot(coarse.clamp(max=Mg - 1), Mg).bool() & (coarse < Mg)[..., None] & pres[..., None]  # [BT, D, Mg]
        adj[:, 1 : 1 + D, 1 + D :] = on
        adj[:, 1 + D :, 1 : 1 + D] = on.transpose(1, 2)
        adj[:, 1 : 1 + D, 1 : 1 + D] = mb.edge_exists.flatten(0, 1)
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        adj |= eye
        alive = torch.cat(
            [
                torch.ones(BT, 1, dtype=torch.bool, device=x.device),
                pres,
                torch.ones(BT, Mg, dtype=torch.bool, device=x.device),
            ],
            1,
        )
        adj &= alive[:, None, :] | eye  # nobody attends to an absent car
        eb = self.edge_bias(edge_inputs(mb).flatten(0, 1)).permute(0, 3, 1, 2)  # [BT, H, D, D]
        bias = Fn.pad(eb, (1, Mg, 1, Mg))
        bias = bias.masked_fill(~adj[:, None], float("-inf"))
        for blk in self.blocks:
            x = blk(x, bias)
        drv_out = x[:, 1 : 1 + D] * pres[..., None]
        z = torch.cat([x[:, 0], drv_out.flatten(1), x[:, 1 + D :].mean(1)], -1)
        return self.read(z).view(B, T, -1)


ENCODERS = {"flatten": FlattenEncoder, "sector": SectorEncoder, "gnn": GNNEncoder}
