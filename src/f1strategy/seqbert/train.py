"""
train.py - pre-training of the temporal BERT.

    python -m f1strategy.seqbert.train --cache cache --out runs/short_flatten --scale short --graph-encoder flatten
    python -m f1strategy.seqbert.train ... --scale medium --batch-size 16
    python -m f1strategy.seqbert.train ... --scale long --batch-size 4 --graph-encoder sector

Every step is either a reconstruction step (each sample gets one of B/T/P/Z) or, with probability --p-contrast, a
contrastive step (pairs of windows of the same circuit + race phase; other circuits are the negatives).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from f1strategy.seqbert import data as D
from f1strategy.seqbert.config import PRESETS, SCALES
from f1strategy.seqbert.losses import (
    MetricAcc,
    accumulate_metrics,
    build_targets,
    build_weights,
    contrastive_loss,
    reconstruction_loss,
)  # fmt: skip
from f1strategy.seqbert.masking import TASKS, SeqMaskConfig, SeqMasker
from f1strategy.seqbert.model import SeqBertConfig, SeqF1Bert


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", choices=list(SCALES), default="short")
    p.add_argument("--n-tokens", type=int, default=None, help="override the number of tokens of the scale")
    p.add_argument("--span-s", type=float, default=None, help="override the window duration in seconds")
    p.add_argument("--span-laps", type=float, default=None, help="override the window duration in laps")
    p.add_argument("--block-s", type=float, default=180.0, help="validation block length when < 5 races (seconds)")
    p.add_argument("--graph-encoder", choices=["flatten", "sector", "gnn"], default="flatten")
    p.add_argument("--preset", choices=list(PRESETS), default="small")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=None, help="default: 32 short / 16 medium / 8 long")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA)")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=64)
    p.add_argument("--select-tasks", default="BTPZ", help="best.pt = lowest mean validation MSE over these tasks")
    p.add_argument("--val-frac", type=float, default=0.1)
    # tasks
    p.add_argument("--task-probs", default="B=0.25,T=0.25,P=0.25,Z=0.25")
    p.add_argument("--zone-frac", type=float, default=0.35)
    p.add_argument("--zone-position", choices=["middle", "random"], default="middle")
    p.add_argument("--mask-fill", choices=["alt", "zero"], default="alt")
    p.add_argument("--pit-window", default="18,24", help="seconds behind the reference car, 'lo,hi'")
    p.add_argument("--ghost-edge-mode", choices=["relational", "topology", "none"], default="relational")
    p.add_argument("--pit-edge-mode", choices=["relational", "topology", "none"], default="topology")
    p.add_argument("--loss-kind", choices=["mse", "huber", "cosine"], default="mse")
    p.add_argument("--cos-weight", type=float, default=0.0)
    # contrastive
    p.add_argument("--p-contrast", type=float, default=0.0, help="share of contrastive steps (0 = off)")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--phase-bins", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--contrast-weight", type=float, default=1.0)
    return p.parse_args()


def parse_probs(s: str):
    d = dict(kv.split("=") for kv in s.split(","))
    v = [float(d.get(t, 0.0)) for t in TASKS]
    tot = sum(v)
    return tuple(x / tot for x in v)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    return a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))


def cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, masker, ds, a, device, scale_id):
    model.eval()
    acc = MetricAcc()
    n = min(a.eval_windows, len(ds))
    sub = torch.utils.data.Subset(ds, np.linspace(0, len(ds) - 1, n).astype(int).tolist())
    for k, _name in enumerate(TASKS):
        g = torch.Generator(device=device).manual_seed(1234 + k)  # same masks at every evaluation
        for raw in DataLoader(sub, batch_size=a.batch_size, shuffle=False):
            mb = masker(raw, task=k, generator=g)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp):
                out = model(mb, scale_id)
            accumulate_metrics(acc, mb, out["recon"])
    model.train()
    return acc.result()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(
        "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    )
    scale = SCALES[a.scale]
    if a.n_tokens:
        scale = replace(scale, n_tokens=a.n_tokens)
    if a.span_s or a.span_laps:
        scale = replace(scale, span_s=a.span_s, span_laps=a.span_laps)
    a.batch_size = a.batch_size or {"short": 32, "medium": 16, "long": 8}[a.scale]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    races = D.open_races(a.cache)
    ix = D.WindowIndex(races, scale)
    tr, va, vdesc = D.split_windows(ix, a.val_frac, block_s=a.block_s, seed=a.seed)
    if not tr or not va:
        raise SystemExit(f"not enough windows for scale {a.scale}: train {len(tr)}, val {len(va)} (add races)")
    print(f"[data] {len(races)} races | {a.scale}: {scale.n_tokens} tokens | windows train {len(tr)} / val {len(va)}")
    train_ds, val_ds = D.WindowDataset(ix, tr), D.WindowDataset(ix, va)
    w = D.window_weights(ix, tr)
    loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=WeightedRandomSampler(w, a.steps * a.batch_size, True),
                        num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)  # fmt: skip
    pair_iter = None
    if a.p_contrast > 0:
        groups = D.window_groups(ix, tr, a.phase_bins)
        print(f"[contrastive] {len({g[0] for g in groups})} circuit(s), {len(groups)} (circuit, phase) groups")
        ps = D.PairBatchSampler(groups, ix, tr, a.pairs, a.steps, seed=a.seed)
        pair_iter = cycle(DataLoader(train_ds, batch_sampler=ps, num_workers=a.workers))

    mcfg = SeqMaskConfig(parse_probs(a.task_probs), a.zone_frac, a.zone_position, a.mask_fill,
                         tuple(float(x) for x in a.pit_window.split(",")), a.ghost_edge_mode, a.pit_edge_mode)  # fmt: skip
    masker = SeqMasker(mcfg, device)
    cfg = SeqBertConfig(graph_encoder=a.graph_encoder, n_track=int(races[0].track.shape[0]),
                        n_drivers=int(races[0].a['drv'].shape[1]),
                        **PRESETS[a.preset]
                        )  # fmt: skip
    model = SeqF1Bert(cfg).to(device)
    print(
        f"[model] {a.graph_encoder} encoder, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, device {device}"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    (out / "args.json").write_text(
        json.dumps({**vars(a), "model": cfg.as_dict(), "n_tokens": scale.n_tokens, "val": vdesc}, indent=1)
    )
    rng = np.random.default_rng(a.seed)
    model.train()
    t0 = time.time()
    best_val = float("inf")
    it = iter(loader)

    with open(out / "log.jsonl", "a") as log:
        for step in range(a.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a)
            contrast = pair_iter is not None and rng.random() < a.p_contrast
            rec = {"step": step}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp):
                if contrast:
                    raw = next(pair_iter)
                    mb = masker.plain(raw, token_dropout=0.15)
                    loss, acc1 = contrastive_loss(model.embed(mb, scale.scale_id), a.temperature)
                    loss = loss * a.contrast_weight
                    rec.update(kind="contrast", loss=loss.item(), top1=acc1.item())
                else:
                    raw = next(it)
                    mb = masker(raw)
                    out_ = model(mb, scale.scale_id)
                    loss, per = reconstruction_loss(
                        out_["recon"], build_targets(mb), build_weights(mb), mb.task, a.loss_kind, a.cos_weight
                    )
                    rec.update(
                        kind="recon",
                        loss=loss.detach().item(),
                        **{f"loss_{k}": v.detach().item() for k, v in per.items()},
                    )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            if step % 25 == 0:
                print(f"step {step:5d} {rec['kind']:8s} loss {rec['loss']:.4f}  {time.time() - t0:.0f}s")
            if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
                rec["eval"] = evaluate(model, masker, val_ds, a, device, scale.scale_id)
                print(
                    "[eval]",
                    {k: round(v, 3) for k, v in rec["eval"].items() if k.endswith(("mae", "acc", "interp", "mse"))},
                )
                sel = [rec["eval"][f"{t}/mse"] for t in a.select_tasks if f"{t}/mse" in rec["eval"]]
                val_loss = float(np.mean(sel)) if sel else float("inf")
                is_best = val_loss < best_val
                best_val = min(best_val, val_loss)
                rec.update(val_loss=val_loss, best_val=best_val, is_best=is_best)
                ck = {"model": model.state_dict(), "cfg": cfg.as_dict(), "scale": a.scale, "n_tokens": scale.n_tokens,
                      "step": step, "val_loss": val_loss, "eval": rec["eval"]}  # fmt: skip
                torch.save(ck, out / "ckpt.pt")  # the latest evaluation point
                if is_best:
                    torch.save(ck, out / "best.pt")
                print(
                    f"[val] mean MSE over {a.select_tasks}: {val_loss:.4f} (best {best_val:.4f}{' - new best.pt' if is_best else ''})"
                )
            log.write(json.dumps(rec) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
