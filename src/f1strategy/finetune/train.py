"""
train.py - supervised fine-tuning of a pre-trained temporal checkpoint.

    python -m f1strategy.finetune.train --cache cache --init runs/mixed_gnn/best.pt --out runs/ft_medium_frozen \
        --scale medium --mode frozen --head linear

Splits are by season (``--train-years`` / ``--val-years``); the test season is never touched here (``evaluate.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from f1strategy.finetune import metrics as M
from f1strategy.finetune.dataset import (
    DEFAULT_HOP_S,
    FtDataset,
    finetune_index,
    open_labelled_races,
    parse_years,
    targets_to_torch,
    to_masked_batch,
    window_targets,
)  # fmt: skip
from f1strategy.finetune.losses import ft_loss, warmup_cosine
from f1strategy.finetune.model import MODES, POOLS, TASKS, FtModel, HeadConfig, load_backbone
from f1strategy.finetune.scrub import LEVELS, scrub_columns
from f1strategy.seqbert.config import SCALES
from f1strategy.seqbert.masking import SeqMasker

BATCH = {"short": 64, "medium": 32, "long": 16}


def resolve_scale(name: str, ck: dict):
    scale = SCALES[name]
    n_tok = (ck.get("scales") or {}).get(name) or ck.get("n_tokens")
    if isinstance(n_tok, int) and n_tok != scale.n_tokens:
        scale = dataclasses.replace(scale, n_tokens=n_tok)
    return scale


def param_groups(model: FtModel, lr_head: float, lr_backbone: float, wd: float) -> list[dict]:
    def split(params):
        dec = [p for p in params if p.ndim >= 2]
        nod = [p for p in params if p.ndim < 2]
        return [{"params": dec, "weight_decay": wd}, {"params": nod, "weight_decay": 0.0}]

    groups = []
    for g in split(list(model.head.parameters())):
        groups.append(g | {"lr": lr_head, "base_lr": lr_head, "kind": "head"})
    bb = model.backbone_params()
    if bb:
        for g in split(bb):
            groups.append(g | {"lr": lr_backbone, "base_lr": lr_backbone, "kind": "backbone"})
    return [g for g in groups if g["params"]]


class FeatDataset(Dataset):
    def __init__(self, feat: torch.Tensor, y: dict):
        self.feat, self.y = feat, y

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, i):
        return self.feat[i], {k: v[i] for k, v in self.y.items()}


@torch.no_grad()
def extract_features(model: FtModel, loader, masker, device, scale_id, cols, fill, amp) -> torch.Tensor:
    model.eval()
    out = []
    for b in loader:
        mb = to_masked_batch(b["x"], masker, device, cols, fill)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(model.features(mb, scale_id).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def predict_head(model: FtModel, feats: torch.Tensor, device, bs: int = 1024) -> dict[str, np.ndarray]:
    model.eval()
    outs = []
    for i in range(0, len(feats), bs):
        outs.append({k: v.cpu() for k, v in model.decode(model.head_forward(feats[i : i + bs].to(device))).items()})
    return {k: torch.cat([o[k] for o in outs]).numpy() for k in outs[0]}


@torch.no_grad()
def predict_windows(model: FtModel, loader, masker, device, scale_id, cols, fill, amp) -> dict[str, np.ndarray]:
    feats = extract_features(model, loader, masker, device, scale_id, cols, fill, amp)
    return predict_head(model, feats, device)


def decode_for_metrics(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"pit": p["pit"], "laps": p["laps"], "cmp": p["cmp"], "nomore": p["nomore"]}


def select_score(rep: dict, tasks, horizons) -> float:
    if "pit" in tasks:
        return rep[f"pit{horizons[0]}"]["auc"]
    if "laps" in tasks:
        return -rep["laps"]["mae"]
    if "cmp" in tasks:
        return rep["cmp"]["acc"]
    return rep["nomore"]["auc"]


def build_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument(
        "--init", required=True, help="pre-trained checkpoint (best.pt); architecture source also for --mode scratch"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", default="medium", choices=["medium", "long"], help="medium = 1 lap, long = 10 laps")
    ap.add_argument("--mode", default="frozen", choices=MODES)
    ap.add_argument("--unfreeze-last", type=int, default=2, help="partial: transformer layers that learn")
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma list of: pit, laps, cmp, nomore")
    ap.add_argument("--horizons", default="5", help="laps for 'pits within the next H laps' (comma list)")
    ap.add_argument("--loss-weights", default="", help="e.g. pit=1,laps=1,cmp=1,nomore=0.5")
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--head-hidden", type=int, default=512)
    ap.add_argument("--head-layers", type=int, default=1)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--pool", default="cls_last", choices=sorted(POOLS))
    ap.add_argument("--scrub", default="basic", choices=sorted(LEVELS))
    ap.add_argument("--scrub-fill", default="zero", choices=["zero", "nan"])
    ap.add_argument("--train-years", default="2019-2024")
    ap.add_argument("--val-years", default="2025")
    ap.add_argument(
        "--hop-s", type=float, default=None, help="seconds between window ends (default 30 medium / 60 long)"
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument(
        "--head-warmup-steps", type=int, default=0, help="full / partial: steps with the backbone at lr 0 first (LP-FT)"
    )
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=None, help="override the backbone dropout (fine-tuning modes)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(
        "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = tuple(t for t in a.tasks.split(",") if t)
    horizons = tuple(int(h) for h in a.horizons.split(","))
    weights = {k: float(v) for k, v in (kv.split("=") for kv in a.loss_weights.split(",") if kv)}
    hc = HeadConfig(horizons=horizons, pool=a.pool, kind=a.head, hidden=a.head_hidden, layers=a.head_layers,
                    dropout=a.head_dropout, tasks=tasks)  # fmt: skip

    backbone, ck = load_backbone(
        a.init, pretrained=a.mode != "scratch", dropout=a.dropout if a.mode != "frozen" else None
    )
    scale = resolve_scale(a.scale, ck)
    model = FtModel(backbone, hc)
    model.set_mode(a.mode, a.unfreeze_last)
    model = model.to(device)

    tr_races, tr_lab = open_labelled_races(a.cache, parse_years(a.train_years))
    va_races, va_lab = open_labelled_races(a.cache, parse_years(a.val_years))
    if not tr_races or not va_races:
        raise SystemExit("no labelled races: build the labels first (python -m f1strategy.finetune.labels)")
    names = [str(x) for x in next(iter(tr_lab.values()))["driver_names"]]
    cols = scrub_columns(names, a.scrub)
    hop_s = a.hop_s or DEFAULT_HOP_S[a.scale]
    tr_ix = finetune_index(tr_races, scale, hop_s)
    va_ix = finetune_index(va_races, scale, hop_s)
    tr_ds, va_ds = FtDataset(tr_ix, tr_lab), FtDataset(va_ix, va_lab)
    bs = a.batch_size or BATCH[a.scale]
    print(f"{a.mode}: {len(tr_races)} train races / {len(tr_ds)} windows, {len(va_races)} val races / {len(va_ds)} windows, "
          f"scrub={a.scrub} ({len(cols)} columns), device={device}")  # fmt: skip
    masker = SeqMasker(device=device)
    mk = lambda ds, shuffle: DataLoader(ds, bs, shuffle=shuffle, num_workers=a.workers, drop_last=shuffle)  # noqa: E731
    va_t = window_targets(va_ix, va_lab)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    (out / "args.json").write_text(json.dumps({"args": vars(a), "head": hc.as_dict(), "backbone": backbone.cfg.as_dict(),
                                               "scale": scale.name if hasattr(scale, "name") else a.scale,
                                               "n_tokens": scale.n_tokens, "driver_names": names, "scrub_cols": cols,
                                               "params_trainable": n_train, "params_total": n_all}, indent=1, default=str))  # fmt: skip
    print(f"parameters: {n_train:,} trainable of {n_all:,}")

    sid = scale.scale_id
    best, log = -1e9, open(out / "log.jsonl", "a")

    def save(step: int, score: float):
        torch.save({"model": model.state_dict(), "head": hc.as_dict(), "backbone_cfg": backbone.cfg.as_dict(), "scale": a.scale,
                    "n_tokens": scale.n_tokens, "mode": a.mode, "scrub": a.scrub, "scrub_fill": a.scrub_fill, "scrub_cols": cols,
                    "driver_names": names, "step": step, "val_score": score, "init": a.init}, out / "best.pt")  # fmt: skip

    def evaluate(preds: dict[str, np.ndarray]) -> dict:
        return M.summarize(decode_for_metrics(preds), va_t, horizons)

    t0 = time.time()
    if a.mode == "frozen":
        # the embedding never changes: compute it once, then the head trains in seconds
        tr_feat = extract_features(model, mk(tr_ds, False), masker, device, sid, cols, a.scrub_fill, a.amp)
        va_feat = extract_features(model, mk(va_ds, False), masker, device, sid, cols, a.scrub_fill, a.amp)
        tr_y = targets_to_torch(window_targets(tr_ix, tr_lab))
        print(f"features: {tuple(tr_feat.shape)} train, {tuple(va_feat.shape)} val in {time.time() - t0:.0f}s")
        loader = DataLoader(FeatDataset(tr_feat, tr_y), 256, shuffle=True, drop_last=True)
    else:
        loader = mk(tr_ds, True)
    opt = torch.optim.AdamW(param_groups(model, a.lr_head, a.lr_backbone, a.weight_decay), betas=(0.9, 0.98))
    step = 0
    while step < a.steps:
        for b in loader:
            model.train()
            f = warmup_cosine(step, a.steps, a.warmup)
            for g in opt.param_groups:
                hold = g["kind"] == "backbone" and step < a.head_warmup_steps
                g["lr"] = 0.0 if hold else g["base_lr"] * f
            if a.mode == "frozen":
                feat, y = b[0].to(device), {k: v.to(device) for k, v in b[1].items()}
                loss, parts = ft_loss(model.head_forward(feat), y, hc, weights)
            else:
                mb = to_masked_batch(b["x"], masker, device, cols, a.scrub_fill)
                y = {k: v.to(device) for k, v in b["y"].items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp):
                    out_ = model(mb, sid)
                loss, parts = ft_loss(out_, y, hc, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], a.grad_clip)
            if not torch.isfinite(gn):
                print(f"step {step}: non-finite gradient, skipped")
                step += 1
                continue
            opt.step()
            step += 1
            rec = {"step": step, "loss": float(loss), "gnorm": float(gn), **{f"l_{k}": v for k, v in parts.items()}}
            if step % a.eval_every == 0 or step == a.steps:
                feats = (
                    va_feat
                    if a.mode == "frozen"
                    else extract_features(model, mk(va_ds, False), masker, device, sid, cols, a.scrub_fill, a.amp)
                )
                rep = evaluate(predict_head(model, feats, device))
                score = select_score(rep, tasks, horizons)
                rec |= {"val_score": score, "is_best": score > best} | M.flatten_report(rep)
                if score > best:
                    best = score
                    save(step, score)
                print(f"step {step:5d} loss {float(loss):.4f} val {score:.4f}{' *' if rec['is_best'] else ''}  "
                      + "  ".join(f"{k} {v:.3f}" for k, v in M.flatten_report(rep).items() if k.endswith(("auc", "mae", "acc"))))  # fmt: skip
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step >= a.steps:
                break
    print(f"done in {time.time() - t0:.0f}s, best validation score {best:.4f} -> {out / 'best.pt'}")
    return {"best": best}


if __name__ == "__main__":
    main()
