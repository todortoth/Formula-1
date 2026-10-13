"""
train.py - self-supervised pre-training of the F1 graph BERT (no fine-tuning, no labels).

    python -m f1strategy.pretrain.cache --src parquet_datasets --dst cache --stride 5
    python -m f1strategy.pretrain.train --cache cache --out runs/pre1 --preset small --steps 100000 --batch-size 64

Each step draws frames (flagged frames oversampled), masks every sample for one of the four pretext tasks on the
device, runs the encoder and optimises the weighted sum of the per-task losses. Validation runs every task on a fixed
validation subset with fixed masks, so the per-task metrics are comparable between evaluations.
Checkpoints (``last.pt``, ``best.pt``) hold the config and weights; ``model.load_encoder`` rebuilds the model later.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from f1strategy.pretrain import schema as S
from f1strategy.pretrain.data import PretrainDataset, frame_weights, open_races, split_races
from f1strategy.pretrain.losses import Metrics, compute_losses
from f1strategy.pretrain.masking import TASKS, MaskConfig, Masker
from f1strategy.pretrain.model import F1Bert, F1BertConfig


def lr_at(step: int, base: float, warmup: int, total: int, floor: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def make_val_items(races, items, n: int, flagged_share: float = 0.5) -> list[tuple[int, int]]:
    """Fixed validation subset: evenly spaced frames plus (if there are any) flagged ones, which are rare."""
    flagged = [r.flagged_frames() for r in races]
    fl = [it for it in items if flagged[it[0]][it[1]]]
    rest = [it for it in items if not flagged[it[0]][it[1]]]
    n_fl = min(len(fl), int(n * flagged_share))
    pick = lambda xs, k: [xs[i] for i in np.linspace(0, len(xs) - 1, k).astype(int)] if k > 0 and xs else []  # noqa: E731
    return pick(fl, n_fl) + pick(rest, n - n_fl)


@torch.no_grad()
def evaluate(model, masker, races, val_items, device, batch_size, task_weights, seed=1234) -> dict[str, float]:
    model.eval()
    loader = DataLoader(PretrainDataset(races, val_items), batch_size=batch_size, shuffle=False)
    out: dict[str, float] = {}
    tot, n_tasks = 0.0, 0
    for k, name in enumerate(TASKS):
        met = Metrics()
        losses, nb = 0.0, 0
        g = torch.Generator(device=device).manual_seed(seed + k)  # same masks at every evaluation
        for raw in loader:
            mb = masker(raw, task=k, generator=g)
            o = model(mb)
            _, st = compute_losses(o, mb, task_weights)
            if f"loss_{name}" in st:
                losses += st[f"loss_{name}"]
                nb += 1
            met.update(o, mb)
        if nb:
            out[f"val/loss_{name}"] = losses / nb
            tot += task_weights[k] * losses / nb
            n_tasks += 1
        out.update({f"val/{m}": v for m, v in met.result().items()})
    out["val/loss"] = tot
    model.train()
    return out


def save_ckpt(path: Path, model, opt, step, cfg: F1BertConfig, args, best: float) -> None:
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg.to_dict(), "best_val": best,
         "args": vars(args), "schema": {"driver": S.DRIVER_FEATURES, "global": S.GLOBAL_FEATURES, "edge": S.EDGE_FEATURES, "track": S.TRACK_STATIC}},
        path,
    )  # fmt: skip


def parse_task_values(text: str, normalise: bool) -> tuple[float, float, float, float]:
    d = dict(kv.split("=") for kv in text.split(","))
    p = np.array([float(d.get(t, 0)) for t in TASKS])
    return tuple((p / p.sum() if normalise else p).tolist())  # type: ignore[return-value]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--task-probs", default="A=0.35,B=0.25,C=0.20,D=0.20", help="share of samples per pretext task")
    ap.add_argument("--task-weights", default="A=1,B=1,C=1,D=1", help="loss weight per task")
    ap.add_argument("--ghost-edge-mode", default="relational", choices=["relational", "topology", "none"])
    ap.add_argument("--p-cell", type=float, default=0.15)
    ap.add_argument("--p-group", type=float, default=0.15)
    ap.add_argument("--flag-oversample", type=float, default=5.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--embargo-s", type=float, default=60.0, help="few-race setting: seconds dropped on both sides of a validation block")
    ap.add_argument("--block-s", type=float, default=180.0, help="few-race setting: length of the held-out time blocks")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-frames", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (CUDA)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    races = open_races(a.cache)
    split = split_races(races, a.val_frac, embargo_s=a.embargo_s, block_s=a.block_s, seed=a.seed)
    if not split.train or not split.val:
        raise SystemExit("empty train or validation split")
    weights = frame_weights(races, split.train, a.flag_oversample)
    val_items = make_val_items(races, split.val, a.eval_frames)
    print(f"{len(races)} races | train frames {len(split.train)} (flagged x{a.flag_oversample:g}) | val frames {len(val_items)} of {len(split.val)}")
    print(f"validation: {split.val_races}")

    task_w = parse_task_values(a.task_weights, normalise=False)
    mcfg = MaskConfig(task_probs=parse_task_values(a.task_probs, normalise=True), p_cell=a.p_cell, p_group=a.p_group, ghost_edge_mode=a.ghost_edge_mode)
    masker = Masker(mcfg, device)
    cfg = F1BertConfig.preset(a.preset, dropout=a.dropout, n_track_tokens=races[0].meta["n_track_tokens"])
    model = F1Bert(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    step, best = 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best_val"]
        print(f"resumed from step {step}")
    print(f"model {a.preset}: {model.n_params() / 1e6:.2f} M parameters on {device}")

    n_samples = (a.steps - step) * a.batch_size
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=n_samples, replacement=True)
    loader = DataLoader(
        PretrainDataset(races, split.train), batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
        drop_last=True, persistent_workers=a.workers > 0, pin_memory=device.type == "cuda",
    )  # fmt: skip
    log = open(out / "log.jsonl", "a")
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if a.amp else torch.autocast(device.type, enabled=False)
    run: dict[str, float] = {}
    t0 = time.time()
    model.train()
    for raw in loader:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.lr, a.warmup, a.steps)
        mb = masker(raw)
        with amp:
            o = model(mb)
        loss, st = compute_losses(o, mb, task_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        step += 1
        for k, v in st.items():
            run[k] = run.get(k, 0.0) * 0.98 + v * 0.02 if k in run else v  # EMA for the console
        if step % a.log_every == 0:
            msg = " ".join(f"{k.replace('loss_', '')}={run[k]:.3f}" for k in ("loss", *[f"loss_{t}" for t in TASKS]) if k in run)
            print(f"step {step:>7} | {msg} | gnorm {gn:.2f} | lr {opt.param_groups[0]['lr']:.2e} | {(time.time() - t0) / max(1, step):.2f}s/step", flush=True)
            log.write(json.dumps({"step": step, **{k: float(v) for k, v in st.items()}, "gnorm": float(gn)}) + "\n")
            log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, masker, races, val_items, device, a.batch_size, task_w)
            print(f"  [val @ {step}] loss {ev['val/loss']:.4f} | " + " ".join(f"{t}={ev.get(f'val/loss_{t}', float('nan')):.3f}" for t in TASKS))
            log.write(json.dumps({"step": step, **ev}) + "\n")
            log.flush()
            if ev["val/loss"] < best:
                best = ev["val/loss"]
                save_ckpt(out / "best.pt", model, opt, step, cfg, a, best)
        if step % a.save_every == 0 or step == a.steps:
            save_ckpt(out / "last.pt", model, opt, step, cfg, a, best)
    print("done")


if __name__ == "__main__":
    main()
