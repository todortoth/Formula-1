"""
train.py - pre-training of the temporal BERT.

    python -m f1strategy.seqbert.train --cache cache --out runs/short_gnn --scale short --graph-encoder gnn
    python -m f1strategy.seqbert.train ... --scale medium --batch-size 16
    python -m f1strategy.seqbert.train ... --scale mixed --graph-encoder gnn --time-embed   # all three scales at once

Every step is either a reconstruction step (each sample gets one of B/T/P/Z) or, with probability --p-contrast, a
contrastive step (pairs of windows of the same circuit + race phase; other circuits are the negatives).

``--scale mixed`` trains ONE model on short, medium and long windows at the same time: every step draws one scale (a batch
needs a single sequence length) with probability ``--scale-probs`` and uses that scale's batch size and scale embedding.
Validation, ``best.pt`` (mean over the scales) and ``best_<scale>.pt`` are tracked per scale.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from f1strategy.seqbert import data as D
from f1strategy.seqbert.config import PRESETS, SCALES, Scale
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

SINGLE_BATCH = {"short": 32, "medium": 16, "long": 8}  # defaults of a single-scale run
MIXED_BATCH = {"short": 128, "medium": 32, "long": 16}  # ~3,000 frames per step at every scale
MIXED_PROBS = {"short": 0.5, "medium": 0.35, "long": 0.15}  # long has only ~3,000 windows: do not over-sample it


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--scale", choices=[*SCALES, "mixed"], default="short", help="one scale, or 'mixed' = several at once"
    )
    p.add_argument("--scales", default="short,medium,long", help="the scales of --scale mixed, comma separated")
    p.add_argument(
        "--scale-probs", default=None, help="mixed: share of steps per scale, e.g. short=0.5,medium=0.35,long=0.15"
    )
    p.add_argument("--batch-sizes", default=None, help="mixed: batch size per scale, e.g. short=128,medium=32,long=16")
    p.add_argument("--n-tokens", default=None, help="override the number of tokens: 192, or per scale short=20,long=96")
    p.add_argument("--span-s", default=None, help="override the window duration in seconds (same forms)")
    p.add_argument("--span-laps", default=None, help="override the window duration in laps (same forms)")
    p.add_argument("--block-s", type=float, default=180.0, help="validation block length when < 5 races (seconds)")
    p.add_argument("--graph-encoder", choices=["flatten", "sector", "gnn"], default="flatten")
    p.add_argument("--preset", choices=list(PRESETS), default="small")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=None, help="single scale: default 32 short / 16 medium / 8 long")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1, help="dropout of the temporal encoder")
    p.add_argument(
        "--time-embed",
        action="store_true",
        help="add an embedding of the elapsed seconds (recommended for mixed scales: token i means a different time per scale)",
    )
    p.add_argument(
        "--resume", default=None, help="continue a run from its ckpt.pt (weights, optimiser, step, schedule)"
    )
    p.add_argument(
        "--init-from", default=None, help="start from the weights of a checkpoint (fresh optimiser and step)"
    )
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA)")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2, help="DataLoader workers (shared between the scales)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=64, help="validation windows per task (and scale)")
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
    p.add_argument(
        "--loss-norm",
        choices=["none", "feature"],
        default="none",
        help="feature: divide every column's error by its target variance, so slow variables count as much as noisy ones",
    )
    # contrastive
    p.add_argument("--p-contrast", type=float, default=0.0, help="share of contrastive steps (0 = off)")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--phase-bins", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--contrast-weight", type=float, default=1.0)
    return p.parse_args()


def parse_kv(s, names, cast, what):
    """'a=1,b=2' -> {'a': 1, 'b': 2}; a bare value applies to every name; None -> {}."""
    if s is None:
        return {}
    if "=" not in s:
        return {n: cast(s) for n in names}
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        if k.strip() not in names:
            raise SystemExit(f"--{what}: unknown scale {k!r} (this run has {', '.join(names)})")
        out[k.strip()] = cast(v)
    return out


def parse_probs(s: str):
    d = dict(kv.split("=") for kv in s.split(","))
    v = [float(d.get(t, 0.0)) for t in TASKS]
    tot = sum(v)
    return tuple(x / tot for x in v)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    return a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))


class Stream:
    """Endless iterator over a DataLoader (restarts it when the sampler is exhausted)."""

    def __init__(self, loader):
        self.loader, self.it = loader, iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


@dataclass
class ScaleRun:
    """Everything one scale needs during training."""

    name: str
    scale: Scale
    batch: int
    prob: float
    index: D.WindowIndex
    train_ds: Dataset
    val_ds: Dataset
    desc: list
    stream: Stream | None = None
    pairs: Stream | None = None


@torch.no_grad()
def evaluate(model, masker, ds, a, device, scale_id, batch):
    model.eval()
    acc = MetricAcc()
    n = min(a.eval_windows, len(ds))
    sub = torch.utils.data.Subset(ds, np.linspace(0, len(ds) - 1, n).astype(int).tolist())
    for k, _name in enumerate(TASKS):
        g = torch.Generator(device=device).manual_seed(1234 + k)  # same masks at every evaluation
        for raw in DataLoader(sub, batch_size=batch, shuffle=False):
            mb = masker(raw, task=k, generator=g)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp):
                out = model(mb, scale_id)
            accumulate_metrics(acc, mb, out["recon"])
    model.train()
    return acc.result()


def resolve_scales(a):
    """-> {name: Scale}, {name: batch size}, {name: probability} for a single scale or a mixed run."""
    names = [n.strip() for n in a.scales.split(",")] if a.scale == "mixed" else [a.scale]
    for n in names:
        if n not in SCALES:
            raise SystemExit(f"unknown scale {n!r}")
    toks = parse_kv(a.n_tokens, names, int, "n-tokens")
    secs = parse_kv(a.span_s, names, float, "span-s")
    laps = parse_kv(a.span_laps, names, float, "span-laps")
    scales = {}
    for n in names:
        sc = SCALES[n]
        if n in toks:
            sc = replace(sc, n_tokens=toks[n])
        if n in secs or n in laps:
            sc = replace(sc, span_s=secs.get(n), span_laps=laps.get(n))
        scales[n] = sc
    if a.scale == "mixed":
        given = parse_kv(a.batch_sizes, names, int, "batch-sizes")
        batches = {n: given.get(n, a.batch_size or MIXED_BATCH[n]) for n in names}
        probs = parse_kv(a.scale_probs, names, float, "scale-probs") or {n: MIXED_PROBS[n] for n in names}
        if set(probs) != set(names):
            raise SystemExit("--scale-probs must list every scale of the run")
    else:
        batches = {names[0]: a.batch_size or SINGLE_BATCH[names[0]]}
        probs = {names[0]: 1.0}
    tot = sum(probs.values())
    return scales, batches, {n: v / tot for n, v in probs.items()}


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(
        "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    )
    mixed = a.scale == "mixed"
    scales, batches, probs = resolve_scales(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    races = D.open_races(a.cache)
    start = int(torch.load(a.resume, map_location="cpu")["step"]) + 1 if a.resume else 0
    workers = -(-a.workers // len(scales)) if a.workers > 0 else 0  # the workers are shared between the scales
    runs: list[ScaleRun] = []
    for name, scale in scales.items():
        ix = D.WindowIndex(races, scale)
        tr, va, vdesc = D.split_windows(ix, a.val_frac, block_s=a.block_s, seed=a.seed)
        if not tr or not va:
            raise SystemExit(f"not enough windows for scale {name}: train {len(tr)}, val {len(va)} (add races)")
        print(f"[data] {len(races)} races | {name}: {scale.n_tokens} tokens | windows train {len(tr)} / val {len(va)}")
        train_ds, val_ds = D.WindowDataset(ix, tr), D.WindowDataset(ix, va)
        r = ScaleRun(name, scale, batches[name], probs[name], ix, train_ds, val_ds, vdesc)
        n_steps = max(
            1, math.ceil((a.steps - start) * r.prob * 1.3) + 20
        )  # expected draws plus slack (Stream restarts)
        sampler = WeightedRandomSampler(D.window_weights(ix, tr), n_steps * r.batch, True)
        r.stream = Stream(DataLoader(train_ds, batch_size=r.batch, sampler=sampler, num_workers=workers,
                                     drop_last=True, persistent_workers=workers > 0))  # fmt: skip
        if a.p_contrast > 0:
            groups = D.window_groups(ix, tr, a.phase_bins)
            print(
                f"[contrastive] {name}: {len({g[0] for g in groups})} circuit(s), {len(groups)} (circuit, phase) groups"
            )
            ps = D.PairBatchSampler(groups, ix, tr, a.pairs, a.steps, seed=a.seed)
            r.pairs = Stream(DataLoader(train_ds, batch_sampler=ps, num_workers=workers))
        runs.append(r)
    if mixed:
        print("[mixed] steps per scale: " + ", ".join(f"{r.name} {r.prob:.0%} (batch {r.batch})" for r in runs))

    mcfg = SeqMaskConfig(parse_probs(a.task_probs), a.zone_frac, a.zone_position, a.mask_fill,
                         tuple(float(x) for x in a.pit_window.split(",")), a.ghost_edge_mode, a.pit_edge_mode)  # fmt: skip
    masker = SeqMasker(mcfg, device)
    cfg = SeqBertConfig(
        graph_encoder=a.graph_encoder,
        n_track=int(races[0].track.shape[0]),
        n_drivers=int(races[0].a["drv"].shape[1]),
        mask_fill=a.mask_fill,
        dropout=a.dropout,
        time_embed=a.time_embed,
        max_len=max(256, *(r.scale.n_tokens for r in runs)),
        **PRESETS[a.preset],
    )
    model = SeqF1Bert(cfg).to(device)
    print(
        f"[model] {a.graph_encoder} encoder, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, device {device}"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    best_val = float("inf")
    best_scale = {r.name: float("inf") for r in runs}
    if a.resume or a.init_from:
        ck0 = torch.load(a.resume or a.init_from, map_location=device)
        model.load_state_dict(ck0["model"])
        if a.resume:
            opt.load_state_dict(ck0["opt"])
            best_val = float(ck0.get("best_val", float("inf")))
            best_scale.update(ck0.get("best_scale", {}))
            print(f"[resume] from step {ck0['step']} (best val so far {best_val:.4f})")
        else:
            print(f"[init-from] initialized model weights from {a.init_from}; optimizer and step start fresh")
    n_tokens = {r.name: r.scale.n_tokens for r in runs}
    (out / "args.json").write_text(
        json.dumps(
            {
                **vars(a),
                "model": cfg.as_dict(),
                "n_tokens": n_tokens,
                "batch_sizes": batches,
                "scale_probs": probs,
                "windows": {r.name: [len(r.train_ds), len(r.val_ds)] for r in runs},
                "val": runs[0].desc,
            },
            indent=1,
        )  # fmt: skip
    )
    rng = np.random.default_rng(a.seed + start)
    model.train()
    t0 = time.time()

    with open(out / "log.jsonl", "a") as log:
        for step in range(start, a.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a)
            run = runs[int(rng.choice(len(runs), p=[r.prob for r in runs]))] if mixed else runs[0]
            sid = run.scale.scale_id
            contrast = run.pairs is not None and rng.random() < a.p_contrast
            rec = {"step": step, "scale": run.name}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp):
                if contrast:
                    mb = masker.plain(next(run.pairs), token_dropout=0.15)
                    loss, acc1 = contrastive_loss(model.embed(mb, sid), a.temperature)
                    loss = loss * a.contrast_weight
                    rec.update(kind="contrast", loss=loss.item(), top1=acc1.item())
                else:
                    mb = masker(next(run.stream))
                    out_ = model(mb, sid)
                    loss, per = reconstruction_loss(
                        out_["recon"],
                        build_targets(mb),
                        build_weights(mb),
                        mb.task,
                        a.loss_kind,
                        a.cos_weight,
                        a.loss_norm,
                    )
                    rec.update(
                        kind="recon",
                        loss=loss.detach().item(),
                        **{f"loss_{k}": v.detach().item() for k, v in per.items()},
                    )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip))
            rec["grad_norm"] = gn
            if math.isfinite(gn):
                opt.step()
            else:  # a non-finite gradient would corrupt the weights: skip the step
                rec["skipped"] = True
                print(f"step {step}: non-finite gradient, step skipped")
            if step % 25 == 0:
                tag = f" [{run.name}]" if mixed else ""
                print(
                    f"step {step:5d} {rec['kind']:8s} loss {rec['loss']:.4f}  gnorm {gn:.2f}  {time.time() - t0:.0f}s{tag}"
                )
            if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
                ev_all, by_scale = {}, {}
                for r in runs:
                    ev = evaluate(model, masker, r.val_ds, a, device, r.scale.scale_id, r.batch)
                    sel = [ev[f"{t}/mse"] for t in a.select_tasks if f"{t}/mse" in ev]
                    by_scale[r.name] = float(np.mean(sel)) if sel else float("inf")
                    ev_all.update({(f"{r.name}/{k}" if mixed else k): v for k, v in ev.items()})
                    shown = {
                        k: round(v, 3) for k, v in ev.items() if k.endswith(("mae", "acc", "interp", "mean", "mse"))
                    }
                    print(f"[eval{' ' + r.name if mixed else ''}]", shown)
                rec["eval"] = ev_all
                val_loss = float(np.mean(list(by_scale.values())))
                is_best = val_loss < best_val
                best_val = min(best_val, val_loss)
                rec.update(val_loss=val_loss, best_val=best_val, is_best=is_best)
                if mixed:
                    rec["val_loss_by_scale"] = by_scale
                ck = {"model": model.state_dict(), "cfg": cfg.as_dict(), "scale": a.scale, "scales": n_tokens,
                      "n_tokens": n_tokens[runs[0].name] if not mixed else None,
                      "step": step, "val_loss": val_loss, "eval": ev_all}  # fmt: skip
                if is_best:
                    torch.save(ck, out / "best.pt")  # weights only
                if mixed:  # the best point of every scale on its own (the mean over scales can hide a scale's optimum)
                    for r in runs:
                        if by_scale[r.name] < best_scale[r.name]:
                            best_scale[r.name] = by_scale[r.name]
                            torch.save({**ck, "best_for": r.name}, out / f"best_{r.name}.pt")
                torch.save(
                    {**ck, "opt": opt.state_dict(), "best_val": best_val, "best_scale": best_scale}, out / "ckpt.pt"
                )
                detail = ", ".join(f"{n} {v:.4f}" for n, v in by_scale.items()) if mixed else ""
                print(
                    f"[val] mean MSE over {a.select_tasks}: {val_loss:.4f} (best {best_val:.4f}{' - new best.pt' if is_best else ''})"
                    + (f"  [{detail}]" if detail else "")
                )
            log.write(json.dumps(rec) + "\n")
            log.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
