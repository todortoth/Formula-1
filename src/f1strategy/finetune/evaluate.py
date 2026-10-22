"""
evaluate.py - test a fine-tuned run (default: the 2026 season) next to the baselines, on identical samples.

    python -m f1strategy.finetune.evaluate --cache cache --run runs/ft_medium_full/best.pt --years 2026 --report report_medium.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from f1strategy.finetune import metrics as M
from f1strategy.finetune.baselines import run_baselines
from f1strategy.finetune.dataset import (
    DEFAULT_HOP_S,
    FtDataset,
    finetune_index,
    open_labelled_races,
    parse_years,
    window_targets,
)
from f1strategy.finetune.model import load_finetuned
from f1strategy.finetune.train import decode_for_metrics, predict_windows, resolve_scale
from f1strategy.seqbert.masking import SeqMasker


def subset(t: dict, mask_windows: np.ndarray) -> dict:
    s = dict(t)
    s["valid"] = t["valid"] & mask_windows[:, None]
    return s


def table(rows: dict[str, dict], horizons) -> str:
    h = horizons[0]
    cols = [
        f"pit{h}/auc",
        f"pit{h}/f1",
        f"pit{h}/precision",
        f"pit{h}/recall",
        f"pit{h}/positive_rate",
        "laps/mae",
        "cmp/acc",
        "cmp/macro_f1",
        "nomore/auc",
        "nomore/f1",
        "nomore/precision",
        "nomore/recall",
        "nomore/positive_rate",
    ]

    head = f"{'method':<20}" + "".join(f"{c:>13}" for c in cols)
    lines = [head, "-" * len(head)]

    for name, report in rows.items():
        flat = M.flatten_report(report)
        values = []

        for col in cols:
            value = flat.get(col, float("nan"))
            if col.endswith("/positive_rate"):
                values.append(f"{value:>12.1%} ")
            else:
                values.append(f"{value:>13.3f}")

        lines.append(f"{name:<20}" + "".join(values))

    return "\n".join(lines)


def print_label_distribution(report: dict) -> None:
    print("\nLabel distribution (ground truth)")
    print("-" * 78)

    for task, values in report.items():
        if not isinstance(values, dict):
            continue

        if "positive_count" in values:
            print(
                f"{task:<12} "
                f"positive={values['positive_count']:>8} "
                f"({values['positive_rate']:>7.2%})  "
                f"negative={values['negative_count']:>8} "
                f"({values['negative_rate']:>7.2%})  "
                f"n={values['n']:>8}"
            )

        elif "class_counts" in values:
            print(f"{task}: n={values['n']}")
            for name, count, rate in zip(
                values["classes"],
                values["class_counts"],
                values["class_rates"],
                strict=False,
            ):
                print(f"  {name:<18} {count:>8} ({rate:>7.2%})")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True, help="best.pt of a fine-tuning run")
    ap.add_argument("--years", default="2026", help="test seasons")
    ap.add_argument("--train-years", default="2019-2024", help="the baselines are fitted on these seasons")
    ap.add_argument("--baselines", default="prior,rule,gbdt")
    ap.add_argument("--report", default="")
    ap.add_argument("--hop-s", type=float, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    device = torch.device(
        "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    )

    model, ck = load_finetuned(a.run, device)
    horizons = tuple(model.hc.horizons)
    scale = resolve_scale(ck["scale"], {"n_tokens": ck["n_tokens"]})
    hop_s = a.hop_s or DEFAULT_HOP_S[ck["scale"]]

    te_races, te_lab = open_labelled_races(a.cache, parse_years(a.years))
    if not te_races:
        raise SystemExit(f"no labelled races in {a.years}: scrape, cache and label them first")
    te_ix = finetune_index(te_races, scale, hop_s)
    te_t = window_targets(te_ix, te_lab)
    ds = FtDataset(te_ix, te_lab)
    preds = predict_windows(model, DataLoader(ds, 16, num_workers=a.workers), SeqMasker(device=device), device, scale.scale_id,
                            ck["scrub_cols"], ck.get("scrub_fill", "zero"), a.amp)  # fmt: skip
    results = {f"model ({ck['mode']})": decode_for_metrics(preds)}

    names = [n for n in a.baselines.split(",") if n]
    if names:
        tr_races, tr_lab = open_labelled_races(a.cache, parse_years(a.train_years))
        tr_ix = finetune_index(tr_races, scale, hop_s)
        results |= run_baselines(window_targets(tr_ix, tr_lab), te_t, horizons, tuple(names))

    reports = {n: M.summarize(p, te_t, horizons) for n, p in results.items()}
    print(
        f"\n{len(te_races)} races {a.years}, {len(te_ix)} windows, {reports[next(iter(reports))]['n_samples']} (window, car) samples, scale {ck['scale']}\n"
    )
    print(table(reports, horizons))

    print_label_distribution(reports[next(iter(reports))])

    by_race = {}
    for ri, r in enumerate(te_races):
        m = te_t["race"] == ri
        by_race[r.name] = {n: M.flatten_report(M.summarize(p, subset(te_t, m), horizons)) for n, p in results.items()}
    out = {"run": a.run, "years": a.years, "scale": ck["scale"], "horizons": horizons, "mode": ck["mode"], "scrub": ck["scrub"],
           "n_windows": len(te_ix), "reports": reports, "by_race": by_race}  # fmt: skip
    if a.report:
        Path(a.report).write_text(json.dumps(out, indent=1, default=float))
        print(f"\nreport -> {a.report}")
    return out


if __name__ == "__main__":
    main()
