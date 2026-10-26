"""
evaluate.py - test a fine-tuned run (default: the 2026 season) next to the baselines, on identical samples.

    python -m f1strategy.finetune.evaluate --cache cache --run runs/ft_medium_full/best.pt --years 2026 --report report_medium.json

F1 / precision / recall of the binary tasks are measured at a decision threshold that is fitted PER METHOD on a validation season
(``--val-years``, default 2025: the threshold that maximises F1 there) and then applied unchanged to the test season.  A method is
thereby not punished for probabilities that are merely scaled differently (a weighted loss, an unweighted gradient boosting).
``f1_best`` in the report is the same F1 with the threshold chosen on the test season itself: an upper bound, not a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from f1strategy.finetune import metrics as M
from f1strategy.finetune.baselines import fit_baselines, predict_baselines
from f1strategy.finetune.dataset import (
    DEFAULT_HOP_S,
    FtDataset,
    finetune_index,
    open_labelled_races,
    parse_years,
    window_targets,
)
from f1strategy.finetune.ensembling import ensemble, group_members, parse_groups
from f1strategy.finetune.model import load_finetuned
from f1strategy.finetune.train import decode_for_metrics, predict_windows, resolve_scale
from f1strategy.seqbert.masking import SeqMasker


def subset(t: dict, mask_windows: np.ndarray) -> dict:
    s = dict(t)
    s["valid"] = t["valid"] & mask_windows[:, None]
    return s


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument(
        "--run",
        required=True,
        help="best.pt of a fine-tuning run; several, comma-separated, give one table "
        "(same scale; rows are named after the run directory)",
    )
    ap.add_argument("--years", default="2026", help="test seasons")
    ap.add_argument(
        "--val-years", default="2025", help="season on which the decision thresholds are fitted ('' = fixed 0.5)"
    )
    ap.add_argument("--train-years", default="2019-2024", help="the baselines are fitted on these seasons")
    ap.add_argument("--baselines", default="prior,rule,gbdt")
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="add a row with the mean prediction of all the given runs (e.g. the seeds)",
    )
    ap.add_argument(
        "--groups",
        default="",
        help='extra ensemble rows, e.g. "B=0,1,2;E=3,4,5;Bgbdt=0,1,2,gbdt,gbdt,gbdt" '
        "(numbers = positions in --run, baseline names allowed, repeated = heavier); rows are named ens_<name>",
    )
    ap.add_argument("--report", default="")
    ap.add_argument("--hop-s", type=float, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    device = torch.device(
        "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    )

    runs = [r for r in a.run.split(",") if r]
    loaded = [(Path(r).parent.name or Path(r).stem, *load_finetuned(r, device)) for r in runs]
    ck0 = loaded[0][2]
    if any((c["scale"], c["n_tokens"]) != (ck0["scale"], ck0["n_tokens"]) for _, _, c in loaded):
        raise SystemExit("all runs of one table must have the same scale")
    horizons = tuple(h for h in loaded[0][1].hc.horizons if all(h in m.hc.horizons for _, m, _ in loaded))
    if not horizons:
        raise SystemExit("the runs have no pit horizon in common")
    scale = resolve_scale(ck0["scale"], {"n_tokens": ck0["n_tokens"]})
    hop_s = a.hop_s or DEFAULT_HOP_S[ck0["scale"]]
    masker = SeqMasker(device=device)

    def run_raw(model, ck, ix, lab):
        loader = DataLoader(FtDataset(ix, lab), 16, num_workers=a.workers)
        p = predict_windows(
            model, loader, masker, device, scale.scale_id, ck["scrub_cols"], ck.get("scrub_fill", "zero"), a.amp
        )
        idx = [
            list(model.hc.horizons).index(h) for h in horizons
        ]  # the horizons all runs share, in the order of the first run
        return p | {"pit": p["pit"][..., idx]}

    te_races, te_lab = open_labelled_races(a.cache, parse_years(a.years))
    if not te_races:
        raise SystemExit(f"no labelled races in {a.years}: scrape, cache and label them first")
    te_ix = finetune_index(te_races, scale, hop_s)
    te_t = window_targets(te_ix, te_lab)
    raws = {name: run_raw(m, c, te_ix, te_lab) for name, m, c in loaded}
    te_runs = [raws[n] for n, _, _ in loaded]
    groups = parse_groups(a.groups)
    if a.ensemble and len(raws) > 1:
        raws["ensemble"] = ensemble(list(raws.values()))
    results = {n: decode_for_metrics(r) for n, r in raws.items()}

    # --- validation season: thresholds
    val_results, va_t = {}, None
    val_years = parse_years(a.val_years)
    if val_years:
        va_races, va_lab = open_labelled_races(a.cache, val_years)
        if not va_races:
            raise SystemExit(f"no labelled races in --val-years {a.val_years}")
        va_ix = finetune_index(va_races, scale, hop_s)
        va_t = window_targets(va_ix, va_lab)
        va_raws = {name: run_raw(m, c, va_ix, va_lab) for name, m, c in loaded}
        va_runs = [va_raws[n] for n, _, _ in loaded]
        if a.ensemble and len(va_raws) > 1:
            va_raws["ensemble"] = ensemble(list(va_raws.values()))
        val_results |= {n: decode_for_metrics(r) for n, r in va_raws.items()}
        if val_years & parse_years(a.years):
            print(
                "note: the validation and the test season overlap, so the thresholds are fitted on the evaluated races (optimistic)"
            )

    names = [n for n in a.baselines.split(",") if n]
    base_te, base_va = {}, {}
    if names:
        tr_races, tr_lab = open_labelled_races(a.cache, parse_years(a.train_years))
        tr_ix = finetune_index(tr_races, scale, hop_s)
        models = fit_baselines(window_targets(tr_ix, tr_lab), horizons, tuple(names))
        base_te = predict_baselines(models, te_t, horizons)
        results |= base_te
        if va_t is not None:
            base_va = predict_baselines(models, va_t, horizons)
            val_results |= base_va
    try:
        for g, toks in groups.items():
            results[f"ens_{g}"] = ensemble(group_members(toks, te_runs, base_te))
            if va_t is not None:
                val_results[f"ens_{g}"] = ensemble(group_members(toks, va_runs, base_va))
    except ValueError as e:
        raise SystemExit(f"--groups: {e}") from e

    thresholds = (
        {n: M.fit_thresholds(val_results[n], va_t, horizons) for n in results if n in val_results}
        if va_t is not None
        else {}
    )
    reports = {n: M.summarize(p, te_t, horizons, thresholds.get(n)) for n, p in results.items()}
    print(f"\n{len(te_races)} races {a.years}, {len(te_ix)} windows, scale {ck0['scale']}")
    print(M.report_text(reports, horizons, a.val_years if thresholds else ""))
    stats = M.label_stats(te_t, horizons)
    print("\n" + M.label_text({f"test {a.years}": stats}, horizons))

    by_race = {}
    for ri, r in enumerate(te_races):
        m = te_t["race"] == ri
        by_race[r.name] = {
            n: M.flatten_report(M.summarize(p, subset(te_t, m), horizons, thresholds.get(n)))
            for n, p in results.items()
        }
    out = {"run": a.run, "years": a.years, "val_years": a.val_years, "scale": ck0["scale"], "horizons": horizons,
           "modes": {n: c["mode"] for n, _, c in loaded}, "scrub": {n: c["scrub"] for n, _, c in loaded}, "n_windows": len(te_ix), "thresholds": thresholds, "label_stats": stats, "reports": reports,
           "by_race": by_race}  # fmt: skip
    if a.report:
        Path(a.report).write_text(json.dumps(out, indent=1, default=float))
        print(f"\nreport -> {a.report}")
    return out


if __name__ == "__main__":
    main()
