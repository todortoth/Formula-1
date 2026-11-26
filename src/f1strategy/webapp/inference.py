"""Inference of the web app: the final ensemble BE (3 x B + 3 x E, 10-lap windows), exactly as in `finetune.evaluate`.

For one race it scores every window of every car with the six member runs, averages them (`finetune.ensembling.ensemble`)
and returns, per car, the lap-by-lap predictions plus the ground truth. Results are cached per race in <F1_WEB_CACHE>/predictions/.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from f1strategy.finetune.dataset import DEFAULT_HOP_S, FtDataset, finetune_index, window_targets
from f1strategy.finetune.ensembling import ensemble
from f1strategy.finetune.labels import CMP_NAMES, TAB_NAMES, load_race_labels
from f1strategy.finetune.model import load_finetuned
from f1strategy.finetune.train import predict_windows, resolve_scale
from f1strategy.pretrain.data import RaceCache
from f1strategy.seqbert.masking import SeqMasker

CACHE = Path(os.environ.get("F1_MODEL_CACHE", "cache"))  # race caches (cache/<year>_round_<NN>/)
RUNS = Path(os.environ.get("F1_RUNS", "runs"))
WEB_CACHE = Path(os.environ.get("F1_WEB_CACHE", "webapp_cache"))  # predictions and GIFs made by the app
PREDICTIONS = WEB_CACHE / "predictions"
MEMBERS = [RUNS / f"long_{k}_s{s}" / "best.pt" for k in "BE" for s in (0, 1, 2)]  # system BE of the README
THRESHOLD_REPORT = RUNS / "eval" / "test2026_groups_long.json"  # thresholds fitted on 2025
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAP, LAPS_TOTAL = TAB_NAMES.index("lap_number"), TAB_NAMES.index("laps_total")
_lock = threading.Lock()
_members: list | None = None


def members():
    global _members
    if _members is None:
        _members = [load_finetuned(str(p), DEVICE) for p in MEMBERS]
    return _members


def thresholds() -> dict:
    """Decision thresholds of the ensemble (ens_BE), fitted on the 2025 season by evaluate.py; 0.5 if the report is missing."""
    try:
        return json.loads(THRESHOLD_REPORT.read_text())["thresholds"]["ens_BE"]
    except Exception:
        return {"pit5": 0.5}


def score_race(name: str) -> Path:
    """Run the ensemble over all windows of one race; cache to <F1_WEB_CACHE>/predictions/<race>.npz."""
    f = PREDICTIONS / f"{name}.npz"
    if f.exists():
        return f
    with _lock:
        if f.exists():
            return f
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        ms = members()
        ck0 = ms[0][1]
        scale = resolve_scale(ck0["scale"], {"n_tokens": ck0["n_tokens"]})
        horizons = tuple(h for h in ms[0][0].hc.horizons if all(h in m.hc.horizons for m, _ in ms))
        race = RaceCache(CACHE / name)
        lab = {name: load_race_labels(CACHE / name)}
        ix = finetune_index([race], scale, DEFAULT_HOP_S[ck0["scale"]])
        tgt = window_targets(ix, lab)
        masker, preds = SeqMasker(device=DEVICE), []
        for model, ck in ms:
            p = predict_windows(
                model,
                DataLoader(FtDataset(ix, lab), 8),
                masker,
                DEVICE,
                scale.scale_id,
                ck["scrub_cols"],
                ck.get("scrub_fill", "zero"),
                False,
            )
            idx = [list(model.hc.horizons).index(h) for h in horizons]
            preds.append(p | {"pit": p["pit"][..., idx]})
        e = ensemble(preds)
        np.savez_compressed(
            f,
            horizons=np.array(horizons),
            pit=e["pit"],
            laps=e["laps"],
            nomore=e["nomore"],
            cmp_prob=e["cmp_prob"],
            valid=tgt["valid"],
            true_laps=tgt["laps"],
            true_cmp=tgt["cmp"],
            lap=tgt["tab"][..., LAP],
            laps_total=tgt["tab"][..., LAPS_TOTAL],
            codes=np.array(race.meta["driver_codes"]),
        )
    return f


_total: dict[str, int] = {}


def total_laps(name: str) -> int:
    """Race length in laps, from the labels' strategy table (no model needed)."""
    if name not in _total:
        tab = load_race_labels(CACHE / name)["tab"]
        _total[name] = int(np.nanmax(tab[..., LAPS_TOTAL]))
    return _total[name]


def cars(name: str) -> list[str]:
    """Driver codes that have at least one scored window in this race."""
    z = np.load(score_race(name))
    return [str(c) for k, c in enumerate(z["codes"]) if z["valid"][:, k].any()]


def _runs(lap, pit, thr, max_gap=3.0):
    """Index runs of consecutive above-threshold windows (the bumps of the probability curve); one run = one expected stop.
    A run ends when the curve is below the threshold for more than `max_gap` laps."""
    runs, cur, last = [], [], None
    for i in range(len(lap)):
        if pit[i] < thr:
            continue
        if cur and lap[i] - last > max_gap:
            runs.append(cur)
            cur = []
        cur.append(i)
        last = lap[i]
    return runs + ([cur] if cur else [])


# Stop-lap estimators. Measured on the 16 test races (423 real stops, 307 predicted, 252 matched within 10 laps):
#   pit-within-1 first alarm   MAE 2.73 laps, median 1, 53% within 1 lap, but defined for only 168 of the 252 matched stops
#   pit-within-3 first alarm   MAE 3.31, median 2, 32% within 1 lap (205 stops)
#   pit-within-5 first alarm   MAE 3.66, median 3, 26% within 1 lap (223 stops)
#   laps-to-stop head          MAE 3.65, median 3, 22% within 1 lap (248 stops)
# The default is therefore p1, and the laps-to-stop head wherever p1 never fires.
LOCATORS = {
    "p1": "pit-within-1 first alarm (default)",
    "p3": "pit-within-3 first alarm",
    "p5": "pit-within-5 first alarm",
    "reg": "laps-to-stop head",
}
DEFAULT_LOCATOR = "p1"
MATCH_LAPS = 10  # a predicted stop is compared with the nearest real stop if it is at most this many laps away


def driver(name: str, code: str, thr: float | None = None, locator: str = DEFAULT_LOCATOR) -> dict:
    """Stops of one car. The pit5 curve DETECTS the stops (one bump = one stop); the stop LAP is placed by an estimator:
    p{h} = first window of the bump where the pit-within-h head crosses its own threshold, stop lap = that window's lap + h - 1
    (causal: nothing after the alarm is used); reg = median of the laps-to-stop head over the bump. If the chosen p{h} never
    fires inside the bump, the laps-to-stop head is used (`used` tells which one was)."""
    z = np.load(score_race(name))
    k = [str(c) for c in z["codes"]].index(code)
    th = thresholds()
    thr = th.get("pit5", 0.5) if thr is None else thr
    horizons = [int(h) for h in z["horizons"]]
    ok = z["valid"][:, k] & np.isfinite(z["lap"][:, k])
    j = np.where(ok)[0]
    j = j[np.argsort(z["lap"][j, k], kind="stable")]
    lap, laps = z["lap"][j, k], z["laps"][j, k]
    curves = {h: z["pit"][j, k, horizons.index(h)] for h in horizons}
    pit = curves[5]
    thrs = {h: (thr if h == 5 else th.get(f"pit{h}", 0.5)) for h in horizons}
    # stop lap = (laps completed) + laps to go; a car that has done lap-1 laps and stops at the end of lap L has L-(lap-1) to go
    true: dict[int, list[int]] = {}
    for i in j:
        if np.isfinite(z["true_laps"][i, k]):
            true.setdefault(int(round(z["lap"][i, k] - 1 + z["true_laps"][i, k])), []).append(int(z["true_cmp"][i, k]))
    actual = [
        {"lap": lp, "compound": CMP_NAMES[max(set(c), key=c.count)] if max(c) >= 0 else "UNKNOWN"}
        for lp, c in sorted(true.items())
    ]

    def nearest_error(lp):
        if lp is None or not actual:
            return None
        e = lp - min((a["lap"] for a in actual), key=lambda x: abs(x - lp))
        return e if abs(e) <= MATCH_LAPS else None

    pred = []
    for r in _runs(lap, pit, thr):
        if len(r) < 2 and pit[r].max() < 0.5:  # one weak window is noise; one confident window is a stop
            continue
        mean = z["cmp_prob"][j[r], k].mean(0)
        c = int(mean.argmax())
        est = {"reg": int(round(float(np.median(lap[r] - 1 + laps[r]))))}
        for h in (1, 3, 5):
            if h in curves:
                hit = [i for i in r if curves[h][i] >= thrs[h]]
                est[f"p{h}"] = int(round(float(lap[hit[0]] + h - 1))) if hit else None
        used = locator if est.get(locator) is not None else "reg"
        pred.append(
            {
                "lap": est[used],
                "used": used,
                "estimates": est,
                "error": {key: nearest_error(lp) for key, lp in est.items()},
                "compound": CMP_NAMES[c],
                "confidence": round(float(mean[c]), 2),
            }
        )
    total = int(np.nanmax(z["laps_total"][j, k])) if len(j) else 0

    def rounded(a):
        return [round(float(x), 3) for x in a]

    return {
        "curve": {
            "laps": [round(float(x), 2) for x in lap],
            "pit_prob": rounded(pit),
            "no_more": rounded(z["nomore"][j, k]),
        },
        "curves": {str(h): rounded(cv) for h, cv in curves.items()},
        "thresholds": {str(h): round(float(t), 3) for h, t in thrs.items()},
        "total_laps": total,
        "threshold": round(float(thr), 3),
        "peak": round(float(pit.max()), 3) if len(pit) else 0.0,
        "n_above": int((pit >= thr).sum()),
        "n_windows": int(len(pit)),
        "predicted": pred,
        "actual": actual,
        "locator": locator,
        "locators": LOCATORS,
    }


def field(name: str, thr: float | None = None, locator: str = DEFAULT_LOCATOR) -> dict:
    """Predicted and actual stops of every scored car of a race, plus how well they agree (same rules as `driver`)."""
    rows, n_actual, n_pred, n_match, n_missed = [], 0, 0, 0, 0
    for code in cars(name):
        o = driver(name, code, thr, locator)
        pred = [{"lap": p["lap"], "compound": p["compound"], "used": p["used"]} for p in o["predicted"]]
        n_actual += len(o["actual"])
        n_pred += len(pred)
        n_match += sum(1 for p in o["predicted"] if p["error"].get(p["used"]) is not None)
        n_missed += sum(1 for a in o["actual"] if not any(abs(p["lap"] - a["lap"]) <= MATCH_LAPS for p in pred))
        rows.append({"code": code, "predicted": pred, "actual": o["actual"]})
    return {
        "drivers": rows,
        "total_laps": total_laps(name),
        "threshold": round(float(thresholds().get("pit5", 0.5) if thr is None else thr), 3),
        "locator": locator,
        "locators": LOCATORS,
        "match_laps": MATCH_LAPS,
        "summary": {"actual": n_actual, "predicted": n_pred, "matched": n_match, "missed": n_missed},
    }
