"""F1 strategy web app: pick a race and a driver, see the race replay and the strategy predicted by the final ensemble BE.

    python -m f1strategy.webapp.app          # from the repository root, then open http://127.0.0.1:2000

Environment: F1_MODEL_CACHE (race caches, default ./cache), F1_RUNS (default ./runs), F1_WEB_CACHE (predictions and GIFs, default
./webapp_cache), F1_YEARS (default 2019-2026), F1_PORT (default 2000).
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path

import fastf1
from flask import Flask, jsonify, request, send_file, send_from_directory

from . import inference as inf

cache = os.path.expanduser("~/fastf1_cache")
os.makedirs(cache, exist_ok=True)
fastf1.Cache.enable_cache(cache)

PORT = int(os.environ.get("F1_PORT", "2000"))
GIFS = inf.WEB_CACHE / "gifs"
app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "static"))


def _years(spec: str) -> set[int]:
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


YEARS = _years(os.environ.get("F1_YEARS", "2019-2026"))  # every season that has labelled races in the cache


def split_of(year: int) -> str:
    """The season roles of the README: training 2019-2024, validation 2025, test 2026."""
    return "test" if year >= 2026 else "validation" if year == 2025 else "training"


def split_id(rid: str) -> tuple[int, int]:
    y, r = re.match(r"(\d{4})_round_(\d{2})", rid).groups()
    return int(y), int(r)


@functools.cache
def event_name(rid: str) -> str:
    y, r = split_id(rid)
    try:
        return fastf1.get_event(y, r)["EventName"]
    except Exception:
        return f"Round {r}"


COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
_info: dict[str, dict] = {}


def session_info(rid: str) -> dict:
    """Driver names, team colours and starting tyres of a race, from the FastF1 cache. Parts that are unavailable stay empty;
    failures are not remembered, so a later request can still fill them in."""
    if rid in _info:
        return _info[rid]
    info: dict = {"names": {}, "colors": {}, "start": {}}
    try:
        y, r = split_id(rid)
        s = fastf1.get_session(y, r, "R")
        try:
            s.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception:
            s.load(laps=False, telemetry=False, weather=False, messages=False)
        for x in s.results.itertuples():  # results are in finishing order
            info["names"][x.Abbreviation] = (x.FullName, x.TeamName)
            col = str(getattr(x, "TeamColor", "") or "").lstrip("#")
            if len(col) == 6:
                info["colors"][x.Abbreviation] = "#" + col
        try:
            first = s.laps.sort_values("LapNumber").dropna(subset=["Compound"]).groupby("Driver")["Compound"].first()
            info["start"] = {d: str(c).upper() for d, c in first.items() if str(c).upper() in COMPOUNDS}
        except Exception:
            pass
        if info["names"]:
            _info[rid] = info
    except Exception:
        pass
    return info


def driver_dicts(rid: str, codes: list[str]) -> list[dict]:
    """Drivers of a race in finishing order (unknown ones last, alphabetically) with name, team and team colour."""
    info = session_info(rid)
    order = [c for c in info["names"] if c in codes] + sorted(c for c in codes if c not in info["names"])
    return [
        {
            "code": c,
            "name": info["names"].get(c, (c, ""))[0],
            "team": info["names"].get(c, ("", ""))[1],
            "color": info["colors"].get(c, "#8a8a94"),
        }
        for c in order
    ]


def race_ids() -> list[str]:
    return sorted(
        p.name
        for p in inf.CACHE.glob("[0-9][0-9][0-9][0-9]_round_[0-9][0-9]")
        if split_id(p.name)[0] in YEARS and (p / "ft_labels.npz").exists()
    )


@app.get("/api/races")
def races():
    return jsonify(
        [
            {
                "id": i,
                "year": split_id(i)[0],
                "round": split_id(i)[1],
                "name": event_name(i),
                "split": split_of(split_id(i)[0]),
            }
            for i in race_ids()
        ]
    )


@app.get("/api/race")
def race():
    """Step 1: load a race -> its drivers (from the cache metadata, no model run) and its lap bounds."""
    rid = request.args["race"]
    try:
        codes = list(json.loads((inf.CACHE / rid / "meta.json").read_text())["driver_codes"])
        drivers = driver_dicts(rid, codes)
        return jsonify(id=rid, name=event_name(rid), total_laps=inf.total_laps(rid), drivers=drivers)
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.post("/api/predict")
def predict():
    b = request.get_json()
    try:
        out = inf.driver(b["race"], b["driver"], b.get("threshold"), b.get("locator") or inf.DEFAULT_LOCATOR)
        out["split"] = split_of(split_id(b["race"])[0])
        out["driver"] = driver_dicts(b["race"], [b["driver"]])[0]
        out["start_compound"] = session_info(b["race"])["start"].get(b["driver"])
        return jsonify(out)
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/api/field")
def field():
    """Predicted and actual stops of the whole field (one row per car, finishing order) with a short agreement summary."""
    rid = request.args["race"]
    try:
        out = inf.field(
            rid, request.args.get("threshold", type=float), request.args.get("locator") or inf.DEFAULT_LOCATOR
        )
        info = {d["code"]: d for d in driver_dicts(rid, [d["code"] for d in out["drivers"]])}
        order = list(info)
        out["drivers"] = [
            {**info[d["code"]], **d} for d in sorted(out["drivers"], key=lambda d: order.index(d["code"]))
        ]
        out["split"] = split_of(split_id(rid)[0])
        return jsonify(out)
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/api/gif")
def gif():
    """The GIF as the CLI's continuous mode makes it: 1 s sampling, decimated to at most 300 frames, 10 fps."""
    rid, a, z = request.args["race"], int(request.args.get("from", 1)), int(request.args.get("to", 10))
    total = inf.total_laps(rid)
    if not 1 <= a <= z <= total:
        return jsonify(error=f"Laps must satisfy 1 <= from <= to <= {total} (got {a}-{z})."), 400
    drv = request.args.get("driver", "")
    out = GIFS / f"{rid}_{a}-{z}_{drv or 'all'}.gif"
    if not out.exists():
        try:
            from f1strategy.session import SessionAnalyser

            from .gif import make_gif

            GIFS.mkdir(parents=True, exist_ok=True)
            y, r = split_id(rid)
            make_gif(SessionAnalyser(y, r, "R"), a, z, drv, f"{event_name(rid)} - laps {a}-{z}", out)
        except Exception as e:
            return jsonify(error=f"GIF failed: {type(e).__name__}: {e}"), 500
    return send_file(out.resolve(), mimetype="image/gif")


@app.get("/")
def index():
    f = Path(app.static_folder) / "index.html"
    if not f.exists():
        return f"index.html not found. Expected it at: {f}", 500, {"Content-Type": "text/plain"}
    return send_from_directory(app.static_folder, "index.html")


def main() -> None:
    print(f"open http://127.0.0.1:{PORT}  |  races found: {len(race_ids())}")
    app.run(host="127.0.0.1", port=PORT, debug=False)


if __name__ == "__main__":
    main()
