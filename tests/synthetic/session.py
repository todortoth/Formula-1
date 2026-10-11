"""
synthetic_session.py - offline stand-in for a loaded FastF1 race session
=========================================================
Mimics the *structure* of ``fastf1.core.Session`` (column names, units, dtypes, quirks such as all-zero
position samples, pos/car data on different clocks) around a track whose geometry is known exactly, so
tests can compare the pipeline against ground truth without network access.

Timeline (session seconds): lights out 30 | DRS enabled 100 | yellow + sector-3 flags 95-128 |
SC 180-245 ("in this lap" 235) | DRS off 180, on 250 | VSC 330, ending 355, clear 360.
NOR pits at the end of lap 3, SAI retires during lap 4.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

T0_DATE = pd.Timestamp("2025-07-27 12:00:00")
RACE_START = 30.0
N_LAPS = 6
DRIVERS = [  # code, number, pace factor, start offset behind the line (m), compound lap1-3, compound lap4-6
    ("VER", "1", 1.000, 8.0, "MEDIUM", "HARD"),
    ("NOR", "4", 0.999, 16.0, "MEDIUM", "HARD"),
    ("LEC", "16", 0.998, 24.0, "MEDIUM", "HARD"),
    ("HAM", "44", 0.995, 32.0, "SOFT", "HARD"),
    ("ALO", "14", 0.990, 40.0, "MEDIUM", "HARD"),
    ("SAI", "55", 0.985, 48.0, "HARD", "HARD"),
]
EXTRA_DRIVERS = [  # joined behind the six main cars when n_drivers > 6 (14 more -> a 20 car grid)
    (code, number, 0.984 - 0.001 * k, 56.0 + 8.0 * k, "MEDIUM", "HARD")
    for k, (code, number) in enumerate(
        [("PIA", "81"), ("RUS", "63"), ("GAS", "10"), ("ALB", "23"), ("TSU", "22"), ("OCO", "31"), ("STR", "18"),
         ("HUL", "27"), ("BOR", "5"), ("HAD", "6"), ("LAW", "30"), ("BEA", "87"), ("COL", "43"), ("ANT", "12")]
    )
]  # fmt: skip
SECTOR_FRACS = (0.33, 0.68)
DRS_ZONES_FRAC = [(0.05, 0.15), (0.55, 0.62)]
MARSHAL_MARKERS_FRAC = [
    0.02 + j / 12 for j in range(12)
]  # sector 1 starts at 2 % of the lap, sector 12 wraps over the line
TRACK_STATUS = [
    (0.0, "1", "AllClear"),
    (95.0, "2", "Yellow"),
    (128.0, "1", "AllClear"),
    (180.0, "4", "SCDeployed"),
    (245.0, "1", "AllClear"),
    (330.0, "6", "VSCDeployed"),
    (355.0, "7", "VSCEnding"),
    (360.0, "1", "AllClear"),
]


def _td(seconds) -> pd.Timedelta:
    # NB: pd.Timedelta(seconds=<float>) emits a NumPy 'generic unit' DeprecationWarning with pandas 2.3 + NumPy 2.5
    return pd.to_timedelta(float(seconds), unit="s")


def _build_track(target_length: float = 5000.0):
    th = np.linspace(0, 2 * np.pi, 40001)
    r = 1 + 0.30 * np.cos(2 * th) + 0.18 * np.sin(3 * th) + 0.06 * np.cos(5 * th)
    x, y, z = r * np.cos(th), r * np.sin(th), 6.0 * np.sin(2 * th)
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    k = target_length / d[-1]
    x, y, d = x * k, y * k, d * k
    L = float(d[-1])
    s = np.arange(0.0, L, 1.0)
    X, Y, Z = np.interp(s, d, x), np.interp(s, d, y), np.interp(s, d, z)
    xs, ys = gaussian_filter1d(X, 10, mode="wrap"), gaussian_filter1d(Y, 10, mode="wrap")
    dx, dy = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)
    v = gaussian_filter1d(np.clip(np.sqrt(30.0 / np.maximum(kappa, 1e-4)), 25.0, 88.0), 25, mode="wrap")
    return L, s, X, Y, Z, v, kappa


def _in_drs_zone(s_mod: np.ndarray, L: float) -> np.ndarray:
    m = np.zeros(s_mod.shape, bool)
    for a, b in DRS_ZONES_FRAC:
        m |= (s_mod >= a * L) & (s_mod <= b * L)
    return m


def _drs_enabled(t: np.ndarray) -> np.ndarray:
    return ((t >= 100) & (t < 180)) | (t >= 250)


def make_synthetic_session(
    seed: int = 0,
    *,
    with_circuit_info: bool = True,
    with_drs: bool = True,
    with_weather: bool = True,
    with_messages: bool = True,
    with_track_status: bool = True,
    with_sector_times: bool = True,
    n_drivers: int = 6,
) -> SimpleNamespace:
    """Build the synthetic session. The keyword flags switch off one data source each, to test degraded input;
    ``n_drivers`` (6-20) adds back markers for realistic scale."""
    if not 6 <= n_drivers <= 6 + len(EXTRA_DRIVERS):
        raise ValueError(f"n_drivers must be between 6 and {6 + len(EXTRA_DRIVERS)}")
    drivers_list = DRIVERS + EXTRA_DRIVERS[: n_drivers - 6]
    rng = np.random.default_rng(seed)
    drs_on = with_drs
    L, s_dense, X, Y, Z, v, kappa = _build_track()
    per = lambda a, sm: np.interp(sm % L, np.append(s_dense, L), np.append(a, a[0]))  # periodic lookup

    pos_data, car_data, rows, S_fn, t_end_driver = {}, {}, [], {}, {}
    retire_s = 3.4 * L  # SAI stops here

    # ---- motion model: cumulative distance S(t) per driver by integrating dt = ds / v
    for code, number, pace, off, c1, c2 in drivers_list:
        s_grid = np.arange(-off, (N_LAPS + 1) * L, 1.0)
        dt = 1.0 / (per(v, s_grid) * pace)
        t_grid = RACE_START + np.concatenate([[0.0], np.cumsum(dt[:-1])])
        if code == "SAI":
            keep = s_grid <= retire_s
            s_grid, t_grid = s_grid[keep], t_grid[keep]
        S_fn[code] = lambda tt, sg=s_grid, tg=t_grid: np.interp(tt, tg, sg, left=sg[0], right=sg[-1])
        t_end_driver[code] = float(t_grid[-1])

    t_total = max(t_end_driver.values()) + 20.0

    # ---- position + car data streams
    for code, number, pace, off, c1, c2 in drivers_list:
        tp = np.arange(0.0, t_total, 0.22) + rng.uniform(-0.01, 0.01, int(np.ceil(t_total / 0.22)))
        tp = np.sort(tp[tp >= 0])
        S = S_fn[code](tp)
        xs = per(X, S) + rng.normal(0, 0.15, len(tp))
        ys = per(Y, S) + rng.normal(0, 0.15, len(tp))
        zs = per(Z, S)
        xi, yi, zi = np.round(xs * 10), np.round(ys * 10), np.round(zs * 10)
        xi[:2] = yi[:2] = zi[:2] = 0  # FastF1 'no signal' artefact
        pos_data[number] = pd.DataFrame(
            {
                "Date": T0_DATE + pd.to_timedelta(tp, unit="s"),
                "Status": "OnTrack",
                "X": xi,
                "Y": yi,
                "Z": zi,
                "Source": "pos",
                "Time": pd.to_timedelta(tp, unit="s"),
                "SessionTime": pd.to_timedelta(tp, unit="s"),
            }
        )

        tc = np.arange(0.1, t_total, 0.24)
        Sc = S_fn[code](tc)
        spd = per(v, Sc) * pace
        moving = np.gradient(Sc, tc) > 1.0
        acc = np.gradient(spd, tc)
        in_zone = _in_drs_zone(Sc % L, L)
        drs = np.zeros(len(tc))
        if drs_on:
            drs[in_zone & _drs_enabled(tc) & moving] = 12
        eligible = _in_drs_zone((Sc + 60) % L, L) & ~in_zone & _drs_enabled(tc) & moving
        if drs_on:
            drs[eligible] = 8
        speed_kmh = np.where(moving, spd * 3.6, 0.0)
        car_data[number] = pd.DataFrame(
            {
                "Date": T0_DATE + pd.to_timedelta(tc, unit="s"),
                "RPM": np.where(moving, 6000 + speed_kmh * 20, 0.0),
                "Speed": speed_kmh,
                "nGear": np.clip((speed_kmh // 45).astype(int) + 1, 1, 8) * moving,
                "Throttle": np.where(acc >= -1.0, 100.0, 0.0) * moving,
                "Brake": acc < -3.0,
                "DRS": drs,
                "Source": "car",
                "Time": pd.to_timedelta(tc, unit="s"),
                "SessionTime": pd.to_timedelta(tc, unit="s"),
            }
        )

    # ---- laps table
    def status_str(a: float, b: float) -> str:
        codes, cur = [], "1"
        for ts, code, _ in TRACK_STATUS:
            if ts <= a:
                cur = code
            elif ts < b:
                if not codes:
                    codes.append(cur)
                codes.append(code)
        return "".join(codes) if codes else cur

    lap_end_times: dict[int, dict[str, float]] = {k: {} for k in range(1, N_LAPS + 1)}

    def t_at_S(code: str, S_target: float) -> float:
        Sf, lo, hi = S_fn[code], 0.0, t_end_driver[code]
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if Sf(mid) < S_target else (lo, mid)
        return hi

    for code, *_ in drivers_list:  # lap k ends when the cumulative distance reaches k * L (if the car got that far)
        for k in range(1, N_LAPS + 1):
            if S_fn[code](t_end_driver[code]) >= k * L - 1e-6:
                lap_end_times[k][code] = t_at_S(code, k * L)

    for code, number, pace, off, c1, c2 in drivers_list:
        prev_end = RACE_START
        for k in range(1, N_LAPS + 1):
            end = lap_end_times[k].get(code)
            if end is None:
                if code == "SAI" and k == 4:  # retirement lap: ends where the car stopped
                    end = t_end_driver[code]
                else:
                    break
            s1 = t_at_S(code, (k - 1) * L + SECTOR_FRACS[0] * L) if end is not None else np.nan
            s2 = t_at_S(code, (k - 1) * L + SECTOR_FRACS[1] * L)
            if code == "SAI" and k == 4:
                s1, s2 = (s1 if s1 < end else np.nan), np.nan
            comp, age, stint = (c1, k, 1) if k <= 3 else (c2, k - 3, 2)
            if code == "SAI":
                comp, age, stint = "HARD", k, 1
            pos_rank = (
                1 + sum(1 for c, tt in lap_end_times[k].items() if tt < end)
                if k in lap_end_times and code in lap_end_times[k]
                else np.nan
            )
            retired_lap = code == "SAI" and k == 4
            rows.append(
                {
                    "Time": _td(end),
                    "Driver": code,
                    "DriverNumber": number,
                    "LapTime": _td(end - prev_end) if not retired_lap else pd.NaT,
                    "LapNumber": float(k),
                    "Stint": float(stint),
                    "PitOutTime": _td(prev_end + 18.0) if (code == "NOR" and k == 4) else pd.NaT,
                    "PitInTime": _td(end - 6.0) if (code == "NOR" and k == 3) else pd.NaT,
                    "Sector1SessionTime": _td(s1) if (with_sector_times and np.isfinite(s1)) else pd.NaT,
                    "Sector2SessionTime": _td(s2) if (with_sector_times and np.isfinite(s2)) else pd.NaT,
                    "Sector3SessionTime": _td(end) if not retired_lap else pd.NaT,
                    "Compound": comp,
                    "TyreLife": float(age),
                    "FreshTyre": True,
                    "Team": "Synthetic",
                    "LapStartTime": _td(prev_end),
                    "TrackStatus": status_str(prev_end, end),
                    "Position": float(pos_rank) if not retired_lap else np.nan,
                    "IsAccurate": not retired_lap and k > 1,
                    "Deleted": False,
                }
            )
            prev_end = end
    laps = pd.DataFrame(rows).sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    # ---- race control
    ts_df = pd.DataFrame(
        {
            "Time": pd.to_timedelta([e[0] for e in TRACK_STATUS], unit="s"),
            "Status": [e[1] for e in TRACK_STATUS],
            "Message": [e[2] for e in TRACK_STATUS],
        }
    )

    def msg(t, cat, text, flag=None, scope=None, sector=None, status=None):
        return {
            "Time": T0_DATE + _td(t),
            "Category": cat,
            "Message": text,
            "Status": status,
            "Flag": flag,
            "Scope": scope,
            "Sector": sector,
            "RacingNumber": None,
            "Lap": None,
        }

    messages = pd.DataFrame(
        [
            msg(5, "Other", "GREEN LIGHT - PIT EXIT OPEN"),
            msg(95, "Flag", "YELLOW IN TRACK SECTOR 3", "YELLOW", "Sector", 3),
            msg(110, "Flag", "DOUBLE YELLOW IN TRACK SECTOR 3", "DOUBLE YELLOW", "Sector", 3),
            msg(100, "Drs", "DRS ENABLED", status="ENABLED"),
            msg(120, "Flag", "YELLOW IN TRACK SECTOR 8", "YELLOW", "Sector", 8),
            msg(128, "Flag", "CLEAR IN TRACK SECTOR 3", "CLEAR", "Sector", 3),
            msg(128, "Flag", "CLEAR IN TRACK SECTOR 8", "CLEAR", "Sector", 8),
            msg(180, "SafetyCar", "SAFETY CAR DEPLOYED"),
            msg(180, "Drs", "DRS DISABLED", status="DISABLED"),
            msg(235, "SafetyCar", "SAFETY CAR IN THIS LAP"),
            msg(250, "Drs", "DRS ENABLED", status="ENABLED"),
        ]
    )
    weather = pd.DataFrame(
        {
            "Time": pd.to_timedelta(np.arange(0, t_total, 60.0), unit="s"),
            "AirTemp": 24.0,
            "Humidity": 50.0,
            "Pressure": 1010.0,
            "Rainfall": False,
            "TrackTemp": 38.0,
            "WindDirection": 90,
            "WindSpeed": 2.5,
        }
    )

    # ---- circuit info (X/Y in 1/10 m like the real thing)
    mk_s = np.array(MARSHAL_MARKERS_FRAC) * L
    corner_s = s_dense[np.argsort(-kappa)[:400]]
    corner_pick = []
    for c in corner_s:  # five well separated curvature peaks
        if all(min(abs(c - q), L - abs(c - q)) > 300 for q in corner_pick):
            corner_pick.append(float(c))
        if len(corner_pick) == 5:
            break
    corner_pick.sort()
    to_df = lambda ss, nums: pd.DataFrame(
        {
            "X": per(X, np.array(ss)) * 10,
            "Y": per(Y, np.array(ss)) * 10,
            "Number": nums,
            "Letter": "",
            "Angle": 0.0,
            "Distance": np.nan,
        }
    )
    circuit_info = SimpleNamespace(
        corners=to_df(corner_pick, list(range(1, len(corner_pick) + 1))),
        marshal_sectors=to_df(mk_s, list(range(1, 13))),
        marshal_lights=None,
        rotation=0.0,
    )

    def _no_circuit_info():
        raise ConnectionError("circuit info unavailable (synthetic outage)")

    session = SimpleNamespace(
        laps=laps,
        pos_data=pos_data,
        car_data=car_data,
        track_status=ts_df if with_track_status else None,
        race_control_messages=messages if with_messages else None,
        weather_data=weather if with_weather else None,
        t0_date=T0_DATE,
        total_laps=N_LAPS,
        session_start_time=_td(RACE_START),
        drivers=[d[1] for d in drivers_list],
        event={"EventName": "Synthetic Grand Prix"},
        get_driver=lambda n: {"Abbreviation": {d[1]: d[0] for d in drivers_list}[n]},
        get_circuit_info=(lambda: circuit_info) if with_circuit_info else _no_circuit_info,
    )

    truth = dict(
        L=L,
        S=lambda code, t: S_fn[code](np.asarray(t)),
        per=per,
        X=X,
        Y=Y,
        v=v,
        kappa=kappa,
        corner_s=corner_pick,
        marshal_s=mk_s,
        t_end=t_end_driver,
        lap_end=lap_end_times,
        retire_s=retire_s,
        t_total=t_total,
    )
    return SimpleNamespace(session=session, truth=truth)
