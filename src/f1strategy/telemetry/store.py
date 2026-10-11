"""
TelemetryStoreClass.py · Formula 1 Strategic Decisions
=========================================================
Random access to every driver's telemetry at arbitrary session times.

FastF1 delivers car data (speed, throttle, ...) at ~240 ms and position data (X/Y/Z) at ~220 ms
on *different* clocks, so neither can be read "at time t" directly. This class interpolates both
(linear for continuous channels, nearest sample for gear/brake/DRS) and additionally projects the
position onto the TrackModel, which gives:

* ``track_s_m``       exact distance from the start/finish line + lateral offset from the racing line
* ``progress_m``      cumulative race distance (monotone-ish, laps * length + track_s_m)
* time gaps           the *true* gap between two cars: how long ago was the car ahead where the car
                      behind is now?  (``gap_seconds``) - no lap-end approximation.

The progress curve of every driver is computed once, lazily, over the full position stream.
Everything that is returned for a time t only depends on samples <= t (the inverse lookup
``time_at_progress`` is only ever asked for progress values the car already passed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from f1strategy.track.model import TrackModel
from f1strategy.utils import pos_frame, td_seconds, wrap_angle

_NAN = float("nan")
MIN_HEADING_SPEED_MPS = 2.0  # below this the velocity vector is too noisy to define a heading


@dataclass
class DriverSample:
    """Interpolated telemetry of one driver at one instant."""

    code: str
    t: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    speed_kmh: float
    heading_rad: float
    heading_valid: bool
    heading_rel_track_rad: float
    throttle: float
    brake: float
    gear: float
    rpm: float
    drs_raw: float
    on_track: bool
    track_node: int
    track_s_m: float
    lateral_offset_m: float
    progress_m: float


class TelemetryStore:
    """Per-driver interpolated telemetry + progress along the track."""

    def __init__(
        self, session: Any, track: TrackModel, driver_map: dict[str, str], race_start_s: float | None = None
    ) -> None:
        self.track = track
        self.race_start_s = race_start_s if race_start_s is not None else self._guess_race_start(session)
        self._pos: dict[str, dict[str, np.ndarray]] = {}
        self._car: dict[str, dict[str, np.ndarray]] = {}
        self._progress: dict[str, dict[str, np.ndarray]] = {}

        for number, code in driver_map.items():
            pos = session.pos_data.get(number) if session.pos_data else None
            if pos is None or len(pos) == 0:
                continue
            frame = pos_frame(pos)
            if len(frame["t"]) < 2:
                continue
            self._pos[code] = frame

            car = session.car_data.get(number) if session.car_data else None
            if car is not None and len(car) > 0:
                tc = td_seconds(car["SessionTime"])
                ok = np.isfinite(tc)
                tc, idx = np.unique(tc[ok], return_index=True)
                self._car[code] = {
                    "t": tc,
                    **{
                        name: car[col].to_numpy(float)[ok][idx]
                        for name, col in (
                            ("speed", "Speed"),
                            ("rpm", "RPM"),
                            ("gear", "nGear"),
                            ("throttle", "Throttle"),
                            ("brake", "Brake"),
                            ("drs", "DRS"),
                        )
                    },
                }

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def _guess_race_start(session: Any) -> float:
        try:
            v = session.session_start_time
            if pd.notna(v):
                return float(pd.Timedelta(v).total_seconds())
        except Exception:
            pass
        starts = td_seconds(session.laps["LapStartTime"])
        starts = starts[np.isfinite(starts)]
        return float(starts.min()) if len(starts) else 0.0

    @property
    def codes(self) -> list[str]:
        return list(self._pos.keys())

    def _prog(self, code: str) -> dict[str, np.ndarray]:
        """Cumulative progress (m) for every position sample; anchored so that progress 0 = start/finish line
        at the race start (cars on the grid, behind the line, therefore start slightly negative)."""
        if code in self._progress:
            return self._progress[code]
        p = self._pos[code]
        L = self.track.length_m
        proj = self.track.project(p["x"], p["y"])
        s = proj.s
        ds = (np.diff(s) + L / 2) % L - L / 2  # wrapped step in (-L/2, L/2]
        unwrapped = s[0] + np.concatenate([[0.0], np.cumsum(ds)])
        # Anchor on the last real sample at/before lights out. Never interpolate the wrapped ``s`` here: a pole
        # sitter can cross the line between two samples (s jumps L-8 -> 4) and the average would be nonsense.
        i0 = int(np.clip(np.searchsorted(p["t"], self.race_start_s, side="right") - 1, 0, len(s) - 1))
        anchor = s[i0] - L if s[i0] > L / 2 else s[i0]  # grid slots are behind the line -> slightly negative
        progress = unwrapped + (anchor - unwrapped[i0])
        mono = np.maximum.accumulate(progress) + np.arange(len(progress)) * 1e-9  # strictly increasing for inversion
        out = {"progress": progress, "mono": mono}
        self._progress[code] = out
        return out

    # ------------------------------------------------------------------ queries
    def progress_at(self, code: str, t: float) -> float:
        return float(np.interp(t, self._pos[code]["t"], self._prog(code)["progress"]))

    def time_at_progress(self, code: str, progress_m: float) -> float:
        """Session time at which ``code`` first reached ``progress_m`` (NaN if it never did / before data)."""
        return float(np.interp(progress_m, self._prog(code)["mono"], self._pos[code]["t"], left=np.nan, right=np.nan))

    def gap_seconds(self, behind: str, ahead: str, t: float) -> float:
        """Time by which ``behind`` trails ``ahead`` at time t (>0 when really behind)."""
        t_ahead = self.time_at_progress(ahead, self.progress_at(behind, t))
        return _NAN if np.isnan(t_ahead) else t - t_ahead

    def sample(self, code: str, t: float, velocity_dt: float = 0.3) -> DriverSample:
        p = self._pos[code]
        tp = p["t"]
        x, y, z = (float(np.interp(t, tp, p[k])) for k in ("x", "y", "z"))
        x0, y0 = float(np.interp(t - velocity_dt, tp, p["x"])), float(np.interp(t - velocity_dt, tp, p["y"]))
        x1, y1 = float(np.interp(t + velocity_dt, tp, p["x"])), float(np.interp(t + velocity_dt, tp, p["y"]))
        vx, vy = (x1 - x0) / (2 * velocity_dt), (y1 - y0) / (2 * velocity_dt)
        v_pos = float(np.hypot(vx, vy))

        # project the interpolated point itself (interpolating node ids would break at the start/finish line)
        proj = self.track.project([x], [y])
        proj_node, proj_s, proj_lat = int(proj.node[0]), float(proj.s[0]), float(proj.lateral[0])

        tangent = float(self.track.heading[proj_node]) if proj_node >= 0 else _NAN
        heading_valid = v_pos >= MIN_HEADING_SPEED_MPS
        heading = float(np.arctan2(vy, vx)) if heading_valid else tangent
        rel = float(wrap_angle(heading - tangent)) if np.isfinite(heading) and np.isfinite(tangent) else _NAN

        speed = rpm = gear = throttle = brake = drs = _NAN
        c = self._car.get(code)
        if c is not None:
            tc = c["t"]
            speed = float(np.interp(t, tc, c["speed"]))
            rpm = float(np.interp(t, tc, c["rpm"]))
            throttle = float(np.interp(t, tc, c["throttle"]))
            j = int(np.clip(np.searchsorted(tc, t), 1, len(tc) - 1)) if len(tc) > 1 else 0
            if len(tc) > 1 and abs(tc[j - 1] - t) < abs(tc[j] - t):
                j -= 1
            gear, brake, drs = float(c["gear"][j]), float(c["brake"][j]), float(c["drs"][j])
        else:
            speed = v_pos * 3.6

        return DriverSample(
            code=code,
            t=t,
            x_m=x,
            y_m=y,
            z_m=z,
            vx_mps=vx,
            vy_mps=vy,
            speed_kmh=speed,
            heading_rad=heading,
            heading_valid=heading_valid,
            heading_rel_track_rad=rel,
            throttle=throttle,
            brake=brake,
            gear=gear,
            rpm=rpm,
            drs_raw=drs,
            on_track=bool(np.interp(t, tp, p["on"].astype(float)) > 0.5),
            track_node=proj_node,
            track_s_m=proj_s,
            lateral_offset_m=proj_lat,
            progress_m=self.progress_at(code, t),
        )
