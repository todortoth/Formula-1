"""
TrackModelClass.py · Formula 1 Strategic Decisions
=========================================================
Builds the static "backbone" of the race graph: a closed centre line of the circuit resampled
every ~10 m, where every point (track node) carries geometry and track-layout information.

How the centre line is obtained
-------------------------------
FastF1 has no track outline. It is rebuilt from the position telemetry (X/Y/Z, 1/10 m units) of a
clean, fast reference lap (no pit, no SC/yellow if possible). Node 0 sits on the start/finish line.
Because the lap is closed, the real spacing is length / round(length / 10 m), i.e. 10 m +- ~0.5 %.

Static per-node information
---------------------------
x/y/z, tangent heading, signed curvature (+ = left), gradient, official sector (S1/S2/S3, from the
reference lap's sector times), corner number (circuit info), marshal sector number (circuit info,
needed to place sector-specific yellow flags), DRS zone membership (inferred from the DRS channel).

DRS detection points
--------------------
Neither the F1 live-timing feed nor FastF1 contain the detection points. They can be passed in manually
(``drs_detection_points_m``) or derived as "activation start - offset" (``drs_detection_offset_m``); the
nearest 10 m node is then flagged ``is_drs_detection`` (max. 5 m from the exact position).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.spatial import cKDTree

from f1strategy.utils import POS_UNIT_TO_M, circular_runs, pos_arrays, run_length, td_seconds, wrap_angle

log = logging.getLogger(__name__)


@dataclass
class Projection:
    """Result of projecting points onto the centre line."""

    node: np.ndarray  # nearest node index (int, -1 where the input was NaN)
    s: np.ndarray  # distance along the lap in m, [0, length)
    lateral: np.ndarray  # signed distance to the centre line in m (+ = left of the direction of travel)


class TrackModel:
    """Closed, evenly resampled centre line with static per-node features."""

    def __init__(self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, length_m: float) -> None:
        self.n = int(len(x_m))
        self.length_m = float(length_m)
        self.spacing_m = self.length_m / self.n
        self.s = np.arange(self.n) * self.spacing_m
        self.x = np.asarray(x_m, float)
        self.y = np.asarray(y_m, float)
        self.z = np.asarray(z_m, float)

        # layout layers, filled by the attach_* / set_* methods
        self.sector_idx = np.full(self.n, -1, dtype=int)  # 0,1,2  (-1 unknown)
        self.sector_end_s: tuple[float, float] | None = None  # end of S1 / S2 along the lap
        self.corner_number = np.zeros(self.n, dtype=int)  # 0 = no corner marker on this node
        self.marshal_sector = np.full(self.n, -1, dtype=int)  # -1 = unknown
        self.drs_zone = np.zeros(self.n, dtype=int)  # 0 = none, 1..K
        self.is_drs_start = np.zeros(self.n, dtype=bool)
        self.is_drs_end = np.zeros(self.n, dtype=bool)
        self.is_drs_detection = np.zeros(self.n, dtype=bool)
        self.drs_detection_s: list[float] = []
        self.meta: dict[str, Any] = {}

        self._compute_geometry()
        self._xy = np.column_stack([self.x, self.y])
        self._tree = cKDTree(self._xy)

    # ------------------------------------------------------------------ geometry
    def _compute_geometry(self) -> None:
        h = self.spacing_m
        sigma = max(1.0, 12.0 / h)  # ~12 m smoothing, only used for derivatives (positions stay raw)
        xs = gaussian_filter1d(self.x, sigma, mode="wrap")
        ys = gaussian_filter1d(self.y, sigma, mode="wrap")
        zs = gaussian_filter1d(self.z, 2.0 * sigma, mode="wrap")
        dx = (np.roll(xs, -1) - np.roll(xs, 1)) / (2 * h)
        dy = (np.roll(ys, -1) - np.roll(ys, 1)) / (2 * h)
        self.heading = np.arctan2(dy, dx)
        self.curvature = wrap_angle(np.roll(self.heading, -1) - np.roll(self.heading, 1)) / (2 * h)
        self.gradient = (np.roll(zs, -1) - np.roll(zs, 1)) / (2 * h)

    # ------------------------------------------------------------------ projection
    def circular_delta(self, s_from: Any, s_to: Any) -> Any:
        """Signed shortest distance along the lap from ``s_from`` to ``s_to`` (+ = ahead), in (-L/2, L/2]."""
        L = self.length_m
        return (np.asarray(s_to) - np.asarray(s_from) + L / 2) % L - L / 2

    def node_at_s(self, s: Any) -> Any:
        return np.round((np.asarray(s) % self.length_m) / self.spacing_m).astype(int) % self.n

    def project(self, x_m: Any, y_m: Any) -> Projection:
        """Project points (metres) onto the centre line; vectorised, NaN-safe."""
        x = np.atleast_1d(np.asarray(x_m, float))
        y = np.atleast_1d(np.asarray(y_m, float))
        node = np.full(x.shape, -1, dtype=int)
        s_out = np.full(x.shape, np.nan)
        lat_out = np.full(x.shape, np.nan)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            return Projection(node, s_out, lat_out)

        pts = np.column_stack([x[valid], y[valid]])
        _, nearest = self._tree.query(pts)
        n, h = self.n, self.spacing_m
        best_d = np.full(len(pts), np.inf)
        best_s = np.zeros(len(pts))
        best_lat = np.zeros(len(pts))
        for seg in ((nearest - 1) % n, nearest):  # the two segments touching the nearest vertex
            a = self._xy[seg]
            b = self._xy[(seg + 1) % n]
            ab = b - a
            t = np.clip(((pts - a) * ab).sum(1) / np.maximum((ab**2).sum(1), 1e-12), 0.0, 1.0)
            diff = pts - (a + t[:, None] * ab)
            d = np.hypot(diff[:, 0], diff[:, 1])
            cross = ab[:, 0] * diff[:, 1] - ab[:, 1] * diff[:, 0]
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_s = np.where(better, (seg + t) * h, best_s)
            best_lat = np.where(better, np.where(cross >= 0, d, -d), best_lat)

        s_valid = best_s % self.length_m
        node[valid] = self.node_at_s(s_valid)
        s_out[valid] = s_valid
        lat_out[valid] = best_lat
        return Projection(node, s_out, lat_out)

    # ------------------------------------------------------------------ layout layers
    def set_sector_boundaries(self, s1_end: float, s2_end: float) -> None:
        """Official timing sectors; node i belongs to S1 if s < s1_end, S2 if s < s2_end, else S3."""
        self.sector_end_s = (float(s1_end), float(s2_end))
        self.sector_idx = np.where(self.s < s1_end, 0, np.where(self.s < s2_end, 1, 2))

    def set_marshal_markers(self, numbers: Sequence[int], s_values: Sequence[float]) -> None:
        """Marshal sector N spans from its marker to the next marker (marker = sector start).
        Nodes before the first marker belong to the last sector (wrap around the start/finish line)."""
        order = np.argsort(s_values)
        s_sorted = np.asarray(s_values, float)[order]
        num_sorted = np.asarray(numbers, int)[order]
        idx = np.searchsorted(s_sorted, self.s, side="right") - 1  # -1 wraps to the last marker
        self.marshal_sector = num_sorted[idx]

    def set_corners(self, numbers: Sequence[int], s_values: Sequence[float]) -> None:
        self.corner_number[:] = 0
        for num, s in zip(numbers, s_values, strict=True):
            k = int(self.node_at_s(s))
            if self.corner_number[k] == 0:
                self.corner_number[k] = int(num)

    def set_drs_zones_from_points(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        min_fraction: float = 0.10,
        min_zone_nodes: int = 10,
        merge_gap_nodes: int = 5,
    ) -> int:
        """Infer DRS activation zones from positions at which any car had DRS open.

        Nodes whose hit count reaches ``min_fraction`` of the busiest node are "active". Gaps up to
        ``merge_gap_nodes`` are closed, zones shorter than ``min_zone_nodes`` (100 m) are dropped.
        Returns the number of zones found."""
        self.drs_zone[:] = 0
        self.is_drs_start[:] = False
        self.is_drs_end[:] = False
        if len(x_m) < 50:
            return 0
        counts = np.bincount(self.project(x_m, y_m).node[np.isfinite(x_m)], minlength=self.n)[: self.n]
        if counts.max() < 5:
            return 0
        active = counts >= max(3, min_fraction * counts.max())
        tiled = np.tile(active, 3)
        closed = binary_closing(tiled, structure=np.ones(merge_gap_nodes + 1, dtype=bool))
        active = closed[self.n : 2 * self.n]

        runs = [r for r in circular_runs(active) if run_length(r, self.n) >= min_zone_nodes]
        runs.sort(key=lambda r: r[0])
        for zid, (a, b) in enumerate(runs, start=1):
            for k in range(run_length((a, b), self.n)):
                self.drs_zone[(a + k) % self.n] = zid
            self.is_drs_start[a] = True
            self.is_drs_end[b] = True
        return len(runs)

    def set_drs_detection_points(self, s_values: Sequence[float]) -> None:
        self.is_drs_detection[:] = False
        self.drs_detection_s = [float(s) % self.length_m for s in s_values]
        for s in self.drs_detection_s:
            self.is_drs_detection[int(self.node_at_s(s))] = True

    @property
    def drs_zone_starts_s(self) -> list[float]:
        return [float(self.s[k]) for k in np.flatnonzero(self.is_drs_start)]

    # ------------------------------------------------------------------ export helpers
    def to_frame(self) -> pd.DataFrame:
        """One row per track node with all static features."""
        return pd.DataFrame(
            {
                "s_m": self.s,
                "x_m": self.x,
                "y_m": self.y,
                "z_m": self.z,
                "heading_rad": self.heading,
                "curvature": self.curvature,
                "gradient": self.gradient,
                "sector": self.sector_idx + 1,  # 1..3, 0 = unknown
                "corner_number": self.corner_number,
                "marshal_sector": self.marshal_sector,
                "drs_zone": self.drs_zone,
                "is_drs_start": self.is_drs_start,
                "is_drs_end": self.is_drs_end,
                "is_drs_detection": self.is_drs_detection,
            }
        )

    def __repr__(self) -> str:
        return (
            f"TrackModel({self.n} nodes, length={self.length_m:.0f} m, spacing={self.spacing_m:.2f} m, "
            f"drs_zones={int(self.drs_zone.max())}, corners={int((self.corner_number > 0).sum())})"
        )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_polyline(cls, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray, spacing_m: float = 10.0) -> TrackModel:
        """Resample a (nearly) closed polyline - first and last point at the start/finish line - to ~spacing_m."""
        x, y, z = (np.asarray(a, float).copy() for a in (x_m, y_m, z_m))
        gap = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        if gap > 60.0:
            log.warning(
                "Reference lap does not close well (gap %.1f m); the track outline may be off near the line.", gap
            )
        x[-1], y[-1], z[-1] = x[0], y[0], z[0]  # force an exactly closed loop
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        keep = np.concatenate([[True], np.diff(d) > 0.05])
        keep[-1] = True
        x, y, z, d = x[keep], y[keep], z[keep], d[keep]
        length = float(d[-1])
        n = max(int(round(length / spacing_m)), 8)
        s_new = np.arange(n) * (length / n)
        return cls(
            PchipInterpolator(d, x)(s_new), PchipInterpolator(d, y)(s_new), PchipInterpolator(d, z)(s_new), length
        )

    @classmethod
    def from_session(
        cls,
        session: Any,
        spacing_m: float = 10.0,
        drs_detection_points_m: Sequence[float] | None = None,
        drs_detection_offset_m: float | None = None,
        use_circuit_info: bool = True,
        infer_drs: bool = True,
    ) -> TrackModel:
        """Build the track from a loaded FastF1 session (needs laps + telemetry)."""
        ref = _select_reference_lap(session.laps)
        number = str(ref["DriverNumber"])
        pos = session.pos_data[number]
        t, x, y, z = pos_arrays(pos)
        t0 = float(td_seconds(pd.Series([ref["LapStartTime"]]))[0])
        t1 = float(td_seconds(pd.Series([ref["Time"]]))[0])

        inside = (t > t0) & (t < t1)
        tt = np.concatenate([[t0], t[inside], [t1]])
        track = cls.from_polyline(np.interp(tt, t, x), np.interp(tt, t, y), np.interp(tt, t, z), spacing_m)
        track.meta.update(
            reference_driver=str(ref.get("Driver", number)),
            reference_lap=int(ref["LapNumber"]),
            reference_lap_time_s=float(td_seconds(pd.Series([ref["LapTime"]]))[0]),
        )

        # official sectors from the reference lap's sector crossing times
        s_times = [ref.get("Sector1SessionTime"), ref.get("Sector2SessionTime")]
        if all(pd.notna(v) for v in s_times):
            ts = td_seconds(pd.Series(s_times))
            ps = track.project(np.interp(ts, t, x), np.interp(ts, t, y)).s
            if ps[0] < ps[1]:
                track.set_sector_boundaries(ps[0], ps[1])
        if track.sector_end_s is None:
            log.warning("No usable sector times on the reference lap; falling back to equal thirds for sectors.")
            track.set_sector_boundaries(track.length_m / 3, 2 * track.length_m / 3)

        if use_circuit_info:
            track._attach_circuit_info(session)
        if infer_drs:
            track._attach_drs_from_telemetry(session)
        if drs_detection_points_m is not None:
            track.set_drs_detection_points(drs_detection_points_m)
        elif drs_detection_offset_m is not None and track.is_drs_start.any():
            track.set_drs_detection_points([s - drs_detection_offset_m for s in track.drs_zone_starts_s])
        return track

    def _attach_circuit_info(self, session: Any) -> None:
        try:
            info = session.get_circuit_info()
        except Exception as exc:  # network / MultiViewer outage etc.
            log.warning("Circuit info unavailable (%s): corners and marshal sectors will be unknown.", exc)
            return
        if info is None:
            log.warning("Circuit info unavailable: corners and marshal sectors will be unknown.")
            return
        for attr, setter in (("corners", self.set_corners), ("marshal_sectors", self.set_marshal_markers)):
            df = getattr(info, attr, None)
            if df is None or len(df) == 0:
                continue
            s = self.project(df["X"].to_numpy(float) * POS_UNIT_TO_M, df["Y"].to_numpy(float) * POS_UNIT_TO_M).s
            setter(df["Number"].to_numpy(int), s)

    def _attach_drs_from_telemetry(self, session: Any) -> None:
        xs, ys = [], []
        for number, car in session.car_data.items():
            pos = session.pos_data.get(number)
            if pos is None or len(pos) == 0 or "DRS" not in car:
                continue
            open_mask = np.isin(car["DRS"].to_numpy(), (10, 12, 14))
            if not open_mask.any():
                continue
            t_c = td_seconds(car["SessionTime"])[open_mask]
            t_p, x_p, y_p, _ = pos_arrays(pos)
            xs.append(np.interp(t_c, t_p, x_p))
            ys.append(np.interp(t_c, t_p, y_p))
        if not xs:
            log.info("No DRS-open samples found: track has no DRS zones (wet race / old season?).")
            return
        n_zones = self.set_drs_zones_from_points(np.concatenate(xs), np.concatenate(ys))
        log.info("Inferred %d DRS zone(s).", n_zones)


# ---------------------------------------------------------------------- module helpers
def _select_reference_lap(laps: pd.DataFrame) -> pd.Series:
    """Fastest 'clean' lap: timed, no pit in/out, accurate, not lap 1, preferably under green flag only."""
    valid = laps.dropna(subset=["LapStartTime", "Time", "LapTime"])
    valid = valid[valid["PitInTime"].isna() & valid["PitOutTime"].isna()]
    # each preference is only applied if at least one lap survives it
    preferences = [
        lambda d: d["IsAccurate"] == True if "IsAccurate" in d else None,  # noqa: E712
        lambda d: d["LapNumber"] > 1,
        lambda d: d["TrackStatus"].astype(str) == "1" if "TrackStatus" in d else None,
    ]
    for pref in preferences:
        mask = pref(valid)
        if mask is not None and mask.any():
            valid = valid[mask]
    if valid.empty:
        raise ValueError("No usable lap to derive the track outline from.")
    return valid.loc[valid["LapTime"].idxmin()]
