"""
RaceControlTimelineClass.py · Formula 1 Strategic Decisions
=========================================================
Turns the two Race Control feeds of FastF1 into a time-indexed state that can be queried at any
session time:

* ``session.track_status``          -> global state: clear / yellow / SC / red / VSC / VSC ending
* ``session.race_control_messages`` -> sector specific yellow / double-yellow flags (marshal sector N),
                                       "TRACK CLEAR", "DRS ENABLED/DISABLED", "SAFETY CAR IN THIS LAP"

The track status feed only says "yellow somewhere" ('sectors are unknown'). The affected marshal
sectors only exist in the messages, so both feeds are needed. Marshal sector numbers are matched to
track nodes by TrackModel.marshal_sector.

Replay rules (events are replayed in time order, status events before messages on ties):
  * status '1' (all clear)            -> every sector flag is cleared, SC-in-this-lap is reset
  * Scope 'Sector' + YELLOW/DOUBLE YELLOW -> flag set for that marshal sector
  * Scope 'Sector' + CLEAR/GREEN          -> flag removed for that marshal sector
  * Scope 'Track'  + CLEAR                -> every sector flag is cleared
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from f1strategy.domain.sector import (
    STATUS_CLEAR,
    STATUS_RED,
    STATUS_SAFETY_CAR,
    STATUS_VSC,
    STATUS_VSC_ENDING,
    STATUS_YELLOW,
)


@dataclass(frozen=True)
class RaceControlState:
    """Race control situation at one moment."""

    time_s: float
    track_status: str = STATUS_CLEAR
    track_message: str = "AllClear"
    sector_flags: dict[int, str] = field(default_factory=dict)  # marshal sector -> 'YELLOW' | 'DOUBLE YELLOW'
    drs_enabled: bool | None = None  # None = no message feed at all (unknown); False until a message enables DRS
    sc_in_this_lap: bool = False

    @property
    def is_clear(self) -> bool:
        return self.track_status == STATUS_CLEAR and not self.sector_flags

    @property
    def is_yellow(self) -> bool:
        return self.track_status == STATUS_YELLOW or bool(self.sector_flags)

    @property
    def is_safety_car(self) -> bool:
        return self.track_status == STATUS_SAFETY_CAR

    @property
    def is_vsc(self) -> bool:
        return self.track_status == STATUS_VSC

    @property
    def is_vsc_ending(self) -> bool:
        return self.track_status == STATUS_VSC_ENDING

    @property
    def is_red(self) -> bool:
        return self.track_status == STATUS_RED

    @property
    def yellow_unlocalised(self) -> bool:
        """Track status says yellow but no marshal sector is known (messages missing)."""
        return self.track_status == STATUS_YELLOW and not self.sector_flags


def _naive_utc(ts: Any) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


class RaceControlTimeline:
    """Replays race control information into a bisect-searchable list of states."""

    def __init__(self, track_status: pd.DataFrame | None, messages: pd.DataFrame | None, t0_date: Any | None) -> None:
        events: list[tuple[float, int, str, Any]] = []  # (time_s, priority, kind, payload)

        if track_status is not None and len(track_status) > 0:
            t = pd.to_timedelta(track_status["Time"]).dt.total_seconds().to_numpy(float)
            for ti, code, msg in zip(
                t, track_status["Status"].astype(str), track_status["Message"].astype(str), strict=True
            ):
                events.append((float(ti), 0, "status", (code, msg)))

        # Is there a usable message feed? Whether *DRS* messages exist later in the session is deliberately not
        # consulted: that would be knowledge from the future. With a feed, DRS starts out disabled (as in every
        # race) until a message enables it; without any feed the DRS state is unknown (None).
        self.has_message_feed = messages is not None and len(messages) > 0 and t0_date is not None
        self.has_drs_messages = False  # informational: at least one DRS message was seen
        if self.has_message_feed:
            t0 = _naive_utc(t0_date)
            for row in messages.itertuples(index=False):
                if pd.isna(row.Time):
                    continue
                ti = (_naive_utc(row.Time) - t0).total_seconds()
                events.extend(self._message_events(ti, row))

        events.sort(key=lambda e: (e[0], e[1]))
        self._times: list[float] = []
        self._states: list[RaceControlState] = []
        self._replay(events)
        self._initial = RaceControlState(time_s=-np.inf, drs_enabled=False if self.has_message_feed else None)

    # ------------------------------------------------------------------ parsing
    def _message_events(self, ti: float, row: Any) -> list[tuple[float, int, str, Any]]:
        category = str(getattr(row, "Category", "") or "").upper()
        message = str(getattr(row, "Message", "") or "").upper()
        flag = str(getattr(row, "Flag", "") or "").upper()
        scope = str(getattr(row, "Scope", "") or "").upper()
        sector = getattr(row, "Sector", None)
        out: list[tuple[float, int, str, Any]] = []

        if category == "DRS" or message.startswith("DRS "):
            if "ENABLED" in message:
                out.append((ti, 1, "drs", True))
                self.has_drs_messages = True
            elif "DISABLED" in message:
                out.append((ti, 1, "drs", False))
                self.has_drs_messages = True
        elif category == "SAFETYCAR" and "IN THIS LAP" in message:
            out.append((ti, 1, "sc_in", True))
        elif category == "FLAG" or flag:
            if scope == "SECTOR" and sector is not None and not pd.isna(sector):
                if flag in ("YELLOW", "DOUBLE YELLOW"):
                    out.append((ti, 1, "sector", (int(sector), flag)))
                elif flag in ("CLEAR", "GREEN"):
                    out.append((ti, 1, "sector", (int(sector), None)))
            elif scope == "TRACK" and flag == "CLEAR":
                out.append((ti, 1, "clear_all", None))
        return out

    # ------------------------------------------------------------------ replay
    def _replay(self, events: list[tuple[float, int, str, Any]]) -> None:
        status, message = STATUS_CLEAR, "AllClear"
        sectors: dict[int, str] = {}
        drs: bool | None = False if self.has_message_feed else None
        sc_in = False
        for ti, _, kind, payload in events:
            if kind == "status":
                status, message = payload
                if status == STATUS_CLEAR:
                    sectors, sc_in = {}, False
            elif kind == "sector":
                sec, flag = payload
                if flag is None:
                    sectors.pop(sec, None)
                else:
                    sectors[sec] = flag
            elif kind == "clear_all":
                sectors = {}
            elif kind == "drs":
                drs = payload
            elif kind == "sc_in":
                sc_in = True
            self._times.append(ti)
            self._states.append(RaceControlState(ti, status, message, dict(sectors), drs, sc_in))

    # ------------------------------------------------------------------ queries
    def state_at(self, t_s: float) -> RaceControlState:
        """State in force at session time ``t_s`` (seconds)."""
        i = bisect.bisect_right(self._times, t_s) - 1
        return self._states[i] if i >= 0 else self._initial

    def to_frame(self) -> pd.DataFrame:
        """Debug view: one row per state change."""
        return pd.DataFrame(
            [
                {
                    "time_s": s.time_s,
                    "track_status": s.track_status,
                    "message": s.track_message,
                    "sector_flags": dict(s.sector_flags),
                    "drs_enabled": s.drs_enabled,
                    "sc_in_this_lap": s.sc_in_this_lap,
                }
                for s in self._states
            ]
        )

    @classmethod
    def from_session(cls, session: Any) -> RaceControlTimeline:
        def _get(name: str) -> pd.DataFrame | None:
            try:
                return getattr(session, name)
            except Exception:  # FastF1 raises DataNotLoadedError if the part was not loaded
                return None

        t0 = None
        try:
            t0 = session.t0_date
        except Exception:
            pass
        return cls(_get("track_status"), _get("race_control_messages"), t0)
