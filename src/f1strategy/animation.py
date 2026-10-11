"""
animation.py · Formula 1 Strategic Decisions
=========================================================
Animated representation of a :class:`~f1strategy.sequence.RaceSequence` (GIF).

It is drawn from the sequence arrays - the same numbers a model is trained on - not from a separate code path:
track nodes coloured by flag state (yellow / double yellow / SC / VSC / red), DRS zones, cars with a short trail,
a leaderboard (position, gap, tyre, pit), the lap and a flag banner.

Efficiency: everything static is drawn once; per frame only colours, offsets and texts are updated. Frames are
rendered with matplotlib's Agg canvas (no pyplot, no GUI) and written with Pillow using **one shared palette** for
the whole GIF, so the flag colours do not flicker from frame to frame.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from f1strategy.plotting import FLAG_COLORS
from f1strategy.sequence.dataset import FLAG_BITS, RaceSequence

log = logging.getLogger(__name__)

_COMPOUND_LETTER = {"soft": "S", "medium": "M", "hard": "H", "intermediate": "I", "wet": "W"}
_FALLBACK_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#808000", "#000075", "#e6beff", "#aaffc3", "#ffd8b1", "#a9a9a9", "#1b9e77",
]  # fmt: skip


def team_colors(session: Any, codes: Sequence[str]) -> dict[str, str]:
    """``code -> '#rrggbb'`` from ``session.results['TeamColor']`` where available, a fixed palette otherwise.

    Team mates share a colour (the labels tell them apart)."""
    colors = {c: _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i, c in enumerate(codes)}
    try:
        res = session.results
        for abbr, col in zip(res["Abbreviation"], res["TeamColor"], strict=True):
            col = str(col).lstrip("#")
            if abbr in colors and len(col) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in col):
                colors[abbr] = f"#{col}"
    except Exception:  # no results table / no TeamColor column (e.g. stand-ins, older data)
        log.debug("No team colours available, using the fallback palette")
    return colors


class RaceAnimator:
    """Renders frames of a ``RaceSequence`` and writes them as an animated GIF."""

    def __init__(
        self,
        sequence: RaceSequence,
        colors: Mapping[str, str] | None = None,
        trail: int = 6,
        figsize: tuple[float, float] = (10.0, 6.25),
        dpi: int = 72,
        title: str = "",
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import to_rgba
        from matplotlib.figure import Figure

        if sequence.n_frames == 0:
            raise ValueError("cannot animate an empty sequence")
        self.seq = sequence
        self.trail = max(0, int(trail))
        self.title = title
        codes = sequence.driver_codes
        base = {c: _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i, c in enumerate(codes)}
        self.colors = {**base, **(dict(colors) if colors else {})}
        self._car_rgba = np.array([to_rgba(self.colors[c]) for c in codes])

        names = sequence.driver_names
        self._ix = {n: names.index(n) for n in ("x_m", "y_m", "position", "gap_to_leader_s", "progress_m", "in_pit",
                                                "finished_or_retired", "speed_kmh")}  # fmt: skip
        self._compound_cols = [(names.index(f"compound_{k}"), v) for k, v in _COMPOUND_LETTER.items()]
        gnames = sequence.global_names
        self._g = {n: gnames.index(n) for n in ("leader_lap", "laps_remaining", "is_sc", "is_vsc", "is_vsc_ending",
                                                "is_red", "is_yellow", "drs_enabled")}  # fmt: skip
        self._flag_rgba = {k: np.array(to_rgba(v)) for k, v in FLAG_COLORS.items()}

        static = {n: sequence.track_static[:, j] for j, n in enumerate(sequence.track_static_names)}
        self._tx, self._ty = static["x_m"], static["y_m"]

        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        grid = self.fig.add_gridspec(1, 2, width_ratios=[3.1, 1.0], left=0.01, right=0.99, top=0.93, bottom=0.02,
                                     wspace=0.02)  # fmt: skip
        self.ax = self.fig.add_subplot(grid[0])
        self.board = self.fig.add_subplot(grid[1])
        self.board.axis("off")
        self._draw_static(static)

    # ------------------------------------------------------------------ static drawing
    def _draw_static(self, static: Mapping[str, np.ndarray]) -> None:
        ax = self.ax
        ax.set_aspect("equal")
        ax.axis("off")
        pad = 0.06 * max(np.ptp(self._tx), np.ptp(self._ty))
        ax.set_xlim(self._tx.min() - pad, self._tx.max() + pad)
        ax.set_ylim(self._ty.min() - pad, self._ty.max() + pad)

        self._nodes = ax.scatter(self._tx, self._ty, s=9, c=[self._flag_rgba["CLEAR"]] * len(self._tx), zorder=2,
                                 linewidths=0)  # fmt: skip
        drs = static["in_drs_zone"] > 0.5
        ax.scatter(
            self._tx[drs], self._ty[drs], s=42, facecolors="none", edgecolors="#2e7d32", linewidths=0.8, zorder=1
        )
        det = static["is_drs_detection"] > 0.5
        if det.any():
            ax.scatter(self._tx[det], self._ty[det], marker="D", s=26, c="#2e7d32", zorder=3)
        for k in np.flatnonzero(static["corner_number"] > 0):
            ax.annotate(str(int(static["corner_number"][k])), (self._tx[k], self._ty[k]), fontsize=6, color="#777",
                        xytext=(4, 4), textcoords="offset points")  # fmt: skip

        sector = np.argmax(np.stack([static["sector_1"], static["sector_2"], static["sector_3"]]), axis=0)
        marks = [(0, "S/F")] + [(int(np.argmax(sector == s)), lbl) for s, lbl in ((1, "S1|S2"), (2, "S2|S3"))
                                if (sector == s).any()]  # fmt: skip
        for k, label in marks:
            ax.annotate(label, (self._tx[k], self._ty[k]), fontsize=7, fontweight="bold", xytext=(-10, -11),
                        textcoords="offset points")  # fmt: skip

        D = self.seq.n_drivers
        self._cars = ax.scatter(np.zeros(D), np.zeros(D), s=62, c=self._car_rgba, edgecolors="white", linewidths=0.8,
                                zorder=6)  # fmt: skip
        self._trail = ax.scatter([], [], s=9, zorder=5, linewidths=0)
        self._labels = [ax.text(0, 0, c, fontsize=7, fontweight="bold", zorder=7, ha="left", va="bottom")
                        for c in self.seq.driver_codes]  # fmt: skip
        self._banner = ax.text(0.01, 0.99, "", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top",
                               bbox={"boxstyle": "round,pad=0.35", "fc": "#9aa0a6", "ec": "none"})  # fmt: skip
        self._drs = ax.text(0.01, 0.915, "", transform=ax.transAxes, fontsize=8, va="top", color="#2e7d32")
        self._title = self.fig.text(0.5, 0.965, "", ha="center", va="center", fontsize=11, fontweight="bold")
        pitch = 0.94 / 20  # fixed row height (sized for a full 20-car grid), independent of the number of cars
        self._rows = [self.board.text(0.02, 0.97 - r * pitch, "", family="monospace", fontsize=7.5,
                                      va="top", transform=self.board.transAxes) for r in range(D)]  # fmt: skip
        self.board.text(0.02, 1.0, " P  DRV    GAP  T", family="monospace", fontsize=7.5, va="bottom",
                        fontweight="bold", transform=self.board.transAxes)  # fmt: skip

    # ------------------------------------------------------------------ per-frame state
    def _node_colors(self, i: int) -> np.ndarray:
        bits = self.seq.track_flags[i]
        on = {n: ((bits >> k) & 1).astype(bool) for k, n in enumerate(FLAG_BITS)}
        order = [("RED", "flag_red"), ("SC", "flag_sc"), ("VSC ENDING", "flag_vsc_ending"), ("VSC", "flag_vsc"),
                 ("DOUBLE YELLOW", "flag_double_yellow"), ("YELLOW", "flag_yellow")]  # fmt: skip
        out = np.tile(self._flag_rgba["CLEAR"], (len(bits), 1))
        for label, flag in reversed(order):  # the most restrictive condition is painted last and wins
            out[on[flag]] = self._flag_rgba[label]
        return out

    def _banner_text(self, g: np.ndarray) -> tuple[str, str]:
        gi = self._g
        if g[gi["is_red"]] > 0.5:
            return "RED FLAG", "RED"
        if g[gi["is_sc"]] > 0.5:
            return "SAFETY CAR", "SC"
        if g[gi["is_vsc_ending"]] > 0.5:
            return "VSC ENDING", "VSC ENDING"
        if g[gi["is_vsc"]] > 0.5:
            return "VIRTUAL SAFETY CAR", "VSC"
        if g[gi["is_yellow"]] > 0.5:
            return "YELLOW FLAG", "YELLOW"
        return "GREEN FLAG", "CLEAR"

    def _gap_text(self, i: int, s: int, leader_progress: float) -> str:
        x, ix = self.seq.driver_x[i, s], self._ix
        if x[ix["position"]] == 1:
            return "LEADER"
        laps_down = int((leader_progress - x[ix["progress_m"]]) // self.seq.track_length_m)
        if laps_down >= 1:
            return f"+{laps_down}L"
        gap = x[ix["gap_to_leader_s"]]
        return f"+{gap:.1f}s" if self.seq.driver_mask[i, s, ix["gap_to_leader_s"]] else "  n/a"

    def update(self, i: int) -> None:
        """Move all dynamic artists to frame ``i``."""
        seq, ix = self.seq, self._ix
        self._nodes.set_facecolor(self._node_colors(i))
        present = seq.driver_present[i]
        xy = np.where(present[:, None], seq.driver_x[i][:, [ix["x_m"], ix["y_m"]]], np.nan)
        self._cars.set_offsets(xy)
        rank = {int(slot): r for r, slot in enumerate(seq.driver_order(i))}
        for s, text in enumerate(self._labels):
            text.set_visible(bool(present[s]))
            if present[s]:
                # alternate the side by race position so labels of cars running in a pack do not sit on top of each other
                up = rank[s] % 2 == 0
                text.set_position((xy[s, 0] + (24 if up else -24), xy[s, 1] + (24 if up else -24)))
                text.set_ha("left" if up else "right")
                text.set_va("bottom" if up else "top")

        if self.trail and i > 0:
            lo = max(0, i - self.trail)
            pts, cols = [], []
            for j in range(lo, i):
                age = (i - j) / (self.trail + 1)
                rgba = self._car_rgba.copy()
                rgba[:, 3] = 0.55 * (1 - age)
                ok = seq.driver_present[j]
                pts.append(seq.driver_x[j][ok][:, [ix["x_m"], ix["y_m"]]])
                cols.append(rgba[ok])
            self._trail.set_offsets(np.concatenate(pts) if pts else np.empty((0, 2)))
            self._trail.set_facecolor(np.concatenate(cols) if cols else np.empty((0, 4)))

        g = seq.global_x[i]
        text, label = self._banner_text(g)
        self._banner.set_text(text)
        self._banner.get_bbox_patch().set_facecolor(self._flag_rgba[label])
        self._banner.set_color("black" if label in ("CLEAR", "YELLOW", "DOUBLE YELLOW", "VSC ENDING") else "white")
        drs = g[self._g["drs_enabled"]]
        self._drs.set_text(
            "DRS enabled" if drs > 0.5 else ("DRS disabled" if seq.global_mask[i, self._g["drs_enabled"]] else "")
        )
        lap = int(g[self._g["leader_lap"]])
        rem = g[self._g["laps_remaining"]]
        total = f"/{lap + int(rem)}" if seq.global_mask[i, self._g["laps_remaining"]] else ""
        head = f"{self.title}   " if self.title else ""
        self._title.set_text(f"{head}Lap {lap}{total}   t = {seq.times[i]:.1f} s")

        order = seq.driver_order(i)
        leader_progress = seq.driver_x[i, order[0], ix["progress_m"]] if len(order) else 0.0
        for r, row in enumerate(self._rows):
            if r >= len(order):
                row.set_text("")
                continue
            s = order[r]
            x = seq.driver_x[i, s]
            tyre = next((letter for col, letter in self._compound_cols if x[col] > 0.5), "?")
            status = "OUT" if x[ix["finished_or_retired"]] > 0.5 else ("PIT" if x[ix["in_pit"]] > 0.5 else "")
            row.set_text(
                f"{r + 1:>2}  {seq.driver_codes[s]}  {self._gap_text(i, s, leader_progress):>7}  {tyre} {status}"
            )
            row.set_color(self._car_rgba[s])

    # ------------------------------------------------------------------ output
    def render(self, i: int) -> np.ndarray:
        """Frame ``i`` as an ``[H, W, 3]`` uint8 RGB array."""
        self.update(i)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()

    def save_png(self, i: int, path: str | Path) -> None:
        """Write a single frame as PNG."""
        from PIL import Image

        Image.fromarray(self.render(i)).save(path)

    def save_gif(
        self,
        path: str | Path,
        every: int = 1,
        fps: float = 10.0,
        max_frames: int = 800,
        palette_colors: int = 128,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Write the animation as a looping GIF.

        Args:
            every: use every n-th frame of the sequence (the playback speed is ``every * step_s * fps`` x real time).
            fps: frames per second of the GIF (GIF timing has 10 ms resolution; browsers clamp very fast GIFs).
            max_frames: refuse to build more frames than this (a GIF holds all frames in memory while writing);
                raise ``every`` or slice the sequence instead.
            palette_colors: size of the shared palette (<= 256).
        """
        from PIL import Image

        if every < 1 or fps <= 0 or not 2 <= palette_colors <= 256:
            raise ValueError("every must be >= 1, fps > 0 and palette_colors in [2, 256]")
        indices = list(range(0, self.seq.n_frames, every))
        if len(indices) > max_frames:
            raise ValueError(
                f"{len(indices)} frames exceed max_frames={max_frames}: use a larger `every` "
                f"(e.g. every={-(-self.seq.n_frames // max_frames)}), a shorter sequence, or raise max_frames"
            )

        # one palette for the whole GIF, from a few evenly spaced frames (colour changes of the flags included)
        picks = sorted({indices[round(k * (len(indices) - 1) / 5)] for k in range(6)}) if len(indices) > 1 else indices
        samples = [Image.fromarray(self.render(j)) for j in picks]
        sheet = Image.new("RGB", (samples[0].width, samples[0].height * len(samples)))
        for k, im in enumerate(samples):
            sheet.paste(im, (0, k * im.height))
        palette = sheet.quantize(colors=palette_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

        frames = []
        for n, j in enumerate(indices):
            frames.append(Image.fromarray(self.render(j)).quantize(palette=palette, dither=Image.Dither.NONE))
            if progress:
                progress(n + 1, len(indices))
        out = Path(path)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=max(1, round(1000 / fps)), loop=0,
                       optimize=False)  # fmt: skip
        log.info("Wrote %s (%d frames, %.1f MB)", out, len(frames), out.stat().st_size / 1e6)
        return out


def make_gif(
    sequence: RaceSequence,
    path: str | Path,
    *,
    colors: Mapping[str, str] | None = None,
    title: str = "",
    every: int = 1,
    fps: float = 10.0,
    **kwargs: Any,
) -> Path:
    """Shortcut: ``RaceAnimator(sequence, colors, title=title).save_gif(path, every=every, fps=fps, **kwargs)``."""
    return RaceAnimator(sequence, colors=colors, title=title).save_gif(path, every=every, fps=fps, **kwargs)
