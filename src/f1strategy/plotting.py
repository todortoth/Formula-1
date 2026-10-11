"""
plotting.py · Formula 1 Strategic Decisions
=========================================================
Draws a race graph on the *real* track layout (X/Y in metres) instead of an abstract spring layout.

Uses ``matplotlib.figure.Figure`` directly (no pyplot, no global state, no GUI backend needed), so it works
headless, e.g. in CI, and is safe to call repeatedly.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

FLAG_COLORS = {
    "CLEAR": "#9aa0a6",
    "YELLOW": "#f2c200",
    "DOUBLE YELLOW": "#ff8c00",
    "SC": "#6d4c41",
    "VSC": "#8e24aa",
    "VSC ENDING": "#ce93d8",
    "RED": "#d32f2f",
}


def plot_race_graph(G: nx.DiGraph, filename: str, title: str = "") -> None:
    """Save a track map of ``G`` to ``filename``.

    Shows: track nodes coloured by flag state, DRS zones (green rings) and detection points (diamonds),
    sector borders, corner numbers, cars with their heading, and the proximity edges between cars.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    track = G.graph["track"]
    tn = sorted((d for _, d in G.nodes(data=True) if d.get("node_type") == "track"), key=lambda d: d["index"])
    fig = Figure(figsize=(12, 9))
    ax = fig.subplots()

    xs, ys = np.array([d["x_m"] for d in tn]), np.array([d["y_m"] for d in tn])
    ax.scatter(xs, ys, c=[FLAG_COLORS[d["flag_label"]] for d in tn], s=14, zorder=2)
    drs = np.array([d["drs_zone"] > 0 for d in tn])
    ax.scatter(xs[drs], ys[drs], s=60, facecolors="none", edgecolors="#2e7d32", linewidths=1.0, zorder=1)
    for d in tn:
        if d["is_drs_detection"]:
            ax.scatter(d["x_m"], d["y_m"], marker="D", c="#2e7d32", s=45, zorder=3)
        if d["corner_number"]:
            ax.annotate(
                str(d["corner_number"]),
                (d["x_m"], d["y_m"]),
                fontsize=8,
                color="#555",
                xytext=(6, 6),
                textcoords="offset points",
            )

    markers = [(int(track.node_at_s(0.0)), "S/F")]
    if track.sector_end_s:
        markers += [
            (int(track.node_at_s(s)), name) for s, name in zip(track.sector_end_s, ("S1|S2", "S2|S3"), strict=True)
        ]
    for k, name in markers:
        ax.annotate(name, (xs[k], ys[k]), fontsize=9, fontweight="bold", xytext=(-14, -14), textcoords="offset points")

    for u, v, e in G.edges(data=True):
        if e.get("edge_type") == "proximity" and u < v:
            a, b = G.nodes[u], G.nodes[v]
            ax.plot([a["x_m"], b["x_m"]], [a["y_m"], b["y_m"]], color="#1565c0", lw=1, alpha=0.5, zorder=4)
    for code in G.graph["driver_codes"]:
        d = G.nodes[code]
        ax.scatter(d["x_m"], d["y_m"], c="#1565c0", s=90, zorder=5, edgecolors="white")
        ax.arrow(
            d["x_m"],
            d["y_m"],
            60 * np.cos(d["heading_rad"]),
            60 * np.sin(d["heading_rad"]),
            head_width=18,
            color="#0d47a1",
            zorder=6,
            length_includes_head=True,
        )
        ax.annotate(
            f"P{d['position']} {code}\n{d['speed_kmh']:.0f} km/h",
            (d["x_m"], d["y_m"]),
            fontsize=8,
            xytext=(8, 8),
            textcoords="offset points",
            zorder=7,
        )

    handles = [Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in FLAG_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", title="Track flag state", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.savefig(filename, dpi=130, bbox_inches="tight")
