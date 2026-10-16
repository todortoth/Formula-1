"""
generate_readme.py - renders README.md from docs/readme_template.md

The feature tables, the command line table and the storage-size table are generated from the code, so they cannot
drift from it; ``tests/unit/test_docs.py`` fails when README.md differs from what this script produces.

    python scripts/generate_readme.py            # rewrite README.md
    python scripts/generate_readme.py --check    # exit 1 if README.md is out of date
"""

from __future__ import annotations

import sys
from pathlib import Path

from f1strategy.cli import build_parser
from f1strategy.graph.exporter import (
    COMPOUNDS,
    DRIVER_FEATURES,
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    TRACK_FEATURES,
)
from f1strategy.sequence.dataset import FLAG_BITS, TRACK_DYNAMIC
from f1strategy.sequence.sampling import DEFAULT_CHUNK_FRAMES

ROOT = Path(__file__).resolve().parents[1]

T = {
    "s_m": "Distance from the start/finish line along the lap, in metres.",
    "x_m": "World x of the node (frame of the FastF1 position feed), metres.",
    "y_m": "World y of the node, metres.",
    "z_m": "Elevation of the node, metres.",
    "heading_sin": "Direction of the track tangent, `sin(atan2(dy, dx))`.",
    "heading_cos": "Direction of the track tangent, `cos(atan2(dy, dx))`.",
    "curvature": "Signed curvature in 1/m, positive = left turn (from a ~12 m smoothed centre line).",
    "gradient": "Slope `dz/ds` (metre per metre).",
    "sector_1": "One-hot official timing sector 1 (boundaries come from the reference lap's sector times).",
    "sector_2": "One-hot official timing sector 2.",
    "sector_3": "One-hot official timing sector 3.",
    "is_corner": "1 on the node closest to a numbered corner marker.",
    "corner_number": "Corner number of that marker (0 = none).",
    "marshal_sector": "Marshal sector the node lies in (-1 = unknown) - the unit in which sector yellow flags are issued.",
    "in_drs_zone": "Node is inside an inferred DRS activation zone.",
    "is_drs_start": "First node of a DRS zone (the activation line).",
    "is_drs_end": "Last node of a DRS zone.",
    "is_drs_detection": "Nearest node to a DRS detection point (only when you supply them, see Limitations).",
    "flag_yellow": "Yellow flag in this node's marshal sector.",
    "flag_double_yellow": "Double yellow in this node's marshal sector.",
    "flag_sc": "Safety car deployed (applies to every node).",
    "flag_vsc": "Virtual safety car deployed (every node).",
    "flag_vsc_ending": "VSC ending (every node).",
    "flag_red": "Red flag (every node).",
    "n_drivers": "Number of cars whose nearest track node is this one.",
}
D = {
    "x_m": "Car position at the snapshot time (interpolated), metres.",
    "y_m": "Car position y, metres.",
    "z_m": "Car elevation, metres.",
    "speed_kmh": "Speed from the car data channel, interpolated.",
    "vx_mps": "Velocity x from the position feed (central difference over +-0.3 s), m/s.",
    "vy_mps": "Velocity y, m/s.",
    "heading_sin": "Direction of travel `sin(atan2(vy, vx))`; below 2 m/s the track tangent is used instead.",
    "heading_cos": "Direction of travel, cosine.",
    "heading_rel_track_sin": "Heading relative to the track tangent, sine (about 0 when aligned; large when spinning or in the pit lane).",
    "heading_rel_track_cos": "Heading relative to the track tangent, cosine.",
    "throttle": "Throttle in percent.",
    "brake": "Brake on (0/1).",
    "gear": "Gear.",
    "rpm": "Engine revs.",
    "drs_open": "DRS flap open (raw DRS channel 10, 12 or 14).",
    "drs_eligible": 'Raw DRS channel equals 8 ("detected / eligible"; semantics are not officially documented).',
    "track_s_m": "Distance from the start/finish line of the car's projection onto the centre line, metres.",
    "lap_fraction": "`track_s_m` divided by the track length (jumps from 1 to 0 at the start/finish line).",
    "lap_fraction_sin": "`sin(2 pi lap_fraction)`: the wrap-safe encoding of the lap position, continuous across the line.",
    "lap_fraction_cos": "`cos(2 pi lap_fraction)`.",
    "lateral_offset_m": "Signed distance from the reference line, metres (+ = left).",
    "progress_m": "Cumulative race distance (laps x length + position); 0 at the line at lights-out, grid slots are slightly negative.",
    "lap_number": "Lap the car is on, from the timing table.",
    "position": "Live race position, ranked by `progress_m`.",
    "official_position": "Classified position at the end of the car's last completed lap (NaN before the first).",
    "gap_to_leader_s": "True time gap to the leader: how long ago was the leader where this car is now.",
    "gap_ahead_s": "Same, to the car directly ahead (NaN for the leader).",
    "gap_behind_s": "Same, of the car directly behind to this car (NaN for the last car).",
    "tyre_age": "Laps on the current set (FastF1 `TyreLife`).",
    "fresh_tyre": "Current set was new when fitted.",
    "stint": "Stint number.",
    "pit_stops": "Pit lane entries so far (a pure count of entries up to the snapshot time).",
    "in_pit": "Car is in the pit lane right now (between pit entry and the next lap's pit exit).",
    "on_track": "Position feed status is not `OffTrack`.",
    "pos_age_s": "Seconds since the newest *real* position sample behind this row (0 exactly at a sample, up to ~0.22 s; NaN before the first sample, grows if the stream stops).",
    "car_age_s": "Same for the car-data stream (speed, throttle, ...; ~0.24 s interval).",
    "pace_last3_s": "Mean of the last three completed representative laps (no in/out laps, none with SC/VSC/red), seconds.",
    "last_lap_time_s": "Time of the last completed lap, seconds.",
    "finished_or_retired": "More than 5 s have passed since the car's final lap timestamp.",
    **{f"compound_{c.lower()}": f"One-hot: current tyre compound is {c}." for c in COMPOUNDS},
    "compound_unknown": "One-hot: compound unknown / not in the list above.",
}
G = {
    "time_s": "Snapshot time in session seconds.",
    "leader_lap": "Lap of the race leader.",
    "laps_remaining": "Scheduled laps minus leader lap (NaN if unknown).",
    "race_fraction": "Leader lap divided by scheduled laps.",
    "is_green": "No flag anywhere: track status clear and no sector flags.",
    "is_yellow": "Yellow somewhere: track status yellow or any sector flagged.",
    "is_sc": "Safety car deployed.",
    "is_vsc": "Virtual safety car deployed.",
    "is_vsc_ending": "VSC ending.",
    "is_red": "Red flag.",
    "sc_in_this_lap": 'Race control announced "SAFETY CAR IN THIS LAP" (reset at all-clear).',
    "yellow_unlocalised": "Track status says yellow but no marshal sector is known (messages missing).",
    "n_yellow_sectors": "Number of marshal sectors currently under yellow / double yellow.",
    "drs_enabled": "Race control has enabled DRS. Disabled until the first enabling message; unknown (masked) if there is no message feed at all.",
    "air_temp": "Air temperature, deg C.",
    "track_temp": "Track temperature, deg C.",
    "humidity": "Relative humidity, percent.",
    "pressure": "Air pressure, mbar.",
    "rainfall": "Rain (0/1).",
    "wind_speed": "Wind speed, m/s.",
    "wind_direction_sin": "Wind direction, sine.",
    "wind_direction_cos": "Wind direction, cosine.",
    "n_drivers": "Number of cars in the snapshot.",
}
E = {
    "gap_s": "Time gap seen from the source car; **positive = the target is ahead**. NaN when undefined (e.g. a lap apart, never reached that point).",
    "progress_gap_m": "Difference in cumulative race distance, target minus source, metres.",
    "circular_gap_m": "Shortest signed distance along the track ignoring laps, in (-L/2, L/2]; this is what connects lapped traffic.",
    "lap_diff": "Whole laps the target is ahead of the source (negative = behind).",
    "rel_speed_kmh": "Target speed minus source speed.",
    "order_neighbor": "The two cars are consecutive in the running order.",
    "drs_window": "Source is the follower, within 1.0 s of the target and on the same lap (approximates the DRS rule with the true gap).",
}
for name, feats, desc in (
    ("track", TRACK_FEATURES, T),
    ("driver", DRIVER_FEATURES, D),
    ("global", GLOBAL_FEATURES, G),
    ("edge", EDGE_FEATURES, E),
):
    missing = [f for f in feats if f not in desc]
    extra = [k for k in desc if k not in feats]
    assert not missing and not extra, (name, missing, extra)


def table(feats, desc):
    return "| Feature | Meaning |\n|---|---|\n" + "\n".join(f"| `{f}` | {desc[f]} |" for f in feats)


CLI = {
    "--year": ("2025", "Season."),
    "--event": ("`Belgian Grand Prix`", "Event name (fuzzy-matched by FastF1) or, if all digits, the round number."),
    "--session": ("`R`", "Session identifier: `R`, `Q`, `S`, `SQ`, `FP1`..."),
    "--lap": (
        "44",
        "Lap whose sector crossings define the sector snapshots (taken when the **leader** crosses the sector line).",
    ),
    "--sectors": ("all three", 'Any of `"Sector 1"` `"Sector 2"` `"Sector 3"`.'),
    "--out-dir": (
        "`graphs`",
        "Where PNGs and `.npz` files go; in continuous mode also the dataset and GIF (`sequence_lap<from>-<to>`).",
    ),
    "--spacing": ("10", "Track node spacing in metres."),
    "--max-tokens": ("512", "Token budget used to choose the track pooling factor."),
    "--drs-detection-offset": ("none", "Estimate DRS detection points as *zone start minus this many metres*."),
    "--cache-dir": ("`$F1_CACHE_DIR` or `~/fastf1_cache`", "FastF1 cache directory."),
    "--continuous": (
        "off",
        "**Continuous mode**: also sample the race graph densely over a lap window, save it and render a GIF.",
    ),
    "--from-lap": ("`--lap`", "First lap of the continuous window (starts when the leader starts it)."),
    "--to-lap": (
        "`--from-lap`",
        "Last lap of the window (ends when the leader completes it); above the session's last lap it is clamped, so `99` means the whole race.",
    ),
    "--step": ("1.0", "Seconds between two frames (or give `--hz`)."),
    "--hz": (
        "none",
        "Frames per second of *data*, e.g. `10`; an alternative to `--step` (not the GIF rate: that is `--fps`).",
    ),
    "--format": (
        "`parquet`",
        "Dataset format: `parquet` (a directory, streamed in chunks), `npz` (one file, in memory) or `both`.",
    ),
    "--chunk-frames": (
        str(DEFAULT_CHUNK_FRAMES),
        "Frames per Parquet part / in-memory chunk; this is what bounds the memory use.",
    ),
    "--gif-every": ("auto", "Use every n-th frame in the GIF; automatic means at most 300 frames."),
    "--fps": ("10", "Playback frames per second of the GIF."),
    "--no-gif": ("off", "Continuous mode: write only the dataset, no GIF."),
    "--no-snapshots": ("off", "Skip the sector snapshots (tables, plots, npz); needs `--continuous`."),
    "--no-tables": ("off", "Skip the console tables (sector snapshots)."),
    "--no-plots": ("off", "Skip the PNG plots."),
    "--no-export": ("off", "Skip the `.npz` export."),
    "--verbose": ("off", "Debug logging (`-v`)."),
}
parser_flags = {o for a in build_parser()._actions for o in a.option_strings if o.startswith("--") and o != "--help"}
assert parser_flags == set(CLI), parser_flags ^ set(CLI)
cli_table = "| Option | Default | Meaning |\n|---|---|---|\n" + "\n".join(
    f"| `{k}` | {v[0]} | {v[1]} |" for k, v in CLI.items()
)


def size_table(frames: int = 5400, drivers: int = 20, nodes: int = 695) -> tuple[str, float, float]:
    """Uncompressed in-memory size of a RaceSequence for a 90 min race at 1 frame / s (Spa-like node count)."""
    fd, fg, fe = len(DRIVER_FEATURES), len(GLOBAL_FEATURES), len(EDGE_FEATURES)
    rows = [
        ("`track_flags`", "`[T, N]` uint8", frames * nodes),
        ("`driver_x` + `driver_mask`", f"`[T, D, {fd}]` float32 + bool", frames * drivers * fd * 5),
        (
            "`edge_attr` + `edge_attr_mask` + `edge_exists`",
            f"`[T, D, D, {fe}]` float32 + bool, `[T, D, D]` bool",
            frames * drivers * drivers * (fe * 5 + 1),
        ),
        ("`global_x` + `global_mask`", f"`[T, {fg}]`", frames * fg * 5),
        ("`driver_to_track`, `driver_present`, `times`", "`[T, D]`, `[T]`", frames * drivers * 5 + frames * 8),
    ]
    total = sum(r[2] for r in rows)
    lines = ["| Array | Shape / dtype | Size |", "|---|---|---|"]
    lines += [f"| {name} | {shape} | {size / 1e6:.1f} MB |" for name, shape, size in rows]
    lines.append(f"| **Total** | T = {frames}, D = {drivers}, N = {nodes} | **{total / 1e6:.0f} MB** |")
    naive = frames * nodes * len(TRACK_FEATURES) * 5 / 1e6
    return "\n".join(lines), naive, total / 1e6


def render() -> str:
    """The README text as it should be, given the template and the current code."""
    text = (ROOT / "docs" / "readme_template.md").read_text(encoding="utf-8")
    sizes, naive, _ = size_table()
    _, _, total_10hz = size_table(frames=54000)
    values = {
        "TRACK": table(TRACK_FEATURES, T),
        "DRIVER": table(DRIVER_FEATURES, D),
        "GLOBAL": table(GLOBAL_FEATURES, G),
        "EDGE": table(EDGE_FEATURES, E),
        "CLI": cli_table,
        "SIZES": sizes,
        "NAIVE_TRACK_MB": f"{naive:.0f}",
        "DENSE_10HZ_MB": f"{total_10hz / 1000:.1f} GB",
        "DRIVER_ROWS_10HZ": f"{54000 * 20:,}",
        "N_DRIVER_FEATURES": str(len(DRIVER_FEATURES)),
        "CHUNK_FRAMES": str(DEFAULT_CHUNK_FRAMES),
        "FLAG_BITS": ", ".join(f"`{n}`" for n in FLAG_BITS),
        "TRACK_DYNAMIC": ", ".join(f"`{n}`" for n in TRACK_DYNAMIC),
    }
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", val)
    leftover = [w for w in text.split() if w.startswith("{{")]
    assert not leftover, f"unreplaced placeholders: {leftover}"
    return text


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    readme = ROOT / "README.md"
    new = render()
    if "--check" in argv:
        if readme.read_text(encoding="utf-8") != new:
            print("README.md is out of date: run `python scripts/generate_readme.py`", file=sys.stderr)
            return 1
        return 0
    readme.write_text(new, encoding="utf-8", newline="\n")
    print(f"README.md written ({len(new.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
