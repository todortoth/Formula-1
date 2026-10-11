"""plot_race_graph: writes a real PNG, reflects the flag state, leaves no global matplotlib state behind."""

from __future__ import annotations

import hashlib

import pytest

from f1strategy.plotting import FLAG_COLORS, plot_race_graph

PNG = b"\x89PNG\r\n\x1a\n"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_writes_a_png_image(G_sc, tmp_path):
    out = tmp_path / "g.png"
    plot_race_graph(G_sc, str(out), "title")
    data = out.read_bytes()
    assert data.startswith(PNG) and len(data) > 20_000


def test_different_flag_states_give_different_pictures(analyser, tmp_path):
    paths = {}
    for name, t in {"green": 100.0, "yellow": 121.0, "sc": 184.0, "vsc": 340.0}.items():
        paths[name] = tmp_path / f"{name}.png"
        plot_race_graph(analyser.build_graph_at_time(t), str(paths[name]), name)
    assert len({_digest(p) for p in paths.values()}) == 4


def test_title_is_optional(G_sc, tmp_path):
    plot_race_graph(G_sc, str(tmp_path / "x.png"))
    assert (tmp_path / "x.png").stat().st_size > 0


def test_works_without_sector_boundaries_or_drs_info(analyser, tmp_path, monkeypatch):
    monkeypatch.setattr(analyser.track, "sector_end_s", None)
    plot_race_graph(analyser.build_graph_at_time(100.0), str(tmp_path / "x.png"))
    assert (tmp_path / "x.png").read_bytes().startswith(PNG)


def test_leaves_no_open_pyplot_figures(G_sc, tmp_path):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    for i in range(3):
        plot_race_graph(G_sc, str(tmp_path / f"{i}.png"))
    assert plt.get_fignums() == before


def test_every_flag_label_the_builder_can_emit_has_a_colour(analyser):
    labels = set()
    for t in (100.0, 110.0, 121.0, 184.0, 340.0, 356.0):
        G = analyser.build_graph_at_time(t)
        labels |= {d["flag_label"] for _, d in G.nodes(data=True) if d["node_type"] == "track"}
    assert labels <= set(FLAG_COLORS) and {"CLEAR", "YELLOW", "DOUBLE YELLOW", "SC", "VSC", "VSC ENDING"} <= labels


def test_unwritable_path_raises(G_sc, tmp_path):
    with pytest.raises(OSError):
        plot_race_graph(G_sc, str(tmp_path / "missing_dir" / "x.png"))
