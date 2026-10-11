"""GraphExporter / GraphArrays: schema, encodings, pooling, normalisation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import numpy as np
import pytest

from f1strategy import GraphExporter, RaceGraphBuilder, SessionAnalyser, suggest_pool_factor
from f1strategy.graph.exporter import (
    COMPOUNDS,
    DRIVER_FEATURES,
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    TRACK_FEATURES,
    GraphArrays,
)
from tests.helpers import clone_session


class TestExporter:
    def test_shapes_and_schema(self, arrays, analyser):
        n = analyser.track.n
        assert arrays.track_x.shape == (n, len(TRACK_FEATURES)) == arrays.track_mask.shape
        assert arrays.driver_x.shape == (6, len(DRIVER_FEATURES)) == arrays.driver_mask.shape
        assert arrays.global_x.shape == (1, len(GLOBAL_FEATURES))
        assert arrays.driver_edge_attr.shape[1] == len(EDGE_FEATURES)
        assert arrays.track_edge_index.shape == (2, 2 * n)
        assert (
            arrays.driver_edge_index.max() < 6 and arrays.driver_edge_index.shape[1] == arrays.driver_edge_attr.shape[0]
        )
        assert arrays.n_tokens == 1 + 6 + n
        for a in (arrays.track_x, arrays.driver_x, arrays.global_x, arrays.driver_edge_attr):
            assert a.dtype == np.float32 and np.isfinite(a).all()

    def test_encodings(self, arrays):
        names = arrays.driver_names
        onehot = arrays.driver_x[
            :, [names.index(f"compound_{c.lower()}") for c in COMPOUNDS] + [names.index("compound_unknown")]
        ]
        assert (onehot.sum(1) == 1).all()
        order = arrays.driver_codes  # VER MEDIUM, HAM SOFT, SAI HARD at this moment
        assert arrays.driver_x[order.index("VER"), names.index("compound_medium")] == 1
        assert arrays.driver_x[order.index("HAM"), names.index("compound_soft")] == 1
        assert arrays.driver_x[order.index("SAI"), names.index("compound_hard")] == 1
        tn = arrays.track_names
        assert (arrays.track_x[:, [tn.index("sector_1"), tn.index("sector_2"), tn.index("sector_3")]].sum(1) == 1).all()
        m = arrays.driver_mask
        sin, cos = arrays.driver_x[:, names.index("heading_sin")], arrays.driver_x[:, names.index("heading_cos")]
        ok = m[:, names.index("heading_sin")]
        assert np.allclose(sin[ok] ** 2 + cos[ok] ** 2, 1.0, atol=1e-5)
        assert (
            not m[0, names.index("gap_ahead_s")] and m[1, names.index("gap_ahead_s")]
        )  # leader has no car ahead -> masked
        assert arrays.driver_x[0, names.index("gap_ahead_s")] == 0.0

    def test_flags_and_values(self, arrays):
        tn = arrays.track_names
        assert (arrays.track_x[:, tn.index("flag_sc")] == 1).all() and (
            arrays.track_x[:, tn.index("flag_yellow")] == 0
        ).all()
        assert arrays.global_x[0, arrays.global_names.index("is_sc")] == 1
        assert arrays.global_x[0, arrays.global_names.index("drs_enabled")] == 0
        assert arrays.track_x[:, tn.index("n_drivers")].sum() == 6
        assert (arrays.driver_to_track >= 0).all() and arrays.driver_to_track.max() < len(arrays.track_x)

    def test_pooling_keeps_events_and_driver_links(self, analyser):
        G = analyser.build_graph_at_time(121.0)
        a = GraphExporter().export(G)
        factor = suggest_pool_factor(len(a.track_x), 6, max_tokens=128)
        assert factor == 5 and suggest_pool_factor(500, 6, 512) == 1
        p = a.pool_track(factor)
        assert p.n_tokens <= 128 and len(p.track_x) == 100 and p.pool_factor == 5
        tn = a.track_names
        assert p.track_x[:, tn.index("n_drivers")].sum() == a.track_x[:, tn.index("n_drivers")].sum()
        assert (
            p.track_x[:, tn.index("flag_double_yellow")].sum() > 0 and p.track_x[:, tn.index("flag_yellow")].sum() > 0
        )
        assert p.track_x[:, tn.index("is_drs_start")].sum() == a.track_x[:, tn.index("is_drs_start")].sum() == 2
        assert (p.driver_to_track == a.driver_to_track // 5).all()
        assert p.track_spacing_m == pytest.approx(a.track_spacing_m * 5)
        assert p.track_edge_index.max() == 99 and a.pool_track(1) is a
        with pytest.raises(ValueError):
            suggest_pool_factor(500, 600)

    def test_token_layout(self, arrays):
        lay = arrays.token_layout()
        assert len(lay["token_type_ids"]) == len(lay["position_ids"]) == arrays.n_tokens
        assert list(lay["token_type_ids"][:8]) == [0] + [1] * 6 + [2]

    def test_normalisation_bounds(self, analyser):
        a = GraphExporter(normalise=True).export(analyser.build_session_graph(3, "Sector 2"))
        tn, dn = a.track_names, a.driver_names
        assert np.abs(a.track_x[:, [tn.index("x_m"), tn.index("y_m")]]).max() <= 1.0 + 1e-5
        assert np.abs(a.driver_x[:, [dn.index("x_m"), dn.index("y_m")]]).max() <= 1.1
        assert 0 <= a.driver_x[:, dn.index("speed_kmh")].max() <= 1.0
        assert a.track_x[:, tn.index("s_m")].max() < 1.0

    def test_serialisable(self, arrays):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.npz")
            np.savez_compressed(path, **arrays.to_dict())
            with np.load(path) as back:
                assert back["track_x"].shape == arrays.track_x.shape and float(back["time_s"]) == pytest.approx(
                    arrays.time_s
                )

    def test_sequence_of_snapshots_for_training(self, analyser):
        """The whole point: many snapshots with an identical schema."""
        shapes = {
            GraphExporter().export(analyser.build_graph_at_time(t)).driver_x.shape for t in np.linspace(40, 370, 12)
        }
        assert shapes == {(6, len(DRIVER_FEATURES))}


class TestExporterRegressionsAndEdges:
    def test_regression_pooled_mask_has_the_pooled_shape(self, analyser):
        """Found on a real Spa export: track_x was pooled (348 rows) but track_mask kept all 695 rows."""
        a = GraphExporter().export(analyser.build_graph_at_time(121.0))
        p = a.pool_track(5)
        assert p.track_mask.shape == p.track_x.shape == (100, len(a.track_names))
        assert p.track_mask.dtype == bool and p.driver_to_track.max() < len(p.track_x)

    def test_pooling_with_a_factor_that_does_not_divide_the_track(self, arrays):
        n = len(arrays.track_x)
        p = arrays.pool_track(7)
        assert len(p.track_x) == -(-n // 7) and p.track_mask.shape == p.track_x.shape
        assert p.track_x[:, arrays.track_names.index("n_drivers")].sum() == 6  # nothing lost in the ragged last group

    def test_pooling_to_a_single_token(self, arrays):
        p = arrays.pool_track(10_000)
        assert len(p.track_x) == 1 and p.track_x[0, arrays.track_names.index("n_drivers")] == 6
        assert (p.driver_to_track == 0).all()

    @pytest.mark.parametrize("factor", [0, 1, -3])
    def test_non_positive_or_unit_factor_is_a_no_op(self, arrays, factor):
        assert arrays.pool_track(factor) is arrays

    def test_pooling_twice_multiplies_the_factor(self, arrays):
        p = arrays.pool_track(2).pool_track(5)
        assert p.pool_factor == 10 and p.track_spacing_m == pytest.approx(arrays.track_spacing_m * 10)
        assert len(p.track_x) == 50

    def test_pooling_does_not_modify_the_original(self, arrays):
        before = arrays.track_x.copy()
        arrays.pool_track(5)
        assert np.array_equal(arrays.track_x, before)

    def test_pooled_flag_and_continuous_semantics(self, analyser):
        a = GraphExporter().export(analyser.build_graph_at_time(184.0))  # safety car: every node flagged
        p = a.pool_track(5)
        tn = a.track_names
        assert (p.track_x[:, tn.index("flag_sc")] == 1).all()
        # continuous features are averaged: pooled s_m is the group's mean
        assert p.track_x[0, tn.index("s_m")] == pytest.approx(a.track_x[:5, tn.index("s_m")].mean(), rel=1e-4)

    def test_validate_accepts_good_arrays_and_returns_self(self, arrays):
        assert arrays.validate() is arrays

    @pytest.mark.parametrize(
        "field, bad",
        [
            ("track_mask", lambda a: a.track_mask[:-1]),
            ("driver_mask", lambda a: a.driver_mask[:, :-1]),
            ("global_x", lambda a: np.zeros((2, a.global_x.shape[1]), np.float32)),
            ("driver_edge_attr", lambda a: a.driver_edge_attr[:, :-1]),
            ("driver_edge_index", lambda a: a.driver_edge_index + 100),
            ("track_edge_index", lambda a: a.track_edge_index + 10_000),
            ("driver_to_track", lambda a: a.driver_to_track + 10_000),
            ("driver_x", lambda a: a.driver_x[:-1]),
        ],
    )
    def test_validate_rejects_inconsistent_arrays(self, arrays, field, bad):
        with pytest.raises(ValueError, match="inconsistent GraphArrays"):
            replace(arrays, **{field: bad(arrays)}).validate()

    def test_progress_is_normalised_by_the_race_distance(self, analyser):
        a = GraphExporter(normalise=True).export(analyser.build_graph_at_time(300.0))
        j = a.driver_names.index("progress_m")
        assert (a.driver_x[:, j] > 0).all() and (a.driver_x[:, j] <= 1.0).all()  # lap ~5 of 6 -> ~0.8, not a lap count
        raw = GraphExporter().export(analyser.build_graph_at_time(300.0))
        L = raw.track_length_m
        assert a.driver_x[0, j] == pytest.approx(raw.driver_x[0, j] / (6 * L), rel=1e-4)

    def test_progress_normalisation_without_total_laps_uses_a_default(self, syn):
        sess = clone_session(syn.session, total_laps=None)
        G = SessionAnalyser.from_session(sess).build_graph_at_time(300.0)
        a = GraphExporter(normalise=True).export(G)
        assert a.driver_x[:, a.driver_names.index("progress_m")].max() < 1.0

    def test_normalised_and_raw_exports_share_masks_and_shapes(self, analyser):
        G = analyser.build_graph_at_time(150.0)
        raw, norm = GraphExporter().export(G), GraphExporter(normalise=True).export(G)
        assert raw.driver_x.shape == norm.driver_x.shape and np.array_equal(raw.driver_mask, norm.driver_mask)
        assert np.array_equal(raw.track_mask, norm.track_mask)
        # categorical / flag columns are untouched by normalisation
        j = raw.driver_names.index("compound_medium")
        assert np.array_equal(raw.driver_x[:, j], norm.driver_x[:, j])

    def test_missing_values_are_zero_filled_and_masked(self, arrays):
        for x, m in ((arrays.driver_x, arrays.driver_mask), (arrays.track_x, arrays.track_mask),
                     (arrays.global_x, arrays.global_mask), (arrays.driver_edge_attr, arrays.driver_edge_mask)):  # fmt: skip
            assert np.isfinite(x).all() and (x[~m] == 0).all()

    def test_unknown_compound_gets_its_own_one_hot(self, analyser):
        G = analyser.build_graph_at_time(150.0)
        G.nodes["VER"]["compound"] = "TEST_UNKNOWN"
        a = GraphExporter().export(G)
        row = a.driver_x[a.driver_codes.index("VER")]
        assert row[a.driver_names.index("compound_unknown")] == 1
        assert sum(row[a.driver_names.index(f"compound_{c.lower()}")] for c in COMPOUNDS) == 0

    @pytest.mark.parametrize("compound", COMPOUNDS)
    def test_each_known_compound_sets_exactly_one_flag(self, analyser, compound):
        G = analyser.build_graph_at_time(150.0)
        G.nodes["VER"]["compound"] = compound.lower()  # case-insensitive
        a = GraphExporter().export(G)
        row = a.driver_x[a.driver_codes.index("VER")]
        flags = {c: row[a.driver_names.index(f"compound_{c.lower()}")] for c in COMPOUNDS}
        assert flags[compound] == 1 and sum(flags.values()) == 1 and row[a.driver_names.index("compound_unknown")] == 0

    def test_nan_heading_is_masked_in_all_four_angle_columns(self, analyser):
        G = analyser.build_graph_at_time(150.0)
        G.nodes["LEC"]["heading_rad"] = float("nan")
        G.nodes["LEC"]["heading_rel_track_rad"] = float("nan")
        a = GraphExporter().export(G)
        i = a.driver_codes.index("LEC")
        for name in ("heading_sin", "heading_cos", "heading_rel_track_sin", "heading_rel_track_cos"):
            assert not a.driver_mask[i, a.driver_names.index(name)]

    def test_no_proximity_edges_gives_empty_but_well_shaped_edge_arrays(self, analyser):
        G = RaceGraphBuilder(analyser.session, analyser.track, analyser.telemetry, analyser.timing, analyser.timeline,
                             analyser._driver_map, proximity_window_s=0.0, proximity_window_m=0.0).build(150.0)  # fmt: skip
        a = GraphExporter().export(G)
        assert a.driver_edge_index.shape == (2, 0) and a.driver_edge_attr.shape == (0, len(EDGE_FEATURES))
        a.validate()

    def test_edge_indices_refer_to_driver_rows_and_match_graph_edges(self, G_sc, arrays):
        n_edges = sum(1 for *_, e in G_sc.edges(data=True) if e["edge_type"] == "proximity")
        assert arrays.driver_edge_index.shape[1] == n_edges
        i = arrays.driver_edge_index[:, 0]
        u, v = arrays.driver_codes[i[0]], arrays.driver_codes[i[1]]
        assert arrays.driver_edge_attr[0, 0] == pytest.approx(G_sc.edges[u, v]["gap_s"], rel=1e-4)

    def test_f_helper_converts_none_to_nan(self):
        assert np.isnan(GraphExporter._f(None)) and GraphExporter._f(3) == 3.0 and GraphExporter._f(True) == 1.0

    def test_to_dict_contains_only_numeric_payload(self, arrays):
        d = arrays.to_dict()
        assert {"track_x", "driver_x", "global_x", "driver_edge_index", "time_s", "pool_factor"} <= set(d)
        assert not {"track_names", "driver_codes"} & set(d)
        assert all(isinstance(v, (np.ndarray, int, float)) for v in d.values())

    def test_to_torch_roundtrip(self, arrays):
        torch = pytest.importorskip("torch")
        t = arrays.to_torch()
        assert t["track_x"].shape == tuple(arrays.track_x.shape) and t["driver_edge_index"].dtype == torch.int64

    def test_suggest_pool_factor_edge_cases(self):
        assert suggest_pool_factor(695, 20, max_tokens=512) == 2  # real Spa numbers
        assert suggest_pool_factor(1, 1, max_tokens=512) == 1
        # budget = 512 - 1 (global token) - 6 (drivers) = 505 track tokens
        assert suggest_pool_factor(505, 6, max_tokens=512) == 1
        assert suggest_pool_factor(506, 6, max_tokens=512) == 2  # one over budget -> pool
        with pytest.raises(ValueError):
            suggest_pool_factor(100, 511, max_tokens=512)

    def test_token_layout_orders_global_drivers_track(self, arrays):
        lay = arrays.token_layout()
        d, n = len(arrays.driver_codes), len(arrays.track_x)
        assert lay["token_type_ids"].tolist() == [0] + [1] * d + [2] * n
        assert lay["position_ids"].tolist() == [0] + list(range(d)) + list(range(n))
        assert lay["token_type_ids"].dtype == np.int64

    def test_graph_arrays_is_a_dataclass_instance(self, arrays):
        assert (
            isinstance(arrays, GraphArrays)
            and arrays.pool_factor == 1
            and arrays.time_s == pytest.approx(184.3, abs=0.1)
        )
