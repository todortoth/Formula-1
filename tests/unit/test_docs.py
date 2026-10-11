"""Documentation and CI configuration must not drift away from the code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from f1strategy.cli import build_parser
from f1strategy.graph.exporter import DRIVER_FEATURES, EDGE_FEATURES, GLOBAL_FEATURES, TRACK_FEATURES

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _block(name: str) -> str:
    m = re.search(rf"<!-- BEGIN:{name} -->(.*?)<!-- END:{name} -->", README, re.S)
    assert m, f"README is missing the <!-- BEGIN:{name} --> block"
    return m.group(1)


def _first_column_names(block: str) -> list[str]:
    return re.findall(r"^\| `([^`]+)` \|", block, re.M)


@pytest.mark.parametrize(
    "block, expected",
    [
        ("track-features", TRACK_FEATURES),
        ("driver-features", DRIVER_FEATURES),
        ("global-features", GLOBAL_FEATURES),
        ("edge-features", EDGE_FEATURES),
    ],
)
def test_feature_tables_match_the_exporter_schema_in_name_and_order(block, expected):
    assert _first_column_names(_block(block)) == list(expected)


def test_every_feature_has_a_non_empty_description():
    for block in ("track-features", "driver-features", "global-features", "edge-features"):
        rows = re.findall(r"^\| `[^`]+` \| (.*) \|$", _block(block), re.M)
        assert rows and all(len(r.strip()) > 3 for r in rows), block


def test_cli_table_matches_the_real_parser():
    documented = set(_first_column_names(_block("cli-options")))
    actual = {o for a in build_parser()._actions for o in a.option_strings if o.startswith("--") and o != "--help"}
    assert documented == actual


def test_cli_defaults_in_the_readme_match_the_parser():
    args = build_parser().parse_args([])
    text = _block("cli-options")
    for needle in (str(args.year), args.event, str(args.lap), args.out_dir, str(args.max_tokens)):
        assert needle in text


def test_every_source_module_appears_in_the_project_structure():
    block = _block("project-structure")
    for path in (ROOT / "src" / "f1strategy").rglob("*.py"):
        if path.name != "__init__.py":
            assert path.name in block, f"{path.relative_to(ROOT)} missing from README project structure"


def test_structure_block_lists_only_existing_files():
    block = _block("project-structure")
    for name in re.findall(r"[├└]── ([\w.]+\.(?:py|toml|yml))", block):
        assert list(ROOT.rglob(name)), f"README mentions {name} which does not exist"


def test_readme_images_exist():
    for rel in re.findall(r'src="([^"]+\.png)"', README):
        assert (ROOT / rel).is_file(), rel


def test_readme_mentions_every_public_name():
    import f1strategy

    for name in f1strategy.__all__:
        if name != "__version__":
            assert name in README, f"{name} is not mentioned in the README"


def test_migration_table_covers_every_original_module():
    for old in ("SectorNodeClass", "DriverNodeClass", "TrackModelClass", "RaceControlTimelineClass", "TelemetryStoreClass",
                "DriverTimingClass", "RaceGraphBuilderClass", "GraphExporterClass", "SessionAnalyserClass"):  # fmt: skip
        assert f"`{old}.py`" in README


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _triggers(wf):
    return wf.get("on") or wf.get(True)  # PyYAML parses a bare `on:` key as boolean True


def test_ci_matrix_covers_every_python_version_the_package_claims():
    proj = _pyproject()["project"]
    claimed = {c.rsplit(" ", 1)[1] for c in proj["classifiers"] if re.search(r"Python :: 3\.\d+$", c)}
    matrix = set(_workflow("ci.yml")["jobs"]["test"]["strategy"]["matrix"]["python-version"])
    assert claimed <= matrix, f"CI does not test {sorted(claimed - matrix)}"
    assert proj["requires-python"].startswith(">=3.10") and "3.10" in matrix


def test_ci_runs_on_the_default_branch_and_pull_requests():
    trig = _triggers(_workflow("ci.yml"))
    assert "master" in trig["push"]["branches"] and "pull_request" in trig


def test_ci_has_lint_test_min_deps_and_build_jobs_and_runs_ruff_and_pytest():
    jobs = _workflow("ci.yml")["jobs"]
    assert {"lint", "test", "minimum-dependencies", "build"} <= set(jobs)
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert (
        "ruff check" in text
        and "ruff format --check" in text
        and "pytest" in text
        and "--resolution lowest-direct" in text
    )


def test_live_workflow_is_scheduled_and_selects_the_live_marker():
    wf = _workflow("live-data.yml")
    assert "schedule" in _triggers(wf) and "workflow_dispatch" in _triggers(wf)
    assert "-m live" in (ROOT / ".github" / "workflows" / "live-data.yml").read_text()


def test_live_tests_are_deselected_by_default_and_marker_is_registered():
    ini = _pyproject()["tool"]["pytest"]["ini_options"]
    assert "not live" in ini["addopts"] and any(m.startswith("live") for m in ini["markers"])


def test_readme_badge_points_at_an_existing_workflow():
    names = re.findall(r"actions/workflows/([\w.-]+)/badge\.svg", README)
    assert names and all((ROOT / ".github" / "workflows" / n).is_file() for n in names)


def test_runtime_dependencies_are_declared_with_lower_bounds():
    deps = _pyproject()["project"]["dependencies"]
    assert {re.split(r"[<>=]", d)[0] for d in deps} >= {"fastf1", "numpy", "pandas", "scipy", "networkx", "matplotlib"}
    assert all(">=" in d for d in deps)  # the minimum-dependencies CI job relies on explicit lower bounds


def test_version_is_single_sourced():
    import f1strategy

    cfg = _pyproject()
    assert (
        "version" in cfg["project"]["dynamic"]
        and cfg["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "f1strategy.__version__"
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", f1strategy.__version__)
