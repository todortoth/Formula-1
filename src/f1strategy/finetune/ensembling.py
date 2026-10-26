"""ensembling.py - mean predictions of several runs (and baselines) as extra table rows (numpy only).

``--ensemble`` of ``evaluate`` averages all the given runs; ``--groups`` defines any number of named subsets, e.g.

    --run B0,B1,B2,E0,E1,E2 --groups "B=0,1,2;E=3,4,5;BE=0,1,2,3,4,5;Bgbdt=0,1,2,gbdt,gbdt,gbdt"

Numbers are positions in ``--run``; a baseline name (``gbdt``, ``rule``) takes that baseline as a member, and repeating a token
gives it a larger weight (above: gbdt carries half of the mean, like E in ``BE``).  Rows are named ``ens_<name>``; with
``scripts/compare_by_race.py report ens_E ens_B`` two groups are compared race by race.
"""

from __future__ import annotations

import numpy as np


def ensemble(members: list[dict]) -> dict:
    """Mean of the pit probabilities, laps and no-more probabilities; the compound probabilities of the members that have them."""
    m = {k: np.mean([r[k] for r in members], 0) for k in ("pit", "laps", "nomore")}
    cps = [r["cmp_prob"] for r in members if "cmp_prob" in r]
    if cps:
        m["cmp_prob"] = np.mean(cps, 0)
        m["cmp"] = m["cmp_prob"].argmax(-1)
    return m


def parse_groups(spec: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, sep, toks = part.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"group {part!r}: write name=member,member,...")
        out[name.strip()] = [t.strip() for t in toks.split(",") if t.strip()]
    return out


def group_members(tokens: list[str], run_preds: list[dict], baseline_preds: dict[str, dict]) -> list[dict]:
    """The prediction dicts a group is made of: ``run_preds[i]`` for a number, ``baseline_preds[name]`` for a baseline name."""
    members = []
    for t in tokens:
        if t.isdigit():
            if int(t) >= len(run_preds):
                raise ValueError(f"group member {t}: only {len(run_preds)} runs were given")
            members.append(run_preds[int(t)])
        elif t in baseline_preds:
            members.append(baseline_preds[t])
        else:
            raise ValueError(f"group member {t!r}: neither a run position nor a fitted baseline")
    return members
