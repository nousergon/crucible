"""`arms_all_scored` does not demand a cycle the day's own grade declared it could not build.

`alpha-engine-config-I11084`. The 09-18 replay filled every U/R/M cycle and
left the clause red on S alone: `experiment.grade[s]` read `ok` on 2026-09-18
with `no_settled_session` (S's first day, nothing settled yet) and on
2026-09-25 with `upstream_champion_absent` (no M pointer when it ran). No
re-run can produce either cycle, and `promote[s]` already reads the same
declaration (`crucible.slots.strategy.declared_grade_outcome`). The gate now
reads it too, and names every day it stopped asking for.
"""

from __future__ import annotations

import json

import pytest
from test_arm_filing_day import (
    EARLY_FRIDAY,
    INCUMBENT,
    LATE_FRIDAY,
    RELEASE,
    _publish_release,
    _put_cycle,
    _put_register,
    _row,
    _seed_arc_manifest,
)

from crucible.gate import _clause_arms_all_scored
from crucible.keys import arena_cycle_key
from crucible.manifest import manifest_key
from crucible.slots.strategy import NO_SETTLED_SESSION_METRIC, UPSTREAM_CHAMPION_ABSENT_METRIC
from crucible.store import LocalStore


def _store(tmp_path) -> LocalStore:
    store = LocalStore(tmp_path)
    _publish_release(store, RELEASE, "u", "s")
    for day in (EARLY_FRIDAY, LATE_FRIDAY):
        _seed_arc_manifest(store, day, RELEASE)
        _put_cycle(store, day, [INCUMBENT])
    _put_register(store, [_row("incumbent", filed_on="2026-01-02", created_date="2026-01-02")])
    return store


def _put_s_grade(store, day, *, status="ok", metric=NO_SETTLED_SESSION_METRIC, claims_cycle=False):
    outputs = [{"key": arena_cycle_key("s", day.isoformat())}] if claims_cycle else []
    metrics = (
        [{"name": metric, "status": "unmeasurable", "status_reason": "fixture"}] if metric else []
    )
    store.put_bytes(
        manifest_key("experiment.grade", day.isoformat(), discriminator="s"),
        json.dumps({"status": status, "outputs": outputs, "metrics": metrics}).encode(),
    )


def test_both_declared_days_are_not_required_and_named(tmp_path) -> None:
    store = _store(tmp_path)
    _put_s_grade(store, EARLY_FRIDAY, metric=NO_SETTLED_SESSION_METRIC)
    _put_s_grade(store, LATE_FRIDAY, metric=UPSTREAM_CHAMPION_ABSENT_METRIC)
    clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
    assert clause.met, clause.detail
    assert f"s@{EARLY_FRIDAY.isoformat()} ({NO_SETTLED_SESSION_METRIC})" in clause.detail
    assert f"s@{LATE_FRIDAY.isoformat()} ({UPSTREAM_CHAMPION_ABSENT_METRIC})" in clause.detail


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "failed"},
        {"metric": None},
        {"claims_cycle": True},
    ],
    ids=["failed-grade", "no-declaration", "claims-a-cycle-that-is-absent"],
)
def test_anything_short_of_a_declaration_is_still_a_gap(tmp_path, kwargs) -> None:
    store = _store(tmp_path)
    _put_s_grade(store, EARLY_FRIDAY)
    _put_s_grade(store, LATE_FRIDAY, **kwargs)
    clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
    assert not clause.met
    assert f"s@{LATE_FRIDAY.isoformat()}: no arena_cycle artifact" in clause.detail


def test_no_grade_manifest_is_still_a_gap(tmp_path) -> None:
    store = _store(tmp_path)
    _put_s_grade(store, EARLY_FRIDAY)
    clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
    assert not clause.met
    assert f"s@{LATE_FRIDAY.isoformat()}: no arena_cycle artifact" in clause.detail
