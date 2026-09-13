"""`alpha-engine-config-I10636`: a slot below `min_active_arms` must render a
finding AND page — but not once per trading day the standing state persists.

`crucible.alerts.evaluate_min_active_arms` reuses the FAILURE condition
(§4.6 admits exactly two) and anchors the page's `trading_day` at the
incident's earliest observed below-floor cycle, so the SAME incident dedups
across every later sweep the way `evaluate_absence`/`evaluate_failure`
already do for a persisting condition on one day.
"""

from __future__ import annotations

import datetime as dt
import json

from crucible.alerts import PAGE_CONDITIONS, evaluate_min_active_arms
from crucible.keys import arena_cycle_key
from crucible.slots.cycle import MIN_ACTIVE_ARMS_FINDING_METRIC
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
PRIOR_FRIDAY = dt.date(2026, 8, 21)
NOW = dt.datetime(2026, 8, 29, 23, 0, tzinfo=dt.UTC)


def _write_cycle(store: LocalStore, slot: str, day: dt.date, *, status: str, count: int) -> None:
    store.put_bytes(
        arena_cycle_key(slot, day.isoformat()),
        json.dumps(
            {
                "active_arms": [f"{slot}:control_planted_{slot}:x", f"{slot}:control_null_{slot}:x"]
                + [f"{slot}:real_{i}:x" for i in range(count)],
                MIN_ACTIVE_ARMS_FINDING_METRIC: {
                    "status": status,
                    "min_active_arms": 3,
                    "promotable_arm_count": count,
                    "promotable_arms": [f"{slot}:real_{i}:x" for i in range(count)],
                    "reason": f"{count} promotable arm(s) against a floor of 3",
                },
            }
        ).encode("utf-8"),
    )


class TestEvaluateMinActiveArms:
    def test_fires_for_a_below_floor_slot(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_cycle(store, "s", FRIDAY, status="BELOW_FLOOR", count=1)
        pages = evaluate_min_active_arms(store, now=NOW)
        s_pages = [p for p in pages if p.job == "slots.s"]
        assert len(s_pages) == 1
        assert s_pages[0].condition == "failure"
        assert s_pages[0].condition in PAGE_CONDITIONS
        assert "floor of 3" in s_pages[0].reason

    def test_does_not_fire_for_a_slot_at_the_floor(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_cycle(store, "s", FRIDAY, status="OK", count=3)
        pages = evaluate_min_active_arms(store, now=NOW)
        assert [p for p in pages if p.job == "slots.s"] == []

    def test_a_slot_with_no_graded_cycle_pages_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        assert evaluate_min_active_arms(store, now=NOW) == []

    def test_the_incident_anchors_at_the_earliest_below_floor_cycle_not_today(
        self, tmp_path
    ) -> None:
        """A standing state observed two cycles running must dedup to ONE
        incident — the anchor is the earlier day, so the second sweep's
        group resolves to the SAME `incident_key` the first one produced
        rather than a fresh one every week."""
        store = LocalStore(tmp_path)
        _write_cycle(store, "s", PRIOR_FRIDAY, status="BELOW_FLOOR", count=1)
        _write_cycle(store, "s", FRIDAY, status="BELOW_FLOOR", count=1)
        pages = evaluate_min_active_arms(store, now=NOW)
        s_pages = [p for p in pages if p.job == "slots.s"]
        assert len(s_pages) == 1
        assert s_pages[0].trading_day == PRIOR_FRIDAY

    def test_a_slot_that_cleared_the_floor_does_not_anchor_past_the_recovery(
        self, tmp_path
    ) -> None:
        """The below-floor run before a recovery must not extend the anchor
        through a cycle that read OK."""
        store = LocalStore(tmp_path)
        _write_cycle(store, "s", PRIOR_FRIDAY, status="OK", count=3)
        _write_cycle(store, "s", FRIDAY, status="BELOW_FLOOR", count=1)
        pages = evaluate_min_active_arms(store, now=NOW)
        s_pages = [p for p in pages if p.job == "slots.s"]
        assert len(s_pages) == 1
        assert s_pages[0].trading_day == FRIDAY
