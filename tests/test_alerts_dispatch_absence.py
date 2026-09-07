"""alpha-engine-config-I10134 deliverable 2: a dispatch record with no
manifest past its horizon is an ABSENCE, for the on-demand jobs
`components.yaml` carries no schedule or deadline for at all.

Normative source: `crucible/AGENTS.md` §Alerting, plan §4.6 (exactly two
page conditions — this is a third INPUT to the first, not a third
condition).
"""

from __future__ import annotations

import datetime as dt
import json

from crucible.alerts import (
    DISPATCH_ABSENCE_HORIZON,
    Page,
    evaluate_dispatch_absence,
    sweep,
)
from crucible.keys import dispatch_key, manifest_key
from crucible.store import LocalStore

#: A Friday, well after that trading day's close (~20:00 UTC — see
#: `test_alerts_grouping.py`'s SATURDAY_NIGHT for the same boundary).
DISPATCHED_AT = dt.datetime(2026, 8, 28, 4, 3, tzinfo=dt.UTC)
#: Resolves to the PRIOR session (2026-08-27): before close on the 28th.
DISPATCH_TRADING_DAY = dt.date(2026, 8, 27)

#: Ten hours after DISPATCHED_AT — the exact gap the issue was filed over,
#: comfortably past DISPATCH_ABSENCE_HORIZON.
PAST_HORIZON = DISPATCHED_AT + dt.timedelta(hours=10)
#: One hour after DISPATCHED_AT — inside the horizon; the box may still be
#: mid-run.
WITHIN_HORIZON = DISPATCHED_AT + dt.timedelta(hours=1)


def _write_dispatch(
    store: LocalStore,
    *,
    job: str = "data.heal",
    dispatch_id: str = "01abc",
    args: str = "--from 2025-01-21 --to 2025-01-21",
    instance_id: str = "i-0cb52a780eb7eb90c",
    dispatched_at: dt.datetime = DISPATCHED_AT,
) -> None:
    store.put_bytes(
        dispatch_key(job, dispatch_id),
        json.dumps(
            {
                "schema_version": "dispatch_record.v1",
                "job": job,
                "args": args,
                "instance_id": instance_id,
                "requested_by": "crucible-v2-dispatcher",
                "dispatched_at_utc": dispatched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ).encode(),
    )


class TestDispatchAbsence:
    def test_a_dispatch_past_horizon_with_no_manifest_pages_absence(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        pages = evaluate_dispatch_absence(store, now=PAST_HORIZON)
        assert pages == [
            Page(
                condition="absence",
                job="data.heal",
                trading_day=DISPATCH_TRADING_DAY,
                reason=pages[0].reason,
            )
        ]
        assert "i-0cb52a780eb7eb90c" in pages[0].reason
        assert "--from 2025-01-21 --to 2025-01-21" in pages[0].reason

    def test_a_dispatch_still_inside_the_horizon_is_not_yet_due(self, tmp_path) -> None:
        """The same 'only past deadlines are evaluated' rule
        `evaluate_absence` states for scheduled jobs: a box mid-run must not
        page for being slow."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        assert evaluate_dispatch_absence(store, now=WITHIN_HORIZON) == []

    def test_a_manifest_at_the_expected_key_clears_the_dispatch(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TRADING_DAY.isoformat()),
            json.dumps({"status": "ok"}).encode(),
        )
        assert evaluate_dispatch_absence(store, now=PAST_HORIZON) == []

    def test_a_discriminated_manifest_also_clears_the_dispatch(self, tmp_path) -> None:
        """Same rule `evaluate_absence` follows: a discriminated manifest
        under the job's prefix is still a manifest for that trading day."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TRADING_DAY.isoformat(), discriminator="r1of4"),
            json.dumps({"status": "ok"}).encode(),
        )
        assert evaluate_dispatch_absence(store, now=PAST_HORIZON) == []

    def test_a_dispatch_record_with_an_unparseable_time_pages_rather_than_vanishing(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            dispatch_key("data.heal", "badtime"),
            json.dumps(
                {
                    "job": "data.heal",
                    "args": "--from x",
                    "instance_id": "i-bad",
                    "dispatched_at_utc": "not-a-timestamp",
                }
            ).encode(),
        )
        pages = evaluate_dispatch_absence(store, now=PAST_HORIZON)
        assert len(pages) == 1
        assert "cannot be graded against the absence horizon" in pages[0].reason

    def test_this_is_the_evidence_a_matching_scheduled_absence_groups_with(self, tmp_path) -> None:
        """§9.3: both share `cause_key` `absence:{trading_day}`, so an
        operator sees one incident naming both — never two separate pages
        for one bad day (`crucible.alerts.cause_key`)."""
        from crucible.alerts import cause_key

        store = LocalStore(tmp_path)
        _write_dispatch(store)
        [dispatch_page] = evaluate_dispatch_absence(store, now=PAST_HORIZON)
        scheduled_page = Page(
            condition="absence",
            job="data.daily",
            trading_day=DISPATCH_TRADING_DAY,
            reason="no manifest under runs/data.daily/2026-08-27/; due ...",
        )
        assert cause_key(dispatch_page) == cause_key(scheduled_page)

    def test_dry_run_sweep_never_writes_a_bus_row_for_a_dispatch_absence(self, tmp_path) -> None:
        """Same `alpha-engine-config-I9922` R2-1 guarantee `sweep(dry_run=True)`
        already gives scheduled absences: reading, never emitting."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        summary = sweep(store, now=PAST_HORIZON, sweep_run_id="run1", dry_run=True)
        assert summary["pages_emitted"] == 0
        assert summary["incidents_open"] == 1
        assert list(store.list_keys("alerts/")) == []


def test_the_horizon_is_stated_and_bounded() -> None:
    """Deliverable 2: 'a horizon that is stated, not implied.'"""
    assert DISPATCH_ABSENCE_HORIZON == dt.timedelta(hours=3)
