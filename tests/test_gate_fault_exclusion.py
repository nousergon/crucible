"""`arc_runs_ok`/`replays_ok` exclude a manifest a fault record excuses
(`alpha-engine-config-I10322`).

The whole design constraint, restated from the issue: a failed manifest is
excused ONLY when a `faults/{day}/{fault}.json` record names its exact
`run_id` — never the trading day alone, so a genuine failure on a day a
fault was once induced still fails these clauses. Both directions are
asserted below, per the issue's own `closes-when`.

Seeded by hand, in the same style as `tests/test_gate_manifest_key_contract.py`
— a minimal one-component synthetic registry, so a five-week `replays_ok`
window costs five manifests, not a full twelve-stage arc times five.
"""

from __future__ import annotations

import datetime as dt
import json

from crucible.components import Component, Deadline
from crucible.gate import _clause_arc_runs_ok, _clause_replays_ok
from crucible.keys import fault_injection_key
from crucible.manifest import manifest_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
RUN_ID = "01M23BQV1C6EPDR8DEA5WS59M4"
OTHER_RUN_ID = "01M1WM0CEY4NKR10TWXJZAHD1R"

_OK_BYTES = (
    b'{"status": "ok", "reason": "", "inputs": [], "outputs": [], '
    b'"release_sha": "' + b"a" * 40 + b'", "run_id": "%s"}'
)


def _registry() -> dict[str, Component]:
    """One arc-dispatched component — `data.weekly`, matching the real
    `crucible.components.yaml` row's name so `manifest_key("data.weekly", ...)`
    reads exactly the key the real job would write."""
    deadline = Deadline.from_yaml(
        {"anchor": "next_calendar_day_at", "cadence": "weekly", "at": "09:00"}
    )
    component = Component(
        name="data.weekly",
        description="x",
        lifecycle="ACTIVE",
        signals={
            "execution": "run.json",
            "cost": None,
            "resource": None,
            "lineage": None,
            "outcome": None,
        },
        log_location="/x",
        log_retention_days=1,
        alert_channel="x",
        console_surface="x",
        artifact_retention="forever",
        schedule="weekly",
        deadline=deadline,
        dispatch="arc",
    )
    return {"data.weekly": component}


def _seed_ok(store: LocalStore, day: dt.date, run_id: str = "0" * 26) -> None:
    store.put_bytes(
        manifest_key("data.weekly", day.isoformat()),
        _OK_BYTES % run_id.encode(),
    )


def _seed_failed(store: LocalStore, day: dt.date, run_id: str) -> None:
    store.put_bytes(
        manifest_key("data.weekly", day.isoformat()),
        (
            b'{"status": "failed", "reason": "MissingSourceError: withheld", '
            b'"inputs": [], "outputs": [], "release_sha": "' + b"a" * 40 + b'", '
            b'"run_id": "' + run_id.encode() + b'"}'
        ),
    )


def _seed_fault_record(
    store: LocalStore,
    day: dt.date,
    fault_id: str,
    run_id: str | None,
    *,
    outcome: str = "induced",
) -> None:
    """A conforming record, per outcome kind.

    `induced` is the ONLY kind whose `run_id` reaches the excusal
    (`alpha-engine-config-I10327`), so the other two are seeded here too and
    asserted NOT to excuse anything.
    """
    store.put_bytes(
        fault_injection_key(fault_id, day.isoformat()),
        json.dumps(
            {
                "schema_version": "fault_record.v1",
                "fault_id": fault_id,
                "outcome": outcome,
                "trading_day": day.isoformat(),
                "run_id": run_id,
                "manifest_key": (
                    manifest_key("data.weekly", day.isoformat())
                    if outcome != "unreachable"
                    else None
                ),
                "bus_key": (
                    f"alerts/{day.isoformat()}/failure.data.weekly.json"
                    if outcome == "induced"
                    else None
                ),
                "attempt": (
                    {"n": 2, "reason": "spot_interruption"} if outcome == "absorbed" else None
                ),
                "closed_paths": (
                    [
                        {
                            "path": "the pointer comes to name an unpublished sha",
                            "mechanism": "crucible.release.pin",
                            "probe": "pin_refuses_an_unpublished_sha",
                            "expected": "StaleReleasePointerError, raised before any write",
                            "observed": "StaleReleasePointerError: was never published",
                            "checked_at_utc": "2026-09-09T12:00:00Z",
                        }
                    ]
                    if outcome == "unreachable"
                    else None
                ),
                "recorded_at_utc": "2026-09-09T12:00:00Z",
            }
        ).encode("utf-8"),
    )


class TestAMatchingFaultRecordExcusesTheFailure:
    def test_arc_runs_ok_reads_met(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _seed_failed(store, FRIDAY, RUN_ID)
        _seed_fault_record(store, FRIDAY, "data_source_withheld", RUN_ID)

        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())

        assert clause.met, clause.detail
        assert "excused" in clause.detail


class TestAGenuineFailureWithNoFaultRecordStillFails:
    """`-I10322`'s closes-when, direction 1: verified to fail WITHOUT the
    fault record — the exclusion never fires on its own."""

    def test_arc_runs_ok_reads_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _seed_failed(store, FRIDAY, RUN_ID)
        # No fault record filed at all.

        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())

        assert not clause.met
        assert "1 failed" in clause.detail


class TestOnlyAnInducedRecordExcusesAnything:
    """`alpha-engine-config-I10327`. The record grew two more outcome kinds and
    neither may reach the excusal: `absorbed` names a manifest reading `ok` and
    `unreachable` names no run at all. Keying the exclusion on `run_id` AND
    narrowing it by `outcome` is what stops a record filed for a fault the
    system SURVIVED from excusing a failure it did not cause.
    """

    def test_an_absorbed_record_naming_the_failed_run_does_not_excuse_it(self, tmp_path) -> None:
        """The attack this closes: an `absorbed` record whose `run_id` happens
        to name a FAILED manifest. The record is well-formed for its own kind,
        and it still excuses nothing."""
        store = LocalStore(tmp_path)
        _seed_failed(store, FRIDAY, RUN_ID)
        _seed_fault_record(store, FRIDAY, "spot_terminated_mid_job", RUN_ID, outcome="absorbed")

        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())

        assert not clause.met
        assert "1 failed" in clause.detail
        assert "excused" not in clause.detail

    def test_an_unreachable_record_excuses_nothing(self, tmp_path) -> None:
        """Structurally, not by policy: it carries no `run_id` to match."""
        store = LocalStore(tmp_path)
        _seed_failed(store, FRIDAY, RUN_ID)
        _seed_fault_record(store, FRIDAY, "stale_release_pointer", None, outcome="unreachable")

        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())

        assert not clause.met
        assert "1 failed" in clause.detail

    def test_an_induced_record_still_excuses(self, tmp_path) -> None:
        """The other direction, so the narrowing is shown not to have broken
        the mechanism it narrows."""
        store = LocalStore(tmp_path)
        _seed_failed(store, FRIDAY, RUN_ID)
        _seed_fault_record(store, FRIDAY, "data_source_withheld", RUN_ID, outcome="induced")

        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())

        assert clause.met, clause.detail
        assert "excused" in clause.detail


class TestAFaultRecordNamingADifferentRunIdNeverExcuses:
    """`-I10322`'s closes-when, direction 2: a fault record on the SAME
    trading day, naming a DIFFERENT run_id, must not excuse a genuine
    failure — matched on run_id, never on the day alone."""

    def test_arc_runs_ok_reads_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _seed_failed(store, FRIDAY, RUN_ID)
        # A fault record exists for this exact day, but for a DIFFERENT run.
        _seed_fault_record(store, FRIDAY, "data_source_withheld", OTHER_RUN_ID)

        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())

        assert not clause.met
        assert "1 failed" in clause.detail


class TestReplaysOkInheritsTheSameExclusion:
    """`_clause_replays_ok` delegates to `_clause_arc_runs_ok` verbatim
    (unchanged by this PR) — this proves the exclusion reaches it too, over
    its own five-Friday replay window."""

    def test_replays_ok_reads_met_with_a_matching_record(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        registry = _registry()
        # weekly_anchor(2026-08-29) == 2026-08-28 (the Friday strictly
        # before it); weekly_window(2026-08-29, 5) is the five Fridays
        # ending there.
        render_day = dt.date(2026, 8, 29)
        fridays = [
            dt.date(2026, 7, 31),
            dt.date(2026, 8, 7),
            dt.date(2026, 8, 14),
            dt.date(2026, 8, 21),
            dt.date(2026, 8, 28),
        ]
        for day in fridays[:-1]:
            _seed_ok(store, day)
        _seed_failed(store, fridays[-1], RUN_ID)
        _seed_fault_record(store, fridays[-1], "data_source_withheld", RUN_ID)

        clause = _clause_replays_ok(store, [render_day], registry)

        assert clause.met, clause.detail
        assert clause.name == "replays_ok"

    def test_replays_ok_reads_unmet_without_a_matching_record(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        registry = _registry()
        render_day = dt.date(2026, 8, 29)
        fridays = [
            dt.date(2026, 7, 31),
            dt.date(2026, 8, 7),
            dt.date(2026, 8, 14),
            dt.date(2026, 8, 21),
            dt.date(2026, 8, 28),
        ]
        for day in fridays[:-1]:
            _seed_ok(store, day)
        _seed_failed(store, fridays[-1], RUN_ID)
        # No fault record.

        clause = _clause_replays_ok(store, [render_day], registry)

        assert not clause.met


class TestAMalformedFaultRecordExcusesNothing:
    """A record that fails to parse is skipped, not raised — a malformed or
    hand-broken record must never be able to excuse anything, per
    `-I10322`'s own worst-case framing."""

    def test_arc_runs_ok_still_reads_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _seed_failed(store, FRIDAY, RUN_ID)
        store.put_bytes(
            fault_injection_key("data_source_withheld", FRIDAY.isoformat()),
            b"{not json",
        )

        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())

        assert not clause.met
