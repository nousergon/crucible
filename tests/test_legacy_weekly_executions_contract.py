"""The v1-weekly-executions producer/consumer contract, and the ruled metric.

Normative source: Brian's ruling of 2026-09-04 on `alpha-engine-config-I9756`,
implemented under `alpha-engine-config-I9962`; the M0 contract discipline in
`AGENTS.md` ("every new cross-repo artifact gets a versioned schema +
producer/consumer contract test at birth").

**What was wrong.** `_clause_old_weekly_within_cadence` counted raw
`StartExecution` against a ceiling of one a week. `alpha-engine-saturday`'s
`cron(0 9 ? * THU-SAT *)` fires three times a week BY DESIGN — two of those
firings Succeed-skip at `WeeklyRunDayGate` in about three seconds and are the
holiday fail-open margin `sf-pipeline-policy.md` §5 line 213 names outright.
So the clause reported a finding against a system behaving exactly as ruled,
and it was the only clause holding phase 0, which put every phase above it in
`OUT_OF_ORDER`. Brian ruled the METRIC changes and the cron stays.

**Why one clause change could not implement that.** The document the clause
reads carried a single integer, under which a 3.0-second Succeed-skip and a
five-hour run are the same event. The producer
(`nous-ergon-ops/scripts/legacy_weekly_executions_producer.py`) had to emit
the distinction first — hence the schema version, and hence this file, which
grades both sides of it:

* the **schema** (`crucible/schemas/legacy_weekly_executions.v2.json`) is the
  declaration, and every assertion about it here tests it REFUSING something;
* the **consumer** grades the ruled metric, treats a `v1` document as
  UNMEASURABLE rather than as a pass, and fails a `watch-rerun-*` execution
  regardless of duration;
* the §7.4 demonstration at the bottom shows the fix CHANGES a reading: the
  same week reads MET under the amended clause and UNMET under the superseded
  one, asserted as a difference rather than as two independent facts.

**Why the producer lives in another repository, and how drift is caught.**
`states:ListExecutions` against a live v1 state machine is operated-system
assembly, and a gate's own repository must never hold the identity that can
reach live AWS on its behalf. There is no shared package between the two, so
the contract is carried by the version string: the producer declares
`legacy-weekly-executions.v2`, the consumer refuses every other value, and a
producer that changes the shape without bumping the version makes its own
documents unmeasurable rather than silently mis-graded. `nous-ergon-ops`'s
`tests/test_legacy_weekly_executions_producer.py` pins the same literal from
the writing side.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from crucible.gate import (
    LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
    LEGACY_WEEKLY_RERUN_NAME_PREFIX,
    WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS,
    _clause_old_weekly_within_cadence,
    legacy_weekly_executions_key,
    weekly_anchor,
)
from crucible.store import LocalStore

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "crucible" / "schemas"
SCHEMA_PATH = SCHEMA_PATH / "legacy_weekly_executions.v2.json"

FRIDAY = dt.date(2026, 8, 28)
#: Two consecutive weeks, the window phase 0's gate reads.
WINDOW = [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(2))]


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _errors(payload: dict[str, Any]) -> list[str]:
    return [e.message for e in Draft202012Validator(_schema()).iter_errors(payload)]


def _execution(
    name: str,
    start: str,
    duration_seconds: float | None,
    status: str = "SUCCEEDED",
) -> dict[str, Any]:
    started = dt.datetime.fromisoformat(start)
    stopped = None if duration_seconds is None else started + dt.timedelta(seconds=duration_seconds)
    return {
        "name": name,
        "start": started.isoformat(),
        "stop": None if stopped is None else stopped.isoformat(),
        "duration_seconds": duration_seconds,
        "status": status,
    }


#: The measured pair the ruling names, plus the second skip the THU-SAT cron
#: produces: Thursday 2026-09-03 ran 3.0 seconds and SUCCEEDED (a
#: `WeeklyRunDayGate` Succeed-skip), Saturday 2026-08-29 ran 5h 01m and FAILED
#: (the real cycle). This is one clean week under the amended metric and three
#: starts under the superseded one.
def _ruled_week(anchor: dt.date) -> dict[str, Any]:
    day = anchor.isoformat()
    return {
        "schema_version": LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
        "executions_started": 3,
        "executions": [
            _execution("uuid_thursday", f"{day}T09:00:49.259000+00:00", 3.016),
            _execution("uuid_friday", f"{day}T09:00:49.100000+00:00", 2.8),
            _execution("uuid_saturday", f"{day}T09:00:49.233000+00:00", 18111.6, status="FAILED"),
        ],
        "source": "states:ListExecutions, fixture",
    }


def _v1_week(starts: int = 3) -> dict[str, Any]:
    """The superseded document: one integer, no version, no per-execution fact."""
    return {"executions_started": starts, "source": "a filed count, not a live API call"}


def _seed(tmp_path: Path, document_for) -> LocalStore:
    store = LocalStore(tmp_path)
    for day in WINDOW:
        anchor = weekly_anchor(day)
        key = legacy_weekly_executions_key(anchor.isoformat())
        store.put_bytes(key, json.dumps(document_for(anchor)).encode("utf-8"))
    return store


def _clause(store: LocalStore, **kwargs):
    return _clause_old_weekly_within_cadence(store, WINDOW, **kwargs)


# ── the schema, shown REFUSING ──────────────────────────────────────────────


class TestTheSchemaConstrainsTheDocument:
    def test_the_ruled_week_validates(self) -> None:
        assert _errors(_ruled_week(FRIDAY)) == []

    def test_a_v1_document_is_refused_by_the_v2_schema(self) -> None:
        """The version is the whole contract: a v1 body must not validate as
        v2, or a producer could stop filing `executions` and nothing would
        notice."""
        assert _errors(_v1_week()) != []

    @pytest.mark.parametrize(
        "field", ["schema_version", "executions_started", "executions", "source"]
    )
    def test_every_required_field_is_required(self, field: str) -> None:
        payload = _ruled_week(FRIDAY)
        del payload[field]
        assert _errors(payload) != []

    @pytest.mark.parametrize("field", ["name", "start", "stop", "duration_seconds", "status"])
    def test_every_per_execution_field_is_required(self, field: str) -> None:
        payload = _ruled_week(FRIDAY)
        del payload["executions"][0][field]
        assert _errors(payload) != []

    def test_a_field_the_consumer_does_not_understand_is_refused(self) -> None:
        """`additionalProperties: false` — a field the producer expected the
        consumer to act on, that the consumer has never heard of, is a silent
        contract break otherwise."""
        payload = _ruled_week(FRIDAY)
        payload["gate_passing"] = 1
        assert _errors(payload) != []

    def test_the_version_is_a_const_not_a_free_string(self) -> None:
        payload = _ruled_week(FRIDAY)
        payload["schema_version"] = "legacy-weekly-executions.v3"
        assert _errors(payload) != []

    def test_a_negative_duration_is_refused(self) -> None:
        payload = _ruled_week(FRIDAY)
        payload["executions"][0]["duration_seconds"] = -1
        assert _errors(payload) != []

    def test_a_null_duration_is_accepted_because_running_has_none(self) -> None:
        payload = _ruled_week(FRIDAY)
        payload["executions"][0]["duration_seconds"] = None
        payload["executions"][0]["stop"] = None
        payload["executions"][0]["status"] = "RUNNING"
        assert _errors(payload) == []


# ── the consumer grades the ruled metric ────────────────────────────────────


class TestTheConsumerGradesTheRuledMetric:
    def test_two_weeks_of_one_run_plus_two_skips_are_MET(self, tmp_path) -> None:
        clause = _clause(_seed(tmp_path, _ruled_week))
        assert clause.met, clause.detail
        assert not clause.unmeasurable
        assert "gate-passing execution" in clause.detail

    def test_two_gate_passing_executions_in_a_week_are_UNMET(self, tmp_path) -> None:
        """The punitive cadence the ruling still forbids: a second real cycle
        in one week is exactly what the ceiling exists to catch, and the
        amended metric must not have made it invisible."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["executions"].append(
                _execution("uuid_second_run", f"{anchor.isoformat()}T20:00:00+00:00", 4000.0)
            )
            payload["executions_started"] = len(payload["executions"])
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met
        assert not clause.unmeasurable
        assert "2 gate-passing executions, ceiling 1" in clause.detail

    def test_the_reason_line_names_gate_passing_executions_not_raw_starts(self, tmp_path) -> None:
        """`alpha-engine-config-I9962`'s closes-when. The live reading said
        "14 starts, ceiling 1" while grading a retired metric; an operator
        reading that goes and looks for reruns that are not there."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["executions"].append(
                _execution("uuid_second_run", f"{anchor.isoformat()}T20:00:00+00:00", 4000.0)
            )
            payload["executions_started"] = len(payload["executions"])
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert "starts, ceiling" not in clause.detail
        assert "Succeed-skips excluded" in clause.detail

    def test_a_watch_rerun_fails_regardless_of_duration(self, tmp_path) -> None:
        """The name identifies a rerun ISSUER — the operator cost deliverable
        1 removes. A rerun that happened to Succeed-skip in two seconds is
        still a rerun, and the duration rule must not launder it."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["executions"].append(
                _execution(
                    # Named for THIS anchor's week: since
                    # `alpha-engine-config-I9756` (2026-09-04) a rerun is
                    # graded against the week it retries, and this test is
                    # about duration being irrelevant, not about attribution.
                    f"{LEGACY_WEEKLY_RERUN_NAME_PREFIX}{anchor.isoformat()}-1",
                    f"{anchor.isoformat()}T21:00:00+00:00",
                    1.0,
                )
            )
            payload["executions_started"] = len(payload["executions"])
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met
        assert not clause.unmeasurable
        assert "watch-rerun" in clause.detail
        assert "regardless of duration" in clause.detail

    def test_a_long_rerun_also_fails(self, tmp_path) -> None:
        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["executions"].append(
                _execution(
                    f"{LEGACY_WEEKLY_RERUN_NAME_PREFIX}{anchor.isoformat()}-2",
                    f"{anchor.isoformat()}T21:00:00+00:00",
                    9000.0,
                    status="FAILED",
                )
            )
            payload["executions_started"] = len(payload["executions"])
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met
        assert "watch-rerun" in clause.detail

    def test_a_running_execution_counts_as_a_run_not_as_a_fast_skip(self, tmp_path) -> None:
        """`duration_seconds: null` is "no answer", not "zero seconds". Read
        as zero it would be the fastest possible Succeed-skip and would
        exclude a live cycle from the count grading it."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["executions"].append(
                _execution(
                    "uuid_in_flight", f"{anchor.isoformat()}T22:00:00+00:00", None, status="RUNNING"
                )
            )
            payload["executions_started"] = len(payload["executions"])
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met
        assert "2 gate-passing executions" in clause.detail

    def test_a_succeed_skip_at_exactly_the_threshold_is_a_run(self, tmp_path) -> None:
        """Strictly less than. The threshold sits in an empty band three
        orders of magnitude wide (3.0s versus 5h01m), so the boundary case
        cannot occur in practice — and where it cannot be measured, the rule
        counts the execution rather than excusing it."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["executions"].append(
                _execution(
                    "uuid_exactly_at_threshold",
                    f"{anchor.isoformat()}T23:00:00+00:00",
                    WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS,
                )
            )
            payload["executions_started"] = len(payload["executions"])
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met

    def test_a_failed_short_execution_is_not_a_skip(self, tmp_path) -> None:
        """The Succeed-skip is a SUCCEEDED terminal state at the gate. A
        three-second FAILURE is the pipeline dying at launch, which is a
        finding, not a fail-open margin."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["executions"].append(
                _execution(
                    "uuid_died_at_launch",
                    f"{anchor.isoformat()}T23:30:00+00:00",
                    2.0,
                    status="FAILED",
                )
            )
            payload["executions_started"] = len(payload["executions"])
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met


# ── an old-schema week is UNMEASURABLE, never a pass ────────────────────────


class TestAnOldSchemaWeekIsUnmeasurable:
    def test_a_v1_document_reads_unmeasurable_with_a_named_reason(self, tmp_path) -> None:
        """`[?]`, not `[x]` on the retired integer and not a bare `[ ]`. The
        week cannot answer the question now being asked, and "no data" is
        never rendered green (principle 7)."""
        clause = _clause(_seed(tmp_path, lambda _anchor: _v1_week(starts=1)))
        assert not clause.met
        assert clause.unmeasurable
        assert "schema_version is None" in clause.detail
        assert "unmeasurable, not a pass" in clause.detail

    def test_a_v1_document_under_the_old_ceiling_is_still_not_met(self, tmp_path) -> None:
        """The specific way this could mislead a second time: one start is
        within the OLD ceiling, so a reader that kept grading
        `executions_started` would call this week met while measuring the
        metric Brian retired."""
        clause = _clause(_seed(tmp_path, lambda _anchor: _v1_week(starts=1)))
        assert not clause.met

    def test_a_mixed_window_is_a_finding_not_a_shrug(self, tmp_path) -> None:
        """One stale week beside one week genuinely over the ceiling. `[?]`
        there would hide a real finding behind an unreadable week."""

        def document_for(anchor: dt.date) -> dict[str, Any]:
            if anchor == weekly_anchor(WINDOW[-1]):
                payload = _ruled_week(anchor)
                payload["executions"].append(
                    _execution("uuid_second_run", f"{anchor.isoformat()}T20:00:00+00:00", 4000.0)
                )
                payload["executions_started"] = len(payload["executions"])
                return payload
            return _v1_week()

        clause = _clause(_seed(tmp_path, document_for))
        assert not clause.met
        assert not clause.unmeasurable
        assert "unmeasurable" in clause.detail
        assert "ceiling 1" in clause.detail

    def test_an_unknown_future_version_is_also_unmeasurable(self, tmp_path) -> None:
        """A v3 producer this reader has never seen is refused the same way —
        the version check is the contract, not a v1 special case."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            payload["schema_version"] = "legacy-weekly-executions.v3"
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met
        assert clause.unmeasurable


# ── malformed v2 documents are red readings, not exceptions ─────────────────


class TestAMalformedVersionTwoDocumentIsARedReading:
    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda d: d.update(executions="three"), "`executions` is 'three'"),
            (lambda d: d["executions"].append("not-an-object"), "not an object"),
            (lambda d: d["executions"][0].update(name=""), "not an execution name"),
            (lambda d: d["executions"][0].update(status=None), "not a status"),
            (lambda d: d["executions"][0].update(duration_seconds=True), "not a duration"),
            (lambda d: d.update(executions_started=99), "disagrees with itself"),
            (lambda d: d.update(executions_started="3"), "not a count"),
        ],
        ids=[
            "executions is not a list",
            "an entry is not an object",
            "an empty name",
            "a null status",
            "a boolean duration",
            "the count contradicts the list",
            "the count is a string",
        ],
    )
    def test_each_shape_is_named_and_unmet(self, tmp_path, mutate, expected: str) -> None:
        """This document is written by a producer outside this repository. An
        exception here does not fail one clause: it propagates out of
        `evaluate` and takes `crucible gate`, `build_ladder` AND the board
        render down together."""

        def week(anchor: dt.date) -> dict[str, Any]:
            payload = _ruled_week(anchor)
            mutate(payload)
            return payload

        clause = _clause(_seed(tmp_path, week))
        assert not clause.met
        assert not clause.unmeasurable, "a malformed document is our producer's fault, not our read"
        assert expected in clause.detail


# ── phase 4 keeps counting every start ──────────────────────────────────────


class TestPhaseFourStillCountsEveryStart:
    """Decommissioned means the state machine emits NOTHING. Excluding
    Succeed-skips at phase 4 would have silently weakened it while fixing
    phase 0 — a surviving skip is evidence the trigger is still alive."""

    def test_a_week_of_skips_alone_is_unmet_at_the_zero_ceiling(self, tmp_path) -> None:
        def week(anchor: dt.date) -> dict[str, Any]:
            day = anchor.isoformat()
            return {
                "schema_version": LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
                "executions_started": 2,
                "executions": [
                    _execution("uuid_thursday", f"{day}T09:00:49+00:00", 3.0),
                    _execution("uuid_friday", f"{day}T09:00:49+00:00", 2.8),
                ],
                "source": "fixture",
            }

        store = _seed(tmp_path, week)
        phase4 = _clause(
            store,
            name="old_sf_execution_count_zero",
            maximum=0,
            minimum=0,
            skips_count_as_runs=True,
        )
        assert not phase4.met
        assert "2 starts, ceiling 0" in phase4.detail
        # The SAME week at phase 0's reading is UNMET for the OPPOSITE reason:
        # phase 4 sees two starts where it requires none, phase 0 sees zero
        # gate-passing runs where it requires one. Skips alone mean the gate
        # declined every day of the week, so the weekly pipeline never ran —
        # worse than a duplicate (`sf-pipeline-policy.md` §5). Before
        # `alpha-engine-config-I9962`'s lower bound this read MET, which is
        # the whole defect: absent graded as clean.
        phase0 = _clause(store)
        assert not phase0.met
        assert not phase0.unmeasurable
        assert "did not run this week" in phase0.detail

    def test_an_empty_week_meets_the_zero_ceiling(self, tmp_path) -> None:
        def week(_anchor: dt.date) -> dict[str, Any]:
            return {
                "schema_version": LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
                "executions_started": 0,
                "executions": [],
                "source": "fixture",
            }

        clause = _clause(
            _seed(tmp_path, week),
            name="old_sf_execution_count_zero",
            maximum=0,
            minimum=0,
            skips_count_as_runs=True,
        )
        assert clause.met


# ── policy §7.4: the guard is verified to FAIL without the fix ──────────────


def _superseded_reading(store: LocalStore, maximum: int = 1) -> bool:
    """The clause exactly as it read before `alpha-engine-config-I9962`.

    A byte-faithful reconstruction of the superseded rule — read
    `executions_started`, compare it to the ceiling — so the demonstration
    below compares two READINGS of one store rather than asserting the new
    one in isolation. Kept in the test file, never in `gate.py`: the retired
    metric must not be reachable from production code.
    """
    for day in WINDOW:
        key = legacy_weekly_executions_key(weekly_anchor(day).isoformat())
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        started = document.get("executions_started")
        if not isinstance(started, int) or started > maximum:
            return False
    return True


class TestTheFixChangesTheReading:
    """`policy §7.4`: a guard nobody has seen fail is a guard nobody knows
    works. The fixture is the week the ruling describes — one real run plus
    two sub-10s `WeeklyRunDayGate` Succeed-skips."""

    def test_the_same_week_reads_MET_after_and_UNMET_before(self, tmp_path) -> None:
        store = _seed(tmp_path, _ruled_week)
        after = _clause(store).met
        before = _superseded_reading(store)
        assert after is True, "the amended clause must accept the ruled week"
        assert before is False, "the superseded clause must have rejected it"
        assert after != before, (
            "the two readings are identical, so this fixture demonstrates nothing — "
            "the fix is not exercised by it"
        )

    def test_the_superseded_reading_would_have_called_it_three_starts(self, tmp_path) -> None:
        """Where the difference comes from, asserted rather than asserted-of:
        the retired metric sees three starts in a week whose real cycle count
        is one."""
        store = _seed(tmp_path, _ruled_week)
        document = json.loads(
            store.get_bytes(legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat())).decode(
                "utf-8"
            )
        )
        assert document["executions_started"] == 3
        assert len([e for e in document["executions"] if e["duration_seconds"] < 10]) == 2

    def test_every_mutation_this_module_makes_actually_changes_the_document(self) -> None:
        """The specific way a demonstration proves nothing silently: a fixture
        edit that matched nothing, leaving both readings taken over the same
        untouched document. Asserted here for the one fixture both readings
        share."""
        base = _ruled_week(FRIDAY)
        mutated = copy.deepcopy(base)
        mutated["executions"].append(
            _execution("uuid_second_run", f"{FRIDAY.isoformat()}T20:00:00+00:00", 4000.0)
        )
        mutated["executions_started"] = len(mutated["executions"])
        assert mutated != base
        assert mutated["executions_started"] != base["executions_started"]


# ── policy §7.4 for the 2026-09-04 amendment: one week, and a floor ─────────
#
# Brian ruled on 2026-09-04: "we can't wait a week on phase 0. it should clear
# after this week's weekly sf." Two changes went in together and only one of
# them is safe alone, so each is demonstrated as a DIFFERENCE between two
# readings of one store.
#
# What the second graded week bought was evidence that the disabled rerun
# issuer had not silently returned. That protection is not deleted, it is
# replaced by a CONTINUOUS one: `nousergon-data-PR1637` adds a
# `trigger-undeclared` finding to `automation_pause.py --check`, which runs
# daily and reports any live enabled trigger declared in neither manifest
# block. A standing detector beats one extra week of watching.

ONE_WEEK_WINDOW = [FRIDAY]
TWO_WEEK_WINDOW = [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(2))]


def _seed_window(tmp_path: Path, window: list[dt.date], document_for) -> LocalStore:
    store = LocalStore(tmp_path)
    for day in window:
        anchor = weekly_anchor(day)
        key = legacy_weekly_executions_key(anchor.isoformat())
        store.put_bytes(key, json.dumps(document_for(anchor)).encode("utf-8"))
    return store


def _skips_only_week(anchor: dt.date) -> dict[str, Any]:
    """A week in which `WeeklyRunDayGate` declined every day it was asked.

    Three sub-10s Succeed-skips and no cycle at all: the weekly pipeline never
    ran. `sf-pipeline-policy.md` §5 line 213 — "missing a weekly run is worse
    than a duplicate; the mutex handles duplicates".
    """
    day = anchor.isoformat()
    return {
        "schema_version": LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
        "executions_started": 3,
        "executions": [
            _execution("uuid_thursday", f"{day}T09:00:49.259000+00:00", 3.016),
            _execution("uuid_friday", f"{day}T09:00:49.312000+00:00", 3.907),
            _execution("uuid_saturday", f"{day}T09:00:49.100000+00:00", 2.8),
        ],
        "source": "states:ListExecutions, fixture",
    }


def _superseded_upper_bound_only(store: LocalStore, window: list[dt.date]) -> bool:
    """The amended clause as it stood BEFORE the lower bound — "at most one".

    A faithful reconstruction of the ruled predicate read literally: count the
    gate-passing executions, pass if there are no more than one. Zero passes,
    which is the defect. Kept in the test file, never in `gate.py`.
    """
    for day in window:
        key = legacy_weekly_executions_key(weekly_anchor(day).isoformat())
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        runs = [
            e
            for e in document["executions"]
            if not (
                e["status"] == "SUCCEEDED"
                and e["duration_seconds"] is not None
                and e["duration_seconds"] < WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS
            )
        ]
        if any(e["name"].startswith(LEGACY_WEEKLY_RERUN_NAME_PREFIX) for e in runs):
            return False
        if len(runs) > 1:
            return False
    return True


class TestTheOneWeekWindowChangesTheReading:
    """Demonstration (a): the WINDOW change, shown as a difference."""

    def test_one_clean_week_is_MET_at_one_week_and_not_at_the_old_two(self, tmp_path) -> None:
        store = _seed_window(tmp_path, ONE_WEEK_WINDOW, _ruled_week)
        after = _clause_old_weekly_within_cadence(store, ONE_WEEK_WINDOW, minimum=1)
        before = _clause_old_weekly_within_cadence(store, TWO_WEEK_WINDOW, minimum=1)
        assert after.met is True, "the ruled week must clear phase 0 on its own"
        assert before.met is False, "the superseded two-week window had a second week to fill"
        assert after.met != before.met, (
            "both windows read the same, so this fixture demonstrates nothing — the "
            "window change is not exercised by it"
        )
        # And the difference is the WINDOW, not a broken store: fill the second
        # week and the two-week reading agrees. Without this the test would pass
        # just as well against a store nothing could ever satisfy.
        filled = _seed_window(tmp_path / "filled", TWO_WEEK_WINDOW, _ruled_week)
        assert _clause_old_weekly_within_cadence(filled, TWO_WEEK_WINDOW, minimum=1).met

    def test_the_registered_phase_zero_window_is_one_and_phase_four_keeps_two(self) -> None:
        from crucible.gate import GATES

        assert GATES["phase0"][0] == 1
        assert GATES["phase4"][0] == 2


class TestTheLowerBoundChangesTheReading:
    """Demonstration (b): the FLOOR, which is what makes (a) safe.

    "At most one gate-passing execution per calendar week" is satisfied by
    ZERO. With a two-week window a silent week was unlikely; with one week it
    is one bad Saturday, and phase 0 would have exited on a week the weekly
    pipeline never ran.
    """

    def test_a_week_with_no_run_is_UNMET_now_and_was_MET_before(self, tmp_path) -> None:
        store = _seed_window(tmp_path, ONE_WEEK_WINDOW, _skips_only_week)
        after = _clause_old_weekly_within_cadence(store, ONE_WEEK_WINDOW, minimum=1)
        before = _superseded_upper_bound_only(store, ONE_WEEK_WINDOW)
        assert after.met is False, "a week with no weekly run must not read MET"
        assert before is True, "the upper-bound-only rule must have accepted it"
        assert after.met != before, (
            "the two readings are identical, so this fixture demonstrates nothing — "
            "the lower bound is not exercised by it"
        )
        # UNMET, not UNMEASURABLE. "We read the document and it lists no
        # gate-passing execution" is a reading; `[?]` is for a document we
        # could not obtain, and collapsing the two would hide a missing weekly
        # run behind the symbol an absent input gets.
        assert after.unmeasurable is False
        assert "did not run this week" in after.detail
        assert "floor 1" in after.detail

    def test_phase_four_still_reads_a_silent_week_as_MET(self, tmp_path) -> None:
        """The floor is a per-call-site bound, not a global rule. Phase 4
        grades a DECOMMISSIONED pipeline: no executions at all is the pass, so
        its call site passes `minimum=0`. A branch on `maximum == 0` inside the
        reader would have produced the same answer here while hiding one
        phase's semantics inside the other's."""
        store = _seed_window(tmp_path, TWO_WEEK_WINDOW, lambda a: _legacy_week_empty())
        phase4 = _clause_old_weekly_within_cadence(
            store,
            TWO_WEEK_WINDOW,
            name="old_sf_execution_count_zero",
            maximum=0,
            minimum=0,
            skips_count_as_runs=True,
        )
        assert phase4.met is True

    def test_an_unsatisfiable_bound_pair_is_refused_at_the_call_site(self, tmp_path) -> None:
        """`minimum > maximum` can never be met, and would read UNMET every
        week forever with a reason that looks like a finding about the
        pipeline. Loud, not a permanently red row."""
        store = _seed_window(tmp_path, ONE_WEEK_WINDOW, _ruled_week)
        with pytest.raises(ValueError, match="no week can satisfy"):
            _clause_old_weekly_within_cadence(store, ONE_WEEK_WINDOW, maximum=0, minimum=1)


def _legacy_week_empty() -> dict[str, Any]:
    return {
        "schema_version": LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
        "executions_started": 0,
        "executions": [],
        "source": "fixture",
    }


class TestTheAmendmentFixturesActuallyDiffer:
    """The specific way a §7.4 demonstration silently proves nothing: a
    fixture edit that matched nothing, leaving both readings taken over the
    same untouched document. Every mutation these two demonstrations rely on
    is asserted to have changed something."""

    def test_the_skips_only_week_really_differs_from_the_ruled_week(self) -> None:
        ruled = _ruled_week(FRIDAY)
        silent = _skips_only_week(FRIDAY)
        assert ruled != silent
        long_runs = [
            e
            for e in ruled["executions"]
            if e["duration_seconds"] >= WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS
        ]
        silent_runs = [
            e
            for e in silent["executions"]
            if e["duration_seconds"] >= WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS
        ]
        assert len(long_runs) == 1
        assert silent_runs == []

    def test_the_two_windows_really_name_different_evidence(self, tmp_path) -> None:
        store = _seed_window(tmp_path, TWO_WEEK_WINDOW, _ruled_week)
        one = _clause_old_weekly_within_cadence(store, ONE_WEEK_WINDOW, minimum=1)
        two = _clause_old_weekly_within_cadence(store, TWO_WEEK_WINDOW, minimum=1)
        assert set(one.evidence) < set(two.evidence)
        assert len(one.evidence) == 1 and len(two.evidence) == 2
