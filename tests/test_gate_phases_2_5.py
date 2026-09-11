"""Every plan §6 phase is graded BY CLAUSE, and no clause reads green on absence.

Normative source: plan §6 (one gate per phase, "a gate reads; it never runs"),
§12 rule 8 (instruments before product), and `alpha-engine-config-I9913`.

The defect these tests exist for: `GATES` registered `phase0` and `phase1`
only, so four of six ladder rows rendered with no instrument behind them.
Blank and "no data yet" render identically, so a reader could not tell whether
the instrument for phase 3 EXISTED or merely had nothing to read.

Two properties are asserted here and they pull in opposite directions, which
is the point:

1. **Structural** — every phase carries a registered clause list, and every
   registered gate is read by a phase. Neither direction alone is enough: the
   first lets a gate be registered that nothing renders, the second lets a
   phase go blank.
2. **Per clause** — each clause has a MET, an UNMET and an UNMEASURABLE
   reading, and the UNMEASURABLE case is the EMPTY store. A clause with no
   exercised unmeasurable path is one that will report a number when it should
   report that it could not look.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import pathlib

import pytest

import crucible.alerts as alerts_module
import crucible.autonomy as autonomy_module
import crucible.cost as cost_module
import crucible.gate as gate_module
import crucible.llm as llm_module
from crucible.gate import (
    GATES,
    MANIFEST_RUN_MODE_FIELD,
    MANIFEST_RUN_MODE_LIVE,
    PHASE2_AUTONOMY_MIN_DAILY_CYCLES,
    PHASE2_LIVE_SATURDAYS,
    PHASE2_MAX_PAGES,
    PHASE2_MAX_TAGGED_USD,
    PHASE2_WINDOW_WEEKS,
    PHASE4_MAX_TOTAL_USD,
    PHASES,
    REPOSITORY_GRADED_CLAUSES,
    autonomy_daily_cycles_in_span,
    autonomy_earliest_satisfiable_render_day,
    evaluate,
    weekly_anchor,
)
from crucible.keys import (
    champion_key,
    manifest_key,
    parse_bus_key,
    runs_prefix,
    strategy_arm_key,
    verdict_key,
)
from crucible.manifest import PREDECESSOR_SCHEMA_VERSION, RUN_MANIFEST_SCHEMA_VERSION
from crucible.release import POINTER_KEY
from crucible.slots import SLOTS
from crucible.store import LocalStore, S3Store

FRIDAY = dt.date(2026, 8, 28)
SHA = "a" * 40


def _window(weeks: int) -> list[dt.date]:
    return [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(weeks))]


#: The window `evaluate` hands every phase-2 clause — `PHASE2_WINDOW_WEEKS`
#: wide, NOT `PHASE2_LIVE_SATURDAYS` wide. The two were one constant until
#: `alpha-engine-config-I10324`, so a fixture built off the Saturday count read
#: identically and could not have caught the substitution.
PHASE2_WINDOW = _window(PHASE2_WINDOW_WEEKS)

#: A render day that is NOT a weekly anchor, and the window `_window` builds
#: off it. Weekly work binds to a Friday close while the ladder renders DAILY,
#: so a clause that keys on the render weekday and one that anchors are
#: indistinguishable when the fixture renders on a Friday — the shape that hid
#: `alpha-engine-config-I9904` from this suite (PR68 adversarial review, F1).
WEDNESDAY = dt.date(2026, 9, 9)
PHASE2_RENDER_WINDOW = [
    WEDNESDAY - dt.timedelta(weeks=n) for n in reversed(range(PHASE2_WINDOW_WEEKS))
]

#: Where a weekly run's manifest is actually filed for the render days the live
#: clause GRADES: the Friday close strictly before each, resolved through the
#: trading calendar. The tail slice mirrors the clause's own
#: `window[-PHASE2_LIVE_SATURDAYS:]` — the window is `PHASE2_WINDOW_WEEKS` wide
#: and only its last `PHASE2_LIVE_SATURDAYS` weeks are live-Saturday evidence.
PHASE2_ANCHORS = [weekly_anchor(day) for day in PHASE2_RENDER_WINDOW[-PHASE2_LIVE_SATURDAYS:]]


@pytest.fixture
def store(tmp_path) -> LocalStore:
    """An EMPTY store. Every unmeasurable case below reads this one."""
    return LocalStore(tmp_path)


def _put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------


class TestNoPlanPhaseCanRenderBlank:
    def test_every_plan_phase_carries_a_gate(self) -> None:
        """The guard that closes `alpha-engine-config-I9913` for good.

        Adding a phase to `PHASES` without writing its clause list is now a
        red CI run rather than a silent blank row on the board.
        """
        blank = [p.id for p in PHASES if p.gate is None]
        assert not blank, f"{blank} would render UNMEASURED — blank, not red"

    def test_every_registered_gate_is_read_by_a_phase(self) -> None:
        """The other direction. A gate no phase names is a clause list nobody
        renders — measured, and invisible."""
        assert set(GATES) == {p.gate for p in PHASES}

    @pytest.mark.parametrize("gate", ["phase2", "phase3", "phase4", "phase5"])
    def test_an_empty_store_grades_every_clause_and_meets_none(
        self, store: LocalStore, gate: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Against an empty store: clauses exist, none is met, and at least one
        says it could not be read rather than reporting a zero.

        The AWS readers are stubbed to RAISE — the reading identity has no
        `ce:GetCostAndUsage` and CI has no CloudTrail archive, and a test that
        depended on which of those two is true would pass for the wrong
        reason on a laptop that happens to hold credentials.

        `REPOSITORY_GRADED_CLAUSES` is excluded from the "none is met" half
        and ONLY from that half: an empty store says nothing about a clause
        whose artifact is the checkout, and asserting it unmet would be
        asserting that the runbook and the §2 suite are missing from the very
        tree the test is running in. The set is checked in the other direction
        immediately below, so it cannot become a place to park a green clause.
        """
        monkeypatch.setattr(gate_module, "_ce_client", _raising_client)
        monkeypatch.setattr(gate_module, "_s3_client", _raising_client)
        result = evaluate(store, gate=gate, trading_day=FRIDAY)
        assert result.clauses, f"{gate} was graded by zero clauses"
        assert not result.met
        assert [c for c in result.clauses if c.unmeasurable], (
            f"{gate} reported a reading for every clause against an EMPTY store"
        )
        for clause in result.clauses:
            if clause.name not in REPOSITORY_GRADED_CLAUSES:
                assert not clause.met
            assert clause.detail.strip(), f"{gate}/{clause.name} gave no reason"

    def test_every_repository_graded_clause_is_met_against_this_checkout(
        self, store: LocalStore
    ) -> None:
        """The other direction of the exclusion above, and the one that keeps
        it honest: each named clause must actually be MET here, with an empty
        store, because the artifact it grades is this tree. A clause added to
        the set to silence a red reading fails here instead."""
        graded = {
            clause.name: clause
            for gate in GATES
            for clause in evaluate(store, gate=gate, trading_day=FRIDAY).clauses
            if clause.name in REPOSITORY_GRADED_CLAUSES
        }
        assert set(graded) == set(REPOSITORY_GRADED_CLAUSES), (
            f"named but never rendered by any gate: "
            f"{sorted(set(REPOSITORY_GRADED_CLAUSES) - set(graded))}"
        )
        for name, clause in sorted(graded.items()):
            assert clause.met, f"{name} is not met against this checkout: {clause.detail}"


def _raising_client() -> object:
    raise RuntimeError("no credentials are configured for this reading identity")


# ---------------------------------------------------------------------------
# phase 2 — live Saturdays
# ---------------------------------------------------------------------------


def _schema_without_run_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build whose run-manifest schema declares no live/replay field.

    That was the real state until `run_manifest.v2` landed
    (alpha-engine-config-I9918), and the guard reading it is kept — a clause
    that answered from `crucible.gate`'s own constant rather than from the
    CONTRACT would grade every manifest as malformed on a build whose schema
    had lost the field. Patched rather than deleted so the unmeasurable branch
    stays exercised now that the real schema satisfies it.
    """
    monkeypatch.setattr(
        gate_module,
        "_manifest_property_names",
        lambda: frozenset({"status", "attempts"}),
    )


def _weekly(day: dt.date, *, mode: str = MANIFEST_RUN_MODE_LIVE, attempts: int = 1) -> dict:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": "ok",
        "reason": "",
        MANIFEST_RUN_MODE_FIELD: mode,
        "attempts": [{"n": n + 1} for n in range(attempts)],
    }


class TestLiveSaturdaysAreReadFromTheManifestNeverTheDate:
    """Every case here renders on a WEDNESDAY and files on the Friday anchors.

    The render day is deliberately not the anchor. `_window` steps back in raw
    calendar weeks, so a Friday render day makes the window days and the Friday
    anchors identical — a clause keyed on the render weekday would then pass
    every case in this class while being unsatisfiable on the four other
    weekdays the ladder renders on (PR68 adversarial review, F1;
    `alpha-engine-config-I9904`).
    """

    def test_the_fixture_days_and_the_render_window_are_different_keys(self) -> None:
        """The precondition every other case in this class depends on. If this
        ever passes trivially again, the class has stopped testing anything."""
        assert not set(PHASE2_ANCHORS) & set(PHASE2_RENDER_WINDOW)

    def test_unmeasurable_when_the_manifest_cannot_say_live_or_replay(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reading on a build whose schema declares no live/replay field:
        the clause refuses to answer rather than inferring liveness from the
        trading day — which the accelerated replay schedule would satisfy."""
        _schema_without_run_mode(monkeypatch)
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert MANIFEST_RUN_MODE_FIELD in clause.detail

    def test_the_shipped_schema_makes_this_clause_measurable(self, store: LocalStore) -> None:
        """The half `alpha-engine-config-I9918` actually closed: with NO
        monkeypatch, against the schema this build ships, the clause grades
        the store instead of reporting that it could not look."""
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.unmeasurable and not clause.met
        assert "never ran" in clause.detail

    def test_met_when_every_live_saturday_is_live_and_first_attempt_ok(
        self, store: LocalStore
    ) -> None:
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert clause.met and not clause.unmeasurable

    def test_the_same_live_saturdays_read_met_on_every_render_weekday(
        self, store: LocalStore
    ) -> None:
        """The regression that closes the class rather than the instance.

        The flawless live Saturdays are filed at their Friday closes and left
        alone; only the day somebody rendered the ladder moves. A clause
        windowed on the render weekday reads MET on the Friday and `never ran`
        on the other four — the board renders DAILY, so that is a contract
        unsatisfiable four days in five.
        """
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day))
        for render in (
            dt.date(2026, 9, 7),
            dt.date(2026, 9, 9),
            dt.date(2026, 9, 10),
            dt.date(2026, 9, 11),
        ):
            window = [render - dt.timedelta(weeks=n) for n in reversed(range(PHASE2_WINDOW_WEEKS))]
            clause = gate_module._clause_live_saturdays_first_attempt_ok(store, window)
            assert clause.met, f"{render.isoformat()}: {clause.detail}"

    def test_a_replay_is_unmet_not_met(self, store: LocalStore) -> None:
        """The whole reason the clause list was withheld. A perfect replay
        Saturday is UNMET here, and says which day was not live."""
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day, mode="replay"))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "not live" in clause.detail

    def test_a_manifest_predating_the_field_cannot_be_counted_live(self, store: LocalStore) -> None:
        """The grandfathering half. A weekly manifest written under
        `run_manifest.v1` is not malformed and is not condemned — it simply
        cannot establish liveness, so it reads UNMET with that said in terms.
        Counting it live on the strength of its date is the one thing this
        clause exists not to do.
        """
        for day in PHASE2_ANCHORS:
            document = _weekly(day)
            del document[MANIFEST_RUN_MODE_FIELD]
            document["schema_version"] = PREDECESSOR_SCHEMA_VERSION
            _put(store, manifest_key("weekly", day.isoformat()), document)
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "predate the live/replay field" in clause.detail
        assert PREDECESSOR_SCHEMA_VERSION in clause.detail

    def test_a_retried_run_is_not_a_first_attempt_ok(self, store: LocalStore) -> None:
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day, attempts=2))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met
        assert "retried" in clause.detail

    def test_an_absent_manifest_is_unmet_by_the_anchor_key_not_the_render_day(
        self, store: LocalStore
    ) -> None:
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met
        assert "never ran" in clause.detail
        for anchor in PHASE2_ANCHORS:
            assert manifest_key("weekly", anchor.isoformat()) in clause.detail
        for rendered in PHASE2_RENDER_WINDOW:
            assert manifest_key("weekly", rendered.isoformat()) not in clause.detail

    def test_the_clearing_date_is_the_next_qualifying_fridays_close(
        self, store: LocalStore
    ) -> None:
        """`alpha-engine-config-I10494` deliverable 1: one week past the
        anchor this clause already grades, resolved through the trading
        calendar — never `weekly_anchor` itself, which steps STRICTLY BEFORE
        its argument and would hand back the same anchor given a Friday
        input."""
        from crucible.calendar import resolve_trading_day

        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met
        expected = resolve_trading_day(
            dt.datetime.combine(PHASE2_ANCHORS[-1] + dt.timedelta(weeks=1), dt.time(23, 59))
        )
        assert clause.earliest_satisfiable == expected
        assert clause.earliest_satisfiable > PHASE2_ANCHORS[-1]

    def test_a_met_reading_carries_no_earliest_satisfiable_date(self, store: LocalStore) -> None:
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert clause.met
        assert clause.earliest_satisfiable is None


class TestReplaysReuseThePhaseOnePredicate:
    def test_the_clause_is_phase_ones_reading_renamed_over_five_saturdays(
        self, store: LocalStore
    ) -> None:
        """Not a restatement. The requirement quotes phase 1's own sentence and
        the window is the five replay Saturdays, so phase 2 and phase 1 cannot
        disagree about what a good replay is."""
        clause = gate_module._clause_replays_ok(store, PHASE2_RENDER_WINDOW, {})
        assert clause.name == "replays_ok"
        assert not clause.met
        # Anchored to weekly closes, not to the render weekday
        # (`alpha-engine-config-I9904`): a Wednesday render reads the five
        # Fridays strictly before it, and names none of the Wednesdays.
        anchors = gate_module.weekly_window(WEDNESDAY, gate_module.PHASE2_REPLAY_SATURDAYS)
        assert len(anchors) == gate_module.PHASE2_REPLAY_SATURDAYS
        assert all(day.weekday() == 4 for day in anchors)
        assert f"{anchors[0].isoformat()}..{anchors[-1].isoformat()}" in clause.requirement
        for anchor in anchors:
            assert manifest_key("data.weekly", anchor.isoformat()) in clause.evidence
        assert manifest_key("data.weekly", WEDNESDAY.isoformat()) not in clause.evidence

    def test_the_replay_window_is_the_same_on_every_render_weekday(self, store: LocalStore) -> None:
        clauses = [
            gate_module._clause_replays_ok(
                store,
                [WEDNESDAY + dt.timedelta(days=offset)],
                {},
            )
            for offset in (-2, -1, 0, 1, 2)  # Mon..Fri of the render week
        ]
        assert len({c.requirement for c in clauses}) == 1
        assert len({c.evidence for c in clauses}) == 1


# ---------------------------------------------------------------------------
# phase 2 — autonomy, pages, cost
# ---------------------------------------------------------------------------


def _archive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure an archive location so the reader gets past the "no trail"
    branch. A test-only value: no bucket name is written into this tree."""
    monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://a-test-archive/trail")


#: A store bucket and a CloudTrail archive that name nothing real. No
#: infrastructure identifier is written into this tree, tests included
#: (`tests/test_no_infra_literals.py`, `alpha-engine-config-I10156`).
TEST_BUCKET = "a-test-store"

#: The instant the graded system last changed, in every case that wants a
#: window long enough to grade: eighteen days before the render day, so the
#: minimum span clears and the render window's weekly close (2026-08-21) falls
#: after it. A fixed literal, never `today` arithmetic — a fixture whose
#: subject moves with the clock stops testing the same thing.
CHANGE_LONG_BEFORE = dt.datetime(2026, 8, 10, 9, 0, tzinfo=dt.UTC)

#: A change on the render day itself, an hour before the read. The case
#: `alpha-engine-config-I10324` requires to be unsatisfiable: without a
#: declared minimum span this leaves a one-hour window containing nothing and
#: the clause reads MET — a change making the clause EASIER.
CHANGE_AN_HOUR_BEFORE_THE_READ = dt.datetime(2026, 8, 28, 13, 0, tzinfo=dt.UTC)

#: A change seven calendar days before the render day, landing ON the window's
#: only weekly close. This is the case the deleted `PHASE2_AUTONOMY_MIN_SPAN =
#: 7 days` floor let through to the day and the CYCLE guard caught — the reason
#: `alpha-engine-config-I10327` could derive the span from the cycle
#: requirement instead of declaring it: the cycle guard was already the
#: binding one.
CHANGE_ON_THE_CYCLE_CLOSE = dt.datetime(2026, 8, 21, 0, 0, tzinfo=dt.UTC)


class _Counted:
    def __init__(self, count: int = 0, actions: tuple = ()) -> None:
        self.actions = actions
        self.count = count or len(actions)
        self.objects_read = 3
        self.records_scanned = 40


def _action(when: str, event_name: str = "UpdateStack") -> autonomy_module.OperatorAction:
    return autonomy_module.OperatorAction(
        event_time=when,
        event_name=event_name,
        event_source="cloudformation.amazonaws.com",
        principal="a-human",
        principal_type="IAMUser",
        request_id="r-1",
    )


class _HeadOnlyS3:
    """An S3 client that answers `head_object` and nothing else."""

    def __init__(self, response: dict | Exception) -> None:
        self._response = response

    def head_object(self, **_: object) -> dict:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _store_whose_pointer_flipped(at: dt.datetime | None, *, raises: Exception | None = None):
    """An `S3Store` whose pointer HeadObject reports ``at``.

    A real `S3Store` with a substituted client, not a stub of the store: the
    key the head is taken against (`store._s3_key(POINTER_KEY)`) is part of
    what is being tested, and a store stub would assert the reader's own
    idea of it.
    """
    response: dict | Exception = raises if raises is not None else {"LastModified": at}
    return S3Store(TEST_BUCKET, "crucible", client=_HeadOnlyS3(response))


class _DescribeOnlyCfn:
    def __init__(self, response: dict | Exception) -> None:
        self._response = response

    def describe_stacks(self, **_: object) -> dict:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _stack_applied(monkeypatch: pytest.MonkeyPatch, at: dt.datetime | None, **extra) -> None:
    stack: dict = {"LastUpdatedTime": at} if at is not None else {}
    stack.update(extra)
    monkeypatch.setattr(
        autonomy_module, "_cfn_client", lambda: _DescribeOnlyCfn({"Stacks": [stack]})
    )


def _cfn_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        autonomy_module,
        "_cfn_client",
        lambda: _DescribeOnlyCfn(RuntimeError("ExpiredToken")),
    )


def _counting(monkeypatch: pytest.MonkeyPatch, counted: _Counted) -> None:
    monkeypatch.setattr(gate_module, "_s3_client", lambda: object())
    monkeypatch.setattr(autonomy_module, "count_operator_actions", lambda *a, **k: counted)


class TestTheAutonomyWindowStartsAtTheSystemsLastChange:
    """`alpha-engine-config-I10324`. The clause used to read the phase's
    rolling calendar window, which straddled its own fixes: measured 2026-09-09
    it read 167 human mutating calls over 2026-09-01..09-08, dominated by the
    operator-gated applies that MADE THE SYSTEM WORK. Every case here is about
    where the window starts and what makes it refuse to be graded at all.
    """

    def test_the_later_of_the_two_inputs_starts_the_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stack apply changes the box's environment with no release flip —
        today's two fixes were exactly that shape — so the stack's instant must
        be able to win."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE - dt.timedelta(days=30))
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.met and not clause.unmeasurable
        assert CHANGE_LONG_BEFORE.isoformat() in clause.detail
        assert "stack last applied" in clause.detail

    def test_a_release_flip_with_no_stack_apply_also_starts_the_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other direction, and the reason both inputs are read: a wheel
        flip changes what the box RUNS while the stack is untouched."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE - dt.timedelta(days=30))
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.met
        assert "release pointer flip" in clause.detail
        assert CHANGE_LONG_BEFORE.isoformat() in clause.detail

    def test_the_reading_prints_both_inputs_whichever_won(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A window start nobody can reconstruct is a number, not a reading
        (principle 1). Both instants and the pointer key appear."""
        _archive(monkeypatch)
        earlier = CHANGE_LONG_BEFORE - dt.timedelta(days=3)
        _stack_applied(monkeypatch, earlier)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert earlier.isoformat() in clause.detail
        assert CHANGE_LONG_BEFORE.isoformat() in clause.detail
        assert POINTER_KEY in clause.evidence

    def test_a_change_an_hour_before_the_read_is_not_a_satisfiable_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard that keeps the rewrite from making the clause EASIER.

        With the window starting at the last change and no floor, an operator
        apply an hour before the read leaves an hour-long window containing
        nothing, and the clause reads MET — a change satisfying the clause it
        should reset. UNMET, not unmeasurable: the inputs read perfectly and
        what they said is that no cycle has run since.
        """
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_AN_HOUR_BEFORE_THE_READ)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_AN_HOUR_BEFORE_THE_READ)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "no complete weekly cycle has run unattended" in clause.detail
        satisfiable_on = autonomy_earliest_satisfiable_render_day(
            CHANGE_AN_HOUR_BEFORE_THE_READ.date()
        )
        assert satisfiable_on.isoformat() in clause.detail
        assert satisfiable_on > PHASE2_WINDOW[-1]
        # `alpha-engine-config-I10494` deliverable 1: the field, not just the
        # sentence, and it agrees with the same derivation.
        assert clause.earliest_satisfiable == satisfiable_on

    def test_a_seven_day_span_is_not_enough_the_cycle_must_close_after_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard that made the deleted seven-day constant redundant. A
        change landing exactly on the window's weekly close cleared that floor
        to the day and still has no COMPLETE unattended cycle behind it, so the
        cycle requirement was always the binding one — which is why
        `alpha-engine-config-I10327` could derive the span from it."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_ON_THE_CYCLE_CLOSE)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_ON_THE_CYCLE_CLOSE)
        assert PHASE2_WINDOW[-1] - CHANGE_ON_THE_CYCLE_CLOSE.date() == dt.timedelta(days=7)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "no complete weekly cycle has run unattended" in clause.detail

    def test_an_operator_apply_after_the_change_is_still_a_violation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is deliberately NO carve-out for operator-gated applies. Under
        this construction an apply DEFINES the window's start; one that happens
        after it is an intervention inside a window that was supposed to be
        hands-off, and it counts."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        _counting(monkeypatch, _Counted(actions=(_action("2026-08-20T12:00:00Z"),)))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "1 human mutating call" in clause.detail
        assert "UpdateStack" in clause.detail
        # `alpha-engine-config-I10494` deliverable 1: a fresh violation
        # INSIDE the window re-derives the floor from the offending call, not
        # from the stale change day — the call is 2026-08-20, after
        # CHANGE_LONG_BEFORE, so it is the later of the two that must win.
        assert clause.earliest_satisfiable is not None
        assert clause.earliest_satisfiable == autonomy_earliest_satisfiable_render_day(
            dt.date(2026, 8, 20)
        )

    def test_calls_on_the_change_day_that_predate_the_change_are_excluded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The archive is day-partitioned, so the change's own day arrives
        carrying the operator's applies. They are before the start instant and
        are not in the window."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        before = CHANGE_LONG_BEFORE - dt.timedelta(hours=2)
        _counting(monkeypatch, _Counted(actions=(_action(before.isoformat()),)))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.met and not clause.unmeasurable
        assert "1 call(s) on the change day itself predate the change" in clause.detail

    def test_an_unparseable_event_time_is_counted_not_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Over-counting is investigated; under-counting is a gate that reads
        clean because it could not read. Same direction
        `crucible.autonomy._touches` chose."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        _counting(monkeypatch, _Counted(actions=(_action("not-a-timestamp"),)))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable

    def test_unmeasurable_when_the_pointer_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never a window starting at zero: an unreadable pointer would give the
        widest possible window, which is the reading that looks most
        authoritative and is least established."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(None, raises=RuntimeError("Denied"))
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "last change could not be established" in clause.detail

    def test_unmeasurable_when_the_stack_cannot_be_described(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _archive(monkeypatch)
        _cfn_unreadable(monkeypatch)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "describe_stacks" in clause.detail

    def test_a_never_updated_stack_falls_back_to_its_creation_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stack nobody has updated since creation still HAS a last change,
        and it is the day it was created — not an error, and not zero."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, None, CreationTime=CHANGE_LONG_BEFORE)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE - dt.timedelta(days=1))
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.met
        assert CHANGE_LONG_BEFORE.isoformat() in clause.detail

    def test_a_naive_change_instant_is_refused_rather_than_assumed_utc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assuming UTC would move the window boundary by the local offset — a
        boundary error that reads as a clean number."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, dt.datetime(2026, 8, 10, 9, 0))
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "without a timezone" in clause.detail

    def test_a_local_store_cannot_establish_a_flip_instant(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `LocalStore` directory's mtime is a fact about a laptop, not a
        deploy. Grading a live autonomy window off it would be an answer about
        the workbench — the defect this clause was rewritten to stop."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "not S3" in clause.detail

    def test_unmeasurable_when_no_archive_is_configured(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`0` and `no trail` must never be the same answer. There is no
        default archive on purpose: a guessed bucket name produces a
        `NoSuchBucket` that reads like a permissions problem."""
        monkeypatch.delenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", raising=False)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "no CloudTrail archive is configured" in clause.detail

    def test_unmeasurable_when_the_archive_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        monkeypatch.setattr(gate_module, "_s3_client", _raising_client)
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "RuntimeError" in clause.detail


class TestTheMinimumSpanIsDerivedFromTheCycleRequirement:
    """`alpha-engine-config-I10327`. `-I10324` shipped BOTH a
    `PHASE2_AUTONOMY_MIN_SPAN = 7 days` constant and a "one complete weekly
    cycle after the change" requirement. For the attack the constant was
    written against they are redundant — one hour cannot contain a weekly
    cycle — and the independent constant put the earliest satisfiable render
    day four days past the weekly Step Function phase 2 must exit on.

    Brian's 2026-09-04 phase-0 ruling is the precedent: *"we can't wait a week
    on phase 0, it should clear after this week's weekly"*, where the second
    week's protection was replaced by a daily guard rather than deleted.
    """

    def test_no_independent_span_constant_survives(self) -> None:
        """The point of the change: the bar is stated in cycles, the unit the
        claim is denominated in, and not also in calendar days."""
        assert not hasattr(gate_module, "PHASE2_AUTONOMY_MIN_SPAN")

    def test_a_wednesday_change_is_satisfiable_on_that_weeks_saturday(self) -> None:
        """The arithmetic that matters. 2026-09-09 is the Wednesday the
        crucible-v2 stack was last applied; under the deleted seven-day floor
        the earliest satisfiable render day was 2026-09-16, four days past the
        2026-09-12 weekly."""
        change = dt.date(2026, 9, 9)
        assert autonomy_earliest_satisfiable_render_day(change) == dt.date(2026, 9, 12)
        assert change + dt.timedelta(days=7) > dt.date(2026, 9, 12)

    def test_the_derived_day_carries_a_weekly_close_after_the_change(self) -> None:
        change = dt.date(2026, 9, 9)
        day = autonomy_earliest_satisfiable_render_day(change)
        assert weekly_anchor(day) > change

    def test_the_derived_day_carries_the_required_daily_cycles(self) -> None:
        change = dt.date(2026, 9, 9)
        day = autonomy_earliest_satisfiable_render_day(change)
        assert autonomy_daily_cycles_in_span(change, day) >= PHASE2_AUTONOMY_MIN_DAILY_CYCLES

    def test_the_day_before_the_derived_day_satisfies_neither_requirement(self) -> None:
        """Derived means TIGHT: the day before is genuinely unsatisfiable, so
        this is the earliest and not merely a day that happens to work."""
        change = dt.date(2026, 9, 9)
        day = autonomy_earliest_satisfiable_render_day(change) - dt.timedelta(days=1)
        assert weekly_anchor(day) <= change

    def test_a_change_an_hour_before_the_read_is_still_unsatisfiable(self) -> None:
        """The anti-gaming property, asserted on the derivation itself and not
        only through the clause: a change on the render day cannot be graded on
        that day, whatever the constant was."""
        change = CHANGE_AN_HOUR_BEFORE_THE_READ.date()
        assert autonomy_earliest_satisfiable_render_day(change) > change

    def test_the_daily_cycle_count_skips_a_holiday(self) -> None:
        """Counted through the trading calendar, never derived from a
        calendar-day span — 2026-07-03 is an observed Independence Day, a
        CLOSED weekday, and a `weekday() < 5` count would score it."""
        assert autonomy_daily_cycles_in_span(dt.date(2026, 7, 2), dt.date(2026, 7, 3)) == 0

    def test_a_render_day_on_or_before_the_change_counts_no_daily_cycles(self) -> None:
        assert autonomy_daily_cycles_in_span(dt.date(2026, 9, 9), dt.date(2026, 9, 9)) == 0

    def test_the_daily_implication_is_self_checked_and_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The daily half is an IMPLICATION of the weekly one under the NYSE
        calendar — `weekly_anchor(render_day)` is itself a trading day in the
        span — so it is checked rather than assumed, and a violation reads
        UNMEASURABLE naming the contradiction. Never a quiet UNMET: that would
        render a calendar defect as a system finding."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        monkeypatch.setattr(gate_module, "autonomy_daily_cycles_in_span", lambda *_: 0)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "trading calendar contradicted itself" in clause.detail

    def test_a_met_reading_names_both_cycles_it_spanned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A window start nobody can reconstruct is a number, not a reading
        (principle 1) — and the same is true of the bar it cleared."""
        _archive(monkeypatch)
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        _counting(monkeypatch, _Counted(0))
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE)
        clause = gate_module._clause_zero_human_mutating_calls(store, PHASE2_WINDOW)
        assert clause.met
        assert "spanning the weekly cycle closing" in clause.detail
        assert "daily cycle(s)" in clause.detail


class TestTheSaturdayCountAndTheWindowWidthAreSeparateConstants:
    """`alpha-engine-config-I10324` defect 2. `PHASE2_LIVE_SATURDAYS` was read
    BOTH as the number of consecutive live Saturdays and, in `GATES["phase2"]`,
    as phase 2's window width in weeks — so narrowing the Saturday count would
    silently have narrowed the window `zero_human_mutating_calls`,
    `pages_within_ceiling` and `replays_ok` are counted over.
    """

    @staticmethod
    def _names_in_function(name: str) -> set[str]:
        tree = ast.parse(pathlib.Path(gate_module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        raise AssertionError(f"{name} is not a function in crucible.gate")

    @staticmethod
    def _names_in_gates_assignment() -> set[str]:
        tree = ast.parse(pathlib.Path(gate_module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = getattr(node, "targets", []) or (
                [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            if any(isinstance(t, ast.Name) and t.id == "GATES" for t in targets):
                return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        raise AssertionError("GATES is not assigned in crucible.gate")

    def test_the_window_width_is_not_reachable_from_the_live_saturday_clause(self) -> None:
        """Structural, and value-independent: the day the two numbers happen to
        be equal again, an assertion comparing them would pass while the
        substitution was back."""
        names = self._names_in_function("_clause_live_saturdays_first_attempt_ok")
        assert "PHASE2_LIVE_SATURDAYS" in names
        assert "PHASE2_WINDOW_WEEKS" not in names

    def test_the_saturday_count_is_not_reachable_from_the_window_registration(self) -> None:
        names = self._names_in_gates_assignment()
        assert "PHASE2_WINDOW_WEEKS" in names
        assert "PHASE2_LIVE_SATURDAYS" not in names

    def test_the_gate_window_is_the_window_constant_and_the_clause_grades_the_other(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behavioural counterpart: the window `evaluate` builds is
        `PHASE2_WINDOW_WEEKS` wide while the live clause names exactly
        `PHASE2_LIVE_SATURDAYS` manifest keys."""
        monkeypatch.setattr(gate_module, "_ce_client", _raising_client)
        monkeypatch.setattr(gate_module, "_s3_client", _raising_client)
        result = evaluate(store, gate="phase2", trading_day=FRIDAY)
        assert len(result.window) == PHASE2_WINDOW_WEEKS
        live = next(c for c in result.clauses if c.name == "live_saturdays_first_attempt_ok")
        assert len(live.evidence) == PHASE2_LIVE_SATURDAYS

    def test_the_page_ceiling_window_still_excludes_the_week_before(self) -> None:
        """What the split PROTECTS: `pages_within_ceiling` counts over the
        window, and a ceiling of two pages over one week asserts almost
        nothing. The window stayed two weeks while the Saturday count moved to
        one."""
        assert len(PHASE2_WINDOW) == PHASE2_WINDOW_WEEKS
        assert PHASE2_WINDOW[-1] - PHASE2_WINDOW[0] == dt.timedelta(weeks=PHASE2_WINDOW_WEEKS - 1)


class TestPagesAreCountedOnlyOnceSomethingHasSwept:
    def _sweep(self, store: LocalStore) -> None:
        _put(store, manifest_key("alerts.sweep", FRIDAY.isoformat()), {"status": "ok"})

    def test_an_empty_bus_with_no_sweep_is_unmeasurable_not_a_clean_month(
        self, store: LocalStore
    ) -> None:
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert runs_prefix("alerts.sweep") in clause.detail

    def test_met_when_the_sweep_ran_and_the_bus_is_under_the_ceiling(
        self, store: LocalStore
    ) -> None:
        self._sweep(store)
        _put(store, f"alerts/{FRIDAY.isoformat()}/one.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.met and not clause.unmeasurable

    def test_unmet_over_the_ceiling(self, store: LocalStore) -> None:
        self._sweep(store)
        for n in range(PHASE2_MAX_PAGES + 1):
            _put(store, f"alerts/{FRIDAY.isoformat()}/incident{n}.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable

    def test_the_clearing_date_is_the_oldest_offending_page_plus_the_window_span(
        self, store: LocalStore
    ) -> None:
        """`alpha-engine-config-I10494` deliverable 1. The window is inclusive
        of both ends (`window[0]..window[-1]`), so its span is
        `(window[-1] - window[0]).days + 1` — 8 calendar days at
        `PHASE2_WINDOW_WEEKS = 2` — and the clause clears the day the
        OLDEST offending page ages past that span, never a hardcoded 8 that
        could drift from the window it grades.
        """
        self._sweep(store)
        for n in range(PHASE2_MAX_PAGES + 1):
            _put(store, f"alerts/{FRIDAY.isoformat()}/incident{n}.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        window_span_days = (PHASE2_WINDOW[-1] - PHASE2_WINDOW[0]).days + 1
        assert window_span_days == 8
        assert clause.earliest_satisfiable == FRIDAY + dt.timedelta(days=window_span_days)

    def test_the_oldest_page_not_the_newest_sets_the_clearing_date(self, store: LocalStore) -> None:
        """Two offending pages on different days: the ceiling clears once the
        OLDER one ages out, dropping the count back to the ceiling — not once
        the newer one does."""
        self._sweep(store)
        older = PHASE2_WINDOW[0]
        for n in range(PHASE2_MAX_PAGES + 1):
            day = older if n == 0 else FRIDAY
            _put(store, f"alerts/{day.isoformat()}/incident{n}.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        window_span_days = (PHASE2_WINDOW[-1] - PHASE2_WINDOW[0]).days + 1
        assert clause.earliest_satisfiable == older + dt.timedelta(days=window_span_days)
        assert clause.earliest_satisfiable < FRIDAY + dt.timedelta(days=window_span_days)

    def test_a_met_reading_carries_no_earliest_satisfiable_date(self, store: LocalStore) -> None:
        self._sweep(store)
        _put(store, f"alerts/{FRIDAY.isoformat()}/one.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.met
        assert clause.earliest_satisfiable is None

    def test_the_gate_and_the_alerts_module_count_the_same_bus(self, store: LocalStore) -> None:
        """One implementation of "pages over a span", not two.

        The clause used to carry its own copy of the count, and the copies
        disagreed at the FIRST day of the window (`start <= day` against the
        module's `start < day`), so the gate and the
        `pages_per_20_trading_days` metric could report different numbers for
        the same bus (PR68 adversarial review, F4). An incident filed on
        exactly `window[0]` is the case that separates them.
        """
        self._sweep(store)
        edge = PHASE2_WINDOW[0]
        _put(store, f"alerts/{edge.isoformat()}/edge.json", {"severity": "page"})
        counted = alerts_module.pages_in_range(store, start=edge, end=PHASE2_WINDOW[-1])
        assert counted == [f"alerts/{edge.isoformat()}/edge.json"]
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.evidence == tuple(counted)
        assert f"{len(counted)} paged incident(s)" in clause.detail

    def test_a_key_that_is_not_a_bus_row_is_not_counted(self, store: LocalStore) -> None:
        """Parsed through `crucible.keys.parse_bus_key`, never by a positional
        index or an arity restated as an integer outside that module — the
        class where `len(parts) != 4` silently dropped every discriminated
        manifest (`alpha-engine-config-I9879`)."""
        assert parse_bus_key(f"alerts/{FRIDAY.isoformat()}/one.json") == (
            FRIDAY.isoformat(),
            "one",
        )
        assert parse_bus_key(f"alerts/{FRIDAY.isoformat()}/nested/one.json") is None
        assert parse_bus_key(f"alerts/{FRIDAY.isoformat()}/one.txt") is None
        assert parse_bus_key(f"runs/weekly/{FRIDAY.isoformat()}/run.json") is None
        self._sweep(store)
        _put(store, f"alerts/{FRIDAY.isoformat()}/nested/one.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.met and clause.evidence == ()
        assert f"ceiling {PHASE2_MAX_PAGES}" in clause.detail


class _CostClient:
    """A Cost Explorer stand-in speaking the real response shape.

    ``amount`` answers the MONTHLY month-to-date request; ``daily`` answers the
    DAILY trailing-window request with one period per day in the interval,
    cycling the list. The default daily figure is a quiet `$1.00`/day so the
    pre-existing month-to-date cases keep grading the month-to-date half.
    """

    def __init__(
        self,
        amount: str,
        daily: list[str] | None = None,
        allocation_tags: list[dict] | Exception | None = None,
    ) -> None:
        self.amount = amount
        self.daily = daily or ["1.00"]
        self.requests: list[dict] = []
        self._allocation_tags = allocation_tags

    def list_cost_allocation_tags(self, **request) -> dict:
        if isinstance(self._allocation_tags, Exception):
            raise self._allocation_tags
        return {"CostAllocationTags": self._allocation_tags or []}

    def get_cost_and_usage(self, **request) -> dict:
        self.requests.append(request)
        if request["Granularity"] == "DAILY":
            start = dt.date.fromisoformat(request["TimePeriod"]["Start"])
            end = dt.date.fromisoformat(request["TimePeriod"]["End"])
            days = (end - start).days
            return {
                "ResultsByTime": [
                    {"Total": {"UnblendedCost": {"Amount": self.daily[n % len(self.daily)]}}}
                    for n in range(days)
                ]
            }
        return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": self.amount}}}]}


class TestACostCeilingIsNeverMetByAnUnreadableApi:
    def test_unmeasurable_when_cost_explorer_is_denied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`ce:GetCostAndUsage` is not granted to the reading identity today.
        A denial must not read as `$0.00 — under the ceiling`."""
        monkeypatch.setattr(gate_module, "_ce_client", _raising_client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW, name="aws_cost_within_ceiling", ceiling_usd=40.0, tagged=True
        )
        assert clause.unmeasurable and not clause.met

    def test_a_tag_filtered_zero_with_the_key_never_activated_names_the_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An untagged estate and an estate whose key was never activated as
        a cost-allocation tag return the same `$0.00`, so the figure alone
        measures the FILTER rather than the spend. `alpha-engine-config-I10076`:
        the reason must name the activation state and, when it is not Active,
        the exact command to fix it — never point the reader at
        `audit_stack_tags`, which cannot answer this question."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("0"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met
        assert "absent as a cost-allocation tag" in clause.detail
        assert (
            "aws ce update-cost-allocation-tags-status "
            "--cost-allocation-tags-status TagKey=system,Status=Active" in clause.detail
        )

    def test_a_tag_filtered_zero_with_the_key_inactive_names_the_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _CostClient(
            "0",
            allocation_tags=[
                {"TagKey": "system", "Status": "Inactive", "LastUpdatedDate": "2026-09-01"}
            ],
        )
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met
        assert "Inactive as a cost-allocation tag" in clause.detail
        assert "TagKey=system,Status=Active" in clause.detail

    def test_a_tag_filtered_zero_with_the_key_active_reads_forward_indexing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`Active since <date>` is not a failure state — Cost Explorer
        indexes forward from activation, so a genuinely quiet window right
        after activation reads $0.00 for a while. The reason must say so
        rather than imply something is still broken."""
        client = _CostClient(
            "0",
            allocation_tags=[
                {"TagKey": "system", "Status": "Active", "LastUpdatedDate": "2026-09-06"}
            ],
        )
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met
        assert "Active since 2026-09-06" in clause.detail
        assert "indexes forward from activation" in clause.detail
        assert "genuinely zero or not yet indexed" in clause.detail

    def test_a_denied_list_cost_allocation_tags_is_unmeasurable_naming_the_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denied `ce:ListCostAllocationTags` must never be read as
        `Inactive` — that is a different finding with a different remedy."""
        error = RuntimeError("User is not authorized to perform ce:ListCostAllocationTags")
        client = _CostClient("0", allocation_tags=error)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met
        assert "ce:ListCostAllocationTags" in clause.detail
        assert "ce:ListCostAllocationTags" in clause.evidence

    def test_an_untagged_zero_is_unmeasurable_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same trap one level up, and the half that shipped untracked.

        For a live AWS estate an ACCOUNT total of exactly `$0.00` means Cost
        Explorer answered with nothing chargeable — a broken reading, not a
        free month — so phase 4's row must not go green on it (PR68
        adversarial review, F3).
        """
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("0"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.unmeasurable and not clause.met
        assert "no spend recorded for the whole account" in clause.detail.lower()

    def test_met_under_the_tagged_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read on the 28th. `month_to_date_usd` asks for [1st, today) —
        yesterday's close is the last complete day Cost Explorer has — so 27
        days are in; a quiet $1.00/day estate totals $30.00 over the trailing
        thirty, under $40."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("12.50"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.met and not clause.unmeasurable
        assert "$12.50 month-to-date" in clause.detail
        assert "27 of 31 days" in clause.detail
        assert "trailing 30 complete days (2026-07-29..2026-08-28) $30.00" in clause.detail
        assert "leading 7-day mean $1.00/day × 30 = $30.00" in clause.detail

    def test_a_month_boundary_lump_does_not_read_a_compliant_month_as_over(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR80 review, B1. This account posts `$15.59` on the 1st and
        `$1.83`/day after, closing at `$69.90` under a `$70` ceiling. A
        pro-rata line read that month OVER on days 1–26. Over thirty days the
        lump is one day in thirty: $15.59 + 29 x $1.83 = $68.66, under."""
        client = _CostClient("15.59", daily=["15.59"] + ["1.83"] * 29)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 2)],  # one complete day in: the lump
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "$68.66" in clause.detail
        assert clause.detail.endswith("under")

    def test_a_weekly_batch_estate_over_budget_is_unmet_not_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR80 round 2, B1. The estate's real shape: eight of eleven
        registry rows are Saturday-weekly and the Saturday spot run is the
        expensive compute. `$60` every seventh day and `$0.20` otherwise is
        ~`$245`/month; a trailing-week MEDIAN read it as a `$0.20`/day estate
        (MET, "projected $66") until month-to-date itself crossed the ceiling
        on the 10th. Thirty days' total carries four or five batch days."""
        client = _CostClient("60.40", daily=["60"] + ["0.20"] * 6)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 3)],  # two complete days in: one batch, one quiet
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable, clause.detail
        assert clause.detail.endswith("OVER")
        # 30 days = 5 batch days ($300) + 25 quiet days ($5.00); the leading
        # 7-day mean sees one batch day ($8.74/day × 30 = $262.29).
        assert "$305.00" in clause.detail
        assert "leading 7-day mean $8.74/day × 30 = $262.29" in clause.detail

    def test_accelerating_pace_is_unmet_while_trailing_total_is_still_under(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR80 round 3, I9927 FAIL demo. Day-5 MTD `$40` (~`$240`/mo implied)
        with a trailing-30 total of `$65` under a `$70` ceiling graded MET when
        only the trailing total was the verdict. Leading 7-day mean × 30 must
        hard-UNMET that acceleration — prefer a short false-UNMET after a
        month-boundary lump over a false-MET during a blowout."""
        # 22 × $0.39 + $0.42 + 7 × $8.00 = $65.00; leading mean $8 × 30 = $240.
        daily = ["0.39"] * 22 + ["0.42"] + ["8.00"] * 7
        client = _CostClient("40.00", daily=daily)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 6)],  # five complete days in: MTD $40
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable, clause.detail
        assert clause.detail.endswith("OVER (leading pace)")
        assert "$65.00" in clause.detail
        assert "leading 7-day mean $8.00/day × 30 = $240.00" in clause.detail

    def test_four_free_days_in_the_window_cannot_make_it_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A median over a window with four `$0.00` days degenerates to zero
        and a projection to month-to-date — a clause that cannot fail. A
        total cannot be pulled down by quiet days: the three `$30` days are
        still in it."""
        client = _CostClient("30.00", daily=["0"] * 4 + ["30"] * 3)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 3)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met, clause.detail

    def test_over_budget_with_one_quiet_day_is_unmet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _CostClient("6.10", daily=["3.00"] * 29 + ["0.10"])
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 3)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable
        assert "$87.10" in clause.detail
        assert clause.detail.endswith("OVER")

    def test_a_month_already_over_the_ceiling_is_unmet_whatever_the_trailing_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _CostClient("70.01", daily=["0.01"])
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 30)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable
        assert "already over" in clause.detail
        # The fast path: no trailing read was needed, and none was made.
        assert all(r["Granularity"] == "MONTHLY" for r in client.requests)

    def test_the_trailing_window_is_thirty_complete_days_and_may_cross_the_month(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The window is the ceiling's own period and a property of the
        estate, not of the calendar month it is read in — on the 2nd it
        reaches back into August, so the clause is measurable on day 1 and
        there is no "young month" branch."""
        client = _CostClient("1.50")
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 2)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.met and not clause.unmeasurable
        (daily,) = [r for r in client.requests if r["Granularity"] == "DAILY"]
        assert daily["TimePeriod"] == {"Start": "2026-08-03", "End": "2026-09-02"}
        assert "2026-08-03..2026-09-02" in clause.detail

    def test_the_tag_filter_reaches_the_trailing_read_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _CostClient("1.50")
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert len(client.requests) == 2
        assert all("Filter" in r for r in client.requests)

    def test_an_unreadable_trailing_window_is_unmeasurable_not_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Month-to-date readable and under, the daily read denied: the clause
        cannot grade the month and says so."""

        class _DailyDenied(_CostClient):
            def get_cost_and_usage(self, **request) -> dict:
                if request["Granularity"] == "DAILY":
                    raise RuntimeError("AccessDenied: ce:GetCostAndUsage DAILY")
                return super().get_cost_and_usage(**request)

        monkeypatch.setattr(gate_module, "_ce_client", lambda: _DailyDenied("1.50"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.unmeasurable and not clause.met
        assert "30-day total could not be read" in clause.detail
        assert "$1.50 month-to-date" in clause.detail

    def test_fewer_daily_periods_than_days_is_unmeasurable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Short(_CostClient):
            def get_cost_and_usage(self, **request) -> dict:
                if request["Granularity"] == "DAILY":
                    return {
                        "ResultsByTime": [
                            {"Total": {"UnblendedCost": {"Amount": "1.00"}}} for _ in range(3)
                        ]
                    }
                return super().get_cost_and_usage(**request)

        monkeypatch.setattr(gate_module, "_ce_client", lambda: _Short("1.50"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.unmeasurable and not clause.met
        assert "3 daily period(s)" in clause.detail

    def test_a_free_trailing_month_is_unmeasurable_not_a_zero_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Thirty days of exactly `$0.00` is what the tagged scope returns on
        an untagged estate — the `$0.00` trap on the trailing window."""
        client = _CostClient("1.50", daily=["0"])
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met
        assert "read exactly $0.00" in clause.detail

    def test_unmet_over_the_account_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Phase 4's reading: same reader, no tag filter, the other ceiling."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("230.21"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable
        assert "230.21" in clause.detail


class _ClosedMonthCostClient:
    """Distinguishes the month-to-date MONTHLY request from the closed-month
    one by the requested interval: `month_to_date_usd` always asks for fewer
    than 28 days (it is graded early in the month, which is the only time the
    closed-month reader ever fires); `closed_month_usd` always asks for a
    full prior calendar month, `>= 28` days starting on the 1st. DAILY always
    answers a flat, quiet `$1.00`/day so only the closed-month behaviour is
    under test."""

    def __init__(self, amount: str, closed_amount: str, *, estimated: bool = False) -> None:
        self.amount = amount
        self.closed_amount = closed_amount
        self.estimated = estimated
        self.requests: list[dict] = []

    def get_cost_and_usage(self, **request) -> dict:
        self.requests.append(request)
        start = dt.date.fromisoformat(request["TimePeriod"]["Start"])
        end = dt.date.fromisoformat(request["TimePeriod"]["End"])
        if request["Granularity"] == "DAILY":
            days = (end - start).days
            return {
                "ResultsByTime": [
                    {"Total": {"UnblendedCost": {"Amount": "1.00"}}} for _ in range(days)
                ]
            }
        if start.day == 1 and (end - start).days >= 28:
            return {
                "ResultsByTime": [
                    {
                        "Total": {"UnblendedCost": {"Amount": self.closed_amount}},
                        "Estimated": self.estimated,
                    }
                ]
            }
        return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": self.amount}}}]}


class TestAClosedCalendarMonthIsAlsoGraded:
    """alpha-engine-config-I9946: `month_to_date_usd` and `trailing_daily_usd`
    both grade a month IN PROGRESS. No render ever graded a COMPLETED
    calendar month before this — a month that projected under all the way
    through and then closed over would leave no red row anywhere. Measured
    before this landed: this whole class red, `_clause_aws_cost_within_ceiling`
    made no request whose interval reached 28 days on any render day."""

    def test_a_closed_month_over_the_ceiling_is_unmet_regardless_of_the_new_month(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _ClosedMonthCostClient(amount="0.50", closed_amount="80.00")
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 10, 1)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable, clause.detail
        assert "closed month 2026-09-01..2026-10-01" in clause.detail
        assert "$80.00" in clause.detail
        assert "over the $70.00" in clause.detail
        # Hard UNMET on the closed reading alone — no in-progress read needed.
        assert all(r["Granularity"] == "MONTHLY" for r in client.requests)

    def test_a_render_past_the_third_of_the_month_never_attempts_the_closed_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _CostClient("1.50")
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 15)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.met and not clause.unmeasurable
        assert "closed month" not in clause.detail
        # One month-to-date read, one trailing-30 read — no third request for
        # the closed-month figure at all, past the 3rd of the month.
        assert len(client.requests) == 2

    def test_a_provisional_closed_month_is_named_but_not_graded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cost Explorer finalises a month a few days into the next; a closed
        month still marked `Estimated: true` can still move, so it is stated
        in the detail rather than turning a render UNMET on a number that is
        not final yet."""
        client = _ClosedMonthCostClient(amount="0.50", closed_amount="80.00", estimated=True)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 10, 2)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "PROVISIONAL" in clause.detail
        assert "$80.00" in clause.detail

    def test_a_zero_closed_month_is_not_graded_as_a_free_month(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _ClosedMonthCostClient(amount="0.50", closed_amount="0")
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 10, 1)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "not graded" in clause.detail


# ---------------------------------------------------------------------------
# phase 3
# ---------------------------------------------------------------------------


class TestASlotHoldsItsPointerOnEvidenceOrSaysNothingLooked:
    def test_neither_artifact_is_unmeasurable_not_unmet(self, store: LocalStore) -> None:
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.unmeasurable and not clause.met
        assert champion_key("r") in clause.detail

    def test_met_on_an_evidence_won_promotion_whose_run_reads_ok(self, store: LocalStore) -> None:
        producing = manifest_key("promote", FRIDAY.isoformat())
        _put(
            store,
            champion_key("r"),
            {
                "arm_id": "r:momentum:ab12cd",
                "promotion_source": "evidence",
                "manifest_key": producing,
            },
        )
        _put(store, producing, {"status": "ok", "reason": "", "metrics": []})
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.met and not clause.unmeasurable

    @pytest.mark.parametrize(
        "render_day",
        [
            dt.date(2026, 8, 31),
            dt.date(2026, 9, 1),
            dt.date(2026, 9, 2),
            dt.date(2026, 9, 3),
            dt.date(2026, 9, 4),
        ],
        ids=["mon", "tue", "wed", "thu", "fri"],
    )
    def test_the_promote_manifest_is_found_on_every_render_weekday(
        self, store: LocalStore, render_day: dt.date
    ) -> None:
        """PR80 review, B2 — `alpha-engine-config-I9904` in phase 3. Promote
        runs on Saturday and files at the Friday close; over the raw render
        window the clause found `runs/promote/2026-08-28/run.json` on a
        Friday render only. Phase 3 is now in `WEEKLY_ANCHORED_GATES`."""
        producing = manifest_key("promote", FRIDAY.isoformat())
        _put(
            store,
            champion_key("r"),
            {
                "arm_id": "r:momentum:ab12cd",
                "promotion_source": "evidence",
                "manifest_key": producing,
            },
        )
        _put(store, producing, {"status": "ok", "reason": "", "metrics": []})
        result = evaluate(store, gate="phase3", trading_day=render_day)
        assert result.window[-1] == FRIDAY
        clause = next(
            c for c in result.clauses if c.name == "r_promotion_or_verdict_backed_non_promotion"
        )
        assert clause.met and not clause.unmeasurable, clause.detail

    def test_a_bootstrap_pointer_is_not_a_promotion_the_system_won(self, store: LocalStore) -> None:
        _put(
            store,
            champion_key("r"),
            {
                "arm_id": "r:momentum:ab12cd",
                "promotion_source": "bootstrap",
                "manifest_key": manifest_key("promote", FRIDAY.isoformat()),
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert not clause.met and not clause.unmeasurable
        assert "not `evidence`" in clause.detail

    def test_met_on_a_verdict_backed_non_promotion(self, store: LocalStore) -> None:
        """A pointer that did NOT move, with a stated reason, is the other way
        phase 3's clause is satisfied — the plan grades the DECISION, not the
        change."""
        from crucible.keys import arena_cycle_key

        _put(
            store,
            manifest_key("promote", FRIDAY.isoformat()),
            {
                "status": "ok",
                "reason": "",
                "metrics": [
                    {
                        "name": "pointer_moved",
                        "source_path": arena_cycle_key("r", FRIDAY.isoformat()),
                        "value": 0,
                        "status_reason": "the challenger lost 3 of 4 paired weeks",
                    }
                ],
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.met and not clause.unmeasurable

    def test_a_non_promotion_with_no_stated_reason_is_unmet(self, store: LocalStore) -> None:
        from crucible.keys import arena_cycle_key

        _put(
            store,
            manifest_key("promote", FRIDAY.isoformat()),
            {
                "status": "ok",
                "reason": "",
                "metrics": [
                    {
                        "name": "pointer_moved",
                        "source_path": arena_cycle_key("r", FRIDAY.isoformat()),
                        "value": 0,
                        "status_reason": "",
                    }
                ],
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert not clause.met and not clause.unmeasurable
        assert "status_reason" in clause.detail

    def test_a_filed_manifest_that_names_no_verdict_is_not_reported_as_absent(
        self, store: LocalStore
    ) -> None:
        """A promote run that was filed, read `ok`, and said nothing about this
        slot is a PRODUCER gap, not "nothing looked".

        Reporting it as "no promote run manifest was filed" states something
        false about the store — the manifest is right there — and principle 1
        asks that someone reconstruct why from the artifact alone (PR68
        adversarial review, F2).
        """
        from crucible.keys import arena_cycle_key

        promote = manifest_key("promote", FRIDAY.isoformat())
        _put(
            store,
            promote,
            {
                "status": "ok",
                "reason": "",
                "metrics": [
                    {
                        "name": "pointer_moved",
                        "source_path": arena_cycle_key("m", FRIDAY.isoformat()),
                        "value": 0,
                        "status_reason": "the challenger lost 3 of 4 paired weeks",
                    }
                ],
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.unmeasurable and not clause.met
        assert promote in clause.detail
        assert "pointer_moved" in clause.detail
        assert arena_cycle_key("r", FRIDAY.isoformat()) in clause.detail
        assert "no promote run manifest was filed" not in clause.detail

    def test_one_clause_per_registered_slot(self, store: LocalStore) -> None:
        clauses = gate_module._phase3(store, _window(4), {}, trading_day=FRIDAY)
        assert [c.name for c in clauses] == [
            f"{slot}_promotion_or_verdict_backed_non_promotion" for slot in sorted(SLOTS)
        ]


# ---------------------------------------------------------------------------
# phase 4
# ---------------------------------------------------------------------------


TRADER_EVIDENCE = "consumers/trader/v2_champion_week.json"


class TestTheTraderIsGradedThroughItsContractOrNotAtAll:
    def test_unmeasurable_while_the_contract_declares_no_artifact(self, store: LocalStore) -> None:
        """The harness may not reach into the trader, so an undeclared
        artifact is a missing CONTRACT, not a missing file."""
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert clause.unmeasurable and not clause.met
        assert "declares no consumer-evidence artifact" in clause.detail

    def test_unmet_once_declared_and_absent(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "TRADER_EVIDENCE_KEY", TRADER_EVIDENCE)
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert not clause.met and not clause.unmeasurable
        assert "is absent" in clause.detail

    def test_met_on_a_full_week(self, store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gate_module, "TRADER_EVIDENCE_KEY", TRADER_EVIDENCE)
        _put(store, TRADER_EVIDENCE, {"trading_days": 5})
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert clause.met and not clause.unmeasurable

    def test_unmet_on_a_short_week(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "TRADER_EVIDENCE_KEY", TRADER_EVIDENCE)
        _put(store, TRADER_EVIDENCE, {"trading_days": 2})
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert not clause.met and not clause.unmeasurable


def _legacy_week_v2(executions: list[dict]) -> dict:
    """A `legacy-weekly-executions.v2` week — the shape the producer files
    since `alpha-engine-config-I9962`. A `v1` document (one integer) is read as
    UNMEASURABLE at every ceiling, so a phase-4 fixture must be v2 too."""
    return {
        "schema_version": gate_module.LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
        "executions_started": len(executions),
        "executions": executions,
        "source": "fixture",
    }


def _v2_execution(name: str, duration_seconds: float | None, status: str = "SUCCEEDED") -> dict:
    return {
        "name": name,
        "start": "2026-08-22T09:00:49+00:00",
        "stop": None if duration_seconds is None else "2026-08-22T09:00:52+00:00",
        "duration_seconds": duration_seconds,
        "status": status,
    }


class TestDecommissionedMeansZeroNotWithinCadence:
    def test_the_phase_four_reading_is_the_phase_zero_reader_at_a_zero_ceiling(
        self, store: LocalStore
    ) -> None:
        """One reader, two ceilings. A second copy of this count is how phase 0
        and phase 4 would come to disagree about what a v1 start is."""
        clause = gate_module._clause_old_weekly_within_cadence(
            store,
            _window(2),
            name="old_sf_execution_count_zero",
            maximum=0,
            minimum=0,
            skips_count_as_runs=True,
        )
        assert clause.name == "old_sf_execution_count_zero"
        assert not clause.met
        assert "at most 0" in clause.requirement

    def test_one_start_is_unmet_at_the_phase_four_ceiling(self, store: LocalStore) -> None:
        from crucible.gate import legacy_weekly_executions_key, weekly_anchor

        for day in _window(2):
            _put(
                store,
                legacy_weekly_executions_key(weekly_anchor(day).isoformat()),
                _legacy_week_v2([_v2_execution("uuid_run", 18000.0)]),
            )
        clause = gate_module._clause_old_weekly_within_cadence(
            store,
            _window(2),
            name="old_sf_execution_count_zero",
            maximum=0,
            minimum=0,
            skips_count_as_runs=True,
        )
        assert not clause.met
        assert "ceiling 0" in clause.detail

    def test_zero_starts_is_met(self, store: LocalStore) -> None:
        from crucible.gate import legacy_weekly_executions_key, weekly_anchor

        for day in _window(2):
            _put(
                store,
                legacy_weekly_executions_key(weekly_anchor(day).isoformat()),
                _legacy_week_v2([]),
            )
        clause = gate_module._clause_old_weekly_within_cadence(
            store,
            _window(2),
            name="old_sf_execution_count_zero",
            maximum=0,
            minimum=0,
            skips_count_as_runs=True,
        )
        assert clause.met

    def test_the_registered_phase_four_clause_counts_succeed_skips(self, store: LocalStore) -> None:
        """The clause the PHASE reads, not one this test constructs.

        `alpha-engine-config-I9962` changed phase 0's metric to exclude
        `WeeklyRunDayGate` Succeed-skips. Phase 4 asks a different question —
        is the v1 pipeline DECOMMISSIONED — and a surviving Succeed-skip is
        evidence the trigger is still firing, so excluding them here would
        have silently weakened phase 4 while fixing phase 0.
        """
        from crucible.gate import legacy_weekly_executions_key, weekly_anchor

        for day in _window(2):
            _put(
                store,
                legacy_weekly_executions_key(weekly_anchor(day).isoformat()),
                _legacy_week_v2([_v2_execution("uuid_skip", 3.0)]),
            )
        (clause,) = [
            c
            for c in gate_module._phase4(store, _window(2), {}, trading_day=_window(2)[-1])
            if c.name == "old_sf_execution_count_zero"
        ]
        assert not clause.met, "a 3.0s Succeed-skip still means the v1 trigger is alive"
        assert "1 start, ceiling 0" in clause.detail


# ---------------------------------------------------------------------------
# phase 5
# ---------------------------------------------------------------------------


class _CallSite:
    """Enough of a registered call site for the membership test."""


class TestAnEmptyArmSetIsNeverAPass:
    def test_unmeasurable_while_no_llm_call_site_is_registered(self, store: LocalStore) -> None:
        """The empty-set trap, stated as a reading. A property over an empty
        set is vacuously true, and that is how a v1 row went green."""
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.unmeasurable and not clause.met
        assert "empty" in clause.detail

    def test_unmeasurable_while_an_arm_recipe_cannot_name_a_call_site(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second precondition, named separately so the reason is
        actionable: call sites exist, but the register cannot say which arms
        use one. The field exists on `main` since alpha-engine-config-I9920,
        so this is the reading a build that LOST it would give — forced by
        setting the constant back to `None`."""
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", None)
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.unmeasurable and not clause.met
        assert "not " in clause.detail

    def test_the_gate_reads_the_same_params_key_the_loader_validates(self) -> None:
        """One key, two modules: the gate restates it by value so it never
        imports the slot machinery; this is what keeps the two from drifting."""
        from crucible.slots.arms import LLM_CALLSITE_PARAM

        assert gate_module.LLM_ARM_CALLSITE_FIELD == LLM_CALLSITE_PARAM == "llm_callsite"

    def test_unmeasurable_when_no_active_arm_declares_one(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", "callsite_id")
        self._register(store, "r", [("plain", {})])
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.unmeasurable and not clause.met

    def test_unmet_when_an_llm_arm_has_no_verdict_in_the_cycle(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", "callsite_id")
        self._register(store, "r", [("llm", {"callsite_id": "research.rank"})])
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert not clause.met and not clause.unmeasurable
        assert "no verdict" in clause.detail

    def test_met_when_every_llm_arm_has_one(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", "callsite_id")
        arm_id = self._register(store, "r", [("llm", {"callsite_id": "research.rank"})])
        _put(store, verdict_key(arm_id, FRIDAY.isoformat()), {"status": "ok"})
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.met and not clause.unmeasurable

    def test_an_active_arm_with_no_synced_recipe_is_unmet_not_skipped(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The register says an arm is active; the strategy tree cannot say
        what it is. Silently treating it as a non-LLM arm is how the set this
        property quantifies over would lose exactly the arms it is for."""
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        self._register(store, "r", [("llm", {"callsite_id": "research.rank"})], sync=False)
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert not clause.met and not clause.unmeasurable
        assert "no arm recipes" in clause.detail or "no recipe" in clause.detail

    def test_the_m_recipe_has_no_place_for_a_call_site(self) -> None:
        """What justifies `LLM_ARM_RECIPE_SLOTS` omitting the M slot: the
        `ModelRecipe` contract carries no `params` and no call-site field, so
        no M arm can be an LLM arm by declaration. If either appears, this
        fails and the tuple is widened deliberately."""
        import dataclasses

        from crucible.slots.model import ModelRecipe

        names = {f.name for f in dataclasses.fields(ModelRecipe)}
        assert "params" not in names
        assert gate_module.LLM_ARM_CALLSITE_FIELD not in names
        assert set(gate_module.LLM_ARM_RECIPE_SLOTS) == {"u", "r"}

    @staticmethod
    def _register(
        store: LocalStore, slot: str, arms: list[tuple[str, dict]], *, sync: bool = True
    ) -> str:
        """Register each arm the way production does — a REAL `ArmSpec`, its
        id the hash of its spec, folded into the register (which carries only
        the `spec_hash`) — and sync its recipe into the store's strategy tree,
        which is where the gate reads `params.llm_callsite` from.

        The call-site field is `LLM_ARM_CALLSITE_FIELD` on the spec's params;
        a fixture key of `callsite_id` is rewritten to it so the cases above
        read as "this arm names that site".
        """
        from nousergon_lib.arena.arms import ArmRegister

        from crucible.slots import get_slot
        from crucible.slots.arms import ArmSpec, control_specs, write_register

        register = ArmRegister()
        # Production registers the slot's two controls beside every filed arm
        # (`crucible.slots.cycle`, both call sites), and controls are never in
        # the strategy tree — so the fixture carries them too, or the join
        # below is proven only on a register production never writes.
        for control in control_specs(get_slot(slot)):
            # `control=True` (`alpha-engine-config-I10044`): the gate's own
            # `is_control_arm(SLOTS[slot], a, register)` call is now
            # register-backed, so a fixture that wants the register's record
            # to say "control" — the way `register_arms` does in production —
            # has to register it that way, not rely on the name colliding
            # with `SlotSpec.control_arms`.
            register, _ = register.register(
                slot=slot,
                name=control.name,
                spec=control.spec,
                created_date=control.registered_at,
                control=True,
            )
        arm_id = ""
        for name, params in arms:
            field = gate_module.LLM_ARM_CALLSITE_FIELD
            params = {(field if k == "callsite_id" and field else k): v for k, v in params.items()}
            spec = ArmSpec(
                name=name,
                slot=slot,
                ranker="momentum_sleeve",
                params={"top_n": 8, **params},
                registered_at="2026-01-02",
            )
            register, _ = register.register(
                slot=slot, name=name, spec=spec.spec, created_date=spec.registered_at
            )
            arm_id = spec.arm_id
            if sync:
                lines = [
                    f"name: {name}",
                    f"slot: {slot}",
                    "ranker: momentum_sleeve",
                    "registered_at: '2026-01-02'",
                    "params:",
                ] + [f"  {k}: {json.dumps(v)}" for k, v in spec.params.items()]
                store.put_bytes(
                    strategy_arm_key(slot, name), ("\n".join(lines) + "\n").encode("utf-8")
                )
        write_register(store, slot, register)
        return arm_id


# ---------------------------------------------------------------------------
# the cost reader itself
# ---------------------------------------------------------------------------


class TestCostExplorerReadsRaiseRatherThanReturnZero:
    def test_a_denied_call_raises_rather_than_returning_zero(self) -> None:
        class _Denied:
            def get_cost_and_usage(self, **_):
                raise RuntimeError("AccessDeniedException")

        with pytest.raises(cost_module.CostUnreadableError) as caught:
            cost_module.month_to_date_usd(_Denied(), today=FRIDAY, tagged=True)
        assert "AccessDeniedException" in str(caught.value)

    def test_an_empty_response_is_not_a_spend_of_zero(self) -> None:
        class _Empty:
            def get_cost_and_usage(self, **_):
                return {"ResultsByTime": []}

        with pytest.raises(cost_module.CostUnreadableError):
            cost_module.month_to_date_usd(_Empty(), today=FRIDAY, tagged=False)

    def test_the_first_of_the_month_asks_for_a_non_empty_interval(self) -> None:
        """Cost Explorer's `End` is exclusive and must be strictly after
        `Start`, so a naive month-to-date reads `$0.00` on exactly one day a
        month."""
        captured: dict = {}

        class _Capturing:
            def get_cost_and_usage(self, **request):
                captured.update(request)
                return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": "1.00"}}}]}

        reading = cost_module.month_to_date_usd(
            _Capturing(), today=dt.date(2026, 9, 1), tagged=False
        )
        assert captured["TimePeriod"] == {"Start": "2026-09-01", "End": "2026-09-02"}
        assert reading.amount_usd == 1.0

    def test_the_tag_filter_is_the_only_difference_between_the_two_scopes(self) -> None:
        captured: list[dict] = []

        class _Capturing:
            def get_cost_and_usage(self, **request):
                captured.append(request)
                return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": "2.00"}}}]}

        client = _Capturing()
        cost_module.month_to_date_usd(client, today=FRIDAY, tagged=True)
        cost_module.month_to_date_usd(client, today=FRIDAY, tagged=False)
        assert "Filter" in captured[0]
        assert "Filter" not in captured[1]
        assert captured[0]["TimePeriod"] == captured[1]["TimePeriod"]


# ---------------------------------------------------------------------------
# `alpha-engine-config-I10494`: the gate-level projection and the setback
# comparison it enables.
# ---------------------------------------------------------------------------


class TestTheGateLevelEarliestSatisfiableDate:
    """Deliverable 2: the MAX over the unmet clauses' own dates, and the
    clause naming it — a VIEW over `GateResult.clauses`, never a second
    reduction."""

    def _clause(self, name: str, *, met: bool, earliest=None) -> gate_module.Clause:
        return gate_module.Clause(name, "req", met, "detail", (), earliest_satisfiable=earliest)

    def test_none_when_no_clause_carries_a_date(self) -> None:
        result = gate_module.GateResult(
            gate="phase2",
            trading_day=FRIDAY,
            window=[FRIDAY],
            clauses=[self._clause("a", met=True), self._clause("b", met=False)],
        )
        assert result.earliest_satisfiable is None
        assert result.earliest_satisfiable_clause is None

    def test_the_max_wins_and_names_its_own_clause(self) -> None:
        earlier = FRIDAY + dt.timedelta(days=3)
        later = FRIDAY + dt.timedelta(days=10)
        result = gate_module.GateResult(
            gate="phase2",
            trading_day=FRIDAY,
            window=[FRIDAY],
            clauses=[
                self._clause("early", met=False, earliest=earlier),
                self._clause("late", met=False, earliest=later),
            ],
        )
        assert result.earliest_satisfiable == later
        assert result.earliest_satisfiable_clause == "late"

    def test_met_clauses_never_contribute_a_date(self) -> None:
        met_but_dated = self._clause("stale", met=True, earliest=FRIDAY + dt.timedelta(days=99))
        result = gate_module.GateResult(
            gate="phase2", trading_day=FRIDAY, window=[FRIDAY], clauses=[met_but_dated]
        )
        # A MET clause carrying a stale date is a construction the real
        # clauses never produce (every dated branch above is an UNMET
        # branch), but the gate-level projection must not be fooled by one
        # if it ever did.
        assert result.earliest_satisfiable == FRIDAY + dt.timedelta(days=99)

    def test_the_artifact_carries_both_fields(self) -> None:
        earliest = FRIDAY + dt.timedelta(days=5)
        result = gate_module.GateResult(
            gate="phase2",
            trading_day=FRIDAY,
            window=[FRIDAY],
            clauses=[self._clause("setting", met=False, earliest=earliest)],
        )
        document = result.to_dict()
        assert document["earliest_satisfiable"] == earliest.isoformat()
        assert document["earliest_satisfiable_clause"] == "setting"
        assert document["clauses"][0]["earliest_satisfiable"] == earliest.isoformat()

    def test_render_names_the_date_and_the_clause(self) -> None:
        earliest = FRIDAY + dt.timedelta(days=5)
        result = gate_module.GateResult(
            gate="phase2",
            trading_day=FRIDAY,
            window=[FRIDAY],
            clauses=[self._clause("setting", met=False, earliest=earliest)],
        )
        assert f"exits no earlier than: {earliest.isoformat()} (set by setting)" in result.render()


class TestTheSetbackComparisonIsPureAndSilentUntilAWorseDate:
    """Deliverable 3: detection and the recorded comparison. NOT wired to a
    page — this class only exercises the comparison itself."""

    def _reading(self, earliest, clause_name="pages_within_ceiling") -> gate_module.GateResult:
        clauses = []
        if earliest is not None:
            clauses.append(
                gate_module.Clause(
                    clause_name, "req", False, "detail", (), earliest_satisfiable=earliest
                )
            )
        return gate_module.GateResult(
            gate="phase2", trading_day=FRIDAY, window=[FRIDAY], clauses=clauses
        )

    def test_no_previous_reading_is_silent(self) -> None:
        current = self._reading(FRIDAY + dt.timedelta(days=5))
        assert gate_module.detect_earliest_satisfiable_setback(None, current) is None

    def test_a_previous_reading_for_a_different_gate_is_silent(self) -> None:
        current = self._reading(FRIDAY + dt.timedelta(days=5))
        previous = {"gate": "phase1", "earliest_satisfiable": FRIDAY.isoformat()}
        assert gate_module.detect_earliest_satisfiable_setback(previous, current) is None

    def test_an_earlier_or_unchanged_date_is_silent(self) -> None:
        previous = {
            "gate": "phase2",
            "earliest_satisfiable": (FRIDAY + dt.timedelta(days=10)).isoformat(),
        }
        unchanged = self._reading(FRIDAY + dt.timedelta(days=10))
        earlier = self._reading(FRIDAY + dt.timedelta(days=3))
        assert gate_module.detect_earliest_satisfiable_setback(previous, unchanged) is None
        assert gate_module.detect_earliest_satisfiable_setback(previous, earlier) is None

    def test_a_current_reading_with_no_date_is_silent(self) -> None:
        """Every dated clause now reads MET, which cannot itself be a
        setback."""
        previous = {"gate": "phase2", "earliest_satisfiable": FRIDAY.isoformat()}
        current = self._reading(None)
        assert gate_module.detect_earliest_satisfiable_setback(previous, current) is None

    def test_a_later_date_is_a_setback_naming_the_clause_and_the_cause(self) -> None:
        previous = {
            "gate": "phase2",
            "earliest_satisfiable": FRIDAY.isoformat(),
        }
        moved_to = FRIDAY + dt.timedelta(days=7)
        current = self._reading(moved_to, clause_name="zero_human_mutating_calls")
        setback = gate_module.detect_earliest_satisfiable_setback(
            previous, current, cause="a stack apply"
        )
        assert setback is not None
        assert setback.gate == "phase2"
        assert setback.previous_date == FRIDAY
        assert setback.current_date == moved_to
        assert setback.clause == "zero_human_mutating_calls"
        assert setback.cause == "a stack apply"
        document = setback.to_dict()
        assert document["previous_earliest_satisfiable"] == FRIDAY.isoformat()
        assert document["current_earliest_satisfiable"] == moved_to.isoformat()


class TestLastSystemChangeProvenanceNeverRaises:
    def test_unmeasurable_becomes_a_named_reason_not_an_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _archive(monkeypatch)
        store = _store_whose_pointer_flipped(None, raises=RuntimeError("Denied"))
        provenance = gate_module.last_system_change_provenance(store)
        assert provenance is not None
        assert "could not be established" in provenance

    def test_a_readable_change_names_the_source_and_the_instant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stack_applied(monkeypatch, CHANGE_LONG_BEFORE)
        store = _store_whose_pointer_flipped(CHANGE_LONG_BEFORE - dt.timedelta(days=30))
        provenance = gate_module.last_system_change_provenance(store)
        assert provenance is not None
        assert "stack last applied" in provenance
        assert CHANGE_LONG_BEFORE.isoformat() in provenance
