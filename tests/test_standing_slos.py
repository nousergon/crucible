"""The three calendar-floored readings that stopped gating phase 2.

**Brian's ruling, 2026-09-13 (verbatim):** *"lets remove the time related
gates for phase 2, so this should close phase 2 now. lets then proceed as
recommended, advancing crucible v2 through phases 3 and 4 as actionable."*
Context: no outside users, paper trading; ship behind measured SLOs, gate only
irreversible steps on evidence.

`live_saturdays_first_attempt_ok`, `zero_human_mutating_calls` and
`pages_within_ceiling` are the three phase-2 clauses whose floor was a
CALENDAR — none of them could be turned green by anything the system did, only
by waiting. They left the exit gate. They did NOT stop being measured, and
this file is what makes the second half true: the removal and the retention
are asserted together, because a ruling that says "keep measuring" is
indistinguishable from a deletion the day nothing renders the numbers.

Three properties, one per way this could silently rot:

* the phase-2 clause set is EXACTLY the remaining seven — a later edit that
  re-adds a calendar floor, or drops one of the seven, is a red run;
* all three readings render as board rows, in a source of their own;
* the autonomy clause's anti-gaming properties survive its removal — it is
  still contained, still derives its own `earliest_satisfiable`, and the row
  that renders it is never green over a count nobody could take.
"""

from __future__ import annotations

import datetime as dt

import pytest

import crucible.board as board_module
import crucible.gate as gate_module
from crucible.board import (
    RED_STATES,
    SOURCES,
    STANDING_SLO_DECLARATIONS,
    HumanTouchReading,
    _check_standing_declarations,
    _standing_rows,
    build_board,
)
from crucible.gate import (
    CLAUSE_FUNCTION_PREFIX,
    GATE_DELIVERABLES,
    GATES,
    STANDING_SLOS,
    evaluate,
    standing_slo_clauses,
)
from crucible.store import LocalStore

#: A Friday, so `weekly_anchor` resolves without a holiday walk-back.
FRIDAY = dt.date(2026, 9, 11)

#: The seven clauses phase 2 grades after the ruling, in registration order.
#: Written out rather than derived from the gate it is testing — a set derived
#: from the thing under test asserts nothing about it.
PHASE2_CLAUSES_AFTER_THE_RULING: tuple[str, ...] = (
    "replays_ok",
    "pages_commissioned",
    "two_page_conditions_on_real_channel",
    "transient_retry_class_in_runner",
    "fault_injection_against_scheduled_path",
    "runbook_in_readme",
    "aws_cost_within_ceiling",
)

#: The three readings the ruling took off the gate.
REMOVED_FROM_THE_GATE: tuple[str, ...] = (
    "live_saturdays_first_attempt_ok",
    "zero_human_mutating_calls",
    "pages_within_ceiling",
)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _touch(count: int = 0, *, measured: bool = True) -> HumanTouchReading:
    return HumanTouchReading(
        month="2026-09",
        count=count,
        actions=(),
        detail=f"{count} human mutating call(s) over 2026-09-01..2026-09-11",
        measured=measured,
    )


class TestThePhase2ClauseSetIsExactlyTheSeven:
    def test_the_gate_grades_the_seven_and_nothing_else(self, store, monkeypatch) -> None:
        monkeypatch.setattr(gate_module, "_ce_client", _raising)
        monkeypatch.setattr(gate_module, "_s3_client", _raising)
        result = evaluate(store, gate="phase2", trading_day=FRIDAY)
        assert tuple(c.name for c in result.clauses) == PHASE2_CLAUSES_AFTER_THE_RULING

    @pytest.mark.parametrize("name", REMOVED_FROM_THE_GATE)
    def test_no_registered_gate_grades_a_removed_clause(self, name, store, monkeypatch) -> None:
        """Removed from phase 2, and not quietly re-homed on another rung."""
        monkeypatch.setattr(gate_module, "_ce_client", _raising)
        monkeypatch.setattr(gate_module, "_s3_client", _raising)
        for gate in GATES:
            result = evaluate(store, gate=gate, trading_day=FRIDAY)
            assert name not in {c.name for c in result.clauses}

    def test_the_clause_functions_still_exist(self) -> None:
        """Removed from the gate, never deleted: the ruling kept the readings,
        and a deletion is the change that stops measuring."""
        for name in REMOVED_FROM_THE_GATE:
            assert callable(getattr(gate_module, f"{CLAUSE_FUNCTION_PREFIX}{name}"))

    def test_coverage_still_grades_all_six_deliverables(self, store, monkeypatch) -> None:
        """None of the three ever graded a declared deliverable, so the
        coverage line is unchanged by their removal. That is the property
        making this a change of CONSEQUENCE rather than of coverage — and it
        is asserted rather than asserted-in-a-comment."""
        monkeypatch.setattr(gate_module, "_ce_client", _raising)
        monkeypatch.setattr(gate_module, "_s3_client", _raising)
        result = evaluate(store, gate="phase2", trading_day=FRIDAY)
        total = len(GATE_DELIVERABLES["phase2"])
        # The tracker reference is DERIVED from `gate.PHASES`, never written
        # as a literal (`alpha-engine-config-I9839`, enforced by
        # `tests/test_no_stale_tracker_literals.py`).
        tracker = next(p.tracker for p in gate_module.PHASES if p.gate == "phase2")
        assert result.coverage == f"grades all {total} of {total} {tracker} deliverables"


class TestTheStandingReader:
    def test_it_returns_exactly_the_declared_slos(self, store) -> None:
        names = tuple(c.name for c in standing_slo_clauses(store, trading_day=FRIDAY))
        assert names == STANDING_SLOS

    def test_it_reads_the_same_window_width_the_gate_does(self, store, monkeypatch) -> None:
        """The ceiling in `pages_within_ceiling`'s requirement is denominated
        in phase 2's window. A standing row counted over a different span
        would publish a different claim under the same name."""
        seen: list[list[dt.date]] = []
        original = gate_module._window

        def _record(trading_day, weeks):
            window = original(trading_day, weeks)
            seen.append(window)
            return window

        monkeypatch.setattr(gate_module, "_window", _record)
        standing_slo_clauses(store, trading_day=FRIDAY)
        assert seen and all(len(w) == gate_module.PHASE2_WINDOW_WEEKS for w in seen)

    def test_a_reader_returning_the_wrong_set_is_refused(self, store, monkeypatch) -> None:
        """The guard itself is made to fire — a detector nobody has made fail
        is a detector nobody knows works."""
        monkeypatch.setattr(gate_module, "STANDING_SLOS", ("something_else",))
        with pytest.raises(gate_module.ClauseMisconfiguredError):
            standing_slo_clauses(store, trading_day=FRIDAY)


class TestTheThreeRowsRender:
    def test_standing_is_a_declared_source(self) -> None:
        assert "standing" in SOURCES

    def test_all_three_rows_are_on_a_built_board(self, store) -> None:
        board = build_board(store, trading_day=FRIDAY, human_touch=_touch())
        ids = {row.id for row in board.rows}
        assert "standing:human_touch_count" in ids
        for name in STANDING_SLOS:
            assert f"standing:{name}" in ids

    def test_every_standing_row_names_an_artifact_and_says_what_red_means(self, store) -> None:
        rows = _standing_rows(store, FRIDAY.isoformat(), _touch())
        assert len(rows) == 3
        for row in rows:
            assert row.source == "standing"
            assert row.artifact.strip()
            assert row.means_when_red.strip()

    def test_an_unmet_standing_row_is_red_and_gates_no_phase(self, store) -> None:
        """The whole point of the ruling: the number still reads RED, and the
        phase row beside it is unaffected by it."""
        rows = {r.id: r for r in _standing_rows(store, FRIDAY.isoformat(), _touch(count=3))}
        assert rows["standing:human_touch_count"].state == "UNMET"
        assert rows["standing:human_touch_count"].red
        # An empty store has never run a live Saturday, so the live row is red too.
        assert rows["standing:live_saturdays_first_attempt_ok"].state in RED_STATES

    def test_a_count_nobody_could_take_is_never_green(self, store) -> None:
        """`measured=False` is "we could not check", and a zero we could not
        verify must never render as a clean month."""
        row = next(
            r
            for r in _standing_rows(store, FRIDAY.isoformat(), _touch(measured=False))
            if r.id == "standing:human_touch_count"
        )
        assert row.state == "UNMEASURABLE"
        assert row.red

    def test_the_autonomy_row_never_names_the_archive_uri(self, store, monkeypatch) -> None:
        """The board is published; the archive bucket is an infrastructure
        identifier this tree does not carry."""
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://a-real-looking-bucket/trail")
        row = next(
            r
            for r in _standing_rows(store, FRIDAY.isoformat(), _touch())
            if r.id == "standing:human_touch_count"
        )
        assert "a-real-looking-bucket" not in row.artifact
        assert "a-real-looking-bucket" not in row.detail

    def test_the_row_reuses_the_reading_rather_than_walking_the_archive_again(
        self, store, monkeypatch
    ) -> None:
        """One archive, one number. A second walk would put two answers to
        one question on one page."""

        def _forbidden(*_a, **_k):
            raise AssertionError("the standing row must not re-read the CloudTrail archive")

        monkeypatch.setattr(board_module, "_read_human_touch_count", _forbidden)
        rows = _standing_rows(store, FRIDAY.isoformat(), _touch())
        assert any(r.id == "standing:human_touch_count" for r in rows)

    def test_a_declaration_table_out_of_step_with_the_slos_is_refused(self) -> None:
        with pytest.raises(ValueError, match="renders nowhere"):
            _check_standing_declarations(("live_saturdays_first_attempt_ok", "a_new_slo"))

    def test_every_declared_slo_has_a_row_declaration(self) -> None:
        assert set(STANDING_SLOS) == set(STANDING_SLO_DECLARATIONS)


class TestTheAutonomyClauseSurvivesAsAReading:
    """The anti-gaming properties are what the clause is FOR; losing them to
    a change that only removed a consequence would be the real regression."""

    def test_it_is_absent_from_the_standing_reader_by_design(self) -> None:
        """Not an omission: `crucible.board._read_human_touch_count` reads the
        same archive through the same counter over a trailing month."""
        assert "zero_human_mutating_calls" not in STANDING_SLOS

    def test_it_still_derives_its_own_earliest_satisfiable_floor(self) -> None:
        """`autonomy_earliest_satisfiable_render_day` is the anti-gaming
        property: a change an hour before the read still yields a window with
        no complete weekly cycle in it."""
        change = dt.date(2026, 9, 10)
        day = gate_module.autonomy_earliest_satisfiable_render_day(change)
        assert day > change
        assert gate_module.weekly_anchor(day) > change

    def test_it_is_still_contained_like_every_other_clause(self) -> None:
        """`_contain_clause_exceptions` wraps every `_clause_*` in the module,
        registered or not — so an unregistered clause read by a board row
        still cannot take the render down."""
        clause = gate_module._clause_zero_human_mutating_calls(_RaisingStore(), [FRIDAY])
        assert clause.unmeasurable is True
        assert clause.met is False


class _RaisingStore:
    def list_keys(self, *_a, **_k):
        raise RuntimeError("denied")

    def read(self, *_a, **_k):
        raise RuntimeError("denied")


def _raising(*_a, **_k):
    raise RuntimeError("no AWS client is constructed in these tests")
