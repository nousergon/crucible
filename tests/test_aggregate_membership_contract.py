"""No aggregate hides a member — `alpha-engine-config-I10417`.

Two measured v1 failures, one class: the evaluator's report card disagreed
with itself by 37.8 points (two graders writing one field name), and a
weighted tile scored RED/F was silently excluded from the headline grade.
Both are reductions that dropped a member.

One contract test per reducer, per the issue's closes-when:

* `crucible.report.attribution_grade` — the report-card grade.
* `crucible.board._read_attribution` — board status, reduced from the same
  rows.
* `crucible.gate.GateResult` — one gate's clause reading.
* `crucible.gate.Ladder` / `build_ladder` — the phase ladder.

Each test asserts (1) `members[]` is present and shaped `id`/`value`/`status`,
(2) a member that is failed/unmeasured/null propagates — the parent is never
greener than its worst member, and (3) the parent is reconstructible from
`members[]` alone. `TestBoardNeverRendersMetOverARedMember` is the fixture the
issue's own closes-when names: it fails against the reader as it stood before
this PR (a RED row did not prevent `attribution_table_is_complete` reading
MET) and passes against the fix in `crucible/board.py`.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.aggregation import MemberRow, member_dicts, worst_member
from crucible.board import Declaration, read_declaration
from crucible.gate import CLAUSE_MEMBER_RANK, Clause, GateResult, Ladder, PhaseRow
from crucible.gate import PHASES as GATE_PHASES
from crucible.report import GRADE_RANK, attribution_grade, attribution_members
from crucible.store import LocalStore

# ── crucible.aggregation itself ─────────────────────────────────────────


class TestWorstMember:
    def test_an_empty_reduction_raises_rather_than_returning_a_verdict(self) -> None:
        """Principle 7: a reduction over nothing has nothing to be
        no-greener-than, so it must never render a parent status at all."""
        with pytest.raises(ValueError, match="zero members"):
            worst_member([], rank={"GREEN": 0})

    def test_a_status_with_no_declared_rank_raises_rather_than_defaulting_to_best(
        self,
    ) -> None:
        members = [MemberRow(id="a", value=1, status="MYSTERY")]
        with pytest.raises(ValueError, match="MYSTERY"):
            worst_member(members, rank={"GREEN": 0})

    def test_the_worst_ranked_member_wins_regardless_of_input_order(self) -> None:
        rank = {"GREEN": 0, "WATCH": 1, "RED": 2}
        members = [
            MemberRow(id="a", value=1, status="GREEN"),
            MemberRow(id="b", value=2, status="RED"),
            MemberRow(id="c", value=3, status="WATCH"),
        ]
        assert worst_member(members, rank=rank).id == "b"

    def test_member_dicts_renders_the_id_value_status_shape(self) -> None:
        members = [MemberRow(id="a", value=0.5, status="GREEN")]
        assert member_dicts(members) == [{"id": "a", "value": 0.5, "status": "GREEN"}]


# ── crucible.report: the report-card grade ─────────────────────────────


def _rows(*statuses: str) -> list[dict]:
    return [
        {"name": f"row{i}", "value": 0.5, "status": status} for i, status in enumerate(statuses)
    ]


class TestAttributionGrade:
    def test_all_green_rows_grade_green(self) -> None:
        grade, _ = attribution_grade(_rows("GREEN", "GREEN", "GREEN"))
        assert grade == "GREEN"

    def test_a_red_member_forces_the_grade_red_not_green(self) -> None:
        """The exact class the issue names: a red tile must never be
        excluded from the headline grade."""
        rows = _rows("GREEN", "GREEN", "RED", "GREEN", "GREEN")
        grade, reason = attribution_grade(rows)
        assert grade == "RED"
        assert "row2" in reason

    def test_an_unverified_member_ranks_worse_than_red_never_better(self) -> None:
        """`null`/`UNVERIFIED` must never render better than a real failure
        (I10417 clause 2) — "we could not check" is not "we checked and it
        passed"."""
        rows = _rows("RED", "N/A-NOT-RUN")
        grade, _ = attribution_grade(rows)
        assert grade == "UNVERIFIED"

    def test_watch_outranks_green_but_not_red(self) -> None:
        assert attribution_grade(_rows("GREEN", "WATCH"))[0] == "WATCH"
        assert attribution_grade(_rows("WATCH", "RED"))[0] == "RED"

    def test_members_are_reconstructible_from_the_rows_alone(self) -> None:
        """Clause 3: recomputing from `members[]` alone reproduces the grade."""
        rows = _rows("GREEN", "RED", "WATCH")
        members = attribution_members(rows)
        assert member_dicts(members) == [
            {"id": "row0", "value": 0.5, "status": "GREEN"},
            {"id": "row1", "value": 0.5, "status": "RED"},
            {"id": "row2", "value": 0.5, "status": "WATCH"},
        ]
        worst = worst_member(members, rank=GRADE_RANK)
        grade, _ = attribution_grade(rows)
        assert worst.status == grade

    def test_every_krepis_status_is_ranked(self) -> None:
        """`GRADE_RANK` must cover the whole vocabulary `_row` can emit — an
        unranked status must raise (`worst_member`), not silently sort as
        best."""
        from typing import get_args

        from krepis.metrics import StatusLiteral

        assert set(get_args(StatusLiteral)) <= set(GRADE_RANK)

    def test_build_attribution_publishes_members_grade_and_grade_reason(self, tmp_path) -> None:
        """Closes-when: `report/{date}/attribution.json` carries `members[]`."""
        from crucible.report import build_attribution

        store = LocalStore(tmp_path)
        document, _ = build_attribution(
            store,
            trading_day=dt.date(2026, 8, 28),
            now=dt.datetime(2026, 8, 29, tzinfo=dt.UTC),
            run_id="R" * 26,
        )
        assert "members" in document
        assert "grade" in document
        assert "grade_reason" in document
        assert [m["id"] for m in document["members"]] == [r["name"] for r in document["rows"]]
        # An empty store: every row is not-measured, so the grade must read
        # UNVERIFIED, never GREEN — principle 7 restated for the grade field.
        assert document["grade"] == "UNVERIFIED"


# ── crucible.board: the board status reduced from the same rows ────────


class TestBoardNeverRendersMetOverARedMember:
    """The issue's own fixture: a member is failed and the parent must not
    render green.

    Before this PR, `crucible.board._read_attribution` asked only "did every
    row report a non-null, non-N/A value" — a row that reported RED still
    counted as "measured" and the reader returned MET. This fails against
    that prior behaviour and passes against the fix.
    """

    @staticmethod
    def _declaration() -> Declaration:
        return Declaration(
            id="attribution",
            source="objective",
            title="a probe",
            surface="crucible/board",
            reader="attribution_table_is_complete",
            artifact="report/{trading_day}/attribution.json",
            means_when_red="the table is incomplete",
        )

    @staticmethod
    def _table(statuses: list[str]) -> bytes:
        rows = [
            {"name": f"row{i}", "status": status, "value": 0.5} for i, status in enumerate(statuses)
        ]
        return json.dumps(
            {"trading_day": "2026-08-28", "generated_utc": "2026-08-28T00:00:00Z", "rows": rows}
        ).encode()

    def test_a_red_member_makes_the_reading_unmet_not_met(self, tmp_path) -> None:
        from crucible.report import ROWS

        store = LocalStore(tmp_path)
        statuses = ["GREEN"] * (len(ROWS) - 1) + ["RED"]
        store.put_bytes("report/2026-08-28/attribution.json", self._table(statuses))
        reading = read_declaration(store, self._declaration(), "2026-08-28")
        assert reading.state == "UNMET"
        assert "RED" in reading.detail

    def test_all_green_still_reads_met(self, tmp_path) -> None:
        from crucible.report import ROWS

        store = LocalStore(tmp_path)
        store.put_bytes("report/2026-08-28/attribution.json", self._table(["GREEN"] * len(ROWS)))
        reading = read_declaration(store, self._declaration(), "2026-08-28")
        assert reading.state == "MET"

    def test_the_readers_verdict_is_reconstructible_from_the_documents_own_rows(
        self, tmp_path
    ) -> None:
        """Clause 3, at the board layer: the same rows the artifact carries,
        run back through `attribution_grade`, reproduce whether the board
        reader called it MET."""
        from crucible.report import ROWS, attribution_grade

        store = LocalStore(tmp_path)
        statuses = ["GREEN"] * (len(ROWS) - 1) + ["WATCH"]
        store.put_bytes("report/2026-08-28/attribution.json", self._table(statuses))
        document = json.loads(store.get_bytes("report/2026-08-28/attribution.json"))
        grade, _ = attribution_grade(document["rows"])
        reading = read_declaration(store, self._declaration(), "2026-08-28")
        assert (reading.state == "MET") == (grade == "GREEN")


# ── crucible.gate: one gate's clause reading ────────────────────────────


class TestGateResultMembers:
    DAY = dt.date(2026, 8, 28)

    def test_members_mirror_the_clauses_id_value_status(self) -> None:
        clauses = [
            Clause("a", "req a", True, "met"),
            Clause("b", "req b", False, "unmet"),
        ]
        result = GateResult(gate="phase1", trading_day=self.DAY, window=[self.DAY], clauses=clauses)
        assert [m.to_dict() for m in result.members] == [
            {"id": "a", "value": True, "status": "MET"},
            {"id": "b", "value": False, "status": "UNMET"},
        ]

    def test_an_unmet_clause_forces_the_gate_result_unmet_never_met(self) -> None:
        """The fixture shape: one failed member, checked against `.met`."""
        clauses = [Clause("a", "req a", True, "met"), Clause("b", "req b", False, "unmet")]
        result = GateResult(gate="phase1", trading_day=self.DAY, window=[self.DAY], clauses=clauses)
        assert result.met is False
        worst = worst_member(result.members, rank=CLAUSE_MEMBER_RANK)
        assert worst.status != "MET"

    def test_an_unmeasurable_clause_ranks_worse_than_unmet_never_averaged_in(self) -> None:
        clauses = [
            Clause("a", "req a", True, "met"),
            Clause("b", "req b", False, "cannot read", unmeasurable=True),
        ]
        result = GateResult(gate="phase1", trading_day=self.DAY, window=[self.DAY], clauses=clauses)
        assert result.met_ratio is None
        worst = worst_member(result.members, rank=CLAUSE_MEMBER_RANK)
        assert worst.status == "UNMEASURABLE"

    def test_to_dict_carries_members_reconstructible_into_met_ratio(self) -> None:
        clauses = [Clause("a", "req a", True, "met"), Clause("b", "req b", True, "met")]
        result = GateResult(gate="phase1", trading_day=self.DAY, window=[self.DAY], clauses=clauses)
        document = result.to_dict()
        met_count = sum(1 for m in document["members"] if m["status"] == "MET")
        assert met_count / len(document["members"]) == document["met_ratio"]

    def test_every_clause_member_status_is_ranked(self) -> None:
        assert set(CLAUSE_MEMBER_RANK) == {"MET", "UNMET", "UNMEASURABLE"}


# ── crucible.gate: the phase ladder ─────────────────────────────────────


def _phase_row(phase, *, gate_state: str, state: str | None = None) -> PhaseRow:
    return PhaseRow(
        phase=phase,
        state=state or gate_state,
        gate_state=gate_state,
        detail="fixture",
        clauses_met=1 if gate_state == "MET" else 0,
        clauses_total=1,
        clauses_unmeasurable=0,
        met_ratio=1.0 if gate_state == "MET" else 0.0,
        read_on="2026-08-28",
        blocked_by=None,
    )


class TestPhaseLadderMembers:
    GENERATED = "2026-08-29T00:00:00Z"

    def test_phases_met_is_reconstructible_from_the_rows_alone(self) -> None:
        rows = [
            _phase_row(GATE_PHASES[0], gate_state="MET"),
            _phase_row(GATE_PHASES[1], gate_state="UNMET"),
        ]
        ladder = Ladder(trading_day=dt.date(2026, 8, 28), generated_utc=self.GENERATED, rows=rows)
        document = ladder.to_dict()
        assert document["phases_met"] == sum(1 for r in rows if r.gate_state == "MET")
        assert document["phases_total"] == len(rows)

    def test_an_unmet_phase_keeps_current_phase_from_reading_complete(self) -> None:
        """The fixture shape at the ladder layer: one failed member (an
        UNMET phase) must stop `current_phase` from reading `complete`."""
        rows = [
            _phase_row(GATE_PHASES[0], gate_state="MET"),
            _phase_row(GATE_PHASES[1], gate_state="UNMET"),
        ]
        ladder = Ladder(trading_day=dt.date(2026, 8, 28), generated_utc=self.GENERATED, rows=rows)
        assert ladder.current_phase == GATE_PHASES[1].id
        assert ladder.current_phase != "complete"

    def test_unmeasured_count_is_reconstructible_from_the_rows_alone(self) -> None:
        rows = [
            _phase_row(GATE_PHASES[0], gate_state="MET"),
            _phase_row(GATE_PHASES[1], gate_state="UNMEASURED"),
            _phase_row(GATE_PHASES[2], gate_state="UNMEASURED"),
        ]
        ladder = Ladder(trading_day=dt.date(2026, 8, 28), generated_utc=self.GENERATED, rows=rows)
        assert ladder.unmeasured == sum(1 for r in rows if r.gate_state == "UNMEASURED")
        assert ladder.to_dict()["unmeasured"] == ladder.unmeasured
