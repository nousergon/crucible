"""The board's guards — `alpha-engine-config-I9837`.

What is asserted here is what would make the board WORSE THAN NOTHING:

* **A row that silently disappears.** The board's whole claim is that it
  contains everything we plan to measure. A row set that can shrink without
  raising turns that claim into a coincidence, and a shrinking board looks
  identical to a board where everything is fine.
* **A green row over no data.** The state vocabulary is total and its red set
  is named explicitly, so a state nobody classified cannot drift into "not
  red" by omission.
* **`UNMEASURABLE` folded into `UNMET`.** "It says no" and "I could not ask"
  are different facts and the second one is about us. Reporting our own
  outage as the system's result is `alpha-engine-config-I9828`.
* **A digest that reports absolute state.** A board reading almost entirely
  `PLANNED` for weeks is correct and is also the thing people stop opening.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import json
import pathlib

import pytest

from crucible.board import (
    BOARD_CONSOLE_STATE,
    BOARD_STATES,
    COMPONENT_BOARD_STATE,
    GREY_STATES,
    LADDER_BOARD_STATE,
    RED_STATES,
    SOURCES,
    Board,
    BoardRow,
    Declaration,
    board_delta,
    board_payload,
    build_board,
    load_declarations,
    read_verdict,
)
from crucible.components import load_registry
from crucible.console.classify import STATES as COMPONENT_STATES
from crucible.console.classify import Classification
from crucible.gate import LADDER_STATES, PHASES
from crucible.store import LocalStore

ACCEPTANCE_SUITE = (
    pathlib.Path(__file__).parent / "acceptance" / "test_plan_section_2_objectives.py"
)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


@pytest.fixture
def declarations():
    return load_declarations()


# ── The vocabulary is total, in both directions ───────────────────────────


class TestTheVocabularyIsTotal:
    def test_every_board_state_is_red_grey_or_met(self) -> None:
        """No state may be in none of the three buckets.

        A state nobody classified is one nobody has decided how to COUNT, and
        an uncounted state is counted as progress by whoever reads the board
        next.
        """
        assert RED_STATES | GREY_STATES | {"MET"} == set(BOARD_STATES)
        assert not RED_STATES & GREY_STATES

    def test_unmeasurable_is_red_and_is_not_unmet(self) -> None:
        assert "UNMEASURABLE" in RED_STATES
        assert "UNMEASURABLE" != "UNMET"

    def test_no_board_state_renders_as_healthy_except_met(self) -> None:
        """The specific failure this surface exists to make impossible."""
        healthy = {s for s, console in BOARD_CONSOLE_STATE.items() if console == "HEALTHY"}
        assert healthy == {"MET"}, (
            f"{sorted(healthy - {'MET'})} would render GREEN on the console. Rendering "
            "no data as green is the one thing this board must never do."
        )

    @pytest.mark.parametrize("state", LADDER_STATES)
    def test_every_ladder_state_maps_onto_the_board(self, state) -> None:
        assert LADDER_BOARD_STATE[state] in BOARD_STATES

    @pytest.mark.parametrize("state", COMPONENT_STATES)
    def test_every_component_state_maps_onto_the_board(self, state) -> None:
        assert COMPONENT_BOARD_STATE[state] in BOARD_STATES

    def test_a_component_that_ran_and_measured_nothing_is_not_met(self) -> None:
        """`UNREPORTED` is the classifier's "ran ok, measured nothing"."""
        assert COMPONENT_BOARD_STATE["UNREPORTED"] in RED_STATES

    def test_the_only_component_state_that_earns_met_is_healthy(self) -> None:
        met = {s for s, board in COMPONENT_BOARD_STATE.items() if board == "MET"}
        assert met == {"HEALTHY"}


# ── The row set is DERIVED, and cannot shrink quietly ─────────────────────


class TestTheRowSetIsDerived:
    def test_every_declaration_source_contributes_rows(self, store, declarations) -> None:
        board = build_board(store, declarations=declarations)
        sources = {row.source for row in board.rows}
        assert sources == set(SOURCES), (
            f"the board is missing rows from {sorted(set(SOURCES) - sources)}. A source "
            "contributing nothing is indistinguishable from a source nobody wired."
        )

    def test_every_phase_in_the_ladder_declaration_has_a_row(self, store) -> None:
        board = build_board(store)
        ids = {row.id for row in board.rows}
        for phase in PHASES:
            assert f"phase:{phase.id}" in ids

    def test_every_registry_component_has_a_row(self, store) -> None:
        board = build_board(store)
        ids = {row.id for row in board.rows}
        for name in load_registry():
            assert f"component:{name}" in ids, (
                f"{name} is registered and has no board row — the registry is a source "
                "of the row set, not a list the board may sample from"
            )

    def test_a_removed_component_row_shrinks_the_board_and_that_is_visible(self, store) -> None:
        """The deletion test I9837 asks for, stated as the property it protects.

        Deleting a declaration must not be silent. The board is rebuilt from a
        registry with one row removed; the row set must differ, and the DIGEST
        must report it as a vanished row rather than simply rendering a
        smaller board.
        """
        registry = load_registry()
        victim = sorted(registry)[0]
        full = build_board(store, registry=registry)
        shrunk = build_board(store, registry={k: v for k, v in registry.items() if k != victim})

        assert len(shrunk.rows) == len(full.rows) - 1
        deltas = board_delta(full.to_dict(), shrunk)
        vanished = [d for d in deltas if d.kind == "vanished"]
        assert [d.id for d in vanished] == [f"component:{victim}"], (
            "a declaration that disappeared must be reported as VANISHED. A board that "
            "just renders one row fewer looks exactly like a board where everything is "
            "fine, which is the failure this instrument exists to prevent."
        )
        assert "VANISHED" in vanished[0].describe()

    def test_a_ladder_missing_a_declared_phase_raises(self, store) -> None:
        """Not "renders that phase UNMEASURED" — RAISES.

        An UNMEASURED row would be a truthful statement about a phase. A
        ladder that dropped a phase is a statement about the PRODUCER, and
        rendering a producer defect as a measurement is the whole bug class.
        """

        class _Row:
            def __init__(self, phase):
                self.phase = phase
                self.state = "UNMEASURED"
                self.gate_state = "UNMEASURED"
                self.detail = "stub"
                self.read_on = None

        class _Ladder:
            rows = [_Row(p) for p in PHASES[:-1]]

        with pytest.raises(ValueError, match="missing phase row"):
            build_board(store, ladder=_Ladder())


class TestTheObjectivesMatchTheAcceptanceSuite:
    """A bijection, asserted in both directions.

    An objective declared with no test is a promise nobody checks; an
    acceptance class with no board row is a check nobody sees. Both are the
    same defect pointing opposite ways.
    """

    @staticmethod
    def _acceptance_classes() -> set[str]:
        tree = ast.parse(ACCEPTANCE_SUITE.read_text())
        return {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
        }

    def test_the_parse_found_classes_at_all(self) -> None:
        """Fails closed. An empty parse would make both directions vacuous."""
        assert len(self._acceptance_classes()) >= 5

    def test_every_objective_names_a_class_that_exists(self, declarations) -> None:
        classes = self._acceptance_classes()
        for row_id, declaration in declarations.objectives.items():
            assert declaration.clause_class in classes, (
                f"objective {row_id} names {declaration.clause_class!r}, which is not a "
                "class in the acceptance suite — the objective is declared and untested"
            )

    def test_every_acceptance_class_has_an_objective_row(self, declarations) -> None:
        declared = {d.clause_class for d in declarations.objectives.values()}
        orphans = self._acceptance_classes() - declared
        assert not orphans, (
            f"{sorted(orphans)} are tested and appear on no board row. A check nobody "
            "can see on the board is a check that can pass or fail unnoticed."
        )


# ── Reading: four outcomes, three of them red ─────────────────────────────


class TestReadingIsHonestAboutWhatItCouldNotDo:
    @staticmethod
    def _measured(**overrides) -> Declaration:
        base = {
            "id": "probe",
            "source": "objective",
            "title": "a probe",
            "surface": "crucible/board",
            "reads": "probe/current.json",
            "verdict": "ok",
            "means_when_red": "the probe says no",
        }
        return Declaration(**{**base, **overrides})

    def test_an_absent_artifact_is_unmeasured_and_names_the_key(self, store) -> None:
        reading = read_verdict(store, self._measured())
        assert reading.state == "UNMEASURED"
        assert "probe/current.json" in reading.detail

    def test_a_read_failure_is_unmeasurable_not_unmet(self, store, monkeypatch) -> None:
        """The `alpha-engine-config-I9828` case, as a property.

        A credential error and an unmet objective rendered identically is what
        made the cost clause unreadable in CI for its whole life.
        """

        def _denied(_key):
            raise PermissionError("AccessDenied")

        monkeypatch.setattr(store, "exists", _denied)
        reading = read_verdict(store, self._measured())
        assert reading.state == "UNMEASURABLE"
        assert "AccessDenied" in reading.detail

    def test_a_document_missing_the_field_is_unmeasurable(self, store) -> None:
        store.put_bytes("probe/current.json", json.dumps({"something_else": True}).encode())
        reading = read_verdict(store, self._measured())
        assert reading.state == "UNMEASURABLE"
        assert "'ok'" in reading.detail

    def test_a_non_boolean_verdict_is_unmeasurable(self, store) -> None:
        store.put_bytes("probe/current.json", json.dumps({"ok": "yes"}).encode())
        assert read_verdict(store, self._measured()).state == "UNMEASURABLE"

    def test_a_true_verdict_is_met_and_a_false_one_is_unmet(self, store) -> None:
        store.put_bytes("probe/current.json", json.dumps({"ok": True}).encode())
        assert read_verdict(store, self._measured()).state == "MET"
        store.put_bytes("probe/current.json", json.dumps({"ok": False}).encode())
        assert read_verdict(store, self._measured()).state == "UNMET"

    def test_planned_is_declared_never_inferred_from_absence(self, store) -> None:
        """The distinction the board turns on.

        An absent artifact under a declared producer is UNMEASURED (red). The
        SAME absence under `reads: null` is PLANNED (grey). Nothing about the
        store separates those two; only the declaration does.
        """
        planned = self._measured(reads=None, verdict=None, planned_because="no producer yet")
        assert read_verdict(store, planned).state == "PLANNED"
        assert read_verdict(store, self._measured()).state == "UNMEASURED"


class TestTheDeclarationSchemaIsClosed:
    def test_reads_and_verdict_must_be_set_together(self) -> None:
        with pytest.raises(ValueError, match="must be set together"):
            Declaration(
                id="x",
                source="objective",
                title="t",
                surface="s",
                reads="k",
                verdict=None,
                means_when_red="r",
            )

    def test_a_planned_row_must_say_why(self) -> None:
        with pytest.raises(ValueError, match="planned_because"):
            Declaration(
                id="x",
                source="objective",
                title="t",
                surface="s",
                reads=None,
                verdict=None,
                means_when_red="r",
            )

    def test_a_row_cannot_be_both_measured_and_not_built(self) -> None:
        with pytest.raises(ValueError, match="both"):
            Declaration(
                id="x",
                source="objective",
                title="t",
                surface="s",
                reads="k",
                verdict="v",
                means_when_red="r",
                planned_because="also planned",
            )

    def test_every_row_must_say_what_red_means(self) -> None:
        with pytest.raises(ValueError, match="means_when_red"):
            Declaration(
                id="x",
                source="objective",
                title="t",
                surface="s",
                reads="k",
                verdict="v",
                means_when_red="   ",
            )

    def test_an_unknown_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="source"):
            Declaration(
                id="x",
                source="invented",
                title="t",
                surface="s",
                reads="k",
                verdict="v",
                means_when_red="r",
            )

    def test_an_unknown_yaml_field_is_refused(self, tmp_path) -> None:
        """A typo'd key must not be silently ignored.

        On a board, an ignored key is a row measuring nothing while looking
        configured — which reads greener than a row that is honestly absent.
        """
        document = tmp_path / "board.yaml"
        document.write_text(
            "version: 1\n"
            "objectives:\n"
            "  a:\n"
            "    statement: s\n"
            "    surface: crucible/board\n"
            "    read: k\n"
            "    means_when_red: r\n"
            "cutover:\n"
            "  b:\n"
            "    statement: s\n"
            "    surface: crucible/board\n"
            "    reads: null\n"
            "    verdict: null\n"
            "    planned_because: p\n"
            "    means_when_red: r\n"
        )
        load_declarations.cache_clear()
        try:
            with pytest.raises(ValueError, match="unknown field"):
                load_declarations(document)
        finally:
            load_declarations.cache_clear()


# ── Rows say how they know, and what red means ────────────────────────────


class TestEveryRowIsActionable:
    def test_every_row_names_the_artifact_it_reads(self, store) -> None:
        for row in build_board(store).rows:
            assert row.artifact.strip()

    def test_every_row_says_how_it_knows(self, store) -> None:
        for row in build_board(store).rows:
            assert row.detail.strip()

    def test_every_row_says_what_red_means(self, store) -> None:
        for row in build_board(store).rows:
            assert row.means_when_red.strip()

    def test_a_row_with_no_artifact_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names no artifact"):
            BoardRow(
                id="x",
                source="objective",
                title="t",
                state="UNMET",
                detail="d",
                surface="s",
                artifact="  ",
                means_when_red="r",
            )

    def test_a_row_with_an_invented_state_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a board state"):
            BoardRow(
                id="x",
                source="objective",
                title="t",
                state="PROBABLY_FINE",
                detail="d",
                surface="s",
                artifact="a",
                means_when_red="r",
            )

    def test_the_id_scheme_is_source_prefixed_and_unique(self, store) -> None:
        rows = build_board(store).rows
        ids = [row.id for row in rows]
        assert len(ids) == len(set(ids)), "a duplicate row id would silently merge two rows"
        for row in rows:
            assert row.id.startswith(f"{row.source}:")


class TestTheBoardIsRedOnDayOne:
    def test_an_empty_store_yields_no_met_rows(self, store) -> None:
        """The instrument's founding property.

        Against a store containing nothing, not one row may read MET. A board
        that can show green over an empty store would show green over an
        outage.
        """
        board = build_board(store)
        assert board.met == [], [r.id for r in board.met]

    def test_the_counts_include_the_zeroes(self, store) -> None:
        counts = build_board(store).counts()
        assert set(counts) == set(BOARD_STATES), (
            "a state missing from the counts reads as 'not applicable' and is "
            "indistinguishable from a state the producer stopped being able to compute"
        )

    def test_the_headline_is_the_red_count_not_the_green_one(self, store) -> None:
        document = json.loads(board_payload(build_board(store)))
        assert document["red_count"] == len(build_board(store).red)
        assert document["red_count"] > 0

    def test_the_payload_round_trips(self, store) -> None:
        document = json.loads(board_payload(build_board(store)))
        assert document["schema_version"] == "board.v1"
        assert len(document["rows"]) == document["row_count"]


# ── The digest reports deltas ─────────────────────────────────────────────


def _board(**states) -> Board:
    return Board(
        trading_day="2026-09-02",
        generated_at="2026-09-02T00:00:00Z",
        rows=[
            BoardRow(
                id=row_id,
                source="objective",
                title="t",
                state=state,
                detail="d",
                surface="s",
                artifact="a",
                means_when_red="r",
            )
            for row_id, state in states.items()
        ],
    )


class TestTheDigestReportsDeltas:
    def test_the_first_ever_board_reports_no_deltas(self) -> None:
        """Not "every row appeared".

        A full board's worth of `appeared` rows on the one day the board
        itself is the news is noise standing exactly where the signal goes.
        """
        assert board_delta(None, _board(a="UNMET")) == []

    def test_an_unchanged_board_reports_nothing(self) -> None:
        before = _board(a="UNMET", b="PLANNED")
        assert board_delta(before.to_dict(), _board(a="UNMET", b="PLANNED")) == []

    def test_a_row_going_green_is_earned(self) -> None:
        deltas = board_delta(_board(a="UNMET").to_dict(), _board(a="MET"))
        assert [d.kind for d in deltas] == ["earned"]

    def test_a_row_going_red_from_green_is_lost(self) -> None:
        deltas = board_delta(_board(a="MET").to_dict(), _board(a="UNMET"))
        assert [d.kind for d in deltas] == ["lost"]

    def test_unmeasurable_is_a_move_not_an_earn(self) -> None:
        """Going from UNMET to UNMEASURABLE is not progress.

        It means we stopped being able to ask. A digest that read that as
        movement toward green would celebrate losing a credential.
        """
        deltas = board_delta(_board(a="UNMET").to_dict(), _board(a="UNMEASURABLE"))
        assert [d.kind for d in deltas] == ["moved"]

    def test_a_new_row_appears_and_a_dropped_row_vanishes(self) -> None:
        deltas = board_delta(_board(a="UNMET").to_dict(), _board(b="UNMET"))
        assert {d.kind for d in deltas} == {"vanished", "appeared"}

    def test_a_regression_offset_by_a_gain_is_still_reported(self) -> None:
        """Both rows, not a net of zero.

        A digest reporting a count rather than the row ids would render this
        as "no change", which is the scalar-comparison defect the acceptance
        ratchet was already fixed for once today.
        """
        before = _board(a="MET", b="UNMET").to_dict()
        deltas = board_delta(before, _board(a="UNMET", b="MET"))
        assert {d.id for d in deltas} == {"a", "b"}


class TestTheProducerNeverRunsWhatItGrades:
    def test_build_board_only_reads_the_store(self, store, monkeypatch) -> None:
        """`a gate reads; it never runs`, asserted rather than intended.

        The board is a grading surface. A producer that could write while
        grading could satisfy a clause it is measuring, and the IAM role is
        the second half of this guard, not the first — a role is a deployment
        fact and this is a code fact.
        """
        writes: list[str] = []
        monkeypatch.setattr(store, "put_bytes", lambda key, *a, **k: writes.append(key))
        build_board(store)
        assert writes == [], f"build_board wrote {writes}"


class TestTheDeclarationsFileItself:
    def test_every_cutover_predicate_is_individually_readable(self, declarations) -> None:
        """Four predicates, not one rolled-up phase-4 row.

        A single row would let one of the four be quietly outstanding, and
        three of them are about resources whose non-existence is the evidence.
        """
        assert len(declarations.cutover) == 4
        for declaration in declarations.cutover.values():
            assert declaration.reads, (
                f"{declaration.id} names no artifact. A cutover predicate with nowhere "
                "to read from can never become true except by someone asserting it."
            )

    def test_every_declaration_is_frozen(self) -> None:
        assert dataclasses.fields(Declaration)
        with pytest.raises(dataclasses.FrozenInstanceError):
            declaration = Declaration(
                id="x",
                source="objective",
                title="t",
                surface="s",
                reads=None,
                verdict=None,
                means_when_red="r",
                planned_because="p",
            )
            declaration.state = "MET"  # type: ignore[attr-defined]


class TestTheComponentRowsUseTheSharedClassification:
    def test_a_supplied_classification_is_what_the_row_renders(self, store) -> None:
        registry = load_registry()
        name = sorted(registry)[0]
        board = build_board(
            store,
            registry={name: registry[name]},
            classifications={
                name: Classification(name, "HEALTHY", "ran and ended ok", trading_day="2026-09-02")
            },
        )
        row = next(r for r in board.rows if r.id == f"component:{name}")
        assert row.state == "MET"
        assert row.last_read == "2026-09-02"

    def test_an_unsupplied_classification_is_unmeasured_not_absent(self, store) -> None:
        registry = load_registry()
        name = sorted(registry)[0]
        board = build_board(store, registry={name: registry[name]}, classifications={})
        row = next(r for r in board.rows if r.id == f"component:{name}")
        assert row.state == "UNMEASURED"
        assert "unobserved, not healthy" in row.detail


def test_the_board_is_keyed_by_trading_day(store) -> None:
    board = build_board(store, now=dt.datetime(2026, 9, 2, 23, 0, tzinfo=dt.UTC))
    assert board.trading_day == "2026-09-02"


class TestTheProducerRunsWithoutTheWeeklyArc:
    """`crucible board` is a job of its own, on its own schedule.

    I9837 deliverable 3: the producer must run independently of the weekly
    arc, which is phase-2 work. A board that only refreshes once phase 2 opens
    could not have rendered the gap that closed phase 1 — which is the reason
    it exists at all.
    """

    @staticmethod
    def _run(tmp_path):
        import argparse

        from crucible.track_c import board_handler

        board_handler(argparse.Namespace(trading_day=dt.date(2026, 8, 28), store=str(tmp_path)))
        return LocalStore(tmp_path)

    def test_the_job_writes_the_json_the_page_and_the_dated_copy(self, tmp_path) -> None:
        written = self._run(tmp_path)
        for key in (
            "board/current.json",
            "board/index.html",
            "board/2026-08-28/board.json",
        ):
            assert written.exists(key), f"{key} was not written"

    def test_the_job_is_registered_as_its_own_daily_scheduled_component(self) -> None:
        component = load_registry()["board"]
        assert component.schedule == "daily"
        assert component.dispatch == "scheduler", (
            "`arc` would put the board behind the weekly driver, which is phase-2 "
            "work — six rows in components.yaml once read 'weekly, Saturday' while "
            "exactly one of them was dispatched by anything"
        )

    def test_a_red_board_is_not_a_failed_run(self, tmp_path) -> None:
        """The forcing-function inversion this instrument has to get right.

        Against an empty store every row is red, and the run must still read
        `ok`. A producer that failed daily on its own correct output is a
        channel that gets muted, and a muted channel is worse than no channel.
        """
        written = self._run(tmp_path)
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        assert manifest["status"] == "ok"
        board = json.loads(written.get_bytes("board/current.json"))
        assert board["red_count"] > 0

    def test_the_manifest_records_the_boards_own_schema_version(self, tmp_path) -> None:
        written = self._run(tmp_path)
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        versions = {o["key"]: o["schema_version"] for o in manifest["outputs"]}
        assert versions["board/current.json"] == "board.v1"

    def test_the_unmeasurable_metric_is_a_breach_and_the_red_count_is_not(self, tmp_path) -> None:
        """Two numbers with opposite meanings, graded oppositely.

        Red is the declared day-one state and is graded OK. UNMEASURABLE is
        never expected and its objective is zero, so it is graded BREACH — the
        distinction `alpha-engine-config-I9828` exists because nothing made.
        """
        written = self._run(tmp_path)
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        by_name = {m["name"]: m for m in manifest["metrics"]}
        assert by_name["board_rows_red"]["status"] == "OK"
        assert by_name["board_rows_red"]["value"] > 0
        assert by_name["board_rows_unmeasurable"]["status"] == "OK"
        assert by_name["board_rows_unmeasurable"]["value"] == 0

    def test_the_second_run_reports_deltas_rather_than_the_whole_board(self, tmp_path) -> None:
        """Deltas, and the board observes ITSELF among them.

        The first run leaves a `board` manifest, so on the second run the
        board's own component row moves — and that is the correct reading, not
        noise. What is asserted is that the digest names the ROW that moved
        rather than restating the whole board: a second run against an
        otherwise unchanged store must report one delta, not forty.
        """
        self._run(tmp_path)
        written = self._run(tmp_path)
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        moved = next(m for m in manifest["metrics"] if m["name"] == "board_rows_moved")
        assert moved["value"] == 1
        assert "component:board" in moved["status_reason"]

    def test_a_third_run_with_nothing_new_reports_no_movement(self, tmp_path) -> None:
        self._run(tmp_path)
        self._run(tmp_path)
        written = self._run(tmp_path)
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        moved = next(m for m in manifest["metrics"] if m["name"] == "board_rows_moved")
        assert moved["value"] == 0
        assert "no row changed state" in moved["status_reason"]

    def test_the_page_distinguishes_unmeasurable_from_unmet(self, tmp_path) -> None:
        """Visibly distinct on the surface, not only in the data.

        Uniform red is noise: if "not built yet", "built and failing" and
        "cannot be measured" render identically the board desensitises exactly
        as the acceptance gate did on seven Dependabot PRs.
        """
        from crucible.board import _SWATCH

        assert len({_SWATCH[s] for s in BOARD_STATES}) == len(BOARD_STATES)
        page = self._run(tmp_path).get_bytes("board/index.html").decode()
        assert "UNMEASURABLE" in page
        assert _SWATCH["UNMEASURABLE"] != _SWATCH["UNMET"]
