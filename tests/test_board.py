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
from abc import ABC
from typing import Any

import pytest

from crucible.board import (
    BOARD_CONSOLE_STATE,
    BOARD_STATES,
    COMPONENT_BOARD_STATE,
    GREY_STATES,
    LADDER_BOARD_STATE,
    READER_ARTIFACTS,
    READERS,
    RED_STATES,
    SOURCES,
    Board,
    BoardRow,
    Declaration,
    board_delta,
    board_payload,
    build_board,
    load_declarations,
    pointer_may_move,
    read_declaration,
)
from crucible.components import load_registry
from crucible.console.classify import STATES as COMPONENT_STATES
from crucible.console.classify import Classification
from crucible.gate import LADDER_STATES, PHASES, evaluate
from crucible.keys import acceptance_reading_key
from crucible.store import LocalStore, Store

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
    """The four ways a read can end, and the three that are red.

    `attribution_table_is_complete` is the one registered reader, so it is the
    one exercised here. What is asserted is the SHAPE every reader must share,
    not one reader's arithmetic.
    """

    @staticmethod
    def _declaration(**overrides) -> Declaration:
        base = {
            "id": "attribution",
            "source": "objective",
            "title": "a probe",
            "surface": "crucible/board",
            "reader": "attribution_table_is_complete",
            "artifact": "report/{trading_day}/attribution.json",
            "means_when_red": "the table is incomplete",
        }
        return Declaration(**{**base, **overrides})

    @staticmethod
    def _table(rows: int, *, blind: int = 0, day: str = "2026-08-28") -> bytes:
        """A table in the shape the REAL producer writes.

        The first version of this helper emitted `"UNREPORTED"` and `"OK"` —
        neither of which `crucible.report` ever writes — and no `value` field
        at all. So the blindness check was asserted against a vocabulary the
        producer does not use, and the MET case was asserted over rows
        `build_attribution` would never produce. The statuses below come from
        `krepis.metrics.derive_status`'s real `N/A-*` family, and
        `TestTheAttributionReaderMatchesItsRealProducer` pins that against the
        producer's actual output rather than against this fixture.
        """
        body = [
            {"name": f"row{i}", "status": "N/A-NOT-RUN", "value": None}
            if i < blind
            else {"name": f"row{i}", "status": "OK", "value": 0.5}
            for i in range(rows)
        ]
        return json.dumps(
            {"trading_day": day, "generated_utc": "2026-08-28T00:00:00Z", "rows": body}
        ).encode()

    def _read(self, store, day="2026-08-28"):
        return read_declaration(store, self._declaration(), day)

    def test_an_absent_artifact_is_unmeasured_and_names_the_key(self, store) -> None:
        reading = self._read(store)
        assert reading.state == "UNMEASURED"
        assert "report/2026-08-28/attribution.json" in reading.detail

    def test_a_read_failure_is_unmeasurable_not_unmet(self, store, monkeypatch) -> None:
        """The `alpha-engine-config-I9828` case, as a property.

        A credential error and an unmet objective rendered identically is what
        made the cost clause unreadable in CI for its whole life.
        """

        def _denied(_key):
            raise PermissionError("AccessDenied")

        monkeypatch.setattr(store, "exists", _denied)
        reading = self._read(store)
        assert reading.state == "UNMEASURABLE"
        assert "AccessDenied" in reading.detail

    def test_a_document_that_does_not_answer_the_question_is_unmeasurable(self, store) -> None:
        store.put_bytes(
            "report/2026-08-28/attribution.json",
            json.dumps({"trading_day": "2026-08-28", "something_else": True}).encode(),
        )
        reading = self._read(store)
        assert reading.state == "UNMEASURABLE"
        assert "rows" in reading.detail

    def test_a_document_for_another_day_is_unmeasured_not_met(self, store) -> None:
        """The staleness case, and the reason it is UNMEASURED rather than MET.

        Every readable artifact here is keyed by trading day, so a document
        stamped with a different day can only mean the key was resolved for
        one day and filled by another. `crucible.gate.last_read` already
        records what happens without this check: a freshly written artifact
        full of never-measured content looks entirely fresh.
        """
        store.put_bytes(
            "report/2026-08-28/attribution.json",
            self._table(len(_attribution_rows()), day="2019-01-04"),
        )
        reading = self._read(store)
        assert reading.state == "UNMEASURED"
        assert "2019-01-04" in reading.detail

    def test_a_complete_table_is_met_and_carries_its_provenance(self, store) -> None:
        store.put_bytes("report/2026-08-28/attribution.json", self._table(len(_attribution_rows())))
        reading = self._read(store)
        assert reading.state == "MET"
        assert reading.last_read == "2026-08-28T00:00:00Z", (
            "a MET row with no provenance is a green dot that cannot say when it was "
            "last true, which is the state I9837 deliverable 3 exists to forbid"
        )

    def test_a_short_table_is_unmet(self, store) -> None:
        store.put_bytes(
            "report/2026-08-28/attribution.json", self._table(len(_attribution_rows()) - 1)
        )
        assert self._read(store).state == "UNMET"

    def test_a_full_table_of_blank_rows_is_unmet_not_met(self, store) -> None:
        """Counting rows is not reading them.

        A table with every declared row present and every one of them
        measuring nothing is the case `build_attribution` cannot refuse — it
        refuses the wrong ROW COUNT — and it is the one that reads green if
        the check is a row count.
        """
        n = len(_attribution_rows())
        store.put_bytes("report/2026-08-28/attribution.json", self._table(n, blind=n))
        reading = self._read(store)
        assert reading.state == "UNMET"
        assert "measured nothing" in reading.detail

    def test_a_row_with_a_measured_status_but_a_null_value_is_not_met(self, store) -> None:
        """Either condition alone is beatable.

        A row can carry a measured-looking status and a null value, and a null
        value is not a measurement whatever the status says.
        """
        n = len(_attribution_rows())
        document = json.loads(self._table(n))
        document["rows"][0]["value"] = None
        store.put_bytes("report/2026-08-28/attribution.json", json.dumps(document).encode())
        assert self._read(store).state == "UNMET"

    def test_planned_is_declared_never_inferred_from_absence(self, store) -> None:
        """The distinction the board turns on.

        An absent artifact under a registered reader is UNMEASURED (red). The
        SAME absence with `reader: null` is PLANNED (grey). Nothing about the
        store separates those two; only the declaration does.
        """
        planned = self._declaration(reader=None, planned_because="no producer yet")
        assert read_declaration(store, planned, "2026-08-28").state == "PLANNED"
        assert self._read(store).state == "UNMEASURED"


def _attribution_rows():
    from crucible.report import ROWS

    return ROWS


class TestNoDeclarationInventsAKey:
    """`crucible/keys.py` exists so a key shape is written once.

    The first draft of `board.yaml` carried a literal S3 key per row and
    invented EIGHT OF TEN of them — `ledger/cost/current.json` against the
    real `ledger/trials.jsonl`, `features/current.json` against a
    `features/{version}/{day}.parquet`, `arms/controls/current.json` against
    an `arms/{slot}/register.jsonl`. Those rows would have read UNMEASURED
    forever, and the red would have been about the KEY rather than about the
    objective — indistinguishable from the gap the board exists to show.
    """

    def test_every_reader_declares_the_artifact_it_resolves(self) -> None:
        assert set(READERS) == set(READER_ARTIFACTS), (
            "a reader with no declared artifact template cannot be checked against the "
            "key it really derives, which is how the human-facing string drifts"
        )

    def test_every_declared_artifact_matches_the_key_its_reader_uses(self, declarations) -> None:
        """The board.yaml string equals what the reader actually resolves.

        Asserted against the REAL key for a sample day, so the template on the
        surface cannot become decoration.
        """
        day = "2026-08-28"
        for declaration in declarations.all.values():
            if declaration.reader is None:
                continue
            expected = READER_ARTIFACTS[declaration.reader].replace("{trading_day}", day)
            assert declaration.artifact.replace("{trading_day}", day) == expected, (
                f"{declaration.id} declares artifact {declaration.artifact!r}, but its "
                f"reader resolves {READER_ARTIFACTS[declaration.reader]!r}"
            )

    def test_the_attribution_reader_uses_the_owning_modules_key_function(self) -> None:
        from crucible.report import attribution_key

        assert READER_ARTIFACTS["attribution_table_is_complete"].replace(
            "{trading_day}", "2026-08-28"
        ) == attribution_key("2026-08-28")

    def test_a_declaration_naming_an_unregistered_reader_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not registered"):
            Declaration(
                id="x",
                source="objective",
                title="t",
                surface="s",
                reader="no_such_reader",
                artifact="k",
                means_when_red="r",
            )


class TestTheDeclarationSchemaIsClosed:
    @staticmethod
    def _declaration(**overrides) -> Declaration:
        base = {
            "id": "x",
            "source": "objective",
            "title": "t",
            "surface": "s",
            "reader": "attribution_table_is_complete",
            "artifact": "report/{trading_day}/attribution.json",
            "means_when_red": "r",
        }
        return Declaration(**{**base, **overrides})

    def test_a_planned_row_must_say_why(self) -> None:
        with pytest.raises(ValueError, match="planned_because"):
            self._declaration(reader=None)

    def test_a_row_cannot_be_both_measured_and_not_built(self) -> None:
        with pytest.raises(ValueError, match="both measured and not built"):
            self._declaration(planned_because="also planned")

    def test_every_row_must_say_what_red_means(self) -> None:
        with pytest.raises(ValueError, match="means_when_red"):
            self._declaration(means_when_red="   ")

    def test_every_row_must_name_an_artifact_even_when_planned(self) -> None:
        """A grey dot with no key is not actionable.

        `PLANNED` still names the thing nobody writes yet — that is the
        difference between "declared and waiting on X" and "somebody forgot
        this row".
        """
        with pytest.raises(ValueError, match="names no artifact"):
            self._declaration(reader=None, planned_because="p", artifact="  ")

    def test_an_unknown_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="source"):
            self._declaration(source="invented")

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
            "    reader: null\n"
            "    artifact: k\n"
            "    planned_because: p\n"
            "    read: k\n"
            "    means_when_red: r\n"
            "cutover:\n"
            "  b:\n"
            "    statement: s\n"
            "    surface: crucible/board\n"
            "    reader: null\n"
            "    artifact: k\n"
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


def _refusing_store(tmp_path) -> LocalStore:
    """A store whose every declared mutator raises, installed FROM the declaration.

    Two earlier versions of this guard were wrong in the same direction. The
    first monkeypatched `put_bytes` alone and was beaten by
    `compare_and_swap`, which writes via `open(tmp,"xb")` plus `os.replace`
    and never touches it. The second hand-listed both names and then compared
    that list against `dir(Store)` FILTERED BY MEMBERSHIP IN THE SAME LIST —
    so it computed `mutators ∩ dir(Store)` and asserted only that those two
    names still exist. Adding a third abstract mutator to `Store` left it
    green: a denylist wearing an allowlist's docstring, reintroduced inside
    the fix for exactly that.

    The refusals are now installed by iterating `Store.MUTATORS`, which is the
    interface's own declaration. A mutator added there is refused here without
    anyone editing this file; a mutator added WITHOUT a line there fails
    `test_the_mutator_declaration_covers_every_abstract_write`.
    """

    class _Refusing(LocalStore):
        pass

    for name in Store.MUTATORS:

        def _refuse(*_args, _name: str = name, **_kwargs):
            raise AssertionError(f"build_board wrote via {_name}")

        setattr(_Refusing, name, _refuse)
    return _Refusing(tmp_path)


class TestTheProducerNeverRunsWhatItGrades:
    def test_build_board_only_reads_the_store(self, tmp_path) -> None:
        """`a gate reads; it never runs`, asserted rather than intended.

        The board is a grading surface. A producer that could write while
        grading could satisfy a clause it is measuring, and the IAM role is
        the second half of this guard, not the first — a role is a deployment
        fact and this is a code fact.
        """
        build_board(_refusing_store(tmp_path))

    @pytest.mark.parametrize("name", Store.MUTATORS)
    def test_the_double_really_refuses_each_declared_mutator(self, tmp_path, name) -> None:
        """The guard's own guard, one case per mutator.

        A double that silently permitted a write would make the test above
        pass over nothing — the vacuous-guard shape this board argues against
        everywhere else.
        """
        refusing = _refusing_store(tmp_path)
        with pytest.raises(AssertionError, match=name):
            getattr(refusing, name)("k", "etag", b"{}")

    def test_the_mutator_declaration_partitions_the_interface(self) -> None:
        """`MUTATORS` and `READERS` together cover every public method on `Store`.

        A PARTITION, not a signature heuristic. The previous version derived
        the write set from `"payload" in signature.parameters` — a guess at a
        parameter NAME — and adding `delete(self, key)` (no payload by
        definition) or `append_bytes(self, key, data)` left it returning the
        same two names, the guard green, and `_refusing_store` installing no
        refusal for either. `test_build_board_only_reads_the_store` would then
        have passed over a board that deletes from the store.

        Under a partition a new method cannot be missed: it belongs to one
        list or the other, and belonging to neither fails here. Classifying it
        wrongly is a deliberate act visible in a diff, which is the most any
        declaration can promise.
        """
        # The whole MRO, not `vars(Store)`. `vars` sees only what is defined
        # ON the class, so a public write inherited from a base or a mixin was
        # invisible: adding `class Store(_LegacyMixin, ABC)` with a `purge`
        # method left this green and `_refusing_store` installing no refusal
        # for it. `ABC` and `object` are excluded because their members are
        # not this interface's surface.
        inherited = set(dir(ABC)) | set(dir(object))
        public = {
            name
            for name in dir(Store)
            if not name.startswith("_")
            and name not in inherited
            and callable(getattr(Store, name, None))
        }
        declared = set(Store.MUTATORS) | set(Store.READERS)
        assert public, "no public methods found on Store — this derivation has gone blind"
        unclassified = public - declared
        assert not unclassified, (
            f"{sorted(unclassified)} are on Store and in neither MUTATORS nor READERS. "
            "Every method must be declared one or the other: an unclassified method is "
            "one `_refusing_store` will not refuse, which makes every read-only "
            "assertion in this class blind to it."
        )
        phantom = declared - public
        assert not phantom, (
            f"{sorted(phantom)} are declared on Store and do not exist — a stale entry "
            "makes the partition look complete while covering nothing."
        )
        assert not (set(Store.MUTATORS) & set(Store.READERS))


class TestAReplayDoesNotClobberThePointer:
    """`board/current.json` is what the fleet console reads.

    Measured before the guard existed: `crucible board --date 2026-09-01` then
    `--date 2026-08-28` left the pointer reading 2026-08-28. The dated board
    is always written; only the pointer is guarded, the same way
    `crucible.release` guards its release pointer rather than its releases.
    """

    @staticmethod
    def _board(day: str) -> Board:
        return Board(trading_day=day, generated_at="2026-09-02T00:00:00Z", rows=[])

    def test_the_first_board_may_move_the_pointer(self) -> None:
        may, _ = pointer_may_move(None, self._board("2026-08-28"))
        assert may

    def test_a_newer_board_may_move_the_pointer(self) -> None:
        may, _ = pointer_may_move({"trading_day": "2026-08-28"}, self._board("2026-09-01"))
        assert may

    def test_the_same_day_may_move_the_pointer(self) -> None:
        """A same-day re-render is a refresh, not a regression."""
        may, _ = pointer_may_move({"trading_day": "2026-09-01"}, self._board("2026-09-01"))
        assert may

    def test_a_replay_may_not_move_the_pointer_backwards(self) -> None:
        may, reason = pointer_may_move({"trading_day": "2026-09-01"}, self._board("2026-08-28"))
        assert not may
        assert "2026-09-01" in reason and "2026-08-28" in reason

    def test_an_unreadable_incumbent_does_not_license_a_move(self) -> None:
        """ "I could not read what is there" is not "what is there is older".

        On a pointer those two want opposite actions, and defaulting to move
        would let one corrupt read overwrite a good pointer with a replay.
        """
        may, reason = pointer_may_move({"generated_at": "x"}, self._board("2026-08-28"))
        assert not may
        assert "no trading_day" in reason


class TestTheDeclarationsFileItself:
    def test_every_cutover_predicate_is_individually_readable(self, declarations) -> None:
        """Four predicates, not one rolled-up phase-4 row.

        A single row would let one of the four be quietly outstanding, and
        three of them are about resources whose non-existence is the evidence.
        """
        assert len(declarations.cutover) == 4
        for declaration in declarations.cutover.values():
            assert declaration.artifact.strip(), (
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
                reader=None,
                artifact="k",
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
        # `scheduler` is what this row said from 2026-09-02T21:47Z until
        # alpha-engine-config-I9878, and it was false: there is no
        # AWS::Scheduler::Schedule for this job anywhere in crucible-v2.yaml.
        assert component.dispatch == "github-actions", (
            "`arc` would put the board behind the weekly driver, which is phase-2 "
            "work — six rows in components.yaml once read 'weekly, Saturday' while "
            "exactly one of them was dispatched by anything. The starter here is "
            "the `on.schedule` cron in .github/workflows/board.yml"
        )

    def test_the_row_names_the_workflow_that_starts_it(self) -> None:
        """The link is DECLARED, not inferred from shell text.

        The first version of the cross-repo guard matched `crucible board` in
        board.yml's `run:` bodies. An adversarial review commented the command
        out, left `echo skipped` in its place, and the guard read green — the
        I9878 defect reproduced through the value added to fix it.
        `tests/test_workflow_triggers.py` already carries the general result
        from five rounds against a different guard: any predicate over a
        `run:` body is a partial shell parser, and a partial shell parser is a
        denylist of the syntax someone thought of.
        """
        assert load_registry()["board"].dispatch_workflow == "board.yml"

    def test_a_github_actions_row_that_names_no_workflow_is_refused(self) -> None:
        """The field is not optional on the rows that need it. A
        `github-actions` row naming no workflow would be verifiable against
        nothing, which is a hole in the vocabulary rather than a value in it.
        """
        import dataclasses as _dc

        from crucible.components import Component as _Component

        row = load_registry()["board"]
        with pytest.raises(ValueError, match="dispatch_workflow is required"):
            _dc.replace(row, dispatch_workflow=None)
        assert isinstance(row, _Component)

    def test_a_non_github_actions_row_may_not_name_a_workflow(self) -> None:
        """The converse. A workflow name on a `scheduler` or `arc` row is a
        second declaration of the starter that nothing reads and nothing
        checks, so it can say anything and drift silently."""
        import dataclasses as _dc

        arc_row = next(r for r in load_registry().values() if r.dispatch == "arc")
        with pytest.raises(ValueError, match="forbidden on any other"):
            _dc.replace(arc_row, dispatch_workflow="board.yml")

    def test_the_workflow_named_by_the_row_is_the_one_that_actually_crons_it(self) -> None:
        """The half of the claim this repository can check on its own.

        The cross-repo lockstep guard
        (`nous-ergon-ops/tests/crossrepo/test_crucible_dispatch_lockstep.py`)
        asserts the `github-actions` vocabulary in both directions over every
        row and every workflow. This asserts the one instance here too,
        because a guard living only in the repository that cannot break it is
        the exact defect I9878 was filed for.
        """
        import yaml

        named = load_registry()["board"].dispatch_workflow
        workflow = yaml.safe_load(
            (
                pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows" / named
            ).read_text(encoding="utf-8")
        )
        # `on` parses as the boolean True under YAML 1.1 — the key is not the
        # string "on" here, which is the single most common way a workflow
        # trigger assertion passes vacuously.
        triggers = workflow[True]
        crons = [entry["cron"] for entry in triggers["schedule"]]
        assert crons == ["30 21 * * *"], (
            "the `board` row declares `dispatch: github-actions`; board.yml is "
            f"the workflow that has to carry the cron, and it has {crons}"
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

        Against an empty store that BREACH is now non-empty, and deliberately:
        `alpha-engine-config-I9913` registered a clause list for every plan §6
        phase, and four of those clauses read UNMEASURABLE today because the
        artifact they read does not exist yet (no live/replay field on the run
        manifest, no Cost Explorer grant, no trader consumer-evidence
        contract, no LLM-arm call-site field — each filed as its own issue).
        That is the intended reading: a phase with nothing to read is
        unmeasurable BY CLAUSE with a named reason, never blank.

        What is asserted here is the GRADING, which is unchanged: red stays OK
        because red is the declared day-one state, and unmeasurable stays
        BREACH because no data is never green. The board's own objective for
        this metric is a separate question from whether the reading is honest.
        """
        written = self._run(tmp_path)
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        by_name = {m["name"]: m for m in manifest["metrics"]}
        assert by_name["board_rows_red"]["status"] == "OK"
        assert by_name["board_rows_red"]["value"] > 0
        assert by_name["board_rows_unmeasurable"]["status"] == "BREACH"
        assert by_name["board_rows_unmeasurable"]["value"] > 0
        # And the run is still `ok`: an unmeasurable row is a reading the board
        # publishes, not a failure of the job that published it. Same inversion
        # `test_a_red_board_is_not_a_failed_run` guards.
        assert manifest["status"] == "ok"

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


class TestTheDigestNeverClaimsNothingMovedOverAFailedRead:
    """A positive claim asserted on no evidence is the one thing forbidden here.

    The earlier `_read_previous_board` swallowed any exception into `None`,
    which fed `board_delta(None, …) -> []` and made the digest publish
    `board_rows_moved = 0, "no row changed state"` over a read that had
    FAILED — on the only surface that reports a VANISHED declaration.
    """

    @staticmethod
    def _run(tmp_path, day):
        import argparse

        from crucible.track_c import board_handler

        board_handler(argparse.Namespace(trading_day=day, store=str(tmp_path)))
        return LocalStore(tmp_path)

    def test_an_unreadable_previous_board_is_a_breach_not_a_quiet_zero(self, tmp_path) -> None:
        self._run(tmp_path, dt.date(2026, 9, 1))
        LocalStore(tmp_path).put_bytes("board/current.json", b"{not json")
        written = self._run(tmp_path, dt.date(2026, 9, 2))
        manifest = json.loads(written.get_bytes("runs/board/2026-09-02/run.json"))
        moved = next(m for m in manifest["metrics"] if m["name"] == "board_rows_moved")
        assert moved["status"] == "BREACH"
        assert "NOT a claim that nothing moved" in moved["status_reason"]

    def test_a_replay_leaves_the_pointer_alone_and_still_writes_its_dated_board(
        self, tmp_path
    ) -> None:
        self._run(tmp_path, dt.date(2026, 9, 1))
        written = self._run(tmp_path, dt.date(2026, 8, 28))
        pointer = json.loads(written.get_bytes("board/current.json"))
        assert pointer["trading_day"] == "2026-09-01", (
            "a replay repointed board/current.json backwards — that key is what the "
            "fleet-console adapter reads"
        )
        assert written.exists("board/2026-08-28/board.json")


class TestThePhaseRowsAreActionable:
    def test_no_row_leaves_an_unsubstituted_placeholder(self, store) -> None:
        """The fleet's documented `{date}`-placeholder gotcha.

        A literal `{trading_day}` on the surface is a key nobody can copy, and
        it is indistinguishable from one that was never written.
        """
        for row in build_board(store).rows:
            assert "{trading_day}" not in row.artifact, row.id

    def test_an_unregistered_phase_says_so_rather_than_naming_a_fabricated_key(
        self, store, monkeypatch
    ) -> None:
        """`phase.gate or phase.id` invented a key for five of six phases.

        `gate_key` is keyed by the REGISTERED gate name, so `gates/phase2/…` is
        a path no producer will ever write. Naming it would send a reader to
        an empty key and tell them nothing about why it is empty.

        Every real phase now carries a gate (`alpha-engine-config-I9913`), so
        the unregistered phase is INJECTED rather than borrowed from `PHASES`.
        The property survives the condition that first exposed it; the old
        shape would have gone quiet the moment the last blank phase was
        filled in, and a guard that stops testing anything is worse than one
        that fails.
        """
        import crucible.gate as gate_module

        phase6 = gate_module.Phase("phase6", 6, "A phase whose gate is unwritten", 9761, None)
        monkeypatch.setattr(gate_module, "PHASES", (*PHASES, phase6))
        rows = {r.id: r for r in build_board(store).rows if r.source == "phase"}
        unregistered = [phase6]
        for phase in unregistered:
            row = rows[f"phase:{phase.id}"]
            assert "no gate is registered" in row.artifact
            assert "it does not exist" in row.means_when_red


class TestADryRunDoesNotTouchThePointer:
    """`--dry-run` is ignored by all five track-C handlers.

    `alpha-engine-config-I9863` owns the repo-wide fix, and honouring the flag
    for `board` alone would normally be the wrong shape — one handler behaving
    differently from the four beside it. The exception is narrow and specific:
    this is the one job whose `--dry-run` clobbers `board/current.json`, the
    key the fleet-console adapter reads.
    """

    @staticmethod
    def _run(tmp_path, day, *, dry_run=False):
        import argparse

        from crucible.track_c import board_handler

        board_handler(argparse.Namespace(trading_day=day, store=str(tmp_path), dry_run=dry_run))
        return LocalStore(tmp_path)

    def test_a_dry_run_writes_neither_the_pointer_nor_the_page(self, tmp_path) -> None:
        written = self._run(tmp_path, dt.date(2026, 9, 1), dry_run=True)
        assert not written.exists("board/current.json")
        assert not written.exists("board/index.html")
        assert not written.exists("board/2026-09-01/board.json")

    def test_a_dry_run_files_no_run_manifest_either(self, tmp_path) -> None:
        """alpha-engine-config-I9922: `board`'s own local `dry_run` branch
        above already kept it from clobbering `board/current.json`, but
        `run_job` itself wrote `runs/board/2026-09-01/run.json` regardless —
        a real firing for a run that touched nothing, which is the exact
        shape `alerts.sweep` and the board read as genuine."""
        written = self._run(tmp_path, dt.date(2026, 9, 1), dry_run=True)
        from crucible.manifest import manifest_key

        assert not written.exists(manifest_key("board", "2026-09-01"))

    def test_a_dry_run_does_not_overwrite_a_good_pointer(self, tmp_path) -> None:
        self._run(tmp_path, dt.date(2026, 9, 1))
        written = self._run(tmp_path, dt.date(2026, 8, 28), dry_run=True)
        assert json.loads(written.get_bytes("board/current.json"))["trading_day"] == "2026-09-01"

    def test_a_held_pointer_reports_no_comparison_rather_than_phantom_regressions(
        self, tmp_path
    ) -> None:
        """A replay must not publish deltas computed forward in time.

        `board_delta` against a NEWER incumbent yields a set of "regressions"
        that are an artefact of the comparison direction, not of anything that
        moved. The digest says no comparison was made and why.
        """
        self._run(tmp_path, dt.date(2026, 9, 1))
        written = self._run(tmp_path, dt.date(2026, 8, 28))
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        moved = next(m for m in manifest["metrics"] if m["name"] == "board_rows_moved")
        assert moved["value"] == 0
        assert "no comparison was made" in moved["status_reason"]
        assert "NOT a claim that nothing moved" in moved["status_reason"]
        assert moved["status"] == "OK", (
            "a replay is a deliberate operator action; grading it BREACH would teach the "
            "reader to discount the one status that means something"
        )


class TestBothHalvesOfTheBlindnessCheckAreTested:
    """`_row_measured_nothing` has two conditions and only one had a test.

    Round 2 added the null-value case and left the `N/A-*` case with no test
    that could fail — deleting the status check left the entire blocking suite
    green. That is the exact half round 1 got wrong, and the repo's binding
    rule is that a detector nobody has made fail is a detector nobody knows
    works.

    The two are not redundant. `crucible/report.py` describes `N/A-LOW-N` as a
    row whose *value may read 1.0* while `n_samples = 1` — a real emission
    carrying a non-null value under a not-measured status. Either condition
    alone lets that row through.
    """

    @staticmethod
    def _row(**overrides):
        from crucible.board import _row_measured_nothing

        base = {"name": "r", "status": "OK", "value": 0.5}
        return _row_measured_nothing({**base, **overrides})

    def test_a_measured_row_is_not_blind(self) -> None:
        assert not self._row()

    def test_a_null_value_is_blind_whatever_the_status_says(self) -> None:
        assert self._row(value=None)

    @pytest.mark.parametrize(
        "status", ["N/A-NOT-RUN", "N/A-NOT-IMPL", "N/A-MISSING-INPUT", "N/A-LOW-N"]
    )
    def test_an_n_a_status_is_blind_even_carrying_a_value(self, status) -> None:
        """The half that had no test.

        `N/A-LOW-N` with `value: 1.0` is the documented real case: the number
        is there and it is not evidence. A check on the value alone passes it.
        """
        assert self._row(status=status, value=1.0)

    def test_a_row_that_is_not_a_dict_is_blind(self) -> None:
        from crucible.board import _row_measured_nothing

        assert _row_measured_nothing("not a row")

    def test_the_two_conditions_are_independently_load_bearing(self) -> None:
        """Neither condition is redundant, asserted rather than argued.

        One case that only the status check catches, one that only the value
        check catches. Delete either condition and one of these goes green.
        """
        assert self._row(status="N/A-LOW-N", value=1.0)
        assert self._row(status="OK", value=None)


class TestTheHeldPointerDeltaGuardIsTested:
    """The delta-on-held-pointer fix shipped with no test that could fail.

    Replacing `board_delta(previous, board) if may_move else []` with the
    unconditional call left the entire blocking suite green, and
    `grep -rn "may_move" tests/` returned nothing. One of the four fixes the
    round-2 body named was asserted only by the body.
    """

    @staticmethod
    def _run(tmp_path, day, **kwargs):
        import argparse

        from crucible.track_c import board_handler

        board_handler(argparse.Namespace(trading_day=day, store=str(tmp_path), **kwargs))
        return LocalStore(tmp_path)

    def test_a_replay_publishes_no_phantom_regressions(self, tmp_path) -> None:
        """The concrete failure the guard prevents.

        The incumbent is doctored so that an UNCONDITIONAL delta would be
        non-zero — one row's state is flipped in `board/current.json`. A
        replay of an older day then compares forward in time and would publish
        that row as a regression, which is an artefact of the comparison
        direction and not of anything that moved.

        Doctoring is necessary: with a real ladder, two boards a few days
        apart over the same store happen to agree, so a test that merely runs
        two days cannot tell the guarded case from the unguarded one. That is
        why the round-2 fix shipped with no test that could fail.
        """
        self._run(tmp_path, dt.date(2026, 9, 1))
        store = LocalStore(tmp_path)
        incumbent = json.loads(store.get_bytes("board/current.json"))
        flipped = next(r for r in incumbent["rows"] if r["state"] == "UNMEASURED")
        flipped["state"] = "MET"
        store.put_bytes("board/current.json", json.dumps(incumbent).encode())

        written = self._run(tmp_path, dt.date(2026, 8, 28))
        manifest = json.loads(written.get_bytes("runs/board/2026-08-28/run.json"))
        moved = next(m for m in manifest["metrics"] if m["name"] == "board_rows_moved")
        assert moved["value"] == 0, (
            f"the replay published {moved['value']} phantom delta(s) — {moved['status_reason']}"
        )
        assert "no comparison was made" in moved["status_reason"]

    def test_a_forward_step_still_reports_its_deltas(self, tmp_path) -> None:
        """The guard must not silence a REAL move.

        The incumbent is doctored for the same reason the replay test doctors
        it: two real boards a few days apart AGREE, so a forward step over an
        undoctored store can never observe a non-zero delta, and an assertion
        over it proves nothing.

        The earlier version asserted only that the status_reason lacked "no
        comparison was made" — a string selected by `may_move`, a different
        variable, never by the delta content. Replacing the delta with a
        literal `[]` (the mute button this test claims to prevent) left it
        passing. It now asserts the COUNT.
        """
        self._run(tmp_path, dt.date(2026, 8, 28))
        store = LocalStore(tmp_path)
        incumbent = json.loads(store.get_bytes("board/current.json"))
        flipped = next(r for r in incumbent["rows"] if r["state"] == "UNMEASURED")
        flipped["state"] = "MET"
        store.put_bytes("board/current.json", json.dumps(incumbent).encode())

        written = self._run(tmp_path, dt.date(2026, 9, 1))
        manifest = json.loads(written.get_bytes("runs/board/2026-09-01/run.json"))
        moved = next(m for m in manifest["metrics"] if m["name"] == "board_rows_moved")
        assert moved["value"] > 0, (
            "a forward step must still diff — a guard that silences every delta is "
            "not a guard, it is a mute button"
        )
        assert flipped["id"] in moved["status_reason"]
        assert "no comparison was made" not in moved["status_reason"]

    def test_a_replay_does_not_leave_the_next_day_blind(self, tmp_path) -> None:
        """The two fixes compose.

        Because the replay did not clobber the pointer, the next forward board
        still diffs against the newest real board rather than against the
        replayed one — so a regression that happened while a replay ran is
        still reported the next day.
        """
        self._run(tmp_path, dt.date(2026, 9, 1))
        self._run(tmp_path, dt.date(2026, 8, 28))
        written = LocalStore(tmp_path)
        assert json.loads(written.get_bytes("board/current.json"))["trading_day"] == "2026-09-01"


# ── `_schedule_rows` — plan §6.1 milestones on the board (I9914 review) ────


class TestScheduleRows:
    """`_schedule_rows`'s own state machine, exercised directly rather than
    through `build_board` — the adversarial review on PR63 found both of its
    blocking defects here: the date boundary and the UNMEASURED fall-through
    were never given a fixture of their own (`crucible/board.py::_schedule_rows`).

    Fixed date literals throughout, per `crucible/AGENTS.md` test discipline.
    """

    @staticmethod
    def _row(phase_id: str, gate_state: str, *, met: int | None = None, total: int | None = None):
        (phase,) = [p for p in PHASES if p.id == phase_id]

        class _Row:
            pass

        row = _Row()
        row.phase = phase
        row.gate_state = gate_state
        row.clauses_met = met
        row.clauses_total = total
        row.read_on = None
        return row

    @staticmethod
    def _ladder(*rows):
        class _Ladder:
            pass

        ladder = _Ladder()
        ladder.rows = list(rows)
        return ladder

    @staticmethod
    def _schedule_row(rows_ladder, trading_day: str, milestone_id: str):
        from crucible.board import _schedule_rows

        want = f"schedule:{milestone_id}"
        (row,) = [r for r in _schedule_rows(rows_ladder, trading_day) if r.id == want]
        return row

    # -- finding 1: the deadline is the anchor day itself, not the day before --

    def test_the_anchor_day_itself_is_not_yet_overdue(self) -> None:
        """day5_replays: plan 2026-09-06 (Sun), anchor 2026-09-04 (Fri). The
        review's false red fired ON the anchor day (`>=`); the fix must not."""
        ladder = self._ladder(self._row("phase1", "UNMET", met=0, total=6))
        row = self._schedule_row(ladder, "2026-09-04", "day5_replays")
        assert row.state == "PLANNED"
        assert "OVERDUE" not in row.detail

    def test_the_day_before_the_anchor_is_planned(self) -> None:
        ladder = self._ladder(self._row("phase1", "UNMET", met=0, total=6))
        row = self._schedule_row(ladder, "2026-09-03", "day5_replays")
        assert row.state == "PLANNED"
        assert "OVERDUE" not in row.detail

    def test_the_day_after_the_anchor_is_overdue(self) -> None:
        ladder = self._ladder(self._row("phase1", "UNMET", met=0, total=6))
        row = self._schedule_row(ladder, "2026-09-05", "day5_replays")
        assert row.state == "UNMET"
        assert row.detail.startswith("OVERDUE")

    def test_a_saturday_milestone_is_not_overdue_on_its_friday_anchor(self) -> None:
        """live1: plan 2026-09-12 (Sat), anchor 2026-09-11 (Fri). The review's
        false red fired a full calendar day before the Saturday it measures."""
        ladder = self._ladder(self._row("phase2", "UNMET", met=1, total=5))
        row = self._schedule_row(ladder, "2026-09-11", "live1")
        assert row.state == "PLANNED"
        assert "OVERDUE" not in row.detail

    def test_a_saturday_milestone_is_overdue_the_monday_after(self) -> None:
        ladder = self._ladder(self._row("phase2", "UNMET", met=1, total=5))
        row = self._schedule_row(ladder, "2026-09-14", "live1")
        assert row.state == "UNMET"
        assert row.detail.startswith("OVERDUE")

    def test_the_overdue_detail_quotes_the_plans_own_calendar_date(self) -> None:
        """Finding 1: the rendered text must not name only the anchored
        session — a reader has to be able to check it against plan §6.1."""
        ladder = self._ladder(self._row("phase2", "UNMET", met=1, total=5))
        row = self._schedule_row(ladder, "2026-09-14", "live1")
        assert "2026-09-12" in row.detail

    # -- finding 2: an UNMEASURED phase reading is never rendered as a breach --

    def test_an_unmeasured_phase_reading_renders_unmeasured_past_its_due_date(self) -> None:
        """live1 names phase2, which has no registered gate (`PHASES`) and
        reads UNMEASURED on the ladder. Rendered well past the anchor, this
        must stay UNMEASURED, never a date-driven UNMET/OVERDUE."""
        ladder = self._ladder(self._row("phase2", "UNMEASURED"))
        row = self._schedule_row(ladder, "2026-09-14", "live1")
        assert row.state == "UNMEASURED"
        assert not row.detail.startswith("OVERDUE")
        assert "phase2" in row.detail

    def test_an_unmeasured_phase_reading_renders_unmeasured_before_its_due_date(self) -> None:
        ladder = self._ladder(self._row("phase2", "UNMEASURED"))
        row = self._schedule_row(ladder, "2026-09-05", "live1")
        assert row.state == "UNMEASURED"

    def test_an_unmeasured_reading_never_asserts_a_breach(self) -> None:
        """The exact defect: UNMET is an assertion the milestone was missed,
        derived from a reading that does not exist."""
        ladder = self._ladder(self._row("phase2", "UNMEASURED"))
        row = self._schedule_row(ladder, "2026-09-14", "live1")
        assert row.state != "UNMET"

    # -- the MET and PLANNED paths, and the no-ladder-supplied path --

    def test_a_met_phase_renders_met_regardless_of_date(self) -> None:
        ladder = self._ladder(self._row("phase1", "MET", met=6, total=6))
        row = self._schedule_row(ladder, "2026-09-01", "day5_replays")
        assert row.state == "MET"

    def test_no_ladder_supplied_renders_unmeasured(self) -> None:
        from crucible.board import _schedule_rows

        (row,) = [r for r in _schedule_rows(None, "2026-09-14") if r.id == "schedule:live1"]
        assert row.state == "UNMEASURED"


# ── the page is the detailed artifact (alpha-engine-config-I9921) ──────────


class TestThePageIsTheDetailedArtifact:
    """Brian, 2026-09-03: *"I don't find the report detailed enough."*

    The Telegram message is capped at 4096 characters and must truncate; this
    page is where the detail lives, so what is asserted here is that each
    thing the message drops is actually PRESENT — and that an absence is
    stated rather than rendered as a gap.
    """

    DAY = dt.date(2026, 8, 28)

    @staticmethod
    def _page(tmp_path, *, acceptance: dict[str, Any] | None = None) -> str:
        import argparse

        from crucible.track_c import board_handler

        store = LocalStore(tmp_path)
        if acceptance is not None:
            store.put_bytes(
                acceptance_reading_key(TestThePageIsTheDetailedArtifact.DAY.isoformat()),
                json.dumps(acceptance).encode(),
            )
        board_handler(
            argparse.Namespace(
                trading_day=TestThePageIsTheDetailedArtifact.DAY, store=str(tmp_path)
            )
        )
        return store.get_bytes("board/index.html").decode()

    def test_every_source_kind_is_grouped_onto_the_page(self, tmp_path) -> None:
        page = self._page(tmp_path)
        for source in SOURCES:
            assert f"<h2>{source} (" in page, f"the {source} rows are not grouped on the page"

    def test_every_row_carries_its_store_key_and_when_it_was_last_read(self, tmp_path) -> None:
        """A row with no key is a red dot a reader has to ask an agent about,
        and a reading with no stamp cannot be told from one taken in March.

        The header names `last read` because the cell renders
        `BoardRow.last_read` (review F5): the column was headed `generated at`
        while carrying a different fact, which makes every stamp under it a
        misquotation rather than a missing one.
        """
        page = self._page(tmp_path)
        assert "<th>store key</th>" in page
        assert "<th>last read</th>" in page
        assert "<th>generated at</th>" not in page, (
            "the column renders row.last_read; a `generated at` header misnames every cell"
        )
        # Every row renders a stamp cell; the unread ones say so in words
        # rather than leaving a cell that reads as a formatting gap.
        assert "never read" in page

    def test_a_phase_row_carries_every_clause_with_its_own_state_and_reason(self, tmp_path) -> None:
        page = self._page(tmp_path)
        reading = evaluate(LocalStore(tmp_path), gate="phase0", trading_day=self.DAY)
        clauses = [c.name for c in reading.clauses]
        assert clauses, "the fixture must exercise a gate that declares clauses"
        for name in clauses:
            assert name in page, f"clause {name} is not on the page"
        assert "requires:" in page, "a clause with no requirement stated is not actionable"

    def test_a_phase_row_with_no_reading_says_so_rather_than_showing_no_clauses(self) -> None:
        """`None` and `()` are different facts. A page that rendered them
        identically would make an unread gate look like a gate with no
        conditions — 0 of 0, which reads as complete."""
        from crucible.board import _clause_list_html

        unread = _row_fixture(clauses=None)
        empty = _row_fixture(clauses=())
        assert "no gate reading was supplied" in _clause_list_html(unread)
        assert "declares no clauses" in _clause_list_html(empty)
        assert _clause_list_html(unread) != _clause_list_html(empty)

    def test_a_non_phase_row_with_no_clause_list_renders_nothing_extra(self) -> None:
        """Only phase rows have gates. Printing "no reading" under every
        objective and component row would be noise wearing honesty's clothes."""
        from crucible.board import _clause_list_html

        assert _clause_list_html(_row_fixture(source="objective", clauses=None)) == ""

    def test_the_acceptance_section_is_present_even_when_the_artifact_is_absent(
        self, tmp_path
    ) -> None:
        """§12 rule 3's only progress figure is the last number this page may
        go quiet about: an omitted section and a broken producer look the same."""
        page = self._page(tmp_path)
        assert "acceptance (plan §2" in page
        assert "No acceptance reading on this store" in page

    def test_a_filed_acceptance_reading_is_rendered_with_its_clause_ids(self, tmp_path) -> None:
        page = self._page(
            tmp_path,
            acceptance={
                "met": 3,
                "unmet": 2,
                "unmeasurable": 1,
                "commit": "abc123",
                "measured_at": "2026-08-28T12:00:00Z",
                "unmet_clauses": ["clause_replays_run", "clause_cost_measured"],
                "unmeasurable_clauses": ["clause_cost_reachable"],
            },
        )
        assert "3 met / 2 unmet / 1 unmeasurable</strong> of 6 clauses" in page
        assert "clause_replays_run" in page
        assert "clause_cost_reachable" in page

    def test_a_partial_acceptance_reading_is_refused_rather_than_half_rendered(
        self, tmp_path
    ) -> None:
        """Three of four fields of the only number the plan calls progress is
        a fabrication that looks exactly like a measurement."""
        page = self._page(tmp_path, acceptance={"met": 3, "commit": "abc123"})
        assert "answers a different question" in page
        assert "3 met" not in page

    def test_an_acceptance_reading_with_no_clause_ids_says_so(self, tmp_path) -> None:
        """The counts alone are honest; claiming to list clauses that are not
        on the artifact would not be."""
        page = self._page(
            tmp_path,
            acceptance={"met": 3, "unmet": 2, "unmeasurable": 1, "commit": "abc123"},
        )
        assert "names no unmet clause ids" in page

    def test_the_clause_states_on_the_page_are_the_ladders_own_reading(self, tmp_path) -> None:
        """One evaluation, two surfaces. A page that re-evaluated the gate
        could disagree with the row it sits inside."""
        self._page(tmp_path)
        board = json.loads(LocalStore(tmp_path).get_bytes("board/current.json"))
        phase = next(r for r in board["rows"] if r["id"] == "phase:phase0")
        met = sum(1 for c in phase["clauses"] if c["met"])
        assert f"{met}/{len(phase['clauses'])} clauses met" in phase["detail"]


def _row_fixture(*, source: str = "phase", clauses: Any = None) -> BoardRow:
    return BoardRow(
        id=f"{source}:x",
        source=source,
        title="x",
        state="UNMET",
        detail="d",
        surface="crucible/board",
        artifact="k.json",
        means_when_red="r",
        clauses=clauses,
    )
