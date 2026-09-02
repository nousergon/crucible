"""The phase ladder is the ONE durable surface for the plan §6 gate reading.

The defect these tests exist for: on `alpha-engine-config-I9757` the phase-1
reading was published only as prose in issue comments, by hand, and moved
18 met / 6 unmet -> 19 met / 4 unmet -> 21 of 23 across three of them, one
explicitly correcting another. Phase 1 was then CLOSED on 2026-09-02 while
phase 0's gate had never been read at all, and no surface showed that.

Every test below was seen failing before the code that makes it pass, and
each asserts something the ladder REFUSES: a never-measured phase refusing to
render green, a never-measured phase refusing to render zero, and a later
phase refusing to look ordinary while an earlier gate is unmet.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.console.render import STATUS_COLORS, build_page, render_html, write_page
from crucible.gate import (
    GATES,
    LADDER_CONSOLE_STATE,
    LADDER_KEY,
    LADDER_STATES,
    PHASES,
    Ladder,
    build_ladder,
    gate_key,
    ladder_payload,
    last_read,
)
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def _file_reading(store: LocalStore, gate: str, day: dt.date, *, met: bool) -> None:
    """A gate reading in the shape `crucible gate` files, at its real key."""
    store.put_bytes(
        gate_key(gate, day.isoformat()),
        json.dumps({"gate": gate, "trading_day": day.isoformat(), "met": met}).encode("utf-8"),
    )


class TestTheLadderIsDeclaredWhole:
    def test_every_plan_phase_has_a_rung(self) -> None:
        """Six phases, 0..5, each bound to its tracker issue.

        A ladder that listed only the phases with gates written would be a
        ladder on which an unwritten gate is invisible rather than UNMEASURED.
        """
        assert [p.number for p in PHASES] == [0, 1, 2, 3, 4, 5]
        assert [p.issue for p in PHASES] == [9756, 9757, 9758, 9759, 9760, 9761]

    def test_a_phase_row_is_identified_by_its_TRACKER_ref(self, store: LocalStore) -> None:
        """`console-policy` §2.1: the identifier is the tracker ref the host
        assigns, so a `git-host` claim about the same issue MERGES onto this
        row instead of rendering the phase a second time."""
        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        ids = [row["decision_id"] for row in ladder.to_dict()["phases"]]
        assert ids[0] == "alpha-engine-config-I9756"
        assert all(i.startswith("alpha-engine-config-I") for i in ids)

    def test_every_declared_gate_name_is_registered_or_absent(self) -> None:
        """A phase may name no gate, or a gate that exists. It may never name
        one that does not — that would render a phase permanently UNMEASURED
        for a reason nobody could find."""
        for phase in PHASES:
            assert phase.gate is None or phase.gate in GATES

    def test_every_ladder_state_declares_how_it_renders(self) -> None:
        assert set(LADDER_STATES) == set(LADDER_CONSOLE_STATE)
        assert set(LADDER_STATES) <= set(STATUS_COLORS)

    def test_a_ladder_state_with_no_console_mapping_raises_ValueError_not_AssertionError(
        self,
    ) -> None:
        """`alpha-engine-config-I9826`. `assert` is compiled out under
        `python -O`/`PYTHONOPTIMIZE` — the guard that stops an undeclared
        ladder state reaching a console surface must not be the one construct
        guaranteed absent in an optimized interpreter. This is the exact
        function the module calls at import time, so a real gap fails the
        same way whether or not the interpreter is optimized."""
        from crucible.gate import _check_ladder_console_coverage

        with pytest.raises(ValueError, match="A_NEW_STATE"):
            _check_ladder_console_coverage(
                (*LADDER_STATES, "A_NEW_STATE"), dict(LADDER_CONSOLE_STATE)
            )
        # No gap: the real, current vocabulary passes without raising.
        _check_ladder_console_coverage(LADDER_STATES, dict(LADDER_CONSOLE_STATE))


class TestAbsenceRendersAsAbsence:
    def test_an_unwritten_gate_is_UNMEASURED_not_MET(self, store: LocalStore) -> None:
        """Principle 7. Phases 0 and 2-5 have no clause list today; not one of
        them may render as a phase that passed."""
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        for phase_id in ("phase0", "phase2", "phase3", "phase4", "phase5"):
            assert rows[phase_id]["state"] == "UNMEASURED"
            assert rows[phase_id]["console_state"] == "UNREPORTED"

    def test_an_unmeasured_phase_reports_a_NULL_ratio_never_zero(self, store: LocalStore) -> None:
        """Zero is a measurement; absence is not. A ladder that published
        `met_ratio: 0.0` for a phase nobody has ever graded is publishing a
        figure it did not measure."""
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase0"]["met_ratio"] is None
        assert rows["phase0"]["clauses_total"] is None
        assert rows["phase0"]["clauses_met"] is None

    def test_a_row_says_when_it_was_last_read_and_null_when_never(self, store: LocalStore) -> None:
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase1"]["read_on"] is None

        _file_reading(store, "phase1", FRIDAY - dt.timedelta(weeks=1), met=False)
        _file_reading(store, "phase1", FRIDAY, met=False)
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase1"]["read_on"] == FRIDAY.isoformat()
        assert last_read(store, "phase1") == FRIDAY.isoformat()
        assert last_read(store, "phase0") is None

    def test_the_unmeasured_COUNT_is_published_not_left_to_be_counted(
        self, store: LocalStore
    ) -> None:
        """Same reasoning as the transparency gap: a number nobody publishes
        is a number nobody is held to."""
        document = build_ladder(store, trading_day=FRIDAY).to_dict()
        assert document["unmeasured"] == 5
        assert document["phases_met"] == 0

    def test_the_html_renders_never_measured_not_a_blank_or_a_zero(self, store: LocalStore) -> None:
        page = build_page(store, now=NOW)
        html = render_html(page)
        assert "Phase ladder" in html
        assert "never measured" in html
        assert "alpha-engine-config-I9756" in html


class TestTheLadderKnowsWhereItIs:
    def test_current_phase_is_the_LOWEST_unmet_phase(self, store: LocalStore) -> None:
        """Not the highest phase with work in it. Phase 0's gate is unwritten,
        so the ladder is at phase 0 however much phase-1 code has landed."""
        assert build_ladder(store, trading_day=FRIDAY).current_phase == "phase0"

    def test_a_gate_with_no_clauses_is_UNMEASURED_not_vacuously_MET(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`all([])` is True. A gate whose clause list emptied would otherwise
        report every phase behind it as exited."""
        monkeypatch.setitem(GATES, "phase1", (5, lambda *_a, **_k: []))
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase1"]["state"] == "UNMEASURED"
        assert rows["phase1"]["met_ratio"] is None

    def test_the_ladder_row_and_the_dated_gate_artifact_cannot_disagree(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`alpha-engine-config-I9824`. The durable, dated artifact at
        `gates/{gate}/{trading_day}/gate.json` (`GateResult.to_dict`) and the
        overwritten `gates/ladder.json` row (`PhaseRow.to_dict`) must publish
        the identical `met_ratio` for the same reading — by construction,
        because `build_ladder` reads `GateResult.met_ratio` rather than
        re-deriving it, not merely by two authors agreeing.

        Checked at both ends of the vocabulary: an emptied clause list
        (`None` on both surfaces) and a real 0/5 measurement (`0.0` on both
        surfaces, which is the correct, MEASURED zero — not the false one
        this issue is about).
        """
        from crucible.gate import Clause, evaluate

        monkeypatch.setitem(GATES, "phase1", (5, lambda *_a, **_k: []))
        gate_reading = evaluate(store, gate="phase1", trading_day=FRIDAY)
        ladder_rows = {
            r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]
        }
        assert gate_reading.to_dict()["met_ratio"] is None
        assert ladder_rows["phase1"]["met_ratio"] is None
        assert gate_reading.to_dict()["met_ratio"] == ladder_rows["phase1"]["met_ratio"]

        monkeypatch.setitem(
            GATES,
            "phase1",
            (5, lambda *_a, **_k: [Clause("c", "req", False, "unmet", ())]),
        )
        gate_reading = evaluate(store, gate="phase1", trading_day=FRIDAY)
        ladder_rows = {
            r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]
        }
        assert gate_reading.to_dict()["met_ratio"] == 0.0
        assert ladder_rows["phase1"]["met_ratio"] == 0.0

    def test_a_supplied_reading_is_reused_not_re_evaluated(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`alpha-engine-config-I9826` finding 2: `gate_handler` used to call
        `evaluate` for `phase1` once itself and again inside `build_ladder`,
        doubling every store read the clause set makes. `readings` lets the
        caller hand in what it already computed; the supplied gate must not
        be evaluated a second time."""
        from crucible.gate import GateResult, evaluate

        calls = {"n": 0}
        real_evaluate = evaluate

        def counting_evaluate(*args, **kwargs):
            calls["n"] += 1
            return real_evaluate(*args, **kwargs)

        monkeypatch.setattr("crucible.gate.evaluate", counting_evaluate)
        reading = real_evaluate(store, gate="phase1", trading_day=FRIDAY)
        calls["n"] = 0  # only count calls made during build_ladder below

        ladder = build_ladder(store, trading_day=FRIDAY, readings={"phase1": reading})
        assert calls["n"] == 0

        rows = {r["phase"]: r for r in ladder.to_dict()["phases"]}
        assert rows["phase1"]["clauses_met"] == sum(1 for c in reading.clauses if c.met)
        assert rows["phase1"]["clauses_total"] == len(reading.clauses)
        assert isinstance(reading, GateResult)


class TestAnOutOfOrderPhaseIsVisibleAsSuch:
    def test_a_graded_later_phase_over_an_unmet_earlier_one_is_OUT_OF_ORDER(
        self, store: LocalStore
    ) -> None:
        """Brian's ruling of 2026-09-02: phase 0's gate must READ before phase
        2 opens. Phase 1 has readings filed and phase 0 has never been
        measured, so phase 1 is being graded ahead of the ladder."""
        _file_reading(store, "phase1", FRIDAY, met=False)
        ladder = build_ladder(store, trading_day=FRIDAY)
        rows = {r["phase"]: r for r in ladder.to_dict()["phases"]}
        assert rows["phase1"]["state"] == "OUT_OF_ORDER"
        assert rows["phase1"]["blocked_by"] == "phase0"
        assert rows["phase1"]["console_state"] == "FAILED"
        assert ladder.out_of_order == ["phase1"]

    def test_the_underlying_gate_reading_survives_the_out_of_order_verdict(
        self, store: LocalStore
    ) -> None:
        """`gate_state` is kept beside `state`. An ordering breach must not
        erase the measurement it was found in — that would trade one blind
        surface for another."""
        _file_reading(store, "phase1", FRIDAY, met=False)
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase1"]["state"] == "OUT_OF_ORDER"
        assert rows["phase1"]["gate_state"] == "UNMET"
        assert rows["phase1"]["clauses_total"] == 5

    def test_an_UNGRADED_later_phase_is_not_called_out_of_order(self, store: LocalStore) -> None:
        """Phases 2-5 have never been read. They are UNMEASURED, which is the
        honest word; calling them out of order would page on the ladder simply
        being early."""
        _file_reading(store, "phase1", FRIDAY, met=False)
        ladder = build_ladder(store, trading_day=FRIDAY)
        assert ladder.out_of_order == ["phase1"]

    def test_the_reason_names_the_phase_that_blocks_it(self, store: LocalStore) -> None:
        _file_reading(store, "phase1", FRIDAY, met=False)
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert "phase0" in rows["phase1"]["detail"]
        assert "alpha-engine-config-I9756" in rows["phase1"]["detail"]


class TestTheLadderIsPublishedWhereSomethingReadsIt:
    def test_the_console_job_writes_the_ladder_artifact(self, store: LocalStore) -> None:
        """Deliverable 4: the reading refreshes on the `console` job's weekly
        arc cadence, not only when somebody types `crucible gate`."""
        page = build_page(store, now=NOW)
        keys = write_page(store, page)
        assert LADDER_KEY in keys
        published = json.loads(store.get_bytes(LADDER_KEY).decode("utf-8"))
        assert published["schema_version"] == "phase_ladder.v1"
        assert len(published["phases"]) == len(PHASES)

    def test_the_published_bytes_are_the_same_shape_from_both_publishers(
        self, store: LocalStore
    ) -> None:
        """`crucible gate` and `crucible console` both write `gates/ladder.json`.
        Two publishers of one key that disagreed on its shape would give the
        console's adapter a row whose fields depend on which job ran last."""
        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        page = build_page(store, now=NOW)
        write_page(store, page)
        from_gate = json.loads(ladder_payload(ladder).decode("utf-8"))
        from_console = json.loads(store.get_bytes(LADDER_KEY).decode("utf-8"))
        assert set(from_gate) == set(from_console)
        assert from_gate["phases"][0].keys() == from_console["phases"][0].keys()

    def test_the_render_survives_a_ladder_that_was_never_built(self) -> None:
        """An empty ladder renders as an absence, never as a complete ladder
        with no rows in it."""
        from crucible.console.render import ConsolePage

        html = render_html(ConsolePage(trading_day="2026-08-28", generated_utc="x"))
        assert "No ladder was built" in html


class TestThePhaseLadderSchemaContract:
    """`alpha-engine-config-I9825`. `gates/ladder.json` has two producers and
    a declared version; this class is the producer/consumer contract test
    the M0 rule (`~/Development/CLAUDE.md`) requires at its birth."""

    def test_the_schema_file_exists_and_is_a_valid_schema(self) -> None:
        from jsonschema import Draft202012Validator

        from crucible.gate import ladder_schema

        Draft202012Validator.check_schema(ladder_schema())

    def test_bytes_from_both_producers_validate_against_the_schema(self, store: LocalStore) -> None:
        from crucible.gate import validate_ladder_document

        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        validate_ladder_document(json.loads(ladder_payload(ladder).decode("utf-8")))

        page = build_page(store, now=NOW)
        write_page(store, page)
        validate_ladder_document(json.loads(store.get_bytes(LADDER_KEY).decode("utf-8")))

    def test_the_gate_jobs_manifest_records_the_ladders_own_schema_version(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Measured on `crucible-PR30` head `a124381`: `gate_handler` recorded
        `record_output`'s bare default (`v1`), not `phase_ladder.v1`, for
        `gates/ladder.json`."""
        import argparse

        from crucible.gate import GATES, Clause
        from crucible.store import LocalStore
        from crucible.track_f import gate_handler

        monkeypatch.setitem(
            GATES, "phase1", (5, lambda *_a, **_k: [Clause("c", "req", True, "met", ())])
        )
        args = argparse.Namespace(
            gate="phase1", trading_day=FRIDAY, weeks=None, store=str(tmp_path)
        )
        gate_handler(args)

        store = LocalStore(tmp_path)
        manifest = json.loads(store.get_bytes(f"runs/gate/{FRIDAY.isoformat()}/run.json"))
        (ladder_output,) = [o for o in manifest["outputs"] if o["key"] == LADDER_KEY]
        assert ladder_output["schema_version"] == "phase_ladder.v1"

    def test_the_console_jobs_manifest_also_records_the_ladders_own_schema_version(
        self, tmp_path
    ) -> None:
        """Measured on `crucible-PR30` head `a124381`: `console_handler`
        stamped every `write_page` key, ladder included, with the single
        literal `console_page.v1`."""
        import argparse

        from crucible.store import LocalStore
        from crucible.track_c import console_handler

        args = argparse.Namespace(trading_day=FRIDAY, store=str(tmp_path))
        console_handler(args)

        store = LocalStore(tmp_path)
        manifest = json.loads(store.get_bytes(f"runs/console/{FRIDAY.isoformat()}/run.json"))
        (ladder_output,) = [o for o in manifest["outputs"] if o["key"] == LADDER_KEY]
        assert ladder_output["schema_version"] == "phase_ladder.v1"

    def test_an_out_of_vocabulary_state_is_refused_by_the_schema(self, store: LocalStore) -> None:
        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        document = ladder.to_dict()
        document["phases"][0]["state"] = "PENDING"

        from crucible.gate import validate_ladder_document

        with pytest.raises(ValueError, match="does not conform"):
            validate_ladder_document(document)


class TestTheLadderRendersForAHuman:
    def test_render_names_the_current_phase_and_the_last_read(self, store: LocalStore) -> None:
        _file_reading(store, "phase1", FRIDAY, met=False)
        text = build_ladder(store, trading_day=FRIDAY, now=NOW).render()
        assert "at phase0" in text
        assert "last read never" in text
        assert f"last read {FRIDAY.isoformat()}" in text

    def test_a_complete_ladder_says_complete(self, store: LocalStore) -> None:
        ladder = Ladder(trading_day=FRIDAY, generated_utc="2026-08-29T12:00:00Z")
        assert ladder.current_phase == "complete"
