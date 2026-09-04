"""The closing reading a phase issue must be closed with.

Tracker: `alpha-engine-config-I9967`. Plan §6 rule 2.

`alpha-engine-config-I9757` was closed on 2026-09-02 while its own gate read
1 of 6 clauses met and the phase beneath it was unmet. The convention that was
supposed to prevent that — paste the reading into the closing comment — had
already failed once. These tests grade the machine-checkable replacement, and
every one of them is written as a REFUSAL: the interesting property is not
that a real MET reading is accepted, it is that a block claiming MET while its
own transcript says otherwise is not.

The three mutation guards (`test_refuses_a_forged_*`) exist because the
obvious way to defeat this mechanism is to hand-edit `gate_state` to `"MET"`
in a block that was rendered UNMET. Each one flips exactly that one field and
asserts the refusal still fires off a field the forger did not think to
change.
"""

from __future__ import annotations

import copy
import datetime as dt
import json

import pytest

from crucible.gate import (
    CLOSING_READING_FENCE,
    CLOSING_READING_SCHEMA_VERSION,
    PHASES,
    Clause,
    GateResult,
    closing_reading,
    closing_reading_refusals,
    gate_state_for,
    parse_closing_comment,
    phase_for_gate,
    render_closing_comment,
    validate_closing_reading_document,
)

# Fixed literals, never `today` arithmetic (AGENTS.md test discipline).
DAY = dt.date(2026, 8, 28)
WINDOW = [dt.date(2026, 8, 27), DAY]
STORE = "s3://example-store/crucible"
COMMIT = "b0021a2ef19c4d5a6b7c8d9e0f1a2b3c4d5e6f70"


def _clause(name: str, *, met: bool = True, unmeasurable: bool = False) -> Clause:
    return Clause(
        name=name,
        requirement=f"{name} holds",
        met=met,
        detail=f"{name}: {'met' if met else 'not met'}",
        unmeasurable=unmeasurable,
    )


def _reading(*clauses: Clause, gate: str = "phase1") -> GateResult:
    return GateResult(gate=gate, trading_day=DAY, window=list(WINDOW), clauses=list(clauses))


def _met_reading() -> GateResult:
    return _reading(_clause("arc_runs_ok"), _clause("arms_all_scored"))


def _met_document() -> dict:
    return closing_reading(_met_reading(), store_uri=STORE, commit=COMMIT)


# --------------------------------------------------------------------------
# gate_state_for — the one derivation, shared with the ladder row
# --------------------------------------------------------------------------


def test_gate_state_is_unmeasurable_when_any_clause_is() -> None:
    reading = _reading(_clause("a"), _clause("b", met=False, unmeasurable=True))
    assert gate_state_for(reading) == "UNMEASURABLE"


def test_gate_state_unmeasurable_outranks_a_gate_that_would_otherwise_read_unmet() -> None:
    # "it says no" and "I could not ask" are different facts and the second is
    # about us — the property `alpha-engine-config-I9869` round 3 established
    # for the ladder, asserted here because the closing block now reads it too.
    reading = _reading(_clause("a", met=False), _clause("b", met=False, unmeasurable=True))
    assert gate_state_for(reading) == "UNMEASURABLE"


def test_gate_state_of_an_empty_clause_list_is_never_met() -> None:
    assert gate_state_for(_reading()) == "UNMET"


def test_the_ladder_row_and_the_block_read_the_same_state_function() -> None:
    # Not a re-derivation: `build_ladder` calls `gate_state_for`, so this
    # asserts the two surfaces cannot drift by construction rather than
    # asserting today's two values happen to match.
    import inspect

    from crucible import gate as gate_module

    source = inspect.getsource(gate_module.build_ladder)
    assert "gate_state_for(reading)" in source


# --------------------------------------------------------------------------
# phase_for_gate — no invented tracker numbers
# --------------------------------------------------------------------------


def test_phase_for_gate_resolves_every_registered_phase() -> None:
    for phase in PHASES:
        assert phase.gate is not None
        assert phase_for_gate(phase.gate) is phase


def test_phase_for_gate_refuses_a_gate_no_phase_reads() -> None:
    with pytest.raises(KeyError, match="no registered phase reads"):
        phase_for_gate("phase99")


# --------------------------------------------------------------------------
# closing_reading — a transcript, not a claim
# --------------------------------------------------------------------------


def test_closing_reading_carries_the_tracker_derived_from_phases() -> None:
    document = _met_document()
    assert document["tracker"] == phase_for_gate("phase1").tracker
    assert document["schema_version"] == CLOSING_READING_SCHEMA_VERSION
    assert document["gate_artifact"] == f"gates/phase1/{DAY.isoformat()}/gate.json"
    assert document["store"] == STORE
    assert document["commit"] == COMMIT


def test_closing_reading_renders_an_unmet_gate_rather_than_refusing_to() -> None:
    # "No block" must mean exactly one thing — nobody measured. If the renderer
    # refused an unmet gate it would also mean "measured and failing", and the
    # sweep could not tell the two apart.
    reading = _reading(_clause("a"), _clause("b", met=False))
    document = closing_reading(reading, store_uri=STORE, commit=COMMIT)
    assert document["gate_state"] == "UNMET"
    assert document["clauses_met"] == 1
    assert document["clauses_total"] == 2


def test_closing_reading_met_ratio_is_null_not_zero_when_nothing_was_measured() -> None:
    document = closing_reading(_reading(), store_uri=STORE, commit=COMMIT)
    assert document["met_ratio"] is None
    assert document["clauses_total"] == 0


def test_closing_reading_refuses_a_missing_store() -> None:
    with pytest.raises(ValueError, match="needs the store URI"):
        closing_reading(_met_reading(), store_uri="", commit=COMMIT)


@pytest.mark.parametrize("bad", ["", "unknown", "HEAD", "b0021a2", "B0021A2EF19C"])
def test_closing_reading_refuses_a_commit_that_is_not_real_provenance(bad: str) -> None:
    with pytest.raises(ValueError, match="lowercase hex"):
        closing_reading(_met_reading(), store_uri=STORE, commit=bad)


def test_closing_reading_validates_against_its_own_schema() -> None:
    validate_closing_reading_document(_met_document())


def test_validation_refuses_an_unknown_field() -> None:
    document = _met_document()
    document["exit_approved_by"] = "me"
    with pytest.raises(ValueError, match="does not conform"):
        validate_closing_reading_document(document)


# --------------------------------------------------------------------------
# render / parse round trip
# --------------------------------------------------------------------------


def test_render_then_parse_round_trips_the_document() -> None:
    document = _met_document()
    comment = render_closing_comment(document)
    assert f"```{CLOSING_READING_FENCE}" in comment
    assert parse_closing_comment(comment) == document


def test_render_puts_the_verdict_in_prose_as_well_as_json() -> None:
    comment = render_closing_comment(_met_document())
    assert "**MET**" in comment
    assert "2/2 clauses met" in comment
    assert STORE in comment
    assert COMMIT in comment


def test_parse_returns_none_when_there_is_no_block() -> None:
    assert parse_closing_comment("closing — the build PRs all merged") is None
    assert parse_closing_comment("") is None


def test_parse_ignores_a_plain_json_fence_that_is_not_a_reading() -> None:
    # A closing comment may legitimately quote JSON. Only the declared fence
    # counts, or "there is a reading here" becomes "someone pasted a dict".
    assert parse_closing_comment('```json\n{"gate_state": "MET"}\n```') is None


def test_parse_raises_on_a_present_but_malformed_block() -> None:
    # A corrupted paste is not an absent reading. Returning None here would
    # let a mangled block read as "not measured yet" and, once the sweep is
    # green again, as nothing at all.
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_closing_comment(f"```{CLOSING_READING_FENCE}\n{{not json}}\n```")


def test_parse_raises_when_the_block_is_json_but_not_an_object() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_closing_comment(f"```{CLOSING_READING_FENCE}\n[1, 2, 3]\n```")


def test_parse_takes_the_last_block_when_a_comment_carries_several() -> None:
    first = _met_document()
    second = copy.deepcopy(first)
    second["trading_day"] = "2026-08-27"
    text = (
        f"```{CLOSING_READING_FENCE}\n{json.dumps(first)}\n```\n"
        f"corrected:\n```{CLOSING_READING_FENCE}\n{json.dumps(second)}\n```"
    )
    assert parse_closing_comment(text)["trading_day"] == "2026-08-27"


# --------------------------------------------------------------------------
# closing_reading_refusals — the predicate the cross-repo sweep enforces
# --------------------------------------------------------------------------


def test_a_real_met_reading_is_accepted() -> None:
    assert closing_reading_refusals(_met_document(), phase_id="phase1") == []


def test_no_block_at_all_is_refused_and_names_the_command_that_makes_one() -> None:
    problems = closing_reading_refusals(None, phase_id="phase1")
    assert len(problems) == 1
    assert "--closing-comment" in problems[0]


def test_an_unmet_reading_is_refused() -> None:
    document = closing_reading(
        _reading(_clause("a"), _clause("b", met=False)), store_uri=STORE, commit=COMMIT
    )
    problems = closing_reading_refusals(document, phase_id="phase1")
    assert any("'UNMET'" in p for p in problems)


def test_an_unmeasurable_reading_is_refused() -> None:
    document = closing_reading(
        _reading(_clause("a"), _clause("b", met=False, unmeasurable=True)),
        store_uri=STORE,
        commit=COMMIT,
    )
    problems = closing_reading_refusals(document, phase_id="phase1")
    assert any("UNMEASURABLE" in p for p in problems)


def test_a_gate_with_no_clauses_is_refused_rather_than_vacuously_met() -> None:
    document = closing_reading(_reading(), store_uri=STORE, commit=COMMIT)
    problems = closing_reading_refusals(document, phase_id="phase1")
    assert any("measured nothing" in p for p in problems)


def test_a_block_for_another_phase_is_refused() -> None:
    problems = closing_reading_refusals(_met_document(), phase_id="phase2")
    assert any("wrong issue" in p for p in problems)


def test_an_unknown_schema_version_is_refused_outright() -> None:
    document = _met_document()
    document["schema_version"] = "phase_closing_reading.v99"
    problems = closing_reading_refusals(document, phase_id="phase1")
    assert len(problems) == 1
    assert "schema_version" in problems[0]


def test_a_block_with_no_store_is_refused() -> None:
    document = _met_document()
    document["store"] = ""
    assert any("names no store" in p for p in closing_reading_refusals(document, phase_id="phase1"))


@pytest.mark.parametrize("bad", ["", "unknown", "b0021a2", None])
def test_a_block_with_no_usable_commit_is_refused(bad: object) -> None:
    document = _met_document()
    document["commit"] = bad
    assert any("lowercase hex" in p for p in closing_reading_refusals(document, phase_id="phase1"))


# --- the mutation guards: a forged `gate_state: MET` --------------------------


def _forged_from_unmet() -> dict:
    """An honest UNMET reading with `gate_state` hand-edited to MET."""
    document = closing_reading(
        _reading(_clause("a"), _clause("b", met=False)), store_uri=STORE, commit=COMMIT
    )
    document["gate_state"] = "MET"
    return document


def test_refuses_a_forged_state_because_the_counts_disagree() -> None:
    problems = closing_reading_refusals(_forged_from_unmet(), phase_id="phase1")
    assert any("records 1 of 2 clauses met" in p for p in problems)


def test_refuses_a_forged_state_because_met_ratio_is_not_one() -> None:
    problems = closing_reading_refusals(_forged_from_unmet(), phase_id="phase1")
    assert any("met_ratio is 0.5" in p for p in problems)


def test_refuses_a_forged_state_because_a_clause_row_says_otherwise() -> None:
    # The forger who fixed the counts and the ratio still has to fix every
    # per-clause verdict. This is the last of the three, and the one that makes
    # the block a transcript rather than a summary.
    document = _forged_from_unmet()
    document["clauses_met"] = 2
    document["met_ratio"] = 1.0
    problems = closing_reading_refusals(document, phase_id="phase1")
    assert any("not met on the block's own transcript" in p for p in problems)


def test_refuses_a_block_whose_clause_list_is_shorter_than_its_claim() -> None:
    document = _met_document()
    document["clauses"] = document["clauses"][:1]
    problems = closing_reading_refusals(document, phase_id="phase1")
    assert any("does not match its own summary" in p for p in problems)


def test_refuses_a_block_whose_clause_rows_were_deleted_entirely() -> None:
    document = _met_document()
    document["clauses"] = None
    problems = closing_reading_refusals(document, phase_id="phase1")
    assert any("no clause list" in p for p in problems)
