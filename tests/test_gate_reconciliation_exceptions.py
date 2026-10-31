"""Declared exceptions to `broker_reconciliation_control_arm_passed` (`alpha-engine-config-I12376`).

Brian ruled the 2026-10-26 (+$0.55) and 2026-10-27 (-$0.01) reconcile breaks
exceptions for the phase-4 clause. Each is a declared record matched on the
exact run and the exact break it recorded, the shape crucible-executor uses for
`-I12411`'s ledger gaps. These tests pin the two halves of that: the ruled run
counts, and nothing that merely resembles it does.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from test_gate_phase4_trader_clauses import (
    SESSIONS,
    WINDOW,
    _reconcile_run,
    _reconciliation,
)

from crucible.gate import (
    RECONCILIATION_EXCEPTIONS_PATH,
    ReconciliationException,
    _clause_broker_reconciliation_control_arm_passed,
    load_reconciliation_exceptions,
)
from crucible.store import LocalStore

EXCEPTED = SESSIONS[2]
RUN_ID = "01JG" + EXCEPTED.replace("-", "") + "00000000000000"
BREAK = {"klass": "cash_delta", "instrument": "USD", "delta": 0.55, "side": "broker_over"}
RULING = "the ruling, comment 1"


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path / "store")


def _exceptions(**overrides: Any) -> dict[dt.date, ReconciliationException]:
    fields: dict[str, Any] = {
        "session": dt.date.fromisoformat(EXCEPTED),
        "run_id": RUN_ID,
        "discrepancies": (("cash_delta", "USD", 0.55, "broker_over"),),
        "ruling": RULING,
        "ruled_by": "Brian 2026-10-27T15:11Z",
    }
    fields.update(overrides)
    return {fields["session"]: ReconciliationException(**fields)}


def _broken(day: str, *, discrepancies: list[dict] | None = None, **kwargs: Any) -> dict:
    document = _reconciliation(day, **kwargs)
    document["discrepancies"] = [dict(BREAK)] if discrepancies is None else discrepancies
    return document


def _week_with_break(store: LocalStore, **overrides: Any) -> None:
    document = overrides.pop("document", None) or _broken(EXCEPTED)
    for day in SESSIONS:
        if day == EXCEPTED:
            _reconcile_run(store, day, status="failed", document=document, **overrides)
        else:
            _reconcile_run(store, day)


def _clause(store: LocalStore, exceptions: dict[dt.date, ReconciliationException] | None):
    return _clause_broker_reconciliation_control_arm_passed(store, WINDOW, exceptions=exceptions)


class TestDeclaredExceptionApplies:
    def test_the_ruled_run_counts_and_the_detail_names_the_ruling(self, store) -> None:
        _week_with_break(store)
        clause = _clause(store, _exceptions())
        assert clause.met and not clause.unmeasurable, clause.detail
        assert EXCEPTED in clause.detail and RULING in clause.detail

    def test_without_the_exception_the_same_week_is_unmet(self, store) -> None:
        _week_with_break(store)
        clause = _clause(store, {})
        assert not clause.met
        assert "status `failed`" in clause.detail

    def test_a_clean_run_on_an_excepted_session_needs_nothing(self, store) -> None:
        for day in SESSIONS:
            _reconcile_run(store, day)
        clause = _clause(store, _exceptions())
        assert clause.met, clause.detail
        assert "ruled exception" not in clause.detail


class TestNothingThatResemblesTheRuledRunCounts:
    def test_a_rerun_with_another_run_id_is_unmet(self, store) -> None:
        _week_with_break(store)
        clause = _clause(store, _exceptions(run_id="01JGSOMEOTHERRUN000000000000"))
        assert not clause.met
        assert "does not apply" in clause.detail

    def test_a_different_delta_is_a_new_break(self, store) -> None:
        moved = dict(BREAK, delta=0.56)
        _week_with_break(store, document=_broken(EXCEPTED, discrepancies=[moved]))
        clause = _clause(store, _exceptions())
        assert not clause.met
        assert "a different break is a new finding" in clause.detail

    def test_an_extra_discrepancy_is_a_new_break(self, store) -> None:
        extra = {"klass": "share_delta", "instrument": "XYZ", "delta": 1, "side": "broker_over"}
        _week_with_break(store, document=_broken(EXCEPTED, discrepancies=[dict(BREAK), extra]))
        clause = _clause(store, _exceptions())
        assert not clause.met
        assert "a different break is a new finding" in clause.detail

    def test_an_exception_never_excuses_a_blind_control_arm(self, store) -> None:
        _week_with_break(store, document=_broken(EXCEPTED, passed=False, void=False))
        clause = _clause(store, _exceptions())
        assert not clause.met
        assert "control_arm.passed is False" in clause.detail

    def test_an_exception_never_excuses_a_replay(self, store) -> None:
        _week_with_break(store, run_mode="replay")
        clause = _clause(store, _exceptions())
        assert not clause.met
        assert "replay" in clause.detail

    def test_an_exception_for_one_session_leaves_another_failed_session_unmet(self, store) -> None:
        _week_with_break(store)
        _reconcile_run(store, SESSIONS[-1], status="failed", document=_broken(SESSIONS[-1]))
        clause = _clause(store, _exceptions())
        assert not clause.met
        assert SESSIONS[-1] in clause.detail


class TestNullDeltaIdentity:
    """The 2026-10-30 form: an incomplete row the reconciler records with a null delta."""

    INCOMPLETE = {
        "klass": "corporate_action_check_incomplete",
        "instrument": "SPY",
        "delta": None,
        "side": "trader",
    }

    def _null(self) -> dict[dt.date, ReconciliationException]:
        return _exceptions(
            discrepancies=(("corporate_action_check_incomplete", "SPY", None, "trader"),)
        )

    def test_the_ruled_null_delta_row_counts(self, store) -> None:
        _week_with_break(store, document=_broken(EXCEPTED, discrepancies=[dict(self.INCOMPLETE)]))
        clause = _clause(store, self._null())
        assert clause.met, clause.detail

    def test_a_null_delta_does_not_match_a_numeric_one(self, store) -> None:
        numeric = dict(self.INCOMPLETE, delta=0.0)
        _week_with_break(store, document=_broken(EXCEPTED, discrepancies=[numeric]))
        clause = _clause(store, self._null())
        assert not clause.met
        assert "a different break is a new finding" in clause.detail

    def test_a_null_and_a_numeric_row_on_one_instrument_still_order(self, store) -> None:
        rows = [dict(self.INCOMPLETE, delta=1.0), dict(self.INCOMPLETE)]
        _week_with_break(store, document=_broken(EXCEPTED, discrepancies=rows))
        clause = _clause(store, self._null())
        assert not clause.met
        assert "a different break is a new finding" in clause.detail


class TestControlArmDefectExcusal:
    """The 2026-10-29 form: a ruled harness defect in the control arm itself."""

    def _defect(self, **overrides: Any) -> dict[dt.date, ReconciliationException]:
        fields: dict[str, Any] = {"excused_missed": ("cash_delta",), "fixed_by": "fix#1"}
        fields.update(overrides)
        return _exceptions(**fields)

    def test_the_ruled_void_run_counts_and_names_the_fix(self, store) -> None:
        _week_with_break(store, document=_broken(EXCEPTED, passed=False))
        clause = _clause(store, self._defect())
        assert clause.met, clause.detail
        assert "fixed by fix#1" in clause.detail

    def test_a_break_only_exception_still_never_excuses_a_void_run(self, store) -> None:
        _week_with_break(store, document=_broken(EXCEPTED, passed=False))
        clause = _clause(store, _exceptions())
        assert not clause.met
        assert "the cycle is VOID" in clause.detail

    def test_a_different_missed_class_is_not_excused(self, store) -> None:
        _week_with_break(store, document=_broken(EXCEPTED, passed=False))
        clause = _clause(store, self._defect(excused_missed=("share_delta",)))
        assert not clause.met
        assert "the cycle is VOID" in clause.detail

    def test_a_failed_control_that_is_not_marked_void_is_not_excused(self, store) -> None:
        _week_with_break(store, document=_broken(EXCEPTED, passed=False, void=False))
        clause = _clause(store, self._defect())
        assert not clause.met

    def test_the_discrepancies_must_still_match(self, store) -> None:
        moved = dict(BREAK, delta=0.56)
        document = _broken(EXCEPTED, discrepancies=[moved], passed=False)
        _week_with_break(store, document=document)
        clause = _clause(store, self._defect())
        assert not clause.met
        assert "a different break is a new finding" in clause.detail

    def test_a_defect_ruling_on_a_run_whose_control_passed_disagrees(self, store) -> None:
        _week_with_break(store)
        clause = _clause(store, self._defect())
        assert not clause.met
        assert "the record and the ruling disagree" in clause.detail

    @pytest.mark.parametrize(
        ("defect", "message"),
        [
            ({"missed": [], "fixed_by": "x#1"}, "must list the missed plant classes"),
            ({"missed": ["cash_delta"], "fixed_by": ""}, "must name the harness fix"),
            ({"missed": ["cash_delta"]}, "must name the harness fix"),
            ("cash_delta", "is not an object"),
        ],
    )
    def test_a_malformed_defect_raises(self, tmp_path, defect, message) -> None:
        path = _write(
            tmp_path, {"schema_version": 1, "exceptions": [_entry(control_arm_defect=defect)]}
        )
        with pytest.raises(ValueError, match=message):
            load_reconciliation_exceptions(path)


def _write(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "session": "2026-10-26",
        "run_id": "01M5WWTX24SJ1TJ8507ABJT0AA",
        "discrepancies": [dict(BREAK)],
        "ruling": RULING,
        "ruled_by": "Brian 2026-10-27T15:11Z",
        "reason": "test",
    }
    entry.update(overrides)
    return entry


class TestTheDeclaredRecord:
    def test_the_shipped_record_carries_exactly_the_four_rulings(self) -> None:
        declared = load_reconciliation_exceptions()
        assert {d.isoformat(): e.run_id for d, e in declared.items()} == {
            "2026-10-26": "01M5WWTX24SJ1TJ8507ABJT0AA",
            "2026-10-27": "01M5ZF8Q6DYRQS33T98XQ60026",
            "2026-10-29": "01M64M1VK9ZRRKJPPHY97XWSRK",
            "2026-10-30": "01M676EVWXC2N6E3QS45CVR7YR",
        }
        assert declared[dt.date(2026, 10, 30)].discrepancies == (
            ("corporate_action_check_incomplete", "SPY", None, "trader"),
        )
        defect = declared[dt.date(2026, 10, 29)]
        assert defect.excused_missed == ("cash_delta",)
        assert defect.fixed_by == "crucible-trader#67"
        assert defect.discrepancies == (("corporate_action", "MU", -11.86, "broker"),)
        assert not declared[dt.date(2026, 10, 26)].excused_missed
        assert declared[dt.date(2026, 10, 26)].discrepancies == (
            ("cash_delta", "USD", 0.55, "broker_over"),
        )
        assert declared[dt.date(2026, 10, 27)].discrepancies == (
            ("cash_delta", "USD", -0.01, "broker_under"),
        )
        assert all(e.ruled_by.startswith("Brian ") for e in declared.values())

    def test_the_shipped_record_ships_beside_the_gate(self) -> None:
        assert RECONCILIATION_EXCEPTIONS_PATH.parent.name == "crucible"
        assert RECONCILIATION_EXCEPTIONS_PATH.is_file()

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            (_entry(discrepancies=[]), "names no break"),
            (_entry(session="2026-10-25"), "not an NYSE session"),
            (_entry(ruling=""), "`ruling` must be a non-empty string"),
            (_entry(discrepancies=[dict(BREAK, delta=True)]), "names no break"),
            (
                _entry(discrepancies=[{k: v for k, v in BREAK.items() if k != "delta"}]),
                "names no break",
            ),
            (_entry(discrepancies=[dict(BREAK, delta="0.55")]), "names no break"),
        ],
    )
    def test_a_malformed_entry_raises(self, tmp_path, entry, message) -> None:
        path = _write(tmp_path, {"schema_version": 1, "exceptions": [entry]})
        with pytest.raises(ValueError, match=message):
            load_reconciliation_exceptions(path)

    def test_two_entries_for_one_session_raise(self, tmp_path) -> None:
        path = _write(tmp_path, {"schema_version": 1, "exceptions": [_entry(), _entry()]})
        with pytest.raises(ValueError, match="a second exception"):
            load_reconciliation_exceptions(path)

    def test_an_unknown_schema_version_raises(self, tmp_path) -> None:
        path = _write(tmp_path, {"schema_version": 2, "exceptions": []})
        with pytest.raises(ValueError, match="schema_version"):
            load_reconciliation_exceptions(path)
