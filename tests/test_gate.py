"""A phase gate is a MEASUREMENT over artifacts, and cannot be met by a merge.

Normative source: plan §6, §11.1, and `alpha-engine-config-I9757`'s
`closes-when`. The defect these tests exist for: the phase-1 issue was closed
on 2026-09-01 the moment its build PRs merged, with zero replay runs performed
and nine acceptance clauses failing.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.gate import GATES, evaluate, gate_key
from crucible.keys import arena_cycle_key, arm_register_key
from crucible.manifest import manifest_key
from crucible.report import attribution_key
from crucible.slots import SLOTS
from crucible.store import LocalStore
from crucible.weekly import arc_stages

FRIDAY = dt.date(2026, 8, 28)
WINDOW = [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(5))]
SHA = "a" * 40


def _put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


def _register_lines(slot: str, arms: list[tuple[str, str]]) -> bytes:
    """A real `ArmRegister` event log, in the shape `write_register` emits."""
    lines = []
    for name, spec_hash in arms:
        arm_id = f"{slot}:{name}:{spec_hash}"
        lines.append(
            json.dumps(
                {
                    "kind": "registered",
                    "arm_id": arm_id,
                    "date": "2026-01-02",
                    "reason": "",
                    "record": {
                        "arm_id": arm_id,
                        "slot": slot,
                        "name": name,
                        "spec_hash": spec_hash,
                        "created_date": "2026-01-02",
                    },
                },
                sort_keys=True,
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _manifest(status: str = "ok", **over) -> dict:
    document = {"status": status, "reason": "", "inputs": [], "outputs": [], "release_sha": SHA}
    document.update(over)
    return document


def _seed_met(tmp_path) -> LocalStore:
    """A store in which every phase-1 clause is satisfied."""
    store = LocalStore(tmp_path)
    for day in WINDOW:
        for stage in arc_stages(day):
            _put(store, manifest_key(stage.job, day.isoformat()), _manifest())
        for slot, spec in SLOTS.items():
            # The REGISTERED id form, `{slot}:{name}:{hash}` — the form the
            # §10.1 control filter has to match, and the one two reviews found
            # `promotable_arms` comparing bare names against.
            arms = [f"{slot}:alpha:abc123"] + [
                f"{slot}:{c.arm_id}:c0ffee" for c in spec.control_arms
            ]
            _put(
                store,
                arena_cycle_key(slot, day.isoformat()),
                {"scored_arms": arms, "active_arms": arms},
            )
        _put(
            store,
            manifest_key("explain", day.isoformat()),
            _manifest(inputs=[{"key": f"experiments/{slot}~alpha~abc123/{day}/verdict.json"}]),
        )
    for slot in SLOTS:
        store.put_bytes(arm_register_key(slot), _register_lines(slot, [("alpha", "abc123")]))
    _put(
        store,
        attribution_key(FRIDAY.isoformat()),
        {
            "rows": [
                {"name": f"row{n}", "value": 0.1, "status": "OK", "status_reason": "measured"}
                for n in range(5)
            ]
        },
    )
    _put(store, "releases/current", {"sha": SHA})
    _put(store, manifest_key("smoke", FRIDAY.isoformat()), _manifest())
    return store


class TestVacuity:
    def test_an_empty_store_is_not_met(self, tmp_path) -> None:
        """The one answer a gate must never give. Every clause is unmet with
        the missing artifact named, so the operator's next action is in the
        output."""
        result = evaluate(LocalStore(tmp_path), gate="phase1", trading_day=FRIDAY)
        assert not result.met
        assert result.met_ratio == 0.0
        assert all(not c.met for c in result.clauses)
        assert all(c.detail for c in result.clauses)

    def test_a_gate_with_no_clauses_is_zero_not_one(self) -> None:
        """`all([])` is True, and a gate whose clause list emptied would read
        as a pass over nothing — the vacuous truth that let a phase close
        unmeasured."""
        from crucible.gate import GateResult

        empty = GateResult(gate="phase1", trading_day=FRIDAY, window=WINDOW)
        assert empty.met_ratio == 0.0
        assert not empty.met

    def test_an_unregistered_gate_raises_rather_than_passing(self) -> None:
        with pytest.raises(KeyError, match="phase1"):
            evaluate(LocalStore("/tmp"), gate="phase9", trading_day=FRIDAY)

    def test_a_window_of_zero_weeks_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="measures nothing"):
            evaluate(LocalStore(tmp_path), gate="phase1", trading_day=FRIDAY, weeks=0)


class TestPhaseOne:
    def test_the_seeded_store_meets_every_clause(self, tmp_path) -> None:
        result = evaluate(_seed_met(tmp_path), gate="phase1", trading_day=FRIDAY)
        assert result.met, result.render()
        assert result.met_ratio == 1.0
        assert len(result.window) == GATES["phase1"][0]

    def test_one_failed_stage_in_one_week_fails_the_gate(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        _put(
            store,
            manifest_key("report", WINDOW[2].isoformat()),
            _manifest("failed", reason="MissingSourceError: yfinance returned no rows"),
        )
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        assert not result.met
        clause = next(c for c in result.clauses if c.name == "arc_runs_ok")
        assert not clause.met
        assert "yfinance" in clause.detail

    def test_one_missing_stage_manifest_fails_the_gate(self, tmp_path) -> None:
        """Absence and failure are different facts and both are unmet. An
        absent manifest read as `no news` is how five weeks of nothing pass."""
        store = _seed_met(tmp_path)
        (tmp_path / manifest_key("drift", WINDOW[0].isoformat())).unlink()
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arc_runs_ok")
        assert not clause.met
        assert "never ran" in clause.detail

    def test_a_registered_arm_that_was_not_scored_fails_the_gate(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        store.put_bytes(
            arm_register_key("r"), _register_lines("r", [("alpha", "abc123"), ("beta", "def456")])
        )
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arms_all_scored")
        assert not clause.met
        assert "r:beta:def456" in clause.detail

    def test_a_cycle_with_no_control_arm_fails_the_gate(self, tmp_path) -> None:
        """§10.1: an unscored control is an unverified grader, and a cycle it
        did not run is a cycle whose verdicts are unbacked."""
        store = _seed_met(tmp_path)
        _put(
            store,
            arena_cycle_key("m", FRIDAY.isoformat()),
            {"scored_arms": ["m:alpha:abc123"], "active_arms": ["m:alpha:abc123"]},
        )
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arms_all_scored")
        assert not clause.met
        assert "control" in clause.detail

    def test_a_report_row_with_neither_value_nor_explanation_fails(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        rows = [
            {"name": f"row{n}", "value": 0.1, "status": "OK", "status_reason": "measured"}
            for n in range(4)
        ]
        rows.append({"name": "row4", "value": None, "status": "OK", "status_reason": ""})
        _put(store, attribution_key(FRIDAY.isoformat()), {"rows": rows})
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        assert not next(c for c in result.clauses if c.name == "attribution_renders").met

    def test_an_explain_run_that_read_no_verdict_does_not_satisfy_the_walk(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        for day in WINDOW:
            _put(store, manifest_key("explain", day.isoformat()), _manifest(inputs=[]))
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        assert not next(c for c in result.clauses if c.name == "explain_walks_a_verdict").met

    def test_a_pointer_with_no_matching_smoke_fails(self, tmp_path) -> None:
        """The pointer must have flipped ON a smoke. A pointer moved by hand
        and a pointer moved on evidence are the same bytes."""
        store = _seed_met(tmp_path)
        _put(store, "releases/current", {"sha": "b" * 40})
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        assert not next(c for c in result.clauses if c.name == "pointer_flipped_on_smoke").met

    def test_a_failed_smoke_does_not_satisfy_the_pointer_clause(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        _put(store, manifest_key("smoke", FRIDAY.isoformat()), _manifest("failed", reason="boom"))
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        assert not next(c for c in result.clauses if c.name == "pointer_flipped_on_smoke").met


class TestArtifact:
    def test_the_reading_serializes_with_its_window_and_every_clause(self, tmp_path) -> None:
        document = evaluate(_seed_met(tmp_path), gate="phase1", trading_day=FRIDAY).to_dict()
        assert document["window"] == [d.isoformat() for d in WINDOW]
        assert document["met"] is True
        assert {c["name"] for c in document["clauses"]} == {
            "arc_runs_ok",
            "arms_all_scored",
            "attribution_renders",
            "explain_walks_a_verdict",
            "pointer_flipped_on_smoke",
        }
        assert all(c["requirement"] and c["detail"] for c in document["clauses"])

    def test_the_key_is_the_trading_day_like_everything_else(self) -> None:
        assert (
            gate_key("phase1", FRIDAY.isoformat()) == f"gates/phase1/{FRIDAY.isoformat()}/gate.json"
        )

    def test_the_render_names_the_unmet_clauses(self, tmp_path) -> None:
        text = evaluate(LocalStore(tmp_path), gate="phase1", trading_day=FRIDAY).render()
        assert "NOT MET" in text
        assert "arc_runs_ok" in text
