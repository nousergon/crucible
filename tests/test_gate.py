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

from crucible.gate import (
    ACCEPTANCE_RATCHET_PATH,
    GATE_DELIVERABLES,
    GATES,
    PHASE0_DELIVERABLES,
    PHASES,
    SOURCE_SCAN_SCOPE,
    Deliverable,
    build_ladder,
    coverage_note,
    evaluate,
    gate_key,
    legacy_weekly_executions_key,
    weekly_anchor,
)
from crucible.keys import (
    arena_cycle_key,
    arm_register_key,
    gate_prefix,
    review_key,
    review_prefix,
    runs_prefix,
)
from crucible.manifest import manifest_key
from crucible.report import attribution_key
from crucible.review import review_document
from crucible.slots import SLOTS
from crucible.store import LocalStore
from crucible.weekly import arc_stages

FRIDAY = dt.date(2026, 8, 28)
WINDOW = [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(5))]
SHA = "a" * 40
REVIEWER = "session_01ReviewerBBBBB"


def _review() -> dict:
    """One independent adversarial review, built by the REAL producer.

    `crucible.review.review_document`, not a hand-written dict: a fixture that
    restates the shape is a second contract, and this repository has already
    found one of those drifting inside the change that introduced it.
    """
    return review_document(
        phase="phase1",
        verdict="pass",
        reviewer=REVIEWER,
        # The author set is read out of the commits, never supplied by the
        # session asking to be passed.
        commits=[
            {
                "commit": {
                    "message": (
                        "fix: a thing\n\nClaude-Session: "
                        "https://claude.ai/code/session_01AuthorAAAAAAAA\n"
                    ),
                    "author": {"email": "someone@example.invalid"},
                    "committer": {"email": "someone@example.invalid"},
                },
                "author": {"login": "cipher813"},
                "committer": {"login": "cipher813"},
            }
        ],
        head_sha=SHA,
        pr_number=46,
        summary="no findings against plan section 2",
        reviewed_at=dt.datetime(2026, 8, 28, 18, 0, tzinfo=dt.UTC),
    )


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
            _put(
                store,
                manifest_key(stage.job, day.isoformat(), discriminator=stage.slot),
                _manifest(),
            )
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
    # The independent adversarial review (plan §11 risk 1,
    # alpha-engine-config-I9794). Every refusal this clause makes is graded in
    # `tests/test_gate_independent_review.py`; here it only has to be present
    # and independent, so `_seed_met` still means "every phase-1 clause is
    # satisfied" rather than "every clause except the newest one".
    _review_document = _review()
    _put(
        store,
        review_key("phase1", FRIDAY.isoformat(), REVIEWER, "pass"),
        _review_document,
    )
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

    def test_a_gate_with_no_clauses_is_unmeasured_not_zero(self) -> None:
        """`all([])` is True, and a gate whose clause list emptied would read
        as a pass over nothing — the vacuous truth that let a phase close
        unmeasured. `met_ratio` must be `None`, not `0.0`: `0.0` says "we
        measured, and nothing passed", which is a different, false, claim
        from "we never measured" (alpha-engine-config-I9824)."""
        from crucible.gate import GateResult

        empty = GateResult(gate="phase1", trading_day=FRIDAY, window=WINDOW)
        assert empty.met_ratio is None
        assert empty.to_dict()["met_ratio"] is None
        assert not empty.met
        assert empty.to_dict()["met"] is False

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


#: `alpha-engine-config-I9869` malformation shapes, shared with the phase-0
#: coverage above: an unreadable body, a JSON array, literal `null`, and a
#: bare string — every shape that is not an object with fields.
MALFORMED_BODIES = [
    (b"{not json", "not readable JSON"),
    (b"[1, 2]", "parsed to list"),
    (b"null", "literal `null`"),
    (b'"a string"', "parsed to str"),
]
MALFORMED_IDS = ["truncated json", "a list", "literal null", "a bare string"]


class TestPhaseOneGuardedReads:
    """`alpha-engine-config-I9869`: phase-1 clauses now read first-party
    `run_manifest.v1` / arena artifacts through the SAME guarded reader as the
    phase-0 clauses over external documents. Every malformed shape must
    become an UNMET clause naming the key and the fault, never an exception
    out of `evaluate`, `build_ladder`, or the board render — and absence must
    stay a distinct reading from malformed."""

    @pytest.mark.parametrize(("body", "expected"), MALFORMED_BODIES, ids=MALFORMED_IDS)
    def test_arc_runs_ok_malformed_manifest_is_a_red_reading(
        self, tmp_path, body: bytes, expected: str
    ) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("report", WINDOW[2].isoformat())
        store.put_bytes(key, body)
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arc_runs_ok")
        assert not clause.met
        assert key in clause.detail
        assert expected in clause.detail

    @pytest.mark.parametrize(("body", "expected"), MALFORMED_BODIES, ids=MALFORMED_IDS)
    def test_arms_all_scored_malformed_cycle_is_a_red_reading(
        self, tmp_path, body: bytes, expected: str
    ) -> None:
        store = _seed_met(tmp_path)
        key = arena_cycle_key("m", FRIDAY.isoformat())
        store.put_bytes(key, body)
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arms_all_scored")
        assert not clause.met
        assert key in clause.detail
        assert expected in clause.detail

    @pytest.mark.parametrize(("body", "expected"), MALFORMED_BODIES, ids=MALFORMED_IDS)
    def test_attribution_renders_malformed_document_is_a_red_reading(
        self, tmp_path, body: bytes, expected: str
    ) -> None:
        store = _seed_met(tmp_path)
        key = attribution_key(FRIDAY.isoformat())
        store.put_bytes(key, body)
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "attribution_renders")
        assert not clause.met
        assert key in clause.detail
        assert expected in clause.detail

    @pytest.mark.parametrize(("body", "expected"), MALFORMED_BODIES, ids=MALFORMED_IDS)
    def test_explain_walks_a_verdict_malformed_manifest_is_a_red_reading(
        self, tmp_path, body: bytes, expected: str
    ) -> None:
        """Every day in the window is corrupted, not only the day named in
        the assertion — a single still-readable day elsewhere in the window
        would satisfy the clause on its own and hide the malformed one."""
        store = _seed_met(tmp_path)
        key = manifest_key("explain", FRIDAY.isoformat())
        for day in WINDOW:
            store.put_bytes(manifest_key("explain", day.isoformat()), body)
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "explain_walks_a_verdict")
        assert not clause.met
        assert key in clause.detail
        assert expected in clause.detail

    @pytest.mark.parametrize(("body", "expected"), MALFORMED_BODIES, ids=MALFORMED_IDS)
    def test_pointer_malformed_is_a_red_reading(self, tmp_path, body: bytes, expected: str) -> None:
        store = _seed_met(tmp_path)
        store.put_bytes("releases/current", body)
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "pointer_flipped_on_smoke")
        assert not clause.met
        assert "releases/current" in clause.detail
        assert expected in clause.detail

    @pytest.mark.parametrize(("body", "expected"), MALFORMED_BODIES, ids=MALFORMED_IDS)
    def test_smoke_manifest_malformed_is_a_red_reading(
        self, tmp_path, body: bytes, expected: str
    ) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("smoke", FRIDAY.isoformat())
        store.put_bytes(key, body)
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "pointer_flipped_on_smoke")
        assert not clause.met
        assert key in clause.detail
        assert expected in clause.detail

    def test_arc_runs_ok_missing_status_field_is_malformed_not_absent(self, tmp_path) -> None:
        """The key EXISTS. Reporting it as "never ran" would name the wrong
        remedy — the stage did run and wrote something, just not a manifest
        this reader can trust."""
        store = _seed_met(tmp_path)
        key = manifest_key("report", WINDOW[2].isoformat())
        _put(store, key, {"reason": ""})
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arc_runs_ok")
        assert not clause.met
        assert key in clause.detail
        assert "status" in clause.detail
        assert "never ran" not in clause.detail

    def test_arc_runs_ok_missing_reason_on_failure_is_malformed(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("report", WINDOW[2].isoformat())
        _put(store, key, {"status": "failed"})
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arc_runs_ok")
        assert not clause.met
        assert key in clause.detail
        assert "reason" in clause.detail

    def test_arms_all_scored_missing_scored_arms_field_is_malformed(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = arena_cycle_key("m", FRIDAY.isoformat())
        _put(store, key, {"active_arms": []})
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arms_all_scored")
        assert not clause.met
        assert key in clause.detail
        assert "scored_arms" in clause.detail

    def test_explain_missing_status_field_is_malformed(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("explain", FRIDAY.isoformat())
        for day in WINDOW:
            _put(store, manifest_key("explain", day.isoformat()), {"inputs": []})
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "explain_walks_a_verdict")
        assert not clause.met
        assert key in clause.detail
        assert "status" in clause.detail

    def test_smoke_missing_status_field_is_malformed(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("smoke", FRIDAY.isoformat())
        _put(store, key, {"release_sha": SHA})
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "pointer_flipped_on_smoke")
        assert not clause.met
        assert key in clause.detail
        assert "status" in clause.detail

    def test_a_malformed_phase1_document_does_not_take_the_ladder_down_with_it(
        self, tmp_path
    ) -> None:
        """The doctrine both modules now state identically: an unreadable
        input is a red READING on the surface, never absence from it, and
        never an exception that takes the whole ladder down."""
        store = _seed_met(tmp_path)
        store.put_bytes(manifest_key("report", WINDOW[2].isoformat()), b"{not json")
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase1"]["state"] == "UNMET"
        assert rows["phase1"]["clauses_total"] == 6

    def test__read_json_is_retired(self) -> None:
        """One reader, not two with different failure semantics."""
        import crucible.gate as gate_module

        assert not hasattr(gate_module, "_read_json")


class _AccessDenied(LocalStore):
    """A store whose `get_bytes` raises for one key — a permission denial /
    AccessDenied, not a content problem. `exists` still answers normally, the
    same asymmetry a real S3 401/403 exhibits."""

    def __init__(self, root, denied_key: str) -> None:
        super().__init__(root)
        self._denied_key = denied_key

    def get_bytes(self, key: str) -> bytes:
        if key == self._denied_key:
            raise PermissionError(f"access denied: {key}")
        return super().get_bytes(key)


class _ListDenied(LocalStore):
    """A store whose `list_keys` raises for one prefix — round 3's finding
    2: `store.list_keys` can fail an access check exactly the way
    `get_bytes` can, and three call sites read it unguarded."""

    def __init__(self, root, denied_prefix: str) -> None:
        super().__init__(root)
        self._denied_prefix = denied_prefix

    def list_keys(self, prefix: str = ""):
        if prefix == self._denied_prefix:
            raise PermissionError(f"access denied listing: {prefix}")
        return super().list_keys(prefix)


class TestPhaseOneGuardedReadsRound2:
    """`alpha-engine-config-I9869` round 2, findings from independent
    adversarial review of round 1's PR: the register was still unguarded, the
    guard checked container type but not field type, the status vocabulary
    was compared with `!= "ok"` instead of against the schema, and a store
    access failure read identically to a broken build."""

    # -- finding 1: BLOCKING — `_register_arms` was a bare `json.loads` per
    # line, unguarded. ------------------------------------------------------

    def test_a_malformed_register_line_is_a_red_reading_not_an_exception(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = arm_register_key("m")
        store.put_bytes(key, b"{not json\n")
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arms_all_scored")
        assert not clause.met
        assert key in clause.detail
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase1"]["clauses_total"] == 6

    def test_a_malformed_register_line_names_the_line_number(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = arm_register_key("m")
        store.put_bytes(key, b'{"kind": "registered"}\n[1, 2]\n')
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arms_all_scored"
        )
        assert not clause.met
        assert f"{key}:2" in clause.detail

    def test_the_register_can_still_be_read_via_the_cli_dry_run_path(self, tmp_path) -> None:
        """The reproduction named in the review: `crucible gate ... --dry-run`
        over a store seeded with a malformed register must not raise."""
        store = _seed_met(tmp_path)
        store.put_bytes(arm_register_key("m"), b"{not json\n")
        # `evaluate` is exactly what the CLI's gate command calls; asserting
        # it returns (rather than raises) is the same guarantee the CLI
        # reproduction depends on, without shelling out to a second process.
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        assert not result.met

    # -- finding 2: BLOCKING — the guard checked container type only; a
    # wrong-typed FIELD still crashed a downstream index. -------------------

    def test_attribution_rows_field_wrong_type_is_a_red_reading(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = attribution_key(FRIDAY.isoformat())
        _put(store, key, {"rows": 5})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "attribution_renders"
        )
        assert not clause.met
        assert "rows" in clause.detail

    def test_attribution_row_element_wrong_type_is_a_red_reading(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = attribution_key(FRIDAY.isoformat())
        _put(store, key, {"rows": ["a", "b", "c", "d", "e"]})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "attribution_renders"
        )
        assert not clause.met

    def test_arena_cycle_scored_arms_element_wrong_type_is_a_red_reading(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = arena_cycle_key("m", FRIDAY.isoformat())
        _put(store, key, {"scored_arms": [{"a": 1}], "active_arms": []})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arms_all_scored"
        )
        assert not clause.met
        assert "scored_arms" in clause.detail

    def test_pointer_sha_wrong_type_is_a_red_reading(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        _put(store, "releases/current", {"sha": 12345})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "pointer_flipped_on_smoke"
        )
        assert not clause.met
        assert "sha" in clause.detail

    def test_pointer_sha_absent_is_a_red_reading_not_an_empty_string(self, tmp_path) -> None:
        """Round 4 finding 3: presence was unchecked, only type — an absent
        `sha` silently became `""` and was compared against every smoke
        manifest's `release_sha` instead of being reported as malformed."""
        store = _seed_met(tmp_path)
        _put(store, "releases/current", {})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "pointer_flipped_on_smoke"
        )
        assert not clause.met
        assert "releases/current" in clause.detail
        assert "missing required field" in clause.detail
        assert "sha" in clause.detail

    def test_explain_inputs_field_wrong_type_is_a_red_reading_not_a_typeerror(
        self, tmp_path
    ) -> None:
        """Round 4 finding 1 (BLOCKING): `inputs` was the one required field
        the PR body named that never got routed through `_field` — a present
        but non-iterable `inputs` raised straight out of `evaluate`, and the
        whole gate command died with no ladder and no artifact published."""
        store = _seed_met(tmp_path)
        key = manifest_key("explain", FRIDAY.isoformat())
        for day in WINDOW:
            _put(store, manifest_key("explain", day.isoformat()), _manifest(inputs=5))
        result = evaluate(store, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "explain_walks_a_verdict")
        assert not clause.met
        assert key in clause.detail
        assert "inputs" in clause.detail

    # -- finding 3: SHOULD-FIX — status vocabulary compared with `!= "ok"`
    # instead of against the schema's exhaustive enum. ----------------------

    def test_a_status_outside_the_schema_vocabulary_is_malformed_not_failed(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("report", WINDOW[2].isoformat())
        _put(store, key, {"status": "degraded", "reason": ""})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arc_runs_ok"
        )
        assert not clause.met
        assert "degraded" in clause.detail
        assert "failed" not in clause.detail.split(":")[0]

    def test_a_non_string_status_is_malformed(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("report", WINDOW[2].isoformat())
        _put(store, key, {"status": 3, "reason": ""})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arc_runs_ok"
        )
        assert not clause.met
        assert key in clause.detail

    def test_status_ok_with_a_non_empty_reason_is_malformed(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("report", WINDOW[2].isoformat())
        _put(store, key, {"status": "ok", "reason": "x"})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arc_runs_ok"
        )
        assert not clause.met, "status `ok` with a non-empty reason read MET"

    def test_a_null_status_is_malformed_not_failed_with_an_empty_cause(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("report", WINDOW[2].isoformat())
        _put(store, key, {"status": None, "reason": ""})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arc_runs_ok"
        )
        assert not clause.met
        assert key in clause.detail

    def test_the_status_vocabulary_check_applies_to_explain_too(self, tmp_path) -> None:
        """`!= "ok"` alone would already read this unmet (for the wrong
        reason — a vacuous `continue`, same as an absent manifest); the
        assertion that actually distinguishes the vocabulary check is that
        `degraded` is NAMED, not silently folded into "no ok manifest"."""
        store = _seed_met(tmp_path)
        key = manifest_key("explain", FRIDAY.isoformat())
        for day in WINDOW:
            _put(
                store,
                manifest_key("explain", day.isoformat()),
                {"status": "degraded", "reason": ""},
            )
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "explain_walks_a_verdict"
        )
        assert not clause.met
        assert key in clause.detail
        assert "degraded" in clause.detail

    def test_the_status_vocabulary_check_applies_to_smoke_too(self, tmp_path) -> None:
        store = _seed_met(tmp_path)
        key = manifest_key("smoke", FRIDAY.isoformat())
        _put(store, key, {"status": "degraded", "reason": ""})
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "pointer_flipped_on_smoke"
        )
        assert not clause.met
        assert key in clause.detail
        assert "degraded" in clause.detail

    # -- finding 4: SHOULD-FIX — a store read that raises (access, not
    # content) must read UNMEASURABLE, never folded into UNMET-malformed. ---

    def test_an_access_failure_reads_unmeasurable_not_malformed(self, tmp_path) -> None:
        key = manifest_key("report", WINDOW[2].isoformat())
        _seed_met(tmp_path)
        denied = _AccessDenied(tmp_path, denied_key=key)
        result = evaluate(denied, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arc_runs_ok")
        assert not clause.met
        assert clause.unmeasurable
        assert "access" in clause.detail
        # An unmeasurable clause is never met, and it is never silently
        # dropped from the reading either.
        assert not result.met

    def test_an_access_failure_does_not_take_the_ladder_down_with_it(self, tmp_path) -> None:
        key = manifest_key("report", WINDOW[2].isoformat())
        _seed_met(tmp_path)
        denied = _AccessDenied(tmp_path, denied_key=key)
        result = evaluate(denied, gate="phase1", trading_day=FRIDAY)
        clause = next(c for c in result.clauses if c.name == "arc_runs_ok")
        assert clause.unmeasurable
        rows = {r["phase"]: r for r in build_ladder(denied, trading_day=FRIDAY).to_dict()["phases"]}
        # Round 3 (`alpha-engine-config-I9869`, finding 4): an unmeasurable
        # clause now renders the ROW as UNMEASURABLE, not UNMET — "we could
        # not measure" and "we measured and it fell short" are different
        # facts.
        assert rows["phase1"]["state"] == "UNMEASURABLE"
        assert rows["phase1"]["clauses_total"] == 6
        assert rows["phase1"]["clauses_unmeasurable"] == 1
        assert rows["phase1"]["met_ratio"] is None

    def test_an_access_failure_on_the_register_reads_unmeasurable(self, tmp_path) -> None:
        key = arm_register_key("m")
        _seed_met(tmp_path)
        denied = _AccessDenied(tmp_path, denied_key=key)
        clause = next(
            c
            for c in evaluate(denied, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arms_all_scored"
        )
        assert not clause.met
        assert clause.unmeasurable

    def test_clause_to_dict_carries_the_unmeasurable_flag(self, tmp_path) -> None:
        key = manifest_key("report", WINDOW[2].isoformat())
        _seed_met(tmp_path)
        denied = _AccessDenied(tmp_path, denied_key=key)
        clause = next(
            c
            for c in evaluate(denied, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arc_runs_ok"
        )
        assert clause.to_dict()["unmeasurable"] is True

    def test_a_met_clause_defaults_unmeasurable_false(self, tmp_path) -> None:
        result = evaluate(_seed_met(tmp_path), gate="phase1", trading_day=FRIDAY)
        assert all(not c.unmeasurable for c in result.clauses)


class TestPhaseOneGuardedReadsRound3:
    """`alpha-engine-config-I9869` round 3, findings from independent
    adversarial re-verification of round 2's PR: round 2's claim 4
    (unmeasurable) was defeated at both boundaries — a still-unguarded
    LISTING (finding 2) and an unmeasurable flag that lost precedence to a
    real content gap (finding 6) — plus a still-unguarded register re-read
    (finding 5, a correctness bug independent of round 2) and the same
    listing gap inside the independent-review clause (finding 3)."""

    # -- finding 2: BLOCKING — `store.list_keys` unguarded at three sites. --

    def test_pointer_clause_smoke_listing_access_failure_is_unmeasurable(self, tmp_path) -> None:
        _seed_met(tmp_path)
        denied = _ListDenied(tmp_path, denied_prefix=runs_prefix("smoke"))
        clause = next(
            c
            for c in evaluate(denied, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "pointer_flipped_on_smoke"
        )
        assert not clause.met
        assert clause.unmeasurable
        # `evaluate` returning at all (rather than raising) is the assertion
        # that matters most here — round 2's own bug class, applied to a listing.

    def test_independently_reviewed_listing_access_failure_is_unmeasurable(self, tmp_path) -> None:
        _seed_met(tmp_path)
        denied = _ListDenied(tmp_path, denied_prefix=review_prefix("phase1"))
        clause = next(
            c
            for c in evaluate(denied, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "independently_reviewed"
        )
        assert not clause.met
        assert clause.unmeasurable

    def test_last_read_reports_an_access_failure_distinct_from_never_read(self, tmp_path) -> None:
        from crucible.gate import last_read

        _seed_met(tmp_path)
        denied = _ListDenied(tmp_path, denied_prefix=gate_prefix("phase1"))
        assert last_read(denied, "phase1") == (None, True)
        # A gate genuinely never read is a DIFFERENT answer — same shape,
        # `access_problem` false — not confusable with the listing above.
        assert last_read(LocalStore(tmp_path / "empty"), "phase1") == (None, False)

    def test_a_listing_access_failure_does_not_take_the_ladder_down(self, tmp_path) -> None:
        _seed_met(tmp_path)
        denied = _ListDenied(tmp_path, denied_prefix=runs_prefix("smoke"))
        rows = {r["phase"]: r for r in build_ladder(denied, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase1"]["state"] == "UNMEASURABLE"

    # -- finding 3: BLOCKING — `_clause_independently_reviewed` ignored
    # `read.access_problem`, rendering `[ ]` with an access sentence buried
    # in `detail` rather than `unmeasurable=True`. ---------------------------

    def test_a_review_read_access_failure_is_unmeasurable_not_plain_unmet(self, tmp_path) -> None:
        _seed_met(tmp_path)
        key = review_key("phase1", FRIDAY.isoformat(), REVIEWER, "pass")
        denied = _AccessDenied(tmp_path, denied_key=key)
        clause = next(
            c
            for c in evaluate(denied, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "independently_reviewed"
        )
        assert not clause.met
        assert clause.unmeasurable

    # A combined content-problem-alongside-access-failure case for THIS
    # clause is not a separate test: pre-fix, `_clause_independently_reviewed`
    # never set `unmeasurable=True` at all (finding 3), so an assertion of
    # `not clause.unmeasurable` in a combined case would be true before this
    # fix for the wrong reason and prove nothing. Finding 6's precedence rule
    # is exercised where it is genuinely at risk of flipping the wrong way:
    # `arc_runs_ok` and `arms_all_scored`, below.

    # -- finding 5: MEDIUM — `_clause_arms_all_scored` re-read the register
    # inside the day loop; one malformed register filled `gaps[:4]` up to
    # `len(window)` times and could hide an unrelated, genuinely missing
    # arena cycle. -----------------------------------------------------------

    def test_a_malformed_register_and_a_missing_cycle_are_both_named(self, tmp_path) -> None:
        """The missing cycle is seeded on the LAST window day deliberately:
        under the round-2 shape (the register re-read inside the day loop),
        the malformed register filed its problem once per day BEFORE this
        day's slot was ever reached, so by the last day `gaps[:4]` was
        already full of duplicates of the SAME register problem and the
        genuinely missing `s` cycle — found only on this final day — was
        truncated away entirely."""
        store = _seed_met(tmp_path)
        register_key = arm_register_key("m")
        store.put_bytes(register_key, b"{not json\n")
        missing_key = arena_cycle_key("s", WINDOW[-1].isoformat())
        (tmp_path / missing_key).unlink()
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arms_all_scored"
        )
        assert not clause.met
        assert register_key in clause.detail
        assert f"s@{WINDOW[-1].isoformat()}" in clause.detail

    def test_the_malformed_register_is_named_only_once(self, tmp_path) -> None:
        """The bug round 2 shipped: reading the register inside the day loop
        filed the SAME problem once per day in the window (5x here), which
        the round-1-style `gaps[:4]` truncation could crowd an unrelated
        finding out of entirely. Hoisting the read to once-per-slot removes
        the duplication at its source."""
        store = _seed_met(tmp_path)
        register_key = arm_register_key("m")
        store.put_bytes(register_key, b"{not json\n")
        clause = next(
            c
            for c in evaluate(store, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arms_all_scored"
        )
        assert clause.detail.count(register_key) == 1

    # -- finding 6: LOW — `unmeasurable=bool(unmeasurable)` rendered `?` even
    # when a real content gap was ALSO present, across every phase-1 clause
    # that combines the two. ------------------------------------------------

    def test_arc_runs_ok_content_gap_alongside_access_failure_is_not_unmeasurable(
        self, tmp_path
    ) -> None:
        store = _seed_met(tmp_path)
        denied_key = manifest_key("report", WINDOW[2].isoformat())
        malformed_key = manifest_key("drift", WINDOW[1].isoformat())
        _put(store, malformed_key, {"status": "degraded", "reason": ""})
        denied = _AccessDenied(tmp_path, denied_key=denied_key)
        clause = next(
            c
            for c in evaluate(denied, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arc_runs_ok"
        )
        assert not clause.met
        assert not clause.unmeasurable
        assert "access" in clause.detail or "could not be read" in clause.detail
        assert malformed_key in clause.detail

    def test_arms_all_scored_content_gap_alongside_access_failure_is_not_unmeasurable(
        self, tmp_path
    ) -> None:
        _seed_met(tmp_path)
        register_key = arm_register_key("m")
        denied = _AccessDenied(tmp_path, denied_key=register_key)
        missing_key = arena_cycle_key("s", WINDOW[0].isoformat())
        (tmp_path / missing_key).unlink()
        clause = next(
            c
            for c in evaluate(denied, gate="phase1", trading_day=FRIDAY).clauses
            if c.name == "arms_all_scored"
        )
        assert not clause.met
        assert not clause.unmeasurable


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
            "independently_reviewed",
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


class TestTheGateJobPublishesAnHonestMetric:
    """`alpha-engine-config-I9824`, deliverable #3: an unmeasured gate must
    publish `gate_clauses_met_ratio` as an explicit N/A-shaped status with a
    reason, never as a numeric `0.0` that reads as "measured, none passed"."""

    def test_an_empty_clause_list_publishes_NA_not_0_0(self, tmp_path, monkeypatch) -> None:
        import argparse

        from crucible.gate import GATES
        from crucible.store import LocalStore
        from crucible.track_f import gate_handler

        monkeypatch.setitem(GATES, "phase1", (5, lambda *_a, **_k: []))
        store_uri = str(tmp_path)
        args = argparse.Namespace(gate="phase1", trading_day=FRIDAY, weeks=None, store=store_uri)
        exit_code = gate_handler(args)
        assert exit_code == 1  # unmet, per the fail-loud-on-exit-code invariant

        store = LocalStore(tmp_path)
        manifest = json.loads(store.get_bytes(f"runs/gate/{FRIDAY.isoformat()}/run.json"))
        (metric,) = [m for m in manifest["metrics"] if m["name"] == "gate_clauses_met_ratio"]
        assert metric["value"] is None
        assert metric["status"].startswith("N/A")
        assert metric["status_reason"]
        assert metric["status"] != "0.0"

        gate_artifact = json.loads(store.get_bytes(gate_key("phase1", FRIDAY.isoformat())))
        assert gate_artifact["met_ratio"] is None

    def test_a_real_measurement_still_publishes_its_ratio_as_a_number(
        self, tmp_path, monkeypatch
    ) -> None:
        """The N/A branch must not swallow a real reading — a gate that
        measured something and met none of it still reports `0.0`, and it is
        a numeric OK/FAIL status, not N/A."""
        import argparse

        from crucible.gate import GATES, Clause
        from crucible.store import LocalStore
        from crucible.track_f import gate_handler

        monkeypatch.setitem(
            GATES,
            "phase1",
            (5, lambda *_a, **_k: [Clause("c", "req", False, "unmet", ())]),
        )
        store_uri = str(tmp_path)
        args = argparse.Namespace(gate="phase1", trading_day=FRIDAY, weeks=None, store=store_uri)
        gate_handler(args)

        store = LocalStore(tmp_path)
        manifest = json.loads(store.get_bytes(f"runs/gate/{FRIDAY.isoformat()}/run.json"))
        (metric,) = [m for m in manifest["metrics"] if m["name"] == "gate_clauses_met_ratio"]
        assert metric["value"] == 0.0
        assert not metric["status"].startswith("N/A")


PHASE0_WINDOW = [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(2))]


def _seed_phase0_met(tmp_path, starts: int = 1) -> LocalStore:
    """A store in which the v1 weekly cadence clause is satisfied.

    The acceptance clause reads the committed ratchet, not the store, so a
    seeded store plus the real repository is the whole met state.
    """
    store = LocalStore(tmp_path)
    for day in PHASE0_WINDOW:
        _put(
            store,
            legacy_weekly_executions_key(weekly_anchor(day).isoformat()),
            {"executions_started": starts, "source": "a filed count, not a live API call"},
        )
    return store


def _clause(result, name: str):
    return next(c for c in result.clauses if c.name == name)


class TestPhaseZeroIsRegisteredAtAll:
    """`alpha-engine-config-I9804`. The ladder read `phase0 UNMEASURED` while
    phase 0 was the phase in flight, which made phase 1 render OUT_OF_ORDER for
    a reason nothing in the harness could ever clear."""

    def test_phase0_has_a_clause_list_and_the_ladder_rung_names_it(self) -> None:
        assert "phase0" in GATES
        assert PHASES[0].id == "phase0"
        assert PHASES[0].gate == "phase0"

    def test_the_window_is_the_two_consecutive_weeks_the_issue_asks_for(self, tmp_path) -> None:
        """I9756's closes-when is "<= 1 start per calendar week for two
        consecutive weeks". One quiet week is a gap between reruns."""
        assert GATES["phase0"][0] == 2
        result = evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY)
        assert result.window == PHASE0_WINDOW


class TestPhaseZeroOldWeeklyCadence:
    def test_an_unfiled_count_is_unmet_with_the_missing_key_named(self, tmp_path) -> None:
        """The count is not filed anywhere today. That reads UNMET naming the
        key — never unmeasurable-as-pass, and never a live
        `states:ListExecutions` call, which no replay and no test could make."""
        result = evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY)
        clause = _clause(result, "old_weekly_within_cadence")
        assert not clause.met
        assert legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat()) in clause.detail
        assert not result.met

    def test_two_quiet_weeks_meet_the_clause_and_the_gate(self, tmp_path) -> None:
        result = evaluate(_seed_phase0_met(tmp_path), gate="phase0", trading_day=FRIDAY)
        assert _clause(result, "old_weekly_within_cadence").met
        assert result.met, result.render()

    def test_a_week_over_the_cadence_fails_the_gate(self, tmp_path) -> None:
        """19 executions since 2026-08-26 was the live reading on 2026-09-02
        (`alpha-engine-config-I9831`); a gate that called that met would be
        measuring nothing."""
        store = _seed_phase0_met(tmp_path)
        _put(
            store,
            legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat()),
            {"executions_started": 19},
        )
        result = evaluate(store, gate="phase0", trading_day=FRIDAY)
        clause = _clause(result, "old_weekly_within_cadence")
        assert not clause.met
        assert "19 starts" in clause.detail
        assert not result.met

    def test_one_quiet_week_is_not_a_cadence(self, tmp_path) -> None:
        """A single filed week may not carry the clause: two consecutive weeks
        is the requirement, and an absent second week is an absence."""
        store = LocalStore(tmp_path)
        _put(
            store,
            legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat()),
            {"executions_started": 1},
        )
        result = evaluate(store, gate="phase0", trading_day=FRIDAY)
        clause = _clause(result, "old_weekly_within_cadence")
        assert not clause.met
        assert (
            legacy_weekly_executions_key(weekly_anchor(PHASE0_WINDOW[0]).isoformat())
            in clause.detail
        )

    def test_every_week_of_the_window_is_named_as_evidence(self, tmp_path) -> None:
        result = evaluate(_seed_phase0_met(tmp_path), gate="phase0", trading_day=FRIDAY)
        clause = _clause(result, "old_weekly_within_cadence")
        assert set(clause.evidence) == {
            legacy_weekly_executions_key(weekly_anchor(d).isoformat()) for d in PHASE0_WINDOW
        }

    def test_the_evidence_key_is_the_same_whatever_day_the_gate_is_read_on(self, tmp_path) -> None:
        """The defect an adversarial review found on this branch: `_window`
        steps back in raw calendar weeks from whatever day the caller passed,
        and the ladder is rendered DAILY, so a raw-day key sent Monday's read
        and Tuesday's read to two different objects. A producer filing one
        document per week would have made phase 0 read MET on one weekday and
        UNMET on the other four, forever."""
        store = LocalStore(tmp_path)
        readings = {
            day: _clause(
                evaluate(store, gate="phase0", trading_day=dt.date.fromisoformat(day)),
                "old_weekly_within_cadence",
            ).evidence
            for day in ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04")
        }
        assert len(set(readings.values())) == 1, readings

    def test_the_week_that_has_not_closed_yet_is_not_asked_for(self) -> None:
        """Strictly BEFORE, not on-or-before. The week ending this Friday is
        not counted until its close has passed and the weekly producer has
        run, so anchoring to it would make the newest week absent for a day
        and flap once a week instead of four times a week."""
        assert weekly_anchor(dt.date(2026, 9, 4)) == dt.date(2026, 8, 28)
        assert weekly_anchor(dt.date(2026, 9, 5)) == dt.date(2026, 9, 4)

    def test_two_window_weeks_collapsing_onto_one_anchor_is_unmet(
        self, tmp_path, monkeypatch
    ) -> None:
        """Two anchors 7 days apart cannot normally collide, but a future
        change to the anchor could make them. One document graded twice is a
        cadence claim backed by half the evidence it names, so the collapse is
        a red reading rather than a silently shorter window."""
        monkeypatch.setattr("crucible.gate.weekly_anchor", lambda day: FRIDAY)
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "old_weekly_within_cadence",
        )
        assert not clause.met
        assert "collapsed" in clause.detail

    def test_a_malformed_filed_count_is_a_red_reading_not_an_exception(self, tmp_path) -> None:
        """This document is written by a producer outside this repository. A
        bare `document["executions_started"]` would raise out of `evaluate`
        and take `crucible gate`, `build_ladder` AND the board render down
        together — so one malformed upstream file would publish NOTHING
        rather than a red reading."""
        store = _seed_phase0_met(tmp_path)
        key = legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat())
        _put(store, key, {"count": 1})
        clause = _clause(
            evaluate(store, gate="phase0", trading_day=FRIDAY), "old_weekly_within_cadence"
        )
        assert not clause.met
        assert key in clause.detail
        assert "executions_started" in clause.detail

    def test_a_count_that_is_not_a_whole_number_of_starts_is_malformed(self, tmp_path) -> None:
        """`True` is an `int` in Python and a boolean is not a count. A
        document filing `executions_started: true` would otherwise read as
        one start and satisfy the cadence."""
        store = _seed_phase0_met(tmp_path)
        key = legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat())
        _put(store, key, {"executions_started": True})
        clause = _clause(
            evaluate(store, gate="phase0", trading_day=FRIDAY), "old_weekly_within_cadence"
        )
        assert not clause.met

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (b"{not json", "not readable JSON"),
            (b"[1, 2]", "parsed to list"),
            (b"null", "literal `null`"),
            (b'"1"', "parsed to str"),
        ],
        ids=["truncated json", "a list", "literal null", "a bare string"],
    )
    def test_every_malformed_body_is_a_red_reading_not_an_exception(
        self, tmp_path, body: bytes, expected: str
    ) -> None:
        """Round-2 covered exactly ONE malformed shape — a well-formed object
        with the wrong field — and the invariant was asserted for all of them.
        Every other shape raised out of `evaluate` and took `crucible gate`,
        `build_ladder` and the board render down together, publishing nothing
        where a red reading belongs."""
        store = _seed_phase0_met(tmp_path)
        key = legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat())
        store.put_bytes(key, body)
        clause = _clause(
            evaluate(store, gate="phase0", trading_day=FRIDAY), "old_weekly_within_cadence"
        )
        assert not clause.met
        assert key in clause.detail
        assert expected in clause.detail

    def test_present_but_null_is_unreadable_not_absent(self, tmp_path) -> None:
        """The key EXISTS. Reporting it as "no filed count" names the wrong
        remedy: the operator would go and write a producer that is already
        running."""
        store = _seed_phase0_met(tmp_path)
        key = legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat())
        store.put_bytes(key, b"null")
        clause = _clause(
            evaluate(store, gate="phase0", trading_day=FRIDAY), "old_weekly_within_cadence"
        )
        assert "no filed count" not in clause.detail
        assert "unreadable, not absent" in clause.detail

    def test_a_malformed_document_does_not_take_the_ladder_down_with_it(self, tmp_path) -> None:
        """The doctrine both modules state: an unreadable input is a red
        READING on the surface, never absence from it."""
        store = _seed_phase0_met(tmp_path)
        # Unparseable BYTES, not a well-formed object with the wrong field:
        # the field check was already guarded, and `json.loads` was not.
        store.put_bytes(
            legacy_weekly_executions_key(weekly_anchor(FRIDAY).isoformat()), b"{not json"
        )
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase0"]["state"] == "UNMET"
        assert rows["phase0"]["clauses_total"] == 2


def _fake_acceptance_tree(
    tmp_path, monkeypatch, *, met: list[str], unmet: dict[str, str], define: list[str] | None = None
) -> None:
    """A ratchet and the suite it claims to describe, both under `tmp_path`.

    Both globals are redirected: the clause reads the ratchet AND parses the
    suite beside it, and a test that moved only one of them would be grading
    this repository's real suite against a synthetic ratchet.
    """
    suite = tmp_path / "acceptance"
    suite.mkdir(exist_ok=True)
    defined = [*met, *unmet] if define is None else define
    body: list[str] = []
    for cid in defined:
        cls, method = cid.split("[", 1)[0].split("::")
        body.append(f"class {cls}:\n    def {method}(self):\n        raise AssertionError")
    (suite / "test_generated.py").write_text("\n\n".join(body) + "\n", encoding="utf-8")
    ratchet = suite / "ratchet.json"
    ratchet.write_text(json.dumps({"met": met, "unmet": unmet}), encoding="utf-8")
    monkeypatch.setattr("crucible.gate.ACCEPTANCE_RATCHET_PATH", ratchet)
    monkeypatch.setattr("crucible.gate.ACCEPTANCE_SUITE_DIR", suite)


class TestPhaseZeroAcceptanceSuite:
    def test_the_clause_reads_the_committed_ratchet_of_this_repository(self, tmp_path) -> None:
        """`tests/acceptance/ratchet.json` is the durable reading: it commits
        the exact id sets, `tests/test_acceptance_reading.py` fails when they
        drift from what the suite collects, and CI fails a push to main on any
        movement."""
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert clause.met, clause.detail
        assert str(ACCEPTANCE_RATCHET_PATH) in clause.evidence
        committed = json.loads(ACCEPTANCE_RATCHET_PATH.read_text(encoding="utf-8"))
        assert f"{len(committed['met']) + len(committed['unmet'])} §2 clauses" in clause.detail

    def test_an_absent_ratchet_is_unmet_with_the_path_named(self, tmp_path, monkeypatch) -> None:
        """A wheel install ships no `tests/`. That is an absence, and an
        absence is never a pass."""
        missing = tmp_path / "nowhere" / "ratchet.json"
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_RATCHET_PATH", missing)
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert str(missing) in clause.detail

    def test_a_ratchet_naming_no_clauses_is_dark_not_green(self, tmp_path, monkeypatch) -> None:
        """`all([])` again: an empty clause set would otherwise satisfy every
        assertion about the clauses it contains."""
        _fake_acceptance_tree(tmp_path, monkeypatch, met=[], unmet={})
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert "no clauses" in clause.detail

    def test_an_unmet_clause_with_no_stated_reason_fails(self, tmp_path, monkeypatch) -> None:
        """ "Failing honestly" is the requirement. A red board whose rows say
        nothing about why is the same absence as no board."""
        _fake_acceptance_tree(
            tmp_path, monkeypatch, met=["TestX::test_a"], unmet={"TestX::test_b": "   "}
        )
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert "TestX::test_b" in clause.detail

    def test_a_suite_that_went_fully_green_still_meets_the_clause(
        self, tmp_path, monkeypatch
    ) -> None:
        """The clause asks that the clauses were WRITTEN, not that they fail.
        Keying it on redness would make it go unmet the day the system
        satisfied it — a gate that inverts."""
        _fake_acceptance_tree(tmp_path, monkeypatch, met=["TestX::test_a"], unmet={})
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert clause.met, clause.detail

    def test_a_ratchet_beside_a_DELETED_suite_is_unmet(self, tmp_path, monkeypatch) -> None:
        """The defect an adversarial review found: `rm tests/acceptance/*.py`
        left the ratchet parsing perfectly and the clause reading MET over 24
        clauses that existed nowhere. The requirement says the suite EXISTS,
        so existence is read from source rather than assumed."""
        _fake_acceptance_tree(
            tmp_path, monkeypatch, met=["TestX::test_a"], unmet={"TestX::test_b": "phase 3"}
        )
        for module in (tmp_path / "acceptance").glob("test_*.py"):
            module.unlink()
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert "defined nowhere" in clause.detail
        assert "TestX::test_a" in clause.detail

    def test_a_clause_the_suite_defines_but_the_ratchet_omits_is_unmet(
        self, tmp_path, monkeypatch
    ) -> None:
        """The other direction, and the quieter one: a clause added to the
        suite and never recorded is a clause no reading is held to."""
        _fake_acceptance_tree(
            tmp_path,
            monkeypatch,
            met=["TestX::test_a"],
            unmet={},
            define=["TestX::test_a", "TestX::test_unrecorded"],
        )
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert "TestX::test_unrecorded" in clause.detail

    def test_an_absent_suite_directory_is_unmet(self, tmp_path, monkeypatch) -> None:
        _fake_acceptance_tree(tmp_path, monkeypatch, met=["TestX::test_a"], unmet={})
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_SUITE_DIR", tmp_path / "gone")
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert "not here" in clause.detail

    def test_a_parametrised_clause_id_matches_the_method_that_defines_it(
        self, tmp_path, monkeypatch
    ) -> None:
        """A source parse cannot see `[case]`; the ratchet's real ids carry
        it. Comparing the raw strings would report every parametrised clause
        as defined nowhere."""
        _fake_acceptance_tree(
            tmp_path,
            monkeypatch,
            met=["TestFault::test_each[a data source withheld]"],
            unmet={},
            define=["TestFault::test_each"],
        )
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert clause.met, clause.detail


class TestPhaseZeroSaysWhatItDoesNotGrade:
    """Partial coverage reported as complete is the defect this repo keeps
    finding. Phase 0 carries five deliverables and plan §6 makes two of them
    the gate, so the subset has to reach every surface the reading reaches —
    not only the clause list of a dated artifact somebody has to open."""

    def test_the_coverage_line_is_on_the_reading_and_on_the_durable_artifact(
        self, tmp_path
    ) -> None:
        result = evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY)
        assert "2 of 5" in result.coverage
        assert result.to_dict()["coverage"] == result.coverage
        assert result.coverage in result.render()
        for deliverable in PHASE0_DELIVERABLES:
            if deliverable.graded_by is None:
                assert deliverable.id in result.coverage

    def test_the_ladder_row_detail_carries_it_so_a_MET_phase_still_says_so(self, tmp_path) -> None:
        """A phase whose gate grades a SUBSET renders MET on the ladder and on
        the board with nothing saying so unless the row itself carries it."""
        store = _seed_phase0_met(tmp_path)
        rows = {r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY).to_dict()["phases"]}
        assert rows["phase0"]["state"] == "MET"
        assert "not gate-readable" in rows["phase0"]["detail"]
        assert "dead_lambdas_deleted" in rows["phase0"]["detail"]

    def test_a_gate_with_no_declared_deliverables_gets_no_coverage_line(self, tmp_path) -> None:
        """Silence is "not declared", never "grades everything". Phase 1
        declares no deliverable table, so it publishes no claim about one."""
        result = evaluate(LocalStore(tmp_path), gate="phase1", trading_day=FRIDAY)
        assert result.coverage is None
        assert result.to_dict()["coverage"] is None

    def test_a_deliverable_naming_a_clause_the_reading_lacks_RAISES(self) -> None:
        """The edit that would quietly shrink what "phase 0 is met" means: a
        clause renamed or dropped while the deliverable still claims it. Rule
        5's default is raise — a reading published from an inconsistent table
        grades less than it says it does."""
        with pytest.raises(ValueError, match="old_weekly_once_per_week"):
            coverage_note("phase0", [])

    def test_a_deliverable_added_with_no_decision_RAISES(self, monkeypatch) -> None:
        monkeypatch.setitem(
            GATE_DELIVERABLES,
            "phase0",
            (*PHASE0_DELIVERABLES, Deliverable("a_sixth_thing", "unruled", None, "")),
        )
        with pytest.raises(ValueError, match="a_sixth_thing"):
            evaluate(LocalStore("/tmp"), gate="phase0", trading_day=FRIDAY)

    def test_every_ungraded_deliverable_carries_a_written_reason(self) -> None:
        for deliverable in PHASE0_DELIVERABLES:
            if deliverable.graded_by is None:
                assert deliverable.reason.strip(), deliverable.id
            else:
                assert not deliverable.reason


class TestTheAcceptanceReadIsAlsoGuarded:
    """Same class of defect, second and third site. The ratchet and the suite
    are checked-in files a person edits by hand, so a malformed one is routine
    — and an exception raised while reading one takes `crucible gate`,
    `build_ladder` and the board render down together."""

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (b"{not json", "not readable JSON"),
            (b"[1, 2]", "parsed to list"),
            (b"null", "literal `null`"),
            (b'{"met": []}', "`unmet` is NoneType"),
            (b'{"met": [], "unmet": ["A::test_a"]}', "`unmet` is list"),
            (b'{"met": "A::test_a", "unmet": {}}', "`met` is str"),
        ],
        ids=[
            "truncated json",
            "a list",
            "literal null",
            "no unmet field",
            "unmet is a list",
            "met is a string",
        ],
    )
    def test_a_malformed_ratchet_is_a_red_clause_not_an_exception(
        self, tmp_path, monkeypatch, body: bytes, expected: str
    ) -> None:
        ratchet = tmp_path / "ratchet.json"
        ratchet.write_bytes(body)
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_RATCHET_PATH", ratchet)
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_SUITE_DIR", tmp_path)
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert expected in clause.detail

    def test_a_malformed_ratchet_does_not_take_the_ladder_down_with_it(
        self, tmp_path, monkeypatch
    ) -> None:
        ratchet = tmp_path / "ratchet.json"
        ratchet.write_bytes(b"{not json")
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_RATCHET_PATH", ratchet)
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_SUITE_DIR", tmp_path)
        rows = {
            r["phase"]: r
            for r in build_ladder(LocalStore(tmp_path), trading_day=FRIDAY).to_dict()["phases"]
        }
        assert rows["phase0"]["state"] == "UNMET"
        assert rows["phase0"]["clauses_total"] == 2

    def test_a_suite_module_that_does_not_parse_is_a_red_clause(
        self, tmp_path, monkeypatch
    ) -> None:
        """A `SyntaxError` out of the source walk is the same crash by another
        route, and "the suite does not parse" is an UNKNOWN clause set, never
        an empty one."""
        _fake_acceptance_tree(tmp_path, monkeypatch, met=["TestX::test_a"], unmet={})
        (tmp_path / "acceptance" / "test_broken.py").write_text("class Test(:\n", encoding="utf-8")
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert "test_broken.py could not be parsed" in clause.detail
        assert "SyntaxError" in clause.detail


class TestTheSourceScanSeesWhatPytestCollects:
    """A latent false UNMET on a clause that GATES THE PHASE is worse than one
    on a test: nobody is holding the gate wrong, and the phase simply stops
    exiting. Reproduced during review on a probe file."""

    def test_an_inherited_test_is_collected_under_the_CHILDS_name(
        self, tmp_path, monkeypatch
    ) -> None:
        """pytest collects `TestChild::test_inherited`. A scan reporting only
        `TestBase::test_inherited` flips the clause to UNMET against a
        perfectly correct ratchet."""
        suite = tmp_path / "acceptance"
        suite.mkdir()
        (suite / "test_inherit.py").write_text(
            "class TestBase:\n"
            "    def test_inherited(self):\n"
            "        raise AssertionError\n"
            "\n"
            "\n"
            "class TestChild(TestBase):\n"
            "    def test_own(self):\n"
            "        raise AssertionError\n",
            encoding="utf-8",
        )
        ratchet = suite / "ratchet.json"
        ratchet.write_text(
            json.dumps(
                {
                    "met": [
                        "TestBase::test_inherited",
                        "TestChild::test_inherited",
                        "TestChild::test_own",
                    ],
                    "unmet": {},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_RATCHET_PATH", ratchet)
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_SUITE_DIR", suite)
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert clause.met, clause.detail

    def test_a_base_class_in_another_module_of_the_suite_is_resolved(
        self, tmp_path, monkeypatch
    ) -> None:
        suite = tmp_path / "acceptance"
        suite.mkdir()
        (suite / "test_base.py").write_text(
            "class TestShared:\n    def test_shared(self):\n        raise AssertionError\n",
            encoding="utf-8",
        )
        (suite / "test_child.py").write_text(
            "class TestUser(TestShared):\n    def test_own(self):\n        raise AssertionError\n",
            encoding="utf-8",
        )
        ratchet = suite / "ratchet.json"
        ratchet.write_text(
            json.dumps(
                {
                    "met": [
                        "TestShared::test_shared",
                        "TestUser::test_shared",
                        "TestUser::test_own",
                    ],
                    "unmet": {},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_RATCHET_PATH", ratchet)
        monkeypatch.setattr("crucible.gate.ACCEPTANCE_SUITE_DIR", suite)
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert clause.met, clause.detail

    def test_a_module_level_test_function_is_NAMED_not_invisible(
        self, tmp_path, monkeypatch
    ) -> None:
        """pytest collects it; the ratchet's `Class::method` grammar cannot
        record it. Invisible would mean a clause nothing is ever held to, so
        the clause reads UNMET and names the function."""
        _fake_acceptance_tree(tmp_path, monkeypatch, met=["TestX::test_a"], unmet={})
        (tmp_path / "acceptance" / "test_loose.py").write_text(
            "def test_loose():\n    raise AssertionError\n", encoding="utf-8"
        )
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert not clause.met
        assert "test_loose.py::test_loose" in clause.detail

    def test_the_requirement_states_exactly_what_the_scan_can_see(self, tmp_path) -> None:
        """A future clause author reads the requirement, not this module. The
        one shape no static parse can see is named there rather than left to
        be discovered as a red gate."""
        clause = _clause(
            evaluate(LocalStore(tmp_path), gate="phase0", trading_day=FRIDAY),
            "acceptance_suite_committed",
        )
        assert SOURCE_SCAN_SCOPE in clause.requirement
        assert "inherit" in clause.requirement
        assert "setattr" in clause.requirement
