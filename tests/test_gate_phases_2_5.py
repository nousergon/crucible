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

import datetime as dt
import json

import pytest

import crucible.autonomy as autonomy_module
import crucible.cost as cost_module
import crucible.gate as gate_module
import crucible.llm as llm_module
from crucible.gate import (
    GATES,
    MANIFEST_RUN_MODE_FIELD,
    MANIFEST_RUN_MODE_LIVE,
    PHASE2_LIVE_SATURDAYS,
    PHASE2_MAX_PAGES,
    PHASE2_MAX_TAGGED_USD,
    PHASE4_MAX_TOTAL_USD,
    PHASES,
    evaluate,
)
from crucible.keys import (
    arm_register_key,
    champion_key,
    manifest_key,
    runs_prefix,
    verdict_key,
)
from crucible.slots import SLOTS
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SHA = "a" * 40


def _window(weeks: int) -> list[dt.date]:
    return [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(weeks))]


PHASE2_WINDOW = _window(PHASE2_LIVE_SATURDAYS)


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
            assert not clause.met
            assert clause.detail.strip(), f"{gate}/{clause.name} gave no reason"


def _raising_client() -> object:
    raise RuntimeError("no credentials are configured for this reading identity")


# ---------------------------------------------------------------------------
# phase 2 — live Saturdays
# ---------------------------------------------------------------------------


def _live_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend `run_manifest.v1` declares the live/replay field.

    It does NOT today — that gap is filed as its own issue and is exactly what
    the unmeasurable case below asserts. Patched here so the MET and UNMET
    branches are reachable and tested BEFORE the schema lands, rather than
    shipping two untested branches that first run in production.
    """
    monkeypatch.setattr(
        gate_module,
        "_manifest_property_names",
        lambda: frozenset({MANIFEST_RUN_MODE_FIELD, "status", "attempts"}),
    )


def _weekly(day: dt.date, *, mode: str = MANIFEST_RUN_MODE_LIVE, attempts: int = 1) -> dict:
    return {
        "status": "ok",
        "reason": "",
        MANIFEST_RUN_MODE_FIELD: mode,
        "attempts": [{"n": n + 1} for n in range(attempts)],
    }


class TestLiveSaturdaysAreReadFromTheManifestNeverTheDate:
    def test_unmeasurable_when_the_manifest_cannot_say_live_or_replay(
        self, store: LocalStore
    ) -> None:
        """Today's reading, and the one that matters: the schema declares no
        live/replay field, so the clause refuses to answer rather than
        inferring liveness from the trading day — which the accelerated replay
        schedule would satisfy."""
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert MANIFEST_RUN_MODE_FIELD in clause.detail

    def test_met_when_both_saturdays_are_live_and_first_attempt_ok(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _live_schema(monkeypatch)
        for day in PHASE2_WINDOW:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_WINDOW)
        assert clause.met and not clause.unmeasurable

    def test_a_replay_is_unmet_not_met(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole reason the clause list was withheld. A perfect replay
        Saturday is UNMET here, and says which day was not live."""
        _live_schema(monkeypatch)
        for day in PHASE2_WINDOW:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day, mode="replay"))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "not live" in clause.detail

    def test_a_retried_run_is_not_a_first_attempt_ok(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _live_schema(monkeypatch)
        for day in PHASE2_WINDOW:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day, attempts=2))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_WINDOW)
        assert not clause.met
        assert "retried" in clause.detail

    def test_an_absent_manifest_is_unmet_by_name(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _live_schema(monkeypatch)
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_WINDOW)
        assert not clause.met
        assert "never ran" in clause.detail


class TestReplaysReuseThePhaseOnePredicate:
    def test_the_clause_is_phase_ones_reading_renamed_over_five_saturdays(
        self, store: LocalStore
    ) -> None:
        """Not a restatement. The requirement quotes phase 1's own sentence and
        the window is the five replay Saturdays, so phase 2 and phase 1 cannot
        disagree about what a good replay is."""
        clause = gate_module._clause_replays_ok(store, PHASE2_WINDOW, {})
        assert clause.name == "replays_ok"
        assert not clause.met
        assert (FRIDAY - dt.timedelta(weeks=4)).isoformat() in clause.requirement


# ---------------------------------------------------------------------------
# phase 2 — autonomy, pages, cost
# ---------------------------------------------------------------------------


def _archive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure an archive location so the reader gets past the "no trail"
    branch. A test-only value: no bucket name is written into this tree."""
    monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://a-test-archive/trail")


class _Counted:
    def __init__(self, count: int) -> None:
        self.count = count
        self.actions = []
        self.objects_read = 3
        self.records_scanned = 40


class TestHumanMutatingCallsAreCountedOrNotAnswered:
    def test_unmeasurable_when_no_archive_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Today's reading, and `0` and `no trail` must never be the same
        answer. There is no default archive on purpose: a guessed bucket name
        produces a `NoSuchBucket` that reads like a permissions problem."""
        monkeypatch.delenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", raising=False)
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "no CloudTrail archive is configured" in clause.detail

    def test_unmeasurable_when_the_archive_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _archive(monkeypatch)
        monkeypatch.setattr(gate_module, "_s3_client", _raising_client)
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "RuntimeError" in clause.detail

    def test_met_on_a_clean_archive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _archive(monkeypatch)
        monkeypatch.setattr(gate_module, "_s3_client", lambda: object())
        monkeypatch.setattr(autonomy_module, "count_operator_actions", lambda *a, **k: _Counted(0))
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert clause.met and not clause.unmeasurable

    def test_unmet_when_a_human_mutated_something(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _archive(monkeypatch)
        monkeypatch.setattr(gate_module, "_s3_client", lambda: object())
        monkeypatch.setattr(autonomy_module, "count_operator_actions", lambda *a, **k: _Counted(3))
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "3 human mutating call" in clause.detail


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
        assert f"ceiling {PHASE2_MAX_PAGES}" in clause.detail


class _CostClient:
    """A Cost Explorer stand-in speaking the real response shape."""

    def __init__(self, amount: str) -> None:
        self.amount = amount

    def get_cost_and_usage(self, **_) -> dict:
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

    def test_a_tag_filtered_zero_is_unmeasurable_not_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An untagged estate and a free month return the same `$0.00`, so the
        figure measures the FILTER rather than the spend."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("0"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met

    def test_met_under_the_tagged_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("12.50"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.met and not clause.unmeasurable

    def test_unmet_over_the_account_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Phase 4's reading: same reader, no tag filter, the other ceiling.
        An unfiltered zero is a real reading and is NOT special-cased."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("230.21"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable
        assert "230.21" in clause.detail


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

    def test_one_clause_per_registered_slot(self, store: LocalStore) -> None:
        clauses = gate_module._phase3(store, _window(4), {})
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


class TestDecommissionedMeansZeroNotWithinCadence:
    def test_the_phase_four_reading_is_the_phase_zero_reader_at_a_zero_ceiling(
        self, store: LocalStore
    ) -> None:
        """One reader, two ceilings. A second copy of this count is how phase 0
        and phase 4 would come to disagree about what a v1 start is."""
        clause = gate_module._clause_old_weekly_within_cadence(
            store, _window(2), name="old_sf_execution_count_zero", maximum=0
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
                {"executions_started": 1},
            )
        clause = gate_module._clause_old_weekly_within_cadence(
            store, _window(2), name="old_sf_execution_count_zero", maximum=0
        )
        assert not clause.met
        assert "ceiling 0" in clause.detail

    def test_zero_starts_is_met(self, store: LocalStore) -> None:
        from crucible.gate import legacy_weekly_executions_key, weekly_anchor

        for day in _window(2):
            _put(
                store,
                legacy_weekly_executions_key(weekly_anchor(day).isoformat()),
                {"executions_started": 0},
            )
        clause = gate_module._clause_old_weekly_within_cadence(
            store, _window(2), name="old_sf_execution_count_zero", maximum=0
        )
        assert clause.met


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
        use one."""
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.unmeasurable and not clause.met
        assert "not " in clause.detail

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

    @staticmethod
    def _register(store: LocalStore, slot: str, arms: list[tuple[str, dict]]) -> str:
        """One `registered` event per arm, carrying the recipe `spec`.

        The `spec.params` carrier is the shape the call-site field WILL take
        once the arm-recipe gap is closed; the register does not emit it
        today, which is why the two clauses above read unmeasurable. Writing
        the fixture in that shape is what makes the met and unmet branches
        tested before they first run rather than after.
        """
        lines = []
        arm_id = ""
        for name, params in arms:
            arm_id = f"{slot}:{name}:ab12cd"
            lines.append(
                json.dumps(
                    {
                        "kind": "registered",
                        "arm_id": arm_id,
                        "date": "2026-01-02",
                        "reason": "",
                        "spec": {"name": name, "slot": slot, "params": params},
                        "record": {
                            "arm_id": arm_id,
                            "slot": slot,
                            "name": name,
                            "spec_hash": "ab12cd",
                            "created_date": "2026-01-02",
                        },
                    },
                    sort_keys=True,
                )
            )
        store.put_bytes(arm_register_key(slot), ("\n".join(lines) + "\n").encode("utf-8"))
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
