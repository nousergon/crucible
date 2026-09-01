"""Plan §2 — "Objectives, made testable", as executable assertions.

Each test below is one row of the §2 table, or one §6 phase gate, or one §10
component whose absence makes a verdict wrong. **They fail today.** See
`tests/acceptance/README.md`: no xfail, no skip, no marker — the repository
carries no suppression collection at all (§11.1), and a gate that was marked
as expected-to-fail is a gate nobody notices going green.

Every test performs the REAL check. Where the implementation does not exist
it raises `NotImplementedError`, and `_unmet` converts that into a failure
naming the clause and the owning track — so a track landing the code turns
the test green with no marker to remove.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any, NoReturn

import pytest

from crucible.slots import SLOTS, get_slot


def _unmet(clause: str, requirement: str, exc: BaseException | None = None) -> NoReturn:
    """Fail this acceptance clause with everything needed to act on it."""
    detail = f" Blocked on: {exc}" if exc is not None else ""
    pytest.fail(
        f"UNMET — {clause}\n"
        f"  Required: {requirement}\n"
        f"  Status:   not yet satisfied (crucible v2 phase 1, "
        f"alpha-engine-config-I9757).{detail}",
        pytrace=False,
    )


def _attempt(clause: str, requirement: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except NotImplementedError as exc:
        _unmet(clause, requirement, exc)


class TestAutonomy:
    """§2 row 1: 'Runs autonomously, minimal input'."""

    def test_four_consecutive_saturdays_first_attempt_ok(self) -> None:
        clause = "plan §2 row 1 / §6 phase-2 exit gate"
        requirement = (
            "4 consecutive weekly runs, first attempt, status: ok, each writing "
            "runs/weekly/{trading_day}/run.json, with zero human-originated mutating "
            "calls on v2 resources over the window."
        )
        from crucible.cli import HANDLERS

        _attempt(clause, requirement, lambda: HANDLERS["report"](_args()))

    def test_zero_human_mutating_calls_is_read_from_the_cloudtrail_archive(
        self,
    ) -> None:
        """§11 risk 8: `lookup-events` truncates its username lookup to ~2
        days, so a gate querying it reads clean because it could not see the
        week. The gate must read the S3 CloudTrail archive over the full
        window."""
        clause = "plan §2 row 1, closed by §11 risk 8"
        requirement = (
            "The operator-action count is computed from the CloudTrail S3 archive "
            "over the full 4-week window, never from `aws cloudtrail lookup-events`."
        )
        _unmet(clause, requirement)


class TestOneCommand:
    """§2 row 2: 'Easy to run experiments'."""

    def test_experiment_run_returns_a_verdict_in_one_command(self) -> None:
        clause = "plan §2 row 2"
        requirement = (
            "`crucible experiment.run --slot r --arm <id>` exits 0 from a laptop and "
            "writes experiments/{arm}/{trading_day}/verdict.json, using no AWS "
            "resource beyond S3 read/write."
        )
        from crucible.cli import HANDLERS

        _attempt(clause, requirement, lambda: HANDLERS["experiment.run"](_args(slot="r")))


class TestCost:
    """§2 row 3: 'Low API + AWS cost'."""

    def test_the_llm_spend_cap_is_declared_and_enforced(self) -> None:
        clause = "plan §2 row 3"
        requirement = (
            "The per-weekly-run LLM cap is declared in config and enforced through "
            "krepis.usage_pacing; a run that would exceed it FAILS rather than "
            "overspending, and run.json.cost_usd carries the spend."
        )
        _unmet(clause, requirement)

    def test_every_v2_resource_is_tagged_for_cost_attribution(self) -> None:
        clause = "plan §2 row 3 / §6 phase-0"
        requirement = (
            "Every v2 AWS resource carries tag system=crucible-v2, so the monthly "
            "cost row has a denominator and the <= $40/mo ceiling is measurable "
            "rather than asserted (§11 risk 7)."
        )
        _unmet(clause, requirement)


class TestNoThirdState:
    """§2 row 4: 'Works flawlessly from day 1'. This row is the one the
    foundation already satisfies — the schema makes the third state
    unrepresentable — so it is asserted here rather than deferred."""

    def test_the_manifest_schema_admits_exactly_two_statuses(self) -> None:
        import json

        from crucible.manifest import SCHEMA_PATH

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert schema["properties"]["status"]["enum"] == ["ok", "failed"], (
            "plan §2 row 4: the schema must forbid the third state, not policy."
        )

    def test_no_job_can_declare_itself_skipped(self) -> None:
        from crucible.runner import RunContext

        ctx = RunContext(
            run_id="01JG0000000000000000000000",
            job="data.daily",
            trading_day=dt.date(2026, 8, 28),
            calendar_date=dt.date(2026, 8, 28),
            store=None,  # type: ignore[arg-type]
            seed=0,
            started=dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC),
        )
        with pytest.raises(ValueError):
            ctx.set_status("skipped")


class TestAttribution:
    """§2 row 5: 'Pinpoint underperformance'."""

    def test_the_attribution_table_has_five_rows(self) -> None:
        clause = "plan §2 row 5 / §4.5"
        requirement = (
            "`crucible report` writes report/{trading_day}/attribution.json with five "
            "MetricRecord rows — data freshness/coverage, signal IC (R), prediction IC "
            "(M), portfolio alpha (S), execution shortfall — each with value, ci, n, "
            "baseline and status."
        )
        from crucible.cli import HANDLERS

        _attempt(clause, requirement, lambda: HANDLERS["report"](_args()))

    def test_alpha_is_factor_neutral_not_raw_excess_return(self) -> None:
        """§10.3: raw 'alpha vs SPY' is mostly beta and sector tilt, and a
        champion promoted on it is promoted on exposure."""
        clause = "plan §10 component 3"
        requirement = (
            "Each slot's return is decomposed into market beta, sector, size and "
            "residual; the attribution table reports RESIDUAL alpha."
        )
        _unmet(clause, requirement)


class TestAlerting:
    """§2 row 6: 'Alerts only when wrong'."""

    def test_there_are_exactly_two_page_conditions(self) -> None:
        from crucible.alerts import PAGE_CONDITIONS

        assert PAGE_CONDITIONS == ("absence", "failure"), (
            "plan §2 row 6 / §4.6: exactly two page conditions, no others."
        )

    def test_the_absence_condition_reads_the_declared_deadline_table(self) -> None:
        clause = "plan §2 row 6 / §4.6"
        requirement = (
            "Absence pages are computed from crucible/components.yaml's deadline "
            "table, resolved against the trading calendar so a market holiday raises "
            "no page."
        )
        from crucible.alerts import evaluate_absence

        _attempt(clause, requirement, evaluate_absence)

    def test_the_weekly_alert_count_is_a_metric_with_a_ceiling(self) -> None:
        clause = "plan §2 row 6 / §11 risk 2"
        requirement = (
            "Pages per week is itself a MetricRecord with a declared ceiling "
            "(target <= 2 pages/month); exceeding it is a defect in the alerting "
            "module, never a reason to add a suppression."
        )
        _unmet(clause, requirement)


class TestTransparency:
    """§2 row 7."""

    def test_every_llm_call_site_is_in_the_registry(self) -> None:
        clause = "plan §2 row 7 / §4.8"
        requirement = (
            "LLM_CALLSITE_REGISTRY coverage of v2 call sites is 100%, measured by a "
            "test that enumerates call sites from the code rather than from a list."
        )
        _unmet(clause, requirement)

    def test_explain_walks_a_verdict_back_to_what_produced_it(self) -> None:
        clause = "plan §10 component 8 / §10.8"
        requirement = (
            "`crucible explain <run_id|verdict>` prints the chain from a verdict to "
            "the arms, features, data snapshot, code sha, cost and LLM calls that "
            "produced it — principle 1 in five seconds rather than an S3 "
            "archaeology session."
        )
        from crucible.cli import HANDLERS

        _attempt(clause, requirement, lambda: HANDLERS["explain"](_args(target="x")))


class TestControlArms:
    """§10 component 1 — a hard precondition for any replay run counting
    toward a gate. The audit's central finding is a grading loop that ran for
    months while measuring nothing; a control that can produce a negative
    result is the only way to know the harness itself works."""

    def test_every_slot_declares_a_planted_and_a_null_control(self) -> None:
        for slot in SLOTS:
            kinds = {c.kind for c in get_slot(slot).control_arms}
            assert kinds == {"planted", "null"}, (
                f"plan §10 component 1: slot {slot} must declare both controls; got {kinds}."
            )

    def test_the_grader_ranks_planted_above_real_above_null(self) -> None:
        clause = "plan §10 component 1"
        requirement = (
            "In every cycle the grader ranks planted > real-or-null > null with the "
            "expected margin. If it does not, the GRADER is broken and the cycle's "
            "verdicts are void — the cycle fails rather than publishing them."
        )
        _unmet(clause, requirement)


class TestFaultInjection:
    """§10 component 7 — 'flawless from day 1' is proven on the failure path.
    The old rehearsal pipeline failed 7 of 7 because nobody had made it fail
    on purpose first."""

    @pytest.mark.parametrize(
        "fault",
        [
            "spot instance terminated mid-job",
            "a data source withheld",
            "the LLM router returns 500",
            "the S3 release pointer is stale",
        ],
    )
    def test_each_scripted_fault_produces_one_failed_run_and_exactly_one_page(
        self, fault: str
    ) -> None:
        clause = "plan §10 component 7 / §6 phase-2 gate"
        requirement = (
            f"With fault injected ({fault}), the run produces status: failed with the "
            "right reason, full telemetry in the manifest, and exactly one page."
        )
        _unmet(clause, requirement)


class TestFeatureLayer:
    """§10 component 4 — without one materialized, hashed feature layer, R
    and M recompute features from different code and 'the signal degraded'
    cannot be separated from 'the feature changed'."""

    def test_r_and_m_read_the_same_versioned_feature_artifact(self) -> None:
        clause = "plan §10 component 4"
        requirement = (
            "features/{version}/{trading_day}.parquet exists with a registry carrying "
            "units and lineage, and both R and M record it as the same input hash in "
            "their manifests."
        )
        _unmet(clause, requirement)


def _args(**overrides: Any) -> Any:
    """A minimal argparse-like namespace for calling a handler directly."""
    import argparse

    ns = argparse.Namespace(
        job="report",
        date=None,
        trading_day=dt.date(2026, 8, 28),
        dry_run=False,
        store=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns
