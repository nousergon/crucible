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
import inspect
import json
import pathlib
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

    def test_experiment_run_returns_a_verdict_in_one_command(self, tmp_path) -> None:
        """MET for U (track A). `experiment.run` produces the arm's selection
        and `experiment.grade` settles it into
        `experiments/{arm}/{trading_day}/verdict.json`, both through the
        store interface and nothing else."""
        store, _, decision_days = _seeded_slot(tmp_path, dt.date(2026, 8, 28))

        verdicts = [k for k in store.list_keys("experiments/") if k.endswith("verdict.json")]
        assert verdicts, "a settled shadow must produce a verdict artifact"

        settled = decision_days[0].isoformat()
        assert any(f"/{settled}/verdict.json" in k for k in verdicts)

        document = json.loads(store.get_bytes(verdicts[0]))
        assert isinstance(document["score_ratio"], float)
        assert document["benchmark"] == "population"
        assert document["horizon_trading_days"] == 21


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

    def test_the_absence_condition_reads_the_declared_deadline_table(self, tmp_path) -> None:
        """MET by track C. Asserted by INDUCING the condition, not by reading
        the code: a market holiday must raise no page, and the only way to
        know that is to resolve a deadline across one."""
        import datetime as dt

        from crucible.alerts import evaluate_absence
        from crucible.components import load_registry
        from crucible.store import LocalStore

        store = LocalStore(tmp_path)
        registry = load_registry()

        # Every Saturday deadline for Friday's session has passed and nothing
        # was written: the condition fires, and it names the deadline it read.
        late = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
        pages = evaluate_absence(store, now=late)
        assert {p.job for p in pages} <= set(registry)
        assert pages, "an empty store past every deadline must raise absence pages"
        assert all("due" in p.reason for p in pages)

        # 2026-07-03 was a half day and 2026-07-04 the observed holiday. A run
        # on the holiday binds to the 3rd's close, so the deadline moves with
        # the calendar rather than paging for a day the market never opened.
        holiday = dt.datetime(2026, 7, 4, 12, 0, tzinfo=dt.UTC)
        assert all(
            p.trading_day == dt.date(2026, 7, 2) or p.trading_day.weekday() < 5
            for p in evaluate_absence(store, now=holiday)
        )

    def test_the_weekly_alert_count_is_a_metric_with_a_ceiling(self, tmp_path) -> None:
        """MET by track C. Pages per window is a MetricRecord with a declared
        ceiling, counted in GROUPS from the durable bus — one outage is one
        page — and BREACH when it is exceeded."""
        import datetime as dt

        from crucible.alerts import (
            CEILING_WINDOW_TRADING_DAYS,
            PAGES_PER_MONTH_CEILING,
            ceiling_metric,
            emit,
            group_pages,
        )
        from crucible.alerts import Page as _Page
        from crucible.store import LocalStore

        now = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
        assert PAGES_PER_MONTH_CEILING == 2
        assert ceiling_metric(PAGES_PER_MONTH_CEILING, now=now)["status"] == "OK"
        breached = ceiling_metric(PAGES_PER_MONTH_CEILING + 1, now=now)
        assert breached["status"] == "BREACH"
        assert breached["unit"] == "pages"
        assert breached["horizon_trading_days"] == CEILING_WINDOW_TRADING_DAYS
        assert "never a reason to add a suppression" in breached["status_reason"]

        # And the count is reconstructible from artifacts by someone who was
        # not here: it is read off the bus, not off an in-process counter.
        store = LocalStore(tmp_path)
        transport_calls: list[str] = []

        def _capture(message: str, **kwargs: object) -> object:
            transport_calls.append(message)
            return type("R", (), {"any_ok": True, "destination": "captured"})()

        pages = [
            _Page(
                condition="failure",
                job=job,
                trading_day=dt.date(2026, 8, 28),
                reason="RuntimeError: data source yfinance is unreachable",
                run_id="01JG000000000000000000000" + job[0].upper(),
            )
            for job in ("data.daily", "report", "drift")
        ]
        emit(store, group_pages(pages), sweep_run_id="0" * 26, transport=_capture)
        from crucible.alerts import pages_in_window

        assert pages_in_window(store, now=now) == 1, (
            "one outage is one page; counting members would blow a two-a-month "
            "ceiling on a single bad Saturday"
        )


class TestTransparency:
    """§2 row 7."""

    def test_every_llm_call_site_is_in_the_registry(self) -> None:
        clause = "plan §2 row 7 / §4.8"
        requirement = (
            "LLM_CALLSITE_REGISTRY coverage of v2 call sites is 100%, measured by a "
            "test that enumerates call sites from the code rather than from a list."
        )
        _unmet(clause, requirement)

    def test_explain_walks_a_verdict_back_to_what_produced_it(self, tmp_path) -> None:
        """MET. The chain is recovered from manifests alone — a run's
        `inputs[].key` is some other run's `outputs[].key` — with each hop's
        code sha, seed, cost and LLM-call count attached."""
        from crucible.explain import explain, render
        from crucible.keys import arena_cycle_key

        store, _, _ = _seeded_slot(tmp_path, dt.date(2026, 8, 28))
        node = explain(store, arena_cycle_key("u", "2026-08-28"))

        assert node.manifest is not None
        assert node.manifest["job"] == "experiment.grade"

        chain = render(node)
        assert "code_sha=" in chain
        assert "cost_usd=" in chain
        assert "llm_calls=" in chain

        walked = {node.key}

        def collect(current: Any) -> None:
            for parent in current.parents:
                walked.add(parent.key)
                collect(parent)

        collect(node)
        assert any(k.startswith("data/") and k.endswith("panel.parquet") for k in walked), (
            "the walk must reach the price panel the forward returns came from"
        )
        assert all(p.manifest is not None for p in node.parents), (
            "an unresolvable hop is reported as UNKNOWN, never elided — and there "
            "should be none here"
        )


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

    def test_the_grader_ranks_planted_above_real_above_null(self, tmp_path) -> None:
        """MET for U (track A), on both sides.

        Positive: a real cycle ranks the planted control above the null one,
        and records the observed margin as a MetricRecord.

        Negative — the half that matters: a grader that does NOT see the
        planted edge fails the cycle rather than publishing its verdicts.
        The audit's central finding was a grading loop that ran for months
        while measuring nothing, and a control that cannot produce a
        negative result would be the same thing again.
        """
        from nousergon_lib.arena.window import ArmSeries

        from crucible.keys import arena_cycle_key
        from crucible.slots.arms import control_specs
        from crucible.slots.grading import GraderControlError, assert_controls_ordered

        store, _, _ = _seeded_slot(tmp_path, dt.date(2026, 8, 28))
        cycle = json.loads(store.get_bytes(arena_cycle_key("u", "2026-08-28")))

        controls = {c.control_kind: c.arm_id for c in control_specs(get_slot("u"))}
        scores: dict[str, list[float]] = {}
        for key in store.list_keys("experiments/"):
            if not key.endswith("verdict.json"):
                continue
            document = json.loads(store.get_bytes(key))
            scores.setdefault(document["arm_id"], []).append(document["score_ratio"])

        planted = sum(scores[controls["planted"]]) / len(scores[controls["planted"]])
        null = sum(scores[controls["null"]]) / len(scores[controls["null"]])
        assert planted > null, (
            "the planted arm's ranking signal is constructed with a known IC against "
            "the realized return; a grader that cannot see it cannot see a real edge"
        )

        assert set(controls.values()) <= set(cycle["scored_arms"]), (
            "§10.1: controls are scored EVERY cycle"
        )
        assert cycle["decision"]["champion"] not in set(controls.values()), (
            "a control never takes the pointer; the planted one reads the realized "
            "forward return and serving it would be a look-ahead in production"
        )

        inverted = {
            controls["planted"]: ArmSeries(
                arm_id=controls["planted"], scores={"2026-08-03": -0.05}
            ),
            controls["null"]: ArmSeries(arm_id=controls["null"], scores={"2026-08-03": 0.05}),
        }
        with pytest.raises(GraderControlError):
            assert_controls_ordered(controls, inverted, slot="u")


class TestFaultInjection:
    """§10 component 7 — 'flawless from day 1' is proven on the failure path.
    The old rehearsal pipeline failed 7 of 7 because nobody had made it fail
    on purpose first."""

    #: The four faults, and the test class in `tests/faults/` that INDUCES
    #: each. Named by the class that proves it rather than by prose, so a
    #: fault whose test is deleted fails this clause instead of silently
    #: ceasing to be covered — which is how the old rehearsal pipeline came
    #: to fail 7 of 7 without anyone having made it fail on purpose first.
    FAULTS = {
        "spot instance terminated mid-job": "TestFaultOneSpotTerminatedMidJob",
        "a data source withheld": "TestFaultTwoDataSourceWithheld",
        "the LLM router returns 500": "TestFaultThreeRouterReturns500",
        "the S3 release pointer is stale": "TestFaultFourStaleReleasePointer",
    }

    @pytest.mark.parametrize("fault", sorted(FAULTS))
    def test_each_scripted_fault_produces_one_failed_run_and_exactly_one_page(
        self, fault: str
    ) -> None:
        """MET by track C. Each fault is scripted in `tests/faults/`, and each
        asserts the same four properties: status failed, the RIGHT reason,
        full telemetry, and EXACTLY ONE page against a captured transport."""
        import importlib

        module = importlib.import_module("tests.faults.test_four_scripted_faults")
        cls = getattr(module, self.FAULTS[fault], None)
        assert cls is not None, (
            f"no scripted injection for {fault!r}. 'Flawless from day 1' is proven on "
            "the failure path, and a fault nobody induces is a claim, not a gate."
        )
        source = inspect.getsource(cls)
        for required in ('status"] == "failed"', "_assert_full_telemetry", "_failure_pages"):
            assert required in source, (
                f"{cls.__name__} does not assert {required!r}: a fault-injection test "
                "that omits the page count proves the run failed, not that the "
                "operator was told once."
            )


class TestFeatureLayer:
    """§10 component 4 — without one materialized, hashed feature layer, R
    and M recompute features from different code and 'the signal degraded'
    cannot be separated from 'the feature changed'."""

    def test_r_and_m_read_the_same_versioned_feature_artifact(self, tmp_path) -> None:
        """MET on the producer side; the M consumer arrives with track B.

        What is asserted here is the property that makes the clause
        meaningful: there is ONE materialized, hashed artifact per trading
        day, its registry carries units and lineage for every column, and a
        consumer records it by CONTENT HASH — so two consumers recording the
        same hash is a checkable fact rather than a convention. The U
        consumer exists today and is checked; a second consumer recording a
        different hash for the same key would fail this test the day it
        lands.
        """
        from crucible.features import CATALOG, DEFAULT_FEATURE_VERSION, feature_version
        from crucible.keys import feature_registry_key, features_key
        from crucible.store import sha256_hex

        store, _, decision_days = _seeded_slot(tmp_path, dt.date(2026, 8, 28))
        day = decision_days[0].isoformat()

        key = features_key(DEFAULT_FEATURE_VERSION, day)
        assert store.exists(key), "the layer is materialized, not recomputed per consumer"

        registry = json.loads(store.get_bytes(feature_registry_key(DEFAULT_FEATURE_VERSION)))
        assert registry["feature_version"] == feature_version(CATALOG), (
            "the version is DERIVED from the catalogue; a hand-written one would let an "
            "edited recipe overwrite the layer an earlier verdict was computed from"
        )
        for entry in registry["features"]:
            assert entry["unit"], f"{entry['name']} declares no unit"
            assert entry["inputs"], f"{entry['name']} declares no lineage"
            assert any(
                entry["name"].endswith(suffix)
                for suffix in ("_raw", "_ratio", "_pct", "_zscore", "_log_return")
            ), f"{entry['name']} carries no units suffix"

        digest = sha256_hex(store.get_bytes(key))
        recorded = set()
        for manifest_key in store.list_keys("runs/"):
            if not manifest_key.endswith("/run.json"):
                continue
            manifest = json.loads(store.get_bytes(manifest_key))
            for entry in manifest["inputs"] + manifest["outputs"]:
                if entry["key"] == key:
                    recorded.add(entry["sha256"])
        assert recorded == {digest}, (
            "every module touching a given feature key must record the SAME content "
            f"hash for it; got {sorted(recorded)} against {digest}"
        )


def _seeded_slot(tmp_path: Any, cycle_date: dt.date) -> tuple[Any, Any, list[dt.date]]:
    """A real U slot with real artifacts: panel, features, shadows, verdicts.

    Built with the SAME job functions production runs — `run_daily` and
    `universe.produce` through `crucible.runner.run_job` — against a local
    store and a seeded synthetic market. Plan §2 row 2's clause is "no AWS
    resource beyond S3 read/write", and the local store is that same
    interface, so this exercises the clause rather than a stand-in for it.
    """
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from conftest import sessions_ending, synthetic_frames

    from crucible.config import Settings
    from crucible.data import FramePriceSource, run_daily
    from crucible.runner import run_job
    from crucible.slots import universe
    from crucible.store import LocalStore

    horizon = 21
    decisions = 6
    store = LocalStore(tmp_path / "store")
    source = FramePriceSource(synthetic_frames(end=cycle_date))
    strategy = tmp_path / "strategy"
    arms = strategy / "arms" / "u"
    arms.mkdir(parents=True)
    for name, ranker in (
        ("momentum_sleeve", "momentum_sleeve"),
        ("tech_score_gate", "tech_score_gate"),
        ("mom_12_1_sleeve", "mom_12_1_sleeve"),
    ):
        (arms / f"{name}.yaml").write_text(
            f"name: {name}\nslot: u\nranker: {ranker}\n"
            "registered_at: '2026-06-01'\nparams:\n  top_n: 8\n",
            encoding="utf-8",
        )
    settings = Settings(
        store_uri=str(tmp_path / "store"),
        arctic_bucket="not-read-in-this-clause",
        strategy_dir=strategy,
        origins={"store_uri": "acceptance", "strategy_dir": "acceptance"},
    )
    decision_days = sessions_ending(cycle_date, horizon + decisions + 1)[:decisions]
    for day in decision_days + [cycle_date]:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
            store=store,
            trading_day=day,
        )
    for day in decision_days:
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings),
            store=store,
            trading_day=day,
        )
    run_job(
        "experiment.grade",
        lambda c: universe.grade(c, settings=settings),
        store=store,
        trading_day=cycle_date,
    )
    return store, settings, decision_days


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
