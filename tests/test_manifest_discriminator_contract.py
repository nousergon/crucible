"""alpha-engine-config-I10967: a job's manifest key distinguishes its invocations.

**The class, and why the issue as filed was mis-scoped.** I10967 was filed as
"nothing detects a store mutation that wrote no manifest", off the measurement
that seven arm registrations on 2026-09-17 left no `runs/experiment.register/`
prefix. Measured since: `runs/experiment.new/2026-09-11/run.json` carries
SEVEN object versions at 00:23:41-00:24:29Z, matching the seven register
appends second for second. The appends DID go through `run_job` and
`experiment.new` manifested all seven times. What defeated the record is that
it wrote every one of them to the same key: `crucible.keys.manifest_key`
writes `runs/{job}/{trading_day}/run.json` when a job passes no
``discriminator``, so N invocations on one trading day collapse to one
manifest, last writer wins. Six runs' lineage was overwritten by the seventh.

So the class worth detecting is **a manifest key that N runs overwrite**, and
the property this file asserts is: a job's discriminator distinguishes its
invocations, or something in the registry bounds it to one invocation per
trading day.

**Derived over the whole component registry, never an enumerated list.** The
required side comes from `crucible/components.yaml`'s own `dispatch` column
and the actual side comes from an AST scan of every `run_job` call in the
package. Nothing here is a list of jobs someone remembered: a new row joins
the domain the moment it is added to the registry. That is the same shape
`tests/test_store_consumer_contract.py` uses for the mirror-image reader
property, and the opposite of the bare `{"deploy"}` literal
`alpha-engine-config-I10969` was filed over.

The runtime half lives in `crucible.alerts.indistinguishable_invocation_findings`
and reports on the sweep's own manifest as `unmeasurable`, never as green.
"""

from __future__ import annotations

import ast
import datetime as dt
import importlib
import json
from pathlib import Path

import pytest

from crucible.alerts import (
    indistinguishable_invocation_findings,
    indistinguishable_invocation_metric,
    on_demand_graded_jobs,
)
from crucible.cli import JOBS
from crucible.components import load_registry
from crucible.keys import manifest_key
from crucible.models import TRADER_JOB_VALUES, WORKFLOW_JOB_VALUES
from crucible.store import LocalStore

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "crucible"

#: The three states a job's manifest key can be in. Closed and TOTAL — a job
#: the classifier cannot place is not silently skipped, it raises. A
#: fall-through here would be the same defect one level up: a component the
#: classifier cannot place renders `UNREPORTED`, which is loud
#: (`observability-policy` §8.3).
DISCRIMINATED = "discriminated"
BOUNDED_BY_ITS_STARTER = "bounded_by_its_starter"
INDISTINGUISHABLE = "indistinguishable"


def _job_name(node: ast.Call, module: ast.Module, module_name: str | None = None) -> str | None:
    """The job a `run_job(...)` call names, resolved through a module-level
    constant when it is one (`FAULT_RECORD_JOB`, `MORNING_JOB`, ...).

    Returns ``None`` only for a call whose job cannot be resolved statically
    at all — and the caller ASSERTS there are none, rather than skipping
    them. A scanner that silently drops what it cannot parse is a scanner
    whose coverage is unknown.
    """
    argument: ast.expr | None = None
    for keyword in node.keywords:
        if keyword.arg == "job":
            argument = keyword.value
    if argument is None and node.args:
        argument = node.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    if isinstance(argument, ast.Name):
        for statement in module.body:
            if isinstance(statement, ast.Assign):
                targets = [t.id for t in statement.targets if isinstance(t, ast.Name)]
                if argument.id in targets and isinstance(statement.value, ast.Constant):
                    return str(statement.value.value)
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == argument.id
                and isinstance(statement.value, ast.Constant)
            ):
                return str(statement.value.value)
        if module_name is not None:
            # The constant is IMPORTED (`from crucible.faults import
            # FAULT_RECORD_JOB`). Resolved by importing the module and reading
            # the attribute rather than by chasing the import graph in the
            # AST: the package imports cleanly, and a partial import resolver
            # is a denylist of the import syntax someone thought of.
            resolved = getattr(importlib.import_module(module_name), argument.id, None)
            if isinstance(resolved, str):
                return resolved
    return None


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)


def _call_sites(tree: ast.Module) -> list[tuple[ast.Call, bool]]:
    sites: list[tuple[ast.Call, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "run_job":
            continue
        sites.append((node, any(k.arg == "discriminator" for k in node.keywords)))
    return sites


def scan_run_job_call_sites() -> tuple[dict[str, bool], list[str]]:
    """``{job: passes a discriminator}`` over the whole package, plus every
    call site whose job could not be resolved."""
    discriminated: dict[str, bool] = {}
    unresolved: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call, has_discriminator in _call_sites(tree):
            job = _job_name(call, tree, _module_name(path))
            if job is None:
                unresolved.append(f"{path.relative_to(PACKAGE_ROOT.parent)}:{call.lineno}")
                continue
            # ANY call site passing one makes the key discriminated for that
            # job; a job with two entry points where only one discriminates
            # is caught by the per-call-site assertion below, not here.
            discriminated[job] = discriminated.get(job, False) or has_discriminator
    return discriminated, unresolved


def classify(job: str, dispatch: str | None, discriminated: bool) -> str:
    if discriminated:
        return DISCRIMINATED
    if dispatch is not None:
        # A cron, the scheduler or the arc fires the row once per trading day,
        # so the single undiscriminated key holds the one invocation there was.
        return BOUNDED_BY_ITS_STARTER
    return INDISTINGUISHABLE


@pytest.fixture(scope="module")
def scanned() -> tuple[dict[str, bool], list[str]]:
    return scan_run_job_call_sites()


class TestTheScanIsComplete:
    def test_every_run_job_call_site_names_a_resolvable_job(self, scanned) -> None:
        _discriminated, unresolved = scanned
        assert not unresolved, (
            f"run_job call sites whose job could not be resolved statically: {unresolved}. "
            "A scanner that silently drops what it cannot parse has unknown coverage, "
            "which is the defect this file exists to remove one level up."
        )

    def test_every_cli_job_has_a_run_job_call_site(self, scanned) -> None:
        """Both directions, derived. A CLI job with no `run_job` call is a job
        that writes no manifest at all — the thing I10967 was originally filed
        about, asserted here as a property rather than looked for by hand."""
        discriminated, _unresolved = scanned
        missing = sorted(set(JOBS) - set(discriminated))
        assert not missing, f"CLI jobs with no run_job call site in the package: {missing}"

    def test_every_scanned_job_is_a_registry_row(self, scanned) -> None:
        discriminated, _unresolved = scanned
        registry = set(load_registry())
        extra = sorted(set(discriminated) - registry)
        assert not extra, (
            f"run_job call sites naming a job with no components.yaml row: {extra}. "
            "A manifest written by a job nobody registered is unobserved, not healthy."
        )


class TestEveryJobIsClassified:
    def test_the_classification_is_total_over_the_registry(self, scanned) -> None:
        """No fall-through. Every row lands in exactly one of three states,
        and the two written outside this repository — the trader's jobs and
        the deploy workflow's — are excluded by the DECLARED tuples, not by a
        list written here."""
        discriminated, _unresolved = scanned
        written_elsewhere = set(TRADER_JOB_VALUES) | set(WORKFLOW_JOB_VALUES)
        states: dict[str, str] = {}
        for name, component in load_registry().items():
            if name in written_elsewhere:
                continue
            states[name] = classify(name, component.dispatch, discriminated.get(name, False))
        assert set(states) == set(JOBS)
        assert set(states.values()) <= {
            DISCRIMINATED,
            BOUNDED_BY_ITS_STARTER,
            INDISTINGUISHABLE,
        }

    def test_a_clock_started_row_is_bounded_by_a_deadline_that_grades_it(self, scanned) -> None:
        """What makes `bounded_by_its_starter` a real bound and not an
        assumption: a row with a `dispatch` fires once per trading day AND
        carries a deadline, so the single key's absence is graded. A row that
        claimed the bound without the deadline would be relying on a cadence
        nothing watches."""
        registry = load_registry()
        discriminated, _unresolved = scanned
        for name, component in registry.items():
            if classify(name, component.dispatch, discriminated.get(name, False)) != (
                BOUNDED_BY_ITS_STARTER
            ):
                continue
            assert component.deadline is not None, (
                f"{name} relies on its starter to bound it to one invocation per "
                "trading day, and declares no deadline — so nothing grades whether "
                "that one invocation happened"
            )


class TestTheIndistinguishableSetIsMeasuredNotHidden:
    """The detector's domain is the whole registry, proven end to end.

    This is the assertion that makes the finding a measurement rather than a
    note: for every job the static classification calls
    `indistinguishable`, a store seeded with that job's undiscriminated
    manifest produces a finding naming that exact key. A detector built
    around an enumerated list would pass the static half and fail here the
    moment the registry grew.
    """

    def test_the_sweep_names_every_indistinguishable_job(self, scanned, tmp_path) -> None:
        discriminated, _unresolved = scanned
        registry = load_registry()
        day = dt.date(2026, 9, 11)
        expected = {
            name
            for name, component in registry.items()
            if name not in set(TRADER_JOB_VALUES) | set(WORKFLOW_JOB_VALUES)
            and classify(name, component.dispatch, discriminated.get(name, False))
            == INDISTINGUISHABLE
        }
        assert expected, (
            "no job classifies as indistinguishable — this guard has stopped "
            "measuring anything, and the assertion below would be vacuous"
        )
        store = LocalStore(tmp_path)
        for name in expected:
            store.put_bytes(
                manifest_key(name, day.isoformat()),
                json.dumps({"status": "ok", "run_id": f"01{name}"}).encode(),
            )
        findings = indistinguishable_invocation_findings(store, days=[day], registry=registry)
        assert {f.job for f in findings} == expected

    def test_a_discriminated_manifest_produces_no_finding(self, tmp_path) -> None:
        """The other direction, so the detector cannot be a count of
        everything: a job that DOES distinguish its invocations is not a
        finding, even though it is on-demand."""
        registry = load_registry()
        day = dt.date(2026, 9, 11)
        on_demand = on_demand_graded_jobs(registry)
        assert on_demand
        store = LocalStore(tmp_path)
        for name in on_demand:
            store.put_bytes(
                manifest_key(name, day.isoformat(), discriminator="r-01ABC"),
                json.dumps({"status": "ok", "run_id": "01ABC"}).encode(),
            )
        assert indistinguishable_invocation_findings(store, days=[day], registry=registry) == []

    def test_a_clock_started_row_is_outside_the_domain(self, tmp_path) -> None:
        """`data.daily` writes one undiscriminated manifest per trading day
        and that is correct: its cron fires it once, and its deadline grades
        whether it did. Grading it here would page for the normal case."""
        registry = load_registry()
        day = dt.date(2026, 9, 11)
        store = LocalStore(tmp_path)
        store.put_bytes(
            manifest_key("data.daily", day.isoformat()),
            json.dumps({"status": "ok", "run_id": "01ABC"}).encode(),
        )
        assert indistinguishable_invocation_findings(store, days=[day], registry=registry) == []

    def test_an_empty_store_reads_as_a_zero_not_as_a_silence(self, tmp_path) -> None:
        """A zero is a reading; a missing metric is not. The metric is emitted
        whether or not there are findings, and its `n_samples` is the number
        of rows really graded."""
        registry = load_registry()
        graded = on_demand_graded_jobs(registry)
        metric = indistinguishable_invocation_metric(
            [], graded_jobs=graded, now=dt.datetime(2026, 9, 17, tzinfo=dt.UTC)
        )
        assert metric["value"] == 0.0
        assert metric["status"] == "OK"
        assert metric["n_samples"] == len(graded)

    def test_a_finding_is_unmeasurable_never_ok(self, tmp_path) -> None:
        """`unmeasurable` is its own state and is never folded into green. A
        key N runs overwrite does not say something went wrong; it says the
        record cannot answer the question."""
        registry = load_registry()
        day = dt.date(2026, 9, 11)
        store = LocalStore(tmp_path)
        store.put_bytes(
            manifest_key("experiment.new", day.isoformat()),
            json.dumps({"status": "ok", "run_id": "01SEVENTH"}).encode(),
        )
        findings = indistinguishable_invocation_findings(store, days=[day], registry=registry)
        assert [f.job for f in findings] == ["experiment.new"]
        metric = indistinguishable_invocation_metric(
            findings,
            graded_jobs=on_demand_graded_jobs(registry),
            now=dt.datetime(2026, 9, 17, tzinfo=dt.UTC),
        )
        assert metric["status"] == "unmeasurable"
        assert metric["status"] != "OK"
        assert "01SEVENTH" in metric["status_reason"]


class TestTheScannerItselfIsShownWorking:
    """`crucible/AGENTS.md` test discipline: a detector nobody has made fail is
    a detector nobody knows works."""

    def test_the_scanner_sees_a_call_site_with_no_discriminator(self) -> None:
        tree = ast.parse('JOB = "x.y"\n\ndef f(ctx):\n    run_job(JOB, body, store=s)\n')
        [(call, has_discriminator)] = _call_sites(tree)
        assert _job_name(call, tree) == "x.y"
        assert has_discriminator is False
        assert classify("x.y", None, has_discriminator) == INDISTINGUISHABLE

    def test_the_scanner_sees_a_call_site_with_one(self) -> None:
        tree = ast.parse('def f(ctx):\n    run_job("x.y", body, discriminator=ctx.slot)\n')
        [(call, has_discriminator)] = _call_sites(tree)
        assert _job_name(call, tree) == "x.y"
        assert has_discriminator is True
        assert classify("x.y", None, has_discriminator) == DISCRIMINATED

    def test_a_clock_started_row_classifies_as_bounded(self) -> None:
        assert classify("data.daily", "scheduler", False) == BOUNDED_BY_ITS_STARTER
        assert classify("promote", "arc", False) == BOUNDED_BY_ITS_STARTER

    def test_an_unresolvable_job_is_reported_not_dropped(self) -> None:
        tree = ast.parse("def f(ctx):\n    run_job(pick(), body)\n")
        [(call, _has)] = _call_sites(tree)
        assert _job_name(call, tree) is None
