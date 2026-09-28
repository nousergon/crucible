"""`crucible test.integration` — the integration tier's registered CLI job.

Normative source: `alpha-engine-config-I10459`. `.github/workflows/
integration-nightly.yml` used to run `tests/integration` in one step and
write a plain `integration_summary.v1` JSON object via a raw
`store.put_bytes` call in a SECOND step, hand-rolled in the workflow itself
— never a `run_manifest.v2` document, so the summary was not
manifest-or-it-didn't-happen (rule 1) and had no `components.yaml` row: no
declared log location, alert channel, retention or console surface (§9.2).

This module is what replaces both steps with one job: a thin handler that
shells out to `pytest tests/integration` and reports pass/fail through
`crucible.runner.run_job`, writing a real
run manifest under `runs/test.integration/{trading_day}/` like every other
job (one per invocation, `crucible.runner.invocation_discriminator`). It
deliberately does not re-implement anything `tests/integration` already
does — it invokes the suite exactly as `integration-nightly.yml` did, and
reports the process's own exit code.

**Staged after the tier itself, not landed with it** (see
`tests/integration/README.md`, "The summary artifact, and its DELIBERATE
delta from `run.json`"): `crucible.cli.JOBS`/`HANDLERS`, `components.yaml`
and `run_manifest.v2.json`'s closed `job` enum are shared, heavily-trafficked
surfaces, and the PR that introduced the tier was scoped to avoid colliding
with concurrent work on them.

**No schedule, no deadline.** `components.yaml`'s row declares both `null`:
this job stays workflow-triggered by `integration-nightly.yml`'s own
`schedule`/`workflow_call`/`workflow_dispatch` triggers, never a second,
independent EventBridge schedule or spot-dispatched launch — the
tier already has exactly one starter, and giving this job a second would be
two clocks disagreeing about when "nightly" is.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys

from crucible.runner import RunContext

__all__ = ["INTEGRATION_TEST_JOB", "integration_test_body", "integration_test_handler"]

#: The job name registered in `crucible.cli.JOBS`, `crucible/components.yaml`
#: and `crucible.models.JOB_VALUES` — a constant imported by `cli.py` rather
#: than a literal restated at each of those sites, the same shape
#: `crucible.fault_probe.FAULT_PROBE_JOB` uses.
INTEGRATION_TEST_JOB = "test.integration"

#: How many trailing lines of combined stdout/stderr are kept in the
#: manifest `reason` on a failure. A full pytest log can run to thousands of
#: lines; the manifest is not the log's home (`log_location` in
#: `components.yaml` is), and a `reason` this large would make every reader
#: of the manifest pay for the whole transcript to learn "it failed".
_REASON_TAIL_LINES = 40


def integration_test_body(ctx: RunContext, *, pytest_args: tuple[str, ...] = ()) -> None:
    """Shell out to `pytest tests/integration`, report pass/fail.

    Separated from the handler so the whole body is exercisable against a
    :class:`~crucible.runner.RunContext` without a CLI or a store URI — the
    same split `crucible.fault_probe.probe_body` uses.

    Raises on a non-zero exit so `run_job` records `status: failed` with the
    process's own tail as `reason` — never swallows a failing suite as `ok`,
    and never re-implements pytest's own pass/fail judgment.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/integration",
        "--tb=short",
        *pytest_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(combined.strip().splitlines()[-_REASON_TAIL_LINES:])
    ctx.record_metric(
        {
            "name": "tests_integration_exit_code",
            "module": "crucible.integration_summary",
            "metric_type": "gauge",
            "value": float(result.returncode),
            "unit": "count",
            "n_floor": 1,
            "status": "OK" if result.returncode == 0 else "FAIL",
            "status_reason": tail[:2000] or "no output",
            "source_path": "tests/integration",
            "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if result.returncode != 0:
        raise RuntimeError(f"tests/integration failed (exit {result.returncode}):\n{tail}")


def integration_test_handler(args: argparse.Namespace) -> int:
    """`crucible test.integration [--store URI]`.

    Writes nothing but its own manifest and whatever `tests/integration`
    itself writes to the dedicated integration store it is already pointed
    at via `CRUCIBLE_INTEGRATION_*` environment — this job's `--store`
    argument (and its manifest) is separate from that: it is where THIS
    job's own `run.json` lands, mirroring `crucible board`/`crucible drift`.
    """
    from crucible.cli import _resolve_store
    from crucible.runner import invocation_discriminator, run_job

    store = _resolve_store(args)

    def job(ctx: RunContext) -> None:
        integration_test_body(ctx)

    ctx = run_job(
        INTEGRATION_TEST_JOB,
        job,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(getattr(args, "dry_run", False)),
        run_mode=getattr(args, "run_mode", None),
        # One manifest per INVOCATION, not per trading day: this job is on
        # demand, so nothing bounds how often it runs on one day
        # (`alpha-engine-config-I11033`; `crucible.runner.invocation_discriminator`).
        discriminator=invocation_discriminator,
    )
    print(json.dumps({"run_id": ctx.run_id, "job": ctx.job}, indent=2))
    return 0
