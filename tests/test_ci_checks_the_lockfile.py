"""CI verifies that `uv.lock` agrees with `pyproject.toml`, and keeps doing so.

The step this guards is one line in `ci.yml`. Its absence is invisible: every
job would still install, every test would still pass, and the lockfile would
silently stop being a description of what `pyproject.toml` asks for. So the
step's PRESENCE is asserted here rather than trusted.

**What happened, measured 2026-09-02.** Seven Dependabot PRs (#31–#37) raised
dependency floors in `pyproject.toml` — `numpy>=2.1` → `>=2.5.2`,
`pyarrow>=14.0` → `>=25.0.1`, plus `jsonschema`, `PyYAML` and `ruff` — and not
one of them regenerated `uv.lock`. All seven merged with fully green CI, and
`main` then carried a lockfile whose `[package.metadata] requires-dist` still
named the old floors.

`uv sync --frozen` does not catch it. Verified against `main` at `44d00a5`: it
exits 0. `--frozen` installs the *locked* versions without re-checking them
against `pyproject.toml`, so the drift is invisible to the one command every
job in this repository runs. `uv lock --check` is the check that does see it —
exit 1 against the stale lockfile, exit 0 once regenerated, both measured.

**Why this is a workflow assertion and not a pytest comparison.** A pytest
guard was written first and discarded. `uv run pytest` implicitly re-locks
before running, so that guard would have rewritten the very file it was
grading and then reported green — a guard that reads as coverage while being
blind, which is exactly the class this repository keeps finding elsewhere.
The lesson generalises: a check on a file that the test runner itself
regenerates cannot live inside that runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

#: The exact command, pinned rather than pattern-matched. A predicate over a
#: shell string is a partial shell parser, and a partial shell parser is a
#: denylist of the syntax someone thought of — the structure `AGENTS.md`
#: rule 4 forbids, and the one that cost five adversarial review rounds on
#: `crucible-PR38`. If the command legitimately changes, this literal changes
#: with it in the same diff.
LOCK_CHECK_COMMAND = "uv lock --check"

#: The install command every job runs, and the reason the check above is
#: needed at all. Asserted so that a future edit swapping `--frozen` for a
#: re-resolve is visible: a CI run that resolves its own dependencies has not
#: tested what ships.
FROZEN_INSTALL_COMMAND = "uv sync --frozen"


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    document = yaml.safe_load(CI_WORKFLOW.read_text())
    assert isinstance(document, dict), f"{CI_WORKFLOW} did not parse to a mapping"
    return document


def _steps(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every step in every job, tagged with its job name.

    Flattened rather than read from one named job on purpose: naming the job
    would make this guard blind the moment the step moves to another job,
    which is a legitimate refactor that must not silently remove the check.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                found.append((job_name, step))
    return found


def test_the_parse_found_steps_at_all(workflow) -> None:
    """Fails CLOSED.

    If the workflow shape changes and `_steps` returns nothing, every
    assertion below would pass over an empty list — green while checking
    nothing.
    """
    assert len(_steps(workflow)) >= 5


def test_ci_runs_uv_lock_check(workflow) -> None:
    running = [
        (job, step)
        for job, step in _steps(workflow)
        if LOCK_CHECK_COMMAND in (step.get("run") or "")
    ]
    assert running, (
        f"no step in ci.yml runs `{LOCK_CHECK_COMMAND}`. Without it, a dependency "
        "floor raised in pyproject.toml with no `uv lock` merges green and the "
        "lockfile quietly stops describing what pyproject.toml asks for — measured "
        "on this repository across seven Dependabot merges on 2026-09-02."
    )


def test_every_job_that_installs_also_checks_the_lockfile_first(workflow) -> None:
    """Per job, and it FAILS rather than skipping when the check is absent.

    The earlier version `continue`d when a job had an install and no check —
    which made the live state pass: `uv lock --check` was in the `test` job
    only, while the `acceptance` job (the phase gate, on `push: [main]`)
    installed from an unverified lockfile. A guard that skips the case it
    exists to catch is the vacuous-guard shape this repository keeps finding
    elsewhere.

    Repeated per job rather than inherited on purpose: GitHub jobs share
    nothing, so "the other job checked it" is not a property of this runner.
    """
    for job_name, job in (workflow.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        steps = [s for s in (job.get("steps") or []) if isinstance(s, dict)]
        commands = [s.get("run") or "" for s in steps]
        install_at = next((i for i, c in enumerate(commands) if FROZEN_INSTALL_COMMAND in c), None)
        if install_at is None:
            continue
        check_at = next((i for i, c in enumerate(commands) if LOCK_CHECK_COMMAND in c), None)
        assert check_at is not None, (
            f"job {job_name!r} installs from the lockfile and never runs "
            f"`{LOCK_CHECK_COMMAND}`. It would install from a lockfile nothing verified "
            "against pyproject.toml."
        )
        assert check_at < install_at, (
            f"in job {job_name!r}, `{LOCK_CHECK_COMMAND}` runs at step {check_at} and "
            f"`{FROZEN_INSTALL_COMMAND}` at step {install_at}. The check must come "
            "first, or it grades a lockfile the job has already installed from."
        )


def test_every_install_is_frozen(workflow) -> None:
    """No job may re-resolve at install time.

    The lockfile is the supply chain. A job that resolves its own
    dependencies has not tested what ships, and a bare `uv sync` would
    additionally REWRITE the lockfile in the runner — which would let CI
    quietly repair the very drift `uv lock --check` had just refused.
    """
    for job_name, step in _steps(workflow):
        command = step.get("run") or ""
        if "uv sync" not in command:
            continue
        assert FROZEN_INSTALL_COMMAND in command, (
            f"job {job_name!r} runs `uv sync` without `--frozen`: {command.strip()!r}"
        )


def test_every_uv_run_is_frozen(workflow) -> None:
    """`uv run` re-locks too, and that is the harder half.

    Measured 2026-09-02: rolling `uv.lock` back to `main`'s stale version and
    running `uv run pytest` changed the file's md5 and made a subsequent
    `uv lock --check` exit 0. So an unfrozen `uv run` anywhere after the
    install silently re-resolves against a different set than the one
    `uv sync --frozen` installed, and repairs the drift in the runner while it
    is at it.

    The install-side assertion above does not cover this: its predicate is
    `uv sync`, so every `uv run` step was uncovered. Both halves of "a CI run
    that resolves its own dependencies has not tested what ships" are now
    enforced.
    """
    for job_name, step in _steps(workflow):
        command = step.get("run") or ""
        if "uv run" not in command:
            continue
        for line in command.splitlines():
            if "uv run" not in line:
                continue
            assert "uv run --frozen" in line, (
                f"job {job_name!r} runs `uv run` without `--frozen`: {line.strip()!r}. "
                "It would re-resolve, and rewrite uv.lock in the runner."
            )
