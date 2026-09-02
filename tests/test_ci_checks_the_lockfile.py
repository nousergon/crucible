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


def test_the_lock_check_runs_before_anything_installs(workflow) -> None:
    """Order is load-bearing.

    `uv sync --frozen` installs from the lockfile. Checking the lockfile
    afterwards would grade a file the job has already acted on, and any step
    between the two runs against dependencies nobody verified were the
    declared ones.
    """
    for job_name, job in (workflow.get("jobs") or {}).items():
        steps = [s for s in (job.get("steps") or []) if isinstance(s, dict)]
        commands = [s.get("run") or "" for s in steps]
        check_at = next((i for i, c in enumerate(commands) if LOCK_CHECK_COMMAND in c), None)
        install_at = next((i for i, c in enumerate(commands) if FROZEN_INSTALL_COMMAND in c), None)
        if check_at is None or install_at is None:
            continue
        assert check_at < install_at, (
            f"in job {job_name!r}, `{LOCK_CHECK_COMMAND}` runs at step {check_at} and "
            f"`{FROZEN_INSTALL_COMMAND}` at step {install_at}. The check must come "
            "first, or it grades a lockfile the job has already installed from."
        )


def test_every_install_is_frozen(workflow) -> None:
    """No job may re-resolve.

    The lockfile is the supply chain. A job that resolves its own
    dependencies has not tested what ships, and a bare `uv sync` would
    additionally REWRITE the lockfile in the runner — which would make
    `uv lock --check` in a later job pass over a file CI had just fixed for
    itself.
    """
    for job_name, step in _steps(workflow):
        command = step.get("run") or ""
        if "uv sync" not in command:
            continue
        assert FROZEN_INSTALL_COMMAND in command, (
            f"job {job_name!r} runs `uv sync` without `--frozen`: {command.strip()!r}"
        )
