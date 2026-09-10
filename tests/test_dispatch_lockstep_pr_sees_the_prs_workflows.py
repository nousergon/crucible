"""`crucible-dispatch-lockstep-pr` must grade the PR's workflows, not the
base ref's — and must refuse to grade an empty set.

**The defect this pins, measured 2026-09-10 (`alpha-engine-config-I10467`).**
The job reads `components.yaml` from the PR head as data and, until this
change, took every other file from the BASE ref checkout that
`pull_request_target` gives it. The guard's whole job is to compare
`components.yaml` against the workflows declaring an `on.schedule` cron — so
it was comparing one side from the PR against the other side from `main`.

A PR whose entire content is a workflow fix could therefore never turn the
check green. `crucible-PR198` removed the exact cron the guard was
complaining about, and the check failed anyway, naming the removed cron. A
required check that a correct fix cannot satisfy is worse than no check: it
teaches people that merging past a red is normal.

**Why a mirror rather than a checkout.** `pull_request_target` combined with
`with: ref: github.event.pull_request.head.sha` runs fork-authored code with
base-repo permissions — the footgun `dispatch-lockstep.yml` documents at
length and `test_workflow_triggers.py::
test_the_pull_request_target_job_never_checks_out_pr_code` enforces. So the
workflows arrive the same way `components.yaml` already does: DATA over the
read-only Contents API, base64-decoded straight to files, nothing executed.
This job's own definition was loaded before any step ran, so replacing files
under `.github/workflows/` in the workspace changes what the TEST reads and
cannot change what runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "dispatch-lockstep.yml"


@pytest.fixture(scope="module")
def pr_job_steps() -> list[dict]:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"]["crucible-dispatch-lockstep-pr"]["steps"]


def _step(steps: list[dict], needle: str) -> dict:
    for step in steps:
        if needle in (step.get("name") or ""):
            return step
    raise AssertionError(f"no step named like {needle!r} in the PR job")


def test_the_pr_job_mirrors_the_prs_workflows(pr_job_steps: list[dict]) -> None:
    step = _step(pr_job_steps, "Mirror the PR's .github/workflows")
    body = step["run"]
    assert "contents/.github/workflows?ref=${HEAD_SHA}" in body
    assert "base64 -d" in body


def test_it_reads_them_at_the_pr_head_not_the_base_ref(pr_job_steps: list[dict]) -> None:
    """The whole point. Grading one side from the PR and the other from
    `main` is what made the check unsatisfiable.
    """
    step = _step(pr_job_steps, "Mirror the PR's .github/workflows")
    assert step["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"


def test_it_refuses_an_empty_listing_rather_than_mirroring_nothing(
    pr_job_steps: list[dict],
) -> None:
    """The direction that fails OPEN, and so the one worth pinning. An API
    hiccup returning zero files would delete every workflow from the
    workspace and leave the guard grading an empty set — which reads GREEN.
    A guard that passes over no data is the failure this file exists to
    prevent.
    """
    body = _step(pr_job_steps, "Mirror the PR's .github/workflows")["run"]
    assert "! -s /tmp/pr-workflows.txt" in body
    assert "refusing to mirror an empty set" in body
    assert "exit 1" in body


def test_it_mirrors_deletions_too(pr_job_steps: list[dict]) -> None:
    """The other half of the same blindness: a PR that DELETES a cron
    workflow would still fail over the base ref's copy of it.
    """
    body = _step(pr_job_steps, "Mirror the PR's .github/workflows")["run"]
    assert "rm -f" in body
    assert "grep -qxF" in body


def test_the_mirror_never_becomes_a_checkout_of_pr_code(pr_job_steps: list[dict]) -> None:
    """Restating the constraint the mirror had to satisfy, at the step that
    could most easily violate it. `test_workflow_triggers.py` asserts the
    CHECKOUT never gains a `ref:`; this asserts the mirror never becomes one.
    """
    step = _step(pr_job_steps, "Mirror the PR's .github/workflows")
    assert "uses" not in step, "the mirror must stay a `run:` over the Contents API"
    body = step["run"]
    for forbidden in ("actions/checkout", "git clone", "git fetch"):
        assert forbidden not in body
    # And nothing decoded is executed.
    for forbidden in ("| bash", "| sh", "source ", "eval "):
        assert forbidden not in body
