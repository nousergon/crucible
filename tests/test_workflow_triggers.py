"""A check grades the diff; a live-state check does not belong on a PR at all.

`scm-platform-policy.md` §3.1, Brian ruling 2026-08-10: a check whose subject is
live infrastructure — or the state of the system at a phase boundary — has two
failure modes on a `pull_request`, and both are wrong there. It reports the
author's own change as drift, or it reports someone else's out-of-band change,
which the author cannot clear. The second survives every fix aimed at the first,
which is why the rule sits at the TRIGGER and not at the comparison.

The fleet holds this with a per-repo guard test (`nousergon-data-PR1293`,
`nous-ergon-ops-PR386`, `crucible-predictor-PR460`). This repo was created after
those landed and did not carry one, so the acceptance gate — which calls
`cloudformation:ListStackResources` — ran on every PR here for the repo's whole
life and made every one of them read red (measured 2026-09-02: seven open
Dependabot PRs, all UNSTABLE on that single check).

**This guard is an ALLOWLIST, and that is load-bearing.** The first version
matched a list of live-state markers (`aws `, `boto3`, `tests/acceptance`) in
each step's `run:` body. Six shapes walked straight past it, all demonstrated:
a job-level `uses:` reusable workflow (no `steps` to scan at all), a marker
passed in `with:` instead of `run:`, a marker behind a called shell script, a
tab instead of the space in `"aws "`, an `aws-actions/configure-aws-credentials`
step followed by a `crucible` command that reaches S3, and `pull_request_target`
(which the trigger check missed entirely, and which is *more* dangerous because
it runs with base-repo secrets).

A marker list is a list of the known-bad that only ever grows — the exact
structure `AGENTS.md` rule 4 and plan §11.1 forbid for suppression collections.
Shipping one as the enforcement mechanism for a policy rule is that defect
wearing a different hat. So the question is inverted: every job reachable on a
PR is live-state UNTIL it is named here as grading the tree. Adding a job means
adding a line to `PR_REACHABLE_JOBS` and saying why, which is a decision someone
makes deliberately rather than a filter someone slips past.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"
# GitHub accepts both suffixes; `test_no_suppressions.py` already scans both.
WORKFLOWS = sorted(p for p in WORKFLOW_DIR.glob("*.y*ml") if p.suffix in {".yml", ".yaml"})

# `pull_request_target` runs with base-repo write-scoped secrets, so it is the
# trigger someone reaches for when they want AWS credentials on a PR. It is
# held to the same rule, not a laxer one.
PR_EVENTS = frozenset({"pull_request", "pull_request_target"})

# The exact `if:` expression that takes a job off the PR path. Compared as a
# whole string, not as a substring: `... != 'pull_request' || <anything>` is a
# disjunction that runs on PRs, and a substring test passes it.
PR_EXCLUSION = "github.event_name != 'pull_request'"

#: `<workflow>:<job>` pairs that may run on a PR because their subject is the
#: tree under review. Every entry names why. Nothing else may run on a PR.
PR_REACHABLE_JOBS: dict[str, str] = {
    "ci.yml:test": (
        "lint, format, the foundation suite, and `--collect-only` over "
        "tests/acceptance. Every step reads the checkout and nothing else; "
        "--collect-only imports the clause modules without executing a body, "
        "so it touches no live AWS."
    ),
    "adversarial-review-gate.yml:post-pending-check": (
        "posts a pending check-run on the PR it is running on. Its subject IS "
        "the pull request, which is the diff under review, and the author "
        "clears it by getting the review done."
    ),
    "adversarial-review-gate.yml:resolve": (
        "workflow_dispatch-only in practice; gated on the dispatch inputs and "
        "records the review verdict for one named PR."
    ),
    "ci.yml:notify-main-failure": (
        "sends a notification; it grades nothing and posts no check. It also "
        "cannot run on a PR: it `needs: [acceptance]`, and that job is excluded "
        "from the PR path, so it is skipped. Its `if:` is a compound "
        "`failure() && ...` expression, which this guard deliberately refuses "
        "to read as an exclusion — a disjunction there would run on PRs and a "
        "substring test would wave it through — so it is named here instead."
    ),
}


def _events(workflow: dict) -> set[str]:
    # PyYAML resolves the bare key `on` to the boolean True (YAML 1.1), so
    # reading workflow["on"] finds nothing and passes on every file — a dark
    # guard, which is the failure mode this file exists to prevent.
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    if isinstance(on, dict):
        return set(on)
    raise AssertionError(f"unparseable `on:` block: {on!r}")


def _excluded_from_pull_request(job: dict) -> bool:
    condition = str(job.get("if", "")).strip()
    # Accept the bare expression and the `${{ ... }}` wrapping of it, and
    # nothing else. A compound condition is not an exclusion.
    return condition in {PR_EXCLUSION, "${{ " + PR_EXCLUSION + " }}"}


def test_at_least_one_workflow_is_scanned() -> None:
    # A guard that scanned nothing is dark, not green (principle 7).
    assert WORKFLOWS, "no workflows found — this guard is not measuring anything"


def test_every_allowlisted_job_still_exists() -> None:
    """An allowlist entry for a deleted job is a hole waiting for a name collision."""
    present = set()
    for path in WORKFLOWS:
        workflow = yaml.safe_load(path.read_text())
        for name in workflow.get("jobs") or {}:
            present.add(f"{path.name}:{name}")
    stale = sorted(set(PR_REACHABLE_JOBS) - present)
    assert not stale, (
        f"PR_REACHABLE_JOBS names jobs that no longer exist: {stale}. "
        "Remove them — a stale allowlist entry silently exempts the next job "
        "that takes the same name."
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_live_state_job_runs_on_the_pull_request_path(path: pathlib.Path) -> None:
    workflow = yaml.safe_load(path.read_text())
    if not (_events(workflow) & PR_EVENTS):
        return

    for name, job in (workflow.get("jobs") or {}).items():
        key = f"{path.name}:{name}"
        if key in PR_REACHABLE_JOBS or _excluded_from_pull_request(job):
            continue
        pytest.fail(
            f"{key} can run on a pull request and is not declared as grading the "
            "tree.\nscm-platform-policy.md §3.1 (Brian ruling 2026-08-10): a check "
            "grades the diff; a check whose subject is live infrastructure does not "
            "belong on the pull_request path at all.\n"
            "Either move it to `push: [main]` plus a schedule, give the job "
            f"`if: {PR_EXCLUSION}` exactly, or — if its subject genuinely is the "
            f"diff — add `{key}` to PR_REACHABLE_JOBS in this file with the reason."
        )
