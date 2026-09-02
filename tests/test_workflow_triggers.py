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

**The allowlist is the primary control, and it is load-bearing.** The first
version of this guard matched live-state markers (`aws `, `boto3`,
`tests/acceptance`) in each step's `run:` body. Six shapes walked straight past
it, all demonstrated: a job-level `uses:` reusable workflow with no steps to
scan, a marker in `with:` rather than `run:`, a marker behind a called script, a
tab instead of the space in `"aws "`, configure-aws-credentials plus a
`crucible` command reaching S3, and `pull_request_target`. A marker list is a
list of the known-bad that only ever grows — the structure `AGENTS.md` rule 4
and plan §11.1 forbid for suppression collections. So the question is inverted:
every job reachable on a PR event is live-state UNTIL it is named in
`PR_REACHABLE_JOBS` with a reason.

**The marker scan is kept as a second layer INSIDE allowlisted jobs.** The
allowlist says a job may run on a PR; it cannot say that every step someone
adds to that job later still grades the tree. An allowlist alone let a
`run: aws s3 ls` inserted into `ci.yml:test` pass, which the old denylist
caught. Neither layer is sufficient; both are cheap.

**`pull_request_target` is held to the same rule, and needs its own exclusion
string.** On a `pull_request_target` event `github.event_name` is
`'pull_request_target'`, so `github.event_name != 'pull_request'` evaluates
TRUE and the job runs. An earlier version of this file accepted that expression
as an exclusion on any workflow, which meant the guard failed open on the one
trigger it claims to hold most tightly — the one that runs with base-repo
write-scoped secrets. The exclusion must cover every PR event the workflow
actually declares.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"
# GitHub accepts both suffixes; `test_no_suppressions.py` already scans both.
WORKFLOWS = sorted(p for p in WORKFLOW_DIR.glob("*.y*ml") if p.suffix in {".yml", ".yaml"})

# `pull_request_target` runs with base-repo write-scoped secrets, so it is the
# trigger someone reaches for when they want AWS credentials on a PR.
PR_EVENTS = frozenset({"pull_request", "pull_request_target"})

_BOTH = '!contains(fromJSON(\'["pull_request","pull_request_target"]\'), github.event_name)'


def _exclusions_for(events: frozenset[str]) -> set[str]:
    """`if:` expressions that provably keep a job off every PR event present.

    An expression naming one event does not exclude the other, which is the
    hole this function exists to close.
    """
    single = {event: f"github.event_name != '{event}'" for event in sorted(events)}
    accepted = {_BOTH}
    if len(events) == 1:
        accepted.add(next(iter(single.values())))
    else:
        accepted.add(" && ".join(single[event] for event in sorted(events)))
    return accepted | {"${{ " + form + " }}" for form in accepted}


# A step whose `run:` body mentions any of these is reading state that lives
# outside the tree under review. Second layer only: the allowlist below is what
# actually decides reachability. `gh api` is deliberately absent — the
# adversarial-review gate reads the state of the pull request it runs on, which
# IS the subject under review.
#
# Matched as a substring bounded by non-identifier characters, NOT as a whole
# shell word. Whole-word matching was demonstrated to miss `/usr/bin/aws`,
# `$(echo aws)`, and `python -c '__import__("boto3")...'` — every one of which
# reaches live AWS from inside an allowlisted job. `awslogs` still does not
# match, which is the only thing word matching was buying.
LIVE_STATE_MARKERS = ("aws", "boto3", "cloudformation", "tests/acceptance")

# `--collect-only` imports the acceptance modules without executing a clause
# body, which is why it is allowed on the PR path at all. It is the one way
# `tests/acceptance` may be RUN in a step there.
ACCEPTANCE_CARVE_OUT = "--collect-only"

# `--ignore=tests/acceptance` is the opposite of running it, and the foundation
# suite carries it on every PR. Stripped before scanning so an exclusion is not
# read as an execution.
IGNORE_TOKEN = re.compile(r"--ignore=\S+")

#: `<workflow>:<job>` pairs that may run on a PR because their subject is the
#: tree, or the pull request itself. Every entry names why.
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
        "to read as an exclusion, so it is named here instead. The two "
        "properties that make it safe are asserted by "
        "`test_the_notify_job_still_depends_on_the_excluded_job`, because an "
        "allowlist keyed on a job NAME otherwise lets that job be repurposed "
        "into something else entirely."
    ),
}


#: Allowlisted jobs that are legitimately a job-level `uses:` and so have no
#: steps for the second layer to scan. Each needs its safety argued somewhere
#: this file asserts, because a called workflow declares `workflow_call` rather
#: than a PR event and is therefore never reached by the scan below.
REUSABLE_WORKFLOW_JOBS = frozenset(
    {
        # Pinned by SHA to a nousergon-lib workflow that posts a notification
        # and nothing else, and unreachable on a PR because it needs the
        # excluded `acceptance` job — both asserted by
        # `test_the_notify_job_still_depends_on_the_excluded_job`.
        "ci.yml:notify-main-failure",
    }
)


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


def _excluded(job: dict, pr_events: frozenset[str]) -> bool:
    # Whole-string comparison, not a substring test: `... != 'pull_request' ||
    # <anything>` is a disjunction that runs on PRs, and a substring test waves
    # it through.
    return str(job.get("if", "")).strip() in _exclusions_for(pr_events)


def _live_state_steps(job: dict) -> list[str]:
    hits = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        run = IGNORE_TOKEN.sub("", (step.get("run") or "").lower())
        for marker in LIVE_STATE_MARKERS:
            if marker == "tests/acceptance" and ACCEPTANCE_CARVE_OUT in run:
                continue
            for match in re.finditer(re.escape(marker), run):
                before = run[match.start() - 1] if match.start() else " "
                after = run[match.end()] if match.end() < len(run) else " "
                # Bounded by non-identifier characters on both sides, so
                # `/usr/bin/aws`, `$(echo aws)` and `"boto3"` all match while
                # `awslogs` and `bawsic` do not.
                if not (before.isalnum() or before == "_") and not (
                    after.isalnum() or after == "_"
                ):
                    hits.append(f"{step.get('name', '<unnamed step>')}: mentions `{marker}`")
                    break
            else:
                continue
            break
    return hits


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


def test_the_notify_job_still_depends_on_the_excluded_job() -> None:
    """The allowlist is keyed on a job NAME, so the name can be repurposed.

    `ci.yml:notify-main-failure` is allowlisted on the strength of two
    properties: it depends on a job that is excluded from the PR path, and it
    only runs on failure. Rewriting the job body while keeping the name would
    inherit the exemption. Assert the properties, not the name.
    """
    workflow = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text())
    job = workflow["jobs"]["notify-main-failure"]
    assert "acceptance" in (job.get("needs") or []), (
        "notify-main-failure no longer depends on `acceptance`, which is what "
        "keeps it off the PR path. Re-derive its allowlist entry."
    )
    condition = str(job.get("if", ""))
    assert "failure()" in condition, (
        "notify-main-failure no longer runs only on failure; its allowlist entry claims it does."
    )
    # It is in REUSABLE_WORKFLOW_JOBS, so the step scan cannot see what it
    # calls. Pin the target: repointing it at another workflow is exactly the
    # repurposing this entry exists to prevent.
    assert str(job.get("uses", "")).startswith(
        "nousergon/nousergon-lib/.github/workflows/notify-ci-failure.yml@"
    ), (
        "notify-main-failure calls something other than the pinned nousergon-lib "
        "notification workflow; its allowlist entry assumes that target, and "
        "nothing here can scan a called workflow."
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_live_state_job_runs_on_the_pull_request_path(path: pathlib.Path) -> None:
    workflow = yaml.safe_load(path.read_text())
    pr_events = frozenset(_events(workflow) & PR_EVENTS)
    if not pr_events:
        return

    for name, job in (workflow.get("jobs") or {}).items():
        key = f"{path.name}:{name}"
        if _excluded(job, pr_events):
            continue
        if key in PR_REACHABLE_JOBS:
            # A job-level `uses:` has no steps to scan and delegates to a
            # workflow this guard never reaches, because that workflow declares
            # `workflow_call` rather than a PR event. Repointing an allowlisted
            # job at one turned the entire guard green with live AWS running on
            # every PR — demonstrated. An allowlist entry describes what a job
            # DOES, so a job that no longer does its own work forfeits it.
            assert "uses" not in job or key in REUSABLE_WORKFLOW_JOBS, (
                f"{key} is allowlisted but is now a job-level `uses:` calling "
                f"{job['uses']!r}. The called workflow declares workflow_call, not "
                "a PR event, so nothing here scans it. Inline the steps, or remove "
                "the allowlist entry and take the job off the PR path."
            )
            assert job.get("steps") or key in REUSABLE_WORKFLOW_JOBS, (
                f"{key} is allowlisted and has no steps. An allowlist entry names "
                "what a job does; a job that does nothing scannable cannot keep one."
            )
            # Second layer: allowlisted for its subject, still not licensed to
            # reach live AWS in a step someone adds later.
            hits = _live_state_steps(job)
            assert not hits, (
                f"{key} is allowlisted as grading the tree, but a step reads live "
                f"state:\n  " + "\n  ".join(hits) + "\n"
                "Move that step to a job off the PR path — the allowlist entry "
                "covers the job's stated subject, not anything later added to it."
            )
            continue
        pytest.fail(
            f"{key} can run on {sorted(pr_events)} and is not declared as grading "
            "the tree.\nscm-platform-policy.md §3.1 (Brian ruling 2026-08-10): a "
            "check grades the diff; a check whose subject is live infrastructure "
            "does not belong on the pull_request path at all.\n"
            "Either move it to `push: [main]` plus a schedule, give the job one of "
            f"these exact `if:` expressions — {sorted(_exclusions_for(pr_events))} — "
            f"or, if its subject genuinely is the diff, add `{key}` to "
            "PR_REACHABLE_JOBS in this file with the reason."
        )
