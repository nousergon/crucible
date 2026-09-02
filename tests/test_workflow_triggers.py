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

**There is deliberately no second layer here scanning step bodies.** Five
independent adversarial reviews attacked one, and each fix opened the next
round's hole — a `--collect-only` substring licensing the next line, then a
per-command splitter beaten by a single `&` and by a trailing `# comment`,
then a boundary rule that rejected `AWS_ACCESS_KEY_ID`. The class is that any
predicate over a `run:` body is a partial shell parser, and a partial shell
parser is a denylist of the syntax someone thought of — the structure
`AGENTS.md` rule 4 forbids.

That scan is defence in depth, not the §3.1 requirement, and it belongs where
it will be reused: `alpha-engine-config-I9830` lifts this guard into
`nousergon-lib` for the whole fleet, built on a typed workflow model rather
than text. Hardening it once there beats hardening it five times here. The
gap until then is narrower than what this file already closes: a live-AWS
step added INSIDE the already-allowlisted `ci.yml:test` job.

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

import inspect
import json
import pathlib
import re
import shutil
import subprocess
from typing import NamedTuple

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
#: steps for the second layer to scan, mapped to the test that argues each
#: one's safety. A named-member exemption set with no forcing function is the
#: shape AGENTS.md rule 4 forbids, so membership REQUIRES the named test to
#: exist — `test_every_reusable_workflow_exemption_has_a_property_test`
#: enforces it, and a second member cannot be added by editing this set alone.
REUSABLE_WORKFLOW_JOBS: dict[str, str] = {
    # Pinned by SHA to a nousergon-lib workflow that posts a notification and
    # nothing else, and unreachable on a PR because it needs the excluded
    # `acceptance` job.
    "ci.yml:notify-main-failure": "test_the_notify_job_still_depends_on_the_excluded_job",
}


class Job(BaseModel):
    """One job, as much of it as this guard reasons about."""

    model_config = ConfigDict(extra="allow")

    condition: str = Field(default="", alias="if")
    needs: list[str] = Field(default_factory=list)
    uses: str = ""
    steps: list[dict] = Field(default_factory=list)

    @field_validator("needs", mode="before")
    @classmethod
    def _needs_accepts_the_scalar_form(cls, value: object) -> object:
        """`needs: build` is legal GitHub syntax, as is `needs: [build]`.

        Modelling only the list form makes a legal edit error the whole guard
        rather than evaluate it. That fails closed, so it is not a hole — but
        a guard that refuses to run is a guard nobody keeps, which is how the
        rule it enforces gets removed instead of fixed.
        """
        return [value] if isinstance(value, str) else value


class Workflow(BaseModel):
    """A GitHub Actions workflow, parsed rather than indexed.

    The `on:` key is why this is a model and not `dict.get`. PyYAML resolves a
    bare `on` to the BOOLEAN True under YAML 1.1, so `workflow["on"]` finds
    nothing, every workflow reads as having no triggers, and the guard passes
    on all of them — dark, which is the exact failure this file exists to
    prevent. An alias states that once; the alternative is remembering it at
    every call site, which is how it gets forgotten.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    triggers: dict | list | str = Field(default_factory=dict)
    jobs: dict[str, Job] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalise_the_on_key(cls, data: object) -> object:
        """`on:` arrives as the boolean True, or as the string, or not at all.

        A pydantic alias cannot name a non-string key, so the normalisation is
        explicit — and it RAISES when neither form is present, because a
        workflow with no triggers is a document this guard did not understand,
        not a workflow with nothing to check.
        """
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        for key in (True, "on"):
            if key in payload:
                payload["triggers"] = payload.pop(key)
                return payload
        if "triggers" in payload:
            return payload
        raise ValueError("workflow has no `on:` block — unparseable, not trigger-free")

    @property
    def events(self) -> set[str]:
        on = self.triggers
        if isinstance(on, str):
            return {on}
        return set(on)

    @classmethod
    def load(cls, path: pathlib.Path) -> Workflow:
        return cls.model_validate(yaml.safe_load(path.read_text()))


def _excluded(job: Job, pr_events: frozenset[str]) -> bool:
    # Whole-string comparison, not a substring test: `... != 'pull_request' ||
    # <anything>` is a disjunction that runs on PRs, and a substring test waves
    # it through.
    return job.condition.strip() in _exclusions_for(pr_events)


def test_at_least_one_workflow_is_scanned() -> None:
    # A guard that scanned nothing is dark, not green (principle 7).
    assert WORKFLOWS, "no workflows found — this guard is not measuring anything"


def test_every_reusable_workflow_exemption_has_a_property_test() -> None:
    """An exemption set with no forcing function grows by editing one line.

    `REUSABLE_WORKFLOW_JOBS` exempts a job from the step scan entirely, because
    a called workflow declares `workflow_call` and is never reached from here.
    That is the strongest exemption this file grants, so each member names the
    test that argues its safety, and that test must exist.
    """
    named = list(REUSABLE_WORKFLOW_JOBS.values())
    assert len(named) == len(set(named)), (
        "two members of REUSABLE_WORKFLOW_JOBS name the same test. Pointing a "
        "new member at an existing member's test is how the set grows without "
        "anything new being argued."
    )
    for key, test_name in REUSABLE_WORKFLOW_JOBS.items():
        test = globals().get(test_name)
        assert test is not None, (
            f"{key} is exempt from the step scan on the strength of "
            f"`{test_name}`, which does not exist in this module. Write it, or "
            "remove the exemption — an exemption nothing argues is a hole."
        )
        # The test must actually be about this job. Naming any existing test
        # satisfied the earlier version of this check.
        job = key.split(":", 1)[1]
        assert job in inspect.getsource(test), (
            f"`{test_name}` never mentions `{job}`, so it does not argue "
            f"{key}'s safety. An exemption is only as good as the property "
            "someone asserted about the thing exempted."
        )


def test_every_allowlisted_job_still_exists() -> None:
    """An allowlist entry for a deleted job is a hole waiting for a name collision."""
    present = set()
    for path in WORKFLOWS:
        for name in Workflow.load(path).jobs:
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
    job = Workflow.load(WORKFLOW_DIR / "ci.yml").jobs["notify-main-failure"]
    assert "acceptance" in job.needs, (
        "notify-main-failure no longer depends on `acceptance`, which is what "
        "keeps it off the PR path. Re-derive its allowlist entry."
    )
    condition = job.condition
    assert "failure()" in condition, (
        "notify-main-failure no longer runs only on failure; its allowlist entry claims it does."
    )
    # It is in REUSABLE_WORKFLOW_JOBS, so the step scan cannot see what it
    # calls. Pin the target: repointing it at another workflow is exactly the
    # repurposing this entry exists to prevent.
    assert re.fullmatch(
        r"nousergon/nousergon-lib/\.github/workflows/notify-ci-failure\.yml@[0-9a-f]{40}",
        job.uses,
    ), (
        "notify-main-failure must call the nousergon-lib notification workflow "
        "pinned to a 40-character SHA. Its allowlist entry assumes that target, "
        "nothing here can scan a called workflow, and the job carries "
        "`secrets: inherit` — a moving ref like `@main` is a supply-chain hole, "
        "and a prefix check accepted one."
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_live_state_job_runs_on_the_pull_request_path(path: pathlib.Path) -> None:
    workflow = Workflow.load(path)
    pr_events = frozenset(workflow.events & PR_EVENTS)
    if not pr_events:
        return

    for name, job in workflow.jobs.items():
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
            assert not job.uses or key in REUSABLE_WORKFLOW_JOBS, (
                f"{key} is allowlisted but is now a job-level `uses:` calling "
                f"{job.uses!r}. The called workflow declares workflow_call, not "
                "a PR event, so nothing here scans it. Inline the steps, or remove "
                "the allowlist entry and take the job off the PR path."
            )
            assert job.steps or key in REUSABLE_WORKFLOW_JOBS, (
                f"{key} is allowlisted and has no steps. An allowlist entry names "
                "what a job does; a job that does nothing scannable cannot keep one."
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


def test_the_job_model_accepts_both_legal_needs_forms() -> None:
    """`needs: build` and `needs: [build]` are both legal GitHub syntax."""
    assert Job.model_validate({"needs": "build"}).needs == ["build"]
    assert Job.model_validate({"needs": ["build", "test"]}).needs == ["build", "test"]
    assert Job.model_validate({}).needs == []


# the queue-race defect -- round 2. An independent adversarial review of
# crucible-PR41 (round 1) found the concurrency/self-heal rework itself was
# real (each of its four changes was individually reverted and the suite
# went red every time), but flagged what round 1 did NOT close:
#
# 1. `resolve`'s lookup filtered `.status=="in_progress"` -- so a check-run
#    an EARLIER resolve dispatch had already completed was invisible, and a
#    second dispatch (a verdict correction, a re-review, a retried
#    dispatch -- the most ordinary operator action there is) took the
#    self-heal path and POSTed a duplicate. Fixed: select the newest
#    check-run of that name by id (`max_by(.id)`) regardless of status, and
#    PATCH it either way.
# 2. The independence refusal -- the one structural control this entire
#    workflow exists for -- had zero tests. Deleting the refusal block, or
#    adding `if: always()` to the completion step so it writes the verdict
#    AFTER a refusal, both left the suite green. Added below.
# 3. The collision guard's numeric check failed OPEN on a non-numeric read:
#    `[ "$existing" -gt 0 ]` on an empty or malformed value prints a stderr
#    line but does not raise under `set -e` inside an `if` condition, so the
#    guard silently evaluated false and posted a duplicate anyway. Fixed
#    with an explicit numeric-format check that exits 1 on anything else.
# 4. The guard is check-THEN-post, not atomic (no conditional create in the
#    Checks API) -- narrowed, not closed, and the workflow comment now says
#    so explicitly rather than presenting it as an absolute invariant.
# 5. `resolve` never read `.draft` -- dispatching it against a draft let the
#    self-heal post a green check-run before `post-pending-check` ever ran.
#    Fixed: `resolve` now refuses when the PR is still a draft.
# 6. The `gh` stub matched on jq-filter substrings without validating the
#    URL, so it silently answered requests against a WRONG endpoint --
#    meaning the collision guard's `/commits/{sha}/check-runs` endpoint had
#    never actually executed against anything resembling real `gh`
#    validation. The stub below routes on the URL path first and rejects
#    anything unrecognized, the same way it already rejected unhandled
#    verbs.


_FAKE_GH = """#!/usr/bin/env bash
set -euo pipefail
argv="$*"

# Every invocation is logged here unconditionally, mutating or not -- used
# to prove the independence refusal makes NO gh api call at all (round-2
# finding 2), not just no *mutating* one.
printf '%s\0' "$argv" >> "$ALL_CALL_LOG"

if [ "$1" != "api" ]; then
  echo "unhandled fake gh invocation (not 'api'): $argv" >&2
  exit 99
fi
shift
url="$1"
shift

method="GET"
query=""
while [ $# -gt 0 ]; do
  case "$1" in
    -X)
      method="$2"
      shift 2
      ;;
    -q)
      query="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# the queue-race defect round-2 finding 6: mutating calls are routed on
# METHOD + URL PREFIX, not on a substring of the whole argv. A prior version
# of this stub matched on jq-filter text anywhere in argv, which meant a
# call against a wrong or misspelled endpoint could still be "answered" by
# whichever case happened to match its query string -- so the collision
# guard's own endpoint had never actually been validated by this harness.
# Anything that doesn't match a known shape hits the trailing `exit 99` and
# fails the test loudly, the same way an unhandled verb already did.
case "$method" in
  POST)
    case "$url" in
      /repos/*/check-runs)
        printf 'POST %s\0' "$argv" >> "$CALL_LOG"
        if [ "$query" = ".id" ]; then echo "999"; fi
        exit 0
        ;;
      *)
        echo "unhandled fake gh POST endpoint: $url" >&2
        exit 99
        ;;
    esac
    ;;
  PATCH)
    case "$url" in
      /repos/*/check-runs/*)
        printf 'PATCH %s\0' "$argv" >> "$CALL_LOG"
        exit 0
        ;;
      *)
        echo "unhandled fake gh PATCH endpoint: $url" >&2
        exit 99
        ;;
    esac
    ;;
esac

case "$url" in
  /repos/*/pulls/*)
    case "$query" in
      .user.login) echo "gh-author" ;;
      .head.sha) echo "$FAKE_SHA" ;;
      .draft) echo "$FAKE_PR_DRAFT" ;;
      *)
        echo "unhandled fake gh pulls query: $query" >&2
        exit 99
        ;;
    esac
    ;;
  /repos/*/commits/*/check-runs)
    case "$query" in
      *"| length")
        echo "$FAKE_EXISTING_COUNT"
        ;;
      *"max_by(.id).id"*)
        echo "$FAKE_RUN_ID"
        ;;
      *)
        echo "unhandled fake gh check-runs query: $query" >&2
        exit 99
        ;;
    esac
    ;;
  *)
    echo "unhandled fake gh GET endpoint: $url" >&2
    exit 99
    ;;
esac
"""


class _StepResult(NamedTuple):
    result: subprocess.CompletedProcess
    outputs: dict[str, str]
    calls: list[str]
    all_calls: list[str]


def _run_step(
    tmp_path: pathlib.Path,
    script: str,
    step_env: dict[str, str],
    fake_sha: str = "deadbeef",
    fake_run_id: str = "",
    fake_existing_count: str = "0",
    fake_pr_draft: str = "false",
) -> _StepResult:
    """Execute an extracted workflow step's real `run:` script under bash,
    with `gh` stubbed out (endpoint-and-method-validated -- see _FAKE_GH),
    and return the process result, GITHUB_OUTPUT as a dict, the list of gh
    invocations that mutated state via POST/PATCH, and the list of EVERY gh
    invocation (mutating or not).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(_FAKE_GH)
    fake_gh.chmod(0o755)

    github_output = tmp_path / "github_output"
    github_output.write_text("")
    call_log = tmp_path / "call_log"
    call_log.write_text("")
    all_call_log = tmp_path / "all_call_log"
    all_call_log.write_text("")

    step_process_env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "GITHUB_OUTPUT": str(github_output),
        "CALL_LOG": str(call_log),
        "ALL_CALL_LOG": str(all_call_log),
        "FAKE_SHA": fake_sha,
        "FAKE_RUN_ID": fake_run_id,
        "FAKE_EXISTING_COUNT": fake_existing_count,
        "FAKE_PR_DRAFT": fake_pr_draft,
        "GH_TOKEN": "fake-token",
        "REPO": "nousergon/crucible",
    }
    step_process_env.update(step_env)

    result = subprocess.run(
        ["bash", "-c", script],
        env=step_process_env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    output_lines = [line for line in github_output.read_text().splitlines() if "=" in line]
    outputs = dict(line.split("=", 1) for line in output_lines)
    # NUL-separated, not newline-separated -- see _FAKE_GH's comment on why.
    calls = [call for call in call_log.read_text().split("\0") if call]
    all_calls = [call for call in all_call_log.read_text().split("\0") if call]
    return _StepResult(result, outputs, calls, all_calls)


def test_post_pending_check_has_no_job_level_concurrency() -> None:
    """A job-level `concurrency:` block is void in GitHub Actions -- the
    workflow-level group and `cancel-in-progress` apply to the whole run,
    and a job cannot opt itself out. Asserting one here would assert a
    no-op; the real defence is the workflow-level group's shape, checked by
    `test_workflow_level_concurrency_group_separates_resolve_from_pull_request_runs`.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    job = workflow.jobs["post-pending-check"]
    assert "concurrency" not in (job.model_extra or {}), (
        "post-pending-check carries a job-level `concurrency:` block. "
        "GitHub Actions has no such per-job override -- it is silently "
        "void -- so this either does nothing (misleading) or, if GitHub "
        "ever rejects it, breaks the workflow outright. The fix belongs at "
        "the workflow level."
    )


def test_workflow_level_concurrency_group_separates_resolve_from_pull_request_runs() -> None:
    """The workflow-level group must key `resolve` (workflow_dispatch)
    separately from `pull_request` runs of the same PR, so a push cannot
    cancel an in-flight verdict-recording dispatch -- and must key
    `pull_request` runs on the head SHA, so two different pushes to the same
    PR cannot cancel each other's `post-pending-check` run. Neither
    collision has an observed instance (an independent review measured,
    against the Actions API, that no run in this repo has ever been
    cancelled) -- this is defence in depth, not the I9848 root-cause fix.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    group = workflow.model_extra["concurrency"]["group"]
    assert "workflow_dispatch" in group and "resolve" in group, (
        "the workflow-level concurrency group no longer branches on "
        "`workflow_dispatch` to give `resolve` its own key -- a push to the "
        "PR can cancel an in-flight resolve dispatch again."
    )
    assert "pull_request.head.sha" in group, (
        "the workflow-level concurrency group is no longer keyed on the "
        "pull_request head sha -- two different pushes to the same PR "
        "number can collide again."
    )


def test_resolve_dispatch_group_never_cancels_a_verdict_write() -> None:
    """Round-2 finding 7: a durable verdict write should queue behind
    another verdict write, never be discarded mid-flight. `cancel-in-
    progress` must be an expression that reads FALSE for a workflow_dispatch
    event, not a flat `true` that applies to `resolve` the same as it does
    to the cheap, re-postable `pull_request` job.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    cancel_in_progress = workflow.model_extra["concurrency"]["cancel-in-progress"]
    assert isinstance(cancel_in_progress, str) and "workflow_dispatch" in cancel_in_progress, (
        "cancel-in-progress is not an expression keyed on workflow_dispatch -- a "
        f"resolve dispatch can be cancelled by an in-flight push again. Got: {cancel_in_progress!r}"
    )


def test_resolve_self_heals_a_missing_check_run_instead_of_exiting(
    tmp_path: pathlib.Path,
) -> None:
    """Execute the real `resolve` lookup step with a stubbed `gh` reporting
    no check-run on the sha at all (the queue-race condition measured on
    crucible-PR38). The step must exit 0 and record an empty `run_id`
    output -- not fail. Running the actual script (rather than pattern-
    matching its text) means a regression that swaps `exit 1` for `false`
    (both non-zero under `set -e`, and both were shown to defeat an earlier
    text-matching version of this test) is still caught, because the
    subprocess genuinely exits non-zero either way.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    resolve = workflow.jobs["resolve"]
    lookup_script = resolve.steps[0]["run"]

    step = _run_step(
        tmp_path,
        lookup_script,
        step_env={
            "PR_NUM": "41",
            "REVIEWER": "session-a",
            "AUTHOR": "session-b",
        },
        fake_sha="7ac3cb6",
        fake_run_id="",
    )
    assert step.result.returncode == 0, (
        "the check-run lookup step exited non-zero when no check-run was "
        "found on the sha -- resolve fails hard again instead of self-healing. "
        f"stderr:\n{step.result.stderr}"
    )
    assert step.outputs.get("run_id", "") == "", (
        f"expected an empty run_id output when no check-run exists; got outputs={step.outputs!r}"
    )
    assert step.outputs.get("sha") == "7ac3cb6", (
        f"expected the resolved sha to be forwarded as an output; got outputs={step.outputs!r}"
    )


def test_resolve_lookup_selects_the_newest_check_run_regardless_of_status() -> None:
    """Round-2 finding 1: the original filter restricted the lookup to
    `.status=="in_progress"`, so a check-run an EARLIER resolve dispatch had
    already completed was invisible -- and a second dispatch (a verdict
    correction, a re-review, a retried dispatch) took the self-heal path
    and POSTed a duplicate check-run of the same name, which is exactly what
    the restated closes-when forbids.

    This runs the ACTUAL jq filter embedded in the workflow (extracted, not
    retyped) through real `jq`, against a fixture with an OLDER in_progress
    run and a NEWER completed one -- the precise ordering that broke. A
    predicate over the filter's *text* (e.g. asserting the substring
    "in_progress" is absent) would not catch a filter that still excludes
    completed runs some other way; running the filter is the only way to
    know what it actually selects.
    """
    # This repo forbids suppression collections (plan §11.1,
    # tests/test_no_suppressions.py) -- a silent test-skip is one of the
    # forbidden shapes, so a missing `jq` fails this test loudly rather
    # than passing over it quietly. `jq` is present on `ubuntu-latest`
    # runners and expected on any dev machine working this repo.
    assert shutil.which("jq") is not None, (
        "`jq` is required to run the real embedded filter in this test -- "
        "install it rather than skip the property it verifies."
    )

    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    lookup_script = workflow.jobs["resolve"].steps[0]["run"]
    match = re.search(r'/check-runs"\s+-q\s+\'([^\']*)\'', lookup_script)
    assert match is not None, (
        "expected the check-runs lookup's `-q '...'` filter to be "
        "extractable from the lookup step's script -- its shape changed; "
        "update this extraction and re-verify the property by hand."
    )
    jq_filter = match.group(1)

    older_in_progress_newer_completed = {
        "check_runs": [
            {"id": 100, "name": "adversarial-review-gate", "status": "in_progress"},
            {"id": 200, "name": "adversarial-review-gate", "status": "completed"},
            {"id": 999, "name": "some-other-check", "status": "completed"},
        ]
    }
    result = subprocess.run(
        ["jq", jq_filter],
        input=json.dumps(older_in_progress_newer_completed),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"jq failed on the embedded filter: {result.stderr}"
    assert result.stdout.strip() == "200", (
        "expected the filter to select the NEWEST run (id 200, already "
        f"completed) regardless of status; got {result.stdout!r} -- a filter "
        "that still prefers in_progress, or picks the lowest id, reintroduces "
        "round-2 finding 1."
    )

    no_matching_check_runs = {"check_runs": [{"id": 1, "name": "some-other-check"}]}
    empty_result = subprocess.run(
        ["jq", jq_filter],
        input=json.dumps(no_matching_check_runs),
        capture_output=True,
        text=True,
    )
    assert empty_result.returncode == 0, f"jq failed on the empty case: {empty_result.stderr}"
    assert empty_result.stdout.strip() == "", (
        "expected no output (empty) when no check-run of this name exists -- "
        f"got {empty_result.stdout!r}, which would break the self-heal branch's -z test"
    )


def test_resolve_completion_step_creates_a_completed_check_run_when_run_id_is_empty(
    tmp_path: pathlib.Path,
) -> None:
    """With no existing check-run to PATCH (`run_id` empty, the self-heal
    condition), the completion step must POST a new, already-`completed`
    check-run rather than doing nothing or failing. Verified by executing
    the real script and inspecting the actual `gh api` call it made, not by
    matching its text.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    resolve = workflow.jobs["resolve"]
    complete_script = resolve.steps[1]["run"]

    step = _run_step(
        tmp_path,
        complete_script,
        step_env={
            "RUN_ID": "",
            "SHA": "7ac3cb6",
            "GH_AUTHOR": "cipher813",
            "REVIEWER": "session-a",
            "AUTHOR": "session-b",
            "CONCLUSION": "success",
            "SUMMARY": "no findings",
        },
    )
    assert step.result.returncode == 0, f"self-heal completion step failed: {step.result.stderr}"
    assert len(step.calls) == 1, (
        f"expected exactly one mutating gh api call (the self-heal POST); got {step.calls!r}"
    )
    (call,) = step.calls
    assert call.startswith("POST"), f"expected a POST (create), got: {call!r}"
    assert "-f status=completed" in call, (
        f"the self-heal call must create the check-run already completed, not pending: {call!r}"
    )
    assert "-f head_sha=7ac3cb6" in call, f"the self-heal call is not scoped to the sha: {call!r}"


def test_resolve_completion_step_patches_the_existing_run_when_run_id_is_present(
    tmp_path: pathlib.Path,
) -> None:
    """The normal path -- a check-run already exists -- must still PATCH it
    rather than creating a second one, whether that run is `in_progress`
    (the usual case) or already `completed` (round-2 finding 1: a
    re-dispatch or verdict correction against a run a prior dispatch
    already resolved)."""
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    resolve = workflow.jobs["resolve"]
    complete_script = resolve.steps[1]["run"]

    step = _run_step(
        tmp_path,
        complete_script,
        step_env={
            "RUN_ID": "123456",
            "SHA": "7ac3cb6",
            "GH_AUTHOR": "cipher813",
            "REVIEWER": "session-a",
            "AUTHOR": "session-b",
            "CONCLUSION": "success",
            "SUMMARY": "no findings",
        },
    )
    assert step.result.returncode == 0, f"the PATCH-existing-run step failed: {step.result.stderr}"
    assert len(step.calls) == 1, f"expected exactly one gh api call; got {step.calls!r}"
    (call,) = step.calls
    assert call.startswith("PATCH"), f"expected a PATCH of the existing run, got: {call!r}"
    assert "check-runs/123456" in call, f"the PATCH did not target the existing run id: {call!r}"


def test_post_pending_check_skips_posting_when_a_check_run_already_exists(
    tmp_path: pathlib.Path,
) -> None:
    """If `resolve` self-healed while this job was still queued (the
    measured PR38 ordering -- `resolve` dispatched ~3 minutes before this
    job started), a check-run named `adversarial-review-gate` already
    exists on the sha when this job finally runs. It must NOT post a
    second, `in_progress` one -- that would become the current reading and
    block the PR forever. Verified by executing the real step script with a
    stubbed `gh` reporting an existing check-run, and asserting no POST
    happened.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    post_pending = workflow.jobs["post-pending-check"]
    script = post_pending.steps[0]["run"]

    step = _run_step(
        tmp_path,
        script,
        step_env={
            "PR_NUM": "41",
            "SHA": "7ac3cb6",
            "IS_DRAFT": "false",
        },
        fake_existing_count="1",
    )
    assert step.result.returncode == 0, f"expected a clean skip, got: {step.result.stderr}"
    assert step.calls == [], (
        "post-pending-check posted a check-run even though one already "
        f"existed on the sha. calls={step.calls!r}"
    )


def test_post_pending_check_posts_when_no_check_run_exists_yet(
    tmp_path: pathlib.Path,
) -> None:
    """The normal path -- nothing has self-healed yet -- must still post
    the pending check-run. Guards against a collision fix that
    over-corrects into never posting.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    post_pending = workflow.jobs["post-pending-check"]
    script = post_pending.steps[0]["run"]

    step = _run_step(
        tmp_path,
        script,
        step_env={
            "PR_NUM": "41",
            "SHA": "7ac3cb6",
            "IS_DRAFT": "false",
        },
        fake_existing_count="0",
    )
    assert step.result.returncode == 0, f"expected a clean post, got: {step.result.stderr}"
    assert len(step.calls) == 1, (
        f"expected exactly one gh api call (the pending POST); got {step.calls!r}"
    )
    (call,) = step.calls
    assert call.startswith("POST"), f"expected a POST, got: {call!r}"
    assert "-f status=in_progress" in call, (
        f"post-pending-check must still post the check-run as in_progress: {call!r}"
    )


def test_post_pending_check_skips_entirely_while_the_pr_is_a_draft(
    tmp_path: pathlib.Path,
) -> None:
    """Unrelated to the collision fix, but a regression here would be silent
    without a real execution test: a draft PR must not get any check-run at
    all, self-heal or otherwise.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    post_pending = workflow.jobs["post-pending-check"]
    script = post_pending.steps[0]["run"]

    step = _run_step(
        tmp_path,
        script,
        step_env={
            "PR_NUM": "41",
            "SHA": "7ac3cb6",
            "IS_DRAFT": "true",
        },
        fake_existing_count="0",
    )
    assert step.result.returncode == 0, f"draft skip should exit 0, got: {step.result.stderr}"
    assert step.calls == [], f"a draft PR must not get any check-run posted; calls={step.calls!r}"


def test_post_pending_check_collision_guard_fails_closed_on_a_non_numeric_read(
    tmp_path: pathlib.Path,
) -> None:
    """Round-2 finding 3: `existing="$(gh api ...)"` followed by
    `[ "$existing" -gt 0 ]` fails OPEN on a malformed read -- bash prints
    "integer expression expected" to stderr, but `set -e` does not fire
    inside an `if` condition, so the guard silently treats the malformed
    value as "no existing check-run" and posts a duplicate. Fed an
    intentionally non-numeric response (simulating a malformed API read),
    the step must now exit non-zero and post NOTHING, per AGENTS.md rule 5
    (fail loud, default RAISE).
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    post_pending = workflow.jobs["post-pending-check"]
    script = post_pending.steps[0]["run"]

    step = _run_step(
        tmp_path,
        script,
        step_env={
            "PR_NUM": "41",
            "SHA": "7ac3cb6",
            "IS_DRAFT": "false",
        },
        fake_existing_count="not-a-number",
    )
    assert step.result.returncode != 0, (
        "a non-numeric existing-check-run count was silently treated as "
        "safe (fail-OPEN) instead of raising -- this is round-2 finding 3, "
        f"and 'set -e' does not fire inside an 'if' condition. stdout={step.result.stdout!r}"
    )
    assert step.calls == [], (
        "a non-numeric read must never result in a POST -- the malformed "
        f"case posted a check-run anyway. calls={step.calls!r}"
    )


def test_resolve_refuses_when_reviewer_and_author_are_the_same_identity(
    tmp_path: pathlib.Path,
) -> None:
    """The one structural control this entire workflow exists for (plan §11
    risk 1) had zero tests before round 2 -- an independent review measured
    that deleting this block, or making the completion step run regardless
    (`if: always()`), both left the whole suite green. Executes the real
    lookup step with an equal reviewer/author pair (case-varied, since the
    comparison is documented as case-insensitive) and asserts it refuses.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    lookup_script = workflow.jobs["resolve"].steps[0]["run"]

    step = _run_step(
        tmp_path,
        lookup_script,
        step_env={
            "PR_NUM": "41",
            "REVIEWER": "Session-Agent-7",
            "AUTHOR": "session-agent-7",
        },
    )
    assert step.result.returncode != 0, (
        "resolve did not refuse a same-identity (case-insensitive) "
        f"reviewer/author pair. stdout={step.result.stdout!r} stderr={step.result.stderr!r}"
    )


def test_resolve_refusal_makes_no_gh_api_call_at_all(tmp_path: pathlib.Path) -> None:
    """The independence refusal must fire before any `gh api` call is made
    -- not merely before a MUTATING one. Checked against the full
    invocation log (`ALL_CALL_LOG`), not just `CALL_LOG` (which records only
    POST/PATCH) -- a version that reads the PR first and refuses second
    would pass a "no mutating call" check while still leaking read access
    to an unauthorized dispatch.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    lookup_script = workflow.jobs["resolve"].steps[0]["run"]

    step = _run_step(
        tmp_path,
        lookup_script,
        step_env={
            "PR_NUM": "41",
            "REVIEWER": "same-session",
            "AUTHOR": "same-session",
        },
    )
    assert step.result.returncode != 0, "expected the refusal to fire (see the sibling test)"
    assert step.all_calls == [], (
        "the independence refusal made at least one gh api call before "
        f"refusing. all_calls={step.all_calls!r}"
    )


def test_resolve_completion_step_has_no_if_override(tmp_path: pathlib.Path) -> None:
    """Round-2 finding 2's second half: adding `if: always()` to the
    completion step makes it run even after the lookup step's refusal
    (`exit 1`), writing the verdict anyway -- GitHub Actions' default is to
    skip a later step once an earlier one in the same job fails, which is
    exactly the mechanism this workflow relies on to make the refusal
    effective; `if: always()` (or `if: success() || failure()`) defeats it.
    This is a structural assertion on the step definition itself (there is
    nothing to execute -- the property is "no override exists"), backed by
    the mutation proof recorded in this PR: adding `if: always()` and
    re-running this test was measured to fail before being reverted.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    complete_step = workflow.jobs["resolve"].steps[1]
    step_if = complete_step.get("if")
    assert step_if is None, (
        "the completion step carries an `if:` override "
        f"({step_if!r}) -- GitHub Actions' default (run only if every "
        "prior step in the job succeeded) is what makes the lookup step's "
        "independence refusal (exit 1) actually block this step from "
        "writing the verdict. An `always()`/`success() || failure()` "
        "override defeats that. remove it."
    )


def test_resolve_refuses_when_the_pr_is_a_draft(tmp_path: pathlib.Path) -> None:
    """Round-2 finding 5: dispatching `resolve` against a draft let the
    self-heal POST a completed, green check-run before `post-pending-check`
    ever ran; marking the PR ready then skipped posting entirely (the
    collision guard sees the self-healed run and backs off), so the PR
    reached ready-for-review with a green required check and no pending
    stage ever posted. `resolve` must refuse while `.draft` is true.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    lookup_script = workflow.jobs["resolve"].steps[0]["run"]

    step = _run_step(
        tmp_path,
        lookup_script,
        step_env={
            "PR_NUM": "41",
            "REVIEWER": "session-a",
            "AUTHOR": "session-b",
        },
        fake_pr_draft="true",
    )
    assert step.result.returncode != 0, (
        f"resolve did not refuse a draft PR. stdout={step.result.stdout!r}"
    )
    assert step.calls == [], (
        f"a draft-PR dispatch must never mutate a check-run; calls={step.calls!r}"
    )


def test_resolve_proceeds_when_the_pr_is_not_a_draft(tmp_path: pathlib.Path) -> None:
    """Guards against a draft-refusal fix that over-corrects into refusing
    every PR."""
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    lookup_script = workflow.jobs["resolve"].steps[0]["run"]

    step = _run_step(
        tmp_path,
        lookup_script,
        step_env={
            "PR_NUM": "41",
            "REVIEWER": "session-a",
            "AUTHOR": "session-b",
        },
        fake_sha="7ac3cb6",
        fake_pr_draft="false",
    )
    assert step.result.returncode == 0, (
        f"a non-draft PR was refused. stdout={step.result.stdout!r} stderr={step.result.stderr!r}"
    )


def test_fake_gh_stub_rejects_an_unrecognized_endpoint(tmp_path: pathlib.Path) -> None:
    """Round-2 finding 6: this harness is only as real as its stub's
    endpoint validation. Runs a script against a bogus endpoint directly and
    asserts the stub itself refuses (exit 99) -- proving the routing added
    in `_FAKE_GH` actually rejects a wrong path rather than answering it via
    a jq-filter-substring coincidence, which is exactly what let the
    collision guard's `/commits/{sha}/check-runs` endpoint go unvalidated
    through round 1.
    """
    bogus_script = (
        'gh api "/repos/$REPO/bogus-endpoint/check-runs" '
        "-q '[.check_runs[] | select(.name==\"adversarial-review-gate\")] | length'"
    )
    step = _run_step(tmp_path, bogus_script, step_env={}, fake_existing_count="0")
    assert step.result.returncode == 99, (
        "the fake gh stub answered a request against an unrecognized "
        f"endpoint instead of rejecting it. stdout={step.result.stdout!r} "
        f"stderr={step.result.stderr!r}"
    )
