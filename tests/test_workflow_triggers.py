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
import pathlib
import re
import subprocess

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


# alpha-engine-config-I9848 -- these tests were reworked after an independent
# adversarial review of crucible-PR41 measured (against the Actions API) that
# the workflow's `cancel-in-progress` concurrency group had NEVER cancelled a
# run in this repo; the run named as root cause (33657407462) completed
# successfully. The actual crucible-PR38 failure was a queue race: `resolve`
# was dispatched ~3 minutes before a busy runner started the queued
# `post-pending-check` job. The tests below assert the corrected shape:
#
# 1. `post-pending-check` carries no job-level `concurrency:` (GitHub Actions
#    has no such per-job override -- a job-level block there is silently
#    void, which the earlier version of this file did not know and asserted
#    on anyway); the real defence against cross-run collision lives at the
#    WORKFLOW level, keyed so a `pull_request` run and a `resolve` dispatch
#    can never share a group.
# 2. `resolve` self-heals a missing check-run by EXECUTING the actual step
#    scripts under a stubbed `gh`, not by pattern-matching the shell text --
#    a predicate over a shell command string is a partial shell parser and
#    therefore a denylist, the exact lesson from PR38's own adversarial
#    review rounds (this file's module docstring). Running the real script
#    means a regression that reintroduces a non-zero exit on this path
#    (`exit 1`, or `false` under `set -e` -- both were tried against an
#    earlier, text-matching version of this test and both slipped through)
#    is caught because the subprocess genuinely fails, not because a string
#    is absent.
# 3. `post-pending-check`'s new collision guard (skip posting if a
#    check-run by this name already exists on the sha) is exercised the
#    same way, both branches.


_FAKE_GH = """#!/usr/bin/env bash
set -euo pipefail
argv="$*"
# Mutating calls (-X POST / -X PATCH) are checked FIRST and unconditionally
# logged: a read-only lookup query can legitimately contain the substrings
# "in_progress" or "length" as query text, but a POST that sets
# `status=in_progress` also contains the literal substring "in_progress" in
# ITS argv (`-f status="in_progress"`) — matching that against the read-only
# case below is exactly the mismatch this ordering avoids (measured: an
# earlier version of this stub matched the POST call against the
# in_progress-lookup case and never logged the call at all). Each call is
# logged NUL-separated, not newline-separated — the real `full_summary`
# payload embeds literal newlines, which would otherwise fragment one call
# into several log lines.
case "$argv" in
  *"-X POST"*)
    printf 'POST %s\\0' "$argv" >> "$CALL_LOG"
    if [[ "$argv" == *"-q .id"* ]]; then echo "999"; fi ;;
  *"-X PATCH"*)
    printf 'PATCH %s\\0' "$argv" >> "$CALL_LOG" ;;
  *"/pulls/"*"-q .user.login"*)
    echo "gh-author" ;;
  *"/pulls/"*"-q .head.sha"*)
    echo "$FAKE_SHA" ;;
  *"check-runs"*"in_progress"*)
    echo "$FAKE_RUN_ID" ;;
  *"check-runs"*"| length"*)
    echo "$FAKE_EXISTING_COUNT" ;;
  *)
    echo "unhandled fake gh invocation: $argv" >&2
    exit 99 ;;
esac
"""


def _run_step(
    tmp_path: pathlib.Path,
    script: str,
    step_env: dict[str, str],
    fake_sha: str = "deadbeef",
    fake_run_id: str = "",
    fake_existing_count: str = "0",
) -> tuple[subprocess.CompletedProcess, dict[str, str], list[str]]:
    """Execute an extracted workflow step's real `run:` script under bash,
    with `gh` stubbed out, and return (result, GITHUB_OUTPUT as a dict, the
    list of gh invocations that mutated state via POST/PATCH).
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

    step_process_env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "GITHUB_OUTPUT": str(github_output),
        "CALL_LOG": str(call_log),
        "FAKE_SHA": fake_sha,
        "FAKE_RUN_ID": fake_run_id,
        "FAKE_EXISTING_COUNT": fake_existing_count,
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
    # NUL-separated, not newline-separated — see _FAKE_GH's comment on why.
    calls = [call for call in call_log.read_text().split("\0") if call]
    return result, outputs, calls


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


def test_resolve_self_heals_a_missing_check_run_instead_of_exiting(
    tmp_path: pathlib.Path,
) -> None:
    """Execute the real `resolve` lookup step with a stubbed `gh` reporting
    no in_progress check-run on the sha (the queue-race condition measured
    on crucible-PR38). The step must exit 0 and record an empty `run_id`
    output -- not fail. Running the actual script (rather than pattern-
    matching its text) means a regression that swaps `exit 1` for `false`
    (both non-zero under `set -e`, and both were shown to defeat an earlier
    text-matching version of this test) is still caught, because the
    subprocess genuinely exits non-zero either way.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    resolve = workflow.jobs["resolve"]
    lookup_script = resolve.steps[0]["run"]

    result, outputs, _calls = _run_step(
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
    assert result.returncode == 0, (
        "the check-run lookup step exited non-zero when no in_progress "
        "check-run was found on the sha -- resolve fails hard again instead "
        f"of self-healing. stderr:\n{result.stderr}"
    )
    assert outputs.get("run_id", "") == "", (
        f"expected an empty run_id output when no in_progress check-run "
        f"exists; got outputs={outputs!r}"
    )
    assert outputs.get("sha") == "7ac3cb6", (
        f"expected the resolved sha to be forwarded as an output; got outputs={outputs!r}"
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

    result, _outputs, calls = _run_step(
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
    assert result.returncode == 0, f"self-heal completion step failed: {result.stderr}"
    assert len(calls) == 1, (
        f"expected exactly one mutating gh api call (the self-heal POST); got {calls!r}"
    )
    (call,) = calls
    assert call.startswith("POST"), f"expected a POST (create), got: {call!r}"
    assert "-f status=completed" in call, (
        f"the self-heal call must create the check-run already completed, not pending: {call!r}"
    )
    assert "-f head_sha=7ac3cb6" in call, f"the self-heal call is not scoped to the sha: {call!r}"


def test_resolve_completion_step_patches_the_existing_run_when_run_id_is_present(
    tmp_path: pathlib.Path,
) -> None:
    """The normal path -- a pending check-run exists -- must still PATCH it
    rather than creating a second one (a second POST left the original
    check-run pending forever; this is the finding-4-on-this-PR regression
    the workflow's own comments already document).
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    resolve = workflow.jobs["resolve"]
    complete_script = resolve.steps[1]["run"]

    result, _outputs, calls = _run_step(
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
    assert result.returncode == 0, f"the PATCH-existing-run step failed: {result.stderr}"
    assert len(calls) == 1, f"expected exactly one gh api call; got {calls!r}"
    (call,) = calls
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
    block the PR forever, the collision finding 2 named. Verified by
    executing the real step script with a stubbed `gh` reporting an
    existing check-run, and asserting no POST happened.
    """
    workflow = Workflow.load(WORKFLOW_DIR / "adversarial-review-gate.yml")
    post_pending = workflow.jobs["post-pending-check"]
    script = post_pending.steps[0]["run"]

    result, _outputs, calls = _run_step(
        tmp_path,
        script,
        step_env={
            "PR_NUM": "41",
            "SHA": "7ac3cb6",
            "IS_DRAFT": "false",
        },
        fake_existing_count="1",
    )
    assert result.returncode == 0, f"expected a clean skip, got: {result.stderr}"
    assert calls == [], (
        "post-pending-check posted a check-run even though one already "
        f"existed on the sha -- this is the collision finding 2 named. calls={calls!r}"
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

    result, _outputs, calls = _run_step(
        tmp_path,
        script,
        step_env={
            "PR_NUM": "41",
            "SHA": "7ac3cb6",
            "IS_DRAFT": "false",
        },
        fake_existing_count="0",
    )
    assert result.returncode == 0, f"expected a clean post, got: {result.stderr}"
    assert len(calls) == 1, f"expected exactly one gh api call (the pending POST); got {calls!r}"
    (call,) = calls
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

    result, _outputs, calls = _run_step(
        tmp_path,
        script,
        step_env={
            "PR_NUM": "41",
            "SHA": "7ac3cb6",
            "IS_DRAFT": "true",
        },
        fake_existing_count="0",
    )
    assert result.returncode == 0, f"draft skip should exit 0, got: {result.stderr}"
    assert calls == [], f"a draft PR must not get any check-run posted; calls={calls!r}"
