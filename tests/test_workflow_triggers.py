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
    "adversarial-review-gate.yml:adversarial-review-gate": (
        "the required status check. Its subject IS the pull request — it reads "
        "the commits under review and the commit statuses on their head sha, "
        "both properties of the diff, and the author clears it by getting an "
        "independent review done. It reaches no live AWS and holds no OIDC "
        "token: it could not, since every crucible role's trust condition pins "
        "`ref:refs/heads/main` and a pull_request job runs on refs/pull/N/merge."
    ),
    "adversarial-review-gate.yml:record-verdict": (
        "workflow_dispatch-only; gated on the dispatch inputs and records one "
        "named PR's review verdict as a commit status."
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


# ---------------------------------------------------------------------------
# The adversarial review gate, after the redesign.
#
# What was deleted here, and why: this file used to carry ~700 lines executing
# the `post-pending-check` and `resolve` shell scripts under a stubbed `gh` —
# their collision guard, their self-heal, their duplicate-check-run handling.
# Every one of them graded the management of a check-run created through
# `POST /check-runs`, and a repository-ruleset required status check is NOT
# satisfied by one of those. Measured by removal on 2026-09-02: dropping the
# context from `required_status_checks` and changing nothing else merged
# instantly; restoring it re-blocked, on a head whose check-run read
# `completed/success`. So the whole mechanism those tests protected could
# never satisfy the rule it existed for, and the tests were a detailed,
# well-argued grading of a control that did not work. The required context is
# a real job now, and the comparison it makes lives in
# `.github/scripts/adversarial_review.py`, whose refusals are exercised
# directly in `tests/test_adversarial_review.py`.
# ---------------------------------------------------------------------------

REVIEW_WORKFLOW = WORKFLOW_DIR / "adversarial-review-gate.yml"

#: The context named in the `main-protection` ruleset's `required_status_checks`.
#: A required context naming a job that does not exist is a check that never
#: reports, which blocks every PR in the repository with no red anywhere to
#: explain why — so this string is asserted against the workflow, not trusted.
REQUIRED_CONTEXT = "adversarial-review-gate"


def _review_workflow() -> Workflow:
    return Workflow.load(REVIEW_WORKFLOW)


def test_the_required_context_is_emitted_by_a_real_job() -> None:
    """The whole repair. A job whose rendered check-run name IS the required
    context, running on `pull_request` — not an API-created check-run, which a
    ruleset does not accept."""
    job = _review_workflow().jobs[REQUIRED_CONTEXT]
    assert job.model_extra["name"] == REQUIRED_CONTEXT, (
        "the job's `name:` is what GitHub renders as the check-run name. If it "
        f"is not exactly {REQUIRED_CONTEXT!r}, the required context is never "
        "reported and every PR blocks forever."
    )
    assert "pull_request" in _review_workflow().events


def test_nothing_in_this_workflow_creates_its_own_check_run() -> None:
    """The measured defect, held closed. A check-run POSTed through the REST
    API cannot satisfy a ruleset required check, so a step that posts one is
    either dead weight or a second check-run colliding with the real job's."""
    workflow = _review_workflow()
    # The EXECUTABLE surface only. The file's header comment explains at
    # length why `POST /check-runs` cannot satisfy a ruleset required check,
    # and a scan over raw text would refuse the explanation along with the
    # thing it explains.
    for name, job in workflow.jobs.items():
        assert "checks" not in (job.model_extra.get("permissions") or {}), name
        for step in job.steps:
            assert "/check-runs" not in step.get("run", ""), name


def test_the_gate_job_holds_no_write_permission_of_any_kind() -> None:
    """It reads a verdict; it never records one. A gate that could write its
    own input is the shape `crucible/gate.py` exists to refuse."""
    permissions = _review_workflow().jobs[REQUIRED_CONTEXT].model_extra["permissions"]
    assert set(permissions.values()) == {"read"}, permissions


def test_the_dispatch_has_no_author_input() -> None:
    """The second defect. `author` was a free-text input supplied by the very
    session asking to be passed — a gate whose input is supplied by the thing
    it grades measures nothing. The author set is derived from the commits
    now, so there is nothing for a dispatcher to assert."""
    inputs = _review_workflow().triggers["workflow_dispatch"]["inputs"]
    assert "author" not in inputs
    assert "reviewer" in inputs


def test_both_jobs_call_the_one_shared_implementation() -> None:
    """One independence comparison, not two. Two shell reimplementations of
    "is this reviewer an author" is a contract restated twice, and one of them
    drifts."""
    jobs = _review_workflow().jobs
    gate_script = "\n".join(step.get("run", "") for step in jobs[REQUIRED_CONTEXT].steps)
    record_script = "\n".join(step.get("run", "") for step in jobs["record-verdict"].steps)
    assert "adversarial_review.py gate" in gate_script
    assert "adversarial_review.py context" in record_script
    assert (
        (pathlib.Path(__file__).resolve().parents[1] / ".github" / "scripts")
        .joinpath("adversarial_review.py")
        .is_file()
    )


def test_the_verdict_is_never_written_before_the_independence_check() -> None:
    """Ordering, asserted rather than assumed: the refusal must sit between the
    dispatch inputs and the status API, or a self-review reaches the record."""
    script = "\n".join(
        step.get("run", "") for step in _review_workflow().jobs["record-verdict"].steps
    )
    assert script.index("adversarial_review.py context") < script.index("/statuses/")


def test_the_record_job_refreshes_the_gate_it_just_satisfied() -> None:
    """Detect -> act -> VERIFY -> close, without a human. A recorded verdict
    that leaves the required check red until someone re-runs a job by hand is
    an operator step, and an operator step an agent could have run itself is a
    defect (principle 3)."""
    script = "\n".join(
        step.get("run", "") for step in _review_workflow().jobs["record-verdict"].steps
    )
    assert "/rerun" in script


def test_a_verdict_write_is_never_cancelled_by_an_in_flight_push() -> None:
    """`cancel-in-progress` must be an expression that reads FALSE for a
    workflow_dispatch event: a durable verdict write should queue behind
    another verdict write, never be discarded mid-flight."""
    cancel_in_progress = _review_workflow().model_extra["concurrency"]["cancel-in-progress"]
    assert isinstance(cancel_in_progress, str) and "workflow_dispatch" in cancel_in_progress


_FAKE_GH = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$*" >> "$CALL_LOG"
if [ "$1" != "api" ]; then
  echo "unhandled fake gh invocation (not 'api'): $*" >&2
  exit 99
fi
shift
url=""
method="GET"
jq=""
while [ $# -gt 0 ]; do
  case "$1" in
    --paginate) shift ;;
    -X) method="$2"; shift 2 ;;
    --jq|-q) jq="$2"; shift 2 ;;
    -f) shift 2 ;;
    *) if [ -z "$url" ]; then url="$1"; fi; shift ;;
  esac
done
case "$method:$url" in
  POST:*/statuses/*) exit 0 ;;
  GET:*/pulls/*/commits) cat "$FAKE_COMMITS" ;;
  GET:*/pulls/*) echo "$FAKE_SHA" ;;
  *) echo "unhandled fake gh endpoint: $method $url" >&2; exit 99 ;;
esac
"""


class _StepResult(NamedTuple):
    result: subprocess.CompletedProcess
    calls: list[str]


def _run_record_step(tmp_path: pathlib.Path, commits: list[dict], reviewer: str) -> _StepResult:
    """Execute the record-verdict job's REAL `run:` script under bash.

    The script is extracted from the workflow rather than restated, so a step
    that stops calling the independence check fails this harness rather than
    passing a copy of itself.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    (tmp_path / ".github" / "scripts").mkdir(parents=True)
    shutil.copy(
        root / ".github" / "scripts" / "adversarial_review.py",
        tmp_path / ".github" / "scripts" / "adversarial_review.py",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(_FAKE_GH)
    (bin_dir / "gh").chmod(0o755)
    commits_path = tmp_path / "fake_commits.json"
    commits_path.write_text(json.dumps(commits), encoding="utf-8")
    call_log = tmp_path / "call_log"
    call_log.write_text("")
    github_env = tmp_path / "github_env"
    github_env.write_text("")

    script = _review_workflow().jobs["record-verdict"].steps[1]["run"]
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "CALL_LOG": str(call_log),
            "GITHUB_ENV": str(github_env),
            "FAKE_SHA": "c" * 40,
            "FAKE_COMMITS": str(commits_path),
            "GH_TOKEN": "fake-token",
            "REPO": "nousergon/crucible",
            "PR_NUM": "46",
            "REVIEWER": reviewer,
            "CONCLUSION": "success",
            "SUMMARY": "no findings against plan section 2",
            "RUN_URL": "https://example.invalid/run/1",
        },
    )
    return _StepResult(result, [c for c in call_log.read_text().split("\0") if c])


_AUTHOR_SESSION = "session_01AuthorAAAAAAAA"
_COMMITS = [
    {
        "commit": {
            "message": f"fix: a thing\n\nClaude-Session: https://claude.ai/code/{_AUTHOR_SESSION}\n",
            "author": {"email": "someone@example.invalid"},
            "committer": {"email": "someone@example.invalid"},
        },
        "author": {"login": "cipher813"},
        "committer": {"login": "cipher813"},
    }
]


def test_the_authoring_session_cannot_record_a_verdict_on_its_own_change(
    tmp_path: pathlib.Path,
) -> None:
    """The one structural control this workflow exists for, executed rather
    than read: the refusal must happen, and it must happen BEFORE anything is
    written."""
    step = _run_record_step(tmp_path, _COMMITS, reviewer=_AUTHOR_SESSION)
    assert step.result.returncode != 0, step.result.stdout
    assert "independent of the author" in step.result.stderr
    assert not [call for call in step.calls if "/statuses/" in call], step.calls


def test_free_text_is_not_accepted_as_a_reviewer_identity(tmp_path: pathlib.Path) -> None:
    step = _run_record_step(tmp_path, _COMMITS, reviewer="the reviewing agent")
    assert step.result.returncode != 0
    assert not [call for call in step.calls if "/statuses/" in call]


def test_an_independent_session_records_the_verdict(tmp_path: pathlib.Path) -> None:
    """Guards against a refusal that over-corrects into refusing everything —
    the shape that made the previous design's independence check unsatisfiable
    on every dispatch."""
    step = _run_record_step(tmp_path, _COMMITS, reviewer="session_01ReviewerBBBBB")
    assert step.result.returncode == 0, step.result.stderr
    posted = [call for call in step.calls if "/statuses/" in call]
    assert posted, step.calls
    assert "adversarial-review/session_01reviewerbbbbb" in posted[0]
