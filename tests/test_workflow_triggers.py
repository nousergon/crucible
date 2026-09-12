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
import subprocess
import sys
from typing import NamedTuple

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crucible.store import LocalStore

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
    # alpha-engine-config-I10119.
    "dispatch-lockstep.yml:crucible-dispatch-lockstep-pr": (
        "runs on `pull_request_target`, which "
        "executes THIS workflow's definition from main, never the PR's. It "
        "checks out no PR ref (the self-checkout takes pull_request_target's "
        "default base ref), executes no PR-supplied code, and mints a "
        "GitHub App token narrowed with `repositories=['nous-ergon-ops']` "
        "(nousergon-lib-PR388) to read only the trusted ops test module. The "
        "one piece of PR content it reads — `crucible/components.yaml` — "
        "arrives over the read-only Contents API as data and is parsed as "
        "YAML by that trusted test, never executed. "
        "`test_the_lockstep_pr_job_checks_out_no_pr_head_and_executes_no_pr_supplied_code` "
        "below is the structural guard that a later edit cannot quietly "
        "reintroduce a PR-head checkout or PR-code execution here."
    ),
    # alpha-engine-config-I10128 deliverable 2 (ported from
    # alpha-engine-config's own gate-label-guard.yml, the fleet reference
    # implementation).
    "gate-label-guard.yml:gate-label-guard": (
        "reads only `GITHUB_EVENT_PATH` -- the PR's own labels and body, "
        "already present in the triggering event payload -- and does no "
        "checkout, no AWS call and no network I/O of any kind. Its subject "
        "is the diff's own metadata, not live infrastructure, so "
        "scm-platform-policy.md section 3.1 places it squarely on the "
        "pull_request path rather than excluding it."
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


def test_no_pull_request_target_job_checks_out_pr_authored_code() -> None:
    """`pull_request_target` is safe here ONLY because nothing fetches PR code.

    The footgun is not the trigger, it is the trigger COMBINED with a checkout
    of the PR HEAD: `pull_request_target` runs the workflow definition with
    base-repo permissions and secrets in scope, so an `actions/checkout` with
    `ref: github.event.pull_request.head.sha` executes fork-authored code with
    those secrets in reach. `dispatch-lockstep.yml`'s PR job deliberately
    overrides neither `ref` nor `repository` for its own checkout, and reads
    the one piece of PR-supplied content it needs as DATA over the read-only
    Contents API.

    **What this does NOT forbid:** checking out a DIFFERENT, TRUSTED repo. The
    push-path job clones `nousergon/nous-ergon-ops` with a minted token, which
    is not PR-authored content and is not the risk. An earlier draft of this
    test asserted on `repository`/`ref` being present at all and failed on
    exactly that job — and it reached it only because the job's `if:` NAMES
    `pull_request_target` in order to EXCLUDE it. A guard that cannot tell an
    include from an exclusion grades the wrong jobs.

    The argument lived only in a comment until 2026-09-07
    (`alpha-engine-config-I10156`), when the trigger was deleted outright on
    the reasoning that a comment plus an `if:` is not a control — right about
    the comment, wrong about the fix: deleting it made the PR job's `if:`
    unsatisfiable, so a required check silently stopped running while still
    appearing in the file. This test is the control that comment stood in for.
    """
    checked = 0
    for path in WORKFLOWS:
        document = yaml.safe_load(path.read_text())
        triggers = document.get(True) or document.get("on") or {}
        if "pull_request_target" not in triggers:
            continue
        for name, job in (document.get("jobs") or {}).items():
            condition = str(job.get("if", ""))
            # A job that names the event only to exclude itself from it does
            # not run on it. `!contains(...)` and `!=` are the two exclusion
            # shapes this repo uses.
            excluded = "!contains" in condition or "!=" in condition
            affirmative = "pull_request_target" in condition and not excluded
            if condition and not affirmative:
                continue
            checked += 1
            for step in job.get("steps") or []:
                if "actions/checkout" not in str(step.get("uses", "")):
                    continue
                with_block = step.get("with") or {}
                supplied = " ".join(str(v) for v in with_block.values())
                assert "pull_request.head" not in supplied, (
                    f"{path.name}:{name} runs on pull_request_target AND checks out "
                    f"the PR head ({with_block}). That combination executes "
                    "fork-authored code with base-repo permissions and secrets in "
                    "scope. Read PR content as data over the Contents API instead, "
                    "as this job already does for components.yaml."
                )
    assert checked, (
        "no pull_request_target job was graded — the guard walked nothing, so its "
        "clean result means nothing. If the trigger was removed, remove this test "
        "deliberately rather than letting it pass vacuously."
    )


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


def _lockstep_pr_job() -> Job:
    return Workflow.load(WORKFLOW_DIR / "dispatch-lockstep.yml").jobs[
        "crucible-dispatch-lockstep-pr"
    ]


def test_the_lockstep_pr_job_checks_out_no_pr_head_and_executes_no_pr_supplied_code() -> None:
    """alpha-engine-config-I10119: the one job in this repo that deliberately
    runs on a PR event with a live-AWS credential in scope. Its safety rests
    on two structural properties -- no checkout of the PR head, no execution
    of PR-supplied content -- asserted here rather than left to the
    allowlist reason in prose, so a later edit that reintroduces either
    fails this test."""
    job = _lockstep_pr_job()
    for step in job.steps:
        with_block = step.get("with") or {}
        ref = str(with_block.get("ref", ""))
        assert "pull_request" not in ref, (
            "a step checks out a PR-derived ref -- this job may only check "
            "out the implicit default (pull_request_target's base ref) or "
            "another trusted repository's own main branch"
        )
        body = step.get("run", "")
        assert "uv run" not in body and "pip install -e" not in body, (
            "a step executes tree-derived code -- this job must never run "
            "anything beyond the fixed pytest invocation against "
            "nous-ergon-ops main"
        )


def test_the_lockstep_pr_job_self_checkout_takes_the_implicit_base_ref() -> None:
    job = _lockstep_pr_job()
    self_checkouts = [
        step
        for step in job.steps
        if step.get("uses", "").startswith("actions/checkout@")
        and "repository" not in (step.get("with") or {})
    ]
    assert len(self_checkouts) == 1, self_checkouts
    assert "ref" not in (self_checkouts[0].get("with") or {}), (
        "the self-checkout must take the implicit default ref, not an explicit one"
    )


def test_the_lockstep_pr_job_fetches_components_yaml_as_data_not_via_checkout() -> None:
    job = _lockstep_pr_job()
    fetch_steps = [s for s in job.steps if "crucible/components.yaml" in s.get("run", "")]
    assert len(fetch_steps) == 1, fetch_steps
    body = fetch_steps[0]["run"]
    assert "repos/nousergon/crucible/contents/crucible/components.yaml" in body, body
    assert "base64 -d > crucible/components.yaml" in body, (
        "the fetched content must be decoded straight to a file, never piped "
        "into a shell or an interpreter"
    )
    for forbidden in ("| bash", "| sh", "| python", "eval "):
        assert forbidden not in body, forbidden


def test_the_lockstep_pr_job_mints_a_token_narrowed_to_nous_ergon_ops() -> None:
    job = _lockstep_pr_job()
    mint_steps = [s for s in job.steps if "installation_token(" in s.get("run", "")]
    assert len(mint_steps) == 1, mint_steps
    assert "repositories=['nous-ergon-ops']" in mint_steps[0]["run"], mint_steps[0]["run"]


def test_the_lockstep_pr_job_and_the_push_job_never_both_run() -> None:
    workflow = Workflow.load(WORKFLOW_DIR / "dispatch-lockstep.yml")
    push_job = workflow.jobs["crucible-dispatch-lockstep"]
    pr_job = workflow.jobs["crucible-dispatch-lockstep-pr"]
    assert push_job.condition.strip() in _exclusions_for(frozenset({"pull_request_target"}))
    assert pr_job.condition.strip() == "github.event_name == 'pull_request_target'"


# ---------------------------------------------------------------------------
# alpha-engine-config-I10468: the PR job's two data-fetch steps ("Fetch the
# PR's components.yaml as data" and "Mirror the PR's .github/workflows as
# data") replace a `pull_request_target` job's implicit base-ref checkout
# with the PR HEAD's tree, fetched over the read-only Contents API. Every
# test above this section proves that replacement is SAFE (no PR-head
# checkout, no PR-code execution). None of them prove it actually WORKS —
# that the resulting on-disk tree is graded as the PR's, not main's, which
# is the exact defect `alpha-engine-config-I10467`/I10468 measured on
# `crucible-PR198`: a correct fix (dropping a cron) still failed because the
# guard read the workflow files from the base ref while reading
# `components.yaml` from the PR.
#
# This extracts BOTH real `run:` bodies (not a restatement) and executes
# them under bash exactly as `test_the_off_main_guard_actually_exits_non_zero`
# does, against a fake `gh api` that serves a fixture "PR head" tree that
# differs from the "base" tree already on disk (mirroring what
# `actions/checkout@...`'s self-checkout would have left there). The
# assertions are on the FILES the two steps produce — the same files the
# push job's `Lockstep guard` step (`tests/crossrepo/test_crucible_dispatch_lockstep.py`
# in nous-ergon-ops, via `CRUCIBLE_ROOT`) reads — so a later edit that
# reintroduces the base-tree read (e.g. dropping the Mirror step, or scoping
# it to only NEW files) fails here without needing a live PR to notice.


def _pr_job_step(name: str) -> str:
    job = Workflow.load(WORKFLOW_DIR / "dispatch-lockstep.yml").jobs[
        "crucible-dispatch-lockstep-pr"
    ]
    matches = [s["run"] for s in job.steps if s.get("name") == name and "run" in s]
    assert len(matches) == 1, f"expected exactly one {name!r} step, found {len(matches)}"
    return matches[0]


_FAKE_GH_CONTENTS_API = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "$1" != "api" ]; then
  echo "unhandled fake gh invocation (not 'api'): $*" >&2
  exit 99
fi
shift
url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --jq) shift 2 ;;
    *) if [ -z "$url" ]; then url="$1"; fi; shift ;;
  esac
done
case "$url" in
  repos/nousergon/crucible/contents/crucible/components.yaml\?ref=*)
    cat "$FAKE_COMPONENTS_B64" ;;
  repos/nousergon/crucible/contents/.github/workflows\?ref=*)
    cat "$FAKE_LISTING" ;;
  repos/nousergon/crucible/contents/.github/workflows/*\?ref=*)
    rest="${url#repos/nousergon/crucible/contents/.github/workflows/}"
    name="${rest%%\?*}"
    fixture="$FAKE_WORKFLOWS_DIR/${name}.b64"
    if [ ! -f "$fixture" ]; then
      echo "unhandled fake gh endpoint (no fixture for ${name}): $url" >&2
      exit 99
    fi
    cat "$fixture" ;;
  *) echo "unhandled fake gh endpoint: $url" >&2; exit 99 ;;
esac
"""


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode()).decode()


def _run_pr_job_data_fetch(
    tmp_path: pathlib.Path,
    *,
    base_workflows: dict[str, str],
    base_components: str,
    head_workflows: dict[str, str],
    head_components: str,
) -> subprocess.CompletedProcess:
    """Seed `tmp_path` as the PR job's self-checkout (base ref) would have
    left it, then run the real "Fetch ... components.yaml" and "Mirror ...
    workflows" steps against a fake Contents API answering with the HEAD
    fixture. Returns the last step's `CompletedProcess`."""
    (tmp_path / "crucible").mkdir()
    (tmp_path / "crucible" / "components.yaml").write_text(base_components)
    workflows_dir = tmp_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    for name, content in base_workflows.items():
        (workflows_dir / name).write_text(content)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(_FAKE_GH_CONTENTS_API)
    (bin_dir / "gh").chmod(0o755)

    fixtures_dir = tmp_path / "_fixtures"
    fixtures_dir.mkdir()
    components_fixture = fixtures_dir / "components.b64"
    components_fixture.write_text(_b64(head_components))
    listing_fixture = fixtures_dir / "listing.txt"
    listing_fixture.write_text("\n".join(sorted(head_workflows)) + "\n")
    workflows_fixture_dir = fixtures_dir / "workflows"
    workflows_fixture_dir.mkdir()
    for name, content in head_workflows.items():
        (workflows_fixture_dir / f"{name}.b64").write_text(_b64(content))

    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "GH_TOKEN": "fake-token",
        "HEAD_SHA": "d" * 40,
        "FAKE_COMPONENTS_B64": str(components_fixture),
        "FAKE_LISTING": str(listing_fixture),
        "FAKE_WORKFLOWS_DIR": str(workflows_fixture_dir),
    }
    result = None
    for step_name in (
        "Fetch the PR's components.yaml as data",
        "Mirror the PR's .github/workflows as data",
    ):
        result = subprocess.run(
            ["bash", "-c", _pr_job_step(step_name)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"{step_name!r} exited {result.returncode}. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    assert result is not None
    return result


_BASE_BOARD_YML = "name: Board\non:\n  workflow_dispatch:\n"
_HEAD_NEW_CRON_YML = "name: Integration Nightly\non:\n  schedule:\n    - cron: '0 6 * * *'\n"
_BASE_COMPONENTS = "components:\n  board:\n    dispatch: scheduler\n"


def test_the_lockstep_pr_job_mirrors_the_prs_tree_not_the_base_tree(
    tmp_path: pathlib.Path,
) -> None:
    """The exact defect measured on `crucible-PR198`
    (`alpha-engine-config-I10467`/I10468): a workflow the PR ADDS, with a
    cron, must appear in the graded tree; a workflow the PR DELETES must not
    survive as base-tree leftover; an unrelated workflow must pass through
    unchanged; and `components.yaml` must be the PR's content, not main's."""
    head_components = _BASE_COMPONENTS + (
        "  integration_nightly:\n"
        "    dispatch: github-actions\n"
        "    dispatch_workflow: integration-nightly.yml\n"
    )
    _run_pr_job_data_fetch(
        tmp_path,
        base_workflows={
            "board.yml": _BASE_BOARD_YML,
            "stale-cron.yml": "name: Stale\non:\n  schedule:\n    - cron: '0 0 * * *'\n",
        },
        base_components=_BASE_COMPONENTS,
        head_workflows={
            "board.yml": _BASE_BOARD_YML,  # unrelated — must survive unchanged
            "integration-nightly.yml": _HEAD_NEW_CRON_YML,  # added by the PR
            # stale-cron.yml omitted: the PR deleted it
        },
        head_components=head_components,
    )
    workflows_dir = tmp_path / ".github" / "workflows"
    assert (workflows_dir / "board.yml").read_text() == _BASE_BOARD_YML, (
        "an unrelated workflow must pass through the mirror unchanged"
    )
    assert (workflows_dir / "integration-nightly.yml").read_text() == _HEAD_NEW_CRON_YML, (
        "a workflow the PR adds must be present with the PR's own content, not absent "
        "the way a base-tree read would leave it"
    )
    assert not (workflows_dir / "stale-cron.yml").exists(), (
        "a workflow the PR deletes must not survive as base-tree leftover — leaving it "
        "would keep the guard failing over a file the PR already removed"
    )
    assert (tmp_path / "crucible" / "components.yaml").read_text() == head_components, (
        "components.yaml must be the PR's content, not main's — the other half of the "
        "same-commit-pair fix"
    )


@pytest.mark.parametrize(
    "declares_the_row,expect_named",
    [
        pytest.param(True, True, id="with-registry-row"),
        pytest.param(False, False, id="without-registry-row"),
    ],
)
def test_the_mirrored_tree_grades_a_new_cron_workflow_both_directions(
    tmp_path: pathlib.Path, declares_the_row: bool, expect_named: bool
) -> None:
    """Both directions of `alpha-engine-config-I10468`'s deliverable 2,
    against the ACTUAL mirrored tree rather than a restated fixture: a PR
    adding `integration-nightly.yml` WITH its `components.yaml` row must
    graded-pass; the identical PR WITHOUT the row must grade-fail. Before
    the mirror fix, neither case could ever have reached this predicate
    correctly — the workflow the PR added was invisible (read from the base
    tree), so `cron_workflows` never contained it and direction 2 below
    passed vacuously regardless of the row.

    The predicate itself — "every workflow declaring an `on.schedule` cron
    is named by some row's `dispatch_workflow`" — is
    `nous-ergon-ops/tests/crossrepo/test_crucible_dispatch_lockstep.py`'s
    `_check_every_cron_workflow_is_named_by_a_row`, restated here in its
    minimal pure form so this test has no dependency on that repo's checkout
    being present; that file's own `TestTheGuardItselfFires` proves the same
    two directions against hand-built fixtures instead of a mirrored tree.
    """
    head_components = _BASE_COMPONENTS
    if declares_the_row:
        head_components += (
            "  integration_nightly:\n"
            "    dispatch: github-actions\n"
            "    dispatch_workflow: integration-nightly.yml\n"
        )
    _run_pr_job_data_fetch(
        tmp_path,
        base_workflows={"board.yml": _BASE_BOARD_YML},
        base_components=_BASE_COMPONENTS,
        head_workflows={
            "board.yml": _BASE_BOARD_YML,
            "integration-nightly.yml": _HEAD_NEW_CRON_YML,
        },
        head_components=head_components,
    )

    workflows_dir = tmp_path / ".github" / "workflows"
    cron_workflows = set()
    for path in workflows_dir.iterdir():
        parsed = yaml.safe_load(path.read_text())
        triggers = parsed.get(True, parsed.get("on"))
        if isinstance(triggers, dict) and triggers.get("schedule"):
            cron_workflows.add(path.name)
    assert cron_workflows == {"integration-nightly.yml"}, (
        "sanity: the mirrored tree must actually carry the new cron, or this test "
        "proves nothing about the row"
    )

    components = yaml.safe_load((tmp_path / "crucible" / "components.yaml").read_text())
    named = {
        row.get("dispatch_workflow")
        for row in components["components"].values()
        if row.get("dispatch") == "github-actions"
    }
    undeclared = cron_workflows - named
    is_named = "integration-nightly.yml" not in undeclared
    assert is_named == expect_named, (
        f"declares_the_row={declares_the_row}: expected "
        f"'integration-nightly.yml' named={expect_named}, got {is_named} "
        f"(named={named}, undeclared={undeclared})"
    )


# ---------------------------------------------------------------------------
# The adversarial review recording workflow.
#
# What was deleted here, and why: this file used to carry ~700 lines executing
# the `post-pending-check` and `resolve` shell scripts under a stubbed `gh`.
# Every one of them graded the management of a check-run created through
# `POST /check-runs`, and a repository-ruleset required status check is NOT
# satisfied by one of those (measured 2026-09-02 by removal, on a head whose
# check-run read `completed/success`). Then Brian ruled on 2026-09-03 that the
# required-check FORM is abandoned permanently — an `if:`-skipped job still
# posts a check-run under its own name and a `skipped` conclusion counts as
# success, so the graded party could green the gate by pressing a dispatch
# button. So the tests went with the mechanism, twice over.
#
# Two properties of that block were real and are rebuilt below rather than
# lost: the `gh` stub's own self-test (a harness that silently answers an
# unhandled endpoint proves nothing about the script it runs), and the
# draft-PR refusal. The harness executes EVERY `run:` body in the surviving
# job — an earlier version ran one step of one job, which is precisely why a
# timeout-budget defect and a missing verification step were invisible to it.
#
# The independence comparison itself is NOT graded here: it lives in
# `crucible/review.py` and is exercised as ordinary Python in
# `tests/test_review.py`. This file grades the workflow's shape, its ordering,
# and the fact that its shell actually reaches that module.
# ---------------------------------------------------------------------------

RECORD_WORKFLOW = WORKFLOW_DIR / "adversarial-review-record.yml"


def _record() -> Workflow:
    return Workflow.load(RECORD_WORKFLOW)


def _record_job() -> Job:
    return _record().jobs["record-verdict"]


def _record_steps() -> list[str]:
    return [step["run"] for step in _record_job().steps if "run" in step]


def _record_step(name_fragment: str) -> str:
    """One step's `run:` body, selected by NAME rather than by index.

    An index moves the moment a step is inserted, and a harness that then runs
    the wrong body still passes — which is how an earlier version of this file
    executed one step of one job while reading as if it covered both.
    """
    matches = [
        step["run"]
        for step in _record_job().steps
        if name_fragment.lower() in step.get("name", "").lower() and "run" in step
    ]
    assert len(matches) == 1, f"{name_fragment!r} matched {len(matches)} steps, expected 1"
    return matches[0]


def test_no_workflow_posts_a_check_run_or_a_commit_status() -> None:
    """The review is a practice that is RUN, not a check that BLOCKS (Brian
    ruling 2026-09-03).

    Two measured properties made a required check worse than none, because
    each converts an unexamined PR into a green one: an `if:`-skipped job
    still posts a check-run under its own name and a `skipped` conclusion
    counts as successful, so any workflow with a gate job plus a second
    trigger let the graded party green it by pressing a button; and the author
    side of the comparison is read out of a commit message the author writes.

    This is asserted over EVERY workflow, not just the review one: the failure
    mode is somebody re-adding the blocking half somewhere else because the
    reason it went away was not written down where they would look.
    """
    for path in WORKFLOWS:
        workflow = Workflow.load(path)
        for name, job in workflow.jobs.items():
            granted = job.model_extra.get("permissions") or {}
            assert "checks" not in granted, f"{path.name}:{name}"
            assert "statuses" not in granted, f"{path.name}:{name}"
            for step in job.steps:
                body = step.get("run", "")
                assert "/check-runs" not in body, f"{path.name}:{name}"
                assert "/statuses/" not in body, f"{path.name}:{name}"


def test_the_recording_workflow_never_runs_on_a_pull_request() -> None:
    """A `pull_request` trigger here would put a `record-verdict` check-run on
    every PR head — a red check on an unreviewed PR, which is exactly the
    noise the ruling removed. An unreviewed PR is a normal state now."""
    assert _record().events == {"workflow_dispatch"}


def test_the_recording_job_carries_no_if_condition() -> None:
    """A single trigger makes an `if:` unnecessary, and an `if:` reappearing
    here is the tell that a second trigger was added — the shape in which a
    skipped job's check-run became satisfiable by a button press."""
    assert _record_job().condition == ""


def test_the_dispatch_has_no_author_input() -> None:
    """`author` was a free-text input supplied by the very session asking to be
    passed — a gate whose input is supplied by the thing it grades measures
    nothing. The author set is read out of the commits now, so there is
    nothing for a dispatcher to assert."""
    inputs = _record().triggers["workflow_dispatch"]["inputs"]
    assert "author" not in inputs
    assert "reviewer" in inputs


def test_the_dispatch_requires_head_sha() -> None:
    """`alpha-engine-config-I9861`: the reviewed sha is supplied as data, the
    same way `reviewer` is, rather than inferred from the PR's head at
    dispatch time — a push landing between the review finishing and the
    dispatch firing must not stamp the verdict on unreviewed code."""
    inputs = _record().triggers["workflow_dispatch"]["inputs"]
    assert inputs.get("head_sha", {}).get("required") is True


def test_the_head_sha_mismatch_guard_runs_before_the_draft_check() -> None:
    """The moved-under-review refusal is the FIRST thing this step checks —
    before the draft check, before the commits fetch, before the independence
    comparison — so a stale sha is caught before any other work happens on
    it."""
    script = _record_step("refuse a self-review")
    assert script.index('"$HEAD_SHA" != "$live_sha"') < script.index('"$is_draft" = "True"')


def test_the_workflow_calls_crucible_review_rather_than_reimplementing_it() -> None:
    """One independence comparison and one key shape, not a shell copy of
    each. The module is imported by the gate clause too, so a drift between
    producer and consumer is impossible rather than merely unlikely."""
    script = "\n".join(_record_steps())
    assert "python -m crucible.review check" in script
    assert "python -m crucible.review record" in script


def test_the_self_review_refusal_runs_before_any_credential_is_issued() -> None:
    """A session that may not record a verdict has no business holding a token
    that could write one. Ordering asserted against the step list, not
    assumed."""
    steps = _record_job().steps
    checked = next(i for i, s in enumerate(steps) if "crucible.review check" in s.get("run", ""))
    credentials = next(
        i for i, s in enumerate(steps) if "configure-aws-credentials" in s.get("uses", "")
    )
    assert checked < credentials


def test_the_dispatch_is_refused_off_main() -> None:
    """`alpha-engine-config-I9882`: the review role's OIDC trust pins
    `ref:refs/heads/main` while `workflow_dispatch` accepts any ref, so a
    dispatch off a branch failed at the assume — after the independence check
    had passed, with an error STS gives no useful text for. The guard is the
    FIRST step, so it costs nothing and its message carries the remedy."""
    first = _record_job().steps[0]
    assert "github.ref != 'refs/heads/main'" in first.get("if", "")
    assert "--ref main" in first.get("run", "")
    # Finding 1 (adversarial review, this PR): the guard's `exit 1` was moved
    # into the NEXT step ("Repository variables are set") when that step was
    # inserted above it, so this step echoed an `::error::` and exited 0 — a
    # dispatch off a branch then ran on to `configure-aws-credentials` and
    # died on the opaque STS error `alpha-engine-config-I9882` exists to
    # prevent. `test_the_off_main_guard_actually_exits_non_zero` below
    # executes this exact `run:` body under bash and proves the exit code;
    # this assertion is the cheap static half.
    assert "exit 1" in first.get("run", "")


def test_the_job_holds_no_write_permission_beyond_the_oidc_token() -> None:
    granted = _record_job().model_extra["permissions"]
    assert granted == {"contents": "read", "pull-requests": "read", "id-token": "write"}


def test_the_job_verifies_the_lockfile_before_installing() -> None:
    """`uv sync --frozen` installs the locked versions without re-checking
    them against pyproject.toml, so lockfile drift is invisible to it. Seven
    Dependabot PRs merged green over exactly that."""
    script = "\n".join(_record_steps())
    assert "uv lock --check" in script
    assert script.index("uv lock --check") < script.index("uv sync --frozen")


def test_a_verdict_write_is_never_cancelled_by_an_in_flight_dispatch() -> None:
    """A durable verdict write should queue behind another verdict write,
    never be discarded mid-flight."""
    assert _record().model_extra["concurrency"]["cancel-in-progress"] is False


# `<workflow>:<job>` pairs carrying the "Repository variables are set" guard
# introduced by `alpha-engine-config-I9906` — every AWS-touching job in the
# five workflows this repo has. Finding 2 (adversarial review, this PR): the
# guard was asserted for its `if:`/message text but never for actually
# failing, which is exactly how Finding 1's `exit 1` went missing from the
# off-main guard and was never caught. Each entry here is executed under real
# bash, once with the two variables empty (must fail, must name the `gh
# variable set` remedy) and once with them set (must succeed).
#
# Keyed by `<workflow>:<job>`, not by workflow file. `gate-close.yml` carries
# TWO AWS-touching jobs since `alpha-engine-config-I10508` — `gate-publish`
# and `gate-close`, under two different identities — and a dict keyed by the
# file could only ever exercise one of them, silently.
VARIABLE_GUARD_JOBS: dict[str, str] = {
    "board.yml:board": "board",
    "gate-close.yml:gate-publish": "gate-publish",
    "gate-close.yml:gate-close": "gate-close",
    "ci.yml:acceptance": "acceptance",
    "deploy.yml:release": "release",
    "morning-report.yml:report": "report",
    "adversarial-review-record.yml:record-verdict": "record-verdict",
}


def _variable_guard_run(workflow_file: str, job_name: str) -> str:
    workflow = Workflow.load(WORKFLOW_DIR / workflow_file)
    job = workflow.jobs[job_name]
    matches = [
        step["run"]
        for step in job.steps
        if step.get("name") == "Repository variables are set" and "run" in step
    ]
    assert len(matches) == 1, (
        f"{workflow_file}:{job_name} — expected exactly one "
        f"'Repository variables are set' step, found {len(matches)}"
    )
    return matches[0]


def test_deploy_smoke_gate_installs_and_imports_the_arcticdb_extra() -> None:
    """alpha-engine-config-I10069: `releases/current -> e329f205` was smoked
    green and died on its first replay arc with `No module named
    'arcticdb'` — the smoke installed and imported the harness without the
    `[arcticdb]` extra, so the one dependency the data layer cannot run
    without was outside what "smoked" measured. `deploy.yml`'s install-proof
    step (the same step `test_the_flip_is_gated_on_a_real_pip_install...` in
    `tests/test_deploy.py` extracts) must install the extra pyproject.toml
    declares and actually import it, before the smoke that gates the flip
    runs — and the extra name must be DERIVED from pyproject.toml, never
    restated (crucible/AGENTS.md: no suppression collections)."""
    workflow = Workflow.load(WORKFLOW_DIR / "deploy.yml")
    steps = workflow.jobs["release"].steps
    proof = next(s for s in steps if "pip install" in s.get("run", ""))
    script = proof["run"]
    assert "tomllib" in script and "optional-dependencies" in script, (
        "the extra name must be read out of pyproject.toml's own "
        "[project.optional-dependencies], not hardcoded in the workflow"
    )
    assert "import nousergon_lib.arcticdb, arcticdb" in script, (
        "the proof must actually import the module the data layer needs on this "
        "x86_64 runner — the box's architecture after nous-ergon-ops-PR1054"
    )
    smoke = next(i for i, s in enumerate(steps) if "crucible smoke" in s.get("run", ""))
    assert steps.index(proof) < smoke, (
        "the extras must be proven before the smoke that gates the flip runs"
    )


def test_every_variable_guard_job_still_exists() -> None:
    """A stale entry in `VARIABLE_GUARD_JOBS` would silently stop exercising a
    guard the moment its job was renamed — the same shape of hole
    `test_every_allowlisted_job_still_exists` closes for `PR_REACHABLE_JOBS`."""
    for label, job_name in VARIABLE_GUARD_JOBS.items():
        workflow_file = label.split(":", 1)[0]
        workflow = Workflow.load(WORKFLOW_DIR / workflow_file)
        assert job_name in workflow.jobs, f"{label} no longer exists"


@pytest.mark.parametrize("label,job_name", sorted(VARIABLE_GUARD_JOBS.items()))
def test_the_variable_guard_actually_exits_non_zero_when_unset(
    tmp_path: pathlib.Path, label: str, job_name: str
) -> None:
    workflow_file = label.split(":", 1)[0]
    script = _variable_guard_run(workflow_file, job_name)
    env = {"PATH": "/usr/bin:/bin", "AWS_ACCOUNT_ID": "", "STORE_URI": ""}
    result = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, env=env
    )
    assert result.returncode != 0, (
        f"{label} 'Repository variables are set' step exited "
        f"0 with both variables unset. stdout={result.stdout!r}"
    )
    assert "gh variable set" in result.stdout, result.stdout


@pytest.mark.parametrize("label,job_name", sorted(VARIABLE_GUARD_JOBS.items()))
def test_the_variable_guard_passes_when_both_variables_are_set(
    tmp_path: pathlib.Path, label: str, job_name: str
) -> None:
    workflow_file = label.split(":", 1)[0]
    script = _variable_guard_run(workflow_file, job_name)
    env = {
        "PATH": "/usr/bin:/bin",
        "AWS_ACCOUNT_ID": "111111111111",
        "STORE_URI": "s3://fake-bucket/prefix",
        # ci.yml:acceptance carries a third variable (Finding 3,
        # CFN_TEMPLATE_BUCKET) the other four guards do not — harmless as an
        # unused env var on the other four.
        "CFN_TEMPLATE_BUCKET": "fake-cfn-bucket",
    }
    result = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, (
        f"{label} 'Repository variables are set' step failed "
        f"with both variables set. stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_the_off_main_guard_actually_exits_non_zero(tmp_path: pathlib.Path) -> None:
    """The static half is `test_the_dispatch_is_refused_off_main`'s `"exit 1"
    in ...` assertion; this is the executable half Finding 2 asked for —
    proving the step fails rather than merely containing the string."""
    first = _record_job().steps[0]
    # GitHub evaluates `${{ github.ref }}` to a plain string BEFORE handing
    # the script to bash — the runtime shell never sees the `${{ }}` syntax.
    # A real off-branch dispatch is substituted here so the harness runs the
    # same shell bash would actually receive, rather than choking on GHA
    # expression syntax as a (invalid) parameter expansion.
    script = first["run"].replace("${{ github.ref }}", "refs/heads/feat/some-branch")
    env = {"PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, env=env
    )
    assert result.returncode != 0, (
        f"off-main guard exited 0 for a dispatch off main. stdout={result.stdout!r}"
    )
    assert "dispatched from" in result.stdout


_FAKE_GH = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$*" >> "$CALL_LOG"
if [ "$1" != "api" ]; then
  echo "unhandled fake gh invocation (not 'api'): $*" >&2
  exit 99
fi
shift
url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --paginate) shift ;;
    -X|--jq|-q|-f) shift 2 ;;
    *) if [ -z "$url" ]; then url="$1"; fi; shift ;;
  esac
done
# Routed on the URL, and anything unrecognised exits 99 rather than being
# answered by whichever branch happens to match. A stub that quietly answers an
# endpoint the script never really called proves nothing about the script;
# `test_the_fake_gh_stub_rejects_an_unrecognised_endpoint` holds that property.
case "$url" in
  */pulls/*/commits) cat "$FAKE_COMMITS" ;;
  */pulls/*) cat "$FAKE_PR" ;;
  *) echo "unhandled fake gh endpoint: $url" >&2; exit 99 ;;
esac
"""

_AUTHOR_SESSION = "session_01AuthorAAAAAAAA"
_REVIEWER_SESSION = "session_01ReviewerBBBBB"
_HEAD_SHA = "c" * 40
_COMMITS = [
    {
        "commit": {
            "message": (
                f"fix: a thing\n\nClaude-Session: https://claude.ai/code/{_AUTHOR_SESSION}\n"
            ),
            "author": {"email": "someone@example.invalid"},
            "committer": {"email": "someone@example.invalid"},
        },
        "author": {"login": "cipher813"},
        "committer": {"login": "cipher813"},
    }
]


class _StepResult(NamedTuple):
    result: subprocess.CompletedProcess
    calls: list[str]
    exported: str


def _run_step(
    tmp_path: pathlib.Path,
    script: str | list[str],
    overrides: dict[str, str],
    *,
    draft: bool = False,
) -> _StepResult:
    """Execute workflow steps' REAL `run:` bodies under bash, with `gh` stubbed.

    The scripts are extracted from the workflow rather than restated, so a step
    that stops calling the independence check fails this harness rather than
    passing a copy of itself. `uv` is stubbed to exec its argument, which runs
    the genuine `crucible.review` out of this checkout — the module is the
    thing under test, and mocking it would leave the shell grading nothing.

    Several steps may be passed, and they run in order in ONE working
    directory with `GITHUB_ENV` carried forward — so the handoffs between them
    (`commits.json` on disk, `REVIEW_SHA` exported) are exercised rather than
    faked. A harness that fabricated those inputs would grade each step against
    a world the previous step does not actually produce. The result is the LAST
    step's, and the run stops at the first non-zero exit, exactly as GitHub
    would.
    """
    scripts = [script] if isinstance(script, str) else script
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(_FAKE_GH)
    (bin_dir / "gh").chmod(0o755)
    (bin_dir / "uv").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n[ "$1" = "run" ] || exit 0\nshift\nexec "$@"\n'
    )
    (bin_dir / "uv").chmod(0o755)
    # The workflow text under test calls `uv run python ...`, which is what CI
    # runs — a machine that only has `python3` on PATH (this laptop, measured)
    # must still exercise that exact text rather than a rewritten one. The
    # harness supplies the interpreter the workflow asks for, in its own temp
    # `bin/`, pointed at the running interpreter's own executable, rather than
    # relying on whatever the host happens to name its interpreter
    # (`alpha-engine-config-I9953`).
    (bin_dir / "python").write_text(f'#!/usr/bin/env bash\nexec {sys.executable!r} "$@"\n')
    (bin_dir / "python").chmod(0o755)

    def _dump(name: str, payload: object) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    call_log = tmp_path / "call_log"
    call_log.write_text("")
    github_env = tmp_path / "github_env"
    github_env.write_text("")
    step_summary = tmp_path / "step_summary"
    step_summary.write_text("")

    root = pathlib.Path(__file__).resolve().parents[1]
    variables = {
        "PATH": f"{bin_dir}:{pathlib.Path(sys.executable).parent}:/usr/bin:/bin",
        "PYTHONPATH": str(root),
        "CALL_LOG": str(call_log),
        "GITHUB_ENV": str(github_env),
        "GITHUB_STEP_SUMMARY": str(step_summary),
        "FAKE_PR": _dump("pr.json.fixture", {"head": {"sha": _HEAD_SHA}, "draft": draft}),
        "FAKE_COMMITS": _dump("commits.json.fixture", _COMMITS),
        "GH_TOKEN": "fake-token",
        "REPO": "nousergon/crucible",
        "PR_NUM": "48",
        "REVIEWER": _REVIEWER_SESSION,
        "HEAD_SHA": _HEAD_SHA,
        "PHASE": "phase1",
        "VERDICT": "pass",
        "SUMMARY": "no findings against plan section 2",
        "STORE_URI": str(tmp_path / "store"),
    }
    variables.update(overrides)
    for body in scripts:
        result = subprocess.run(
            ["bash", "-c", body],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=variables,
        )
        # `GITHUB_ENV` is how a step hands a value to the next one. Replayed
        # here rather than assumed, so a step that stops exporting what the
        # next one reads fails this harness.
        for line in github_env.read_text().splitlines():
            if "=" in line:
                name, _, value = line.partition("=")
                variables[name] = value
        if result.returncode != 0:
            break
    return _StepResult(
        result,
        [c for c in call_log.read_text().split("\0") if c],
        github_env.read_text(),
    )


def _assert_step_refused_for_the_reason_under_test(step: _StepResult) -> None:
    """Fail loudly when the step died before reaching the control under test.

    A step can exit non-zero for a reason that has nothing to do with the
    guard a test names — a missing interpreter, a shell syntax slip, an
    unstubbed endpoint — and an assertion that only checks for a substring in
    `stderr` cannot tell that failure apart from the real one. Measured,
    `alpha-engine-config-I9953`: every refusal case below "passed" on a
    machine with no `python` binary on `PATH`, because the shell died with
    `exec: python: not found` before the code under test ever ran, and no
    assertion here distinguished that from the guard actually firing.

    Every intentional refusal in this workflow prints an `::error::`-prefixed
    line — bash's own `echo "::error::..."` for the draft guard, or
    `crucible.review.main`'s `print(f"::error::{exc}", file=sys.stderr)` for a
    `ReviewError`. Its absence means the step never reached that code, and an
    assertion on the surrounding text at that point proves nothing about the
    guard.
    """
    combined = step.result.stdout + step.result.stderr
    assert "::error::" in combined, (
        "the step exited without an `::error::`-marked message — it died "
        "before reaching the guard this test asserts on, not because the "
        f"guard refused it. stdout={step.result.stdout!r} "
        f"stderr={step.result.stderr!r}"
    )


def test_the_reached_its_subject_guard_actually_fires_on_a_dead_shell(
    tmp_path: pathlib.Path,
) -> None:
    """`_assert_step_refused_for_the_reason_under_test` is worth nothing
    unless something has been seen making it fail (Test discipline: "a
    detector nobody has made fail is a detector nobody knows works"). A step
    that dies at the shell level before any `::error::` line — exactly what a
    missing interpreter produced on this laptop before
    `alpha-engine-config-I9953` — must trip it."""
    step = _run_step(tmp_path, 'echo "no error marker here" >&2; exit 1', {})
    with pytest.raises(AssertionError, match="died before reaching the guard"):
        _assert_step_refused_for_the_reason_under_test(step)


def test_the_fake_gh_stub_rejects_an_unrecognised_endpoint(tmp_path: pathlib.Path) -> None:
    """This harness is only as real as its stub's routing. A stub that answers
    a wrong path by coincidence is how a step's real endpoint goes unexercised
    while the test reads green — which is what happened to the predecessor of
    this file."""
    step = _run_step(tmp_path, 'gh api "/repos/$REPO/bogus-endpoint"', {})
    assert step.result.returncode == 99, step.result.stdout


def test_the_authoring_session_cannot_record_a_verdict_on_its_own_change(
    tmp_path: pathlib.Path,
) -> None:
    """The one structural control this workflow exists for, executed through
    the real shell and the real module rather than read off the file."""
    step = _run_step(tmp_path, _record_step("refuse a self-review"), {"REVIEWER": _AUTHOR_SESSION})
    assert step.result.returncode != 0, step.result.stdout
    _assert_step_refused_for_the_reason_under_test(step)
    assert "independent of the author" in step.result.stderr


def test_free_text_is_not_accepted_as_a_reviewer_identity(tmp_path: pathlib.Path) -> None:
    step = _run_step(
        tmp_path, _record_step("refuse a self-review"), {"REVIEWER": "the reviewing agent"}
    )
    assert step.result.returncode != 0
    _assert_step_refused_for_the_reason_under_test(step)
    assert "not a Claude session id" in step.result.stderr


def test_a_verdict_is_refused_against_a_draft_pr(tmp_path: pathlib.Path) -> None:
    """A verdict recorded against a draft sits on a sha the author is still
    editing, and is honoured the moment the PR is marked ready — no further
    push, no re-review."""
    step = _run_step(tmp_path, _record_step("refuse a self-review"), {}, draft=True)
    assert step.result.returncode != 0
    _assert_step_refused_for_the_reason_under_test(step)
    assert "draft" in step.result.stdout.lower() + step.result.stderr.lower()


def test_a_verdict_is_refused_when_the_pr_moved_under_review(tmp_path: pathlib.Path) -> None:
    """`alpha-engine-config-I9861`: a push landing between the review
    finishing and the dispatch firing must not stamp the verdict on
    unreviewed code. `_HEAD_SHA` is what the fake PR's live head reports;
    dispatching with a DIFFERENT sha simulates exactly that push, and the
    step must refuse rather than silently recording against whichever sha is
    live."""
    stale_sha = "d" * 40
    step = _run_step(tmp_path, _record_step("refuse a self-review"), {"HEAD_SHA": stale_sha})
    assert step.result.returncode != 0, step.result.stdout
    _assert_step_refused_for_the_reason_under_test(step)
    combined = step.result.stdout + step.result.stderr
    assert stale_sha in combined, combined
    assert _HEAD_SHA in combined, combined
    assert "re-review" in combined.lower(), combined
    assert "re-dispatch" in combined.lower(), combined


def test_an_independent_session_passes_the_check_and_exports_the_sha(
    tmp_path: pathlib.Path,
) -> None:
    """Guards against a refusal that over-corrects into refusing everything —
    the shape that made the previous design's independence check unsatisfiable
    on every dispatch. Also pins the step's one output: the filing step reads
    `REVIEW_SHA`, and a step that stopped exporting it would file a review of
    nothing."""
    step = _run_step(tmp_path, _record_step("refuse a self-review"), {})
    assert step.result.returncode == 0, step.result.stderr
    assert f"REVIEW_SHA={_HEAD_SHA}" in step.exported


def test_the_filing_step_writes_the_artifact_the_gate_clause_reads(
    tmp_path: pathlib.Path,
) -> None:
    """The second `run:` body, executed end to end into a real store — the
    step that actually produces the deliverable. An earlier version of this
    harness ran one step only, which is why two defects in the other one were
    invisible."""
    step = _run_step(
        tmp_path,
        [_record_step("refuse a self-review"), _record_step("File the review artifact")],
        {},
    )
    assert step.result.returncode == 0, step.result.stdout + step.result.stderr
    filed = sorted(LocalStore(tmp_path / "store").list_keys("reviews/"))
    assert len(filed) == 1, filed
    assert filed[0].startswith("reviews/phase1/")
    assert filed[0].endswith(f"/{_REVIEWER_SESSION.lower()}/pass.json")


def test_the_filing_step_records_a_fail_under_its_own_key(tmp_path: pathlib.Path) -> None:
    """The durability guarantee, end to end: a `fail` and a `pass` from one
    reviewer on one session are different objects, so re-recording a pass
    cannot erase the finding."""
    _run_step(
        tmp_path,
        [_record_step("refuse a self-review"), _record_step("File the review artifact")],
        {"VERDICT": "fail"},
    )
    filed = sorted(LocalStore(tmp_path / "store").list_keys("reviews/"))
    assert filed and filed[0].endswith("/fail.json"), filed
