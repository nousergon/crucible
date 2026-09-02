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

import inspect
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

#: The exact `run:` bodies that may mention `tests/acceptance` on the PR path,
#: keyed `<workflow>:<job>:<step name>`. Compared verbatim.
#:
#: Three carve-outs have now been tried and defeated, each by ordinary shell
#: syntax: a `--collect-only` substring anywhere in the step (which licensed
#: `pytest tests/acceptance` on the next line), then the same test applied per
#: command with a naive splitter — beaten by a single `&` instead of `&&`, and
#: by `# --collect-only` in a trailing comment. The class is not "the splitter
#: is incomplete". The class is TEXT MATCHING OVER SHELL: any predicate over a
#: command string is a partial shell parser, and a partial shell parser is a
#: denylist of the syntax someone thought of.
#:
#: So there is no predicate. There is exactly one legitimate command, and it is
#: pinned. Editing it means editing this file, which is the point — the whole
#: reason `tests/acceptance` may appear on the PR path at all is that this
#: precise invocation collects without executing a clause body.
PINNED_ACCEPTANCE_COMMANDS: dict[str, str] = {
    "ci.yml:test:Acceptance clauses still import": (
        "uv run pytest tests/acceptance --collect-only -q"
    ),
}

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


def _mentions(text: str, marker: str) -> bool:
    """`marker` appears in `text` bounded by non-identifier characters.

    Substring-with-boundary, not whole-word: whole-word matching was
    demonstrated to miss `/usr/bin/aws`, `$(echo aws)` and
    `python -c '__import__("boto3")...'`. `awslogs` still does not match.
    """
    for match in re.finditer(re.escape(marker), text):
        before = text[match.start() - 1] if match.start() else " "
        after = text[match.end()] if match.end() < len(text) else " "
        # An underscore AFTER the marker is a separator, not a continuation:
        # `aws_access_key_id`, `aws_secret_access_key` and `$aws_profile` are
        # the single most common way live AWS reaches a workflow, and treating
        # `_` as an identifier character rejected every one of them while
        # accepting `/usr/bin/aws`. `awslogs` still does not match, because
        # `l` is not a separator.
        if not (before.isalnum() or before == "_") and not after.isalnum():
            return True
    return False


def _live_state_steps(job: dict, workflow_job: str = "") -> list[str]:
    """EVERY step in `job` that reads state outside the tree under review.

    Every one, not the first: naming one and stopping tells the reader to fix
    it and re-run to discover the next.

    The whole step is scanned, not just its `run:` body. `uses:`, `with:`,
    `env:`, `container:` and `services:` reach live state just as well, and
    moving a marker from `$(echo aws)` into `env: {TOOL: /usr/bin/aws}` was
    demonstrated to walk straight past a run-only scan — the same evasion
    class, one field along.
    """
    hits = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        run = IGNORE_TOKEN.sub("", (step.get("run") or "").lower())
        rest = yaml.safe_dump({k: v for k, v in step.items() if k != "run"}).lower()
        label = step.get("name") or step.get("uses") or "<unnamed step>"
        for marker in LIVE_STATE_MARKERS:
            if marker == "tests/acceptance":
                # No predicate over the command text — the pinned command, or
                # a hit. See PINNED_ACCEPTANCE_COMMANDS for why.
                pinned = PINNED_ACCEPTANCE_COMMANDS.get(f"{workflow_job}:{step.get('name')}")
                if pinned is not None and (step.get("run") or "").strip() == pinned:
                    continue
            if _mentions(run, marker) or _mentions(rest, marker):
                hits.append(f"{label}: mentions `{marker}`")
                break
    return hits


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
    assert re.fullmatch(
        r"nousergon/nousergon-lib/\.github/workflows/notify-ci-failure\.yml@[0-9a-f]{40}",
        str(job.get("uses", "")),
    ), (
        "notify-main-failure must call the nousergon-lib notification workflow "
        "pinned to a 40-character SHA. Its allowlist entry assumes that target, "
        "nothing here can scan a called workflow, and the job carries "
        "`secrets: inherit` — a moving ref like `@main` is a supply-chain hole, "
        "and a prefix check accepted one."
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
            hits = _live_state_steps(job, key)
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


def test_every_pinned_acceptance_command_is_the_step_that_is_there() -> None:
    """A pin that no longer matches the workflow silently permits nothing.

    Worse, an entry left behind after a step is renamed becomes a licence
    waiting for the next step to take that name.
    """
    for key, command in PINNED_ACCEPTANCE_COMMANDS.items():
        workflow_name, job_name, step_name = key.split(":", 2)
        workflow = yaml.safe_load((WORKFLOW_DIR / workflow_name).read_text())
        job = (workflow.get("jobs") or {}).get(job_name)
        assert job, f"{key} pins a command in a job that does not exist"
        steps = [s for s in (job.get("steps") or []) if s.get("name") == step_name]
        assert len(steps) == 1, (
            f"{key} pins a command to a step named {step_name!r}, and the job has "
            f"{len(steps)} of them. Two steps with one name make the pin ambiguous."
        )
        assert (steps[0].get("run") or "").strip() == command, (
            f"{key} no longer runs the pinned command.\n  pinned: {command!r}\n"
            f"  actual: {(steps[0].get('run') or '').strip()!r}\n"
            "tests/acceptance may appear on the PR path only as this exact "
            "invocation, which collects without executing a clause body. Change "
            "both together, deliberately, or not at all."
        )
