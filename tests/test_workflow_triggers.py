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
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml"))

# A step whose run script matches any of these is reading state that lives
# outside the tree under review: the live AWS account, or the acceptance suite,
# whose clauses grade the system at a phase boundary rather than the diff.
#
# `gh api` is deliberately NOT a marker. The adversarial-review gate reads the
# state of the pull request it is running on — that IS the subject under review,
# and the author can clear it.
LIVE_STATE_MARKERS = (
    "tests/acceptance",
    "aws ",
    "boto3",
)

# `--ignore=tests/acceptance` is the opposite of running it. Strip every
# --ignore= token before matching, or the blocking suite trips its own guard.
IGNORE_TOKEN = re.compile(r"--ignore=\S+")


def _events(workflow: dict) -> set[str]:
    # PyYAML resolves the bare key `on` to the boolean True (YAML 1.1).
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    if isinstance(on, dict):
        return set(on)
    raise AssertionError(f"unparseable `on:` block: {on!r}")


def _excluded_from_pull_request(job: dict) -> bool:
    condition = str(job.get("if", ""))
    return "github.event_name != 'pull_request'" in condition


def _live_state_steps(job: dict) -> list[str]:
    hits = []
    for step in job.get("steps", []) or []:
        run = IGNORE_TOKEN.sub("", step.get("run") or "")
        for marker in LIVE_STATE_MARKERS:
            if marker in run:
                hits.append(f"{step.get('name', '<unnamed step>')}: matches {marker!r}")
    return hits


def test_at_least_one_workflow_is_scanned() -> None:
    # A guard that scanned nothing is dark, not green (principle 7).
    assert WORKFLOWS, "no workflows found — this guard is not measuring anything"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_live_state_job_runs_on_the_pull_request_path(path: pathlib.Path) -> None:
    workflow = yaml.safe_load(path.read_text())
    if "pull_request" not in _events(workflow):
        return

    for name, job in (workflow.get("jobs") or {}).items():
        hits = _live_state_steps(job)
        if not hits:
            continue
        assert _excluded_from_pull_request(job), (
            f"{path.name} job `{name}` reads state outside the tree under review "
            f"and can run on a pull_request:\n  " + "\n  ".join(hits) + "\n"
            "scm-platform-policy.md §3.1 — move it to `push: [main]` plus a schedule, "
            "or gate the job with `if: github.event_name != 'pull_request'`."
        )
