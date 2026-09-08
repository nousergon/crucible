"""A workflow that runs `crucible` against the live store declares the topic names.

`crucible.alerts._required_name` refuses to guess a topic name. That refusal is
correct — this repository is public and publishing a page to a topic nobody is
watching is worse than raising — but it means the names are **required env**,
and nothing asserted that any workflow supplies them.

Nothing did. The 2026-09-07 public-repo scrub (`alpha-engine-config-I10156`)
removed the literals; the repository variables were created the same day; and
no workflow's `env:` block was given them. Measured consequence:

    Board       2026-09-07T23:37:45Z  failure
    Gate close  2026-09-08T00:56:02Z  failure

both `RuntimeError: CRUCIBLE_MUTED_TOPIC is unset`, raised from
`gate._clause_old_alerts_muted` while formatting its own UNMET detail.

So the v2 board and the phase-exit filer were dark from the moment the repo went
public — on the two days phase 0 and phase 1 both turned MET, which is exactly
when the filer's one job comes due.

**Why CI did not catch it.** The raising branch is only reached when a clause
reads UNMET against the LIVE store, and no CI job grades the live store. A test
over the workflow files is the only place this is checkable before it fires in
production, which is the whole reason this file exists rather than another
assertion inside the gate.

**Derived, never hand-listed** — `AGENTS.md` rule 4. The required names come
from `crucible.alerts` itself, so a third required variable added there fails
this test until every workflow that could need it declares it. A literal list
here would go stale silently, which is the failure being fixed.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from crucible import alerts

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOWS = sorted(p for p in WORKFLOW_DIR.glob("*.y*ml") if p.suffix in {".yml", ".yaml"})

#: Every `*_TOPIC_VAR` `crucible.alerts` declares. Read off the module so the
#: set cannot drift from the code that raises on it.
REQUIRED_TOPIC_VARS = frozenset(
    value
    for name, value in vars(alerts).items()
    if name.endswith("_TOPIC_VAR") and isinstance(value, str)
)

#: A `crucible <job>` invocation that reaches the live store. `--store` is the
#: discriminator, not the subcommand: every job that reads or writes the store
#: takes it, and a job that does not take it cannot reach an alert clause.
_LIVE_INVOCATION = re.compile(r"crucible\s+[\w.]+[^\n]*--store", re.M)


def _run_bodies(document: dict) -> str:
    out = []
    for job in (document.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            if isinstance(step, dict) and step.get("run"):
                out.append(str(step["run"]))
    return "\n".join(out)


def test_the_required_set_is_not_empty():
    """A derived set that derives nothing would make every assertion below vacuous."""
    assert REQUIRED_TOPIC_VARS, (
        "crucible.alerts declares no *_TOPIC_VAR — either the module was "
        "refactored and this guard now measures nothing, or the names moved "
        "somewhere this test cannot see them."
    )
    assert WORKFLOWS, "no workflows found — this guard is not measuring anything"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_reaching_the_live_store_declares_every_topic_name(path):
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not _LIVE_INVOCATION.search(_run_bodies(document)):
        return

    declared = set((document.get("env") or {}).keys())
    for job in (document.get("jobs") or {}).values():
        declared |= set((job.get("env") or {}).keys())
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                declared |= set((step.get("env") or {}).keys())

    missing = sorted(REQUIRED_TOPIC_VARS - declared)
    assert not missing, (
        f"{path.name} runs `crucible ... --store` against the live store but "
        f"does not declare {missing}. `crucible.alerts._required_name` raises "
        "rather than guessing a topic name, and the raising branch is only "
        "reached against live state — so this workflow fails in production and "
        "in no CI job. Add the variable(s) to its `env:` block, reading from "
        "`${{ vars.<NAME> }}`; the repository variables already exist."
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_topic_names_are_read_from_variables_and_never_written_as_literals(path):
    """The names are exactly what the public-repo scrub removed from the tree.

    Re-introducing one as a workflow literal would satisfy the test above while
    putting back the thing `AGENTS.md` forbids, and workflow files are as public
    as any other file here.
    """
    # The scrub that removed the literals is alpha-engine-config-I10156; it is
    # cited here in a comment rather than in the assertion message because
    # `test_no_stale_tracker_literals` forbids a tracker number in package
    # source outside prose.
    text = path.read_text(encoding="utf-8")
    for var in sorted(REQUIRED_TOPIC_VARS):
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(f"{var}:"):
                continue
            value = stripped.split(":", 1)[1].strip()
            assert value.startswith("${{") and "vars." in value, (
                f"{path.name} sets {var} to {value!r}. It must read "
                f"`${{{{ vars.{var} }}}}` — a literal topic name in a public "
                "repository is what the public-repo scrub removed (see the "
                "comment above)."
            )
