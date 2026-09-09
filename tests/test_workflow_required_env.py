r"""Every workflow that runs against the LIVE store declares every required
environment variable the package will refuse to run without.

Normative source: `crucible/AGENTS.md` (Visibility), `alpha-engine-config-I10156`.

## The defect this file exists to make impossible

`alpha-engine-config-I10156` moved five role names and two topic names out of
this public tree and into required environment, in one commit. Nothing then
declared them to a workflow. `Board` failed 2026-09-07T23:37Z and `Gate close`
failed 2026-09-08T00:56Z, both on `RuntimeError: CRUCIBLE_MUTED_TOPIC is
unset` — so the v2 board and the phase-exit filer were dark on the two days
phase 0 and phase 1 turned MET.

The repair declared the two TOPIC names to those workflows. It did not declare
`CRUCIBLE_MACHINE_PRINCIPALS`, extracted by the same commit for the same
reason, and so phase 2's `zero_human_mutating_calls` clause read UNMEASURABLE
in EVERY context from that day on — including `board.yml`, the only scheduled
identity that evaluates the ladder at all. A phase-2 exit clause that no
scheduled reader can ever measure is not a slow clause; it is a clause that
cannot turn green, and nothing said so, because `_unmeasurable` is the honest
rendering of exactly that state.

Fixing two of the three instances of one's own class is the failure
`engagement-protocol-policy` §5 names: the fix must survive the class, not the
instance. So this test derives BOTH sides rather than listing either.

## Both sides are derived, so a sixth extraction is caught by the commit

* **The required set** comes from AST call sites of
  :func:`crucible.required.require_env` across `crucible/`, resolving a
  `SOMETHING_VAR` module constant to its literal. Routing a new value through
  that resolver — which is the only way to get the refuse-rather-than-guess
  behaviour — automatically enrols it here.
* **The workflows** are those whose steps run a crucible command with
  `--run-mode live`. That is the property that matters: a job reading or
  writing the production store can reach a required resolver, and a job that
  cannot is not at risk. It is read from the file, so a new live workflow is
  covered on the commit that adds it.

Declaring a variable a given job never reads costs nothing and is the safe
direction to err in; the reverse is a dark board.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "crucible"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: The resolver whose call sites define "required to run at all".
RESOLVER = "require_env"

#: A workflow step that names this is talking to the production store.
LIVE_MARKER = "--run-mode live"


def _required_variables() -> dict[str, list[str]]:
    """Variable name -> the ``crucible/`` files whose call sites require it.

    A call site passing something this cannot resolve to a literal — a
    computed name, a parameter — FAILS the test rather than being skipped: an
    unresolvable required variable is precisely the one that would slip
    through, and silence about it would rebuild the hole this file fills.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        constants = {
            target.id: node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != RESOLVER or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                variable = argument.value
            elif isinstance(argument, ast.Name) and argument.id in constants:
                variable = constants[argument.id]
            else:
                pytest.fail(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} calls {RESOLVER}() with "
                    "a variable name this test cannot resolve to a literal, so it cannot "
                    "check that any workflow declares it. Pass a string literal or a "
                    "module-level constant assigned one."
                )
            if variable.startswith("CRUCIBLE_"):
                found.setdefault(variable, []).append(str(path.relative_to(REPO_ROOT)))
    return found


def _live_workflows() -> list[Path]:
    return sorted(p for p in WORKFLOWS.glob("*.yml") if LIVE_MARKER in p.read_text())


def _declared(text: str) -> set[str]:
    """Variables the workflow binds in an ``env:`` mapping, comments excluded.

    Read with a regex on purpose: the point is what the FILE declares, and a
    YAML load would also accept a name that only appears inside a comment
    block explaining why it is absent — which is how this class survived its
    first repair.
    """
    stripped = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return set(re.findall(r"^\s*(CRUCIBLE_[A-Z0-9_]+):", stripped, flags=re.MULTILINE))


def test_a_required_variable_is_actually_found() -> None:
    """The derivation itself is load-bearing; an empty set would pass vacuously."""
    required = _required_variables()
    assert required, (
        f"no {RESOLVER}() call site was found under crucible/, so every assertion below "
        "would pass over an empty set. Either the resolver was renamed or the required "
        "values stopped going through it — both defeat this file."
    )
    assert _live_workflows(), (
        f"no workflow under .github/workflows/ contains {LIVE_MARKER!r}, so no workflow "
        "would be checked. The live jobs were renamed or removed."
    )


@pytest.mark.parametrize("workflow", _live_workflows(), ids=lambda p: p.name)
def test_live_workflow_declares_every_required_variable(workflow: Path) -> None:
    required = _required_variables()
    declared = _declared(workflow.read_text())
    missing = sorted(set(required) - declared)
    assert not missing, (
        f"{workflow.relative_to(REPO_ROOT)} runs a crucible command with {LIVE_MARKER!r} "
        f"but does not declare {', '.join(missing)}. The package REFUSES to run without "
        f"each of these (see crucible/required.py), so this job fails at the first call "
        "site, or — worse, and this is what actually happened on "
        "`CRUCIBLE_MACHINE_PRINCIPALS` — renders a gate clause UNMEASURABLE forever "
        "while exiting 0. Required by: "
        + "; ".join(f"{v} ({', '.join(required[v])})" for v in missing)
        + f". Fix: add `{missing[0]}: ${{{{ vars.{missing[0]} }}}}` to this workflow's "
        "top-level env, and set the repository variable if it does not exist."
    )
