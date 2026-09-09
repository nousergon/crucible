"""The one place a REQUIRED environment variable is resolved, and the one
place the set of them can be derived from.

Normative source: `crucible/AGENTS.md` (Visibility) and
`alpha-engine-config-I10156`. This repository is public and carries no
infrastructure identifier as a literal, so several values a live gate reads —
topic names — arrive only as environment. Each of them RAISES when unset
rather than falling back, because every fallback available is silently
wrong: a page published to a guessed topic reaches nobody.

**Why the resolution is centralised rather than open-coded per module.**
`alpha-engine-config-I10156` moved five role names and two topic names out of
the tree and into required environment in one commit. Nothing then declared
them to a workflow, and `Board` and `Gate close` both went dark on
`RuntimeError: CRUCIBLE_MUTED_TOPIC is unset` — on the two days phase 0 and
phase 1 turned MET. The repair declared the two TOPIC names to those
workflows and missed `CRUCIBLE_MACHINE_PRINCIPALS`, extracted by the same
commit, so phase 2's `zero_human_mutating_calls` clause read UNMEASURABLE in
every context for two days, including the only scheduled identity that
evaluates the ladder. A repair that fixes two of the three instances of its
own class is the failure `engagement-protocol-policy` §5 names.

So the class is closed at the source instead: every required variable is
resolved through :func:`require_env`, and
`tests/test_workflow_required_env.py` derives the call sites by AST and fails
when a workflow that evaluates live gates does not declare one of them. A
sixth extraction is caught by that test on the commit that makes it, not by a
dark board a week later.

**The machine-principal allowlist no longer routes through this resolver**
(`alpha-engine-config-I10307`, corrected 2026-09-09): the fifth extraction
named above was itself the hand-kept-twin defect one layer up —
`CRUCIBLE_MACHINE_PRINCIPALS` was a comma-separated list an operator had to
keep in sync with the `crucible-v2` stack by hand, and it drifted the moment
the stack grew past the five roles it was extracted from
(`crucible.autonomy.machine_principals` derives it from the stack's own
`AWS::IAM::Role` resources instead). This module's "several values" is
therefore topic names only; a value that can be read from a source the
account itself keeps truthful belongs there, not here.
"""

from __future__ import annotations

import os

__all__ = ["require_env"]


def require_env(variable: str, *, refusing_to: str) -> str:
    """The value of ``variable``, or a ``RuntimeError`` naming what was refused.

    ``refusing_to`` completes the sentence "refusing to ..." and is what makes
    the failure legible to whoever finds it in a workflow log: the reader
    needs to know which wrong answer we declined to invent, not merely that a
    name is absent.
    """
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(
            f"{variable} is unset. This repository is public and carries no "
            f"infrastructure identifier as a literal; refusing to {refusing_to}."
        )
    return value
