#!/usr/bin/env python3
"""Grade the acceptance gate's READING against the committed ratchet.

The acceptance suite is red by design — plan §6 phase-0 exit requires the §2
clauses written as failing pytest, and they go green one at a time as the
phases land. So "did the suite pass?" is the wrong question to page on: the
answer is no for months, and a check that is red whenever it is working
trains the operator to ignore it.

The right question is whether the READING MOVED, and the reading is the SET of
unmet clauses, never a count of them. Counting is how a regression hides: one
clause going green while another regresses leaves `met` unchanged, and a
scalar comparison reports "unchanged" on the day something broke. Demonstrated
against an earlier version of this file, which did exactly that.

So the comparison is set-shaped:

* a clause in `unmet` that now passes — progress, and it is not recorded until
  this file moves with it, in the PR that earns it;
* a clause not in `unmet` that now fails — a REGRESSION, and the only event on
  this arm that is genuinely someone's fault;
* a different collected count — a clause was added, removed, or is failing to
  import. A collection error surfaces as an `error` case, not as "no tests
  ran", which is the shape the old grep-based guard could not see.

Exit 0 means the reading is exactly what the repository claims it is. That is
the state in which this job is green while the suite is red, and it is the
distinction the arm exists to make.

Absence is never a pass (principle 7): a missing, unparseable or empty report
is an error, not a silent success.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
RATCHET = HERE / "ratchet.json"


def _fail(message: str) -> int:
    """Emit a GitHub annotation when running in Actions, and always to stderr."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error::{message}")
    print(message, file=sys.stderr)
    return 1


def _summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def clause_id(classname: str, name: str) -> str:
    """`TestAutonomy::test_x` — the class and method, without the module path.

    JUnit's `classname` is the dotted module path plus the class. The module
    path moves when a file is renamed, and a renamed file is not a moved
    reading, so it is dropped.
    """
    return f"{classname.rsplit('.', 1)[-1]}::{name}"


def read_report(report: pathlib.Path) -> tuple[int, set[str]]:
    """Return (collected, set of unmet clause ids) from a JUnit XML report."""
    if not report.exists():
        raise SystemExit(_fail(f"no acceptance report at {report} — the gate is dark, not green"))
    try:
        tree = ET.parse(report)
    except ET.ParseError as exc:
        raise SystemExit(_fail(f"acceptance report {report} does not parse: {exc}")) from exc

    collected = 0
    unmet: set[str] = set()
    seen: set[str] = set()
    for case in tree.getroot().iter("testcase"):
        collected += 1
        cid = clause_id(case.get("classname", "?"), case.get("name", "?"))
        # Two same-named classes in different modules collapse to one id, and
        # then one of them can regress while the other stays unmet: the set is
        # unchanged, the count is unchanged, and the run is green while the
        # true reading dropped. Refuse the ambiguity rather than resolve it —
        # a clause id that is not unique is not an identifier.
        if cid in seen:
            raise SystemExit(
                _fail(
                    f"two acceptance clauses share the id {cid!r}. Clause ids are "
                    "Class::method, so a duplicate makes one clause invisible to "
                    "this grader — rename one of the classes."
                )
            )
        seen.add(cid)
        if any(child.tag in {"failure", "error", "skipped"} for child in case):
            unmet.add(cid)
    if collected == 0:
        raise SystemExit(
            _fail("the acceptance suite collected no tests — the gate is dark, not green")
        )
    return collected, unmet


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return _fail(f"usage: {argv[0]} <junit-xml>")
    collected, unmet = read_report(pathlib.Path(argv[1]))
    ratchet = json.loads(RATCHET.read_text())
    want_collected = int(ratchet["collected"])
    want_unmet = set(ratchet["unmet"])

    _summary(
        [
            "### Acceptance gate (plan §2)",
            "",
            f"`{collected - len(unmet)} of {collected} clauses met` — "
            f"ratchet says `{want_collected - len(want_unmet)} of {want_collected}`",
            "",
            *(f"- unmet: `{name}`" for name in sorted(unmet)),
        ]
    )

    if collected != want_collected:
        return _fail(
            f"the acceptance suite collected {collected} clauses, ratchet.json says "
            f"{want_collected}. A clause was added, removed, or failed to import "
            f"(a collection error reports as an error case, not as 'no tests ran'). "
            f"Update tests/acceptance/ratchet.json in the PR that changes the suite."
        )

    regressed = sorted(unmet - want_unmet)
    earned = sorted(want_unmet - unmet)
    if regressed:
        return _fail(
            "REGRESSION: a plan §2 objective that was satisfied no longer is: "
            + ", ".join(regressed)
            + (f" (and {', '.join(earned)} now passes)" if earned else "")
        )
    if earned:
        return _fail(
            "clauses now pass that ratchet.json still lists as unmet: "
            + ", ".join(earned)
            + ". Progress is not recorded until the ratchet moves with it — drop "
            "them from `unmet` (and bump `last_moved`) in the PR that earns them."
        )
    print(f"the reading is unchanged: {collected - len(unmet)} of {collected} clauses met")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
