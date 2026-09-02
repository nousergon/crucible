#!/usr/bin/env python3
"""Grade the acceptance gate's READING against the committed ratchet.

The acceptance suite is red by design — plan §6 phase-0 exit requires the §2
clauses written as failing pytest, and they go green one at a time as the
phases land. So "did the suite pass?" is the wrong question to page on: the
answer is no for months, and a check that is red whenever it is working
trains the operator to ignore it.

The right question is whether the READING MOVED. This script answers it:

* fewer clauses met than `ratchet.json` says — a regression, and the only
  event on this arm that is genuinely someone's fault;
* more clauses met — progress that was not recorded. Bumping the ratchet is
  part of earning the clause, in the same PR, or the number nobody updates
  stops meaning anything;
* a different collected count — a clause was added, removed, or is failing to
  import. A collection error prints `Interrupted`, not `no tests ran`, so a
  count check is the only thing that catches it.

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


def _fail(message: str) -> None:
    """Emit a GitHub annotation when running in Actions, and always to stderr."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error::{message}")
    print(message, file=sys.stderr)


def _summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def read_report(report: pathlib.Path) -> tuple[int, int, list[str]]:
    """Return (met, collected, names of unmet clauses) from a JUnit XML report."""
    if not report.exists():
        raise SystemExit(_die(f"no acceptance report at {report} — the gate is dark, not green"))
    try:
        tree = ET.parse(report)
    except ET.ParseError as exc:
        raise SystemExit(_die(f"acceptance report {report} does not parse: {exc}")) from exc

    cases = tree.getroot().iter("testcase")
    collected = 0
    unmet: list[str] = []
    for case in cases:
        collected += 1
        outcome = [child.tag for child in case if child.tag in {"failure", "error", "skipped"}]
        if outcome:
            unmet.append(f"{case.get('classname', '?')}::{case.get('name', '?')}")
    if collected == 0:
        raise SystemExit(
            _die("the acceptance suite collected no tests — the gate is dark, not green")
        )
    return collected - len(unmet), collected, unmet


def _die(message: str) -> int:
    _fail(message)
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return _die(f"usage: {argv[0]} <junit-xml>")
    met, collected, unmet = read_report(pathlib.Path(argv[1]))
    ratchet = json.loads(RATCHET.read_text())
    want_met, want_collected = int(ratchet["met"]), int(ratchet["collected"])

    _summary(
        [
            "### Acceptance gate (plan §2)",
            "",
            f"`{met} of {collected} clauses met` — ratchet says `{want_met} of {want_collected}`",
            "",
            *(f"- unmet: `{name}`" for name in unmet),
            "",
            "Passing clauses are satisfied objectives. Failing clauses are the",
            "phase definition of done — see `tests/acceptance/README.md`.",
        ]
    )

    if collected != want_collected:
        return _die(
            f"the acceptance suite collected {collected} clauses, ratchet.json says "
            f"{want_collected}. A clause was added, removed, or failed to import "
            f"(a collection error reports as an error case, not as 'no tests ran'). "
            f"Update tests/acceptance/ratchet.json in the PR that changes the suite."
        )
    if met < want_met:
        return _die(
            f"REGRESSION: {met} clauses met, ratchet.json says {want_met}. "
            f"A plan §2 objective that was satisfied no longer is. Unmet now: " + ", ".join(unmet)
        )
    if met > want_met:
        return _die(
            f"{met} clauses met, ratchet.json says {want_met}. Progress is not "
            f"recorded until the ratchet moves with it — bump `met` (and "
            f"`last_moved`, and prune `unmet`) in the PR that earns the clause."
        )
    print(f"the reading is unchanged: {met} of {collected} clauses met")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
