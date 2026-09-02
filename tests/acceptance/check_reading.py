#!/usr/bin/env python3
"""Grade the acceptance gate's READING against the committed ratchet.

The acceptance suite is red by design — plan §6 phase-0 exit requires the §2
clauses written as failing pytest, and they go green one at a time as the
phases land. So "did the suite pass?" is the wrong question to page on: the
answer is no for months, and a check that is red whenever it is working trains
the operator to ignore it.

The right question is whether the READING MOVED, and the reading is two SETS of
clause ids, never counts. Counting failed twice, at two different levels: a
regression offset by a gain leaves a count unchanged, and committing `collected`
as a number let any of the met clauses be renamed or deleted with a one-digit
edit while every check read green.

So the comparison is set-shaped:

* a clause that vanished or appeared — added, renamed, removed, or failing to
  import. A collection error surfaces as an `error` case, not as "no tests ran",
  which is the shape the old grep-based guard could not see;
* a clause in `unmet` that now passes — progress, and it is not recorded until
  the ratchet moves with it, in the PR that earns it;
* a clause not in `unmet` that now fails — a REGRESSION, and the only event on
  this arm that is genuinely someone's fault.

Exit 0 means the reading is exactly what the repository claims it is. That is
the state in which this job is green while the suite is red, and it is the
distinction the arm exists to make.

**Both documents are parsed into validated models, not dict indexing.** An
earlier version read `ratchet["met"]` straight off `json.loads` and a missing
field surfaced as a KeyError traceback in the Actions log — indistinguishable
from the grader itself being broken. Absence is never a pass (principle 7), and
a malformed input is an absence: it fails here with the field named.
"""

from __future__ import annotations

import os
import pathlib
import sys
import xml.etree.ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

HERE = pathlib.Path(__file__).resolve().parent
RATCHET = HERE / "ratchet.json"

#: `Class::method`. The module path is deliberately absent: renaming a file is
#: not a moved reading, and JUnit's `classname` carries the dotted module path
#: in front of the class.
CLAUSE_ID = r"^[A-Za-z_][\w]*::[A-Za-z_][\w]*(\[.*\])?$"


class Ratchet(BaseModel):
    """The committed reading: which clauses exist, and which are unmet."""

    model_config = ConfigDict(extra="forbid")

    unmet: dict[str, str] = Field(description="clause id -> why it is unmet, with its phase")
    met: list[str] = Field(description="clause ids that pass today")
    note: str = ""
    last_moved: str = ""

    @model_validator(mode="after")
    def _ids_are_well_formed_and_disjoint(self) -> Ratchet:
        import re

        for cid in [*self.unmet, *self.met]:
            if not re.match(CLAUSE_ID, cid):
                raise ValueError(f"{cid!r} is not a `Class::method` clause id")
        if len(set(self.met)) != len(self.met):
            raise ValueError("`met` lists the same clause twice")
        both = set(self.met) & set(self.unmet)
        if both:
            raise ValueError(f"listed as both met and unmet: {sorted(both)}")
        for cid, reason in self.unmet.items():
            if not reason.strip():
                raise ValueError(f"{cid} carries no reason")
        return self

    @property
    def clauses(self) -> set[str]:
        return set(self.met) | set(self.unmet)


class Reading(BaseModel):
    """What the suite actually reported, parsed out of the JUnit report."""

    model_config = ConfigDict(extra="forbid")

    collected: set[str]
    unmet: set[str]

    @model_validator(mode="after")
    def _unmet_is_a_subset(self) -> Reading:
        if not self.collected:
            raise ValueError("the acceptance suite collected no tests — dark, not green")
        stray = self.unmet - self.collected
        if stray:
            raise ValueError(f"unmet clauses that were never collected: {sorted(stray)}")
        return self


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
    """`TestAutonomy::test_x` — the class and method, without the module path."""
    return f"{classname.rsplit('.', 1)[-1]}::{name}"


def read_report(report: pathlib.Path) -> Reading:
    """Parse a JUnit XML report into a `Reading`."""
    if not report.exists():
        raise SystemExit(_fail(f"no acceptance report at {report} — the gate is dark, not green"))
    try:
        tree = ET.parse(report)
    except ET.ParseError as exc:
        raise SystemExit(_fail(f"acceptance report {report} does not parse: {exc}")) from exc

    collected: set[str] = set()
    unmet: set[str] = set()
    for case in tree.getroot().iter("testcase"):
        cid = clause_id(case.get("classname", "?"), case.get("name", "?"))
        # Two same-named classes in different modules collapse to one id, and
        # one can then regress while the other stays unmet — set unchanged,
        # count unchanged, run green, reading wrong. An id that is not unique
        # is not an identifier, so refuse the ambiguity rather than resolve it.
        if cid in collected:
            raise SystemExit(
                _fail(
                    f"two acceptance clauses share the id {cid!r}. Clause ids are "
                    "Class::method, so a duplicate makes one clause invisible to "
                    "this grader — rename one of the classes."
                )
            )
        collected.add(cid)
        if any(child.tag in {"failure", "error", "skipped"} for child in case):
            unmet.add(cid)
    try:
        return Reading(collected=collected, unmet=unmet)
    except ValidationError as exc:
        raise SystemExit(_fail(f"the acceptance report is not a usable reading: {exc}")) from exc


def load_ratchet(path: pathlib.Path = RATCHET) -> Ratchet:
    try:
        return Ratchet.model_validate_json(path.read_text())
    except ValidationError as exc:
        raise SystemExit(_fail(f"{path.name} is not a valid ratchet: {exc}")) from exc


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return _fail(f"usage: {argv[0]} <junit-xml>")
    reading = read_report(pathlib.Path(argv[1]))
    ratchet = load_ratchet()

    _summary(
        [
            "### Acceptance gate (plan §2)",
            "",
            f"`{len(reading.collected) - len(reading.unmet)} of "
            f"{len(reading.collected)} clauses met` — ratchet says "
            f"`{len(ratchet.met)} of {len(ratchet.clauses)}`",
            "",
            *(f"- unmet: `{name}`" for name in sorted(reading.unmet)),
        ]
    )

    vanished = sorted(ratchet.clauses - reading.collected)
    appeared = sorted(reading.collected - ratchet.clauses)
    if vanished or appeared:
        return _fail(
            "the acceptance suite no longer collects what ratchet.json describes."
            + (f" Gone: {', '.join(vanished)}." if vanished else "")
            + (f" New: {', '.join(appeared)}." if appeared else "")
            + " A clause was added, renamed, removed, or failed to import."
            " Update tests/acceptance/ratchet.json in the PR that changes the suite."
        )

    regressed = sorted(reading.unmet - set(ratchet.unmet))
    earned = sorted(set(ratchet.unmet) - reading.unmet)
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
    print(
        f"the reading is unchanged: {len(reading.collected) - len(reading.unmet)} "
        f"of {len(reading.collected)} clauses met"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
