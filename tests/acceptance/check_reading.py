#!/usr/bin/env python3
"""Grade the acceptance gate's READING against the committed ratchet.

The acceptance suite is red by design — plan §6 phase-0 exit requires the §2
clauses written as failing pytest, and they go green one at a time as the
phases land. So "did the suite pass?" is the wrong question to page on: the
answer is no for months, and a check that is red whenever it is working trains
the operator to ignore it.

The right question is whether the READING MOVED, and the reading is THREE SETS
of clause ids, never counts. Counting failed twice, at two different levels: a
regression offset by a gain leaves a count unchanged, and committing `collected`
as a number let any of the met clauses be renamed or deleted with a one-digit
edit while every check read green.

**Failing is not one outcome.** A clause that reads live infrastructure fails
two ways that must never render the same (alpha-engine-config-I9828): UNMET —
the read succeeded and the property does not hold — and UNMEASURABLE — the
read itself failed (no credentials, no region, AccessDenied, an unreachable
endpoint). Both fail the run; neither is ever a pass (principle 7, "no data is
never rendered as green" — and an unreadable clause must not render as a
genuine gap either, because that inflates the denominator with something no
amount of code can fix). `tests/acceptance/test_plan_section_2_objectives.py`'s
`_unmeasurable` tags a failure `UNMEASURABLE — ` in its JUnit `message`, which
this grader reads to sort the clause into the right bucket.

**`unmeasurable` is a SUBSET of `unmet`, not a fourth disjoint set.**
`crucible/gate.py`'s own phase-0 clause reads this same file and requires
`met ∪ unmet` to be every clause the suite defines — that two-bucket contract
predates this file gaining a third outcome and is out of scope here
(alpha-engine-config-I9869 owns `gate.py`). So `unmeasurable` narrows an
existing `unmet` entry rather than replacing it: a clause is always counted
as not-met in the coarse (gate.py) view, and this grader additionally reports
which of the not-met clauses could not be read at all, in the finer view the
step summary and `ratchet.json` both carry.

So the comparison is set-shaped, over collected / unmet / unmeasurable-within-unmet:

* a clause that vanished or appeared — added, renamed, removed, or failing to
  import. A collection error surfaces as an `error` case, not as "no tests ran",
  which is the shape the old grep-based guard could not see;
* a clause in `unmet` (unmeasurable or not) that now passes — progress, and it
  is not recorded until the ratchet moves with it, in the PR that earns it;
* a clause not in `unmet` that now fails and reads fine (property false) — a
  REGRESSION, the only event on this arm that is genuinely someone's fault;
* a clause not in `unmet` that now fails because the read itself broke — same
  effect on the coarse gate, reported separately because it is not a code
  fault;
* a clause whose failure reason moved between plain-unmet and
  unmeasurable-within-unmet without the ratchet moving with it — neither
  progress nor a regression, but still a drifted reading.

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
    """The committed reading: which clauses exist, and their outcome bucket."""

    model_config = ConfigDict(extra="forbid")

    unmet: dict[str, str] = Field(description="clause id -> why it is unmet, with its phase")
    #: A SUBSET of `unmet`'s keys: the read itself failed (credentials,
    #: region, access, network) rather than the property being read and
    #: found false. Still counted in `unmet` for `gate.py`'s coarse met/unmet
    #: view (alpha-engine-config-I9869 owns that file; its phase-0 clause
    #: requires `met | unmet` to equal every clause the suite defines) — this
    #: is a finer-grained annotation on top, for the reading this grader
    #: publishes and compares.
    unmeasurable: dict[str, str] = Field(
        default_factory=dict, description="clause id -> why the read could not happen"
    )
    met: list[str] = Field(description="clause ids that pass today")
    note: str = ""
    last_moved: str = ""

    @model_validator(mode="after")
    def _ids_are_well_formed_and_consistent(self) -> Ratchet:
        import re

        for cid in [*self.unmet, *self.unmeasurable, *self.met]:
            if not re.match(CLAUSE_ID, cid):
                raise ValueError(f"{cid!r} is not a `Class::method` clause id")
        if len(set(self.met)) != len(self.met):
            raise ValueError("`met` lists the same clause twice")
        both = set(self.met) & set(self.unmet)
        if both:
            raise ValueError(f"listed as both met and unmet: {sorted(both)}")
        stray = set(self.unmeasurable) - set(self.unmet)
        if stray:
            raise ValueError(
                f"`unmeasurable` names {sorted(stray)}, not present in `unmet` — "
                "unmeasurable is a subset of unmet, not a fourth bucket"
            )
        for cid, reason in {**self.unmet, **self.unmeasurable}.items():
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
    #: A SUBSET of `unmet`: failed with the `UNMEASURABLE — ` marker — the
    #: read failed, not the property.
    unmeasurable: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def _unmeasurable_is_a_subset_of_unmet(self) -> Reading:
        if not self.collected:
            raise ValueError("the acceptance suite collected no tests — dark, not green")
        stray = self.unmet - self.collected
        if stray:
            raise ValueError(f"unmet clauses that were never collected: {sorted(stray)}")
        stray = self.unmeasurable - self.unmet
        if stray:
            raise ValueError(f"unmeasurable clauses not classified as unmet too: {sorted(stray)}")
        return self

    @property
    def met(self) -> set[str]:
        return self.collected - self.unmet

    @property
    def plain_unmet(self) -> set[str]:
        """`unmet` minus `unmeasurable` — the read succeeded and the property
        does not hold. This is the `M` in `N met / M unmet / K unmeasurable`."""
        return self.unmet - self.unmeasurable


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


#: The marker `_unmeasurable` (test_plan_section_2_objectives.py) puts at the
#: front of its `pytest.fail` message. pytest prefixes JUnit's `<failure
#: message=...>` with its own "Failed: " (or similar), and `--tb=no` still
#: preserves the message — only the traceback body is dropped — so a plain
#: substring search survives both. A dedicated pytest outcome or marker would
#: need a plugin this suite does not otherwise carry, and the marker would be
#: invisible in the raw `pytest -q` terminal output, which this string is not.
UNMEASURABLE_MARKER = "UNMEASURABLE — "


def _is_unmeasurable(case: ET.Element) -> bool:
    for child in case:
        if child.tag not in {"failure", "error"}:
            continue
        haystack = (child.get("message") or "") + (child.text or "")
        if UNMEASURABLE_MARKER in haystack:
            return True
    return False


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
    unmeasurable: set[str] = set()
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
            # Every failing case is `unmet`; `unmeasurable` is the subset
            # whose read itself broke. A `skipped` case carries no marker to
            # find — this repo forbids a skip outcome entirely (a suppression
            # per `tests/test_no_suppressions.py`) — so it lands in plain
            # `unmet`, never silently dropped from the reading.
            unmet.add(cid)
            if _is_unmeasurable(case):
                unmeasurable.add(cid)
    try:
        return Reading(collected=collected, unmet=unmet, unmeasurable=unmeasurable)
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

    met_n, plain_unmet_n, unmeasurable_n = (
        len(reading.met),
        len(reading.plain_unmet),
        len(reading.unmeasurable),
    )
    ratchet_unmeasurable_n = len(ratchet.unmeasurable)
    ratchet_plain_unmet_n = len(ratchet.unmet) - ratchet_unmeasurable_n
    _summary(
        [
            "### Acceptance gate (plan §2)",
            "",
            f"`{met_n} met / {plain_unmet_n} unmet / {unmeasurable_n} unmeasurable` "
            f"(of {len(reading.collected)} clauses) — ratchet says `{len(ratchet.met)} "
            f"met / {ratchet_plain_unmet_n} unmet / {ratchet_unmeasurable_n} unmeasurable`",
            "",
            *(f"- unmet: `{name}`" for name in sorted(reading.plain_unmet)),
            *(f"- unmeasurable: `{name}`" for name in sorted(reading.unmeasurable)),
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

    met_ratchet = set(ratchet.met)
    unmet_ratchet = set(ratchet.unmet)  # unmeasurable ids included, by construction
    unmeasurable_ratchet = set(ratchet.unmeasurable)

    regressed = sorted(reading.plain_unmet & met_ratchet)
    turned_unmeasurable = sorted(reading.unmeasurable & met_ratchet)
    earned = sorted(unmet_ratchet & reading.met)
    reclassified_to_unmeasurable = sorted(
        reading.unmeasurable & (unmet_ratchet - unmeasurable_ratchet)
    )
    reclassified_to_unmet = sorted(reading.plain_unmet & unmeasurable_ratchet)

    if regressed:
        return _fail(
            "REGRESSION: a plan §2 objective that was satisfied no longer is: "
            + ", ".join(regressed)
            + (f" (and {', '.join(earned)} now passes)" if earned else "")
        )
    if turned_unmeasurable:
        return _fail(
            "a plan §2 objective that was satisfied can no longer be READ: "
            + ", ".join(turned_unmeasurable)
            + ". This is not a code regression, but the ratchet no longer describes "
            "what the suite reports — move these from `met` into `unmet` (and add "
            "them to `unmeasurable`) in tests/acceptance/ratchet.json, naming the "
            "read failure."
        )
    if earned:
        return _fail(
            "clauses now pass that ratchet.json still lists as unmet: "
            + ", ".join(earned)
            + ". Progress is not recorded until the ratchet moves with it — drop "
            "them from `unmet` (and `unmeasurable`, and bump `last_moved`) in the "
            "PR that earns them."
        )
    if reclassified_to_unmeasurable or reclassified_to_unmet:
        return _fail(
            "a clause's failure reason changed without the ratchet moving with it: "
            + (
                f"now unmeasurable: {', '.join(reclassified_to_unmeasurable)}. "
                if reclassified_to_unmeasurable
                else ""
            )
            + (
                f"now unmet (readable again): {', '.join(reclassified_to_unmet)}. "
                if reclassified_to_unmet
                else ""
            )
            + "Neither a regression nor progress, but the ratchet no longer describes "
            "what the suite reports — update tests/acceptance/ratchet.json's "
            "`unmeasurable` set in the PR that changes it."
        )
    print(
        f"the reading is unchanged: {met_n} met / {plain_unmet_n} unmet / "
        f"{unmeasurable_n} unmeasurable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
