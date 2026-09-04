"""Typed models for the documents this package READS.

Normative source: `alpha-engine-config-I9847`; Brian, 2026-09-02 — *"I want to
make sure we are using pydantic rather than plain text to ensure we don't run
into any issues down the line."*

This module is the first boundary of that migration and the place its design
is written down. It is deliberately not a 24-module rewrite: `I9847`'s own
constraint is one boundary per PR, each with its contract test, because a
sweeping refactor cannot be adversarially reviewed and the reviews are the
only reason `crucible-PR38`'s defects were found.

── WHY, precisely ────────────────────────────────────────────────────────

The defect is not that untyped reads are ugly. It is **where a malformed
document surfaces**:

* Typed: a named field error at the boundary, naming the document, the row
  and the field.
* Untyped: a `KeyError` three functions in, indistinguishable in a log from
  the reader itself being broken. Measured — `tests/acceptance/check_reading.py`
  read `ratchet["met"]` straight off `json.loads`, and a missing field
  produced exactly that traceback in an adversarial review.

And the sharper half, which no amount of care at the call site fixes: an
untyped read **silently ignores a key it does not know**. A row carrying
`consol_surface:` is not a typo the reader complains about; it is a field
that does nothing, in a file whose whole purpose is to declare what is
observed. `extra="forbid"` is the point of this module at least as much as
the type annotations are.

── WHAT THIS MODULE IS NOT ───────────────────────────────────────────────

**Pydantic does not replace a published JSON Schema.** The versioned schemas
in `crucible/schemas/` are the cross-repo contract surface: other repos read
them, and plan §4.2 and the M0 contract discipline both depend on their being
readable without a Python import. Where an artifact has both, the schema is
**generated from the model** so there is one source of truth rather than two
that must agree — `tests/test_typed_boundaries.py` fails when the committed
file and the generated one differ, which is `I9847` deliverable 3.

**A model is not a validator of semantics.** Cross-field rules that carry a
measured incident in their message — "six rows read `weekly, Saturday` while
the scheduler dispatched exactly one of them" — stay written out, as
`model_validator`s here rather than as hand-rolled checks in the reader. They
move; they are not diluted.

── THE ORDER THE REST LANDS IN ───────────────────────────────────────────

Boundaries are taken in descending order of *how far a malformed document
travels before anything notices*, not in file order. `components.yaml` is
first because it is the declaration that decides what is watched at all: a
row silently missing a field makes a component unobserved, and unobserved
reads exactly like healthy. The remaining boundaries are filed as children of
`alpha-engine-config-I9847`.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ComponentRow",
    "ComponentsDocument",
    "DeadlineRow",
    "RegistryDefaults",
    "SignalsRow",
]


class _Strict(BaseModel):
    """Every model here forbids what it does not declare.

    `extra="forbid"` is the half of this migration that a careful call site
    cannot substitute for: a key the reader does not know is an edit somebody
    made and nothing performed.
    """

    model_config = ConfigDict(extra="forbid")


class SignalsRow(_Strict):
    """§9.2's five signal classes, always all five.

    A class the job legitimately does not emit is `null` — DECLARED, not
    omitted, because an omitted class is indistinguishable from a forgotten
    one. That is why every field is required and nullable rather than
    optional with a default: a default would let an omission read as a
    declaration.
    """

    execution: str | None
    cost: str | None
    resource: str | None
    lineage: str | None
    outcome: str | None


class DeadlineRow(_Strict):
    """The structured deadline `crucible.components.Deadline` is built from.

    Two anchors, exhaustively; a third is a design change visible in a diff,
    for the same reason the two page conditions are a closed set.
    """

    anchor: Literal["close_plus", "next_calendar_day_at"]
    offset_hours: float | None = None
    at: dt.time | None = None

    @model_validator(mode="after")
    def _the_anchor_carries_the_field_it_needs(self) -> DeadlineRow:
        if self.anchor == "close_plus" and self.offset_hours is None:
            raise ValueError("a close_plus deadline needs offset_hours")
        if self.anchor == "next_calendar_day_at" and self.at is None:
            raise ValueError("a next_calendar_day_at deadline needs `at`")
        return self


class ComponentRow(_Strict):
    """One CLI job's observability declaration.

    A component with no row is unobserved, not healthy: its absence cannot
    page (§4.6 reads `deadline` from this file), its logs have no declared
    location or retention, and the console has no surface to render it on.
    """

    description: str = Field(min_length=1)
    lifecycle: Literal["ACTIVE", "DISABLED", "RETIRED"] = "ACTIVE"
    signals: SignalsRow
    log_location: str = Field(min_length=1)
    #: CALENDAR days — one of §4.12's exhaustive exceptions, because retention
    #: is an AWS property billed by calendar time.
    log_retention_days: Annotated[int, Field(ge=1)] | Literal["forever"]
    alert_channel: str = Field(min_length=1)
    console_surface: str = Field(min_length=1)
    artifact_retention: str = Field(min_length=1)
    #: Prose. `dispatch` below is the wiring, and they are separate fields
    #: because six rows read "weekly, Saturday" while exactly one of them was
    #: dispatched by anything.
    schedule: str | None
    #: WHO starts a scheduled row. Required and nullable: `null` is
    #: "on-demand", an assertion, and it must not be spellable by leaving the
    #: key out.
    dispatch: Literal["arc", "scheduler", "github-actions"] | None
    dispatch_workflow: str | None = None
    absence_watched_by: str = "alerts.sweep"
    deadline: DeadlineRow | None

    @model_validator(mode="after")
    def _a_scheduled_row_names_who_starts_it(self) -> ComponentRow:
        """Every scheduled row names its starter; no other row does.

        The failure this prevents is one that shipped: six rows read
        `schedule: weekly, Saturday` while the scheduler dispatched exactly
        one of them, so five components were deadlined, watched for absence,
        and started by nobody.
        """
        scheduled = self.schedule is not None
        if scheduled and self.dispatch is None:
            raise ValueError(
                "is scheduled but its dispatch is null. Something has to start it, "
                "and a row naming no starter is a job whose absence pages every "
                "cycle for work nobody was going to run."
            )
        if not scheduled and self.dispatch is not None:
            raise ValueError(
                f"is on-demand but declares dispatch {self.dispatch!r}; an "
                "unscheduled job is started by a person or another job, and naming "
                "a starter here would claim a cadence it does not have."
            )
        return self

    @model_validator(mode="after")
    def _an_on_demand_row_declares_no_deadline(self) -> ComponentRow:
        if self.schedule is None and self.deadline is not None:
            raise ValueError(
                "is on-demand but declares a deadline; its absence is not a fact "
                "about the system and the deadline would page for nothing."
            )
        return self


class RegistryDefaults(_Strict):
    """§4.6: exactly two page conditions, and one channel either reaches."""

    page_channel: str = Field(min_length=1)
    quiet_channel: str = Field(min_length=1)


class ComponentsDocument(_Strict):
    """`crucible/components.yaml`, whole.

    Read by this package AND, across the repo boundary, by
    `nous-ergon-ops/tests/crossrepo/test_crucible_dispatch_lockstep.py`, which
    is why it is a versioned artifact with a published schema rather than a
    private config file: the M0 contract discipline applies to every
    cross-repo artifact, and this is one.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"$id": "components_registry.v1"})

    version: Literal[1]
    defaults: RegistryDefaults
    components: dict[str, ComponentRow] = Field(min_length=1)
