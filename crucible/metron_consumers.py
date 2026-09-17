"""The Metron consumer register — `alpha-engine-config-I10739`.

Component 4 (Metron) reads artifacts that components 1, 2 and 3 produce.
Phase 4 disables the three v1 Step Functions, and measured 2026-09-14 every
one of those artifacts but the two intraday files was produced only inside
them. `architecture.d/146` rule 2 is the ruling that makes this a defect
rather than a coincidence: each component runs on its own schedule and its
own stack, and a component-2 decommission may not silently stop a
component-4 product.

This module reads `metron_consumers.yaml`. The grading lives in
:func:`crucible.gate._clause_metron_reads_have_surviving_producer`; the
analysis lives in the private-docs inventory the document names.

A missing or malformed register RAISES. It does not read as empty: an
unreadable register read as empty would make the set of Metron reads empty
and the phase-4 property over it vacuously true — the register would report
full coverage of nothing, which is the one failure mode this whole guard
exists to prevent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from crucible.models import (
    PHASE4_RETIRED_SCHEDULE_OWNERS,
    MetronConsumerRegisterDocument,
    MetronConsumerRow,
)

__all__ = [
    "METRON_CONSUMER_REGISTER_PATH",
    "PHASE4_RETIRED_SCHEDULE_OWNERS",
    "MetronConsumerRegisterDocument",
    "MetronConsumerRow",
    "load_register",
    "orphaned_by_phase4",
    "paused_rows",
]

METRON_CONSUMER_REGISTER_PATH = Path(__file__).parent / "metron_consumers.yaml"


@lru_cache(maxsize=1)
def load_register() -> MetronConsumerRegisterDocument:
    """Parse and validate `metron_consumers.yaml` as one document."""
    if not METRON_CONSUMER_REGISTER_PATH.is_file():
        raise FileNotFoundError(
            f"the Metron consumer register is missing at "
            f"{METRON_CONSUMER_REGISTER_PATH}. It ships inside the package; its "
            "absence is a broken build, not an empty register."
        )
    raw = yaml.safe_load(METRON_CONSUMER_REGISTER_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{METRON_CONSUMER_REGISTER_PATH} is not a mapping; got {type(raw).__name__}"
        )
    try:
        return MetronConsumerRegisterDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{METRON_CONSUMER_REGISTER_PATH}: {exc}") from exc


def orphaned_by_phase4(
    register: MetronConsumerRegisterDocument,
) -> list[MetronConsumerRow]:
    """Rows phase 4 would leave without a producer, worst case first.

    Two ways to be orphaned, and they are deliberately separate conditions
    rather than one:

    * the producer's schedule is one phase 4 disables and no ruling retired
      the feature — the original finding; and
    * the disposition is `unresolved` whatever the owner — nobody has decided,
      and an undecided artifact grades as undecided. This second condition is
      what makes the guard survive the class instead of the instance: a Metron
      read added tomorrow lands here by DEFAULT, with no edit to this
      function.
    """
    return [
        row
        for row in register.reads
        if row.phase4_disposition == "unresolved"
        or (
            row.schedule_owner in PHASE4_RETIRED_SCHEDULE_OWNERS
            and row.phase4_disposition != "retired_by_ruling"
        )
    ]


def paused_rows(register: MetronConsumerRegisterDocument) -> list[MetronConsumerRow]:
    """Rows whose trigger exists outside the v1 stack but is disabled.

    Not a phase-4 failure — phase 4 neither causes nor fixes an
    administrative pause — and never silently absorbed into a green reading
    either. The clause names these in its detail so "no data" is never
    rendered as the same thing as "fine" (principle 7).
    """
    return [row for row in register.reads if row.schedule_owner == "paused_outside_v1"]
