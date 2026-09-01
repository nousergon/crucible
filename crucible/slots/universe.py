"""The U slot: which names reach the predictor.

Normative source: plan §4.4, table row 1.

The first draft of the plan folded the universe cut into the data layer; the
champion/challenger policy is explicit that it is its own slot with its own
scoring path, and conflating it with the selection producer produces a
meaningless number. So it is a slot, with a register, a ladder, controls and
a pointer of its own.

**Benchmark: the population it drew from, count-matched. Never SPY.** The
library's `ArenaConfig` refuses SPY for a selection-stage slot outright —
on 2026-08-17 arms were graded against SPY while SPY trailed the population
they were drawn from by 140bp at 21 sessions, which inverted wins and losses.

**Champion feed:** `universe/{trading_day}/members.json`. The serving path
resolves the pointer and writes the pointed-to arm's selection; nothing
imports a ranking function directly.

This module is a binding, not an implementation: the loop is
`crucible.slots.cycle`, shared with R, so one policy has one implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crucible.config import Settings
from crucible.keys import universe_members_key
from crucible.slots.cycle import run_grade, run_produce

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crucible.runner import RunContext

__all__ = ["SLOT", "grade", "produce"]

SLOT = "u"


def produce(ctx: RunContext, *, settings: Settings, **kwargs: Any) -> dict[str, Any]:
    """Run every registered U arm's cut for one trading day."""
    return run_produce(
        ctx,
        slot=SLOT,
        settings=settings,
        feed_key_for=universe_members_key,
        **kwargs,
    )


def grade(ctx: RunContext, *, settings: Settings, **kwargs: Any) -> dict[str, Any]:
    """Score every settled U cut and run the slot's arena cycle."""
    return run_grade(ctx, slot=SLOT, settings=settings, **kwargs)
