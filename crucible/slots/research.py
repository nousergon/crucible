"""The R slot: how names are scored into signals.

Normative source: plan §4.4, table row 2.

**Benchmark: top-N realized alpha against the population drawn from — never
SPY**; rank IC is secondary and is not what moves the pointer.

**`ChallengerShadowGapError` is retired.** In v1, `producers/runner.py`
raised it when any arm failed, which starved every other arm's cohort: one
broken producer took the whole board down, and the workaround was a script
that bypassed the runner. In v2 a failed arm is its own failed row — the
comparison is pairwise on each pair's own window (`I9745`, closed by
construction in `nousergon_lib.arena.window`), so an arm with no shadow
cannot null another arm's figure and there is nothing to gate on.

The one thing that DOES fail the whole slot is a compromised **input**: an
arm whose ranker's declared feature column is absent means the cycle's
inputs are defective, and that is a `TrainingIntegrityError`, not a miss
(plan §4.4, Brian ruling 2026-08-29). "This arm legitimately had nothing to
say" and "this arm's inputs were broken" must never render alike.

**Champion feed:** `signals/{trading_day}/signals.json`.

This module is a binding, not an implementation: the loop is
`crucible.slots.cycle`, shared with U.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crucible.config import Settings
from crucible.keys import signals_key
from crucible.slots.cycle import run_grade, run_produce

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crucible.runner import RunContext

__all__ = ["SLOT", "grade", "produce"]

SLOT = "r"


def produce(ctx: RunContext, *, settings: Settings, **kwargs: Any) -> dict[str, Any]:
    """Run every registered R arm's selection for one trading day."""
    return run_produce(
        ctx,
        slot=SLOT,
        settings=settings,
        feed_key_for=signals_key,
        **kwargs,
    )


def grade(ctx: RunContext, *, settings: Settings, **kwargs: Any) -> dict[str, Any]:
    """Score every settled R selection and run the slot's arena cycle."""
    return run_grade(ctx, slot=SLOT, settings=settings, **kwargs)
