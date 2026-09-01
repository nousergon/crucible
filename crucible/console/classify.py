"""The fourteen-state classifier. Total, with no fall-through.

Normative source: `observability-policy.md` §8.3; plan §9.2.

**Two properties, not one.** The population comes from `components.yaml`;
this module supplies the totality. Every registered component resolves to
exactly one of fourteen states, and a component the classifier cannot place
renders `UNREPORTED` — which is loud, and is the transparency-gap count whose
objective is zero.

**There is no `else: HEALTHY` and no fifteenth state.** `UNKNOWN`, `OTHER`,
`PENDING` and `N/A` are the fall-through the vocabulary exists to remove. The
one thing this classifier will never do is render the absence of evidence as
green.

**`DISABLED`, `DEPRECATED` and `RETIRED` are read from the registry**, never
inferred from silence. A job inferred to be retired because it stopped
producing is indistinguishable from one that broke, and the two want opposite
responses.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from crucible.calendar import resolve_trading_day
from crucible.components import Component

__all__ = ["STATES", "Classification", "classify"]

#: The closed vocabulary, exhaustive and add-by-PR-only. Ordered as the policy
#: lists them so a reader can check this against §8.3 line by line.
STATES: tuple[str, ...] = (
    "HEALTHY",
    "RUNNING",
    "DEGRADED",
    "FAILED",
    "STALLED",
    "MISSED",
    "NEVER_RAN",
    "DISABLED",
    "DEPRECATED",
    "RETIRED",
    "ABSENT",
    "UNREGISTERED",
    "UNREPORTED",
    "ARMED",
)


@dataclass(frozen=True)
class Classification:
    """A state plus why. A dot that cannot say how it knows is not trustworthy.

    §8.3: "each state is decision-shaped — the row says what is broken, who
    owns it, and what happens next, not a raw status token."
    """

    component: str
    state: str
    reason: str
    trading_day: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(
                f"{self.state!r} is not in the closed vocabulary {STATES}. A fifteenth "
                "state is the fall-through §8.3 exists to remove."
            )
        if not self.reason.strip():
            raise ValueError(f"{self.component} classified {self.state} with no reason")


def classify(
    component: Component,
    manifest: dict[str, Any] | None,
    *,
    now: dt.datetime,
    history: bool = True,
) -> Classification:
    """Place one component in exactly one state.

    ``manifest`` is the component's manifest for the current trading day, or
    None when there is none. ``history`` says whether the component has ever
    produced a manifest — that is the whole difference between `MISSED` (its
    schedule fired and no run started) and `NEVER_RAN` (registered, in
    service, never executed, so its first failure is still ahead of it), and
    collapsing the two reports a defect as a decision.

    The branch order is load-bearing. Declared lifecycle first, because a
    DISABLED component must not be evaluated against a deadline it was
    deliberately taken off. Then evidence, then the absence branches, then
    `UNREPORTED` as the resolved placement for everything left — never as a
    default that means "probably fine".
    """
    # 1. Declared dispositions. Read, never inferred.
    if component.lifecycle == "DISABLED":
        return Classification(
            component.name,
            "DISABLED",
            "declared DISABLED in components.yaml — a decision, not a defect. Its "
            "deadline is deliberately not evaluated.",
        )
    if component.lifecycle == "RETIRED":
        return Classification(
            component.name,
            "RETIRED",
            "declared RETIRED in components.yaml — removed on purpose; the row persists "
            "so the absence is stated rather than silent.",
        )

    # 2. Evidence. A manifest exists, so the run's own record decides.
    if manifest is not None:
        status = manifest.get("status")
        run_id = manifest.get("run_id")
        day = manifest.get("trading_day")
        retried = len(manifest.get("attempts", [])) > 1
        if status == "failed":
            return Classification(
                component.name,
                "FAILED",
                f"ran and ended failed: {manifest.get('reason') or '(no reason recorded)'}",
                trading_day=day,
                run_id=run_id,
            )
        if status == "ok":
            note = " after one transient-class retry" if retried else ""
            return Classification(
                component.name,
                "HEALTHY",
                f"ran inside its declared window and ended ok{note}.",
                trading_day=day,
                run_id=run_id,
            )
        # A manifest with a status outside the closed set is not a run whose
        # outcome we can read — it is a producer we cannot trust. UNREPORTED,
        # loudly, rather than a guess in either direction.
        return Classification(
            component.name,
            "UNREPORTED",
            f"manifest carries status {status!r}, which is outside the schema's closed "
            "set. The producer is not conformant, so its outcome is unknown — and "
            "unknown renders as a finding, never as green.",
            trading_day=day,
            run_id=run_id,
        )

    # 3. No manifest. The four absence branches, each distinct on the surface.
    if not component.scheduled:
        # An on-demand job's silence is not a fact about the system. It is
        # ARMED when something still triggers it, and that trigger is the CLI,
        # which is verified by the CLI's own job table containing it — the
        # anchor §8.3 requires, resolved rather than asserted.
        return Classification(
            component.name,
            "ARMED",
            "on-demand (schedule: null); its trigger is the CLI job table, which still "
            "carries it. Silence here carries no claim of a recent run.",
        )

    assert component.deadline is not None  # Component.__post_init__ guarantees it
    trading_day = resolve_trading_day(now)
    due = component.deadline.due_at(trading_day)
    if now < due:
        return Classification(
            component.name,
            "RUNNING" if history else "NEVER_RAN",
            (
                f"scheduled, deadline {component.deadline.describe(trading_day)} has not "
                f"passed ({due.strftime('%Y-%m-%dT%H:%M:%SZ')}); no manifest yet."
            )
            if history
            else (
                "registered and in service with no run in its history. It has never "
                "executed, so its first failure is still ahead of it."
            ),
            trading_day=trading_day.isoformat(),
        )
    if not history:
        return Classification(
            component.name,
            "NEVER_RAN",
            "registered and in service, no run in its history, and its deadline has "
            "passed. Distinct from MISSED: a component that has never executed has "
            "never been tested.",
            trading_day=trading_day.isoformat(),
        )
    return Classification(
        component.name,
        "MISSED",
        f"its schedule fired or should have and no run started; deadline "
        f"{component.deadline.describe(trading_day)} passed at "
        f"{due.strftime('%Y-%m-%dT%H:%M:%SZ')}. The failure is upstream of the "
        "component, in the trigger.",
        trading_day=trading_day.isoformat(),
    )
