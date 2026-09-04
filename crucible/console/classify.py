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
    unreadable: str | None = None,
) -> Classification:
    """Place one component in exactly one state.

    ``manifest`` is the component's manifest for the current trading day, or
    None when there is none. ``unreadable`` is set instead when a manifest for
    this component EXISTS and could not be read — the fault, naming the key.
    That is a third input, not a variant of ``manifest=None``: an unreadable
    manifest is a producer we cannot trust, and rendering it as "no run today"
    would send an operator to the trigger when the artifact is the problem
    (`alpha-engine-config-I9900`). ``history`` says whether the component has ever
    produced a manifest — that is the whole difference between `MISSED` (its
    schedule fired and no run started) and `NEVER_RAN` (registered, in
    service, never executed, so its first failure is still ahead of it), and
    collapsing the two reports a defect as a decision.

    The branch order is load-bearing. Declared lifecycle first, because a
    DISABLED component must not be evaluated against a deadline it was
    deliberately taken off. Then the unreadable branch, because a fault we
    cannot read past outranks both the evidence we do not have and the
    absence we would otherwise infer. Then evidence, then the absence
    branches, then
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

    # 2. An artifact exists and we could not read it. Ahead of every evidence
    #    and absence branch because it is the one case where we know something
    #    was filed and know nothing about what it says. UNREPORTED, which is
    #    red and counts toward the §8.4 transparency gap: principle 7 — a
    #    component whose record we cannot parse is unobserved, never healthy,
    #    and never quietly reported as not having run.
    if unreadable:
        return Classification(
            component.name,
            "UNREPORTED",
            f"a manifest for this trading day exists and could not be read: {unreadable}. "
            "Distinct from MISSED: something was filed, so the failure is in the "
            "artifact or in our access to it, not in the trigger.",
        )

    # 3. Evidence. A manifest exists, so the run's own record decides.
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
            # alpha-engine-config-I9757 (C5): the run's own exit status is
            # not the whole story — its metrics can carry no value even
            # though the run "succeeded". Reading only `manifest["status"]`
            # let a drift cycle whose inputs were structurally present but
            # empty (feature_psi_max_ratio and ic_decay_ratio both
            # UNREPORTED) render HEALTHY, which is principle 7 violated:
            # a component emitting nothing is unobserved, not healthy.
            metrics = manifest.get("metrics") or []
            unreported_metrics = [
                m.get("name", "?") for m in metrics if m.get("status") == "UNREPORTED"
            ]
            # alpha-engine-config-I9798 (round 2, adversarial review): a
            # `status: ok` manifest can carry a metric whose OWN status is
            # `BREACH` — `crucible.release_lock_sweep`'s
            # `release_objects_unlocked` inside `alerts.sweep` is the first
            # producer of one — and reading only `manifest["status"]` (or
            # only the UNREPORTED branch above) let it render HEALTHY: a
            # correctness failure hidden inside a run that otherwise
            # succeeded, the same C5 shape as the UNREPORTED case, one
            # status word over.
            breached_metrics = [m.get("name", "?") for m in metrics if m.get("status") == "BREACH"]
            if metrics and len(unreported_metrics) == len(metrics):
                # Total blindness: the run reports OK and declared metrics,
                # and every one of them carries no value. Its own status is
                # not evidence when nothing behind it was actually
                # measured, so this renders the same as an unreadable
                # producer — UNREPORTED, never green. (`drift_metrics`
                # already refuses to let this combination reach the
                # manifest for the drift job specifically, by raising
                # before the caller can record `status: ok`; this branch is
                # the systemic backstop for every other metric-emitting
                # component, present or future, that has not been given
                # the same producer-side guard.)
                return Classification(
                    component.name,
                    "UNREPORTED",
                    f"ran and ended ok, but all {len(metrics)} of its declared "
                    f"metric(s) carry no value ({', '.join(unreported_metrics)}). A "
                    "run that measured nothing is not evidence of health, whatever "
                    "its own exit status claims.",
                    trading_day=day,
                    run_id=run_id,
                )
            if breached_metrics or unreported_metrics:
                # Partial blindness, a breached ceiling, or both: the run
                # did measure something, so this is not the total-blindness
                # case above — but it is not a clean HEALTHY either, or the
                # breached/silent metric is invisible on the one row that
                # owns it.
                clauses = []
                if breached_metrics:
                    clauses.append(
                        f"{len(breached_metrics)} declared metric(s) breached their "
                        f"ceiling ({', '.join(breached_metrics)})"
                    )
                if unreported_metrics:
                    clauses.append(
                        f"{len(unreported_metrics)} of its {len(metrics)} declared "
                        f"metric(s) carry no value ({', '.join(unreported_metrics)})"
                    )
                return Classification(
                    component.name,
                    "DEGRADED",
                    f"ran inside its declared window and ended ok{note}, but "
                    + "; and ".join(clauses)
                    + ".",
                    trading_day=day,
                    run_id=run_id,
                )
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

    # 4. No manifest. The four absence branches, each distinct on the surface.
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
