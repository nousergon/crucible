"""Every attempted configuration, registered before it runs.

Normative source: `alpha-engine-config-I12563` (audit CR-02,
`nous-ergon-ops/audits/2026-10-22-core-ne`), SYSTEM_OPTIMIZED §§4.2, 5.2.

The trial ledger (:mod:`crucible.ledger`) is the DSR denominator for GRADED
arm-cycles: its rows are produced from ``cycle.scored_arms`` after
``grade_slot`` returns. A configuration whose fit raised, whose grade
failed, that was stopped early, or whose worker was killed never reaches
that append, so the family a selection correction is applied to is the
family that SURVIVED — the selection bias the correction exists to price.

This module is the other half. An **attempt** is one configuration of one
family dispatched for one decision date. Its lifecycle is two immutable
events, each its own create-only object (:func:`crucible.keys.attempt_event_key`):

* ``dispatched`` — written BEFORE the work starts, carrying the family and
  config identity, data/code/recipe provenance, and the committed allocation;
* ``terminal`` — written when the work ends, carrying the outcome, its
  reason, the actual cost, and the links to whatever it produced.

An attempt with a ``dispatched`` event and no ``terminal`` one is
**unterminated**: the worker was killed, or is still running. It is reported
as such, never inferred to be a success or dropped, because a killed worker
is exactly the attempt a survivors-only ledger forgets.

**The identity is deterministic** — a digest of (family, config, as_of) — so
a replay of the same decision date re-dispatches the SAME attempt and the
denominator moves once, not once per replay. A replay that finds the
attempt already terminal leaves it terminal; a replay that reaches a
DIFFERENT terminal outcome raises :class:`AttemptConflictError` rather than
rewriting history.

**Create-only, never read-modify-write.** Two workers recording two attempts
touch two disjoint keys, so neither can lose the other's event (the defect
`alpha-engine-config-I12562` found in the trial ledger's rewrite). The trial
ledger keeps its per-arm-cycle grade history for its existing purpose; this
module does not redefine an arena cycle as a search attempt, it LINKS to one
through the terminal event's ``arena_cycle_key``.

**Unknown is named, never guessed.** A cost or provenance field this
process cannot measure is written as ``{"status": "unavailable", "reason":
...}``, so a reader can tell "zero" from "not measured".
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crucible.documents import load_store_document
from crucible.keys import ATTEMPTS_PREFIX, attempt_event_key
from crucible.store import ETAG_ABSENT, PointerConflictError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from crucible.store import Store

__all__ = [
    "ATTEMPT_SCHEMA",
    "OUTCOMES",
    "OUTCOME_EARLY_STOPPED",
    "OUTCOME_FAILED",
    "OUTCOME_KILLED",
    "OUTCOME_SUCCEEDED",
    "STATE_UNTERMINATED",
    "Attempt",
    "AttemptConflictError",
    "AttemptLifecycle",
    "attempt_id",
    "dispatch_attempt",
    "family_attempts",
    "n_attempts",
    "read_attempts",
    "terminate_attempt",
    "unavailable",
]

ATTEMPT_SCHEMA = "attempt_event.v1"

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"
OUTCOME_EARLY_STOPPED = "early_stopped"
OUTCOME_KILLED = "killed"
OUTCOMES = (OUTCOME_SUCCEEDED, OUTCOME_FAILED, OUTCOME_EARLY_STOPPED, OUTCOME_KILLED)

#: The state of an attempt that was dispatched and never reported an end.
STATE_UNTERMINATED = "unterminated"


class AttemptConflictError(RuntimeError):
    """A second, DIFFERENT event for a phase that has already landed."""


def unavailable(reason: str) -> dict[str, str]:
    """The explicit marker for a value this process cannot measure."""
    return {"status": "unavailable", "reason": reason}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def attempt_id(*, family: str, config_id: str, as_of: str) -> str:
    """The deterministic identity of one configuration attempted for one date.

    A digest, not a fresh ULID: a replay of the same decision date must map
    to the SAME attempt, or every replay would add a hypothesis nobody new
    proposed and the multiplicity correction would drift with retry count.
    """
    for name, value in (("family", family), ("config_id", config_id), ("as_of", as_of)):
        if not value:
            raise ValueError(f"an attempt's {name} is required")
    canonical = json.dumps(
        {"family": family, "config_id": config_id, "as_of": as_of},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _create_once(store: Store, key: str, event: dict[str, Any], *, same: tuple[str, ...]) -> bool:
    """Create ``key`` with ``event``; True if this call wrote it.

    If the object already exists, it is accepted only when it agrees with
    ``event`` on the ``same`` fields — that is a replay of the event that
    already landed. Any disagreement raises: the phase is immutable.
    """
    payload = json.dumps(event, sort_keys=True).encode("utf-8")
    try:
        store.compare_and_swap(key, ETAG_ABSENT, payload)
    except PointerConflictError:
        prior = load_store_document(store, key)
        disagree = {f: (prior.get(f), event.get(f)) for f in same if prior.get(f) != event.get(f)}
        if disagree:
            raise AttemptConflictError(
                f"{key!r} already records a different event; this call would change "
                f"{sorted(disagree)} ({disagree}). An attempt's lifecycle events are "
                "immutable once written — a replay that reaches a different outcome is a "
                "finding to investigate, not a correction to apply."
            ) from None
        return False
    return True


def dispatch_attempt(
    store: Store,
    *,
    family: str,
    config_id: str,
    as_of: str,
    run_id: str,
    control: bool = False,
    provenance: Mapping[str, Any] | None = None,
    committed_cost: Mapping[str, Any] | None = None,
) -> str:
    """Record that ``config_id`` of ``family`` is about to run. Returns its id.

    Call it BEFORE the work: an attempt registered afterwards is registered
    only if it survived, which is the bias this module exists to remove.
    Idempotent for a replay of the same (family, config, as_of).
    """
    aid = attempt_id(family=family, config_id=config_id, as_of=as_of)
    event = {
        "schema_version": ATTEMPT_SCHEMA,
        "phase": "dispatched",
        "attempt_id": aid,
        "family": family,
        "config_id": config_id,
        "as_of": as_of,
        "control": bool(control),
        "provenance": dict(provenance) if provenance is not None else {},
        "committed_cost": (
            dict(committed_cost)
            if committed_cost is not None
            else unavailable("no allocation is declared for this attempt")
        ),
        "run_id": run_id,
        "written_at_utc": _utc_now(),
    }
    _create_once(
        store,
        attempt_event_key(aid, "dispatched"),
        event,
        same=("family", "config_id", "as_of", "control"),
    )
    return aid


def terminate_attempt(
    store: Store,
    aid: str,
    *,
    outcome: str,
    reason: str,
    run_id: str,
    actual_cost: Mapping[str, Any] | None = None,
    links: Mapping[str, Any] | None = None,
) -> bool:
    """Record how attempt ``aid`` ended. True if this call wrote the event.

    Refuses an attempt that was never dispatched: a terminal event with no
    dispatch would count an outcome whose configuration nobody registered.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"an attempt's outcome is one of {OUTCOMES}, not {outcome!r}")
    if not reason:
        raise ValueError("an attempt's terminal event carries a reason")
    if not store.exists(attempt_event_key(aid, "dispatched")):
        raise AttemptConflictError(
            f"attempt {aid} has no dispatched event; register it with dispatch_attempt "
            "before the work starts, not after it ends"
        )
    event = {
        "schema_version": ATTEMPT_SCHEMA,
        "phase": "terminal",
        "attempt_id": aid,
        "outcome": outcome,
        "reason": reason,
        "actual_cost": (
            dict(actual_cost)
            if actual_cost is not None
            else unavailable("this process did not measure the attempt's cost")
        ),
        "links": dict(links) if links is not None else {},
        "run_id": run_id,
        "written_at_utc": _utc_now(),
    }
    return _create_once(store, attempt_event_key(aid, "terminal"), event, same=("outcome",))


@dataclass(frozen=True)
class Attempt:
    """One attempt, folded from its events."""

    attempt_id: str
    dispatched: dict[str, Any]
    terminal: dict[str, Any] | None = field(default=None)

    @property
    def family(self) -> str:
        return str(self.dispatched["family"])

    @property
    def config_id(self) -> str:
        return str(self.dispatched["config_id"])

    @property
    def control(self) -> bool:
        return bool(self.dispatched.get("control"))

    @property
    def state(self) -> str:
        """The terminal outcome, or :data:`STATE_UNTERMINATED`."""
        if self.terminal is None:
            return STATE_UNTERMINATED
        return str(self.terminal["outcome"])


def read_attempts(store: Store) -> list[Attempt]:
    """Every attempt in the store, ordered by id. Unterminated ones included.

    A terminal event with no dispatched event is an orphan this module never
    writes; it is refused here rather than silently counted or dropped.
    """
    events: dict[str, dict[str, dict[str, Any]]] = {}
    for key in store.list_keys(ATTEMPTS_PREFIX):
        rest = key[len(ATTEMPTS_PREFIX) :]
        aid, _, leaf = rest.partition("/")
        phase = leaf.removesuffix(".json")
        if phase not in ("dispatched", "terminal") or key != attempt_event_key(aid, phase):
            raise ValueError(f"{key!r} is not an attempt event key")
        events.setdefault(aid, {})[phase] = load_store_document(store, key)
    out: list[Attempt] = []
    for aid in sorted(events):
        phases = events[aid]
        if "dispatched" not in phases:
            raise ValueError(
                f"attempt {aid} has a terminal event and no dispatched event; the family "
                "and config it ended cannot be counted"
            )
        out.append(Attempt(aid, phases["dispatched"], phases.get("terminal")))
    return out


def family_attempts(store: Store, family: str, *, include_controls: bool = False) -> list[Attempt]:
    """The COMPLETE declared family a selection correction consumes.

    Every attempt dispatched for ``family``, whatever its outcome — failed,
    stopped early and unterminated attempts included — because a correction
    applied to the survivors alone is the bias it exists to price.
    """
    return [
        a
        for a in read_attempts(store)
        if a.family == family and (include_controls or not a.control)
    ]


def n_attempts(store: Store, *, family: str | None = None, include_controls: bool = False) -> int:
    """Distinct configurations ATTEMPTED — the survivorship-free denominator.

    Distinct (family, config), not attempts: a configuration attempted on
    forty dates is one hypothesis tried forty times, the same rule
    :func:`crucible.ledger.n_trials` applies to graded rows. Controls are
    excluded unless asked for, for the same reason.
    """
    return len(
        {
            (a.family, a.config_id)
            for a in read_attempts(store)
            if (family is None or a.family == family) and (include_controls or not a.control)
        }
    )


class AttemptLifecycle:
    """The attempts one job dispatched, so it can end all of them in one place.

    A job dispatches as it learns which configurations it will run, then
    calls :meth:`succeed` on the way out, or :meth:`end_open` from an
    exception handler. Whatever a job never ends — a SIGKILL — stays
    unterminated in the store, which is the point.
    """

    def __init__(self, store: Store, *, family: str, as_of: str, run_id: str) -> None:
        self.store = store
        self.family = family
        self.as_of = as_of
        self.run_id = run_id
        self._open: dict[str, str] = {}

    @property
    def open_attempts(self) -> dict[str, str]:
        """config_id -> attempt_id for every attempt not yet ended here."""
        return dict(self._open)

    def dispatch(
        self,
        config_ids: Iterable[str],
        *,
        control_ids: Iterable[str] = (),
        provenance: Mapping[str, Any] | None = None,
        committed_cost: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        controls = set(control_ids)
        for config_id in sorted(set(config_ids)):
            if config_id in self._open:
                continue
            self._open[config_id] = dispatch_attempt(
                self.store,
                family=self.family,
                config_id=config_id,
                as_of=self.as_of,
                run_id=self.run_id,
                control=config_id in controls,
                provenance=provenance,
                committed_cost=committed_cost,
            )
        return self.open_attempts

    def _end(self, config_id: str, outcome: str, reason: str, links: Mapping[str, Any]) -> None:
        aid = self._open.pop(config_id)
        terminate_attempt(
            self.store,
            aid,
            outcome=outcome,
            reason=reason,
            run_id=self.run_id,
            # Not metered here, and not guessed: the job's wall-clock timings
            # live on its run manifest (named by `run_id`), and a per-attempt
            # number in this event would make a replay's bytes differ from
            # the run it replays. Named unavailable until a metered source
            # exists for the attempt itself.
            actual_cost=unavailable(
                "per-attempt cost is not metered; the job's timings are on the run "
                "manifest this event's run_id names"
            ),
            links=links,
        )

    def succeed(self, reason: str, *, links: Mapping[str, Any] | None = None) -> None:
        for config_id in sorted(self._open):
            self._end(config_id, OUTCOME_SUCCEEDED, reason, links or {})

    def end_open(self, outcome: str, reason: str) -> None:
        """End every attempt still open with ``outcome``. Never raises over the cause.

        Called from an exception handler, so a failure to record the end is
        reported on the attempt's absence (it stays unterminated) rather than
        by replacing the exception that actually ended the job.
        """
        for config_id in sorted(self._open):
            try:
                self._end(config_id, outcome, reason, {})
            except Exception:  # noqa: BLE001 - see docstring: the original error wins
                self._open.pop(config_id, None)
