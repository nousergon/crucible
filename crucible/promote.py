"""`crucible promote <slot>` — move the pointer, record every verdict.

Normative sources: `champion-challenger-policy.md` §5 (promotion), §5.0 (the
anytime-valid sequence and the `promote_min_weeks` delta), §5.2 (free
movement), §5.3 (serving preconditions), §6 (Condorcet retirement), §11 (the
`arena_cycle` artifact); plan §4.4, §4.12, §9.1.

**This module decides nothing statistical.** The ladder, the paired windows,
the confidence sequence, the Condorcet ranking, the pointer decision and the
cap-with-grace retirement rule all live in `nousergon_lib.arena` and are
CALLED (policy §10 makes re-implementing §§3–6 a defect). What crucible adds
is exactly three things the library does not own:

1. **The eligibility age**, Brian's 2026-09-01 ruling: an arm is promotable
   only after `promote_min_weeks` (4) paired weeks against the incumbent —
   **20 paired TRADING days** (§4.12), not 28 calendar ones. It is applied
   *after* the library's decision, as a post-filter, and it can only ever
   turn a MOVE into a HOLD. It never causes a promotion and never changes a
   hold, which is what makes it a delay rather than a second decision rule.

   It is deliberately **not** implemented as a `ServingPrecondition`. A
   precondition excludes an arm from serving at all, so an incumbent inside
   its own first four weeks would force the pointer off itself — the exact
   inversion of a rule meant to slow promotions down.

2. **The durable artifacts**: the `arena_cycle`, the champion pointer, the
   append-only retirement log, and the generated `EXPERIMENTS` feed.

3. **The revert**, which is an operator action and is recorded as one.

Every artifact is written only when a store is supplied. A caller grading in
memory (a test, a `--dry-run`) gets the same decision and writes nothing —
there is no second code path for the dry case, which is how a dry run stays
evidence about the wet one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from nousergon_lib.arena import (
    ArenaCycle,
    ArmRegister,
    ArmSeries,
    PointerDecision,
    ServingPrecondition,
    TrainingStatus,
)
from nousergon_lib.arena.arms import EVENT_RETIRED, ArmEvent, ImmutableArmError
from nousergon_lib.arena.engine import run_cycle

from crucible.arena_io import write_arena_cycle
from crucible.calendar import TRADING_DAYS_PER_WEEK, assert_trading_day
from crucible.champion import (
    PROMOTION_SOURCES,
    ChampionPointer,
    read_champion,
    read_champion_etag,
    write_champion,
)
from crucible.keys import (
    arm_register_key,
    arm_series_key,
    champion_key,
    experiments_key,
    retirement_log_key,
)
from crucible.keys import manifest_key as _promote_manifest_key
from crucible.slots import SlotSpec, is_control_arm
from crucible.store import ETAG_ABSENT, Store

__all__ = [
    "PROMOTION_SOURCES",
    "PromotionRefused",
    "PromotionResult",
    "SlotInputs",
    "apply_eligibility_age",
    # Re-exported from `crucible.keys`, which is the single producer of every
    # key shape (plan §4.12). They appear here because a promote consumer
    # imports them from this module, NOT because promote.py also declares
    # them: `arm_register_key` did, in two places producing the same string,
    # until `alpha-engine-config-I9757` F11 collapsed them; `arm_series_key`
    # did the same until I9784 moved it here too.
    "arm_register_key",
    "arm_series_key",
    "experiments_key",
    "load_slot_inputs",
    "paired_days_required",
    "retirement_log_key",
    "revert_champion",
    "run_promotion",
]


class PromotionRefused(RuntimeError):
    """A promotion or revert that must not be written.

    Always a raise. A refused promotion that returned a value would be
    indistinguishable at the call site from one that happened.
    """


def paired_days_required(spec: SlotSpec) -> int:
    """`promote_min_weeks` expressed in **paired trading days**.

    §4.12 is explicit: "`promote_min_weeks = 4` is 20 paired trading days,
    and a holiday week is still one rung". Counting calendar days here would
    make the bar drift by whatever holidays happened to fall inside the
    window — an eligibility clock nobody declared.
    """
    return spec.promote_min_weeks * TRADING_DAYS_PER_WEEK


@dataclass(frozen=True)
class SlotInputs:
    """Everything one cycle needs, read from the store."""

    register: ArmRegister
    series_by_arm: dict[str, ArmSeries]
    incumbent: str | None
    #: The champion key's version at the moment ``incumbent`` was read, to be
    #: handed back to :func:`run_promotion` so the cycle's write is
    #: conditional on the premise the cycle was decided against. Never
    #: ``None``: an absent pointer is :data:`~crucible.store.ETAG_ABSENT`,
    #: which is the create case and not a third thing to handle.
    pointer_etag: str = ETAG_ABSENT


def load_slot_inputs(store: Store, slot: str) -> SlotInputs:
    """Read the register, every arm's series, and the current pointer.

    **A registered arm with no series raises.** The engine's own contract is
    that every registered arm is scored every cycle and that a missing series
    is a defect rather than an omission; a loader that quietly presented the
    smaller cohort would satisfy the engine while changing what was compared.
    """
    events = [
        json.loads(line)
        for line in store.get_bytes(arm_register_key(slot)).decode().splitlines()
        if line.strip()
    ]
    register = ArmRegister.from_dicts(events)

    series_by_arm: dict[str, ArmSeries] = {}
    for arm_id in register.all_arms():
        try:
            payload = json.loads(store.get_bytes(arm_series_key(slot, arm_id)))
        except KeyError as exc:
            raise KeyError(
                f"arm {arm_id!r} is registered in slot {slot!r} but has no series at "
                f"{arm_series_key(slot, arm_id)!r}. Every registered arm is scored every "
                "cycle (champion-challenger-policy.md §3); a missing series is a "
                "grading defect, and running the cycle without it would silently "
                "change the cohort every verdict rests on."
            ) from exc
        series_by_arm[arm_id] = ArmSeries(
            arm_id=payload["arm_id"],
            scores={k: float(v) for k, v in payload["scores"].items()},
            misses=frozenset(payload.get("misses") or ()),
        )

    # The etag FIRST, then the value it describes. A token captured after
    # the read would be newer than the bytes in hand, and the conditional
    # write at the end of the cycle would sail through and overwrite a
    # pointer this process never saw. Captured first, a concurrent write
    # makes the token stale and the swap fails — the safe direction.
    pointer_etag = read_champion_etag(store, slot)
    incumbent = _current_arm(store, slot)
    return SlotInputs(
        register=register,
        series_by_arm=series_by_arm,
        incumbent=incumbent,
        pointer_etag=pointer_etag,
    )


@dataclass(frozen=True)
class PromotionResult:
    """One cycle's outcome: the artifact, the pointer, and what was written."""

    cycle: ArenaCycle
    decision: PointerDecision
    pointer: ChampionPointer | None
    keys_written: tuple[str, ...]
    eligibility_held: bool
    eligibility_reason: str


# ---------------------------------------------------------------------------
# The eligibility age.
# ---------------------------------------------------------------------------


def apply_eligibility_age(
    *,
    spec: SlotSpec,
    register: ArmRegister,
    decision: PointerDecision,
) -> PointerDecision:
    """Hold the incumbent when the winner has not yet served its 20 paired days.

    Returns ``decision`` unchanged unless the library moved the pointer to an
    arm whose paired window against the incumbent is shorter than
    :func:`paired_days_required`. The one thing this can do is turn a
    ``decided``/``moved`` decision into a ``held`` one; a hold, a bootstrap,
    an ``unmeasurable`` and an ``unservable`` all pass through untouched.
    """
    if not decision.moved or decision.status != "decided":
        return decision
    if decision.champion is None or decision.incumbent is None:
        return decision

    required = paired_days_required(spec)
    comparison = next((c for c in decision.comparisons if c.challenger == decision.champion), None)
    if comparison is None:
        # The library moved the pointer to an arm it recorded no comparison
        # for. That is a library contract violation, not an eligibility
        # question, and guessing an answer here would hide it.
        raise PromotionRefused(
            f"slot {spec.slot}: the pointer moved to {decision.champion!r} but the "
            "decision carries no comparison for it, so the paired-window length "
            "cannot be read. The eligibility age is not evaluable and the promotion "
            "is refused rather than assumed."
        )

    paired = comparison.window.n_dates
    if paired >= required:
        return decision

    created = register.state(decision.champion).record.created_date
    return replace(
        decision,
        champion=decision.incumbent,
        moved=False,
        status="held",
        reason=(
            f"{decision.champion} leads {decision.incumbent} on a supported window of "
            f"{paired} paired trading day(s), but promote_min_weeks="
            f"{spec.promote_min_weeks} requires {required} paired trading days "
            f"(4 paired weeks; Brian ruling 2026-09-01). The arm registered on "
            f"{created} is scored, laddered and reported exactly as any arm, and its "
            "bound is emitted — the pointer simply may not move to it yet. Original "
            f"decision: {decision.reason}"
        ),
    )


# ---------------------------------------------------------------------------
# One cycle, end to end.
# ---------------------------------------------------------------------------


def run_promotion(
    *,
    spec: SlotSpec,
    as_of: str,
    register: ArmRegister,
    series_by_arm: dict[str, ArmSeries],
    incumbent: str | None,
    preconditions: dict[str, tuple[ServingPrecondition, ...]] | None = None,
    training: dict[str, TrainingStatus] | None = None,
    store: Store | None = None,
    pointer_etag: str | None = None,
    manifest_key: str | None = None,
    run_id: str | None = None,
    code_sha: str | None = None,
    attestation: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> PromotionResult:
    """Score the slot, decide the pointer, record every verdict.

    ``training`` is passed straight through to the engine: any active arm
    reporting an unsound fit — or no status at all — raises
    :class:`~nousergon_lib.arena.engine.TrainingIntegrityError` and this
    function does not return. The exception is deliberately not caught: the
    whole slot's run fails (policy §3, Brian ruling 2026-08-29), the runner's
    `finally` writes a `failed` manifest, and no artifact from a compromised
    cycle reaches the store.

    ``pointer_etag`` is the champion key's version at the moment ``incumbent``
    was read — :attr:`SlotInputs.pointer_etag`. The cycle's pointer write is
    conditional on it, so a decision taken against an incumbent that has since
    moved fails loudly instead of clobbering the writer that moved it. A
    caller that does not supply one gets a token read here, which still
    covers the whole scoring window; supplying it from the same read as
    ``incumbent`` covers the gap before that too.

    **The controls never reach the pointer.** §10.1's planted control reads
    next-period returns by construction, so it is injected as a FAILED
    :class:`ServingPrecondition` before the engine decides — the same
    mechanism §5.3 already uses to keep an arm off the serving path while it
    is still scored, laddered and ranked exactly like any other. Excluding it
    at the decision rather than at the write is what makes the exclusion bind:
    the artifact then records *why* the control was not served, rather than a
    decision naming it and a writer silently declining to act on it.
    """
    assert_trading_day(as_of, context=f"promote --slot {spec.slot} as_of")

    cycle = run_cycle(
        config=spec.arena,
        as_of=as_of,
        register=register,
        series_by_arm=series_by_arm,
        incumbent=incumbent,
        preconditions=_with_control_vetoes(spec, series_by_arm, preconditions),
        training=training,
    )

    library_decision = cycle.decision
    decision = apply_eligibility_age(spec=spec, register=register, decision=library_decision)
    held_by_age = decision is not library_decision

    # The artifact carries the decision as it STANDS, not as the library
    # reached it — otherwise the console would render a promotion that did
    # not happen. The library's own reason is preserved inside the amended
    # reason string, so nothing is lost.
    cycle = replace(cycle, decision=decision)

    written: list[str] = []
    pointer: ChampionPointer | None = None
    if store is not None:
        expected = (
            pointer_etag if pointer_etag is not None else read_champion_etag(store, spec.slot)
        )
        written.append(write_arena_cycle(store, cycle))
        written.append(_append_retirement_events(store, spec.slot, as_of, cycle))
        applied = _apply_retirements_to_register(store, spec.slot, as_of, cycle, register)
        if applied is not None:
            written.append(applied)
        written.append(_append_experiment_events(store, as_of, spec, cycle, held_by_age))
        pointer = _write_pointer_if_moved(
            store=store,
            spec=spec,
            cycle=cycle,
            expected=expected,
            manifest_key=manifest_key,
            run_id=run_id,
            code_sha=code_sha,
            attestation=attestation,
            now=now,
        )
        if pointer is not None:
            written.append(champion_key(spec.slot))

    return PromotionResult(
        cycle=cycle,
        decision=decision,
        pointer=pointer,
        keys_written=tuple(written),
        eligibility_held=held_by_age,
        eligibility_reason=decision.reason if held_by_age else "",
    )


def revert_champion(
    *,
    spec: SlotSpec,
    register: ArmRegister,
    store: Store,
    arm_id: str,
    as_of: str,
    operator: str,
    reason: str,
    manifest_key: str | None = None,
    run_id: str | None = None,
    code_sha: str | None = None,
    attestation: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> ChampionPointer:
    """`promote <slot> --revert-to <arm>` — one command, recorded as operator.

    A revert is not a promotion and must never read as one. It writes
    ``promotion_source: operator_bootstrap``, names the operator and the
    reason in the evidence, and appends an event to the EXPERIMENTS feed, so
    the console's "has this pointer ever moved on evidence?" question keeps
    answering honestly (policy §11).

    The write is conditional on the pointer's version as read at the top of
    this call: an operator reverting a pointer that has moved underneath is
    reverting to a premise that no longer holds, and
    :class:`~crucible.store.PointerConflictError` propagates rather than the
    revert overwriting whatever just published.
    """
    assert_trading_day(as_of, context=f"promote --slot {spec.slot} --revert-to {arm_id}")
    if not reason:
        raise PromotionRefused(
            "a revert requires a reason; an unexplained operator override is the one "
            "kind of pointer movement nobody can reconstruct later (principles.md §2.1)"
        )
    if arm_id not in register:
        raise PromotionRefused(
            f"cannot revert slot {spec.slot} to {arm_id!r}: it is not in the register. "
            "An unregistered arm has no score history, so reverting to it would put "
            "the live path on something the harness has never measured."
        )
    # Read the version before the value it describes; see `load_slot_inputs`.
    expected = read_champion_etag(store, spec.slot)
    state = register.state(arm_id)
    if state.retired_date is not None:
        raise PromotionRefused(
            f"cannot revert slot {spec.slot} to {arm_id!r}: it was retired on "
            f"{state.retired_date}. Reverting to a retired arm is a registration "
            "decision, not a pointer decision — register it again if it should serve."
        )

    decided_at = _utc(now)
    pointer = ChampionPointer(
        slot=spec.slot,
        arm_id=arm_id,
        as_of=as_of,
        decided_at=decided_at,
        run_id=run_id or _placeholder_run_id(),
        code_sha=code_sha or "0" * 40,
        promotion_source="operator_bootstrap",
        manifest_key=manifest_key or _promote_manifest_key("promote", as_of),
        evidence={
            "operator": operator,
            "reason": reason,
            "status": "operator_revert",
            "moved": True,
            "incumbent": _current_arm(store, spec.slot),
        },
        attestation=attestation,
    )
    write_champion(store, pointer, expected=expected)
    _append_events(
        store,
        experiments_key(as_of),
        [
            {
                "kind": "operator_revert",
                "slot": spec.slot,
                "as_of": as_of,
                "arm_id": arm_id,
                "operator": operator,
                "reason": reason,
                # The revert's own timestamp, so two genuinely distinct
                # reverts of the same arm on the same day stay two events
                # while a re-entered one collapses to one (`_append_events`).
                "decided_at": decided_at,
            }
        ],
    )
    return pointer


# ---------------------------------------------------------------------------
# Artifact writers.
# ---------------------------------------------------------------------------


CONTROL_VETO = "not_a_control_arm"


def _with_control_vetoes(
    spec: SlotSpec,
    series_by_arm: dict[str, ArmSeries],
    preconditions: dict[str, tuple[ServingPrecondition, ...]] | None,
) -> dict[str, tuple[ServingPrecondition, ...]] | None:
    """Add a FAILED serving precondition to every control arm being scored.

    §10.1 / policy §5.3. A control is scored, laddered and ranked exactly
    like a real arm — that is its entire purpose, since a grader that cannot
    rank planted > real > null is a broken grader — and is barred from one
    thing only: **serving**. A failed precondition is precisely that shape,
    so the exclusion is expressed in the mechanism the policy already has,
    the cycle artifact records the veto with its reason, and no second notion
    of eligibility is invented alongside the library's.

    Returns ``None`` unchanged when there is nothing to add, so a slot with
    no registered controls takes the same path it did before.

    Caller-supplied preconditions are preserved and the veto is appended:
    an arm can fail more than one gate, and dropping the caller's would hide
    a behavioural veto behind a control flag.
    """
    controls = {arm for arm in series_by_arm if is_control_arm(spec, arm)}
    if not controls:
        return preconditions
    merged: dict[str, tuple[ServingPrecondition, ...]] = {
        arm: tuple(checks) for arm, checks in (preconditions or {}).items()
    }
    for arm in sorted(controls):
        merged[arm] = merged.get(arm, ()) + (
            ServingPrecondition(
                name=CONTROL_VETO,
                passed=False,
                reason=(
                    f"{arm} is a slot {spec.slot!r} control arm (§10.1): it is scored "
                    "every cycle to prove the grader ranks planted > real > null, and "
                    "it never serves. The planted control reads next-period returns by "
                    "construction, so promoting it would put a look-ahead arm on the "
                    "one contract the trader reads."
                ),
            ),
        )
    return merged


def _write_pointer_if_moved(
    *,
    store: Store,
    spec: SlotSpec,
    cycle: ArenaCycle,
    expected: str,
    manifest_key: str | None,
    run_id: str | None,
    code_sha: str | None,
    attestation: dict[str, Any] | None,
    now: dt.datetime | None,
) -> ChampionPointer | None:
    decision = cycle.decision
    if decision.champion is None:
        return None
    if decision.status not in ("decided", "bootstrap"):
        return None
    if not decision.moved:
        return None
    if is_control_arm(spec, decision.champion):
        # Unreachable while `_with_control_vetoes` runs ahead of the engine,
        # and kept because it sits on the write to the one contract the
        # trader reads. A RAISE rather than a silent `return None`: if a
        # control ever reaches here the decision layer has stopped excluding
        # it, and a writer that quietly declined would leave the pointer on
        # the previous arm while every artifact recorded a promotion that did
        # not happen — policy §7.2's dominant bug class, self-inflicted.
        raise PromotionRefused(
            f"slot {spec.slot}: the pointer decision names control arm "
            f"{decision.champion!r} as champion. A control arm must never serve "
            "(§10.1): the planted control reads next-period returns, so this would "
            "put a look-ahead arm on champions/{slot}/current.json. The exclusion "
            "belongs at the decision (`_with_control_vetoes`); reaching this guard "
            "means it did not fire."
        )

    comparison = next((c for c in decision.comparisons if c.challenger == decision.champion), None)
    evidence: dict[str, Any] = {
        "incumbent": decision.incumbent,
        "status": decision.status,
        "reason": decision.reason,
        "moved": decision.moved,
        "promote_min_weeks": spec.promote_min_weeks,
        "paired_dates_required": paired_days_required(spec),
        "eligible_arms": sorted(
            arm for arm in cycle.active_arms if arm not in (decision.ineligible or {})
        ),
    }
    if comparison is not None:
        evidence.update(
            {
                "paired_dates": comparison.window.n_dates,
                "window_start": comparison.window.start_date,
                "window_end": comparison.window.end_date,
                "mean_diff": comparison.window.mean_diff if comparison.window.measurable else None,
                "confidence_sequence": comparison.bound.to_dict() if comparison.bound else None,
            }
        )

    pointer = ChampionPointer(
        slot=spec.slot,
        arm_id=decision.champion,
        as_of=decision.as_of,
        decided_at=_utc(now),
        run_id=run_id or _placeholder_run_id(),
        code_sha=code_sha or "0" * 40,
        promotion_source="evidence" if decision.status == "decided" else "bootstrap",
        manifest_key=manifest_key or _promote_manifest_key("promote", decision.as_of),
        evidence=evidence,
        attestation=attestation,
    )
    write_champion(store, pointer, expected=expected)
    return pointer


def _apply_retirements_to_register(
    store: Store, slot: str, as_of: str, cycle: ArenaCycle, register: ArmRegister
) -> str | None:
    """Append this cycle's RETIRE verdicts to the slot's arm register.

    Policy §6 makes retirement automatic, and §6.3 makes ``retired_date`` a
    **queryable fact** rather than an inference. Both require the verdict to
    reach `arms/{slot}/register.jsonl`, because that log is what
    :func:`load_slot_inputs` folds into the register the next cycle runs
    against. Without this, a decided retirement was written to the event log
    and to the EXPERIMENTS feed and to nothing that any later cycle reads:
    the arm stayed in ``active_arms()`` forever, was re-retired every week,
    ``retired_trailing_cycles`` never started counting, and
    :func:`revert_champion`'s ``retired_date is not None`` guard could never
    fire (`alpha-engine-config-I9757`, F3).

    **Idempotent against the STORE, not against the in-memory register.** The
    runner retries a transient failure by re-entering the whole job with a
    fresh context, so a second pass can arrive holding a register loaded
    before the first pass wrote. The already-retired set is therefore folded
    from the bytes now in the store, and an arm the log already retires is
    skipped. Appending a second ``retired`` event would not merely duplicate
    a line — `ArmRegister._fold` raises ``ImmutableArmError`` on it, so the
    slot would be unloadable from the next cycle onward.

    **Appended, never rewritten from the in-memory register.** Rewriting the
    whole log would silently drop any ``registered`` event another writer
    (`experiment.new`) appended since this cycle read it — last-writer-wins
    on a shared log, which is the failure this module exists to refuse.

    Returns the key when something was written, ``None`` when this cycle
    retired nobody, so a caller's `keys_written` names only real writes.
    """
    retiring = [verdict for verdict in cycle.retirements if verdict.retire]
    if not retiring:
        return None

    key = arm_register_key(slot)
    existing = _read_register_events(store, key)
    # Both sources, because they answer the question at different moments:
    # the STORE's log covers a first pass whose write this re-entered pass
    # cannot see in its own register, and the in-memory register covers an
    # arm already retired before this cycle began.
    already_retired = {e["arm_id"] for e in existing if e.get("kind") == EVENT_RETIRED}
    already_retired |= {
        arm for arm in register.all_arms() if register.state(arm).retired_date is not None
    }

    rows: list[dict[str, Any]] = []
    for verdict in retiring:
        if verdict.arm_id in already_retired:
            continue
        rows.append(
            ArmEvent(
                kind=EVENT_RETIRED,
                arm_id=verdict.arm_id,
                date=as_of,
                reason=verdict.reason,
            ).to_dict()
        )
    if not rows:
        return None

    # Fold the log as it WILL be before writing it. The register is the
    # document every later cycle reads; a malformed one is discovered at the
    # next promote run otherwise, a week later and one slot at a time.
    try:
        ArmRegister.from_dicts(existing + rows)
    except ImmutableArmError as exc:
        raise PromotionRefused(
            f"slot {slot}: this cycle's retirements cannot be appended to {key!r} — "
            f"the folded log refuses them ({exc}). The usual cause is that the cycle "
            "ran against a register the store does not hold: `run_promotion` retires "
            "arms in the SAME document `load_slot_inputs` reads, so a caller that "
            "supplied an in-memory register while the store's log is absent or "
            "narrower has no place to record the verdict. Writing the event anyway "
            "would leave a register no later cycle could load."
        ) from exc
    return _append_lines(store, key, rows)


def _read_register_events(store: Store, key: str) -> list[dict[str, Any]]:
    try:
        raw = store.get_bytes(key)
    except KeyError:
        return []
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def _append_retirement_events(store: Store, slot: str, as_of: str, cycle: ArenaCycle) -> str:
    """Append EVERY retirement verdict, survivors included.

    Policy §6.1: "a retirement list containing only retirements cannot be
    audited". The log is append-only — the previous bytes are read and
    re-written ahead of the new lines rather than overwritten — so the
    history of a slot's retirement decisions is reconstructible even when the
    register itself is rebuilt.
    """
    rows = [{"as_of": as_of, "slot": slot, **verdict.to_dict()} for verdict in cycle.retirements]
    return _append_events(store, retirement_log_key(slot), rows)


def _append_experiment_events(
    store: Store,
    as_of: str,
    spec: SlotSpec,
    cycle: ArenaCycle,
    held_by_age: bool,
) -> str:
    """Generate the EXPERIMENTS feed rows for this cycle.

    A challenger that was measured and did not win is a **negative result**,
    and plan §9.1 requires it to reach a durable feed rather than a private
    doc somebody remembers to edit. Retirements and eligibility holds are
    recorded on the same feed, because "why did nothing happen this week" is
    the question the feed exists to answer.
    """
    decision = cycle.decision
    rows: list[dict[str, Any]] = []
    for comparison in decision.comparisons:
        if comparison.challenger == decision.champion and decision.moved:
            rows.append(
                {
                    "kind": "promotion",
                    "slot": spec.slot,
                    "as_of": as_of,
                    "arm_id": comparison.challenger,
                    "incumbent": comparison.incumbent,
                    "status": comparison.status,
                    "reason": comparison.reason,
                    "window": comparison.window.to_dict(),
                }
            )
            continue
        rows.append(
            {
                "kind": "negative_result",
                "slot": spec.slot,
                "as_of": as_of,
                "arm_id": comparison.challenger,
                "incumbent": comparison.incumbent,
                "status": comparison.status,
                "reason": comparison.reason,
                "window": comparison.window.to_dict(),
                "confidence_sequence": comparison.bound.to_dict() if comparison.bound else None,
            }
        )
    if held_by_age:
        rows.append(
            {
                "kind": "eligibility_hold",
                "slot": spec.slot,
                "as_of": as_of,
                "reason": decision.reason,
                "promote_min_weeks": spec.promote_min_weeks,
                "paired_dates_required": paired_days_required(spec),
            }
        )
    for verdict in cycle.retirements:
        if verdict.retire:
            rows.append(
                {
                    "kind": "retirement",
                    "slot": spec.slot,
                    "as_of": as_of,
                    "arm_id": verdict.arm_id,
                    "reason": verdict.reason,
                }
            )
    if not rows:
        # A cycle with no comparisons at all (a single-arm slot, an
        # unservable slot) still emits its shape, because a feed that is
        # silent on a cycle is indistinguishable from a cycle that never ran.
        rows.append(
            {
                "kind": "no_comparison",
                "slot": spec.slot,
                "as_of": as_of,
                "status": decision.status,
                "reason": decision.reason,
            }
        )
    return _append_events(store, experiments_key(as_of), rows)


def _event_id(key: str, row: dict[str, Any]) -> str:
    """A row's identity: the log it belongs to plus its canonical content.

    Every field a cycle row carries is deterministic given (slot, as_of,
    arm), so a re-entered run regenerates byte-identical rows and they
    collapse onto one id. A genuinely distinct event differs in at least one
    field — an operator revert carries its own timestamp for exactly that
    reason — and keeps its own id.
    """
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{key}\x00{canonical}".encode()).hexdigest()[:16]


def _append_events(store: Store, key: str, rows: list[dict[str, Any]]) -> str:
    """Append JSONL ``rows`` to ``key``, once each, preserving what is there.

    Read-modify-write rather than a true append because the `Store`
    interface is four methods on purpose (principle 8) and S3 has no append.

    **Idempotent, because "one writer" was never the same claim as "one
    pass".** The previous version reasoned "single-writer by construction:
    one `promote` job per slot per cycle", which is true about *concurrency*
    and says nothing about *re-entrancy*: `crucible.runner.run_job` retries a
    declared transient class (spot reclamation, provider 5xx, S3 throttle) by
    re-running the job body from the top with a fresh context. A first pass
    that wrote the retirement log and then hit a throttle on the experiments
    write left both logs to be appended again, doubling every row of the
    first — and a duplicated negative result or retirement verdict is a
    record asserting an action happened twice (policy §7.2).

    So each row carries an ``event_id`` derived from the log key and the
    row's canonical content, and a row whose id is already present is
    skipped. The id is written into the row rather than kept in a sidecar:
    the log is the record, and dedupe state that lived anywhere else could
    drift from it.
    """
    existing_raw = b""
    seen: set[str] = set()
    try:
        existing_raw = store.get_bytes(key)
    except KeyError:
        pass
    for line in existing_raw.decode().splitlines():
        if not line.strip():
            continue
        recorded = json.loads(line).get("event_id")
        if recorded is not None:
            seen.add(str(recorded))

    fresh: list[dict[str, Any]] = []
    for row in rows:
        stamped = {**row, "event_id": _event_id(key, row)}
        if stamped["event_id"] in seen:
            continue
        seen.add(stamped["event_id"])
        fresh.append(stamped)
    if not fresh:
        return key
    return _append_lines(store, key, fresh, existing=existing_raw)


def _append_lines(
    store: Store, key: str, rows: list[dict[str, Any]], existing: bytes | None = None
) -> str:
    """Write ``rows`` as JSONL after whatever ``key`` already holds.

    The raw half of :func:`_append_events`, used directly by the arm
    register, whose rows are `nousergon_lib.arena.ArmEvent` documents: that
    schema is the library's, an ``event_id`` this module invented would be
    dropped the first time `crucible.slots.arms.write_register` reserialized
    the log, and the register has a stronger idempotency key of its own — an
    arm the fold already shows as retired.
    """
    if existing is None:
        try:
            existing = store.get_bytes(key)
        except KeyError:
            existing = b""
    payload = existing + b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows
    )
    store.put_bytes(key, payload)
    return key


def _current_arm(store: Store, slot: str) -> str | None:
    """The arm currently pointed at, recorded as the revert's provenance.

    **A declared swallow, with its three fields** (`AGENTS.md`, fail-loud):
    (a) the failure swallowed is `ChampionUnusableError` — the current pointer
    is refused by the reader — plus `KeyError` for no pointer at all;
    (b) the deliverable survives because this value is *provenance on the new
    pointer*, never an input to the revert: a revert is precisely the action
    taken when the current pointer is unusable, so raising here would make the
    recovery command depend on the thing it is recovering from;
    (c) the recording surface is the written pointer's
    `evidence.incumbent`, which reads `null` when the prior pointer could not
    be resolved — visible in the artifact rather than lost.

    Only those two exceptions. A malformed JSON body or an I/O error still
    propagates: those are not "no usable champion", they are a broken store.
    """
    from crucible.champion import ChampionUnusableError

    try:
        return read_champion(store, slot).arm_id
    except KeyError:
        return None
    except ChampionUnusableError:
        # Refused by the reader, but its arm_id is still the honest answer to
        # "what was this pointing at before the revert".
        payload = json.loads(store.get_bytes(champion_key(slot)))
        arm_id = payload.get("arm_id")
        return str(arm_id) if arm_id is not None else None


def _utc(now: dt.datetime | None) -> str:
    moment = now or dt.datetime.now(dt.UTC)
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _placeholder_run_id() -> str:
    """A ULID-shaped id for a decision taken outside `run_job`.

    All zeroes rather than a fresh random id: a caller that bypassed the
    runner produced no manifest, and inventing a plausible-looking run id
    would make an unrecorded decision look recorded.
    """
    return "0" * 26
