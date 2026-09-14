"""`crucible experiment.backfill --slot --arm --from --to` — one arm's history.

`alpha-engine-config-I10696` (Brian's ruling (a), 2026-09-14).

**The structural gap this closes.** A stacked M arm declares
`predictions[<base>]` and reads that artifact on EVERY row of its training
window, so :func:`crucible.slots.inputs._assert_base_predictions_present`
refuses it until the base has produced for every session in the window — a
gap is a refusal rather than a hole, because a substituted zero is the
2026-08-28 hard-zeroed condition by another door. A weekly arc produces ONE
session per week, so a 504-session window is 504 weeks of weekly runs away:
a stacked arm can never register from the scheduled cadence alone. Measured
2026-09-13 on the M slot's first real run —
`directional_combine_residual_stack` refused with its base
`residual_momentum` holding predictions for 1 of 526 panel sessions.

**Why this is evidence and not leakage.** Nothing here is a shortcut. Each
session is produced by the SAME per-arm produce path `experiment.run` calls
(`crucible.slots.model.produce`, reached through the ``produce`` argument, so
this module owns no fitting code and can never grow a second one): the fit
sees the panel and the features as of that session and nothing after it. A
backfilled session is byte-for-byte the artifact the weekly arc would have
written had it run that week, which is what makes the accumulated history
walk-forward — and what lets `experiment.grade` score these sessions like any
other, with no backfill-aware grading code anywhere.

**In region, or it refuses** — the same guard, the same allowance and the
same single override as `crucible data.heal`, reused rather than restated
(`crucible.data.heal.in_region`, :data:`~crucible.data.heal.LAPTOP_SESSION_ALLOWANCE`).
A backfill is a larger write than a heal, not a smaller one.

**One manifest for the whole range**, keyed to ``--to``, exactly as
`data.heal` writes one for its range: this is one job, and per-session
manifests under a job nothing schedules would be a manifest set no absence
detector can grade. The per-session telemetry the produce path already emits
— `feature_completeness` and the serving completeness rows — lands on that
one manifest, because the rebound context shares its telemetry lists by
reference.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any, Protocol

from crucible.data.heal import (
    LAPTOP_SESSION_ALLOWANCE,
    NotInRegionError,
    in_region,
    sessions_in_range,
)
from crucible.keys import arm_predictions_key, backfill_key, cross_section_key, shadow_key
from crucible.runner import rebind_trading_day
from crucible.slots.inputs import BasePredictionsUnavailableError, SlotUnservableError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crucible.runner import RunContext

__all__ = [
    "BackfillProducedNothingError",
    "SESSION_REFUSALS",
    "UnknownArmError",
    "run_backfill",
]


class UnknownArmError(KeyError):
    """The named arm is not registered in the slot, so there is nothing to backfill."""


class BackfillProducedNothingError(RuntimeError):
    """Every session in the range refused, so the range was not backfilled."""


#: The per-session refusals a backfill RECORDS and continues past, exhaustively.
#:
#: Both mean the same thing: this arm's own declared inputs are not satisfiable
#: on THIS session yet. A stacked arm being backfilled over a range that starts
#: before its own base's history is the live case — the early sessions cannot
#: be produced and the late ones can, and a job that died on the first one
#: would make a two-level stack unbackfillable in either order.
#:
#: Deliberately NOT here, and the reason this is a closed tuple rather than a
#: bare `except Exception`: `TrainingIntegrityError`, a defective feature
#: layer, a non-finite prediction. Those are evidence the inputs are
#: compromised and they fail the whole job, the same way they fail a whole
#: slot run (plan §4.4, Brian ruling 2026-08-29).
#:
#: Recording surface: one `rows_rejected` row per refused session on this
#: run's manifest, naming the session, plus the full reason in the
#: `backfill.v1` result document. A range that refuses ENTIRELY raises
#: :class:`BackfillProducedNothingError` rather than reporting a completed
#: backfill of nothing.
SESSION_REFUSALS: tuple[type[BaseException], ...] = (
    SlotUnservableError,
    BasePredictionsUnavailableError,
)

#: `record_rejected` caps a reason at the schema's length and RAISES over it,
#: so the row carries the session and the exception class — the two facts that
#: are actionable — and the prose reason lives in the result document, which
#: has no cap.
_REJECTED_REASON = "{session}: {kind} — this arm's declared inputs are not satisfiable yet"


class _Produce(Protocol):
    def __call__(self, ctx: Any, *, settings: Any, **kwargs: Any) -> dict[str, Any]: ...


def _arm_id(specs: Sequence[Any], *, slot: str, arm: str) -> str:
    """The registered id of ``arm``, or a refusal naming what the slot carries.

    Resolved ONCE, before any session runs: the id is what the three
    per-session artifact keys are built from, so a backfill that resolved it
    per session could skip a session against one id and write it under
    another.
    """
    for spec in specs:
        if spec.name == arm:
            return str(spec.arm_id)
    raise UnknownArmError(
        f"no arm named {arm!r} is registered in slot {slot!r}; the slot registered "
        f"{sorted(s.name for s in specs)}. Backfilling nothing and exiting 0 would be "
        "indistinguishable from backfilling a range that was already complete."
    )


def _session_keys(arm_id: str, session: str) -> tuple[str, str, str]:
    """The three artifacts one produced session leaves behind.

    All three, not just `arm_predictions`: the stacked arm reads the first
    and `experiment.grade` reads the other two, so a session holding one of
    the three is a session the grader cannot score and the skip test must
    not treat as done.
    """
    return (
        arm_predictions_key(arm_id, session),
        shadow_key(arm_id, session),
        cross_section_key(arm_id, session),
    )


def run_backfill(
    ctx: RunContext,
    *,
    produce: _Produce,
    specs: Sequence[Any],
    settings: Any,
    slot: str,
    arm: str,
    start: dt.date,
    end: dt.date,
    force: bool = False,
    i_am_in_region: bool = False,
) -> dict[str, Any]:
    """Produce ``arm`` for every NYSE session in ``[start, end]``. Idempotent.

    ``produce`` is the slot's own per-arm produce callable — the one
    `experiment.run` dispatches to. Passing it in rather than importing a
    slot module here is what keeps this job from becoming a second fitting
    path: there is no code in this module that could fit anything.
    """
    sessions = sessions_in_range(start, end)
    if not sessions:
        raise ValueError(
            f"the range {start}..{end} contains no NYSE sessions; there is nothing to "
            "backfill, and a backfill that reported success over an empty range would "
            "be a no-op wearing a completed job's clothes"
        )
    arm_id = _arm_id(specs, slot=slot, arm=arm)

    on_ec2, evidence = in_region()
    if len(sessions) > LAPTOP_SESSION_ALLOWANCE and not on_ec2 and not i_am_in_region:
        raise NotInRegionError(
            f"refusing to backfill {len(sessions)} session(s) ({start}..{end}) of "
            f"{slot}:{arm} from this host: {evidence}. Each session is a full "
            "point-in-time fit against the store, so the range is dominated by S3 "
            "round-trip latency the same way a heal is. Run it in region:\n"
            f"    crucible experiment.backfill --slot {slot} --arm {arm} "
            f"--from {start} --to {end} --run-mode replay\n"
            f"Ranges of up to {LAPTOP_SESSION_ALLOWANCE} sessions are allowed locally "
            "as a diagnostic. `--i-am-in-region` overrides this and nothing else does."
        )

    produced: list[str] = []
    already: list[str] = []
    refused: list[dict[str, str]] = []
    for day in sessions:
        session = day.isoformat()
        if not force and all(ctx.store.exists(k) for k in _session_keys(arm_id, session)):
            already.append(session)
            continue
        try:
            produce(rebind_trading_day(ctx, day), settings=settings, arm_name=arm)
        except SESSION_REFUSALS as exc:
            # The ONE swallow in this loop, and it is a record rather than a
            # swallow: see SESSION_REFUSALS for the failure mode absorbed,
            # why the deliverable survives, and the surface it lands on.
            refused.append({"session": session, "reason": str(exc)})
            ctx.record_rejected(
                _REJECTED_REASON.format(session=session, kind=type(exc).__name__), 1
            )
            continue
        produced.append(session)

    if not produced and not already:
        raise BackfillProducedNothingError(
            f"every one of the {len(sessions)} session(s) in {start}..{end} refused for "
            f"{slot}:{arm}: first {refused[0]['session'] if refused else '<none>'}. A "
            "backfill that produced nothing did not have nothing to do — the range is "
            "still missing. First reason: "
            f"{refused[0]['reason'] if refused else '<none>'}"
        )

    ctx.record_rows(rows_in=len(sessions), rows_out=len(produced) + len(already))
    ctx.record_metric(
        {
            "name": "sessions_backfilled",
            "module": "crucible.backfill",
            "metric_type": "repair",
            "value": float(len(produced)),
            "unit": "sessions",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"{slot}:{arm} ({arm_id}): {len(produced)} session(s) produced over "
                f"{start}..{end}, {len(already)} already present, {len(refused)} "
                f"refused. Host: {evidence}."
            ),
            "source_path": f"predictions/{arm_id}/*.json ({start}..{end})",
            "last_updated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    result = {
        "schema_version": "backfill.v1",
        "slot": slot,
        "arm": arm,
        "arm_id": arm_id,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "host": evidence,
        "forced": bool(force),
        "sessions": [d.isoformat() for d in sessions],
        "produced": produced,
        "already_present": already,
        "refused": refused,
    }
    ctx.record_output(
        backfill_key(ctx.trading_day.isoformat(), ctx.run_id),
        json.dumps(result, indent=2, sort_keys=True).encode("utf-8"),
        schema_version="backfill.v1",
    )
    return result
