"""`experiment.retire`: one operator retirement of one arm, on a named ruling.

Retirement is normally the engine's verdict: `crucible.promote` appends a
`retired` event for every arm the weekly cycle retires, and a row for every
arm it judged (evaluation-policy § Retirement). This job is the other door.
It exists for the case that verdict does not cover: a person with authority
over model selection rules that a specific arm leaves the arena now.

It stays narrow on purpose:

* **No ruling, no retirement.** ``ruling`` names where the decision is
  recorded (a tracker issue, a decision card). A retirement nobody can trace
  back to a decision cannot be reviewed later.
* **Never the champion.** Retiring the pointer's arm would leave the slot
  serving a retired arm. Moving the pointer is `promote`'s job, and so is
  retiring what it then leaves behind.
* **Never a control.** The engine never retires a control, and neither does
  this job.
* **Two records, the same two the engine writes.** A `retired` event in
  `arms/{slot}/register.jsonl` is what every later cycle folds. A row in
  `retirements/{slot}/events.jsonl` is the audit trail. The row's reason
  names the ruling.

Re-running is safe. An arm already retired with the same reason gets no
second register event, which the fold would refuse anyway, and
`crucible.promote.append_events` skips a log row that is already there. So a
run the runner retried after the register write still finishes the log.
"""

from __future__ import annotations

from dataclasses import dataclass

from nousergon_lib.arena.arms import ArmRegister

from crucible.documents import load_store_document
from crucible.keys import champion_key, retirement_log_key
from crucible.models import RetirementLogRow
from crucible.promote import append_events
from crucible.slots import arm_name
from crucible.slots.arms import read_register, write_register
from crucible.store import Store

#: The prefix every operator retirement's reason carries. The engine's own
#: verdict reasons never start with it, so a reader can split the two.
OPERATOR_REASON_PREFIX = "operator retirement"


class RetirementRefused(ValueError):
    """This retirement is not one the job may perform. Nothing was written."""


@dataclass(frozen=True)
class RetirementResult:
    slot: str
    arm_id: str
    retired_date: str
    reason: str
    register_written: bool
    log_key: str

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "arm_id": self.arm_id,
            "retired_date": self.retired_date,
            "reason": self.reason,
            "register_written": self.register_written,
            "log_key": self.log_key,
        }


def operator_reason(ruling: str, reason: str) -> str:
    """The one text both records carry, so they cannot disagree."""
    return f"{OPERATOR_REASON_PREFIX} on ruling {ruling}: {reason}"


def _champion_arm_id(store: Store, slot: str) -> str | None:
    key = champion_key(slot)
    if not store.exists(key):
        return None
    pointer = load_store_document(store, key)
    arm_id = pointer.get("arm_id")
    return str(arm_id) if arm_id else None


def resolve_arm(register: ArmRegister, *, slot: str, arm: str) -> str:
    """The one registered id ``arm`` names, by bare name or by full id."""
    if ":" in arm:
        if arm not in register:
            raise RetirementRefused(f"arm {arm!r} is not in slot {slot!r}'s register")
        return arm
    name = arm_name(arm)
    matches = [a for a in register.all_arms() if register.state(a).record.name == name]
    live = [a for a in matches if register.state(a).retired_date is None]
    if len(live) == 1:
        return live[0]
    if not matches:
        raise RetirementRefused(
            f"no arm named {name!r} is registered in slot {slot!r}; the register holds "
            f"{sorted({register.state(a).record.name for a in register.all_arms()})}"
        )
    if len(matches) == 1:
        return matches[0]
    raise RetirementRefused(
        f"{len(matches)} registered arms are named {name!r} in slot {slot!r} "
        f"({matches}); name the one to retire by its full id"
    )


def retire_arm(
    store: Store,
    *,
    slot: str,
    arm: str,
    ruling: str,
    reason: str,
    retired_date: str,
    write: bool = True,
) -> RetirementResult:
    """Retire one arm on ``ruling``. Refuses before writing anything.

    ``write=False`` runs every check and reports what would be written.
    """
    ruling = (ruling or "").strip()
    reason = (reason or "").strip()
    if not ruling:
        raise RetirementRefused(
            "an operator retirement needs --ruling: the issue or decision that authorised "
            "it. Without one, nobody can trace the retirement back to a decision."
        )
    if not reason:
        raise RetirementRefused("an operator retirement needs --reason")

    register = read_register(store, slot)
    arm_id = resolve_arm(register, slot=slot, arm=arm)
    state = register.state(arm_id)
    if state.record.control:
        raise RetirementRefused(f"{arm_id} is a control arm, and a control is never retired")
    champion = _champion_arm_id(store, slot)
    if champion == arm_id:
        raise RetirementRefused(
            f"{arm_id} is slot {slot!r}'s current champion. Retiring it would leave the "
            "slot serving a retired arm. Move the pointer first (`promote`)."
        )

    text = operator_reason(ruling, reason)
    if state.retired_date is not None and state.retired_reason != text:
        raise RetirementRefused(
            f"{arm_id} was already retired on {state.retired_date} ({state.retired_reason!r})"
        )
    needs_register_event = state.retired_date is None
    when = retired_date if needs_register_event else str(state.retired_date)

    row = {
        "as_of": when,
        "slot": slot,
        "arm_id": arm_id,
        "retire": True,
        "reason": text,
        "age_weeks": max(0, state.age_weeks(when)),
        "pairwise_losses": 0,
        "is_champion": False,
    }
    RetirementLogRow.model_validate(row)

    if write:
        if needs_register_event:
            write_register(store, slot, register.retire(arm_id, when, text))
        append_events(store, retirement_log_key(slot), [row])

    return RetirementResult(
        slot=slot,
        arm_id=arm_id,
        retired_date=when,
        reason=text,
        register_written=needs_register_event,
        log_key=retirement_log_key(slot),
    )


__all__ = [
    "OPERATOR_REASON_PREFIX",
    "RetirementRefused",
    "RetirementResult",
    "operator_reason",
    "resolve_arm",
    "retire_arm",
]
