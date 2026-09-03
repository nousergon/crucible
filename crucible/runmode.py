"""Live or replay — read from the INVOCATION, never from the date.

Normative source: plan §6 row 2 and §6.1; the gap is alpha-engine-config-I9918.

Phase 2's exit gate is "2 consecutive LIVE Saturdays whose weekly run
manifest reads `status: ok` on its first attempt". §6.1 builds phase 2 by
*replaying* five historical Saturdays on an accelerated schedule, so a gate
that inferred liveness from `trading_day` or `calendar_date` would be
satisfied by exactly those replays — the "green on replays" failure the
phase-2 clause list was withheld for. The manifest therefore carries the
fact, and this module is the one place the fact is resolved.

**There is no default, at any layer.** `run_manifest.v2` makes `run_mode`
required with no default, and :func:`resolve_run_mode` refuses rather than
assuming: an invocation that says nothing is a bug in the invocation, and
guessing `live` for it is precisely how a replay ends up counted as a live
Saturday. Refusing is loud, immediate and happens before any work is done,
which is the cheapest place for it to happen.

The resolution order is explicit-argument, then environment, then refuse.
The environment variable exists so a scheduled runner declares the mode once
for a whole job — a workflow `env:` block, a systemd unit — rather than every
call site restating it; the explicit argument exists so one command can
override that without editing the schedule.
"""

from __future__ import annotations

import os

__all__ = [
    "RUN_MODES",
    "RUN_MODE_ENV",
    "RUN_MODE_LIVE",
    "RUN_MODE_REPLAY",
    "RunModeError",
    "resolve_run_mode",
]

#: A real execution, on the day it happened. The only value phase 2's live
#: clause counts.
RUN_MODE_LIVE = "live"

#: A re-run of a historical trading day. Graded like any other run, and
#: excluded from every clause that asks whether the system ran live.
RUN_MODE_REPLAY = "replay"

#: The closed vocabulary, exactly `run_manifest.v2`'s `run_mode` enum. Held
#: here and asserted equal to the schema's enum by
#: `tests/test_run_mode_contract.py`, so the producer and the contract cannot
#: drift into two spellings of one vocabulary.
RUN_MODES: tuple[str, ...] = (RUN_MODE_LIVE, RUN_MODE_REPLAY)

#: The environment variable a scheduled runner declares the mode with.
RUN_MODE_ENV = "CRUCIBLE_RUN_MODE"


class RunModeError(ValueError):
    """The invocation did not say whether it was live or a replay.

    Raised, never defaulted around. A `run_mode` this module guessed would be
    a claim about production that nobody made, sitting in the one field
    phase 2's gate reads.
    """


def _refuse(detail: str) -> RunModeError:
    return RunModeError(
        f"{detail} Pass `--run-mode {'|'.join(RUN_MODES)}` on the crucible command, or "
        f"set ${RUN_MODE_ENV} for the whole invocation. There is no default: a run "
        "assumed live could be a replay of a historical Saturday, and phase 2's exit "
        "gate counts live Saturdays. The trading day cannot answer this — the replay "
        "schedule replays real past Saturdays — so it is never consulted here."
    )


def resolve_run_mode(explicit: str | None = None) -> str:
    """The run mode for this invocation, or raise :class:`RunModeError`.

    ``explicit`` wins (the `--run-mode` flag); then ``$CRUCIBLE_RUN_MODE``;
    then the call is refused. A value outside :data:`RUN_MODES` is refused at
    either source rather than passed through to fail later at schema
    validation, where the message would name a field instead of the flag that
    set it.
    """
    if explicit is not None:
        if explicit not in RUN_MODES:
            raise _refuse(f"run mode {explicit!r} is not one of {RUN_MODES}.")
        return explicit
    from_env = os.environ.get(RUN_MODE_ENV)
    if from_env is None or from_env == "":
        raise _refuse("this invocation did not declare whether it is live or a replay.")
    if from_env not in RUN_MODES:
        raise _refuse(f"${RUN_MODE_ENV} is {from_env!r}, which is not one of {RUN_MODES}.")
    return from_env
