"""The fleet trial ledger: everything ever tried, in one append-only log.

Normative source: plan §9.1 row 2.

`I9685`: PBO passed on a three-spec race. A Deflated Sharpe Ratio needs
`n_trials` across **everything tried**, not per run — and a per-run count is
always small, always flattering, and always available, which is why it keeps
being the one used. This log is the denominator: one row per graded
arm-cycle, across every slot, for the life of the system.

**One row per (slot, arm, cycle date).** Not per arm and not per cycle: a
trial is one arm measured once. Re-grading a cycle appends nothing new,
because the row's identity is those three fields and
:func:`append_trials` de-duplicates on it — a replay must not inflate the
multiplicity correction that a replay is supposed to leave untouched.

**Controls carry `control: true` and are counted separately.** A planted
control is not a hypothesis anyone tried; counting it in `n_trials` would
deflate every real arm's Sharpe for the harness's own self-check.

**JSONL, appended by rewrite.** S3 has no append. The append-only property
is a property of the CONTENT — nothing here rewrites or drops an existing
row — and `tests/test_ledger.py` asserts that every write is a strict prefix
extension of the file it replaces.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

from crucible.keys import ledger_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nousergon_lib.arena.engine import ArenaCycle
    from nousergon_lib.arena.window import ArmSeries

    from crucible.store import Store

__all__ = ["LedgerAppendError", "append_trials", "n_trials", "read_trials", "trial_rows"]


class LedgerAppendError(RuntimeError):
    """A write that would drop or rewrite an existing trial row."""


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["slot"], row["arm_id"], row["as_of"])


def trial_rows(
    cycle: ArenaCycle,
    *,
    slot: str,
    as_of: str,
    run_id: str,
    control_ids: set[str],
    series_by_arm: dict[str, ArmSeries],
    arena_cycle_key: str,
) -> list[dict[str, Any]]:
    """One row per arm scored in ``cycle``.

    `n_dates_scored` is carried because a trial measured over two dates and
    a trial measured over forty are not equally informative, and a DSR that
    counted them alike would treat a rumour as a result.
    """
    written = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, Any]] = []
    for arm_id in sorted(cycle.scored_arms):
        series = series_by_arm.get(arm_id)
        scores = dict(series.scores) if series is not None else {}
        rows.append(
            {
                "schema_version": "trial.v1",
                "slot": slot,
                "arm_id": arm_id,
                "as_of": as_of,
                "control": arm_id in control_ids,
                "active": arm_id in cycle.active_arms,
                "benchmark": cycle.benchmark,
                "n_dates_scored": len(scores),
                "first_date": min(scores) if scores else None,
                "last_date": max(scores) if scores else None,
                "mean_score_ratio": (sum(scores.values()) / len(scores)) if scores else None,
                "run_id": run_id,
                "arena_cycle_key": arena_cycle_key,
                "written_at_utc": written,
            }
        )
    return rows


def read_trials(store: Store) -> list[dict[str, Any]]:
    """Every row in the ledger, in write order. An absent ledger is empty."""
    key = ledger_key()
    if not store.exists(key):
        return []
    lines = store.get_bytes(key).decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def append_trials(store: Store, rows: list[dict[str, Any]]) -> int:
    """Append rows the ledger does not already carry. Returns how many landed.

    Idempotent on (slot, arm_id, as_of): a re-grade of the same cycle adds
    nothing, so replaying a date does not inflate the multiplicity
    correction the replay is meant to leave alone.
    """
    existing = read_trials(store)
    seen = {_identity(row) for row in existing}
    fresh = [row for row in rows if _identity(row) not in seen]
    if not fresh:
        return 0
    # The guard that matters is NOT `len(combined) < len(existing)` — `combined`
    # is built by concatenating `existing` with `fresh` two lines below, so
    # that comparison can never be true for any input; it is a predicate that
    # tests a property of its own construction, not of the log. The property
    # that actually matters is append-only in substance: no row `read_trials`
    # returned above may be dropped, reordered or rewritten by this write. A
    # concurrent writer that appended between our read and our write is
    # exactly the case that would violate it — this call would then rewrite
    # the log from a stale `existing` and silently drop whatever the other
    # writer just added. Re-reading immediately before the write catches
    # that race the same way `Store.compare_and_swap` catches it for a
    # pointer, without needing the store to expose CAS for an append target.
    current = read_trials(store)
    if current != existing:
        raise LedgerAppendError(
            f"the ledger changed between read and write: {len(current)} row(s) present "
            f"now, {len(existing)} when this call read it. Writing `existing + fresh` on "
            "top of that would drop or reorder whatever the concurrent writer just "
            "appended. Re-read the ledger and retry."
        )
    combined = existing + fresh
    payload = ("\n".join(json.dumps(row, sort_keys=True) for row in combined) + "\n").encode(
        "utf-8"
    )
    store.put_bytes(ledger_key(), payload)
    return len(fresh)


def n_trials(store: Store, *, slot: str | None = None, include_controls: bool = False) -> int:
    """The DSR multiplicity denominator: distinct (slot, arm) pairs ever tried.

    Distinct ARMS, not rows: an arm graded for forty cycles is one
    hypothesis measured forty times, not forty hypotheses. Controls are
    excluded unless asked for, because the harness's own self-check is not
    something anyone hypothesised.
    """
    pairs = {
        (row["slot"], row["arm_id"])
        for row in read_trials(store)
        if (slot is None or row["slot"] == slot) and (include_controls or not row.get("control"))
    }
    return len(pairs)
