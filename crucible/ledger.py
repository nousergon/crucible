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
row — and `tests/test_ledger_explain_migrate.py` asserts that every write is a
strict prefix extension of the file it replaces.

**The rewrite is a conditional write** (`alpha-engine-config-I12562`, audit
CR-01). A read-modify-write of one object is only append-only if the write
lands on the version it read; an unconditional PUT gives the ledger to
whichever writer finished last and silently drops the other's trial. So
:func:`append_trials` takes the object's version token BEFORE it reads,
writes with :meth:`Store.compare_and_swap` against that token, and on a lost
race re-reads and re-decides — a bounded number of times, then it raises.
`tests/test_ledger_concurrency.py` holds two writers past their final read
on both backends and asserts both trials survive.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from crucible.keys import ledger_key
from crucible.models import TrialRow
from crucible.store import PointerConflictError

if TYPE_CHECKING:
    from nousergon_lib.arena.engine import ArenaCycle
    from nousergon_lib.arena.window import ArmSeries

    from crucible.store import Store

__all__ = [
    "APPEND_ATTEMPTS",
    "LedgerAppendError",
    "append_trials",
    "n_trials",
    "read_trials",
    "trial_rows",
]

#: How many times :func:`append_trials` re-reads and retries after losing the
#: conditional write to a concurrent writer before it raises. Each loss means
#: another writer's append LANDED, so the ledger is making progress; running
#: out means contention no grading cadence produces, and that is reported, not
#: absorbed.
APPEND_ATTEMPTS = 8


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

    ``lineage`` is the slot's own provenance for the series this row counts,
    forwarded verbatim from :attr:`ArmSeries.lineage`
    (`alpha-engine-config-I9963`). The DSR denominator's log carries the same
    fact as the cycle artifact, so "how many trials" and "over which upstream
    versions" are answerable from one file rather than by joining two.
    Additive on `trial.v1`: the row has no JSON Schema, every reader ignores
    unknown keys, and an OLDER row that predates the field carries no
    `lineage` key at all — which is deliberately distinct from `{}`, "this
    slot declares none", exactly as on `arena_cycle.v1`.
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
                "lineage": (
                    {k: list(v) for k, v in sorted(series.lineage.items())}
                    if series is not None
                    else {}
                ),
                "run_id": run_id,
                "arena_cycle_key": arena_cycle_key,
                "written_at_utc": written,
            }
        )
    # `alpha-engine-config-I10045` row 9: each row validated against
    # `crucible.models.TrialRow` before being handed to the caller — the
    # producer-validates-on-the-way-out pattern every other boundary in
    # this migration takes. No schema is generated or published (the module
    # docstring's "the row has no JSON Schema" is a design decision, not a
    # gap): `TrialRow` is `extra="allow"`, exists only to catch a malformed
    # CORE field here, at construction, rather than downstream in
    # `n_trials`. The dicts returned are UNCHANGED — `append_trials`/
    # `read_trials`'s append-only race guard still compares plain dicts.
    for row in rows:
        try:
            TrialRow.model_validate(row)
        except ValidationError as exc:
            raise ValueError(
                f"trial row for arm {row['arm_id']!r} does not conform: {exc}"
            ) from exc
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

    Atomic against concurrent writers (`alpha-engine-config-I12562`): the
    write is a :meth:`Store.compare_and_swap` against the version token read
    BEFORE the ledger body, so it lands only if no other writer appended in
    between. A lost race re-reads, re-deduplicates and tries again, up to
    :data:`APPEND_ATTEMPTS` times; after that it raises
    :class:`LedgerAppendError` rather than returning a count for rows that
    are not in the ledger.
    """
    key = ledger_key()
    for _attempt in range(APPEND_ATTEMPTS):
        # The token is read FIRST. A body read after it is at least as new, so
        # a write conditioned on this token can only land on the version the
        # body below was decided from — never on a newer one it did not see.
        # Reading the body first and the token second would let a writer that
        # landed in between hand us ITS token, and our write would succeed on
        # top of a row we never read. The ledger only grows, so a version
        # token never comes back to an older body (no ABA).
        token = store.etag(key)
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
        # exactly the case that would violate it. This re-read refuses the case
        # it can SEE; the conditional write below is what closes the window it
        # cannot — a writer landing after this re-read (alpha-engine-config-I12562).
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
        try:
            store.compare_and_swap(key, token, payload)
        except PointerConflictError:
            # Deliberate retry, not a swallow: a lost conditional write means
            # another writer's append landed and ours did not. The loop re-reads
            # that writer's rows and re-decides `fresh` against them; exhausting
            # it raises below.
            continue
        return len(fresh)
    raise LedgerAppendError(
        f"lost the conditional write of {key!r} to a concurrent writer {APPEND_ATTEMPTS} "
        f"times in a row; {len(rows)} row(s) were NOT recorded. Each loss means another "
        "writer appended, so the ledger itself is intact — re-run this append."
    )


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
