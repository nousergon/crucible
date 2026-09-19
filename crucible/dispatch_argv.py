"""Reading a dispatch's argv the way the BOX read it.

Two readers now need the same answer to "which manifest does this dispatch
bind to" (`policy-shared-code`'s second-adoption trigger):

* `crucible.alerts`, grading an absence for a dispatch whose box is gone; and
* `crucible.dispatch_exit`, filling `expected_manifest_key` when the run left
  no breadcrumb (`crucible.runner.EXPECTED_MANIFEST_KEY_FILE`).

A second private copy of this derivation is exactly the shape
`alpha-engine-config-I11048` cost a night of false pages: two answers to
"which manifest", one of them keyed off flag order. There is one answer, and
it lives here.
"""

from __future__ import annotations

import datetime as dt
import shlex
from typing import Any

from crucible.calendar import resolve_trading_day
from crucible.keys import RANGE_BOUND_JOBS

__all__ = ["dispatch_target_trading_day", "flag_value"]


def flag_value(args: Any, flag: str) -> str | None:
    """``--flag value`` or ``--flag=value`` out of a dispatch record's argv.

    Positional, via :mod:`shlex`, never a substring test — same rule and same
    reason as `crucible.synthetic.args_synthetic_marker`.
    """
    if not isinstance(args, str) or not args.strip():
        return None
    try:
        tokens = shlex.split(args)
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def dispatch_target_trading_day(job: str, args: Any, dispatched_at: dt.datetime) -> dt.date:
    """The trading day a dispatch's manifest will actually be keyed under.

    `alpha-engine-config-I10134` resolved this from the dispatch's WALL CLOCK
    (`resolve_trading_day(dispatched_at)`), which is right for the on-demand
    jobs it was written for and wrong for every dispatch carrying `--date`.
    `crucible.cli.resolve_date` returns an explicit `--date` verbatim, so the
    manifest lands under THAT day.

    **The precedence is a property of the JOB, not of flag ordering**
    (`alpha-engine-config-I11048`). A job in
    :data:`crucible.keys.RANGE_BOUND_JOBS` binds its one manifest to the END
    of the range (`crucible.track_a.handle_data_heal` and
    `handle_experiment_backfill` both pass `trading_day=end` to
    `crucible.runner.run_job`), so `--to` wins for those and `--date` wins for
    everything else. `tests/test_range_bound_jobs_contract.py` reads
    `track_a.py` so the set cannot drift from the handlers.

    **The strongest form does not derive this at all.** The run records the
    key it actually bound in `$CRUCIBLE_STATE_DIR/manifest-key`
    (`crucible.runner.EXPECTED_MANIFEST_KEY_FILE`), which is the authoritative
    answer written by the party that wrote the manifest. This function is the
    FALLBACK: for a dispatch made before that breadcrumb existed, for a box
    that never reached the code that writes it, and — measured 2026-09-18 —
    for a box whose bootstrap did not export `CRUCIBLE_STATE_DIR` into the
    job's environment, which made the breadcrumb a no-op on every dispatch.

    Falls back to the wall clock when neither flag is present or parseable —
    the original behaviour, which is correct for a dispatch that named no
    day. A `--date` that is not a date is not silently substituted with a
    guess here; it is a malformed dispatch, and `crucible.cli` refuses it at
    the box with the usage exit code the wrapper reports as one.
    """
    ordered = ("--to", "--date") if job in RANGE_BOUND_JOBS else ("--date", "--to")
    for flag in ordered:
        explicit = flag_value(args, flag)
        if explicit is None:
            continue
        try:
            return dt.date.fromisoformat(explicit)
        except ValueError:
            return resolve_trading_day(dispatched_at)
    return resolve_trading_day(dispatched_at)
