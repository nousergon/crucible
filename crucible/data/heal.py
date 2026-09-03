"""`crucible data.heal --from --to` — repair a named gap, idempotently, in region.

Normative source: plan §9.7 row 1, and the fleet's standing in-region rule.

Two properties, and the second is the one that has cost real hours:

**Idempotent.** A heal recompiles each session in the range and writes the
same key the daily job writes. A session already correct is rewritten with
identical bytes, so a rerun is a no-op by content hash — that is the whole
basis of the runbook's `rerun` verb, and it means an interrupted heal is
resumed by running it again rather than by working out where it stopped.
Every session gets a row in the result, `repaired` or `already_present`,
because a heal that reports only what it changed cannot be checked against
what it was asked to do.

**In region, or it refuses.** A ~900-ticker write is dominated by S3
round-trip latency: a 20-40 minute in-region job took over three hours from
a laptop on 2026-07-15. The guard reads EC2 instance metadata (IMDSv2, with
a short timeout so an off-network laptop fails fast rather than hanging), and
the ONLY override is an explicit `--i-am-in-region`. There is no environment
variable and no config key, because the failure mode is an operator in a
hurry, and a flag they must type is the point.

The refusal is scaled to the write: a range under
:data:`LAPTOP_SESSION_ALLOWANCE` sessions is small enough to be a
diagnostic, and diagnostics from a laptop are sanctioned. Above it the job
refuses, names the range and prints the in-region command.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from crucible.calendar import assert_trading_day, is_trading_day
from crucible.data.daily import DEFAULT_LOOKBACK_DAYS, run_daily
from crucible.data.sources import PriceSource
from crucible.keys import data_panel_key, heal_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crucible.runner import RunContext

__all__ = [
    "IMDS_TOKEN_URL",
    "IMDS_URL",
    "LAPTOP_SESSION_ALLOWANCE",
    "NotInRegionError",
    "in_region",
    "run_heal",
    "sessions_in_range",
]

IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
IMDS_URL = "http://169.254.169.254/latest/meta-data/instance-id"

#: Sessions a non-EC2 host may heal without the override. Small enough to be
#: a diagnostic ("did yesterday actually write?"), far short of a backfill.
LAPTOP_SESSION_ALLOWANCE = 3

#: IMDS is link-local. On EC2 it answers in single-digit milliseconds; off
#: EC2 the address is unroutable and the socket fails immediately. The
#: timeout exists for the case where something answers the address slowly.
_IMDS_TIMEOUT_S = 0.5


class NotInRegionError(RuntimeError):
    """A bulk write was attempted from a host that is not in the bucket's region."""


def in_region() -> tuple[bool, str]:
    """Whether this host is an EC2 instance, and the evidence either way.

    Returns the evidence string as well as the verdict so the refusal can
    state WHY it decided the host is a laptop, rather than asserting it.
    IMDSv2 (token first) because IMDSv1 is disabled on the fleet's launch
    templates and a v1 probe would report "not EC2" from inside EC2 — a
    guard that is wrong in the silent direction.
    """
    try:
        token_request = urllib.request.Request(
            IMDS_TOKEN_URL,
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_request, timeout=_IMDS_TIMEOUT_S) as response:
            token = response.read().decode("utf-8").strip()
        instance_request = urllib.request.Request(
            IMDS_URL, headers={"X-aws-ec2-metadata-token": token}
        )
        with urllib.request.urlopen(instance_request, timeout=_IMDS_TIMEOUT_S) as response:
            instance_id = response.read().decode("utf-8").strip()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"IMDSv2 did not answer ({type(exc).__name__}): not an EC2 instance"
    if not instance_id.startswith("i-"):
        return False, f"IMDSv2 answered with {instance_id!r}, which is not an instance id"
    return True, f"EC2 instance {instance_id}"


def sessions_in_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Every NYSE session in ``[start, end]``. Both bounds must be sessions.

    Asserting the bounds rather than snapping them: an operator who typed a
    Saturday has a bug in the range they believe they are healing, and
    silently moving the bound to Friday heals a different range than the one
    they will report having healed.
    """
    assert_trading_day(start, context=f"data.heal --from {start}")
    assert_trading_day(end, context=f"data.heal --to {end}")
    if end < start:
        raise ValueError(f"--to {end} precedes --from {start}")
    day = start
    out: list[dt.date] = []
    while day <= end:
        if is_trading_day(day):
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def run_heal(
    ctx: RunContext,
    *,
    source: PriceSource,
    start: dt.date,
    end: dt.date,
    gap: str,
    i_am_in_region: bool = False,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    expected_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Recompile every session in ``[start, end]``. Idempotent, and loud.

    No ``feature_version`` parameter, for the same reason `run_daily` has
    none (`alpha-engine-config-I9816`).
    """
    sessions = sessions_in_range(start, end)
    if not sessions:
        raise ValueError(
            f"the range {start}..{end} contains no NYSE sessions; there is nothing to "
            "heal, and a heal that reported success over an empty range would be a "
            "no-op wearing a completed job's clothes"
        )

    on_ec2, evidence = in_region()
    if len(sessions) > LAPTOP_SESSION_ALLOWANCE and not on_ec2 and not i_am_in_region:
        raise NotInRegionError(
            f"refusing to heal {len(sessions)} sessions ({start}..{end}) from this host: "
            f"{evidence}. A ~900-ticker write is dominated by S3 round-trip latency — a "
            "20-40 minute in-region job took over three hours from a laptop on "
            "2026-07-15. Run it on the trading box off-market-hours, or on a fresh "
            "in-region instance:\n"
            f"    crucible data.heal --from {start} --to {end} --gap {gap}\n"
            f"Ranges of up to {LAPTOP_SESSION_ALLOWANCE} sessions are allowed locally as "
            "a diagnostic. `--i-am-in-region` overrides this and nothing else does."
        )

    repaired: list[str] = []
    already: list[str] = []
    for day in sessions:
        key = data_panel_key(day.isoformat())
        present = ctx.store.exists(key)
        # The day is recompiled either way. "Present" is not "correct": the
        # gap being healed may be a panel that exists and is wrong, and a
        # heal that skipped every existing key could never repair one.
        day_ctx = _rebind(ctx, day)
        run_daily(
            day_ctx,
            source=source,
            lookback_days=lookback_days,
            expected_symbols=expected_symbols,
        )
        (already if present else repaired).append(day.isoformat())

    ctx.record_rows(rows_in=len(sessions), rows_out=len(sessions))
    ctx.record_metric(
        {
            "name": "sessions_healed",
            "module": "crucible.data.heal",
            "metric_type": "repair",
            "value": float(len(sessions)),
            "unit": "sessions",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"gap {gap!r}: recompiled {len(sessions)} session(s) over {start}..{end}; "
                f"{len(repaired)} had no prior panel, {len(already)} were rewritten in "
                f"place. Host: {evidence}."
            ),
            "source_path": f"data/*/panel.parquet ({start}..{end})",
            "last_updated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    result = {
        "schema_version": "heal.v1",
        "gap": gap,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "host": evidence,
        "sessions": [d.isoformat() for d in sessions],
        "repaired": repaired,
        "already_present": already,
    }
    ctx.record_output(
        heal_key(ctx.trading_day.isoformat(), ctx.run_id),
        json.dumps(result, indent=2, sort_keys=True).encode("utf-8"),
        schema_version="heal.v1",
    )
    return result


def _rebind(ctx: RunContext, day: dt.date) -> RunContext:
    """A view of ``ctx`` bound to another session, sharing its telemetry lists.

    The heal writes ONE manifest for the whole range — it is one job — but
    each session's compile must key its artifacts to its own trading day.
    The lists are shared by reference on purpose, so every session's inputs
    and outputs land in the one manifest rather than in per-day manifests
    that nothing would read.
    """
    from dataclasses import replace

    return replace(ctx, trading_day=day)
