"""The daily v2 pipelines notice: one Telegram message per trading day.

Brian's only daily "Pipelines" message was v1's Step Functions notifier, and it
counts v1 state machines only. On 2026-11-06 it read "all 2 succeeded" on the
morning the v2 `trader.session` never ran; nothing said so, because v2 pages
only on absence or failure (plan §4.6) and sends no daily result at all.

**It is not a page.** Like `crucible.weekly_notice`, it goes out on the
operator chat with ``severity="info"``, Telegram only, through
`nousergon_lib.gates.report.deliver` (no SNS, no dedup), and `crucible.alerts`
is not imported. A failed or absent run still pages through the sweep; this
message only says what happened and whether anything needs a look.

**Which day it covers.** The notice is sent by the 21:00 ET `alerts.sweep`
and reports the PREVIOUS trading day ``P``: `data.daily` P, `serve.daily` P
(every slot in :data:`crucible.daily_chain.DAILY_CHAIN_SERVE_ORDER`),
`trader.reconcile` P, and `trader.session` P -- the session the trader ran
that same morning, since it binds to the last closed session. Every one of
those is past its declared deadline by then. Today's own `data.daily` and
`serve.daily` are not: their deadlines (22:00 and 23:00 ET) fall after the
sweep, so reporting them would call a run "missing" that is not yet due.

**It reads; it never grades.** Each run is its canonical manifest
(`crucible.keys.manifest_key`): `ok` succeeded, `failed` failed, no manifest
is missing. A `--skip-if-ok` no-op files under its own discriminator and is
never read here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from nousergon_lib.gates import report

from crucible.daily_chain import DAILY_CHAIN_SERVE_ORDER
from crucible.documents import read_store_document
from crucible.execution import EXECUTION_METRIC_NAME
from crucible.keys import manifest_key
from crucible.morning import TELEGRAM_DESTINATION_OVERRIDE_VAR
from crucible.store import Store

__all__ = [
    "NOTICE_SOURCE",
    "ORDER_BOOK_METRIC",
    "RunReading",
    "daily_runs",
    "order_book_line",
    "read_runs",
    "render_notice",
    "send_notice",
]

NOTICE_SEVERITY = "info"
NOTICE_SOURCE = "crucible-v2/daily"

#: The trader's order-book row (crucible-trader `crucible_trader.order_book`).
#: Stated here, not imported: the trader depends on this package, never the
#: reverse.
ORDER_BOOK_METRIC = "trader_order_book_written"

TRADER_SESSION_JOB = "trader.session"

Document = Mapping[str, object]


def daily_runs() -> tuple[tuple[str, str | None], ...]:
    """``(job, slot)`` for every run a trading day owes, in pipeline order."""
    return (
        ("data.daily", None),
        *(("serve.daily", slot) for slot in DAILY_CHAIN_SERVE_ORDER),
        (TRADER_SESSION_JOB, None),
        ("trader.reconcile", None),
    )


@dataclass(frozen=True)
class RunReading:
    """One run's state for the day: ``ok``, ``failed``, ``missing`` or
    ``unreadable``, and its manifest when one was read."""

    job: str
    slot: str | None
    state: str
    manifest: Document | None = None

    @property
    def label(self) -> str:
        return f"{self.job} {self.slot}" if self.slot else self.job


def read_runs(store: Store, trading_day: str) -> list[RunReading]:
    """Read each owed run's canonical manifest for ``trading_day``."""
    readings: list[RunReading] = []
    for job, slot in daily_runs():
        key = manifest_key(job, trading_day, discriminator=slot)
        read = read_store_document(store, key)
        if read.problem is not None:
            readings.append(RunReading(job, slot, "unreadable"))
        elif read.document is None:
            readings.append(RunReading(job, slot, "missing"))
        else:
            status = str(read.document.get("status") or "")
            state = status if status in ("ok", "failed") else "unreadable"
            readings.append(RunReading(job, slot, state, read.document))
    return readings


def _metric(manifest: Document, name: str) -> Mapping[str, object] | None:
    rows = manifest.get("metrics")
    for row in cast("list[object]", rows) if isinstance(rows, list) else []:
        if isinstance(row, dict) and cast("dict[str, object]", row).get("name") == name:
            return cast("dict[str, object]", row)
    return None


def _short_cause(reason: str) -> str:
    """The exception class out of a ``"what: Class: detail"`` reason, so the
    message names the cause without copying a resource name into a chat."""
    for part in reason.split(":")[1:]:
        word = part.strip()
        if word and " " not in word:
            return word
    return reason.strip()[:60] or "no reason recorded"


def order_book_line(session: RunReading) -> str:
    """One line on the session's order book: written, missing or failed."""
    if session.manifest is None or session.state != "ok":
        return f"Order book: missing ({TRADER_SESSION_JOB} {session.state})."
    row = _metric(session.manifest, ORDER_BOOK_METRIC)
    if row is None:
        return f"Order book: missing (no {ORDER_BOOK_METRIC} row)."
    if row.get("status") != "OK":
        return f"Order book: FAILED ({_short_cause(str(row.get('status_reason') or ''))})."
    line = "Order book: written"
    fills = _metric(session.manifest, EXECUTION_METRIC_NAME)
    n = fills.get("n_samples") if fills is not None else None
    if isinstance(n, int | float) and not isinstance(n, bool):
        line += f", {int(n)} order(s) filled"
    return line + "."


def render_notice(trading_day: str, readings: Sequence[RunReading]) -> str:
    """The HTML message, two or three lines. Every interpolated string is escaped."""
    ok = [r for r in readings if r.state == "ok"]
    head = f"v2 pipelines {trading_day}: {len(ok)} of {len(readings)} succeeded"
    groups: list[str] = []
    for state in ("failed", "missing", "unreadable"):
        names = [r.label for r in readings if r.state == state]
        if names:
            groups.append(f"{state}: {', '.join(names)}")
    if groups:
        head += " — " + "; ".join(groups)
    session = next(r for r in readings if r.job == TRADER_SESSION_JOB)
    book = order_book_line(session)
    first = next((r for r in readings if r.state != "ok"), None)
    if first is None and not book.startswith("Order book: written"):
        first = session
    action = (
        "No action needed."
        if first is None
        else f"Look at runs/{first.job}/{trading_day}/ first; failures and absences also page."
    )
    return "\n".join([f"<b>{report.escape(head)}</b>", report.escape(book), report.escape(action)])


def send_notice(
    store: Store,
    trading_day: str,
    *,
    console_artifact: str,
    transport: Callable[..., object] | None = None,
) -> str:
    """Read the day's runs, render, deliver. Returns the destination. Raises
    when nothing received the message."""
    message = render_notice(trading_day, read_runs(store, trading_day))
    # The destination resolves exactly as `report.morning`'s does: the
    # operator chat by name, unless the integration tier's override names
    # `console_only` (so a test sweep never reaches a phone).
    return report.deliver(
        message,
        severity=NOTICE_SEVERITY,
        source=NOTICE_SOURCE,
        console_artifact=console_artifact,
        destination_override_var=TELEGRAM_DESTINATION_OVERRIDE_VAR,
        parse_mode="HTML",
        transport=transport,
    )
