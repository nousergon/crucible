"""The weekly arc's completion notice: one Telegram message per finished arc.

Brian, 2026-10-31: "is the weekly sf still running? ... trying to understand
why I don't see a run alert". v1's weekly Step Function sent a Telegram at the
end of every run, and his 2026-10-24 rule made that message list every stage
that was less than perfect. v1's Saturday rule was retired on 2026-10-28 and
v2 pages only on absence or failure (plan §4.6), so a clean v2 weekly said
nothing at all -- the rule was lost in the cutover, not ruled away.

**It is not a page.** Like `crucible.morning`, it goes out on the operator
chat with ``severity="info"``, Telegram only, through
`nousergon_lib.gates.report.deliver` (no SNS, no dedup), and
`crucible.alerts` is not imported. A failed arc still pages through the
alerts sweep; this message only says what happened.

**It reads; it never grades.** Every line comes from the arc's manifest and
the stage manifests it recorded as inputs. A metric status outside
:data:`CLEAN_STATUSES` is listed as an issue, by stage, so the message names
what to look at rather than restating the report.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from nousergon_lib.gates import report

from crucible.documents import load_store_document
from crucible.keys import manifest_key
from crucible.morning import TELEGRAM_DESTINATION_OVERRIDE_VAR
from crucible.store import Store

__all__ = [
    "CLEAN_STATUSES",
    "MAX_ISSUE_LINES",
    "NOTICE_SOURCE",
    "issues",
    "render_notice",
    "send_notice",
]

NOTICE_SEVERITY = "info"
NOTICE_SOURCE = "crucible-v2/weekly"

#: Metric statuses that are not an issue. Everything else (FAIL, BREACH, RED,
#: unmeasurable, unservable, N/A-*) is listed. `held` and `decided` are the
#: promote stage's normal outcomes; GREEN is the report card's pass.
CLEAN_STATUSES = frozenset({"OK", "GREEN", "held", "decided", "moved"})

#: Issue lines past this are summarised as a count, so one noisy week cannot
#: push the message past Telegram's limit.
MAX_ISSUE_LINES = 15

Document = Mapping[str, object]


def _stage_name(key: str) -> str:
    """`runs/experiment.grade/2026-10-30/m/run.json` -> `experiment.grade m`."""
    parts = key.split("/")
    job, rest = parts[1], parts[3:-1]
    return " ".join([job, *rest])


def issues(stages: Sequence[tuple[str, Document]]) -> list[str]:
    """One line per stage problem: a non-ok status, a retry, or a non-clean
    metric status (counted when a metric repeats)."""
    lines: list[str] = []
    for key, doc in stages:
        name = _stage_name(key)
        status = str(doc.get("status") or "")
        if status != "ok":
            lines.append(f"{name}: status {status or 'missing'}")
        attempts = doc.get("attempts")
        if isinstance(attempts, list) and len(cast("list[object]", attempts)) > 1:
            lines.append(f"{name}: needed {len(cast('list[object]', attempts))} attempts")
        counts: dict[tuple[str, str], int] = {}
        metrics = doc.get("metrics")
        for raw in cast("list[object]", metrics) if isinstance(metrics, list) else []:
            if not isinstance(raw, dict):
                continue
            row = cast("dict[str, object]", raw)
            metric_status = str(row.get("status") or "")
            if metric_status in CLEAN_STATUSES:
                continue
            ident = (str(row.get("name") or "?"), metric_status or "missing")
            counts[ident] = counts.get(ident, 0) + 1
        for (metric, metric_status), n in counts.items():
            lines.append(f"{name}: {metric} {metric_status}" + (f" x{n}" if n > 1 else ""))
    return lines


def _minutes(started: object, finished: object) -> str:
    try:
        start = dt.datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
    except ValueError:
        return "duration unknown"
    return f"{round((end - start).total_seconds() / 60)} min"


def render_notice(
    trading_day: str, arc: Document, stages: Sequence[tuple[str, Document]], planned: int
) -> str:
    """The HTML message. Every interpolated string is escaped."""
    status = str(arc.get("status") or "missing")
    head = "OK" if status == "ok" else status.upper()
    sha = str(arc.get("release_sha") or arc.get("code_sha") or "")[:8] or "unknown"
    lines = [
        f"<b>Crucible v2 weekly {report.escape(trading_day)}: {report.escape(head)}</b>",
        report.escape(
            f"{str(arc.get('started') or '')[11:16]}-{str(arc.get('finished') or '')[11:16]}Z "
            f"({_minutes(arc.get('started'), arc.get('finished'))}), "
            f"{len(stages)}/{planned} stages, release {sha}, {arc.get('run_mode') or '?'}"
        ),
    ]
    reason = str(arc.get("reason") or "")
    if status != "ok" and reason:
        lines.append(report.escape(f"Reason: {reason[:300]}"))
    found = issues(stages)
    if found:
        lines.append(f"<b>Issues ({len(found)})</b>")
        lines += [f"- {report.escape(line)}" for line in found[:MAX_ISSUE_LINES]]
        if len(found) > MAX_ISSUE_LINES:
            lines.append(f"- and {len(found) - MAX_ISSUE_LINES} more in the stage manifests")
    else:
        lines.append("No stage issues.")
    return "\n".join(lines)


def _read(store: Store, key: str) -> Document:
    return load_store_document(store, key)


def send_notice(
    store: Store,
    trading_day: str,
    planned: int,
    *,
    transport: Callable[..., object] | None = None,
) -> str:
    """Read the arc's manifest and its stages, render, deliver. Returns the
    destination. Raises when the manifest cannot be read or nothing received
    the message."""
    arc_key = manifest_key("weekly", trading_day)
    arc = _read(store, arc_key)
    stages: list[tuple[str, Document]] = []
    inputs = arc.get("inputs")
    for raw in cast("list[object]", inputs) if isinstance(inputs, list) else []:
        key = str(cast("dict[str, object]", raw).get("key")) if isinstance(raw, dict) else str(raw)
        if key.startswith("runs/") and key.endswith("/run.json"):
            stages.append((key, _read(store, key)))
    message = render_notice(trading_day, arc, stages, planned)
    # The destination resolves exactly as `report.morning`'s does: the
    # operator chat by name, unless the integration tier's override names
    # `console_only` (so a test run of `weekly` never reaches a phone).
    return report.deliver(
        message,
        severity=NOTICE_SEVERITY,
        source=NOTICE_SOURCE,
        console_artifact=arc_key,
        destination_override_var=TELEGRAM_DESTINATION_OVERRIDE_VAR,
        parse_mode="HTML",
        transport=transport,
    )
