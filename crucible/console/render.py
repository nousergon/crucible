"""Build the console page's data, then render it.

Split in two on purpose: :func:`build_page` produces a plain dict that a test
can assert against and an agent can read, and :func:`render_html` turns that
into the page. A renderer that computed its own numbers would put the console
and its JSON out of step, and the JSON is what the next reader parses.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from dataclasses import dataclass, field
from typing import Any

from crucible.calendar import previous_trading_day, resolve_trading_day
from crucible.components import Component, load_registry
from crucible.console.classify import Classification, classify
from crucible.manifest import manifest_key
from crucible.store import Store

__all__ = ["CONSOLE_KEY", "ConsolePage", "build_page", "render_html", "write_page"]

CONSOLE_KEY = "console/index.html"
CONSOLE_JSON_KEY = "console/index.json"


@dataclass
class ConsolePage:
    """Everything the page shows, as data.

    ``unreported`` is a top-level field rather than something a reader counts
    off the rows: §8.4 makes it the transparency-gap count with an objective
    of zero, and a number nobody publishes is a number nobody is held to.
    """

    trading_day: str
    generated_utc: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    attribution: list[dict[str, Any]] = field(default_factory=list)
    champions: dict[str, Any] = field(default_factory=dict)
    deploys: list[dict[str, Any]] = field(default_factory=list)
    week_cost_usd: float = 0.0
    unreported: int = 0
    population: int = 0

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__, indent=2, sort_keys=True).encode("utf-8")


def _read_json(store: Store, key: str) -> dict[str, Any] | None:
    """Read one JSON artifact, or None when it is absent.

    An UNREADABLE artifact raises. Absence and corruption are different facts,
    and returning None for both would let a corrupt manifest render as a
    component that simply had not run.
    """
    if not store.exists(key):
        return None
    return json.loads(store.get_bytes(key).decode("utf-8"))


def _has_history(store: Store, job: str) -> bool:
    """Whether this component has ever produced a manifest.

    The whole difference between MISSED and NEVER_RAN, so it is a real
    listing rather than an assumption. Short-circuits on the first hit — the
    question is existential, not a count.
    """
    prefix = f"runs/{job}/"
    for key in store.list_keys(prefix):
        if key.endswith("/run.json"):
            return True
    return False


def build_page(
    store: Store,
    *,
    now: dt.datetime | None = None,
    registry: dict[str, Component] | None = None,
    run_days: int = 5,
) -> ConsolePage:
    """Assemble the page from the store. Reads artifacts; computes no state.

    ``run_days`` is a count of TRADING days (§4.12), so a holiday week shows
    the same number of sessions as any other rather than four.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    reg = registry if registry is not None else load_registry()
    trading_day = resolve_trading_day(moment)

    days = [trading_day]
    for _ in range(run_days - 1):
        days.append(previous_trading_day(days[-1]))
    day_set = {d.isoformat() for d in days}

    rows: list[dict[str, Any]] = []
    for name, component in sorted(reg.items()):
        manifest = _read_json(store, manifest_key(name, trading_day.isoformat()))
        classification = classify(
            component,
            manifest,
            now=moment,
            history=manifest is not None or _has_history(store, name),
        )
        rows.append(_row(component, classification, manifest))

    week_cost = 0.0
    deploys: list[dict[str, Any]] = []
    for key in store.list_keys("runs/"):
        if not key.endswith("/run.json"):
            continue
        parts = key.split("/")
        if len(parts) != 4 or parts[2] not in day_set:
            continue
        manifest = json.loads(store.get_bytes(key).decode("utf-8"))
        week_cost += float(manifest.get("cost_usd", 0.0))
        if parts[1] == "deploy":
            deploys.append(
                {
                    "trading_day": manifest.get("trading_day"),
                    "status": manifest.get("status"),
                    "release_sha": manifest.get("release_sha"),
                    "reason": manifest.get("reason"),
                    "run_id": manifest.get("run_id"),
                }
            )

    attribution = _read_json(store, f"report/{trading_day.isoformat()}/attribution.json")
    champions = _champions(store)

    return ConsolePage(
        trading_day=trading_day.isoformat(),
        generated_utc=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows=rows,
        # Track A owns `report`; until it lands there is no attribution
        # artifact, and the page says so rather than showing an empty table
        # that reads like five rows of zero.
        attribution=attribution.get("rows", []) if attribution else [],
        champions=champions,
        deploys=sorted(deploys, key=lambda d: str(d.get("trading_day"))),
        week_cost_usd=round(week_cost, 4),
        unreported=sum(1 for r in rows if r["state"] == "UNREPORTED"),
        population=len(rows),
    )


def _row(
    component: Component,
    classification: Classification,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "component": component.name,
        "state": classification.state,
        "reason": classification.reason,
        "lifecycle": component.lifecycle,
        "schedule": component.schedule,
        "deadline": component.deadline.describe() if component.deadline else None,
        "absence_watched_by": component.absence_watched_by,
        "console_surface": component.console_surface,
        "log_location": component.log_location,
        "run_id": classification.run_id,
        "trading_day": classification.trading_day,
        "cost_usd": (manifest or {}).get("cost_usd"),
        "attempts": len((manifest or {}).get("attempts", []) or []) or None,
    }


def _champions(store: Store) -> dict[str, Any]:
    """The champion pointer per slot, or an honest absence.

    Track B owns `promote` and therefore the pointers. Reading them here is a
    one-way dependency on an artifact contract, not on their code.
    """
    out: dict[str, Any] = {}
    for slot in ("u", "r", "m", "s"):
        key = f"champions/{slot}/current.json"
        payload = _read_json(store, key)
        out[slot] = payload if payload else None
    return out


_STYLE = """
:root { color-scheme: light dark; --fg:#111; --bg:#fff; --muted:#666; --line:#ddd; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#111; --muted:#999; --line:#333; }
}
body { font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; color: var(--fg);
       background: var(--bg); margin: 0; padding: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; }
.sub { color: var(--muted); margin: 0 0 16px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-weight: 600; white-space: nowrap; }
code { font: 12px ui-monospace, monospace; }
.reason { color: var(--muted); max-width: 60ch; }
.state { font-weight: 600; white-space: nowrap; }
.s-HEALTHY { color: #1a7f37; } .s-RUNNING, .s-ARMED { color: #6e7781; }
.s-DEGRADED, .s-WATCH { color: #9a6700; }
.s-FAILED, .s-MISSED, .s-STALLED, .s-ABSENT, .s-UNREPORTED,
.s-UNREGISTERED, .s-NEVER_RAN { color: #cf222e; }
.s-DISABLED, .s-DEPRECATED, .s-RETIRED { color: #8250df; }
.gap { font-weight: 600; }
.gap-zero { color: #1a7f37; } .gap-nonzero { color: #cf222e; }
.empty { color: var(--muted); font-style: italic; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_html(page: ConsolePage) -> str:
    """One page, no JavaScript, no polling.

    The transparency-gap count is rendered at the top and is red whenever it
    is non-zero — including when it is non-zero for a good reason. §8.4 makes
    it the number that says the surface is complete, and a count that renders
    calmly is a count nobody acts on.
    """
    gap_class = "gap-zero" if page.unreported == 0 else "gap-nonzero"
    parts = [
        "<title>Crucible v2</title>",
        f"<style>{_STYLE}</style>",
        "<h1>Crucible v2</h1>",
        f'<p class="sub">Trading day <code>{_e(page.trading_day)}</code> · generated '
        f"<code>{_e(page.generated_utc)}</code> · week cost "
        f"<code>${page.week_cost_usd:.2f}</code> · population {page.population} · "
        f'transparency gap <span class="gap {gap_class}">{page.unreported}</span> '
        "(objective 0)</p>",
        "<h2>Components</h2>",
        '<div class="wrap"><table><thead><tr>'
        "<th>Component</th><th>State</th><th>Schedule</th><th>Deadline</th>"
        "<th>Absence watched by</th><th>Run</th><th>Cost</th><th>Why</th>"
        "</tr></thead><tbody>",
    ]
    for row in page.rows:
        cost = "" if row["cost_usd"] is None else f"${float(row['cost_usd']):.4f}"
        attempts = f" ({row['attempts']}x)" if (row["attempts"] or 1) > 1 else ""
        parts.append(
            "<tr>"
            f"<td><code>{_e(row['component'])}</code></td>"
            f'<td class="state s-{_e(row["state"])}">{_e(row["state"])}</td>'
            f"<td>{_e(row['schedule'])}</td>"
            f"<td>{_e(row['deadline'])}</td>"
            f"<td><code>{_e(row['absence_watched_by'])}</code></td>"
            f"<td><code>{_e(row['run_id'])}</code>{attempts}</td>"
            f"<td>{_e(cost)}</td>"
            f'<td class="reason">{_e(row["reason"])}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")

    parts.append("<h2>Attribution</h2>")
    if page.attribution:
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Row</th><th>Value</th>'
            "<th>Status</th><th>Why</th></tr></thead><tbody>"
        )
        for row in page.attribution:
            parts.append(
                "<tr>"
                f"<td><code>{_e(row.get('name'))}</code></td>"
                f"<td>{_e(row.get('value'))} {_e(row.get('unit'))}</td>"
                f'<td class="state s-{_e(row.get("status"))}">{_e(row.get("status"))}</td>'
                f'<td class="reason">{_e(row.get("status_reason"))}</td>'
                "</tr>"
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append(
            '<p class="empty">No attribution artifact for this trading day. This is an '
            "absence, not five rows of zero — <code>crucible report</code> has not "
            "written <code>report/{trading_day}/attribution.json</code>.</p>"
        )

    parts.append("<h2>Champions</h2>")
    parts.append(
        '<div class="wrap"><table><thead><tr><th>Slot</th><th>Champion</th>'
        "<th>Since</th></tr></thead><tbody>"
    )
    for slot, champ in sorted(page.champions.items()):
        if champ:
            parts.append(
                f"<tr><td><code>{_e(slot)}</code></td>"
                f"<td><code>{_e(champ.get('arm_id'))}</code></td>"
                f"<td>{_e(champ.get('promoted_trading_day'))}</td></tr>"
            )
        else:
            parts.append(
                f"<tr><td><code>{_e(slot)}</code></td>"
                '<td class="empty" colspan="2">no pointer written</td></tr>'
            )
    parts.append("</tbody></table></div>")

    parts.append("<h2>Deploys</h2>")
    if page.deploys:
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Trading day</th><th>Status</th>'
            "<th>Release</th><th>Why</th></tr></thead><tbody>"
        )
        for dep in page.deploys:
            state = "HEALTHY" if dep.get("status") == "ok" else "FAILED"
            parts.append(
                f"<tr><td>{_e(dep.get('trading_day'))}</td>"
                f'<td class="state s-{state}">{_e(dep.get("status"))}</td>'
                f"<td><code>{_e(str(dep.get('release_sha'))[:12])}</code></td>"
                f'<td class="reason">{_e(dep.get("reason"))}</td></tr>'
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append('<p class="empty">No deploy manifests in the window.</p>')

    return "\n".join(parts)


def write_page(store: Store, page: ConsolePage) -> tuple[str, str]:
    """Write the HTML and the JSON. Returns both keys.

    The JSON is not an extra: `console-policy` requires every view to serve
    the JSON an agent reads, and a page whose numbers can only be scraped out
    of HTML is a page the next automated reader re-derives incorrectly.
    """
    store.put_bytes(CONSOLE_KEY, render_html(page).encode("utf-8"))
    store.put_bytes(CONSOLE_JSON_KEY, page.to_json())
    return CONSOLE_KEY, CONSOLE_JSON_KEY
