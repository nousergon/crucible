"""The trader's two measured artifacts, as the harness reads them.

Normative sources: plan §4.5 (the fifth attribution row, "execution shortfall
(trader, when present)"), §9.5 (execution-quality row; paper→live entry
condition 2, "the execution-shortfall row populated for every order over that
window, with a declared band"), §10.6 row 6 (shadow books per challenger on
paper); `alpha-engine-config-I10652`, `alpha-engine-config-I10653`.

**Why this module is in the PUBLIC harness.** The PRODUCER of both artifacts is
`nousergon/crucible-trader`, which is private. The consumers are here —
`crucible.report`'s fifth row, and the phase-4 gate clauses that read
presence/status and shadow-book coverage. A contract whose schema lived only
with its producer is a contract the consumer cannot check, so the schemas, the
keys and the validators live here and the trader imports them, exactly as
`trader_evidence.v1` does (`alpha-engine-config-I10648`).

**The harness never reaches into the trader** (plan §3). Nothing here opens a
broker, a fill log or a trader database: each artifact is a document the trader
AGREED to write into the store, validated on read.

Two artifacts, two schemas:

    trader/execution_shortfall/{trading_day}.json   execution_shortfall.v1
    trader/shadow_books/{trading_day}.json          shadow_books.v1

**`shadow_books` is not `shadow.json`.** `crucible.keys.shadow_key` is an arm's
grading-cycle SELECTION; a shadow BOOK is a simulated position with P&L. Two
artifacts sharing one word under one prefix is how `predictions/` came to hold
two shapes distinguishable only by path depth (`alpha-engine-config-I9822`), so
the book lives under its own `trader/shadow_books/` prefix while both are empty.

**Keys live in `crucible.keys`** (`execution_shortfall_key`,
`shadow_books_key`), the one place every store key shape is expressed, and are
re-exported here so a producer imports one module. Both are under the existing
top-level `trader/` prefix the trader's own role writes; the per-key grant for
the two new documents is an IAM change in `nous-ergon-ops` and lands as one.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crucible.documents import UnreadableDocumentError, load_store_document
from crucible.keys import (
    TRADER_EXECUTION_SHORTFALL_PREFIX,
    TRADER_SHADOW_BOOKS_PREFIX,
    execution_shortfall_key,
    shadow_books_key,
)
from crucible.models import MetricRecordRow
from crucible.portfolio import COST_MODEL_KINDS, COST_MODELS
from crucible.store import Store

__all__ = [
    "DECISION_PRICE_BASIS",
    "EXECUTION_METRIC_NAME",
    "EXECUTION_N_FLOOR",
    "EXECUTION_SHORTFALL_SCHEMA_VERSION",
    "SHADOW_BOOKS_SCHEMA_VERSION",
    "SHADOW_BOOK_METRIC_NAME",
    "SHADOW_FILL_BASIS",
    "TRADER_EXECUTION_SHORTFALL_PREFIX",
    "TRADER_SHADOW_BOOKS_PREFIX",
    "ExecutionArtifactError",
    "ExecutionWindow",
    "ShadowCoverage",
    "execution_shortfall_key",
    "reduce_execution_window",
    "shadow_book_coverage",
    "shadow_books_key",
    "validate_execution_shortfall",
    "validate_shadow_books",
    "weighted_shortfall_ci",
]

EXECUTION_SHORTFALL_SCHEMA_VERSION = "execution_shortfall.v1"
SHADOW_BOOKS_SCHEMA_VERSION = "shadow_books.v1"

#: The only decision-price basis the artifact admits: the price at the instant
#: the sizing decision was taken. Recorded in the document, not implied by code.
DECISION_PRICE_BASIS = "sizing_decision"

#: Every simulated shadow fill is at the session close (plan §10.6 row 6).
SHADOW_FILL_BASIS = "close"

#: The row name — identical to `crucible.report.ROWS[-1].name`, asserted by test.
EXECUTION_METRIC_NAME = "execution_shortfall_bps"

#: Per-arm realized paper P&L metric on `shadow_books.v1`.
SHADOW_BOOK_METRIC_NAME = "shadow_book_cumulative_net_return_ratio"

#: Filled orders the fifth row needs over its window before its value is read
#: as a measurement. Twenty is one order per session per name for a four-name
#: book over the five-session window — the floor below which a notional-weighted
#: mean is dominated by one fill. Below half of it the row is N/A-LOW-N.
EXECUTION_N_FLOOR = 20

#: Float tolerance for re-derived quantities. Relative, because notional runs
#: to 1e6 and bps to 1e1 in the same document.
_REL_TOL = 1e-6
_ABS_TOL = 1e-6

_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 20260914

_SCHEMA_DIR = Path(__file__).parent / "schemas"


class ExecutionArtifactError(ValueError):
    """A trader artifact exists and does not honour its contract. RAISED, never skipped."""


def _validator(version: str) -> Draft202012Validator:
    return _checked_validator((_SCHEMA_DIR / f"{version}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _checked_validator(schema_text: str) -> Draft202012Validator:
    """Keyed on the schema TEXT, re-read on every call: an edited schema is
    re-checked, and only the repeat metaschema walk of an unchanged one is
    skipped."""
    schema = json.loads(schema_text)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_errors(document: Mapping[str, Any], version: str, origin: str) -> None:
    errors = sorted(_validator(version).iter_errors(document), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors)
        raise ExecutionArtifactError(f"{origin} does not conform to {version}: {detail}")


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _metric_rows(document: Mapping[str, Any], origin: str) -> list[dict[str, Any]]:
    rows = list(document["metrics"])
    for row in rows:
        try:
            MetricRecordRow.model_validate(row)
        except ValidationError as exc:
            raise ExecutionArtifactError(
                f"{origin}: metric {row.get('name')!r} is not a MetricRecordRow: {exc}"
            ) from exc
    return rows


def _load(store: Store, key: str) -> dict[str, Any]:
    """One present artifact through the package's single strict reader.

    A document that is not a JSON object raises, naming the key — never read
    as absent, because a truncated artifact is a producer defect, not a
    session the trader did not run.
    """
    try:
        return load_store_document(store, key)
    except UnreadableDocumentError as exc:
        raise ExecutionArtifactError(f"{key} is not readable JSON: {exc}") from exc


def _utc(stamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


# -- execution_shortfall.v1 ------------------------------------------------------


def validate_execution_shortfall(
    document: Mapping[str, Any], *, origin: str = EXECUTION_SHORTFALL_SCHEMA_VERSION
) -> dict[str, Any]:
    """Validate one `execution_shortfall.v1` document, schema AND cross-field rules.

    The single validator both sides call: the trader before it writes, the
    report before it reduces. Returns the document as a plain dict.
    """
    _schema_errors(document, EXECUTION_SHORTFALL_SCHEMA_VERSION, origin)
    doc = dict(document)
    outcome = doc["outcome"]
    orders = doc["orders"]
    band = doc["band"]
    if not band["upper_bps"] > band["baseline_bps"]:
        raise ExecutionArtifactError(
            f"{origin}: band upper_bps {band['upper_bps']} does not exceed baseline_bps "
            f"{band['baseline_bps']}; a band with no width cannot tell an acceptable fill "
            "from a breach"
        )

    if outcome == "computed":
        if doc["outcome_reason"] is not None:
            raise ExecutionArtifactError(f"{origin}: a computed document carries no outcome_reason")
        if not orders:
            raise ExecutionArtifactError(
                f"{origin}: outcome 'computed' with zero orders. A session with no orders is "
                "'no_orders' — a fact — and must say so rather than read as a measured zero"
            )
    else:
        if not (doc["outcome_reason"] or "").strip():
            raise ExecutionArtifactError(
                f"{origin}: outcome {outcome!r} must name its reason; an unexplained "
                "not-measured state is indistinguishable from a silent one"
            )
    if outcome == "no_orders" and orders:
        raise ExecutionArtifactError(
            f"{origin}: outcome 'no_orders' carries {len(orders)} order(s)"
        )

    seen: set[str] = set()
    for order in orders:
        _validate_order(order, origin)
        if order["order_id"] in seen:
            raise ExecutionArtifactError(
                f"{origin}: order_id {order['order_id']!r} appears twice; a duplicated fill "
                "double-counts its shortfall"
            )
        seen.add(order["order_id"])

    summary = doc["summary"]
    metrics = _metric_rows(doc, origin)
    if outcome == "not_computed":
        if summary is not None:
            raise ExecutionArtifactError(
                f"{origin}: a not_computed document carries a summary; a figure beside a "
                "producer failure is a figure nobody measured"
            )
        return doc
    if summary is None:
        raise ExecutionArtifactError(f"{origin}: outcome {outcome!r} requires a summary")

    filled = [o for o in orders if o["filled_quantity"] > 0]
    notional = sum(o["notional_usd"] for o in orders)
    decision_notional = sum(o["decision_notional_usd"] for o in orders)
    shortfall_usd = sum(o["shortfall_usd"] for o in orders)
    expected = {
        "n_orders": len(orders),
        "n_filled": len(filled),
    }
    for name, value in expected.items():
        if summary[name] != value:
            raise ExecutionArtifactError(
                f"{origin}: summary.{name} is {summary[name]} but the orders give {value}"
            )
    if not _close(summary["notional_usd"], notional):
        raise ExecutionArtifactError(
            f"{origin}: summary.notional_usd {summary['notional_usd']} != sum of orders {notional}"
        )
    if not _close(summary["decision_notional_usd"], decision_notional):
        raise ExecutionArtifactError(
            f"{origin}: summary.decision_notional_usd {summary['decision_notional_usd']} != "
            f"sum of orders {decision_notional}"
        )
    if not _close(summary["shortfall_usd"], shortfall_usd):
        raise ExecutionArtifactError(
            f"{origin}: summary.shortfall_usd {summary['shortfall_usd']} != sum of orders "
            f"{shortfall_usd}"
        )
    weighted = summary["shortfall_bps_notional_weighted"]
    if decision_notional > 0:
        derived = shortfall_usd / decision_notional * 1e4
        if weighted is None or not _close(weighted, derived):
            raise ExecutionArtifactError(
                f"{origin}: summary.shortfall_bps_notional_weighted {weighted} != "
                f"shortfall_usd / decision_notional_usd × 1e4 = {derived}"
            )
    elif weighted is not None:
        raise ExecutionArtifactError(
            f"{origin}: zero filled notional cannot carry a weighted shortfall ({weighted})"
        )

    named = [m for m in metrics if m["name"] == EXECUTION_METRIC_NAME]
    if outcome == "computed":
        if len(named) != 1:
            raise ExecutionArtifactError(
                f"{origin}: a computed document carries {len(named)} "
                f"{EXECUTION_METRIC_NAME!r} metric(s), exactly one required — the row "
                "arrives as a MetricRecord the way every other producer's rows do"
            )
        value = named[0].get("value")
        if weighted is None:
            if value is not None:
                raise ExecutionArtifactError(
                    f"{origin}: metric value {value} on a session with no filled notional"
                )
        elif value is None or not _close(float(value), weighted):
            raise ExecutionArtifactError(
                f"{origin}: metric {EXECUTION_METRIC_NAME!r} value {value} != summary "
                f"shortfall_bps_notional_weighted {weighted}"
            )
    return doc


def _validate_order(order: Mapping[str, Any], origin: str) -> None:
    oid = order["order_id"]
    if _utc(order["submitted_at_utc"]) < _utc(order["decided_at_utc"]):
        raise ExecutionArtifactError(
            f"{origin}: order {oid!r} submitted at {order['submitted_at_utc']} before its "
            f"sizing decision at {order['decided_at_utc']}. The decision price is the price "
            "at the decision instant; an order that predates its decision was not sized by it"
        )
    if order["filled_quantity"] > order["quantity"] * (1 + _REL_TOL):
        raise ExecutionArtifactError(
            f"{origin}: order {oid!r} filled {order['filled_quantity']} of {order['quantity']}"
        )
    sign = 1.0 if order["side"] == "buy" else -1.0
    if order["filled_quantity"] == 0:
        if order["fill_price"] is not None or order["shortfall_bps"] is not None:
            raise ExecutionArtifactError(
                f"{origin}: unfilled order {oid!r} carries a fill price or shortfall"
            )
        if (
            order["notional_usd"] != 0
            or order["decision_notional_usd"] != 0
            or order["shortfall_usd"] != 0
        ):
            raise ExecutionArtifactError(
                f"{origin}: unfilled order {oid!r} carries non-zero notional or shortfall_usd"
            )
        return
    if order["fill_price"] is None or order["shortfall_bps"] is None:
        raise ExecutionArtifactError(
            f"{origin}: filled order {oid!r} carries no fill price or no shortfall — orders "
            "placed and shortfall not computed is a producer failure ('not_computed')"
        )
    decision = order["decision_price"]
    fill = order["fill_price"]
    qty = order["filled_quantity"]
    checks = {
        "notional_usd": qty * fill,
        "decision_notional_usd": qty * decision,
        "shortfall_bps": sign * (fill - decision) / decision * 1e4,
        "shortfall_usd": sign * (fill - decision) * qty,
    }
    for name, derived in checks.items():
        if not _close(float(order[name]), derived):
            raise ExecutionArtifactError(
                f"{origin}: order {oid!r} {name} {order[name]} != {derived} re-derived from "
                "side, decision_price, fill_price and filled_quantity"
            )


@dataclass(frozen=True)
class ExecutionWindow:
    """What the trader filed over a window of sessions, reduced but not graded."""

    sessions: tuple[str, ...]
    sources: tuple[str, ...]
    missing: tuple[str, ...]
    no_orders: tuple[str, ...]
    not_computed: tuple[tuple[str, str], ...]
    order_bps: tuple[float, ...]
    order_notional: tuple[float, ...]
    order_day: tuple[str, ...]
    notional_usd: float
    decision_notional_usd: float
    shortfall_usd: float
    band: dict[str, Any] | None
    placeholder_band: bool = field(default=False)

    @property
    def filed_any(self) -> bool:
        return bool(self.sources)

    @property
    def n_filled(self) -> int:
        return len(self.order_bps)

    @property
    def weighted_bps(self) -> float | None:
        if self.decision_notional_usd <= 0:
            return None
        return self.shortfall_usd / self.decision_notional_usd * 1e4


def reduce_execution_window(store: Store, sessions: Sequence[str]) -> ExecutionWindow:
    """Read every session's `execution_shortfall.v1` and pool the orders.

    Absence is per session and named (``missing``); a corrupt or
    non-conforming artifact RAISES, naming the key — a truncated document must
    not read as a session the trader simply did not run. Bands must agree
    across the window: pooling shortfall graded against two different red
    lines produces a status measured against neither.
    """
    sources: list[str] = []
    missing: list[str] = []
    no_orders: list[str] = []
    not_computed: list[tuple[str, str]] = []
    bps: list[float] = []
    notional: list[float] = []
    order_day: list[str] = []
    total_notional = 0.0
    total_decision_notional = 0.0
    total_shortfall = 0.0
    band: dict[str, Any] | None = None
    for day in sessions:
        key = execution_shortfall_key(day)
        if not store.exists(key):
            missing.append(day)
            continue
        doc = validate_execution_shortfall(_load(store, key), origin=key)
        if doc["trading_day"] != day:
            raise ExecutionArtifactError(
                f"{key} declares trading_day {doc['trading_day']}; an artifact misfiled under "
                "another session's key would grade that session with this one's fills"
            )
        sources.append(key)
        if band is None:
            band = dict(doc["band"])
        elif dict(doc["band"]) != band:
            raise ExecutionArtifactError(
                f"{key} declares band {doc['band']} but an earlier session in the window "
                f"declared {band}; one row cannot be graded against two red lines"
            )
        if doc["outcome"] == "no_orders":
            no_orders.append(day)
            continue
        if doc["outcome"] == "not_computed":
            not_computed.append((day, str(doc["outcome_reason"])))
            continue
        for order in doc["orders"]:
            if order["filled_quantity"] > 0:
                bps.append(float(order["shortfall_bps"]))
                notional.append(float(order["decision_notional_usd"]))
                order_day.append(day)
        total_notional += float(doc["summary"]["notional_usd"])
        total_decision_notional += float(doc["summary"]["decision_notional_usd"])
        total_shortfall += float(doc["summary"]["shortfall_usd"])
    return ExecutionWindow(
        sessions=tuple(sessions),
        sources=tuple(sources),
        missing=tuple(missing),
        no_orders=tuple(no_orders),
        not_computed=tuple(not_computed),
        order_bps=tuple(bps),
        order_notional=tuple(notional),
        order_day=tuple(order_day),
        notional_usd=total_notional,
        decision_notional_usd=total_decision_notional,
        shortfall_usd=total_shortfall,
        band=band,
        placeholder_band=bool(band and band["placeholder"]),
    )


def weighted_shortfall_ci(
    order_bps: Sequence[float],
    order_notional: Sequence[float],
    order_day: Sequence[str],
) -> tuple[float | None, float | None]:
    """A seeded 95% percentile BLOCK bootstrap of the NOTIONAL-WEIGHTED mean shortfall.

    Orders are not independent: every order filled on the same trading day
    shares that day's decision-to-fill market move, a common factor in all of
    their shortfalls. Resampling orders i.i.d. treats a day of N correlated
    orders as N independent observations and understates the interval width
    by roughly sqrt(orders per day) — on a ~40-name book, close to 6x too
    tight. This resamples whole TRADING DAYS with replacement instead: each
    draw picks k = the number of distinct trading days in the window (with
    replacement), takes every drawn day's orders as one block, and recomputes
    the notional-weighted ratio over the pooled block. That keeps the
    correlation structure inside a day intact across every draw, so the
    interval reflects the actual number of independent observations — trading
    days, not orders.

    ``order_day`` is REQUIRED, not optional: a default would let a call site
    keep resampling orders i.i.d. by omission, silently reintroducing the
    defect this replaces. Fewer than two DISTINCT trading days, or a
    degenerate zero-width result, is no interval (None), never certainty —
    many orders crowded onto a single day is still one observation of that
    day's common factor, not evidence of precision.
    """
    n = len(order_bps)
    if n != len(order_notional) or n != len(order_day):
        raise ValueError(
            f"{n} shortfalls against {len(order_notional)} notionals and "
            f"{len(order_day)} day labels"
        )
    by_day: dict[str, list[int]] = {}
    for i, day in enumerate(order_day):
        by_day.setdefault(day, []).append(i)
    days = sorted(by_day)
    k = len(days)
    if k < 2:
        return (None, None)
    rng = random.Random(_BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        drawn_days = [days[rng.randrange(k)] for _ in range(k)]
        idx = [i for day in drawn_days for i in by_day[day]]
        w = sum(order_notional[i] for i in idx)
        if w <= 0:
            continue
        draws.append(sum(order_bps[i] * order_notional[i] for i in idx) / w)
    if len(draws) < _BOOTSTRAP_RESAMPLES // 2:
        return (None, None)
    draws.sort()
    low = draws[int(0.025 * len(draws))]
    high = draws[int(0.975 * len(draws)) - 1]
    if low == high:
        return (None, None)
    return (low, high)


# -- shadow_books.v1 -------------------------------------------------------------


def validate_shadow_books(
    document: Mapping[str, Any], *, origin: str = SHADOW_BOOKS_SCHEMA_VERSION
) -> dict[str, Any]:
    """Validate one `shadow_books.v1` document, schema AND cross-field rules.

    The cost-model check is the "same cost model the S grade charges" contract
    at the harness boundary: a book's model must be a name in
    `crucible.portfolio.COST_MODELS`, of the kind `COST_MODEL_KINDS` assigns it,
    carrying exactly that model's declared parameters. A shadow book charged by
    anything else measures a different object from the grade.
    """
    _schema_errors(document, SHADOW_BOOKS_SCHEMA_VERSION, origin)
    doc = dict(document)
    active = list(doc["active_arms"])
    if len(set(active)) != len(active):
        raise ExecutionArtifactError(f"{origin}: active_arms lists an arm twice")
    arms = [b["arm_id"] for b in doc["books"]]
    if len(set(arms)) != len(arms):
        raise ExecutionArtifactError(f"{origin}: an arm carries two books")
    uncovered = sorted(set(active) - set(arms))
    if uncovered:
        raise ExecutionArtifactError(
            f"{origin}: active arm(s) {uncovered} carry no book, not even a failed one. An arm "
            "with no shadow book is a finding and is recorded as a failed book — a collection "
            "that quietly shrinks is the min_active_arms defect"
        )
    unregistered = sorted(set(arms) - set(active))
    if unregistered:
        raise ExecutionArtifactError(
            f"{origin}: book(s) for {unregistered}, which are not active in the register read"
        )

    metrics = _metric_rows(doc, origin)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for m in metrics:
        if m["name"] == SHADOW_BOOK_METRIC_NAME:
            by_arm.setdefault(str(m.get("arm_id")), []).append(m)

    for book in doc["books"]:
        _validate_book(book, doc["trading_day"], origin)
        rows = by_arm.get(book["arm_id"], [])
        if book["status"] == "advanced":
            if len(rows) != 1:
                raise ExecutionArtifactError(
                    f"{origin}: advanced book {book['arm_id']!r} carries {len(rows)} "
                    f"{SHADOW_BOOK_METRIC_NAME!r} metric(s), exactly one required"
                )
            value = rows[0].get("value")
            if value is None or not _close(float(value), book["cumulative_net_return_ratio"]):
                raise ExecutionArtifactError(
                    f"{origin}: {book['arm_id']!r} metric value {value} != "
                    f"cumulative_net_return_ratio {book['cumulative_net_return_ratio']}"
                )
    return doc


def _validate_book(book: Mapping[str, Any], trading_day: str, origin: str) -> None:
    arm = book["arm_id"]
    days = list(book["days_advanced"])
    if book["sessions"] != len(days):
        raise ExecutionArtifactError(
            f"{origin}: {arm!r} sessions {book['sessions']} != len(days_advanced) {len(days)}"
        )
    if days != sorted(set(days)):
        raise ExecutionArtifactError(f"{origin}: {arm!r} days_advanced is not strictly increasing")
    numeric = (
        "portfolio_notional_usd",
        "gross_return_ratio",
        "cost_bps",
        "net_return_ratio",
        "cumulative_net_return_ratio",
        "turnover_one_way_ratio",
    )
    if book["status"] == "failed":
        if not (book["failure_reason"] or "").strip():
            raise ExecutionArtifactError(f"{origin}: failed book {arm!r} names no failure_reason")
        if any(book[name] is not None for name in numeric):
            raise ExecutionArtifactError(
                f"{origin}: failed book {arm!r} carries a return figure; a book that did not "
                "advance earned nothing anyone measured"
            )
        return
    if book["failure_reason"] is not None:
        raise ExecutionArtifactError(f"{origin}: advanced book {arm!r} carries a failure_reason")
    missing = [name for name in numeric if book[name] is None]
    if missing or book["cost_model"] is None or book["inception_trading_day"] is None:
        raise ExecutionArtifactError(
            f"{origin}: advanced book {arm!r} is missing {missing or ['cost_model/inception']}"
        )
    if not days or days[-1] != trading_day or days[0] != book["inception_trading_day"]:
        raise ExecutionArtifactError(
            f"{origin}: advanced book {arm!r} days_advanced must run from its inception "
            f"{book['inception_trading_day']} to this session {trading_day}; got "
            f"{days[:1]}..{days[-1:]}"
        )
    model = book["cost_model"]
    name = model["name"]
    if name not in COST_MODELS:
        raise ExecutionArtifactError(
            f"{origin}: {arm!r} is charged by cost model {name!r}, which is not in "
            f"crucible.portfolio.COST_MODELS {sorted(COST_MODELS)}; a shadow book charged "
            "differently from the S grade measures a different object"
        )
    if model["kind"] != COST_MODEL_KINDS[name]:
        raise ExecutionArtifactError(
            f"{origin}: {arm!r} cost model {name!r} recorded as kind {model['kind']!r}; "
            f"the registry says {COST_MODEL_KINDS[name]!r}"
        )
    if set(model["params"]) != set(COST_MODELS[name]):
        raise ExecutionArtifactError(
            f"{origin}: {arm!r} cost model {name!r} declares {sorted(model['params'])}; "
            f"the registry requires exactly {sorted(COST_MODELS[name])}"
        )
    derived_net = book["gross_return_ratio"] - book["cost_bps"] / 1e4
    if not _close(book["net_return_ratio"], derived_net):
        raise ExecutionArtifactError(
            f"{origin}: {arm!r} net_return_ratio {book['net_return_ratio']} != gross − "
            f"cost_bps/1e4 = {derived_net}"
        )


@dataclass(frozen=True)
class ShadowCoverage:
    """A gate-readable reading of shadow-book coverage for one session."""

    met: bool
    detail: str
    sources: tuple[str, ...]


def shadow_book_coverage(
    store: Store,
    trading_day: str,
    *,
    days_served: Sequence[str] | None = None,
) -> ShadowCoverage:
    """Does every active arm carry an advanced shadow book, over every served day?

    The reading a phase-4 clause returns MET/UNMET from (I10653 deliverable 4).
    UNMET, with the reason, when: the session's artifact is absent; any active
    arm's book failed; or — given the trader's own ``days_served`` — an active
    book on or after its inception is missing a day the trader served. A
    non-conforming artifact RAISES rather than reading UNMET, because a
    corrupt document is a producer defect, not a coverage gap.
    """
    key = shadow_books_key(trading_day)
    if not store.exists(key):
        return ShadowCoverage(
            False,
            f"{key} is absent — the trader filed no shadow books for {trading_day}",
            (key,),
        )
    doc = validate_shadow_books(_load(store, key), origin=key)
    if not doc["active_arms"]:
        return ShadowCoverage(
            False,
            f"{key}: the register the trader read lists no active arm, so no shadow book "
            "measures anything",
            (key,),
        )
    findings: list[str] = []
    for book in doc["books"]:
        if book["status"] == "failed":
            findings.append(f"{book['arm_id']}: failed ({book['failure_reason']})")
            continue
        if days_served is not None:
            inception = book["inception_trading_day"]
            advanced = set(book["days_advanced"])
            gap = [d for d in days_served if d >= inception and d not in advanced]
            if gap:
                findings.append(f"{book['arm_id']}: not advanced on served day(s) {gap}")
    if findings:
        return ShadowCoverage(False, f"{key}: " + "; ".join(findings), (key,))
    return ShadowCoverage(
        True,
        f"{key}: {len(doc['books'])} active arm(s), every book advanced"
        + ("" if days_served is None else f" on all {len(days_served)} served day(s)"),
        (key,),
    )
