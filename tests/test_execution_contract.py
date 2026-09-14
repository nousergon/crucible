"""Producer/consumer contract for the trader's two measured artifacts.

`alpha-engine-config-I10652` (execution-shortfall row) and `-I10653` (shadow
books per challenger). M0 rule: every new cross-repo artifact gets a versioned
schema plus a producer/consumer contract test at birth. The PRODUCER is
`nousergon/crucible-trader` (private), whose own suite validates what it writes
through `crucible.execution.validate_*` as shipped in its pinned wheel; the
CONSUMER is `crucible.report`'s fifth row and `crucible.execution.shadow_book_coverage`.

The negative cases are the point: a validator that only ever sees a happy
document grades nothing.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from crucible.execution import (
    DECISION_PRICE_BASIS,
    EXECUTION_METRIC_NAME,
    EXECUTION_N_FLOOR,
    EXECUTION_SHORTFALL_SCHEMA_VERSION,
    SHADOW_BOOK_METRIC_NAME,
    SHADOW_BOOKS_SCHEMA_VERSION,
    SHADOW_FILL_BASIS,
    ExecutionArtifactError,
    execution_shortfall_key,
    reduce_execution_window,
    shadow_book_coverage,
    shadow_books_key,
    validate_execution_shortfall,
    validate_shadow_books,
    weighted_shortfall_ci,
)
from crucible.keys import shadow_key
from crucible.portfolio import COST_MODELS, CostModel
from crucible.report import REPORT_WINDOW_TRADING_DAYS, ROWS, build_attribution
from crucible.store import LocalStore

SCHEMAS = Path(__file__).resolve().parents[1] / "crucible" / "schemas"
DAY = "2026-09-11"
WEEK = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14"]
NOW = dt.datetime(2026, 9, 14, 21, 0, tzinfo=dt.UTC)
BAND = {"baseline_bps": 0.0, "upper_bps": 25.0, "placeholder": True}
ARM = "s:contract_fixture:0123456789ab"
CHALLENGER = "s:contract_fixture:ba9876543210"


def _order(
    oid: str,
    *,
    side: str = "buy",
    decision: float = 100.0,
    fill: float | None = 100.05,
    qty: float = 100.0,
    filled: float | None = None,
) -> dict:
    filled_qty = qty if filled is None else filled
    sign = 1.0 if side == "buy" else -1.0
    if fill is None or filled_qty == 0:
        return {
            "order_id": oid,
            "symbol": "AAA",
            "side": side,
            "quantity": qty,
            "decided_at_utc": "2026-09-11T19:55:00Z",
            "decision_price": decision,
            "submitted_at_utc": "2026-09-11T19:55:02Z",
            "filled_quantity": 0,
            "fill_price": None,
            "notional_usd": 0,
            "decision_notional_usd": 0,
            "shortfall_bps": None,
            "shortfall_usd": 0,
        }
    return {
        "order_id": oid,
        "symbol": "AAA",
        "side": side,
        "quantity": qty,
        "decided_at_utc": "2026-09-11T19:55:00Z",
        "decision_price": decision,
        "submitted_at_utc": "2026-09-11T19:55:02Z",
        "filled_quantity": filled_qty,
        "fill_price": fill,
        "notional_usd": filled_qty * fill,
        "decision_notional_usd": filled_qty * decision,
        "shortfall_bps": sign * (fill - decision) / decision * 1e4,
        "shortfall_usd": sign * (fill - decision) * filled_qty,
    }


def _metric(value: float | None, **extra: object) -> dict:
    row = {
        "name": EXECUTION_METRIC_NAME,
        "module": "crucible_trader.execution_shortfall",
        "metric_type": "ratio",
        "value": value,
        "unit": "bps" if value is not None else None,
        "n_floor": 1,
        "status": "GREEN" if value is not None else "N/A-LOW-N",
        "status_reason": "per-order shortfall against the sizing-decision price",
        "source_path": "trader/execution_shortfall/",
        "last_updated_utc": "2026-09-11T21:00:00Z",
    }
    row.update(extra)
    return row


def _shortfall_doc(day: str = DAY, orders: list[dict] | None = None, **overrides: object) -> dict:
    orders = [_order("o1"), _order("o2", side="sell", fill=99.9)] if orders is None else orders
    notional = sum(o["notional_usd"] for o in orders)
    decision_notional = sum(o["decision_notional_usd"] for o in orders)
    shortfall = sum(o["shortfall_usd"] for o in orders)
    weighted = shortfall / decision_notional * 1e4 if decision_notional > 0 else None
    doc = {
        "schema_version": EXECUTION_SHORTFALL_SCHEMA_VERSION,
        "trading_day": day,
        "calendar_date": day,
        "champion": "m:ridge_21d:0123456789ab",
        "decision_price_basis": DECISION_PRICE_BASIS,
        "shortfall_scope": "filled_quantity_vs_decision_price",
        "outcome": "computed",
        "outcome_reason": None,
        "band": dict(BAND),
        "orders": orders,
        "summary": {
            "n_orders": len(orders),
            "n_filled": sum(1 for o in orders if o["filled_quantity"]),
            "notional_usd": notional,
            "decision_notional_usd": decision_notional,
            "shortfall_usd": shortfall,
            "shortfall_bps_notional_weighted": weighted,
        },
        "metrics": [_metric(weighted)],
    }
    doc.update(overrides)
    return doc


def _no_orders_doc(day: str) -> dict:
    return _shortfall_doc(
        day,
        orders=[],
        outcome="no_orders",
        outcome_reason="the book needed no rebalance",
        summary={
            "n_orders": 0,
            "n_filled": 0,
            "notional_usd": 0.0,
            "decision_notional_usd": 0.0,
            "shortfall_usd": 0.0,
            "shortfall_bps_notional_weighted": None,
        },
        metrics=[],
    )


def _put(store: LocalStore, key: str, doc: dict) -> None:
    store.put_bytes(key, json.dumps(doc).encode("utf-8"))


class TestTheSchemasArePublishedAndVersioned:
    @pytest.mark.parametrize(
        "version", [EXECUTION_SHORTFALL_SCHEMA_VERSION, SHADOW_BOOKS_SCHEMA_VERSION]
    )
    def test_each_schema_is_valid_draft_2020_12_and_names_its_version(self, version) -> None:
        schema = json.loads((SCHEMAS / f"{version}.json").read_text())
        Draft202012Validator.check_schema(schema)
        assert schema["properties"]["schema_version"]["const"] == version

    def test_the_report_row_name_is_the_one_the_producer_emits(self) -> None:
        assert ROWS[-1].name == EXECUTION_METRIC_NAME

    def test_the_shadow_book_key_is_not_the_grading_selection_key(self) -> None:
        """I10653 gotcha: `shadow.json` is an arm's SELECTION; a book is a
        different object and must not share its key shape."""
        assert not shadow_books_key(DAY).startswith("experiments/")
        assert shadow_books_key(DAY) != shadow_key(ARM, DAY)
        assert shadow_books_key(DAY) == f"trader/shadow_books/{DAY}.json"
        assert execution_shortfall_key(DAY) == f"trader/execution_shortfall/{DAY}.json"

    def test_a_key_from_a_non_date_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not an ISO calendar date"):
            execution_shortfall_key("2026-02-31")
        with pytest.raises(ValueError, match="not an ISO calendar date"):
            shadow_books_key("today")


class TestExecutionShortfallValidator:
    def test_a_computed_document_validates(self) -> None:
        assert validate_execution_shortfall(_shortfall_doc())["outcome"] == "computed"

    def test_the_decision_price_basis_is_recorded_and_closed(self) -> None:
        with pytest.raises(ExecutionArtifactError, match="decision_price_basis"):
            validate_execution_shortfall(_shortfall_doc(decision_price_basis="arrival"))

    def test_an_order_submitted_before_its_decision_is_refused(self) -> None:
        doc = _shortfall_doc()
        doc["orders"][0]["submitted_at_utc"] = "2026-09-11T19:54:59Z"
        with pytest.raises(ExecutionArtifactError, match="before its sizing decision"):
            validate_execution_shortfall(doc)

    def test_a_shortfall_not_rederivable_from_the_decision_price_is_refused(self) -> None:
        doc = _shortfall_doc()
        doc["orders"][0]["shortfall_bps"] = 0.0
        with pytest.raises(ExecutionArtifactError, match="re-derived"):
            validate_execution_shortfall(doc)

    def test_computed_with_zero_orders_is_refused(self) -> None:
        doc = _no_orders_doc(DAY)
        doc["outcome"] = "computed"
        doc["outcome_reason"] = None
        with pytest.raises(ExecutionArtifactError, match="zero orders"):
            validate_execution_shortfall(doc)

    def test_no_orders_carrying_orders_is_refused(self) -> None:
        doc = _shortfall_doc(outcome="no_orders", outcome_reason="x")
        with pytest.raises(ExecutionArtifactError, match="carries 2 order"):
            validate_execution_shortfall(doc)

    def test_a_not_measured_outcome_must_name_its_reason(self) -> None:
        doc = _shortfall_doc(outcome="not_computed", outcome_reason=" ", summary=None, metrics=[])
        with pytest.raises(ExecutionArtifactError, match="must name its reason"):
            validate_execution_shortfall(doc)

    def test_not_computed_with_a_summary_is_refused(self) -> None:
        doc = _shortfall_doc(outcome="not_computed", outcome_reason="broker fill feed down")
        with pytest.raises(ExecutionArtifactError, match="carries a summary"):
            validate_execution_shortfall(doc)

    def test_a_filled_order_with_no_shortfall_is_refused(self) -> None:
        doc = _shortfall_doc()
        doc["orders"][0]["shortfall_bps"] = None
        with pytest.raises(ExecutionArtifactError, match="producer failure"):
            validate_execution_shortfall(doc)

    @pytest.mark.parametrize(
        "field", ["notional_usd", "decision_notional_usd", "shortfall_usd", "n_filled"]
    )
    def test_a_summary_that_disagrees_with_its_orders_is_refused(self, field) -> None:
        doc = _shortfall_doc()
        doc["summary"][field] += 1000
        with pytest.raises(ExecutionArtifactError, match=field):
            validate_execution_shortfall(doc)

    def test_a_weighted_figure_on_zero_filled_notional_is_refused(self) -> None:
        doc = _shortfall_doc(orders=[_order("o1", fill=None)], metrics=[_metric(None)])
        doc["summary"]["shortfall_bps_notional_weighted"] = 1.0
        with pytest.raises(ExecutionArtifactError, match="zero filled notional"):
            validate_execution_shortfall(doc)

    def test_a_metric_value_on_zero_filled_notional_is_refused(self) -> None:
        doc = _shortfall_doc(orders=[_order("o1", fill=None)], metrics=[_metric(3.0)])
        with pytest.raises(ExecutionArtifactError, match="no filled notional"):
            validate_execution_shortfall(doc)

    def test_the_weighted_figure_over_identical_orders_is_each_orders_own_bps(self) -> None:
        doc = validate_execution_shortfall(_shortfall_doc(orders=[_order("a"), _order("b")]))
        assert doc["summary"]["shortfall_bps_notional_weighted"] == pytest.approx(
            doc["orders"][0]["shortfall_bps"]
        )

    def test_the_metric_must_equal_the_summary(self) -> None:
        doc = _shortfall_doc()
        doc["metrics"] = [_metric(99.0)]
        with pytest.raises(ExecutionArtifactError, match="!= summary"):
            validate_execution_shortfall(doc)

    def test_a_computed_document_without_its_metric_is_refused(self) -> None:
        with pytest.raises(ExecutionArtifactError, match="exactly one required"):
            validate_execution_shortfall(_shortfall_doc(metrics=[]))

    def test_a_duplicated_order_is_refused(self) -> None:
        orders = [_order("o1"), _order("o1")]
        with pytest.raises(ExecutionArtifactError, match="appears twice"):
            validate_execution_shortfall(_shortfall_doc(orders=orders))

    def test_a_band_with_no_width_is_refused(self) -> None:
        doc = _shortfall_doc(band={"baseline_bps": 5.0, "upper_bps": 5.0, "placeholder": False})
        with pytest.raises(ExecutionArtifactError, match="no width"):
            validate_execution_shortfall(doc)

    def test_an_unfilled_order_carries_no_price_and_is_admitted(self) -> None:
        orders = [_order("o1"), _order("o2", fill=None)]
        doc = validate_execution_shortfall(_shortfall_doc(orders=orders))
        assert doc["summary"]["n_filled"] == 1

    def test_an_unfilled_order_with_a_price_is_refused(self) -> None:
        orders = [_order("o1"), _order("o2", fill=None)]
        orders[1]["fill_price"] = 100.0
        with pytest.raises(ExecutionArtifactError, match="unfilled order"):
            validate_execution_shortfall(_shortfall_doc(orders=orders))

    def test_an_overfilled_order_is_refused(self) -> None:
        orders = [_order("o1", qty=10.0, filled=11.0)]
        with pytest.raises(ExecutionArtifactError, match="filled 11.0 of 10.0"):
            validate_execution_shortfall(_shortfall_doc(orders=orders))

    def test_a_computed_document_with_a_reason_is_refused(self) -> None:
        with pytest.raises(ExecutionArtifactError, match="no outcome_reason"):
            validate_execution_shortfall(_shortfall_doc(outcome_reason="why"))

    def test_a_non_metric_record_is_refused(self) -> None:
        doc = _shortfall_doc()
        doc["metrics"] = [
            {**_metric(doc["summary"]["shortfall_bps_notional_weighted"]), "unit": None}
        ]
        with pytest.raises(ExecutionArtifactError, match="not a MetricRecordRow"):
            validate_execution_shortfall(doc)


def _week_store(tmp_path, docs: dict[str, dict]) -> LocalStore:
    store = LocalStore(tmp_path)
    for day, doc in docs.items():
        _put(store, execution_shortfall_key(day), doc)
    return store


def _execution_row(store: LocalStore) -> dict:
    document, sources = build_attribution(
        store, trading_day=dt.date(2026, 9, 14), now=NOW, run_id="R" * 26
    )
    row = document["rows"][-1]
    assert row["name"] == EXECUTION_METRIC_NAME
    assert len(document["rows"]) == 5, "the table is five rows whether or not the trader ran"
    row["_sources"] = sources
    return row


def _many_orders(n: int, fill: float) -> list[dict]:
    return [_order(f"o{i}", fill=fill) for i in range(n)]


class TestTheFifthRow:
    def test_a_traded_week_populates_the_row_with_baseline_band_and_absolutes(
        self, tmp_path
    ) -> None:
        docs = {
            day: _shortfall_doc(day, orders=_many_orders(5, 100.02 + 0.01 * i))
            for i, day in enumerate(WEEK)
        }
        row = _execution_row(_week_store(tmp_path, docs))
        assert row["status"] in ("GREEN", "WATCH")
        assert row["unit"] == "bps"
        assert row["n_samples"] == 25 >= EXECUTION_N_FLOOR
        assert row["baseline"] == 0.0
        assert row["band"] == BAND
        assert row["red_line"] == 25.0
        assert row["notional_usd"] > 0 and row["shortfall_usd"] > 0
        assert row["decision_price_basis"] == DECISION_PRICE_BASIS
        assert row["window_trading_days"] == REPORT_WINDOW_TRADING_DAYS
        assert "PLACEHOLDER band" in row["status_reason"]
        assert sorted(row["_sources"]) == sorted(execution_shortfall_key(d) for d in WEEK)

    def test_a_week_with_no_orders_renders_a_named_na(self, tmp_path) -> None:
        """Closes-when 2: verified by running the reducer against such a week."""
        row = _execution_row(_week_store(tmp_path, {d: _no_orders_doc(d) for d in WEEK}))
        assert row["status"] == "N/A-LOW-N"
        assert row["value"] is None
        assert "placed no filled order" in row["status_reason"]
        assert row["sessions_no_orders"] == WEEK

    def test_orders_placed_and_not_computed_is_red_not_absent(self, tmp_path) -> None:
        docs = {d: _shortfall_doc(d, orders=_many_orders(5, 100.01)) for d in WEEK[:4]}
        docs[WEEK[4]] = _shortfall_doc(
            WEEK[4],
            outcome="not_computed",
            outcome_reason="broker fill report never arrived",
            summary=None,
            metrics=[],
        )
        row = _execution_row(_week_store(tmp_path, docs))
        assert row["status"] == "RED"
        assert "broker fill report never arrived" in row["status_reason"]
        assert row["sessions_not_computed"] == [WEEK[4]]

    def test_shortfall_beyond_the_band_is_red(self, tmp_path) -> None:
        docs = {d: _shortfall_doc(d, orders=_many_orders(5, 100.40)) for d in WEEK}
        row = _execution_row(_week_store(tmp_path, docs))
        assert row["value"] == pytest.approx(40.0)
        assert row["status"] == "RED"

    def test_missing_sessions_are_named_not_hidden(self, tmp_path) -> None:
        docs = {d: _shortfall_doc(d, orders=_many_orders(6, 100.01)) for d in WEEK[:4]}
        row = _execution_row(_week_store(tmp_path, docs))
        assert row["sessions_missing"] == [WEEK[4]]
        assert WEEK[4] in row["status_reason"]

    def test_a_corrupt_artifact_raises_naming_the_key(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(execution_shortfall_key(WEEK[0]), b"{not json")
        with pytest.raises(ExecutionArtifactError, match=execution_shortfall_key(WEEK[0])):
            _execution_row(store)

    def test_an_artifact_misfiled_under_another_session_is_refused(self, tmp_path) -> None:
        store = _week_store(tmp_path, {WEEK[0]: _shortfall_doc(WEEK[1])})
        with pytest.raises(ExecutionArtifactError, match="misfiled"):
            _execution_row(store)

    def test_two_bands_in_one_window_are_refused(self, tmp_path) -> None:
        other = {"baseline_bps": 0.0, "upper_bps": 10.0, "placeholder": False}
        docs = {WEEK[0]: _shortfall_doc(WEEK[0]), WEEK[1]: _shortfall_doc(WEEK[1], band=other)}
        with pytest.raises(ExecutionArtifactError, match="two red lines"):
            _execution_row(_week_store(tmp_path, docs))


class TestWeightedCI:
    def test_fewer_than_two_orders_has_no_interval(self) -> None:
        assert weighted_shortfall_ci([5.0], [100.0]) == (None, None)

    def test_identical_orders_have_no_interval(self) -> None:
        assert weighted_shortfall_ci([5.0] * 10, [100.0] * 10) == (None, None)

    def test_the_interval_brackets_the_weighted_mean(self) -> None:
        bps = [1.0, 2.0, 8.0, 3.0, 5.0, 4.0]
        notional = [10.0, 30.0, 5.0, 20.0, 15.0, 20.0]
        low, high = weighted_shortfall_ci(bps, notional)
        mean = sum(b * n for b, n in zip(bps, notional, strict=True)) / sum(notional)
        assert low is not None and high is not None and low < mean < high

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="notionals"):
            weighted_shortfall_ci([1.0, 2.0], [1.0])

    def test_all_zero_notional_has_no_interval(self) -> None:
        assert weighted_shortfall_ci([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == (None, None)


def test_the_reducer_reads_nothing_when_nothing_was_filed(tmp_path) -> None:
    reading = reduce_execution_window(LocalStore(tmp_path), WEEK)
    assert not reading.filed_any and reading.missing == tuple(WEEK)
    assert reading.weighted_bps is None


# -- shadow_books.v1 -------------------------------------------------------------

FLAT = CostModel(
    name="flat_bps_v0",
    placeholder=True,
    params={"half_spread_bps": 2.5, "commission_bps": 0.5, "slippage_bps": 10.0},
)


def _book(
    arm: str = ARM,
    *,
    days: list[str] | None = None,
    gross: float = 0.004,
    cost: float = 3.0,
    cumulative: float = 0.012,
) -> dict:
    days = [WEEK[0], WEEK[1], WEEK[2], DAY] if days is None else days
    return {
        "arm_id": arm,
        "status": "advanced",
        "failure_reason": None,
        "cost_model": FLAT.record(),
        "inception_trading_day": days[0],
        "days_advanced": days,
        "sessions": len(days),
        "portfolio_notional_usd": 1_000_000.0,
        "gross_return_ratio": gross,
        "cost_bps": cost,
        "net_return_ratio": gross - cost / 1e4,
        "cumulative_net_return_ratio": cumulative,
        "turnover_one_way_ratio": 0.1,
        "weights": {"AAA": 0.5, "CASH": 0.5},
    }


def _failed(arm: str, reason: str = "no ADV for a participation-aware model") -> dict:
    return {
        "arm_id": arm,
        "status": "failed",
        "failure_reason": reason,
        "cost_model": None,
        "inception_trading_day": None,
        "days_advanced": [],
        "sessions": 0,
        "portfolio_notional_usd": None,
        "gross_return_ratio": None,
        "cost_bps": None,
        "net_return_ratio": None,
        "cumulative_net_return_ratio": None,
        "turnover_one_way_ratio": None,
        "weights": {},
    }


def _shadow_metric(book: dict) -> dict:
    return {
        "name": SHADOW_BOOK_METRIC_NAME,
        "module": "crucible_trader.shadow_books",
        "metric_type": "ratio",
        "value": book["cumulative_net_return_ratio"],
        "unit": "ratio",
        "n_floor": 1,
        "status": "GREEN",
        "status_reason": "cumulative paper P&L, fills at close, net of the grade's cost model",
        "source_path": "trader/shadow_books/",
        "last_updated_utc": "2026-09-11T21:00:00Z",
        "arm_id": book["arm_id"],
    }


def _shadow_doc(books: list[dict] | None = None, active: list[str] | None = None) -> dict:
    books = [_book(ARM), _book(CHALLENGER)] if books is None else books
    return {
        "schema_version": SHADOW_BOOKS_SCHEMA_VERSION,
        "trading_day": DAY,
        "calendar_date": DAY,
        "fill_basis": SHADOW_FILL_BASIS,
        "slot": "s",
        "active_arms": [b["arm_id"] for b in books] if active is None else active,
        "books": books,
        "metrics": [_shadow_metric(b) for b in books if b["status"] == "advanced"],
    }


class TestShadowBooksValidator:
    def test_a_conforming_document_validates(self) -> None:
        assert len(validate_shadow_books(_shadow_doc())["books"]) == 2

    def test_an_active_arm_with_no_book_is_refused_not_dropped(self) -> None:
        doc = _shadow_doc(books=[_book(ARM)], active=[ARM, CHALLENGER])
        with pytest.raises(ExecutionArtifactError, match="carry no book"):
            validate_shadow_books(doc)

    def test_a_book_for_an_inactive_arm_is_refused(self) -> None:
        doc = _shadow_doc(active=[ARM])
        with pytest.raises(ExecutionArtifactError, match="not active"):
            validate_shadow_books(doc)

    def test_a_cost_model_outside_the_grade_registry_is_refused(self) -> None:
        book = _book()
        book["cost_model"] = {**book["cost_model"], "name": "trader_private_flat"}
        with pytest.raises(ExecutionArtifactError, match="COST_MODELS"):
            validate_shadow_books(_shadow_doc(books=[book]))

    def test_a_cost_model_recorded_under_the_wrong_kind_is_refused(self) -> None:
        book = _book()
        book["cost_model"] = {**book["cost_model"], "kind": "sqrt_impact"}
        with pytest.raises(ExecutionArtifactError, match="kind"):
            validate_shadow_books(_shadow_doc(books=[book]))

    def test_a_cost_model_with_other_params_is_refused(self) -> None:
        book = _book()
        book["cost_model"] = {**book["cost_model"], "params": {"half_spread_bps": 1.0}}
        with pytest.raises(ExecutionArtifactError, match="requires exactly"):
            validate_shadow_books(_shadow_doc(books=[book]))

    def test_every_registered_model_is_admissible(self) -> None:
        assert set(COST_MODELS) >= {"flat_bps_v0", FLAT.name}

    def test_net_that_is_not_gross_minus_cost_is_refused(self) -> None:
        book = _book()
        book["net_return_ratio"] = book["gross_return_ratio"]
        with pytest.raises(ExecutionArtifactError, match="gross"):
            validate_shadow_books(_shadow_doc(books=[book]))

    def test_a_session_count_the_days_do_not_support_is_refused(self) -> None:
        book = _book()
        book["sessions"] = 9
        with pytest.raises(ExecutionArtifactError, match="len\\(days_advanced\\)"):
            validate_shadow_books(_shadow_doc(books=[book]))

    def test_days_that_do_not_end_at_this_session_are_refused(self) -> None:
        with pytest.raises(ExecutionArtifactError, match="from its inception"):
            validate_shadow_books(_shadow_doc(books=[_book(days=[WEEK[0], WEEK[1]])]))

    def test_unordered_days_are_refused(self) -> None:
        with pytest.raises(ExecutionArtifactError, match="strictly increasing"):
            validate_shadow_books(_shadow_doc(books=[_book(days=[WEEK[1], WEEK[0], DAY])]))

    def test_an_advanced_book_needs_exactly_one_matching_metric(self) -> None:
        doc = _shadow_doc()
        doc["metrics"] = doc["metrics"][:1]
        with pytest.raises(ExecutionArtifactError, match="exactly one required"):
            validate_shadow_books(doc)
        doc = _shadow_doc()
        doc["metrics"][0]["value"] = 0.5
        with pytest.raises(ExecutionArtifactError, match="cumulative_net_return_ratio"):
            validate_shadow_books(doc)

    def test_a_failed_book_names_its_reason_and_carries_no_figure(self) -> None:
        doc = _shadow_doc(books=[_book(ARM), _failed(CHALLENGER)])
        assert validate_shadow_books(doc)
        bad = copy.deepcopy(doc)
        bad["books"][1]["failure_reason"] = ""
        with pytest.raises(ExecutionArtifactError, match="names no failure_reason"):
            validate_shadow_books(bad)
        bad = copy.deepcopy(doc)
        bad["books"][1]["gross_return_ratio"] = 0.01
        with pytest.raises(ExecutionArtifactError, match="carries a return figure"):
            validate_shadow_books(bad)

    def test_an_advanced_book_missing_a_figure_or_carrying_a_reason_is_refused(self) -> None:
        book = _book()
        book["failure_reason"] = "x"
        with pytest.raises(ExecutionArtifactError, match="carries a failure_reason"):
            validate_shadow_books(_shadow_doc(books=[book]))
        book = _book()
        book["cost_model"] = None
        with pytest.raises(ExecutionArtifactError, match="is missing"):
            validate_shadow_books(_shadow_doc(books=[book]))

    def test_duplicates_are_refused(self) -> None:
        with pytest.raises(ExecutionArtifactError, match="lists an arm twice"):
            validate_shadow_books(_shadow_doc(books=[_book()], active=[ARM, ARM]))
        with pytest.raises(ExecutionArtifactError, match="two books"):
            validate_shadow_books(_shadow_doc(books=[_book(), _book()], active=[ARM]))

    def test_fill_basis_is_close(self) -> None:
        doc = _shadow_doc()
        doc["fill_basis"] = "vwap"
        with pytest.raises(ExecutionArtifactError, match="fill_basis"):
            validate_shadow_books(doc)


class TestShadowBookCoverage:
    def test_absent_is_unmet(self, tmp_path) -> None:
        reading = shadow_book_coverage(LocalStore(tmp_path), DAY)
        assert not reading.met and "absent" in reading.detail

    def test_every_active_arm_advanced_is_met(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _put(store, shadow_books_key(DAY), _shadow_doc())
        reading = shadow_book_coverage(store, DAY, days_served=[WEEK[0], WEEK[1], WEEK[2], DAY])
        assert reading.met, reading.detail
        assert reading.sources == (shadow_books_key(DAY),)

    def test_a_failed_book_is_unmet_and_named(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _put(store, shadow_books_key(DAY), _shadow_doc(books=[_book(ARM), _failed(CHALLENGER)]))
        reading = shadow_book_coverage(store, DAY)
        assert not reading.met and CHALLENGER in reading.detail

    def test_a_book_behind_the_traders_own_day_count_is_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _put(store, shadow_books_key(DAY), _shadow_doc(books=[_book(days=[WEEK[0], DAY])]))
        reading = shadow_book_coverage(store, DAY, days_served=[WEEK[0], WEEK[1], DAY])
        assert not reading.met and WEEK[1] in reading.detail

    def test_an_empty_register_is_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _put(store, shadow_books_key(DAY), _shadow_doc(books=[], active=[]))
        assert not shadow_book_coverage(store, DAY).met

    def test_a_corrupt_document_raises(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(shadow_books_key(DAY), b"\xff")
        with pytest.raises(ExecutionArtifactError, match="not readable JSON"):
            shadow_book_coverage(store, DAY)
