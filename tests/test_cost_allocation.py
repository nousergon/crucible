"""The phase-4 cost scope must reconcile the entire bill before grading."""

import datetime as dt
import io
import json
from pathlib import Path

import pytest

from crucible.cost import (
    CollectorSpendClient,
    CostExplorerCache,
    CostUnreadableError,
    month_to_date_usd,
)

SCOPE = "crucible-and-trader"


def rollup():
    days = [
        {
            "date": (dt.date(2026, 8, 1) + dt.timedelta(days=day)).isoformat(),
            "estimated": False,
            "by_system_usd": {"crucible-v2": 1.0, "(untagged)": 9.0},
        }
        for day in range(59)
    ]
    allocation = {
        "schema_version": "cost_allocation.v1",
        "inventory_complete": True,
        "inventory_evidence": "inventory/verified.json",
        "covered_components": ["harness", "trader", "shared"],
        "declared_scopes": [SCOPE, "other-system"],
        "days": [
            {
                "date": d["date"],
                "account_usd": 10.0,
                "members": [
                    {
                        "id": "harness",
                        "component": "harness",
                        "scope": SCOPE,
                        "kind": "direct",
                        "usd": 1.0,
                        "evidence": "ledger/harness",
                    },
                    {
                        "id": "trader",
                        "component": "trader",
                        "scope": SCOPE,
                        "kind": "direct",
                        "usd": 2.0,
                        "evidence": "ledger/trader",
                    },
                    {
                        "id": "trail",
                        "component": "shared",
                        "scope": SCOPE,
                        "kind": "shared",
                        "usd": 0.5,
                        "evidence": "allocation/trail",
                    },
                    {
                        "id": "other",
                        "component": "other",
                        "scope": "other-system",
                        "kind": "direct",
                        "usd": 6.5,
                        "evidence": "ledger/other",
                    },
                ],
            }
            for d in days
        ],
    }
    return {
        "as_of": "2026-09-29T12:00:00Z",
        "providers": [
            {
                "key": "aws",
                "detail": {
                    "daily_by_system": {"tag_key": "system", "complete": True, "days": days},
                    "cost_allocation": allocation,
                },
            }
        ],
    }


def client(doc):
    class S3:
        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(json.dumps(doc).encode())}

    return CollectorSpendClient(
        "s3://bucket/expenses/latest.json",
        s3=S3(),
        now=lambda: dt.datetime(2026, 9, 29, 13, tzinfo=dt.UTC),
    )


def read(c):
    return month_to_date_usd(c, today=dt.date(2026, 9, 29), tagged=False, cache=CostExplorerCache())


def test_component_includes_trader_and_shared_but_excludes_other_systems():
    c = client(rollup())
    assert read(c).amount_usd == 280.0
    scoped = read(c.for_scope(SCOPE))
    assert scoped.amount_usd == 98.0
    assert scoped.scope == "allocated " + SCOPE


def test_scope_and_account_never_share_cached_results():
    c = client(rollup())
    cache = CostExplorerCache()
    args = dict(today=dt.date(2026, 9, 29), tagged=False, cache=cache)
    assert month_to_date_usd(c, **args).amount_usd == 280.0
    assert month_to_date_usd(c.for_scope(SCOPE), **args).amount_usd == 98.0


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "allocation"),
        ("inventory", "value_error"),
        ("components", "value_error"),
        ("unallocated", "unallocated"),
        ("reconciliation", "value_error"),
        ("account", "account"),
        ("date", "date"),
        ("duplicate", "value_error"),
        ("evidence", "evidence"),
        ("nan", "finite"),
    ],
)
def test_incomplete_or_invalid_allocation_is_never_a_cheap_day(mutation, match):
    doc = rollup()
    detail = doc["providers"][0]["detail"]
    a = detail["cost_allocation"]
    day = a["days"][0]
    if mutation == "missing":
        del detail["cost_allocation"]
    elif mutation == "inventory":
        a["inventory_complete"] = False
    elif mutation == "components":
        a["covered_components"].remove("trader")
    elif mutation == "unallocated":
        day["members"][-1]["kind"] = "unallocated"
        day["members"][-1]["component"] = "unallocated"
    elif mutation == "reconciliation":
        day["members"][-1]["usd"] = 6.0
    elif mutation == "account":
        day["account_usd"] = 11.0
        day["members"][-1]["usd"] = 7.5
    elif mutation == "date":
        a["days"].pop(0)
    elif mutation == "duplicate":
        day["members"][-1]["id"] = "harness"
    elif mutation == "evidence":
        day["members"][-1]["evidence"] = ""
    elif mutation == "nan":
        day["members"][-1]["usd"] = float("nan")
    with pytest.raises(CostUnreadableError, match=match):
        read(client(doc).for_scope(SCOPE))
    # Bad/missing allocation must not erase the independent account reading.
    assert read(client(doc)).amount_usd == 280.0


def test_schema_is_generated_from_the_runtime_contract():
    from crucible.cost_allocation import CostAllocation

    schema = Path(__file__).parents[1] / "crucible" / "schemas" / "cost_allocation.v1.json"
    assert json.loads(schema.read_text()) == CostAllocation.model_json_schema()


def test_phase4_uses_allocated_client_and_keeps_account_slo(monkeypatch):
    from crucible import gate

    monkeypatch.setattr(gate, "_ce_client", lambda: client(rollup()))
    window = [dt.date(2026, 9, 28)]
    scoped = gate._clause_aws_cost_within_ceiling(
        window,
        name="aws_total_within_ceiling",
        ceiling_usd=110.0,
        tagged=False,
        cost_scope=SCOPE,
    )
    assert scoped.met
    assert SCOPE in scoped.requirement
    assert "the whole account" not in scoped.requirement
    assert "aws_account_within_ceiling" in gate.STANDING_SLOS


@pytest.mark.parametrize(
    "mutation", ["typo", "missing-trader", "misrouted-trader", "missing-shared"]
)
def test_component_membership_cannot_silently_drop_billed_cost(mutation):
    doc = rollup()
    members = doc["providers"][0]["detail"]["cost_allocation"]["days"][0]["members"]
    if mutation == "typo":
        members[1]["scope"] = "crucible-and-trade"
    elif mutation == "misrouted-trader":
        members[1]["scope"] = "other-system"
    elif mutation == "missing-trader":
        members[1]["component"] = "other"
        members[1]["scope"] = "other-system"
    else:
        members[2]["component"] = "other"
        members[2]["scope"] = "other-system"
    with pytest.raises(CostUnreadableError):
        read(client(doc).for_scope(SCOPE))


def test_invalid_allocation_diagnostic_never_contains_private_input():
    doc = rollup()
    sentinel = "arn:aws:s3:::private-evidence-bucket/ownership.json"
    member = doc["providers"][0]["detail"]["cost_allocation"]["days"][0]["members"][0]
    member["scope"] = {"private": sentinel}
    member["id"] = sentinel
    member["evidence"] = sentinel
    with pytest.raises(CostUnreadableError) as caught:
        read(client(doc).for_scope(SCOPE))
    assert sentinel not in str(caught.value)
    assert "scope" in str(caught.value)


def test_missing_required_closed_month_cannot_grade_green(monkeypatch):
    from crucible import gate

    doc = rollup()
    detail = doc["providers"][0]["detail"]
    for series in [detail["daily_by_system"]["days"], detail["cost_allocation"]["days"]]:
        del series[30:]
        for index, day in enumerate(series):
            day["date"] = (dt.date(2026, 9, 3) + dt.timedelta(days=index)).isoformat()
    doc["as_of"] = "2026-10-03T12:00:00Z"
    c = client(doc)
    c._now = lambda: dt.datetime(2026, 10, 3, 13, tzinfo=dt.UTC)
    monkeypatch.setattr(gate, "_ce_client", lambda: c)
    clause = gate._clause_aws_cost_within_ceiling(
        [dt.date(2026, 10, 3)],
        name="aws_total_within_ceiling",
        ceiling_usd=110.0,
        tagged=False,
        cost_scope=SCOPE,
    )
    assert not clause.met
    assert clause.unmeasurable
    assert "closed month" in clause.detail


@pytest.mark.parametrize("missing", [False, True])
def test_account_standing_row_preserves_known_overage_and_unknown_reading(
    store, monkeypatch, missing
):
    from crucible import gate
    from crucible.board import HumanTouchReading, _standing_rows

    doc = rollup()
    if missing:
        doc["providers"] = []
    monkeypatch.setattr(gate, "_ce_client", lambda: client(doc))
    rows = _standing_rows(
        store, "2026-09-28", HumanTouchReading("2026-09", 0, (), "measured", True)
    )
    account = next(row for row in rows if row.id == "standing:aws_account_within_ceiling")
    assert account.state == ("UNMEASURABLE" if missing else "UNMET")
    assert account.red


def test_private_unknown_field_name_is_not_a_diagnostic_path():
    doc = rollup()
    sentinel = "arn:aws:s3:::private-evidence-bucket/ownership.json"
    doc["providers"][0]["detail"]["cost_allocation"][sentinel] = True
    with pytest.raises(CostUnreadableError) as caught:
        read(client(doc).for_scope(SCOPE))
    assert sentinel not in str(caught.value)
    assert "unknown_field" in str(caught.value)


def test_malformed_account_numeric_input_is_not_a_diagnostic():
    doc = rollup()
    sentinel = "arn:aws:s3:::private-evidence-bucket/ledger.json"
    doc["providers"][0]["detail"]["daily_by_system"]["days"][0]["by_system_usd"]["(untagged)"] = (
        sentinel
    )
    with pytest.raises(CostUnreadableError) as caught:
        read(client(doc).for_scope(SCOPE))
    assert sentinel not in str(caught.value)
