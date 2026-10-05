"""Versioned allocation boundary between the expense collector and cost gates."""

from __future__ import annotations

import datetime as dt
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class AllocationMember(BaseModel):
    """One accountable part of the provider ledger, including shared services."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    component: Literal["harness", "trader", "shared", "other", "unallocated"]
    kind: Literal["direct", "shared", "unallocated"]
    usd: float
    evidence: str = Field(min_length=1)


class AllocationDay(BaseModel):
    """Member amounts must reconcile to the account, not just to the scope."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date: dt.date
    account_usd: float
    members: list[AllocationMember] = Field(min_length=1)

    @model_validator(mode="after")
    def reconciled(self) -> AllocationDay:
        ids = [m.id for m in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate allocation member")
        if not math.isclose(
            math.fsum(m.usd for m in self.members), self.account_usd, rel_tol=0, abs_tol=0.000001
        ):
            raise ValueError("allocation members do not reconcile to account")
        return self


class CostAllocation(BaseModel):
    """Coverage is explicit; an unknown share never makes a component cheaper."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["cost_allocation.v1"]
    inventory_complete: StrictBool
    inventory_evidence: str = Field(min_length=1)
    covered_components: list[Literal["harness", "trader", "shared"]]
    declared_scopes: list[str] = Field(min_length=1)
    days: list[AllocationDay] = Field(min_length=1)

    @model_validator(mode="after")
    def complete(self) -> CostAllocation:
        if not self.inventory_complete:
            raise ValueError("resource inventory is incomplete")
        if sorted(self.covered_components) != ["harness", "shared", "trader"]:
            raise ValueError("covered components must name harness, trader and shared exactly once")
        if len(set(self.declared_scopes)) != len(self.declared_scopes):
            raise ValueError("duplicate declared scope")
        if "crucible-and-trader" not in self.declared_scopes:
            raise ValueError("canonical scope is not declared")
        for day in self.days:
            covered = set()
            for member in day.members:
                if member.scope not in self.declared_scopes:
                    raise ValueError("member scope is not declared")
                if member.component in {"harness", "trader"}:
                    if member.scope != "crucible-and-trader" or member.kind != "direct":
                        raise ValueError("dedicated component must use canonical direct scope")
                if member.component == "shared" and member.kind != "shared":
                    raise ValueError("shared component must use shared kind")
                if (member.component == "unallocated") != (member.kind == "unallocated"):
                    raise ValueError("unallocated component and kind must agree")
                if member.scope == "crucible-and-trader":
                    if member.component == "other":
                        raise ValueError("other component cannot use canonical scope")
                    covered.add(member.component)
            if not {"harness", "trader", "shared"}.issubset(covered):
                raise ValueError("daily canonical component coverage is incomplete")
        dates = [day.date for day in self.days]
        if len(set(dates)) != len(dates):
            raise ValueError("duplicate allocation date")
        return self

    def amounts(self, account: dict[dt.date, dict], *, scope: str) -> dict[dt.date, float]:
        """Cross-check independent account series before returning a scoped total."""
        if {day.date for day in self.days} != set(account):
            raise ValueError("allocation date coverage differs from the account series")
        amounts = {}
        for day in self.days:
            try:
                values = [float(v) for v in account[day.date]["by_system_usd"].values()]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("nonfinite account amount")
                account_usd = math.fsum(values)
            except (ValueError, TypeError, OverflowError) as exc:
                # Malformed account amounts can be private strings; never echo the input.
                raise ValueError("invalid account ledger numeric value") from exc
            if not math.isclose(day.account_usd, account_usd, rel_tol=0, abs_tol=0.000001):
                raise ValueError(f"allocation account total disagrees on {day.date}")
            unknown = any(m.kind == "unallocated" and m.usd != 0 for m in day.members)
            if unknown:
                raise ValueError("unallocated billed members")
            amounts[day.date] = math.fsum(m.usd for m in day.members if m.scope == scope)
        return amounts
