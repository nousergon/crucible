# Cost allocation contract
Approved boundary: Crucible harness + trader + allocated shared services, unchanged $70/month ceiling (alpha-engine-config-I12022, Brian ruling 2026-10-05). The whole AWS account remains a standing SLO; v1 retirement remains independent.

The expense collector's AWS provider detail may carry `cost_allocation`, conforming to `crucible/schemas/cost_allocation.v1.json`. The runtime model is `crucible.cost_allocation.CostAllocation`. Account and harness-only readers continue to use `daily_by_system`; phase 4 uses the reconciled allocation scope `crucible-and-trader`.

## Producer requirements
- Inventory evidence covers dedicated resources and the trader's shared host, storage, logs, CloudTrail, and billing/commitment/shared-service charges. `covered_components` names harness, trader, shared exactly once, and `inventory_complete` is true only after this coverage is verified.
- `declared_scopes` enumerates every accountable scope and includes `crucible-and-trader`; unknown scopes are rejected. Every member carries a `component` enum (harness/trader/shared/other/unallocated). Each day has canonical harness, trader and shared rows, including explicit evidenced zero rows when truly zero. Dedicated harness/trader members must use the canonical scope and direct kind; shared members use shared kind.
- Each complete UTC day has named allocation members: accountable scope, direct/shared/unallocated kind, USD amount, and a durable source or allocation evidence reference. IDs are unique per day. Shared charges may split into multiple accountable members; their assigned amounts, including credits, must conserve the provider ledger.
- Member sum equals that day's account amount within 0.000001 USD serialization precision. Account amount must independently equal the sum of the collector's per-system ledger. Day sets must match exactly; no missing or duplicate day is zero.
- Positive or negative unresolved amounts are `unallocated`, not silently assigned outside Crucible. Nonzero unallocated members refuse a scoped grade. Never claim inventory coverage solely because tagged harness spend exists.
- Preserve historical billing evidence and allocation-method provenance. A historical tagging gap cannot be retroactively invented as tagged spend.
- Unknown fields, missing evidence, NaN/infinity, duplicate members/days, incomplete inventory, incomplete components, and failed reconciliation are refused. The gate renders UNMEASURABLE; the account amount remains readable independently.

## Migration
Consumer-first. Deploy this reader/schema before the collector emits the optional field. The current export lacks complete trader/shared allocation, so the scoped gate will be UNMEASURABLE until the producer closes that gap. A fake allocation or harness-only filter is not a workaround. No additional Cost Explorer query is introduced by this contract.

Producer/consumer example and negative mutations live in `tests/test_cost_allocation.py`. The collector must pin this schema at its own boundary and run its actual output through the runtime model and ledger reconciliation before publication; JSON Schema alone does not enforce conservation or scope membership.

Validation diagnostics expose bounded field paths and error types, never private member IDs or input values. On the first three days of a month, required prior closed-month coverage must be readable and finalized before MET; known current overages remain UNMET.
