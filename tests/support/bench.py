"""A valid `slot_bench.v1` record, for tests that need a filed bench.

Built through `crucible.bench.bench_document` — the writer's own validator —
so a fixture can never seed a bench the real writer would refuse.
"""

from __future__ import annotations

from typing import Any

from crucible.bench import bench_document, bench_record_bytes
from crucible.keys import slot_bench_key
from crucible.store import Store

#: The ruling's tracker, as a plain `int` (`tests/test_no_stale_tracker_literals.py`).
TRACKER_ISSUE = 12024
RULING_REF = "https://github.com/nousergon/alpha-engine-config/issues/12024#issuecomment-1"
EVIDENCE = (
    "https://github.com/nousergon/alpha-engine-config/pull/12406",
    "runs/promote/2026-10-16/m/run.json",
)


def bench_record(
    champion: str, *, effective: str, filed_on: str | None = None, **overrides: Any
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "slot": "m",
        "champion_at_bench": champion,
        "ruled_by": "brian",
        "ruled_at": "2026-10-18T14:44Z",
        "ruling_ref": RULING_REF,
        "tracker": f"alpha-engine-config-I{TRACKER_ISSUE}",
        "reason": "no M arm passes its serving preconditions; every near-miss had negative OOS IC",
        "evidence": list(EVIDENCE),
        "effective_trading_day": effective,
        "reexam_date": "2026-11-20",
        "filed_by": "tester",
        "filed_at_utc": "2026-10-19T12:00:00Z",
        "filed_on": filed_on or effective,
    }
    fields.update(overrides)
    return bench_document(**fields)


def file_bench(store: Store, champion: str, *, effective: str, **overrides: Any) -> dict:
    document = bench_record(champion, effective=effective, **overrides)
    store.put_bytes(slot_bench_key("m"), bench_record_bytes(document))
    return document
