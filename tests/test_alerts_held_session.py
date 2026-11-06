"""`alpha-engine-config-I13057`: a HELD `trader.session` pages.

The trader records a held session (no champion, feed or construction input
for the day) as `status: ok` with `trader_session_served = 0`, by design: a
hold is not a failed run (`crucible_trader.daily_session`). Neither page
condition saw it: the manifest exists, so there is no ABSENCE, and it is
`ok`, so there is no FAILURE. The 2026-10-06 hold ("no champion pointer for
slot 's'") was found by a person reading manifests (nous-ergon-ops#1998).

`crucible.alerts.evaluate_held_session` reuses the FAILURE condition (§4.6
admits exactly two): a trading day the trader did not serve is a day the job
failed to do the one thing it exists to do.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Iterator

import pytest

from crucible.alerts import (
    PAGE_CONDITIONS,
    SESSION_SERVED_METRIC,
    TRADER_SESSION_JOB,
    StoreAccessError,
    evaluate_held_session,
    sweep,
)
from crucible.components import load_registry
from crucible.manifest import manifest_key
from crucible.runmode import RUN_MODE_REPLAY
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)

#: The hold reason the trader wrote on 2026-10-06, verbatim (measured
#: read-only from `runs/trader.session/2026-10-06/run.json`).
HOLD_REASON = (
    "held: no champion pointer for slot 's'. Nothing has been promoted yet, so there is "
    "nothing to serve -- hold the book and page on the absence deadline rather than "
    "treating an empty slot as a decision to trade nothing."
)


def _served_metric(value: float, reason: str) -> dict:
    return {
        "name": SESSION_SERVED_METRIC,
        "module": "crucible_trader.daily_session",
        "metric_type": "operational",
        "value": value,
        "unit": "flag",
        "n_floor": 1,
        "status": "OK",
        "status_reason": reason,
        "source_path": f"predictions/{FRIDAY.isoformat()}.json",
        "last_updated_utc": "2026-08-31T14:46:10Z",
    }


def _write_session(
    store: LocalStore,
    *,
    status: str = "ok",
    metrics: list[dict] | None = None,
    reason: str = "",
    run_id: str = "01JG0000000000000000000HLD",
    discriminator: str | None = None,
    run_mode: str | None = None,
) -> None:
    payload = {
        "schema_version": "run_manifest.v2",
        "run_id": run_id,
        "job": TRADER_SESSION_JOB,
        "trading_day": FRIDAY.isoformat(),
        "calendar_date": "2026-08-31",
        "status": status,
        "reason": reason,
        "cost_usd": 0.0,
        "metrics": metrics if metrics is not None else [],
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator
    if run_mode is not None:
        payload["run_mode"] = run_mode
    store.put_bytes(
        manifest_key(TRADER_SESSION_JOB, FRIDAY.isoformat(), discriminator=discriminator),
        json.dumps(payload).encode(),
    )


class TestEvaluateHeldSession:
    def test_a_held_session_pages_with_its_hold_reason(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, metrics=[_served_metric(0.0, HOLD_REASON)])
        pages = evaluate_held_session(store, now=SATURDAY_NIGHT)
        assert len(pages) == 1
        (page,) = pages
        assert page.condition == "failure"
        assert page.condition in PAGE_CONDITIONS
        assert page.job == TRADER_SESSION_JOB
        assert page.trading_day == FRIDAY
        assert page.run_id == "01JG0000000000000000000HLD"
        assert "no champion pointer for slot 's'" in page.reason
        assert "Auto-fix:" in page.reason
        assert "Resolution:" in page.reason

    def test_a_served_session_pages_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, metrics=[_served_metric(1.0, "shadow session on m:x")])
        assert evaluate_held_session(store, now=SATURDAY_NIGHT) == []

    def test_a_failed_session_is_left_to_the_failure_condition(self, tmp_path) -> None:
        """A refusal is `status: failed` and already pages through
        `evaluate_failure`; paging it here too would be two members for one
        manifest."""
        store = LocalStore(tmp_path)
        _write_session(store, status="failed", reason="ContractRefused: boom")
        assert evaluate_held_session(store, now=SATURDAY_NIGHT) == []

    def test_no_manifest_is_left_to_the_absence_condition(self, tmp_path) -> None:
        assert evaluate_held_session(LocalStore(tmp_path), now=SATURDAY_NIGHT) == []

    def test_a_later_served_run_clears_an_earlier_hold(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(
            store,
            metrics=[_served_metric(0.0, HOLD_REASON)],
            discriminator="a",
            run_id="01JG0000000000000000000HLA",
        )
        _write_session(
            store,
            metrics=[_served_metric(1.0, "shadow session on m:x")],
            discriminator="b",
            run_id="01JG0000000000000000000HLB",
        )
        assert evaluate_held_session(store, now=SATURDAY_NIGHT) == []

    def test_a_served_replay_does_not_clear_a_held_production_session(self, tmp_path) -> None:
        """Codex finding 85448a6184b9fea4: synthetic and production sessions
        are judged apart, so a served replay neither suppresses the held
        production day nor lends that page its synthetic marker."""
        store = LocalStore(tmp_path)
        _write_session(
            store,
            metrics=[_served_metric(0.0, HOLD_REASON)],
            discriminator="a",
            run_id="01JG0000000000000000000HLA",
        )
        _write_session(
            store,
            metrics=[_served_metric(1.0, "replay session on m:x")],
            discriminator="b",
            run_id="01JG0000000000000000000HLB",
            run_mode=RUN_MODE_REPLAY,
        )
        (page,) = evaluate_held_session(store, now=SATURDAY_NIGHT)
        assert page.run_id == "01JG0000000000000000000HLA"
        assert page.synthetic is None

    def test_an_ok_session_that_does_not_say_whether_it_served_pages(self, tmp_path) -> None:
        """Fail loud: every `ok` path in the trader records the metric, so its
        absence is a contract break that would otherwise blind this rule."""
        store = LocalStore(tmp_path)
        _write_session(store, metrics=[])
        (page,) = evaluate_held_session(store, now=SATURDAY_NIGHT)
        assert SESSION_SERVED_METRIC in page.reason

    def test_a_retired_trader_session_row_pages_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, metrics=[_served_metric(0.0, HOLD_REASON)])
        registry = load_registry()
        retired = {
            **registry,
            TRADER_SESSION_JOB: dataclasses.replace(
                registry[TRADER_SESSION_JOB], lifecycle="RETIRED"
            ),
        }
        assert evaluate_held_session(store, now=SATURDAY_NIGHT, registry=retired) == []

    def test_an_unlistable_prefix_is_an_access_fault_not_silence(self, tmp_path) -> None:
        class _Denied(LocalStore):
            def list_keys(self, prefix: str = "") -> Iterator[str]:
                if prefix.startswith(f"runs/{TRADER_SESSION_JOB}/"):
                    raise PermissionError("AccessDenied")
                return super().list_keys(prefix)

        with pytest.raises(StoreAccessError):
            evaluate_held_session(_Denied(tmp_path), now=SATURDAY_NIGHT)


class TestTheSweepPagesAHeldSession:
    def test_the_sweep_sends_one_page_carrying_the_hold_reason(self, tmp_path, transport) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, metrics=[_served_metric(0.0, HOLD_REASON)])
        result = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        held = [r for r in rows if any(m["job"] == TRADER_SESSION_JOB for m in r["members"])]
        assert len(held) == 1
        assert held[0]["condition"] == "failure"
        assert held[0]["sent"] is True
        assert "no champion pointer for slot 's'" in held[0]["members"][0]["reason"]

    def test_a_served_session_sends_no_page(self, tmp_path, transport) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, metrics=[_served_metric(1.0, "shadow session on m:x")])
        result = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        assert not [r for r in rows if any(m["job"] == TRADER_SESSION_JOB for m in r["members"])]
