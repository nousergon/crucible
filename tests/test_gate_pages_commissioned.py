"""Plan §9.3 "Commissioning": each page condition fired, delivered, stood down.

The row exists in the plan as a phase-2 exit-gate row and had no clause behind
it: `crucible.gate._phase2` graded five clauses and none of them asked whether
the two page conditions had ever been proven end to end. Both had, on
2026-09-04/05, on genuine defects — and nothing could read that.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

import crucible.gate as gate_module
from crucible.alerts import PAGE_CONDITIONS
from crucible.gate import evaluate
from crucible.keys import manifest_key, runs_prefix
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 9, 4)
DAY = FRIDAY.isoformat()


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def _put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


def _swept(store: LocalStore) -> None:
    _put(store, manifest_key("alerts.sweep", DAY), {"status": "ok"})


def _row(store: LocalStore, condition: str, jobs: list[str], *, sent: bool = True) -> str:
    key = f"alerts/{DAY}/{condition}.{jobs[0]}.json"
    _put(
        store,
        key,
        {
            "schema_version": "alert_bus.v2",
            "condition": condition,
            "sent": sent,
            "trading_day": DAY,
            "members": [{"job": j, "run_id": None, "reason": "x"} for j in jobs],
        },
    )
    return key


def _manifest(store: LocalStore, job: str, status: str) -> None:
    _put(
        store,
        manifest_key(job, DAY),
        {"status": status, "reason": "" if status == "ok" else "boom"},
    )


class TestUnmeasurableBeforeAnythingSwept:
    def test_an_empty_store_is_unmeasurable(self, store: LocalStore) -> None:
        clause = gate_module._clause_pages_commissioned(store)
        assert clause.unmeasurable and not clause.met
        assert runs_prefix("alerts.sweep") in clause.detail

    def test_a_swept_but_empty_bus_is_unmet_not_unmeasurable(self, store: LocalStore) -> None:
        """The sweep ran and found nothing: that is a measured "never fired"
        for both conditions, and the clause says so per condition."""
        _swept(store)
        clause = gate_module._clause_pages_commissioned(store)
        assert not clause.met and not clause.unmeasurable
        for condition in PAGE_CONDITIONS:
            assert f"{condition}: has never fired" in clause.detail


class TestBothConditionsMustBeDeliveredAndStoodDown:
    def test_met_when_each_condition_was_delivered_and_every_member_is_now_ok(
        self, store: LocalStore
    ) -> None:
        _swept(store)
        failure = _row(store, "failure", ["weekly"])
        absence = _row(store, "absence", ["drift", "report"])
        for job in ("weekly", "drift", "report"):
            _manifest(store, job, "ok")
        clause = gate_module._clause_pages_commissioned(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert set(clause.evidence) == {failure, absence}
        assert "failure: commissioned by" in clause.detail
        assert "absence: commissioned by" in clause.detail

    def test_one_condition_alone_is_not_enough(self, store: LocalStore) -> None:
        _swept(store)
        _row(store, "failure", ["weekly"])
        _manifest(store, "weekly", "ok")
        clause = gate_module._clause_pages_commissioned(store)
        assert not clause.met
        assert "failure: commissioned by" in clause.detail
        assert "absence: has never fired" in clause.detail

    def test_a_delivered_page_whose_job_is_still_failed_has_not_stood_down(
        self, store: LocalStore
    ) -> None:
        """This is the live 2026-09-06 state for `failure.data.daily@2026-09-03`:
        paged, delivered, and the manifest still `failed` until the rerun."""
        _swept(store)
        _row(store, "failure", ["data.daily"])
        _manifest(store, "data.daily", "failed")
        _row(store, "absence", ["drift"])
        _manifest(store, "drift", "ok")
        clause = gate_module._clause_pages_commissioned(store)
        assert not clause.met
        assert "failure: delivered but not stood down" in clause.detail
        assert "data.daily not yet ok" in clause.detail

    def test_an_absence_stands_down_only_when_every_member_has_a_manifest(
        self, store: LocalStore
    ) -> None:
        _swept(store)
        _row(store, "absence", ["drift", "report", "console"])
        _manifest(store, "drift", "ok")
        _manifest(store, "report", "ok")
        clause = gate_module._clause_pages_commissioned(store)
        assert "console not yet ok" in clause.detail
        _manifest(store, "console", "ok")
        clause = gate_module._clause_pages_commissioned(store)
        assert "absence: commissioned by" in clause.detail

    def test_a_page_that_never_left_proves_the_condition_not_the_path(
        self, store: LocalStore
    ) -> None:
        """`sent: false` is a transport that did not deliver. The condition
        fired; the path is unproven; the row does not commission anything."""
        _swept(store)
        _row(store, "failure", ["weekly"], sent=False)
        _manifest(store, "weekly", "ok")
        clause = gate_module._clause_pages_commissioned(store)
        assert not clause.met
        assert "failure: fired but never delivered" in clause.detail

    def test_a_discriminated_ok_manifest_counts_as_stood_down(self, store: LocalStore) -> None:
        """`experiment.grade` files per slot; the stand-down read lists the
        prefix rather than guessing a discriminator."""
        _swept(store)
        _row(store, "absence", ["experiment.grade"])
        _put(store, manifest_key("experiment.grade", DAY, discriminator="u"), {"status": "ok"})
        _row(store, "failure", ["weekly"])
        _manifest(store, "weekly", "ok")
        assert gate_module._clause_pages_commissioned(store).met


class TestMalformedRowsAreReportedNeverCounted:
    def test_a_row_naming_an_unknown_condition_is_named_in_the_detail(
        self, store: LocalStore
    ) -> None:
        _swept(store)
        _put(store, f"alerts/{DAY}/weird.json", {"condition": "warning", "sent": True})
        clause = gate_module._clause_pages_commissioned(store)
        assert not clause.met
        assert "malformed bus row(s) ignored" in clause.detail
        assert "warning" in clause.detail

    def test_a_row_without_members_is_malformed(self, store: LocalStore) -> None:
        _swept(store)
        _put(store, f"alerts/{DAY}/failure.x.json", {"condition": "failure", "sent": True})
        clause = gate_module._clause_pages_commissioned(store)
        assert "malformed bus row(s) ignored" in clause.detail

    def test_a_non_bus_key_under_the_prefix_is_not_a_row(self, store: LocalStore) -> None:
        _swept(store)
        _put(store, f"alerts/{DAY}/nested/one.json", {"condition": "failure", "sent": True})
        clause = gate_module._clause_pages_commissioned(store)
        assert "malformed" not in clause.detail
        assert "failure: has never fired" in clause.detail


class TestItIsAPhaseTwoClause:
    def test_phase2_reads_it(self, store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raising() -> object:
            raise RuntimeError("no credentials")

        monkeypatch.setattr(gate_module, "_ce_client", _raising)
        monkeypatch.setattr(gate_module, "_s3_client", _raising)
        result = evaluate(store, gate="phase2", trading_day=dt.date(2026, 8, 28))
        assert "pages_commissioned" in [c.name for c in result.clauses]
