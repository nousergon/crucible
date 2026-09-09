"""The producer of `faults/{trading_day}/{fault}.json`
(`alpha-engine-config-I10320`/`-I10322`).

`crucible.gate._clause_fault_injection_against_scheduled_path` (`crucible-
PR169`) declared the key shape and the two required fields
(`manifest_key`/`bus_key`) and found no producer. This is that producer's
own test — written and seen failing before `crucible.faults` existed (fleet
test discipline), then again after `crucible.faults.record_fault` existed
but before its refusals did.

The whole point of this module is the REFUSAL, not the write
(`alpha-engine-config-I10322`): a record that could excuse an arbitrary
failure is worse than the conflict it resolves. Every refusal path below has
its own test, per AGENTS.md's test discipline ("a detector nobody has made
fail is a detector nobody knows works").
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.faults import FaultRecordRefusedError, record_fault
from crucible.keys import fault_injection_key
from crucible.manifest import manifest_key
from crucible.runner import run_job
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)


def _seed_manifest(store: LocalStore, *, job: str, trading_day: dt.date, ok: bool) -> str:
    """A real manifest through the real runner, so its `run_id` is a real
    ULID and its shape is exactly what a producer's own writes look like —
    never a hand-built dict standing in for one."""

    def body(ctx) -> None:
        if not ok:
            raise RuntimeError("MissingSourceError: seeded failure for the producer test")

    if ok:
        run_job(job, body, store=store, trading_day=trading_day, transient_retry=False)
    else:
        with pytest.raises(RuntimeError):
            run_job(job, body, store=store, trading_day=trading_day, transient_retry=False)
    key = manifest_key(job, trading_day.isoformat())
    document = json.loads(store.get_bytes(key))
    return document["run_id"]


class TestRefusals:
    def test_refuses_an_unrecognized_fault_id(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="not_a_scripted_fault",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=run_id,
                bus_key=None,
            )

        with pytest.raises(FaultRecordRefusedError, match="not one of"):
            run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)
        assert not store.exists(fault_injection_key("not_a_scripted_fault", FRIDAY.isoformat()))

    def test_refuses_when_no_manifest_carries_the_named_run_id(self, tmp_path) -> None:
        """The whole `-I10322` design constraint: a record cannot excuse a
        run that does not exist on the store."""
        store = LocalStore(tmp_path)
        _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id="0" * 26,  # a well-shaped ULID that names nothing real
                bus_key=None,
            )

        with pytest.raises(FaultRecordRefusedError, match="no manifest under"):
            run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)
        assert not store.exists(fault_injection_key("data_source_withheld", FRIDAY.isoformat()))

    def test_refuses_a_run_id_that_succeeded(self, tmp_path) -> None:
        """A record naming a successful run is exactly the arbitrary-excuse
        shape `-I10322` forbids: refused, never filed."""
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=True)

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=run_id,
                bus_key=None,
            )

        with pytest.raises(FaultRecordRefusedError, match="not 'failed'"):
            run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)
        assert not store.exists(fault_injection_key("data_source_withheld", FRIDAY.isoformat()))

    def test_refuses_a_bus_key_the_store_does_not_hold(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=run_id,
                bus_key="alerts/2026-08-28/failure.data.weekly.json",
            )

        with pytest.raises(FaultRecordRefusedError, match="store does not hold"):
            run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)
        assert not store.exists(fault_injection_key("data_source_withheld", FRIDAY.isoformat()))

    def test_refuses_a_bus_key_of_the_wrong_shape(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=run_id,
                bus_key="not/a/bus/key/at/all.json",
            )

        with pytest.raises(FaultRecordRefusedError, match="not shaped like"):
            run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)

    def test_refuses_before_writing_anything(self, tmp_path) -> None:
        """The refusal happens before ANY store write — including this
        job's own manifest content beyond `status: failed`. A producer that
        could write partial evidence and then refuse would leave a trace an
        operator could mistake for a filed record."""
        store = LocalStore(tmp_path)
        before = sorted(store.list_keys())

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id="0" * 26,
                bus_key=None,
            )

        with pytest.raises(FaultRecordRefusedError):
            run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)
        # The runner's own manifest for THIS failed fault.record run is the
        # only new key — no faults/ record and nothing else landed.
        after = sorted(store.list_keys())
        new_keys = sorted(set(after) - set(before))
        assert new_keys == [manifest_key("fault.record", FRIDAY.isoformat())]


class TestASuccessfulRecord:
    def test_writes_the_record_naming_the_excused_run(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=run_id,
                bus_key=None,
            )

        ctx = run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)
        assert ctx.outputs  # this job's own manifest carries the write as lineage

        key = fault_injection_key("data_source_withheld", FRIDAY.isoformat())
        document = json.loads(store.get_bytes(key))
        assert document["fault_id"] == "data_source_withheld"
        assert document["trading_day"] == FRIDAY.isoformat()
        assert document["run_id"] == run_id
        assert document["manifest_key"] == manifest_key("data.weekly", FRIDAY.isoformat())
        assert document["bus_key"] is None
        assert document["schema_version"] == "fault_record.v1"

        this_run_manifest = json.loads(
            store.get_bytes(manifest_key("fault.record", FRIDAY.isoformat()))
        )
        assert this_run_manifest["status"] == "ok"
        assert [o["key"] for o in this_run_manifest["outputs"]] == [key]

    def test_re_inducing_overwrites_the_record_in_place(self, tmp_path) -> None:
        """`crucible.keys.fault_injection_key`'s own docstring: the record
        describes the day's state, and re-inducing the same fault on the
        same day and re-filing overwrites correctly."""
        store = LocalStore(tmp_path)
        first_run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)

        def job1(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=first_run_id,
                bus_key=None,
            )

        run_job("fault.record", job1, store=store, trading_day=FRIDAY, transient_retry=False)

        # Stand down, then re-induce: a fresh failed manifest, a fresh run_id.
        second_run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        assert second_run_id != first_run_id

        def job2(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=second_run_id,
                bus_key=None,
            )

        run_job("fault.record", job2, store=store, trading_day=FRIDAY, transient_retry=False)

        key = fault_injection_key("data_source_withheld", FRIDAY.isoformat())
        document = json.loads(store.get_bytes(key))
        assert document["run_id"] == second_run_id

    def test_bus_key_is_recorded_when_given_and_real(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        bus_key = "alerts/2026-08-28/failure.data.weekly.json"
        store.put_bytes(bus_key, b'{"condition": "failure"}')

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="data_source_withheld",
                target_job="data.weekly",
                trading_day=FRIDAY.isoformat(),
                run_id=run_id,
                bus_key=bus_key,
            )

        run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)
        key = fault_injection_key("data_source_withheld", FRIDAY.isoformat())
        document = json.loads(store.get_bytes(key))
        assert document["bus_key"] == bus_key
