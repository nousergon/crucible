"""The producer of `faults/{trading_day}/{fault}.json`
(`alpha-engine-config-I10320`/`-I10322`/`-I10327`).

`crucible.gate._clause_fault_injection_against_scheduled_path` (`crucible-
PR169`) declared the key shape and the two field names
(`manifest_key`/`bus_key`) and found no producer. This is that producer's
own test — written and seen failing before `crucible.faults` existed (fleet
test discipline), then again after `crucible.faults.record_fault` existed
but before its refusals did, and once more per outcome kind before each
kind's refusals existed.

The whole point of this module is the REFUSAL, not the write
(`alpha-engine-config-I10322`): a record that could excuse an arbitrary
failure is worse than the conflict it resolves. Every refusal path below has
its own test, per AGENTS.md's test discipline ("a detector nobody has made
fail is a detector nobody knows works"), and every one of them is asserted in
BOTH directions — the shape that is refused and the shape that is accepted.

The three kinds are not variations on one shape. `induced` requires a FAILED
manifest and a bus row; `absorbed` requires an `ok` manifest carrying a
declared-transient-class retry and FORBIDS a bus row; `unreachable` requires
no run at all and a machine-executed probe per closed path. No kind is
reachable by dropping a requirement from another, which is what the
cross-refusal tests below pin.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.faults import FaultRecordRefusedError, record_fault
from crucible.keys import fault_injection_key
from crucible.manifest import manifest_key
from crucible.release import release_json_key, wheel_key
from crucible.runner import SpotInterruptionError, run_job
from crucible.store import LocalStore, S3Store

FRIDAY = dt.date(2026, 8, 28)
BUS_KEY = "alerts/2026-08-28/failure.data.weekly.json"

#: A well-shaped ULID that names no run on any store.
NO_SUCH_RUN_ID = "0" * 26


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


def _seed_absorbed_manifest(store: LocalStore, *, job: str, trading_day: dt.date) -> str:
    """A manifest that reads `ok` AND records a `spot_interruption` retry —
    produced by the real runner absorbing a real `SpotInterruptionError`, not
    by writing an `attempts[]` row by hand.

    This is the outcome fault 1 is designed to produce, and running it through
    `run_job` is what proves the claim the `absorbed` kind rests on: a spot
    interruption IS in `crucible.runner.TRANSIENT_CLASSIFIERS`, so the runner
    retries it and the run succeeds.
    """
    attempts: list[int] = []

    def body(ctx) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise SpotInterruptionError(
                "spot_interruption: received signal 15; the instance is being reclaimed"
            )

    run_job(job, body, store=store, trading_day=trading_day, transient_retry=True)
    key = manifest_key(job, trading_day.isoformat())
    document = json.loads(store.get_bytes(key))
    assert document["status"] == "ok"
    assert [row["reason"] for row in document["attempts"]] == ["initial", "spot_interruption"]
    return document["run_id"]


def _record(store, **kwargs):
    """Run `fault.record` through `run_job`, as the CLI does."""
    defaults = {
        "target_job": None,
        "trading_day": FRIDAY.isoformat(),
        "run_id": None,
        "bus_key": None,
    }
    defaults.update(kwargs)

    def job(ctx) -> None:
        record_fault(ctx, store, **defaults)

    return run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)


class TestRefusalsCommonToEveryOutcome:
    def test_refuses_an_unrecognized_fault_id(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        store.put_bytes(BUS_KEY, b'{"condition": "failure"}')
        with pytest.raises(FaultRecordRefusedError, match="not one of"):
            _record(
                store,
                fault_id="not_a_scripted_fault",
                outcome="induced",
                target_job="data.weekly",
                run_id=run_id,
                bus_key=BUS_KEY,
            )
        assert not store.exists(fault_injection_key("not_a_scripted_fault", FRIDAY.isoformat()))

    def test_refuses_an_unrecognized_outcome(self, tmp_path) -> None:
        """The three kinds are a closed set, for the same reason the runner has
        two statuses: a fourth is a design change visible in a diff."""
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        with pytest.raises(FaultRecordRefusedError, match="is not one of"):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="mostly_fine",
                target_job="data.weekly",
                run_id=run_id,
            )

    def test_refuses_before_writing_anything(self, tmp_path) -> None:
        """The refusal happens before ANY store write — including this
        job's own manifest content beyond `status: failed`. A producer that
        could write partial evidence and then refuse would leave a trace an
        operator could mistake for a filed record."""
        store = LocalStore(tmp_path)
        before = sorted(store.list_keys())
        with pytest.raises(FaultRecordRefusedError):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="induced",
                target_job="data.weekly",
                run_id=NO_SUCH_RUN_ID,
                bus_key=BUS_KEY,
            )
        # The runner's own manifest for THIS failed fault.record run is the
        # only new key — no faults/ record and nothing else landed.
        after = sorted(store.list_keys())
        assert sorted(set(after) - set(before)) == [
            manifest_key("fault.record", FRIDAY.isoformat())
        ]


class TestInduced:
    """The shape that shipped: a FAILED manifest plus the page it produced."""

    def test_refuses_when_no_manifest_carries_the_named_run_id(self, tmp_path) -> None:
        """The whole `-I10322` design constraint: a record cannot excuse a
        run that does not exist on the store."""
        store = LocalStore(tmp_path)
        _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        store.put_bytes(BUS_KEY, b'{"condition": "failure"}')
        with pytest.raises(FaultRecordRefusedError, match="no manifest under"):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="induced",
                target_job="data.weekly",
                run_id=NO_SUCH_RUN_ID,
                bus_key=BUS_KEY,
            )
        assert not store.exists(fault_injection_key("data_source_withheld", FRIDAY.isoformat()))

    def test_refuses_a_run_id_that_succeeded(self, tmp_path) -> None:
        """A record naming a successful run is exactly the arbitrary-excuse
        shape `-I10322` forbids: refused, never filed. The refusal names
        `absorbed` as the kind for a run the system handled, so the operator
        is pointed at the harder record rather than at a way around this one."""
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=True)
        store.put_bytes(BUS_KEY, b'{"condition": "failure"}')
        with pytest.raises(FaultRecordRefusedError, match="not 'failed'"):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="induced",
                target_job="data.weekly",
                run_id=run_id,
                bus_key=BUS_KEY,
            )
        assert not store.exists(fault_injection_key("data_source_withheld", FRIDAY.isoformat()))

    def test_refuses_a_missing_bus_key(self, tmp_path) -> None:
        """`-I10327` tightened this: `bus_key` was nullable for any record, and
        an induced fault that paged nobody is half of plan §10.7's exercise."""
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        with pytest.raises(FaultRecordRefusedError, match="requires --bus-key"):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="induced",
                target_job="data.weekly",
                run_id=run_id,
            )
        assert not store.exists(fault_injection_key("data_source_withheld", FRIDAY.isoformat()))

    def test_refuses_a_bus_key_the_store_does_not_hold(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        with pytest.raises(FaultRecordRefusedError, match="store does not hold"):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="induced",
                target_job="data.weekly",
                run_id=run_id,
                bus_key=BUS_KEY,
            )
        assert not store.exists(fault_injection_key("data_source_withheld", FRIDAY.isoformat()))

    def test_refuses_a_bus_key_of_the_wrong_shape(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        with pytest.raises(FaultRecordRefusedError, match="not shaped like"):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="induced",
                target_job="data.weekly",
                run_id=run_id,
                bus_key="not/a/bus/key/at/all.json",
            )

    def test_refuses_without_a_target_job(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        with pytest.raises(FaultRecordRefusedError, match="both --target-job and --run-id"):
            _record(
                store,
                fault_id="data_source_withheld",
                outcome="induced",
                run_id=run_id,
                bus_key=BUS_KEY,
            )

    def test_writes_the_record_naming_the_excused_run_and_the_page(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        store.put_bytes(BUS_KEY, b'{"condition": "failure"}')
        ctx = _record(
            store,
            fault_id="data_source_withheld",
            outcome="induced",
            target_job="data.weekly",
            run_id=run_id,
            bus_key=BUS_KEY,
        )
        assert ctx.outputs  # this job's own manifest carries the write as lineage

        key = fault_injection_key("data_source_withheld", FRIDAY.isoformat())
        document = json.loads(store.get_bytes(key))
        assert document["fault_id"] == "data_source_withheld"
        assert document["outcome"] == "induced"
        assert document["trading_day"] == FRIDAY.isoformat()
        assert document["run_id"] == run_id
        assert document["manifest_key"] == manifest_key("data.weekly", FRIDAY.isoformat())
        assert document["bus_key"] == BUS_KEY
        assert document["attempt"] is None
        assert document["closed_paths"] is None
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
        store.put_bytes(BUS_KEY, b'{"condition": "failure"}')
        first_run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        _record(
            store,
            fault_id="data_source_withheld",
            outcome="induced",
            target_job="data.weekly",
            run_id=first_run_id,
            bus_key=BUS_KEY,
        )
        # Stand down, then re-induce: a fresh failed manifest, a fresh run_id.
        second_run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        assert second_run_id != first_run_id
        _record(
            store,
            fault_id="data_source_withheld",
            outcome="induced",
            target_job="data.weekly",
            run_id=second_run_id,
            bus_key=BUS_KEY,
        )
        key = fault_injection_key("data_source_withheld", FRIDAY.isoformat())
        assert json.loads(store.get_bytes(key))["run_id"] == second_run_id


class TestAbsorbed:
    """`alpha-engine-config-I10327`. Fault 1's designed outcome: the fault
    fired and the declared transient class HANDLED it, so the manifest reads
    `ok` and its `attempts[]` records the retry. A stronger result than
    `induced`, and deliberately not reachable by relaxing `induced`'s
    requirements — it has its own, and one of them is the ABSENCE of a page.
    """

    def test_refuses_an_ok_manifest_with_no_retry(self, tmp_path) -> None:
        """The refusal that makes this kind mean anything. A clean
        first-attempt `ok` manifest is evidence the job worked; it is not
        evidence a fault was absorbed, and accepting it would make `absorbed`
        satisfiable by any successful run of any job."""
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=True)
        with pytest.raises(FaultRecordRefusedError, match="not one retry in the declared"):
            _record(
                store,
                fault_id="spot_terminated_mid_job",
                outcome="absorbed",
                target_job="data.weekly",
                run_id=run_id,
            )
        assert not store.exists(fault_injection_key("spot_terminated_mid_job", FRIDAY.isoformat()))

    def test_refuses_a_failed_manifest(self, tmp_path) -> None:
        """The other direction: a run that FAILED was not absorbed, and the
        refusal points at `induced`, which requires the bus row."""
        store = LocalStore(tmp_path)
        run_id = _seed_manifest(store, job="data.weekly", trading_day=FRIDAY, ok=False)
        with pytest.raises(FaultRecordRefusedError, match="not 'ok'"):
            _record(
                store,
                fault_id="spot_terminated_mid_job",
                outcome="absorbed",
                target_job="data.weekly",
                run_id=run_id,
            )

    def test_refuses_a_bus_key_even_when_the_store_holds_it(self, tmp_path) -> None:
        """The refusal `alpha-engine-config-I10327` asks for by name. A page on
        an absorbed fault would mean the retry did NOT work, so the bus row's
        absence is part of what this record asserts — and naming an existing
        row from an unrelated incident is the rubber stamp the `run_id` refusal
        exists to prevent, arriving through the other field. Refused even
        though the key is real and present, which is the case a
        does-the-key-exist check would wave through."""
        store = LocalStore(tmp_path)
        run_id = _seed_absorbed_manifest(store, job="data.weekly", trading_day=FRIDAY)
        store.put_bytes(BUS_KEY, b'{"condition": "failure"}')
        with pytest.raises(FaultRecordRefusedError, match="forbids --bus-key"):
            _record(
                store,
                fault_id="spot_terminated_mid_job",
                outcome="absorbed",
                target_job="data.weekly",
                run_id=run_id,
                bus_key=BUS_KEY,
            )
        assert not store.exists(fault_injection_key("spot_terminated_mid_job", FRIDAY.isoformat()))

    def test_writes_the_record_naming_the_retry_that_absorbed_the_fault(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_id = _seed_absorbed_manifest(store, job="data.weekly", trading_day=FRIDAY)
        _record(
            store,
            fault_id="spot_terminated_mid_job",
            outcome="absorbed",
            target_job="data.weekly",
            run_id=run_id,
        )
        key = fault_injection_key("spot_terminated_mid_job", FRIDAY.isoformat())
        document = json.loads(store.get_bytes(key))
        assert document["outcome"] == "absorbed"
        assert document["run_id"] == run_id
        assert document["manifest_key"] == manifest_key("data.weekly", FRIDAY.isoformat())
        assert document["bus_key"] is None
        assert document["attempt"] == {"n": 2, "reason": "spot_interruption"}
        assert document["closed_paths"] is None

    def test_a_spot_interruption_is_genuinely_in_the_declared_transient_class(
        self, tmp_path
    ) -> None:
        """`alpha-engine-config-I10327`'s "verify before building": if a spot
        interruption FAILED the run instead of being retried, fault 1's outcome
        would be `induced` and this kind would have no producer. Asserted
        against the runner itself rather than read off `TRANSIENT_CLASSIFIERS`,
        so the claim is about behaviour and not about a tuple."""
        store = LocalStore(tmp_path)
        _seed_absorbed_manifest(store, job="data.weekly", trading_day=FRIDAY)
        document = json.loads(store.get_bytes(manifest_key("data.weekly", FRIDAY.isoformat())))
        assert document["status"] == "ok"
        assert document["attempts"][-1]["reason"] == "spot_interruption"


class _Paginator:
    def __init__(self, fake: _FakeS3) -> None:
        self.fake = fake

    def paginate(self, **kw) -> list:
        prefix = kw.get("Prefix") or ""
        keys = sorted(k for k in self.fake.objects if k.startswith(prefix))
        return [{"Contents": [{"Key": k} for k in keys]}]


class _FakeS3:
    """Enough of boto3's S3 surface for the `unreachable` probes: the release
    listing, `head_object`, and `get_object_retention`.

    An `S3Store` with a substituted client, not a stub of the store — the
    probes read Object Lock retention, which `LocalStore` has no concept of at
    all, and that is exactly why they refuse to file a record from a laptop
    store. Modelling only what this module's probes call, per the convention
    the other release fixtures follow.
    """

    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.retentions: dict[str, dict | str] = {}
        self.blobs: dict[str, bytes] = {}

    def publish(self, sha: str, *, locked: bool = True, retain_years: int = 10) -> None:
        published_at = dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC)
        for key in (f"crucible/{release_json_key(sha)}", f"crucible/{wheel_key(sha)}"):
            self.objects[key] = {"LastModified": published_at}
            self.retentions[key] = (
                {
                    "Mode": "GOVERNANCE",
                    "RetainUntilDate": published_at + dt.timedelta(days=365 * retain_years),
                }
                if locked
                else "NoSuchObjectLockConfiguration"
            )

    def head_object(self, **kw) -> dict:
        from botocore.exceptions import ClientError

        key = kw["Key"]
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "head_object")
        return self.objects[key]

    def get_object_retention(self, **kw) -> dict:
        from botocore.exceptions import ClientError

        outcome = self.retentions.get(kw["Key"])
        if isinstance(outcome, str):
            raise ClientError({"Error": {"Code": outcome}}, "get_object_retention")
        return {"Retention": outcome or {}}

    def put_object(self, **kw) -> dict:
        key = kw["Key"]
        self.blobs[key] = kw["Body"]
        self.objects.setdefault(key, {"LastModified": dt.datetime(2026, 8, 1, tzinfo=dt.UTC)})
        return {"ETag": '"fake"'}

    def get_object(self, **kw) -> dict:
        from botocore.exceptions import ClientError

        key = kw["Key"]
        if key not in self.blobs:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "get_object")

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(self.blobs[key])}

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)


def _s3_store_with_a_published_release(**kw) -> tuple[S3Store, _FakeS3]:
    client = _FakeS3()
    client.publish("a" * 40, **kw)
    return S3Store("a-test-store", "crucible", client=client), client


class TestUnreachable:
    """`alpha-engine-config-I10327`. Fault 4's state cannot be entered at all,
    so there is no run to name — which removes the `run_id` refusal that guards
    every other kind. That makes `unreachable` the HARDEST record here to
    write, not the easiest: the evidence is a probe EXECUTED per closed path,
    and every probe must observe what it required or the record is refused.
    """

    def test_refuses_a_fault_with_no_declared_probe(self, tmp_path) -> None:
        """The enforcement of "no attestation a human types". A fault with no
        registered probe has no machine-checkable evidence, so the clause stays
        UNMEASURABLE rather than being filed over."""
        store = LocalStore(tmp_path)
        with pytest.raises(FaultRecordRefusedError, match="no machine-checkable evidence"):
            _record(store, fault_id="data_source_withheld", outcome="unreachable")

    def test_refuses_a_run_id(self, tmp_path) -> None:
        """The structural half: a record that claims a state cannot be entered
        must not also be able to excuse a manifest."""
        store = LocalStore(tmp_path)
        with pytest.raises(FaultRecordRefusedError, match="names no run"):
            _record(
                store,
                fault_id="stale_release_pointer",
                outcome="unreachable",
                target_job="data.weekly",
                run_id=NO_SUCH_RUN_ID,
            )

    def test_refuses_a_bus_key(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        with pytest.raises(FaultRecordRefusedError, match="nothing for a page to be about"):
            _record(
                store,
                fault_id="stale_release_pointer",
                outcome="unreachable",
                bus_key=BUS_KEY,
            )

    def test_refuses_against_a_local_store_because_the_probe_cannot_read_retention(
        self, tmp_path
    ) -> None:
        """The evidence has to come from the store the claim is about. A
        `LocalStore` has no Object Lock concept, so the retention probe reports
        every key UNMEASURABLE and the record is refused — this record cannot be
        filed from a laptop."""
        store = LocalStore(tmp_path)
        store.put_bytes(release_json_key("a" * 40), b"{}")
        with pytest.raises(FaultRecordRefusedError, match="could not be read"):
            _record(store, fault_id="stale_release_pointer", outcome="unreachable")

    def test_refuses_when_no_release_has_been_published(self) -> None:
        """ "every published object is locked" over zero objects is vacuously
        true and evidence of nothing."""
        store = S3Store("a-test-store", "crucible", client=_FakeS3())
        with pytest.raises(FaultRecordRefusedError, match="not one published release object"):
            _record(store, fault_id="stale_release_pointer", outcome="unreachable")

    def test_refuses_when_a_published_object_is_not_retained(self) -> None:
        """The probe firing on a genuinely open path: an unlocked release object
        means the artifacts CAN be deleted from under a valid pointer, so the
        state is reachable and this is a finding, not a record."""
        store, _client = _s3_store_with_a_published_release(locked=False)
        with pytest.raises(FaultRecordRefusedError, match="are NOT retained to policy"):
            _record(store, fault_id="stale_release_pointer", outcome="unreachable")

    def test_writes_the_record_carrying_every_executed_probe(self) -> None:
        store, _client = _s3_store_with_a_published_release()
        _record(store, fault_id="stale_release_pointer", outcome="unreachable")
        key = fault_injection_key("stale_release_pointer", FRIDAY.isoformat())
        document = json.loads(store.get_bytes(key))
        assert document["outcome"] == "unreachable"
        assert document["run_id"] is None
        assert document["manifest_key"] is None
        assert document["bus_key"] is None
        assert document["attempt"] is None
        probes = {row["probe"] for row in document["closed_paths"]}
        assert probes == {
            "pin_refuses_an_unpublished_sha",
            "published_release_objects_are_retained",
        }
        for row in document["closed_paths"]:
            assert row["observed"] and row["expected"] and row["mechanism"]
            assert row["checked_at_utc"].endswith("Z")

    def test_the_pin_probe_observed_a_real_refusal_not_a_sentence(self) -> None:
        """The probe RAN `release.pin`: the observation is the exception the
        real refusal raised, so the record cites behaviour rather than a
        docstring."""
        store, _client = _s3_store_with_a_published_release()
        _record(store, fault_id="stale_release_pointer", outcome="unreachable")
        document = json.loads(
            store.get_bytes(fault_injection_key("stale_release_pointer", FRIDAY.isoformat()))
        )
        row = next(
            r for r in document["closed_paths"] if r["probe"] == "pin_refuses_an_unpublished_sha"
        )
        assert row["observed"].startswith("StaleReleasePointerError:")
        assert "was never published" in row["observed"]

    def test_the_probe_cannot_move_the_pointer(self) -> None:
        """The probe calls the real `pin`, so it is wrapped in a read-only view
        of the store: no probe may mutate the system it is describing, and a
        `pin` that stopped refusing would hit that wall rather than flip the
        production pointer."""
        store, client = _s3_store_with_a_published_release()
        _record(store, fault_id="stale_release_pointer", outcome="unreachable")
        assert not any("releases/current" in key for key in client.blobs)
