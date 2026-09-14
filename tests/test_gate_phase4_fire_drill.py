"""The phase-4 kill-switch fire drill clause and its register entry (`alpha-engine-config-I10650`).

Mutation-tested against each way a drill record can read as evidence while not
being any: a `passed` flag the evidence contradicts, an order let through after
the fire, a bound overrun, a document with no run behind it, a document edited
after its run, a replay, only announced drills, only one drill — and the two
readings that must never collapse (absence is UNMET; an unreadable store is
UNMEASURABLE).

The store is an `S3Store` over the suite's `FakeS3`, because an unannounced
drill counts only on a sealed schedule whose STORE-SET write time
(`LastModified`) precedes its fire (`alpha-engine-config-I10761`), and a local
directory has no such time. `TestSeal` mutation-tests that seal.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from conftest import FakeS3
from test_gate_phase4_trader_clauses import _manifest, _put

from crucible.faults import ARTIFACT_GRADED_FAULTS, FaultRecordRefusedError, record_fault
from crucible.gate import (
    FIRE_DRILL_FAULT,
    FIRE_DRILL_SCHEMA_VERSION,
    PHASE4_DELIVERABLES,
    RECONCILIATION_FAULT,
    SCRIPTED_FAULTS,
    TRADER_FIRE_DRILL_JOB,
    _clause_kill_switch_fire_drill_passed,
    _phase4,
    kill_switch_fire_drill_reading,
)
from crucible.keys import (
    TRADER_FIRE_DRILL_SCHEDULE_PREFIX,
    TRADER_FIRE_DRILLS_PREFIX,
    manifest_key,
    trader_fire_drill_key,
    trader_fire_drill_schedule_key,
)
from crucible.models import (
    FIRE_DRILL_COMMITMENT_SCHEME,
    FIRE_DRILL_SEAL_TOLERANCE_SECONDS,
    JOB_VALUES,
    TRADER_JOB_VALUES,
    FireDrillDocument,
    FireDrillScheduleDocument,
    fire_drill_commitment,
)
from crucible.store import LocalStore, S3Store, Store, sha256_hex

WINDOW = [dt.date(2026, 9, 1), dt.date(2026, 9, 11)]
FRIDAY = WINDOW[-1]


@pytest.fixture
def store() -> S3Store:
    return S3Store("a-test-store", "crucible", client=FakeS3())


def _utc(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


def _seal(
    store: Store,
    day: str,
    n: int,
    *,
    written_at: dt.datetime | None = None,
    sealed_instant: str | None = None,
    write: bool = True,
    window: tuple[str, str] = ("2026-09-01", "2026-09-11"),
) -> dict:
    """Seal drill ``n`` for 15:00:00Z on ``day``, as the operator's `fire-drill seal`
    does, and return the reveal its drill document carries. ``written_at`` is
    when the FAKE SERVICE stamps the write (default: a day before the fire);
    ``sealed_instant`` commits to a different instant than the one revealed."""
    fire_instant = f"{day}T15:00:00Z"
    schedule_id, nonce = f"{n:032x}", f"{n:064x}"
    if write:
        document = {
            "schema_version": "fire_drill_schedule.v1",
            "schedule_id": schedule_id,
            "window_start": window[0],
            "window_end": window[1],
            "commitment_scheme": FIRE_DRILL_COMMITMENT_SCHEME,
            "commitment": fire_drill_commitment(sealed_instant or fire_instant, nonce),
        }
        if isinstance(store, S3Store):
            stamp = written_at or _utc(fire_instant) - dt.timedelta(days=1)
            store.client.now = lambda: stamp
        _put(store, trader_fire_drill_schedule_key(*window, schedule_id), document)
    return {
        "schedule_id": schedule_id,
        "window_start": window[0],
        "window_end": window[1],
        "fire_instant": fire_instant,
        "nonce": nonce,
    }


def _run_id(n: int) -> str:
    return f"01JG{n:022d}"


def _drill(day: str, run_id: str, **overrides: Any) -> dict:
    """Exactly the fields `crucible_trader.fire_drill.drill_document` writes."""
    document = {
        "schema_version": FIRE_DRILL_SCHEMA_VERSION,
        "kind": "kill_switch_freeze",
        "announced": True,
        "run_id": run_id,
        "trading_day": day,
        "account": "DU0000000",
        "mode": "freeze",
        "fired_at": f"{day}T15:00:00.000000Z",
        "settled_at": f"{day}T15:00:04.000000Z",
        "settle_seconds": 4.0,
        "bound_seconds": 60.0,
        "state_before": {"positions": {"AAA": 10}},
        "state_after": {"positions": {"AAA": 10}},
        "orders_accepted_after_fire": [],
        "hold_decision": None,
        "passed": True,
    }
    document.update(overrides)
    return document


def _drill_run(
    store: Store,
    day: str,
    n: int,
    *,
    record_output: bool = True,
    status: str = "ok",
    run_mode: str = "live",
    seal: bool = True,
    **drill_overrides: Any,
) -> str:
    """One drill and its run. An unannounced drill is SEALED by default (a
    correct seal, written before the fire); ``seal=False`` or an explicit
    ``schedule=`` override is how a test breaks that."""
    run_id = _run_id(n)
    if drill_overrides.get("announced") is False and "schedule" not in drill_overrides and seal:
        drill_overrides["schedule"] = _seal(store, day, n)
    key = trader_fire_drill_key(day, run_id)
    payload = _put(store, key, _drill(day, run_id, **drill_overrides))
    outputs = (
        [{"key": key, "sha256": sha256_hex(payload), "schema_version": FIRE_DRILL_SCHEMA_VERSION}]
        if record_output
        else []
    )
    manifest = _manifest(day, outputs=outputs, status=status, run_mode=run_mode)
    manifest |= {"job": TRADER_FIRE_DRILL_JOB, "run_id": run_id}
    _put(store, manifest_key(TRADER_FIRE_DRILL_JOB, day, discriminator=f"drill-{n}"), manifest)
    return key


def _clause(store: Store):
    return _clause_kill_switch_fire_drill_passed(store, WINDOW)


def _two_good(store: Store) -> None:
    _drill_run(store, "2026-09-08", 1)
    _drill_run(store, "2026-09-10", 2, announced=False)


class TestMet:
    def test_two_passed_one_unannounced_is_met(self, store) -> None:
        _two_good(store)
        clause = _clause(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "2 passed drill(s)" in clause.detail

    def test_a_failed_drill_beside_two_passes_is_named_not_voiding(self, store) -> None:
        _two_good(store)
        _drill_run(store, "2026-09-09", 3, status="failed", settled_at=None, settle_seconds=None)
        clause = _clause(store)
        assert clause.met
        assert "failed:" in clause.detail and _run_id(3) in clause.detail


class TestUnmet:
    def test_an_empty_store_is_unmet_never_unmeasurable(self, store) -> None:
        clause = _clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "undrilled" in clause.detail

    def test_one_drill_is_not_two(self, store) -> None:
        _drill_run(store, "2026-09-08", 1, announced=False)
        assert not _clause(store).met

    def test_only_announced_drills_are_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2)
        clause = _clause(store)
        assert not clause.met and "0 unannounced" in clause.detail

    def test_drills_outside_the_window_do_not_count(self, store) -> None:
        _drill_run(store, "2026-08-27", 1)
        _drill_run(store, "2026-08-28", 2, announced=False)
        assert not _clause(store).met

    @pytest.mark.parametrize(
        "mutation",
        [
            pytest.param({"settled_at": None, "settle_seconds": None}, id="never-settled"),
            pytest.param({"settle_seconds": 61.0}, id="over-its-bound"),
            pytest.param(
                {
                    "orders_accepted_after_fire": [
                        {
                            "order_id": 7,
                            "symbol": "AAA",
                            "action": "BUY",
                            "quantity": 1.0,
                            "order_ref": "",
                            "accepted_at": "2026-09-10T15:00:01.000000Z",
                        }
                    ]
                },
                id="order-accepted-after-the-fire",
            ),
        ],
    )
    def test_a_passed_flag_the_evidence_contradicts_does_not_count(
        self, store, mutation: dict
    ) -> None:
        """The flag says passed; the evidence does not. The evidence wins."""
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, **mutation)
        clause = _clause(store)
        assert not clause.met and not clause.unmeasurable
        assert _run_id(2) in clause.detail

    def test_a_document_with_no_run_behind_it_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        run_id = _run_id(2)
        _put(
            store,
            trader_fire_drill_key("2026-09-10", run_id),
            _drill("2026-09-10", run_id, announced=False),
        )
        clause = _clause(store)
        assert not clause.met and "no run behind it" in clause.detail

    def test_a_document_edited_after_its_run_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        key = _drill_run(store, "2026-09-10", 2, announced=True)
        edited = json.loads(store.get_bytes(key)) | {"announced": False}
        _put(store, key, edited)
        clause = _clause(store)
        assert not clause.met and "changed after the run" in clause.detail

    def test_a_document_the_run_did_not_record_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, record_output=False)
        clause = _clause(store)
        assert not clause.met and "0 output(s)" in clause.detail

    def test_a_replayed_drill_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, run_mode="replay")
        clause = _clause(store)
        assert not clause.met and "replayed drill" in clause.detail

    def test_passed_against_a_failed_manifest_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, status="failed")
        clause = _clause(store)
        assert not clause.met and "must agree" in clause.detail

    def test_a_non_conforming_document_is_unmet_naming_the_key(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        key = trader_fire_drill_key("2026-09-10", _run_id(2))
        _put(store, key, {"schema_version": "fire_drill.v0"})
        clause = _clause(store)
        assert not clause.met and not clause.unmeasurable and key in clause.detail

    def test_a_document_filed_under_another_run_id_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        run_id = _run_id(2)
        payload = _put(
            store,
            trader_fire_drill_key("2026-09-10", _run_id(9)),
            _drill("2026-09-10", run_id, announced=False),
        )
        del payload
        clause = _clause(store)
        assert not clause.met and "which belong at" in clause.detail


class TestSeal:
    """`announced: false` is a claim; the seal is the reading (I10761)."""

    def test_a_correctly_sealed_drill_reads_as_unannounced(self, store) -> None:
        _two_good(store)
        clause = _clause(store)
        assert clause.met, clause.detail
        assert "1 unannounced" in clause.detail and "claim not counted" not in clause.detail
        schedule_key = trader_fire_drill_schedule_key("2026-09-01", "2026-09-11", f"{2:032x}")
        assert schedule_key in clause.evidence
        assert "does not conform" not in clause.detail  # the seal is not read as a drill

    def test_no_schedule_revealed_is_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, seal=False)
        clause = _clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "0 unannounced" in clause.detail and "reveals no sealed schedule" in clause.detail

    def test_a_revealed_schedule_that_was_never_written_is_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-10", 2, write=False)
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        clause = _clause(store)
        assert not clause.met and "no sealed schedule at" in clause.detail

    @pytest.mark.parametrize(
        "written_at",
        [
            pytest.param("2026-09-10T15:00:00+00:00", id="written-at-the-fire-instant"),
            pytest.param("2026-09-10T15:00:05+00:00", id="written-after-the-fire"),
        ],
    )
    def test_a_schedule_written_at_or_after_the_fire_is_unmet(self, store, written_at) -> None:
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-10", 2, written_at=_utc(written_at))
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        clause = _clause(store)
        assert not clause.met and "a seal written after the fire seals nothing" in clause.detail

    def test_the_write_time_is_the_services_not_a_field_the_writer_sets(self, store) -> None:
        """A seal document cannot carry its own write time: the model forbids it."""
        reveal = _seal(store, "2026-09-10", 2, write=False)
        document = {
            "schema_version": "fire_drill_schedule.v1",
            "schedule_id": reveal["schedule_id"],
            "window_start": reveal["window_start"],
            "window_end": reveal["window_end"],
            "commitment_scheme": FIRE_DRILL_COMMITMENT_SCHEME,
            "commitment": "0" * 64,
            "sealed_at": "2026-09-01T00:00:00Z",
        }
        with pytest.raises(ValueError, match="sealed_at"):
            FireDrillScheduleDocument.model_validate(document)

    def test_a_hash_mismatch_is_unmet(self, store) -> None:
        """The seal committed to 15:00; the drill reveals 15:00 with a nonce that
        opens a commitment to a different instant."""
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-10", 2, sealed_instant="2026-09-10T18:30:00Z")
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        clause = _clause(store)
        assert not clause.met and "does not match the commitment" in clause.detail

    def test_a_wrong_nonce_is_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-10", 2) | {"nonce": "f" * 64}
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        assert "does not match the commitment" in _clause(store).detail

    @pytest.mark.parametrize(
        "fired_at",
        [
            pytest.param("2026-09-10T14:59:59.000000Z", id="fired-before-the-sealed-instant"),
            pytest.param(
                (
                    _utc("2026-09-10T15:00:00+00:00")
                    + dt.timedelta(seconds=FIRE_DRILL_SEAL_TOLERANCE_SECONDS + 1)
                ).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                id="fired-past-the-tolerance",
            ),
            pytest.param("not-an-instant", id="unparseable-fire-time"),
        ],
    )
    def test_a_fire_that_is_not_the_sealed_fire_is_unmet(self, store, fired_at) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, fired_at=fired_at)
        clause = _clause(store)
        assert not clause.met and "claim not counted" in clause.detail

    def test_a_seal_on_another_day_or_window_is_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-09", 2)
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        assert "is not on trading_day" in _clause(store).detail
        reveal = _seal(store, "2026-09-10", 3, window=("2026-09-01", "2026-09-04"))
        _drill_run(store, "2026-09-10", 3, announced=False, schedule=reveal)
        assert "outside its seal's window" in _clause(store).detail

    def test_a_seal_filed_under_another_schedule_id_is_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-10", 2)
        key = trader_fire_drill_schedule_key("2026-09-01", "2026-09-11", reveal["schedule_id"])
        moved = json.loads(store.get_bytes(key)) | {"schedule_id": f"{9:032x}"}
        _put(store, key, moved)
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        assert "not the schedule its key names" in _clause(store).detail

    def test_a_non_conforming_seal_is_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-10", 2)
        key = trader_fire_drill_schedule_key("2026-09-01", "2026-09-11", reveal["schedule_id"])
        _put(store, key, {"schema_version": "fire_drill_schedule.v0"})
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        assert "does not conform to fire_drill_schedule.v1" in _clause(store).detail

    def test_one_seal_opens_one_fire(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        reveal = _seal(store, "2026-09-10", 2)
        _drill_run(store, "2026-09-10", 2, announced=False, schedule=reveal)
        _drill_run(store, "2026-09-10", 3, announced=False, schedule=reveal)
        clause = _clause(store)
        assert clause.met and "1 unannounced" in clause.detail
        assert "already opened by another drill" in clause.detail

    def test_a_local_directory_has_no_store_set_write_time(self, tmp_path) -> None:
        local = LocalStore(tmp_path / "store")
        _drill_run(local, "2026-09-08", 1)
        _drill_run(local, "2026-09-10", 2, announced=False)
        clause = _clause(local)
        assert not clause.met and not clause.unmeasurable
        assert "no store-set write time" in clause.detail

    def test_an_unreadable_seal_write_time_is_unmeasurable(self, store) -> None:
        _two_good(store)
        key = trader_fire_drill_schedule_key("2026-09-01", "2026-09-11", f"{2:032x}")

        real = store.client.head_object
        calls: list[str] = []

        def denied(**kw):
            """`exists` HEADs too, so only the write-time HEAD (the second one on
            the seal key) is denied — the document itself stays readable."""
            if kw["Key"].endswith(key):
                calls.append(kw["Key"])
                if len(calls) > 1:
                    raise FakeS3._client_error("AccessDenied")
            return real(**kw)

        store.client.head_object = denied
        clause = _clause(store)
        assert clause.unmeasurable and not clause.met and key in clause.detail

    def test_an_unreadable_seal_document_is_unmeasurable(self, store) -> None:
        _two_good(store)
        real = store.get_bytes

        def denied(key: str) -> bytes:
            if key.startswith(TRADER_FIRE_DRILL_SCHEDULE_PREFIX):
                raise PermissionError("AccessDenied")
            return real(key)

        store.get_bytes = denied
        clause = _clause(store)
        assert clause.unmeasurable and not clause.met

    def test_an_announced_drill_cannot_reveal_a_seal(self, store) -> None:
        reveal = _seal(store, "2026-09-08", 1, write=False)
        with pytest.raises(ValueError, match="contradict"):
            FireDrillDocument.model_validate(
                _drill("2026-09-08", _run_id(1), announced=True, schedule=reveal)
            )

    def test_the_commitment_is_sha256_of_instant_then_nonce(self) -> None:
        import hashlib

        nonce = "ab" * 32
        assert (
            fire_drill_commitment("2026-09-10T15:00:00Z", nonce)
            == hashlib.sha256(f"2026-09-10T15:00:00Z{nonce}".encode()).hexdigest()
        )

    def test_the_schedule_key_shape_and_refusals(self) -> None:
        assert (
            trader_fire_drill_schedule_key("2026-09-14", "2026-09-18", "a" * 32)
            == f"trader/fire_drills/schedule/2026-09-14_2026-09-18/{'a' * 32}.json"
        )
        assert trader_fire_drill_schedule_key("2026-09-14", "2026-09-18", "x").startswith(
            TRADER_FIRE_DRILL_SCHEDULE_PREFIX
        )
        for args in (
            ("2026-09-18", "2026-09-14", "x"),
            ("2026-09-14", "2026-09-18", ""),
            ("2026-09-14", "2026-09-18", "a/b"),
            ("2026-9-14", "2026-09-18", "x"),
        ):
            with pytest.raises(ValueError):
                trader_fire_drill_schedule_key(*args)
        with pytest.raises(ValueError, match="ends before it starts"):
            FireDrillScheduleDocument.model_validate(
                {
                    "schema_version": "fire_drill_schedule.v1",
                    "schedule_id": "a" * 32,
                    "window_start": "2026-09-18",
                    "window_end": "2026-09-14",
                    "commitment_scheme": FIRE_DRILL_COMMITMENT_SCHEME,
                    "commitment": "0" * 64,
                }
            )


class TestUnmeasurable:
    def test_an_unlistable_drill_prefix_is_unmeasurable(self, store, monkeypatch) -> None:
        def denied(prefix: str):
            raise PermissionError(f"AccessDenied listing {prefix}")

        monkeypatch.setattr(store, "list_keys", denied)
        clause = _clause(store)
        assert clause.unmeasurable and not clause.met

    def test_a_denied_read_is_unmeasurable(self, store, monkeypatch) -> None:
        _two_good(store)
        real = store.get_bytes

        def denied(key: str) -> bytes:
            if key.startswith(TRADER_FIRE_DRILLS_PREFIX):
                raise PermissionError("AccessDenied")
            return real(key)

        monkeypatch.setattr(store, "get_bytes", denied)
        clause = _clause(store)
        assert clause.unmeasurable and not clause.met


class TestRegistration:
    def test_both_trader_jobs_are_admitted(self) -> None:
        for job in ("trader.kill_switch", TRADER_FIRE_DRILL_JOB):
            assert job in TRADER_JOB_VALUES and job in JOB_VALUES

    def test_the_deliverable_names_the_clause_phase_4_evaluates(self, store) -> None:
        (row,) = [d for d in PHASE4_DELIVERABLES if d.id == "kill_switch_fire_drill"]
        assert row.graded_by == "kill_switch_fire_drill_passed"
        readings = {c.name: c for c in _phase4(store, WINDOW, {}, trading_day=FRIDAY)}
        assert not readings["kill_switch_fire_drill_passed"].met
        assert not readings["kill_switch_fire_drill_passed"].unmeasurable

    def test_the_public_reading_is_the_contained_clause(self, store) -> None:
        _two_good(store)
        reading = kill_switch_fire_drill_reading(store, WINDOW[0], WINDOW[-1])
        assert reading == _clause(store) and reading.met

        def exploding(prefix: str):
            raise RuntimeError("the store fell over")

        store.list_keys = exploding
        assert kill_switch_fire_drill_reading(store, WINDOW[0], WINDOW[-1]).unmeasurable

    def test_the_fixture_is_the_model(self) -> None:
        """The test fixture above mirrors crucible-trader's writer field for field."""
        FireDrillDocument.model_validate(_drill("2026-09-08", _run_id(1)))


class TestRegisterEntry:
    def test_the_drill_is_fault_fives_sibling(self) -> None:
        assert SCRIPTED_FAULTS[-2:] == (RECONCILIATION_FAULT, FIRE_DRILL_FAULT)

    def test_the_entry_and_the_clause_read_the_same_artifact(self) -> None:
        entry = ARTIFACT_GRADED_FAULTS[FIRE_DRILL_FAULT]
        assert entry.producer_job == TRADER_FIRE_DRILL_JOB
        assert entry.artifact_prefix == TRADER_FIRE_DRILLS_PREFIX
        assert entry.schema_version == FIRE_DRILL_SCHEMA_VERSION
        assert entry.graded_by == _clause_kill_switch_fire_drill_passed.__name__.removeprefix(
            "_clause_"
        )
        assert trader_fire_drill_key("2026-09-08", "r").startswith(entry.artifact_prefix)

    def test_a_fault_record_for_the_drill_is_refused(self, store) -> None:
        with pytest.raises(FaultRecordRefusedError, match="graded from its producer's own"):
            record_fault(
                None,  # type: ignore[arg-type] - refused before the context is touched
                store,
                fault_id=FIRE_DRILL_FAULT,
                outcome="induced",
                target_job=TRADER_FIRE_DRILL_JOB,
                trading_day="2026-09-08",
                run_id=_run_id(1),
                bus_key="alerts/2026-09-08/x.json",
            )
