"""The champion-consumption contract — the one thing the trader reads.

Normative source: plan §3 ("the contract is the only coupling"), §9.1
(contamination attestation), `champion-challenger-policy.md` §11.

The trader is a separate system. Everything it knows about the harness comes
through `champions/{slot}/current.json`, so the refusals live on the READER:
a consumer that trusted whatever the producer last wrote would have no way
to tell a champion decided by a healthy cycle from one written by a run that
died halfway.

Two refusals, both measured defects rather than hypotheticals:

* **The producing run must be `ok`.** A pointer written by a run that then
  failed is the "record asserting an action that never happened" bug class
  (policy §7.2).
* **An S-slot champion needs a `pit_parity` PASS.** Plan §9.1: "a card
  without `attestation: PASS` renders UNVERIFIED, never a grade" — and a
  pointer is a card the trader acts on with money.
"""

from __future__ import annotations

import json

import pytest

from crucible.champion import (
    CHAMPION_SCHEMA_VERSION,
    ChampionPointer,
    ChampionUnusableError,
    champion_key,
    read_champion,
    write_champion,
)
from crucible.store import LocalStore

DAY = "2026-08-28"


def _manifest(status: str, job: str = "promote") -> bytes:
    return json.dumps({"status": status, "job": job, "trading_day": DAY, "reason": ""}).encode()


def _pointer(slot: str = "m", **over) -> ChampionPointer:
    payload = dict(
        schema_version=CHAMPION_SCHEMA_VERSION,
        slot=slot,
        arm_id=f"{slot}:champ:0123456789abcdef",
        as_of=DAY,
        decided_at="2026-08-29T02:00:00Z",
        run_id="01JG0000000000000000000000",
        code_sha="0" * 40,
        promotion_source="evidence",
        manifest_key=f"runs/promote/{DAY}/run.json",
        evidence={"status": "decided", "moved": True, "paired_dates": 40},
        attestation=None,
    )
    payload.update(over)
    return ChampionPointer(**payload)


def _store(tmp_path, pointer: ChampionPointer, manifest_status: str = "ok") -> LocalStore:
    store = LocalStore(tmp_path)
    write_champion(store, pointer)
    store.put_bytes(pointer.manifest_key, _manifest(manifest_status))
    return store


class TestKeyAndSchema:
    def test_the_key_is_the_one_contract_the_trader_reads(self) -> None:
        assert champion_key("m") == "champions/m/current.json"

    def test_a_written_pointer_round_trips(self, tmp_path) -> None:
        pointer = _pointer()
        store = _store(tmp_path, pointer)
        assert read_champion(store, "m") == pointer

    def test_an_unknown_schema_version_is_refused_not_guessed(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            champion_key("m"),
            json.dumps({**_pointer().to_dict(), "schema_version": "champion_pointer.v9"}).encode(),
        )
        with pytest.raises(ChampionUnusableError, match="schema_version"):
            read_champion(store, "m")


class TestManifestGate:
    def test_a_champion_from_a_failed_run_is_refused(self, tmp_path) -> None:
        store = _store(tmp_path, _pointer(), manifest_status="failed")
        with pytest.raises(ChampionUnusableError, match="status"):
            read_champion(store, "m")

    def test_a_champion_whose_manifest_is_missing_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        write_champion(store, _pointer())
        with pytest.raises(ChampionUnusableError, match="manifest"):
            read_champion(store, "m")


class TestAttestation:
    def test_an_s_champion_without_a_pit_parity_pass_is_refused(self, tmp_path) -> None:
        pointer = _pointer(
            slot="s",
            arm_id="s:stock_registry:0123456789abcdef",
            attestation={"kind": "pit_parity", "status": "PARTIAL", "key": "x", "reason": "y"},
        )
        store = _store(tmp_path, pointer)
        with pytest.raises(ChampionUnusableError, match="UNVERIFIED"):
            read_champion(store, "s")

    def test_an_s_champion_with_no_attestation_at_all_is_refused(self, tmp_path) -> None:
        pointer = _pointer(slot="s", arm_id="s:stock_registry:0123456789abcdef")
        store = _store(tmp_path, pointer)
        with pytest.raises(ChampionUnusableError, match="UNVERIFIED"):
            read_champion(store, "s")

    def test_an_s_champion_with_a_pass_is_served(self, tmp_path) -> None:
        pointer = _pointer(
            slot="s",
            arm_id="s:stock_registry:0123456789abcdef",
            attestation={
                "kind": "pit_parity",
                "status": "PASS",
                "key": f"attestations/s/{DAY}/pit_parity.json",
                "reason": "delta not distinguishable from zero",
            },
        )
        store = _store(tmp_path, pointer)
        assert read_champion(store, "s").arm_id == pointer.arm_id


class TestProvenance:
    def test_promotion_source_is_carried_so_a_bootstrap_is_visible_as_one(self, tmp_path) -> None:
        """Policy §11: 'a pointer that has never moved on evidence is a
        finding, not a stable system'. R's champion has read
        `operator_bootstrap` since 2026-07-13 and no surface said so."""
        store = _store(tmp_path, _pointer(promotion_source="operator_bootstrap"))
        assert read_champion(store, "m").promotion_source == "operator_bootstrap"

    def test_an_unknown_promotion_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="promotion_source"):
            _pointer(promotion_source="because_i_said_so")
