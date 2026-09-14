"""`trader/release_pin` carries the smoke it was gated on, and the trader's keys
are declared single-source (`alpha-engine-config-I10649` deliverable 3,
`-I10650`; crucible-trader-PR7)."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from pydantic import ValidationError
from test_trader_release_pin import AFTER_CLOSE, SHA_A, _published, _trader_smoke_manifest

from crucible.keys import (
    TRADER_FIRE_DRILLS_PREFIX,
    TRADER_KILL_SWITCH_EVENTS_PREFIX,
    TRADER_KILL_SWITCH_KEY,
    TRADER_PAPER_SMOKE_PREFIX,
    TRADER_PIN_KEY,
    trader_fire_drill_key,
    trader_kill_switch_event_key,
    trader_paper_smoke_key,
)
from crucible.models import TraderReleasePinDocument
from crucible.release import (
    TraderSmokeEvidence,
    parse_trader_release_pin,
    pin,
    pin_trader,
    read_pointer,
)
from crucible.store import LocalStore

SMOKE_KEY = "runs/trader.smoke/2026-09-04/aaaaaaaaaaaa-21/run.json"


def _document(**overrides) -> dict:
    return {
        "sha": SHA_A,
        "target": "trader",
        "pinned_at": "2026-09-08T21:00:00Z",
        "smoke_run_id": "01JG2026090400000000000000",
        "smoke_status": "ok",
        "smoke_manifest_key": SMOKE_KEY,
    } | overrides


class TestModel:
    def test_round_trip(self) -> None:
        document = _document()
        model = TraderReleasePinDocument.model_validate(document)
        assert model.model_dump() == document
        assert TraderReleasePinDocument.model_validate_json(model.model_dump_json()) == model

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"target": "current"}, id="harness-target"),
            pytest.param({"smoke_status": "failed"}, id="failed-smoke"),
            pytest.param(
                {"smoke_manifest_key": "runs/smoke/2026-09-04/run.json"}, id="harness-smoke"
            ),
            pytest.param({"smoke_manifest_key": "trader/evidence.json"}, id="not-a-manifest"),
            pytest.param({"smoke_run_id": ""}, id="empty-run-id"),
            pytest.param({"extra": 1}, id="extra-field"),
        ],
    )
    def test_a_pin_that_pin_would_never_write_is_refused(self, overrides) -> None:
        with pytest.raises(ValidationError):
            TraderReleasePinDocument.model_validate(_document(**overrides))

    @pytest.mark.parametrize("missing", ["smoke_run_id", "smoke_status", "smoke_manifest_key"])
    def test_a_pin_without_its_evidence_is_refused(self, missing) -> None:
        document = _document()
        del document[missing]
        with pytest.raises(ValueError, match="does not conform to a trader release pin"):
            parse_trader_release_pin(TRADER_PIN_KEY, document)


class TestWriterReaderContract:
    def test_pin_trader_writes_the_evidence_it_was_gated_on(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        key, manifest = _trader_smoke_manifest(store, SHA_A)
        _, evidence = pin_trader(store, SHA_A, now=AFTER_CLOSE)
        written = json.loads(store.get_bytes(TRADER_PIN_KEY))
        model = TraderReleasePinDocument.model_validate(written)
        assert (model.smoke_run_id, model.smoke_status, model.smoke_manifest_key) == (
            manifest["run_id"],
            "ok",
            key,
        )
        assert evidence.manifest_key == key
        assert read_pointer(store, TRADER_PIN_KEY)[0] == SHA_A

    def test_a_three_field_trader_pin_is_refused_on_read(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            TRADER_PIN_KEY,
            json.dumps(
                {"sha": SHA_A, "target": "trader", "pinned_at": "2026-09-08T21:00:00Z"}
            ).encode(),
        )
        with pytest.raises(ValueError, match="trader release pin"):
            read_pointer(store, TRADER_PIN_KEY)

    def test_evidence_that_names_no_smoke_manifest_is_refused_before_the_write(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        bogus = TraderSmokeEvidence("k", "r", SHA_A, "ok", "2026-09-04", "2026-09-04T21:00:00Z")
        with pytest.raises(ValueError, match="trader release pin"):
            pin(store, SHA_A, target="trader", trader_smoke=bogus, now=dt.datetime.now(dt.UTC))
        assert not store.exists(TRADER_PIN_KEY)


class TestTraderKeys:
    """The literal shapes crucible-trader-PR7 writes, owned here from now on."""

    def test_the_shapes_are_literal(self) -> None:
        run_id = "01JG2026090800000000000000"
        assert (
            trader_paper_smoke_key("2026-09-08", SHA_A)
            == f"trader/paper_smoke/2026-09-08/{SHA_A}.json"
        )
        assert TRADER_KILL_SWITCH_KEY == "trader/kill_switch.json"
        assert (
            trader_kill_switch_event_key("2026-09-08", run_id)
            == f"trader/kill_switch/events/2026-09-08/{run_id}.json"
        )
        assert (
            trader_fire_drill_key("2026-09-08", run_id)
            == f"trader/fire_drills/2026-09-08/{run_id}.json"
        )

    def test_each_key_starts_with_its_prefix(self) -> None:
        assert trader_paper_smoke_key("2026-09-08", SHA_A).startswith(TRADER_PAPER_SMOKE_PREFIX)
        assert trader_kill_switch_event_key("2026-09-08", "r").startswith(
            TRADER_KILL_SWITCH_EVENTS_PREFIX
        )
        assert trader_fire_drill_key("2026-09-08", "r").startswith(TRADER_FIRE_DRILLS_PREFIX)

    def test_the_halt_document_is_not_under_the_events_prefix(self) -> None:
        assert not TRADER_KILL_SWITCH_KEY.startswith(TRADER_KILL_SWITCH_EVENTS_PREFIX)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: trader_paper_smoke_key("2026-09-08", "abc"),
            lambda: trader_paper_smoke_key("not-a-day", SHA_A),
            lambda: trader_kill_switch_event_key("2026-09-08", ""),
            lambda: trader_fire_drill_key("2026-09-08", "a/b"),
            lambda: trader_fire_drill_key("09/08/2026", "r"),
        ],
    )
    def test_a_malformed_part_raises(self, call) -> None:
        with pytest.raises(ValueError):
            call()
