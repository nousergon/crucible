"""The phase-3 clause `v1_arms_carried_or_excluded` (`alpha-engine-config-I10716`).

The class fix for `-I10714`: nothing compared v2's registers to v1's live arm
set, so an arm v1 serves that was never registered in v2 was invisible on
every surface. The tests that matter most are the MUTATIONS — each plants
exactly the defect the clause exists to catch into an otherwise-MET world and
asserts the reading moves:

* a live v1 arm with no ledger row -> UNMET naming it;
* a `pending` row -> UNMET naming it;
* a `carried` row whose arm is not in the v2 register -> UNMET naming it;
* any v1 source unreadable (absent, malformed, denied, location unset) ->
  UNMEASURABLE, never MET.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
import yaml

from crucible import carryover
from crucible.carryover import (
    V1_MODEL_ARENA_KEY,
    V1_PRODUCER_ARENA_KEY,
    V1_PRODUCER_CHAMPION_KEY,
    V1_S_SERVING_KEY,
    V1_SCANNER_CUT_KEY,
    V1_SCANNER_SPEC_KEY,
    V1_BUCKET_VAR,
    V1_ZOO_LEADERBOARD_KEY,
    LedgerError,
    parse_ledger,
)
from crucible.gate import _clause_v1_arms_carried_or_excluded as clause_fn
from crucible.gate import phase_tracker, weekly_window
from crucible.keys import arm_register_key, v1_carryover_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 9, 11)
NAME = "v1_arms_carried_or_excluded"

#: Any well-formed reference serves a fixture; the phase-3 tracker is derived
#: rather than typed (`tests/test_no_stale_tracker_literals.py`).
REF = phase_tracker("phase3")


def _window() -> list[dt.date]:
    return weekly_window(FRIDAY, 4)


def _put_json(store: LocalStore, key: str, document: object) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


def _seed_v1(store: LocalStore) -> None:
    """v1's artifacts in their live shapes (measured 2026-09-14), trimmed."""
    _put_json(
        store,
        V1_ZOO_LEADERBOARD_KEY,
        {
            "candidates": [{"spec_id": "residual-momentum"}, {"spec_id": "champion-arch"}],
            "slot_register": {
                "arms": [{"spec_id": "residual-momentum"}, {"spec_id": "horizon-60d"}]
            },
            "champion_arch": {"version_id": "v3.0-meta-2026-09-11-a214ae0a"},
        },
    )
    _put_json(
        store,
        V1_MODEL_ARENA_KEY,
        {"active_arms": ["M:champion-arch:4db81ad0d630", "M:residual-momentum:3ccf955a9c72"]},
    )
    _put_json(
        store,
        V1_PRODUCER_ARENA_KEY,
        {
            "active_arms": [
                "producer:no_agent_quant:979910811b04",
                "producer:single_agent_quant:862d0e3e9356",
            ]
        },
    )
    _put_json(store, V1_PRODUCER_CHAMPION_KEY, {"champion": "no_agent_quant"})
    _put_json(
        store,
        V1_SCANNER_SPEC_KEY,
        {"champion": "momentum_sleeve", "arms": {"momentum_sleeve": {}}},
    )
    _put_json(
        store,
        V1_SCANNER_CUT_KEY,
        {
            "champion": "attractiveness_top_60",
            "arms": {"attractiveness_top_60": {}},
            "arena": {"active_arms": ["universe_cut:attractiveness_top_60:90aa88291c29"]},
        },
    )
    _put_json(store, V1_S_SERVING_KEY, {"min_score": 75})


def _ledger_rows() -> list[dict]:
    return [
        {
            "v1_slot": "model_zoo",
            "v1_arm": "residual-momentum",
            "disposition": "carried",
            "v2_arm": "residual_momentum",
        },
        {
            "v1_slot": "model_zoo",
            "v1_arm": "champion-arch",
            "disposition": "carried",
            "v2_arm": "base_model",
        },
        {
            "v1_slot": "model_zoo",
            "v1_arm": "horizon-60d",
            "disposition": "excluded",
            "reason": "non-canonical horizon",
            "reference": REF,
        },
        {
            "v1_slot": "producer",
            "v1_arm": "no_agent_quant",
            "disposition": "carried",
            "v2_arm": "no_agent_quant",
        },
        {
            "v1_slot": "producer",
            "v1_arm": "single_agent_quant",
            "disposition": "excluded",
            "reason": "LLM arm, phase 5",
            "reference": REF,
        },
        {
            "v1_slot": "scanner_spec",
            "v1_arm": "momentum_sleeve",
            "disposition": "carried",
            "v2_arm": "momentum_sleeve",
        },
        {
            "v1_slot": "scanner_cut",
            "v1_arm": "attractiveness_top_60",
            "disposition": "carried",
            "v2_arm": "attractiveness_top_60",
        },
        {
            "v1_slot": "strategy",
            "v1_arm": "stock_registry",
            "disposition": "carried",
            "v2_arm": "stock_registry",
        },
    ]


def _put_ledger(store: LocalStore, rows: list[dict]) -> None:
    document = {"schema_version": carryover.LEDGER_SCHEMA_VERSION, "arms": rows}
    store.put_bytes(v1_carryover_key(), yaml.safe_dump(document).encode("utf-8"))


def _put_register(store: LocalStore, slot: str, names: list[str]) -> None:
    lines = []
    for name in names:
        arm_id = f"{slot}:{name}:abc123"
        lines.append(
            json.dumps(
                {
                    "kind": "registered",
                    "arm_id": arm_id,
                    "date": "2026-09-04",
                    "reason": "",
                    "record": {
                        "arm_id": arm_id,
                        "slot": slot,
                        "name": name,
                        "spec_hash": "abc123",
                        "created_date": "2026-09-04",
                    },
                },
                sort_keys=True,
            )
        )
    store.put_bytes(arm_register_key(slot), ("\n".join(lines) + "\n").encode("utf-8"))


@pytest.fixture
def v1(tmp_path: Path) -> LocalStore:
    store = LocalStore(tmp_path / "v1")
    _seed_v1(store)
    return store


@pytest.fixture
def v2(tmp_path: Path) -> LocalStore:
    store = LocalStore(tmp_path / "v2")
    _put_ledger(store, _ledger_rows())
    _put_register(store, "m", ["residual_momentum", "base_model"])
    _put_register(store, "r", ["no_agent_quant"])
    _put_register(store, "u", ["momentum_sleeve", "attractiveness_top_60"])
    _put_register(store, "s", ["stock_registry"])
    return store


def _read(v2: LocalStore, v1: LocalStore):
    return clause_fn(v2, _window(), v1_store=v1)


def test_a_complete_ledger_over_registered_arms_is_met(v2: LocalStore, v1: LocalStore) -> None:
    clause = _read(v2, v1)
    assert clause.name == NAME
    assert clause.met and not clause.unmeasurable, clause.detail
    assert v1_carryover_key() in clause.evidence
    assert f"v1:{V1_SCANNER_CUT_KEY}" in clause.evidence


class TestMutations:
    def test_an_unlisted_v1_arm_is_unmet_and_named(self, v2: LocalStore, v1: LocalStore) -> None:
        """THE mutation test: plant a live v1 arm with no ledger row."""
        document = json.loads(v1.get_bytes(V1_SCANNER_CUT_KEY))
        document["arms"]["tech_score_top_60"] = {}
        _put_json(v1, V1_SCANNER_CUT_KEY, document)
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "scanner_cut/tech_score_top_60" in clause.detail
        assert "no ledger row" in clause.detail

    def test_an_arm_only_in_a_v1_arena_is_still_seen(self, v2: LocalStore, v1: LocalStore) -> None:
        _put_json(
            v1,
            V1_PRODUCER_ARENA_KEY,
            {
                "active_arms": [
                    "producer:no_agent_quant:979910811b04",
                    "producer:single_agent_quant:862d0e3e9356",
                    "producer:brand_new_arm:000000000000",
                ]
            },
        )
        clause = _read(v2, v1)
        assert not clause.met and "producer/brand_new_arm" in clause.detail

    def test_a_pending_row_is_unmet_and_named(self, v2: LocalStore, v1: LocalStore) -> None:
        rows = _ledger_rows()
        rows[1] = {
            "v1_slot": "model_zoo",
            "v1_arm": "champion-arch",
            "disposition": "pending",
            "reference": REF,
        }
        _put_ledger(v2, rows)
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert f"model_zoo/champion-arch ({REF})" in clause.detail

    def test_a_carried_arm_missing_from_the_register_is_unmet(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        _put_register(v2, "m", ["residual_momentum"])
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "model_zoo/champion-arch -> m:base_model" in clause.detail

    def test_an_absent_register_holds_no_carried_arm(self, v2: LocalStore, v1: LocalStore) -> None:
        (Path(v2.root) / arm_register_key("s")).unlink()
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "strategy/stock_registry -> s:stock_registry" in clause.detail
        assert "does not exist" in clause.detail

    def test_an_absent_ledger_is_unmet(self, v2: LocalStore, v1: LocalStore) -> None:
        (Path(v2.root) / v1_carryover_key()).unlink()
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "absent" in clause.detail

    def test_a_malformed_ledger_is_unmet_naming_the_violation(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        rows = _ledger_rows()
        rows.append(dict(rows[0]))
        _put_ledger(v2, rows)
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "more than one row" in clause.detail


class TestUnreadableSourcesAreUnmeasurable:
    @pytest.mark.parametrize(
        "key",
        [
            V1_ZOO_LEADERBOARD_KEY,
            V1_MODEL_ARENA_KEY,
            V1_PRODUCER_ARENA_KEY,
            V1_PRODUCER_CHAMPION_KEY,
            V1_SCANNER_SPEC_KEY,
            V1_SCANNER_CUT_KEY,
            V1_S_SERVING_KEY,
        ],
    )
    def test_an_absent_v1_source(self, v2: LocalStore, v1: LocalStore, key: str) -> None:
        (Path(v1.root) / key).unlink()
        clause = _read(v2, v1)
        assert clause.unmeasurable and not clause.met
        assert key in clause.detail

    def test_a_v1_source_of_the_wrong_shape(self, v2: LocalStore, v1: LocalStore) -> None:
        _put_json(v1, V1_SCANNER_CUT_KEY, {"champion": "attractiveness_top_60", "arms": []})
        clause = _read(v2, v1)
        assert clause.unmeasurable and not clause.met
        assert "`arms`" in clause.detail

    def test_a_denied_v1_read(self, v2: LocalStore, v1: LocalStore) -> None:
        class Denied(LocalStore):
            def get_bytes(self, key: str) -> bytes:
                raise PermissionError("AccessDenied")

        clause = _read(v2, Denied(v1.root))
        assert clause.unmeasurable and not clause.met
        assert "AccessDenied" in clause.detail

    def test_an_unset_v1_location(self, v2: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(V1_BUCKET_VAR, raising=False)
        clause = clause_fn(v2, _window())
        assert clause.unmeasurable and not clause.met
        assert V1_BUCKET_VAR in clause.detail

    def test_the_location_is_read_from_the_environment(
        self, v2: LocalStore, v1: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import crucible.config as config

        opened: list[str] = []

        def fake_store_from_uri(uri: str) -> LocalStore:
            opened.append(uri)
            return v1

        monkeypatch.setenv(V1_BUCKET_VAR, "v1-data-bucket")
        monkeypatch.setattr(config, "store_from_uri", fake_store_from_uri)
        clause = clause_fn(v2, _window())
        assert opened == ["s3://v1-data-bucket"]
        assert clause.met, clause.detail


class TestLedgerSchema:
    def _raw(self, rows: list[dict], version: str = carryover.LEDGER_SCHEMA_VERSION) -> bytes:
        return yaml.safe_dump({"schema_version": version, "arms": rows}).encode()

    def test_the_seeded_rows_parse(self) -> None:
        assert len(parse_ledger(self._raw(_ledger_rows()), source="x")) == len(_ledger_rows())

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            (
                {"v1_slot": "nope", "v1_arm": "a", "disposition": "carried", "v2_arm": "a"},
                "v1_slot",
            ),
            ({"v1_slot": "producer", "v1_arm": "a", "disposition": "maybe"}, "disposition"),
            ({"v1_slot": "producer", "v1_arm": "a", "disposition": "carried"}, "v2_arm"),
            (
                {"v1_slot": "producer", "v1_arm": "a", "disposition": "excluded", "reason": "r"},
                "reference",
            ),
            (
                {
                    "v1_slot": "producer",
                    "v1_arm": "a",
                    "disposition": "excluded",
                    "reference": "alpha-engine-config-I1",
                },
                "reason",
            ),
            ({"v1_slot": "producer", "v1_arm": "a", "disposition": "pending"}, "reference"),
            (
                {
                    "v1_slot": "producer",
                    "v1_arm": "a",
                    "disposition": "pending",
                    "reference": "I10715",
                },
                "does not match",
            ),
            (
                {
                    "v1_slot": "producer",
                    "v1_arm": "a",
                    "disposition": "pending",
                    "reference": "alpha-engine-config-I1",
                    "v2_arm": "a",
                },
                "only meaningful",
            ),
            (
                {
                    "v1_slot": "producer",
                    "v1_arm": "a",
                    "disposition": "carried",
                    "v2_arm": "a",
                    "extra": 1,
                },
                "unknown field",
            ),
        ],
    )
    def test_a_violating_row_is_refused(self, row: dict, message: str) -> None:
        with pytest.raises(LedgerError, match=message):
            parse_ledger(self._raw([row]), source="x")

    def test_a_wrong_schema_version_is_refused(self) -> None:
        with pytest.raises(LedgerError, match="schema_version"):
            parse_ledger(self._raw(_ledger_rows(), version="v0"), source="x")
