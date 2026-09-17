"""The phase-3 clause `v1_arms_carried_or_excluded` (`alpha-engine-config-I10716`,
`-I10960`, `-I10961`, `-I10962`, `-I10963`, `-I10964`).

The class fix for `-I10714`: nothing compared v2's registers to v1's live arm
set, so an arm v1 serves that was never registered in v2 was invisible on
every surface. That fix covered ARMS. This suite covers the same question on
every dimension `crucible.carryover.DIMENSIONS` declares — arms, CHAMPIONS,
tuned PARAMETERS — plus the one a carried arm can still fail: whether it has
ever produced.

The tests that matter most are the MUTATIONS — each plants exactly the defect
the clause exists to catch into an otherwise-MET world and asserts the reading
moves:

* a live v1 arm, CHAMPION or tuned PARAMETER with no ledger row -> UNMET naming it;
* a `pending` row -> UNMET naming it;
* a `carried` row whose arm is not in the v2 register -> UNMET naming it;
* an `imported` champion row the live v2 pointer does not name -> UNMET;
* a `deferred` champion row whose condition has CLEARED -> UNMET;
* a `carried` arm registered past its settle window that has produced
  nothing -> UNMET; the same arm inside the window -> not that finding;
* a `carried` parameter v1 has since re-tuned -> UNMET;
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
    V1_BUCKET_VAR,
    V1_EXECUTOR_PARAMS_KEY,
    V1_FACTOR_WEIGHTS_KEY,
    V1_MODEL_ARENA_KEY,
    V1_PRODUCER_ARENA_KEY,
    V1_PRODUCER_CHAMPION_KEY,
    V1_S_SERVING_KEY,
    V1_SCANNER_CUT_KEY,
    V1_SCANNER_PARAMS_KEY,
    V1_SCANNER_SPEC_KEY,
    V1_ZOO_LEADERBOARD_KEY,
    LedgerError,
    parse_ledger,
)
from crucible.gate import _clause_v1_arms_carried_or_excluded as clause_fn
from crucible.gate import phase_tracker, weekly_window
from crucible.keys import arm_register_key, champion_key, shadow_key, v1_carryover_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 9, 11)
NAME = "v1_arms_carried_or_excluded"

#: Long before FRIDAY's 21-session settle window opens: an arm registered here
#: has been asked to produce.
SETTLED = "2026-06-01"
#: Inside it: an arm registered here has not failed to produce, it has not
#: been asked.
FRESH = "2026-09-08"

#: Any well-formed reference serves a fixture; the phase-3 tracker is derived
#: rather than typed (`tests/test_no_stale_tracker_literals.py`).
REF = phase_tracker("phase3")


def _window() -> list[dt.date]:
    return weekly_window(FRIDAY, 4)


def _put_json(store: LocalStore, key: str, document: object) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


def _seed_v1(store: LocalStore) -> None:
    """v1's artifacts in their live shapes (measured 2026-09-14 and
    2026-09-17), trimmed to the fields the readers name."""
    _put_json(
        store,
        V1_ZOO_LEADERBOARD_KEY,
        {
            "candidates": [{"spec_id": "residual-momentum"}, {"spec_id": "champion-arch"}],
            "slot_register": {
                "arms": [{"spec_id": "residual-momentum"}, {"spec_id": "horizon-60d"}]
            },
            "champion_arch": {"version_id": "v3.0-meta-2026-09-11-a214ae0a"},
            "serving_champion": {"served_version": "v3.0-meta-2026-08-14-119e069b"},
            # v1's own `champion` key is a METRICS block, not a name. Seeded
            # because a reader that sniffed for `champion` would read it as an
            # arm id (`alpha-engine-config-I10961`).
            "champion": {"forward_days": 21, "cpcv_mean_ic": 0.105001},
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
    # `config/executor_params.json` is BOTH the S slot's serving proof and a
    # tuned parameter document: one key, two readings.
    _put_json(store, V1_S_SERVING_KEY, {"min_score": 75})
    _put_json(store, V1_SCANNER_PARAMS_KEY, {"tech_score_min": 40, "updated_at": "2026-08-22"})
    _put_json(store, V1_FACTOR_WEIGHTS_KEY, {"weights": {"momentum": 1.0}})


def _arm_rows() -> list[dict]:
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


def _champion_rows() -> list[dict]:
    return [
        {
            "v1_slot": "model_zoo",
            "v1_champion": "champion-arch",
            "disposition": "imported",
            "v2_arm": "base_model",
        },
        {
            "v1_slot": "producer",
            "v1_champion": "no_agent_quant",
            "disposition": "imported",
            "v2_arm": "no_agent_quant",
            "v2_slot": "r",
        },
        {
            "v1_slot": "scanner_spec",
            "v1_champion": "momentum_sleeve",
            "disposition": "imported",
            "v2_arm": "momentum_sleeve",
        },
        {
            # v1's SECOND universe champion. v2 has one U pointer for two v1
            # champions, and which one holds it is a decision with a record
            # rather than an artifact of the import's source list (`-I10962`).
            "v1_slot": "scanner_cut",
            "v1_champion": "attractiveness_top_60",
            "disposition": "excluded",
            "reason": "v2's U slot serves the candidate-ranking decision; the cut is a width",
            "reference": REF,
        },
        {
            "v1_slot": "strategy",
            "v1_champion": "stock_registry",
            "disposition": "deferred",
            "v2_arm": "stock_registry",
            "reason": "S cannot grade before an M champion exists",
            "reference": REF,
        },
    ]


def _parameter_rows() -> list[dict]:
    return [
        {
            "v1_key": V1_SCANNER_PARAMS_KEY,
            "v1_param": "tech_score_min",
            "v1_value": "40",
            "disposition": "excluded",
            "reason": "v2's universe applies one liquidity floor",
            "reference": REF,
        },
        {
            "v1_key": V1_SCANNER_PARAMS_KEY,
            "v1_param": "updated_at",
            "v1_value": '"2026-08-22"',
            "disposition": "excluded",
            "reason": "provenance stamped by v1's optimizer, not a tuned value",
            "reference": REF,
        },
        {
            "v1_key": V1_EXECUTOR_PARAMS_KEY,
            "v1_param": "min_score",
            "v1_value": "75",
            "disposition": "deferred",
            "reason": "the v2 trader is unbuilt, so this value has no destination yet",
            "reference": REF,
        },
        {
            "v1_key": V1_FACTOR_WEIGHTS_KEY,
            "v1_param": "weights.momentum",
            "v1_value": "1.0",
            "disposition": "carried",
            "v2_home": "strategy/arms/u/attractiveness.yaml:momentum_weight",
        },
    ]


def _put_ledger(
    store: LocalStore,
    arms: list[dict] | None = None,
    champions: list[dict] | None = None,
    parameters: list[dict] | None = None,
) -> None:
    document = {
        "schema_version": carryover.LEDGER_SCHEMA_VERSION,
        "arms": _arm_rows() if arms is None else arms,
        "champions": _champion_rows() if champions is None else champions,
        "parameters": _parameter_rows() if parameters is None else parameters,
    }
    store.put_bytes(v1_carryover_key(), yaml.safe_dump(document).encode("utf-8"))


def _arm_id(slot: str, name: str) -> str:
    return f"{slot}:{name}:abc123"


def _put_register(store: LocalStore, slot: str, names: list[str], *, date: str = SETTLED) -> None:
    lines = []
    for name in names:
        arm_id = _arm_id(slot, name)
        lines.append(
            json.dumps(
                {
                    "kind": "registered",
                    "arm_id": arm_id,
                    "date": date,
                    "reason": "",
                    "record": {
                        "arm_id": arm_id,
                        "slot": slot,
                        "name": name,
                        "spec_hash": "abc123",
                        "created_date": date,
                    },
                },
                sort_keys=True,
            )
        )
    store.put_bytes(arm_register_key(slot), ("\n".join(lines) + "\n").encode("utf-8"))


def _put_produced(store: LocalStore, slot: str, name: str) -> None:
    _put_json(store, shadow_key(_arm_id(slot, name), "2026-09-04"), {"names": []})


def _put_champion(store: LocalStore, slot: str, name: str) -> None:
    _put_json(store, champion_key(slot), {"arm_id": _arm_id(slot, name)})


@pytest.fixture
def v1(tmp_path: Path) -> LocalStore:
    store = LocalStore(tmp_path / "v1")
    _seed_v1(store)
    return store


@pytest.fixture
def v2(tmp_path: Path) -> LocalStore:
    """An otherwise-MET world.

    U, R and M hold an imported pointer and arms that have produced. S holds a
    register filed INSIDE the settle window and no pointer — which is the live
    state and is correct: S cannot grade before an M champion exists, so its
    champion row is a `deferred` whose condition has not cleared.
    """
    store = LocalStore(tmp_path / "v2")
    _put_ledger(store)
    _put_register(store, "m", ["residual_momentum", "base_model"])
    _put_register(store, "r", ["no_agent_quant"])
    _put_register(store, "u", ["momentum_sleeve", "attractiveness_top_60"])
    _put_register(store, "s", ["stock_registry"], date=FRESH)
    for slot, name in (
        ("m", "residual_momentum"),
        ("m", "base_model"),
        ("r", "no_agent_quant"),
        ("u", "momentum_sleeve"),
        ("u", "attractiveness_top_60"),
    ):
        _put_produced(store, slot, name)
    _put_champion(store, "m", "base_model")
    _put_champion(store, "r", "no_agent_quant")
    _put_champion(store, "u", "momentum_sleeve")
    return store


def _read(v2: LocalStore, v1: LocalStore):
    return clause_fn(v2, _window(), v1_store=v1)


def test_a_complete_ledger_over_every_dimension_is_met(v2: LocalStore, v1: LocalStore) -> None:
    clause = _read(v2, v1)
    assert clause.name == NAME
    assert clause.met and not clause.unmeasurable, clause.detail
    assert v1_carryover_key() in clause.evidence
    assert f"v1:{V1_SCANNER_CUT_KEY}" in clause.evidence
    assert f"v1:{V1_SCANNER_PARAMS_KEY}" in clause.evidence
    assert champion_key("u") in clause.evidence
    for dimension in ("arms", "champions", "parameters"):
        assert dimension in clause.detail


class TestArmMutations:
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
        rows = _arm_rows()
        rows[1] = {
            "v1_slot": "model_zoo",
            "v1_arm": "champion-arch",
            "disposition": "pending",
            "reference": REF,
        }
        _put_ledger(v2, arms=rows)
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert f"model_zoo/champion-arch ({REF})" in clause.detail
        assert "undecided" in clause.detail

    def test_a_carried_arm_missing_from_the_register_is_unmet(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        _put_register(v2, "m", ["residual_momentum"])
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "model_zoo/champion-arch -> base_model" in clause.detail

    def test_an_absent_register_holds_no_carried_arm_and_is_reported_once(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        """`s:stock_registry` has no register until S first grades. It trips
        the register condition and must NOT also be reported as mute — one
        gap, one finding (`-I10964` gotcha)."""
        (Path(v2.root) / arm_register_key("s")).unlink()
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "strategy/stock_registry -> stock_registry" in clause.detail
        assert "does not exist" in clause.detail
        assert "MUTE" not in clause.detail
        assert "the deferral" not in clause.detail, (
            "the champion row for the same slot is a deferral whose condition plainly holds: "
            "an arm with no register row has no arm id, so no artifact can exist under one"
        )

    def test_an_absent_ledger_is_unmet(self, v2: LocalStore, v1: LocalStore) -> None:
        (Path(v2.root) / v1_carryover_key()).unlink()
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "absent" in clause.detail

    def test_a_malformed_ledger_is_unmet_naming_the_violation(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        rows = _arm_rows()
        rows.append(dict(rows[0]))
        _put_ledger(v2, arms=rows)
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "more than one row" in clause.detail


class TestProductionIsNotRegistration:
    """`-I10964`: a `carried` row was satisfied by REGISTRATION, so an arm that
    registers and can never emit read MET. The live instance is
    `m:v3meta_stack` — the port of v1's serving model, registered 2026-09-17
    and unable to produce on any date (`-I10947`)."""

    def test_a_registered_arm_that_has_produced_nothing_is_unmet(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        (Path(v2.root) / shadow_key(_arm_id("m", "base_model"), "2026-09-04")).unlink()
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "carried-unproduced" in clause.detail
        assert "model_zoo/champion-arch" in clause.detail
        assert "MUTE" in clause.detail

    def test_the_same_arm_inside_the_settle_window_is_not_that_finding(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        """An arm registered three sessions ago has not FAILED to produce; it
        has not been asked. The window is the slot's own horizon constant."""
        (Path(v2.root) / shadow_key(_arm_id("m", "base_model"), "2026-09-04")).unlink()
        (Path(v2.root) / shadow_key(_arm_id("m", "residual_momentum"), "2026-09-04")).unlink()
        _put_register(v2, "m", ["residual_momentum", "base_model"], date=FRESH)
        clause = _read(v2, v1)
        assert clause.met, clause.detail

    def test_an_excluded_arm_that_is_mute_does_not_trip_it(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        """`r:thinktank_coverage` is registered and refuses by name until
        phase 5. It is an `excluded` row, and an exclusion is a decision, not
        a carry that has to emit."""
        document = json.loads(v1.get_bytes(V1_PRODUCER_ARENA_KEY))
        document["active_arms"].append("producer:thinktank_coverage:111111111111")
        _put_json(v1, V1_PRODUCER_ARENA_KEY, document)
        _put_register(v2, "r", ["no_agent_quant", "thinktank_coverage"])
        _put_produced(v2, "r", "no_agent_quant")
        rows = _arm_rows()
        rows.append(
            {
                "v1_slot": "producer",
                "v1_arm": "thinktank_coverage",
                "disposition": "excluded",
                "reason": "LLM arm; refuses by name until phase 5",
                "reference": REF,
            }
        )
        _put_ledger(v2, arms=rows)
        clause = _read(v2, v1)
        assert clause.met, clause.detail

    def test_an_unreadable_production_listing_is_never_a_pass(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        class Denied(LocalStore):
            def list_keys(self, prefix: str):
                raise PermissionError("AccessDenied")

        clause = clause_fn(Denied(v2.root), _window(), v1_store=v1)
        assert not clause.met
        assert "could not be listed" in clause.detail


class TestChampionMutations:
    """`-I10960`: the ledger had no champion row shape at all, and no clause
    compared v1's champion POINTERS to v2's. 2 of v1's 5 live champions had
    crossed and nothing said so."""

    def test_a_v1_champion_with_no_ledger_row_is_unmet_and_named(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        rows = [r for r in _champion_rows() if r["v1_slot"] != "producer"]
        _put_ledger(v2, champions=rows)
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "champions" in clause.detail
        assert "producer/no_agent_quant" in clause.detail
        assert "no ledger row" in clause.detail

    def test_an_imported_row_the_live_pointer_does_not_name_is_unmet(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        """The U seat was held by whichever v1 key the import happened to read
        (`-I10962`). A row claiming a seat the pointer does not hold is now a
        finding rather than a silent disagreement."""
        _put_champion(v2, "u", "attractiveness_top_60")
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "points at u:attractiveness_top_60, not u:momentum_sleeve" in clause.detail

    def test_an_imported_row_with_no_pointer_at_all_is_unmet(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        (Path(v2.root) / champion_key("m")).unlink()
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert f"{champion_key('m')} is absent" in clause.detail

    def test_a_deferral_holds_while_its_arm_is_mute(self, v2: LocalStore, v1: LocalStore) -> None:
        assert _read(v2, v1).met, "S's deferred champion row is the live, correct state"

    def test_a_deferral_that_has_cleared_is_unmet(self, v2: LocalStore, v1: LocalStore) -> None:
        """`-I10961`: `migrate.history` ran `slots=("u","r")` and the deferral
        had NO trigger. Once the deferred arm is registered and producing and
        the pointer is still absent, the import is outstanding work and this
        goes red."""
        _put_register(v2, "s", ["stock_registry"])
        _put_produced(v2, "s", "stock_registry")
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "the deferral has CLEARED" in clause.detail
        assert "migrate.history" in clause.detail


class TestParameterMutations:
    """`-I10963`: v1's live tuned parameter documents had no row shape and no
    disposition — 7 of 8 scanner gates and all of `executor_params.json`."""

    def test_a_tuned_parameter_with_no_ledger_row_is_unmet_and_named(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        document = json.loads(v1.get_bytes(V1_SCANNER_PARAMS_KEY))
        document["max_atr_pct"] = 8.0
        _put_json(v1, V1_SCANNER_PARAMS_KEY, document)
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert f"{V1_SCANNER_PARAMS_KEY}/max_atr_pct" in clause.detail

    def test_a_nested_parameter_is_one_row_not_one_blob(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        _put_json(v1, V1_FACTOR_WEIGHTS_KEY, {"weights": {"momentum": 1.0, "value": 1.0}})
        clause = _read(v2, v1)
        assert not clause.met
        assert f"{V1_FACTOR_WEIGHTS_KEY}/weights.value" in clause.detail

    def test_a_carried_value_v1_has_re_tuned_is_unmet(self, v2: LocalStore, v1: LocalStore) -> None:
        _put_json(v1, V1_FACTOR_WEIGHTS_KEY, {"weights": {"momentum": 0.0}})
        clause = _read(v2, v1)
        assert not clause.met and not clause.unmeasurable
        assert "value-drifted" in clause.detail
        assert "re-tuned" in clause.detail


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
            V1_SCANNER_PARAMS_KEY,
            V1_FACTOR_WEIGHTS_KEY,
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

    def test_a_zoo_leaderboard_with_no_serving_champion_is_the_wrong_shape(
        self, v2: LocalStore, v1: LocalStore
    ) -> None:
        document = json.loads(v1.get_bytes(V1_ZOO_LEADERBOARD_KEY))
        del document["serving_champion"]
        _put_json(v1, V1_ZOO_LEADERBOARD_KEY, document)
        clause = _read(v2, v1)
        assert clause.unmeasurable and not clause.met
        assert "serving_champion" in clause.detail

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
    def _raw(
        self,
        arms: list[dict] | None = None,
        champions: list[dict] | None = None,
        parameters: list[dict] | None = None,
        version: str = carryover.LEDGER_SCHEMA_VERSION,
    ) -> bytes:
        document: dict = {"schema_version": version, "arms": _arm_rows() if arms is None else arms}
        if champions is not None or version == carryover.LEDGER_SCHEMA_VERSION:
            document["champions"] = _champion_rows() if champions is None else champions
        if parameters is not None or version == carryover.LEDGER_SCHEMA_VERSION:
            document["parameters"] = _parameter_rows() if parameters is None else parameters
        return yaml.safe_dump(document).encode()

    def test_the_seeded_rows_parse(self) -> None:
        rows = parse_ledger(self._raw(), source="x")
        assert len(rows) == len(_arm_rows()) + len(_champion_rows()) + len(_parameter_rows())
        assert {r.dimension for r in rows} == {"arms", "champions", "parameters"}

    def test_a_published_v1_document_still_parses(self) -> None:
        """The store holds a published `v1_carryover.v1` document. A reader
        that refused it would turn the clause into a parse error on the day
        the schema bumped, and say nothing about the carry-over at all."""
        rows = parse_ledger(self._raw(version="v1_carryover.v1"), source="x")
        assert {r.dimension for r in rows} == {"arms"}

    @pytest.mark.parametrize(
        ("section", "row", "message"),
        [
            (
                "arms",
                {"v1_slot": "nope", "v1_arm": "a", "disposition": "carried", "v2_arm": "a"},
                "v1_slot",
            ),
            ("arms", {"v1_slot": "producer", "v1_arm": "a", "disposition": "maybe"}, "disposition"),
            ("arms", {"v1_slot": "producer", "v1_arm": "a", "disposition": "carried"}, "v2_arm"),
            (
                "arms",
                {"v1_slot": "producer", "v1_arm": "a", "disposition": "excluded", "reason": "r"},
                "reference",
            ),
            (
                "arms",
                {
                    "v1_slot": "producer",
                    "v1_arm": "a",
                    "disposition": "excluded",
                    "reference": "alpha-engine-config-I1",
                },
                "reason",
            ),
            ("arms", {"v1_slot": "producer", "v1_arm": "a", "disposition": "pending"}, "reference"),
            (
                "arms",
                {
                    "v1_slot": "producer",
                    "v1_arm": "a",
                    "disposition": "pending",
                    "reference": "I10715",
                },
                "does not match",
            ),
            (
                "arms",
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
                "arms",
                {
                    "v1_slot": "producer",
                    "v1_arm": "a",
                    "disposition": "carried",
                    "v2_arm": "a",
                    "extra": 1,
                },
                "unknown field",
            ),
            (
                "champions",
                {"v1_slot": "producer", "v1_champion": "a", "disposition": "carried"},
                "disposition",
            ),
            (
                "champions",
                {"v1_slot": "producer", "v1_champion": "a", "disposition": "imported"},
                "v2_arm",
            ),
            (
                "champions",
                {
                    "v1_slot": "producer",
                    "v1_champion": "a",
                    "disposition": "deferred",
                    "v2_arm": "a",
                    "reason": "waiting",
                },
                "reference",
            ),
            (
                "champions",
                {
                    "v1_slot": "producer",
                    "v1_champion": "a",
                    "disposition": "imported",
                    "v2_arm": "a",
                    "v2_slot": "u",
                },
                "is not 'r'",
            ),
            (
                "parameters",
                {
                    "v1_key": "config/nope.json",
                    "v1_param": "a",
                    "v1_value": "1",
                    "disposition": "carried",
                    "v2_home": "x",
                },
                "v1_key",
            ),
            (
                "parameters",
                {
                    "v1_key": V1_SCANNER_PARAMS_KEY,
                    "v1_param": "a",
                    "disposition": "carried",
                    "v2_home": "x",
                },
                "v1_value",
            ),
            (
                "parameters",
                {
                    "v1_key": V1_SCANNER_PARAMS_KEY,
                    "v1_param": "a",
                    "v1_value": "1",
                    "disposition": "carried",
                },
                "v2_home",
            ),
        ],
    )
    def test_a_violating_row_is_refused(self, section: str, row: dict, message: str) -> None:
        kwargs = {section: [row]}
        with pytest.raises(LedgerError, match=message):
            parse_ledger(self._raw(**kwargs), source="x")

    def test_a_wrong_schema_version_is_refused(self) -> None:
        with pytest.raises(LedgerError, match="schema_version"):
            parse_ledger(self._raw(version="v0"), source="x")

    def test_an_unknown_section_is_refused(self) -> None:
        document = yaml.safe_load(self._raw().decode())
        document["parametres"] = []
        with pytest.raises(LedgerError, match="unknown top-level section"):
            parse_ledger(yaml.safe_dump(document).encode(), source="x")
