"""Only an arm fitted to a SIGNED forward return may serve the trader.

`alpha-engine-config-I12121`. On 2026-10-07 `champions/m/current.json` named
`m:v3meta_volatility_head:97645d421bea`, whose recipe declares
`target: abs_forward_return` — an unsigned |return| magnitude — and
`serve.daily` republished its output as `predicted_alpha` in
`predictions/{day}.json`, which the trader sizes on. It was seated by an
operator revert (`status: operator_revert`, `promotion_source:
operator_bootstrap`), and nothing on either pointer writer, or on the feed,
read the target.

Asserted here, each against the shape that broke:

1. **The set** — :data:`crucible.slots.model.SIGNED_FORWARD_RETURN_TARGETS`
   names the slot default and the factor residual, and nothing unsigned.
2. **The promote write** — a cycle that names a magnitude head as a moved
   champion is refused loudly, naming the arm and its target, and the
   incumbent pointer is left as it was.
3. **The operator revert** — the same refusal on `promote --revert-to`, the
   path that actually seated the head.
4. **The feed** — publishing under such a champion raises and writes no
   `predictions/{day}.json`.
5. **The cross-section** — a one-sided batch of at least
   :data:`crucible.serving.MIN_NAMES_FOR_SIGN_CHECK` names is refused with a
   named reason, whatever the champion's target says.
"""

from __future__ import annotations

import json

import pytest
from nousergon_lib.arena import ArmRegister, ArmSeries

from crucible.champion import (
    CHAMPION_SCHEMA_VERSION,
    ChampionPointer,
    champion_key,
    read_champion_etag,
    write_champion,
)
from crucible.keys import (
    arm_predictions_key,
    experiments_key,
    predictions_key,
    strategy_arm_key,
)
from crucible.promote import PromotionRefused, revert_champion, run_promotion
from crucible.serving import (
    MIN_NAMES_FOR_SIGN_CHECK,
    OneSidedCrossSectionError,
    publish_predictions_feed,
    read_predictions_feed,
)
from crucible.slots import get_slot
from crucible.slots.inputs import ARM_PREDICTIONS_SCHEMA_VERSION
from crucible.slots.model import (
    DEFAULT_TARGET,
    FACTOR_RESIDUAL_TARGET,
    SIGNED_FORWARD_RETURN_TARGETS,
    TARGETS,
    UnsignedChampionTargetError,
    require_signed_target,
)
from crucible.store import LocalStore
from tests.support.panels import trading_days
from tests.support.servable import (
    m_recipe,
    seed_arm_predictions,
    seed_m_recipe,
)

CODE_SHA = "a" * 40
SEAT_MANIFEST = "runs/promote/seed/m/run.json"
UNSIGNED = "abs_forward_return"

#: name -> declared target. `vol_head` is the 2026-10-07 shape.
_ARMS: dict[str, str | None] = {"champ": None, "vol_head": UNSIGNED}


def _register(dates: list[str]) -> tuple[ArmRegister, dict[str, str]]:
    register = ArmRegister()
    ids: dict[str, str] = {}
    for name, target in _ARMS.items():
        register, record = register.register(
            slot="m",
            name=name,
            spec=m_recipe(name, target=target).spec,
            created_date=dates[0],
            filed_on=dates[0],
        )
        ids[name] = record.arm_id
    return register, ids


def _seat(store: LocalStore, arm_id: str, as_of: str) -> bytes:
    store.put_bytes(SEAT_MANIFEST, json.dumps({"status": "ok"}).encode())
    write_champion(
        store,
        ChampionPointer(
            schema_version=CHAMPION_SCHEMA_VERSION,
            slot="m",
            arm_id=arm_id,
            as_of=as_of,
            decided_at="2026-09-01T02:00:00Z",
            run_id="01JG0000000000000000000000",
            code_sha=CODE_SHA,
            promotion_source="evidence",
            manifest_key=SEAT_MANIFEST,
            evidence={"status": "decided", "moved": True},
        ),
        expected=read_champion_etag(store, "m"),
    )
    return store.get_bytes(champion_key("m"))


def _cross_section(store: LocalStore, arm_id: str, day: str, alpha: dict[str, float]) -> None:
    store.put_bytes(
        arm_predictions_key(arm_id, day),
        json.dumps(
            {
                "schema_version": ARM_PREDICTIONS_SCHEMA_VERSION,
                "arm_id": arm_id,
                "trading_day": day,
                "feature_version": "features.v3",
                "predicted_alpha": alpha,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )


@pytest.fixture
def arena(tmp_path):
    """The signed incumbent seated and serving; the magnitude head filed,
    registered and servable for the session — everything but its target
    would let it serve."""
    store = LocalStore(tmp_path)
    dates = trading_days(40)
    register, ids = _register(dates)
    for name, target in _ARMS.items():
        seed_m_recipe(store, name, target=target)
    as_of = dates[-1]
    pointer_bytes = _seat(store, ids["champ"], as_of)
    seed_arm_predictions(store, ids["champ"], as_of)
    seed_arm_predictions(store, ids["vol_head"], as_of)
    publish_predictions_feed(store, trading_day=as_of)
    return store, dates, register, ids, as_of, pointer_bytes


class TestTheSignedSetIsExplicit:
    def test_it_is_the_default_and_the_factor_residual_and_nothing_else(self) -> None:
        assert DEFAULT_TARGET == "forward_return"
        assert SIGNED_FORWARD_RETURN_TARGETS == {"forward_return", FACTOR_RESIDUAL_TARGET}

    def test_abs_forward_return_is_a_declarable_target_and_not_a_signed_one(self) -> None:
        assert UNSIGNED in TARGETS
        assert UNSIGNED not in SIGNED_FORWARD_RETURN_TARGETS

    def test_a_recipe_that_declares_no_target_is_signed(self) -> None:
        require_signed_target(m_recipe("plain"), action="test")

    def test_a_magnitude_head_is_refused_by_name_and_target(self) -> None:
        recipe = m_recipe("vol_head", target=UNSIGNED)
        with pytest.raises(UnsignedChampionTargetError) as refused:
            require_signed_target(recipe, action="seat it")
        assert recipe.arm_id in str(refused.value)
        assert UNSIGNED in str(refused.value)


class TestPromoteRefusesAnUnsignedChampion:
    def test_a_cycle_naming_the_magnitude_head_is_refused_and_the_pointer_holds(
        self, arena
    ) -> None:
        from nousergon_lib.arena.engine import run_cycle

        store, dates, register, ids, as_of, pointer_bytes = arena
        spec = get_slot("m")
        cycle = run_cycle(
            config=spec.arena,
            as_of=as_of,
            register=register,
            series_by_arm={
                ids["champ"]: ArmSeries(arm_id=ids["champ"], scores=dict.fromkeys(dates, 0.0)),
                ids["vol_head"]: ArmSeries(
                    arm_id=ids["vol_head"], scores=dict.fromkeys(dates, 0.045)
                ),
            },
            incumbent=ids["champ"],
        )
        assert cycle.decision.moved and cycle.decision.champion == ids["vol_head"]
        feed_bytes = store.get_bytes(predictions_key(as_of))

        with pytest.raises(PromotionRefused) as refused:
            run_promotion(spec=spec, register=register, cycle=cycle, store=store, code_sha=CODE_SHA)

        message = str(refused.value)
        assert ids["vol_head"] in message
        assert UNSIGNED in message
        assert "left unchanged" in message
        assert isinstance(refused.value.__cause__, UnsignedChampionTargetError)
        assert store.get_bytes(champion_key("m")) == pointer_bytes
        assert store.get_bytes(predictions_key(as_of)) == feed_bytes


class TestTheOperatorRevertRefusesAnUnsignedChampion:
    def _revert(self, store, register, arm_id, as_of):
        return revert_champion(
            spec=get_slot("m"),
            register=register,
            store=store,
            arm_id=arm_id,
            as_of=as_of,
            operator="cipher813",
            reason="restore the v1 volatility head",
            code_sha=CODE_SHA,
        )

    def test_the_2026_10_07_revert_is_refused_and_the_pointer_holds(self, arena) -> None:
        store, _, register, ids, as_of, pointer_bytes = arena
        with pytest.raises(PromotionRefused) as refused:
            self._revert(store, register, ids["vol_head"], as_of)

        message = str(refused.value)
        assert ids["vol_head"] in message
        assert UNSIGNED in message
        assert "operator revert" in message
        assert store.get_bytes(champion_key("m")) == pointer_bytes
        assert not store.exists(experiments_key(as_of)), "a refused revert records no event"

    def test_a_revert_to_an_arm_no_recipe_declares_is_refused(self, arena, tmp_path) -> None:
        """An arm whose target cannot be read cannot be shown to be signed."""
        store, _, register, ids, as_of, pointer_bytes = arena
        (tmp_path / strategy_arm_key("m", "vol_head")).unlink()
        with pytest.raises(PromotionRefused, match="declared by no M recipe"):
            self._revert(store, register, ids["vol_head"], as_of)
        assert store.get_bytes(champion_key("m")) == pointer_bytes

    def test_a_revert_to_a_signed_arm_still_works(self, arena) -> None:
        store, _, register, ids, as_of, _ = arena
        pointer = self._revert(store, register, ids["champ"], as_of)
        assert pointer.arm_id == ids["champ"]
        assert pointer.promotion_source == "operator_bootstrap"


class TestTheFeedRefusesAnUnsignedChampion:
    def test_no_feed_is_written_under_a_magnitude_head(self, tmp_path) -> None:
        """The live state: the pointer ALREADY names the head (seated before
        this guard existed), and the head produced a cross-section."""
        store = LocalStore(tmp_path)
        dates = trading_days(5)
        as_of = dates[-1]
        arm_id = seed_m_recipe(store, "vol_head", target=UNSIGNED)
        _seat(store, arm_id, as_of)
        seed_arm_predictions(store, arm_id, as_of)

        with pytest.raises(UnsignedChampionTargetError) as refused:
            publish_predictions_feed(store, trading_day=as_of)

        assert arm_id in str(refused.value)
        assert UNSIGNED in str(refused.value)
        assert predictions_key(as_of) in str(refused.value)
        assert not store.exists(predictions_key(as_of))


def _alpha(n: int, value: float) -> dict[str, float]:
    return {f"N{i:03d}": value for i in range(n)}


class TestAOneSidedCrossSectionIsRefused:
    @pytest.fixture
    def seated(self, tmp_path):
        store = LocalStore(tmp_path)
        as_of = trading_days(5)[-1]
        arm_id = seed_m_recipe(store, "champ")
        _seat(store, arm_id, as_of)
        return store, arm_id, as_of

    def test_the_threshold_is_fifty_names(self) -> None:
        assert MIN_NAMES_FOR_SIGN_CHECK == 50

    @pytest.mark.parametrize(
        ("value", "empty_side"),
        [(0.01, "below zero"), (-0.01, "above zero"), (0.0, "away from zero")],
    )
    def test_a_one_sided_batch_at_the_threshold_writes_no_feed(
        self, seated, value, empty_side
    ) -> None:
        store, arm_id, as_of = seated
        _cross_section(store, arm_id, as_of, _alpha(MIN_NAMES_FOR_SIGN_CHECK, value))
        with pytest.raises(OneSidedCrossSectionError) as refused:
            publish_predictions_feed(store, trading_day=as_of)
        message = str(refused.value)
        assert message.startswith("one_sided_cross_section:")
        assert f"no name {empty_side}" in message
        assert arm_id in message
        assert not store.exists(predictions_key(as_of))

    def test_a_one_sided_batch_below_the_threshold_is_served(self, seated) -> None:
        store, arm_id, as_of = seated
        _cross_section(store, arm_id, as_of, _alpha(MIN_NAMES_FOR_SIGN_CHECK - 1, 0.01))
        publish_predictions_feed(store, trading_day=as_of)
        assert read_predictions_feed(store, as_of).champion == arm_id

    def test_a_two_sided_batch_at_the_threshold_is_served(self, seated) -> None:
        store, arm_id, as_of = seated
        alpha = _alpha(MIN_NAMES_FOR_SIGN_CHECK, 0.01)
        alpha["N000"] = -0.01
        _cross_section(store, arm_id, as_of, alpha)
        publish_predictions_feed(store, trading_day=as_of)
        assert read_predictions_feed(store, as_of).predicted_alpha == alpha
