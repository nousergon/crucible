"""`promote` never moves the M pointer onto an arm with nothing to serve.

2026-10-05, live: `crucible promote --slot m` for as_of 2026-10-02 seated
`m:v3meta_stack:1bb8d3646649` on point evidence over 2026-06-05..09-02. That
same day's `experiment.run[m]` had REFUSED the arm (`arm_refused_at_registration`,
`unservable`: its base heads lacked predictions for 09-28..10-01), so it wrote
no `arm_predictions/.../2026-10-02.json`. The feed `experiment.run` had already
published named the OLD champion; the pointer named the new one; the trader's
resolver refuses exactly that disagreement.

Three layers, each asserted here against the shape that broke:

1. **The decision** — `crucible.slots.model.grade` supplies a `servable_as_of`
   serving precondition (:func:`crucible.slots.model.evaluate_servable_as_of`),
   so the engine bars the arm, the incumbent holds, and the arm is NAMED in
   `decision.ineligible`.
2. **The write** — a cycle that still names such an arm (one graded before
   the precondition existed) is REFUSED by `promote`, loudly, with the
   incumbent pointer untouched; never a silent `return None`.
3. **The feed** — a legitimate M promotion republishes `predictions/{as_of}.json`
   for the new champion, so the two halves of the trader contract agree.
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
from crucible.keys import predictions_key
from crucible.promote import PromotionRefused, run_promotion
from crucible.serving import publish_predictions_feed, read_predictions_feed
from crucible.slots import get_slot
from crucible.slots.model import SERVABLE_AS_OF_PRECONDITION, evaluate_servable_as_of
from crucible.store import LocalStore
from tests.support.panels import trading_days
from tests.support.servable import seed_arm_predictions

CODE_SHA = "a" * 40
SEAT_MANIFEST = "runs/promote/seed/m/run.json"


def _slot(dates: list[str]) -> tuple[ArmRegister, dict[str, str]]:
    register = ArmRegister()
    ids: dict[str, str] = {}
    for name in ("champ", "stack"):
        register, record = register.register(
            slot="m",
            name=name,
            spec={"name": name},
            created_date=dates[0],
            filed_on=dates[0],
        )
        ids[name] = record.arm_id
    return register, ids


def _series(ids: dict[str, str], dates: list[str]) -> dict[str, ArmSeries]:
    """``stack`` leads ``champ`` on every paired date — on history alone it
    wins the pointer, exactly as `m:v3meta_stack:1bb8d3646649` did."""
    return {
        ids["champ"]: ArmSeries(arm_id=ids["champ"], scores=dict.fromkeys(dates, 0.0)),
        ids["stack"]: ArmSeries(arm_id=ids["stack"], scores=dict.fromkeys(dates, 0.045)),
    }


def _cycle(spec, as_of, register, series_by_arm, *, incumbent, preconditions=None):
    from nousergon_lib.arena.engine import run_cycle

    return run_cycle(
        config=spec.arena,
        as_of=as_of,
        register=register,
        series_by_arm=series_by_arm,
        incumbent=incumbent,
        preconditions=preconditions,
    )


def _seat_incumbent_and_serve_it(store: LocalStore, arm_id: str, as_of: str) -> bytes:
    """The state at 15:00 on the arc day: the incumbent is seated, and
    `experiment.run[m]` already published today's feed for it."""
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
    seed_arm_predictions(store, arm_id, as_of)
    publish_predictions_feed(store, trading_day=as_of)
    return store.get_bytes(champion_key("m"))


@pytest.fixture
def arena(tmp_path):
    store = LocalStore(tmp_path)
    dates = trading_days(40)
    register, ids = _slot(dates)
    as_of = dates[-1]
    pointer_bytes = _seat_incumbent_and_serve_it(store, ids["champ"], as_of)
    return store, dates, register, ids, as_of, pointer_bytes


class TestTheDecisionBarsAnArmWithNothingToServe:
    def test_the_winner_with_no_cross_section_is_barred_and_named(self, arena) -> None:
        store, dates, register, ids, as_of, pointer_bytes = arena
        spec = get_slot("m")
        # `stack` produced nothing on `as_of` — no `arm_predictions` for it.
        checks = evaluate_servable_as_of(store, [ids["stack"]], as_of=as_of)
        assert not checks[ids["stack"]].passed
        assert checks[ids["stack"]].name == SERVABLE_AS_OF_PRECONDITION

        cycle = _cycle(
            spec,
            as_of,
            register,
            _series(ids, dates),
            incumbent=ids["champ"],
            preconditions={arm: [check] for arm, check in checks.items()},
        )
        result = run_promotion(
            spec=spec, register=register, cycle=cycle, store=store, code_sha=CODE_SHA
        )

        assert not result.decision.moved
        assert result.decision.champion == ids["champ"]
        assert result.pointer is None
        assert store.get_bytes(champion_key("m")) == pointer_bytes, "the incumbent holds"
        assert read_predictions_feed(store, as_of).champion == ids["champ"]

        barred = result.decision.ineligible[ids["stack"]]
        assert [p.name for p in barred] == [SERVABLE_AS_OF_PRECONDITION]
        assert ids["stack"] in barred[0].reason
        assert as_of in barred[0].reason
        assert "arm_predictions/" in barred[0].reason

    def test_a_servable_challenger_passes_the_same_precondition(self, arena) -> None:
        store, _, _, ids, as_of, _ = arena
        seed_arm_predictions(store, ids["stack"], as_of)
        check = evaluate_servable_as_of(store, [ids["stack"]], as_of=as_of)[ids["stack"]]
        assert check.passed, check.reason

    def test_a_cross_section_for_another_session_is_not_servable(self, arena) -> None:
        """The serving path's own session check, not an existence test: a
        document misfiled under today's key is look-ahead or stale."""
        store, dates, _, ids, as_of, _ = arena
        key = seed_arm_predictions(store, ids["stack"], dates[-2])
        store.put_bytes(key.replace(dates[-2], as_of), store.get_bytes(key))
        check = evaluate_servable_as_of(store, [ids["stack"]], as_of=as_of)[ids["stack"]]
        assert not check.passed
        assert "trading_day" in check.reason


class TestTheWriteRefusesWhatTheDecisionMissed:
    def test_the_2026_10_02_cycle_is_refused_loudly_and_the_incumbent_stays(self, arena) -> None:
        """A cycle graded WITHOUT the precondition — the live 2026-10-02
        artifact — names the unservable arm as a moved champion."""
        store, dates, register, ids, as_of, pointer_bytes = arena
        spec = get_slot("m")
        cycle = _cycle(spec, as_of, register, _series(ids, dates), incumbent=ids["champ"])
        assert cycle.decision.moved and cycle.decision.champion == ids["stack"]
        feed_bytes = store.get_bytes(predictions_key(as_of))

        with pytest.raises(PromotionRefused) as refused:
            run_promotion(spec=spec, register=register, cycle=cycle, store=store, code_sha=CODE_SHA)

        message = str(refused.value)
        assert ids["stack"] in message
        assert ids["champ"] in message
        assert "servable_as_of" in message
        assert store.get_bytes(champion_key("m")) == pointer_bytes
        assert store.get_bytes(predictions_key(as_of)) == feed_bytes


class TestALegitimatePromotionMovesTheFeedWithThePointer:
    def test_the_feed_names_the_new_champion(self, arena) -> None:
        store, dates, register, ids, as_of, _ = arena
        spec = get_slot("m")
        seed_arm_predictions(store, ids["stack"], as_of)
        assert read_predictions_feed(store, as_of).champion == ids["champ"]
        checks = evaluate_servable_as_of(store, [ids["stack"]], as_of=as_of)

        result = run_promotion(
            spec=spec,
            register=register,
            cycle=_cycle(
                spec,
                as_of,
                register,
                _series(ids, dates),
                incumbent=ids["champ"],
                preconditions={arm: [check] for arm, check in checks.items()},
            ),
            store=store,
            code_sha=CODE_SHA,
        )

        assert result.decision.moved
        assert result.pointer is not None and result.pointer.arm_id == ids["stack"]
        assert predictions_key(as_of) in result.keys_written
        feed = read_predictions_feed(store, as_of)
        assert feed.champion == ids["stack"]
        assert feed.source_key == f"arm_predictions/{ids['stack'].replace(':', '~')}/{as_of}.json"

    def test_a_slot_without_a_feed_writes_no_feed(self, tmp_path) -> None:
        """U, R and S serve under their own keys; only M republishes."""
        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec = get_slot("u")
        register = ArmRegister()
        ids: dict[str, str] = {}
        for name in ("champ", "chal"):
            register, record = register.register(
                slot="u", name=name, spec={"name": name}, created_date=dates[0], filed_on=dates[0]
            )
            ids[name] = record.arm_id
        series_by_arm = {
            ids["champ"]: ArmSeries(arm_id=ids["champ"], scores=dict.fromkeys(dates, 0.0)),
            ids["chal"]: ArmSeries(arm_id=ids["chal"], scores=dict.fromkeys(dates, 0.045)),
        }
        result = run_promotion(
            spec=spec,
            register=register,
            cycle=_cycle(spec, dates[-1], register, series_by_arm, incumbent=ids["champ"]),
            store=store,
            code_sha=CODE_SHA,
        )
        assert result.decision.moved
        assert not store.exists(predictions_key(dates[-1]))
