"""A backfill produces history; it does not serve. `alpha-engine-config-I11005`.

**The live blocker.** `experiment.backfill --slot u --arm attractiveness
--from 2025-11-17 --to 2026-09-09` (`run_mode: live`, run_id
`01M2R98M9FMCVPPFQRXHV23FKF`) failed in about two minutes on the FIRST of 205
sessions with

    MissingArtifactError: the champion pointer for slot 'u' names
    'u:momentum_sleeve:2a5526115a4c', which produced no shadow this cycle.

because the backfill called the slot's production `produce`, whose second
half is the SERVING path. That invariant is correct for a production cycle —
the serving path resolves the pointer rather than importing a ranking
function, so a pointer to an arm that did not produce means production has no
feed today — and it has no business in a historical backfill, which feeds
nothing and serves nothing. Enforced there, it made every NON-champion arm
unbackfillable: a closed loop in which only the incumbent can accumulate the
history a promotion needs.

The fix is a decomposition, never a flag: `produce` is produce-then-serve,
`produce_history` is produce alone, and there is no boolean that could disarm
the serving check from argv. These tests hold BOTH halves: the backfill
succeeds against a pointer that produced nothing, and the serving cycle
against the same store still refuses, with the same message.

The U slot is the one that failed live, so it is the one exercised end to
end here; the M half (a serving path that lives in `crucible.serving`, and a
pointer that is ABSENT rather than unproduced) is in
`tests/test_experiment_backfill.py`.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from conftest import sessions_ending

from crucible.backfill import run_backfill
from crucible.champion import (
    CHAMPION_SCHEMA_VERSION,
    ChampionPointer,
    read_champion_etag,
    write_champion,
)
from crucible.config import Settings
from crucible.data import run_daily
from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.keys import shadow_key, universe_members_key
from crucible.runner import run_job
from crucible.slots import dispatchable_slots, history_producer, universe
from crucible.slots.arms import load_arm_specs
from crucible.slots.cycle import MissingArtifactError
from crucible.store import LocalStore

#: The pointer arm on the live U slot, kept verbatim: it is registered in
#: `crucible/arms/u/register.jsonl` and it is NOT one of this fixture's
#: recipes, which is exactly the live condition — a seated champion the
#: backfilled arm's run does not produce.
ABSENT_CHAMPION = "u:momentum_sleeve:2a5526115a4c"

#: The three sessions the range covers. Three, not two hundred: the produce
#: path is the same on every one of them, and `LAPTOP_SESSION_ALLOWANCE`
#: refuses a larger range off EC2 — a guard this test must not have to
#: override to prove its point.
SESSION_COUNT = 3


@pytest.fixture
def settings(tmp_path, strategy_dir):
    return Settings(
        store_uri=str(tmp_path / "store"),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )


@pytest.fixture
def sessions(store, source, cycle_date):
    """A feature layer and a price panel for each session in the range."""
    days = sessions_ending(cycle_date, SESSION_COUNT)
    for day in days:
        run_job(
            "data.daily",
            lambda c: run_daily(
                c,
                point_in_time=UnavailablePointInTimeSource(
                    reason="synthetic fixture market carries no fundamentals"
                ),
                source=source,
                expected_symbols=source.symbols(),
            ),
            store=store,
            trading_day=day,
        )
    return days


def _seat_champion(store: LocalStore, *, arm_id: str, as_of: dt.date) -> None:
    """A USABLE U pointer naming ``arm_id``.

    `crucible.champion.read_champion` refuses a pointer whose producing run's
    manifest is not `ok`, so the manifest is written; U is not in
    `ATTESTED_SLOTS`, so no attestation is owed.
    """
    manifest = f"runs/promote/{as_of.isoformat()}/run.json"
    store.put_bytes(manifest, json.dumps({"status": "ok", "job": "promote"}).encode("utf-8"))
    write_champion(
        store,
        ChampionPointer(
            schema_version=CHAMPION_SCHEMA_VERSION,
            slot="u",
            arm_id=arm_id,
            as_of=as_of.isoformat(),
            decided_at=f"{as_of.isoformat()}T02:00:00Z",
            run_id="01JG0000000000000000000000",
            code_sha="a" * 40,
            promotion_source="operator_bootstrap",
            manifest_key=manifest,
            evidence={"status": "decided", "moved": True, "paired_dates": 40},
            attestation=None,
        ),
        expected=read_champion_etag(store, "u"),
    )


def _backfill(store, settings, *, arm: str, days: list[dt.date]) -> dict:
    result: dict = {}
    run_job(
        "experiment.backfill",
        lambda c: result.update(
            run_backfill(
                c,
                produce=history_producer(universe),
                specs=load_arm_specs("u", store=c.store, strategy_dir=settings.strategy_dir),
                settings=settings,
                slot="u",
                arm=arm,
                start=days[0],
                end=days[-1],
            )
        ),
        store=store,
        trading_day=days[-1],
        run_mode="replay",
        discriminator=f"u.{arm}",
    )
    return result


class TestANonChampionArmBackfillsAgainstASeatedChampion:
    """Deliverable 1 — RED before the fix, on the live failure's shape."""

    def test_the_range_is_produced_although_the_champion_produced_nothing(
        self, store, settings, sessions
    ) -> None:
        _seat_champion(store, arm_id=ABSENT_CHAMPION, as_of=sessions[-1])
        result = _backfill(store, settings, arm="tech_score_gate", days=sessions)

        assert result["produced"] == [d.isoformat() for d in sessions], result
        arm_id = result["arm_id"]
        assert arm_id != ABSENT_CHAMPION
        for day in sessions:
            assert store.exists(shadow_key(arm_id, day.isoformat()))

    def test_the_manifest_is_ok_and_rejects_nothing(self, store, settings, sessions) -> None:
        from crucible.keys import manifest_key

        _seat_champion(store, arm_id=ABSENT_CHAMPION, as_of=sessions[-1])
        _backfill(store, settings, arm="tech_score_gate", days=sessions)
        document = json.loads(
            store.get_bytes(
                manifest_key("experiment.backfill", sessions[-1], discriminator="u.tech_score_gate")
            ).decode("utf-8")
        )
        assert document["status"] == "ok", document["reason"]
        assert document["rows_rejected"] == []

    def test_no_serving_feed_is_written_for_a_backfilled_session(
        self, store, settings, sessions
    ) -> None:
        """A backfill of a past session must not republish that session's
        champion feed under the serving key — not even when the pointer
        resolves, which is why the champion here IS a registered arm."""
        specs = {s.name: s.arm_id for s in load_arm_specs("u", strategy_dir=settings.strategy_dir)}
        _seat_champion(store, arm_id=specs["momentum_sleeve"], as_of=sessions[-1])
        _backfill(store, settings, arm="momentum_sleeve", days=sessions)
        for day in sessions:
            assert not store.exists(universe_members_key(day.isoformat()))


class TestTheServingCycleStillRefuses:
    """Deliverable 1's other half — the check is unchanged where it belongs."""

    def test_a_serving_cycle_with_an_unproduced_champion_raises(
        self, store, settings, sessions
    ) -> None:
        _seat_champion(store, arm_id=ABSENT_CHAMPION, as_of=sessions[-1])
        with pytest.raises(MissingArtifactError) as excinfo:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=sessions[-1],
                run_mode="replay",
                discriminator="u",
            )
        message = str(excinfo.value)
        # Deliverable 4: the refusal still names the pointer's slot and the
        # arm it resolves to. An operator who reads only this line has to be
        # able to act on it.
        assert "slot 'u'" in message
        assert ABSENT_CHAMPION in message
        assert "produced no shadow this cycle" in message

    def test_a_serving_cycle_whose_champion_produced_still_writes_the_feed(
        self, store, settings, sessions
    ) -> None:
        """The serving path is not merely un-raising after the split — it
        still SERVES, and an assertion only on the refusal would pass over a
        feed that silently stopped being written."""
        specs = {s.name: s.arm_id for s in load_arm_specs("u", strategy_dir=settings.strategy_dir)}
        _seat_champion(store, arm_id=specs["momentum_sleeve"], as_of=sessions[-1])
        served: dict = {}
        run_job(
            "experiment.run",
            lambda c: served.update(universe.produce(c, settings=settings)),
            store=store,
            trading_day=sessions[-1],
            run_mode="replay",
            discriminator="u",
        )
        assert served["champion"] == specs["momentum_sleeve"]
        assert served["feed_key"] == universe_members_key(sessions[-1].isoformat())
        assert store.exists(served["feed_key"])


class TestTheHistoryProducerIsResolvedAndNeverInferred:
    def test_every_dispatchable_slot_declares_one(self) -> None:
        """A slot that reaches the CLI without a `produce_history` would send
        `experiment.backfill` back into the serving path the day its module
        lands, with nothing saying so."""
        for slot, module in dispatchable_slots().items():
            assert callable(history_producer(module)), slot

    def test_a_module_without_one_refuses_rather_than_falling_back(self) -> None:
        """The fallback is the defect wearing a resolver's clothes: it would
        restore the exact coupling this issue removed, silently."""
        import types

        module = types.ModuleType("crucible.slots.nothing")
        module.produce = lambda *a, **k: None  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError) as excinfo:
            history_producer(module)
        assert "produce_history" in str(excinfo.value)
        assert "serving path" in str(excinfo.value)

    def test_the_history_producer_is_not_the_serving_producer(self) -> None:
        """U, R and M each have a serving half a backfill must not enter, so
        their two entry points are two functions. S is the declared
        exception and is asserted separately, by behaviour."""
        for slot, module in dispatchable_slots().items():
            if slot == "s":
                continue
            assert history_producer(module) is not module.produce, slot

    def test_the_s_slot_declares_the_equality_rather_than_inheriting_it(self, monkeypatch) -> None:
        """S writes no champion feed on a production cycle either — its feed
        has no key builder and no schema in this repository — so its history
        path runs its produce path's BODY. Asserted by dispatch, not by a
        docstring: the point is that the equality is written down in the
        module, and a future S serving half that forgot to split would fail
        here.

        The one difference is `alpha-engine-config-I11452`'s declared
        no-M-champion outcome, which only the ARC's `produce` consults: a
        backfill of history that cannot exist must still fail."""
        from types import SimpleNamespace

        from crucible.slots import strategy

        seen: list[str] = []
        monkeypatch.setattr(
            strategy, "_produce_sessions", lambda ctx, **kwargs: seen.append("body") or {}
        )
        monkeypatch.setattr(
            strategy,
            "_registered_arms",
            lambda ctx, *, settings: (strategy.SlotStrategies(registered=(), refused=()), []),
        )
        monkeypatch.setattr(
            strategy,
            "declare_no_m_champion",
            lambda ctx, *, job: seen.append("declared?") or None,
        )
        ctx = SimpleNamespace(trading_day=dt.date(2026, 8, 28))
        strategy.produce(ctx, settings=None)
        assert seen == ["declared?", "body"]
        seen.clear()
        strategy.produce_history(ctx, settings=None)
        assert seen == ["body"]
