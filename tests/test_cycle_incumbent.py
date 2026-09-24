"""`crucible.slots.cycle._incumbent` reads the champion pointer document.

`alpha-engine-config-I10686`: the pointer document
(`crucible.champion.ChampionPointer.to_dict()`) carries the serving arm as
`arm_id`, never `champion`. `_incumbent` read `.get("champion")`, which is
always `None` against a real pointer — so `experiment.grade` took the
library's §9.1 cold-start (bootstrap) path on every cycle, for every slot
that already has a seated champion, forever.

Two tests: a direct unit test against `_incumbent` (no fixture — just a
store and a real pointer), and an integration test that a seeded champion
reaches `run_grade`'s own emitted cycle as `decision.incumbent`, through the
same `_seed_cycle` shape `tests/test_grading.py` already uses for the U
slot.
"""

from __future__ import annotations

import json

from conftest import seed_data_daily, sessions_ending

from crucible.champion import (
    CHAMPION_SCHEMA_VERSION,
    ChampionPointer,
    read_champion_etag,
    write_champion,
)
from crucible.keys import arena_cycle_key
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.cycle import _incumbent
from crucible.store import LocalStore

HORIZON = 21
DECISION_DATES = 6


def _pointer(slot: str, arm_id: str, *, as_of: str) -> ChampionPointer:
    return ChampionPointer(
        schema_version=CHAMPION_SCHEMA_VERSION,
        slot=slot,
        arm_id=arm_id,
        as_of=as_of,
        decided_at=f"{as_of}T02:00:00Z",
        run_id="01JG0000000000000000000000",
        code_sha="a" * 40,
        promotion_source="evidence",
        manifest_key=f"runs/promote/{as_of}/run.json",
        evidence={"status": "decided", "moved": True, "paired_dates": 40},
        attestation=None,
    )


class TestIncumbentReadsArmId:
    def test_a_real_champion_pointer_comes_back_as_the_incumbent(self, tmp_path) -> None:
        """RED against the old code: `.get("champion")` on a real
        `ChampionPointer.to_dict()` returns `None` — this asserts the arm id
        comes back instead."""
        store = LocalStore(tmp_path / "store")
        pointer = _pointer("u", "u:champ:0123456789ab", as_of="2026-08-28")
        write_champion(store, pointer, expected=read_champion_etag(store, "u"))

        assert _incumbent(store, "u") == "u:champ:0123456789ab"

    def test_no_pointer_is_still_no_incumbent(self, tmp_path) -> None:
        store = LocalStore(tmp_path / "store")
        assert _incumbent(store, "u") is None


class TestRunGradeSeesTheRealIncumbent:
    def test_a_seeded_champion_reaches_the_emitted_cycles_decision(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """Integration seam, reusing `test_grading.py`'s `_seed_cycle` shape
        rather than a new large fixture: produce a few decision days for the
        U slot, seed a real champion pointer naming one of the registered
        arms, grade, and assert the emitted `arena_cycle` names that arm as
        `decision.incumbent` — never `bootstrap`."""
        from crucible.config import Settings
        from crucible.slots.arms import load_arm_specs

        settings = Settings(
            store_uri=str(tmp_path / "store"),
            arctic_bucket="unused-in-this-test",
            strategy_dir=strategy_dir,
            origins={"store_uri": "test", "strategy_dir": "test"},
        )
        sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
        decision_days = sessions[:DECISION_DATES]

        seed_data_daily(store, source, [*decision_days, cycle_date])
        for day in decision_days:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=day,
            )

        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        seeded = specs[0].arm_id
        write_champion(
            store,
            _pointer("u", seeded, as_of=decision_days[-1].isoformat()),
            expected=read_champion_etag(store, "u"),
        )

        result: dict = {}

        def job(ctx):
            result.update(universe.grade(ctx, settings=settings))

        run_job("experiment.grade", job, store=store, trading_day=cycle_date)

        cycle = json.loads(store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())))
        decision = cycle["decision"]
        assert decision["incumbent"] == seeded, (
            "run_grade must pass the seeded champion pointer's arm_id into the cycle "
            "as the incumbent, not None"
        )
        assert decision["status"] != "bootstrap", (
            "a seated champion must not take the no-incumbent cold-start path"
        )
