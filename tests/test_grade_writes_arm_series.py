"""`experiment.grade` writes the per-arm score series `promote` reads.

`alpha-engine-config-I10705`. `crucible.keys.arm_series_key`'s own docstring
says the document at `scores/{slot}/{arm}/series.json` is "as produced by
`experiment.grade`", and `crucible.promote.load_slot_inputs` reads exactly
that key for EVERY arm in the register — raising, correctly, when one is
missing ("a missing series is a grading defect"). Nothing under `crucible/`
wrote it. Measured 2026-09-14 against the dedicated integration store: a real
`experiment.grade` that exited `ok` and wrote eleven `verdict.v1` documents
plus an `arena_cycle` left `scores/` EMPTY, and the `promote` that followed it
raised `KeyError: arm 'u:control_null_u:...' is registered in slot 'u' but has
no series`. Every unit test that exercises `promote` hand-writes the series
first, so the producer's absence was invisible from inside this suite — the
first thing to run the two jobs in sequence against a real store found it.

`run_grade` already builds `series_by_arm` in memory (it is what it hands the
arena engine); the fix is to persist it under the key the declared consumer
reads, not to teach `promote` a second way to reconstruct a series from the
verdict documents.
"""

from __future__ import annotations

import json

import pytest
from conftest import sessions_ending

from crucible.config import Settings
from crucible.data import run_daily
from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.keys import arm_register_key, arm_series_key
from crucible.promote import load_slot_inputs
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.arms import ArmRegister

HORIZON = 21
DECISION_DATES = 6


@pytest.fixture
def graded_slot(store, source, strategy_dir, cycle_date, tmp_path):
    """A U slot taken through the real producer chain to a graded cycle.

    Mirrors `tests/test_cycle_first_champion.py::TestRunGradeSubstitutesThe
    Baseline._grade` — the existing shape for "run the real jobs, then assert
    on what they left in the store" — rather than hand-writing artifacts,
    which is precisely the habit that hid this gap.
    """
    settings = Settings(
        store_uri=str(tmp_path / "store"),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )
    sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
    decision_days = sessions[:DECISION_DATES]

    for day in [*decision_days, cycle_date]:
        run_job(
            "data.daily",
            lambda c, day=day: run_daily(
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
    for day in decision_days:
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings),
            store=store,
            trading_day=day,
        )

    result: dict = {}

    def job(ctx):
        result.update(universe.grade(ctx, settings=settings))

    run_job("experiment.grade", job, store=store, trading_day=cycle_date)
    return store, result


def _registered_arms(store) -> list[str]:
    events = [
        json.loads(line)
        for line in store.get_bytes(arm_register_key("u")).decode().splitlines()
        if line.strip()
    ]
    return sorted(ArmRegister.from_dicts(events).all_arms())


class TestGradeWritesTheSeriesPromoteReads:
    def test_every_registered_arm_has_a_series_after_grade(self, graded_slot) -> None:
        """RED before this change: `scores/` held nothing at all."""
        store, _result = graded_slot
        arms = _registered_arms(store)
        assert arms, "the register named no arms, so this test would pass vacuously"
        missing = [a for a in arms if not store.exists(arm_series_key("u", a))]
        assert not missing, (
            f"{len(missing)} of {len(arms)} registered arm(s) have no series document: "
            f"{missing}. `promote` refuses the whole cycle on the first one."
        )

    def test_the_series_document_carries_the_scores_the_cycle_used(self, graded_slot) -> None:
        store, _result = graded_slot
        for arm_id in _registered_arms(store):
            payload = json.loads(store.get_bytes(arm_series_key("u", arm_id)))
            assert payload["arm_id"] == arm_id
            assert isinstance(payload["scores"], dict)
            assert all(isinstance(v, float) for v in payload["scores"].values())
            assert isinstance(payload["misses"], list)

    def test_promote_can_load_the_slot_grade_just_produced(self, graded_slot) -> None:
        """The consumer contract, end to end — the assertion that actually
        failed in production. `load_slot_inputs` is what `crucible promote`
        calls first, and it reads the register and every arm's series."""
        store, _result = graded_slot
        inputs = load_slot_inputs(store, "u")
        assert sorted(inputs.series_by_arm) == _registered_arms(store)

    def test_a_scored_arm_is_not_silently_absent_from_the_series_it_wrote(
        self, graded_slot
    ) -> None:
        """The guard firing (AGENTS.md, Test discipline): delete one series
        and `load_slot_inputs` must refuse the cycle rather than proceed on
        the smaller cohort."""
        store, _result = graded_slot
        victim = _registered_arms(store)[0]
        (store.root / arm_series_key("u", victim)).unlink()
        with pytest.raises(KeyError, match="has no series"):
            load_slot_inputs(store, "u")
