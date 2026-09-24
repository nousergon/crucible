"""`experiment.grade` benchmarks every arm of a date on the common names.

`alpha-engine-config-I10947`, deliverable 3, asserted through the real job
rather than against the pure function that computes the pairing. The lesson
this repository keeps re-learning is that a producer with no production
caller passes every unit test it has: `stack_prediction_columns` itself had
zero callers on 2026-09-01 and the arm that needed it died in `train_arm`.

The fixture narrows ONE arm's recorded population — exactly what a stacked
arm's shadow carries once its base model has no opinion on a young listing —
and reads the benchmark back off the verdict documents the grade wrote.
"""

from __future__ import annotations

import json

import pytest
from conftest import seed_data_daily, sessions_ending

from crucible.config import Settings
from crucible.keys import manifest_key, shadow_key, verdict_key
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.arms import load_arm_specs
from crucible.slots.grading import NAME_PAIRING_METRIC

HORIZON = 21
DECISION_DATES = 6

#: How many names are taken out of the narrowed arm's population. Two rather
#: than one so the intersection is visibly narrower than every arm's own
#: population and a coincidence cannot produce the same reading.
_WITHHELD = 2


@pytest.fixture
def graded_with_a_narrowed_arm(store, source, strategy_dir, cycle_date, tmp_path):
    """Produce a U cycle, narrow one arm's population, then grade it."""
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

    narrowed = load_arm_specs("u", strategy_dir=strategy_dir)[0].arm_id
    withheld: set[str] = set()
    for day in decision_days:
        key = shadow_key(narrowed, day.isoformat())
        document = json.loads(store.get_bytes(key))
        population = list(document["population"])
        # Taken from the END of the sorted population and never from the
        # arm's own selection: the point under test is the BENCHMARK, and
        # dropping a pick would change what this arm is credited with as well.
        drop = [n for n in population[-_WITHHELD:] if n not in document["selection"]]
        withheld.update(drop)
        document["population"] = [n for n in population if n not in drop]
        store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))

    def job(ctx):
        universe.grade(ctx, settings=settings)

    run_job("experiment.grade", job, store=store, trading_day=cycle_date)
    return {
        "store": store,
        "narrowed": narrowed,
        "withheld": withheld,
        "days": [d.isoformat() for d in decision_days],
        "arms": [s.arm_id for s in load_arm_specs("u", strategy_dir=strategy_dir)],
    }


class TestEveryArmOfADateSharesOneBenchmark:
    def test_the_verdicts_of_one_date_were_scored_on_one_population(
        self, graded_with_a_narrowed_arm
    ) -> None:
        graded = graded_with_a_narrowed_arm
        assert graded["withheld"], "the fixture must actually narrow an arm"

        by_day: dict[str, set[int]] = {}
        for arm_id in graded["arms"]:
            for day in graded["days"]:
                key = verdict_key(arm_id, day)
                if not graded["store"].exists(key):
                    continue
                detail = json.loads(graded["store"].get_bytes(key))["detail"]
                by_day.setdefault(day, set()).add(detail["n_population"])

        assert by_day, "the grade produced no verdicts; the fixture is not exercising it"
        for day, sizes in by_day.items():
            assert len(sizes) == 1, (
                f"{day}: arms were benchmarked against populations of {sorted(sizes)}. "
                "Two arms measured against two universes produce two numbers the "
                "engine compares as one measurement"
            )

    def test_the_shared_benchmark_is_the_INTERSECTION_not_the_widest_population(
        self, graded_with_a_narrowed_arm
    ) -> None:
        """The direction matters. Falling back to the widest population would
        put names the narrowed arm never ranked into its benchmark, which is
        the same incomparability with the sign flipped."""
        graded = graded_with_a_narrowed_arm
        day = graded["days"][0]
        widest = max(
            len(json.loads(graded["store"].get_bytes(shadow_key(arm, day)))["population"])
            for arm in graded["arms"]
            if graded["store"].exists(shadow_key(arm, day))
        )
        detail = json.loads(graded["store"].get_bytes(verdict_key(graded["arms"][0], day)))[
            "detail"
        ]

        assert detail["n_population"] < widest
        assert detail["n_common_names"] == detail["n_population"]
        assert detail["n_union_names"] == widest

    def test_the_grade_manifest_carries_the_narrowest_dates_coverage(
        self, graded_with_a_narrowed_arm, cycle_date
    ) -> None:
        """A figure nobody can read is not a published figure.

        `n_floor: 0` and filed on every cycle, narrowed or not: a component
        that emits nothing is unobserved, not healthy, and "no data" is never
        rendered as green (`principles.md`, measurability).
        """
        graded = graded_with_a_narrowed_arm
        manifest = json.loads(
            graded["store"].get_bytes(manifest_key("experiment.grade", cycle_date.isoformat()))
        )

        (row,) = [m for m in manifest["metrics"] if m["name"] == NAME_PAIRING_METRIC]
        assert row["value"] < 1.0, "the narrowed arm shrank the common universe"
        assert row["name_pairing"]["union_name_count"] > row["name_pairing"]["common_name_count"]
        assert graded["narrowed"] in row["name_pairing"]["dropped_by_arm"]
        assert row["unit"] == "ratio"
