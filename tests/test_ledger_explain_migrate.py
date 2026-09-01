"""The trial ledger, the lineage walk, and the v1 history import."""

from __future__ import annotations

import json

import pytest

from crucible.data import run_daily
from crucible.explain import explain, render
from crucible.ledger import append_trials, n_trials, read_trials
from crucible.migrate import SOURCES, MigrationSourceMissing, run_migrate_history
from crucible.runner import run_job


def _row(slot: str, arm: str, as_of: str, control: bool = False) -> dict:
    return {
        "schema_version": "trial.v1",
        "slot": slot,
        "arm_id": arm,
        "as_of": as_of,
        "control": control,
        "active": True,
        "benchmark": "population",
        "n_dates_scored": 4,
        "first_date": "2026-08-03",
        "last_date": "2026-08-06",
        "mean_score_ratio": 0.01,
        "run_id": "01JG0000000000000000000000",
        "arena_cycle_key": f"arena/{slot}/{as_of}/arena_cycle.json",
        "written_at_utc": "2026-08-28T21:00:00Z",
    }


class TestLedger:
    def test_a_regrade_of_the_same_cycle_appends_nothing(self, store) -> None:
        rows = [_row("u", "u:a:1", "2026-08-28"), _row("u", "u:b:2", "2026-08-28")]
        assert append_trials(store, rows) == 2
        assert append_trials(store, rows) == 0, (
            "a replay must not inflate the multiplicity correction it is meant to leave untouched"
        )
        assert len(read_trials(store)) == 2

    def test_a_write_is_a_strict_prefix_extension(self, store) -> None:
        append_trials(store, [_row("u", "u:a:1", "2026-08-28")])
        before = store.get_bytes("ledger/trials.jsonl").decode().splitlines()
        append_trials(store, [_row("u", "u:a:1", "2026-09-04")])
        after = store.get_bytes("ledger/trials.jsonl").decode().splitlines()
        assert after[: len(before)] == before
        assert len(after) == len(before) + 1

    def test_n_trials_counts_distinct_arms_not_rows(self, store) -> None:
        append_trials(
            store,
            [
                _row("u", "u:a:1", "2026-08-21"),
                _row("u", "u:a:1", "2026-08-28"),
                _row("u", "u:b:2", "2026-08-28"),
                _row("r", "r:c:3", "2026-08-28"),
            ],
        )
        assert n_trials(store) == 3, (
            "an arm graded for forty cycles is one hypothesis measured forty times, not "
            "forty hypotheses"
        )
        assert n_trials(store, slot="u") == 2

    def test_controls_are_excluded_from_the_multiplicity_denominator(self, store) -> None:
        append_trials(
            store,
            [
                _row("u", "u:a:1", "2026-08-28"),
                _row("u", "u:control_planted_u:9", "2026-08-28", control=True),
            ],
        )
        assert n_trials(store) == 1, (
            "counting the harness's own self-check would deflate every real arm's Sharpe"
        )
        assert n_trials(store, include_controls=True) == 2

    def test_an_absent_ledger_reads_as_empty_not_as_an_error(self, store) -> None:
        assert read_trials(store) == []
        assert n_trials(store) == 0


class TestExplain:
    def test_the_walk_reaches_the_data_run_from_the_feature_artifact(
        self, store, source, cycle_date
    ) -> None:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
            store=store,
            trading_day=cycle_date,
        )
        from crucible.features import DEFAULT_FEATURE_VERSION
        from crucible.keys import features_key

        node = explain(store, features_key(DEFAULT_FEATURE_VERSION, cycle_date.isoformat()))
        assert node.manifest is not None
        assert node.manifest["job"] == "data.daily"
        rendered = render(node)
        assert "code_sha=" in rendered and "seed=" in rendered and "cost_usd=" in rendered

    def test_a_run_id_resolves_to_what_that_run_read(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        from conftest import sessions_ending

        from crucible.config import Settings
        from crucible.slots import universe

        settings = Settings(
            store_uri=str(tmp_path / "store"),
            arctic_bucket="unused",
            strategy_dir=strategy_dir,
            origins={},
        )
        day = sessions_ending(cycle_date, 1)[0]
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
            store=store,
            trading_day=day,
        )
        ctx = run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings),
            store=store,
            trading_day=day,
        )
        node = explain(store, ctx.run_id)
        parents = [p.key for p in node.parents]
        assert any(p.startswith("features/") for p in parents), (
            "the shadow's lineage must reach the feature layer it ranked on"
        )
        assert any(
            p.manifest is not None and p.manifest["job"] == "data.daily" for p in node.parents
        )

    def test_an_unclaimed_key_renders_as_unknown_rather_than_being_elided(
        self, store, source, cycle_date
    ) -> None:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
            store=store,
            trading_day=cycle_date,
        )
        with pytest.raises(KeyError, match="never guesses a near match"):
            explain(store, "runs/does-not-exist/2026-08-28/run.json")

    def test_an_empty_store_says_nothing_has_run_here(self, store) -> None:
        with pytest.raises(FileNotFoundError, match="no run manifests"):
            explain(store, "anything")


class _FakeCtx:
    """The minimum of a RunContext the migration touches."""

    def __init__(self, store, trading_day):
        self.store = store
        self.trading_day = trading_day
        self.run_id = "01JG0000000000000000000000"
        self.metrics: list = []
        self.outputs: list = []

    def record_metric(self, metric):
        self.metrics.append(metric)

    def record_output(self, key, payload, schema_version="v1"):
        self.store.put_bytes(key, payload)
        self.outputs.append({"key": key, "schema_version": schema_version})


class TestMigrate:
    def test_an_absent_source_is_named_by_key_and_fails_the_run(
        self, store, tmp_path, cycle_date
    ) -> None:
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        with pytest.raises(MigrationSourceMissing) as excinfo:
            run_migrate_history(_FakeCtx(store, cycle_date), v1_store=v1)
        message = str(excinfo.value)
        for source in SOURCES:
            assert source.key in message, "every absent source must be named by its key"

    def test_the_operator_bootstrap_flag_is_carried_across(
        self, store, tmp_path, strategy_dir, cycle_date
    ) -> None:
        from crucible.slots.arms import load_arm_specs, read_register
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "champion": "momentum_sleeve",
                    "promoted_at": "2026-07-13T22:07:09Z",
                    "promotion_source": "operator_bootstrap",
                }
            ).encode(),
        )
        recipes = {s.name: s for s in load_arm_specs("u", strategy_dir=strategy_dir)}
        result = run_migrate_history(
            _FakeCtx(store, cycle_date),
            v1_store=v1,
            slots=("r",),
            arm_recipes=recipes,
            allow_missing=True,
        )
        assert result["arms_imported"]["r"], "the champion must land in the register"

        pointer = json.loads(store.get_bytes("champions/r/current.json"))
        assert pointer["promotion_source"] == "operator_bootstrap", (
            "an operator-installed champion and an evidence-won one must not render alike"
        )
        register = read_register(store, "r")
        arm = register.state(register.all_arms()[0])
        assert arm.record.bootstrap is True
        assert arm.record.created_date == "2026-07-13", (
            "the OOS clock starts when the arm actually started, not at cutover"
        )

    def test_a_champion_with_no_v2_recipe_is_refused_rather_than_guessed(
        self, store, tmp_path, cycle_date
    ) -> None:
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps(
                {
                    "champion": "thinktank_coverage",
                    "promoted_at": "2026-07-13T22:07:09Z",
                    "promotion_source": "gate_engine",
                }
            ).encode(),
        )
        with pytest.raises(MigrationSourceMissing, match="launder provenance"):
            run_migrate_history(
                _FakeCtx(store, cycle_date),
                v1_store=v1,
                slots=("r",),
                arm_recipes={},
                allow_missing=True,
            )

    def test_a_pointer_with_no_date_will_not_start_a_fresh_clock(
        self, store, tmp_path, strategy_dir, cycle_date
    ) -> None:
        from crucible.slots.arms import load_arm_specs
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps({"champion": "momentum_sleeve"}).encode(),
        )
        recipes = {s.name: s for s in load_arm_specs("u", strategy_dir=strategy_dir)}
        with pytest.raises(MigrationSourceMissing, match="no `promoted_at`"):
            run_migrate_history(
                _FakeCtx(store, cycle_date),
                v1_store=v1,
                slots=("r",),
                arm_recipes=recipes,
                allow_missing=True,
            )
