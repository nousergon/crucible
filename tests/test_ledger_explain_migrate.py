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

    def test_a_concurrent_writer_between_read_and_write_is_refused(
        self, store, monkeypatch
    ) -> None:
        """alpha-engine-config-I9757 defect #12: `LedgerAppendError`'s old
        guard (`len(combined) < len(existing)`) can never be true, because
        `combined` is built by concatenating `existing` with `fresh` two
        lines above — the comparison tests a property of its own
        construction, not of the log. The property that matters is that no
        row this call already read gets dropped or reordered by its own
        write; the fix re-reads the log immediately before writing and
        refuses if a concurrent writer got there first."""
        from crucible import ledger as ledger_module

        append_trials(store, [_row("u", "u:a:1", "2026-08-28")])

        real_read_trials = ledger_module.read_trials
        calls = {"n": 0}

        def racing_read(store_arg):
            calls["n"] += 1
            if calls["n"] == 2:
                # Simulate a second writer landing between THIS call's first
                # read and its write: append a row directly, bypassing this
                # call's own bookkeeping, then return the log AS IT NOW
                # STANDS — the same as a real re-read would.
                current = real_read_trials(store_arg)
                combined = current + [_row("r", "r:z:9", "2026-08-28")]
                payload = (
                    "\n".join(__import__("json").dumps(row, sort_keys=True) for row in combined)
                    + "\n"
                ).encode("utf-8")
                store_arg.put_bytes(ledger_module.ledger_key(), payload)
            return real_read_trials(store_arg)

        monkeypatch.setattr(ledger_module, "read_trials", racing_read)
        from crucible.ledger import LedgerAppendError

        with pytest.raises(LedgerAppendError, match="changed between read and write"):
            append_trials(store, [_row("u", "u:b:2", "2026-08-28")])

        # The racing writer's row must survive — the whole point of the guard.
        monkeypatch.undo()
        rows = read_trials(store)
        assert any(r["arm_id"] == "r:z:9" for r in rows), (
            "the guard exists so a race never silently drops the other writer's row"
        )


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

    def test_a_collision_keeps_the_later_run_and_names_the_earlier_one(
        self, store, cycle_date
    ) -> None:
        """alpha-engine-config-I9757 defect #9: the module's own docstring
        claims "the caller sees both `run_id`s in the walk" — `by_output[key]
        = manifest` alone just overwrites, and the caller sees one. Two runs
        both claiming `shared/key.json` must render as one winner (the later
        one, by `finished`) PLUS the earlier run named as a collision, in
        both the dict form (`Lineage.to_dict()["also_claimed_by"]`) and the
        rendered text `crucible explain` prints."""
        import datetime as dt

        earlier = dt.datetime(2026, 8, 28, 10, 0, tzinfo=dt.UTC)
        later = dt.datetime(2026, 8, 28, 14, 0, tzinfo=dt.UTC)

        first = run_job(
            "data.heal",
            lambda c: c.record_output("shared/key.json", b'{"from": "first"}'),
            store=store,
            trading_day=cycle_date,
            now=earlier,
        )
        second = run_job(
            "data.daily",
            lambda c: c.record_output("shared/key.json", b'{"from": "second"}'),
            store=store,
            trading_day=cycle_date,
            now=later,
        )

        node = explain(store, "shared/key.json")
        assert node.manifest["run_id"] == second.run_id, (
            "the LATER run (by `finished`) is the winner the walk resolves to"
        )
        assert node.collisions == (first.run_id,), (
            "the earlier claimant must not be silently dropped from the walk"
        )
        as_dict = node.to_dict()
        assert as_dict["also_claimed_by"] == [first.run_id]

        rendered = render(node)
        assert "ALSO CLAIMED BY" in rendered
        assert first.run_id in rendered
        assert second.run_id in rendered


def _run_migrate(store, cycle_date, **kwargs):
    """Run `migrate.history` through the REAL runner, not a hand-rolled ctx.

    alpha-engine-config-I9757 defect #7b: every migrate test used to
    construct a `_FakeCtx` whose `record_output` fabricated an entry with no
    `sha256` and never called `compare_and_swap` — migrate was never run
    through `run_job` in the suite, so its manifest was never schema-
    validated and the champion-pointer write's CAS/lineage behaviour was
    never exercised at all. This wrapper is the fix: it returns
    `(RunContext, result)`, where `result` is `run_migrate_history`'s own
    return value, captured via the closure since `run_job` does not forward
    a job callable's return value onto the `RunContext` it returns.
    """
    from crucible.runner import run_job

    captured: dict = {}

    def job(ctx):
        captured["result"] = run_migrate_history(ctx, **kwargs)

    ctx = run_job("migrate.history", job, store=store, trading_day=cycle_date)
    return ctx, captured.get("result")


class TestMigrate:
    def test_an_absent_source_is_named_by_key_and_fails_the_run(
        self, store, tmp_path, cycle_date
    ) -> None:
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        with pytest.raises(MigrationSourceMissing) as excinfo:
            _run_migrate(store, cycle_date, v1_store=v1)
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
        ctx, result = _run_migrate(
            store,
            cycle_date,
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

        # defect #7: the pointer must be real lineage, not a bare PUT the
        # manifest never heard about.
        output_keys = [o["key"] for o in ctx.outputs]
        assert "champions/r/current.json" in output_keys, (
            "the champion pointer must be recorded in outputs[], or `crucible "
            "explain champions/r/current.json` cannot name the run that wrote it"
        )
        pointer_output = next(o for o in ctx.outputs if o["key"] == "champions/r/current.json")
        assert len(pointer_output["sha256"]) == 64, (
            "a fabricated output entry with no real content hash would defeat both the "
            "manifest schema's artifactRef contract and idempotency-by-content-hash"
        )

        node = explain(store, "champions/r/current.json")
        assert node.manifest["run_id"] == ctx.run_id, "explain must resolve the pointer's lineage"

    def test_allow_missing_records_fail_not_a_third_state(
        self, store, tmp_path, strategy_dir, cycle_date
    ) -> None:
        """alpha-engine-config-I9757 defect #3: `--allow-missing` must not
        spell a third state (`DEGRADED_BY_OPERATOR_CONSENT`) at the metric
        level inside a manifest whose own `status` is `ok`. `FAIL` is the
        honest, schema-legal word; the operator's consent is prose in
        `status_reason`, not a fabricated status token."""
        from crucible.slots.arms import load_arm_specs
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps(
                {
                    "champion": "momentum_sleeve",
                    "promoted_at": "2026-07-13T22:07:09Z",
                    "promotion_source": "operator_bootstrap",
                }
            ).encode(),
        )
        recipes = {s.name: s for s in load_arm_specs("u", strategy_dir=strategy_dir)}
        ctx, result = _run_migrate(
            store,
            cycle_date,
            v1_store=v1,
            slots=("r",),
            arm_recipes=recipes,
            allow_missing=True,
        )
        assert result["sources_missing"], "this fixture leaves sources absent on purpose"
        migrate_metric = next(m for m in ctx.metrics if m["name"] == "arms_migrated")
        assert migrate_metric["status"] == "FAIL"
        assert "ABSENT" in migrate_metric["status_reason"]
        manifest_key = f"runs/migrate.history/{cycle_date.isoformat()}/run.json"
        manifest = json.loads(store.get_bytes(manifest_key))
        assert manifest["status"] == "ok", (
            "the RUN succeeded (the operator consented to the gap); the schema forbids "
            "the metric from spelling anything other than ok/failed/OK/FAIL/BREACH/... "
            "— never a degraded-flavoured word invented for this one caller"
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
            _run_migrate(
                store,
                cycle_date,
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
            _run_migrate(
                store,
                cycle_date,
                v1_store=v1,
                slots=("r",),
                arm_recipes=recipes,
                allow_missing=True,
            )

    def test_a_rerun_does_not_lose_the_pointer_to_a_stale_cas_token(
        self, store, tmp_path, strategy_dir, cycle_date
    ) -> None:
        """Migration is documented one-shot and idempotent: `register_arms`
        already refuses to duplicate an arm. The champion pointer write must
        be equally safe to repeat — a CAS token captured once and reused on
        a second run would raise `PointerConflictError` on the exact rerun
        the module's own docstring promises is safe."""
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps(
                {
                    "champion": "momentum_sleeve",
                    "promoted_at": "2026-07-13T22:07:09Z",
                    "promotion_source": "operator_bootstrap",
                }
            ).encode(),
        )
        from crucible.slots.arms import load_arm_specs

        recipes = {s.name: s for s in load_arm_specs("u", strategy_dir=strategy_dir)}
        kwargs = dict(v1_store=v1, slots=("r",), arm_recipes=recipes, allow_missing=True)
        _run_migrate(store, cycle_date, **kwargs)
        # A second run reuses the same store; run_job refuses a second
        # manifest at the same key on the same trading day only if the CALLER
        # re-derives its own etag each time, which is what `run_migrate_history`
        # must do internally rather than caching one from an earlier read.
        pointer_before = json.loads(store.get_bytes("champions/r/current.json"))
        _run_migrate(store, cycle_date, **kwargs)
        pointer_after = json.loads(store.get_bytes("champions/r/current.json"))
        assert pointer_after == pointer_before
