"""The trial ledger, the lineage walk, and the v1 history import."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from crucible.data import run_daily
from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.explain import NoSettledVerdictError, explain, render, select_newest_settled_verdict
from crucible.keys import manifest_key
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
            lambda c: run_daily(
                c,
                point_in_time=UnavailablePointInTimeSource(
                    reason="synthetic fixture market carries no fundamentals"
                ),
                source=source,
                expected_symbols=source.symbols(),
            ),
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

    def test_a_run_id_resolves_to_its_discriminated_key_not_the_bare_shape(
        self, store, cycle_date
    ) -> None:
        """alpha-engine-config-I9807 review: `explain.py` reconstructed the root
        key as the bare `runs/{job}/{day}/run.json` shape, ignoring the
        `discriminator` `crucible.keys.manifest_key` has supported since
        I9781 — wrong for every `experiment.run`/`experiment.grade`/
        `alerts.sweep` manifest, which is every job that ever legitimately
        writes more than one manifest per trading day. This pins the fix:
        the resolved node's identifier must be the key the manifest was
        actually written under, discriminator included.
        """
        import crucible.keys as store_keys

        ctx = run_job(
            "experiment.run",
            lambda c: None,
            store=store,
            trading_day=cycle_date,
            discriminator="u",
        )
        expected = store_keys.manifest_key(
            "experiment.run", cycle_date.isoformat(), discriminator="u"
        )
        node = explain(store, ctx.run_id)
        assert node.key == expected
        assert store.exists(node.key), (
            "the identifier explain() names for this run must be the one it was "
            "actually written under, not a reconstruction that happens to look plausible"
        )

    def test_an_unclaimed_key_renders_as_unknown_rather_than_being_elided(
        self, store, source, cycle_date
    ) -> None:
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
            trading_day=cycle_date,
        )
        with pytest.raises(KeyError, match="never guesses a near match"):
            explain(store, "runs/does-not-exist/2026-08-28/run.json")

    def test_an_empty_store_says_nothing_has_run_here(self, store) -> None:
        with pytest.raises(FileNotFoundError, match="no CONFORMANT run manifests"):
            explain(store, "anything")

    def test_a_placeholder_code_sha_manifest_elsewhere_does_not_break_the_walk(
        self, store, source, cycle_date
    ) -> None:
        """`alpha-engine-config-I10626`: one manifest under `runs/` carrying
        the all-zero `code_sha` placeholder used to make `load_manifests`
        raise on it, taking down a walk that never touches it. It must now
        be named as unreadable, never elided, and the walk over the REST of
        the store still completes."""
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
            trading_day=cycle_date,
        )
        from crucible.features import DEFAULT_FEATURE_VERSION
        from crucible.keys import features_key
        from crucible.manifest import RUN_MANIFEST_SCHEMA_VERSION

        broken_key = manifest_key("smoke", "2026-08-20")
        broken = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": "01JG0000000000000000BR0KEN",
            "job": "smoke",
            "run_mode": "live",
            "trading_day": "2026-08-20",
            "calendar_date": "2026-08-20",
            "status": "ok",
            "reason": "",
            "started": "2026-08-20T14:00:00Z",
            "finished": "2026-08-20T14:01:00Z",
            "code_sha": "0" * 40,
            "release_sha": "0" * 40,
            "seed": 20260820,
            "inputs": [],
            "outputs": [],
            "rows_in": 0,
            "rows_out": 0,
            "rows_rejected": [],
            "cost_usd": 0.0,
            "llm_calls": [],
            "resource": {
                "instance_type": "c7i.xlarge",
                "spot": True,
                "escalated_to_on_demand": False,
                "interruptions": 0,
                "mem_peak_mb": 0.0,
                "disk_free_mb": 0.0,
            },
            "metrics": [],
            "attempts": [{"n": 1, "reason": "initial"}],
        }
        store.put_bytes(broken_key, json.dumps(broken).encode("utf-8"))

        node = explain(store, features_key(DEFAULT_FEATURE_VERSION, cycle_date.isoformat()))

        assert node.manifest is not None and node.manifest["job"] == "data.daily"
        assert node.unreadable == {broken_key: node.unreadable[broken_key]}
        assert "placeholder" in node.unreadable[broken_key]
        rendered = render(node)
        assert "1 manifest(s) in this store could not be validated." in rendered
        assert broken_key in rendered

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


def _seed_produced(store: Any, recipe: Any) -> None:
    """One shadow for ``recipe``, so the migration will ADMIT its slot.

    `crucible.migrate.admission_refusal` seats a slot only once the arm the v1
    champion resolves to has PRODUCED (`alpha-engine-config-I10961`,
    `-I10964`): registration is the weaker fact, and an imported pointer on a
    mute arm points the trader's contract at silence.
    """
    from crucible.keys import shadow_key

    store.put_bytes(shadow_key(recipe.arm_id, "2026-07-13"), json.dumps({"names": []}).encode())


class TestMigrate:
    def test_an_absent_source_is_named_by_key_and_fails_an_ASSERTED_run(
        self, store, tmp_path, cycle_date
    ) -> None:
        """The operator path: a caller that NAMED its slots said the sources
        were there, so a partial import under that claim is refused loudly."""
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        with pytest.raises(MigrationSourceMissing) as excinfo:
            _run_migrate(store, cycle_date, v1_store=v1, slots=("u", "r"))
        message = str(excinfo.value)
        for source in SOURCES:
            assert source.key in message, "every absent source must be named by its key"

    def test_an_absent_source_is_recorded_and_the_SCHEDULED_run_still_exits_ok(
        self, store, tmp_path, cycle_date
    ) -> None:
        """`alpha-engine-config-I10961` deliverable 4. `migrate.history` is an
        arc stage now, and `run_arc` stops at the first raise - so a stage that
        raised because v1 (a system being decommissioned) had stopped writing a
        source would kill every stage after it. The absence is recorded, every
        slot defers, and nothing is written."""
        from crucible.keys import champion_key
        from crucible.store import LocalStore

        v1 = LocalStore(tmp_path / "v1")
        _ctx, result = _run_migrate(store, cycle_date, v1_store=v1)
        assert [m["key"] for m in result["sources_missing"]] == [s.key for s in SOURCES]
        assert set(result["deferred"]) == set(result["slots_considered"])
        for slot in result["slots_considered"]:
            assert not store.exists(champion_key(slot))

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
        # The fixture tree holds U recipes and the v1 source here is R's pointer;
        # a recipe for another slot is refused, so the same recipe is declared R.
        recipes = {
            s.name: replace(s, slot="r") for s in load_arm_specs("u", strategy_dir=strategy_dir)
        }
        _seed_produced(store, recipes["momentum_sleeve"])
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
        # The fixture tree holds U recipes and the v1 source here is R's pointer;
        # a recipe for another slot is refused, so the same recipe is declared R.
        recipes = {
            s.name: replace(s, slot="r") for s in load_arm_specs("u", strategy_dir=strategy_dir)
        }
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
        # The fixture tree holds U recipes and the v1 source here is R's pointer;
        # a recipe for another slot is refused, so the same recipe is declared R.
        recipes = {
            s.name: replace(s, slot="r") for s in load_arm_specs("u", strategy_dir=strategy_dir)
        }
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

        # The fixture tree holds U recipes and the v1 source here is R's pointer;
        # a recipe for another slot is refused, so the same recipe is declared R.
        recipes = {
            s.name: replace(s, slot="r") for s in load_arm_specs("u", strategy_dir=strategy_dir)
        }
        kwargs = dict(v1_store=v1, slots=("r",), arm_recipes=recipes, allow_missing=True)
        _seed_produced(store, recipes["momentum_sleeve"])
        _run_migrate(store, cycle_date, **kwargs)
        # A second run reuses the same store; run_job refuses a second
        # manifest at the same key on the same trading day only if the CALLER
        # re-derives its own etag each time, which is what `run_migrate_history`
        # must do internally rather than caching one from an earlier read.
        pointer_before = json.loads(store.get_bytes("champions/r/current.json"))
        _run_migrate(store, cycle_date, **kwargs)
        pointer_after = json.loads(store.get_bytes("champions/r/current.json"))
        # `run_id`/`decided_at`/`manifest_key` name the run that WROTE the
        # pointer (`alpha-engine-config-I10691`) and legitimately differ
        # between two separate `migrate.history` invocations — the CAS not
        # raising is the real assertion the docstring names; the fields the
        # v1 source actually determines must still agree.
        for field in ("slot", "arm_id", "as_of", "promotion_source"):
            assert pointer_after[field] == pointer_before[field], field


class TestSelectNewestSettledVerdict:
    """`crucible.explain.select_newest_settled_verdict`
    (`alpha-engine-config-I10858`) — the deterministic target the scheduled
    weekly `explain` arc stage walks, since an arc stage names no operator
    target on its own argv (`crucible.weekly.Stage.argv`)."""

    def test_the_newest_trading_day_wins_across_slots_and_arms(self, store) -> None:
        store.put_bytes(
            "experiments/r~arm_a~aaa/2026-07-31/verdict.json",
            json.dumps({"trading_day": "2026-07-31"}).encode(),
        )
        store.put_bytes(
            "experiments/u~arm_b~bbb/2026-08-12/verdict.json",
            json.dumps({"trading_day": "2026-08-12"}).encode(),
        )
        store.put_bytes(
            "experiments/m~arm_c~ccc/2026-08-05/verdict.json",
            json.dumps({"trading_day": "2026-08-05"}).encode(),
        )
        expected = "experiments/u~arm_b~bbb/2026-08-12/verdict.json"
        assert select_newest_settled_verdict(store) == expected

    def test_a_tie_on_trading_day_breaks_on_the_key_deterministically(self, store) -> None:
        store.put_bytes(
            "experiments/r~arm_a~aaa/2026-08-12/verdict.json",
            json.dumps({"trading_day": "2026-08-12"}).encode(),
        )
        store.put_bytes(
            "experiments/u~arm_z~zzz/2026-08-12/verdict.json",
            json.dumps({"trading_day": "2026-08-12"}).encode(),
        )
        # The larger key string wins, both ways round — the selection does
        # not depend on write order or store iteration order.
        assert (
            select_newest_settled_verdict(store)
            == "experiments/u~arm_z~zzz/2026-08-12/verdict.json"
        )

    def test_a_non_verdict_document_under_experiments_is_never_selected(self, store) -> None:
        store.put_bytes(
            "experiments/r~arm_a~aaa/2026-08-12/shadow.json",
            json.dumps({"trading_day": "2026-09-01"}).encode(),
        )
        store.put_bytes(
            "experiments/r~arm_a~aaa/2026-08-05/verdict.json",
            json.dumps({"trading_day": "2026-08-05"}).encode(),
        )
        expected = "experiments/r~arm_a~aaa/2026-08-05/verdict.json"
        assert select_newest_settled_verdict(store) == expected

    def test_no_verdict_anywhere_raises_rather_than_returning_none(self, store) -> None:
        store.put_bytes("experiments/r~arm_a~aaa/2026-08-12/shadow.json", b"{}")
        with pytest.raises(NoSettledVerdictError, match="no settled verdict.json"):
            select_newest_settled_verdict(store)

    def test_an_empty_store_raises_rather_than_returning_none(self, store) -> None:
        with pytest.raises(NoSettledVerdictError):
            select_newest_settled_verdict(store)


class TestExplainWalksAVerdict:
    """Plan §10.8, as the phase-1 gate measures it (`explain_walks_a_verdict`):
    an `explain` RUN whose manifest records a verdict.json as an input.

    Measured 2026-09-05 on the first replay arc: `crucible explain
    <verdict key>` on the box read "neither a run_id nor a key any run claims
    as an output" while the verdict sat in the store — `experiment.grade`
    wrote verdicts but claimed only the arena cycle as an output — and the
    handler wrote no manifest at all, so the clause had nothing to read
    either way.
    """

    def _graded_cycle(self, store, source, strategy_dir, cycle_date, tmp_path):
        from conftest import sessions_ending

        from crucible.config import Settings
        from crucible.slots import universe
        from crucible.slots.grading import DEFAULT_HORIZON_TRADING_DAYS

        settings = Settings(
            store_uri=str(tmp_path / "store"),
            arctic_bucket="unused",
            strategy_dir=strategy_dir,
            origins={},
        )
        sessions = sessions_ending(cycle_date, DEFAULT_HORIZON_TRADING_DAYS + 4)
        decision_days = sessions[:3]
        for day in [*decision_days, cycle_date]:
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
        for day in decision_days:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=day,
                discriminator="u",
            )
        grade = run_job(
            "experiment.grade",
            lambda c: universe.grade(c, settings=settings),
            store=store,
            trading_day=cycle_date,
            discriminator="u",
        )
        return grade, decision_days

    def test_the_grade_claims_every_verdict_it_writes_as_an_output(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        grade, _days = self._graded_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        verdicts = [o["key"] for o in grade.outputs if o["key"].endswith("/verdict.json")]
        assert verdicts, "a graded cycle wrote verdicts and claimed none of them"
        for key in verdicts:
            assert store.exists(key), key
        node = explain(store, verdicts[0])
        assert node.manifest is not None
        assert node.manifest["run_id"] == grade.run_id

    def test_explain_runs_through_run_job_and_records_the_verdict_it_walked(
        self, store, source, strategy_dir, cycle_date, tmp_path, monkeypatch
    ) -> None:
        import json

        from crucible.cli import main
        from crucible.keys import manifest_key

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        grade, _days = self._graded_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        verdict = next(o["key"] for o in grade.outputs if o["key"].endswith("/verdict.json"))
        day = cycle_date.isoformat()

        rc = main(
            ["explain", "--date", day, "--run-mode", "replay", "--store", str(store.root), verdict]
        )

        assert rc == 0
        manifest = json.loads(store.get_bytes(manifest_key("explain", day)))
        assert manifest["status"] == "ok"
        assert manifest["outputs"] == [], "explain changes nothing; it records what it walked"
        walked = [i["key"] for i in manifest["inputs"]]
        assert verdict in walked, walked
        assert any("verdict.json" in k for k in walked)

    def test_select_newest_verdict_walks_the_deterministic_target(
        self, store, source, strategy_dir, cycle_date, tmp_path, monkeypatch
    ) -> None:
        """`alpha-engine-config-I10858`: `--select-newest-verdict` (the
        scheduled arc stage's own flag, `crucible.weekly.
        SELECT_NEWEST_VERDICT_JOBS`) is equivalent, end to end, to an
        operator naming the same key explicitly — it walks it, records it as
        an input, and the manifest reads exactly as
        `test_explain_runs_through_run_job_and_records_the_verdict_it_walked`
        does for an explicit target."""
        import json

        from crucible.cli import main
        from crucible.explain import select_newest_settled_verdict
        from crucible.keys import manifest_key

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        grade, _days = self._graded_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        assert any(o["key"].endswith("/verdict.json") for o in grade.outputs)
        # The grade wrote more than one arm's verdict; the selection is by
        # `trading_day`, never "the first output listed" — computed
        # independently, the same way the scheduled arc stage's own call
        # will resolve it against this same store state.
        expected = select_newest_settled_verdict(store)
        day = cycle_date.isoformat()

        rc = main(
            [
                "explain",
                "--date",
                day,
                "--run-mode",
                "replay",
                "--store",
                str(store.root),
                "--select-newest-verdict",
            ]
        )

        assert rc == 0
        manifest = json.loads(store.get_bytes(manifest_key("explain", day)))
        assert manifest["status"] == "ok"
        walked = [i["key"] for i in manifest["inputs"]]
        assert expected in walked, walked

    def test_select_newest_verdict_and_an_explicit_target_are_mutually_exclusive(
        self, store, tmp_path, monkeypatch, cycle_date
    ) -> None:
        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        with pytest.raises(SystemExit, match="mutually exclusive"):
            main(
                [
                    "explain",
                    "--date",
                    cycle_date.isoformat(),
                    "--run-mode",
                    "replay",
                    "--store",
                    str(store.root),
                    "--select-newest-verdict",
                    "some/verdict.json",
                ]
            )

    def test_neither_a_target_nor_select_newest_verdict_is_refused(
        self, store, tmp_path, monkeypatch, cycle_date
    ) -> None:
        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        with pytest.raises(SystemExit, match="requires a target"):
            main(
                [
                    "explain",
                    "--date",
                    cycle_date.isoformat(),
                    "--run-mode",
                    "replay",
                    "--store",
                    str(store.root),
                ]
            )

    def test_dry_run_explain_prints_the_walk_and_files_nothing(
        self, store, source, strategy_dir, cycle_date, tmp_path, monkeypatch, capsys
    ) -> None:
        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        grade, _days = self._graded_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        verdict = next(o["key"] for o in grade.outputs if o["key"].endswith("/verdict.json"))
        before = sorted(store.list_keys())
        rc = main(
            [
                "explain",
                "--date",
                cycle_date.isoformat(),
                "--run-mode",
                "replay",
                "--store",
                str(store.root),
                "--dry-run",
                verdict,
            ]
        )
        assert rc == 0
        assert sorted(store.list_keys()) == before
        assert verdict in capsys.readouterr().out

    def test_an_unclaimed_key_still_fails_loud_with_a_failed_manifest(
        self, store, source, cycle_date, tmp_path, monkeypatch
    ) -> None:
        import json

        from crucible.cli import main
        from crucible.keys import manifest_key

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
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
            trading_day=cycle_date,
        )
        day = cycle_date.isoformat()
        with pytest.raises(KeyError, match="neither a run_id nor a key"):
            main(
                [
                    "explain",
                    "--date",
                    day,
                    "--run-mode",
                    "replay",
                    "--store",
                    str(store.root),
                    "nobody/wrote/this.json",
                ]
            )
        manifest = json.loads(store.get_bytes(manifest_key("explain", day)))
        assert manifest["status"] == "failed"
        assert "neither a run_id" in manifest["reason"]


def _money_path_manifest(
    run_id: str,
    *,
    job: str = "promote",
    outputs: list[str] | None = None,
    finished: str = "2026-08-31T13:04:11Z",
    trading_day: str = "2026-08-28",
) -> dict[str, Any]:
    """A conformant v2 manifest at the floor, plus whatever outputs a test
    needs — the same shape `tests/test_money_path_chain.py::_manifest` uses,
    duplicated here rather than imported across test modules."""
    from crucible.manifest import RUN_MANIFEST_SCHEMA_VERSION
    from crucible.store import sha256_hex

    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "job": job,
        "run_mode": "live",
        "trading_day": trading_day,
        "calendar_date": "2026-08-31",
        "status": "ok",
        "reason": "",
        "started": "2026-08-31T13:00:00Z",
        "finished": finished,
        "code_sha": "a" * 40,
        "release_sha": "b" * 40,
        "seed": 7,
        "inputs": [],
        "outputs": [
            {"key": k, "sha256": sha256_hex(k.encode()), "schema_version": "v1"}
            for k in (outputs or [])
        ],
        "rows_in": 0,
        "rows_out": 0,
        "rows_rejected": [],
        "cost_usd": 0.0,
        "llm_calls": [],
        "resource": {
            "instance_type": "local",
            "spot": False,
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 1.0,
            "disk_free_mb": 1.0,
        },
        "metrics": [],
        "attempts": [{"n": 1, "reason": "initial"}],
    }


def _write_money_path_manifest(store, manifest: dict[str, Any]) -> str:
    """Write ``manifest`` through the single writer; return its manifest key."""
    from crucible.manifest import write_manifest

    key = manifest_key(
        manifest["job"], manifest["trading_day"], discriminator=manifest.get("discriminator")
    )
    write_manifest(store, key, manifest)
    return key


class TestVerifyChainFlag:
    """`--verify-chain` (`alpha-engine-config-I10625`), wired to
    `crucible.explain`'s real `verify_money_path_chain` / `ChainVerification`
    / `Lineage.chain` (`crucible-PR240`, `alpha-engine-config-I10414`) — no
    monkeypatched stand-in: these tests build a real money-path chain with
    `crucible.manifest.write_manifest` and walk it through the CLI.
    """

    TRADING_DAY = "2026-08-28"

    def _chain(self, store) -> list[str]:
        from crucible.keys import champion_key, predictions_key

        return [
            _write_money_path_manifest(
                store,
                _money_path_manifest(
                    "01JG0000000000000000000001",
                    outputs=[champion_key("m")],
                    finished="2026-08-31T13:00:11Z",
                ),
            ),
            _write_money_path_manifest(
                store,
                _money_path_manifest(
                    "01JG0000000000000000000002",
                    job="experiment.run",
                    outputs=[predictions_key(self.TRADING_DAY)],
                    finished="2026-08-31T14:00:11Z",
                ),
            ),
        ]

    def test_verify_chain_is_silent_ok_when_the_walk_never_crossed_the_money_path(
        self, store, tmp_path, monkeypatch
    ) -> None:
        self._chain(store)
        report_key = "report/2026-08-28/attribution.json"
        _write_money_path_manifest(
            store,
            _money_path_manifest(
                "01JG000000000000000000000B",
                job="report",
                outputs=[report_key],
                finished="2026-08-31T15:00:11Z",
            ),
        )
        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        rc = main(
            [
                "explain",
                "--date",
                self.TRADING_DAY,
                "--run-mode",
                "replay",
                "--store",
                str(store.root),
                "--verify-chain",
                report_key,
            ]
        )
        assert rc == 0

    def test_verify_chain_exits_zero_on_an_intact_chain(self, store, tmp_path, monkeypatch) -> None:
        from crucible.cli import main
        from crucible.keys import predictions_key

        self._chain(store)
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        rc = main(
            [
                "explain",
                "--date",
                self.TRADING_DAY,
                "--run-mode",
                "replay",
                "--store",
                str(store.root),
                "--verify-chain",
                predictions_key(self.TRADING_DAY),
            ]
        )
        assert rc == 0

    def test_verify_chain_exits_non_zero_and_names_the_break_on_a_broken_chain(
        self, store, tmp_path, monkeypatch
    ) -> None:
        from crucible.cli import main
        from crucible.keys import predictions_key
        from crucible.manifest import MoneyPathChainError

        keys = self._chain(store)
        tampered = json.loads(store.get_bytes(keys[0]))
        tampered["seed"] = 999
        store.put_bytes(keys[0], json.dumps(tampered, indent=2, sort_keys=True).encode("utf-8"))

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        with pytest.raises(MoneyPathChainError, match="CHAIN BROKEN"):
            main(
                [
                    "explain",
                    "--date",
                    self.TRADING_DAY,
                    "--run-mode",
                    "replay",
                    "--store",
                    str(store.root),
                    "--verify-chain",
                    predictions_key(self.TRADING_DAY),
                ]
            )

    def test_verify_chain_omitted_never_raises_on_a_broken_chain(
        self, store, tmp_path, monkeypatch
    ) -> None:
        """Without the flag, a broken chain is never even consulted —
        `explain` prints the walk and exits 0 regardless, exactly as it does
        today over a store carrying no chain at all."""
        from crucible.cli import main
        from crucible.keys import predictions_key

        keys = self._chain(store)
        tampered = json.loads(store.get_bytes(keys[0]))
        tampered["seed"] = 999
        store.put_bytes(keys[0], json.dumps(tampered, indent=2, sort_keys=True).encode("utf-8"))

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        rc = main(
            [
                "explain",
                "--date",
                self.TRADING_DAY,
                "--run-mode",
                "replay",
                "--store",
                str(store.root),
                predictions_key(self.TRADING_DAY),
            ]
        )
        assert rc == 0


class TestExplainDryRunRecordsZeroMutations:
    """`--dry-run` writes nothing — asserted against a store that RECORDS
    every mutating call, the same contract shape `alpha-engine-config-I10576`
    /`crucible-PR225` established for `crucible gate`.

    `explain` itself is NOT a no-write read in the shape `crucible gate`
    became: `alpha-engine-config-I9757` (`crucible-PR115`, measured
    2026-09-05, `TestExplainWalksAVerdict`'s own docstring above) ruled the
    opposite way on purpose — the phase-1 gate clause
    `_clause_explain_walks_a_verdict` has nothing to read unless a real
    `explain` invocation files its manifest, so `explain` writes its own
    `runs/explain/{day}/run.json` like every other job (`AGENTS.md` rule 1)
    and reverting that would reopen I9757. `--dry-run` is rule 1's one
    declared exception, and this test pins zero mutating calls through it —
    not zero mutating calls unconditionally.
    """

    def test_dry_run_makes_no_mutating_call_at_all(
        self, source, cycle_date, tmp_path, monkeypatch
    ) -> None:
        from crucible.cli import main
        from crucible.store import LocalStore

        class RecordingStore(LocalStore):
            def __init__(self, root: Any) -> None:
                super().__init__(root)
                self.mutations: list[str] = []

            def put_bytes(self, key: str, payload: bytes, **kwargs: Any) -> Any:
                self.mutations.append(key)
                return super().put_bytes(key, payload, **kwargs)

            def compare_and_swap(self, key: str, expected: str, payload: bytes, **kw: Any) -> Any:
                self.mutations.append(key)
                return super().compare_and_swap(key, expected, payload, **kw)

        recorder = RecordingStore(tmp_path / "store")
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
            store=recorder,
            trading_day=cycle_date,
        )
        recorder.mutations.clear()

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        # `crucible.track_a` resolves its store through `Settings.store()`,
        # which builds one from the URI via `crucible.config.store_from_uri`
        # — patched here (not `open_store`, which `track_a` never calls) so
        # the RESOLVED store is the recorder, dry-run wrapping included.
        monkeypatch.setattr("crucible.config.store_from_uri", lambda uri: recorder)

        from crucible.keys import manifest_key

        manifest = json.loads(
            recorder.get_bytes(manifest_key("data.daily", cycle_date.isoformat()))
        )
        target = manifest["outputs"][0]["key"]
        rc = main(
            [
                "explain",
                "--date",
                cycle_date.isoformat(),
                "--run-mode",
                "replay",
                "--store",
                str(recorder.root),
                "--dry-run",
                target,
            ]
        )
        assert rc == 0
        assert recorder.mutations == []
