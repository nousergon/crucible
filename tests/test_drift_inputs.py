"""`drift` computes its three inputs from the store; it does not read documents
nobody writes.

Measured 2026-09-05 on the first phase-1 replay arc to reach `drift`
(weekly@2026-08-07, x86 box): `FileNotFoundError: drift inputs absent ...
['features', 'ic', 'predictions']`. No job in this repository had ever
written `drift/{day}/input_*.json`; the only reference to
`drift_input_key` outside `keys.py` was the reader.
"""

from __future__ import annotations

import datetime as dt
import json
import random

import pytest

from crucible.data.daily import run_daily
from crucible.drift_inputs import (
    DRIFT_INPUT_SCHEMA_VERSION,
    compute_drift_inputs,
    feature_days,
)
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import (
    cross_section_key,
    cross_section_settled_key,
    drift_input_key,
    features_key,
    manifest_key,
)
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.arms import load_arm_specs
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)


def _settings(strategy_dir, store_root):
    from crucible.config import Settings

    return Settings(
        store_uri=str(store_root),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )


def _compile(store, source, days):
    for day in days:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=source.symbols()),
            store=store,
            trading_day=day,
        )


class TestFeatureDays:
    def test_lists_only_dated_frames_ascending(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        for day in ("2026-08-05", "2026-08-03", "2026-08-04"):
            store.put_bytes(features_key(DEFAULT_FEATURE_VERSION, day), b"x")
        store.put_bytes(f"features/{DEFAULT_FEATURE_VERSION}/registry.json", b"{}")
        assert [d.isoformat() for d in feature_days(store, DEFAULT_FEATURE_VERSION)] == [
            "2026-08-03",
            "2026-08-04",
            "2026-08-05",
        ]

    def test_an_undated_parquet_under_the_prefix_is_a_writer_defect(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(f"features/{DEFAULT_FEATURE_VERSION}/latest.parquet", b"x")
        with pytest.raises(ValueError, match="not a dated frame"):
            feature_days(store, DEFAULT_FEATURE_VERSION)


class TestFeaturesInput:
    def test_an_absent_feature_layer_for_the_day_raises_nothing_to_measure(self, tmp_path):
        store = LocalStore(tmp_path)
        with pytest.raises(FileNotFoundError, match="nothing to measure"):
            compute_drift_inputs(store, FRIDAY)

    def test_the_first_compiled_day_is_unmeasured_with_a_reason(
        self, store, source, cycle_date
    ) -> None:
        _compile(store, source, [cycle_date])
        computed = compute_drift_inputs(store, cycle_date)
        assert computed.features["psi_by_feature"] == {}
        assert "no compiled feature day precedes" in computed.features["unmeasured_reason"]
        assert computed.features["reference_sessions"] == []

    def test_psi_is_computed_per_catalogue_column_against_the_trailing_window(
        self, store, source, cycle_date
    ) -> None:
        from conftest import sessions_ending

        days = sessions_ending(cycle_date, 4)
        _compile(store, source, days)
        computed = compute_drift_inputs(store, cycle_date)
        doc = computed.features
        assert doc["schema_version"] == DRIFT_INPUT_SCHEMA_VERSION
        assert doc["reference_sessions"] == [d.isoformat() for d in days[:-1]]
        assert doc["psi_by_feature"], "at least one catalogue column was compared"
        for name, value in doc["psi_by_feature"].items():
            assert value >= 0.0, name
            assert doc["reference_rows_by_feature"][name] > 0
        # the frames it read are the manifest's lineage
        assert features_key(DEFAULT_FEATURE_VERSION, cycle_date.isoformat()) in computed.sources
        for day in days[:-1]:
            assert features_key(DEFAULT_FEATURE_VERSION, day.isoformat()) in computed.sources


class TestPredictionsInput:
    def test_no_cross_sections_is_unmeasured_with_a_reason(self, store, source, cycle_date):
        _compile(store, source, [cycle_date])
        doc = compute_drift_inputs(store, cycle_date).predictions
        assert doc["psi"] is None
        assert "no arm has both" in doc["unmeasured_reason"]

    def test_worst_arm_psi_against_its_own_earlier_cross_section(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        from conftest import sessions_ending

        settings = _settings(strategy_dir, tmp_path / "store")
        days = sessions_ending(cycle_date, 3)
        _compile(store, source, days)
        for day in days:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=day,
            )
        doc = compute_drift_inputs(store, cycle_date).predictions
        arms = {s.arm_id for s in load_arm_specs("u", strategy_dir=strategy_dir)}
        assert set(doc["psi_by_arm"]) == arms
        assert doc["worst_arm"] in arms
        assert doc["psi"] == max(doc["psi_by_arm"].values())
        for arm_id in arms:
            assert doc["reference_dates_by_arm"][arm_id] == [d.isoformat() for d in days[:-1]]
        assert "unmeasured_reason" not in doc

    def test_a_shifted_cross_section_reads_as_drift(self, store, source, cycle_date) -> None:
        """Synthetic: the same arm, an earlier cross-section drawn from one
        distribution and today's from a shifted one, must breach the band —
        a prediction PSI that cannot move is worse than none."""
        from crucible.slots.arms import ArmRegister, write_register

        _compile(store, source, [cycle_date])
        register = ArmRegister()
        register, record = register.register(
            slot="u", name="probe", spec={"x": 1}, created_date="2026-06-01"
        )
        arm_id = record.arm_id
        write_register(store, "u", register)
        rng = random.Random(7)
        earlier = dt.date(2026, 8, 21)
        for day, shift in ((earlier, 0.0), (cycle_date, 3.0)):
            ranks = [
                {"ticker": f"T{i:03d}", "score": rng.gauss(shift, 1.0), "rank": i + 1}
                for i in range(300)
            ]
            store.put_bytes(
                cross_section_key(arm_id, day.isoformat()),
                json.dumps(
                    {
                        "schema_version": "cross_section.v2",
                        "arm_id": arm_id,
                        "trading_day": day.isoformat(),
                        "population_size": 300,
                        "ranks": ranks,
                    }
                ).encode(),
            )
        doc = compute_drift_inputs(store, cycle_date).predictions
        assert doc["psi_by_arm"][arm_id] > 0.25, doc


class TestICInput:
    def _settled(self, store, arm_id, day, horizon, slope):
        rng = random.Random(day.toordinal())
        ranks = []
        for i in range(200):
            score = float(i)
            ranks.append(
                {
                    "ticker": f"T{i:03d}",
                    "score": score,
                    "rank": 200 - i,
                    "realized_forward_return_ratio": slope * score + rng.gauss(0, 30.0),
                }
            )
        store.put_bytes(
            cross_section_settled_key(arm_id, day.isoformat()),
            json.dumps(
                {
                    "schema_version": "cross_section_settled.v1",
                    "arm_id": arm_id,
                    "trading_day": day.isoformat(),
                    "population_size": 200,
                    "n_settled": 200,
                    "horizon_trading_days": horizon,
                    "settled_on": "2026-09-30",
                    "ranks": ranks,
                }
            ).encode(),
        )

    def test_nothing_settled_is_unmeasured_with_a_reason(self, store, source, cycle_date):
        _compile(store, source, [cycle_date])
        doc = compute_drift_inputs(store, cycle_date).ic
        assert doc["decay_by_horizon"] == {}
        assert "settlement needs horizon_trading_days" in doc["unmeasured_reason"]

    def test_decay_is_latest_against_the_mean_of_earlier_settled_dates(
        self, store, source, cycle_date
    ) -> None:
        from crucible.slots.arms import ArmRegister, write_register

        _compile(store, source, [cycle_date])
        register = ArmRegister()
        register, record = register.register(
            slot="u", name="probe", spec={"x": 1}, created_date="2026-06-01"
        )
        arm_id = record.arm_id
        write_register(store, "u", register)
        # two strong earlier dates, one weak latest date -> most of the edge lost
        self._settled(store, arm_id, dt.date(2026, 7, 31), 21, slope=1.0)
        self._settled(store, arm_id, dt.date(2026, 8, 7), 21, slope=1.0)
        self._settled(store, arm_id, dt.date(2026, 8, 14), 21, slope=0.05)
        doc = compute_drift_inputs(store, cycle_date).ic
        assert "21" in doc["decay_by_horizon"]
        assert doc["decay_by_horizon"]["21"] > 0.5, doc
        assert doc["worst_arm_by_horizon"]["21"] == arm_id
        assert doc["baseline_ic_by_arm"][arm_id]["21"] > doc["current_ic_by_arm"][arm_id]["21"]

    def test_a_single_settled_date_is_too_few_not_a_zero(self, store, source, cycle_date):
        from crucible.slots.arms import ArmRegister, write_register

        _compile(store, source, [cycle_date])
        register = ArmRegister()
        register, record = register.register(
            slot="u", name="probe", spec={"x": 1}, created_date="2026-06-01"
        )
        arm_id = record.arm_id
        write_register(store, "u", register)
        self._settled(store, arm_id, dt.date(2026, 8, 7), 21, slope=1.0)
        doc = compute_drift_inputs(store, cycle_date).ic
        assert doc["decay_by_horizon"] == {}
        assert doc["settled_dates_too_few_by_arm"][arm_id]["21"] == 1


class TestTheHandlerFilesWhatItComputed:
    def test_drift_writes_its_three_inputs_and_three_rows_and_ends_ok(
        self, store, source, strategy_dir, cycle_date, tmp_path, monkeypatch
    ) -> None:
        from conftest import sessions_ending

        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        settings = _settings(strategy_dir, tmp_path / "store")
        days = sessions_ending(cycle_date, 3)
        _compile(store, source, days)
        for day in days:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=day,
            )
        day = cycle_date.isoformat()
        argv = ["drift", "--date", day, "--run-mode", "replay", "--store", str(store.root)]
        assert main(argv) == 0
        for name in ("features", "predictions", "ic"):
            assert store.exists(drift_input_key(name, day)), name
        manifest = json.loads(store.get_bytes(manifest_key("drift", day)))
        assert manifest["status"] == "ok"
        names = {m["name"] for m in manifest["metrics"]}
        assert names == {"feature_psi_max_ratio", "prediction_psi_ratio", "ic_decay_ratio"}
        by_name = {m["name"]: m for m in manifest["metrics"]}
        assert by_name["feature_psi_max_ratio"]["status"] in {"OK", "WATCH", "BREACH"}
        assert by_name["prediction_psi_ratio"]["status"] in {"OK", "WATCH", "BREACH"}
        assert by_name["ic_decay_ratio"]["status"] == "UNREPORTED"
        assert "horizon_trading_days" in by_name["ic_decay_ratio"]["status_reason"]
        output_keys = {o["key"] for o in manifest["outputs"]}
        assert {drift_input_key(n, day) for n in ("features", "predictions", "ic")} <= output_keys
        input_keys = {i["key"] for i in manifest["inputs"]}
        assert features_key(DEFAULT_FEATURE_VERSION, day) in input_keys

    def test_dry_run_computes_but_writes_nothing(
        self, store, source, cycle_date, tmp_path, monkeypatch
    ) -> None:
        from conftest import sessions_ending

        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        _compile(store, source, sessions_ending(cycle_date, 2))
        before = sorted(store.list_keys())
        argv = [
            "drift",
            "--date",
            cycle_date.isoformat(),
            "--run-mode",
            "replay",
            "--store",
            str(store.root),
            "--dry-run",
        ]
        assert main(argv) == 0
        assert sorted(store.list_keys()) == before

    def test_an_absent_feature_layer_fails_the_job_with_the_reason_on_the_manifest(
        self, tmp_path, monkeypatch
    ) -> None:
        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path / "s")
        day = FRIDAY.isoformat()
        with pytest.raises(FileNotFoundError, match="nothing to measure"):
            main(["drift", "--date", day, "--run-mode", "replay", "--store", str(store.root)])
        manifest = json.loads(store.get_bytes(manifest_key("drift", day)))
        assert manifest["status"] == "failed"
        assert "nothing to measure" in manifest["reason"]
