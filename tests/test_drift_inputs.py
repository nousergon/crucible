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
from crucible.drift import BANDS
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


class TestMarketWideVsCrossSectional:
    """`alpha-engine-config-I10071`: a market-wide column (one value repeated
    across every ticker on a day, by declared construction) is compared
    ALONG TIME against the trailing window's daily values, never as a
    cross-section against pooled cross-sections — the latter reads a point
    mass against 1-20 pooled point masses and BREACHES whatever the market
    did. A synthetic fixture panel, not the real catalogue, so the reference
    and current values are exact numbers this test controls."""

    #: A tiny catalogue: one market-wide column, one cross-sectional column,
    #: both real names from `crucible.features.CATALOG` (so `market_wide` is
    #: read off the SAME declaration production reads) but with the rest of
    #: the catalogue absent from the fixture frames on purpose — a smaller,
    #: exact fixture rather than the full synthetic price panel.
    _TICKERS = ("T000", "T001", "T002", "T003", "T004")

    def _write_day(
        self, store, day: dt.date, *, market_return: float, close_by_ticker: list[float]
    ) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "ticker": list(self._TICKERS),
                "close_raw": close_by_ticker,
                # Identical for every ticker, by construction — the market-wide
                # shape this fixture exists to exercise.
                "market_return_1d_log_return": [market_return] * len(self._TICKERS),
            }
        )
        store.put_bytes(features_key(DEFAULT_FEATURE_VERSION, day.isoformat()), frame.to_parquet())

    def test_a_market_wide_column_at_an_ordinary_value_reads_ok(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        from conftest import sessions_ending

        days = sessions_ending(FRIDAY, 21)
        # Small daily returns clustered near zero, and today's value sits
        # inside that same cluster — an ordinary trading day, not a regime
        # shift.
        rng = random.Random(11)
        for i, day in enumerate(days[:-1]):
            self._write_day(
                store,
                day,
                market_return=rng.gauss(0.0003, 0.0009),
                close_by_ticker=[100.0 + i + t for t in range(len(self._TICKERS))],
            )
        self._write_day(
            store,
            days[-1],
            market_return=0.0004,
            close_by_ticker=[103.0, 104.0, 105.0, 106.0, 107.0],
        )
        doc = compute_drift_inputs(store, days[-1]).features
        assert doc["method_by_feature"]["market_return_1d_log_return"].startswith("along-time")
        assert (
            doc["psi_by_feature"]["market_return_1d_log_return"]
            < BANDS["feature_psi_max_ratio"].watch
        ), doc

    def test_a_market_wide_column_with_a_regime_jump_reads_breach(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        from conftest import sessions_ending

        days = sessions_ending(FRIDAY, 21)
        rng = random.Random(11)
        for i, day in enumerate(days[:-1]):
            self._write_day(
                store,
                day,
                market_return=rng.gauss(0.0003, 0.0009),
                close_by_ticker=[100.0 + i + t for t in range(len(self._TICKERS))],
            )
        # Today: a genuine regime jump, an order of magnitude past anything
        # in the trailing window.
        self._write_day(
            store, days[-1], market_return=0.08, close_by_ticker=[103.0, 104.0, 105.0, 106.0, 107.0]
        )
        doc = compute_drift_inputs(store, days[-1]).features
        assert doc["method_by_feature"]["market_return_1d_log_return"].startswith("along-time")
        assert (
            doc["psi_by_feature"]["market_return_1d_log_return"]
            >= BANDS["feature_psi_max_ratio"].breach
        ), doc

    def test_a_cross_sectional_column_is_unaffected(self, tmp_path) -> None:
        """`close_raw` varies per ticker; it must still read the
        cross-sectional comparison and be unmoved by the market-wide fix."""
        store = LocalStore(tmp_path)
        from conftest import sessions_ending

        days = sessions_ending(FRIDAY, 21)
        rng = random.Random(13)
        for day in days[:-1]:
            self._write_day(
                store,
                day,
                market_return=rng.gauss(0.0003, 0.0009),
                close_by_ticker=[100.0 + rng.gauss(0.0, 2.0) for _ in self._TICKERS],
            )
        self._write_day(
            store, days[-1], market_return=0.0002, close_by_ticker=[99.0, 101.0, 100.0, 102.0, 98.0]
        )
        doc = compute_drift_inputs(store, days[-1]).features
        assert doc["method_by_feature"]["close_raw"].startswith("cross-sectional")
        # A cross-sectional column's reference is pooled ACROSS tickers and
        # days, not one point per day — orders of magnitude more reference
        # rows than the market-wide column's trailing-session count.
        assert doc["reference_rows_by_feature"]["close_raw"] > len(days) * len(self._TICKERS) / 2
        assert doc["reference_rows_by_feature"]["market_return_1d_log_return"] == len(days) - 1


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
