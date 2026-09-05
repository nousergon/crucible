"""An arm whose ranker reads a column the feature catalogue never declared is
refused BY NAME — per arm, as a value — and its siblings run.

The U/R half of `alpha-engine-config-I9955`. Measured 2026-09-05 on the
first phase-1 replay arc (`weekly@2026-08-07`, x86 box): `data.weekly` and
the U slot passed, then `experiment.run[r]` raised
`TrainingIntegrityError` for `scanner_predictor_direct`, whose ranker reads
`predicted_alpha_ratio` — a column the phase-1 catalogue has never carried
(the M slot materialises it, phase 3) and whose own recipe says it "refuses
BY NAME until then". One arm that could never have been produced took the
whole R slot, and with it the arc, down.

The distinction under test is the one plan §4.4 draws: a column the
catalogue DECLARES but today's frame lacks is a compromised input and still
fails the slot (`TestACatalogueDeclaredColumnMissingFromTheFrameStillFailsTheSlot`);
a column the catalogue never declared is a refusal at registration.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.data.daily import run_daily
from crucible.keys import arm_register_key, manifest_key, shadow_key
from crucible.runner import run_job
from crucible.slots import research
from crucible.slots.arms import load_arm_specs, read_register
from crucible.slots.cycle import ARM_REFUSED_METRIC, partition_by_catalog
from crucible.slots.inputs import SlotUnservableError


def _write_r_recipe(root, name: str, ranker: str, **params) -> None:
    arms = root / "arms" / "r"
    arms.mkdir(parents=True, exist_ok=True)
    body = [f"name: {name}", "slot: r", f"ranker: {ranker}", "registered_at: '2026-06-15'"]
    if params:
        body.append("params:")
        body += [f"  {k}: {v}" for k, v in params.items()]
    else:
        body.append("params: {}")
    body.append(f"notes: fixture recipe for {name}")
    (arms / f"{name}.yaml").write_text("\n".join(body) + "\n", encoding="utf-8")


def _settings(strategy_dir, store_root):
    from crucible.config import Settings

    return Settings(
        store_uri=str(store_root),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )


@pytest.fixture
def r_strategy_dir(strategy_dir):
    """The conftest U tree plus the two R shapes that matter: one arm fully
    expressible on the phase-1 catalogue, one that ranks on a predictor
    column the catalogue never declared."""
    _write_r_recipe(
        strategy_dir,
        "no_agent_quant",
        "quant_composite",
        top_n=15,
        momentum_weight=0.5,
        trend_weight=0.3,
        reversal_weight=0.2,
    )
    _write_r_recipe(strategy_dir, "scanner_predictor_direct", "predicted_alpha_direct", top_n=10)
    return strategy_dir


class TestPartitionByCatalog:
    def test_an_undeclared_column_is_a_refusal_naming_arm_and_column(self, r_strategy_dir):
        specs = load_arm_specs("r", strategy_dir=r_strategy_dir)
        from crucible.slots.rankers import get_ranker

        # exactly what the producible arm's ranker reads, and nothing the
        # predictor-ranked arm needs — the catalogue is the only discriminator
        producible, refused = partition_by_catalog(
            specs, catalog_columns=list(get_ranker("quant_composite").reads)
        )
        assert [s.name for s in producible] == ["no_agent_quant"]
        assert [r.arm for r in refused] == ["scanner_predictor_direct"]
        (refusal,) = refused
        assert refusal.unresolvable == ("predicted_alpha_ratio",)
        assert "predicted_alpha_direct" in refusal.reason
        assert "BY NAME" in refusal.reason

    def test_a_fully_declared_arm_is_producible(self, r_strategy_dir):
        specs = load_arm_specs("r", strategy_dir=r_strategy_dir)
        from crucible.features import CATALOG

        producible, refused = partition_by_catalog(specs, catalog_columns=[f.name for f in CATALOG])
        assert [s.name for s in producible] == ["no_agent_quant"]
        assert [r.arm for r in refused] == ["scanner_predictor_direct"]

    def test_an_unknown_ranker_still_raises_it_is_malformed_not_refused(self, strategy_dir):
        _write_r_recipe(strategy_dir, "ghost", "no_such_ranker", top_n=10)
        # The loader already refuses it (its spec hash would be the hash of a
        # recipe nothing can run); the partition never sees it. Either way it
        # is a raise, never an InputRefusal.
        with pytest.raises(KeyError, match="unknown ranker"):
            specs = load_arm_specs("r", strategy_dir=strategy_dir)
            partition_by_catalog(specs, catalog_columns=[])

    def test_the_metric_name_matches_the_m_slots(self):
        from crucible.slots.model import ARM_REFUSED_METRIC as M_NAME

        assert ARM_REFUSED_METRIC == M_NAME


class TestTheSlotRunsAroundTheRefusedArm:
    def test_siblings_produce_the_refused_arm_does_not_register_and_the_manifest_says_so(
        self, store, source, r_strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings = _settings(r_strategy_dir, tmp_path / "store")
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=source.symbols()),
            store=store,
            trading_day=cycle_date,
        )
        specs = {s.name: s for s in load_arm_specs("r", strategy_dir=r_strategy_dir)}

        ctx = run_job(
            "experiment.run",
            lambda c: research.produce(c, settings=settings),
            store=store,
            trading_day=cycle_date,
            discriminator="r",
        )

        day = cycle_date.isoformat()
        assert store.exists(shadow_key(specs["no_agent_quant"].arm_id, day))
        assert not store.exists(shadow_key(specs["scanner_predictor_direct"].arm_id, day))

        register = read_register(store, "r")
        assert specs["no_agent_quant"].arm_id in register.all_arms()
        assert specs["scanner_predictor_direct"].arm_id not in register.all_arms(), (
            "a refused arm must not register: `arms_all_scored` demands every ACTIVE "
            "registered arm be scored, and an arm that can never produce would hold the "
            "phase gate red forever"
        )
        assert store.exists(arm_register_key("r"))

        manifest = json.loads(
            store.get_bytes(manifest_key("experiment.run", day, discriminator="r"))
        )
        assert manifest["status"] == "ok"
        refusals = [m for m in manifest["metrics"] if m["name"] == ARM_REFUSED_METRIC]
        assert len(refusals) == 1
        assert refusals[0]["status"] == "unservable"
        assert "scanner_predictor_direct" in refusals[0]["status_reason"]
        assert "predicted_alpha_ratio" in refusals[0]["status_reason"]
        assert ctx is not None

    def test_a_slot_with_every_arm_refused_is_unservable_and_fails_loud(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        _write_r_recipe(
            strategy_dir, "scanner_predictor_direct", "predicted_alpha_direct", top_n=10
        )
        settings = _settings(strategy_dir, tmp_path / "store")
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=source.symbols()),
            store=store,
            trading_day=cycle_date,
        )
        with pytest.raises(SlotUnservableError):
            run_job(
                "experiment.run",
                lambda c: research.produce(c, settings=settings),
                store=store,
                trading_day=cycle_date,
                discriminator="r",
            )
        manifest = json.loads(
            store.get_bytes(
                manifest_key("experiment.run", cycle_date.isoformat(), discriminator="r")
            )
        )
        assert manifest["status"] == "failed"
        assert "SlotUnservableError" in manifest["reason"]


class TestACatalogueDeclaredColumnMissingFromTheFrameStillFailsTheSlot:
    def test_compromised_input_is_not_softened_into_a_refusal(self, r_strategy_dir) -> None:
        """The catalogue declares the column, the frame lacks it: that is the
        `TrainingIntegrityError` path, untouched. Shown at the partition
        level — the arm is PRODUCIBLE (not refused) when the column is in the
        catalogue, so the only thing that can stop it later is the frame."""
        specs = [
            s
            for s in load_arm_specs("r", strategy_dir=r_strategy_dir)
            if s.name == "no_agent_quant"
        ]
        from crucible.features import CATALOG

        producible, refused = partition_by_catalog(specs, catalog_columns=[f.name for f in CATALOG])
        assert producible and not refused

    def test_the_frame_level_refusal_is_still_training_integrity(self, cycle_date) -> None:
        import pandas as pd

        from crucible.slots.rankers import MissingFeatureError, get_ranker

        frame = pd.DataFrame({"liquidity_pass_raw": [1.0]}, index=["AAA"])
        with pytest.raises(MissingFeatureError):
            get_ranker("quant_composite").score(frame, {})
        assert cycle_date == dt.date(2026, 8, 28)
