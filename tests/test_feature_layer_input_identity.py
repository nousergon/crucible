"""R and M read the SAME feature artifact — asserted from their manifests.

Normative source: plan §10 component 4, `alpha-engine-config-I9765`
deliverable 3, `-I9772` "Closes when".

The feature layer's whole reason to exist is that "the signal degraded" and
"the feature changed" must be separable from artifacts alone. That property
is not the existence of `features/{version}/{trading_day}.parquet` — it is
the identity of what the two consumers recorded reading. A layer both slots
read from *different versions*, or a consumer that recorded a key without its
hash, would satisfy every other test in this repository and lose exactly the
property §10.4 was written for.

So this file compares the `inputs[]` entry the R slot's produce path records
against the one the M slot's :class:`FeatureLayerSource` records, on the same
trading day, and requires them equal in all three fields.

It lives in the BLOCKING suite, not in `tests/acceptance/`: both producers
exist now, so a divergence here is a regression rather than an unbuilt
clause.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from crucible.data import run_daily
from crucible.features import CATALOG, DEFAULT_FEATURE_VERSION
from crucible.keys import features_key
from crucible.runner import run_job
from crucible.slots import research
from crucible.slots.cycle import MissingArtifactError
from crucible.slots.model import FeatureLayerSource

#: The four columns the `residual_momentum` M arm declares. Reconciled with
#: the catalogue by `alpha-engine-config-I9765`.
M_ARM_COLUMNS: tuple[str, ...] = (
    "residual_momentum_252d_skip21d_zscore",
    "residual_vol_20d_ratio",
    "momentum_change_21d_zscore",
    "beta_60d_raw",
)


@pytest.fixture
def r_strategy_dir(tmp_path):
    """Three R arms that do not share a ranking callable.

    Three because `min_active_arms` is 3; distinct callables because an arm
    sharing the champion's callable is `inapplicable` and refused at import.
    """
    root = tmp_path / "strategy-r"
    arms = root / "arms" / "r"
    arms.mkdir(parents=True)
    for name, ranker in (
        ("momentum_sleeve", "momentum_sleeve"),
        ("tech_score_gate", "tech_score_gate"),
        ("mom_12_1_sleeve", "mom_12_1_sleeve"),
    ):
        (arms / f"{name}.yaml").write_text(
            "\n".join(
                [
                    f"name: {name}",
                    "slot: r",
                    f"ranker: {ranker}",
                    "registered_at: '2026-06-01'",
                    "params:",
                    "  top_n: 8",
                    f"notes: fixture recipe for {name}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def _settings(strategy_dir, store_root) -> Any:
    from crucible.config import Settings

    return Settings(
        store_uri=str(store_root),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )


class TestTheTwoSlotsRecordTheSameInput:
    def test_the_m_input_entry_is_identical_to_the_r_input_entry(
        self, store, source, cycle_date: dt.date, r_strategy_dir, tmp_path
    ) -> None:
        run_job(
            "data.daily",
            lambda ctx: run_daily(ctx, source=source),
            store=store,
            trading_day=cycle_date,
        )
        settings = _settings(r_strategy_dir, tmp_path / "store")

        r_ctx = run_job(
            "experiment.run",
            lambda ctx: research.produce(ctx, settings=settings),
            store=store,
            trading_day=cycle_date,
        )
        r_inputs = [i for i in r_ctx.inputs if i["key"].startswith("features/")]
        assert len(r_inputs) == 1, "the R slot reads exactly one feature artifact per cycle"

        def m_slot_reads_its_columns(ctx: Any) -> None:
            """The M slot's produce path, reduced to the feature read."""
            FeatureLayerSource(store=store, registry=CATALOG).panel(
                trading_day=cycle_date.isoformat(),
                columns=M_ARM_COLUMNS,
                ctx=ctx,
            )

        m_ctx = run_job(
            "experiment.run",
            m_slot_reads_its_columns,
            store=store,
            trading_day=cycle_date,
        )
        m_inputs = [i for i in m_ctx.inputs if i["key"].startswith("features/")]
        assert len(m_inputs) == 1

        assert m_inputs[0] == r_inputs[0], (
            "the M slot and the R slot must record the SAME features/{version}/"
            "{trading_day} key, sha256 and schema version. Two slots reading two "
            "layers is exactly the condition that makes 'the signal degraded' "
            "indistinguishable from 'the feature changed' (plan §10.4)."
        )
        assert m_inputs[0]["key"] == features_key(DEFAULT_FEATURE_VERSION, cycle_date.isoformat())

    def test_neither_slot_names_a_version_of_its_own(self) -> None:
        """The version is DERIVED, so an edited catalogue moves both consumers.

        A consumer holding a hand-written `"v1"` would keep reading the old
        prefix after a recipe edit, and the two slots would silently diverge
        without either one failing.
        """
        assert FeatureLayerSource(store=None).version == DEFAULT_FEATURE_VERSION

    def test_the_m_slot_refuses_an_absent_layer_rather_than_recomputing(
        self, store, cycle_date: dt.date
    ) -> None:
        """No local fallback. The absence is named with the command that fixes it."""
        layer = FeatureLayerSource(store=store, registry=CATALOG)
        with pytest.raises(MissingArtifactError, match="feature layer is absent"):
            layer.panel(trading_day=cycle_date.isoformat(), columns=M_ARM_COLUMNS)
