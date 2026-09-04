"""A stacked M arm trains END TO END, through the production seam.

The 2026-09-01 adversarial review of `crucible-PR28` found that
`stack_prediction_columns` and `write_arm_predictions` had **zero production
callers**: a well-formed stacked arm registered cleanly and then died in
`train_arm` with `KeyError: feature 'predicted_alpha_base_raw' is not in this
panel`. That is `alpha-engine-config-I9777` reproduced one frame later — the
refusal moved from `FeatureLayerSource.panel()` to `FeaturePanel.column()`
and the defect did not move at all.

So this file drives the real path with no stubs anywhere: a real feature
layer on disk, the real loader, the real `design_panel` seam, the real
producer, and the real `train_arm`/`grade_arm`. Two arms, the second stacked
on the first. If the wiring is ever removed again, this file fails at the
line that trains, not at a line that asserts a call site exists.

Dates are fixed literals derived from one pinned start session, never `today`
arithmetic (AGENTS.md test discipline).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.slots.inputs import (
    BasePredictionsUnavailableError,
    UnproducibleInputError,
    UnresolvedInputError,
    arm_predictions_key,
    prediction_column,
    read_arm_predictions,
    stack_prediction_columns,
)
from crucible.slots.model import (
    FeatureLayerSource,
    design_panel,
    grade_arm,
    load_model_recipes,
    predict_cross_section,
    produce_arm_predictions,
    train_arm,
)
from crucible.store import LocalStore

#: The two catalogue columns the fixture arms declare. Real names from
#: `crucible.features.CATALOG`, so the default catalogue the loader resolves
#: is the one under test — a fixture catalogue would let a registration check
#: pass against a registry production never sees.
BASE_COLUMN = "momentum_20d_zscore"
STACKED_COLUMN = "volatility_20d_ratio"

#: 45 sessions from a pinned Monday. Weekends are dropped so the axis reads
#: like a session axis; the panel's axis is the store's keys either way.
_START = dt.date(2026, 6, 1)
_SESSIONS = 45
_NAMES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")


def _sessions() -> tuple[str, ...]:
    days: list[str] = []
    day = _START
    while len(days) < _SESSIONS:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    return tuple(days)


@pytest.fixture
def layer(tmp_path):
    """A real feature-layer store: parquet per session, under the real prefix."""
    import pandas as pd

    store = LocalStore(tmp_path)
    rng = np.random.default_rng(20260901)
    for i, day in enumerate(_sessions()):
        frame = pd.DataFrame(
            {
                "ticker": list(_NAMES),
                # A drifting positive price series, so forward returns are
                # finite and non-constant rather than a flat block the
                # trainability guard would (correctly) refuse.
                "close_raw": 100.0 + i * 0.5 + rng.normal(0.0, 1.0, len(_NAMES)),
                BASE_COLUMN: rng.normal(0.0, 1.0, len(_NAMES)),
                STACKED_COLUMN: rng.normal(1.0, 0.3, len(_NAMES)),
            }
        )
        store.put_bytes(
            f"features/{DEFAULT_FEATURE_VERSION}/{day}.parquet",
            frame.to_parquet(index=False),
        )
    return store


def _write_recipe(directory, name, *, features, inputs=()):
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["slot: m", f"name: {name}", "spec:", f"  features: [{', '.join(features)}]"]
    if inputs:
        lines.append("  inputs:")
        lines += [f"    - {entry}" for entry in inputs]
    lines += [
        "  estimator: {kind: ridge, alpha: 1.0}",
        "  label_horizon_trading_days: 2",
        "  refit_cadence_trading_days: 5",
        "  training_window: {kind: expanding, min_trading_days: 10}",
        "  cpcv: {n_groups: 4, k_test: 1, embargo_trading_days: 1}",
        "registered_at: '2026-06-01'",
    ]
    (directory / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def recipes(tmp_path):
    """`base`, and `stacked` which consumes `predictions[base]`."""
    arms = tmp_path / "strategy" / "arms" / "m"
    _write_recipe(arms, "base", features=[BASE_COLUMN])
    _write_recipe(arms, "stacked", features=[STACKED_COLUMN], inputs=("predictions[base]",))
    loaded = load_model_recipes(arms)
    assert loaded.refused == (), "both arms are producible; a refusal here is a fixture bug"
    return {r.name: r for r in loaded.registered}, loaded.registered


class _Ctx:
    """The two `RunContext` methods the M path uses, and nothing else."""

    def __init__(self, store):
        self.store = store
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []

    def record_input(self, key, payload, schema_version="v1") -> None:
        self.inputs.append({"key": key, "schema_version": schema_version})

    def record_output(self, key, payload, schema_version="v1") -> None:
        self.store.put_bytes(key, payload)
        self.outputs.append({"key": key, "schema_version": schema_version})


def _base_panel(layer, base, ctx=None):
    source = FeatureLayerSource(store=layer)
    return design_panel(
        base,
        source=source,
        trading_day=_sessions()[-1],
        lookback_trading_days=_SESSIONS - 1,
        ctx=ctx,
    )


def _produce_base_history(layer, base, *, ctx):
    """Fit `base` and write its cross-section for EVERY session on the panel."""
    panel = _base_panel(layer, base)
    fit = train_arm(base, panel, as_of=_sessions()[-1])
    written = [
        produce_arm_predictions(ctx, fit=fit, panel=panel, trading_day=day) for day in panel.dates
    ]
    return panel, fit, written


class TestTheStackedArmTrains:
    def test_a_stacked_arm_trains_and_grades_through_the_seam(self, layer, recipes) -> None:
        """The review's failing case, end to end, with nothing stubbed.

        Before this wiring existed the last line of this test raised
        `KeyError: feature 'predicted_alpha_base_raw' is not in this panel`.
        """
        by_name, loaded = recipes
        ctx = _Ctx(layer)
        _, base_fit, written = _produce_base_history(layer, by_name["base"], ctx=ctx)
        assert len(written) == _SESSIONS
        assert written[0] == arm_predictions_key(base_fit.arm_id, _sessions()[0])

        panel = design_panel(
            by_name["stacked"],
            source=FeatureLayerSource(store=layer),
            trading_day=_sessions()[-1],
            lookback_trading_days=_SESSIONS - 1,
            recipes=loaded,
            ctx=ctx,
        )
        column = prediction_column("base")
        assert column in panel.features, "the seam materialised the stacked column"
        assert panel.resolved_inputs == ("predictions[base]",)
        assert by_name["stacked"].design_columns == (STACKED_COLUMN, column)

        fit = train_arm(by_name["stacked"], panel, as_of=_sessions()[-1])
        assert fit.coefficients.shape == (2,), "one coefficient per design column"

        grade = grade_arm(by_name["stacked"], panel, as_of=_sessions()[-1])
        assert grade.status == "ok", grade.reason
        assert grade.oos_n > 0

    def test_row_t_carries_session_ts_base_prediction_on_the_real_path(
        self, layer, recipes
    ) -> None:
        """Point-in-time along the whole axis, checked against the artifacts."""
        by_name, loaded = recipes
        ctx = _Ctx(layer)
        base_panel, base_fit, _ = _produce_base_history(layer, by_name["base"], ctx=ctx)
        panel = design_panel(
            by_name["stacked"],
            source=FeatureLayerSource(store=layer),
            trading_day=_sessions()[-1],
            lookback_trading_days=_SESSIONS - 1,
            recipes=loaded,
        )
        stacked_column = panel.column(prediction_column("base"))
        for row, day in enumerate(panel.dates):
            expected = read_arm_predictions(layer, arm_id=base_fit.arm_id, trading_day=day)
            assert stacked_column[row, :].tolist() == [expected[n] for n in panel.names]
            assert expected == predict_cross_section(base_fit, base_panel, trading_day=day)

    def test_the_base_arms_reads_are_manifest_inputs_so_explain_walks_back(
        self, layer, recipes
    ) -> None:
        by_name, loaded = recipes
        _produce_base_history(layer, by_name["base"], ctx=_Ctx(layer))
        ctx = _Ctx(layer)
        design_panel(
            by_name["stacked"],
            source=FeatureLayerSource(store=layer),
            trading_day=_sessions()[-1],
            lookback_trading_days=_SESSIONS - 1,
            recipes=loaded,
            ctx=ctx,
        )
        base_id = by_name["base"].arm_id
        recorded = {i["key"] for i in ctx.inputs}
        assert {arm_predictions_key(base_id, d) for d in _sessions()} <= recorded


class TestTheGapCannotReopen:
    """The wiring defect, asserted as a refusal rather than as a call site."""

    def test_a_panel_built_without_the_seam_names_the_seam_not_the_parquet_layer(
        self, layer, recipes
    ) -> None:
        """The review's exact reproduction, now refused with an actionable message.

        A panel straight from `FeatureLayerSource.panel()` carries feature
        columns only. Training on it used to raise `KeyError` about a missing
        parquet column — sending an operator to the feature layer for a
        column the feature layer can never produce.
        """
        by_name, _ = recipes
        stacked = by_name["stacked"]
        raw = FeatureLayerSource(store=layer).panel(
            trading_day=_sessions()[-1],
            columns=(STACKED_COLUMN,),
            lookback_trading_days=_SESSIONS - 1,
            label_horizon_trading_days=stacked.label_horizon_trading_days,
        )
        assert raw.resolved_inputs == ()
        with pytest.raises(UnresolvedInputError) as excinfo:
            train_arm(stacked, raw, as_of=_sessions()[-1])
        message = str(excinfo.value)
        assert "design_panel" in message
        assert "predictions[base]" in message
        assert prediction_column("base") in message

    def test_a_base_arm_that_has_never_run_names_the_producer_and_the_command(
        self, layer, recipes
    ) -> None:
        """Registration proves the base arm EXISTS; only the store knows it ran.

        The review's second half of finding 1: a stacked arm over a long
        window needs one base artifact per session, and nothing said whether
        they existed. One refusal, with the count and the command.
        """
        by_name, loaded = recipes
        with pytest.raises(BasePredictionsUnavailableError) as excinfo:
            design_panel(
                by_name["stacked"],
                source=FeatureLayerSource(store=layer),
                trading_day=_sessions()[-1],
                lookback_trading_days=_SESSIONS - 1,
                recipes=loaded,
            )
        message = str(excinfo.value)
        assert f"{_SESSIONS} of {_SESSIONS} panel session(s)" in message
        assert "crucible experiment.run --slot m --arm base" in message

    def test_a_partial_base_history_is_refused_with_the_count_of_what_is_missing(
        self, layer, recipes
    ) -> None:
        by_name, loaded = recipes
        ctx = _Ctx(layer)
        _produce_base_history(layer, by_name["base"], ctx=ctx)
        gap = layer.root / arm_predictions_key(by_name["base"].arm_id, _sessions()[3])
        gap.unlink()
        with pytest.raises(BasePredictionsUnavailableError, match="1 of 45 panel session"):
            design_panel(
                by_name["stacked"],
                source=FeatureLayerSource(store=layer),
                trading_day=_sessions()[-1],
                lookback_trading_days=_SESSIONS - 1,
                recipes=loaded,
            )


class TestTheDeclaredNameBindsToTheArmActuallyRead:
    """Finding 2: `base_arm_ids` was caller-supplied and unverified."""

    def test_another_arms_predictions_cannot_be_stacked_under_a_declared_base(
        self, layer, recipes
    ) -> None:
        """The review's E3: `predicted_alpha_base_raw` carried a stranger's opinion."""
        by_name, loaded = recipes
        ctx = _Ctx(layer)
        _produce_base_history(layer, by_name["base"], ctx=ctx)
        panel = _base_panel(layer, by_name["base"])
        with pytest.raises(UnproducibleInputError) as excinfo:
            stack_prediction_columns(
                panel,
                store=layer,
                recipe=by_name["stacked"],
                base_arm_ids={"base": "m:totally_different_arm:deadbe"},
            )
        message = str(excinfo.value)
        assert "totally_different_arm" in message
        assert prediction_column("base") in message

    def test_the_seam_derives_the_id_so_a_caller_cannot_supply_a_wrong_one(
        self, layer, recipes
    ) -> None:
        """`design_panel` takes recipes, not ids — there is nothing to get wrong."""
        by_name, loaded = recipes
        _produce_base_history(layer, by_name["base"], ctx=_Ctx(layer))
        panel = design_panel(
            by_name["stacked"],
            source=FeatureLayerSource(store=layer),
            trading_day=_sessions()[-1],
            lookback_trading_days=_SESSIONS - 1,
            recipes=loaded,
        )
        assert prediction_column("base") in panel.features
        with pytest.raises(UnresolvedInputError, match="load_model_recipes"):
            design_panel(
                by_name["stacked"],
                source=FeatureLayerSource(store=layer),
                trading_day=_sessions()[-1],
                lookback_trading_days=_SESSIONS - 1,
                recipes=(by_name["stacked"],),
            )

    def test_an_id_that_is_not_slot_name_hash_is_refused(self, layer, recipes) -> None:
        by_name, _ = recipes
        panel = _base_panel(layer, by_name["base"])
        with pytest.raises(UnproducibleInputError, match="spec_hash"):
            stack_prediction_columns(
                panel,
                store=layer,
                recipe=by_name["stacked"],
                base_arm_ids={"base": "base"},
            )
