"""Stacked M arms: the predictions-input contract (`alpha-engine-config-I9777`).

The defect these tests lock down was measured on 2026-09-01:
`sota_directional_combine` declared `gbm_directional_score_zscore` and
`sentiment_directional_score_zscore` — two columns that are another model's
OUTPUT and have no feature-layer producer — and the recipe **loaded**.
`load_model_recipes` validated shape, not producibility, so the arm registered
cleanly and then raised deep inside `FeatureLayerSource.panel()` at grading
time. Registered fine, could never be graded.

Every test below asserts a REFUSAL, because an acceptance-shaped test that
only shows the loader taking a well-formed file has not shown it constrains
anything (AGENTS.md, test discipline).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crucible.slots.inputs import (
    ARM_PREDICTIONS_SCHEMA_VERSION,
    ArmPredictionsContractError,
    InputCycleError,
    InputRef,
    UnproducibleInputError,
    arm_predictions_key,
    parse_input_ref,
    prediction_column,
    read_arm_predictions,
    stack_prediction_columns,
    write_arm_predictions,
)
from crucible.slots.model import FeaturePanel, load_model_recipes
from crucible.store import LocalStore

_LAYER = ("mom_21d_ratio", "vol_21d_ratio")


def _write(directory: Path, name: str, *, features: str, inputs: tuple[str, ...] = ()) -> None:
    """A recipe file, written the way the private strategy tree writes one.

    `inputs` entries go in BLOCK form. `predictions[base]` is a legal plain
    scalar in a block sequence and an illegal one inside a flow sequence, so
    the shape the tree actually uses is the shape tested here.
    """
    lines = [
        "slot: m",
        f"name: {name}",
        "spec:",
        f"  features: {features}",
    ]
    if inputs:
        lines.append("  inputs:")
        lines += [f"    - {entry}" for entry in inputs]
    lines += [
        "  estimator: {kind: ridge, alpha: 1.0}",
        "  label_horizon_trading_days: 21",
        "  refit_cadence_trading_days: 5",
        "  training_window: {kind: expanding, min_trading_days: 504}",
        "  cpcv: {n_groups: 6, k_test: 2, embargo_trading_days: 2}",
        "registered_at: '2026-06-01'",
    ]
    (directory / f"{name}.yaml").write_text("\n".join(lines), encoding="utf-8")


class TestTheInputGrammar:
    def test_a_bare_column_name_is_refused_rather_than_assumed_to_be_a_feature(self) -> None:
        with pytest.raises(UnproducibleInputError, match="not a typed reference"):
            parse_input_ref("gbm_directional_score_zscore")

    def test_an_unknown_kind_is_refused_by_name(self) -> None:
        with pytest.raises(UnproducibleInputError, match="input kind 'signals'"):
            parse_input_ref("signals[thinktank]")

    def test_a_typed_reference_parses_to_its_kind_and_its_design_column(self) -> None:
        ref = parse_input_ref("predictions[gbm_directional]")
        assert ref == InputRef(kind="predictions", ref="gbm_directional")
        assert ref.column == prediction_column("gbm_directional")
        assert ref.column.endswith("_raw"), "a design column carries an explicit units suffix"


class TestRegistrationRefusesWhatCannotBeProduced:
    """Deliverable 3: the failure moves from grading to registration."""

    def test_a_model_output_column_declared_as_a_feature_does_not_register(self, tmp_path) -> None:
        """The measured I9777 condition, verbatim: this used to LOAD."""
        _write(
            tmp_path,
            "sota_directional_combine",
            features="[mom_21d_ratio, gbm_directional_score_zscore]",
        )
        with pytest.raises(UnproducibleInputError) as excinfo:
            load_model_recipes(tmp_path, feature_columns=_LAYER)
        message = str(excinfo.value)
        assert "sota_directional_combine" in message, "the refusal names the ARM"
        assert "gbm_directional_score_zscore" in message, "the refusal names the COLUMN"

    def test_a_features_input_the_layer_does_not_produce_is_refused_too(self, tmp_path) -> None:
        """Both declaration channels are checked, not one of two."""
        _write(
            tmp_path,
            "two_channel",
            features="[mom_21d_ratio]",
            inputs=("features[nowhere_ratio]",),
        )
        with pytest.raises(UnproducibleInputError, match="nowhere_ratio"):
            load_model_recipes(tmp_path, feature_columns=_LAYER)

    def test_a_prediction_input_naming_no_registered_arm_does_not_register(self, tmp_path) -> None:
        _write(
            tmp_path,
            "stacked",
            features="[mom_21d_ratio]",
            inputs=("predictions[gbm_directional]",),
        )
        with pytest.raises(UnproducibleInputError) as excinfo:
            load_model_recipes(tmp_path, feature_columns=_LAYER)
        assert "stacked" in str(excinfo.value)
        assert "gbm_directional" in str(excinfo.value)

    def test_a_column_declared_twice_across_features_and_inputs_is_refused(self, tmp_path) -> None:
        _write(
            tmp_path,
            "doubled",
            features="[mom_21d_ratio]",
            inputs=("features[mom_21d_ratio]",),
        )
        with pytest.raises(ValueError, match="more than once"):
            load_model_recipes(tmp_path, feature_columns=_LAYER)

    def test_a_well_formed_stacked_arm_registers_and_carries_both_column_kinds(
        self, tmp_path
    ) -> None:
        _write(tmp_path, "base", features="[mom_21d_ratio]")
        _write(
            tmp_path,
            "stacked",
            features="[vol_21d_ratio]",
            inputs=("predictions[base]",),
        )
        loaded = load_model_recipes(tmp_path, feature_columns=_LAYER)
        assert loaded.refused == ()
        recipes = {r.name: r for r in loaded.registered}
        assert recipes["stacked"].design_columns == (
            "vol_21d_ratio",
            prediction_column("base"),
        )
        assert recipes["stacked"].spec["inputs"] == ["predictions[base]"]


class TestTheCycleGuard:
    """Deliverable 4."""

    def test_an_arm_cannot_declare_itself_as_an_input(self, tmp_path) -> None:
        _write(
            tmp_path,
            "ouroboros",
            features="[mom_21d_ratio]",
            inputs=("predictions[ouroboros]",),
        )
        with pytest.raises(InputCycleError, match="ouroboros -> ouroboros"):
            load_model_recipes(tmp_path, feature_columns=_LAYER)

    def test_a_two_arm_cycle_is_refused_and_the_message_names_the_path(self, tmp_path) -> None:
        _write(tmp_path, "alpha", features="[mom_21d_ratio]", inputs=("predictions[beta]",))
        _write(tmp_path, "beta", features="[vol_21d_ratio]", inputs=("predictions[alpha]",))
        with pytest.raises(InputCycleError) as excinfo:
            load_model_recipes(tmp_path, feature_columns=_LAYER)
        assert "alpha -> beta -> alpha" in str(excinfo.value)

    def test_a_three_arm_chain_that_closes_is_refused(self, tmp_path) -> None:
        _write(tmp_path, "a1", features="[mom_21d_ratio]", inputs=("predictions[a2]",))
        _write(tmp_path, "a2", features="[vol_21d_ratio]", inputs=("predictions[a3]",))
        _write(tmp_path, "a3", features="[mom_21d_ratio]", inputs=("predictions[a1]",))
        with pytest.raises(InputCycleError):
            load_model_recipes(tmp_path, feature_columns=_LAYER)

    def test_a_diamond_is_not_a_cycle(self, tmp_path) -> None:
        """Two arms stacking on one base is legitimate; only a CYCLE is not."""
        _write(tmp_path, "base", features="[mom_21d_ratio]")
        _write(tmp_path, "left", features="[vol_21d_ratio]", inputs=("predictions[base]",))
        _write(tmp_path, "right", features="[mom_21d_ratio]", inputs=("predictions[base]",))
        loaded = load_model_recipes(tmp_path, feature_columns=_LAYER)
        assert [r.name for r in loaded.registered] == ["base", "left", "right"]
        assert loaded.refused == ()


class TestTheArmIdDoesNotMoveForArmsWithNoInputs:
    def test_an_existing_recipes_id_is_unchanged_by_this_field_existing(self) -> None:
        """A key that always appeared would orphan every registered series.

        The literal is the id derived before `inputs` existed (measured
        2026-09-01), and re-pinned 2026-09-02 (`alpha-engine-config-I9801`)
        when `feature_version` left the hashed spec — an intentional, one-time
        move since it changes what the id is a hash OF, not the presence of
        `inputs`, which is what this test actually pins. It is pinned rather
        than recomputed: a test that recomputes the hash with the code under
        test proves only that the code agrees with itself.
        """
        from tests.test_slot_model import _recipe

        recipe = _recipe()
        assert recipe.arm_id == "m:residual_momentum:ab013b0f2225"
        assert "inputs" not in recipe.spec

    def test_declaring_an_input_produces_a_different_arm(self) -> None:
        from tests.test_slot_model import _recipe

        stacked = _recipe(inputs=(InputRef(kind="predictions", ref="base"),))
        assert stacked.arm_id != _recipe().arm_id


class TestThePredictionsArtifactContract:
    """Deliverable 5: a versioned schema with both halves tested at birth."""

    def test_the_key_is_per_arm_and_carries_the_arms_spec_hash(self) -> None:
        """Under `arm_predictions/`, not `predictions/` — the latter is the
        trader's champion serving feed and the two shared a prefix until
        `alpha-engine-config-I9822`."""
        key = arm_predictions_key("m:base:abc123", "2026-08-28")
        assert key == "arm_predictions/m~base~abc123/2026-08-28.json"

    def test_the_producer_output_round_trips_through_the_consumer(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        ctx = _Ctx(store)
        key = write_arm_predictions(
            ctx,
            arm_id="m:base:abc123",
            trading_day="2026-08-28",
            feature_version="v1",
            predicted_alpha={"AAPL": 0.01, "MSFT": -0.02},
        )
        assert ctx.outputs[0]["schema_version"] == ARM_PREDICTIONS_SCHEMA_VERSION
        assert ctx.outputs[0]["key"] == key
        assert read_arm_predictions(store, arm_id="m:base:abc123", trading_day="2026-08-28") == {
            "AAPL": 0.01,
            "MSFT": -0.02,
        }

    def test_the_producer_refuses_to_write_an_empty_cross_section(self, tmp_path) -> None:
        with pytest.raises(ArmPredictionsContractError):
            write_arm_predictions(
                _Ctx(LocalStore(tmp_path)),
                arm_id="m:base:abc123",
                trading_day="2026-08-28",
                feature_version="v1",
                predicted_alpha={},
            )

    def test_a_document_from_another_trading_day_is_refused_not_trained_on(self, tmp_path) -> None:
        """The point-in-time guarantee, as a property of the code.

        The key names the session and the document repeats it. A payload for
        a LATER session sitting under this key is look-ahead; the consumer
        refuses rather than reading it, so a base arm's opinion about a
        future day is unreachable rather than merely unused.
        """
        store = LocalStore(tmp_path)
        store.put_bytes(
            arm_predictions_key("m:base:abc123", "2026-08-28"),
            json.dumps(
                {
                    "schema_version": ARM_PREDICTIONS_SCHEMA_VERSION,
                    "arm_id": "m:base:abc123",
                    "trading_day": "2026-09-04",
                    "feature_version": "v1",
                    "predicted_alpha": {"AAPL": 0.9},
                }
            ).encode("utf-8"),
        )
        with pytest.raises(ArmPredictionsContractError, match="2026-09-04"):
            read_arm_predictions(store, arm_id="m:base:abc123", trading_day="2026-08-28")

    def test_a_misfiled_document_from_another_arm_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            arm_predictions_key("m:base:abc123", "2026-08-28"),
            json.dumps(
                {
                    "schema_version": ARM_PREDICTIONS_SCHEMA_VERSION,
                    "arm_id": "m:other:def456",
                    "trading_day": "2026-08-28",
                    "feature_version": "v1",
                    "predicted_alpha": {"AAPL": 0.9},
                }
            ).encode("utf-8"),
        )
        with pytest.raises(ArmPredictionsContractError, match="m:other:def456"):
            read_arm_predictions(store, arm_id="m:base:abc123", trading_day="2026-08-28")

    def test_a_nonconforming_document_is_refused_on_read(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            arm_predictions_key("m:base:abc123", "2026-08-28"),
            json.dumps({"schema_version": "arm_predictions.v1"}).encode("utf-8"),
        )
        with pytest.raises(ArmPredictionsContractError, match="does not conform"):
            read_arm_predictions(store, arm_id="m:base:abc123", trading_day="2026-08-28")

    def test_an_absent_prediction_raises_rather_than_returning_none(self, tmp_path) -> None:
        with pytest.raises(KeyError):
            read_arm_predictions(
                LocalStore(tmp_path), arm_id="m:base:abc123", trading_day="2026-08-28"
            )


class TestStackingOntoAPanel:
    """Deliverables 1 and 2: same-day resolution, and one lineage entry per arm."""

    def test_each_panel_row_reads_its_own_sessions_prediction(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        panel = _panel()
        for day, value in zip(panel.dates, (0.1, 0.2, 0.3), strict=True):
            write_arm_predictions(
                _Ctx(store),
                arm_id="m:base:abc123",
                trading_day=day,
                feature_version="v1",
                predicted_alpha={n: value for n in panel.names},
            )
        ctx = _Ctx(store)
        stacked = stack_prediction_columns(
            panel,
            store=store,
            recipe=_StubRecipe((InputRef(kind="predictions", ref="base"),)),
            base_arm_ids={"base": "m:base:abc123"},
            ctx=ctx,
        )
        column = stacked.column(prediction_column("base"))
        assert column[:, 0].tolist() == [0.1, 0.2, 0.3], (
            "row t carries session t's prediction — a single anchor-day read would "
            "put one day's opinion on every training row"
        )
        assert [i["key"] for i in ctx.inputs] == [
            arm_predictions_key("m:base:abc123", d) for d in panel.dates
        ], "every read is a manifest input, so `crucible explain` walks to the base arm"
        assert {i["schema_version"] for i in ctx.inputs} == {ARM_PREDICTIONS_SCHEMA_VERSION}

    def test_a_base_arm_that_did_not_score_every_name_is_a_refusal_not_a_hole(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        panel = _panel()
        for day in panel.dates:
            write_arm_predictions(
                _Ctx(store),
                arm_id="m:base:abc123",
                trading_day=day,
                feature_version="v1",
                predicted_alpha={panel.names[0]: 0.1},
            )
        with pytest.raises(ArmPredictionsContractError, match="missing"):
            stack_prediction_columns(
                panel,
                store=store,
                recipe=_StubRecipe((InputRef(kind="predictions", ref="base"),)),
                base_arm_ids={"base": "m:base:abc123"},
            )

    def test_an_arm_with_no_prediction_inputs_gets_the_panel_back_unchanged(self, tmp_path) -> None:
        panel = _panel()
        assert (
            stack_prediction_columns(
                panel,
                store=LocalStore(tmp_path),
                recipe=_StubRecipe(()),
                base_arm_ids={},
            )
            is panel
        )


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------


class _Ctx:
    """The two `RunContext` methods this contract uses, and nothing else."""

    def __init__(self, store: LocalStore) -> None:
        self.store = store
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []

    def record_input(self, key: str, payload: bytes, schema_version: str = "v1") -> None:
        self.inputs.append({"key": key, "schema_version": schema_version})

    def record_output(self, key: str, payload: bytes, schema_version: str = "v1") -> None:
        self.store.put_bytes(key, payload)
        self.outputs.append({"key": key, "schema_version": schema_version})


class _StubRecipe:
    """A recipe reduced to the one attribute :func:`stack_prediction_columns` reads."""

    def __init__(self, inputs: tuple[InputRef, ...]) -> None:
        self.inputs = inputs


def _panel() -> FeaturePanel:
    dates = ("2026-08-26", "2026-08-27", "2026-08-28")
    names = ("AAPL", "MSFT")
    shape = (len(dates), len(names))
    return FeaturePanel(
        dates=dates,
        names=names,
        features={"mom_21d_ratio": np.zeros(shape)},
        forward_returns=np.zeros(shape),
        feature_version="v1",
    )
