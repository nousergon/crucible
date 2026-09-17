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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crucible.slots import SLOTS, get_slot
from crucible.slots.inputs import (
    ARM_PREDICTIONS_SCHEMA_VERSION,
    BASE_COVERAGE_METRIC,
    ArmPredictionsContractError,
    BaseCoverageBelowFloorError,
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


class TestAPureMetaLearnerCanRegister:
    """alpha-engine-config-I9821: `ModelRecipe.__post_init__`'s no-features
    guard predated `spec.inputs` (I9777) and still tested `self.features`
    alone, refusing a PURE meta-learner — every design column a
    `predictions[...]` input, no feature-layer columns at all — the
    canonical stacking ensemble. Demonstrated before the fix:

        === a PURE meta-learner (only prediction inputs) cannot be declared ===
          REFUSED: ValueError arm 'pure' declares no features
    """

    def test_a_pure_meta_learner_with_only_prediction_inputs_registers(self, tmp_path) -> None:
        _write(tmp_path, "base", features="[mom_21d_ratio]")
        _write(
            tmp_path,
            "pure",
            features="[]",
            inputs=("predictions[base]",),
        )
        loaded = load_model_recipes(tmp_path, feature_columns=_LAYER)
        assert loaded.refused == ()
        recipes = {r.name: r for r in loaded.registered}
        assert recipes["pure"].design_columns == (prediction_column("base"),)

    def test_an_arm_with_no_features_and_no_inputs_is_still_refused_by_name(self, tmp_path) -> None:
        """The refusal itself is correct and must survive the fix: an arm
        with an empty design matrix — no feature-layer columns AND no
        inputs — cannot be fit. Construction-time refusals (this one, like
        the missing-required-field refusal above it in `load_model_recipes`)
        raise and end the load, rather than becoming a per-arm
        `InputRefusal` — a recipe this malformed never reaches the
        producibility check `InputRefusal` exists to report on."""
        _write(tmp_path, "empty", features="[]")
        with pytest.raises(ValueError, match=r"arm 'empty' declares no design columns"):
            load_model_recipes(tmp_path, feature_columns=_LAYER)


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

    def test_a_base_arm_far_below_the_coverage_floor_is_a_refusal_not_a_hole(
        self, tmp_path
    ) -> None:
        """One name of two is a 50% shortfall — a collapse, not a young listing.

        The general case moved with `alpha-engine-config-I10947` (see
        `TestTheStackScoresTheIntersection`); what stays a refusal is a base
        whose opinion covers too little of the panel to be the same universe.
        """
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


def _wide_panel(n_names: int = 20) -> FeaturePanel:
    """A panel wide enough for a one-name shortfall to sit ABOVE the floor.

    `_panel` carries two names, so dropping one of them is a 50% shortfall
    and every case it can express is a refusal. The ruling's ordinary case —
    a young listing the base model has no window for — is a shortfall of a
    few names in nine hundred, and a fixture that cannot express it would
    test the floor and call it the intersection.
    """
    dates = ("2026-08-26", "2026-08-27", "2026-08-28")
    names = tuple(f"N{i:02d}" for i in range(n_names))
    shape = (len(dates), len(names))
    return FeaturePanel(
        dates=dates,
        names=names,
        features={"mom_21d_ratio": np.zeros(shape)},
        forward_returns=np.zeros(shape),
        feature_version="v1",
    )


def _write_base(store, panel, *, scored, arm_id="m:base:abc123", value=0.1):
    """``scored`` names get an opinion on every session; the rest get none."""
    for day in panel.dates:
        write_arm_predictions(
            _Ctx(store),
            arm_id=arm_id,
            trading_day=day,
            feature_version="v1",
            predicted_alpha={n: value for n in scored},
        )


def _stack(store, panel, *, ctx=None):
    return stack_prediction_columns(
        panel,
        store=store,
        recipe=_StubRecipe((InputRef(kind="predictions", ref="base"),)),
        base_arm_ids={"base": "m:base:abc123"},
        ctx=ctx,
    )


class TestTheStackScoresTheIntersection:
    """Brian's ruling 2026-09-17, `alpha-engine-config-I10947` option (a).

    The contract these tests replace demanded the base's opinion on 100% of
    the panel. Measured over 2024-05-01..2026-06-04 the base scored the whole
    panel on NO date — it correctly drops the rows whose features are null —
    so every stacked arm was unproducible on every date, and the M slot could
    never win a champion.

    What is asserted here is the pair of properties that make the relaxation
    safe: the missing name is EXCLUDED (never substituted), and the shortfall
    is a published number checked against a floor the slot declares.
    """

    def test_a_name_the_base_did_not_score_is_excluded_never_substituted(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        panel = _wide_panel()
        _write_base(store, panel, scored=panel.names[1:])

        column = _stack(store, panel).column(prediction_column("base"))

        assert np.isnan(column[:, 0]).all(), (
            "the unscored name carries NOT-A-NUMBER on every session — the one "
            "encoding the training design and the serving path already read as "
            "'this row cannot be used'"
        )
        assert (column[:, 0] != 0.0).all() | np.isnan(column[:, 0]).all(), (
            "and it is not a zero: a zero is an OPINION ('exactly average') invented "
            "for a model that said nothing — the 2026-08-28 hard-zeroed-features "
            "condition through a different door"
        )
        assert np.isfinite(column[:, 1:]).all(), "every scored name keeps its value"

    def test_the_panel_carries_a_coverage_record_naming_scored_panel_and_missing(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        panel = _wide_panel()
        _write_base(store, panel, scored=panel.names[2:])

        (coverage,) = _stack(store, panel).input_coverage

        assert coverage.panel_names == 20
        assert coverage.scored_on("2026-08-28") == 18
        assert coverage.missing_on("2026-08-28") == 2
        assert coverage.coverage_on("2026-08-28") == pytest.approx(0.9)
        assert coverage.missing_names == ("N00", "N01")
        assert coverage.floor == get_slot("m").stacked_base_coverage_floor
        assert coverage.base_arm_id == "m:base:abc123"

    def test_the_coverage_figure_is_per_session_never_one_panel_average(self, tmp_path) -> None:
        """One collapsed session inside a healthy window must not average away."""
        store = LocalStore(tmp_path)
        panel = _wide_panel()
        for day in panel.dates:
            scored = panel.names if day != "2026-08-27" else panel.names[:19]
            write_arm_predictions(
                _Ctx(store),
                arm_id="m:base:abc123",
                trading_day=day,
                feature_version="v1",
                predicted_alpha={n: 0.1 for n in scored},
            )

        (coverage,) = _stack(store, panel).input_coverage

        assert coverage.worst_session == "2026-08-27"
        assert coverage.coverage_on("2026-08-26") == 1.0
        assert coverage.worst_coverage == pytest.approx(0.95)

    def test_a_coverage_figure_for_a_session_never_read_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        panel = _wide_panel()
        _write_base(store, panel, scored=panel.names)
        (coverage,) = _stack(store, panel).input_coverage

        with pytest.raises(KeyError, match="was not measured"):
            coverage.coverage_on("2026-08-25")

    def test_the_metric_row_satisfies_the_manifest_schema(self, tmp_path) -> None:
        from jsonschema import Draft202012Validator

        store = LocalStore(tmp_path)
        panel = _wide_panel()
        _write_base(store, panel, scored=panel.names[1:])
        (coverage,) = _stack(store, panel).input_coverage

        row = coverage.as_metric(slot="m", trading_day="2026-08-28")
        schema = json.loads(
            Path("crucible/schemas/run_manifest.v2.json").read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema["$defs"]["MetricRecordRow"])

        assert sorted(validator.iter_errors(row), key=lambda e: list(e.path)) == []
        assert row["name"] == BASE_COVERAGE_METRIC
        assert row["value"] == pytest.approx(0.95)
        assert row["baseline"] == coverage.floor
        assert row["base_coverage"]["worst_session_missing"] == 1
        assert "19 of 20" in row["status_reason"]


class TestTheFloorIsDeclaredOnTheSlot:
    def test_a_session_below_the_floor_is_refused_rather_than_produced(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        panel = _wide_panel()
        _write_base(store, panel, scored=panel.names[:5])

        with pytest.raises(BaseCoverageBelowFloorError, match="below the 0.9 floor"):
            _stack(store, panel)

    def test_the_floor_read_is_the_one_the_slot_declares(self, tmp_path, monkeypatch) -> None:
        """Move the slot's declared value and the behaviour moves with it.

        The floor is a property of the slot, and a literal at this call site
        would be a second declaration of it — the shape that lets two readers
        of one rule drift apart. Nineteen of twenty names PRODUCES under the
        declared 0.90 and REFUSES under 0.99, with nothing else changed.
        """
        store = LocalStore(tmp_path)
        panel = _wide_panel()
        _write_base(store, panel, scored=panel.names[1:])

        assert _stack(store, panel).input_coverage, "0.95 coverage clears the 0.90 floor"

        monkeypatch.setitem(SLOTS, "m", replace(SLOTS["m"], stacked_base_coverage_floor=0.99))
        with pytest.raises(BaseCoverageBelowFloorError, match="below the 0.99 floor"):
            _stack(store, panel)

    def test_the_refusal_measures_against_the_panel_of_the_day_being_read(self, tmp_path) -> None:
        """The non-inferable gotcha of `alpha-engine-config-I10947`.

        The refusal this replaced reported 887 scored names against a
        908-name panel while the session it named carried 903 — a count that
        described no day, so an operator could not tell which universe was
        short. The message names the DAY and the count that day was measured
        against.
        """
        store = LocalStore(tmp_path)
        panel = _wide_panel()
        _write_base(store, panel, scored=panel.names[:5])

        with pytest.raises(BaseCoverageBelowFloorError) as raised:
            _stack(store, panel)

        message = str(raised.value)
        assert "scored 5 of the 20 name(s) the panel carries on 2026-08-26" in message
        assert "15 missing" in message
        assert "excluded name, never a zero" in message
