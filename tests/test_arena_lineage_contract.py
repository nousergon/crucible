"""A graded arm's lineage reaches the verdict surface, and stays honest there.

`alpha-engine-config-I9903` filed the defect: a champion's score series could
not be read, from the verdict artifact alone, as spanning one feature-layer
version or several. `nousergon-lib-PR386` built the channel — an opaque
`dimension -> distinct values` map on `ArmSeries`, passed through by the
arena engine unread and emitted per ladder entry on `arena_cycle.v1`.
`alpha-engine-config-I9963` is crucible's half, and this module is its
consumer contract.

**Three properties, and the third is the one that is easy to lose.**

* A cycle carrying a populated `lineage` validates against the LIBRARY's
  schema — `crucible.arena_io.validate_arena_cycle` reads
  `nousergon_lib/contracts/arena_cycle.schema.json`, never a local copy, so a
  crucible-side restatement cannot pass while the real contract has moved.
* A ladder entry declaring a dimension with an EMPTY value list is refused. A
  dimension that took no value is a well-formed record of nothing, and the
  whole point of this surface is that nothing is recorded as nothing.
* **ABSENT and `{}` stay distinguishable.** A document with no `lineage` key
  was written before the field existed; `{}` is a slot declaring none. The
  schema keeps the property optional for exactly that reason, and collapsing
  the two would make "we never recorded this" and "there is nothing to
  record" render identically — the defect class this surface exists to
  remove.

**The M half is a contract, not an observation.** `crucible.slots.model` has
no production caller at all (`alpha-engine-config-I9957`, pinned by
`tests/test_slot_inputs_wiring.py`), so `grade_arm`'s lineage reaches no
`arena_cycle` any scheduled run writes. What is asserted here is that the
writer populates it and that the value is the panel's — not that a live M
cycle has ever emitted one.
"""

from __future__ import annotations

import copy
import datetime as dt
import json

import numpy as np
import pytest
from conftest import sessions_ending
from nousergon_lib.arena.ladder import build_ladder
from nousergon_lib.arena.window import ArmSeries

from crucible.arena_io import ArenaCycleValidationError, validate_arena_cycle
from crucible.config import Settings
from crucible.data import run_daily
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import arena_cycle_key, shadow_key
from crucible.ledger import read_trials
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.arms import control_specs
from crucible.slots.grading import ShadowSelection, produce_shadow
from crucible.slots.model import FeaturePanel, ModelRecipe, grade_arm

HORIZON = 21
DECISION_DATES = 6


def _settings(strategy_dir, store_root) -> Settings:
    return Settings(
        store_uri=str(store_root),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )


def _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path):
    """One real U cycle: shadows on six past sessions, then the grade."""
    settings = _settings(strategy_dir, tmp_path / "store")
    sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
    decision_days = sessions[:DECISION_DATES]

    for day in [*decision_days, cycle_date]:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=source.symbols()),
            store=store,
            trading_day=day,
        )
    for day in decision_days:
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings),
            store=store,
            trading_day=day,
        )
    run_job(
        "experiment.grade",
        lambda c: universe.grade(c, settings=settings),
        store=store,
        trading_day=cycle_date,
    )
    payload = json.loads(store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())))
    return settings, decision_days, payload


class TestTheProducedShadowRecordsTheVersionItWasProducedUnder:
    """Deliverable 3's first half: the per-day fact, recorded where it is known."""

    def test_every_shadow_a_cycle_writes_names_its_feature_layer_version(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        _, decision_days, _ = _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path)
        from crucible.slots.arms import load_arm_specs

        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        seen = set()
        for spec in specs:
            for day in decision_days:
                doc = json.loads(store.get_bytes(shadow_key(spec.arm_id, day.isoformat())))
                assert doc["feature_version"], (
                    f"{spec.name} on {day}: the shadow records no feature-layer version, "
                    "so the series assembled from it can carry no lineage"
                )
                seen.add(doc["feature_version"])
        assert seen == {DEFAULT_FEATURE_VERSION}

    def test_a_producer_that_drops_the_version_is_refused_not_defaulted(self) -> None:
        """Fail loud. An empty version is a producer that HAD the value, not
        an artifact that predates the field — and a default would let it
        reach the verdict surface as a series whose lineage silently cannot
        be assembled."""
        with pytest.raises(ValueError, match="feature-layer version"):
            ShadowSelection(
                arm_id="u:a:aaa",
                trading_day="2026-08-03",
                selection=("A",),
                population=("A", "B"),
                ranker="momentum_sleeve",
                params={},
                feature_version="",
            )

    def test_produce_shadow_takes_the_version_as_a_required_keyword(self) -> None:
        """No default: a second caller would otherwise produce a shadow whose
        lineage claims a version it never read."""
        import inspect

        parameter = inspect.signature(produce_shadow).parameters["feature_version"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


class TestTheCycleArtifactCarriesTheLineage:
    """Deliverable 3's second half: the per-day facts, assembled per series."""

    def test_a_real_arms_ladder_entry_names_the_version_its_dates_were_produced_under(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        _, _, payload = _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path)
        control_ids = {c.arm_id for c in control_specs(_slot("u"))}
        ladders = {entry["arm_id"]: entry for entry in payload["ladders"]}
        real = {a: e for a, e in ladders.items() if a not in control_ids}
        assert real, "the cycle graded no non-control arm; there is no lineage to read"
        for arm_id, entry in real.items():
            assert entry["lineage"] == {"feature_version": [DEFAULT_FEATURE_VERSION]}, (
                f"{arm_id}: the ladder entry does not name the feature-layer version "
                f"its scored dates were produced under: {entry['lineage']}"
            )

    def test_a_control_declares_no_dimension_rather_than_an_invented_one(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """A control is generated at grade time from the settled returns and
        has no shadow, so it read no feature layer. `{}` is the honest
        reading; naming the grade-date version would be a fabrication."""
        _, _, payload = _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path)
        control_ids = {c.arm_id for c in control_specs(_slot("u"))}
        entries = [e for e in payload["ladders"] if e["arm_id"] in control_ids]
        assert entries, "no control reached the cycle artifact"
        for entry in entries:
            assert entry["lineage"] == {}

    def test_the_trial_ledger_row_carries_the_same_map_as_the_cycle(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """Deliverable 2. The DSR denominator's own log answers the same
        question as the verdict artifact, so "how many trials" and "over which
        upstream versions" are readable from one file."""
        _, _, payload = _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path)
        ladders = {entry["arm_id"]: entry["lineage"] for entry in payload["ladders"]}
        rows = {
            row["arm_id"]: row["lineage"]
            for row in read_trials(store)
            if row["as_of"] == cycle_date.isoformat()
        }
        assert rows, "the cycle wrote no trial rows"
        for arm_id, lineage in rows.items():
            assert lineage == ladders[arm_id], (
                f"{arm_id}: the trial row and the cycle artifact disagree about the "
                f"arm's lineage — {lineage} vs {ladders[arm_id]}"
            )


class TestTheModelWriterPopulatesTheLineage:
    """Deliverable 1. A CONTRACT: no scheduled job calls `grade_arm`."""

    def test_a_graded_m_arm_declares_the_panels_feature_version(self) -> None:
        grade = grade_arm(_m_recipe(), _m_panel(version="v-cat-abc"), as_of="2026-08-28")
        assert grade.status == "ok", grade.reason
        assert grade.series.lineage == {"feature_version": ("v-cat-abc",)}

    def test_the_version_is_the_panels_not_one_a_caller_supplied(self) -> None:
        """Two panels differing only in their version produce two lineages,
        so the value is read from the panel rather than from a constant that
        happens to agree with it today."""
        first = grade_arm(_m_recipe(), _m_panel(version="v-one"), as_of="2026-08-28")
        second = grade_arm(_m_recipe(), _m_panel(version="v-two"), as_of="2026-08-28")
        assert first.series.lineage != second.series.lineage
        assert second.series.lineage == {"feature_version": ("v-two",)}

    def test_an_unmeasurable_arm_declares_nothing_rather_than_a_version(self) -> None:
        """An arm with no out-of-sample date has no scored dates, and the
        contract is "the distinct values across the dates in `scores`".
        Declaring the panel's version anyway would record a version the
        series never rested on."""
        recipe = _m_recipe(registered_at="2030-01-02")
        grade = grade_arm(recipe, _m_panel(version="v-cat-abc"), as_of="2026-08-28")
        assert grade.status == "unmeasurable"
        assert grade.series.scores == {}
        assert grade.series.lineage == {}


class TestTheLibrarySchemaIsTheContract:
    """Deliverable 5. Validated against the LIBRARY's schema, never a copy."""

    def test_a_populated_lineage_validates(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        _, _, payload = _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path)
        assert any(e["lineage"] for e in payload["ladders"])
        validate_arena_cycle(payload)  # the artifact as written, unmodified

    def test_a_dimension_with_an_empty_value_list_is_refused(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """Policy §7.4: the same validator, over two documents, asserting the
        readings DIFFER — and asserting the mutation actually changed the
        document rather than being a no-op that passes for the wrong reason.
        """
        _, _, payload = _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path)
        validate_arena_cycle(payload)

        mutated = copy.deepcopy(payload)
        target = next(e for e in mutated["ladders"] if e["lineage"])
        target["lineage"]["feature_version"] = []
        assert mutated != payload, "the mutation did not change the document"

        with pytest.raises(ArenaCycleValidationError, match="lineage"):
            validate_arena_cycle(mutated)

    def test_absent_and_empty_lineage_are_both_valid_and_are_not_the_same_document(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """The distinction the whole field rests on. ABSENT is a document
        written before `lineage` existed; `{}` is a slot that declares none.
        Both validate — the property is optional for exactly this reason —
        and they are not equal, so a reader can still tell them apart."""
        _, _, payload = _seed_and_grade(store, source, strategy_dir, cycle_date, tmp_path)

        declares_none = copy.deepcopy(payload)
        for entry in declares_none["ladders"]:
            entry["lineage"] = {}
        predates_the_field = copy.deepcopy(payload)
        for entry in predates_the_field["ladders"]:
            entry.pop("lineage")

        validate_arena_cycle(declares_none)
        validate_arena_cycle(predates_the_field)
        assert declares_none != predates_the_field, (
            "an `{}` lineage and an absent one serialise identically, so 'we never "
            "recorded this' and 'there is nothing to record' cannot be told apart"
        )
        assert all("lineage" not in e for e in predates_the_field["ladders"])
        assert all(e["lineage"] == {} for e in declares_none["ladders"])

    def test_the_library_builds_the_ladder_entry_from_the_series_unread(self) -> None:
        """The channel is opaque by construction: a dimension crucible has
        never heard of survives the engine untouched, which is what keeps a
        slot-agnostic engine from learning about one slot's feature layer
        (`principles.md` §2.8)."""
        series = ArmSeries(
            arm_id="u:a:aaa",
            scores={"2026-08-03": 0.01, "2026-08-04": 0.02},
            lineage={"a_dimension_crucible_never_declared": ("x", "y")},
        )
        ladder = build_ladder(series, as_of="2026-08-04")
        assert ladder.to_dict()["lineage"] == {"a_dimension_crucible_never_declared": ["x", "y"]}


# ---------------------------------------------------------------------------
# M fixtures. A small synthetic panel with a learnable signal, so `grade_arm`
# reaches `status: ok` and the series has dates for the lineage to describe.
# ---------------------------------------------------------------------------


def _slot(slot: str):
    from crucible.slots import get_slot

    return get_slot(slot)


def _m_recipe(registered_at: str = "2026-01-02") -> ModelRecipe:
    from crucible.slots.model import CPCVSpec, EstimatorSpec, TrainingWindowSpec

    return ModelRecipe(
        slot="m",
        name="fixture_arm",
        features=("mom_21d_ratio",),
        estimator=EstimatorSpec(kind="ols", params={}),
        label_horizon_trading_days=1,
        refit_cadence_trading_days=5,
        training_window=TrainingWindowSpec(kind="expanding", min_trading_days=5),
        cpcv=CPCVSpec(n_groups=4, k_test=1, embargo_trading_days=1),
        registered_at=registered_at,
    )


def _m_panel(*, version: str) -> FeaturePanel:
    """Sessions ending 2026-08-28, one feature that predicts the next return."""
    dates = [d.isoformat() for d in sessions_ending(dt.date(2026, 8, 28), 40)]
    names = tuple(f"T{i:02d}" for i in range(12))
    rng = np.random.default_rng(11)
    feature = rng.normal(size=(len(dates), len(names)))
    forward = feature * 0.5 + rng.normal(scale=0.1, size=feature.shape)
    return FeaturePanel(
        dates=tuple(dates),
        names=names,
        features={"mom_21d_ratio": feature},
        forward_returns=forward,
        feature_version=version,
    )
