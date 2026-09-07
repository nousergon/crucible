"""The M cycle job: `experiment.run --slot m` and `experiment.grade --slot m`.

`alpha-engine-config-I9957`. Until this job existed `crucible/slots/model.py`
was reachable only from `tests/`: `load_model_recipes` had no production
caller, so `SlotRecipes.refusal_metrics` — one `unservable` row per refused
arm — landed on no manifest a scheduled run ever wrote, and a standing
refusal was a fact nobody was shown.

Every test here drives the REAL path: a real feature layer on disk, the real
loader, the real `crucible.runner.run_job` wrapper, and the real manifest
written to a real store. The tracker's closes-when is a property of a written
`run.json`, and a test that asserted a call site instead would pass on a job
whose rows reached nothing — which is the defect one frame further out.

Dates are fixed literals off a pinned session axis, never `today` arithmetic
(AGENTS.md test discipline).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from crucible.calendar import is_trading_day
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import (
    arm_predictions_key,
    arm_register_key,
    cross_section_key,
    data_panel_key,
    manifest_key,
    shadow_key,
)
from crucible.runner import run_job
from crucible.slots.inputs import SlotUnservableError
from crucible.slots.model import (
    ARM_REFUSED_METRIC,
    CPCV_OOS_IC_METRIC,
    SLOT,
    ModelRecipe,
    RegisteredModelArm,
    SlotRecipes,
    SupersededArmUndeclaredError,
    _in_dependency_order,
    grade,
    load_model_recipes,
    produce,
    registration_specs,
)
from crucible.store import LocalStore

#: Real names from `crucible.features.CATALOG`. A fixture catalogue would let
#: a registration check pass against a registry production never sees.
BASE_COLUMN = "momentum_20d_zscore"
SECOND_COLUMN = "volatility_20d_ratio"

#: A pinned start session and the count needed for a 10-session training
#: window plus a 2-session label horizon plus room to settle at grade time.
_START = dt.date(2026, 6, 1)
_SESSIONS = 60
#: Twelve names, not eight: an M shadow selects `M_SELECTION_TOP_N` = 10, and
#: the slot's controls are count-matched to it (`_control_top_n`), so a
#: population smaller than the selection makes the CONTROL refuse rather than
#: the arm — a fixture that could never exercise the grader it is checking.
_NAMES = (
    "AAA",
    "BBB",
    "CCC",
    "DDD",
    "EEE",
    "FFF",
    "GGG",
    "HHH",
    "III",
    "JJJ",
    "KKK",
    "LLL",
)


def _sessions() -> tuple[str, ...]:
    """Real NYSE sessions, so a produce date is never refused as a non-session.

    Taken from `crucible.calendar`, not from `weekday() < 5`: 2026-07-03 is a
    closed weekday and an axis that carried it would make every date literal
    below name a day the market did not trade.
    """
    days: list[str] = []
    day = _START
    while len(days) < _SESSIONS:
        if is_trading_day(day):
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    return tuple(days)


SESSIONS = _sessions()
#: The produce/grade date used by every test that needs one artifact deep
#: enough to train on and shallow enough to leave sessions unsettled.
RUN_DAY = SESSIONS[45]

#: The sessions a stacked arm needs its base to have already predicted:
#: `min_trading_days + label_horizon` rows, plus the anchor.
WARMUP = SESSIONS[32:46]

#: The last session the whole slot produced for, so every arm — the stack
#: included — has a panel at grade time and a settled shadow behind it.
GRADE_DAY = SESSIONS[51]


@pytest.fixture
def store(tmp_path):
    """A real store carrying a real feature layer and a real price panel."""
    import pandas as pd

    backing = LocalStore(tmp_path / "store")
    rng = np.random.default_rng(20260906)
    rows = []
    for i, day in enumerate(SESSIONS):
        closes = 100.0 + i * 0.5 + rng.normal(0.0, 1.0, len(_NAMES))
        frame = pd.DataFrame(
            {
                "ticker": list(_NAMES),
                "close_raw": closes,
                BASE_COLUMN: rng.normal(0.0, 1.0, len(_NAMES)),
                SECOND_COLUMN: rng.normal(1.0, 0.3, len(_NAMES)),
            }
        )
        backing.put_bytes(
            f"features/{DEFAULT_FEATURE_VERSION}/{day}.parquet", frame.to_parquet(index=False)
        )
        rows.extend(
            {"trading_day": dt.date.fromisoformat(day), "ticker": t, "close_raw": float(c)}
            for t, c in zip(_NAMES, closes, strict=True)
        )
    panel = pd.DataFrame(rows)
    for day in SESSIONS:
        backing.put_bytes(data_panel_key(day), panel.to_parquet(index=False))
    return backing


def _write_recipe(directory, name, *, features, inputs=(), supersedes=None, min_days=10):
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["slot: m", f"name: {name}"]
    if supersedes:
        lines.append(f"supersedes: {supersedes}")
    lines += ["spec:", f"  features: [{', '.join(features)}]"]
    if inputs:
        lines.append("  inputs:")
        lines += [f"    - {entry}" for entry in inputs]
    lines += [
        "  estimator: {kind: ridge, alpha: 1.0}",
        "  label_horizon_trading_days: 2",
        "  refit_cadence_trading_days: 5",
        f"  training_window: {{kind: expanding, min_trading_days: {min_days}}}",
        "  cpcv: {n_groups: 4, k_test: 1, embargo_trading_days: 1}",
        f"registered_at: '{SESSIONS[0]}'",
    ]
    (directory / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Settings:
    """The two attributes the M job reads off `crucible.config.Settings`."""

    def __init__(self, strategy_dir=None):
        self.strategy_dir = strategy_dir


@pytest.fixture
def strategy(tmp_path):
    """`base`, `stacked` (consuming `predictions[base]`), and one REFUSED arm.

    `refused_leg` declares a `predictions[...]` input naming an arm this slot
    does not carry — the live shape of `sota_directional_combine`, whose two
    legs are unregisterable until phase 5.
    """
    arms = tmp_path / "strategy" / "arms" / SLOT
    _write_recipe(arms, "base", features=[BASE_COLUMN])
    _write_recipe(arms, "stacked", features=[SECOND_COLUMN], inputs=["predictions[base]"])
    _write_recipe(arms, "refused_leg", features=[BASE_COLUMN], inputs=["predictions[nowhere]"])
    return _Settings(tmp_path / "strategy")


def _run_produce(store, settings, day=RUN_DAY, **kwargs):
    result: dict = {}
    ctx = run_job(
        "experiment.run",
        lambda c: result.update(produce(c, settings=settings, **kwargs)),
        store=store,
        trading_day=dt.date.fromisoformat(day),
        run_mode="replay",
        discriminator=SLOT,
    )
    return ctx, result


def _manifest(store, job, day):
    return json.loads(store.get_bytes(manifest_key(job, day, discriminator=SLOT)).decode("utf-8"))


def _warm_the_base(store, settings):
    """Run `base` alone over the sessions the stacked arm's window covers.

    A stacked arm reads its base's prediction on EVERY row of its training
    window, so it cannot produce until the base has actually run that many
    cycles. Nothing here is a shortcut: each of these is the real produce job
    for one real session, which is exactly how the history accumulates in
    production.
    """
    for day in WARMUP:
        _run_produce(store, settings, day=day, arm_name="base")


class TestTheRefusalRowReachesARealManifest:
    """The tracker's closes-when, as a property of a written `run.json`."""

    def test_produce_writes_the_refusal_metric_on_its_own_manifest(self, store, strategy) -> None:
        _warm_the_base(store, strategy)
        _run_produce(store, strategy)
        document = _manifest(store, "experiment.run", RUN_DAY)
        rows = [m for m in document["metrics"] if m["name"] == ARM_REFUSED_METRIC]
        assert [r["status"] for r in rows] == ["unservable"]
        assert "refused_leg" in rows[0]["status_reason"]
        assert "predictions[nowhere]" in rows[0]["status_reason"]
        assert document["status"] == "ok", document["reason"]

    def test_the_refused_arm_produces_no_artifact_while_its_siblings_do(
        self, store, strategy
    ) -> None:
        """The blast radius, asserted rather than assumed: one refused arm
        must not take the two producible arms down with it."""
        _warm_the_base(store, strategy)
        _, result = _run_produce(store, strategy)
        assert [r["arm"] for r in result["refused"]] == ["refused_leg"]
        assert len(result["arms"]) == 2
        for arm_id in result["arms"]:
            assert store.exists(shadow_key(arm_id, RUN_DAY))
            assert store.exists(cross_section_key(arm_id, RUN_DAY))
            assert store.exists(arm_predictions_key(arm_id, RUN_DAY))

    def test_the_shadow_and_the_predictions_artifact_rank_the_same_session(
        self, store, strategy
    ) -> None:
        """One `predict_cross_section` call feeds both writes, so the artifact
        a stacked arm trains on and the shadow the arena scores can never be
        two different rankings of one day."""
        _warm_the_base(store, strategy)
        _, result = _run_produce(store, strategy)
        arm_id = result["arms"][0]
        shadow = json.loads(store.get_bytes(shadow_key(arm_id, RUN_DAY)).decode("utf-8"))
        predictions = json.loads(
            store.get_bytes(arm_predictions_key(arm_id, RUN_DAY)).decode("utf-8")
        )["predicted_alpha"]
        ranked = sorted(predictions.items(), key=lambda item: (-item[1], item[0]))
        assert shadow["selection"] == [t for t, _ in ranked[: len(shadow["selection"])]]
        assert shadow["population"] == sorted(predictions)


class TestTheUnservableSlotPagesEndToEnd:
    """Deliverable 2: the whole-slot refusal reaches a `failed` manifest and
    the existing failure page — by running, not by construction."""

    def test_every_arm_refused_fails_the_run_after_recording_every_row(
        self, store, tmp_path
    ) -> None:
        from crucible.alerts import evaluate_failure

        arms = tmp_path / "only-refused" / "arms" / SLOT
        _write_recipe(arms, "leg_a", features=[BASE_COLUMN], inputs=["predictions[nowhere]"])
        _write_recipe(arms, "leg_b", features=[SECOND_COLUMN], inputs=["predictions[elsewhere]"])
        with pytest.raises(SlotUnservableError):
            _run_produce(store, _Settings(tmp_path / "only-refused"))
        document = _manifest(store, "experiment.run", RUN_DAY)
        assert document["status"] == "failed"
        rows = sorted(
            m["status_reason"] for m in document["metrics"] if m["name"] == ARM_REFUSED_METRIC
        )
        assert len(rows) == 2, "the whole-slot case must still name every arm"
        assert any("leg_a" in r for r in rows) and any("leg_b" in r for r in rows)
        pages = evaluate_failure(store, now=dt.datetime.fromisoformat(f"{RUN_DAY}T23:00:00+00:00"))
        assert [p.job for p in pages] == ["experiment.run"], (
            "an unservable slot must page through the ordinary failed-manifest "
            "condition — no third page condition, ever"
        )


class TestRegistrationCarriesADeclaredLineage:
    """A recipe's `supersedes` and a register row's `supersedes` are two facts.

    Conflating them broke the M slot outright the first time a real call was
    made: `directional_combine_residual_stack` supersedes a sibling that is
    refused until phase 5, so its parent has no register row and never will.
    """

    def _loaded(self, tmp_path, *, supersedes):
        arms = tmp_path / "lineage" / "arms" / SLOT
        _write_recipe(arms, "parent", features=[BASE_COLUMN], inputs=["predictions[nowhere]"])
        _write_recipe(arms, "child", features=[BASE_COLUMN], supersedes=supersedes)
        return load_model_recipes(arms)

    def test_a_refused_parent_is_provenance_not_a_register_link(self, tmp_path) -> None:
        loaded = self._loaded(tmp_path, supersedes="m:parent:0123456789ab")
        specs = registration_specs(loaded)
        child = next(s for s in specs if s.name == "child")
        assert child.supersedes is None, "a refused parent has no row to link to"
        assert "m:parent:0123456789ab" in child.notes

    def test_the_registration_actually_succeeds_with_a_refused_parent(
        self, store, tmp_path
    ) -> None:
        """The measured failure, made to pass. Before this the whole slot
        raised `ValueError: ... which is not in the register`."""
        arms = tmp_path / "lineage" / "arms" / SLOT
        _write_recipe(arms, "parent", features=[BASE_COLUMN], inputs=["predictions[nowhere]"])
        _write_recipe(arms, "child", features=[BASE_COLUMN], supersedes="m:parent:0123456789ab")
        _run_produce(store, _Settings(tmp_path / "lineage"))
        register = store.get_bytes(arm_register_key(SLOT)).decode("utf-8")
        assert "child" in register

    def test_a_parent_the_slot_never_declared_still_raises(self, tmp_path) -> None:
        """The guard, made to fire. Moving the check from REGISTERED to
        DECLARED must not have removed it."""
        loaded = self._loaded(tmp_path, supersedes="m:typo:0123456789ab")
        with pytest.raises(SupersededArmUndeclaredError, match="typo"):
            registration_specs(loaded)


class TestBaseArmsProduceBeforeTheArmsThatStackOnThem:
    def test_the_stacked_arm_reads_a_prediction_written_this_cycle(self, store, strategy) -> None:
        """Produced in file-name order the stack would refuse on the one
        session it could have been satisfied: `base`'s artifact for today does
        not exist until `base` has run."""
        _warm_the_base(store, strategy)
        _, result = _run_produce(store, strategy)
        assert len(result["arms"]) == 2
        base_id = next(a for a in result["arms"] if ":base:" in a)
        assert result["arms"].index(base_id) == 0

    def test_the_order_is_derived_and_not_alphabetical(self, tmp_path) -> None:
        """A detector for the ordering itself: `zzz` is the base, so an
        alphabetical order would produce it last and refuse `aaa`."""
        arms = tmp_path / "order" / "arms" / SLOT
        _write_recipe(arms, "zzz", features=[BASE_COLUMN])
        _write_recipe(arms, "aaa", features=[SECOND_COLUMN], inputs=["predictions[zzz]"])
        specs = registration_specs(load_model_recipes(arms))
        assert [s.name for s in _in_dependency_order(specs)] == ["zzz", "aaa"]
        assert sorted(s.name for s in specs) == ["aaa", "zzz"], "the input was not pre-sorted"


class TestTheLoaderReadsTheStoreAsWellAsACheckout:
    """Blocker 3 of the tracker's 2026-09-04 re-scope: the loader took a
    filesystem path and nothing else, and a spot box has no checkout."""

    def test_recipes_load_from_the_synced_strategy_tree(self, store, tmp_path) -> None:
        arms = tmp_path / "checkout" / "arms" / SLOT
        _write_recipe(arms, "base", features=[BASE_COLUMN])
        store.put_bytes(
            f"strategy/current/arms/{SLOT}/base.yaml", (arms / "base.yaml").read_bytes()
        )
        loaded = load_model_recipes(store=store)
        assert [r.name for r in loaded.registered] == ["base"]
        assert loaded.registered[0].source_key == f"strategy/current/arms/{SLOT}/base.yaml"

    def test_the_job_reads_the_store_when_no_checkout_is_configured(self, store, tmp_path) -> None:
        arms = tmp_path / "checkout" / "arms" / SLOT
        _write_recipe(arms, "base", features=[BASE_COLUMN])
        store.put_bytes(
            f"strategy/current/arms/{SLOT}/base.yaml", (arms / "base.yaml").read_bytes()
        )
        _, result = _run_produce(store, _Settings(None))
        assert len(result["arms"]) == 1

    @pytest.mark.parametrize(("directory", "store_arg"), [(None, None), ("some/dir", "a store")])
    def test_neither_source_and_both_sources_are_both_refused(self, directory, store_arg) -> None:
        """Neither is a caller that resolved no source and would register
        nothing; both is two trees that can disagree."""
        with pytest.raises(ValueError, match="EITHER a checkout"):
            load_model_recipes(directory, store=store_arg)


class TestGradeRunsTheSharedEngine:
    """`crucible.slots.cycle.run_grade`, unchanged, over M's own recipe set.

    `load_arm_specs` refuses slot `m` by name, so the specs are supplied. A
    second copy of the grader for one slot is the four-drifting-
    implementations shape v2 exists to remove.
    """

    def _produce_a_series(self, store, strategy):
        _warm_the_base(store, strategy)
        for day in SESSIONS[46:52]:
            _run_produce(store, strategy, day=day)

    def _grade(self, store, strategy, day):
        result: dict = {}
        ctx = run_job(
            "experiment.grade",
            lambda c: result.update(grade(c, settings=strategy)),
            store=store,
            trading_day=dt.date.fromisoformat(day),
            run_mode="replay",
            discriminator=SLOT,
        )
        return ctx, result

    def test_a_cycle_is_written_and_the_cpcv_ic_reaches_the_manifest(self, store, strategy) -> None:
        self._produce_a_series(store, strategy)
        _, result = self._grade(store, strategy, GRADE_DAY)
        assert result["arena_cycle_key"]
        assert store.exists(result["arena_cycle_key"])
        document = _manifest(store, "experiment.grade", GRADE_DAY)
        assert document["status"] == "ok", document["reason"]
        cpcv = [m for m in document["metrics"] if m["name"] == CPCV_OOS_IC_METRIC]
        assert len(cpcv) == 2, "one out-of-sample IC row per registered arm"
        assert {m["horizon_trading_days"] for m in cpcv} == {2}

    def test_the_refusal_row_reaches_the_grade_manifest_too(self, store, strategy) -> None:
        """A slot that became unservable between produce and grade must page
        from whichever job ran."""
        self._produce_a_series(store, strategy)
        self._grade(store, strategy, GRADE_DAY)
        document = _manifest(store, "experiment.grade", GRADE_DAY)
        assert any(m["name"] == ARM_REFUSED_METRIC for m in document["metrics"])

    def test_no_arm_may_serve_while_its_veto_cannot_be_computed(self, store, strategy) -> None:
        """Policy §5.1: an uncomputed gate is not a pass. With no incumbent
        and no producer for three of the four §5.3 metrics, the veto reads
        `insufficient` and the serving precondition FAILS — visible on the
        cycle artifact rather than assumed."""
        self._produce_a_series(store, strategy)
        _, result = self._grade(store, strategy, GRADE_DAY)
        assert {g["veto"] for g in result["model_grades"].values()} == {"insufficient"}
        cycle = json.loads(store.get_bytes(result["arena_cycle_key"]).decode("utf-8"))
        assert "behavioural_veto" in json.dumps(cycle)

    def test_the_control_exclusion_survives_a_caller_supplied_precondition(
        self, store, strategy
    ) -> None:
        """§10.1 is the harness's rule, not a slot's: a caller able to
        displace it is a look-ahead arm one dict key away from the pointer."""
        self._produce_a_series(store, strategy)
        _, result = self._grade(store, strategy, GRADE_DAY)
        cycle = json.loads(store.get_bytes(result["arena_cycle_key"]).decode("utf-8"))
        assert "not_a_control_arm" in json.dumps(cycle)
        assert not [a for a in result["promotable_arms"] if "control" in a]


class TestTheOutOfSampleClockStartsOnASession:
    def test_a_recipe_registered_on_a_non_session_is_refused(self) -> None:
        """`ModelRecipe` checks the field is an ISO date and stops there; a
        date and a session are two different claims, and every ladder rung is
        counted from this one."""
        from crucible.calendar import NonTradingDayKeyError
        from crucible.slots.model import CPCVSpec, EstimatorSpec, TrainingWindowSpec

        recipe = ModelRecipe(
            name="saturday",
            features=(BASE_COLUMN,),
            estimator=EstimatorSpec(kind="ridge"),
            label_horizon_trading_days=2,
            refit_cadence_trading_days=5,
            training_window=TrainingWindowSpec(min_trading_days=10),
            cpcv=CPCVSpec(n_groups=4, k_test=1, embargo_trading_days=1),
            registered_at="2026-08-29",
        )
        with pytest.raises(NonTradingDayKeyError):
            RegisteredModelArm(recipe)


class TestTheIdIsTheRecipesOwn:
    def test_the_adapter_never_re_derives_an_arm_id(self, tmp_path) -> None:
        """Wrapping a `ModelRecipe` in a real `ArmSpec` would hash
        `ranker`/`params` and give one arm two identities — the register, the
        shadows and the series would each speak about a different one."""
        arms = tmp_path / "id" / "arms" / SLOT
        _write_recipe(arms, "base", features=[BASE_COLUMN])
        loaded = load_model_recipes(arms)
        spec = registration_specs(loaded)[0]
        assert spec.arm_id == loaded.registered[0].arm_id
        assert spec.spec == loaded.registered[0].spec


class TestExperimentNewRegistersTheMSlot:
    def test_the_command_no_longer_refuses_m_by_name(self, store, strategy, capsys) -> None:
        """`experiment.new --slot m` was broken for the whole life of the
        command: it reached `load_arm_specs`, failed on a missing `ranker`,
        and read as a malformed recipe tree."""
        import argparse

        from crucible.track_a import _recipes_for_registration

        specs = _recipes_for_registration(
            SLOT, config=argparse.Namespace(strategy_dir=strategy.strategy_dir), store=store
        )
        assert sorted(s.name for s in specs) == ["base", "stacked"]
        assert "refused_leg" in capsys.readouterr().out

    def test_s_is_still_refused_by_name(self, store) -> None:
        import argparse

        from crucible.slots.arms import ForeignRecipeSchemaError
        from crucible.track_a import _recipes_for_registration

        with pytest.raises(ForeignRecipeSchemaError):
            _recipes_for_registration(
                "s", config=argparse.Namespace(strategy_dir=None), store=store
            )


class TestRefusalMetricsAreRecordedBeforeAnyFittingWork:
    def test_a_run_that_raises_later_still_carries_every_refusal(self, store, tmp_path) -> None:
        """Order matters: the rows are recorded before any fitting, so they
        reach the manifest whether the rest of the run succeeds or raises.
        Here the fit is impossible — the declared window exceeds the sessions
        the layer carries — and the refusal row is on the failed manifest."""
        arms = tmp_path / "deep" / "arms" / SLOT
        _write_recipe(arms, "base", features=[BASE_COLUMN], min_days=5000)
        _write_recipe(arms, "refused_leg", features=[BASE_COLUMN], inputs=["predictions[gone]"])
        with pytest.raises(Exception, match="min_trading_days"):
            _run_produce(store, _Settings(tmp_path / "deep"))
        document = _manifest(store, "experiment.run", RUN_DAY)
        assert document["status"] == "failed"
        assert [m["name"] for m in document["metrics"]] == [ARM_REFUSED_METRIC]


class TestTheSlotRecipesRefusalMetricNamesTheSlot:
    def test_the_two_metric_names_are_one_literal(self) -> None:
        """`crucible.slots.cycle` restates the name rather than importing it,
        because importing this module would pull the fitting stack onto the
        U/R path. The two must stay equal."""
        from crucible.slots.cycle import ARM_REFUSED_METRIC as CYCLE_NAME

        assert CYCLE_NAME == ARM_REFUSED_METRIC

    def test_an_empty_slot_is_absence_and_not_unservable(self) -> None:
        """A directory nobody has written an arm for registers nothing and
        refuses nothing; calling that `unservable` would page for a condition
        `crucible.alerts` already owns."""
        assert SlotRecipes(registered=()).unservable is False


class TestAStackedArmWaitsForItsBaseWithoutTakingTheSlotDown:
    """The per-arm warm-up refusal, made to fire on both jobs.

    A stacked arm reads its base's prediction on every row of its training
    window, which is unsatisfiable by construction on the base arm's first
    `min_trading_days` cycles. Left slot-wide it makes the whole M slot
    unproducible for as long as any stacked arm is filed — and one is, under
    Brian's `alpha-engine-config-I9808` ruling (b). This is the same blast
    radius `alpha-engine-config-I9955` removed one layer in, at registration.
    """

    def test_a_cold_slot_produces_its_base_and_refuses_the_stack_by_name(
        self, store, strategy
    ) -> None:
        _, result = _run_produce(store, strategy)
        assert [a.split(":")[1] for a in result["arms"]] == ["base"]
        document = _manifest(store, "experiment.run", RUN_DAY)
        assert document["status"] == "ok", document["reason"]
        rows = [m for m in document["metrics"] if m["name"] == ARM_REFUSED_METRIC]
        assert {r["status"] for r in rows} == {"unservable"}
        waiting = [r for r in rows if "arm 'stacked' is refused" in r["status_reason"]]
        assert len(waiting) == 1
        assert "predictions[base]" in waiting[0]["status_reason"]
        assert "no predictions artifact" in waiting[0]["status_reason"]

    def test_a_slot_of_nothing_but_waiting_stacks_is_unservable_and_pages(
        self, store, tmp_path
    ) -> None:
        """`ok` would be a lie: a slot that produced no cross-section did not
        have nothing to do."""
        arms = tmp_path / "cold" / "arms" / SLOT
        _write_recipe(arms, "base", features=[BASE_COLUMN])
        _write_recipe(arms, "only", features=[SECOND_COLUMN], inputs=["predictions[base]"])
        with pytest.raises(SlotUnservableError):
            _run_produce(store, _Settings(tmp_path / "cold"), arm_name="only")
        document = _manifest(store, "experiment.run", RUN_DAY)
        assert document["status"] == "failed"
        assert [m["name"] for m in document["metrics"]] == [ARM_REFUSED_METRIC]

    def test_the_grade_job_refuses_the_waiting_stack_the_same_way(self, store, strategy) -> None:
        """Same condition, same row, on the other job — and the cycle still
        runs for the arms that do have a panel."""
        for day in SESSIONS[15:21]:
            _run_produce(store, strategy, day=day)
        result: dict = {}
        run_job(
            "experiment.grade",
            lambda c: result.update(grade(c, settings=strategy)),
            store=store,
            trading_day=dt.date.fromisoformat(GRADE_DAY),
            run_mode="replay",
            discriminator=SLOT,
        )
        document = _manifest(store, "experiment.grade", GRADE_DAY)
        assert document["status"] == "ok", document["reason"]
        waiting = [
            m
            for m in document["metrics"]
            if m["name"] == ARM_REFUSED_METRIC and "arm 'stacked' is refused" in m["status_reason"]
        ]
        assert len(waiting) == 1
        assert list(result["model_grades"]) == [
            a for a in result["model_grades"] if ":base:" in a
        ], "an arm with no panel has no CPCV reading and no precondition to evaluate"

    def test_a_defective_feature_layer_is_still_slot_wide(self, store, tmp_path) -> None:
        """What the swallow does NOT absorb. A training window the layer
        cannot supply fails the whole slot, exactly as plan §4.4 requires."""
        from nousergon_lib.arena.engine import TrainingIntegrityError

        arms = tmp_path / "deep" / "arms" / SLOT
        _write_recipe(arms, "base", features=[BASE_COLUMN], min_days=5000)
        with pytest.raises(TrainingIntegrityError, match="min_trading_days"):
            _run_produce(store, _Settings(tmp_path / "deep"))


class TestTheIncumbentsMetricsComeFromThePointer:
    """`_incumbent_serving_metrics`: the champion's own cross-section, or `{}`.

    `{}` is not a neutral default — every dispersion rule is a RATIO against
    the incumbent, so an absent one makes the veto `insufficient` rather than
    a pass.
    """

    def _grade_at(self, store, strategy, day):
        result: dict = {}
        run_job(
            "experiment.grade",
            lambda c: result.update(grade(c, settings=strategy)),
            store=store,
            trading_day=dt.date.fromisoformat(day),
            run_mode="replay",
            discriminator=SLOT,
        )
        return result

    def test_a_champion_pointer_supplies_the_dispersion_denominator(self, store, strategy) -> None:
        from crucible.keys import champion_key
        from crucible.slots.model import _incumbent_serving_metrics

        _warm_the_base(store, strategy)
        for day in SESSIONS[46:52]:
            _run_produce(store, strategy, day=day)
        register = store.get_bytes(arm_register_key(SLOT)).decode("utf-8")
        champion = next(
            json.loads(line)["arm_id"]
            for line in register.splitlines()
            if line.strip() and ":base:" in line
        )
        store.put_bytes(champion_key(SLOT), json.dumps({"champion": champion}).encode("utf-8"))
        metrics = _incumbent_serving_metrics(store, as_of=GRADE_DAY)
        assert set(metrics) == {"alpha_stdev"}
        assert metrics["alpha_stdev"] > 0.0

    def test_a_pointer_to_an_arm_that_did_not_predict_today_supplies_nothing(
        self, store, strategy
    ) -> None:
        """Absent, never fabricated: a stale dispersion figure would let a
        collapsed candidate clear a ratio against a day it never ran."""
        from crucible.keys import champion_key
        from crucible.slots.model import _incumbent_serving_metrics

        store.put_bytes(
            champion_key(SLOT), json.dumps({"champion": "m:absent:0123456789ab"}).encode("utf-8")
        )
        assert _incumbent_serving_metrics(store, as_of=GRADE_DAY) == {}

    def test_a_champion_document_with_no_pointer_supplies_nothing(self, store) -> None:
        from crucible.keys import champion_key
        from crucible.slots.model import _incumbent_serving_metrics

        store.put_bytes(champion_key(SLOT), json.dumps({"champion": ""}).encode("utf-8"))
        assert _incumbent_serving_metrics(store, as_of=GRADE_DAY) == {}


class TestAnUnmeasurableCpcvCarriesNoNumber:
    def test_the_row_omits_value_and_unit_rather_than_reporting_a_zero(
        self, store, tmp_path
    ) -> None:
        """`unmeasurable` is a first-class status (plan §7). A number beside
        it would be read as a measurement nobody made — and `CPCVResult.
        mean_ic` refuses to be read at all in that state, which is what
        surfaced this."""
        arms = tmp_path / "thin" / "arms" / SLOT
        _write_recipe(arms, "base", features=[BASE_COLUMN])
        settings = _Settings(tmp_path / "thin")
        for day in SESSIONS[15:21]:
            _run_produce(store, settings, day=day)
        result: dict = {}
        run_job(
            "experiment.grade",
            lambda c: result.update(grade(c, settings=settings)),
            store=store,
            trading_day=dt.date.fromisoformat(GRADE_DAY),
            run_mode="replay",
            discriminator=SLOT,
        )
        document = _manifest(store, "experiment.grade", GRADE_DAY)
        rows = [m for m in document["metrics"] if m["name"] == CPCV_OOS_IC_METRIC]
        assert len(rows) == 1
        row = rows[0]
        if row["status"] == "unmeasurable":
            assert "value" not in row and "unit" not in row
            assert (
                result["model_grades"][next(iter(result["model_grades"]))]["cpcv_mean_ic"] is None
            )
        else:
            assert row["unit"] == "rank_ic" and isinstance(row["value"], float)
