"""One unproducible arm refuses ITSELF, not the slot (`alpha-engine-config-I9955`).

The measured condition, 2026-09-04, against `alpha-engine-config` branch
`feat/m-stacked-arm-recipe-i9777` with crucible `main` at `e41667f`::

    UnproducibleInputError: arm 'sota_directional_combine' declares input
    'predictions[gbm_directional]', but no recipe named 'gbm_directional'
    exists in this slot; the slot declares
    ['directional_combine_residual_stack', 'residual_momentum',
     'sota_directional_combine'].

Three arms in the directory. **One** unproducible. **None** of the three
loaded — including `residual_momentum`, the subject of the only genuine
promotion in the fleet's history, and
`directional_combine_residual_stack`, the arm Brian's
`alpha-engine-config-I9808` ruling (b) created on 2026-09-04 for the express
purpose of letting the M slot accumulate evidence before phase 5. Option (b)
was chosen over (c) — carry the slot as `unservable` until phase 5 — for
exactly that reason, and a directory-wide refusal defeats it.

**This is not a softening, and every test below is written to catch one.**
A refused arm is still refused, still names the exact input nothing produces,
is RECORDED as a metric on the manifest of the job that loaded the slot, and
when NOTHING registers it raises and PAGES through the existing failure
condition. What changed is the blast radius. Two conditions deliberately stay
slot-wide, and are pinned here: a dependency cycle, and two recipes sharing a
name — both are properties of the graph rather than of one member.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from crucible.alerts import evaluate_failure
from crucible.manifest import read_manifest
from crucible.runner import run_job
from crucible.slots.inputs import (
    InputCycleError,
    InputRefusal,
    SlotUnservableError,
    UnproducibleInputError,
)
from crucible.slots.model import ARM_REFUSED_METRIC, SlotRecipes, load_model_recipes
from crucible.store import LocalStore

_LAYER = ("mom_21d_ratio", "vol_21d_ratio")

#: A fixed session (§4.12, test discipline: never `today` arithmetic). Friday
#: 2026-08-28 is an ordinary NYSE session and is the day the rest of this
#: suite pins its manifests to.
FRIDAY = dt.date(2026, 8, 28)


def _write(directory: Path, name: str, *, features: str, inputs: tuple[str, ...] = ()) -> None:
    """A recipe file, written the way the private strategy tree writes one."""
    directory.mkdir(parents=True, exist_ok=True)
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


def _mixed_slot(tmp_path: Path) -> Path:
    """The measured shape: one producible arm, one that cannot be built.

    Named for what it stands in for — `residual_momentum` beside
    `sota_directional_combine` — rather than for the mechanism, so the test
    that FAILS against today's `main` reads as the real condition it is.
    """
    arms = tmp_path / "arms"
    _write(arms, "residual_momentum_like", features="[mom_21d_ratio]")
    _write(
        arms,
        "sota_directional_combine_like",
        features="[vol_21d_ratio]",
        inputs=("predictions[gbm_directional]",),
    )
    return arms


class TestOneRefusedArmDoesNotTakeTheSlotDown:
    """The defect, stated as the property that was missing.

    This class is the §7.4 demonstration: every test in it FAILS against
    `main` before the fix, with `UnproducibleInputError` raised out of
    `load_model_recipes` for the whole directory.
    """

    def test_the_producible_arm_registers_while_its_sibling_is_refused(self, tmp_path) -> None:
        loaded = load_model_recipes(_mixed_slot(tmp_path), feature_columns=_LAYER)

        assert [r.name for r in loaded.registered] == ["residual_momentum_like"], (
            "the producible arm loads. Before I9955 the whole directory raised, so an arm "
            "with nothing wrong with it could not be graded because a SIBLING named a "
            "base model that does not exist until phase 5."
        )
        assert not loaded.unservable, "one arm registered, so the slot serves"

    def test_the_refusal_names_the_arm_and_the_exact_unresolvable_input(self, tmp_path) -> None:
        """The refusal is not softened — it is as precise as the exception was."""
        loaded = load_model_recipes(_mixed_slot(tmp_path), feature_columns=_LAYER)

        assert len(loaded.refused) == 1
        refusal = loaded.refused[0]
        assert refusal.arm == "sota_directional_combine_like"
        assert refusal.unresolvable == ("predictions[gbm_directional]",)
        assert "no recipe named ['gbm_directional']" in refusal.reason
        assert "does not register" in refusal.reason

    def test_a_refused_arm_is_absent_from_the_registered_set(self, tmp_path) -> None:
        """`registered` is what may be graded, and nothing else is in it.

        A partition that reported the refusal AND still handed the arm back
        would be the original defect with a report attached.
        """
        loaded = load_model_recipes(_mixed_slot(tmp_path), feature_columns=_LAYER)
        assert "sota_directional_combine_like" not in {r.name for r in loaded.registered}

    def test_a_missing_feature_column_refuses_only_its_own_arm(self, tmp_path) -> None:
        """The other declaration channel, partitioned the same way.

        `spec.features` naming a column the layer does not produce is the
        original I9777 condition; it refused the whole directory too.
        """
        arms = tmp_path / "arms"
        _write(arms, "good", features="[mom_21d_ratio]")
        _write(arms, "bad", features="[mom_21d_ratio, gbm_directional_score_zscore]")

        loaded = load_model_recipes(arms, feature_columns=_LAYER)

        assert [r.name for r in loaded.registered] == ["good"]
        assert [r.arm for r in loaded.refused] == ["bad"]
        assert "gbm_directional_score_zscore" in loaded.refused[0].reason


class TestARefusedArmIsRecordedOnTheManifest:
    """Deliverable 2. A refusal that produces no record is the defect moved."""

    def test_each_refused_arm_produces_one_metric_naming_it_and_its_input(self, tmp_path) -> None:
        loaded = load_model_recipes(_mixed_slot(tmp_path), feature_columns=_LAYER)

        metrics = loaded.refusal_metrics(slot="m")

        assert len(metrics) == 1, "one row per refused arm — not one row for the slot"
        metric = metrics[0]
        assert metric["name"] == ARM_REFUSED_METRIC
        assert metric["status"] == "unservable", (
            "the arena's own first-class status (plan §7), forwarded verbatim rather than "
            "re-encoded into a second word that would drift from the first"
        )
        assert "sota_directional_combine_like" in metric["status_reason"]
        assert "predictions[gbm_directional]" in metric["status_reason"]
        assert metric["source_path"].endswith("sota_directional_combine_like.yaml")

    def test_a_slot_with_nothing_refused_emits_no_refusal_metric(self, tmp_path) -> None:
        """Rule 7 read the other way: a metric that always fires says nothing."""
        arms = tmp_path / "arms"
        _write(arms, "clean", features="[mom_21d_ratio]")
        assert load_model_recipes(arms, feature_columns=_LAYER).refusal_metrics() == []

    def test_the_metric_survives_the_run_manifest_schema_on_a_real_run(self, tmp_path) -> None:
        """The whole path, not the shape: recorded through `run_job`, read back.

        A metric asserted only as a dict proves the producer agrees with
        itself. This one is written by the runner, validated against
        `run_manifest.v2.json` on the way out, and read off the store.
        """
        store = LocalStore(tmp_path)
        arms = _mixed_slot(tmp_path)

        def _fn(ctx):
            loaded = load_model_recipes(arms, feature_columns=_LAYER)
            for metric in loaded.refusal_metrics(slot="m"):
                ctx.record_metric(metric)
            return {"registered": [r.name for r in loaded.registered]}

        run_job(
            "experiment.run",
            _fn,
            store=store,
            trading_day=FRIDAY,
            now=dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC),
            run_mode="replay",
        )

        # `read_manifest` validates against `run_manifest.v2.json` on the way
        # in, so a metric row the schema would refuse fails here.
        manifest = read_manifest(store, "experiment.run", FRIDAY.isoformat())
        assert manifest["status"] == "ok", "one arm registered, so the slot served"
        rows = [m for m in manifest["metrics"] if m["name"] == ARM_REFUSED_METRIC]
        assert len(rows) == 1
        assert "sota_directional_combine_like" in rows[0]["status_reason"]
        assert "predictions[gbm_directional]" in rows[0]["status_reason"]


class TestASlotThatCanServeNothingIsUnservableAndPages:
    """Deliverable 3, and the reason this change is not a softening.

    `unservable` is a first-class status that PAGES (plan §5.3, §7). It pages
    through the existing FAILURE condition — the raise reaches `run_job`'s
    `try/finally`, which writes `status: failed` with the cause — so no third
    page condition is invented (`crucible/AGENTS.md`, Alerting).
    """

    def test_every_arm_refused_raises_slot_wide(self, tmp_path) -> None:
        arms = tmp_path / "arms"
        _write(arms, "one", features="[mom_21d_ratio]", inputs=("predictions[nowhere]",))
        _write(arms, "two", features="[nowhere_ratio]")

        with pytest.raises(SlotUnservableError) as excinfo:
            load_model_recipes(arms, feature_columns=_LAYER)

        message = str(excinfo.value)
        assert "unservable" in message
        assert {r.arm for r in excinfo.value.refusals} == {"one", "two"}
        assert "nowhere" in message and "nowhere_ratio" in message, (
            "the whole-slot refusal still names every arm and every unresolvable input"
        )

    def test_a_single_arm_slot_whose_only_arm_is_refused_still_raises(self, tmp_path) -> None:
        """The I9777 behaviour, unchanged where it was already correct.

        A one-arm directory has no sibling to protect, so the per-arm
        partition and the slot-wide refusal are the same thing — and a
        subclass of `UnproducibleInputError` is what keeps every caller that
        already handled that exception working.
        """
        arms = tmp_path / "arms"
        _write(arms, "solo", features="[mom_21d_ratio]", inputs=("predictions[gbm_directional]",))
        with pytest.raises(UnproducibleInputError, match="gbm_directional"):
            load_model_recipes(arms, feature_columns=_LAYER)

    def test_the_unservable_slot_writes_a_failed_manifest_and_pages(self, tmp_path) -> None:
        """The guard, made to fire end to end.

        A refused arm is NOT quieter than it was: an unservable slot still
        produces a page, and this test walks the whole path — loader raises,
        `run_job` writes `status: failed`, `evaluate_failure` returns a page
        naming the run.
        """
        store = LocalStore(tmp_path)
        arms = tmp_path / "arms"
        _write(arms, "only", features="[mom_21d_ratio]", inputs=("predictions[gbm_directional]",))

        with pytest.raises(SlotUnservableError):
            run_job(
                "experiment.run",
                lambda ctx: load_model_recipes(arms, feature_columns=_LAYER),
                store=store,
                trading_day=FRIDAY,
                now=dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC),
                run_mode="replay",
                transient_retry=False,
            )

        manifest = read_manifest(store, "experiment.run", FRIDAY.isoformat())
        assert manifest["status"] == "failed"
        assert "unservable" in manifest["reason"]
        assert "gbm_directional" in manifest["reason"]

        pages = evaluate_failure(store, now=dt.datetime(2026, 8, 28, 23, 0, tzinfo=dt.UTC))
        assert [p.job for p in pages] == ["experiment.run"], (
            "an unservable M slot pages. It does so through the FAILURE condition, which "
            "already exists — there are exactly two page conditions and this change adds "
            "none."
        )

    def test_an_empty_directory_is_not_unservable(self, tmp_path) -> None:
        """Nothing registered AND nothing refused is an absence, not a refusal.

        Calling it `unservable` would page for a slot nobody has written an
        arm for yet — a condition `crucible.alerts`'s absence half already
        owns.
        """
        arms = tmp_path / "arms"
        arms.mkdir()
        loaded = load_model_recipes(arms, feature_columns=_LAYER)
        assert loaded.registered == ()
        assert loaded.refused == ()
        assert not loaded.unservable


class TestWhatStaysSlotWide:
    """A cycle and an ambiguous name belong to the GRAPH, not to a member."""

    def test_a_genuine_cycle_still_refuses_the_whole_slot(self, tmp_path) -> None:
        """Even beside an arm that would otherwise register.

        Partitioning a cycle into "each of these arms is individually
        refused" loses the one fact an operator needs, which is that they
        point at each other.
        """
        arms = tmp_path / "arms"
        _write(arms, "innocent", features="[mom_21d_ratio]")
        _write(arms, "alpha", features="[mom_21d_ratio]", inputs=("predictions[beta]",))
        _write(arms, "beta", features="[vol_21d_ratio]", inputs=("predictions[alpha]",))

        with pytest.raises(InputCycleError) as excinfo:
            load_model_recipes(arms, feature_columns=_LAYER)
        assert "alpha -> beta -> alpha" in str(excinfo.value)

    def test_a_cycle_is_reported_even_when_another_arm_is_unproducible(self, tmp_path) -> None:
        """The cycle check runs BEFORE the partition, deliberately.

        Dropping refused arms first could make a real cycle disappear from
        the graph, and a slot that quietly loads two arms pointing at each
        other is worse than one that refuses to load at all.
        """
        arms = tmp_path / "arms"
        _write(arms, "alpha", features="[mom_21d_ratio]", inputs=("predictions[beta]",))
        _write(arms, "beta", features="[vol_21d_ratio]", inputs=("predictions[alpha]",))
        _write(arms, "unbuildable", features="[nowhere_ratio]")

        with pytest.raises(InputCycleError):
            load_model_recipes(arms, feature_columns=_LAYER)

    def test_two_recipes_sharing_a_name_still_refuse_the_whole_slot(self, tmp_path) -> None:
        """An ambiguous graph cannot be partitioned honestly.

        Two files declaring one name: a `predictions[...]` reference resolves
        to whichever loaded last, so no arm in the directory can be trusted
        to have read the graph the operator wrote.
        """
        arms = tmp_path / "arms"
        _write(arms, "first", features="[mom_21d_ratio]")
        (arms / "second.yaml").write_text(
            (arms / "first.yaml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        with pytest.raises(UnproducibleInputError, match="share the name"):
            load_model_recipes(arms, feature_columns=_LAYER)


class TestRefusalIsTransitive:
    """An arm stacking on a refused arm cannot be graded either."""

    def test_an_arm_stacking_on_a_refused_arm_is_refused_too(self, tmp_path) -> None:
        """Otherwise `registers fine, dies at grading` moves up one level.

        `middle` names a base that does not exist, so it is refused. `top`
        names `middle`, which DOES exist as a recipe — its own declaration
        parses cleanly — and it still has no producer, because the arm it
        consumes has none.
        """
        arms = tmp_path / "arms"
        _write(arms, "fine", features="[mom_21d_ratio]")
        _write(arms, "middle", features="[mom_21d_ratio]", inputs=("predictions[absent]",))
        _write(arms, "top", features="[vol_21d_ratio]", inputs=("predictions[middle]",))

        loaded = load_model_recipes(arms, feature_columns=_LAYER)

        assert [r.name for r in loaded.registered] == ["fine"]
        assert [r.arm for r in loaded.refused] == ["middle", "top"]
        assert loaded.refused[1].unresolvable == ("predictions[middle]",)
        assert "itself refused" in loaded.refused[1].reason


class TestTheRefusalValueRefusesToBeEmpty:
    """The record cannot be shaped so that it says nothing."""

    def test_a_refusal_naming_no_unresolvable_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names no unresolvable input"):
            InputRefusal(arm="x", unresolvable=(), reason="because")

    def test_a_refusal_with_no_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not name the arm"):
            InputRefusal(arm="  ", unresolvable=("predictions[y]",), reason="because")

    def test_an_arm_cannot_be_both_registered_and_refused(self) -> None:
        """A set saying both renders as healthy on whichever half is read first."""
        from tests.test_slot_model import _recipe

        with pytest.raises(ValueError, match="both registered and refused"):
            SlotRecipes(
                registered=(_recipe(),),
                refused=(
                    InputRefusal(
                        arm=_recipe().name, unresolvable=("predictions[y]",), reason="because"
                    ),
                ),
            )
