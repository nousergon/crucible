"""The S slot: exit-rule recipes, walk-forward grading, and the attestation.

Normative sources: plan §4.4 (S is graded market-relative vs SPY, net of the
cost model), §9.1 (contamination attestation — "a card without
`attestation: PASS` renders UNVERIFIED, never a grade"), §10.2 (the real cost
model is phase 3; phase 1 declares a NAMED placeholder in the arm file).

S is the one slot benchmarked against SPY, and that is deliberate: its output
IS a market position, so the population rule that protects U and R would be
the wrong axis here. The test below pins that so the population rule is not
over-applied into a second defect.
"""

from __future__ import annotations

import pytest

from crucible.slots import get_slot
from crucible.slots.strategy import (
    ATTESTATION_STATUSES,
    build_walk_forward_folds,
    grade_arm,
    load_strategy_recipes,
    pit_parity,
    render_verdict,
)

SEVEN = (
    "position_loss_floor",
    "catalyst_hard_exit",
    "atr_trailing_stop",
    "fallback_stop",
    "profit_take",
    "momentum_exit",
    "time_decay",
)


class TestBenchmark:
    def test_s_is_benchmarked_against_spy_and_the_others_are_not(self) -> None:
        assert get_slot("s").benchmark == "SPY"
        assert get_slot("s").slot_kind == "strategy"
        for selection in ("u", "r"):
            assert get_slot(selection).benchmark == "population"


class TestRecipes:
    def test_the_champion_recipe_carries_the_seven_stock_registry_rules(self, arm_dir) -> None:
        recipes = load_strategy_recipes(arm_dir)
        champion = next(r for r in recipes if r.name == "stock_registry")
        assert tuple(rule.rule_id for rule in champion.rules) == SEVEN, (
            "the 7 exit rules run as an ordered first-decision-wins chain; the ORDER "
            "is part of the recipe, so a reordering is a different arm"
        )

    def test_an_edited_recipe_is_a_new_arm(self, arm_dir) -> None:
        recipes = load_strategy_recipes(arm_dir)
        champion = next(r for r in recipes if r.name == "stock_registry")
        tighter = champion.with_rule_params("profit_take", {"profit_take_pct": 0.20})
        assert tighter.arm_id != champion.arm_id
        assert tighter.supersedes == champion.arm_id

    def test_a_reordered_chain_is_a_new_arm(self, arm_dir) -> None:
        recipes = load_strategy_recipes(arm_dir)
        champion = next(r for r in recipes if r.name == "stock_registry")
        flipped = champion.with_rules(tuple(reversed(champion.rules)))
        assert flipped.arm_id != champion.arm_id

    def test_the_cost_model_is_a_named_constant_declared_in_the_arm_file(self, arm_dir) -> None:
        """§10.2's real cost model is phase 3. Phase 1 must therefore NAME the
        placeholder in the recipe rather than bury a number in the grader —
        a grade net of an unnamed cost is a grade nobody can reproduce."""
        champion = next(r for r in load_strategy_recipes(arm_dir) if r.name == "stock_registry")
        assert champion.cost_model.name == "flat_bps_v0"
        assert champion.cost_model.placeholder is True
        assert champion.cost_model.params["half_spread_bps"] == 2.5

    def test_a_recipe_with_an_unknown_rule_is_refused(self, tmp_path) -> None:
        (tmp_path / "bogus.yaml").write_text(
            "slot: s\nname: bogus\nspec:\n"
            "  cost_model: {name: flat_bps_v0, placeholder: true, params: "
            "{half_spread_bps: 2.5, commission_bps: 0.5, slippage_bps: 10.0}}\n"
            "  rules:\n    - {rule_id: teleport, params: {}}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="teleport"):
            load_strategy_recipes(tmp_path)


class TestWalkForward:
    def test_folds_purge_and_embargo_around_the_test_block(self) -> None:
        dates = [f"d{i:03d}" for i in range(300)]
        folds = build_walk_forward_folds(dates, test_window=21, min_train=100, purge=21, embargo=2)
        assert folds
        for fold in folds:
            assert fold.train_end_idx <= fold.test_start_idx - 21
            assert fold.train_start_idx == 0  # expanding

    def test_a_rolling_window_is_bounded_by_the_test_window(self) -> None:
        dates = [f"d{i:03d}" for i in range(300)]
        folds = build_walk_forward_folds(
            dates, test_window=21, min_train=100, purge=21, embargo=2, train_mode="rolling"
        )
        assert all(f.train_start_idx > 0 for f in folds[1:])

    def test_an_unknown_train_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="train_mode"):
            build_walk_forward_folds(
                ["a", "b", "c"],
                test_window=1,
                min_train=1,
                purge=0,
                embargo=0,
                train_mode="teleport",
            )


class TestPitParity:
    def test_a_material_delta_is_a_fail(self) -> None:
        verdict = pit_parity(
            contaminated=[0.02] * 60,
            point_in_time=[0.001] * 60,
            coverage={"coverage_fraction": 1.0, "budget_stopped": False, "measured": True},
        )
        assert verdict.status == "FAIL"

    def test_no_coverage_is_unknown_never_a_pass(self) -> None:
        verdict = pit_parity(
            contaminated=[],
            point_in_time=[],
            coverage={"coverage_fraction": 0.0, "budget_stopped": False, "measured": False},
        )
        assert verdict.status == "UNKNOWN"
        assert verdict.reason

    def test_partial_coverage_is_partial_never_a_pass(self) -> None:
        verdict = pit_parity(
            contaminated=[0.001] * 60,
            point_in_time=[0.001] * 60,
            coverage={"coverage_fraction": 0.6, "budget_stopped": True, "measured": True},
        )
        assert verdict.status == "PARTIAL"

    def test_an_indistinguishable_delta_over_full_coverage_passes(self) -> None:
        verdict = pit_parity(
            contaminated=[0.001, 0.0012, 0.0009] * 20,
            point_in_time=[0.001, 0.0011, 0.001] * 20,
            coverage={"coverage_fraction": 1.0, "budget_stopped": False, "measured": True},
        )
        assert verdict.status == "PASS"

    def test_the_status_vocabulary_is_closed(self) -> None:
        assert ATTESTATION_STATUSES == ("PASS", "FAIL", "PARTIAL", "UNKNOWN")


class TestAttestationGatesTheCard:
    @pytest.mark.parametrize("status", ["FAIL", "PARTIAL", "UNKNOWN"])
    def test_a_card_without_a_pass_renders_unverified(self, status: str) -> None:
        card = render_verdict(
            arm_id="s:stock_registry:0123456789abcdef",
            as_of="2026-08-28",
            alpha_vs_spy=0.031,
            attestation={"kind": "pit_parity", "status": status, "reason": "x"},
        )
        assert card["rendered"] == "UNVERIFIED"
        assert "grade" not in card, "an unverified card must not carry a grade at all"

    def test_a_card_with_a_pass_renders_its_grade(self) -> None:
        card = render_verdict(
            arm_id="s:stock_registry:0123456789abcdef",
            as_of="2026-08-28",
            alpha_vs_spy=0.031,
            attestation={"kind": "pit_parity", "status": "PASS", "reason": "x"},
        )
        assert card["rendered"] == "GRADED"
        assert card["grade"]["alpha_vs_spy"] == 0.031


class TestGrade:
    def test_the_grade_is_net_of_the_declared_cost_model(self, arm_dir, book) -> None:
        champion = next(r for r in load_strategy_recipes(arm_dir) if r.name == "stock_registry")
        gross = grade_arm(champion, book, as_of="2026-08-28", apply_costs=False)
        net = grade_arm(champion, book, as_of="2026-08-28")
        assert net.total_cost_bps > 0
        assert sum(net.series.scores.values()) < sum(gross.series.scores.values())
        assert net.cost_model_name == "flat_bps_v0"

    def test_the_series_is_market_relative(self, arm_dir, book) -> None:
        champion = next(r for r in load_strategy_recipes(arm_dir) if r.name == "stock_registry")
        graded = grade_arm(champion, book, as_of="2026-08-28")
        assert graded.benchmark == "SPY"
        assert set(graded.series.scores) <= set(book.dates)


@pytest.fixture
def arm_dir(tmp_path):
    from tests.support.arms import write_strategy_arms

    return write_strategy_arms(tmp_path)


@pytest.fixture
def book():
    from tests.support.panels import synthetic_book

    return synthetic_book(n_days=60, seed=11)
