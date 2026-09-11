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

import random

import pytest

from crucible.portfolio import CostModel
from crucible.slots import get_slot
from crucible.slots.strategy import (
    ATTESTATION_STATUSES,
    build_walk_forward_folds,
    grade_arm,
    load_strategy_recipes,
    pit_parity,
    render_verdict,
)


def _cost_model() -> CostModel:
    """The declared flat stand-in, as the filed recipe carries it."""
    return CostModel(
        name="flat_bps_v0",
        placeholder=True,
        params={"half_spread_bps": 2.5, "commission_bps": 0.5, "slippage_bps": 10.0},
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

    def test_an_unknown_spec_key_is_refused_by_name(self, tmp_path) -> None:
        """`alpha-engine-config-I9944`: an S recipe author writing
        `llm_callsite:` under `spec:` — the natural place for it — would
        otherwise register an arm the phase-5 LLM-arm gate silently never
        counts. Every unrecognised `spec` key is refused, named."""
        (tmp_path / "sneaky.yaml").write_text(
            "slot: s\nname: sneaky\nspec:\n"
            "  cost_model: {name: flat_bps_v0, placeholder: true, params: "
            "{half_spread_bps: 2.5, commission_bps: 0.5, slippage_bps: 10.0}}\n"
            "  rules:\n    - {rule_id: position_loss_floor,"
            " params: {position_loss_floor_pct: 0.08}}\n"
            "  llm_callsite: research.thinktank\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="llm_callsite") as exc:
            load_strategy_recipes(tmp_path)
        assert "phase-5 gate" in str(exc.value)

    def test_an_unknown_top_level_key_is_refused_by_name(self, tmp_path) -> None:
        (tmp_path / "extra.yaml").write_text(
            "slot: s\nname: extra\nbogus_field: nope\nspec:\n"
            "  cost_model: {name: flat_bps_v0, placeholder: true, params: "
            "{half_spread_bps: 2.5, commission_bps: 0.5, slippage_bps: 10.0}}\n"
            "  rules:\n    - {rule_id: position_loss_floor,"
            " params: {position_loss_floor_pct: 0.08}}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="bogus_field"):
            load_strategy_recipes(tmp_path)


class TestWalkForward:
    """The fold geometry, asserted as PROPERTIES rather than as arithmetic.

    Both tests in this class previously restated the implementation instead of
    the invariant, so both were true of the buggy implementation:

    * the purge test asserted ``train_end_idx <= test_start_idx - 21``, which
      is exactly what ``train_end_idx = fold_start_idx - purge`` produces. It
      encoded the off-by-one: the last training row's 21-day label lands ON
      the first test day, and the assertion written to the code could not see
      it. The invariant is stated below as the leak itself — no training
      row's label may extend into the test window.
    * the rolling test asserted ``train_start_idx > 0``, which is true of a
      bounded window of ANY length, including one bounded by the wrong
      quantity. It never checked the bound.
    """

    PURGE = 21

    def _folds(self, **kwargs):
        dates = [f"d{i:03d}" for i in range(600)]
        params = {"test_window": 21, "min_train": 100, "purge": self.PURGE, "embargo": 2}
        params.update(kwargs)
        return build_walk_forward_folds(dates, **params)

    def test_no_training_rows_label_reaches_into_the_test_window(self) -> None:
        """`purge` is the LABEL HORIZON, so the last trainable row is the one
        whose label is fully realized before the test window opens. Stated as
        the leak rather than as the arithmetic, because the arithmetic is
        what was wrong: `train_end_idx = fold_start_idx - purge` leaves a row
        whose label is realized ON `test_start_idx`."""
        folds = self._folds()
        assert folds
        for fold in folds:
            assert fold.train_end_idx + self.PURGE < fold.test_start_idx, (
                f"the training block ends at {fold.train_end_idx}, whose {self.PURGE}-day "
                f"label spans {fold.train_end_idx}..{fold.train_end_idx + self.PURGE} and "
                f"overlaps a test window opening at {fold.test_start_idx}"
            )
            assert fold.test_start_idx - fold.train_end_idx - 1 == self.PURGE, (
                "the gap must be exactly `purge` excluded rows; a wider one silently "
                "discards trainable data and a narrower one leaks"
            )

    def test_the_same_purge_arithmetic_as_the_cpcv_purge(self) -> None:
        """Two purge implementations in one repository must not differ by a
        day. `crucible.slots.model.cpcv_oos_ic` excludes `[a - h, a - 1]`,
        leaving `a - h - 1` as the last trainable row for a test block opening
        at `a`. The walk-forward folds must agree exactly."""
        for horizon in (1, 5, 21, 63):
            folds = build_walk_forward_folds(
                [f"d{i:03d}" for i in range(600)],
                test_window=21,
                min_train=100,
                purge=horizon,
                embargo=2,
            )
            assert folds
            for fold in folds:
                assert fold.train_end_idx == fold.test_start_idx - horizon - 1

    def test_an_expanding_fold_trains_on_at_least_min_train_rows(self) -> None:
        """`min_train` is a minimum training LENGTH. A fold that cannot reach
        it is skipped, never emitted short — a spec declaring 100 training
        rows that trains on 79 of them is a geometry nobody declared."""
        min_train = 100
        folds = self._folds(min_train=min_train)
        assert folds
        for fold in folds:
            assert fold.train_start_idx == 0, "expanding"
            length = fold.train_end_idx - fold.train_start_idx + 1
            assert length >= min_train, (
                f"fold trains on {length} rows against a declared min_train={min_train}"
            )

    def test_a_rolling_window_is_exactly_min_train_rows_long(self) -> None:
        """The property the previous test NAMED but did not check: the bound.
        `train_start_idx > 0` holds for a window of any length; it holds just
        as well for the 21-row window the lifted source produced against a
        `min_train=504` spec. The bound is asserted as a length, and the
        window is asserted to actually ROLL rather than expand."""
        min_train = 100
        folds = self._folds(min_train=min_train, train_mode="rolling")
        assert len(folds) > 1
        for fold in folds:
            length = fold.train_end_idx - fold.train_start_idx + 1
            assert length == min_train, (
                f"rolling window is {length} rows against a declared min_train={min_train}"
            )
        starts = [f.train_start_idx for f in folds]
        assert starts == sorted(starts) and starts[-1] > starts[0], (
            "a rolling window must move forward; a constant start is an expanding window"
        )

    def test_an_embargo_wider_than_the_purge_skips_the_next_block(self) -> None:
        """The second property the fold geometry exists for: when
        `embargo > purge` the next fold starts later, so the serial
        correlation immediately after a test block is not scored either."""
        purge, embargo, test_window = 21, 40, 21
        folds = build_walk_forward_folds(
            [f"d{i:03d}" for i in range(900)],
            test_window=test_window,
            min_train=100,
            purge=purge,
            embargo=embargo,
        )
        assert len(folds) > 1
        for prev, nxt in zip(folds, folds[1:], strict=False):
            skipped = nxt.test_start_idx - prev.test_end_idx - 1
            assert skipped >= embargo - purge, (
                f"only {skipped} day(s) between test blocks for an embargo of {embargo} "
                f"over a purge of {purge}"
            )

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


FULL_COVERAGE = {"coverage_fraction": 1.0, "budget_stopped": False, "measured": True}


def _days(n: int) -> list[str]:
    return [f"d{i:03d}" for i in range(n)]


def _series(values: list[float]) -> dict[str, float]:
    return dict(zip(_days(len(values)), values, strict=True))


class TestPitParity:
    def test_a_material_delta_is_a_fail(self) -> None:
        verdict = pit_parity(
            contaminated=_series([0.02] * 60),
            point_in_time=_series([0.001] * 60),
            coverage=FULL_COVERAGE,
        )
        assert verdict.status == "FAIL"
        assert "MATERIAL contamination" in verdict.reason

    def test_no_coverage_is_unknown_never_a_pass(self) -> None:
        verdict = pit_parity(
            contaminated={},
            point_in_time={},
            coverage={"coverage_fraction": 0.0, "budget_stopped": False, "measured": False},
        )
        assert verdict.status == "UNKNOWN"
        assert verdict.reason

    def test_partial_coverage_is_partial_never_a_pass(self) -> None:
        verdict = pit_parity(
            contaminated=_series([0.001] * 60),
            point_in_time=_series([0.001] * 60),
            coverage={"coverage_fraction": 0.6, "budget_stopped": True, "measured": True},
        )
        assert verdict.status == "PARTIAL"

    def test_an_indistinguishable_delta_over_full_coverage_passes(self) -> None:
        verdict = pit_parity(
            contaminated=_series([0.001, 0.0012, 0.0009] * 20),
            point_in_time=_series([0.001, 0.0011, 0.001] * 20),
            coverage=FULL_COVERAGE,
        )
        assert verdict.status == "PASS"

    def test_the_status_vocabulary_is_closed(self) -> None:
        assert ATTESTATION_STATUSES == ("PASS", "FAIL", "PARTIAL", "UNKNOWN")

    def test_pairing_is_by_date_so_a_reordered_series_is_the_same_verdict(self) -> None:
        """Positional pairing makes the ORDER of the two mappings load-bearing.
        Date-keyed pairing does not, and that is the point."""
        contaminated = _series([0.02] * 60)
        point_in_time = _series([0.001] * 60)
        shuffled = dict(reversed(list(point_in_time.items())))
        assert (
            pit_parity(
                contaminated=contaminated, point_in_time=shuffled, coverage=FULL_COVERAGE
            ).mean_delta
            == pit_parity(
                contaminated=contaminated, point_in_time=point_in_time, coverage=FULL_COVERAGE
            ).mean_delta
        )

    def test_one_extra_date_on_one_side_raises_instead_of_pairing_off_by_one(self) -> None:
        """The demonstrated defect, with the demonstrated numbers.

        `random.seed(2)`, 60 days of N(0, 0.05) point-in-time returns plus a
        FLAT +150bp/day of real look-ahead alpha, full proven coverage. Paired
        correctly this renders FAIL (mean delta 0.0150, interval excluding
        zero). Under the previous positional pairing, prepending ONE element
        to the contaminated side truncated both sides to 60 and shifted every
        pair by a day, rendering PASS (mean delta 0.0155, interval straddling
        zero) with no exception — and `champion._assert_attested` treats that
        PASS as the sole gate on an S champion.

        There is no verdict to render from two passes that scored different
        windows, so this raises rather than answering."""
        random.seed(2)
        point_in_time = {d: random.gauss(0.0, 0.05) for d in _days(60)}
        contaminated = {d: v + 0.015 for d, v in point_in_time.items()}

        aligned = pit_parity(
            contaminated=contaminated, point_in_time=point_in_time, coverage=FULL_COVERAGE
        )
        assert aligned.status == "FAIL"
        assert aligned.mean_delta == pytest.approx(0.015)

        with pytest.raises(ValueError, match="scored different dates"):
            pit_parity(
                contaminated={"d999": 0.0, **contaminated},
                point_in_time=point_in_time,
                coverage=FULL_COVERAGE,
            )

    def test_a_missing_date_on_the_contaminated_side_raises(self) -> None:
        contaminated = _series([0.001] * 60)
        point_in_time = _series([0.001] * 60)
        contaminated.pop("d030")
        with pytest.raises(ValueError, match="scored different dates"):
            pit_parity(
                contaminated=contaminated, point_in_time=point_in_time, coverage=FULL_COVERAGE
            )

    def test_a_positional_sequence_is_refused_outright(self) -> None:
        """The signature is the fix: positional pairing must be unexpressible,
        not merely unused."""
        with pytest.raises(TypeError, match="mapping"):
            pit_parity(
                contaminated=[0.02] * 60,  # type: ignore[arg-type]
                point_in_time=_series([0.001] * 60),
                coverage=FULL_COVERAGE,
            )

    def test_an_inverted_delta_fails_and_names_a_broken_harness(self) -> None:
        """The materiality test is two-sided ON PURPOSE, and that is kept: a
        point-in-time pass beating its own look-ahead pass is not a clean
        result, so rendering PASS would be a fail-open on the one gate an S
        champion has. What was wrong is the MESSAGE — a broken harness
        reported as "MATERIAL contamination" sends the operator to hunt a leak
        that is not there."""
        verdict = pit_parity(
            contaminated=_series([0.001] * 60),
            point_in_time=_series([0.02] * 60),
            coverage=FULL_COVERAGE,
        )
        assert verdict.status == "FAIL"
        assert verdict.mean_delta is not None and verdict.mean_delta < 0
        assert "INVERTED" in verdict.reason
        assert "contamination finding" in verdict.reason
        assert "MATERIAL contamination" not in verdict.reason


class TestAttestationGatesTheCard:
    @pytest.mark.parametrize("status", ["FAIL", "PARTIAL", "UNKNOWN"])
    def test_a_card_without_a_pass_renders_unverified(self, status: str) -> None:
        card = render_verdict(
            arm_id="s:stock_registry:0123456789abcdef",
            as_of="2026-08-28",
            alpha_vs_spy=0.031,
            attestation={"kind": "pit_parity", "status": status, "reason": "x"},
            cost_model=_cost_model().record(),
        )
        assert card["rendered"] == "UNVERIFIED"
        assert "grade" not in card, "an unverified card must not carry a grade at all"
        assert card["cost_model"]["name"] == "flat_bps_v0", (
            "an unverified card still names the cost model: what a run was charged is "
            "a fact about the run, not a decoration on a grade"
        )

    def test_a_card_with_a_pass_renders_its_grade(self) -> None:
        card = render_verdict(
            arm_id="s:stock_registry:0123456789abcdef",
            as_of="2026-08-28",
            alpha_vs_spy=0.031,
            attestation={"kind": "pit_parity", "status": "PASS", "reason": "x"},
            cost_model=_cost_model().record(),
        )
        assert card["rendered"] == "GRADED"
        assert card["grade"]["alpha_vs_spy"] == 0.031
        assert card["cost_model"]["placeholder"] is True, (
            "a grade taken net of a stand-in says so on the card's face"
        )

    def test_a_card_cannot_be_rendered_without_naming_its_cost_model(self) -> None:
        """`alpha-engine-config-I10503`. A net-of-cost number is a claim about
        what was charged, and a card able to make it without naming the model is
        the shape in which an optimizer ran on a cost model nobody configured.
        """
        with pytest.raises(ValueError, match="must name the cost model"):
            render_verdict(
                arm_id="s:stock_registry:0123456789abcdef",
                as_of="2026-08-28",
                alpha_vs_spy=0.031,
                attestation={"kind": "pit_parity", "status": "PASS", "reason": "x"},
                cost_model={},
            )


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
