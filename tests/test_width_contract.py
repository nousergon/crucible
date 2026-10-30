"""Variable-width selection families bind to the information ratio (alpha-engine-config-I12565).

Audit CR-05: the pinned R recipes declare `no_agent_quant` at 15 names and
every sibling at 10, the R `SlotSpec` passed no `promote_statistic`, so the
pointer ranked those arms on the library default — the mean paired
difference — which rewards concentration: mean alpha per name falls with
depth whenever a ranking carries signal, so the narrowest arm wins a
comparison it did not earn. These tests hold the CR-05 closes-when:

* every family's width semantics is validated, and a family whose widths
  vary under no declaration is REFUSED (the carryover gate);
* a 10-vs-15-name family where the narrow arm leads on the mean but trails on
  IR is decided by IR when width is experimental;
* a ranking-only family must be equal-width;
* the contract that decided a cycle is stamped on the cycle artifact.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from nousergon_lib.arena.engine import promotion_statistic
from nousergon_lib.arena.window import ArmSeries, pair_on_common_window

from crucible.slots import (
    EVIDENCE_ANYTIME_VALID,
    EVIDENCE_POINT,
    PROMOTE_AGAINST_INCUMBENT,
    SLOTS,
    STATISTIC_INFORMATION_RATIO,
    STATISTIC_MEAN_DIFF,
    WIDTH_EXPERIMENTAL,
    WIDTH_RANKING_ONLY,
    WidthContractError,
    assert_width_contract,
    selection_widths,
)


def _spec(name: str, top_n: int | None, *, control: bool = False) -> SimpleNamespace:
    params = {} if top_n is None else {"top_n": top_n}
    return SimpleNamespace(name=name, arm_id=f"r:{name}:0", params=params, control=control)


#: The pinned R family the audit cited (alpha-engine-config strategy/arms/r/).
R_FAMILY = [
    _spec("no_agent_quant", 15),
    _spec("scanner_predictor_direct", 10),
    _spec("scanner_top20_predictor", 10),
    _spec("sleeve_top10", 10),
    _spec("thinktank_coverage", 10),
]


def _slot(**changes):
    return dataclasses.replace(SLOTS["r"], **changes)


class TestTheRBinding:
    """Brian's 2026-10-23 ruling on I12565 bound option (a): R is experimental."""

    def test_r_declares_experimental_widths_on_ir_and_point(self) -> None:
        r = SLOTS["r"]
        assert r.width_semantics == WIDTH_EXPERIMENTAL
        assert r.promote_statistic == STATISTIC_INFORMATION_RATIO
        assert r.promote_evidence == EVIDENCE_POINT

    def test_the_pinned_r_family_now_passes_the_gate(self) -> None:
        record = assert_width_contract(SLOTS["r"], R_FAMILY)
        assert record["width_semantics"] == WIDTH_EXPERIMENTAL
        assert record["widths"]["r:no_agent_quant:0"] == 15


class TestCarryoverGate:
    def test_an_undeclared_variable_width_family_is_refused(self) -> None:
        with pytest.raises(WidthContractError, match="declares no width_semantics"):
            assert_width_contract(_slot(width_semantics=None), R_FAMILY)

    def test_a_ranking_only_family_must_be_equal_width(self) -> None:
        with pytest.raises(WidthContractError, match="same number of names"):
            assert_width_contract(_slot(width_semantics=WIDTH_RANKING_ONLY), R_FAMILY)
        equal = [_spec(s.name, 10) for s in R_FAMILY]
        record = assert_width_contract(_slot(width_semantics=WIDTH_RANKING_ONLY), equal)
        assert set(record["widths"].values()) == {10}

    def test_an_experimental_family_must_rank_on_ir(self) -> None:
        with pytest.raises(WidthContractError, match="information_ratio"):
            _slot(width_semantics=WIDTH_EXPERIMENTAL, promote_statistic=STATISTIC_MEAN_DIFF)

    def test_an_experimental_family_passes_and_records_its_contract(self) -> None:
        slot = _slot(
            width_semantics=WIDTH_EXPERIMENTAL,
            promote_statistic=STATISTIC_INFORMATION_RATIO,
            promote_evidence=EVIDENCE_POINT,
        )
        record = assert_width_contract(slot, R_FAMILY)
        assert record == {
            "width_semantics": WIDTH_EXPERIMENTAL,
            "widths": {s.arm_id: s.params["top_n"] for s in R_FAMILY},
            "promote_statistic": STATISTIC_INFORMATION_RATIO,
            "promote_evidence": EVIDENCE_POINT,
        }
        # The statistic reaches the library config the engine decides with.
        assert slot.arena.promote_statistic == STATISTIC_INFORMATION_RATIO

    def test_an_unknown_semantics_is_refused(self) -> None:
        with pytest.raises(WidthContractError, match="width_semantics is one of"):
            _slot(width_semantics="sometimes")

    def test_an_equal_width_family_needs_no_declaration(self) -> None:
        equal = [_spec(s.name, 10) for s in R_FAMILY]
        assert assert_width_contract(_slot(width_semantics=None), equal)["width_semantics"] is None

    def test_controls_and_widthless_recipes_are_not_governed(self) -> None:
        specs = [_spec("a", 10), _spec("planted", 99, control=True), _spec("book", None)]
        assert selection_widths(specs) == {"r:a:0": 10}

    @pytest.mark.parametrize("slot", sorted(SLOTS))
    def test_every_declared_slot_is_a_valid_contract(self, slot: str) -> None:
        """Construction validates the declaration; the library validates the statistic."""
        spec = SLOTS[slot]
        assert spec.width_semantics in (None, WIDTH_EXPERIMENTAL, WIDTH_RANKING_ONLY)
        assert spec.arena.promote_statistic == spec.promote_statistic


def _series(arm: str, scores: list[float]) -> ArmSeries:
    dates = [f"2026-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(len(scores))]
    return ArmSeries(arm_id=arm, scores=dict(zip(dates, scores, strict=True)))


class TestTenVersusFifteen:
    """A narrow arm with the higher mean and the lower IR, against a wide one.

    10 names: mean 0.010, swinging +/-0.03 — concentration. 15 names: mean
    0.007, swinging +/-0.002 — breadth. On the mean the narrow arm leads; on
    IR the wide arm leads by an order of magnitude.
    """

    narrow = _series("r:narrow10:0", [0.04, -0.02] * 10)
    wide = _series("r:wide15:0", [0.009, 0.005] * 10)

    def _challenger_leads(self, slot) -> bool:
        # Challenger = wide, incumbent = narrow: a positive statistic means wide leads.
        window = pair_on_common_window(self.wide, self.narrow)
        value = promotion_statistic(slot.arena, window)
        assert value is not None
        return value > 0

    def test_experimental_width_selects_on_paired_ir(self) -> None:
        slot = _slot(
            width_semantics=WIDTH_EXPERIMENTAL,
            promote_statistic=STATISTIC_INFORMATION_RATIO,
            promote_evidence=EVIDENCE_POINT,
        )
        assert self._challenger_leads(slot), "IR must favour the broad, steadier arm"

    def test_the_default_mean_statistic_would_have_picked_concentration(self) -> None:
        """The CR-05 defect, kept as the reason the gate exists."""
        # R's pre-ruling declaration: no width semantics, the library default
        # statistic, the default evidence rule and the default comparison set.
        undeclared = _slot(
            width_semantics=None,
            promote_statistic=STATISTIC_MEAN_DIFF,
            promote_evidence=EVIDENCE_ANYTIME_VALID,
            promote_against=PROMOTE_AGAINST_INCUMBENT,
        )
        assert not self._challenger_leads(undeclared)


HORIZON = 21
DECISION_DATES = 6


def _grade(store, source, strategy_dir, cycle_date, tmp_path) -> None:
    from conftest import seed_data_daily, sessions_ending

    from crucible.config import Settings
    from crucible.runner import run_job
    from crucible.slots import universe

    settings = Settings(
        store_uri=str(tmp_path / "store"),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )
    sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
    decision_days = sessions[:DECISION_DATES]
    seed_data_daily(store, source, [*decision_days, cycle_date])
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


class TestRunGrade:
    def test_the_cycle_artifact_carries_the_contract(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        import json

        from crucible.keys import arena_cycle_key
        from crucible.slots.cycle import WIDTH_CONTRACT_FIELD

        _grade(store, source, strategy_dir, cycle_date, tmp_path)
        cycle = json.loads(store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())))
        contract = cycle[WIDTH_CONTRACT_FIELD]
        assert set(contract["widths"].values()) == {8}
        assert contract["promote_statistic"] == SLOTS["u"].promote_statistic
        assert contract["promote_evidence"] == SLOTS["u"].promote_evidence
        # The library's own config on the same artifact names the same statistic.
        assert cycle["config"]["promote_statistic"] == contract["promote_statistic"]

    def test_an_undeclared_variable_width_family_is_not_graded(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        recipe = strategy_dir / "arms" / "u" / "tech_score_gate.yaml"
        recipe.write_text(
            recipe.read_text(encoding="utf-8").replace("top_n: 8", "top_n: 12"),
            encoding="utf-8",
        )
        with pytest.raises(WidthContractError, match="declares no width_semantics"):
            _grade(store, source, strategy_dir, cycle_date, tmp_path)
