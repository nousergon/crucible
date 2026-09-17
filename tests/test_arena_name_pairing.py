"""Challengers are paired on common NAMES as well as common dates.

`alpha-engine-config-I10947`, deliverable 3 — the cost the ruling names and
requires to ship WITH it, not after it. Brian's ruling of 2026-09-17 lets a
stacked M arm score the intersection of the panel and the names its base
model had an opinion on. Its population is therefore a strict SUBSET of an
unstacked sibling's, and `crucible.slots.grading.score_selection` measures a
selection against the mean of the population it drew from: two arms
benchmarked against two different populations hand the engine two numbers it
then compares as though they were one measurement, and the arm that skipped
the hard names is favoured whenever the names it skipped are the ones that
moved.

The engine still owns the ladder, the paired WINDOW, the confidence sequence
and the pointer (`AGENTS.md`: the arena is called, never re-implemented).
What is added is the name axis of the pairing, applied where the scalar per
date is constructed — before any of it reaches a series.

Every test asserts a refusal or a distortion that is now impossible; the
first one below is the defect itself, measured on its own fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.slots.grading import (
    NAME_PAIRING_METRIC,
    NamePairing,
    pair_on_common_names,
    score_selection,
)

_DAY = "2026-08-28"

#: One date's realized forward returns. `HARD0`/`HARD1` are the two names a
#: stacked arm's base model has no opinion on — young listings, in the real
#: case — and they are the two that moved, which is exactly when a benchmark
#: difference decides a comparison.
_RETURNS = {
    "AAA": 0.02,
    "BBB": 0.01,
    "CCC": 0.00,
    "DDD": -0.01,
    "HARD0": -0.20,
    "HARD1": -0.18,
}
_WIDE = tuple(sorted(_RETURNS))
_NARROW = tuple(n for n in _WIDE if not n.startswith("HARD"))


class TestTheDistortionThisRemoves:
    def test_two_arms_with_the_same_picks_score_differently_on_their_own_populations(
        self,
    ) -> None:
        """The defect, stated as a measurement rather than as a worry.

        Same date, same picks, same realized returns — and two different
        numbers, because the benchmark each is measured against is a
        different universe. This is what reached the engine before the name
        axis existed.
        """
        selection = ("AAA", "BBB")
        wide, _ = score_selection(selection, _WIDE, _RETURNS)
        narrow, _ = score_selection(selection, _NARROW, _RETURNS)

        assert wide != pytest.approx(narrow)
        assert wide > narrow, (
            "the arm whose population still carries the two names that fell is "
            "measured against a lower benchmark and books a higher alpha for "
            "identical picks"
        )

    def test_on_the_common_names_the_same_picks_score_identically(self) -> None:
        pairing = pair_on_common_names({_DAY: {"wide": _WIDE, "narrow": _NARROW}})[_DAY]
        selection = ("AAA", "BBB")

        wide_sel, _ = pairing.restrict(selection)
        narrow_sel, _ = pairing.restrict(selection)
        wide, _ = score_selection(wide_sel, pairing.common, _RETURNS)
        narrow, _ = score_selection(narrow_sel, pairing.common, _RETURNS)

        assert wide == pytest.approx(narrow), (
            "one benchmark per date, over the names every arm on that date could "
            "have picked — the comparison is then about the picks"
        )


class TestThePairing:
    def test_the_common_set_is_the_intersection_and_the_union_is_reported(self) -> None:
        pairing = pair_on_common_names({_DAY: {"wide": _WIDE, "narrow": _NARROW}})[_DAY]

        assert pairing.common == _NARROW
        assert pairing.union_size == len(_WIDE)
        assert pairing.dropped_by_arm == {"wide": 2, "narrow": 0}
        assert pairing.coverage_ratio == pytest.approx(4 / 6)
        assert pairing.arms == ("narrow", "wide")

    def test_a_pick_outside_the_common_set_is_excluded_never_credited(self) -> None:
        pairing = pair_on_common_names({_DAY: {"wide": _WIDE, "narrow": _NARROW}})[_DAY]

        kept, dropped = pairing.restrict(("AAA", "HARD0"))

        assert kept == ("AAA",), (
            "a name at least one rival on this date could not have chosen is not "
            "scored — counting its return grades this arm on a universe the "
            "comparison does not cover"
        )
        assert dropped == 1

    def test_arms_that_ranked_the_same_names_lose_nothing(self) -> None:
        pairing = pair_on_common_names({_DAY: {"a": _WIDE, "b": _WIDE}})[_DAY]

        assert pairing.common == _WIDE
        assert pairing.coverage_ratio == 1.0
        assert set(pairing.dropped_by_arm.values()) == {0}

    def test_each_date_is_paired_on_its_own_names(self) -> None:
        """A name absent on one date does not narrow another date.

        The universe moves — a listing appears, a ticker is delisted — and a
        pairing carried across dates would grade every session against the
        worst one.
        """
        pairings = pair_on_common_names(
            {
                "2026-08-27": {"a": ("AAA", "BBB"), "b": ("AAA", "BBB")},
                _DAY: {"a": ("AAA", "BBB"), "b": ("AAA",)},
            }
        )

        assert pairings["2026-08-27"].common == ("AAA", "BBB")
        assert pairings[_DAY].common == ("AAA",)


class TestTheFigureIsOnTheManifest:
    def test_the_metric_row_satisfies_the_manifest_schema(self) -> None:
        from jsonschema import Draft202012Validator

        pairing = pair_on_common_names({_DAY: {"wide": _WIDE, "narrow": _NARROW}})[_DAY]
        row = pairing.as_metric(
            slot="m", source_path="arena/m/2026-08-28.json", now="2026-08-28T21:00:00Z"
        )
        schema = json.loads(
            Path("crucible/schemas/run_manifest.v2.json").read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema["$defs"]["MetricRecordRow"])

        assert sorted(validator.iter_errors(row), key=lambda e: list(e.path)) == []
        assert row["name"] == NAME_PAIRING_METRIC
        assert row["value"] == pytest.approx(4 / 6)
        assert row["name_pairing"]["dropped_by_arm"] == {"narrow": 0, "wide": 2}

    def test_the_row_names_the_arm_that_narrowed_the_slot(self) -> None:
        pairing = pair_on_common_names({_DAY: {"wide": _WIDE, "narrow": _NARROW}})[_DAY]
        row = pairing.as_metric(
            slot="m", source_path="arena/m/2026-08-28.json", now="2026-08-28T21:00:00Z"
        )

        assert "'narrow' lost 0" not in row["status_reason"]
        assert "the narrowest arm 'wide' lost 2" in row["status_reason"], (
            "an arm that narrows the whole slot's benchmark is named on the "
            "manifest rather than inferred from a number that quietly moved"
        )


class TestTheEmptyCases:
    def test_no_arms_on_a_date_is_an_empty_pairing_not_a_crash(self) -> None:
        pairing = pair_on_common_names({_DAY: {}})[_DAY]

        assert pairing.common == ()
        assert pairing.coverage_ratio == 0.0

    def test_two_arms_that_share_no_name_pair_on_nothing(self) -> None:
        """Not silently allowed to proceed: `score_selection` refuses a pool of
        fewer than two names as a compromised input for the whole cycle, which
        is what a slot whose arms rank disjoint universes is."""
        pairing = pair_on_common_names({_DAY: {"a": ("AAA",), "b": ("BBB",)}})[_DAY]

        assert pairing.common == ()
        with pytest.raises(Exception, match="not a benchmark"):
            score_selection(("AAA",), pairing.common, _RETURNS)


def test_the_pairing_type_is_a_value_not_a_dict() -> None:
    """Constructed directly, so a consumer can be tested without a store."""
    pairing = NamePairing(trading_day=_DAY, common=("AAA",), union_size=2, dropped_by_arm={"a": 1})
    assert pairing.coverage_ratio == 0.5
