"""The attractiveness rankers: v1's `scanner_cut` arms carried into the U slot.

`alpha-engine-config-I10715`. Pins the four properties the carry-over rests on:
each variant is a DISTINCT callable (the vacuity guard compares callables),
each reads exactly the pillars its hypothesis names, the blend is v1's
coverage-renormalised winsorised z-mean, and a degenerate or absent pillar is a
named refusal rather than a quietly re-weighted composite.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from crucible.slots.cycle import partition_by_catalog
from crucible.slots.rankers import (
    ATTRACTIVENESS_PILLAR_COLUMNS,
    MOMENTUM_12_1_PILLAR_COLUMN,
    RANKERS,
    DegenerateFeatureError,
    MissingFeatureError,
    rank_with,
    ranker_identity,
)

FAMILY = (
    "attractiveness_blend",
    "attractiveness_momzero_blend",
    "attractiveness_mom121_blend",
    "attractiveness_hard3_blend",
)
PILLARS = ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")


def _frame(
    rows: dict[str, dict[str, float]], liquid: dict[str, float] | None = None
) -> pd.DataFrame:
    tickers = list(rows)
    data: dict[str, list] = {"ticker": tickers}
    data["liquidity_pass_raw"] = [(liquid or {}).get(t, 1.0) for t in tickers]
    columns = [*ATTRACTIVENESS_PILLAR_COLUMNS.values(), MOMENTUM_12_1_PILLAR_COLUMN]
    for column in columns:
        data[column] = [rows[t].get(column, float("nan")) for t in tickers]
    return pd.DataFrame(data)


def _spread_frame(n: int = 6) -> pd.DataFrame:
    """Distinct, dispersed values in every pillar column."""
    rows = {}
    for i in range(n):
        rows[f"T{i}"] = {
            column: float(((i + k) % n) * (k + 1)) + 0.001 * i
            for k, column in enumerate(
                [*ATTRACTIVENESS_PILLAR_COLUMNS.values(), MOMENTUM_12_1_PILLAR_COLUMN]
            )
        }
    return _frame(rows)


def _weights(name: str, value: float = 1.0) -> dict[str, float]:
    return {p: value for p in RANKERS[name].params if p != "top_n"}


def _expected_blend(frame: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    block = frame[frame["liquidity_pass_raw"] == 1.0]
    zs = []
    for column in columns.values():
        v = block[column].astype(float)
        zs.append(((v - v.mean()) / v.std(ddof=0)).clip(-3, 3))
    stacked = pd.concat(zs, axis=1)
    out = stacked.mean(axis=1, skipna=True).dropna()
    out.index = block.loc[out.index, "ticker"].to_numpy()
    return out.sort_values(ascending=False)


class TestRegistration:
    def test_every_variant_is_registered_as_a_distinct_callable(self) -> None:
        assert set(FAMILY) <= set(RANKERS)
        identities = {ranker_identity(name) for name in FAMILY}
        assert len(identities) == len(FAMILY), (
            "two attractiveness variants share a callable; the vacuity guard would "
            "refuse them as not two arms"
        )

    def test_each_variant_reads_exactly_the_pillars_its_hypothesis_names(self) -> None:
        col = ATTRACTIVENESS_PILLAR_COLUMNS
        expected = {
            "attractiveness_blend": {col[p] for p in PILLARS},
            "attractiveness_momzero_blend": {col[p] for p in PILLARS if p != "momentum"},
            "attractiveness_mom121_blend": {col[p] for p in PILLARS if p != "momentum"}
            | {MOMENTUM_12_1_PILLAR_COLUMN},
            "attractiveness_hard3_blend": {col["value"], col["momentum"], col["defensiveness"]},
        }
        for name, pillars in expected.items():
            reads = set(RANKERS[name].reads)
            assert "liquidity_pass_raw" in reads
            assert reads - {"liquidity_pass_raw"} == pillars, name

    def test_every_pillar_column_carries_a_units_suffix(self) -> None:
        for column in [*ATTRACTIVENESS_PILLAR_COLUMNS.values(), MOMENTUM_12_1_PILLAR_COLUMN]:
            assert column.endswith("_pct")

    def test_an_arm_is_refused_by_name_while_the_catalogue_lacks_its_pillars(self) -> None:
        class _Spec:
            def __init__(self, name: str, ranker: str) -> None:
                self.name = name
                self.ranker = ranker

        catalogue = ["liquidity_pass_raw", "momentum_20d_zscore", "return_60d_zscore"]
        specs = [
            _Spec("momentum_sleeve", "momentum_sleeve"),
            _Spec("attractiveness", "attractiveness_blend"),
        ]
        producible, refused = partition_by_catalog(specs, catalog_columns=catalogue)
        assert [s.name for s in producible] == ["momentum_sleeve"]
        assert [r.arm for r in refused] == ["attractiveness"]
        assert set(refused[0].unresolvable) == set(ATTRACTIVENESS_PILLAR_COLUMNS.values())


class TestBlend:
    @pytest.mark.parametrize(
        ("name", "columns"),
        [
            ("attractiveness_blend", {p: ATTRACTIVENESS_PILLAR_COLUMNS[p] for p in PILLARS}),
            (
                "attractiveness_mom121_blend",
                {
                    p: (
                        MOMENTUM_12_1_PILLAR_COLUMN
                        if p == "momentum"
                        else ATTRACTIVENESS_PILLAR_COLUMNS[p]
                    )
                    for p in PILLARS
                },
            ),
            (
                "attractiveness_hard3_blend",
                {
                    p: ATTRACTIVENESS_PILLAR_COLUMNS[p]
                    for p in ("value", "momentum", "defensiveness")
                },
            ),
        ],
    )
    def test_equal_weights_reproduce_the_v1_z_mean(self, name, columns) -> None:
        frame = _spread_frame()
        got = rank_with(name, frame, {"top_n": 3, **_weights(name)})
        want = _expected_blend(frame, columns)
        assert list(got.index) == list(want.index)
        for ticker in want.index:
            assert math.isclose(got[ticker], want[ticker], rel_tol=1e-12, abs_tol=1e-12)

    def test_weights_are_scale_free(self) -> None:
        frame = _spread_frame()
        a = rank_with(
            "attractiveness_blend", frame, {"top_n": 3, **_weights("attractiveness_blend", 1.0)}
        )
        b = rank_with(
            "attractiveness_blend", frame, {"top_n": 3, **_weights("attractiveness_blend", 7.0)}
        )
        pd.testing.assert_series_equal(a, b)

    def test_momzero_does_not_read_the_momentum_pillar(self) -> None:
        frame = _spread_frame()
        before = rank_with(
            "attractiveness_momzero_blend",
            frame,
            {"top_n": 3, **_weights("attractiveness_momzero_blend")},
        )
        frame[ATTRACTIVENESS_PILLAR_COLUMNS["momentum"]] = (
            5.0  # constant: would be degenerate if read
        )
        after = rank_with(
            "attractiveness_momzero_blend",
            frame,
            {"top_n": 3, **_weights("attractiveness_momzero_blend")},
        )
        pd.testing.assert_series_equal(before, after)

    def test_illiquid_names_are_excluded_before_scoring(self) -> None:
        frame = _spread_frame()
        frame.loc[frame["ticker"] == "T5", "liquidity_pass_raw"] = 0.0
        got = rank_with(
            "attractiveness_blend", frame, {"top_n": 3, **_weights("attractiveness_blend")}
        )
        assert "T5" not in got.index
        want = _expected_blend(frame, {p: ATTRACTIVENESS_PILLAR_COLUMNS[p] for p in PILLARS})
        assert list(got.index) == list(want.index)

    def test_a_missing_pillar_value_renormalises_over_the_available_ones(self) -> None:
        frame = _spread_frame()
        col = ATTRACTIVENESS_PILLAR_COLUMNS["quality"]
        frame.loc[frame["ticker"] == "T2", col] = float("nan")
        got = rank_with(
            "attractiveness_blend", frame, {"top_n": 3, **_weights("attractiveness_blend")}
        )
        want = _expected_blend(frame, {p: ATTRACTIVENESS_PILLAR_COLUMNS[p] for p in PILLARS})
        assert math.isclose(got["T2"], want["T2"], rel_tol=1e-12)

    def test_a_name_with_no_pillar_is_dropped_not_ranked_last(self) -> None:
        frame = _spread_frame()
        for column in ATTRACTIVENESS_PILLAR_COLUMNS.values():
            frame.loc[frame["ticker"] == "T1", column] = float("nan")
        got = rank_with(
            "attractiveness_blend", frame, {"top_n": 3, **_weights("attractiveness_blend")}
        )
        assert "T1" not in got.index

    def test_pillar_z_is_winsorised_at_three(self) -> None:
        n = 40
        rows = {
            f"T{i}": {c: float(i % 7) for c in ATTRACTIVENESS_PILLAR_COLUMNS.values()}
            for i in range(n)
        }
        rows["T0"][ATTRACTIVENESS_PILLAR_COLUMNS["value"]] = 1e6
        frame = _frame(rows)
        got = rank_with(
            "attractiveness_hard3_blend",
            frame,
            {"top_n": 3, **_weights("attractiveness_hard3_blend")},
        )
        block = frame.set_index("ticker")
        z = []
        for p in ("value", "momentum", "defensiveness"):
            v = block[ATTRACTIVENESS_PILLAR_COLUMNS[p]]
            z.append(min(3.0, max(-3.0, (v["T0"] - v.mean()) / v.std(ddof=0))))
        assert z[0] == 3.0
        assert math.isclose(got["T0"], sum(z) / 3, rel_tol=1e-12)


class TestRefusals:
    def test_an_absent_pillar_column_is_a_missing_feature(self) -> None:
        frame = _spread_frame().drop(columns=[ATTRACTIVENESS_PILLAR_COLUMNS["growth"]])
        with pytest.raises(MissingFeatureError, match="growth_pillar_pct"):
            rank_with(
                "attractiveness_blend", frame, {"top_n": 3, **_weights("attractiveness_blend")}
            )

    def test_a_constant_pillar_is_degenerate_and_routes_as_a_missing_feature(self) -> None:
        frame = _spread_frame()
        frame[ATTRACTIVENESS_PILLAR_COLUMNS["value"]] = 50.0
        with pytest.raises(DegenerateFeatureError, match="value_pillar_pct") as info:
            rank_with(
                "attractiveness_hard3_blend",
                frame,
                {"top_n": 3, **_weights("attractiveness_hard3_blend")},
            )
        assert isinstance(info.value, MissingFeatureError)

    def test_an_all_null_pillar_is_degenerate(self) -> None:
        frame = _spread_frame()
        frame[ATTRACTIVENESS_PILLAR_COLUMNS["stewardship"]] = float("nan")
        with pytest.raises(DegenerateFeatureError, match="stewardship"):
            rank_with(
                "attractiveness_blend", frame, {"top_n": 3, **_weights("attractiveness_blend")}
            )

    def test_an_omitted_weight_is_refused(self) -> None:
        params = {"top_n": 3, **_weights("attractiveness_blend")}
        del params["growth_weight"]
        with pytest.raises(ValueError, match="growth_weight"):
            rank_with("attractiveness_blend", _spread_frame(), params)

    @pytest.mark.parametrize("bad", [0, 0.0, -1.0, float("nan"), float("inf"), True, "1"])
    def test_a_non_positive_or_non_numeric_weight_is_refused(self, bad) -> None:
        params = {"top_n": 3, **_weights("attractiveness_blend"), "value_weight": bad}
        with pytest.raises(ValueError, match="value_weight"):
            rank_with("attractiveness_blend", _spread_frame(), params)

    def test_a_weight_for_a_pillar_the_variant_does_not_read_is_refused(self) -> None:
        params = {"top_n": 3, **_weights("attractiveness_momzero_blend"), "momentum_weight": 1.0}
        with pytest.raises(ValueError, match="unknown parameter"):
            rank_with("attractiveness_momzero_blend", _spread_frame(), params)
