"""The feature layer's contract: units, version derivation, and no look-ahead."""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.features import (
    CATALOG,
    UNIT_SUFFIXES,
    FeatureSpec,
    build_features,
    feature_version,
    registry_payload,
)
from crucible.features.compute import LIQUIDITY_FLOOR_USD


class TestUnits:
    def test_every_catalogue_column_carries_a_units_suffix(self) -> None:
        for spec in CATALOG:
            assert any(spec.name.endswith(s) for s in UNIT_SUFFIXES), (
                f"{spec.name} carries no units suffix. The fleet rule has no grandfather "
                "list here, because this layer has no history."
            )

    def test_a_bare_name_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="carries no units suffix"):
            FeatureSpec(
                name="avg_volume_20d",
                unit="shares",
                expression="mean(volume, 20)",
                description="the column that caused the rule",
                inputs=("volume_raw",),
            )

    def test_a_feature_with_no_lineage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="declares no inputs"):
            FeatureSpec(
                name="mystery_ratio",
                unit="ratio",
                expression="?",
                description="no lineage",
                inputs=(),
            )

    def test_a_calendar_window_cannot_be_expressed(self) -> None:
        with pytest.raises(ValueError, match="count of SESSIONS"):
            FeatureSpec(
                name="momentum_1m_log_return",
                unit="log_return",
                expression="log(close / close.shift(30))",
                description="a calendar month",
                inputs=("close_raw",),
                window_trading_days=0,
            )


class TestVersion:
    def test_the_version_is_derived_from_the_catalogue(self) -> None:
        assert feature_version(CATALOG) == registry_payload(CATALOG)["feature_version"]

    def test_changing_a_column_changes_the_version(self) -> None:
        edited = CATALOG[:-1] + (
            FeatureSpec(
                name="mom_12_1_zscore",
                unit="zscore",
                expression="zscore(mom_12_1_log_return)  # window changed",
                description="a different recipe under the same name",
                inputs=("mom_12_1_log_return",),
                cross_sectional=True,
            ),
        )
        assert feature_version(edited) != feature_version(CATALOG), (
            "a changed recipe must write to a different features/{version}/ prefix; "
            "otherwise it overwrites the layer an earlier verdict was computed from"
        )


class TestNoLookAhead:
    def test_truncating_the_panel_at_the_day_reproduces_the_same_features(
        self, source, cycle_date: dt.date
    ) -> None:
        """The property a look-ahead breaks.

        Features for day *d* computed from a panel that also contains *d+1..*
        must equal features for *d* computed from a panel truncated at *d*.
        Any window reaching forward changes when the future is removed.
        """
        full = source.load_panel(end=cycle_date, lookback_days=1200)
        earlier = sorted({d for d in full["trading_day"].unique()})[-15]

        from_full, _ = build_features(full, as_of=earlier)
        truncated = full[full["trading_day"] <= earlier]
        from_truncated, _ = build_features(truncated, as_of=earlier)

        import pandas.testing as pdt

        pdt.assert_frame_equal(from_full, from_truncated)


class TestValues:
    def test_the_liquidity_gate_is_a_column_not_a_per_arm_threshold(
        self, source, cycle_date
    ) -> None:
        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        features, _ = build_features(panel)
        liquid = features[features["liquidity_pass_raw"] == 1.0]
        assert (liquid["dollar_volume_20d_raw"] >= LIQUIDITY_FLOOR_USD).all()
        illiquid = features[features["liquidity_pass_raw"] == 0.0]
        assert (illiquid["dollar_volume_20d_raw"] < LIQUIDITY_FLOOR_USD).all()

    def test_nothing_is_forward_filled(self, source, cycle_date) -> None:
        """A short-history ticker gets a null, not a carried value."""
        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        days = sorted({d for d in panel["trading_day"].unique()})
        short = panel[panel["trading_day"] >= days[-30]]
        features, _ = build_features(short)
        assert features["mom_12_1_log_return"].isna().all(), (
            "a 252-session window over 30 sessions of history has no value; filling it "
            "would put a non-measurement into a column an arm ranks on"
        )

    def test_the_cross_section_is_one_trading_day(self, source, cycle_date) -> None:
        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        features, _ = build_features(panel)
        assert set(features["trading_day"].unique()) == {cycle_date}
