"""Three drift metrics with control bands. §10 component 5.

The tests that matter here are the ones asserting the module REFUSES: a
drift monitor whose failure mode is a quiet zero is the blind spot it exists
to close.
"""

from __future__ import annotations

import datetime as dt
import random

import pytest

from crucible.drift import (
    BANDS,
    Band,
    drift_metrics,
    feature_psi,
    ic_by_horizon,
    ic_decay,
    psi,
)

NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.UTC)
FRIDAY = dt.date(2026, 8, 28)


def _sample(n: int, mu: float, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


class TestPSI:
    def test_the_same_distribution_scores_near_zero(self) -> None:
        assert psi(_sample(2000, 0, 1, 1), _sample(2000, 0, 1, 2)) < 0.10

    def test_a_shifted_distribution_breaches(self) -> None:
        assert psi(_sample(2000, 0, 1, 1), _sample(2000, 1.5, 1, 2)) > 0.25

    def test_an_empty_current_sample_raises_rather_than_scoring_zero(self) -> None:
        """A PSI of zero for a feature that produced nothing is a monitor
        reporting health from the absence of data — the reason this module
        exists."""
        with pytest.raises(ValueError, match="lineage failure"):
            psi([1.0, 2.0, 3.0], [])

    def test_an_empty_reference_raises(self) -> None:
        with pytest.raises(ValueError, match="measures nothing"):
            psi([], [1.0])

    def test_an_emptied_bin_does_not_send_psi_to_infinity(self) -> None:
        """Without the epsilon floor one rare value that stopped appearing
        renders as a catastrophe every cycle."""
        value = psi([0.0] * 50 + [9.0] * 50, [0.0] * 100)
        assert value == pytest.approx(value)  # finite
        assert value < 100


class TestFeaturePSI:
    def test_a_feature_that_vanished_raises_rather_than_being_skipped(self) -> None:
        """Skipping it would let a feature disappear from the pipeline
        entirely while the drift row stayed green."""
        with pytest.raises(KeyError, match="largest drift there is"):
            feature_psi({"momentum_zscore": [1.0, 2.0]}, {})

    def test_it_scores_every_feature(self) -> None:
        out = feature_psi(
            {"a_ratio": _sample(500, 0, 1, 1), "b_ratio": _sample(500, 0, 1, 3)},
            {"a_ratio": _sample(500, 0, 1, 2), "b_ratio": _sample(500, 2, 1, 4)},
        )
        assert set(out) == {"a_ratio", "b_ratio"}
        assert out["b_ratio"] > out["a_ratio"]


class TestIC:
    def test_a_perfectly_ranked_prediction_scores_one(self) -> None:
        assert ic_by_horizon([1.0, 2.0, 3.0], {21: [10.0, 20.0, 30.0]})[21] == pytest.approx(1.0)

    def test_a_mismatched_pairing_raises(self) -> None:
        """A truncated pairing silently drops the tail of the cross-section
        and biases the IC."""
        with pytest.raises(ValueError, match="truncated pairing"):
            ic_by_horizon([1.0, 2.0, 3.0], {21: [1.0, 2.0]})

    def test_a_constant_prediction_vector_is_a_producer_defect_not_a_zero_ic(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            ic_by_horizon([1.0, 1.0, 1.0], {21: [1.0, 2.0, 3.0]})

    def test_a_calendar_horizon_is_not_expressible(self) -> None:
        with pytest.raises(ValueError, match="positive trading-day count"):
            ic_by_horizon([1.0, 2.0], {0: [1.0, 2.0]})


class TestICDecay:
    def test_a_halved_ic_reads_as_half_the_edge_lost(self) -> None:
        assert ic_decay({21: 0.02}, {21: 0.04})[21] == pytest.approx(0.5)

    def test_an_improvement_is_not_negative_decay(self) -> None:
        assert ic_decay({21: 0.08}, {21: 0.04})[21] == 0.0

    def test_a_sign_flip_reads_worse_than_total_loss(self) -> None:
        """The signal is not weaker, it is backwards."""
        assert ic_decay({21: -0.04}, {21: 0.04})[21] > 1.0

    def test_a_horizon_that_stopped_being_measured_raises(self) -> None:
        with pytest.raises(KeyError, match="renders green if it is skipped"):
            ic_decay({}, {21: 0.04})

    def test_a_baseline_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a baseline"):
            ic_decay({21: 0.01}, {21: 0.0})


class TestBands:
    def test_no_value_renders_unreported_never_ok(self) -> None:
        """A component emitting nothing is unobserved, not healthy."""
        assert Band(0.1, 0.25, "x").status(None) == "UNREPORTED"

    @pytest.mark.parametrize("name", sorted(BANDS))
    def test_every_band_carries_its_rationale(self, name: str) -> None:
        assert len(BANDS[name].rationale) > 40
        assert BANDS[name].watch < BANDS[name].breach


class TestRecords:
    def test_there_are_exactly_three_rows_per_cycle(self) -> None:
        """A drift module that grows a row per feature is the 185-rule fleet
        again with a different noun."""
        records = drift_metrics(
            trading_day=FRIDAY,
            feature_psi_by_name={"a_ratio": 0.02, "b_ratio": 0.31},
            prediction_psi=0.05,
            ic_decay_by_horizon={21: 0.1, 63: 0.6},
            now=NOW,
        )
        assert [r["name"] for r in records] == [
            "feature_psi_max_ratio",
            "prediction_psi_ratio",
            "ic_decay_ratio",
        ]

    def test_the_worst_offender_is_named_in_the_row(self) -> None:
        records = drift_metrics(
            trading_day=FRIDAY,
            feature_psi_by_name={"a_ratio": 0.02, "b_ratio": 0.31},
            prediction_psi=0.05,
            ic_decay_by_horizon={21: 0.1},
            now=NOW,
        )
        assert "b_ratio" in records[0]["status_reason"]
        assert records[0]["status"] == "BREACH"

    def test_every_value_carries_a_unit(self) -> None:
        """The units-suffix contract: `avg_volume_20d` was emitted as a
        normalized ratio and consumed as raw shares for months."""
        for record in drift_metrics(
            trading_day=FRIDAY,
            feature_psi_by_name={"a_ratio": 0.02},
            prediction_psi=0.05,
            ic_decay_by_horizon={252: 0.1},
            now=NOW,
        ):
            assert record["unit"]
            assert record["name"].endswith("_ratio")

    def test_the_ic_row_carries_its_horizon_in_trading_days(self) -> None:
        records = drift_metrics(
            trading_day=FRIDAY,
            feature_psi_by_name={"a_ratio": 0.02},
            prediction_psi=0.05,
            ic_decay_by_horizon={21: 0.1, 252: 0.9},
            now=NOW,
        )
        assert records[2]["horizon_trading_days"] == 252

    def test_no_features_compared_renders_unreported_not_ok(self) -> None:
        records = drift_metrics(
            trading_day=FRIDAY,
            feature_psi_by_name={},
            prediction_psi=0.05,
            ic_decay_by_horizon={},
            now=NOW,
        )
        assert records[0]["status"] == "UNREPORTED"
        assert records[2]["status"] == "UNREPORTED"
