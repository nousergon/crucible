"""The feature layer completeness detector — `alpha-engine-config-I10693`.

Normative source: `alpha-engine-config-I10693`, parent `-I10688`.

Measured 2026-09-13: `check_feature_layer_depth` read GREEN over
`features/v6df3c0a27b70` while `residual_momentum_252d_skip21d_ratio` and its
z-score were null on every row of every session for the whole history of that
version — depth counts OBJECTS, never CONTENTS, so a column that looks
computed and measured nothing is invisible to it. `crucible-PR266` fixed the
producer (`min_panel_trading_days`, `PanelDepthError`); this module is the
detector that would have caught every session compiled BEFORE that guard
landed, and any future column in the same shape.

The mutation test (`TestMutationAgainstTheOldDetector`) is the load-bearing
one: it proves the pre-existing `check_feature_layer_depth` reads GREEN over
the exact input this module reads RED over, so a regression back to
depth-only grading is caught here even if every other test in this file were
deleted.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from crucible.board import build_board
from crucible.features.depth import (
    FEATURES_PREFIX,
    NULL_RATIO_CEILING,
    check_feature_layer_completeness,
    check_feature_layer_depth,
)
from crucible.features.registry import feature_names, feature_version
from crucible.store import LocalStore

LIVE_VERSION = "v6df3c0a27b70"


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _session_frame(*, tickers: tuple[str, ...], null_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """A session cross-section shaped like `build_features`'s output: one row
    per ticker, every catalogue column present. Columns named in
    ``null_columns`` are all-NaN on every row — the measured `-I10688` shape.
    """
    columns = feature_names()
    data: dict[str, list[object]] = {
        "trading_day": ["2026-09-11"] * len(tickers),
        "ticker": list(tickers),
    }
    for i, col in enumerate(columns):
        if col in null_columns:
            data[col] = [None] * len(tickers)
        else:
            data[col] = [float(i + 1)] * len(tickers)
    return pd.DataFrame(data)


def _put_session(store: LocalStore, version: str, day: str, frame: pd.DataFrame) -> None:
    buf = io.BytesIO()
    frame.to_parquet(buf)
    store.put_bytes(f"{FEATURES_PREFIX}{version}/{day}.parquet", buf.getvalue())


class TestCheckFeatureLayerCompleteness:
    def test_red_when_a_catalogue_column_is_null_on_every_row(self, store) -> None:
        """The exact `-I10688` condition: `residual_momentum_252d_skip21d_ratio`
        and its z-score null for every ticker, on the only session present.
        """
        dead = (
            "residual_momentum_252d_skip21d_ratio",
            "residual_momentum_252d_skip21d_zscore",
        )
        _put_session(
            store,
            LIVE_VERSION,
            "2026-09-11",
            _session_frame(tickers=("AAA", "BBB", "CCC"), null_columns=dead),
        )

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "RED"
        assert set(dead) <= set(reading.dead_columns)
        for col in dead:
            assert reading.null_ratios[col] == 1.0
            assert col in reading.detail

    def test_green_when_no_column_is_fully_null(self, store) -> None:
        _put_session(
            store,
            LIVE_VERSION,
            "2026-09-11",
            _session_frame(tickers=("AAA", "BBB", "CCC")),
        )

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "GREEN"
        assert reading.dead_columns == ()

    def test_a_column_null_for_one_ticker_of_several_is_not_red(self, store) -> None:
        """A single young ticker legitimately null under a deep column
        (`catalog_column_depths()`'s head-of-history case) is a PARTIAL null
        ratio, reported but not the RED condition — only an all-ticker null
        is.
        """
        frame = _session_frame(tickers=("AAA", "BBB", "CCC"))
        col = "residual_momentum_252d_skip21d_ratio"
        frame.loc[frame["ticker"] == "AAA", col] = None

        _put_session(store, LIVE_VERSION, "2026-09-11", frame)

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "GREEN"
        assert col not in reading.dead_columns
        assert reading.null_ratios[col] == pytest.approx(1.0 / 3.0)

    def test_reports_the_null_ratio_of_every_catalogue_column(self, store) -> None:
        """Item 3 of the deliverable: a partial degradation is visible on the
        reading even when it never reaches the RED ceiling.
        """
        frame = _session_frame(tickers=("AAA", "BBB", "CCC", "DDD"))
        col = "beta_60d_raw"
        frame.loc[frame["ticker"] == "AAA", col] = None

        _put_session(store, LIVE_VERSION, "2026-09-11", frame)

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.null_ratios[col] == pytest.approx(0.25)
        assert reading.state == "GREEN"

    def test_reads_the_most_recent_session_of_the_live_version(self, store) -> None:
        dead = ("beta_60d_raw",)
        _put_session(
            store,
            LIVE_VERSION,
            "2026-09-10",
            _session_frame(tickers=("AAA",), null_columns=dead),
        )
        _put_session(
            store,
            LIVE_VERSION,
            "2026-09-11",
            _session_frame(tickers=("AAA",)),
        )

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.session == "2026-09-11"
        assert reading.state == "GREEN"

    def test_red_when_the_live_version_has_no_session_at_all(self, store) -> None:
        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "RED"
        assert reading.session is None

    def test_defaults_to_the_code_s_own_feature_version_when_none_is_passed(self, store) -> None:
        live = feature_version()
        _put_session(store, live, "2026-09-11", _session_frame(tickers=("AAA",)))

        reading = check_feature_layer_completeness(store)

        assert reading.live_version == live
        assert reading.state == "GREEN"

    def test_null_ratio_ceiling_is_named_and_half(self) -> None:
        """The declared ceiling this module grades against: a column at or
        above this ratio is the RED condition, for every catalogue column
        alike — never a per-column allowlist (`crucible/AGENTS.md` rule 4).
        Half: the measured head-of-history null rate after the producer fix
        is 0.44% of tickers, and no per-ticker depth explains half the
        universe null at once.
        """
        assert NULL_RATIO_CEILING == 0.5

    def test_red_when_a_column_is_null_for_half_the_universe(self, store) -> None:
        """Mutation guard on the ceiling: a column dead for half the tickers
        is the same blindness as one dead for all of them — the old
        all-rows-only condition read this GREEN."""
        frame = _session_frame(tickers=("AAA", "BBB", "CCC", "DDD"))
        col = "residual_momentum_252d_skip21d_ratio"
        frame.loc[frame["ticker"].isin(["AAA", "BBB"]), col] = None

        _put_session(store, LIVE_VERSION, "2026-09-11", frame)

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "RED"
        assert col in reading.dead_columns
        assert reading.null_ratios[col] == pytest.approx(0.5)


class TestMutationAgainstTheOldDetector:
    """Proves the class of bug, not just the instance: the PRE-EXISTING
    `check_feature_layer_depth` reads GREEN over the same store this module
    reads RED over — the measured `-I10688`/`-I10693` defect, reproduced as a
    mutation test so a future change that reverts to depth-only grading is
    caught here.
    """

    def test_old_depth_detector_reads_green_over_an_all_null_column(self, store) -> None:
        dead = (
            "residual_momentum_252d_skip21d_ratio",
            "residual_momentum_252d_skip21d_zscore",
        )
        _put_session(
            store,
            LIVE_VERSION,
            "2026-09-11",
            _session_frame(tickers=("AAA", "BBB", "CCC"), null_columns=dead),
        )

        depth_reading = check_feature_layer_depth(store, live_version=LIVE_VERSION)
        completeness_reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert depth_reading.state == "GREEN", (
            "the old detector must still read GREEN here — that is the measured "
            "detection blindness this issue exists to close, not a bug in the test"
        )
        assert completeness_reading.state == "RED"


class TestTheBoardRow:
    def test_the_board_carries_exactly_one_feature_layer_completeness_row(self, store) -> None:
        board = build_board(store)
        rows = [r for r in board.rows if r.id == "component:feature_layer_completeness"]
        assert len(rows) == 1

    def test_the_row_is_red_when_a_column_is_fully_null(self, store) -> None:
        dead = ("residual_momentum_252d_skip21d_ratio",)
        _put_session(
            store,
            LIVE_VERSION,
            "2026-09-11",
            _session_frame(tickers=("AAA", "BBB"), null_columns=dead),
        )

        board = build_board(store)
        row = next(r for r in board.rows if r.id == "component:feature_layer_completeness")

        assert row.red
        assert row.state == "UNMET"

    def test_the_row_never_raises_when_the_store_is_unreadable(self, store, monkeypatch) -> None:
        def _boom(self, prefix=""):
            raise RuntimeError("simulated listing failure")

        monkeypatch.setattr(type(store), "list_keys", _boom)

        board = build_board(store)
        row = next(r for r in board.rows if r.id == "component:feature_layer_completeness")

        assert row.state == "UNMEASURABLE"
        assert row.red
        assert "simulated listing failure" in row.detail
