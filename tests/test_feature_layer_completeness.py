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
    COMPLETENESS_SAMPLE_SIZE,
    FEATURES_PREFIX,
    NULL_RATIO_CEILING,
    check_feature_layer_completeness,
    check_feature_layer_depth,
    sample_sessions,
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

    def test_a_dead_column_behind_a_clean_newest_session_is_caught(self, store) -> None:
        """The blindness `alpha-engine-config-I10733` walked through, as a test.

        A degenerate `data/inst_ownership/2025Q3/latest.parquet` was live for
        under two hours on 2026-09-14; every feature session healed inside that
        window took `institutional_accumulation_raw` null for 903 of 903
        tickers, ~50 sessions of them. The newest session was healed after the
        window and is fine — so a reading that looked only there was GREEN over
        the whole band. It is the older session that must decide the verdict.
        """
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

        assert reading.state == "RED"
        assert reading.session == "2026-09-10"
        assert reading.dead_columns == dead
        assert reading.dead_sessions == ("2026-09-10",)

    def test_a_clean_layer_reports_the_newest_session_and_what_it_sampled(self, store) -> None:
        for day in ("2026-09-10", "2026-09-11"):
            _put_session(store, LIVE_VERSION, day, _session_frame(tickers=("AAA",)))

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "GREEN"
        assert reading.session == "2026-09-11"
        assert reading.sessions_read == ("2026-09-10", "2026-09-11")
        assert reading.sessions_total == 2

    def test_the_reading_states_the_sample_rather_than_implying_a_sweep(self, store) -> None:
        """Measurability, one layer up: a sample reported as a full read is the
        same false green the sample exists to catch. The detail has to say how
        many sessions were read of how many exist, and how long a band can
        hide between two of them.
        """
        for day in ("2026-09-10", "2026-09-11"):
            _put_session(store, LIVE_VERSION, day, _session_frame(tickers=("AAA",)))

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert "2 of 2 session(s) sampled" in reading.detail
        assert "can sit between two samples unseen" in reading.detail

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


class TestSampleSessions:
    """The sampler's guarantee is the whole basis of the reading's claim, so it
    is asserted here rather than left to the caller's comment.
    """

    def test_a_layer_at_or_under_the_sample_size_is_read_whole(self) -> None:
        sessions = [f"2026-01-{day:02d}" for day in range(1, 11)]
        assert sample_sessions(sessions, size=40) == sessions

    def test_the_newest_and_oldest_are_always_sampled(self) -> None:
        sessions = [str(i) for i in range(1181)]
        picked = sample_sessions(sessions)
        assert picked[0] == sessions[0]
        assert picked[-1] == sessions[-1]
        assert picked == sorted(picked, key=sessions.index)

    def test_no_band_longer_than_the_stride_can_hide(self) -> None:
        """The claim the detail makes, checked against every window of that
        length rather than one example — an off-by-one in the stride would
        otherwise show up only on the band that happens to be missed.
        """
        sessions = [str(i) for i in range(1181)]
        picked = sample_sessions(sessions)
        positions = [sessions.index(candidate) for candidate in picked]
        widest_gap = max(b - a for a, b in zip(positions, positions[1:], strict=False))
        # The guarantee the reading states, checked over every window of that
        # length rather than one example — an off-by-one in the stride would
        # otherwise show up only on the band that happens to be missed.
        chosen = set(picked)
        for start in range(len(sessions) - widest_gap + 1):
            window = sessions[start : start + widest_gap]
            assert chosen.intersection(window), f"band at {start} sampled nothing"
        assert widest_gap <= len(sessions) // COMPLETENESS_SAMPLE_SIZE + 2

    def test_it_is_reproducible(self) -> None:
        """Two board runs over an unchanged layer must not disagree about it."""
        sessions = [str(i) for i in range(500)]
        assert sample_sessions(sessions) == sample_sessions(sessions)

    def test_a_non_positive_sample_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sample size must be positive"):
            sample_sessions(["2026-01-02"], size=0)
