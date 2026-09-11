"""The feature layer depth detector — `alpha-engine-config-I10498` deliverable 2.

Red-first, against local store fixtures shaped like the measured production
state (2026-09-11): `feature_version()` -> `v6df3c0a27b70` holding 8 objects
(7 sessions + `registry.json`), while `vf795db2b5049` holds 533 objects (532
sessions + `registry.json`) one prefix over. No test here hits S3 — every
fixture is a `LocalStore` over `tmp_path`, per this repo's own test
convention (`tests/test_board.py::store`).
"""

from __future__ import annotations

import pytest

from crucible.board import build_board
from crucible.features.depth import (
    FEATURES_PREFIX,
    check_feature_layer_depth,
    count_session_objects_by_version,
)
from crucible.store import LocalStore


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _put_sessions(store: LocalStore, version: str, days: list[str], *, with_registry=True) -> None:
    for day in days:
        store.put_bytes(f"{FEATURES_PREFIX}{version}/{day}.parquet", b"parquet-bytes")
    if with_registry:
        store.put_bytes(f"{FEATURES_PREFIX}{version}/registry.json", b"{}")


LIVE_VERSION = "v6df3c0a27b70"
DEEP_VERSION = "vf795db2b5049"


class TestCountSessionObjectsByVersion:
    def test_counts_parquet_only_never_registry_json(self, store) -> None:
        _put_sessions(store, LIVE_VERSION, ["2026-07-31"])
        counts = count_session_objects_by_version(store)
        assert counts == {LIVE_VERSION: 1}, (
            "registry.json sits at the same prefix depth as the session parquet files "
            "and must never be counted as a session"
        )

    def test_a_version_with_no_objects_at_all_is_absent_from_the_dict(self, store) -> None:
        _put_sessions(store, LIVE_VERSION, ["2026-07-31"])
        counts = count_session_objects_by_version(store)
        assert "vnonexistent000" not in counts


class TestCheckFeatureLayerDepth:
    def test_red_when_the_live_version_is_shallower_than_the_deepest(self, store) -> None:
        """The exact `-I10498` condition: the live version's prefix (8
        objects / 7 sessions) is shallower than another version's (533
        objects / 532 sessions) in the same store.
        """
        seven_sessions = [f"2026-07-{d:02d}" for d in range(24, 31)]
        five_thirty_two_sessions = [f"session-{i:04d}" for i in range(532)]
        _put_sessions(store, LIVE_VERSION, seven_sessions)
        _put_sessions(store, DEEP_VERSION, five_thirty_two_sessions)

        reading = check_feature_layer_depth(store, live_version=LIVE_VERSION)

        assert reading.state == "RED"
        assert reading.live_count == 7
        assert reading.deepest_version == DEEP_VERSION
        assert reading.deepest_count == 532
        assert LIVE_VERSION in reading.detail
        assert DEEP_VERSION in reading.detail

    def test_green_when_the_live_version_is_the_deepest(self, store) -> None:
        seven_sessions = [f"2026-07-{d:02d}" for d in range(24, 31)]
        five_thirty_two_sessions = [f"session-{i:04d}" for i in range(532)]
        _put_sessions(store, LIVE_VERSION, five_thirty_two_sessions)
        _put_sessions(store, DEEP_VERSION, seven_sessions)

        reading = check_feature_layer_depth(store, live_version=LIVE_VERSION)

        assert reading.state == "GREEN"
        assert reading.live_count == 532
        assert reading.deepest_version == LIVE_VERSION
        assert reading.deepest_count == 532

    def test_green_when_the_live_version_is_the_only_version_present(self, store) -> None:
        _put_sessions(store, LIVE_VERSION, ["2026-07-31"])

        reading = check_feature_layer_depth(store, live_version=LIVE_VERSION)

        assert reading.state == "GREEN"
        assert reading.deepest_version == LIVE_VERSION

    def test_red_when_the_live_prefix_is_entirely_absent(self, store) -> None:
        """A store that has never seen the live version at all — some OTHER
        version is present, but not the one the running code resolves to.
        """
        _put_sessions(store, DEEP_VERSION, [f"session-{i:04d}" for i in range(532)])

        reading = check_feature_layer_depth(store, live_version=LIVE_VERSION)

        assert reading.state == "RED"
        assert reading.live_count == 0
        assert reading.deepest_version == DEEP_VERSION
        assert LIVE_VERSION in reading.detail

    def test_red_when_the_store_has_no_feature_layer_prefix_at_all(self, store) -> None:
        """Absence of every version, not just the live one — the store has
        never had a feature layer built into it.
        """
        reading = check_feature_layer_depth(store, live_version=LIVE_VERSION)

        assert reading.state == "RED"
        assert reading.deepest_version is None
        assert reading.deepest_count == 0

    def test_defaults_to_the_code_s_own_feature_version_when_none_is_passed(self, store) -> None:
        from crucible.features.registry import feature_version

        live = feature_version()
        _put_sessions(store, live, ["2026-07-31"])

        reading = check_feature_layer_depth(store)

        assert reading.live_version == live
        assert reading.state == "GREEN"


class TestTheBoardRow:
    """`alpha-engine-config-I10498` deliverable 2's second half: one row on
    `crucible board`, never a crash of the render.
    """

    def test_the_board_carries_exactly_one_feature_layer_depth_row(self, store) -> None:
        board = build_board(store)
        rows = [r for r in board.rows if r.id == "component:feature_layer_depth"]
        assert len(rows) == 1

    def test_the_row_is_red_when_the_layer_is_shallow(self, store) -> None:
        seven_sessions = [f"2026-07-{d:02d}" for d in range(24, 31)]
        _put_sessions(store, LIVE_VERSION, seven_sessions)
        _put_sessions(store, DEEP_VERSION, [f"session-{i:04d}" for i in range(532)])

        board = build_board(store)
        row = next(r for r in board.rows if r.id == "component:feature_layer_depth")

        assert row.red
        assert row.state == "UNMET"

    def test_the_row_never_raises_when_the_store_is_unreadable(self, store, monkeypatch) -> None:
        """A listing failure is a statement about OUR access — the render
        must still complete, with the row UNMEASURABLE, never a raise out of
        `build_board`.
        """

        def _boom(self, prefix=""):
            raise RuntimeError("simulated listing failure")

        monkeypatch.setattr(type(store), "list_keys", _boom)

        board = build_board(store)
        row = next(r for r in board.rows if r.id == "component:feature_layer_depth")

        assert row.state == "UNMEASURABLE"
        assert row.red
        assert "simulated listing failure" in row.detail
