"""ReadThroughStore: staging reads prod research history (alpha-engine-config-I13900)."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible import store as st
from crucible.store import ETAG_ABSENT, LocalStore, ReadThroughStore

Pair = tuple[LocalStore, LocalStore, ReadThroughStore]


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    primary = LocalStore(tmp_path / "staging")
    fallback = LocalStore(tmp_path / "prod")
    return primary, fallback, ReadThroughStore(primary, fallback, before="2026-11-07")


def test_history_before_the_cutoff_reads_through(pair: Pair) -> None:
    primary, fallback, store = pair
    fallback.put_bytes("data/2026-10-01/panel.parquet", b"prod")
    fallback.put_bytes("features/v1/registry.json", b"reg")
    assert store.get_bytes("data/2026-10-01/panel.parquet") == b"prod"
    assert store.exists("features/v1/registry.json")
    assert store.etag("data/2026-10-01/panel.parquet") != ETAG_ABSENT
    assert not primary.exists("data/2026-10-01/panel.parquet")


def test_staging_own_days_and_records_never_read_through(pair: Pair) -> None:
    _, fallback, store = pair
    for key in (
        "data/2026-11-07/panel.parquet",
        "runs/data.daily/2026-10-01/run.json",
        "champions/m/current.json",
        "trader/release_pin",
    ):
        fallback.put_bytes(key, b"prod")
        assert not store.exists(key)
        assert store.etag(key) == ETAG_ABSENT
        with pytest.raises(KeyError):
            store.get_bytes(key)


def test_primary_wins_and_writes_go_to_primary_only(pair: Pair) -> None:
    primary, fallback, store = pair
    fallback.put_bytes("data/2026-10-01/panel.parquet", b"prod")
    store.put_bytes("data/2026-10-01/panel.parquet", b"staging")
    assert store.get_bytes("data/2026-10-01/panel.parquet") == b"staging"
    assert fallback.get_bytes("data/2026-10-01/panel.parquet") == b"prod"
    tag = store.compare_and_swap("champions/m/current.json", ETAG_ABSENT, b"{}")
    assert tag and primary.exists("champions/m/current.json")
    assert not fallback.exists("champions/m/current.json")


def test_listing_unions_history_only(pair: Pair) -> None:
    primary, fallback, store = pair
    primary.put_bytes("data/2026-11-10/panel.parquet", b"s")
    fallback.put_bytes("data/2026-10-01/panel.parquet", b"p")
    fallback.put_bytes("data/2026-11-10/panel.parquet", b"p")
    fallback.put_bytes("data/2026-11-12/panel.parquet", b"p")
    fallback.put_bytes("runs/weekly/2026-10-01/run.json", b"p")
    assert sorted(store.list_keys("data/")) == [
        "data/2026-10-01/panel.parquet",
        "data/2026-11-10/panel.parquet",
    ]
    assert sorted(store.list_keys("data/2026-10")) == ["data/2026-10-01/panel.parquet"]
    assert "runs/weekly/2026-10-01/run.json" not in set(store.list_keys(""))
    assert list(store.list_keys("runs/")) == []


def test_settled_shadows_read_through_for_grading(pair: Pair) -> None:
    """The staging weekly grades shadows whose horizon settled weeks before it existed."""
    primary, fallback, store = pair
    old = "experiments/m-arm/2026-10-01/shadow.json"
    own = "experiments/m-arm/2026-11-10/shadow.json"
    fallback.put_bytes(old, b"prod")
    fallback.put_bytes("experiments/m-arm/2026-11-12/shadow.json", b"prod")
    primary.put_bytes(own, b"staging")
    assert store.get_bytes(old) == b"prod"
    assert sorted(store.list_keys("experiments/m-arm/")) == [old, own]
    store.put_bytes("experiments/m-arm/2026-10-01/verdict.json", b"graded")
    assert primary.exists("experiments/m-arm/2026-10-01/verdict.json")
    assert not fallback.exists("experiments/m-arm/2026-10-01/verdict.json")


def test_a_bad_cutoff_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        ReadThroughStore(LocalStore(tmp_path / "a"), LocalStore(tmp_path / "b"), before="11/07")


def test_env_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = LocalStore(tmp_path / "s")
    monkeypatch.delenv(st.READ_THROUGH_URI_ENV, raising=False)
    monkeypatch.delenv(st.READ_THROUGH_BEFORE_ENV, raising=False)
    assert st.with_read_through(base) is base
    monkeypatch.setenv(st.READ_THROUGH_URI_ENV, str(tmp_path / "p"))
    with pytest.raises(ValueError, match="or neither"):
        st.with_read_through(base)
    monkeypatch.setenv(st.READ_THROUGH_BEFORE_ENV, "2026-11-07")
    wrapped = st.with_read_through(base)
    assert isinstance(wrapped, ReadThroughStore) and wrapped.primary is base
    monkeypatch.setenv(st.READ_THROUGH_URI_ENV, "s3://bucket/crucible")
    s3 = st.with_read_through(base)
    assert isinstance(s3, ReadThroughStore)
    assert isinstance(s3.fallback, st.S3Store)


def test_presigned_url_is_primary_only(pair: Pair) -> None:
    _, fallback, store = pair
    fallback.put_bytes("data/2026-10-01/panel.parquet", b"p")
    with pytest.raises(KeyError):
        store.presigned_url("data/2026-10-01/panel.parquet", 60)


def test_primary_presence_and_write_refusal_delegate(pair: Pair) -> None:
    primary, _, store = pair
    primary.put_bytes("runs/data.daily/2026-11-10/run.json", b"{}")
    assert store.exists("runs/data.daily/2026-11-10/run.json")
    with pytest.raises(ValueError):
        store._assert_writable_key("../escape")  # pyright: ignore[reportPrivateUsage]
