"""The S3 backend's three load-bearing behaviours, and the local CAS.

Normative source: plan §4.11, §4.12.

Each test here corresponds to a defect the fleet has already paid for once:
a truncated listing read as complete, an unreadable object reported as
absent, and a pointer written by the last writer rather than the checking one.
"""

from __future__ import annotations

import pytest

from crucible.store import (
    ETAG_ABSENT,
    LocalStore,
    PointerConflictError,
    S3Store,
    open_store,
    sha256_hex,
)


class TestPagination:
    def test_a_listing_longer_than_one_page_is_returned_whole(self, fake_s3) -> None:
        """The `--limit N` bug class, at the store layer. Five sweeps once
        read a 2049-issue backlog under a limit and one published report was
        7 of 9 false; a store that returned the first page would reproduce
        that for every absence check."""
        store = S3Store("bucket", "crucible", client=fake_s3)
        for i in range(7):
            store.put_bytes(f"runs/data.daily/2026-08-{20 + i}/run.json", b"{}")
        assert fake_s3.page_size < 7, "the fake must actually paginate or this proves nothing"
        assert len(list(store.list_keys("runs/"))) == 7

    def test_keys_come_back_without_the_prefix(self, fake_s3) -> None:
        """A manifest records store KEYS, never URIs, so the same manifest is
        portable between backends. A listing that leaked the bucket prefix
        would make an S3-produced manifest unreadable against a local store."""
        store = S3Store("bucket", "crucible", client=fake_s3)
        store.put_bytes("runs/smoke/2026-08-28/run.json", b"{}")
        assert list(store.list_keys("runs/")) == ["runs/smoke/2026-08-28/run.json"]


class TestAbsentVersusUnreadable:
    def test_a_missing_key_is_absent(self, fake_s3) -> None:
        store = S3Store("bucket", client=fake_s3)
        assert store.exists("runs/data.daily/2026-08-28/run.json") is False

    def test_an_unreadable_key_raises_rather_than_reading_as_absent(self, fake_s3) -> None:
        """Absence is a page condition (§4.6). An AccessDenied returned as
        False is an absence page for an artifact that is there, and the
        operator spends the morning looking for a producer that ran fine."""
        from botocore.exceptions import ClientError

        store = S3Store("bucket", client=fake_s3)
        store.put_bytes("runs/data.daily/2026-08-28/run.json", b"{}")
        fake_s3.denied.add("runs/data.daily/2026-08-28/run.json")
        with pytest.raises(ClientError):
            store.exists("runs/data.daily/2026-08-28/run.json")

    def test_a_missing_get_raises_keyerror_never_returns_none(self, fake_s3) -> None:
        store = S3Store("bucket", client=fake_s3)
        with pytest.raises(KeyError):
            store.get_bytes("nope.json")


class TestConditionalWrite:
    def test_creating_a_pointer_twice_loses_the_second_time(self, fake_s3) -> None:
        store = S3Store("bucket", client=fake_s3)
        store.compare_and_swap("releases/current", ETAG_ABSENT, b'{"sha": "a"}')
        with pytest.raises(PointerConflictError):
            store.compare_and_swap("releases/current", ETAG_ABSENT, b'{"sha": "b"}')

    def test_a_swap_against_a_stale_version_loses(self, fake_s3) -> None:
        """The last-writer-wins failure, refused. Two deploys racing must not
        give the pointer to whichever finished last."""
        store = S3Store("bucket", client=fake_s3)
        store.compare_and_swap("releases/current", ETAG_ABSENT, b'{"sha": "a"}')
        stale = store.etag("releases/current")
        store.compare_and_swap("releases/current", stale, b'{"sha": "b"}')
        with pytest.raises(PointerConflictError):
            store.compare_and_swap("releases/current", stale, b'{"sha": "c"}')

    def test_etag_of_an_absent_key_is_the_absent_token(self, fake_s3) -> None:
        assert S3Store("bucket", client=fake_s3).etag("nope") == ETAG_ABSENT


class TestLocalBackendMatchesTheContract:
    """The laptop backend is not a mock: the trading-day walk, the content
    hashing and the key shapes are the same code the S3 backend runs."""

    def test_put_returns_the_content_hash(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        assert store.put_bytes("a/b.json", b"hello") == sha256_hex(b"hello")

    def test_compare_and_swap_refuses_a_stale_version(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.compare_and_swap("releases/current", ETAG_ABSENT, b"one")
        stale = ETAG_ABSENT
        with pytest.raises(PointerConflictError):
            store.compare_and_swap("releases/current", stale, b"two")

    def test_a_traversing_key_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="never traverse"):
            LocalStore(tmp_path).put_bytes("../escape.json", b"x")


class TestOpenStore:
    def test_an_s3_uri_resolves_to_the_s3_backend(self) -> None:
        store = open_store("s3://my-bucket/crucible")
        assert isinstance(store, S3Store)
        assert (store.bucket, store.prefix) == ("my-bucket", "crucible")

    def test_a_path_resolves_to_the_local_backend(self, tmp_path) -> None:
        assert isinstance(open_store(str(tmp_path)), LocalStore)

    def test_no_store_and_no_env_refuses(self, monkeypatch) -> None:
        """There is deliberately no production-bucket fallback: a job that
        wrote to production because a flag was missing is noticed once."""
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        with pytest.raises(ValueError, match="no default production bucket"):
            open_store(None)
