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
    PRESIGN_MAX_S,
    LocalStore,
    PointerConflictError,
    S3Store,
    open_store,
    sha256_hex,
)


class TestPresignedUrl:
    """`alpha-engine-config-I9921` — the link the morning report carries.

    Every assertion here is a REFUSAL or a provenance check. A test that only
    showed the method returning a string would pass over a URL pointing at
    the wrong object, at nothing at all, or with a lifetime S3 rejects on use.
    """

    def test_an_s3_presign_names_the_prefixed_key_and_the_lifetime(self, fake_s3) -> None:
        store = S3Store("bucket", "crucible", client=fake_s3)
        store.put_bytes("board/index.html", b"<html></html>")
        url = store.presigned_url("board/index.html", 3600)
        assert "crucible/board/index.html" in url, "the store prefix must be signed, not dropped"
        assert "X-Amz-Expires=3600" in url

    def test_an_s3_presign_of_a_missing_key_raises_rather_than_linking_to_nothing(
        self, fake_s3
    ) -> None:
        """S3 signs a key that does not exist and the URL 403s on use — a
        broken link on the operator's phone under a report claiming the
        render succeeded."""
        store = S3Store("bucket", "crucible", client=fake_s3)
        with pytest.raises(KeyError):
            store.presigned_url("board/index.html", 3600)

    def test_a_local_presign_is_a_file_uri_for_the_object_on_disk(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes("board/index.html", b"<html></html>")
        url = store.presigned_url("board/index.html", 60)
        assert url.startswith("file://")
        assert url.endswith("/board/index.html")

    def test_a_local_presign_of_a_missing_key_raises(self, tmp_path) -> None:
        with pytest.raises(KeyError):
            LocalStore(tmp_path).presigned_url("board/index.html", 60)

    @pytest.mark.parametrize("expires", [0, -1, PRESIGN_MAX_S + 1])
    def test_a_lifetime_outside_the_sigv4_bound_is_refused_by_both_backends(
        self, tmp_path, fake_s3, expires
    ) -> None:
        """Refused, never clamped. S3 does not shorten an over-long
        `ExpiresIn`; it rejects the URL on use, so a clamp here would hand
        back a link that expires at a time nobody stated."""
        local = LocalStore(tmp_path)
        local.put_bytes("board/index.html", b"x")
        s3 = S3Store("bucket", "crucible", client=fake_s3)
        s3.put_bytes("board/index.html", b"x")
        for store in (local, s3):
            with pytest.raises(ValueError, match="outside 1.."):
                store.presigned_url("board/index.html", expires)

    def test_the_seven_day_maximum_is_itself_accepted(self, tmp_path) -> None:
        """The bound is inclusive — the report asks for exactly this."""
        store = LocalStore(tmp_path)
        store.put_bytes("board/index.html", b"x")
        assert store.presigned_url("board/index.html", PRESIGN_MAX_S).startswith("file://")


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
