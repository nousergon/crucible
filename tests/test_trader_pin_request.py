"""The QUEUED trader pin: a request anyone may write, applied only by a clean
evening, only while it is still the one pending change, and only through the
same gate a human pin goes through (alpha-engine-config-I11545).

Asserted by refusals: a bad sha, an unpublished sha, a stale request, a failed
smoke, an unclean evening and a pin moved mid-apply each leave the pin where it
was, and the run that refused says why.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest
from test_trader_release_pin import (
    AFTER_CLOSE,
    ET,
    SHA_A,
    SHA_B,
    _published,
    _trader_smoke_manifest,
)

from crucible import release, track_c
from crucible.keys import TRADER_PIN_KEY, TRADER_PIN_REQUEST_KEY, manifest_key
from crucible.release import (
    PendingTraderPin,
    StaleReleasePointerError,
    TraderPinRefusedError,
    build_trader_pin_request,
    pending_trader_pin,
    pin_trader,
    read_trader_pin_request,
    write_trader_pin_request,
)
from crucible.store import LocalStore, PointerConflictError

SHA_C = "c" * 40
#: Tuesday 2026-09-08 — the session every apply below binds to.
SESSION = dt.date(2026, 9, 8)
#: 14:00 ET on the session: after the box's midday smoke slot.
ON_SESSION_AFTERNOON = dt.datetime(2026, 9, 8, 14, 0, tzinfo=ET)
#: The Friday before (Monday 2026-09-07 is Labor Day).
BEFORE_SESSION = dt.datetime(2026, 9, 4, 20, 0, tzinfo=ET)


def _request(store, sha=SHA_B, *, at=ON_SESSION_AFTERNOON, by="brian"):
    return write_trader_pin_request(
        store, sha, requested_by=by, requested_at=at, run_url="https://example.invalid/run/1"
    )


def _pinned_to(store, sha):
    """Pin the trader to ``sha`` the way the real pin does: on its passing smoke."""
    _trader_smoke_manifest(store, sha, finished_hour=20)
    pin_trader(store, sha, now=AFTER_CLOSE)


class TestTheRequestWriter:
    def test_a_short_or_uppercase_sha_is_refused_before_anything_is_read(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        for bad in ("abc123", SHA_A.upper(), SHA_A + "0"):
            with pytest.raises(ValueError, match="40-character lowercase"):
                _request(store, bad)
        assert not store.exists(TRADER_PIN_REQUEST_KEY)

    def test_an_unpublished_sha_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        with pytest.raises(StaleReleasePointerError, match="never published"):
            _request(store, SHA_B)
        assert not store.exists(TRADER_PIN_REQUEST_KEY)

    def test_a_published_record_whose_wheel_is_gone_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        record = _published(store, SHA_B)
        (tmp_path / record.wheel_key).unlink()
        with pytest.raises(StaleReleasePointerError, match="no wheel at"):
            _request(store, SHA_B)
        assert not store.exists(TRADER_PIN_REQUEST_KEY)

    def test_from_sha_is_null_when_nothing_was_ever_pinned(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store, SHA_B)
        document = _request(store)
        assert document.from_sha is None
        assert json.loads(store.get_bytes(TRADER_PIN_REQUEST_KEY))["from_sha"] is None

    def test_from_sha_is_the_live_pin_read_at_request_time(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store, SHA_A)
        _published(store, SHA_B)
        _pinned_to(store, SHA_A)
        document = _request(store, SHA_B, at=ON_SESSION_AFTERNOON)
        written = json.loads(store.get_bytes(TRADER_PIN_REQUEST_KEY))
        assert written == {
            "schema_version": "trader_pin_request.v1",
            "sha": SHA_B,
            "from_sha": SHA_A,
            "requested_by": "brian",
            "requested_at": "2026-09-08T18:00:00Z",
            "run_url": "https://example.invalid/run/1",
        }
        assert document.from_sha == SHA_A

    def test_a_naive_request_instant_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store, SHA_B)
        with pytest.raises(ValueError, match="naive"):
            build_trader_pin_request(
                store, SHA_B, requested_by="x", requested_at=dt.datetime(2026, 9, 8), run_url=None
            )

    def test_a_newer_request_overwrites_the_older_one(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store, SHA_B)
        _published(store, SHA_C)
        _request(store, SHA_B)
        _request(store, SHA_C, by="someone-else")
        document, _ = read_trader_pin_request(store)
        assert (document.sha, document.requested_by) == (SHA_C, "someone-else")


class TestTheReader:
    def test_nothing_queued_reads_as_none(self, tmp_path) -> None:
        assert read_trader_pin_request(LocalStore(tmp_path)) is None

    def test_a_document_that_does_not_conform_is_refused_not_ignored(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(TRADER_PIN_REQUEST_KEY, json.dumps({"sha": SHA_B}).encode())
        with pytest.raises(ValueError, match="does not conform to a trader pin request"):
            read_trader_pin_request(store)

    def test_an_unlistable_store_is_a_failure_not_an_absence(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        def refused(prefix=""):
            raise PermissionError("AccessDenied: s3:ListBucket")

        store.list_keys = refused  # type: ignore[method-assign]
        with pytest.raises(PermissionError):
            read_trader_pin_request(store)


class TestPending:
    def _store(self, tmp_path):
        store = LocalStore(tmp_path)
        for sha in (SHA_A, SHA_B, SHA_C):
            _published(store, sha)
        return store

    def test_none(self, tmp_path) -> None:
        pending = pending_trader_pin(self._store(tmp_path))
        assert (pending.state, pending.request) == ("none", None)

    def test_fresh_while_the_pin_still_names_from_sha(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _pinned_to(store, SHA_A)
        _request(store, SHA_B)
        pending = pending_trader_pin(store)
        assert pending.state == "fresh"
        assert pending.pin_sha == SHA_A
        assert pending.pin_version == store.etag(TRADER_PIN_KEY)

    def test_fresh_from_an_unset_pin(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        assert pending_trader_pin(store).state == "fresh"

    def test_noop_once_the_pin_names_the_requested_sha(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _pinned_to(store, SHA_A)
        _request(store, SHA_B)
        _pinned_to(store, SHA_B)
        assert pending_trader_pin(store).state == "noop"

    def test_stale_when_the_pin_moved_elsewhere_after_the_request(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _pinned_to(store, SHA_A)
        _request(store, SHA_B)
        _pinned_to(store, SHA_C)
        pending = pending_trader_pin(store)
        assert pending.state == "stale"
        assert "undo" in pending.describe()

    def test_requesting_the_current_pin_cancels_even_a_stale_request(self, tmp_path) -> None:
        """`noop` is decided before `stale`: cancelling is requesting the pin
        the trader is already on, whatever the request's `from_sha` says."""
        store = self._store(tmp_path)
        _pinned_to(store, SHA_A)
        _request(store, SHA_B)
        _pinned_to(store, SHA_C)
        _request(store, SHA_C)
        assert pending_trader_pin(store).state == "noop"


class TestApply:
    def _store(self, tmp_path):
        store = LocalStore(tmp_path)
        for sha in (SHA_A, SHA_B, SHA_C):
            _published(store, sha)
        _pinned_to(store, SHA_A)
        return store

    def _args(self, tmp_path, postclose="clean", **extra):
        return argparse.Namespace(
            postclose=postclose,
            store=str(tmp_path),
            trading_day=SESSION,
            run_mode="live",
            **extra,
        )

    def _manifest(self, tmp_path):
        return json.loads(
            LocalStore(tmp_path).get_bytes(manifest_key("release.pin_apply", SESSION.isoformat()))
        )

    def _pin(self, store):
        return json.loads(store.get_bytes(TRADER_PIN_KEY))["sha"]

    @pytest.fixture(autouse=True)
    def _after_close(self, monkeypatch):
        monkeypatch.setattr(track_c, "_now", lambda: AFTER_CLOSE)

    def test_nothing_queued_is_an_ok_run_that_moves_nothing(self, tmp_path) -> None:
        store = self._store(tmp_path)
        assert track_c.release_pin_apply_handler(self._args(tmp_path, "unclean")) == 0
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "ok"
        (metric,) = manifest["metrics"]
        assert (metric["name"], metric["value"]) == ("trader_pin_moved", 0.0)
        assert metric["status_reason"].startswith("none:")
        assert self._pin(store) == SHA_A

    def test_noop_is_ok_even_on_an_unclean_evening(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_A)
        track_c.release_pin_apply_handler(self._args(tmp_path, "unclean"))
        assert self._manifest(tmp_path)["metrics"][0]["status_reason"].startswith("noop:")

    def test_a_stale_request_fails_the_run_and_leaves_the_pin(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        _pinned_to(store, SHA_C)
        _trader_smoke_manifest(store, SHA_B, finished_hour=19)
        with pytest.raises(TraderPinRefusedError, match="stale"):
            track_c.release_pin_apply_handler(self._args(tmp_path))
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "failed" and "stale" in manifest["reason"]
        assert TRADER_PIN_REQUEST_KEY in [i["key"] for i in manifest["inputs"]]
        assert self._pin(store) == SHA_C

    def test_a_fresh_request_on_an_unclean_evening_fails_and_waits(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        _trader_smoke_manifest(store, SHA_B, finished_hour=19)
        with pytest.raises(TraderPinRefusedError, match="no\\s+clean post-close"):
            track_c.release_pin_apply_handler(self._args(tmp_path, "unclean"))
        assert self._manifest(tmp_path)["status"] == "failed"
        assert self._pin(store) == SHA_A
        assert read_trader_pin_request(store)[0].sha == SHA_B

    def test_a_failed_smoke_fails_the_run(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        _trader_smoke_manifest(store, SHA_B, status="failed", finished_hour=19)
        with pytest.raises(TraderPinRefusedError, match="FAILED"):
            track_c.release_pin_apply_handler(self._args(tmp_path))
        assert self._pin(store) == SHA_A

    def test_no_smoke_yet_waits_on_the_evening_it_was_queued(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B, at=ON_SESSION_AFTERNOON)
        assert track_c.release_pin_apply_handler(self._args(tmp_path)) == 0
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "ok"
        assert "waiting" in manifest["metrics"][0]["status_reason"]
        assert self._pin(store) == SHA_A

    def test_no_smoke_for_a_request_older_than_the_session_fails(self, tmp_path) -> None:
        """The session's midday smoke should have run it; its absence is a fault."""
        store = self._store(tmp_path)
        _request(store, SHA_B, at=BEFORE_SESSION)
        with pytest.raises(TraderPinRefusedError, match="predates session 2026-09-08"):
            track_c.release_pin_apply_handler(self._args(tmp_path))
        assert self._pin(store) == SHA_A

    def test_a_fresh_request_with_a_passing_smoke_moves_the_pin(self, tmp_path) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        key, document = _trader_smoke_manifest(store, SHA_B, finished_hour=19)
        assert track_c.release_pin_apply_handler(self._args(tmp_path)) == 0
        pinned = json.loads(store.get_bytes(TRADER_PIN_KEY))
        assert (pinned["sha"], pinned["smoke_run_id"]) == (SHA_B, document["run_id"])
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "ok"
        inputs = [i["key"] for i in manifest["inputs"]]
        assert TRADER_PIN_REQUEST_KEY in inputs and key in inputs
        assert [o["key"] for o in manifest["outputs"]] == [TRADER_PIN_KEY]
        assert manifest["metrics"][0]["value"] == 1.0
        # Converged: the next evening reads the same request as done.
        assert pending_trader_pin(store).state == "noop"

    def test_the_apply_still_honours_the_market_hours_blackout(self, tmp_path, monkeypatch) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        _trader_smoke_manifest(store, SHA_B, finished_hour=19)
        monkeypatch.setattr(track_c, "_now", lambda: dt.datetime(2026, 9, 9, 9, 45, tzinfo=ET))
        with pytest.raises(TraderPinRefusedError, match="market_hours"):
            track_c.release_pin_apply_handler(self._args(tmp_path))
        assert self._pin(store) == SHA_A

    def test_a_pin_moved_between_the_read_and_the_swap_loses_the_race(
        self, tmp_path, monkeypatch
    ) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        _trader_smoke_manifest(store, SHA_B, finished_hour=19)
        read_before = pending_trader_pin(store)
        _pinned_to(store, SHA_C)  # a human pin lands after the read

        monkeypatch.setattr(release, "pending_trader_pin", lambda _store: read_before)
        with pytest.raises(PointerConflictError):
            track_c.release_pin_apply_handler(self._args(tmp_path))
        assert self._pin(store) == SHA_C
        assert self._manifest(tmp_path)["status"] == "failed"

    def test_a_dry_run_reports_the_move_and_makes_none(self, tmp_path, capsys) -> None:
        store = self._store(tmp_path)
        _request(store, SHA_B)
        _, document = _trader_smoke_manifest(store, SHA_B, finished_hour=19)
        track_c.release_pin_apply_handler(self._args(tmp_path, dry_run=True))
        assert document["run_id"] in capsys.readouterr().out
        assert self._pin(store) == SHA_A

    def test_the_postclose_reading_is_required_on_the_command_line(self) -> None:
        from crucible.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["release.pin_apply", "--run-mode", "live"])
        with pytest.raises(SystemExit):
            build_parser().parse_args(["release.pin_apply", "--postclose", "maybe"])


class TestRequestHandler:
    def _args(self, tmp_path, sha=SHA_B):
        return argparse.Namespace(
            sha=sha, store=str(tmp_path), trading_day=SESSION, run_mode="live"
        )

    def _manifest(self, tmp_path):
        return json.loads(
            LocalStore(tmp_path).get_bytes(manifest_key("release.pin_request", SESSION.isoformat()))
        )

    def test_the_requester_and_run_come_from_the_actions_environment(
        self, tmp_path, monkeypatch
    ) -> None:
        store = LocalStore(tmp_path)
        _published(store, SHA_B)
        monkeypatch.setattr(track_c, "_now", lambda: ON_SESSION_AFTERNOON)
        monkeypatch.setenv("GITHUB_ACTOR", "octo")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        monkeypatch.setenv("GITHUB_REPOSITORY", "nousergon/crucible")
        monkeypatch.setenv("GITHUB_RUN_ID", "42")
        assert track_c.release_pin_request_handler(self._args(tmp_path)) == 0
        document, _ = read_trader_pin_request(store)
        assert document.requested_by == "octo"
        assert document.run_url == "https://github.com/nousergon/crucible/actions/runs/42"
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "ok"
        assert [o["key"] for o in manifest["outputs"]] == [TRADER_PIN_REQUEST_KEY]
        assert not store.exists(TRADER_PIN_KEY), "a request moves no pointer"

    def test_a_dry_run_builds_the_request_and_writes_nothing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        store = LocalStore(tmp_path)
        _published(store, SHA_B)
        before = sorted(store.list_keys())
        monkeypatch.setenv("GITHUB_ACTOR", "octo")
        args = self._args(tmp_path)
        args.dry_run = True
        assert track_c.release_pin_request_handler(args) == 0
        assert SHA_B in capsys.readouterr().out
        assert sorted(store.list_keys()) == before

    def test_an_anonymous_request_is_refused_with_a_manifest(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _published(store, SHA_B)
        monkeypatch.delenv("GITHUB_ACTOR", raising=False)
        monkeypatch.delenv("USER", raising=False)
        with pytest.raises(ValueError, match="no requester"):
            track_c.release_pin_request_handler(self._args(tmp_path))
        assert self._manifest(tmp_path)["status"] == "failed"
        assert not store.exists(TRADER_PIN_REQUEST_KEY)

    def test_an_unpublished_sha_fails_the_run_and_writes_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTOR", "octo")
        with pytest.raises(StaleReleasePointerError):
            track_c.release_pin_request_handler(self._args(tmp_path, SHA_C))
        assert self._manifest(tmp_path)["status"] == "failed"
        assert not LocalStore(tmp_path).exists(TRADER_PIN_REQUEST_KEY)


def test_pending_is_a_frozen_record() -> None:
    pending = PendingTraderPin("none", None, None, None, "absent")
    with pytest.raises(AttributeError):
        pending.state = "fresh"  # type: ignore[misc]
