"""The trader's pin moves only off-market-hours, and only on its own passing
paper smoke for that exact sha — asserted by refusals, not described.

Normative source: `alpha-engine-config-I10649`; plan §4.11 (Trader row).

`trader.smoke` is a registered job (`crucible.models.JOB_VALUES`,
alpha-engine-config-I10651), so every success-path test below reads the real
manifest validator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from crucible import track_c
from crucible.keys import TRADER_PIN_KEY, manifest_key
from crucible.release import (
    TRADER_SMOKE_JOB,
    TraderPinRefusedError,
    TraderSmokeEvidence,
    passing_trader_smoke,
    pin,
    pin_trader,
    publish_release,
    select_trader_smoke,
    trader_pin_window_refusal,
)
from crucible.runner import run_job
from crucible.store import LocalStore
from tests.support.manifests import only_manifest

SHA_A = "a" * 40
SHA_B = "b" * 40
ET = dt.timezone(dt.timedelta(hours=-4))  # EDT, fixed so the literals below read as ET

#: Tuesday 2026-09-08, 10:00 ET — a regular NYSE session.
IN_SESSION = dt.datetime(2026, 9, 8, 10, 0, tzinfo=ET)
#: The same trading day, after the blackout ends.
AFTER_CLOSE = dt.datetime(2026, 9, 8, 17, 0, tzinfo=ET)


def _published(store, sha=SHA_A):
    return publish_release(
        store,
        sha=sha,
        wheel=b"PK\x03\x04 wheel bytes",
        lockfile=b"# uv.lock",
        test_summary="42 passed",
        workflow_run_url="https://github.com/nousergon/crucible/actions/runs/1",
        now=dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC),
    )


def _trader_smoke_manifest(store, sha, *, status="ok", day="2026-09-04", finished_hour=21):
    """A REAL runner-written manifest, re-filed as the trader's smoke.

    Written by `run_job` under an admitted job so every other field is exactly
    what the runner produces; only `job` (and the key) are the trader's. That
    is the one field the contract dependency above is about.
    """

    def body(ctx):
        if status != "ok":
            raise RuntimeError(
                "broker_session_unavailable: IB Gateway paper refused the connection"
            )

    now = dt.datetime(2026, 9, 4, finished_hour, 0, tzinfo=dt.UTC)
    try:
        ctx = run_job(
            "smoke",
            body,
            store=store,
            trading_day=dt.date.fromisoformat(day),
            now=now,
            release_sha=sha,
            run_mode="live",
        )
    except RuntimeError:
        ctx = None
    source = manifest_key("smoke", day)
    document = json.loads(store.get_bytes(source))
    assert ctx is None or document["run_id"] == ctx.run_id
    document["job"] = TRADER_SMOKE_JOB
    document["finished"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    key = manifest_key(TRADER_SMOKE_JOB, day, discriminator=f"{sha[:12]}-{finished_hour}")
    store.put_bytes(key, json.dumps(document).encode())
    return key, document


class TestWindow:
    def test_inside_the_session_is_refused_with_a_named_reason(self) -> None:
        reason = trader_pin_window_refusal(IN_SESSION)
        assert reason is not None and reason.startswith("market_hours:")

    @pytest.mark.parametrize(
        ("instant", "why"),
        [
            (dt.datetime(2026, 9, 8, 4, 59, tzinfo=ET), "before the blackout opens"),
            (dt.datetime(2026, 9, 8, 16, 30, tzinfo=ET), "the end is exclusive"),
            (dt.datetime(2026, 9, 12, 10, 0, tzinfo=ET), "a Saturday"),
            (dt.datetime(2026, 7, 3, 10, 0, tzinfo=ET), "observed Independence Day"),
        ],
    )
    def test_off_market_hours_is_allowed(self, instant, why) -> None:
        assert trader_pin_window_refusal(instant) is None, why

    def test_the_blackout_opens_at_five(self) -> None:
        assert trader_pin_window_refusal(dt.datetime(2026, 9, 8, 5, 0, tzinfo=ET)) is not None

    def test_a_utc_instant_is_placed_in_eastern_time(self) -> None:
        """14:00Z is 10:00 EDT: refused, although 14:00 'looks' like afternoon."""
        assert trader_pin_window_refusal(dt.datetime(2026, 9, 8, 14, 0, tzinfo=dt.UTC))

    def test_a_naive_instant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            trader_pin_window_refusal(dt.datetime(2026, 9, 8, 10, 0))


class TestSelection:
    def _doc(self, sha, status="ok", finished="2026-09-04T21:00:00Z", job=TRADER_SMOKE_JOB):
        return {
            "job": job,
            "release_sha": sha,
            "status": status,
            "reason": "" if status == "ok" else "boom",
            "run_id": f"run-{finished}",
            "trading_day": "2026-09-04",
            "finished": finished,
        }

    def test_another_shas_pass_is_not_evidence(self) -> None:
        evidence, failures = select_trader_smoke([("k", self._doc(SHA_B))], SHA_A)
        assert evidence is None and failures == []

    def test_the_harness_smoke_is_not_the_trader_smoke(self) -> None:
        evidence, _ = select_trader_smoke([("k", self._doc(SHA_A, job="smoke"))], SHA_A)
        assert evidence is None

    def test_a_failed_smoke_is_a_reason_not_evidence(self) -> None:
        evidence, failures = select_trader_smoke([("k", self._doc(SHA_A, "failed"))], SHA_A)
        assert evidence is None and failures == ["k (run-2026-09-04T21:00:00Z): failed — boom"]

    def test_the_newest_pass_is_chosen(self) -> None:
        evidence, _ = select_trader_smoke(
            [
                ("new", self._doc(SHA_A, finished="2026-09-05T21:00:00Z")),
                ("old", self._doc(SHA_A, finished="2026-09-04T21:00:00Z")),
            ],
            SHA_A,
        )
        assert evidence is not None and evidence.manifest_key == "new"


class TestPinPrimitiveRefusesTheTraderWithoutEvidence:
    def _evidence(self, sha=SHA_A, status="ok"):
        return TraderSmokeEvidence("k", "r", sha, status, "2026-09-04", "2026-09-04T21:00:00Z")

    def test_no_evidence_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        with pytest.raises(TraderPinRefusedError, match="no trader smoke evidence"):
            pin(store, SHA_A, target="trader")
        assert not store.exists(TRADER_PIN_KEY)

    @pytest.mark.parametrize(("sha", "status"), [(SHA_B, "ok"), (SHA_A, "failed")])
    def test_evidence_for_another_build_or_a_failure_is_refused(
        self, tmp_path, sha, status
    ) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        with pytest.raises(TraderPinRefusedError, match="failing open"):
            pin(store, SHA_A, target="trader", trader_smoke=self._evidence(sha, status))
        assert not store.exists(TRADER_PIN_KEY)

    def test_trader_evidence_cannot_gate_the_harness_pointer(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        with pytest.raises(ValueError, match="never by the trader's"):
            pin(store, SHA_A, target="current", trader_smoke=self._evidence())


class TestPassingTraderSmokeFromTheStore:
    def test_no_smoke_ever_is_refused_by_name(self, tmp_path) -> None:
        with pytest.raises(TraderPinRefusedError, match="no_passing_smoke.*has ever run"):
            passing_trader_smoke(LocalStore(tmp_path), SHA_A)

    def test_an_unreadable_smoke_manifest_is_a_refusal(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(manifest_key(TRADER_SMOKE_JOB, "2026-09-04"), b"{not json")
        with pytest.raises(TraderPinRefusedError, match="could not be read"):
            passing_trader_smoke(store, SHA_A)

    def test_an_unlistable_prefix_is_a_refusal_not_an_absence(self, tmp_path) -> None:
        class Denied(LocalStore):
            def list_keys(self, prefix):
                raise PermissionError("AccessDenied")

        with pytest.raises(TraderPinRefusedError, match="could not be read: PermissionError"):
            passing_trader_smoke(Denied(tmp_path), SHA_A)

    def test_a_failed_smoke_names_itself_in_the_refusal(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _trader_smoke_manifest(store, SHA_A, status="failed")
        with pytest.raises(TraderPinRefusedError, match="broker_session_unavailable"):
            passing_trader_smoke(store, SHA_A)

    def test_a_passing_smoke_for_this_sha_is_evidence(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        key, document = _trader_smoke_manifest(store, SHA_A)
        evidence = passing_trader_smoke(store, SHA_A)
        assert (evidence.manifest_key, evidence.run_id) == (key, document["run_id"])


class TestPinTrader:
    def test_in_session_is_refused_before_the_evidence_is_even_read(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        with pytest.raises(TraderPinRefusedError, match="market_hours"):
            pin_trader(store, SHA_A, now=IN_SESSION)
        assert not store.exists(TRADER_PIN_KEY)

    def test_without_a_passing_smoke_it_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        _trader_smoke_manifest(store, SHA_B)
        with pytest.raises(TraderPinRefusedError, match="no_passing_smoke"):
            pin_trader(store, SHA_A, now=AFTER_CLOSE)
        assert not store.exists(TRADER_PIN_KEY)

    def test_with_a_passing_smoke_it_moves_and_a_rollback_reuses_the_record(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        _published(store, SHA_B)
        _trader_smoke_manifest(store, SHA_A, finished_hour=20)
        _trader_smoke_manifest(store, SHA_B, finished_hour=21)
        pin_trader(store, SHA_B, now=AFTER_CLOSE)
        assert json.loads(store.get_bytes(TRADER_PIN_KEY))["sha"] == SHA_B
        # Rollback: SHA_A's smoke passed earlier; no new smoke is required.
        _, evidence = pin_trader(store, SHA_A, now=AFTER_CLOSE)
        assert json.loads(store.get_bytes(TRADER_PIN_KEY))["sha"] == SHA_A
        assert evidence.release_sha == SHA_A


class TestHandler:
    def _args(self, tmp_path, sha=SHA_A, **extra):
        return argparse.Namespace(
            sha=sha,
            target="trader",
            store=str(tmp_path),
            trading_day=dt.date(2026, 9, 8),
            run_mode="live",
            **extra,
        )

    def _manifest(self, tmp_path):
        return only_manifest(LocalStore(tmp_path), "release.pin", "2026-09-08")[1]

    def test_a_frozen_in_session_clock_fails_the_run_with_the_reason(
        self, tmp_path, monkeypatch
    ) -> None:
        _published(LocalStore(tmp_path))
        monkeypatch.setattr(track_c, "_now", lambda: IN_SESSION)
        with pytest.raises(TraderPinRefusedError):
            track_c.release_pin_handler(self._args(tmp_path))
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "failed" and "market_hours" in manifest["reason"]
        assert not LocalStore(tmp_path).exists(TRADER_PIN_KEY)

    def test_no_smoke_fails_the_run_with_the_reason(self, tmp_path, monkeypatch) -> None:
        _published(LocalStore(tmp_path))
        monkeypatch.setattr(track_c, "_now", lambda: AFTER_CLOSE)
        with pytest.raises(TraderPinRefusedError):
            track_c.release_pin_handler(self._args(tmp_path))
        assert "no_passing_smoke" in self._manifest(tmp_path)["reason"]

    def test_a_gated_pin_records_the_smoke_as_its_input(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        key, document = _trader_smoke_manifest(store, SHA_A)
        monkeypatch.setattr(track_c, "_now", lambda: AFTER_CLOSE)
        assert track_c.release_pin_handler(self._args(tmp_path)) == 0
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "ok"
        assert key in [i["key"] for i in manifest["inputs"]]
        (metric,) = [m for m in manifest["metrics"] if m["name"] == "release_pointer_moved"]
        assert document["run_id"] in metric["status_reason"]
        assert metric["status_reason"].startswith("trader moved from (unset)")

    def test_a_dry_run_checks_the_gate_and_moves_nothing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        _, document = _trader_smoke_manifest(store, SHA_A)
        monkeypatch.setattr(track_c, "_now", lambda: AFTER_CLOSE)
        track_c.release_pin_handler(self._args(tmp_path, dry_run=True))
        assert document["run_id"] in capsys.readouterr().out
        assert not store.exists(TRADER_PIN_KEY)

    def test_now_is_timezone_aware(self) -> None:
        assert track_c._now().tzinfo is not None
