"""`pages_topic_confirmed_subscribers`: who would actually receive a page.

alpha-engine-config-I10024: `crucible-v2-pages` had ZERO subscribers from
creation until 2026-09-04 while ten pages were published to nobody. The
template asserted a subscriber; nothing measured one. The weekly heartbeat now
reads the topic's subscription list and files the answer on its manifest.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import (
    PAGES_TOPIC_ARN_VAR,
    PENDING_CONFIRMATION,
    SUBSCRIBERS_METRIC,
    StoreAccessError,
    heartbeat,
    pages_topic_subscribers,
)
from crucible.manifest import manifest_key, validate
from crucible.runner import run_job
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
TOPIC = "arn:aws:sns:us-east-1:123456789012:crucible-v2-pages"


class _Sns:
    """`list_subscriptions_by_topic` with the real response shape, paged."""

    def __init__(self, *pages: list[dict], fail: Exception | None = None) -> None:
        self.pages = list(pages) or [[]]
        self.fail = fail
        self.requests: list[dict] = []

    def list_subscriptions_by_topic(self, **request) -> dict:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        index = int(request.get("NextToken") or 0)
        response: dict = {"Subscriptions": self.pages[index]}
        if index + 1 < len(self.pages):
            response["NextToken"] = str(index + 1)
        return response


def _sub(protocol: str, *, confirmed: bool = True) -> dict:
    return {
        "Protocol": protocol,
        "SubscriptionArn": (
            f"{TOPIC}:00000000-0000-0000-0000-000000000001" if confirmed else PENDING_CONFIRMATION
        ),
        "TopicArn": TOPIC,
    }


def _sweep_ran(store: LocalStore) -> None:
    store.put_bytes(
        manifest_key("alerts.sweep", FRIDAY.isoformat()),
        json.dumps({"status": "ok", "cost_usd": 0.0}).encode(),
    )


class TestTheReading:
    def test_a_confirmed_email_and_a_lambda_leg_is_met(self) -> None:
        reading = pages_topic_subscribers(TOPIC, sns=_Sns([_sub("email"), _sub("lambda")]))
        assert reading.met
        assert reading.confirmed_human_legs == ("email",)
        assert reading.lambda_legs == 1
        assert reading.metric(now=SATURDAY_NIGHT)["status"] == "OK"

    def test_a_pending_email_is_not_a_confirmed_one(self) -> None:
        """The live state from 2026-09-04 to whenever Brian clicked: the leg
        exists in the template and delivers nothing."""
        reading = pages_topic_subscribers(
            TOPIC, sns=_Sns([_sub("email", confirmed=False), _sub("lambda")])
        )
        assert not reading.met
        assert reading.pending_legs == ("email",)
        metric = reading.metric(now=SATURDAY_NIGHT)
        assert metric["status"] == "FAIL"
        assert PENDING_CONFIRMATION in metric["status_reason"]

    def test_a_human_leg_without_the_lambda_leg_is_not_met(self) -> None:
        metric = pages_topic_subscribers(TOPIC, sns=_Sns([_sub("email")])).metric(
            now=SATURDAY_NIGHT
        )
        assert metric["status"] == "FAIL"
        assert "no lambda leg" in metric["status_reason"]

    def test_no_subscribers_at_all_is_the_original_defect(self) -> None:
        metric = pages_topic_subscribers(TOPIC, sns=_Sns([])).metric(now=SATURDAY_NIGHT)
        assert metric["status"] == "FAIL"
        assert "none subscribed" in metric["status_reason"]

    def test_the_listing_is_paged_to_the_end(self) -> None:
        sns = _Sns([_sub("lambda")], [_sub("email")])
        reading = pages_topic_subscribers(TOPIC, sns=sns)
        assert reading.met
        assert [r.get("NextToken") for r in sns.requests] == [None, "1"]

    def test_no_endpoint_address_reaches_the_metric(self) -> None:
        sub = _sub("email")
        sub["Endpoint"] = "someone@example.com"
        metric = pages_topic_subscribers(TOPIC, sns=_Sns([sub, _sub("lambda")])).metric(
            now=SATURDAY_NIGHT
        )
        assert "example.com" not in json.dumps(metric)

    def test_a_denied_listing_raises_rather_than_reading_zero(self) -> None:
        with pytest.raises(PermissionError):
            pages_topic_subscribers(TOPIC, sns=_Sns(fail=PermissionError("AccessDenied")))


class TestTheHeartbeatCarriesIt:
    def test_the_row_is_on_the_summary_and_validates_on_a_manifest(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        monkeypatch.setenv(PAGES_TOPIC_ARN_VAR, TOPIC)
        store = LocalStore(tmp_path)
        _sweep_ran(store)
        sns = _Sns([_sub("email"), _sub("lambda")])
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport, sns=sns)
        assert summary["subscribers"]["name"] == SUBSCRIBERS_METRIC
        assert summary["subscribers"]["status"] == "OK"
        assert [c.kwargs["severity"] for c in transport.calls] == ["info"]

        def body(ctx) -> None:
            ctx.record_metric(summary["metric"])
            ctx.record_metric(summary["subscribers"])

        run_job("heartbeat", body, store=store, trading_day=FRIDAY, now=SATURDAY_NIGHT)
        manifest = json.loads(store.get_bytes(manifest_key("heartbeat", FRIDAY.isoformat())))
        validate(manifest)
        assert SUBSCRIBERS_METRIC in [m["name"] for m in manifest["metrics"]]

    def test_nobody_listening_is_said_on_the_channel_at_error_severity(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        """The Lambda leg still forwards to Telegram when the human leg is
        gone, so the heartbeat says so THERE, not only on a manifest row."""
        monkeypatch.setenv(PAGES_TOPIC_ARN_VAR, TOPIC)
        store = LocalStore(tmp_path)
        _sweep_ran(store)
        summary = heartbeat(
            store,
            now=SATURDAY_NIGHT,
            transport=transport,
            sns=_Sns([_sub("email", confirmed=False), _sub("lambda")]),
        )
        assert summary["subscribers"]["status"] == "FAIL"
        assert transport.calls[0].kwargs["severity"] == "error"
        assert "no CONFIRMED human leg" in transport.calls[0].message

    def test_a_denied_listing_fails_the_run_after_the_heartbeat_is_sent(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        """Same shape as an unlistable manifest prefix (I9960): the proof of
        life goes out, then the run fails naming the action it lacked. The
        row reads `unmeasurable`, never `OK` and never zero."""
        monkeypatch.setenv(PAGES_TOPIC_ARN_VAR, TOPIC)
        store = LocalStore(tmp_path)
        _sweep_ran(store)
        with pytest.raises(StoreAccessError, match="sns:ListSubscriptionsByTopic"):
            heartbeat(
                store,
                now=SATURDAY_NIGHT,
                transport=transport,
                sns=_Sns(fail=PermissionError("AccessDenied")),
            )
        assert transport.pages == 1, "the heartbeat itself still went out"

    def test_no_topic_configured_reads_unmeasurable_and_builds_no_client(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        monkeypatch.delenv(PAGES_TOPIC_ARN_VAR, raising=False)
        store = LocalStore(tmp_path)
        _sweep_ran(store)
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport)
        assert summary["subscribers"]["status"] == "unmeasurable"
        assert PAGES_TOPIC_ARN_VAR in summary["subscribers"]["status_reason"]

    def test_a_dry_run_still_reads_but_sends_nothing(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        monkeypatch.setenv(PAGES_TOPIC_ARN_VAR, TOPIC)
        store = LocalStore(tmp_path)
        _sweep_ran(store)
        sns = _Sns([_sub("email"), _sub("lambda")])
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport, dry_run=True, sns=sns)
        assert summary["subscribers"]["status"] == "OK"
        assert sns.requests, "a read-only listing runs under dry-run"
        assert transport.pages == 0
