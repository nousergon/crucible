"""Shared fixtures: a synthetic market that is a real panel, not a mock.

Every fixture here produces the SAME shapes production produces — NYSE
sessions from `krepis.trading_calendar`, the canonical panel columns, the
real `LocalStore` — so a test against them exercises the contract rather
than a parallel one built to pass.

The prices are seeded and deterministic. That is load-bearing twice over:
the replay gate asserts that grading the same date twice reproduces the same
verdict, and the control arms assert that a planted edge is visible. Neither
claim means anything against a market that moves between runs.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from crucible.calendar import is_trading_day
from crucible.store import LocalStore

#: Enough sessions for the 252-session feature window plus a 21-session
#: horizon plus the dates a ladder needs. Shorter panels make `mom_12_1`
#: null everywhere, which silently disarms the arms that rank on it.
SESSIONS = 320


def sessions_ending(end: dt.date, count: int) -> list[dt.date]:
    """``count`` NYSE sessions ending at ``end``, ascending."""
    out: list[dt.date] = []
    day = end
    while len(out) < count:
        if is_trading_day(day):
            out.append(day)
        day -= dt.timedelta(days=1)
    return sorted(out)


def synthetic_frames(
    *,
    end: dt.date,
    n_tickers: int = 40,
    sessions: int = SESSIONS,
    seed: int = 20260901,
) -> dict[str, object]:
    """A ``{ticker: OHLCV frame}`` mapping with a real, persistent cross-section.

    Each ticker gets its own drift, drawn once. That is what makes momentum
    a signal rather than noise: without persistent per-name drift, a
    momentum ranker is a random selector and the whole grading path would be
    tested against a market in which nothing is measurable.
    """
    import pandas as pd

    days = sessions_ending(end, sessions)
    rng = random.Random(seed)
    frames: dict[str, object] = {}
    for index in range(n_tickers):
        ticker = f"T{index:03d}"
        drift = rng.gauss(0.0004, 0.0009)
        vol = rng.uniform(0.008, 0.02)
        price = rng.uniform(20.0, 300.0)
        rows = []
        for day in days:
            step = drift + rng.gauss(0.0, vol)
            price = max(1.0, price * math.exp(step))
            volume = rng.uniform(3e5, 4e6)
            rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "open_raw": price * 0.998,
                    "high_raw": price * 1.006,
                    "low_raw": price * 0.994,
                    "close_raw": price,
                    "volume_raw": volume,
                }
            )
        frames[ticker] = pd.DataFrame(rows).set_index("trading_day")
    return frames


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


@pytest.fixture
def cycle_date() -> dt.date:
    """A real NYSE session, well clear of a holiday, used as the cycle date."""
    return dt.date(2026, 8, 28)


@pytest.fixture
def frames(cycle_date):
    return synthetic_frames(end=cycle_date)


@pytest.fixture
def source(frames):
    from crucible.data import FramePriceSource

    return FramePriceSource(frames, snapshot="frames:conftest-seed-20260901")


@pytest.fixture
def strategy_dir(tmp_path):
    """A minimal strategy tree: three U arms that do not share a callable.

    Three because `min_active_arms` is 3 — a slot with fewer produces zero
    or one comparison, which is the `no_promotable_challenger` defect the
    engine's own config refuses.
    """
    root = tmp_path / "strategy"
    arms = root / "arms" / "u"
    arms.mkdir(parents=True)
    recipes = {
        "momentum_sleeve": ("momentum_sleeve", {"top_n": 8}),
        "tech_score_gate": ("tech_score_gate", {"top_n": 8}),
        "mom_12_1_sleeve": ("mom_12_1_sleeve", {"top_n": 8}),
    }
    for name, (ranker, params) in recipes.items():
        body = [
            f"name: {name}",
            "slot: u",
            f"ranker: {ranker}",
            "registered_at: '2026-06-01'",
            "params:",
        ]
        body += [f"  {k}: {v}" for k, v in params.items()]
        body.append(f"notes: fixture recipe for {name}")
        (arms / f"{name}.yaml").write_text("\n".join(body) + "\n", encoding="utf-8")
    return root


# ──────────────────────────────────────────────────────────────────────────
# Track C fixtures (alpha-engine-config-I9757): a capturing alert transport
# and a fake S3.
#
# Both are real implementations of the contracts they stand in for, not mocks
# that agree with whatever the caller does. A test double that cannot fail the
# way the real thing fails proves nothing — the audit's central finding was a
# grading loop that ran for months while measuring nothing.
# ──────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402


@dataclass
class CapturedPublish:
    """One call to the transport, recorded verbatim."""

    message: str
    kwargs: dict[str, Any]


@dataclass
class CapturingTransport:
    """Stands in for `krepis.alerts.publish` and counts what was delivered.

    §10.7 requires each scripted fault to produce *exactly one* page, so the
    count has to be observable. It also honours `dedup_key` the way krepis
    does — same key, one delivery — because a transport that delivered twice
    for one key would make the fault-injection assertions pass for the wrong
    reason.
    """

    calls: list[CapturedPublish] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    fail: bool = False

    def __call__(self, message: str, **kwargs: Any) -> Any:
        if self.fail:
            raise RuntimeError("transport unreachable")
        key = kwargs.get("dedup_key")
        if key is not None and key in self.seen:
            # `dedup_skipped=True` alongside `any_ok=True`, because that is
            # what krepis returns and because `crucible.alerts` now reads it
            # to decide whether the bus row may claim delivery. A stand-in
            # that reported a suppressed publish as delivered would make the
            # honesty test pass against a transport that never sent anything.
            return _Result(any_ok=True, destination="deduped", dedup_skipped=True)
        if key is not None:
            self.seen.add(key)
        self.calls.append(CapturedPublish(message, kwargs))
        return _Result(any_ok=True, destination="operator_chat")

    @property
    def pages(self) -> int:
        return len(self.calls)


@dataclass
class _Result:
    any_ok: bool
    destination: str
    #: krepis sets this on a publish its own dedup suppressed. `any_ok` stays
    #: True there — the call did not fail — so `any_ok` alone cannot tell a
    #: delivery from a suppression, and a bus row built on it claims delivery
    #: for a page that never left.
    dedup_skipped: bool = False


@pytest.fixture
def transport() -> CapturingTransport:
    return CapturingTransport()


class FakeS3:
    """An in-memory S3 with the three behaviours the store depends on.

    Pagination, the absent/unreadable distinction, and conditional PUT — the
    three things :class:`crucible.store.S3Store` exists to get right. A fake
    without them would let the store's pagination bug through, which is the
    `--limit N` class that once made a published report 7 of 9 false.
    """

    def __init__(self, page_size: int = 2) -> None:
        self.objects: dict[str, bytes] = {}
        self.page_size = page_size
        self.denied: set[str] = set()

    # -- the boto3 surface the store uses -------------------------------
    def put_object(self, **kw: Any) -> dict[str, Any]:
        key, body = kw["Key"], kw["Body"]
        if "IfNoneMatch" in kw and key in self.objects:
            raise self._client_error("PreconditionFailed")
        if "IfMatch" in kw and self._etag(key) != kw["IfMatch"]:
            raise self._client_error("PreconditionFailed")
        self.objects[key] = body
        return {"ETag": f'"{self._etag(key)}"'}

    def get_object(self, **kw: Any) -> dict[str, Any]:
        key = kw["Key"]
        self._guard(key)
        if key not in self.objects:
            raise self._client_error("NoSuchKey")
        return {"Body": _Body(self.objects[key])}

    def head_object(self, **kw: Any) -> dict[str, Any]:
        key = kw["Key"]
        self._guard(key)
        if key not in self.objects:
            raise self._client_error("404")
        return {"ETag": f'"{self._etag(key)}"'}

    def get_paginator(self, name: str) -> Any:
        assert name == "list_objects_v2"
        return _Paginator(self)

    # -- helpers --------------------------------------------------------
    def _guard(self, key: str) -> None:
        if key in self.denied:
            raise self._client_error("AccessDenied")

    def _etag(self, key: str) -> str:
        import hashlib

        return hashlib.md5(self.objects.get(key, b""), usedforsecurity=False).hexdigest()

    @staticmethod
    def _client_error(code: str) -> Exception:
        from botocore.exceptions import ClientError

        return ClientError({"Error": {"Code": code}}, "op")


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _Paginator:
    def __init__(self, fake: FakeS3) -> None:
        self.fake = fake

    def paginate(self, **kw: Any) -> Any:
        prefix = kw.get("Prefix") or ""
        keys = sorted(k for k in self.fake.objects if k.startswith(prefix))
        size = self.fake.page_size
        for i in range(0, len(keys), size) or [0]:
            yield {"Contents": [{"Key": k} for k in keys[i : i + size]]}


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
