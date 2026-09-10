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

from crucible.alerts import MUTED_TOPIC_VAR, PAGES_TOPIC_VAR
from crucible.calendar import is_trading_day
from crucible.runmode import RUN_MODE_ENV, RUN_MODE_LIVE
from crucible.store import LocalStore
from crucible.tracker import TRACKER_APP_SSM_PREFIX_VAR, TRACKER_TOKEN_VAR

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


@pytest.fixture(autouse=True)
def declared_run_mode(monkeypatch):
    """Every test invocation declares itself LIVE, in one place.

    `run_manifest.v2` requires `run_mode` and `crucible.runmode` refuses to
    guess it, so a job invoked by a test has to declare one exactly as a
    workflow or an operator does. Declaring it here rather than at ~60 call
    sites keeps the requirement in a single readable statement — and it is a
    DECLARATION, not a default: `crucible.runmode.resolve_run_mode` still has
    none, and `tests/test_run_mode_contract.py` clears this variable to show
    the refusal firing, which is what proves the requirement is real.

    `monkeypatch.setenv` unwinds per test, so nothing leaks between them.
    """
    monkeypatch.setenv(RUN_MODE_ENV, RUN_MODE_LIVE)


@pytest.fixture(autouse=True)
def declared_topics(monkeypatch):
    """The account id and every topic name this suite used to carry as
    literals are gone (`alpha-engine-config-I10156`): this repo is public and
    `crucible.alerts.muted_topic`/`pages_topic` now RAISE unless their
    environment variable is set. These values are SYNTHETIC — they name no
    real AWS resource — and exist only so every test declares them exactly as
    a workflow or an operator would, in one place rather than at each call
    site. A test proving the raise-on-unset behaviour itself clears the
    relevant variable with `monkeypatch.delenv`, which unwinds this fixture's
    `setenv` for that one test only.

    `crucible.autonomy.machine_principals` is NOT declared here
    (`alpha-engine-config-I10307`, corrected 2026-09-09): it no longer reads
    an environment variable at all, having been the hand-kept-twin defect
    `alpha-engine-config-I10156` itself introduced. It derives the allowlist
    from a CloudFormation client instead, and `tests/test_autonomy.py`
    supplies a fake one per test rather than through this autouse fixture,
    since (unlike a topic name) the interesting cases are about WHAT the
    stack contains, not merely that a value is present.
    """
    monkeypatch.setenv(MUTED_TOPIC_VAR, "test-muted-topic")
    monkeypatch.setenv(PAGES_TOPIC_VAR, "test-pages-topic")


class ActiveCostAllocationTag:
    """A Cost Explorer stand-in whose `system` key reads `Active`.

    The phase-0 tag clause reads Billing's activation state live
    (`alpha-engine-config-I10076` deliverable 4); a test that expects phase 0
    MET supplies this through `crucible.gate._ce_client`. The autouse fixture
    below makes the default REFUSE, so no test constructs a real client.
    """

    def list_cost_allocation_tags(self, **_request) -> dict:
        return {
            "CostAllocationTags": [
                {"TagKey": "system", "Status": "Active", "LastUpdatedDate": "2026-09-06T14:56:34Z"}
            ]
        }


@pytest.fixture(autouse=True)
def no_live_cost_explorer(monkeypatch):
    """No test reaches Cost Explorer by accident.

    Measured 2026-09-06: two phase-0 tests reached the real API through the
    laptop's default identity and read `AccessDeniedException` -- a suite
    whose result depends on the developer's AWS credentials is not a suite.
    `crucible.gate._ce_client` refuses here; a test that needs a reading
    patches it with `ActiveCostAllocationTag()` or its own stand-in.

    Also resets `crucible.cost`'s process-wide cache + call budget
    (`alpha-engine-config-I10389`) before AND after every test. Without this,
    the cache is a module-level singleton shared by the whole pytest
    process: a reading (or a budget-exhaustion trip) left behind by one test
    would silently short-circuit or fail the next test's Cost Explorer call,
    regardless of which fake client that test installs.
    """
    import crucible.cost as cost_module  # noqa: PLC0415 - local to the fixture
    import crucible.gate as gate_module  # noqa: PLC0415 - local to the fixture

    def _refuse():
        raise RuntimeError(
            "tests must not construct a real Cost Explorer client; patch "
            "crucible.gate._ce_client (tests/conftest.py::ActiveCostAllocationTag is "
            "the Active stand-in)"
        )

    cost_module.reset_default_cache()
    original = gate_module._ce_client
    monkeypatch.setattr(gate_module, "_ce_client", _refuse)
    try:
        # The real constructor, for the one test that asserts what it builds
        # (`tests/test_gate_lazy_clients.py`) without calling the API.
        yield original
    finally:
        cost_module.reset_default_cache()


@pytest.fixture(autouse=True)
def no_tracker_credential(monkeypatch):
    """No test reaches GitHub by accident.

    `crucible.tracker` is the one adapter in this package that talks to
    something other than the store, and it reads its credential from the
    environment. A developer whose shell exports a real token would otherwise
    have the suite post comments to `nousergon/alpha-engine-config` — the
    private tracker — on any test that files a closing record.

    Clearing it here makes the ABSENT case the default everywhere, which is
    also the case every surface must render honestly
    (`tests/test_phase_closing_record.py` grants it explicitly, per test, and
    substitutes the HTTP opener, so the adapter is exercised without a
    socket).
    """
    monkeypatch.delenv(TRACKER_TOKEN_VAR, raising=False)
    # And the App path: with the prefix unset the adapter never imports the
    # minting helper, so no test reaches SSM or GitHub's App endpoint either.
    monkeypatch.delenv(TRACKER_APP_SSM_PREFIX_VAR, raising=False)


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

    def generate_presigned_url(self, operation: str, **kw: Any) -> str:
        """A URL shaped like a real presign, without a real signature.

        boto3 computes this offline from the credential it already holds, so
        the fake does the same: no object lookup, no network. The parameters
        are echoed into the query string so a test can assert WHICH key and
        WHICH lifetime were signed — a fake returning a constant would pass
        for a store that presigned the wrong object.
        """
        params = kw["Params"]
        return (
            f"https://{params['Bucket']}.s3.amazonaws.com/{params['Key']}"
            f"?op={operation}&X-Amz-Expires={kw['ExpiresIn']}&X-Amz-Signature=fake"
        )

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


@pytest.fixture(autouse=True)
def _declared_liquidity_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The liquidity gate has no default in this public tree
    (`crucible.features.compute.liquidity_floor_usd` raises without it), so the
    suite declares a synthetic one. Deliberately NOT the production value: a
    test that would still pass against the real number is a test that would
    also pass if the number leaked back into the tree.
    """
    monkeypatch.setenv("CRUCIBLE_LIQUIDITY_FLOOR_USD", "1000000")
