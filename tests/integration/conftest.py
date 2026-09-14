"""Fixtures for the integration tier — real S3, real ArcticDB.

Normative source: `tests/integration/README.md`, `alpha-engine-config-I10419`.

Every external dependency here is resolved from required environment, never
a literal and never a silent default — the same discipline
`crucible.required.require_env` already carries for the production topic
names, applied to this tier's own dedicated resources. A missing variable
RAISES, naming which one and why, rather than falling back to something that
could resolve to production.

This module is collected on every PR (`pytest --collect-only`, same as
`tests/acceptance`), but its fixtures are never INSTANTIATED there:
`pyproject.toml`'s `addopts` carries `--ignore=tests/integration`, so no PR
or `deploy.yml` invocation reaches a test that would use them. Only
`.github/workflows/integration-nightly.yml` runs this directory for real.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import pytest

from crucible.calendar import is_trading_day
from crucible.features import min_panel_trading_days
from crucible.required import require_env
from crucible.slots import attribution_factor_symbols, declared_benchmark_symbols
from crucible.slots.grading import DEFAULT_HORIZON_TRADING_DAYS
from crucible.store import Store, open_store

__all__ = [
    "INTEGRATION_TRADING_DAY",
    "SETTLED_TRADING_DAY",
    "integration_seed_trading_days",
]

#: Fixed literal, never wall-clock (AGENTS.md, Test discipline). A Tuesday,
#: chosen only because it is not a holiday; re-verified below at collection
#: time via the real trading calendar rather than trusted as a comment.
INTEGRATION_TRADING_DAY = "2026-09-08"

assert is_trading_day(dt.date.fromisoformat(INTEGRATION_TRADING_DAY)), (
    f"INTEGRATION_TRADING_DAY={INTEGRATION_TRADING_DAY!r} is not a real NYSE trading day "
    "per crucible.calendar — every key this tier writes binds to it (plan §4.12), so a "
    "non-trading literal would fail every job's own refusal rather than this assertion."
)

#: A second FIXED literal (`alpha-engine-config-I10633`), exactly
#: `DEFAULT_HORIZON_TRADING_DAYS` (21) NYSE sessions after
#: `INTEGRATION_TRADING_DAY` — never `today` arithmetic, same discipline as
#: the constant above. `experiment.grade`/`promote` score only SETTLED
#: shadows (`crucible.slots.cycle.run_grade`: `forward_returns` needs price
#: data `horizon_trading_days` sessions AFTER the shadow's own date), and a
#: tier with only ONE fixed trading day could never produce one — every
#: nightly `experiment.grade` run would legitimately score zero cuts,
#: forever. Seeding a second, LATER day was chosen over hand-writing shadow
#: history directly into the store (the issue's own alternative (b)):
#: this way the real `data.daily`/`experiment.run` producer path is
#: exercised a second time, at a later date, rather than fabricating the
#: verdicts those producers exist to compute — hand-written history would
#: prove the grader can read a shape it was told to expect, not that the
#: producer chain actually produces a settleable one.
SETTLED_TRADING_DAY = "2026-10-07"

assert is_trading_day(dt.date.fromisoformat(SETTLED_TRADING_DAY)), (
    f"SETTLED_TRADING_DAY={SETTLED_TRADING_DAY!r} is not a real NYSE trading day per "
    "crucible.calendar — every key this tier writes binds to it (plan §4.12)."
)


#: Sessions of headroom above the feature catalogue's own minimum
#: (`crucible.features.min_panel_trading_days`) this tier seeds beyond what
#: `data.daily`'s producer guard strictly requires — never zero: the guard
#: compares the PRODUCER's trailing panel (sized from
#: `crucible.data.daily.DEFAULT_LOOKBACK_DAYS`, itself a calendar-day margin
#: over the same minimum) against this constant's session count, and a seed
#: pinned exactly at the minimum would start failing again the moment either
#: margin shifted by one session.
_SEED_DEPTH_MARGIN_TRADING_DAYS = 20


def integration_seed_trading_days() -> int:
    """Sessions of price history this tier seeds before `INTEGRATION_TRADING_DAY`.

    **Was a literal `300`** (`alpha-engine-config-I10701`, measured against
    run 34797033396): `crucible-PR266` sized the producer's own guard —
    `crucible.data.daily.PanelDepthError` — from
    `crucible.features.min_panel_trading_days()` (313 sessions today), and a
    fixture that still hand-counted 300 failed `test_data_daily`,
    `test_data_weekly` and `test_data_heal` with "the trailing panel ...
    carries 300 session(s), below the 313 the feature catalogue's deepest
    column needs" — a literal that could not notice the catalogue's deepest
    column changing under it, the exact class `DEFAULT_LOOKBACK_DAYS`'s own
    docstring warns against for the producer side. Derived from the same
    producer function the guard itself calls, plus a fixed margin, so this
    tier's seed can never again fall behind that guard without both moving
    together.
    """
    return min_panel_trading_days() + _SEED_DEPTH_MARGIN_TRADING_DAYS


def _sessions_between(start: dt.date, end: dt.date) -> int:
    """Count of NYSE sessions strictly after ``start`` up to and including
    ``end``, walked one calendar day at a time — the same construction
    `integration_arctic_symbols` below already uses, so this assertion
    cannot disagree with what that fixture actually seeds."""
    count = 0
    day = start
    while day < end:
        day += dt.timedelta(days=1)
        if is_trading_day(day):
            count += 1
    return count


assert (
    _sessions_between(
        dt.date.fromisoformat(INTEGRATION_TRADING_DAY), dt.date.fromisoformat(SETTLED_TRADING_DAY)
    )
    == DEFAULT_HORIZON_TRADING_DAYS
), (
    f"SETTLED_TRADING_DAY={SETTLED_TRADING_DAY!r} is not exactly "
    f"{DEFAULT_HORIZON_TRADING_DAYS} NYSE sessions after INTEGRATION_TRADING_DAY="
    f"{INTEGRATION_TRADING_DAY!r} — a shadow produced on the earlier day would not "
    "settle by the later one, and `experiment.grade` would legitimately score zero "
    "cuts against it, same as with only one fixed day."
)

#: The declared universe this tier seeds, writes and grades against — one
#: named constant rather than a literal inside `integration_arctic_symbols`,
#: because the ARM RECIPES `strategy_dir` writes have to be sized from it
#: (`alpha-engine-config-I10705`: a recipe asking for more names than the
#: universe holds makes the control arm's draw unsatisfiable, and
#: `experiment.grade` refuses with "cannot draw a control selection of N from
#: M settled names"). Synthetic tickers naming no real security.
#:
#: **Sixteen names, not three** (`alpha-engine-config-I10705`). Three was
#: enough for `data.daily` to have something real to compile and is not
#: enough to GRADE: `crucible.slots.grading.control_selection` draws
#: `top_n` names, so a recipe whose `top_n` equals the universe makes the
#: planted and the null control pick the identical set and
#: `assert_controls_ordered` correctly refuses the cycle —
#: `GraderControlError: the planted control did not outrank the null control
#: ... margin 0.000000`. A cross-section is what the §10.1 controls measure
#: against; sixteen is the smallest one that leaves a real selection at a
#: `top_n` of :data:`INTEGRATION_ARM_TOP_N` while keeping the ArcticDB seed
#: this fixture writes to a nightly-sized job.
#:
#: **APPEND-ONLY.** The dedicated store is durable across runs, and
#: `experiment.run` writes a shadow whose recorded `population` is the
#: universe as it stood that night; `experiment.grade` settles that shadow
#: later against whatever panel exists then. Drop a ticker and every
#: already-written shadow naming it is STRANDED — measured 2026-09-14,
#: replacing the original three names outright left
#: `PopulationIntegrityError: the population has 1 settled forward return(s)
#: out of 4` on a shadow no later run could ever settle, permanently red. So
#: the original three lead the tuple unchanged and the widening is added
#: after them.
INTEGRATION_UNIVERSE_SYMBOLS: tuple[str, ...] = ("INTGA", "INTGB", "INTGC") + tuple(
    f"INTG{i:02d}" for i in range(13)
)

#: How many names each of this tier's fixture arm recipes selects. Strictly
#: LESS than the universe (a selection of everything is not a selection) and
#: at most the settled names `control_selection` can draw from.
INTEGRATION_ARM_TOP_N = 4

assert 0 < INTEGRATION_ARM_TOP_N < len(INTEGRATION_UNIVERSE_SYMBOLS), (
    f"INTEGRATION_ARM_TOP_N={INTEGRATION_ARM_TOP_N} is not a strict selection from a "
    f"universe of {len(INTEGRATION_UNIVERSE_SYMBOLS)} — the planted and null controls "
    "would pick the same names and every graded cycle would read as a broken grader."
)

#: The three production ArcticDB library names this tier's dedicated library
#: must never collide with — imported, not restated, so a future addition to
#: `nousergon_lib.arcticdb` is caught by this comparison rather than needing
#: a second edit here.
try:
    from nousergon_lib.arcticdb import MACRO_LIB, PRELIMINARY_LIB, UNIVERSE_LIB

    _PRODUCTION_ARCTIC_LIBRARIES = frozenset({UNIVERSE_LIB, MACRO_LIB, PRELIMINARY_LIB})
except ImportError:
    # The `arcticdb` extra is optional at import time everywhere else in this
    # package (`crucible.data.sources.ArcticPriceSource.load_panel`); this
    # tier needs it to run at all, so the collision guard below still needs
    # the three names even when the extra genuinely is not installed here.
    # Never a silent narrower guard: the literal set below is exactly what
    # `nousergon_lib.arcticdb` (0.124.110) declares, so a mismatch is only
    # possible if that package adds a fourth library and this file is not
    # updated with it.
    _PRODUCTION_ARCTIC_LIBRARIES = frozenset({"universe", "macro", "preliminary"})


@pytest.fixture(scope="session")
def integration_store_uri() -> str:
    """The dedicated S3 prefix this whole tier writes under.

    Refused if it does not carry an `integration` path segment — a generic,
    non-literal defence: this repo forbids infrastructure-identifier
    literals (`tests/test_no_infra_literals.py`), so this guard cannot name
    the production prefix to compare against and instead requires the
    dedicated one to say what it is.
    """
    uri = require_env(
        "CRUCIBLE_INTEGRATION_STORE_URI",
        refusing_to="run the integration tier against a guessed or production store",
    )
    segments = uri.replace("s3://", "").strip("/").split("/")
    assert "integration" in segments, (
        f"CRUCIBLE_INTEGRATION_STORE_URI={uri!r} carries no 'integration' path segment — "
        "this tier refuses to write anywhere that is not visibly a dedicated prefix, per "
        "the hard constraint 'never the production artifact prefixes'."
    )
    return uri


@pytest.fixture(scope="session")
def integration_store(integration_store_uri: str) -> Store:
    return open_store(integration_store_uri)


@pytest.fixture(scope="session")
def integration_arctic_bucket() -> str:
    return require_env(
        "CRUCIBLE_INTEGRATION_ARCTIC_BUCKET",
        refusing_to="open an ArcticDB connection with a guessed or production bucket",
    )


@pytest.fixture(scope="session")
def integration_arctic_library(integration_arctic_bucket: str) -> str:
    """A library name dedicated to this tier — never `universe`/`macro`/
    `preliminary`, the three names `nousergon_lib.arcticdb` reserves for
    production reads.
    """
    name = require_env(
        "CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY",
        refusing_to="write ArcticDB test rows with a guessed or production library name",
    )
    assert name not in _PRODUCTION_ARCTIC_LIBRARIES, (
        f"CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY={name!r} collides with a production ArcticDB "
        f"library name ({sorted(_PRODUCTION_ARCTIC_LIBRARIES)}) — this tier must never "
        "write into the library the real jobs read from."
    )
    return name


@pytest.fixture(scope="session")
def arctic_library(integration_arctic_bucket: str, integration_arctic_library: str) -> Any:
    """A real, dedicated ArcticDB library — opened generically, never through
    `nousergon_lib.arcticdb.open_universe_lib` and its siblings, which are
    hard-wired to the three production library names (see README.md, "Not
    exercised, and why"). `create_if_missing=True` mirrors those helpers'
    own cold-start shape.
    """
    from nousergon_lib.arcticdb import open_arctic

    arctic = open_arctic(integration_arctic_bucket)
    return arctic.get_library(integration_arctic_library, create_if_missing=True)


@pytest.fixture(scope="session")
def integration_pages_topic() -> str:
    return require_env(
        "CRUCIBLE_INTEGRATION_PAGES_TOPIC",
        refusing_to="let alerts.sweep/heartbeat publish to a guessed or production topic",
    )


@pytest.fixture(scope="session")
def integration_muted_topic() -> str:
    return require_env(
        "CRUCIBLE_INTEGRATION_MUTED_TOPIC",
        refusing_to="read the old-alerts-muted clause against a guessed or production topic",
    )


def _dedicated_topic_arn(name: str) -> str:
    """The ARN of a topic this tier already knows by NAME.

    `crucible.alerts.topic_arn` is the one adapter that resolves an ARN, and
    it reads it from the environment because the box's own user-data exports
    it straight from the `crucible-v2` stack output
    (`CRUCIBLE_PAGES_TOPIC_ARN`, then derives the NAME from it:
    `export CRUCIBLE_PAGES_TOPIC=${CRUCIBLE_PAGES_TOPIC_ARN##*:}`). This tier
    has the two halves the other way round — the repository variables carry
    the dedicated topics' NAMES — and the integration identity holds neither
    `cloudformation:DescribeStacks` nor `sns:ListTopics`/`GetTopicAttributes`
    (see `crucible-v2.yaml`'s `IntegrationRole`, which grants `sns:Publish`
    and `sns:ListSubscriptionsByTopic` and nothing else), so the ARN is
    composed from the caller's OWN account and region plus the declared name.

    Nothing here is a literal this repo forbids
    (`tests/test_no_infra_literals.py`): the account comes from
    `sts:GetCallerIdentity` at run time, the region from the session, and the
    topic name from the repository variable. And nothing here can silently
    resolve to a production topic — `crucible.alerts.topic_arn` re-reads the
    composed value and REFUSES it unless its last segment equals the name
    `CRUCIBLE_PAGES_TOPIC`/`CRUCIBLE_MUTED_TOPIC` carries, which
    :func:`_dedicated_topic_env` sets from the same dedicated variables.
    """
    import boto3  # noqa: PLC0415 - lazy, one call site

    session = boto3.session.Session()
    region = (session.region_name or os.environ.get("AWS_REGION") or "").strip()
    if not region:
        raise RuntimeError(
            "no AWS region is configured, so the dedicated SNS topic ARNs cannot be "
            "composed. Set AWS_REGION (integration-nightly.yml declares it) rather than "
            "letting a default region decide which account's topic a page reaches."
        )
    account = session.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:sns:{region}:{account}:{name}"


@pytest.fixture(scope="session")
def dedicated_topic_arns(integration_pages_topic: str, integration_muted_topic: str) -> dict:
    """Resolved once per session — one `sts:GetCallerIdentity` for the tier,
    not one per test, while the environment that USES them is re-declared per
    test by :func:`_dedicated_topic_env` below."""
    return {
        "CRUCIBLE_PAGES_TOPIC_ARN": _dedicated_topic_arn(integration_pages_topic),
        "CRUCIBLE_MUTED_TOPIC_ARN": _dedicated_topic_arn(integration_muted_topic),
    }


@pytest.fixture(autouse=True)
def _dedicated_topic_env(
    integration_pages_topic: str,
    integration_muted_topic: str,
    dedicated_topic_arns: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point every job this test invokes at the dedicated topics — names AND ARNs.

    **Function-scoped, and that is the fix** (`alpha-engine-config-I10705`,
    group B). This fixture was session-scoped and set `os.environ` directly,
    which reads as sufficient and is not: `tests/conftest.py::declared_topics`
    is an autouse FUNCTION-scoped fixture that `monkeypatch.setenv`s
    `CRUCIBLE_PAGES_TOPIC`/`CRUCIBLE_MUTED_TOPIC` to the unit suite's two
    SYNTHETIC names, and it applies to this directory too (a `tests/`
    conftest is a parent of `tests/integration/`). Higher-scoped fixtures are
    set up first, so the synthetic names overwrote this tier's dedicated ones
    on every single case — silently, because no case asserted which topic
    name was in the environment and the ARN half was unset, so
    `TopicUnresolvedError` fired first and hid it. Re-declared here at
    function scope, this fixture is set up AFTER its `tests/conftest.py`
    counterpart and wins, and `monkeypatch` unwinds it exactly the same way.

    The two `_ARN` variables are what `crucible.alerts._krepis_publish`
    actually requires — the missing half that failed `test_alerts_sweep` and
    `test_heartbeat` — and `crucible.alerts.topic_arn` cross-checks each
    against the corresponding NAME, so the pair can never be half-dedicated.
    """
    monkeypatch.setenv("CRUCIBLE_PAGES_TOPIC", integration_pages_topic)
    monkeypatch.setenv("CRUCIBLE_MUTED_TOPIC", integration_muted_topic)
    for variable, arn in dedicated_topic_arns.items():
        monkeypatch.setenv(variable, arn)


#: Captured at COLLECTION time, before any fixture has run. The tracker
#: credential this tier is granted (`alpha-engine-config-I10672`:
#: this tier's own identity holds `ssm:GetParameter` on the fleet App's
#: three tracker parameters) arrives as
#: `CRUCIBLE_TRACKER_APP_SSM_PREFIX` in `integration-nightly.yml`'s `env:` —
#: and `tests/conftest.py::no_tracker_credential` DELETES it, autouse, for
#: every test in the tree including this directory. That deletion is correct
#: for the unit suite (no test may post to the private tracker by accident)
#: and wrong for exactly this one, whose whole contract is a REAL
#: `report.morning` against a dedicated public tracker repo. Read from
#: `os.environ` at import rather than inside the fixture because by then the
#: variable is already gone.
_WORKFLOW_TRACKER_ENV = {
    name: os.environ.get(name)
    for name in ("CRUCIBLE_TRACKER_APP_SSM_PREFIX", "CRUCIBLE_TRACKER_TOKEN")
}


@pytest.fixture(autouse=True)
def _real_alert_delivery_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let this tier's publishes actually leave the process.

    `alpha-engine-config-I10705`. `krepis.alerts.publish` carries a
    cross-repo, defence-in-depth guard: with `PYTEST_CURRENT_TEST` set it
    short-circuits BOTH channels to `ok=False, detail="suppressed in test env
    (PYTEST_CURRENT_TEST set)"` before it reaches a transport, so no consumer
    suite can page an operator by forgetting to stub it. Its declared escape
    hatch is `ALPHA_ENGINE_ALLOW_TEST_ALERTS`.

    That guard is right for every other suite in this repo and is the exact
    negation of this one's contract. Two consequences, both measured
    2026-09-14 against the dedicated store:

    * `test_report_morning` FAILED on it — `UndeliveredError ... any_ok=False`
      — because `deliver` refuses to record a report nobody received, and
      under the guard nobody ever does.
    * `test_alerts_sweep` and `test_heartbeat` PASSED under it, and that is
      worse: their own docstrings claim a "real SNS publish path" and a real
      heartbeat "to the dedicated pages topic", and every publish they made
      was short-circuited before reaching SNS. A green case asserting a
      delivery that structurally could not happen is the detection blindness
      this tier exists to remove, not a passing test.

    Safe to lift here and nowhere else: every destination this tier publishes
    to is dedicated and non-notifying — the two zero-subscriber SNS topics
    (`IntegrationRole` grants `sns:Publish` on exactly those two) and krepis'
    `console_only` Telegram destination, which writes a named artifact and
    sends nothing.

    **`NOUSERGON_ALLOW_TEST_EVENTS` is deliberately NOT set.** That is the
    separate switch over `krepis.fleet_events`, whose destination is the
    PRODUCTION intake queue — there is no dedicated integration equivalent of
    it, so the tier leaves `event_emitted` False rather than feeding synthetic
    findings into the fleet's real response plane. `any_ok` reads the two
    channels only, so this changes nothing about what the cases assert.
    """
    monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")


@pytest.fixture(autouse=True)
def _tracker_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give this tier back the tracker credential the root conftest removes.

    `alpha-engine-config-I10705`, group D. `test_report_morning` failed with
    `TrackerError: no tracker credential ($CRUCIBLE_TRACKER_APP_SSM_PREFIX
    and $CRUCIBLE_TRACKER_TOKEN unset)` on run 34801438655 even though
    `integration-nightly.yml` declares the prefix and the repository variable
    has been set since 2026-09-06 — the variable reached the process and an
    autouse fixture two directories up deleted it. Restored per test, from
    what the WORKFLOW actually exported, so an unset prefix still fails this
    case loudly rather than being papered over with a literal.
    """
    for name, value in _WORKFLOW_TRACKER_ENV.items():
        if value:
            monkeypatch.setenv(name, value)


#: `report.morning`'s own dedicated destinations (`alpha-engine-config-
#: I10458`, Brian ruling option (a), narrow) — neither is an infrastructure
#: identifier this repo forbids as a literal (`tests/test_no_infra_
#: literals.py`): one names THIS public repo, the other is krepis' own
#: destination-enum string, not a bucket/role/topic name. Both resolve
#: through `crucible.required.optional_env`, unset in production.
#:
#: **The GitHub half — `nousergon/crucible`, not a second, muted tracker
#: repo.** A "muted" tracker equivalent to `CRUCIBLE_INTEGRATION_MUTED_
#: TOPIC` does not exist for GitHub issues the way it does for an SNS topic
#: with no subscribers — an issue always has a repo of record, and creating
#: a dedicated private repo for this alone would be a second tracker to
#: keep alive for one nightly case. `nousergon/crucible` is already public,
#: already where this workflow runs, and posting a rolling `[v2 board]
#: daily update` issue there keeps synthetic content out of the PRIVATE
#: production tracker without inventing new infrastructure.
#:
#: **The Telegram half — `console_only`, not a muted chat.** "a chat id
#: with no members is not a thing" (the issue's own words): krepis'
#: `console_only` destination, paired with the `console_artifact`
#: `crucible.morning.deliver` now always passes, delivers to a NAMED
#: durable artifact and returns `ok=True` WITHOUT sending anything to
#: Telegram (`krepis.alerts.resolve_destination`) — a real, fully-exercised
#: `publish()` call, not a `--dry-run` stub that would skip the send
#: entirely and prove nothing about it.
CRUCIBLE_MORNING_TRACKER_REPO = "nousergon/crucible"
CRUCIBLE_MORNING_TELEGRAM_DESTINATION = "console_only"


@pytest.fixture(scope="session", autouse=True)
def _morning_destination_env() -> None:
    os.environ["CRUCIBLE_MORNING_TRACKER_REPO"] = CRUCIBLE_MORNING_TRACKER_REPO
    os.environ["CRUCIBLE_MORNING_TELEGRAM_DESTINATION"] = CRUCIBLE_MORNING_TELEGRAM_DESTINATION


@pytest.fixture(scope="session", autouse=True)
def _arctic_bucket_env(integration_arctic_bucket: str) -> None:
    """`crucible.track_a._source` resolves `ArcticPriceSource`'s BUCKET from
    `CRUCIBLE_ARCTIC_BUCKET` — there is no `--arctic-bucket` CLI flag,
    deliberately (`alpha-engine-config-I10457`): the production entry point
    takes no bucket override, only a LIBRARY one, since the bucket is not
    this tier's isolation unit (see README.md, "Why the ArcticDB bucket is
    not itself dedicated" — the LIBRARY is). Pointing `CRUCIBLE_ARCTIC_BUCKET`
    at this tier's bucket for the session is therefore safe as long as every
    ArcticDB-reading case below also passes `--arctic-library`, which is the
    actual isolation guarantee: `ArcticPriceSource(library=...)` never falls
    through to the production `universe`/`macro`/`preliminary` libraries
    (see `crucible.data.sources.ArcticPriceSource`'s own docstring).
    """
    os.environ["CRUCIBLE_ARCTIC_BUCKET"] = integration_arctic_bucket


@pytest.fixture(scope="session", autouse=True)
def _declared_universe_env(
    integration_arctic_symbols: list[str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point `CRUCIBLE_UNIVERSE_URI` at a local membership document naming
    this tier's own dedicated symbols (`crucible.data.universe`).

    Every case above that calls `data.daily`/`data.weekly`/`data.heal`
    directly passes `--symbols` on its own argv and never needed this — the
    explicit argument always wins over the environment
    (`crucible.track_a._declared_universe`). `weekly`
    (`alpha-engine-config-I10633`) is the first case in this tier that does
    NOT: `crucible.weekly.Stage.argv` carries no `--symbols` for any stage,
    by design (`crucible/data/universe.py`'s own docstring — "the weekly
    arc's `Stage.argv` carries no `--symbols`... every scheduled data job —
    and the first stage of every weekly arc — failed by construction with
    `UndeclaredUniverseError`", the exact gap this env var exists to close in
    production). Without it, `test_weekly`'s own `data.weekly` stage would
    refuse before ever reaching the ArcticDB read this tier exists to prove.
    A local path is a fully real exercise of this resolution path, not a
    shortcut around it — `load_declared_universe` reads a local path and an
    `s3://` URI through the same code, differing only in how bytes are
    fetched (`crucible/data/universe.py`'s own docstring).
    """
    document = tmp_path_factory.mktemp("integration-universe") / "constituents.json"
    document.write_text(
        json.dumps({"tickers": list(integration_arctic_symbols)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.environ["CRUCIBLE_UNIVERSE_URI"] = str(document)


@pytest.fixture(scope="session")
def integration_arctic_symbols(arctic_library: Any, strategy_dir: Path) -> list[str]:
    """Synthetic OHLCV rows for three symbols, written into the DEDICATED
    library so `data.daily`/`data.weekly`/`data.heal`
    (`alpha-engine-config-I10457`) have something real to read through
    `ArcticPriceSource(library=...)` — proving the full `crucible.cli.main`
    wire end to end, the same property `test_arctic_connectivity.py` proves
    at the raw `Library.write`/`read` level. Written in the raw
    Open/High/Low/Close/Volume shape `ArcticPriceSource.load_panel`'s
    `_OHLCV_RENAME` expects — never the already-normalized `*_raw` panel
    shape `tests/conftest.py::synthetic_frames` produces for the unit suite's
    `FramePriceSource` fixtures, which is a different contract.

    `integration_seed_trading_days()` sessions BEFORE `INTEGRATION_TRADING_DAY`
    — derived from `crucible.features.min_panel_trading_days()` (the same
    function `crucible.data.daily.MIN_PANEL_TRADING_DAYS` calls to size the
    producer's own `PanelDepthError` guard) plus a fixed margin, never a
    literal session count (`alpha-engine-config-I10701`; see
    `integration_seed_trading_days`'s own docstring for the literal-300
    defect this replaced) — so the feature layer's deepest column (today,
    252-session residual momentum composed over a 61-session residual
    stream) is never starved — plus every session THROUGH `SETTLED_TRADING_DAY`
    (`alpha-engine-config-I10633`), one continuous random walk rather than
    two independent ones, so a `data.daily`/`data.weekly` run at either fixed
    day reads the same coherent series and a shadow produced at
    `INTEGRATION_TRADING_DAY` settles against real forward prices rather than
    a second, unrelated draw. Every session up to and including
    `INTEGRATION_TRADING_DAY` is byte-for-byte what this fixture wrote before
    this extension — the walk only gains a forward tail, nothing before it
    changes. Torn down after the session so a rerun starts from the same
    clean state, mirroring `test_arctic_connectivity.py`'s own idempotent
    teardown.

    **Also seeds every symbol `data.daily`'s producer reads BEYOND the
    declared universe** (`alpha-engine-config-I10701`, measured against run
    34797908861: `test_data_daily`/`weekly`/`heal`/`experiment_run`/`weekly`/
    `experiment_grade` all failed `MissingSourceError: ... returned zero
    symbols`, and the captured `WARNING` this tier's own log never surfaced —
    `crucible.integration_summary.integration_test_body` captures the pytest
    subprocess's stdout/stderr and keeps only the last 40 lines as the
    manifest `reason` — read `... failed for SPY: ... E_NO_SUCH_VERSION ...`).
    `crucible.data.daily.run_daily` fetches TWO declared sources of panel
    symbols beyond the universe it was handed — every slot's non-`"population"`
    `benchmark` (`crucible.slots.declared_benchmark_symbols`, today just S's
    `"SPY"`, `alpha-engine-config-I10635`) and every factor-attribution proxy
    ticker (`crucible.slots.attribution_factor_symbols`, `-I10683`) — and
    raises if either has no panel row, exactly like a ticker missing from the
    universe itself. This fixture predates both: it seeded only the three
    declared-universe tickers, so the very first `data.daily` run in this
    tier's history was always going to fail the moment either producer clause
    landed, and did. Fixed at THIS layer (the seed), not by removing the
    producer's own guard: the guard is correct (a compile-time gap here would
    otherwise surface only as a grading refusal on a box, per that function's
    own comment) and `crucible.slots.strategy_dir` here declares no
    `attribution.yaml`, so `attribution_factor_symbols` returns the same
    empty set it always has — only the benchmark set is non-empty today.
    Derived from the producer's own two functions, never a second hand-kept
    literal (the exact discipline `declared_benchmark_symbols`'s own
    docstring names as this repo's reason for existing), so this fixture
    cannot fall behind a THIRD extra-symbol source the same way it fell
    behind these two. Extra symbols are written into the library — so the
    producer's read finds them — but never returned from this fixture: they
    are not part of the declared UNIVERSE (`CRUCIBLE_UNIVERSE_URI`), the same
    distinction the producer itself draws.
    """
    import math
    import random

    import pandas as pd

    symbols = list(INTEGRATION_UNIVERSE_SYMBOLS)
    extra_symbols = sorted(
        (declared_benchmark_symbols() | attribution_factor_symbols(strategy_dir=strategy_dir))
        - set(symbols)
    )
    seeded_symbols = symbols + extra_symbols
    anchor = dt.date.fromisoformat(INTEGRATION_TRADING_DAY)
    settled = dt.date.fromisoformat(SETTLED_TRADING_DAY)
    days: list[dt.date] = []
    day = anchor
    while len(days) < integration_seed_trading_days():
        if is_trading_day(day):
            days.append(day)
        day -= dt.timedelta(days=1)
    days.sort()
    # Forward tail through the settled day — `anchor` itself is already the
    # last element above, so this starts the day after it.
    day = anchor
    while day < settled:
        day += dt.timedelta(days=1)
        if is_trading_day(day):
            days.append(day)

    rng = random.Random(20260908)
    for symbol in seeded_symbols:
        # A PERSISTENT per-name drift and volatility, drawn once per symbol
        # (`alpha-engine-config-I10705`), mirroring
        # `tests/conftest.py::synthetic_frames` — whose own docstring states
        # why: "that is what makes momentum a signal rather than noise:
        # without persistent per-name drift, a momentum ranker is a random
        # selector and the whole grading path would be tested against a
        # market in which nothing is measurable". This fixture drew one
        # shared (0.0003, 0.01) step distribution for every symbol, which was
        # invisible while only `data.daily` read the panel and became the
        # second half of `experiment.grade`'s refusal the moment the arena
        # cycle ran against it.
        drift = rng.gauss(0.0004, 0.0009)
        vol = rng.uniform(0.008, 0.02)
        price = rng.uniform(20.0, 200.0)
        rows = []
        for _day in days:
            price = max(1.0, price * math.exp(rng.gauss(drift, vol)))
            rows.append(
                {
                    "Open": price * 0.998,
                    "High": price * 1.005,
                    "Low": price * 0.995,
                    "Close": price,
                    "Volume": rng.uniform(3e5, 4e6),
                }
            )
        frame = pd.DataFrame(rows, index=pd.DatetimeIndex(days))
        arctic_library.write(symbol, frame)
    yield symbols
    for symbol in seeded_symbols:
        if arctic_library.has_symbol(symbol):
            arctic_library.delete(symbol)


@pytest.fixture(scope="session")
def strategy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three U-slot arms that do not share a ranking callable.

    Mirrors `tests/conftest.py::strategy_dir` exactly (the same fixture the
    unit suite already exercises `crucible.slots.arms.load_arm_specs`
    against) rather than inventing a second recipe shape: three arms because
    `min_active_arms` is 3 — a slot with fewer produces zero or one
    comparison, the `no_promotable_challenger` refusal.
    """
    root = tmp_path_factory.mktemp("integration-strategy") / "strategy"
    arms = root / "arms" / "u"
    arms.mkdir(parents=True)
    # `top_n` comes from this tier's own declared constant, never the unit
    # suite's literal `8` (`alpha-engine-config-I10705`): the unit fixture
    # this one mirrors runs against `synthetic_frames`' 40-name cross-section,
    # and a recipe asking for eight names out of a three-name universe failed
    # `experiment.grade` with `ValueError: cannot draw a control selection of
    # 8 from 4 settled names`. Named beside the universe it is drawn from, and
    # asserted a strict subset there, so the two can never drift into either
    # the too-large or the select-everything failure again.
    top_n = INTEGRATION_ARM_TOP_N
    recipes = {
        "momentum_sleeve": ("momentum_sleeve", {"top_n": top_n}),
        "tech_score_gate": ("tech_score_gate", {"top_n": top_n}),
        "mom_12_1_sleeve": ("mom_12_1_sleeve", {"top_n": top_n}),
    }
    for name, (ranker, params) in recipes.items():
        body = [
            f"name: {name}",
            "slot: u",
            f"ranker: {ranker}",
            f"registered_at: '{INTEGRATION_TRADING_DAY}'",
            "params:",
        ]
        body += [f"  {k}: {v}" for k, v in params.items()]
        body.append(f"notes: integration-tier fixture recipe for {name}")
        (arms / f"{name}.yaml").write_text("\n".join(body) + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="session", autouse=True)
def seeded_strategy_tree(integration_store: Store, strategy_dir: Path) -> list[str]:
    """Publish this tier's own arm recipes into the dedicated store.

    `alpha-engine-config-I10705`, group A, second half. `crucible.slots.arms
    .load_arm_specs` has two sources: a CHECKOUT (`--strategy-dir`, what a
    developer editing `alpha-engine-config/strategy/` expects to win) and the
    tree SYNCED INTO THE STORE at `strategy/current/arms/{slot}/*.yaml`,
    which is what a spot instance reads because it has no checkout. Only
    `test_experiment_new` passes `--strategy-dir`; `experiment.run`,
    `experiment.grade` and `promote` deliberately do not — the production
    dispatch passes none either — so every one of them read the store branch
    against a prefix nothing had ever written and refused with
    `FileNotFoundError: no arm recipes found for slot 'u'`. The dedicated
    store had no strategy tree at all.

    Seeded through the store keys `crucible.keys.strategy_arm_key` defines,
    so the REAL loader path is what resolves them — this fixture writes
    bytes, it does not teach the loader a second way to find a recipe. The
    bytes are :func:`strategy_dir`'s own three fixture recipes, read back
    from the files that fixture wrote: one source for both branches, so the
    checkout case and the store case can never describe different arms, and
    never a copy of the private `alpha-engine-config/strategy/` tree — these
    are generic fixture recipes with no tuned parameter in them, which is
    what lets them live in a public repo at all.

    Not torn down. The dedicated prefix is disposable and every key here is
    overwritten byte-for-byte by the next run; deleting them would need an
    `s3:DeleteObject` grant on the store prefix that `IntegrationRole`
    deliberately withholds ("an authority nothing exercises is one this
    identity does not get").
    """
    from crucible.keys import strategy_arm_key

    names: list[str] = []
    for path in sorted((strategy_dir / "arms" / "u").glob("*.yaml")):
        integration_store.put_bytes(strategy_arm_key("u", path.stem), path.read_bytes())
        names.append(path.stem)
    assert names, (
        "the strategy_dir fixture wrote no U-slot recipes, so this fixture seeded an "
        "empty strategy tree — every store-branch arm load would still refuse."
    )
    return names


@pytest.fixture(scope="session")
def compiled_week_panels(
    integration_store_uri: str,
    integration_arctic_library: str,
    integration_arctic_symbols: list[str],
) -> list[str]:
    """Compile every OTHER session of `INTEGRATION_TRADING_DAY`'s trading week.

    `alpha-engine-config-I10705`, group A. `crucible.data.weekly.run_weekly`
    compiles only `ctx.trading_day` and then asserts that every other session
    of that week already carries a compiled panel — a real production
    invariant, because each weekday's own scheduled `data.daily` fires and
    the week's denominator is otherwise undeclared. This tier compiled ONE
    trading day, so `data.weekly` was always going to refuse the moment
    `test_data_daily`'s own blocker cleared, and it did
    (`DataGapError ... ['2026-09-02', '2026-09-03', '2026-09-04']`).

    Option (a) of the issue's two: mirror production's daily cadence by
    running the real `data.daily` job for each missing session, rather than
    option (b), re-anchoring the fixture on a week whose first session is
    `INTEGRATION_TRADING_DAY`. (b) would move a FIXED literal that
    `SETTLED_TRADING_DAY` is pinned exactly `DEFAULT_HORIZON_TRADING_DAYS`
    sessions after, and would buy the week's invariant a pass by choosing a
    week where it says nothing — the opposite of exercising it.

    The sessions come from `crucible.data.weekly.week_sessions`, the same
    function `run_weekly` grades against, so this fixture cannot compile a
    different week from the one the job checks.
    """
    from crucible.cli import main as cli_main
    from crucible.data.weekly import week_sessions

    anchor = dt.date.fromisoformat(INTEGRATION_TRADING_DAY)
    compiled: list[str] = []
    for session in week_sessions(anchor):
        if session == anchor:
            continue
        cli_main(
            [
                "data.daily",
                "--store",
                integration_store_uri,
                "--arctic-library",
                integration_arctic_library,
                "--symbols",
                ",".join(integration_arctic_symbols),
                "--run-mode",
                "live",
                "--date",
                session.isoformat(),
            ]
        )
        compiled.append(session.isoformat())
    return compiled
