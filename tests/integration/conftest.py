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


@pytest.fixture(scope="session", autouse=True)
def _dedicated_topic_env(integration_pages_topic: str, integration_muted_topic: str) -> None:
    """Point every job this session invokes at the dedicated topics.

    `crucible.required.require_env` is what every live-paging job reads
    `CRUCIBLE_PAGES_TOPIC`/`CRUCIBLE_MUTED_TOPIC` through — session-scoped
    and autouse, so no test in this directory can forget it and no test
    outside this directory (which never imports this conftest) is affected.
    """
    os.environ["CRUCIBLE_PAGES_TOPIC"] = integration_pages_topic
    os.environ["CRUCIBLE_MUTED_TOPIC"] = integration_muted_topic


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
def integration_arctic_symbols(arctic_library: Any) -> list[str]:
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
    """
    import math
    import random

    import pandas as pd

    symbols = ["INTGA", "INTGB", "INTGC"]
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
    for symbol in symbols:
        price = rng.uniform(20.0, 200.0)
        rows = []
        for _day in days:
            price = max(1.0, price * math.exp(rng.gauss(0.0003, 0.01)))
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
    for symbol in symbols:
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
            f"registered_at: '{INTEGRATION_TRADING_DAY}'",
            "params:",
        ]
        body += [f"  {k}: {v}" for k, v in params.items()]
        body.append(f"notes: integration-tier fixture recipe for {name}")
        (arms / f"{name}.yaml").write_text("\n".join(body) + "\n", encoding="utf-8")
    return root
