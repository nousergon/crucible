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
import os
from pathlib import Path
from typing import Any

import pytest

from crucible.calendar import is_trading_day
from crucible.required import require_env
from crucible.store import Store, open_store

__all__ = [
    "INTEGRATION_TRADING_DAY",
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
