"""The phase gate reads the integration tier's most recent result
(`alpha-engine-config-I10460`, from `-I10419`).

The tier (`tests/integration`, run nightly against the real store and the real
ArcticDB, filed as the registered `test.integration` job by
`alpha-engine-config-I10459`) was the one reading no gate could see. It could
have failed every night for a month without a phase gate being a shade
different, which is the detection blindness the clause closes.

Four readings are asserted here and they are deliberately four, not two:
a fresh `ok` is MET; a fresh non-`ok` is UNMET; a reading older than the gate's
own window is UNMET however green it was; and NO reading at all is UNMET —
never UNMEASURABLE, because a prefix we listed successfully and found empty is
an answer about the system. Only a failed READING (a listing we could not take,
manifests we could not parse) is UNMEASURABLE.

**`alpha-engine-config-I10706`.** The tier writes through
`CRUCIBLE_INTEGRATION_STORE_URI`, the production root plus
`crucible.keys.INTEGRATION_STORE_SUBPREFIX` — never the bare production
prefix. `_seed` below writes where the real writer writes; a dedicated
mutation test below shows a manifest at the OLD bare-prefix location does
not satisfy this clause.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.gate import (
    INTEGRATION_JOB,
    _clause_integration_tier_current,
    weekly_window,
)
from crucible.keys import INTEGRATION_STORE_SUBPREFIX, runs_prefix
from crucible.manifest import manifest_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)

#: Phase 3's window width, which is the gate this clause is registered on.
WEEKS = 4


def _window() -> list[dt.date]:
    return weekly_window(FRIDAY, WEEKS)


def _manifest(day: dt.date, *, status: str = "ok", reason: str = "") -> bytes:
    return json.dumps(
        {
            "job": INTEGRATION_JOB,
            "status": status,
            "reason": reason,
            "trading_day": day.isoformat(),
            "inputs": [],
            "outputs": [],
            "run_id": "0" * 26,
        }
    ).encode()


def _dedicated_key(day: dt.date, *, discriminator: str | None = None) -> str:
    """Where the real `test.integration` writer files a manifest: the
    dedicated integration store's sub-prefix of the production store."""
    key = manifest_key(INTEGRATION_JOB, day.isoformat(), discriminator=discriminator)
    return f"{INTEGRATION_STORE_SUBPREFIX}{key}"


def _seed(store: LocalStore, day: dt.date, *, status: str = "ok", reason: str = "") -> str:
    """Write a manifest exactly where the real integration-tier writer does:
    under the dedicated sub-prefix of the production store this clause reads."""
    key = _dedicated_key(day)
    store.put_bytes(key, _manifest(day, status=status, reason=reason))
    return key


def _seed_bare_production_prefix(
    store: LocalStore, day: dt.date, *, status: str = "ok", reason: str = ""
) -> str:
    """Write a manifest at the OLD, WRONG location — the bare production
    `runs/test.integration/...` prefix the tier never writes to. Used only to
    prove this location no longer satisfies the clause."""
    key = manifest_key(INTEGRATION_JOB, day.isoformat())
    store.put_bytes(key, _manifest(day, status=status, reason=reason))
    return key


class _UnlistableStore(LocalStore):
    """A store whose listing is DENIED. Not an empty store: the whole point of
    the clause's absent branch is that those two must not collapse."""

    def list_keys(self, prefix: str) -> list[str]:
        raise PermissionError("AccessDenied")


class TestAFreshOkReadingIsTheOnlyThingThatPasses:
    def test_an_ok_reading_inside_the_window_is_met(self, tmp_path: object) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        key = _seed(store, _window()[-1])
        clause = _clause_integration_tier_current(store, _window())
        assert clause.met and not clause.unmeasurable, clause.detail
        assert key in clause.evidence

    def test_a_failed_reading_is_unmet_and_names_the_reason(self, tmp_path: object) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed(store, _window()[-1], status="failed", reason="ArcticDB read timed out")
        clause = _clause_integration_tier_current(store, _window())
        assert not clause.met and not clause.unmeasurable
        assert "ArcticDB read timed out" in clause.detail

    def test_the_MOST_RECENT_reading_is_the_one_graded(self, tmp_path: object) -> None:
        """A tier that passed once and has failed every night since must not
        read MET off the pass. The last reading is the reading."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed(store, _window()[0])
        latest = _seed(store, _window()[-1], status="failed", reason="boom")
        clause = _clause_integration_tier_current(store, _window())
        assert not clause.met, clause.detail
        assert latest in clause.detail


class TestStalenessIsThisGatesOwnWindow:
    def test_a_green_reading_older_than_the_window_is_unmet(self, tmp_path: object) -> None:
        """The staleness rule, and the whole reason `-I10419` asked for the
        clause: "the phase gate reads the most recent integration result and
        refuses a stale one"."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        stale_day = _window()[0] - dt.timedelta(weeks=1)
        _seed(store, stale_day)
        clause = _clause_integration_tier_current(store, _window())
        assert not clause.met and not clause.unmeasurable, clause.detail
        assert stale_day.isoformat() in clause.detail
        assert _window()[0].isoformat() in clause.detail

    def test_the_window_edge_itself_is_inside(self, tmp_path: object) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed(store, _window()[0])
        clause = _clause_integration_tier_current(store, _window())
        assert clause.met, clause.detail


class TestAbsenceIsUnmetAndUnreadabilityIsUnmeasurable:
    def test_no_reading_at_all_is_unmet_never_unmeasurable(self, tmp_path: object) -> None:
        """`Clause.unmeasurable` is a fact about OUR READING. We listed the
        prefix and it is empty; that is a finding about the tier."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        clause = _clause_integration_tier_current(store, _window())
        assert not clause.met
        assert not clause.unmeasurable, clause.detail
        assert f"{INTEGRATION_STORE_SUBPREFIX}{runs_prefix(INTEGRATION_JOB)}" in clause.detail

    def test_a_listing_we_could_not_take_is_unmeasurable(self, tmp_path: object) -> None:
        store = _UnlistableStore(tmp_path)  # type: ignore[arg-type]
        clause = _clause_integration_tier_current(store, _window())
        assert clause.unmeasurable and not clause.met
        assert "AccessDenied" in clause.detail

    def test_manifests_none_of_which_parse_is_unmeasurable(self, tmp_path: object) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        store.put_bytes(_dedicated_key(_window()[-1]), b"not json")
        clause = _clause_integration_tier_current(store, _window())
        assert clause.unmeasurable and not clause.met, clause.detail

    def test_a_malformed_status_is_a_content_finding_not_unmeasurable(
        self, tmp_path: object
    ) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        key = _dedicated_key(_window()[-1])
        store.put_bytes(key, json.dumps({"job": INTEGRATION_JOB, "status": "degraded"}).encode())
        clause = _clause_integration_tier_current(store, _window())
        assert not clause.met and not clause.unmeasurable, clause.detail
        assert key in clause.detail


class TestTheDedicatedSubprefixIsTheOnlyLocationThatCounts:
    """`alpha-engine-config-I10706`. The tier writes under
    `INTEGRATION_STORE_SUBPREFIX` of the production store; a manifest at the
    old bare production prefix was never written by the real tier and must
    not satisfy this clause — the mutation this fix exists to catch."""

    def test_a_manifest_at_the_old_bare_prefix_does_not_satisfy_the_clause(
        self, tmp_path: object
    ) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed_bare_production_prefix(store, _window()[-1])
        clause = _clause_integration_tier_current(store, _window())
        assert not clause.met
        assert not clause.unmeasurable, clause.detail
        assert f"{INTEGRATION_STORE_SUBPREFIX}{runs_prefix(INTEGRATION_JOB)}" in clause.detail

    def test_the_dedicated_prefix_reading_ignores_a_bare_prefix_decoy(
        self, tmp_path: object
    ) -> None:
        """A manifest at BOTH locations reads MET off the dedicated one only —
        the bare-prefix decoy is not evidence and is not consulted."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed_bare_production_prefix(store, _window()[-1], status="failed", reason="decoy")
        key = _seed(store, _window()[-1])
        clause = _clause_integration_tier_current(store, _window())
        assert clause.met, clause.detail
        assert key in clause.evidence
        assert "decoy" not in clause.detail


def test_the_dedicated_subprefix_constant_matches_the_workflows_variable_shape() -> None:
    """`alpha-engine-config-I10706`. `CRUCIBLE_INTEGRATION_STORE_URI`
    (`.github/workflows/integration-nightly.yml`) is the production store's
    root plus this sub-prefix — `tests/integration/conftest.py::
    integration_store_uri` independently refuses any URI whose path does not
    carry an `integration` segment, since this repo forbids naming the
    production prefix literally to compare against. Both guards describe the
    SAME shape; this asserts `crucible.keys.INTEGRATION_STORE_SUBPREFIX` is
    that shape, so a future change to the separator has one place to change
    rather than two independently-derived checks silently drifting apart."""
    production_root = "s3://EXAMPLE-BUCKET/crucible"
    dedicated_uri = f"{production_root}/{INTEGRATION_STORE_SUBPREFIX}".rstrip("/")
    assert dedicated_uri.endswith(f"/{INTEGRATION_STORE_SUBPREFIX.rstrip('/')}")
    segments = dedicated_uri.replace("s3://", "").strip("/").split("/")
    assert "integration" in segments, segments
    assert INTEGRATION_STORE_SUBPREFIX.endswith("/"), (
        "must be a prefix (trailing slash), not a bare segment"
    )


@pytest.mark.parametrize("status", ["ok", "failed"])
def test_the_clause_never_raises_into_the_ladder(tmp_path: object, status: str) -> None:
    """Every clause is wrapped by `_contain_clause_exceptions`; asserted here
    against a store whose reads raise, because a clause that raised would take
    every other phase row down with it (`alpha-engine-config-I10328`)."""

    class _Exploding(LocalStore):
        def get_bytes(self, key: str) -> bytes:
            raise RuntimeError(status)

    store = _Exploding(tmp_path)  # type: ignore[arg-type]
    _seed(store, _window()[-1])
    clause = _clause_integration_tier_current(store, _window())
    assert not clause.met
