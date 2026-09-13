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
from crucible.keys import runs_prefix
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


def _seed(store: LocalStore, day: dt.date, *, status: str = "ok", reason: str = "") -> str:
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
        assert runs_prefix(INTEGRATION_JOB) in clause.detail

    def test_a_listing_we_could_not_take_is_unmeasurable(self, tmp_path: object) -> None:
        store = _UnlistableStore(tmp_path)  # type: ignore[arg-type]
        clause = _clause_integration_tier_current(store, _window())
        assert clause.unmeasurable and not clause.met
        assert "AccessDenied" in clause.detail

    def test_manifests_none_of_which_parse_is_unmeasurable(self, tmp_path: object) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        store.put_bytes(manifest_key(INTEGRATION_JOB, _window()[-1].isoformat()), b"not json")
        clause = _clause_integration_tier_current(store, _window())
        assert clause.unmeasurable and not clause.met, clause.detail

    def test_a_malformed_status_is_a_content_finding_not_unmeasurable(
        self, tmp_path: object
    ) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        key = manifest_key(INTEGRATION_JOB, _window()[-1].isoformat())
        store.put_bytes(key, json.dumps({"job": INTEGRATION_JOB, "status": "degraded"}).encode())
        clause = _clause_integration_tier_current(store, _window())
        assert not clause.met and not clause.unmeasurable, clause.detail
        assert key in clause.detail


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
