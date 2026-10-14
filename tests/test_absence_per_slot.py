"""`serve.daily` owes one manifest PER SLOT, and absence grades each one.

`alpha-engine-config-I12054`. `crucible.alerts.evaluate_absence` grades
absence by listing `manifest_prefix(job, day)` and used to pass when ANY
manifest sat under it (the I9781 prefix listing). `serve.daily` runs once per
slot under one job name, discriminated by slot — `runs/serve.daily/{day}/m/`,
`/u/`, `/s/` — so the day `serve-daily-s` (or `-u`) never fired, M's manifest
satisfied the row and nothing paged. The S link is the one the daily shadow
books need, and a `dispatch: scheduler` row has no arc stage list to back it
up the way `experiment.run`/`experiment.grade` do.

The slots owed are read off the job's own declaration,
:func:`crucible.slots.daily_servers` (via :func:`crucible.slots.owed_slot_manifests`),
never from a list in `crucible.alerts`.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import evaluate_absence
from crucible.calendar import is_trading_day
from crucible.components import NYSE_TZ, load_registry
from crucible.keys import manifest_key, manifest_prefix
from crucible.slots import daily_servers, owed_slot_manifests
from crucible.slots.strategy import DAILY_SERVE_JOB
from crucible.store import LocalStore

#: A Tuesday; `serve.daily` is due every trading day.
DAY = dt.date(2026, 10, 6)


def _past_every_deadline(trading_day: dt.date) -> dt.datetime:
    """One minute before the NEXT session's close (as `test_absence_cadence`):
    every deadline for ``trading_day`` has passed and it is still the session
    the sweep evaluates."""
    following = trading_day + dt.timedelta(days=1)
    while not is_trading_day(following):
        following += dt.timedelta(days=1)
    return dt.datetime.combine(following, dt.time(15, 59), tzinfo=NYSE_TZ)


def _write_manifest(store: LocalStore, slot: str | None) -> None:
    store.put_bytes(
        manifest_key(DAILY_SERVE_JOB, DAY.isoformat(), discriminator=slot),
        json.dumps({"job": DAILY_SERVE_JOB, "status": "ok"}).encode("utf-8"),
    )


def _pages(store: LocalStore) -> list:
    registry = {DAILY_SERVE_JOB: load_registry()[DAILY_SERVE_JOB]}
    return [
        p
        for p in evaluate_absence(store, now=_past_every_deadline(DAY), registry=registry)
        if p.job == DAILY_SERVE_JOB
    ]


class TestTheOwedSlotsAreTheJobsOwnDeclaration:
    def test_serve_daily_owes_one_manifest_per_daily_server(self) -> None:
        assert owed_slot_manifests(DAILY_SERVE_JOB) == frozenset(daily_servers())
        assert {"m", "u", "s"} <= owed_slot_manifests(DAILY_SERVE_JOB)

    def test_a_job_with_one_manifest_per_day_owes_no_slot(self) -> None:
        assert owed_slot_manifests("data.daily") is None

    def test_the_arc_jobs_keep_the_arcs_own_backstop(self) -> None:
        """`experiment.run`/`experiment.grade` are graded through
        `_arc_declared_members`; their per-slot set is the arc's to declare."""
        assert owed_slot_manifests("experiment.run") is None
        assert owed_slot_manifests("experiment.grade") is None


class TestAMissingSlotPages:
    def test_an_m_only_day_pages_absence_naming_u_and_s(self, tmp_path) -> None:
        """The defect, reproduced: before I12054 M's manifest cleared the row."""
        store = LocalStore(tmp_path)
        _write_manifest(store, "m")
        (page,) = _pages(store)
        assert page.condition == "absence"
        assert page.trading_day == DAY
        assert "slot(s) s, u" in page.reason
        assert manifest_prefix(DAILY_SERVE_JOB, DAY.isoformat()) in page.reason
        assert "delivered: m" in page.reason

    def test_only_the_s_link_missing_pages_naming_s_alone(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_manifest(store, "m")
        _write_manifest(store, "u")
        (page,) = _pages(store)
        assert "slot(s) s " in page.reason
        assert "u," not in page.reason

    def test_every_slot_delivered_pages_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        for slot in sorted(daily_servers()):
            _write_manifest(store, slot)
        assert _pages(store) == []

    def test_a_failed_slot_run_is_not_absent(self, tmp_path) -> None:
        """A failed manifest is the FAILURE condition's to page; absence asks
        only whether the slot's run left a record at all."""
        store = LocalStore(tmp_path)
        for slot in sorted(daily_servers()):
            _write_manifest(store, slot)
        key = manifest_key(DAILY_SERVE_JOB, DAY.isoformat(), discriminator="s")
        store.put_bytes(key, json.dumps({"job": DAILY_SERVE_JOB, "status": "failed"}).encode())
        assert _pages(store) == []

    def test_nothing_delivered_is_one_page_naming_every_slot(self, tmp_path) -> None:
        """One page per job and day, whatever the number of slots missing: the
        grouping and dedup identity stay the job's, and the reason names the
        slots."""
        (page,) = _pages(LocalStore(tmp_path))
        owed = ", ".join(sorted(daily_servers()))
        assert f"slot(s) {owed}" in page.reason
        assert "delivered: none" in page.reason

    def test_an_undiscriminated_manifest_clears_no_slot(self, tmp_path) -> None:
        """A bare `runs/serve.daily/{day}/run.json` names no slot, so it is no
        slot's run and every owed slot still pages."""
        store = LocalStore(tmp_path)
        _write_manifest(store, None)
        (page,) = _pages(store)
        assert f"slot(s) {', '.join(sorted(daily_servers()))}" in page.reason


@pytest.mark.parametrize("job", ["data.daily"])
def test_a_single_manifest_job_is_graded_as_before(tmp_path, job) -> None:
    """The undiscriminated rows keep the prefix-listing rule unchanged."""
    store = LocalStore(tmp_path)
    store.put_bytes(
        manifest_key(job, DAY.isoformat()),
        json.dumps({"job": job, "status": "ok"}).encode("utf-8"),
    )
    registry = {job: load_registry()[job]}
    pages = evaluate_absence(store, now=_past_every_deadline(DAY), registry=registry)
    assert [p for p in pages if p.job == job] == []
