"""An arm is demanded only on days at or after the day its row was FILED.

`alpha-engine-config-I10948`, measured 2026-09-17. Seven arms were appended
to the live U and M registers in one 48-second window; every `registered`
row took its `date` from `record.created_date` — the date the RECIPE
declares — so they read as filed on 2026-07-27, 2026-08-17, 2026-08-24 and
2026-09-14. `arms_all_scored` then demanded them on every day in its window,
including days whose arena cycles ran weeks before the rows existed, and
phase 1 dropped from 5/6 to 4/6 with no way back for five weekly closes.

This is the same class the same clause already fixed one axis over: the SLOT
axis (`alpha-engine-config-I10628`, crucible-PR248) stopped it grading past
days against the present tree "by a commit that names neither". It survived
as "by a REGISTRATION that names neither". Three things are asserted here:

* the PRODUCER stamps the filing day and keeps `created_date` separate;
* the CLAUSE honours it as a third axis, and NAMES every arm/day pair it
  narrows away, whether it reads MET or UNMET;
* an event date that is absent or unparseable is a MALFORMED register
  reading — red and named — never a silent skip and never "assume it was
  always there".
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import pytest
from nousergon_lib.arena import ArmRegister

from crucible.calendar import NonTradingDayKeyError
from crucible.gate import _clause_arms_all_scored
from crucible.keys import arena_cycle_key, arm_register_key
from crucible.manifest import manifest_key
from crucible.migrate import (
    ARM_FILING_CORRECTIONS,
    run_migrate_arm_filed_on,
)
from crucible.release import ReleaseRecord, release_json_key, wheel_key_for
from crucible.slots import SLOTS
from crucible.slots.arms import ArmSpec, register_arms
from crucible.store import LocalStore

#: Two consecutive weekly closes. The arm under test is filed on the LATER
#: one, so the earlier one is the day it must not be demanded on. Fixed
#: literals: a window that moves with the clock stops testing the same thing.
EARLY_FRIDAY = dt.date(2026, 8, 21)
LATE_FRIDAY = dt.date(2026, 8, 28)

SLOT = "u"
RELEASE = "c" * 40


# ---------------------------------------------------------------------------
# the producer
# ---------------------------------------------------------------------------


def _spec(name: str, registered_at: str) -> ArmSpec:
    return ArmSpec(
        name=name,
        slot=SLOT,
        ranker="momentum",
        params={"lookback": 21, "name": name},
        registered_at=registered_at,
        notes=f"fixture arm {name}",
    )


class TestTheProducerDatesTheEventByTheFilingDay:
    def test_the_event_date_is_the_filing_day_and_the_record_keeps_the_recipes_date(
        self,
    ) -> None:
        """The deliverable-1 assertion: the two dates must not be copied from
        each other. A failure here is the live defect returning."""
        register, _ = register_arms(
            ArmRegister(), [_spec("late_filer", "2026-06-01")], filed_on="2026-08-28"
        )
        event = register.events[-1]
        assert event.date == "2026-08-28"
        assert event.record is not None
        assert event.record.created_date == "2026-06-01"
        # The out-of-sample clock every grace/promotion rung counts from is
        # still the RECIPE's date, unchanged by this fix.
        assert register.state(event.arm_id).record.created_date == "2026-06-01"

    def test_the_filing_day_is_required_not_defaulted(self) -> None:
        """The library accepts its omission with a DeprecationWarning for the
        v1 call sites. This wrapper is the fleet's producer and refuses: every
        caller here has a `ctx.trading_day` in scope, and a default would
        silently reinstate `created_date` as the event date."""
        with pytest.raises(TypeError):
            register_arms(ArmRegister(), [_spec("no_day", "2026-06-01")])  # type: ignore[call-arg]

    def test_a_filing_day_that_is_not_a_session_is_refused(self) -> None:
        """Rule 3: every key, manifest field and window binds to a session.
        A filing day that is not one makes every `filed_on <= day` comparison
        in the clause a comparison against a day the market never traded."""
        with pytest.raises(NonTradingDayKeyError):
            register_arms(ArmRegister(), [_spec("saturday", "2026-06-01")], filed_on="2026-08-29")


# ---------------------------------------------------------------------------
# the clause
# ---------------------------------------------------------------------------


def _publish_release(store: LocalStore, sha: str, *slots: str) -> None:
    wheel_filename = f"crucible-0.1.0+g{sha[:12]}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("crucible/components.yaml", b"version: 1\ndefaults: {}\ncomponents: {}\n")
        for slot in slots:
            archive.writestr(
                f"crucible/slots/{SLOTS[slot].module}.py",
                "def produce():\n    ...\n\n\ndef grade():\n    ...\n",
            )
    payload = buffer.getvalue()
    store.put_bytes(wheel_key_for(sha, wheel_filename), payload)
    store.put_bytes(
        release_json_key(sha),
        ReleaseRecord(
            schema_version="release.v3",
            sha=sha,
            lockfile_sha256="0" * 64,
            wheel_sha256="0" * 64,
            wheel_filename=wheel_filename,
        ).to_json(),
    )


def _seed_arc_manifest(store: LocalStore, day: dt.date, sha: str) -> None:
    store.put_bytes(
        manifest_key("weekly", day.isoformat()),
        json.dumps(
            {
                "job": "weekly",
                "status": "ok",
                "reason": "",
                "inputs": [],
                "outputs": [],
                "release_sha": sha,
                "run_id": "0" * 26,
            }
        ).encode(),
    )


def _row(name: str, *, filed_on: str, created_date: str) -> dict:
    arm_id = f"{SLOT}:{name}:abc123"
    return {
        "kind": "registered",
        "arm_id": arm_id,
        "date": filed_on,
        "reason": "",
        "record": {
            "arm_id": arm_id,
            "slot": SLOT,
            "name": name,
            "spec_hash": "abc123",
            "created_date": created_date,
        },
    }


def _put_register(store: LocalStore, rows: list[dict]) -> None:
    store.put_bytes(
        arm_register_key(SLOT),
        ("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode(),
    )


def _put_cycle(store: LocalStore, day: dt.date, arm_ids: list[str]) -> None:
    controls = [f"{SLOT}:{c.arm_id}:c0ffee" for c in SLOTS[SLOT].control_arms]
    arms = [*arm_ids, *controls]
    store.put_bytes(
        arena_cycle_key(SLOT, day.isoformat()),
        json.dumps({"scored_arms": arms, "active_arms": arms}).encode(),
    )


def _window_store(tmp_path) -> LocalStore:
    """A two-day window whose release declares only U, so the SLOT axis
    narrows the other three away and the ARM axis is what is under test."""
    store = LocalStore(tmp_path)
    _publish_release(store, RELEASE, SLOT)
    for day in (EARLY_FRIDAY, LATE_FRIDAY):
        _seed_arc_manifest(store, day, RELEASE)
    return store


INCUMBENT = f"{SLOT}:incumbent:abc123"
LATECOMER = f"{SLOT}:latecomer:abc123"


class TestAnArmIsNotDemandedBeforeItWasFiled:
    def test_the_regression_an_arm_filed_today_is_not_demanded_on_a_past_day(
        self, tmp_path
    ) -> None:
        """The live regression, reduced: an arm whose row was appended on
        2026-08-28 must not be demanded of 2026-08-21's cycle, which ran
        before the row existed and is a historical artifact that is not
        recomputed."""
        store = _window_store(tmp_path)
        _put_register(
            store,
            [
                _row("incumbent", filed_on="2026-01-02", created_date="2026-01-02"),
                # The defect's shape: created MONTHS before it was filed.
                _row("latecomer", filed_on="2026-08-28", created_date="2026-06-01"),
            ],
        )
        _put_cycle(store, EARLY_FRIDAY, [INCUMBENT])
        _put_cycle(store, LATE_FRIDAY, [INCUMBENT, LATECOMER])
        clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
        assert clause.met and not clause.unmeasurable, clause.detail

    def test_the_narrowing_is_named_even_when_the_clause_reads_MET(self, tmp_path) -> None:
        """`_unregistered_note`'s contract: a requirement the grader narrowed
        is a fact about the reading, never housekeeping."""
        store = _window_store(tmp_path)
        _put_register(
            store,
            [
                _row("incumbent", filed_on="2026-01-02", created_date="2026-01-02"),
                _row("latecomer", filed_on="2026-08-28", created_date="2026-06-01"),
            ],
        )
        _put_cycle(store, EARLY_FRIDAY, [INCUMBENT])
        _put_cycle(store, LATE_FRIDAY, [INCUMBENT, LATECOMER])
        clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
        assert clause.met
        assert "not required" in clause.detail
        assert "not in the slot's register on that day" in clause.detail
        assert f"{SLOT}:{LATECOMER}@{EARLY_FRIDAY.isoformat()}" in clause.detail
        assert "(filed 2026-08-28)" in clause.detail

    def test_the_narrowing_is_named_when_the_clause_reads_UNMET_too(self, tmp_path) -> None:
        """A clause that is red for an unrelated reason must still say which
        arm/day pairs it stopped asking for — otherwise the reading is a
        different reading than it appears to be."""
        store = _window_store(tmp_path)
        _put_register(
            store,
            [
                _row("incumbent", filed_on="2026-01-02", created_date="2026-01-02"),
                _row("latecomer", filed_on="2026-08-28", created_date="2026-06-01"),
            ],
        )
        _put_cycle(store, EARLY_FRIDAY, [INCUMBENT])
        # The latecomer IS demanded here and was not scored: a real gap.
        _put_cycle(store, LATE_FRIDAY, [INCUMBENT])
        clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
        assert not clause.met
        assert f"{SLOT}@{LATE_FRIDAY.isoformat()}" in clause.detail
        assert "not in the slot's register on that day" in clause.detail
        assert f"{SLOT}:{LATECOMER}@{EARLY_FRIDAY.isoformat()}" in clause.detail

    def test_an_arm_filed_ON_the_day_is_still_demanded(self, tmp_path) -> None:
        """The boundary is inclusive: the arc that registers an arm scores it
        in the same arc, so the day of filing is a day it must appear on."""
        store = _window_store(tmp_path)
        _put_register(store, [_row("latecomer", filed_on="2026-08-28", created_date="2026-06-01")])
        _put_cycle(store, EARLY_FRIDAY, [])
        _put_cycle(store, LATE_FRIDAY, [])
        clause = _clause_arms_all_scored(store, [LATE_FRIDAY])
        assert not clause.met
        assert f"{SLOT}@{LATE_FRIDAY.isoformat()}" in clause.detail
        assert LATECOMER in clause.detail

    def test_an_arm_filed_BEFORE_the_window_is_demanded_on_every_day(self, tmp_path) -> None:
        """The narrowing must not leak: nothing about this fix changes the
        reading of an arm that was in the register all along."""
        store = _window_store(tmp_path)
        _put_register(store, [_row("incumbent", filed_on="2026-01-02", created_date="2026-01-02")])
        _put_cycle(store, EARLY_FRIDAY, [])
        _put_cycle(store, LATE_FRIDAY, [INCUMBENT])
        clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
        assert not clause.met
        assert f"{SLOT}@{EARLY_FRIDAY.isoformat()}" in clause.detail
        assert INCUMBENT in clause.detail
        # The ARM axis narrowed nothing (the SLOT axis legitimately narrows
        # r/m/s away here, which is a different note with a different reason).
        assert "not in the slot's register on that day" not in clause.detail


class TestAnUnusableFilingDayIsMalformedNotExcused:
    @pytest.mark.parametrize("bad", ["", "2026-8-28", "not-a-date", None, 20260828])
    def test_a_row_whose_event_date_cannot_be_read_is_red_and_named(self, tmp_path, bad) -> None:
        """Never a silent skip and never "assume it was always there". Either
        would restore the defect for exactly the rows whose provenance is
        already broken — the second demands the arm on every day again, the
        first hides a genuinely unscored arm."""
        store = _window_store(tmp_path)
        row = _row("latecomer", filed_on="2026-08-28", created_date="2026-06-01")
        row["date"] = bad
        _put_register(store, [row])
        _put_cycle(store, EARLY_FRIDAY, [LATECOMER])
        _put_cycle(store, LATE_FRIDAY, [LATECOMER])
        clause = _clause_arms_all_scored(store, [EARLY_FRIDAY, LATE_FRIDAY])
        assert not clause.met
        # A CONTENT gap: the register was read fine, the row is malformed.
        assert not clause.unmeasurable
        assert LATECOMER in clause.detail
        assert arm_register_key(SLOT) in clause.detail


# ---------------------------------------------------------------------------
# the one-time correction
# ---------------------------------------------------------------------------


def _seed_mis_stamped_registers(store: LocalStore) -> dict[str, list[dict]]:
    """The live register shape as measured 2026-09-17: every `registered`
    row carrying `date == record.created_date`."""
    by_slot: dict[str, list[dict]] = {}
    for correction in ARM_FILING_CORRECTIONS:
        _slot, name, spec_hash = correction.arm_id.split(":")
        by_slot.setdefault(correction.slot, []).append(
            {
                "kind": "registered",
                "arm_id": correction.arm_id,
                "date": correction.wrong_date,
                "reason": "",
                "record": {
                    "arm_id": correction.arm_id,
                    "slot": correction.slot,
                    "name": name,
                    "spec_hash": spec_hash,
                    "created_date": correction.wrong_date,
                    "supersedes": None,
                    "bootstrap": False,
                    "notes": "measured fixture",
                    "control": False,
                },
            }
        )
    for slot, rows in by_slot.items():
        store.put_bytes(
            arm_register_key(slot),
            ("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode(),
        )
    return by_slot


def _rows(store: LocalStore, slot: str) -> list[dict]:
    return [
        json.loads(line)
        for line in store.get_bytes(arm_register_key(slot)).decode().splitlines()
        if line.strip()
    ]


class TestTheOneTimeCorrection:
    def test_the_declared_set_is_the_seven_rows_measured_on_2026_09_17(self) -> None:
        """The table is the audit. Four U ports and three M heads, each with
        the S3 object version that introduced it — the only durable record of
        the true filing day, since the appends left no per-arm manifest."""
        assert len(ARM_FILING_CORRECTIONS) == 7
        assert sorted({c.slot for c in ARM_FILING_CORRECTIONS}) == ["m", "u"]
        assert {c.filed_on for c in ARM_FILING_CORRECTIONS} == {"2026-09-17"}
        assert all(
            c.evidence_version_id and c.evidence_last_modified for c in ARM_FILING_CORRECTIONS
        )
        assert len({c.arm_id for c in ARM_FILING_CORRECTIONS}) == 7

    def test_a_dry_run_reports_every_row_and_writes_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        before = _seed_mis_stamped_registers(store)
        report = run_migrate_arm_filed_on(store, dry_run=True)
        assert report.dry_run
        assert len(report.corrected) == 7
        assert not report.refused
        for slot, rows in before.items():
            assert _rows(store, slot) == rows

    def test_it_changes_the_event_date_and_nothing_else(self, tmp_path) -> None:
        """Date-only, and that is asserted field by field: no `arm_id`, no
        `spec_hash`, no `record`, no `created_date`."""
        store = LocalStore(tmp_path)
        before = _seed_mis_stamped_registers(store)
        report = run_migrate_arm_filed_on(store)
        assert len(report.corrected) == 7
        assert not report.refused
        for slot, old_rows in before.items():
            new_rows = _rows(store, slot)
            assert len(new_rows) == len(old_rows)
            for old, new in zip(old_rows, new_rows, strict=True):
                assert new["date"] == "2026-09-17"
                assert new["arm_id"] == old["arm_id"]
                assert new["kind"] == old["kind"]
                assert new["reason"] == old["reason"]
                assert new["record"] == old["record"]
                assert new["record"]["created_date"] == old["record"]["created_date"]

    def test_it_files_its_own_migration_report(self, tmp_path) -> None:
        """The durable record of the repair — the same artifact
        `migrate.code_sha` and `migrate.history` file."""
        store = LocalStore(tmp_path)
        _seed_mis_stamped_registers(store)
        report = run_migrate_arm_filed_on(store)
        keys = [k for k in store.list_keys("migrations/") if k.endswith(".json")]
        assert len(keys) == 1
        filed = json.loads(store.get_bytes(keys[0]).decode())
        assert filed["schema_version"] == "migrate_arm_filed_on.v1"
        assert filed["counts"] == {"corrected": 7, "refused": 0}
        assert filed["migration_run_id"] == report.migration_run_id
        assert {row["arm_id"] for row in filed["corrected"]} == {
            c.arm_id for c in ARM_FILING_CORRECTIONS
        }
        assert all(row["evidence_version_id"] for row in filed["corrected"])

    def test_a_rerun_corrects_nothing_and_says_why(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _seed_mis_stamped_registers(store)
        run_migrate_arm_filed_on(store)
        again = run_migrate_arm_filed_on(store)
        assert not again.corrected
        assert len(again.refused) == 7
        assert all("already carries" in row["reason"] for row in again.refused)

    def test_a_row_carrying_an_unexpected_date_is_refused_not_coerced(self, tmp_path) -> None:
        """The repair rewrites a value this session MEASURED. A row someone
        else has since changed is reported, never overwritten."""
        store = LocalStore(tmp_path)
        by_slot = _seed_mis_stamped_registers(store)
        rows = by_slot["u"]
        rows[0]["date"] = "2026-09-01"
        store.put_bytes(
            arm_register_key("u"),
            ("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode(),
        )
        report = run_migrate_arm_filed_on(store)
        refused = [row for row in report.refused if row["arm_id"] == rows[0]["arm_id"]]
        assert len(refused) == 1
        assert "refusing to overwrite" in refused[0]["reason"]
        assert _rows(store, "u")[0]["date"] == "2026-09-01"

    def test_a_missing_register_refuses_its_rows_rather_than_creating_one(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        report = run_migrate_arm_filed_on(store)
        assert not report.corrected
        assert len(report.refused) == 7
        assert all("does not exist" in row["reason"] for row in report.refused)

    def test_every_declared_row_is_accounted_for_in_exactly_one_list(self, tmp_path) -> None:
        """Exhaustive by construction: the report raises rather than returning
        a count that does not add up."""
        store = LocalStore(tmp_path)
        _seed_mis_stamped_registers(store)
        report = run_migrate_arm_filed_on(store, dry_run=True)
        seen = [row["arm_id"] for row in report.corrected] + [
            row["arm_id"] for row in report.refused
        ]
        assert sorted(seen) == sorted(c.arm_id for c in ARM_FILING_CORRECTIONS)
