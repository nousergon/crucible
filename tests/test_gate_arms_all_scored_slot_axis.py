"""`arms_all_scored` requires a slot's arena cycle of a day only when the
RELEASE that ran that day's arc could dispatch it (`alpha-engine-config-I10628`
extended to the slot axis this clause reads, live regression measured
2026-09-13T17:32Z: `arms_all_scored` read UNMET 5/6 the moment
`dispatchable_slots()` grew to `{u, r, m, s}`, because the clause filtered its
whole window against TODAY's dispatchable set instead of asking, per day,
what that day's own release could have dispatched).

Mirrors `tests/test_gate_registry_history.py`'s job-axis fixtures: a two-slot
synthetic release history, one slot standing for a slot that existed all
along and one for a slot whose entry points landed later. The narrowing is
asserted to apply on the ABSENCE path only, exactly like `_clause_arc_runs_ok`
— a slot the day's release DID carry is still required, a failed/malformed
cycle is never narrowed away, and an unreadable release excuses nothing.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

from crucible.gate import _clause_arms_all_scored
from crucible.keys import arena_cycle_key, arm_register_key
from crucible.manifest import manifest_key
from crucible.release import ReleaseRecord, release_json_key, wheel_key_for
from crucible.slots import SLOTS
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)

#: `u` stands for the slot every release in this test declares; `m` stands
#: for the slot a LATER release grows entry points for. Real slot ids are
#: used (rather than invented ones) because `_slots_declared_by` walks
#: TODAY's `SLOTS` to find each slot's module path.
OLD_SLOT = "u"
NEW_SLOT = "m"

OLD_RELEASE = "c" * 40
NEW_RELEASE = "d" * 40


def _components_yaml() -> bytes:
    # No `dispatch: arc` row needed for these tests -- `arms_all_scored`
    # reads only the SLOT half of `_ReleaseFacts`, never `arc_jobs`. A wheel
    # with no components at all is still valid YAML the release reader can
    # parse.
    return b"version: 1\ndefaults: {}\ncomponents: {}\n"


def _publish_release(store: LocalStore, sha: str, *slots: str) -> None:
    """A release whose published wheel could dispatch exactly ``slots`` —
    the real per-slot module, carrying both `produce` and `grade`, the same
    artifact `crucible.slots.dispatchable_slots` would import.
    """
    wheel_filename = f"crucible-0.1.0+g{sha[:12]}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("crucible/components.yaml", _components_yaml())
        for slot in slots:
            archive.writestr(
                f"crucible/slots/{SLOTS[slot].module}.py",
                "def produce():\n    ...\n\n\ndef grade():\n    ...\n",
            )
    payload = buffer.getvalue()
    store.put_bytes(wheel_key_for(sha, wheel_filename), payload)
    record = ReleaseRecord(
        schema_version="release.v3",
        sha=sha,
        lockfile_sha256="0" * 64,
        wheel_sha256="0" * 64,
        wheel_filename=wheel_filename,
    )
    store.put_bytes(release_json_key(sha), record.to_json())


def _weekly_manifest(sha: str) -> bytes:
    return json.dumps(
        {
            "job": "weekly",
            "status": "ok",
            "reason": "",
            "inputs": [],
            "outputs": [],
            "release_sha": sha,
            "run_id": "0" * 26,
        }
    ).encode()


def _seed_arc_manifest(store: LocalStore, day: dt.date, sha: str) -> None:
    """The day's own arc manifest, which names the release — the only
    artifact `_ArcRegistryHistory` reads to find which release ran the day.
    """
    store.put_bytes(manifest_key("weekly", day.isoformat()), _weekly_manifest(sha))


def _register(slot: str) -> bytes:
    arm_id = f"{slot}:alpha:abc123"
    return (
        json.dumps(
            {
                "kind": "registered",
                "arm_id": arm_id,
                "date": "2026-01-02",
                "reason": "",
                "record": {
                    "arm_id": arm_id,
                    "slot": slot,
                    "name": "alpha",
                    "spec_hash": "abc123",
                    "created_date": "2026-01-02",
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _seed_scored_cycle(store: LocalStore, slot: str, day: dt.date) -> None:
    """A fully-scored, control-included cycle for ``slot``/``day`` — used for
    the OTHER slot in each test so the finding under test is isolated."""
    store.put_bytes(arm_register_key(slot), _register(slot))
    controls = [f"{slot}:{c.arm_id}:c0ffee" for c in SLOTS[slot].control_arms]
    arms = [f"{slot}:alpha:abc123", *controls]
    store.put_bytes(
        arena_cycle_key(slot, day.isoformat()),
        json.dumps({"scored_arms": arms, "active_arms": arms}).encode(),
    )


class TestASlotADaysReleaseDidNotCarryIsNotAFinding:
    def test_the_regression_a_slot_that_only_exists_TODAY_is_not_required_of_a_past_day(
        self, tmp_path: object
    ) -> None:
        """The live regression: `m` grows entry points and joins TODAY's
        `dispatchable_slots()`, and a day whose release never carried it must
        not turn red for a missing arena cycle."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_SLOT)
        _seed_arc_manifest(store, FRIDAY, OLD_RELEASE)
        _seed_scored_cycle(store, OLD_SLOT, FRIDAY)
        # NEW_SLOT's arena cycle is deliberately absent: OLD_RELEASE could
        # not have dispatched it.
        clause = _clause_arms_all_scored(store, [FRIDAY])
        assert clause.met and not clause.unmeasurable, clause.detail

    def test_the_narrowing_is_stated_in_the_detail_rather_than_silent(
        self, tmp_path: object
    ) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_SLOT)
        _seed_arc_manifest(store, FRIDAY, OLD_RELEASE)
        _seed_scored_cycle(store, OLD_SLOT, FRIDAY)
        clause = _clause_arms_all_scored(store, [FRIDAY])
        assert "not required" in clause.detail
        assert f"{NEW_SLOT}@{FRIDAY.isoformat()}" in clause.detail


class TestTheNarrowingRefusesInEveryOtherDirection:
    def test_a_slot_the_days_release_DID_carry_is_still_required(self, tmp_path: object) -> None:
        """The live-arc protection: a release declaring both slots is graded
        on both, exactly as before."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, NEW_RELEASE, OLD_SLOT, NEW_SLOT)
        _seed_arc_manifest(store, FRIDAY, NEW_RELEASE)
        _seed_scored_cycle(store, OLD_SLOT, FRIDAY)
        # NEW_SLOT's arena cycle is absent despite the release carrying it.
        clause = _clause_arms_all_scored(store, [FRIDAY])
        assert not clause.met
        assert f"{NEW_SLOT}@{FRIDAY.isoformat()}: no arena_cycle artifact" in clause.detail
        # NEW_SLOT itself was NOT narrowed away -- it appears once, as the
        # gap, never inside a "not required" note (r/s legitimately ARE
        # narrowed here since NEW_RELEASE declares only OLD_SLOT/NEW_SLOT).
        assert clause.detail.count(f"{NEW_SLOT}@{FRIDAY.isoformat()}") == 1

    def test_a_day_whose_arc_never_ran_at_all_is_excused_from_nothing(
        self, tmp_path: object
    ) -> None:
        """No arc manifest means no release to consult, and "no arc ran" is
        the finding, not a reason to ask for fewer slots."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_SLOT)
        _seed_scored_cycle(store, OLD_SLOT, FRIDAY)
        clause = _clause_arms_all_scored(store, [FRIDAY])
        assert not clause.met
        assert "not required" not in clause.detail
        # Every slot the CLI can run today is asked for -- the fallback that
        # widens, never narrows, a requirement with no release to consult.
        assert "no arena_cycle artifact" in clause.detail

    def test_an_unreadable_release_excuses_nothing_and_says_so(self, tmp_path: object) -> None:
        """A release nobody can read must not be able to clear a clause. The
        miss stands AND the read failure is reported beside it."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed_arc_manifest(store, FRIDAY, OLD_RELEASE)  # no release published
        _seed_scored_cycle(store, OLD_SLOT, FRIDAY)
        clause = _clause_arms_all_scored(store, [FRIDAY])
        assert not clause.met
        assert f"{NEW_SLOT}@{FRIDAY.isoformat()}: no arena_cycle artifact" in clause.detail
        assert "never published" in clause.detail

    def test_a_malformed_cycle_is_never_narrowed_away(self, tmp_path: object) -> None:
        """The narrowing touches an ABSENT cycle only. A cycle that exists
        and is malformed is a real content gap whatever any release
        declared."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_SLOT)
        _seed_arc_manifest(store, FRIDAY, OLD_RELEASE)
        _seed_scored_cycle(store, OLD_SLOT, FRIDAY)
        store.put_bytes(arena_cycle_key(NEW_SLOT, FRIDAY.isoformat()), b"{not json")
        clause = _clause_arms_all_scored(store, [FRIDAY])
        assert not clause.met
        assert "not readable JSON" in clause.detail
        # A PRESENT-but-malformed cycle never reaches the absence-only
        # narrowing at all: NEW_SLOT's key appears only in the malformed
        # reading, never inside a "not required" note.
        assert f"{NEW_SLOT}@{FRIDAY.isoformat()}" not in clause.detail.rsplit("not required", 1)[-1]
