"""`arc_runs_ok`/`replays_ok` require of a day only the stages that day's own
release carried (`alpha-engine-config-I10478`).

The defect, restated from the issue: the clause graded PAST days against the
registry in the grader's own process, so merging one new `dispatch: arc` row
turned four Saturdays that had closed before the row existed retroactively
red — no artifact having changed, attributed to days nobody touched, by a
commit that named neither. `replays_ok` went from MET to NOT MET on a merge.

Both directions are asserted here, plus the two refusals that keep the
narrowing from becoming an excuse-collection: a day with NO arc at all is
excused from nothing, and a release that cannot be READ excuses nothing
either. Seeded by hand in the style of `tests/test_gate_fault_exclusion.py`
— a two-component synthetic registry, one row standing for the stage that
existed all along and one for the stage a later release introduced.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import pytest

from crucible.components import Component, Deadline
from crucible.gate import (
    PHASE2_REPLAY_SATURDAYS,
    _arc_jobs_declared_by,
    _ArcRegistryHistory,
    _clause_arc_runs_ok,
    _clause_replays_ok,
    _slots_declared_by,
    weekly_window,
)
from crucible.manifest import manifest_key
from crucible.release import ReleaseRecord, release_json_key, wheel_key_for
from crucible.slots import SLOTS, dispatchable_slots
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)

#: The stage that existed all along, and the one a later release introduced.
OLD_STAGE = "data.weekly"
NEW_STAGE = "iac.conformance"

OLD_RELEASE = "a" * 40
NEW_RELEASE = "b" * 40


def _component(name: str, at: str) -> Component:
    return Component(
        name=name,
        description="x",
        lifecycle="ACTIVE",
        signals={
            "execution": "run.json",
            "cost": None,
            "resource": None,
            "lineage": None,
            "outcome": None,
        },
        log_location="/x",
        log_retention_days=1,
        alert_channel="x",
        console_surface="x",
        artifact_retention="forever",
        schedule="weekly",
        deadline=Deadline.from_yaml(
            {"anchor": "next_calendar_day_at", "cadence": "weekly", "at": at}
        ),
        dispatch="arc",
    )


def _registry() -> dict[str, Component]:
    """TODAY's registry — both stages. This is the tree the grader runs in,
    and the whole point is that it is not what a past day is graded against."""
    return {
        OLD_STAGE: _component(OLD_STAGE, "09:00"),
        NEW_STAGE: _component(NEW_STAGE, "17:15"),
    }


def _components_yaml(*names: str) -> bytes:
    rows = "\n".join(
        f"  {name}:\n    lifecycle: ACTIVE\n    dispatch: arc\n    schedule: weekly, Saturday\n"
        for name in names
    )
    return f"version: 1\ndefaults: {{}}\ncomponents:\n{rows}".encode()


def _publish_release(store: LocalStore, sha: str, *names: str, slots: tuple[str, ...] = ()) -> None:
    """A release whose published wheel declares ``names`` as its arc and
    ``slots`` as the slots it could dispatch.

    Built through `ReleaseRecord` and `wheel_key_for`, not by hand: the
    grader resolves the wheel through `crucible.release`'s own resolver, so
    a fixture that invented the key shape would stop testing the resolution.

    A slot is shipped as its real module path carrying both entry points —
    the same artifact `crucible.slots.dispatchable_slots` would import, which
    is what makes the release-derived reading a reading of that release
    rather than of the tree the test happens to run in.
    """
    wheel_filename = f"crucible-0.1.0+g{sha[:12]}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("crucible/components.yaml", _components_yaml(*names))
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


def _manifest(job: str, day: dt.date, sha: str, *, status: str = "ok") -> bytes:
    return json.dumps(
        {
            "job": job,
            "status": status,
            "reason": "" if status == "ok" else "boom",
            "inputs": [],
            "outputs": [],
            "release_sha": sha,
            "run_id": "0" * 26,
        }
    ).encode()


def _seed_arc(store: LocalStore, day: dt.date, sha: str, *stages: str) -> None:
    """The day's own arc manifest (which names the release) plus one stage
    manifest per stage that actually ran."""
    store.put_bytes(manifest_key("weekly", day.isoformat()), _manifest("weekly", day, sha))
    for stage in stages:
        store.put_bytes(manifest_key(stage, day.isoformat()), _manifest(stage, day, sha))


class TestAStagePredatingItsReleaseIsNotRequired:
    def test_a_stage_the_days_release_did_not_carry_is_not_a_finding(
        self, tmp_path: object
    ) -> None:
        """The regression: `iac.conformance` joins the registry, and a day
        whose release never carried it must not turn red for it."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_STAGE)
        _seed_arc(store, FRIDAY, OLD_RELEASE, OLD_STAGE)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert clause.met and not clause.unmeasurable, clause.detail

    def test_the_narrowing_is_stated_in_the_detail_rather_than_silent(
        self, tmp_path: object
    ) -> None:
        """A requirement the grader narrowed is a fact about the reading. A
        clause that quietly asked for fewer stages reads identically to one
        that graded them and found them present."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_STAGE)
        _seed_arc(store, FRIDAY, OLD_RELEASE, OLD_STAGE)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert "not required" in clause.detail
        assert f"{NEW_STAGE}@{FRIDAY.isoformat()}" in clause.detail

    def test_replays_ok_inherits_it(self, tmp_path: object) -> None:
        """`_clause_replays_ok` re-reads this predicate verbatim — the clause
        the regression was actually reported on. Its window is the five
        replay Saturdays anchored on the render day, so all five are seeded
        on the release that predates the new stage."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_STAGE)
        for day in weekly_window(FRIDAY, PHASE2_REPLAY_SATURDAYS):
            _seed_arc(store, day, OLD_RELEASE, OLD_STAGE)
        clause = _clause_replays_ok(store, [FRIDAY], _registry())
        assert clause.name == "replays_ok"
        assert clause.met and not clause.unmeasurable, clause.detail


class TestTheNarrowingRefusesInEveryOtherDirection:
    """`arc_runs_ok` would be worthless if it could be talked out of a stage.
    Each of these is a case where the answer is "still required"."""

    def test_a_stage_the_days_release_DID_carry_is_still_required(self, tmp_path: object) -> None:
        """The live-arc protection: a release declaring every current stage
        is graded on every current stage, exactly as before."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, NEW_RELEASE, OLD_STAGE, NEW_STAGE)
        _seed_arc(store, FRIDAY, NEW_RELEASE, OLD_STAGE)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert not clause.met
        assert f"{NEW_STAGE}@{FRIDAY.isoformat()}" in clause.detail
        assert "never ran" in clause.detail

    def test_a_day_whose_arc_never_ran_at_all_is_excused_from_nothing(
        self, tmp_path: object
    ) -> None:
        """No arc manifest means no release to consult — and "no arc ran" is
        the finding, not a reason to ask for fewer stages."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_STAGE)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert not clause.met
        assert "2 never ran" in clause.detail
        assert "not required" not in clause.detail

    def test_an_unreadable_release_excuses_nothing_and_says_so(self, tmp_path: object) -> None:
        """A release nobody can read must not be able to clear a clause. The
        miss stands AND the read failure is reported beside it."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed_arc(store, FRIDAY, OLD_RELEASE, OLD_STAGE)  # no release published
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert not clause.met
        assert f"{NEW_STAGE}@{FRIDAY.isoformat()}" in clause.detail
        assert "never published" in clause.detail

    def test_an_arc_manifest_naming_no_release_excuses_nothing(self, tmp_path: object) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        store.put_bytes(
            manifest_key("weekly", FRIDAY.isoformat()),
            json.dumps(
                {
                    "job": "weekly",
                    "status": "ok",
                    "reason": "",
                    "inputs": [],
                    "outputs": [],
                    "run_id": "0" * 26,
                }
            ).encode(),
        )
        store.put_bytes(
            manifest_key(OLD_STAGE, FRIDAY.isoformat()), _manifest(OLD_STAGE, FRIDAY, OLD_RELEASE)
        )
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert not clause.met
        assert "names no `release_sha`" in clause.detail

    def test_a_wheel_carrying_no_registry_excuses_nothing(self, tmp_path: object) -> None:
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        wheel_filename = f"crucible-0.1.0+g{OLD_RELEASE[:12]}-py3-none-any.whl"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("crucible/__init__.py", b"")
        store.put_bytes(wheel_key_for(OLD_RELEASE, wheel_filename), buffer.getvalue())
        store.put_bytes(
            release_json_key(OLD_RELEASE),
            ReleaseRecord(
                schema_version="release.v3",
                sha=OLD_RELEASE,
                lockfile_sha256="0" * 64,
                wheel_sha256="0" * 64,
                wheel_filename=wheel_filename,
            ).to_json(),
        )
        _seed_arc(store, FRIDAY, OLD_RELEASE, OLD_STAGE)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert not clause.met
        assert "carries no readable" in clause.detail
        assert "not required" not in clause.detail

    def test_a_stage_that_FAILED_is_never_narrowed_away(self, tmp_path: object) -> None:
        """The narrowing touches an ABSENT manifest only. A stage that ran and
        failed is a failure whatever any release declared."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_STAGE)
        _seed_arc(store, FRIDAY, OLD_RELEASE)
        store.put_bytes(
            manifest_key(OLD_STAGE, FRIDAY.isoformat()),
            _manifest(OLD_STAGE, FRIDAY, OLD_RELEASE, status="failed"),
        )
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert not clause.met
        assert "1 failed" in clause.detail


class TestWhatAReleaseDeclaresIsReadNarrowly:
    """The registry a release shipped is read for two fields only — a gate
    clause defeated by a since-required field could not consult the history
    it exists for."""

    def test_only_active_arc_rows_count(self) -> None:
        raw = (
            b"version: 1\ncomponents:\n"
            b"  a:\n    lifecycle: ACTIVE\n    dispatch: arc\n"
            b"  b:\n    lifecycle: RETIRED\n    dispatch: arc\n"
            b"  c:\n    lifecycle: ACTIVE\n    dispatch: scheduler\n"
        )
        assert _arc_jobs_declared_by(raw) == frozenset({"a"})

    def test_a_row_carrying_a_field_todays_loader_requires_still_reads(self) -> None:
        """`crucible.components.load_registry` would raise on this row — it
        declares a deadline with no `cadence`. The history reader must not."""
        raw = (
            b"version: 1\ncomponents:\n"
            b"  a:\n    lifecycle: ACTIVE\n    dispatch: arc\n"
            b"    deadline:\n      anchor: next_calendar_day_at\n      at: '09:00'\n"
        )
        assert _arc_jobs_declared_by(raw) == frozenset({"a"})

    def test_a_release_is_read_once_per_sha(self, tmp_path: object) -> None:
        """Two days on one release cost one wheel read, not two."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, OLD_STAGE)
        history = _ArcRegistryHistory(store)
        other = dt.date(2026, 8, 21)
        _seed_arc(store, FRIDAY, OLD_RELEASE, OLD_STAGE)
        _seed_arc(store, other, OLD_RELEASE, OLD_STAGE)
        first, _ = history.declared_on(FRIDAY)
        second, _ = history.declared_on(other)
        assert first is second is not None


class TestTheHistoryIsNotConsultedWhenNothingIsMissing:
    def test_a_complete_arc_reads_no_release_at_all(self, tmp_path: object) -> None:
        """Lazy by construction: the live arcs are complete, so the structural
        fix costs them nothing. Asserted by giving the store no release to
        read — an eager lookup would report it unreadable."""
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed_arc(store, FRIDAY, OLD_RELEASE, OLD_STAGE, NEW_STAGE)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "could not be read" not in clause.detail


@pytest.mark.parametrize("status", ["ok", "failed"])
def test_the_arc_manifests_own_status_does_not_gate_the_lookup(
    tmp_path: object, status: str
) -> None:
    """A failed arc still names the release that ran it, and that is the
    question being asked — an arc that failed at stage one still could not
    have run a stage its release did not carry."""
    store = LocalStore(tmp_path)  # type: ignore[arg-type]
    _publish_release(store, OLD_RELEASE, OLD_STAGE)
    store.put_bytes(
        manifest_key("weekly", FRIDAY.isoformat()),
        _manifest("weekly", FRIDAY, OLD_RELEASE, status=status),
    )
    store.put_bytes(
        manifest_key(OLD_STAGE, FRIDAY.isoformat()), _manifest(OLD_STAGE, FRIDAY, OLD_RELEASE)
    )
    clause = _clause_arc_runs_ok(store, [FRIDAY], _registry())
    assert clause.met, clause.detail


# ---------------------------------------------------------------------------
# The SLOT axis — `alpha-engine-config-I10628`, the other half of `-I10478`.
#
# `crucible.weekly.arc_stages` expands a slot-scoped job into one stage per
# DISPATCHABLE slot, and `dispatchable_slots()` is read from the grader's own
# process. So promoting the M slot at phase 3 would retroactively add
# `experiment.run[m]@<every past day>` to the requirement set of days whose
# release could not have dispatched it — the same class, a second axis, and no
# live instance yet only because `dispatchable_slots()` has not moved since the
# replay Saturdays ran.
#
# The fixture inverts the live situation deliberately: the grading process
# dispatches two slots, and the release under test shipped one.
# ---------------------------------------------------------------------------

SLOT_JOB = "experiment.run"


def _slot_registry() -> dict[str, Component]:
    """TODAY's registry, with a SLOT-SCOPED arc row. `experiment.run` is in
    `crucible.weekly.ARC_SLOT_JOBS`, so `arc_stages` expands it per slot."""
    return {SLOT_JOB: _component(SLOT_JOB, "12:00")}


def _seed_slot_arc(store: LocalStore, day: dt.date, sha: str, *slots: str) -> None:
    store.put_bytes(manifest_key("weekly", day.isoformat()), _manifest("weekly", day, sha))
    for slot in slots:
        store.put_bytes(
            manifest_key(SLOT_JOB, day.isoformat(), discriminator=slot),
            _manifest(SLOT_JOB, day, sha),
        )


@pytest.fixture
def two_live_slots() -> tuple[str, str]:
    """Two slots this process can dispatch, so the test asks a question about
    the grading tree it is actually running in rather than a hypothetical one.

    Read live, never listed: the dispatchable set GROWS (m at phase 3, s after
    it) and these assertions must not need editing when it does. It never
    shrinks below two — `dispatchable_slots` itself raises on an empty set, and
    u and r have both been dispatchable since phase 1 — so a tree with fewer is
    a broken tree and is asserted rather than stepped around.
    """
    live = list(dispatchable_slots())
    assert len(live) >= 2, (
        f"fewer than two dispatchable slots in this tree: {live}. The slot axis cannot "
        "be graded at all without a promoted slot to grade it against"
    )
    return live[0], live[1]


class TestASlotTheDaysReleaseCouldNotDispatchIsNotRequired:
    def test_a_past_day_is_graded_on_its_own_releases_slots(
        self, tmp_path: object, two_live_slots: tuple[str, str]
    ) -> None:
        """The test that would have caught it: a past day on a release with
        one dispatchable slot, graded by a process with two, reads MET."""
        shipped, promoted = two_live_slots
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, SLOT_JOB, slots=(shipped,))
        _seed_slot_arc(store, FRIDAY, OLD_RELEASE, shipped)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _slot_registry())
        assert clause.met and not clause.unmeasurable, clause.detail
        assert f"{SLOT_JOB}[{promoted}]@{FRIDAY.isoformat()}" in clause.detail
        assert "could not dispatch that slot" in clause.detail

    def test_a_slot_the_days_release_COULD_dispatch_is_still_required(
        self, tmp_path: object, two_live_slots: tuple[str, str]
    ) -> None:
        """The constraint the issue states: do NOT weaken the grading of any
        day whose release DID carry the slot."""
        shipped, promoted = two_live_slots
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, NEW_RELEASE, SLOT_JOB, slots=(shipped, promoted))
        _seed_slot_arc(store, FRIDAY, NEW_RELEASE, shipped)
        clause = _clause_arc_runs_ok(store, [FRIDAY], _slot_registry())
        assert not clause.met
        assert f"{SLOT_JOB}[{promoted}]@{FRIDAY.isoformat()}" in clause.detail
        assert "never ran" in clause.detail


class TestTheSlotNarrowingRefusesInEveryOtherDirection:
    def test_a_day_whose_arc_never_ran_is_excused_from_no_slot(
        self, tmp_path: object, two_live_slots: tuple[str, str]
    ) -> None:
        shipped, _ = two_live_slots
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, SLOT_JOB, slots=(shipped,))
        clause = _clause_arc_runs_ok(store, [FRIDAY], _slot_registry())
        assert not clause.met
        assert "never ran" in clause.detail
        assert "not required" not in clause.detail

    def test_an_unreadable_release_excuses_no_slot_and_says_so(
        self, tmp_path: object, two_live_slots: tuple[str, str]
    ) -> None:
        shipped, promoted = two_live_slots
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _seed_slot_arc(store, FRIDAY, OLD_RELEASE, shipped)  # no release published
        clause = _clause_arc_runs_ok(store, [FRIDAY], _slot_registry())
        assert not clause.met
        assert f"{SLOT_JOB}[{promoted}]@{FRIDAY.isoformat()}" in clause.detail
        assert "never published" in clause.detail
        assert "could not dispatch that slot" not in clause.detail

    def test_a_slot_stage_that_FAILED_is_never_narrowed_away(
        self, tmp_path: object, two_live_slots: tuple[str, str]
    ) -> None:
        """The narrowing touches an ABSENT manifest only — on this axis too."""
        shipped, promoted = two_live_slots
        store = LocalStore(tmp_path)  # type: ignore[arg-type]
        _publish_release(store, OLD_RELEASE, SLOT_JOB, slots=(shipped,))
        _seed_slot_arc(store, FRIDAY, OLD_RELEASE, shipped)
        store.put_bytes(
            manifest_key(SLOT_JOB, FRIDAY.isoformat(), discriminator=promoted),
            _manifest(SLOT_JOB, FRIDAY, OLD_RELEASE, status="failed"),
        )
        clause = _clause_arc_runs_ok(store, [FRIDAY], _slot_registry())
        assert not clause.met
        assert "1 failed" in clause.detail


class TestWhatAReleaseDeclaresAboutSlotsIsReadFromItsSource:
    """Parsed, never imported: executing a past release's slot modules inside
    the grader would be a second copy of the package running, not a reading."""

    @staticmethod
    def _wheel(**members: str) -> zipfile.ZipFile:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for path, source in members.items():
                archive.writestr(path, source)
        return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))

    def test_both_entry_points_are_required(self) -> None:
        slot = next(iter(SLOTS))
        module = SLOTS[slot].module
        with self._wheel(**{f"crucible/slots/{module}.py": "def produce():\n    ...\n"}) as archive:
            assert slot not in _slots_declared_by(archive)

    def test_a_module_defining_both_is_dispatchable(self) -> None:
        slot = next(iter(SLOTS))
        module = SLOTS[slot].module
        source = "async def produce():\n    ...\n\n\nasync def grade():\n    ...\n"
        with self._wheel(**{f"crucible/slots/{module}.py": source}) as archive:
            assert slot in _slots_declared_by(archive)

    def test_a_release_predating_the_module_declares_nothing(self) -> None:
        with self._wheel(**{"crucible/__init__.py": ""}) as archive:
            assert _slots_declared_by(archive) == frozenset()

    def test_a_slot_module_that_will_not_parse_raises_rather_than_narrowing(self) -> None:
        """A member that is THERE and unreadable is a fact about our reading,
        and `_ArcRegistryHistory` reports it as unmeasurable. Silently reading
        it as "not dispatchable" would let a corrupt wheel clear a phase."""
        slot = next(iter(SLOTS))
        module = SLOTS[slot].module
        with self._wheel(**{f"crucible/slots/{module}.py": "def produce(:\n"}) as archive:
            with pytest.raises(SyntaxError):
                _slots_declared_by(archive)


def test_one_wheel_read_answers_both_axes(tmp_path: object) -> None:
    """Both axes come out of the same immutable artifact, so they are cached
    together — reading the wheel twice would double the S3 round trips for one
    question about one release."""
    store = LocalStore(tmp_path)  # type: ignore[arg-type]
    _publish_release(store, OLD_RELEASE, OLD_STAGE, slots=(next(iter(SLOTS)),))
    _seed_arc(store, FRIDAY, OLD_RELEASE, OLD_STAGE)
    history = _ArcRegistryHistory(store)
    jobs, jobs_problem = history.declared_on(FRIDAY)
    slots, slots_problem = history.slots_on(FRIDAY)
    assert jobs == frozenset({OLD_STAGE})
    assert slots == frozenset({next(iter(SLOTS))})
    assert jobs_problem is None and slots_problem is None
    assert len(history._by_sha) == 1
