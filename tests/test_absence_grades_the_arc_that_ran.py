"""An arc member is due on the days the arc that RAN declared it — not on
every day before the registry row existed.

**The measured defect (alpha-engine-config-I10711).** `promote` joined the
weekly arc in crucible-PR257, merged 2026-09-13T19:29Z. The arc for trading
day 2026-09-11 had already run, at 2026-09-12T11:01Z under release
4da73b1, whose registry declared `promote` with `dispatch: null`; its manifest
reads "9 of 9 declared arc stages completed" and lists no `promote` input.
`alerts.sweep` then graded 2026-09-11 against the CURRENT registry and paged
`promote: no manifest under runs/promote/2026-09-11/` — an absence for a
stage no dispatcher in force on that day was ever going to start, and one
that would recur for every future row that joins the arc.

The arc's own manifest is the dispatcher's record of what it declared: a
`weekly` run that is `ok` completed every stage it planned (`run_arc` raises
on the first stage that does not), and records each one as an input. So an
`ok` arc naming no manifest for a member did not declare that member. Every
other shape — no arc manifest, a failed arc, an unreadable one — grades the
member exactly as before, because none of them is evidence of what was
declared.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import StoreAccessError, evaluate_absence
from crucible.components import load_registry
from crucible.keys import manifest_key
from crucible.store import LocalStore
from crucible.weekly import ARC_JOB, arc_stages

FRIDAY = dt.date(2026, 9, 11)
#: Monday 2026-09-14 01:02Z — the instant the live page fired; every Saturday
#: deadline for Friday's session is past, and it still resolves to Friday.
PAGED_AT = dt.datetime(2026, 9, 14, 1, 2, 1, tzinfo=dt.UTC)

#: The input keys of the LIVE `runs/weekly/2026-09-11/run.json`, verbatim
#: (read 2026-09-14 as ne-admin). Nine stages, no `promote`.
LIVE_ARC_INPUTS = (
    "runs/data.weekly/2026-09-11/run.json",
    "runs/experiment.run/2026-09-11/u/run.json",
    "runs/experiment.run/2026-09-11/r/run.json",
    "runs/experiment.grade/2026-09-11/u/run.json",
    "runs/experiment.grade/2026-09-11/r/run.json",
    "runs/drift/2026-09-11/run.json",
    "runs/report/2026-09-11/run.json",
    "runs/console/2026-09-11/run.json",
    "runs/iac.conformance/2026-09-11/run.json",
)


def _write(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document).encode())


def _stage_manifests(store: LocalStore, keys=LIVE_ARC_INPUTS) -> None:
    for key in keys:
        _write(store, key, {"status": "ok", "trading_day": FRIDAY.isoformat()})


def _arc(store: LocalStore, *, status: str = "ok", inputs=LIVE_ARC_INPUTS) -> None:
    _write(
        store,
        manifest_key(ARC_JOB, FRIDAY.isoformat()),
        {
            "job": ARC_JOB,
            "status": status,
            "trading_day": FRIDAY.isoformat(),
            "inputs": [{"key": k, "schema_version": "v1", "sha256": "0" * 64} for k in inputs],
        },
    )


def _absent(store: LocalStore) -> set[str]:
    return {p.job for p in evaluate_absence(store, now=PAGED_AT) if p.trading_day == FRIDAY}


class TestTheLiveShape:
    def test_promote_is_not_absent_on_a_day_the_arc_did_not_declare_it(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        _arc(store)
        assert "promote" not in _absent(store)

    def test_every_member_the_arc_did_declare_is_still_satisfied_only_by_its_manifest(
        self, tmp_path
    ) -> None:
        """The narrowing is by declaration, never by the arc's word that a
        stage ran: delete a declared stage's manifest and it pages."""
        store = LocalStore(tmp_path)
        _stage_manifests(store, keys=[k for k in LIVE_ARC_INPUTS if "/report/" not in k])
        _arc(store)
        absent = _absent(store)
        assert "report" in absent
        assert "promote" not in absent


class TestEveryOtherShapeGradesTheMemberAsBefore:
    """The other half. A narrowing that only ever suppressed would be a
    suppression collection (§11.1) wearing a manifest read."""

    def test_no_arc_manifest_means_the_current_declaration_is_graded(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        assert "promote" in _absent(store)

    def test_a_failed_arc_is_no_evidence_of_what_it_declared(self, tmp_path) -> None:
        """A failed arc stopped at a stage; its inputs are the stages that
        finished, not the ones it planned."""
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        _arc(store, status="failed")
        assert "promote" in _absent(store)

    def test_an_unparseable_arc_manifest_is_no_evidence_either(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        store.put_bytes(manifest_key(ARC_JOB, FRIDAY.isoformat()), b"{truncated")
        assert "promote" in _absent(store)

    def test_an_ok_arc_with_no_inputs_list_is_no_evidence_either(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        _write(
            store,
            manifest_key(ARC_JOB, FRIDAY.isoformat()),
            {"job": ARC_JOB, "status": "ok", "trading_day": FRIDAY.isoformat()},
        )
        assert "promote" in _absent(store)

    def test_an_arc_that_declared_promote_and_wrote_none_pages_it(self, tmp_path) -> None:
        """The producer-defect shape this issue asked about: declared by the
        arc in force, and no manifest. Still an absence."""
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        _arc(store, inputs=(*LIVE_ARC_INPUTS, "runs/promote/2026-09-11/u/run.json"))
        assert "promote" in _absent(store)

    def test_a_non_arc_row_is_never_narrowed_by_the_arc_manifest(self, tmp_path) -> None:
        """`data.daily` is started by the scheduler; the arc's inputs say
        nothing about it."""
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        _arc(store)
        assert "data.daily" in _absent(store)


class TestAnUnreadableArcManifestIsAnAccessFault:
    def test_a_read_that_raises_is_carried_out_and_the_member_still_graded(
        self, tmp_path, monkeypatch
    ) -> None:
        store = LocalStore(tmp_path)
        _stage_manifests(store)
        _arc(store)
        arc_key = manifest_key(ARC_JOB, FRIDAY.isoformat())
        real = store.get_bytes

        def denied(key: str) -> bytes:
            if key == arc_key:
                raise PermissionError("AccessDenied")
            return real(key)

        monkeypatch.setattr(store, "get_bytes", denied)
        faults: list[str] = []
        pages = evaluate_absence(store, now=PAGED_AT, access_faults=faults)
        assert "promote" in {p.job for p in pages}
        assert any(arc_key in f for f in faults)
        with pytest.raises(StoreAccessError):
            evaluate_absence(store, now=PAGED_AT)


class TestEveryAbsenceGradedRowHasAStarter:
    """The class check alpha-engine-config-I10711 asked for: a row the
    absence condition grades must be started by something this tree can
    show. `ComponentRow` already refuses a scheduled row with `dispatch:
    null`; this proves the `arc` value is backed by `arc_stages` itself, and
    `nous-ergon-ops/tests/crossrepo/test_crucible_dispatch_lockstep.py`
    proves `scheduler` and `github-actions` against the template and the
    workflows."""

    def test_every_graded_row_names_a_dispatcher(self) -> None:
        for name, component in load_registry().items():
            if component.deadline is None or component.lifecycle != "ACTIVE":
                continue
            assert component.dispatch is not None, name

    def test_every_arc_row_is_a_stage_of_the_arc(self) -> None:
        registry = load_registry()
        declared = {
            name
            for name, c in registry.items()
            if c.dispatch == "arc" and c.lifecycle == "ACTIVE" and c.deadline is not None
        }
        staged = {stage.job for stage in arc_stages(FRIDAY, registry)}
        assert declared == staged
