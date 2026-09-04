"""Every key `gate.py` reads is a key some real writer actually writes.

Normative source: `alpha-engine-config-I9781`'s closing review. The defect
this file exists for: `crucible/gate.py:192` computed the BARE
`manifest_key` while `crucible/runner.py:530` (via `experiment.run` /
`experiment.grade`'s `--slot` discriminator) wrote the discriminated one —
four slots, one job, one trading day, and the gate read the wrong key for
all four. The full suite was green over it (`tests/test_gate.py`'s own
`_seed_met` seeded the SAME wrong bare key the gate read, so reader and
seed agreed with each other and both disagreed with the real writer). That
is the class defect: a reader and a writer drifting apart with nothing
between them that could notice.

This module is the something between them. It never hand-computes an
expected key from `stage.slot`, `ARC_SLOT_JOBS` or any other reader-side
fact — every key on both sides of the comparison comes from actually
running the real writer (`crucible.cli.HANDLERS[job]`, through `run_job`)
and the real reader (`crucible.gate._clause_arc_runs_ok`, via `evaluate`)
against one `LocalStore`. A future PR that changes either side without
changing the other fails this test, not a copy of the old formula.
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

import crucible.track_a as track_a
from crucible.cli import HANDLERS
from crucible.components import load_registry
from crucible.gate import _clause_arc_runs_ok
from crucible.manifest import manifest_key
from crucible.slots import SLOTS, dispatchable_slots
from crucible.store import LocalStore
from crucible.weekly import ARC_SLOT_JOBS, arc_stages

FRIDAY = dt.date(2026, 8, 28)


class _StubSlotModule:
    """Stands in for `crucible.slots.{universe,research,...}` so the real
    `experiment.run`/`experiment.grade` handlers can run end to end (through
    the real `run_job`, writing a real manifest) without touching Arctic or
    the arena engine — neither of which this contract needs."""

    def produce(self, ctx: Any, *, settings: Any, arm_name: str | None) -> None:
        ctx.record_rows(rows_in=0, rows_out=0)

    def grade(self, ctx: Any, *, settings: Any) -> dict[str, Any]:
        ctx.record_rows(rows_in=0, rows_out=0)
        return {}


def _experiment_args(job: str, slot: str, store_uri: str) -> argparse.Namespace:
    return argparse.Namespace(
        job=job,
        slot=slot,
        arm=None,
        dry_run=False,
        store=store_uri,
        strategy_dir=None,
        trading_day=FRIDAY,
        date=FRIDAY.isoformat(),
    )


def _seed_non_slot_stages(store: LocalStore, registry: dict[str, Any]) -> None:
    """The other four arc stages (`data.weekly`, `report`, `drift`,
    `console`) write at most one manifest per trading day — undisputed by
    the review (`crucible/report.py`, `release.py`, `deploy.py` call sites
    all take no discriminator) and not the shape this contract is about.
    Seeded directly so `_clause_arc_runs_ok`'s full-arc requirement can be
    satisfied while the real writers under test are only the slot jobs."""
    for stage in arc_stages(FRIDAY, registry):
        if stage.slot is not None:
            continue
        store.put_bytes(
            manifest_key(stage.job, FRIDAY.isoformat()),
            b'{"status": "ok", "reason": "", "inputs": [], "outputs": [], '
            b'"release_sha": "' + b"a" * 40 + b'"}',
        )


class TestGateReadsWhatTheRealWriterWrote:
    def test_every_arc_slot_job_writes_where_the_gate_reads_it(self, tmp_path, monkeypatch) -> None:
        """Runs the REAL `experiment.run`/`experiment.grade` handlers — the
        real `run_job`, the real `manifest_key` write — for every slot, then
        asks the real `_clause_arc_runs_ok` to read the same store. Only the
        business logic inside each slot module is stubbed; the manifest
        write path and the gate's read path are both exercised unmodified."""
        assert ARC_SLOT_JOBS == frozenset({"experiment.run", "experiment.grade"}), (
            "this test enumerates ARC_SLOT_JOBS explicitly below — a new "
            "slot-scoped job needs a writer added here too, or this test "
            "would silently stop covering the class it exists for"
        )
        monkeypatch.setattr(track_a, "_slot_module", lambda slot: _StubSlotModule())
        store_uri = str(tmp_path)
        store = LocalStore(tmp_path)
        registry = load_registry()
        _seed_non_slot_stages(store, registry)

        for job in sorted(ARC_SLOT_JOBS):
            for slot in SLOTS:
                args = _experiment_args(job, slot, store_uri)
                code = HANDLERS[job](args)
                assert code == 0

        clause = _clause_arc_runs_ok(store, [FRIDAY], registry)

        assert clause.met, clause.detail
        # And the write really did land under the discriminated key the arc
        # jobs claim — not just "the clause happened to pass".
        for slot in SLOTS:
            for job in sorted(ARC_SLOT_JOBS):
                key = manifest_key(job, FRIDAY.isoformat(), discriminator=slot)
                assert store.exists(key), f"expected a real writer at {key}"

    def test_a_bare_undiscriminated_seed_is_correctly_reported_as_never_ran(self, tmp_path) -> None:
        """The inverse check, so this file cannot pass by accident: a store
        seeded at the OLD, bare key (the shape the pre-fix gate read, and the
        shape `tests/test_gate.py::_seed_met` used to write) must NOT satisfy
        the clause — four slot writers sharing one bare key is exactly the
        collision `alpha-engine-config-I9781` fixed, and a gate that called
        that "ran" would be measuring the collision as success."""
        store = LocalStore(tmp_path)
        registry = load_registry()
        for stage in arc_stages(FRIDAY, registry):
            store.put_bytes(
                manifest_key(stage.job, FRIDAY.isoformat()),
                b'{"status": "ok", "reason": "", "inputs": [], "outputs": [], '
                b'"release_sha": "' + b"a" * 40 + b'"}',
            )
        clause = _clause_arc_runs_ok(store, [FRIDAY], registry)
        assert not clause.met
        # Exactly the slot-scoped stages (every DISPATCHABLE slot x
        # {experiment.run, experiment.grade}) are missing — the 4 non-slot
        # stages, which never had a discriminator to begin with, are unaffected.
        expected = 2 * len(dispatchable_slots())
        assert f"{expected} never ran" in clause.detail, clause.detail

    def test_removing_the_discriminator_from_the_gate_read_reintroduces_the_defect(
        self, tmp_path, monkeypatch
    ) -> None:
        """Mutation-style self-test (repo convention, `tests/test_gate.py`'s
        own pattern): if `_clause_arc_runs_ok` regressed to a bare
        `manifest_key(stage.job, day.isoformat())` read, a store seeded by
        the REAL discriminated writer must fail this test — proving the
        discriminator argument in the gate's read is load-bearing, not
        decorative."""
        monkeypatch.setattr(track_a, "_slot_module", lambda slot: _StubSlotModule())
        store_uri = str(tmp_path)
        store = LocalStore(tmp_path)
        registry = load_registry()
        _seed_non_slot_stages(store, registry)
        for job in sorted(ARC_SLOT_JOBS):
            for slot in SLOTS:
                assert HANDLERS[job](_experiment_args(job, slot, store_uri)) == 0

        def _bare_read(store: Any, window: list[dt.date], registry: Any) -> Any:
            from crucible.gate import Clause

            missing: list[str] = []
            for day in window:
                for stage in arc_stages(day, registry):
                    key = manifest_key(stage.job, day.isoformat())  # the OLD, broken read
                    if not store.exists(key):
                        missing.append(f"{stage.label}@{day.isoformat()}")
            return Clause(
                name="arc_runs_ok",
                requirement="x",
                met=not missing,
                detail=f"{len(missing)} never ran" if missing else "ok",
            )

        broken = _bare_read(store, [FRIDAY], registry)
        assert not broken.met, (
            "a bare-key read against a real discriminated writer must fail — "
            "if it passes, this test's own oracle is wrong"
        )
        real = _clause_arc_runs_ok(store, [FRIDAY], registry)
        assert real.met, real.detail
