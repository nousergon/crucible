"""crucible-state backup verification: the pure comparator behind the nightly
restore drill (alpha-engine-config-I12513, S-25; audit finding SH-02).

Normative source: the design on alpha-engine-config-I12513 (2026-10-22, with the
SH-02 addendum of 2026-10-23), and Brian's ruling of 2026-10-24: **RPO 24h / RTO
4h approved**, "build the nightly backup and restore drill with no new spend".

WHAT THIS MODULE IS
-------------------
The nightly job takes a consistent ``pg_dump`` of crucible-state, restores it
into a throwaway LOCAL Postgres (``initdb``, socket only, never a Neon branch),
and re-measures the restored copy. This module is everything in that job that
is a JUDGMENT rather than a side effect:

* the documents: :class:`Manifest` (captured at dump time, inside the same
  REPEATABLE READ snapshot as the dump) and :class:`RestoredMeasurement` (the
  same fields re-measured on the restored copy, plus the recovery timings);
* :func:`verify`, the six named checks V1-V6, each a PASS/FAIL with the reason;
* :func:`latest_pointer`, the fixed-name ``latest.json`` the freshness monitor
  reads. ``verify_ok`` is true only when V1-V5 all pass;
* the negative controls m1-m6 (:data:`MUTATIONS`). After a PASS, the job corrupts
  the scratch copy one way and re-verifies, and :func:`negative_control_caught`
  says whether the right check noticed. A verifier that has never been shown to
  fail is not yet known to work;
* the S3 layout and the retention rule.

It is pure Python over JSON, so its tests run in the existing test job with no
Postgres service and no AWS. The side effects (``pg_dump``, ``pg_restore``, the
catalog and row-hash queries, the S3 writes) are the job's, and they are built
on top of this. None of them is in this change. There is no runtime change, no
IAM grant and no Neon spend here.

THE CHECKS
----------
V1 schema       the restore exited clean, the restored ``schema_fingerprint`` and
                ``alembic_version`` equal the manifest's, and every manifest table
                is present.
V2 identities   every sequence has ``last_value >= max(col)`` on the restored copy
                (so the next insert cannot collide), and it equals the manifest's.
V3 values       every table's row count and full-row hash, and every PK-range hash,
                equal the manifest's. This catches corruption that leaves the row
                count unchanged: a flipped value, swapped amounts, or N rows
                replaced by N others.
V4 invariants   every financial invariant HOLDS on the restored copy and its
                answer equals the manifest's prod answer.
V5 application  every golden application read equals the manifest's.
V6 recovery     measured RTO (download start to V5 PASS) is at most
                :data:`RTO_OBJECTIVE`, measured RPO (the span of writes the
                previous dump did not hold) is at most :data:`RPO_OBJECTIVE`, and
                the restored copy lost no write the snapshot saw.

V6 is a recovery MEASUREMENT against the approved objectives, not an integrity
check. It is reported separately (:attr:`VerifyReport.objectives_met`) and does
not gate ``verify_ok``. A slow restore is a finding, not a corrupt backup.

Absence is never a pass. A table, sequence, invariant or golden read that the
manifest names and the restored copy does not carry FAILS its check, naming what
is missing. A manifest naming no tables at all fails V1. "Nothing to compare"
never reads as "everything matched".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from crucible import keys

#: Approved 2026-10-24 by Brian ("RPO 24h / RTO 4h approved"), I12513.
RPO_OBJECTIVE: Final[timedelta] = timedelta(hours=24)
RTO_OBJECTIVE: Final[timedelta] = timedelta(hours=4)
OBJECTIVES_SOURCE: Final[str] = (
    "S-25 crucible-state backup objectives, Brian's ruling 2026-10-24: RPO 24h / RTO 4h approved"
)

#: Retention (the I12513 design). Daily dumps are kept 35 days. The Sunday dump is
#: also copied to ``weekly/`` and kept 13 months. Both are enforced by S3
#: lifecycle plus Object Lock GOVERNANCE for the same periods, in the bucket's IaC
#: (not in this change).
DAILY_RETENTION: Final[timedelta] = timedelta(days=35)
WEEKLY_RETENTION_MONTHS: Final[int] = 13

CheckName = Literal["V1", "V2", "V3", "V4", "V5", "V6"]
INTEGRITY_CHECKS: Final[tuple[CheckName, ...]] = ("V1", "V2", "V3", "V4", "V5")
ALL_CHECKS: Final[tuple[CheckName, ...]] = (*INTEGRITY_CHECKS, "V6")


# ── documents ─────────────────────────────────────────────────────────────────


class _Doc(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SequenceState(_Doc):
    """One identity or serial column: the sequence and its table's high-water mark."""

    last_value: int
    is_called: bool
    max_value_in_table: int | None = Field(
        description="max(col) of the owning table, None for an empty table"
    )


class TableHash(_Doc):
    """Per-table count plus an ordered-by-PK full-row hash.

    For tables over 100k rows, ``range_hashes`` holds one hash per PK range of 10k
    rows (keyed ``"<lo>..<hi>"``), so a failure names the range.
    """

    rows: int = Field(ge=0)
    row_hash: str
    range_hashes: dict[str, str] = Field(default_factory=dict)


class InvariantAnswer(_Doc):
    """A declared financial-invariant query's answer: whether it held, and its value."""

    holds: bool
    value: str


class Snapshot(_Doc):
    taken_at: datetime
    wal_lsn: str
    max_written_at: datetime | None = Field(
        description="max(written_at) over the ledger tables; None before any write"
    )


class Manifest(_Doc):
    """``daily/{date}.manifest.json``: written beside the dump, from the same snapshot."""

    schema_version: Literal["crucible_state_backup_manifest.v1"] = (
        "crucible_state_backup_manifest.v1"
    )
    env: str
    dump_date: date
    dump_sha256: str
    pg_dump_version: str
    alembic_version: str
    schema_fingerprint: str
    sequences: dict[str, SequenceState]
    tables: dict[str, TableHash]
    invariants: dict[str, InvariantAnswer]
    golden: dict[str, str]
    snapshot: Snapshot
    previous_max_written_at: datetime | None = Field(
        default=None,
        description="the PREVIOUS dump's snapshot max(written_at), for the RPO measurement",
    )


class RestoredMeasurement(_Doc):
    """The same measurements, taken on the restored scratch copy, plus timings."""

    restore_exit_ok: bool = Field(description="pg_restore --exit-on-error exited 0")
    dump_sha256: str = Field(description="sha256 of the object actually restored")
    alembic_version: str
    schema_fingerprint: str
    sequences: dict[str, SequenceState]
    tables: dict[str, TableHash]
    invariants: dict[str, InvariantAnswer]
    golden: dict[str, str]
    max_written_at: datetime | None
    download_started_at: datetime
    verified_at: datetime = Field(description="when V5 finished on the restored copy")


class CheckResult(_Doc):
    check: CheckName
    passed: bool
    failures: tuple[str, ...] = ()


class VerifyReport(_Doc):
    """``verify/{date}.json``."""

    schema_version: Literal["crucible_state_backup_verify.v1"] = "crucible_state_backup_verify.v1"
    env: str
    dump_date: date
    checks: tuple[CheckResult, ...]
    rto_measured_seconds: float
    rpo_measured_seconds: float | None
    rto_objective_seconds: float
    rpo_objective_seconds: float
    objectives_source: str

    def result(self, check: CheckName) -> CheckResult:
        for r in self.checks:
            if r.check == check:
                return r
        raise KeyError(check)

    @property
    def verify_ok(self) -> bool:
        """Every integrity check (V1-V5) passed. V6 never gates this."""
        return all(self.result(c).passed for c in INTEGRITY_CHECKS)

    @property
    def objectives_met(self) -> bool:
        return self.result("V6").passed


# ── canonical fingerprint ─────────────────────────────────────────────────────


def schema_fingerprint(
    catalog_rows: Iterable[Sequence[str | int | bool | None]], alembic_version: str
) -> str:
    """sha256 over normalized ``pg_catalog`` rows plus the ``alembic_version``.

    The caller passes one row per schema object: tables, columns (name, type,
    nullability, default), constraints (with ``convalidated``), indexes, triggers
    and extensions. The rows are sorted here, so the catalog's return order cannot
    change the fingerprint. Two databases with the same objects fingerprint
    identically, and a lost constraint, index or trigger changes it.
    """
    canon = sorted(json.dumps(list(r), separators=(",", ":")) for r in catalog_rows)
    h = hashlib.sha256()
    h.update(f"alembic_version={alembic_version}\n".encode())
    for line in canon:
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()


# ── the checks ────────────────────────────────────────────────────────────────


def _missing(kind: str, want: Mapping[str, object], got: Mapping[str, object]) -> list[str]:
    return [f"{kind} {k!r} absent from the restored copy" for k in sorted(set(want) - set(got))]


def _v1(m: Manifest, r: RestoredMeasurement) -> CheckResult:
    f: list[str] = []
    if not r.restore_exit_ok:
        f.append("pg_restore --exit-on-error did not exit 0")
    if r.dump_sha256 != m.dump_sha256:
        f.append(f"restored object sha256 {r.dump_sha256} != manifest {m.dump_sha256}")
    if r.schema_fingerprint != m.schema_fingerprint:
        f.append("schema_fingerprint differs: a constraint, index, trigger or column changed")
    if r.alembic_version != m.alembic_version:
        f.append(f"alembic_version {r.alembic_version!r} != manifest {m.alembic_version!r}")
    if not m.tables:
        f.append("manifest names no tables: nothing to compare is never a pass")
    f += _missing("table", m.tables, r.tables)
    return CheckResult(check="V1", passed=not f, failures=tuple(f))


def _v2(m: Manifest, r: RestoredMeasurement) -> CheckResult:
    f = _missing("sequence", m.sequences, r.sequences)
    for name, want in sorted(m.sequences.items()):
        got = r.sequences.get(name)
        if got is None:
            continue
        hw = got.max_value_in_table
        if hw is not None and got.last_value < hw:
            f.append(
                f"sequence {name!r} last_value {got.last_value} is behind max(col) {hw}:"
                " the next insert collides"
            )
        if (got.last_value, got.is_called) != (want.last_value, want.is_called):
            f.append(
                f"sequence {name!r} restored at ({got.last_value}, {got.is_called}),"
                f" manifest ({want.last_value}, {want.is_called})"
            )
    return CheckResult(check="V2", passed=not f, failures=tuple(f))


def _v3(m: Manifest, r: RestoredMeasurement) -> CheckResult:
    f = _missing("table", m.tables, r.tables)
    for name, want in sorted(m.tables.items()):
        got = r.tables.get(name)
        if got is None:
            continue
        if got.rows != want.rows:
            f.append(f"table {name!r} has {got.rows} rows, manifest {want.rows}")
        if got.row_hash != want.row_hash:
            f.append(
                f"table {name!r} row_hash differs"
                + (" with the row count UNCHANGED" if got.rows == want.rows else "")
            )
        for rng in sorted(set(want.range_hashes) | set(got.range_hashes)):
            if want.range_hashes.get(rng) != got.range_hashes.get(rng):
                f.append(f"table {name!r} PK range {rng} hash differs")
    return CheckResult(check="V3", passed=not f, failures=tuple(f))


def _v4(m: Manifest, r: RestoredMeasurement) -> CheckResult:
    f = _missing("invariant", m.invariants, r.invariants)
    for name, want in sorted(m.invariants.items()):
        got = r.invariants.get(name)
        if got is None:
            continue
        if not got.holds:
            f.append(f"invariant {name!r} does not hold on the restored copy ({got.value})")
        if got.value != want.value:
            f.append(f"invariant {name!r} = {got.value!r}, prod answer {want.value!r}")
    return CheckResult(check="V4", passed=not f, failures=tuple(f))


def _v5(m: Manifest, r: RestoredMeasurement) -> CheckResult:
    f = _missing("golden read", m.golden, r.golden)
    for name, want in sorted(m.golden.items()):
        got = r.golden.get(name)
        if got is not None and got != want:
            f.append(f"golden read {name!r} = {got!r}, manifest {want!r}")
    return CheckResult(check="V5", passed=not f, failures=tuple(f))


def _v6(m: Manifest, r: RestoredMeasurement) -> tuple[CheckResult, float, float | None]:
    f: list[str] = []
    rto = r.verified_at - r.download_started_at
    if rto < timedelta(0):
        f.append("verified_at precedes download_started_at: the timings are not a measurement")
    elif rto > RTO_OBJECTIVE:
        f.append(f"measured RTO {rto} exceeds the approved {RTO_OBJECTIVE}")
    rpo: timedelta | None = None
    if m.previous_max_written_at is None or m.snapshot.max_written_at is None:
        f.append("RPO cannot be measured (no previous dump, or no write yet); unknown is not met")
    else:
        # The writes the previous dump did NOT hold, had the primary been lost just
        # before this dump: the design's "restored max(written_at) gap to the last
        # write before the next dump", measured one dump in arrears.
        rpo = m.snapshot.max_written_at - m.previous_max_written_at
        if rpo > RPO_OBJECTIVE:
            f.append(
                f"measured RPO {rpo} (writes the previous dump did not hold) exceeds"
                f" the approved {RPO_OBJECTIVE}"
            )
    if r.max_written_at != m.snapshot.max_written_at:
        f.append(
            f"restored max(written_at) {r.max_written_at} != snapshot"
            f" {m.snapshot.max_written_at}: the restore lost writes the dump saw"
        )
    return (
        CheckResult(check="V6", passed=not f, failures=tuple(f)),
        rto.total_seconds(),
        None if rpo is None else rpo.total_seconds(),
    )


def verify(manifest: Manifest, restored: RestoredMeasurement) -> VerifyReport:
    """Run V1-V6 and return the ``verify/{date}.json`` document."""
    v6, rto, rpo = _v6(manifest, restored)
    checks = (
        _v1(manifest, restored),
        _v2(manifest, restored),
        _v3(manifest, restored),
        _v4(manifest, restored),
        _v5(manifest, restored),
        v6,
    )
    return VerifyReport(
        env=manifest.env,
        dump_date=manifest.dump_date,
        checks=checks,
        rto_measured_seconds=rto,
        rpo_measured_seconds=rpo,
        rto_objective_seconds=RTO_OBJECTIVE.total_seconds(),
        rpo_objective_seconds=RPO_OBJECTIVE.total_seconds(),
        objectives_source=OBJECTIVES_SOURCE,
    )


# ── negative controls (acceptance (a)) ────────────────────────────────────────

MutationId = Literal["m1", "m2", "m3", "m4", "m5", "m6"]

#: Each mutation the job applies to the restored scratch copy after a PASS, and the
#: checks that must then report FAIL. The design names V3 for m3; V4 is required
#: as well whenever a chain invariant is declared.
MUTATIONS: Final[dict[MutationId, tuple[str, tuple[CheckName, ...]]]] = {
    "m1": ("change one numeric value", ("V3",)),
    "m2": ("swap two rows' amounts: count and sum unchanged", ("V3",)),
    "m3": ("break one ledger chain link", ("V3", "V4")),
    "m4": ("set a sequence behind max(id)", ("V2",)),
    "m5": ("drop one CHECK constraint", ("V1",)),
    "m6": ("delete N rows and insert N different rows", ("V3",)),
}


def nightly_mutation(on: date) -> MutationId:
    """The one rotating mutation for a given night. Each runs every sixth night."""
    ids: tuple[MutationId, ...] = ("m1", "m2", "m3", "m4", "m5", "m6")
    return ids[on.toordinal() % len(ids)]


def negative_control_caught(mutation: MutationId, report: VerifyReport) -> tuple[bool, str]:
    """Whether every check the mutation must trip reported FAIL on the mutated copy.

    A run fails unless this is true. A mutation nobody noticed means the verifier
    is blind to that class of corruption, and a passing backup verified by it
    proves nothing about that class.
    """
    _desc, expected = MUTATIONS[mutation]
    missed = [c for c in expected if report.result(c).passed]
    if missed:
        return False, f"{mutation} ({_desc}) was NOT caught by {', '.join(missed)}"
    return True, f"{mutation} caught by {', '.join(expected)}"


# ── layout and the pointer ────────────────────────────────────────────────────
# The key shapes live in crucible.keys (its rule: every store key shape in one
# place). They are re-exported here so the job reads one module.

daily_dump_key = keys.state_backup_daily_dump_key
daily_manifest_key = keys.state_backup_daily_manifest_key
verify_key = keys.state_backup_verify_key
weekly_dump_key = keys.state_backup_weekly_dump_key
latest_key = keys.state_backup_latest_key


def is_weekly_copy(on: date) -> bool:
    """The Sunday dump is also kept under ``weekly/`` for 13 months."""
    return on.weekday() == 6


def latest_pointer(manifest: Manifest, report: VerifyReport, written_at: datetime) -> str:
    """The ``latest.json`` body. The freshness row reads ``written_at`` and
    ``verify_ok``, so a dump whose restore failed never reads as a fresh backup."""
    if (report.env, report.dump_date) != (manifest.env, manifest.dump_date):
        raise ValueError("report and manifest describe different dumps")
    doc = {
        "schema_version": "crucible_state_backup_latest.v1",
        "env": manifest.env,
        "dump_date": manifest.dump_date.isoformat(),
        "dump_key": daily_dump_key(manifest.env, manifest.dump_date),
        "manifest_key": daily_manifest_key(manifest.env, manifest.dump_date),
        "verify_key": verify_key(manifest.env, manifest.dump_date),
        "dump_sha256": manifest.dump_sha256,
        "verify_ok": report.verify_ok,
        "objectives_met": report.objectives_met,
        "failed_checks": [r.check for r in report.checks if not r.passed],
        "written_at": written_at.isoformat(),
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"
