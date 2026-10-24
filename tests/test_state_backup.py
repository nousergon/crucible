"""crucible.state_backup: the restore-drill comparator (alpha-engine-config-I12513).

The fixtures are a synthetic staging-shaped manifest and the restored copy that
matches it. Each negative control below is built the way the nightly job builds
it: the job corrupts the restored scratch copy and re-measures, so a corruption
shows up as a changed measurement on ``RestoredMeasurement``. Every one of m1-m6
must be caught by the check the design names.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from crucible import state_backup as sb

FIX = Path(__file__).parent / "fixtures" / "state_backup"


def _manifest() -> sb.Manifest:
    return sb.Manifest.model_validate_json((FIX / "manifest_2026-10-25.json").read_text())


def _restored() -> sb.RestoredMeasurement:
    return sb.RestoredMeasurement.model_validate_json(
        (FIX / "restored_2026-10-25.json").read_text()
    )


def _with_table(r: sb.RestoredMeasurement, name: str, **upd: object) -> sb.RestoredMeasurement:
    tables = dict(r.tables)
    tables[name] = tables[name].model_copy(update=upd)
    return r.model_copy(update={"tables": tables})


# ── the clean restore ─────────────────────────────────────────────────────────


def test_a_faithful_restore_passes_every_check():
    report = sb.verify(_manifest(), _restored())
    assert [c.check for c in report.checks] == list(sb.ALL_CHECKS)
    assert all(c.passed for c in report.checks), [c for c in report.checks if not c.passed]
    assert report.verify_ok and report.objectives_met
    assert report.rto_measured_seconds == 450.0
    # RPO is the write gap: the writes the previous dump did not hold.
    assert report.rpo_measured_seconds == timedelta(hours=23, minutes=49).total_seconds()


def test_the_approved_objectives_are_24h_and_4h():
    assert sb.RPO_OBJECTIVE == timedelta(hours=24)
    assert sb.RTO_OBJECTIVE == timedelta(hours=4)
    report = sb.verify(_manifest(), _restored())
    assert report.rpo_objective_seconds == 86400 and report.rto_objective_seconds == 14400
    assert "2026-10-24" in report.objectives_source


# ── negative controls m1-m6 (acceptance (a)) ──────────────────────────────────


def _mutated(mutation: str) -> sb.RestoredMeasurement:
    r = _restored()
    if mutation == "m1":  # one numeric value changed: count unchanged, hash moves
        return _with_table(r, "fills", row_hash="c3d2-m1")
    if mutation == "m2":  # two amounts swapped: count and sum unchanged, hash moves
        return _with_table(r, "postings", row_hash="b2e1-m2")
    if mutation == "m3":  # one chain link broken: the row hash and the chain head move
        r = _with_table(r, "ledger_events", row_hash="a1f0-m3")
        inv = dict(r.invariants)
        inv["ledger_chain_head"] = sb.InvariantAnswer(holds=False, value="break at event 2201")
        return r.model_copy(update={"invariants": inv})
    if mutation == "m4":  # a sequence set behind max(id)
        seqs = dict(r.sequences)
        seqs["fills_id_seq"] = seqs["fills_id_seq"].model_copy(update={"last_value": 900})
        return r.model_copy(update={"sequences": seqs})
    if mutation == "m5":  # one CHECK constraint dropped
        return r.model_copy(update={"schema_fingerprint": "f" * 64})
    if mutation == "m6":  # N rows deleted, N different rows inserted, in a ranged table
        r = _with_table(
            r,
            "trial_ledger",
            row_hash="d4c3-m6",
            range_hashes={"1..10000": "r1", "10001..20000": "r2-m6", "20001..120417": "r3"},
        )
        return r
    raise AssertionError(mutation)


@pytest.mark.parametrize("mutation", sorted(sb.MUTATIONS))
def test_every_negative_control_is_caught_by_its_named_check(mutation):
    report = sb.verify(_manifest(), _mutated(mutation))
    caught, why = sb.negative_control_caught(mutation, report)  # type: ignore[arg-type]
    assert caught, why
    assert not report.verify_ok


def test_unchanged_row_count_corruption_fails_and_says_so():
    report = sb.verify(_manifest(), _mutated("m2"))
    v3 = report.result("V3")
    assert not v3.passed
    assert any("row count UNCHANGED" in f for f in v3.failures)


def test_a_ranged_failure_names_the_range():
    v3 = sb.verify(_manifest(), _mutated("m6")).result("V3")
    assert any("PK range 10001..20000" in f for f in v3.failures)


def test_a_blind_verifier_is_reported_as_a_missed_control():
    # The mutation happened, but the re-measurement did not change: the verifier
    # is blind to that class, and the run must fail rather than pass.
    report = sb.verify(_manifest(), _restored())
    caught, why = sb.negative_control_caught("m1", report)
    assert not caught and "NOT caught by V3" in why


def test_the_rotation_covers_all_six_in_six_nights():
    start = date(2026, 10, 25)
    assert {sb.nightly_mutation(start + timedelta(days=i)) for i in range(6)} == set(sb.MUTATIONS)


# ── absence is never a pass ───────────────────────────────────────────────────


def test_a_missing_table_fails_v1_and_v3():
    r = _restored()
    tables = {k: v for k, v in r.tables.items() if k != "fills"}
    report = sb.verify(_manifest(), r.model_copy(update={"tables": tables}))
    assert not report.result("V1").passed and not report.result("V3").passed
    assert any("'fills' absent" in f for f in report.result("V1").failures)


def test_a_manifest_naming_no_tables_fails():
    m = _manifest().model_copy(update={"tables": {}})
    r = _restored().model_copy(update={"tables": {}})
    assert not sb.verify(m, r).result("V1").passed


@pytest.mark.parametrize(
    "field,check", [("invariants", "V4"), ("golden", "V5"), ("sequences", "V2")]
)
def test_a_missing_measurement_fails_its_check(field, check):
    r = _restored().model_copy(update={field: {}})
    result = sb.verify(_manifest(), r).result(check)
    assert not result.passed and any("absent" in f for f in result.failures)


def test_an_invariant_that_does_not_hold_fails_even_if_the_value_matches():
    r = _restored()
    inv = dict(r.invariants)
    inv["postings_balance_per_entry"] = sb.InvariantAnswer(
        holds=False, value="0 unbalanced entries"
    )
    assert not sb.verify(_manifest(), r.model_copy(update={"invariants": inv})).result("V4").passed


def test_restore_exit_and_object_identity_gate_v1():
    assert (
        not sb.verify(_manifest(), _restored().model_copy(update={"restore_exit_ok": False}))
        .result("V1")
        .passed
    )
    assert (
        not sb.verify(_manifest(), _restored().model_copy(update={"dump_sha256": "0" * 64}))
        .result("V1")
        .passed
    )
    assert (
        not sb.verify(_manifest(), _restored().model_copy(update={"alembic_version": "0006"}))
        .result("V1")
        .passed
    )


# ── V6: recovery measured against the approved objectives ────────────────────


def test_a_slow_restore_breaches_rto_but_does_not_fail_the_backup():
    r = _restored()
    late = r.download_started_at + timedelta(hours=4, minutes=1)
    slow = r.model_copy(update={"verified_at": late})
    report = sb.verify(_manifest(), slow)
    assert not report.objectives_met and report.verify_ok
    assert any("RTO" in f for f in report.result("V6").failures)


def test_a_missed_night_breaches_rpo():
    m = _manifest()
    assert m.snapshot.max_written_at is not None
    prev = m.snapshot.max_written_at - timedelta(hours=49)
    late = m.model_copy(update={"previous_max_written_at": prev})
    report = sb.verify(late, _restored())
    assert not report.objectives_met and any("RPO" in f for f in report.result("V6").failures)


def test_a_fixed_nightly_schedule_has_no_rpo_headroom():
    """Pinned on purpose. A nightly ledger writer followed by a nightly dump puts the
    last writes of consecutive nights about 24h apart, so a writer that finishes a
    minute later than the night before already breaches a strict 24h RPO. The check
    stays strict, because no tolerance was approved. Headroom is the job's schedule
    to provide, not a tolerance hidden here."""
    m = _manifest()
    assert m.snapshot.max_written_at is not None
    prev = m.snapshot.max_written_at - timedelta(hours=24, seconds=55)
    report = sb.verify(m.model_copy(update={"previous_max_written_at": prev}), _restored())
    assert not report.objectives_met


def test_an_unmeasurable_rpo_is_not_met():
    m = _manifest().model_copy(update={"previous_max_written_at": None})
    report = sb.verify(m, _restored())
    assert report.rpo_measured_seconds is None and not report.objectives_met


def test_a_restore_that_lost_writes_fails_v6():
    lost = datetime.fromisoformat("2026-10-25T20:00:00+00:00")
    r = _restored().model_copy(update={"max_written_at": lost})
    assert not sb.verify(_manifest(), r).result("V6").passed


# ── fingerprint, layout, pointer ──────────────────────────────────────────────


def test_fingerprint_ignores_catalog_order_and_sees_a_dropped_constraint():
    rows = [
        ("table", "fills"),
        ("constraint", "fills", "ck_qty_positive", "c", True),
        ("index", "fills", "ix_fills_exec_id"),
    ]
    a = sb.schema_fingerprint(rows, "0007")
    assert sb.schema_fingerprint(list(reversed(rows)), "0007") == a
    assert sb.schema_fingerprint(rows[:1] + rows[2:], "0007") != a
    assert sb.schema_fingerprint(rows, "0008") != a


def test_layout_and_weekly_rule():
    on = date(2026, 10, 25)  # a Sunday
    assert sb.daily_dump_key("prod", on) == "state_backups/prod/daily/2026-10-25.dump"
    assert sb.daily_manifest_key("prod", on) == "state_backups/prod/daily/2026-10-25.manifest.json"
    assert sb.verify_key("prod", on) == "state_backups/prod/verify/2026-10-25.json"
    assert sb.weekly_dump_key("prod", on) == "state_backups/prod/weekly/2026-10-25.dump"
    assert sb.latest_key("prod") == "state_backups/prod/latest.json"
    assert sb.is_weekly_copy(on) and not sb.is_weekly_copy(on - timedelta(days=1))
    from crucible import keys

    for bad in ("prod/../x", "", ".."):
        with pytest.raises(ValueError):
            keys.state_backup_prefix(bad)


def test_the_pointer_carries_verify_ok_and_never_claims_a_failed_restore():
    m = _manifest()
    now = datetime.fromisoformat("2026-10-25T22:52:00+00:00")
    good = json.loads(sb.latest_pointer(m, sb.verify(m, _restored()), now))
    assert good["verify_ok"] is True and good["failed_checks"] == []
    assert good["dump_key"] == "state_backups/staging/daily/2026-10-25.dump"
    bad = json.loads(sb.latest_pointer(m, sb.verify(m, _mutated("m2")), now))
    assert bad["verify_ok"] is False and bad["failed_checks"] == ["V3"]


def test_the_pointer_refuses_a_mismatched_report():
    m = _manifest()
    report = sb.verify(m, _restored())
    other = m.model_copy(update={"dump_date": date(2026, 10, 24)})
    with pytest.raises(ValueError):
        sb.latest_pointer(other, report, datetime.fromisoformat("2026-10-25T22:52:00+00:00"))


def test_documents_reject_unknown_fields():
    raw = json.loads((FIX / "manifest_2026-10-25.json").read_text())
    raw["surprise"] = 1
    with pytest.raises(ValueError):
        sb.Manifest.model_validate(raw)
