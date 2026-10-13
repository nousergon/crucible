"""Both pointer writers attach the §9.1 attestation `experiment.grade` measured.

Plan §9.1: "a card without `attestation: PASS` renders UNVERIFIED, never a
grade", and `crucible.champion._assert_attested` refuses an S champion that
does not carry one. Until this file, neither writer in `crucible.promote`
attached one — `crucible.cli._promote` passed no ``attestation=`` on either
path — so every S pointer the harness could write was one the trader would
refuse. The 2026-10-07 operator revert wrote exactly that pointer, and the
2026-10-12 paper session was refused on it.

The attestation is READ off the grade run's own manifest
(`crucible.champion.ATTESTATION_METRIC` rows, one per graded arm), never
computed by `promote` and never typed by an operator; a verdict that is not
`PASS` refuses the write rather than producing a pointer the reader refuses.
"""

from __future__ import annotations

import json

import pytest

from crucible.champion import (
    ChampionUnusableError,
    _assert_attested,
    champion_key,
)
from crucible.promote import (
    PromotionRefused,
    read_graded_attestation,
    revert_champion,
    run_promotion,
)
from crucible.slots import get_slot
from crucible.store import LocalStore
from tests.support.manifests import attestation_row, write_grade_manifest
from tests.test_promote import (
    CODE_SHA,
    _cycle_for,
    narrow,
    register_with,
    series,
    trading_days,
)


def _row(arm_id: str, status: str) -> dict:
    return attestation_row(arm_id, status)


class TestReadGradedAttestation:
    def test_a_measured_pass_is_returned_shaped_for_the_pointer(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        day = trading_days(1)[-1]
        key = write_grade_manifest(store, "s", day, metrics=[_row("s:a:0123456789ab", "PASS")])

        attestation = read_graded_attestation(store, "s", day, "s:a:0123456789ab")

        assert attestation["kind"] == "pit_parity"
        assert attestation["status"] == "PASS"
        assert attestation["key"] == key, "the attestation names the run that measured it"
        assert "fixture PASS" in attestation["reason"]

    @pytest.mark.parametrize("status", ["FAIL", "PARTIAL", "UNKNOWN"])
    def test_anything_but_pass_is_refused(self, tmp_path, status) -> None:
        store = LocalStore(tmp_path)
        day = trading_days(1)[-1]
        write_grade_manifest(store, "s", day, metrics=[_row("s:a:0123456789ab", status)])
        with pytest.raises(PromotionRefused, match=f"attestation of '{status}'"):
            read_graded_attestation(store, "s", day, "s:a:0123456789ab")

    def test_another_arms_pass_is_not_this_arms(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        day = trading_days(1)[-1]
        write_grade_manifest(store, "s", day, metrics=[_row("s:other:0123456789ab", "PASS")])
        with pytest.raises(PromotionRefused, match="0 'pit_parity_attestation' row"):
            read_graded_attestation(store, "s", day, "s:a:0123456789ab")

    def test_a_failed_grade_run_is_refused_whatever_it_recorded(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        day = trading_days(1)[-1]
        write_grade_manifest(
            store,
            "s",
            day,
            status="failed",
            reason="MissingArtifactError",
            metrics=[_row("s:a:0123456789ab", "PASS")],
        )
        with pytest.raises(PromotionRefused, match="did not succeed"):
            read_graded_attestation(store, "s", day, "s:a:0123456789ab")

    def test_no_grade_run_is_refused(self, tmp_path) -> None:
        with pytest.raises(PromotionRefused, match="run manifest"):
            read_graded_attestation(
                LocalStore(tmp_path), "s", trading_days(1)[-1], "s:a:0123456789ab"
            )


class TestTheOperatorRevertCarriesTheMeasuredAttestation:
    """The 2026-10-07 shape: `promote --slot s --revert-to <arm>`."""

    def _revert(self, store, reg, arm_id: str, day: str):
        return revert_champion(
            spec=get_slot("s"),
            register=reg,
            store=store,
            arm_id=arm_id,
            as_of=day,
            operator="cipher813",
            reason="re-seat S with its measured attestation",
            code_sha=CODE_SHA,
        )

    def test_an_s_revert_writes_a_pointer_the_reader_accepts(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        dates = trading_days(5)
        reg, ids = register_with("s", ["champ"], dates[0])
        key = write_grade_manifest(store, "s", dates[-1], metrics=[_row(ids["champ"], "PASS")])

        pointer = self._revert(store, reg, ids["champ"], dates[-1])

        assert pointer.attestation is not None
        assert pointer.attestation["status"] == "PASS"
        assert pointer.attestation["key"] == key
        _assert_attested(pointer)  # the trader's own gate: no raise
        stored = json.loads(store.get_bytes(champion_key("s")))
        assert stored["attestation"]["status"] == "PASS"

    def test_an_s_revert_with_a_failed_attestation_writes_nothing(self, tmp_path) -> None:
        """Measured on the 2026-10-09 S grade before the re-resolution fix:
        FAIL. That must refuse the revert, not seat a pointer the reader
        refuses at 10:45."""
        store = LocalStore(tmp_path)
        dates = trading_days(5)
        reg, ids = register_with("s", ["champ"], dates[0])
        write_grade_manifest(store, "s", dates[-1], metrics=[_row(ids["champ"], "FAIL")])

        with pytest.raises(PromotionRefused, match="not 'PASS'"):
            self._revert(store, reg, ids["champ"], dates[-1])
        assert not store.exists(champion_key("s"))

    def test_an_s_revert_with_no_grade_for_the_day_writes_nothing(self, tmp_path) -> None:
        """Exactly what the 2026-10-07 revert did — and the pointer it wrote
        was refused by the trader. Now it is refused at the write."""
        store = LocalStore(tmp_path)
        dates = trading_days(5)
        reg, ids = register_with("s", ["champ"], dates[0])

        with pytest.raises(PromotionRefused, match="run manifest"):
            self._revert(store, reg, ids["champ"], dates[-1])
        assert not store.exists(champion_key("s"))

    def test_the_old_unattested_pointer_is_still_refused_by_the_reader(self) -> None:
        """The reader's gate is untouched: this change adds the attestation
        at the writer and weakens nothing at the reader."""
        from crucible.champion import ChampionPointer

        pointer = ChampionPointer(
            slot="s",
            arm_id="s:a:0123456789ab",
            as_of="2026-10-07",
            decided_at="2026-10-07T22:12:15Z",
            run_id="0" * 26,
            code_sha=CODE_SHA,
            promotion_source="operator_bootstrap",
            manifest_key="runs/promote/2026-10-07/s/run.json",
            evidence={"operator": "x", "reason": "y"},
            attestation=None,
        )
        with pytest.raises(ChampionUnusableError, match="UNVERIFIED"):
            _assert_attested(pointer)

    def test_a_revert_outside_the_attested_slots_needs_no_grade(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        dates = trading_days(5)
        reg, ids = register_with("u", ["champ"], dates[0])
        pointer = revert_champion(
            spec=get_slot("u"),
            register=reg,
            store=store,
            arm_id=ids["champ"],
            as_of=dates[-1],
            operator="cipher813",
            reason="u revert",
            code_sha=CODE_SHA,
        )
        assert pointer.attestation is None


class TestAPromotionOnEvidenceCarriesTheMeasuredAttestation:
    def _promote(self, store):
        spec = narrow(get_slot("s"))
        dates = trading_days(4 * 5)
        reg, ids = register_with("s", ["champ", "chal"], dates[0])
        cycle = _cycle_for(
            spec,
            dates[-1],
            reg,
            {
                ids["champ"]: series(ids["champ"], dates, 0.0),
                ids["chal"]: series(ids["chal"], dates, 0.01),
            },
            incumbent=ids["champ"],
        )
        assert cycle.decision.moved and cycle.decision.champion == ids["chal"]
        return spec, reg, ids, dates, cycle

    def test_the_promoted_arms_pass_reaches_the_pointer(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        spec, reg, ids, dates, cycle = self._promote(store)
        key = write_grade_manifest(
            store,
            "s",
            dates[-1],
            metrics=[_row(ids["champ"], "FAIL"), _row(ids["chal"], "PASS")],
        )

        result = run_promotion(spec=spec, cycle=cycle, register=reg, store=store, code_sha=CODE_SHA)

        assert result.pointer is not None
        assert result.pointer.arm_id == ids["chal"]
        assert result.pointer.attestation["key"] == key
        _assert_attested(result.pointer)

    def test_a_promotion_with_no_measured_pass_writes_no_pointer(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        spec, reg, ids, dates, cycle = self._promote(store)
        write_grade_manifest(store, "s", dates[-1], metrics=[_row(ids["champ"], "PASS")])

        with pytest.raises(PromotionRefused, match="pit_parity_attestation"):
            run_promotion(spec=spec, cycle=cycle, register=reg, store=store, code_sha=CODE_SHA)
        assert not store.exists(champion_key("s"))
