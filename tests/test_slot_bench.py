"""The ruled slot bench: its record, its reader, its writer (`alpha-engine-config-I12024`).

Brian's ruling of 2026-10-18 (option (a)): cut over to v2 with the M slot
benched at zero weight, recording "no lift". The REFUSALS are the subject
here, as they are in `tests/test_holdout.py`: a bench is a ruled record, and a
writer that filed one without a ruling — or a reader that read a malformed one
as "not benched" — would serve an alpha a human ruled to zero, or zero an
alpha nobody ruled on.

Fixed date literals throughout. Every store is a `LocalStore` under
`tmp_path`.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.bench import (
    BENCH_JOB,
    SLOT_BENCH_SCHEMA_VERSION,
    BenchRulingRequiredError,
    SlotBenchError,
    active_bench,
    bench_record_bytes,
    read_slot_bench,
)
from crucible.cli import USAGE_EXIT_CODE, main
from crucible.holdout import payload_digest
from crucible.keys import champion_key, slot_bench_key
from crucible.manifest import on_money_path
from crucible.store import LocalStore
from tests.support.bench import EVIDENCE, RULING_REF, TRACKER_ISSUE, bench_record, file_bench
from tests.support.manifests import only_manifest

#: Thursday 2026-10-22: the plan's effective session. Filed on Monday 10-19.
EFFECTIVE = "2026-10-22"
FILED_ON = "2026-10-19"
CHAMPION = "m:residual_momentum:5349050484f6"
TRACKER = f"alpha-engine-config-I{TRACKER_ISSUE}"


def _seat(store: LocalStore, arm_id: str = CHAMPION) -> None:
    store.put_bytes(
        champion_key("m"),
        json.dumps(
            {"schema_version": "champion_pointer.v1", "slot": "m", "arm_id": arm_id}
        ).encode(),
    )


class TestTheKey:
    def test_one_dateless_record_per_slot_under_its_own_prefix(self) -> None:
        assert slot_bench_key("m") == "slots/m/bench.json"

    def test_an_unknown_slot_is_refused(self) -> None:
        with pytest.raises(KeyError, match="unknown slot"):
            slot_bench_key("x")

    def test_the_record_is_on_the_money_path(self) -> None:
        """It decides that the M champion's alpha reaches the book at zero."""
        assert on_money_path(slot_bench_key("m"))


class TestTheRecordRefusesWithoutARuling:
    @pytest.mark.parametrize(
        ("field", "flag"),
        [
            ("ruled_by", "--ruled-by"),
            ("ruled_at", "--ruled-at"),
            ("ruling_ref", "--ruling-ref"),
            ("tracker", "--tracker"),
            ("reason", "--reason"),
            ("effective_trading_day", "--effective"),
            ("reexam_date", "--reexam"),
        ],
    )
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_every_ruling_field_is_required_by_name(self, field, flag, value) -> None:
        with pytest.raises(BenchRulingRequiredError, match=flag):
            bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, **{field: value})

    @pytest.mark.parametrize("evidence", [None, [], ["", "  "]])
    def test_no_evidence_is_an_opinion_not_a_record(self, evidence) -> None:
        with pytest.raises(BenchRulingRequiredError, match="--evidence"):
            bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, evidence=evidence)

    def test_a_ruling_time_that_is_not_utc_is_refused(self) -> None:
        with pytest.raises(BenchRulingRequiredError, match="not a UTC instant"):
            bench_record(
                CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, ruled_at="2026-10-18 14:44"
            )

    def test_a_ruling_reference_that_is_not_https_is_refused(self) -> None:
        with pytest.raises(BenchRulingRequiredError, match="not an https link"):
            bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, ruling_ref="I12024")

    def test_a_tracker_that_is_not_a_fleet_reference_is_refused(self) -> None:
        with pytest.raises(BenchRulingRequiredError, match="not a fleet reference"):
            bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, tracker="12024")

    def test_only_m_can_be_benched(self) -> None:
        with pytest.raises(SlotBenchError, match="cannot be benched"):
            bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, slot="s")

    def test_no_champion_means_nothing_to_bench(self) -> None:
        with pytest.raises(SlotBenchError, match="no champion to bench"):
            bench_record(None, effective=EFFECTIVE, filed_on=FILED_ON)  # type: ignore[arg-type]

    def test_the_effective_day_must_be_a_session(self) -> None:
        with pytest.raises(SlotBenchError, match="not"):
            bench_record(CHAMPION, effective="2026-10-24", filed_on=FILED_ON)

    def test_a_reexam_that_is_not_a_date_is_refused(self) -> None:
        with pytest.raises(SlotBenchError):
            bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, reexam_date="soon")

    def test_the_past_cannot_be_benched(self) -> None:
        with pytest.raises(SlotBenchError, match="before the filing session"):
            bench_record(CHAMPION, effective="2026-10-16", filed_on=FILED_ON)

    def test_a_reexam_before_it_takes_effect_is_refused(self) -> None:
        with pytest.raises(SlotBenchError, match="re-examined after"):
            bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON, reexam_date="2026-10-21")

    def test_a_valid_record_carries_zero_weight_and_no_lift(self) -> None:
        document = bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON)
        assert document["schema_version"] == SLOT_BENCH_SCHEMA_VERSION
        assert document["weight"] == 0
        assert document["disposition"] == "no_lift"
        assert document["champion_at_bench"] == CHAMPION
        assert document["evidence"] == list(EVIDENCE)


class TestTheReader:
    def test_absent_is_none_and_only_absent(self, tmp_path) -> None:
        assert read_slot_bench(LocalStore(tmp_path), "m") is None

    def test_a_filed_record_reads_with_its_digest(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        document = file_bench(store, CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON)
        bench = read_slot_bench(store, "m")
        assert bench is not None
        assert bench.key == slot_bench_key("m")
        assert bench.digest == payload_digest(document)
        assert bench.slot == "m"
        assert bench.champion_at_bench == CHAMPION
        assert bench.effective_trading_day == EFFECTIVE
        assert bench.reexam_date == "2026-11-20"
        assert bench.tracker == TRACKER
        assert bench.headline() == (
            f"M benched at zero weight — no lift (ruled 2026-10-18 by brian, {TRACKER})"
        )
        described = bench.describe()
        for fragment in (CHAMPION, EFFECTIVE, RULING_REF, bench.digest, "re-exam 2026-11-20"):
            assert fragment in described

    def test_a_record_that_does_not_validate_raises_rather_than_reading_as_absent(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        document = bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON)
        document["weight"] = 0.5
        store.put_bytes(slot_bench_key("m"), bench_record_bytes(document))
        with pytest.raises(SlotBenchError, match="weight"):
            read_slot_bench(store, "m")

    def test_a_record_that_does_not_parse_raises(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(slot_bench_key("m"), b"{not json")
        with pytest.raises(SlotBenchError):
            read_slot_bench(store, "m")

    def test_a_record_filed_under_another_slots_key_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            slot_bench_key("s"),
            bench_record_bytes(bench_record(CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON)),
        )
        with pytest.raises(SlotBenchError, match="another slot"):
            read_slot_bench(store, "s")

    def test_a_store_that_cannot_answer_is_not_not_benched(self, tmp_path) -> None:
        class Denied(LocalStore):
            def get_bytes(self, key: str) -> bytes:
                raise PermissionError(f"AccessDenied on {key}")

        store = Denied(tmp_path)
        LocalStore(tmp_path).put_bytes(slot_bench_key("m"), b"{}")
        with pytest.raises(PermissionError, match="AccessDenied"):
            read_slot_bench(store, "m")

    def test_a_bench_covers_its_effective_session_and_every_one_after(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_bench(store, CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON)
        assert active_bench(store, "m", "2026-10-21") is None
        assert active_bench(store, "m", EFFECTIVE) is not None
        assert active_bench(store, "m", "2026-10-30") is not None

    @pytest.mark.parametrize("slot", ["u", "r", "s"])
    def test_a_slot_that_cannot_be_benched_never_is(self, tmp_path, slot) -> None:
        store = LocalStore(tmp_path)
        file_bench(store, CHAMPION, effective=EFFECTIVE, filed_on=FILED_ON)
        assert active_bench(store, slot, "2026-10-30") is None


class TestTheWriter:
    @staticmethod
    def _argv(tmp_path, *extra: str, omit: str | None = None) -> list[str]:
        flags = {
            "--ruled-by": "brian",
            "--ruled-at": "2026-10-18T14:44Z",
            "--ruling-ref": RULING_REF,
            "--tracker": TRACKER,
            "--reason": "no M arm passes its serving preconditions",
            "--effective": EFFECTIVE,
            "--reexam": "2026-11-20",
        }
        argv = [
            BENCH_JOB,
            "--slot",
            "m",
            "--date",
            FILED_ON,
            "--store",
            str(tmp_path),
            "--run-mode",
            "live",
            "--operator",
            "tester",
        ]
        for flag, value in flags.items():
            if flag != omit:
                argv += [flag, value]
        if omit != "--evidence":
            for ref in EVIDENCE:
                argv += ["--evidence", ref]
        return [*argv, *extra]

    def test_a_ruled_filing_writes_the_record_under_its_own_manifest(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        _seat(store)

        assert main(self._argv(tmp_path)) == 0

        bench = read_slot_bench(store, "m")
        assert bench is not None
        assert bench.champion_at_bench == CHAMPION, "read from the live pointer, not typed"
        assert bench.document["filed_by"] == "tester"
        _key, manifest = only_manifest(store, BENCH_JOB, FILED_ON)
        assert manifest["status"] == "ok"
        assert [o["key"] for o in manifest["outputs"]] == [slot_bench_key("m")]
        (metric,) = [m for m in manifest["metrics"] if m["name"] == "slot_bench_filed"]
        assert bench.headline() in metric["status_reason"]
        assert "filed slots/m/bench.json" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "omit",
        ["--ruled-by", "--ruled-at", "--ruling-ref", "--tracker", "--reason", "--evidence"],
    )
    def test_a_filing_without_a_ruling_field_exits_usage_and_writes_nothing(
        self, tmp_path, monkeypatch, capsys, omit
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        _seat(store)
        before = sorted(store.list_keys())

        assert main(self._argv(tmp_path, omit=omit)) == USAGE_EXIT_CODE

        assert omit in capsys.readouterr().err
        assert sorted(store.list_keys()) == before, "no record and no manifest"

    def test_a_filing_with_no_champion_pointer_is_refused(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        assert main(self._argv(tmp_path)) == USAGE_EXIT_CODE
        assert not LocalStore(tmp_path).exists(slot_bench_key("m"))

    def test_a_dry_run_prints_the_record_and_writes_nothing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        _seat(store)
        before = sorted(store.list_keys())

        assert main(self._argv(tmp_path, "--dry-run")) == 0

        out = capsys.readouterr().out
        assert '"disposition": "no_lift"' in out
        assert "would file slots/m/bench.json" in out
        assert sorted(store.list_keys()) == before

    def test_an_existing_bench_is_replaced_only_on_request(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        _seat(store)
        assert main(self._argv(tmp_path)) == 0
        first = read_slot_bench(store, "m")

        assert main(self._argv(tmp_path)) == USAGE_EXIT_CODE
        assert "--replace" in capsys.readouterr().err
        assert read_slot_bench(store, "m") == first

        assert main(self._argv(tmp_path, "--replace", "--reexam", "2026-12-18")) == 0
        replaced = read_slot_bench(store, "m")
        assert replaced is not None
        assert replaced.digest != first.digest, "a re-filed bench is a new identity"


def test_the_published_schema_states_the_ruling_fields_as_required() -> None:
    """A second reader of `slot_bench.v1` with no Python import gets the same
    refusal the writer makes."""
    from crucible.bench import _SCHEMA_PATH

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == SLOT_BENCH_SCHEMA_VERSION
    for field in ("ruled_by", "ruled_at", "ruling_ref", "tracker", "evidence", "reexam_date"):
        assert field in schema["required"]
    assert schema["properties"]["weight"] == {**schema["properties"]["weight"], "const": 0}
    assert dt.date.fromisoformat(EFFECTIVE)
