"""`experiment.retire`: one operator retirement on a named ruling
(`alpha-engine-config-I12765`)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from nousergon_lib.arena import ArmRegister

from crucible.keys import arm_register_key, champion_key, retirement_log_key
from crucible.manifest import manifest_key
from crucible.models import RetirementLogRow
from crucible.retire import RetirementRefused, operator_reason, retire_arm
from crucible.slots.arms import read_register, write_register
from crucible.store import LocalStore

DAY = "2026-10-23"
RULING = "decision-card-2026-10-29"


def _seed(store: LocalStore) -> dict[str, str]:
    register = ArmRegister()
    ids: dict[str, str] = {}
    for name, created, control in (
        ("champ", "2026-06-15", False),
        ("challenger", "2026-07-13", False),
        ("planted", "2026-01-02", True),
    ):
        register, record = register.register(
            "r", name, {"ranker": name}, created, control=control, filed_on=created
        )
        ids[name] = record.arm_id
    write_register(store, "r", register)
    store.put_bytes(champion_key("r"), json.dumps({"arm_id": ids["champ"]}).encode())
    return ids


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "store")


def _log_rows(store: LocalStore) -> list[dict]:
    raw = store.get_bytes(retirement_log_key("r")).decode()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_retires_the_arm_in_the_register_and_logs_the_ruling(store: LocalStore) -> None:
    ids = _seed(store)
    result = retire_arm(
        store, slot="r", arm="challenger", ruling=RULING, reason="Brian's call", retired_date=DAY
    )
    assert result.arm_id == ids["challenger"]
    state = read_register(store, "r").state(ids["challenger"])
    assert state.retired_date == DAY
    assert RULING in state.retired_reason
    (row,) = _log_rows(store)
    assert row["arm_id"] == ids["challenger"]
    assert row["retire"] is True
    assert RULING in row["reason"]
    RetirementLogRow.model_validate({k: v for k, v in row.items() if k != "event_id"})
    assert ids["challenger"] not in read_register(store, "r").active_arms()


def test_a_full_arm_id_resolves_too(store: LocalStore) -> None:
    ids = _seed(store)
    result = retire_arm(
        store, slot="r", arm=ids["challenger"], ruling=RULING, reason="x", retired_date=DAY
    )
    assert result.arm_id == ids["challenger"]


@pytest.mark.parametrize("ruling", ["", "   "])
def test_refuses_without_a_ruling_and_writes_nothing(store: LocalStore, ruling: str) -> None:
    _seed(store)
    before = store.get_bytes(arm_register_key("r"))
    with pytest.raises(RetirementRefused, match="--ruling"):
        retire_arm(store, slot="r", arm="challenger", ruling=ruling, reason="x", retired_date=DAY)
    assert store.get_bytes(arm_register_key("r")) == before
    assert not store.exists(retirement_log_key("r"))


def test_refuses_without_a_reason(store: LocalStore) -> None:
    _seed(store)
    with pytest.raises(RetirementRefused, match="--reason"):
        retire_arm(store, slot="r", arm="challenger", ruling=RULING, reason="", retired_date=DAY)


def test_refuses_the_current_champion(store: LocalStore) -> None:
    _seed(store)
    before = store.get_bytes(arm_register_key("r"))
    with pytest.raises(RetirementRefused, match="current champion"):
        retire_arm(store, slot="r", arm="champ", ruling=RULING, reason="x", retired_date=DAY)
    assert store.get_bytes(arm_register_key("r")) == before
    assert not store.exists(retirement_log_key("r"))


def test_refuses_a_control_arm(store: LocalStore) -> None:
    _seed(store)
    with pytest.raises(RetirementRefused, match="control"):
        retire_arm(store, slot="r", arm="planted", ruling=RULING, reason="x", retired_date=DAY)


def test_refuses_an_unknown_arm(store: LocalStore) -> None:
    _seed(store)
    with pytest.raises(RetirementRefused, match="no arm named"):
        retire_arm(store, slot="r", arm="nope", ruling=RULING, reason="x", retired_date=DAY)


def test_a_rerun_writes_no_second_event_or_row(store: LocalStore) -> None:
    _seed(store)
    kwargs = dict(slot="r", arm="challenger", ruling=RULING, reason="x", retired_date=DAY)
    retire_arm(store, **kwargs)
    register_after_first = store.get_bytes(arm_register_key("r"))
    again = retire_arm(store, **kwargs)
    assert again.register_written is False
    assert store.get_bytes(arm_register_key("r")) == register_after_first
    assert len(_log_rows(store)) == 1


def test_a_retry_after_the_register_write_still_writes_the_log(store: LocalStore) -> None:
    ids = _seed(store)
    register = read_register(store, "r")
    write_register(
        store, "r", register.retire(ids["challenger"], DAY, operator_reason(RULING, "x"))
    )
    result = retire_arm(
        store, slot="r", arm="challenger", ruling=RULING, reason="x", retired_date=DAY
    )
    assert result.register_written is False
    assert len(_log_rows(store)) == 1


def test_refuses_an_arm_retired_for_another_reason(store: LocalStore) -> None:
    ids = _seed(store)
    register = read_register(store, "r")
    write_register(store, "r", register.retire(ids["challenger"], "2026-10-16", "engine verdict"))
    with pytest.raises(RetirementRefused, match="already retired"):
        retire_arm(store, slot="r", arm="challenger", ruling=RULING, reason="x", retired_date=DAY)


def test_dry_run_checks_everything_and_writes_nothing(store: LocalStore) -> None:
    _seed(store)
    before = store.get_bytes(arm_register_key("r"))
    with pytest.raises(RetirementRefused):
        retire_arm(
            store, slot="r", arm="champ", ruling=RULING, reason="x", retired_date=DAY, write=False
        )
    result = retire_arm(
        store,
        slot="r",
        arm="challenger",
        ruling=RULING,
        reason="x",
        retired_date=DAY,
        write=False,
    )
    assert result.register_written is True
    assert store.get_bytes(arm_register_key("r")) == before
    assert not store.exists(retirement_log_key("r"))


class TestTheCli:
    def _argv(self, root: Path, *extra: str) -> list[str]:
        return [
            "experiment.retire",
            "--slot",
            "r",
            "--arm",
            "challenger",
            "--date",
            DAY,
            "--store",
            str(root),
            "--run-mode",
            "live",
            *extra,
        ]

    def test_the_job_retires_and_files_its_manifest(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        ids = _seed(store)
        argv = self._argv(store.root, "--ruling", RULING, "--reason", "retired", "on", "ruling")
        assert main(argv) == 0
        state = read_register(store, "r").state(ids["challenger"])
        assert state.retired_date == DAY
        assert state.retired_reason.endswith("retired on ruling")
        key = manifest_key(
            "experiment.retire", dt.date.fromisoformat(DAY), discriminator="r.challenger"
        )
        document = json.loads(store.get_bytes(key))
        assert document["status"] == "ok", document.get("reason")
        assert {o["key"] for o in document["outputs"]} == {
            arm_register_key("r"),
            retirement_log_key("r"),
        }

    def test_a_dry_run_writes_nothing(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from crucible.cli import main
        from crucible.store import begin_capture, end_capture

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        ids = _seed(store)
        before = sorted(store.list_keys())
        begin_capture()
        try:
            argv = self._argv(store.root, "--ruling", RULING, "--reason", "x", "--dry-run")
            assert main(argv) == 0
        finally:
            end_capture()
        assert sorted(store.list_keys()) == before
        assert ids["challenger"] in capsys.readouterr().out
        assert read_register(store, "r").state(ids["challenger"]).retired_date is None

    def test_the_parser_requires_a_ruling(self, store: LocalStore) -> None:
        from crucible.cli import main

        with pytest.raises(SystemExit):
            main(self._argv(store.root, "--reason", "x"))
