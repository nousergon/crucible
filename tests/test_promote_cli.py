"""`crucible promote --slot <s>` end to end, through the runner.

Normative sources: plan §4.1 (one entry point), §4.2 (every job writes a
manifest), §4.4; `champion-challenger-policy.md` §5, §6, §11.

The CLI test is where the three artifacts are asserted together — the
`arena_cycle`, the pointer and the retirement log — because the failure mode
they exist against is a job that writes one of them and reports success.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from nousergon_lib.arena import ArmRegister, ArmSeries

from crucible.arena_io import arena_cycle_key
from crucible.champion import champion_key, read_champion
from crucible.cli import main
from crucible.promote import (
    arm_register_key,
    arm_series_key,
    load_slot_inputs,
    retirement_log_key,
)
from crucible.store import LocalStore
from tests.support.panels import trading_days

DAY = "2026-08-28"


@pytest.fixture
def seeded(tmp_path):
    """A slot with three registered arms and 40 paired trading days each."""
    store = LocalStore(tmp_path)
    dates = trading_days(40, dt.date.fromisoformat(DAY))
    register = ArmRegister()
    ids: dict[str, str] = {}
    for name, value in (("champ", 0.0), ("chal", 0.045), ("laggard", -0.02)):
        register, record = register.register(
            slot="m", name=name, spec={"name": name}, created_date=dates[0]
        )
        ids[name] = record.arm_id
        series = ArmSeries(arm_id=record.arm_id, scores={d: value for d in dates})
        store.put_bytes(
            arm_series_key("m", record.arm_id),
            json.dumps({"arm_id": record.arm_id, "scores": series.scores, "misses": []}).encode(),
        )
    store.put_bytes(
        arm_register_key("m"),
        b"".join(json.dumps(e).encode() + b"\n" for e in register.to_dicts()),
    )
    return store, register, ids, dates


def seat(store, ids, dates, arm: str = "champ") -> None:
    """Seat an incumbent, as `migrate.history` will for the real slots.

    Deliberately `operator_bootstrap`: that is R's real state since
    2026-07-13, and starting the tests from it means the first evidence-won
    promotion is exercised as the transition it actually is.
    """
    from crucible.champion import ChampionPointer, read_champion_etag, write_champion

    prior = dates[-2]
    manifest_key = f"runs/promote/{prior}/run.json"
    store.put_bytes(
        manifest_key,
        json.dumps({"status": "ok", "job": "promote", "trading_day": prior, "reason": ""}).encode(),
    )
    write_champion(
        store,
        ChampionPointer(
            slot="m",
            arm_id=ids[arm],
            as_of=prior,
            decided_at="2026-08-27T21:00:00Z",
            run_id="0" * 26,
            code_sha="a" * 40,
            promotion_source="operator_bootstrap",
            manifest_key=manifest_key,
            evidence={"operator": "cipher813", "reason": "seeded by migrate.history"},
        ),
        expected=read_champion_etag(store, "m"),
    )


class TestLoadSlotInputs:
    def test_the_register_and_every_series_round_trip(self, seeded) -> None:
        store, register, ids, _ = seeded
        loaded = load_slot_inputs(store, "m")
        assert set(loaded.register.all_arms()) == set(register.all_arms())
        assert set(loaded.series_by_arm) == set(ids.values())

    def test_a_registered_arm_with_no_series_is_refused(self, seeded) -> None:
        store, _, ids, _ = seeded
        # Remove one series: the engine's own contract is that a missing
        # series is a defect, not an omission, and the loader must not
        # quietly present a smaller cohort to it.
        (store.root / arm_series_key("m", ids["laggard"])).unlink()
        with pytest.raises(KeyError, match="laggard"):
            load_slot_inputs(store, "m")


class TestPromoteCommand:
    def test_it_writes_the_cycle_the_pointer_and_the_retirement_log(
        self, seeded, monkeypatch
    ) -> None:
        store, _, ids, dates = seeded
        seat(store, ids, dates)
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))

        assert main(["promote", "--slot", "m", "--date", DAY]) == 0

        cycle = json.loads(store.get_bytes(arena_cycle_key("m", DAY)))
        assert cycle["slot"] == "m"
        assert cycle["decision"]["champion"] == ids["chal"]

        pointer = read_champion(store, "m")
        assert pointer.arm_id == ids["chal"]
        assert pointer.promotion_source == "evidence"

        log = store.get_bytes(retirement_log_key("m")).decode().splitlines()
        assert len(log) == 3, "every ACTIVE arm gets a verdict, survivors included"

        manifest = json.loads(store.get_bytes(f"runs/promote/{DAY}/run.json"))
        assert manifest["status"] == "ok"
        assert any(o["key"] == champion_key("m") for o in manifest["outputs"])

    def test_a_cold_slot_bootstraps_and_says_so(self, seeded, monkeypatch) -> None:
        """Policy §9.1 cold start: no incumbent, an arm must serve. The
        pointer records `bootstrap`, not `evidence`, so the first
        evidence-won promotion stays visible as such (§11)."""
        store, _, ids, _ = seeded
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        assert main(["promote", "--slot", "m", "--date", DAY]) == 0
        pointer = read_champion(store, "m")
        assert pointer.promotion_source == "bootstrap"
        assert pointer.arm_id == ids["chal"]

    def test_a_dry_run_decides_and_writes_nothing(self, seeded, monkeypatch) -> None:
        """alpha-engine-config-I9922: before `run_job` gained `dry_run`, this
        call still filed an `ok` run manifest for a promotion that never
        happened — "writes nothing" was true of the pointer and the arena
        cycle but false of `runs/promote/{DAY}/run.json`. Covered here
        alongside the pointer/cycle assertions rather than as a bare
        "no manifest" check, since a dry run that decided nothing writing no
        manifest is a vacuous pass."""
        from crucible.manifest import manifest_key

        store, _, _, _ = seeded
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        assert main(["promote", "--slot", "m", "--date", DAY, "--dry-run"]) == 0
        assert not store.exists(champion_key("m"))
        assert not store.exists(arena_cycle_key("m", DAY))
        assert not store.exists(manifest_key("promote", DAY))

    def test_revert_records_the_operator(self, seeded, monkeypatch) -> None:
        store, _, ids, dates = seeded
        seat(store, ids, dates, arm="chal")
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        assert (
            main(
                [
                    "promote",
                    "--slot",
                    "m",
                    "--date",
                    DAY,
                    "--revert-to",
                    ids["champ"],
                    "--reason",
                    "challenger degraded live",
                ]
            )
            == 0
        )
        pointer = json.loads(store.get_bytes(champion_key("m")))
        assert pointer["arm_id"] == ids["champ"]
        assert pointer["promotion_source"] == "operator_bootstrap"
        assert pointer["evidence"]["reason"] == "challenger degraded live"

    def test_a_revert_under_dry_run_refuses_rather_than_reverting_for_real(
        self, seeded, monkeypatch
    ) -> None:
        """alpha-engine-config-I9922 N1 / I9935. Before this fix,
        `--revert-to --dry-run` silently performed the revert for real —
        `--dry-run` had no effect on that path at all (`cli.py::_promote`'s
        `job()` called `revert_champion(store=store, ...)` unconditionally).
        The interim guarantee (I9922) refused the pointer write at the
        read-only store; I9935 closes the gap properly: the combination is
        refused UP FRONT, with a message naming both flags, before the store
        is opened at all. `_resolve_store` is patched to blow up so the test
        proves "before touching the store" rather than inferring it."""
        import crucible.cli as cli_module

        store, _, ids, dates = seeded
        seat(store, ids, dates, arm="chal")
        before = json.loads(store.get_bytes(champion_key("m")))
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))

        def _store_must_not_be_opened(args):
            raise AssertionError("--revert-to --dry-run must be refused before the store opens")

        monkeypatch.setattr(cli_module, "_resolve_store", _store_must_not_be_opened)
        with pytest.raises(SystemExit, match="--revert-to is an operator action") as excinfo:
            main(
                [
                    "promote",
                    "--slot",
                    "m",
                    "--date",
                    DAY,
                    "--revert-to",
                    ids["champ"],
                    "--reason",
                    "challenger degraded live",
                    "--dry-run",
                ]
            )
        # A refusal is non-zero: `SystemExit(str)` carries exit status 1.
        assert excinfo.value.code != 0
        assert "--dry-run" in str(excinfo.value)
        # `seat()` already pointed the champion at "chal"; the assertion that
        # matters is that the REVERT to `ids["champ"]` never landed — the
        # pointer document is byte-identical to what `seat()` wrote.
        assert json.loads(store.get_bytes(champion_key("m"))) == before

    def test_a_revert_without_a_reason_is_refused(self, seeded, monkeypatch) -> None:
        store, _, ids, _ = seeded
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        with pytest.raises(SystemExit):
            main(["promote", "--slot", "m", "--date", DAY, "--revert-to", ids["champ"]])
