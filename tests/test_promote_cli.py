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
    PromotionRefused,
    arm_register_key,
    arm_series_key,
    load_slot_inputs,
    retirement_log_key,
)
from crucible.store import LocalStore
from tests.support.panels import trading_days

DAY = "2026-08-28"


#: The arms every fixture below registers. `control_null_m` is §10.1's null
#: control and is registered like any other arm: since
#: `alpha-engine-config-I9759` it is the BASELINE a slot with no champion
#: measures its first promotion against, so a fixture without it exercises
#: the refusal rather than the promotion path.
_ARMS: tuple[tuple[str, float], ...] = (
    ("champ", 0.0),
    ("chal", 0.045),
    ("laggard", -0.02),
    ("control_null_m", 0.0),
)


def _seed(tmp_path, arms: tuple[tuple[str, float], ...]):
    store = LocalStore(tmp_path)
    dates = trading_days(40, dt.date.fromisoformat(DAY))
    register = ArmRegister()
    ids: dict[str, str] = {}
    for name, value in arms:
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
    grade(store, register, ids, dates)
    return store, register, ids, dates


def grade(store, register, ids, dates, *, ineligible: dict[str, list] | None = None) -> None:
    """The `arena_cycle` `experiment.grade[m]` writes an hour before promote.

    `promote` reads its `decision.ineligible` for the eligibility grade
    evaluated (`alpha-engine-config-I9759`); without this artifact promote
    refuses, because the M behavioural veto and the S contamination
    attestation exist nowhere else and a promotion decided without them is a
    promotion decided with every slot-evaluated veto silently empty.
    """
    import json as _json

    from nousergon_lib.arena.engine import run_cycle

    from crucible.arena_io import arena_cycle_key as _key
    from crucible.arena_io import write_arena_cycle
    from crucible.slots import get_slot

    series_by_arm = {
        arm_id: ArmSeries(
            arm_id=arm_id,
            scores=_json.loads(store.get_bytes(arm_series_key("m", arm_id)))["scores"],
        )
        for arm_id in ids.values()
    }
    cycle = run_cycle(
        config=get_slot("m").arena,
        as_of=DAY,
        register=register,
        series_by_arm=series_by_arm,
        incumbent=None,
    )
    payload = cycle.to_dict()
    if ineligible is not None:
        payload["decision"]["ineligible"] = ineligible
        store.put_bytes(_key("m", DAY), _json.dumps(payload).encode())
        return
    write_arena_cycle(store, cycle)


@pytest.fixture
def seeded(tmp_path):
    """A slot with three registered arms plus the null control, 40 paired
    trading days each, and the graded cycle promote reads."""
    return _seed(tmp_path, _ARMS)


@pytest.fixture
def seeded_without_control(tmp_path):
    """The same slot with §10.1's null control absent — the cold-start case."""
    return _seed(tmp_path, tuple(a for a in _ARMS if not a[0].startswith("control_")))


def seat(store, ids, dates, arm: str = "champ") -> None:
    """Seat an incumbent, as `migrate.history` will for the real slots.

    Deliberately `operator_bootstrap`: that is R's real state since
    2026-07-13, and starting the tests from it means the first evidence-won
    promotion is exercised as the transition it actually is.
    """
    from crucible.champion import ChampionPointer, read_champion_etag, write_champion

    prior = dates[-2]
    manifest_key = f"runs/promote/{prior}/m/run.json"
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
        assert len(log) == len(_ARMS), "every ACTIVE arm gets a verdict, survivors included"

        # Discriminated by slot (`alpha-engine-config-I9759`): four slots,
        # one job name, one trading day, four writers — and once promote is
        # an arc stage all four run every Saturday, so the bare key would
        # mean the last slot to finish erased the other three.
        manifest = json.loads(store.get_bytes(f"runs/promote/{DAY}/m/run.json"))
        assert manifest["status"] == "ok"
        assert any(o["key"] == champion_key("m") for o in manifest["outputs"])

    def test_a_cold_slot_wins_its_first_champion_on_evidence(self, seeded, monkeypatch) -> None:
        """`alpha-engine-config-I9759`: no incumbent, and the pointer still
        records `evidence`.

        §10.1's null control stands in as the baseline, so the engine's
        ordinary `decided` path runs — the same paired window, the same
        `promote_min_weeks`, the same `promote_evidence` — and the first
        champion is one the system won. Before this, a cold slot took the
        library's §9.1 cold start and wrote `promotion_source: bootstrap`,
        which the §6 phase-3 gate rejects by name: the FIRST pointer of every
        slot was, by construction, the one kind that does not count.
        """
        store, _, ids, _ = seeded
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        assert main(["promote", "--slot", "m", "--date", DAY]) == 0
        pointer = read_champion(store, "m")
        assert pointer.promotion_source == "evidence"
        assert pointer.arm_id == ids["chal"]
        # The baseline is NAMED, so a reader can tell "beat the incumbent"
        # from "beat noise" — materially different claims.
        assert pointer.evidence["baseline_control"] == ids["control_null_m"]
        assert pointer.evidence["had_incumbent"] is False

    def test_the_baseline_control_is_never_itself_promoted(self, seeded, monkeypatch) -> None:
        """The null control is exempted from the control veto so it can BE
        the incumbent; that exemption must not make it promotable."""
        store, _, ids, _ = seeded
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        assert main(["promote", "--slot", "m", "--date", DAY]) == 0
        assert read_champion(store, "m").arm_id != ids["control_null_m"]

    def test_a_cold_slot_with_no_null_control_refuses_rather_than_bootstrapping(
        self, seeded_without_control, monkeypatch
    ) -> None:
        """The one path that would seat a champion on no evidence is a raise.

        Not a `return None`: a writer that quietly declined would leave the
        slot with no pointer and an `ok` manifest, which is indistinguishable
        from a legitimate verdict-backed non-promotion.
        """
        store, _, _, _ = seeded_without_control
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        with pytest.raises(PromotionRefused, match="cold-start"):
            main(["promote", "--slot", "m", "--date", DAY])
        assert not store.exists(champion_key("m"))

    def test_an_arm_grade_refused_to_serve_is_not_promoted_by_promote(
        self, tmp_path, monkeypatch
    ) -> None:
        """`alpha-engine-config-I9759`: promote runs the SAME engine over the
        SAME series as `experiment.grade`, and until this fix it ran it with
        `preconditions=None` — so the M slot's §5.3 behavioural veto and the
        S slot's contamination attestation, both evaluated inside
        `experiment.grade` and expressible nowhere else, did not exist for
        the job that moves the pointer. Grade would refuse to serve an arm
        and promote would serve it, from the same series, in the same hour.
        """
        store, register, ids, dates = _seed(tmp_path, _ARMS)
        grade(
            store,
            register,
            ids,
            dates,
            ineligible={
                ids["chal"]: [
                    {
                        "name": "behavioural_veto",
                        "passed": False,
                        "reason": "an uncomputed gate is not a pass",
                    }
                ]
            },
        )
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        assert main(["promote", "--slot", "m", "--date", DAY]) == 0
        # `chal` is the only arm that leads the null-control baseline, so a
        # veto grade evaluated on it leaves the slot with NO champion — the
        # verdict-backed non-promotion. Before the fix the pointer moved to
        # `chal`, the arm grade had just refused to serve.
        assert not store.exists(champion_key("m"))

    def test_promote_refuses_when_the_graded_cycle_is_absent(self, seeded, monkeypatch) -> None:
        """The arc runs grade at 14:00 and promote at 15:00. An absent cycle
        means grade did not finish, and a promotion decided without it is
        decided with every slot-evaluated veto empty."""
        from crucible.arena_io import arena_cycle_key

        store, _, _, _ = seeded
        (store.root / arena_cycle_key("m", DAY)).unlink()
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        with pytest.raises(PromotionRefused, match="no graded arena cycle"):
            main(["promote", "--slot", "m", "--date", DAY])

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
        graded = store.get_bytes(arena_cycle_key("m", DAY))
        assert main(["promote", "--slot", "m", "--date", DAY, "--dry-run"]) == 0
        assert not store.exists(champion_key("m"))
        # The cycle promote READS is untouched; the one it would have written
        # over it was not written (`run_promotion(store=None)`).
        assert store.get_bytes(arena_cycle_key("m", DAY)) == graded
        assert not store.exists(manifest_key("promote", DAY, discriminator="m"))

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
