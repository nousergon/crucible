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
from tests.support.manifests import write_grade_manifest
from tests.support.panels import trading_days

DAY = "2026-08-28"


#: The arms every fixture below registers. `control_null_m` is §10.1's null
#: control, registered like any other arm.
#:
#: `alpha-engine-config-I9759` gave a slot with no champion a first
#: promotion won on evidence, by substituting the null control as the
#: baseline incumbent BEFORE calling `run_cycle` — that substitution lived
#: inside `crucible.promote.run_promotion`, which called `run_cycle` itself.
#: `alpha-engine-config-I10679` removed that call (`promote` now acts on the
#: cycle `experiment.grade` already computed) and `experiment.grade`
#: (`crucible/slots/cycle.py::run_grade`) does not (yet) apply the
#: substitution — see `alpha-engine-config-I10687`, filed to move it there.
#: So `control_null_m` is still registered here (grade still scores and
#: ladders it every cycle, per §10.1), but it no longer changes what a
#: cold slot's `grade()` decides: a slot with no seated champion grades as a
#: raw §9.1 bootstrap either way until I10687 lands.
_ARMS: tuple[tuple[str, float], ...] = (
    ("champ", 0.0),
    ("chal", 0.045),
    ("laggard", -0.02),
    ("control_null_m", 0.0),
)


def _seed(tmp_path, arms: tuple[tuple[str, float], ...]):
    """Register every arm and its series. Does NOT grade — a real slot's
    champion pointer (or absence of one) exists BEFORE grade runs, and
    `grade()` below must read whatever `seat()` did, exactly as
    `crucible.slots.cycle._incumbent` does in production."""
    store = LocalStore(tmp_path)
    dates = trading_days(40, dt.date.fromisoformat(DAY))
    register = ArmRegister()
    ids: dict[str, str] = {}
    for name, value in arms:
        register, record = register.register(
            slot="m",
            name=name,
            spec={"name": name},
            created_date=dates[0],
            filed_on=dates[0],
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


def grade(store, register, ids, dates, *, preconditions: dict | None = None) -> None:
    """The `arena_cycle` (plus its OWN run manifest) `experiment.grade[m]`
    writes an hour before promote.

    `crucible.promote.read_graded_cycle` (`alpha-engine-config-I10679`) acts
    on this cycle's WHOLE decision, computed here through the same
    `run_cycle` call `run_grade` makes — never hand-patched after the fact,
    since a decision recomputed from a real `preconditions` map is what
    `run_grade` actually produces, and a surgically-edited `ineligible` field
    on an unrelated decision was already the wrong fixture shape before this
    issue. `read_graded_cycle` also refuses unless `experiment.grade`'s own
    manifest claims the cycle, so this fixture writes both, exactly as the
    real job does (`crucible/slots/cycle.py::run_grade` via
    `ctx.record_output`, `crucible/runner.py::run_job` for the manifest).

    Reads the CURRENT champion pointer as the incumbent, mirroring
    `crucible.slots.cycle._incumbent` — so a test that calls `seat()` before
    `grade()` gets a cycle decided against that incumbent, and a test that
    does not gets the slot's genuine cold-start.
    """
    import json as _json

    from nousergon_lib.arena.engine import run_cycle

    from crucible.arena_io import write_arena_cycle
    from crucible.champion import champion_key as _champion_key
    from crucible.documents import load_store_document as _load_store_document
    from crucible.slots import get_slot

    incumbent = None
    if store.exists(_champion_key("m")):
        incumbent = _load_store_document(store, _champion_key("m")).get("arm_id")

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
        incumbent=incumbent,
        preconditions=preconditions,
    )
    write_arena_cycle(store, cycle)
    write_grade_manifest(store, "m", DAY)


@pytest.fixture
def seeded(tmp_path):
    """A slot with three registered arms plus the null control, 40 paired
    trading days each — NOT yet graded; a test grades after seating (or not)
    its own incumbent."""
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
        store, register, ids, dates = seeded
        seat(store, ids, dates)
        grade(store, register, ids, dates)
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

    def test_a_cold_slot_bootstrap_cycle_is_refused_not_seated(self, seeded, monkeypatch) -> None:
        """`alpha-engine-config-I9759` gave a cold slot (no incumbent) a
        first champion won on evidence, by substituting §10.1's null control
        as the baseline incumbent inside `crucible.promote.run_promotion`'s
        own (now-removed) `run_cycle` call. `alpha-engine-config-I10679`
        removed that call — `promote` now acts on the cycle
        `experiment.grade` already computed, and grade does not (yet) apply
        the substitution (`alpha-engine-config-I10687`, filed to move it
        there). So a cold slot's graded cycle is a raw §9.1 bootstrap, and
        `promote` refuses it rather than silently reintroducing the
        un-evidenced `promotion_source: bootstrap` pointer I9759 exists to
        prevent — a real, documented regression until I10687 lands.
        """
        store, register, ids, dates = seeded
        grade(store, register, ids, dates)
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        with pytest.raises(PromotionRefused, match="bootstrap"):
            main(["promote", "--slot", "m", "--date", DAY])
        assert not store.exists(champion_key("m"))

    def test_a_cold_slot_with_no_null_control_refuses_rather_than_bootstrapping(
        self, seeded_without_control, monkeypatch
    ) -> None:
        """The one path that would seat a champion on no evidence is a raise.

        Not a `return None`: a writer that quietly declined would leave the
        slot with no pointer and an `ok` manifest, which is indistinguishable
        from a legitimate verdict-backed non-promotion.
        """
        store, register, ids, dates = seeded_without_control
        grade(store, register, ids, dates)
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        with pytest.raises(PromotionRefused, match="bootstrap"):
            main(["promote", "--slot", "m", "--date", DAY])
        assert not store.exists(champion_key("m"))

    def test_an_arm_grade_refused_to_serve_is_not_promoted_by_promote(
        self, seeded, monkeypatch
    ) -> None:
        """`alpha-engine-config-I9759`/`-I10679`: `promote` acts on the
        eligibility `experiment.grade` evaluated — the M slot's §5.3
        behavioural veto and the S slot's contamination attestation, both
        evaluated inside `experiment.grade` and expressible nowhere else.
        Seeded with a REAL incumbent (`champ`) so this test exercises the
        eligibility veto on its own, decoupled from the currently-refused
        cold-start path (`alpha-engine-config-I10687`)."""
        from nousergon_lib.arena import ServingPrecondition

        store, register, ids, dates = seeded
        seat(store, ids, dates, arm="champ")
        grade(
            store,
            register,
            ids,
            dates,
            preconditions={
                ids["chal"]: (
                    ServingPrecondition(
                        name="behavioural_veto",
                        passed=False,
                        reason="an uncomputed gate is not a pass",
                    ),
                )
            },
        )
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        assert main(["promote", "--slot", "m", "--date", DAY]) == 0
        # `chal` is the only arm that leads `champ`, so a veto grade
        # evaluated on it leaves the incumbent HELD — not the verdict-backed
        # non-promotion this used to test as "no champion at all" (which,
        # under the OLD architecture, meant a cold slot; see the class
        # docstring for why that path is decoupled here).
        assert read_champion(store, "m").arm_id == ids["champ"]

    def test_promote_refuses_when_the_graded_cycle_is_absent(self, seeded, monkeypatch) -> None:
        """The arc runs grade at 14:00 and promote at 15:00. An absent
        manifest means grade did not run or did not finish, and a promotion
        decided without it is decided on an artifact nothing vouches for.
        `seeded` no longer grades on its own (`alpha-engine-config-I10679`
        made grading incumbent-dependent, so a test decides when to grade
        relative to `seat()`) — this test simply never calls `grade()`."""
        store, _, _, _ = seeded
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        with pytest.raises(PromotionRefused, match="experiment.grade` run manifest"):
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

        store, register, ids, dates = seeded
        seat(store, ids, dates)
        grade(store, register, ids, dates)
        monkeypatch.setenv("CRUCIBLE_STORE", str(store.root))
        graded = store.get_bytes(arena_cycle_key("m", DAY))
        assert main(["promote", "--slot", "m", "--date", DAY, "--dry-run"]) == 0
        assert read_champion(store, "m").arm_id == ids["champ"], "the SEATED pointer, untouched"
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

    def test_a_revert_does_not_need_score_series(self, seeded, monkeypatch) -> None:
        """2026-09-23: reverting R off an unservable champion was refused with
        a KeyError because no R series existed yet. A revert reads only the
        register; the series are the grading contract, not the operator's."""
        store, _, ids, dates = seeded
        seat(store, ids, dates, arm="chal")
        for arm_id in ids.values():
            (store.root / arm_series_key("m", arm_id)).unlink()
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
                    "champion unservable",
                ]
            )
            == 0
        )
        assert json.loads(store.get_bytes(champion_key("m")))["arm_id"] == ids["champ"]

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
