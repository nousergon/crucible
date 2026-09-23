"""The weekly arc is DERIVED from the registry and runs the real CLI.

Normative source: plan §4.1, §4.6, and the defect these tests exist for: six
components declared `schedule: weekly, Saturday` while the scheduler
dispatched exactly one of them, so five were deadlined, watched for absence,
and started by nobody.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from crucible.components import DISPATCHES, Component, Deadline, load_registry
from crucible.keys import champion_key
from crucible.runmode import RUN_MODE_ENV, RUN_MODE_LIVE, RUN_MODE_REPLAY
from crucible.slots import SLOTS, dispatchable_slots, get_slot
from crucible.slots.arms import control_specs, load_arm_specs
from crucible.slots.producibility import UnproducibleChampionError
from crucible.store import LocalStore
from crucible.weekly import ARC_SLOT_JOBS, ARCTIC_LIBRARY_JOBS, ArcStageFailed, arc_stages, run_arc

FRIDAY = dt.date(2026, 8, 28)


@pytest.fixture(autouse=True)
def _a_store_the_preflight_can_read(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_arc` reads every dispatched slot's champion pointer before stage 1
    (`alpha-engine-config-I11085`), from `--store` or `$CRUCIBLE_STORE` exactly
    as each stage would. The tests below that pass `store=None` exercise argv
    shape, not the store, so they get an EMPTY store: no pointer, nothing to
    refuse. The check itself is exercised against real stores in
    `TestChampionsAreProducibleBeforeStageOne`."""
    monkeypatch.setenv("CRUCIBLE_STORE", str(tmp_path / "empty-v2-store"))
    monkeypatch.delenv("CRUCIBLE_STRATEGY_DIR", raising=False)


def _expected_stage_count() -> int:
    """How many stages the arc must dispatch, derived from the registry.

    Unscoped rows count once; a row in `ARC_SLOT_JOBS` counts once per
    dispatchable slot. Nothing here is a literal: a row joining or leaving
    `dispatch: arc` changes this number without an edit, which is the whole
    point of `crucible.weekly`'s derivation.
    """
    arc_rows = [
        name
        for name, row in load_registry().items()
        if row.dispatch == "arc" and row.lifecycle == "ACTIVE"
    ]
    scoped = [name for name in arc_rows if name in ARC_SLOT_JOBS]
    return (len(arc_rows) - len(scoped)) + len(scoped) * len(dispatchable_slots())


class TestDerivation:
    def test_the_arc_is_every_row_that_declares_it(self) -> None:
        """Not a list in this file, and not a list in `weekly.py`. Both would
        be written from the jobs someone remembered."""
        registry = load_registry()
        declared = {name for name, row in registry.items() if row.dispatch == "arc"}
        assert {stage.job for stage in arc_stages(FRIDAY)} == declared

    def test_every_weekly_component_is_started_by_something(self) -> None:
        """The defect, as an assertion. A row that is scheduled and dispatched
        by nothing in the exhaustive vocabulary is a page every cycle for work
        nobody was going to run.

        Derived from `DISPATCHES`, not restated: a fourth value added to the
        vocabulary and wired to nothing would otherwise fail here for the
        wrong reason, and a value removed from it would silently keep passing.
        """
        for name, row in load_registry().items():
            if row.scheduled:
                assert row.dispatch in DISPATCHES, f"{name} is scheduled and nothing starts it"

    def test_the_order_is_the_deadline_order(self) -> None:
        """The deadline table IS the dependency order — data at 09:00 is what
        the 12:00 scoring reads — so deriving from it means one declaration
        rather than two that can disagree."""
        stages = arc_stages(FRIDAY)
        assert [s.due_at for s in stages] == sorted(s.due_at for s in stages)
        names = [s.job for s in stages]
        assert names.index("data.weekly") < names.index("experiment.run")
        assert names.index("experiment.grade") < names.index("report")
        assert names.index("report") < names.index("console")

    def test_a_slot_scoped_job_expands_to_every_dispatchable_slot_in_dependency_order(
        self,
    ) -> None:
        """Every slot the CLI can run, in `SLOTS` order — and no other.

        Re-stated with both the M cycle job (`alpha-engine-config-I9957`) and
        the S cycle job (`-I10512`). Until they existed, M and S had no
        `produce`/`grade` and expanding over all of `SLOTS` made every arc
        fail at `experiment.run[m]` (measured 2026-09-04); both now have
        both, so both are stages — which is the derivation working, and is
        exactly why this pin is an equality. The order is `SLOTS` order, the
        data dependency: the universe cut feeds the signal, the signal feeds
        the model, the model feeds the strategy.
        """
        expected = [slot for slot in SLOTS if slot in dispatchable_slots()]
        assert expected == ["u", "r", "m", "s"], expected
        for job in ARC_SLOT_JOBS:
            slots = [s.slot for s in arc_stages(FRIDAY) if s.job == job]
            assert slots == expected, f"{job} must run for every dispatchable slot, in SLOTS order"

    def test_a_slot_without_entry_points_is_not_a_stage(self, monkeypatch) -> None:
        """The derivation is real: take `grade` away from a dispatchable slot's
        module and its stages leave the arc — no list to edit anywhere."""
        from crucible.slots import research

        monkeypatch.delattr(research, "grade")
        assert "r" not in dispatchable_slots()
        slots = {s.slot for s in arc_stages(FRIDAY) if s.job in ARC_SLOT_JOBS}
        assert slots == {"u", "m", "s"}

    def test_the_dispatch_table_and_the_arc_read_one_source(self) -> None:
        """`experiment.run --slot s` refuses by name (track A) and the arc
        never asks for it: both derive from `dispatchable_slots`, so the arc
        cannot schedule a stage the CLI will refuse."""
        from crucible.track_a import _SLOT_MODULES

        assert set(_SLOT_MODULES) == set(dispatchable_slots())
        assert set(_SLOT_MODULES) == {s.slot for s in arc_stages(FRIDAY) if s.slot}

    def test_an_empty_arc_raises_rather_than_reporting_a_quiet_week(self) -> None:
        """A registry in which nothing declares `dispatch: arc` would make the
        weekly job succeed having run nothing at all."""
        registry = {
            name: Component(
                name=name,
                description="x",
                lifecycle="ACTIVE",
                signals={},
                log_location="/x",
                log_retention_days=1,
                alert_channel="none",
                console_surface="x",
                artifact_retention="forever",
                schedule=None,
                deadline=None,
                dispatch=None,
            )
            for name in ("a", "b")
        }
        with pytest.raises(ValueError, match="dispatch: arc"):
            arc_stages(FRIDAY, registry)

    def test_a_retired_arc_row_is_not_run(self) -> None:
        """DISABLED and RETIRED are declared, never inferred — and a retired
        stage that still ran would be work nobody is watching."""
        registry = dict(load_registry())
        victim = next(n for n, c in registry.items() if c.dispatch == "arc")
        row = registry[victim]
        registry[victim] = Component(
            name=row.name,
            description=row.description,
            lifecycle="RETIRED",
            signals=row.signals,
            log_location=row.log_location,
            log_retention_days=row.log_retention_days,
            alert_channel=row.alert_channel,
            console_surface=row.console_surface,
            artifact_retention=row.artifact_retention,
            schedule=row.schedule,
            deadline=row.deadline,
            dispatch=row.dispatch,
        )
        assert victim not in {s.job for s in arc_stages(FRIDAY, registry)}


class TestRunsTheRealCommand:
    def test_the_default_runner_is_the_cli_entry_point(self) -> None:
        """`run_arc`'s injectable `main` is for tests only. If the default
        were anything but `crucible.cli.main`, every test below would be
        testing a parallel path that can be healthy while the real one is
        broken — which is the whole reason the arc exists to be measured."""
        import inspect

        from crucible import cli

        source = inspect.getsource(run_arc)
        assert "from crucible.cli import main as cli_main" in source
        assert callable(cli.main)

    def test_each_stage_is_invoked_as_the_argv_an_operator_would_type(self) -> None:
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        run_arc(FRIDAY, store="/tmp/store", run_mode=RUN_MODE_LIVE, main=fake_main)
        assert seen[0][:7] == [
            "data.weekly",
            "--date",
            FRIDAY.isoformat(),
            "--run-mode",
            RUN_MODE_LIVE,
            "--store",
            "/tmp/store",
        ]
        assert all("--date" in argv and FRIDAY.isoformat() in argv for argv in seen)
        # Every stage, not just the first: each one re-enters `crucible.cli.main`
        # and resolves its own mode, so a stage missing the flag falls back to
        # the environment and can disagree with the arc that launched it.
        for argv in seen:
            assert "--run-mode" in argv, argv
            assert argv[argv.index("--run-mode") + 1] == RUN_MODE_LIVE, argv
        slot_calls = [a for a in seen if a[0] in ARC_SLOT_JOBS]
        assert all("--slot" in a for a in slot_calls)

    def test_a_replay_arc_names_no_stage_live_even_when_the_environment_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The class, not the instance.

        A replay arc launched on a box whose environment declares `live` — the
        production shape, since the dispatcher exports `CRUCIBLE_RUN_MODE=live`
        — must not put `live` on any stage argv. Every stage would otherwise
        resolve `live` from that environment and file twelve LIVE manifests
        under an arc manifest saying `replay`, and `_clause_replays_ok` reads
        the STAGE manifests: §6.1's five replayed Saturdays would each file
        twelve false-live runs.
        """
        monkeypatch.setenv(RUN_MODE_ENV, RUN_MODE_LIVE)
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        run_arc(FRIDAY, store=None, run_mode=RUN_MODE_REPLAY, main=fake_main)
        assert seen, "the arc ran no stages, so this asserts nothing"
        for argv in seen:
            assert RUN_MODE_LIVE not in argv, argv
            assert argv[argv.index("--run-mode") + 1] == RUN_MODE_REPLAY, argv

    def test_a_dry_run_arc_puts_dry_run_on_every_stage(self) -> None:
        """alpha-engine-config-I9922 R3-1 (independent review, 2026-09-03):
        `Stage.argv(dry_run=)` / `run_arc(dry_run=)` had ZERO coverage —
        deleting `argv.append("--dry-run")` (`weekly.py::Stage.argv`) left
        this suite green. `--dry-run` must reach every stage's own argv, not
        just the arc's own `run_job` call: each stage re-enters
        `crucible.cli.main` as its own invocation with no shared `args`
        object, so a stage missing the flag would run for real regardless of
        what `weekly --dry-run` itself did."""
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        run_arc(FRIDAY, store="/tmp/store", run_mode=RUN_MODE_REPLAY, main=fake_main, dry_run=True)
        # One stage per unscoped arc row + one per ARC_SLOT_JOB x every
        # DISPATCHABLE slot. BOTH terms derived (`alpha-engine-config-I10961`):
        # the unscoped count was the literal `6` and went stale the moment
        # `migrate.history` joined the arc at 14:30 — the arithmetic it was
        # meant to spare an editor is exactly the arithmetic that broke, and a
        # count written from the rows someone remembered is the class
        # `weekly.py`'s own header names.
        assert len(seen) == _expected_stage_count()
        for argv in seen:
            assert "--dry-run" in argv, argv
            assert "--run-mode" in argv, argv
            assert argv[argv.index("--run-mode") + 1] == RUN_MODE_REPLAY, argv

    def test_a_real_arc_puts_dry_run_on_no_stage(self) -> None:
        """The converse of the test above — `dry_run=False` (the default)
        must not leak `--dry-run` onto any stage's argv either."""
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        run_arc(FRIDAY, store="/tmp/store", run_mode=RUN_MODE_REPLAY, main=fake_main, dry_run=False)
        assert len(seen) == _expected_stage_count()
        for argv in seen:
            assert "--dry-run" not in argv, argv

    def test_the_explain_stage_carries_select_newest_verdict_and_no_target(self) -> None:
        """`alpha-engine-config-I10858`: the arc names no operator target for
        `explain` — it names the flag that makes the selection deterministic
        instead (`crucible.explain.select_newest_settled_verdict`)."""
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        run_arc(FRIDAY, store="/tmp/store", run_mode=RUN_MODE_REPLAY, main=fake_main)
        explain_calls = [a for a in seen if a[0] == "explain"]
        assert len(explain_calls) == 1, explain_calls
        argv = explain_calls[0]
        assert "--select-newest-verdict" in argv, argv
        # No bare positional target: every other token is either the job
        # name, a flag, or that flag's own value.
        flags = {"--date", "--run-mode", "--store", "--select-newest-verdict", "--dry-run"}
        i = 1
        while i < len(argv):
            token = argv[i]
            assert token in flags, f"unexpected positional {token!r} in {argv}"
            if token != "--select-newest-verdict" and token != "--dry-run":
                i += 1  # skip the flag's value
            i += 1

    def test_a_failed_stage_stops_the_arc_and_names_itself(self) -> None:
        """No `continue`, no partial success. The stages after a failure read
        what it was to write, so running them would produce a report card
        built on an absent input."""
        seen: list[str] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv[0])
            return 0 if argv[0] == "data.weekly" else 3

        # DERIVED, not named: the arc's shape is the deadline table, and a
        # new stage inserted between `data.weekly` and the one this test used
        # to name would make it assert against a stage the arc never reached
        # — green while covering nothing. `experiment.register` was inserted
        # at exactly that position (`alpha-engine-config-I10927`).
        first_failing = next(s for s in arc_stages(FRIDAY) if s.job != "data.weekly")

        with pytest.raises(ArcStageFailed, match=re.escape(first_failing.label)):
            run_arc(FRIDAY, store=None, run_mode=RUN_MODE_LIVE, main=fake_main)
        assert "report" not in seen, "the arc must not continue past a failed stage"

    def test_a_stage_raising_propagates_rather_than_being_swallowed(self) -> None:
        def fake_main(argv: list[str]) -> int:
            raise RuntimeError("provider_5xx: the source is down")

        with pytest.raises(RuntimeError, match="provider_5xx"):
            run_arc(FRIDAY, store=None, run_mode=RUN_MODE_LIVE, main=fake_main)


class TestArcticLibraryThreading:
    """alpha-engine-config-I10633: `weekly` could not be exercised against the
    dedicated ArcticDB library because no stage's argv ever carried
    `--arctic-library` — an arc run inside `tests/integration/` would
    silently fall through to the PRODUCTION `universe` library. Additive and
    production-inert (`Stage.argv`'s own docstring): absent, argv is
    unchanged."""

    def test_stage_argv_omits_the_flag_when_no_library_is_given(self) -> None:
        stages = arc_stages(FRIDAY)
        weekly_stage = next(s for s in stages if s.job == "data.weekly")
        argv = weekly_stage.argv(trading_day=FRIDAY, store=None, run_mode=RUN_MODE_LIVE)
        assert "--arctic-library" not in argv, argv

    def test_stage_argv_appends_the_flag_for_an_arctic_library_job(self) -> None:
        stages = arc_stages(FRIDAY)
        weekly_stage = next(s for s in stages if s.job == "data.weekly")
        argv = weekly_stage.argv(
            trading_day=FRIDAY, store=None, run_mode=RUN_MODE_LIVE, arctic_library="integration-lib"
        )
        assert argv[argv.index("--arctic-library") + 1] == "integration-lib", argv

    def test_stage_argv_never_puts_the_flag_on_a_non_arctic_stage(self) -> None:
        """`report` reads no ArcticDB library; a flag that leaked onto every
        stage's argv regardless of `ARCTIC_LIBRARY_JOBS` would make `report`
        refuse on an argument it does not declare."""
        stages = arc_stages(FRIDAY)
        report_stage = next(s for s in stages if s.job == "report")
        argv = report_stage.argv(
            trading_day=FRIDAY, store=None, run_mode=RUN_MODE_LIVE, arctic_library="integration-lib"
        )
        assert "--arctic-library" not in argv, argv

    def test_run_arc_threads_the_library_onto_every_arctic_library_stage_only(self) -> None:
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        run_arc(
            FRIDAY,
            store="/tmp/store",
            run_mode=RUN_MODE_LIVE,
            main=fake_main,
            arctic_library="integration-lib",
        )
        for argv in seen:
            job = argv[0]
            if job in ARCTIC_LIBRARY_JOBS:
                assert argv[argv.index("--arctic-library") + 1] == "integration-lib", argv
            else:
                assert "--arctic-library" not in argv, argv

    def test_run_arc_omits_the_flag_from_every_stage_when_not_given(self) -> None:
        """The default (`arctic_library=None`) — production's own shape —
        must leave every stage's argv byte-for-byte unchanged from before
        this parameter existed."""
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        run_arc(FRIDAY, store="/tmp/store", run_mode=RUN_MODE_LIVE, main=fake_main)
        for argv in seen:
            assert "--arctic-library" not in argv, argv


class TestDeadlineOfTheArcItself:
    def test_the_arc_deadline_is_later_than_every_stage_it_runs(self) -> None:
        """An arc deadline earlier than one of its own stages would page for a
        run that was still working."""
        registry = load_registry()
        arc_due = registry["weekly"].deadline
        assert isinstance(arc_due, Deadline)
        latest = max(s.due_at for s in arc_stages(FRIDAY, registry))
        assert arc_due.due_at(FRIDAY) > latest


# `alpha-engine-config-I11085`: production's R tree in the part that matters.
# `scanner_predictor_direct` ranks on `predicted_alpha_ratio`, which the
# phase-1 feature catalogue declares no producer for, so `experiment.run`
# refuses it at registration; `no_agent_quant` is fully expressible.
_R_RECIPES = {
    "scanner_predictor_direct": (
        "name: scanner_predictor_direct\nslot: r\nranker: predicted_alpha_direct\n"
        "registered_at: '2026-07-13'\nparams:\n  top_n: 10\n"
    ),
    "no_agent_quant": (
        "name: no_agent_quant\nslot: r\nranker: quant_composite\n"
        "registered_at: '2026-06-15'\nparams:\n  top_n: 15\n  momentum_weight: 0.5\n"
        "  trend_weight: 0.3\n  reversal_weight: 0.2\n"
    ),
}


class TestChampionsAreProducibleBeforeStageOne:
    """`alpha-engine-config-I11085`. On 2026-09-19 the arc died at
    `experiment.run[r]` because R's pointer named an arm whose recipe refuses
    in phase 1, and every stage after it died too. The class must fail in
    seconds, before stage 1, naming the slot, the arm and why."""

    @pytest.fixture
    def v2(self, tmp_path) -> LocalStore:
        store = LocalStore(tmp_path / "v2")
        for name, body in _R_RECIPES.items():
            store.put_bytes(f"strategy/current/arms/r/{name}.yaml", body.encode())
        return store

    @staticmethod
    def _arm_id(store: LocalStore, name: str) -> str:
        return next(s.arm_id for s in load_arm_specs("r", store=store) if s.name == name)

    @staticmethod
    def _point_r_at(store: LocalStore, arm_id: str) -> None:
        store.put_bytes(champion_key("r"), json.dumps({"arm_id": arm_id}).encode())

    def _run(self, store: LocalStore, *, dry_run: bool = False) -> list[list[str]]:
        seen: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv)
            return 0

        self.seen = seen
        run_arc(
            FRIDAY,
            store=str(store.root),
            run_mode=RUN_MODE_LIVE,
            main=fake_main,
            dry_run=dry_run,
        )
        return seen

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_a_pointer_at_an_arm_the_release_refuses_fails_before_stage_one(
        self, v2: LocalStore, dry_run: bool
    ) -> None:
        refused = self._arm_id(v2, "scanner_predictor_direct")
        self._point_r_at(v2, refused)
        with pytest.raises(UnproducibleChampionError) as excinfo:
            self._run(v2, dry_run=dry_run)
        assert self.seen == [], "no stage may run once a served slot is known to have no feed"
        message = str(excinfo.value)
        assert "slot 'r'" in message
        assert refused in message
        assert "predicted_alpha_ratio" in message, "the reason must say WHY it cannot produce"
        assert self._arm_id(v2, "no_agent_quant") in message, (
            "the message must name what the operator can re-point to"
        )
        assert "--revert-to" in message

    def test_a_pointer_at_a_producible_arm_runs_every_stage(self, v2: LocalStore) -> None:
        self._point_r_at(v2, self._arm_id(v2, "no_agent_quant"))
        assert len(self._run(v2)) == _expected_stage_count()

    def test_no_pointer_is_not_a_refusal(self, v2: LocalStore) -> None:
        """A slot that has never promoted serves nothing and says so; that is
        `_serve_champion_feed`'s `(None, None)`, not a failure."""
        assert len(self._run(v2)) == _expected_stage_count()

    def test_a_pointer_at_a_control_arm_is_refused(self, v2: LocalStore) -> None:
        control = control_specs(get_slot("r"))[0].arm_id
        self._point_r_at(v2, control)
        with pytest.raises(UnproducibleChampionError, match="control arm"):
            self._run(v2)
        assert self.seen == []

    def test_a_pointer_at_a_superseded_spec_hash_is_refused(self, v2: LocalStore) -> None:
        """The name exists but the recipe now hashes elsewhere: nothing will
        ever write a shadow under the id the pointer names."""
        current = self._arm_id(v2, "no_agent_quant")
        stale = current.rsplit(":", 1)[0] + ":000000000000"
        self._point_r_at(v2, stale)
        with pytest.raises(UnproducibleChampionError, match=re.escape(current)):
            self._run(v2)
        assert self.seen == []

    def test_a_pointer_at_an_arm_no_recipe_declares_is_refused(self, v2: LocalStore) -> None:
        self._point_r_at(v2, "r:thinktank_coverage:557ccb7e7988")
        with pytest.raises(UnproducibleChampionError, match="no recipe named 'thinktank_coverage'"):
            self._run(v2)
        assert self.seen == []
