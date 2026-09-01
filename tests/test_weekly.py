"""The weekly arc is DERIVED from the registry and runs the real CLI.

Normative source: plan §4.1, §4.6, and the defect these tests exist for: six
components declared `schedule: weekly, Saturday` while the scheduler
dispatched exactly one of them, so five were deadlined, watched for absence,
and started by nobody.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.components import Component, Deadline, load_registry
from crucible.slots import SLOTS
from crucible.weekly import ARC_SLOT_JOBS, ArcStageFailed, arc_stages, run_arc

FRIDAY = dt.date(2026, 8, 28)


class TestDerivation:
    def test_the_arc_is_every_row_that_declares_it(self) -> None:
        """Not a list in this file, and not a list in `weekly.py`. Both would
        be written from the jobs someone remembered."""
        registry = load_registry()
        declared = {name for name, row in registry.items() if row.dispatch == "arc"}
        assert {stage.job for stage in arc_stages(FRIDAY)} == declared

    def test_every_weekly_component_is_started_by_something(self) -> None:
        """The defect, as an assertion. A row that is scheduled and dispatched
        by neither the arc nor a scheduler is a page every cycle for work
        nobody was going to run."""
        for name, row in load_registry().items():
            if row.scheduled:
                assert row.dispatch in ("arc", "scheduler"), (
                    f"{name} is scheduled and nothing starts it"
                )

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

    def test_a_slot_scoped_job_expands_to_every_slot_in_dependency_order(self) -> None:
        for job in ARC_SLOT_JOBS:
            slots = [s.slot for s in arc_stages(FRIDAY) if s.job == job]
            assert slots == list(SLOTS), f"{job} must run for every slot, u then r then m then s"

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

        run_arc(FRIDAY, store="/tmp/store", main=fake_main)
        assert seen[0][:5] == ["data.weekly", "--date", FRIDAY.isoformat(), "--store", "/tmp/store"]
        assert all("--date" in argv and FRIDAY.isoformat() in argv for argv in seen)
        slot_calls = [a for a in seen if a[0] in ARC_SLOT_JOBS]
        assert all("--slot" in a for a in slot_calls)

    def test_a_failed_stage_stops_the_arc_and_names_itself(self) -> None:
        """No `continue`, no partial success. The stages after a failure read
        what it was to write, so running them would produce a report card
        built on an absent input."""
        seen: list[str] = []

        def fake_main(argv: list[str]) -> int:
            seen.append(argv[0])
            return 0 if argv[0] == "data.weekly" else 3

        with pytest.raises(ArcStageFailed, match="experiment.run"):
            run_arc(FRIDAY, store=None, main=fake_main)
        assert "report" not in seen, "the arc must not continue past a failed stage"

    def test_a_stage_raising_propagates_rather_than_being_swallowed(self) -> None:
        def fake_main(argv: list[str]) -> int:
            raise RuntimeError("provider_5xx: the source is down")

        with pytest.raises(RuntimeError, match="provider_5xx"):
            run_arc(FRIDAY, store=None, main=fake_main)


class TestDeadlineOfTheArcItself:
    def test_the_arc_deadline_is_later_than_every_stage_it_runs(self) -> None:
        """An arc deadline earlier than one of its own stages would page for a
        run that was still working."""
        registry = load_registry()
        arc_due = registry["weekly"].deadline
        assert isinstance(arc_due, Deadline)
        latest = max(s.due_at for s in arc_stages(FRIDAY, registry))
        assert arc_due.due_at(FRIDAY) > latest
