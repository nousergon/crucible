"""The CLI's shape and the alerting surface, both fixed before three tracks
build on them.

Normative source: plan §4.1 (one entry point, thirteen jobs) and §4.6 (two
page conditions, no others).

These test the parts that are real today — the dispatch table, the argument
surface, date resolution, the closed page-condition set — and that the parts
that are not real refuse loudly rather than returning a clean zero.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.alerts import PAGE_CONDITIONS, Page, dedup_key
from crucible.cli import HANDLERS, JOBS, build_parser, is_stub, main, resolve_date

FRIDAY = dt.date(2026, 8, 28)


class TestJobSurface:
    def test_the_thirteen_jobs_of_the_plan_are_registered(self) -> None:
        assert set(JOBS) == {
            "data.daily",
            "data.weekly",
            "data.heal",
            "experiment.new",
            "experiment.run",
            "experiment.grade",
            "promote",
            "report",
            "explain",
            "migrate.history",
            "release.pin",
            "smoke",
        }

    @pytest.mark.parametrize("job", sorted(JOBS))
    def test_every_job_parses(self, job: str) -> None:
        parser = build_parser()
        argv = [job]
        if job in ("experiment.run", "experiment.grade", "promote", "experiment.new"):
            argv += ["--slot", "r"]
        if job in ("experiment.run", "experiment.new"):
            argv += ["--arm", "arm_abc"]
        if job == "explain":
            argv += ["01JG0000000000000000000000"]
        if job == "release.pin":
            argv += ["a" * 40]
        if job == "data.heal":
            argv += ["--gap", "2026-08-28"]
        assert parser.parse_args(argv).job == job

    def test_an_unknown_job_exits_rather_than_defaulting(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["data.hourly"])

    def test_no_job_at_all_exits(self) -> None:
        """A bare `crucible` must not pick a job. Guessing here is how a
        laptop invocation runs the weekly against production."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_every_job_has_a_handler(self) -> None:
        """The guard below only covers stubs, so this covers the rest: a job
        in JOBS with no handler at all would silently drop out of both."""
        assert set(HANDLERS) == set(JOBS)

    @pytest.mark.parametrize("job", sorted(j for j in JOBS if is_stub(HANDLERS[j])))
    def test_an_unimplemented_job_raises_and_never_returns_zero(self, job: str) -> None:
        """A stub that exits 0 is indistinguishable from a job that ran and
        had nothing to do — the exact shape §11 says agent-built systems
        drift toward.

        The parametrisation reads `is_stub` off the handler, so a track
        landing an implementation flips this by IMPLEMENTING, with no
        exclusion list for anyone to remember to edit."""
        argv = [job]
        if job in ("experiment.run", "experiment.grade", "promote", "experiment.new"):
            argv += ["--slot", "r"]
        if job in ("experiment.run", "experiment.new"):
            argv += ["--arm", "arm_abc"]
        if job == "explain":
            argv += ["01JG0000000000000000000000"]
        if job == "release.pin":
            argv += ["a" * 40]
        if job == "data.heal":
            argv += ["--gap", "2026-08-28"]
        with pytest.raises(NotImplementedError, match="I9757"):
            main(argv)


class TestDateResolution:
    def test_an_omitted_date_resolves_to_the_last_session(self) -> None:
        assert resolve_date(None, now=dt.datetime(2026, 8, 29, 10, 0)) == FRIDAY

    def test_an_explicit_date_is_returned_as_given(self) -> None:
        """Not silently corrected. The runner is the single place a
        non-trading day is refused, so a backfill cannot route around the
        refusal by calling a different entry point."""
        assert resolve_date("2026-08-29") == dt.date(2026, 8, 29)

    @pytest.mark.parametrize("bad", ["28-08-2026", "2026-8-28", "yesterday", ""])
    def test_a_malformed_date_exits_with_a_message(self, bad: str) -> None:
        with pytest.raises(SystemExit, match="YYYY-MM-DD"):
            resolve_date(bad)


class TestPageConditions:
    def test_there_are_exactly_two(self) -> None:
        assert PAGE_CONDITIONS == ("absence", "failure")

    def test_a_third_condition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a page condition"):
            Page(condition="degraded", job="data.daily", trading_day=FRIDAY, reason="x")

    def test_a_page_without_a_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no reason"):
            Page(condition="absence", job="data.daily", trading_day=FRIDAY, reason="   ")

    def test_a_failure_page_must_carry_its_run_id(self) -> None:
        """The manifest exists, so its correlation identity does. Only an
        ABSENCE page legitimately has none."""
        with pytest.raises(ValueError, match="run_id"):
            Page(
                condition="failure",
                job="data.daily",
                trading_day=FRIDAY,
                reason="RuntimeError: boom",
            )

    def test_an_absence_page_legitimately_has_no_run_id(self) -> None:
        page = Page(
            condition="absence",
            job="data.weekly",
            trading_day=FRIDAY,
            reason="no manifest by 09:00 ET on the session after 2026-08-28",
        )
        assert page.run_id is None

    def test_dedup_is_one_key_per_job_and_trading_day(self) -> None:
        """Not per attempt: a job that fails, is retried by the declared
        transient class and fails again is ONE incident. Paging twice for it
        is how a two-pages-per-month ceiling is blown by one bad Saturday."""
        a = dedup_key("failure", "data.daily", FRIDAY)
        b = dedup_key("failure", "data.daily", FRIDAY)
        assert a == b
        assert a != dedup_key("failure", "data.daily", dt.date(2026, 8, 27))
        assert a != dedup_key("absence", "data.daily", FRIDAY)


class TestUnimplementedAlerting:
    @pytest.mark.parametrize("name", ["evaluate_absence", "evaluate_failure", "heartbeat"])
    def test_the_alerting_stubs_refuse_loudly(self, name: str) -> None:
        import crucible.alerts as alerts

        with pytest.raises(NotImplementedError, match="I9757"):
            getattr(alerts, name)()

    def test_send_refuses_loudly(self) -> None:
        """A delivery path that could quietly no-op is an outage nobody
        hears about."""
        from crucible.alerts import send

        page = Page(
            condition="absence",
            job="data.weekly",
            trading_day=FRIDAY,
            reason="no manifest by deadline",
        )
        with pytest.raises(NotImplementedError, match="I9757"):
            send(page)
