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
from crucible.gate import PHASES

FRIDAY = dt.date(2026, 8, 28)

#: The jobs still carrying a `_todo` placeholder, DERIVED from the dispatch
#: table via `cli.is_stub` rather than listed here. A hand-written list would
#: be written from the jobs someone remembered, and would go stale silently
#: the first time a track landed one.
UNIMPLEMENTED = sorted(job for job in JOBS if is_stub(HANDLERS[job]))


def _minimal_argv(job: str) -> list[str]:
    """The fewest arguments that make ``job`` parse."""
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
        argv += ["--gap", "missing-panel", "--from", "2026-08-24", "--to", "2026-08-28"]
    if job == "gate":
        # track-F: a gate with no name has no clause list, so `--gate` is
        # required rather than defaulted — a defaulted gate would report a
        # pass for a phase nobody asked about.
        argv += ["--gate", "phase1"]
    if job == "smoke":
        # track-C: the pointer flip refuses a smoke manifest belonging to
        # another build, so the sha the smoke is verifying is required.
        argv += ["--release", "a" * 40]
    return argv


#: The jobs track C implemented (alpha-engine-config-I9757). The stub
#: parametrisation above derives itself from `is_stub`, so this list exists
#: only to assert the CONVERSE — that these six are implemented. Without it,
#: track C landing its handlers would simply shrink the stub parametrisation
#: and nothing would assert they now do something: a gate going dark rather
#: than green.
TRACK_C_JOBS = ("alerts.sweep", "console", "drift", "heartbeat", "release.pin", "smoke")


class TestJobSurface:
    def test_the_jobs_of_the_plan_are_registered(self) -> None:
        """The plan's twelve, plus track C's four observing surfaces.

        The four are jobs like any other on purpose: they write manifests on
        the same terms, so the thing that watches the fleet is watched by the
        same registry, the same deadline table and the same console."""
        assert set(JOBS) == {
            "alerts.sweep",
            "heartbeat",
            "drift",
            "console",
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
            "weekly",
            "gate",
        }

    @pytest.mark.parametrize("job", sorted(JOBS))
    def test_every_job_parses(self, job: str) -> None:
        assert build_parser().parse_args(_minimal_argv(job)).job == job

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

    def test_an_unimplemented_job_raises_and_never_returns_zero(self) -> None:
        """A stub that exits 0 is indistinguishable from a job that ran and
        had nothing to do — the exact shape §11 says agent-built systems
        drift toward.

        **A loop, not a parametrisation.** `UNIMPLEMENTED` is derived from
        `is_stub`, so it legitimately empties as tracks land — and pytest
        SKIPS a parametrized test whose parameter set is empty, which in a
        repository declaring zero suppressions (§11.1) is a suppression the
        `tests/test_no_suppressions.py` grep cannot see: it reports
        `1 skipped` and reads as green. Looping keeps the assertion real at
        every size, including zero.
        """
        # Derived, not hardcoded: a stub's message names phase 1's tracker
        # (crucible/cli.py::_WIRING_PHASE), so this stays in sync with it rather
        # than restating the number a second time (alpha-engine-config-I9839).
        wiring_phase = next(p for p in PHASES if p.id == "phase1")
        for job in UNIMPLEMENTED:
            with pytest.raises(NotImplementedError, match=wiring_phase.tracker):
                main(_minimal_argv(job))

    def test_the_stub_set_shrinks_rather_than_being_declared(self) -> None:
        """The other half of the derivation: an implemented job is NOT here.

        Without this, `UNIMPLEMENTED` going empty would leave the test above
        asserting nothing at all — and an empty stub set is exactly the state
        the repository is trying to reach, so the moment it succeeds is the
        moment the guard would go dark rather than green.
        """
        assert set(UNIMPLEMENTED) <= set(JOBS)
        assert set(UNIMPLEMENTED).isdisjoint(
            {"data.daily", "data.weekly", "data.heal", "experiment.run", "explain"}
        ), "track A landed these; they are no longer stubs"

    def test_every_job_in_the_table_is_implemented(self) -> None:
        """The positive form, which is the one that has to hold at the end.

        `UNIMPLEMENTED` shrinking to nothing is the phase-1 goal; stated as an
        assertion, reaching it is a green test rather than an absence of
        tests.
        """
        assert UNIMPLEMENTED == [], (
            f"still stubs: {UNIMPLEMENTED}. A stub is a job whose absence from the "
            "weekly arc nothing else reports."
        )


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


class TestTrackCJobsAreImplemented:
    """The converse of `test_an_unimplemented_job_raises_and_never_returns_zero`.

    Without this, track C landing its handlers would be invisible to the test
    suite: the stub assertion would simply stop covering those jobs and
    nothing would assert they now do something. A parametrisation that
    shrinks silently is a gate going dark rather than green.
    """

    @pytest.mark.parametrize("job", sorted(TRACK_C_JOBS))
    def test_the_handler_is_not_a_stub(self, job: str) -> None:
        handler = HANDLERS[job]
        assert not is_stub(handler), (
            f"{job} still dispatches to a stub. Track C's jobs are implemented; a "
            "parametrisation that merely stopped covering them would be a gate going "
            "dark rather than green."
        )
        assert handler.__module__ == "crucible.track_c", (
            f"{job} dispatches to {handler.__module__}.{handler.__name__}; track C's "
            "handlers live in crucible.track_c."
        )


class TestAlertingSurface:
    """§4.6's two conditions, exercised against a real store rather than
    asserted about. The transport is captured, never a live one."""

    def test_an_absence_page_names_its_deadline_and_the_job(self, tmp_path) -> None:
        from crucible.alerts import evaluate_absence
        from crucible.store import LocalStore

        store = LocalStore(tmp_path)
        # Saturday 23:00 UTC — every Saturday deadline for Friday's session
        # has passed, and nothing has been written.
        now = dt.datetime(2026, 8, 29, 23, 0, tzinfo=dt.UTC)
        pages = evaluate_absence(store, now=now)
        jobs = {p.job for p in pages}
        assert "data.weekly" in jobs
        assert all(p.condition == "absence" for p in pages)
        assert all(p.trading_day == FRIDAY for p in pages)
        assert all("due" in p.reason for p in pages)

    def test_a_deadline_that_has_not_passed_is_not_an_absence(self, tmp_path) -> None:
        """A job whose deadline is still in the future is not absent, it is
        not due. Reporting it would fire the condition every early run."""
        from crucible.alerts import evaluate_absence
        from crucible.store import LocalStore

        store = LocalStore(tmp_path)
        # 16:30 ET on the session itself: the close has passed, so the run
        # binds to Friday, and every deadline anchored on that close is
        # still ahead — including data.daily's, at close + 3h.
        early = dt.datetime(2026, 8, 28, 20, 30, tzinfo=dt.UTC)
        assert [p.job for p in evaluate_absence(store, now=early)] == []

    def test_the_heartbeat_row_is_never_paged_for_by_the_sweep(self, tmp_path) -> None:
        """It declares `absence_watched_by: operator`. A machine watcher
        living inside the alerting path cannot report that path dead."""
        from crucible.alerts import evaluate_absence
        from crucible.store import LocalStore

        store = LocalStore(tmp_path)
        now = dt.datetime(2026, 8, 29, 23, 59, tzinfo=dt.UTC)
        assert "heartbeat" not in {p.job for p in evaluate_absence(store, now=now)}

    def test_send_raises_when_the_transport_does(self) -> None:
        """Delivery failure RAISES. An alert that could not be sent and was
        logged instead is an outage nobody hears about."""
        from crucible.alerts import PageGroup, send

        page = Page(
            condition="absence",
            job="data.weekly",
            trading_day=FRIDAY,
            reason="no manifest by deadline",
        )

        def exploding(*args, **kwargs):
            raise RuntimeError("telegram unreachable")

        with pytest.raises(RuntimeError, match="telegram unreachable"):
            send(PageGroup("absence:x", (page,)), alert_id="0" * 26, transport=exploding)
