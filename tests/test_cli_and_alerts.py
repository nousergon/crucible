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
from crucible.cli import HANDLERS, JOBS, NON_JOB_HANDLERS, build_parser, is_stub, main, resolve_date
from crucible.data.point_in_time import UnavailablePointInTimeSource

FRIDAY = dt.date(2026, 8, 28)

#: The jobs still carrying a `_todo` placeholder, DERIVED from the dispatch
#: table via `cli.is_stub` rather than listed here. A hand-written list would
#: be written from the jobs someone remembered, and would go stale silently
#: the first time a track landed one.
UNIMPLEMENTED = sorted(job for job in JOBS if is_stub(HANDLERS[job]))


def _minimal_argv(job: str) -> list[str]:
    """The fewest arguments that make ``job`` parse."""
    argv = [job]
    if job in (
        "experiment.run",
        "experiment.grade",
        "experiment.backfill",
        "promote",
        "experiment.new",
        "experiment.register",
    ):
        argv += ["--slot", "r"]
    if job in ("experiment.run", "experiment.new", "experiment.backfill"):
        argv += ["--arm", "arm_abc"]
    if job == "experiment.backfill":
        # alpha-engine-config-I10696: a range job, like data.heal. Both bounds
        # are required — a backfill that defaulted one end would produce a
        # range nobody named.
        argv += ["--from", "2026-08-24", "--to", "2026-08-28"]
    if job == "explain":
        argv += ["01JG0000000000000000000000"]
    if job == "release.pin":
        argv += ["a" * 40]
    if job == "release.lock":
        # alpha-engine-config-I9898: repair job, takes the release sha
        # positionally same as release.pin.
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
    if job == "fault.probe":
        # alpha-engine-config-I10343: the request is the whole job, so the
        # class is required rather than defaulted -- a `fault.probe` that
        # picked its own fault-injection class would be a job that could
        # induce a fault nobody asked for.
        argv += ["--fault-capability-class", "chaos_probe"]
    if job == "review.record":
        # alpha-engine-config-I10968: every field of the verdict is required.
        # A recording call that defaulted any of them would file a review
        # naming something other than what was reviewed.
        argv += [
            "--commits",
            "commits.json",
            "--reviewer",
            "session_01Reviewer00",
            "--phase",
            "phase1",
            "--verdict",
            "pass",
            "--head-sha",
            "b" * 40,
            "--pr-number",
            "48",
            "--summary",
            "no findings",
        ]
    if job == "acceptance.publish":
        # The reading is the whole input: this job records a measurement, it
        # does not take one, so there is nothing to default.
        argv += ["--reading", "acceptance-reading.json"]
    if job == "fault.record":
        # `--outcome` is required, and which of the remaining flags are legal
        # is decided by it — inside `crucible.faults.record_fault`, not the
        # parser, since argparse cannot express "one of these three field
        # sets" and a second copy of the matrix would be the half that drifts.
        argv += [
            "--fault",
            "data_source_withheld",
            "--outcome",
            "induced",
            "--target-job",
            "data.weekly",
            "--run-id",
            "01" + "A" * 24,
        ]
    return argv


#: The jobs track C implemented (alpha-engine-config-I9757). The stub
#: parametrisation above derives itself from `is_stub`, so this list exists
#: only to assert the CONVERSE — that these six are implemented. Without it,
#: track C landing its handlers would simply shrink the stub parametrisation
#: and nothing would assert they now do something: a gate going dark rather
#: than green.
TRACK_C_JOBS = (
    "alerts.sweep",
    "board",
    "console",
    "drift",
    "heartbeat",
    "release.pin",
    "smoke",
)


class TestJobSurface:
    def test_the_jobs_of_the_plan_are_registered(self) -> None:
        """The plan's twelve, plus the observing surfaces.

        The observing surfaces are jobs like any other on purpose: they write
        manifests on the same terms, so the thing that watches the fleet is
        watched by the same registry, the same deadline table and the same
        console."""
        assert set(JOBS) == {
            "alerts.sweep",
            # alpha-engine-config-I9837: the fully-declared board. Its own job
            # on its own DAILY schedule, not a stage of the weekly arc.
            "board",
            "heartbeat",
            "drift",
            "console",
            "data.daily",
            "data.weekly",
            "data.heal",
            "experiment.new",
            # alpha-engine-config-I10927: the DERIVED sibling of
            # `experiment.new`. An arc stage at 11:00, before
            # `experiment.run` at 12:00, registering every recipe the pinned
            # release declares and the slot's register lacks. Not a
            # relaxation of `experiment.new --arm`, which stays required: the
            # deliberate act moves up a layer, to the release pin.
            "experiment.register",
            "experiment.run",
            # alpha-engine-config-I10696 (Brian's ruling (a), 2026-09-14):
            # one arm's history over a session range, produced through the
            # slot's OWN per-arm produce path. On-demand — it repairs a
            # history, and a stacked arm's training window is unreachable
            # from the weekly cadence alone.
            "experiment.backfill",
            "experiment.grade",
            "promote",
            "report",
            "explain",
            # alpha-engine-config-I10502: the sealed holdout's reader and its
            # ruling-gated --unseal. Never scheduled — unsealing is a reserved
            # matter (principles.md §3.2), so a clock able to invoke it would
            # be an automation holding an authority reserved to a human.
            "holdout",
            "migrate.history",
            "release.pin",
            # alpha-engine-config-I9898: the repair for a release published
            # before I9787's write-time Object Lock fix.
            "release.lock",
            "smoke",
            "weekly",
            "gate",
            # alpha-engine-config-I10095: the FILING half of the phase-exit
            # loop, on a cadence. `gate` stays on-demand -- a gate on a
            # schedule would page between phases -- but the RECORD of a
            # phase's exit cannot wait for somebody to run one, and the board
            # that detects the gap is read-only over what it grades on
            # purpose, so it cannot close it.
            "gate.close",
            # alpha-engine-config-I9896: the 6am PT accountability delivery.
            # A reporting surface, so it READS the board and never renders
            # one -- the same separation `board` keeps from the artifacts it
            # grades, one layer further out.
            "report.morning",
            # alpha-engine-config-I10320/-I10322: the producer of
            # faults/{trading_day}/{fault}.json -- the record phase 2's
            # fault_injection_against_scheduled_path clause reads. On-demand,
            # like data.heal and experiment.new: filed once per induced
            # fault, never on a schedule.
            "fault.record",
            # alpha-engine-config-I10343: the INDUCER for §10.7 fault 3, kept
            # separate from `fault.record` (the attester). On-demand -- a job
            # that can only ever fail would page every cycle on a schedule.
            "fault.probe",
            # alpha-engine-config-I10418: the two IaC conformance
            # comparisons (account vs template, template vs the plan's
            # declared inventory). `dispatch: arc` -- a stage of the same
            # weekly run, not a new schedule or a separate detector fleet.
            "iac.conformance",
            # alpha-engine-config-I10459: promotes the integration tier's
            # summary artifact to a real registered job -- a thin handler
            # that shells to `pytest tests/integration` and reports
            # pass/fail through `run_job`. Workflow-triggered only, like
            # `iac.conformance`'s `dispatch: arc` sibling -- no schedule or
            # deadline of its own.
            "test.integration",
            # alpha-engine-config-I10968: the two remaining writers of the
            # `test.integration` class -- a store write reached from outside
            # the job table, filing no manifest. `review.record` was
            # `python -m crucible.review record`'s own `store.put_bytes`;
            # `acceptance.publish` was a raw `aws s3 cp` in ci.yml that never
            # went through `crucible.store.Store` at all. Both
            # workflow-triggered, neither scheduled.
            "review.record",
            "acceptance.publish",
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
        in JOBS with no handler at all would silently drop out of both.

        `NON_JOB_HANDLERS` is the one named exception: a handler wired into
        `HANDLERS` for a repair tool that patches an existing document rather
        than writing a new `run_manifest.v2` (`migrate.code_sha`,
        alpha-engine-config-I10626) and so deliberately claims no `JOBS`
        slot. Only names listed there are exempt — anything else missing
        from `JOBS` still fails this assertion."""
        assert set(HANDLERS) - NON_JOB_HANDLERS == set(JOBS)

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
        # Derived, not hardcoded: a stub's message names the EPIC's tracker
        # (crucible/cli.py::_EPIC_ISSUE / ::_epic_tracker), not a phase — a
        # phase closes (phase 1 did, 2026-09-02T01:31Z, while these stubs
        # remained) and the epic does not. Matching the same int, not the
        # same string, keeps this in sync with cli.py rather than restating
        # the number a second time (alpha-engine-config-I9839).
        from crucible.cli import _EPIC_ISSUE

        expected_tracker = f"alpha-engine-config-I{_EPIC_ISSUE}"
        for job in UNIMPLEMENTED:
            with pytest.raises(NotImplementedError, match=expected_tracker):
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


class TestUsageErrorReachesStderr:
    """`alpha-engine-config-I10517`: `UsageError` is a `SystemExit` subclass
    constructed with an INT `.code`. Python's own uncaught-`SystemExit`
    handling only prints something when `.code` is a *string*, so the
    message never reached the console script's stderr -- only `main`'s
    return value (an int, silently swallowed by the interpreter) carried the
    exit code. `main` must print the message itself before returning.
    """

    def test_a_missing_run_mode_prints_to_stderr_and_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_RUN_MODE", raising=False)
        code = main(["experiment.new", "--slot", "r", "--arm", "arm_abc"])
        assert code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "live" in captured.err
        assert "replay" in captured.err

    def test_missing_gate_env_prints_to_stderr_and_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", raising=False)
        monkeypatch.delenv("CRUCIBLE_MUTED_TOPIC", raising=False)
        monkeypatch.setenv("CRUCIBLE_RUN_MODE", "live")
        code = main(["gate", "--gate", "phase2", "--store", "/tmp/does-not-matter"])
        assert code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "refuses to read" in captured.err


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

    def test_evidence_beside_a_manifest_is_neither_a_run_nor_a_failure(self, tmp_path) -> None:
        """alpha-engine-config-I9900, the same false assumption one module
        over: `report.morning` files its delivered `message.txt` under its OWN
        manifest prefix. `evaluate_failure` used to `json.loads` it and page a
        FAILURE every night for a job that succeeded, and `evaluate_absence`
        used to count it as "a manifest exists" and suppress a real absence.
        Both narrow the listing to manifest keys now."""
        from crucible.alerts import evaluate_absence, evaluate_failure
        from crucible.keys import morning_report_key
        from crucible.store import LocalStore

        store = LocalStore(tmp_path)
        store.put_bytes(
            morning_report_key(FRIDAY.isoformat(), "2026-08-29"),
            b"Good morning. Board: 14 of 23 clauses met.\n",
        )
        now = dt.datetime(2026, 8, 29, 23, 0, tzinfo=dt.UTC)
        assert evaluate_failure(store, now=now) == []
        assert "report.morning" in {p.job for p in evaluate_absence(store, now=now)}

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


class TestDryRunNeverWrites:
    """alpha-engine-config-I9922 N1 / R2-1 / R2-2 (independent review,
    2026-09-03, two rounds).

    `--dry-run`'s own CLI help ("write nothing") was true only of the jobs
    whose handler body happened to check `args.dry_run` before touching the
    store. `run_job(dry_run=True)` alone did not fix this: it only stops
    `run_job` from writing ITS OWN manifest — a job body that calls
    `ctx.record_output` (or `store.put_bytes` / `compare_and_swap` directly)
    still reached the real backend, reproduced live: a body calling
    `record_output("board/current.json", ...)` under `dry_run=True` left the
    artifact in the store with no manifest, while the runner printed "no
    outputs recorded".

    **R2-1**: the store-level guard (`crucible.store.read_only`, enforced at
    `open_store`/`Settings.store`) is the BACKSTOP, not the primary path —
    every handler that has a natural report (`gate`, `report`, `console`,
    `drift`, `alerts.sweep`, `heartbeat`, `release.pin`) now checks `dry_run`
    itself and prints that report rather than reaching the guard and dying on
    it, which is what made `crucible gate --gate phase1 --dry-run` (a
    documented, previously-working read) crash with a bare traceback instead
    of printing the reading it always printed before.

    **R2-2**: the round-1 version of this test wrapped every job in
    `try: main(argv) except BaseException: pass`, which can never fail, and
    18 of 21 jobs died on MISSING INPUTS before ever reaching a write or the
    guard, making the empty-store assertion vacuous. This version seeds each
    job far enough to reach its real dry-run path, and every row below
    asserts a CLEAN return (R3-1, non-blocking review note) — none of the
    fifteen jobs tested here is documented to raise under `--dry-run`, so
    accepting `DryRunWriteRefusedError` as an alternative legal outcome
    would let a handler that lost its own `dry_run` branch and started
    relying on the store guard alone keep passing. Any exception at all is a
    test failure, not a swallowed pass.

    Five jobs are excluded, each for a stated reason rather than silently:
    `experiment.new` needs a synced strategy tree with real arm
    recipes; `migrate.history` needs seeded v1 sources; `smoke` needs a
    published release publishing through `crucible.deploy`'s own flow;
    `release.lock` is S3-only (`apply_release_retention` refuses a
    `LocalStore` outright) and is covered instead, against a fake S3 client,
    by `tests/test_release_retention.py::TestReleaseLockHandler::
    test_dry_run_writes_no_manifest_at_all`.

    `weekly` is excluded here for a DIFFERENT reason than the other four: the
    rows above each invoke one job directly, through its own argv — none of
    them exercises `weekly`'s OWN responsibility, which is handing
    `--dry-run` DOWN onto each of its twelve stages' own argv
    (`weekly.py::Stage.argv`/`run_arc`, alpha-engine-config-I9922 R3-1). A
    per-job row here could not observe that hand-down even if `weekly` were
    added to the parametrisation — invoking `weekly` here would only prove
    `weekly`'s OWN store is read-only, which is not the property that
    matters. That property is asserted directly, against a fake `main`, by
    `tests/test_weekly.py::TestRunsTheRealCommand::
    test_a_dry_run_arc_puts_dry_run_on_every_stage` and
    `test_a_real_arc_puts_dry_run_on_no_stage`.
    """

    #: Reach either outcome with NO seeding beyond a fresh, empty store —
    #: verified individually (2026-09-03) by running each against a fresh
    #: `tmp_path` and confirming it returns 0 rather than raising.
    _CLEAN_ON_A_FRESH_STORE = (
        "alerts.sweep",
        "board",
        "console",
        "gate",
        # alpha-engine-config-I10095, verified 2026-09-06 the same way: on a
        # fresh store every registered gate reads UNMET or UNMEASURABLE, so
        # the handler reaches its real --dry-run path (no store write, and no
        # tracker comment -- the `would_file` branch returns BEFORE
        # `file_closing_record` is reached) and returns 0.
        "gate.close",
        "heartbeat",
        "release.pin",
        "report",
    )
    # NOTE (alpha-engine-config-I11012): `experiment.run` left this set. Its
    # dry run no longer prints a sentence restating its own source and
    # returns — it EXECUTES `produce` against a write-capturing store, and on
    # a fresh store that raises `MissingArtifactError` for the absent feature
    # layer, which is precisely the rehearsal fidelity this issue bought. It
    # is covered by a seeded row below instead.

    #: The jobs each covered by their own one-off SEEDED row below rather
    #: than the shared fresh-store parametrisation above -- each needs
    #: something present in the store, or a config env var, before it can
    #: reach its real dry-run path (a bucket name for the three ArcticDB
    #: jobs, drift's three input keys, promote's empty register, morning's
    #: seeded board).
    _COVERED_BY_A_SEEDED_ROW = frozenset(
        {
            "data.daily",
            "data.heal",
            "data.weekly",
            "drift",
            "experiment.run",
            "explain",
            # alpha-engine-config-I11005: its dry run now RESOLVES the arm
            # against the slot's registered recipes, the way the job does, so
            # it needs a strategy tree exactly as `experiment.new` does. It
            # moved out of the fresh-store set for that reason rather than
            # into the excluded set: a dry run that could not refuse an arm
            # the job would refuse is the gap this issue closed.
            "experiment.backfill",
            # alpha-engine-config-I10721: `experiment.grade` left the
            # fresh-store set above for `experiment.run`'s reason, one issue
            # after it. Its dry run no longer prints a sentence restating its
            # own source and returns 0 — it EXECUTES the slot's real `grade`
            # against a write-capturing store, and on a fresh store that
            # raises `MissingArtifactError` for the absent price panel, which
            # is precisely the rehearsal fidelity this move buys.
            "experiment.grade",
            "promote",
            "report.morning",
        }
    )

    #: Every job in `JOBS` NOT covered by a row above, each with the reason
    #: it is excluded rather than tested here -- lifted from this class's own
    #: docstring, which states the reason but never asserted it against
    #: `JOBS` itself.
    _EXCLUDED_WITH_REASON = {
        "experiment.new": "needs a synced strategy tree with real arm recipes",
        "experiment.register": (
            "needs a synced strategy tree with real arm recipes, same as "
            "`experiment.new` — its dry-run property (the diff is reported and the "
            "store gains no key at all, not even a manifest) is asserted directly, "
            "against a seeded LocalStore, by tests/test_experiment_register.py::"
            "TestDryRun::test_a_dry_run_reports_the_diff_and_writes_nothing"
        ),
        "migrate.history": "needs seeded v1 sources",
        "holdout": (
            "its READ form writes no manifest at all and exits 1 on a fresh store (no "
            "holdout is published, which is the honest answer), so the shared "
            "fresh-store row could not distinguish that from a failure; its WRITING "
            "form refuses before it reaches the store unless --ruling and --reason are "
            "given, which no shared row supplies. Both are asserted directly by "
            "tests/test_holdout.py::TestTheCli"
        ),
        "smoke": "needs a published release publishing through crucible.deploy's own flow",
        "release.lock": (
            "S3-only (apply_release_retention refuses a LocalStore outright); covered "
            "instead, against a fake S3 client, by tests/test_release_retention.py::"
            "TestReleaseLockHandler::test_dry_run_writes_no_manifest_at_all"
        ),
        "weekly": (
            "hands --dry-run DOWN onto twelve stages' own argv; a per-job row here could "
            "not observe that hand-down, so it is asserted directly, against a fake main, "
            "by tests/test_weekly.py::TestRunsTheRealCommand::"
            "test_a_dry_run_arc_puts_dry_run_on_every_stage and "
            "test_a_real_arc_puts_dry_run_on_no_stage"
        ),
        "fault.record": (
            "always refuses on a fresh store -- crucible.faults.record_fault looks for a "
            "manifest naming --run-id BEFORE it ever writes, so a fresh store raises "
            "FaultRecordRefusedError, not a clean dry-run print; --dry-run then never "
            "reaches the write path at all (the property this class exists to check) "
            "regardless of seeding. The refusal-before-write property itself is asserted "
            "directly by tests/faults/test_fault_record_producer.py::"
            "TestRefusals::test_refuses_before_writing_anything"
        ),
        "fault.probe": (
            "reaches a real provider on every invocation by design -- its registered call "
            "site asks for a capability class contracted never to serve, so there is no "
            "clean dry-run print, only a transport failure. Its --dry-run property (the "
            "run raises AND no manifest is written) is asserted directly, against a fake "
            "transport, by tests/test_chaos_probe_containment.py::"
            "TestTheProbeJob::test_a_dry_run_probe_still_fails_and_writes_no_manifest"
        ),
        "iac.conformance": (
            "reaches real CloudFormation/tagging/IAM/SSM clients on every invocation to "
            "compute both comparisons -- --dry-run only suppresses the store WRITE "
            "(crucible.iac_conformance.iac_conformance_handler), not the AWS reads "
            "themselves, so a fresh store with no AWS credentials/region configured "
            "raises out of client construction rather than returning a clean 0. Its "
            "--dry-run property (the readings still compute; the store gains no output "
            "key) is asserted directly, against fake clients, by "
            "tests/test_iac_conformance.py"
        ),
        "review.record": (
            "its input is a commits payload the workflow fetches from the GitHub API, "
            "and its dry-run property (the document is built, the independence "
            "comparison runs, and the store gains no key at all) is asserted directly, "
            "against a seeded LocalStore, by tests/test_review_record_job.py::"
            "TestDryRun"
        ),
        "acceptance.publish": (
            "its input is the reading file `tests/acceptance/check_reading.py "
            "--write-json` just wrote, which no shared fresh-store row supplies; its "
            "dry-run property (the reading is read and validated, and the store gains "
            "no key at all) is asserted directly by tests/test_acceptance_publish_job.py"
            "::TestDryRun"
        ),
        "test.integration": (
            "shells out to a real `pytest tests/integration` subprocess on every "
            "invocation -- --dry-run only suppresses THIS job's own manifest write "
            "(run_job(dry_run=True)), never the subprocess call, so a fresh store with "
            "no CRUCIBLE_INTEGRATION_* environment configured fails inside that real "
            "suite's own fixtures rather than returning a clean 0. Covered directly, "
            "against a faked subprocess, by tests/test_integration_summary_job.py"
        ),
    }

    def test_every_job_is_dry_run_tested_or_excluded_with_a_reason(self) -> None:
        """The class fix for alpha-engine-config-I9863.

        `_CLEAN_ON_A_FRESH_STORE` and the seeded rows below are all
        hand-written lists of job names, not derived from `crucible.cli.JOBS`
        -- which is exactly how the sixth handler (and the fourteenth, and
        the twenty-first) gets added with no dry-run coverage at all: nothing
        here would notice, because nothing here reads `JOBS`.

        This is the derivation: every job in `JOBS` is either covered
        directly (by one of the two sets above) or named in
        `_EXCLUDED_WITH_REASON`, and never simply absent from both. A new
        job that is neither seeded here nor given a stated exclusion fails
        this test loudly, by name, rather than silently inheriting zero
        dry-run assurance.
        """
        covered = set(self._CLEAN_ON_A_FRESH_STORE) | self._COVERED_BY_A_SEEDED_ROW
        excluded = set(self._EXCLUDED_WITH_REASON)
        double_counted = covered & excluded
        assert not double_counted, (
            f"{sorted(double_counted)} listed as both dry-run tested AND excluded -- pick one."
        )
        accounted_for = covered | excluded
        missing = set(JOBS) - accounted_for
        assert not missing, (
            f"{sorted(missing)} are registered in crucible.cli.JOBS with NO dry-run "
            "coverage and no stated exclusion. Add a --dry-run test row for each, or a "
            "reason to _EXCLUDED_WITH_REASON if one genuinely cannot be tested here."
        )
        stale = accounted_for - set(JOBS)
        assert not stale, f"{sorted(stale)} are named here but no longer in crucible.cli.JOBS."

    @staticmethod
    def _assert_no_new_keys(tmp_path, before: list[str]) -> None:
        from crucible.store import LocalStore

        assert sorted(LocalStore(tmp_path).list_keys()) == before

    @pytest.mark.parametrize("job", sorted(_CLEAN_ON_A_FRESH_STORE))
    def test_dry_run_completes_cleanly_against_a_fresh_store(
        self, job: str, tmp_path, monkeypatch
    ) -> None:
        """These nine are documented (R2-1) to print their reading/report and
        return — NOT to raise. Asserting a clean return directly, rather than
        accepting `DryRunWriteRefusedError` as an alternative legal outcome,
        is deliberate (R3-1, non-blocking review note): a two-outcome helper
        here would keep passing the moment one of these nine LOST its own
        `dry_run` branch and started relying on the store guard alone — a
        real regression this test exists to catch, since a bare guard-refusal
        is a worse operator experience than the print these jobs promise."""
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        argv = [
            *_minimal_argv(job),
            "--date",
            FRIDAY.isoformat(),
            "--store",
            str(tmp_path),
            "--dry-run",
        ]

        main(argv)  # must not raise at all -- see the docstring above

        self._assert_no_new_keys(tmp_path, [])

    @pytest.mark.parametrize("job", ["data.heal", "data.weekly"])
    def test_dry_run_completes_cleanly_with_an_arctic_bucket_configured(
        self, job: str, tmp_path, monkeypatch
    ) -> None:
        """These two construct an `ArcticPriceSource` (which needs a
        bucket NAME, never a real connection) before their own dry-run
        branch prints and returns.

        `data.daily` was a third row here until alpha-engine-config-I11012
        removed its dry-run branch: its dry run now runs the real compile
        against a write-capturing store, so a bucket NAME is no longer
        enough — it reaches ArcticDB for real, which is the point. It has its
        own seeded row below — a `CRUCIBLE_ARCTIC_BUCKET` config gap,
        not a store-write concern, and unrelated to `--dry-run` itself
        (the same `ValueError` fires with `--dry-run` omitted). Also asserted
        as a clean return, not a two-outcome one — same reasoning as above:
        these three never reach the store guard at all on their real
        dry-run path (they return before `run_job` is even called), so a
        `DryRunWriteRefusedError` here would itself be a regression, not an
        acceptable alternative."""
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        monkeypatch.setenv("CRUCIBLE_ARCTIC_BUCKET", "fake-bucket")
        argv = [
            *_minimal_argv(job),
            "--date",
            FRIDAY.isoformat(),
            "--store",
            str(tmp_path),
            "--dry-run",
        ]

        main(argv)  # must not raise at all -- see the docstring above

        self._assert_no_new_keys(tmp_path, [])

    # ── alpha-engine-config-I11012 ─────────────────────────────────────────
    #
    # `--dry-run` used to resolve a `crucible.store.read_only` store, which
    # made a dry run safe by RAISING at the first write. Safe, and not a
    # rehearsal: no job body ever ran far enough to fail the way its run
    # fails. Measured cost — a laptop dry run of `experiment.backfill --slot u
    # --arm attractiveness --from 2025-11-17 --to 2026-09-09` reported "would
    # produce 205 session(s)" for a command that died on the FIRST session in
    # production, two minutes in.
    #
    # It now resolves a `crucible.store.capturing` store: real reads, writes
    # RECORDED. The rows below assert the two halves — every job's dry run
    # gets one of those stores, and the reported key set is the set the real
    # run writes.

    @staticmethod
    def _capturing_stores_resolved_during(monkeypatch, run):
        """Every store ``run()`` resolved, and whether each was capturing.

        Spies on BOTH resolution sites, because there are two and they have
        disagreed before: `crucible.store.open_store` (the CLI's own
        `_resolve_store`) and `crucible.config.Settings.store` (which
        track-A's eight handlers reach through `_settings`). A spy on one
        would read green over a handler wired to the other.
        """
        from crucible import config as config_module
        from crucible import store as store_module

        resolved: list[object] = []

        def _spy(original):
            def _wrapped(store, **kwargs):
                wrapped = original(store, **kwargs)
                resolved.append(wrapped)
                return wrapped

            return _wrapped

        monkeypatch.setattr(store_module, "capturing", _spy(store_module.capturing))
        monkeypatch.setattr(config_module, "capturing", _spy(config_module.capturing))
        run()
        return resolved

    @pytest.mark.parametrize("job", sorted(_CLEAN_ON_A_FRESH_STORE))
    def test_every_dry_run_resolves_a_write_capturing_store(
        self, job: str, tmp_path, monkeypatch
    ) -> None:
        """The property `--dry-run`'s help text has always claimed and only
        the capturing store makes structural: the object every handler writes
        through cannot reach the backend's write path at all.

        Asserted per job rather than once on `open_store`, because what makes
        a handler safe is which store IT resolved — the measured defect this
        class exists for was a handler reaching the real backend while the
        runner printed "no outputs recorded".
        """
        from crucible.store import is_capturing

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        argv = [
            *_minimal_argv(job),
            "--date",
            FRIDAY.isoformat(),
            "--store",
            str(tmp_path),
            "--dry-run",
        ]

        resolved = self._capturing_stores_resolved_during(monkeypatch, lambda: main(argv))

        assert resolved, f"{job} --dry-run resolved no store through either wrap point"
        assert all(is_capturing(store) for store in resolved)
        self._assert_no_new_keys(tmp_path, [])

    def test_dry_run_experiment_run_executes_produce_and_writes_nothing(
        self, tmp_path, monkeypatch, source, strategy_dir
    ) -> None:
        """`experiment.run --dry-run` runs the slot's real `produce`.

        The version this replaces printed "would call produce and write its
        feed" — a sentence restating the handler's own source, which could
        not be wrong and could not be right. Seeded with two compiled days
        because `produce` reads the feature layer and nothing else (§10.4).
        """
        from conftest import sessions_ending

        from crucible.data.daily import run_daily
        from crucible.runner import run_job
        from crucible.store import LocalStore, begin_capture, end_capture

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path / "store")
        for day in sessions_ending(FRIDAY, 2):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c,
                    point_in_time=UnavailablePointInTimeSource(
                        reason="synthetic fixture market carries no fundamentals"
                    ),
                    source=source,
                    expected_symbols=source.symbols(),
                ),
                store=store,
                trading_day=day,
            )
        before = sorted(store.list_keys())

        ledger = begin_capture()
        try:
            main(
                [
                    "experiment.run",
                    "--slot",
                    "u",
                    "--date",
                    FRIDAY.isoformat(),
                    "--store",
                    str(tmp_path / "store"),
                    "--strategy-dir",
                    str(strategy_dir),
                    "--run-mode",
                    "replay",
                    "--dry-run",
                ]
            )
        finally:
            end_capture()

        assert sorted(store.list_keys()) == before  # nothing NEW landed
        # It reached the WRITE, which the print it replaces never did.
        assert any(key.startswith("runs/experiment.run/") for key in ledger.keys)

    def test_dry_run_data_daily_compiles_for_real_and_writes_nothing(
        self, tmp_path, monkeypatch, source
    ) -> None:
        """`data.daily --dry-run` runs the real compile.

        This is the job whose dry run most needed to become a rehearsal: a
        missing feature column, an unsatisfiable declared input or a refusing
        store grant all live inside the compile, and every one of them was
        invisible to a print naming only the source and the store URI.

        The price and point-in-time sources are substituted because the CLI
        has no flag for a test source — `--source` admits `arctic` alone, on
        purpose. Nothing else about the handler is faked.
        """
        from crucible import track_a
        from crucible.store import LocalStore, begin_capture, end_capture

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        monkeypatch.setattr(track_a, "_source", lambda args, config: source)
        monkeypatch.setattr(
            track_a,
            "_point_in_time_source",
            lambda config: UnavailablePointInTimeSource(
                reason="synthetic fixture market carries no fundamentals"
            ),
        )
        store_root = tmp_path / "store"

        ledger = begin_capture()
        try:
            main(
                [
                    "data.daily",
                    "--date",
                    FRIDAY.isoformat(),
                    "--store",
                    str(store_root),
                    "--run-mode",
                    "replay",
                    "--symbols",
                    ",".join(source.symbols()),
                    "--dry-run",
                ]
            )
        finally:
            end_capture()

        assert sorted(LocalStore(store_root).list_keys()) == []
        assert len(ledger.keys) > 1, "the compile recorded no artifact beyond its manifest"

    def test_the_reported_key_set_is_the_set_a_real_data_daily_writes(
        self, tmp_path, monkeypatch, source
    ) -> None:
        """The Closes-when property, on a second job shape
        (`alpha-engine-config-I11012`): what the rehearsal REPORTS is what the
        run WRITES.

        Two separate store roots rather than one run after the other, so the
        real run cannot see anything the rehearsal left and the rehearsal
        cannot see anything the real run left — there is nothing for either to
        skip as already present.
        """
        from crucible import track_a
        from crucible.store import LocalStore, begin_capture, end_capture

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        monkeypatch.setattr(track_a, "_source", lambda args, config: source)
        monkeypatch.setattr(
            track_a,
            "_point_in_time_source",
            lambda config: UnavailablePointInTimeSource(
                reason="synthetic fixture market carries no fundamentals"
            ),
        )

        def _argv(root, *extra):
            return [
                "data.daily",
                "--date",
                FRIDAY.isoformat(),
                "--store",
                str(root),
                "--run-mode",
                "replay",
                "--symbols",
                ",".join(source.symbols()),
                *extra,
            ]

        ledger = begin_capture()
        try:
            main(_argv(tmp_path / "rehearsal", "--dry-run"))
        finally:
            end_capture()
        assert sorted(LocalStore(tmp_path / "rehearsal").list_keys()) == []

        main(_argv(tmp_path / "real"))
        really_written = sorted(LocalStore(tmp_path / "real").list_keys())

        assert sorted(ledger.keys) == really_written

    def test_dry_run_experiment_backfill_fails_the_way_the_run_fails(
        self, tmp_path, monkeypatch, strategy_dir
    ) -> None:
        """The property `alpha-engine-config-I11012` bought, stated as
        plainly as it can be: on a store with no feature layer, the REHEARSAL
        raises the same `MissingArtifactError` the real run raises, from
        inside the per-session produce call — and still writes nothing.

        The version this replaces returned 0 here. It resolved the slot
        module, the history producer, the range, the arm and the in-region
        guard (`alpha-engine-config-I11005`) and then printed "NOT rehearsed:
        the per-session produce call itself", which is where the measured
        production failure lived. This is the row that would have caught it.

        The store is a subdirectory, not ``tmp_path`` itself, because the
        `strategy_dir` fixture writes its recipe tree there and a store
        rooted above it would count those files as store keys.
        """
        from crucible.slots.cycle import MissingArtifactError

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        argv = [
            "experiment.backfill",
            "--slot",
            "u",
            "--arm",
            "momentum_sleeve",
            "--from",
            "2026-08-24",
            "--to",
            "2026-08-28",
            "--store",
            str(tmp_path / "store"),
            "--strategy-dir",
            str(strategy_dir),
            "--dry-run",
        ]

        with pytest.raises(MissingArtifactError) as excinfo:
            main(argv)
        assert "the feature layer is absent" in str(excinfo.value)

        self._assert_no_new_keys(tmp_path / "store", [])

    def test_dry_run_experiment_backfill_refuses_an_arm_the_job_would_refuse(
        self, tmp_path, monkeypatch, strategy_dir
    ) -> None:
        """The property the row above cannot show: the rehearsal REFUSES.
        `alpha-engine-config-I11005`."""
        from crucible.backfill import UnknownArmError

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        argv = [
            "experiment.backfill",
            "--slot",
            "u",
            "--arm",
            "not_a_registered_arm",
            "--from",
            "2026-08-24",
            "--to",
            "2026-08-28",
            "--store",
            str(tmp_path / "store"),
            "--strategy-dir",
            str(strategy_dir),
            "--dry-run",
        ]
        with pytest.raises(UnknownArmError) as excinfo:
            main(argv)
        assert "not_a_registered_arm" in str(excinfo.value)
        self._assert_no_new_keys(tmp_path / "store", [])

    def test_dry_run_drift_computes_its_inputs_and_writes_nothing(
        self, tmp_path, monkeypatch, source
    ) -> None:
        """`drift` computes its three inputs from the store (a compiled feature
        layer is the one thing it cannot do without — `crucible.drift_inputs`);
        under `--dry-run` it computes, prints the three records, and files
        neither the inputs nor a manifest. One compiled day is enough to reach
        the dry-run path: the feature row reads UNREPORTED (no earlier day to
        drift from), which is a legitimate reading, not an absent input."""
        from conftest import sessions_ending

        from crucible.data.daily import run_daily
        from crucible.runner import run_job
        from crucible.store import LocalStore

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        # two compiled days: with only one, every row is UNREPORTED and
        # `drift_metrics` correctly refuses the cycle (nothing at all measured)
        for day in sessions_ending(FRIDAY, 2):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c,
                    point_in_time=UnavailablePointInTimeSource(
                        reason="synthetic fixture market carries no fundamentals"
                    ),
                    source=source,
                    expected_symbols=source.symbols(),
                ),
                store=store,
                trading_day=day,
            )
        before = sorted(store.list_keys())
        argv = ["drift", "--date", FRIDAY.isoformat(), "--store", str(tmp_path), "--dry-run"]

        main(argv)  # must not raise at all -- the feature layer is present

        assert sorted(store.list_keys()) == before  # nothing NEW landed

    def test_dry_run_explain_walks_and_writes_nothing(self, tmp_path, monkeypatch, source) -> None:
        """`explain` runs through `run_job` since 2026-09-05 (it used to write
        no manifest at all, which left the phase-1 `explain_walks_a_verdict`
        clause with nothing to read). Under `--dry-run` it walks, prints, and
        files neither inputs nor a manifest. Seeded with one `data.daily` run
        so there is a run_id to walk."""
        from crucible.data.daily import run_daily
        from crucible.runner import run_job
        from crucible.store import LocalStore

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        ctx = run_job(
            "data.daily",
            lambda c: run_daily(
                c,
                point_in_time=UnavailablePointInTimeSource(
                    reason="synthetic fixture market carries no fundamentals"
                ),
                source=source,
                expected_symbols=source.symbols(),
            ),
            store=store,
            trading_day=FRIDAY,
        )
        before = sorted(store.list_keys())
        argv = [
            "explain",
            "--date",
            FRIDAY.isoformat(),
            "--store",
            str(tmp_path),
            "--dry-run",
            ctx.run_id,
        ]

        main(argv)  # must not raise at all -- the run id is walkable

        assert sorted(store.list_keys()) == before  # nothing NEW landed

    def test_dry_run_promote_completes_cleanly_with_an_empty_register_seeded(
        self, tmp_path, monkeypatch
    ) -> None:
        """`crucible.promote.load_slot_inputs` does a bare
        `store.get_bytes(arm_register_key(slot))` with no absent-is-empty
        fallback (unlike `crucible.slots.arms.read_register`), so `promote`
        needs the register key to exist at all — an empty one is enough to
        reach the real dry-run path (`run_promotion(store=None)`,
        pre-existing, `cli.py::_promote`).

        The graded `arena_cycle` is seeded for the same reason and is not a
        second fixture doing the same job: since `alpha-engine-config-I9759`
        `promote` READS the eligibility `experiment.grade` evaluated, and
        refuses outright when that artifact is absent, so an empty register
        alone no longer reaches the dry-run path. Seeded here with the
        library's own `run_cycle` over the same empty register, which is
        exactly what `experiment.grade` would have written for a slot with
        nothing scored: `unservable`, no arms, no vetoes. `alpha-engine-config-
        I10679` made `crucible.promote.read_graded_cycle` additionally check
        `experiment.grade`'s own run manifest before trusting that artifact,
        so this fixture writes one too, via the same test helper the promote
        test suites use.
        """
        from nousergon_lib.arena import ArmRegister
        from nousergon_lib.arena.engine import run_cycle

        from crucible.arena_io import write_arena_cycle
        from crucible.promote import arm_register_key
        from crucible.slots import get_slot
        from crucible.store import LocalStore
        from tests.support.manifests import write_grade_manifest

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        store.put_bytes(arm_register_key("r"), b"")
        write_arena_cycle(
            store,
            run_cycle(
                config=get_slot("r").arena,
                as_of=FRIDAY.isoformat(),
                register=ArmRegister(),
                series_by_arm={},
                incumbent=None,
            ),
        )
        write_grade_manifest(store, "r", FRIDAY.isoformat())
        before = sorted(store.list_keys())
        argv = [
            *_minimal_argv("promote"),
            "--date",
            FRIDAY.isoformat(),
            "--store",
            str(tmp_path),
            "--dry-run",
        ]

        main(argv)  # must not raise at all -- an empty register is a valid read

        assert sorted(store.list_keys()) == before  # nothing NEW landed

    def test_dry_run_report_morning_completes_cleanly_with_a_board_seeded(
        self, tmp_path, monkeypatch
    ) -> None:
        """`report.morning` reads `board/current.json`; seeded with the same
        `_board()` fixture `tests/test_morning.py` itself builds against, so
        this reaches `morning.py::morning_handler`'s own pre-existing
        dry-run branch (print, no delivery, no write) rather than the
        `KeyError` an empty store produces before ever reaching it."""
        import json

        from crucible.keys import BOARD_CURRENT_KEY
        from crucible.store import LocalStore
        from tests.test_morning import _board

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        store = LocalStore(tmp_path)
        store.put_bytes(BOARD_CURRENT_KEY, json.dumps(_board()).encode())
        before = sorted(store.list_keys())
        argv = [
            "report.morning",
            "--date",
            FRIDAY.isoformat(),
            "--store",
            str(tmp_path),
            "--dry-run",
        ]

        main(argv)  # must not raise at all -- morning_handler's own dry-run branch

        assert sorted(store.list_keys()) == before  # nothing NEW landed
