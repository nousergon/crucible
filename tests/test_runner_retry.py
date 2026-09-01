"""One declared transient class, one retry, both attempts in the manifest.

Normative source: plan §11 risk 2.

"Autonomous" that only covers the success path is minimal-touch, not
zero-touch: two of August's three Saturday failures were transient, and each
was a 3am page for something that would have succeeded on a fresh instance.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.manifest import TRANSIENT_RETRY_REASONS, manifest_key
from crucible.runner import (
    MAX_ATTEMPTS,
    TRANSIENT_CLASSIFIERS,
    RunContext,
    SpotInterruptionError,
    classify_transient,
    run_job,
)
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC)


def _manifest(store, job="data.daily"):
    return json.loads(store.get_bytes(manifest_key(job, FRIDAY.isoformat())))


class TestTheDeclaredClass:
    def test_the_class_matches_the_schema_enum_exactly(self) -> None:
        """A class the schema cannot record cannot be retried — that is what
        keeps the two from drifting apart."""
        assert {r for r, _, _ in TRANSIENT_CLASSIFIERS} == set(TRANSIENT_RETRY_REASONS)

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (SpotInterruptionError("spot_interruption: signal 15"), "spot_interruption"),
            (TimeoutError("read timeout"), "provider_timeout"),
            (RuntimeError("HTTP 503 service unavailable"), "provider_5xx"),
            (RuntimeError("SlowDown: reduce your request rate"), "s3_throttling"),
        ],
    )
    def test_each_declared_failure_is_recognised(self, exc, expected) -> None:
        assert classify_transient(exc) == expected

    @pytest.mark.parametrize(
        "exc",
        [ValueError("schema drift in the fundamentals frame"), KeyError("missing_column")],
    )
    def test_an_undeclared_failure_is_not_transient(self, exc) -> None:
        """The classifier's fall-through is None. A fall-through of 'probably
        transient' would retry real defects and halve the rate at which they
        are noticed."""
        assert classify_transient(exc) is None


class TestRetry:
    def test_a_transient_failure_is_retried_once_and_both_attempts_are_recorded(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        calls = []

        def flaky(ctx: RunContext) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise SpotInterruptionError("spot_interruption: the instance is being reclaimed")

        run_job("data.daily", flaky, store=store, trading_day=FRIDAY, now=NOW)
        manifest = _manifest(store)
        assert manifest["status"] == "ok"
        assert manifest["attempts"] == [
            {"n": 1, "reason": "initial"},
            {"n": 2, "reason": "spot_interruption"},
        ]

    def test_exactly_one_manifest_is_written_across_both_attempts(self, tmp_path) -> None:
        """A `failed` manifest for attempt 1 and an `ok` one for attempt 2 at
        the same key would leave the store's answer to "did this run work"
        decided by write ordering."""
        store = LocalStore(tmp_path)
        calls = []

        def flaky(ctx: RunContext) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("read timeout")

        run_job("data.daily", flaky, store=store, trading_day=FRIDAY, now=NOW)
        assert len([k for k in store.list_keys("runs/") if k.endswith("run.json")]) == 1

    def test_a_retry_that_also_fails_writes_one_failed_manifest_and_raises(self, tmp_path) -> None:
        """The page fires only when the retry also fails."""
        store = LocalStore(tmp_path)

        def always(ctx: RunContext) -> None:
            raise TimeoutError("read timeout")

        with pytest.raises(TimeoutError):
            run_job("data.daily", always, store=store, trading_day=FRIDAY, now=NOW)
        manifest = _manifest(store)
        assert manifest["status"] == "failed"
        assert len(manifest["attempts"]) == MAX_ATTEMPTS

    def test_an_undeclared_failure_is_not_retried(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        calls = []

        def broken(ctx: RunContext) -> None:
            calls.append(1)
            raise ValueError("schema drift")

        with pytest.raises(ValueError):
            run_job("data.daily", broken, store=store, trading_day=FRIDAY, now=NOW)
        assert len(calls) == 1
        assert _manifest(store)["attempts"] == [{"n": 1, "reason": "initial"}]

    def test_the_retry_ladder_is_one_rung_and_never_a_loop(self, tmp_path) -> None:
        """A transient fault that survives a fresh instance is not transient,
        and a retry budget larger than one burns an afternoon of spot time
        before anyone is told."""
        store = LocalStore(tmp_path)
        calls = []

        def always(ctx: RunContext) -> None:
            calls.append(1)
            raise SpotInterruptionError("spot_interruption")

        with pytest.raises(SpotInterruptionError):
            run_job("data.daily", always, store=store, trading_day=FRIDAY, now=NOW)
        assert len(calls) == MAX_ATTEMPTS == 2

    def test_retry_can_be_disabled_for_a_caller_that_must_see_the_first_failure(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        calls = []

        def flaky(ctx: RunContext) -> None:
            calls.append(1)
            raise SpotInterruptionError("spot_interruption")

        with pytest.raises(SpotInterruptionError):
            run_job("smoke", flaky, store=store, trading_day=FRIDAY, now=NOW, transient_retry=False)
        assert len(calls) == 1

    def test_the_second_attempt_gets_a_fresh_context(self, tmp_path) -> None:
        """Its lineage is its own rather than a merge of two runs."""
        store = LocalStore(tmp_path)
        calls = []

        def flaky(ctx: RunContext) -> None:
            calls.append(1)
            ctx.record_rows(rows_in=10, rows_out=10)
            if len(calls) == 1:
                raise SpotInterruptionError("spot_interruption")

        run_job("data.daily", flaky, store=store, trading_day=FRIDAY, now=NOW)
        assert _manifest(store)["rows_in"] == 10  # not 20


class TestSpotGuard:
    def test_sigterm_without_an_external_guard_still_produces_a_manifest(self, tmp_path) -> None:
        """`run_job` must install `spot_interruption_guard` ITSELF (defect #8,
        alpha-engine-config-I9757): before the fix, `grep -rn
        spot_interruption_guard crucible` found only the definition and
        `track_c.py`'s smoke job — no track-A job installed it, and
        `data.weekly`, the 1-3 hour job plan §4.7 puts on a spot instance,
        ran unguarded. A SIGTERM there killed the process with no manifest
        at all: an ABSENCE page with the cause discarded, which is precisely
        what the guard's own docstring says it prevents.

        A previous version of this test wrapped the `run_job` call in its
        OWN `with spot_interruption_guard():` — which passes whether or not
        `run_job` installs the guard internally, because the external guard
        converts the signal before `run_job` ever sees it. That is a test
        that cannot fail (§11 risk 1): it asserts a property of the test's
        own scaffolding, not of `run_job`.

        This version calls `run_job` with NO external guard, in a
        subprocess: the delivered SIGTERM is real, and if `run_job` does not
        catch it the interpreter dies with the default disposition (process
        termination) rather than raising anything a `pytest.raises` could
        observe in this process — which is itself the bug being tested for,
        so the assertion is on the subprocess's OUTPUT (a manifest file on
        disk), not on how it exited.
        """
        import subprocess
        import sys

        sub_root = tmp_path / "sub"
        script = (
            "import datetime as dt, os, signal\n"
            "from crucible.runner import run_job\n"
            "from crucible.store import LocalStore\n"
            f"store = LocalStore({str(sub_root)!r})\n"
            "def reclaimed(ctx):\n"
            "    os.kill(os.getpid(), signal.SIGTERM)\n"
            "try:\n"
            "    run_job(\n"
            "        'data.weekly', reclaimed, store=store,\n"
            f"        trading_day=dt.date({FRIDAY.year}, {FRIDAY.month}, {FRIDAY.day}),\n"
            f"        now=dt.datetime({NOW.year}, {NOW.month}, {NOW.day}, {NOW.hour}, "
            f"{NOW.minute}, tzinfo=dt.UTC),\n"
            "        transient_retry=False,\n"
            "    )\n"
            "except BaseException:\n"
            "    pass\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )
        manifest_path = sub_root / "runs" / "data.weekly" / FRIDAY.isoformat() / "run.json"
        assert manifest_path.is_file(), (
            "run_job must install the spot-interruption guard itself, so a caller "
            "that installs no guard of its own still gets a manifest rather than a "
            "silent process death. subprocess exit "
            f"{result.returncode}, stderr:\n{result.stderr[-4000:]}"
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "failed"
        assert "spot_interruption" in manifest["reason"]

    def test_the_previous_handler_is_restored(self) -> None:
        import signal

        from crucible.runner import spot_interruption_guard

        before = signal.getsignal(signal.SIGTERM)
        with spot_interruption_guard():
            pass
        assert signal.getsignal(signal.SIGTERM) is before
