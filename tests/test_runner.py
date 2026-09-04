"""The runner's one guarantee: a job that dies still writes its manifest.

Normative source: plan §4.2, §9.2 ("A dying job flushes run.json with
`status: failed`, its spend and cause before exit; a trap in the runner
guarantees it") and §4.6 (a `failed` manifest is one of exactly two page
conditions).

Written before `crucible/runner.py` and seen failing.

The failure path is the subject here, not the success path. "Works flawlessly
from day 1" is proven on the failure path — a harness whose telemetry only
appears when nothing went wrong is a harness with no telemetry at the moment
it is needed.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.manifest import ManifestValidationError, manifest_key, validate
from crucible.runner import RunContext, run_job
from crucible.store import LocalStore

TRADING_DAY = dt.date(2026, 8, 28)


def _read_manifest(store: LocalStore, job: str) -> dict:
    return json.loads(store.get_bytes(manifest_key(job, TRADING_DAY.isoformat())))


class TestSuccessPath:
    def test_a_clean_job_writes_an_ok_manifest(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            ctx.record_output("signals/2026-08-28/signals.json", b'{"names": []}')

        run_job("experiment.run", job, store=store, trading_day=TRADING_DAY)

        doc = _read_manifest(store, "experiment.run")
        validate(doc)
        assert doc["status"] == "ok"
        assert doc["reason"] == ""
        assert doc["outputs"][0]["key"] == "signals/2026-08-28/signals.json"

    def test_the_manifest_is_keyed_by_trading_day_not_calendar_date(self, tmp_path) -> None:
        """A Saturday run writes Friday's key and records Saturday only as
        provenance (§4.12)."""
        store = LocalStore(tmp_path)
        saturday = dt.datetime(2026, 8, 29, 10, 0)

        run_job("data.weekly", lambda ctx: None, store=store, now=saturday)

        doc = _read_manifest(store, "data.weekly")
        assert doc["trading_day"] == "2026-08-28"
        assert doc["calendar_date"] == "2026-08-29"
        assert store.exists("runs/data.weekly/2026-08-28/run.json")
        assert not store.exists("runs/data.weekly/2026-08-29/run.json")

    def test_run_id_is_the_same_on_the_manifest_and_the_context(self, tmp_path) -> None:
        """§9.2: one run_id on every log line, manifest, alert and cost row.
        A context whose id differs from the manifest's makes the correlation
        identity useless in exactly the case it is needed."""
        store = LocalStore(tmp_path)
        seen: list[str] = []

        run_job("smoke", lambda ctx: seen.append(ctx.run_id), store=store, trading_day=TRADING_DAY)

        assert _read_manifest(store, "smoke")["run_id"] == seen[0]


class TestFailurePath:
    def test_an_exception_still_writes_a_failed_manifest_and_re_raises(self, tmp_path) -> None:
        """The whole point of the runner. `try/finally`, not `try/except`:
        the manifest is written AND the exception continues to propagate, so
        the process exit code is non-zero and the scheduler sees a failure."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            raise RuntimeError("ArcticDB library 'macro' returned zero rows")

        with pytest.raises(RuntimeError, match="zero rows"):
            run_job("data.daily", job, store=store, trading_day=TRADING_DAY)

        doc = _read_manifest(store, "data.daily")
        validate(doc)
        assert doc["status"] == "failed"
        assert "zero rows" in doc["reason"]
        assert "RuntimeError" in doc["reason"]

    def test_the_reason_is_never_empty_on_failure(self, tmp_path) -> None:
        """An exception with no message still produces an actionable reason.
        `raise ValueError()` must not become `reason: ""` — that is the shape
        that made three consecutive Saturday failures indistinguishable."""
        store = LocalStore(tmp_path)

        with pytest.raises(ValueError):
            run_job(
                "data.daily",
                lambda ctx: (_ for _ in ()).throw(ValueError()),
                store=store,
                trading_day=TRADING_DAY,
            )

        doc = _read_manifest(store, "data.daily")
        validate(doc)
        assert doc["status"] == "failed"
        assert doc["reason"].strip() != ""
        assert "ValueError" in doc["reason"]

    def test_partial_work_before_the_exception_is_still_recorded(self, tmp_path) -> None:
        """Telemetry recorded before the failure survives it. A failed run
        that reports zero spend and zero rows is a failed run nobody can
        diagnose — and its cost is silently unattributed."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            ctx.record_rows(rows_in=903, rows_out=0)
            ctx.record_rejected("stale_price_history", 12)
            ctx.record_cost(0.17)
            raise TimeoutError("router upstream timed out after 120s")

        with pytest.raises(TimeoutError):
            run_job("experiment.run", job, store=store, trading_day=TRADING_DAY)

        doc = _read_manifest(store, "experiment.run")
        validate(doc)
        assert doc["status"] == "failed"
        assert doc["rows_in"] == 903
        assert doc["rows_rejected"] == [{"reason": "stale_price_history", "count": 12}]
        assert doc["cost_usd"] == pytest.approx(0.17)

    def test_a_keyboard_interrupt_also_writes_the_manifest(self, tmp_path) -> None:
        """BaseException, not Exception. A spot reclamation arrives as a
        signal, and catching only `Exception` is how the most common real
        failure produces no manifest at all — an ABSENCE page instead of a
        FAILURE page, with the cause discarded."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_job("data.weekly", job, store=store, trading_day=TRADING_DAY)

        doc = _read_manifest(store, "data.weekly")
        validate(doc)
        assert doc["status"] == "failed"
        assert "KeyboardInterrupt" in doc["reason"]


class TestNoThirdState:
    def test_the_runner_cannot_be_asked_for_a_third_status(self, tmp_path) -> None:
        """There is no API by which a job declares itself skipped or partial.
        The status is derived from whether the callable returned or raised,
        and nothing else — which is what makes §11.1 structural rather than a
        rule someone has to remember."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            with pytest.raises(ValueError, match="ok|failed"):
                ctx.set_status("skipped")

        run_job("data.daily", job, store=store, trading_day=TRADING_DAY)
        assert _read_manifest(store, "data.daily")["status"] == "ok"

    def test_the_written_manifest_always_validates(self, tmp_path) -> None:
        """The runner validates before writing. A run that could emit a
        non-conformant manifest defeats the schema entirely, and the failure
        path is where an incomplete document would come from."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            run_job("data.daily", job, store=store, trading_day=TRADING_DAY)

        try:
            validate(_read_manifest(store, "data.daily"))
        except ManifestValidationError as exc:  # pragma: no cover - failure detail
            pytest.fail(f"runner wrote a non-conformant manifest: {exc}")


class TestCostAssertion:
    def test_cost_usd_below_the_llm_call_total_is_refused(self, tmp_path) -> None:
        """schemas/run_manifest.v1.json's `cost_usd` description claims "the
        runner asserts that, since a schema cannot" (a schema cannot
        cross-reference two fields of the same document). Before this fix
        nothing in the runner did — it only accumulated and rounded. A job
        whose own bookkeeping (a negative `record_cost`, the plausible real
        case: a refund, a cache-hit credit applied twice) pulls `cost_usd`
        below what `llm_calls[].usd` itself reports must be refused at write
        time, not discovered by a reader doing the arithmetic later."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            ctx.record_llm_call(
                {
                    "callsite_id": "research.rank.v1",
                    "model_requested": "tier:high",
                    "model_served": "glm-4.6",
                    "tokens_in": 100,
                    "tokens_out": 10,
                    "cache_read": 0,
                    "cache_write": 0,
                    "usd": 1.00,
                }
            )
            # bookkeeping bug: pulls cost_usd below the llm total while
            # staying non-negative, so this exercises the cost_usd-vs-
            # llm_calls cross-check specifically rather than tripping the
            # schema's unrelated `cost_usd >= 0` minimum.
            ctx.record_cost(-0.50)

        with pytest.raises(ValueError, match="less than the sum of llm_calls"):
            run_job("experiment.run", job, store=store, trading_day=TRADING_DAY)


class TestDiscriminator:
    """alpha-engine-config-I9781: a discriminator keeps concurrent or
    repeated writers of the same job+trading_day from colliding."""

    def test_two_slots_on_the_same_trading_day_write_two_manifests(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        run_job(
            "experiment.run",
            lambda ctx: ctx.record_output("u.json", b"{}"),
            store=store,
            trading_day=TRADING_DAY,
            discriminator="u",
        )
        run_job(
            "experiment.run",
            lambda ctx: ctx.record_output("r.json", b"{}"),
            store=store,
            trading_day=TRADING_DAY,
            discriminator="r",
        )

        iso = TRADING_DAY.isoformat()
        u_doc = json.loads(store.get_bytes(manifest_key("experiment.run", iso, discriminator="u")))
        r_doc = json.loads(store.get_bytes(manifest_key("experiment.run", iso, discriminator="r")))
        assert u_doc["discriminator"] == "u"
        assert r_doc["discriminator"] == "r"
        assert u_doc["outputs"][0]["key"] == "u.json"
        assert r_doc["outputs"][0]["key"] == "r.json"
        assert not store.exists(manifest_key("experiment.run", TRADING_DAY.isoformat()))

    def test_no_discriminator_writes_the_undiscriminated_key_with_no_field(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_job("data.daily", lambda ctx: None, store=store, trading_day=TRADING_DAY)
        doc = _read_manifest(store, "data.daily")
        assert "discriminator" not in doc
        validate(doc)

    def test_a_callable_discriminator_resolves_against_the_built_context(self, tmp_path) -> None:
        """`alerts.sweep`'s shape: the discriminator (`calendar_date`) is
        only known once the runner has resolved it, not at the call site."""
        store = LocalStore(tmp_path)
        friday_evening = dt.datetime(2026, 8, 28, 21, 0)
        saturday_evening = dt.datetime(2026, 8, 29, 21, 0)
        sunday_evening = dt.datetime(2026, 8, 30, 21, 0)

        for moment in (friday_evening, saturday_evening, sunday_evening):
            run_job(
                "alerts.sweep",
                lambda ctx: None,
                store=store,
                now=moment,
                discriminator=lambda ctx: ctx.calendar_date.isoformat(),
            )

        # All three resolve to Friday's trading day, and each firing left its
        # own manifest rather than the last one overwriting the first two.
        for calendar_date in ("2026-08-28", "2026-08-29", "2026-08-30"):
            doc = json.loads(
                store.get_bytes(
                    manifest_key("alerts.sweep", "2026-08-28", discriminator=calendar_date)
                )
            )
            assert doc["trading_day"] == "2026-08-28"
            assert doc["calendar_date"] == calendar_date
            assert doc["discriminator"] == calendar_date


class TestKeyRefusal:
    def test_the_runner_refuses_a_non_trading_day(self, tmp_path) -> None:
        """§4.12: a caller cannot force a Saturday key by passing one. The
        runner refuses rather than resolving silently — a caller that asked
        for a wrong day has a bug, and quietly correcting it hides the bug."""
        from crucible.calendar import NonTradingDayKeyError

        store = LocalStore(tmp_path)
        with pytest.raises(NonTradingDayKeyError):
            run_job(
                "data.daily",
                lambda ctx: None,
                store=store,
                trading_day=dt.date(2026, 8, 29),
            )


class TestDryRun:
    """alpha-engine-config-I9922: `run_job(dry_run=True)` writes nothing at
    all — before this, `report.morning --dry-run` was documented as
    "renders and files nothing" while `run_job` wrote the manifest anyway,
    leaving a real `ok` firing in the production store for `alerts.sweep`
    and the board to read as a genuine run."""

    def test_dry_run_leaves_the_store_completely_empty(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        run_job("data.daily", lambda ctx: None, store=store, trading_day=TRADING_DAY, dry_run=True)

        assert list(store.list_keys()) == []

    def test_dry_run_false_writes_the_manifest_as_normal(self, tmp_path) -> None:
        """The control: the same job, `dry_run=False` (the default), does
        write — proving the emptiness above is `dry_run`'s effect and not an
        accident of the fixture."""
        store = LocalStore(tmp_path)

        run_job("data.daily", lambda ctx: None, store=store, trading_day=TRADING_DAY, dry_run=False)

        doc = _read_manifest(store, "data.daily")
        validate(doc)
        assert doc["status"] == "ok"

    def test_dry_run_still_runs_fn_but_records_nothing_from_it(self, tmp_path) -> None:
        """A dry run still calls `fn` — a caller like `report.morning` needs
        the rendered result to print it — but whatever `fn` recorded onto the
        `RunContext` (rows, cost, metrics) is discarded rather than folded
        into a manifest, since no manifest is written."""
        store = LocalStore(tmp_path)
        called: list[bool] = []

        def job(ctx: RunContext) -> None:
            called.append(True)
            ctx.record_rows(rows_in=10, rows_out=10)
            ctx.record_cost(1.23)

        run_job("report", job, store=store, trading_day=TRADING_DAY, dry_run=True)

        assert called == [True]
        assert list(store.list_keys()) == []

    def test_dry_run_on_a_raising_job_still_reraises_and_still_writes_nothing(
        self, tmp_path
    ) -> None:
        """Fail loud survives a dry run: the exception is never swallowed
        just because nothing was going to be written. The one thing dry_run
        changes is that the FAILURE, too, produces no manifest — a caller
        that dry-runs a job and hits a real bug in it still sees the
        exception at the terminal."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            run_job("report", job, store=store, trading_day=TRADING_DAY, dry_run=True)

        assert list(store.list_keys()) == []

    def test_dry_run_alone_does_not_stop_fn_writing_through_a_real_store(self, tmp_path) -> None:
        """BLOCKING F1 (independent review of crucible-PR74, 2026-09-03),
        turned into a passing regression test. `run_job(dry_run=True)` skips
        `run_job`'s OWN manifest write — it does not, and was never meant to,
        stop `fn` writing through whatever `store` it was handed. Reproduced
        exactly as the reviewer measured: a body calling
        `ctx.record_output("board/current.json", ...)` under `dry_run=True`
        against a REAL `LocalStore` lands the artifact in the store with no
        manifest.

        This is not a defect in `run_job` — the fix is that no caller ever
        hands a job body a real, writable store under `--dry-run` in the
        first place (`crucible.store.read_only`, wired in at every CLI
        store-construction point; see `tests/test_cli_and_alerts.py::
        TestDryRunNeverWrites` for the store-wrapped, CLI-level guarantee).
        This test documents and pins the boundary: `run_job` itself does not
        and must not try to guess whether `store` is real or wrapped."""
        store = LocalStore(tmp_path)

        def job(ctx: RunContext) -> None:
            ctx.record_output("board/current.json", b"{}")

        run_job("board", job, store=store, trading_day=TRADING_DAY, dry_run=True)

        # The artifact landed for real — `run_job` alone cannot and does not
        # prevent this — while no manifest exists, exactly the split the
        # reviewer's probe demonstrated.
        assert store.exists("board/current.json")
        assert list(store.list_keys("runs/")) == []

    def test_dry_run_against_a_read_only_store_refuses_the_write_loudly(self, tmp_path) -> None:
        """The actual fix, exercised at the `run_job` boundary: when the
        store IS wrapped (`crucible.store.read_only`, as every CLI handler
        now resolves its store under `--dry-run`), the same body from the
        test above raises instead of writing — `run_job` needed no changes
        of its own to get this; it is a property of the store it was
        handed."""
        from crucible.store import DryRunWriteRefusedError, read_only

        store = read_only(LocalStore(tmp_path))

        def job(ctx: RunContext) -> None:
            ctx.record_output("board/current.json", b"{}")

        with pytest.raises(DryRunWriteRefusedError, match="board/current.json"):
            run_job("board", job, store=store, trading_day=TRADING_DAY, dry_run=True)

        assert list(store.list_keys()) == []
