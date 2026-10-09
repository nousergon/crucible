"""The daily chain's two rules: never redo a done day, one writer at a time.

`alpha-engine-config-I12020`. On 2026-10-07 and 2026-10-08 the ArcticDB append
`data.daily` reads landed at 04:07Z and 03:18Z, after the fixed 01:15Z
`data-daily` schedule had already failed with no retry, and two paper sessions
were lost. nous-ergon-ops now also dispatches
`data.daily --skip-if-ok --then-serve` on the collection's own SUCCEEDED
event and keeps the clock schedules as backstops — so one session has several
starters, and these tests pin what makes that safe:

* `--skip-if-ok` files an `ok` no-op of its own over a day whose CANONICAL
  manifest reads `ok`, and re-runs a `failed` or absent one;
* the late-Friday case (an event after midnight ET, a Saturday whose session
  is Friday's) compiles Friday rather than filing the holiday no-op;
* every `data.daily` / `serve.daily` run holds one lease, decides inside it,
  and two runs never overlap;
* `--then-serve` serves M, U, S in that order, only after an `ok` compile.

Every test drives the real handlers against a real `LocalStore`, the real
`run_job` and real manifests. Dates are fixed literals.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import multiprocessing
import time
import types

import pytest

import crucible.daily_chain as daily_chain
import crucible.track_a as track_a
from crucible.cli import build_parser
from crucible.daily_chain import (
    DAILY_CHAIN_SERVE_ORDER,
    DailyChainLockTimeout,
    already_ok,
    daily_chain_lock,
    run_serve_chain,
)
from crucible.keys import DAILY_CHAIN_LOCK_KEY, calendar_day_discriminator
from crucible.manifest import manifest_key
from crucible.runner import run_job
from crucible.store import LocalStore

#: Thursday 2026-10-08: the second of the two lost sessions.
THURSDAY = dt.date(2026, 10, 8)
#: Friday 2026-10-09, and the Saturday a late Friday collection fires on in ET.
FRIDAY = dt.date(2026, 10, 9)
SATURDAY = dt.date(2026, 10, 10)


def _ok(store: LocalStore, job: str, day: dt.date, *, discriminator: str | None = None) -> None:
    run_job(
        job,
        lambda ctx: None,
        store=store,
        trading_day=day,
        run_mode="live",
        discriminator=discriminator,
    )


def _failed(store: LocalStore, job: str, day: dt.date, *, discriminator: str | None = None) -> None:
    def _boom(ctx):
        raise RuntimeError("source returned NO rows for the day")

    with pytest.raises(RuntimeError):
        run_job(
            job, _boom, store=store, trading_day=day, run_mode="live", discriminator=discriminator
        )


def _status(store: LocalStore, job: str, day: dt.date, discriminator: str | None = None) -> str:
    return json.loads(
        store.get_bytes(manifest_key(job, day.isoformat(), discriminator=discriminator))
    )["status"]


class TestAlreadyOk:
    def test_absent_is_not_done(self, tmp_path) -> None:
        assert not already_ok(LocalStore(tmp_path), "data.daily", THURSDAY.isoformat())

    def test_a_failed_manifest_is_not_done(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _failed(store, "data.daily", THURSDAY)
        assert not already_ok(store, "data.daily", THURSDAY.isoformat())

    def test_an_ok_manifest_is_done(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _ok(store, "data.daily", THURSDAY)
        assert already_ok(store, "data.daily", THURSDAY.isoformat())

    def test_a_manifest_nobody_can_validate_is_not_done(self, tmp_path) -> None:
        """The safe direction: a rerun overwrites, a skip would let it stand."""
        store = LocalStore(tmp_path)
        store.put_bytes(manifest_key("data.daily", THURSDAY.isoformat()), b'{"status": "ok"}')
        assert not already_ok(store, "data.daily", THURSDAY.isoformat())

    def test_only_the_slots_own_manifest_counts(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _ok(store, "serve.daily", THURSDAY, discriminator="m")
        assert already_ok(store, "serve.daily", THURSDAY.isoformat(), discriminator="m")
        assert not already_ok(store, "serve.daily", THURSDAY.isoformat(), discriminator="u")


class _AtomicLocalStore(LocalStore):
    """`LocalStore` with its replace-by-version made atomic across processes.

    `LocalStore.compare_and_swap` documents that its REPLACE case is a
    check-then-write two processes can both pass; S3's `IfMatch` is atomic at
    the service, and that is the primitive the lease is built on. An `flock`
    around the call gives the test the production primitive's guarantee, so
    what is graded here is the lease logic and not the laptop backend.
    """

    def compare_and_swap(self, key: str, expected: str, payload: bytes) -> str:
        with open(self.root / ".cas.lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                return super().compare_and_swap(key, expected, payload)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _hold_and_record(root: str, name: str) -> None:
    store = _AtomicLocalStore(root)
    with daily_chain_lock(store, job=name, trading_day="2026-10-08", poll_s=0.01):
        start = time.time()
        time.sleep(0.2)
        end = time.time()
    (store.root / f"span-{name}.txt").write_text(f"{start} {end}")


class TestTheLease:
    def test_acquire_then_release_leaves_a_released_lease(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        with daily_chain_lock(store, job="data.daily", trading_day="2026-10-08") as holder:
            held = json.loads(store.get_bytes(DAILY_CHAIN_LOCK_KEY))
            assert held["holder"] == holder and held["released_utc"] is None
        released = json.loads(store.get_bytes(DAILY_CHAIN_LOCK_KEY))
        assert released["released_utc"] is not None
        assert released["schema_version"] == "daily_chain_lock.v1"
        # And the next run can take it at once.
        with daily_chain_lock(store, job="serve.daily[m]", trading_day="2026-10-08"):
            pass

    def test_a_live_lease_held_by_someone_else_times_out_loudly(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        now = dt.datetime(2026, 10, 9, 2, 0, tzinfo=dt.UTC)
        store.put_bytes(
            DAILY_CHAIN_LOCK_KEY,
            json.dumps(
                {
                    "holder": "data.daily@other-box",
                    "job": "data.daily",
                    "expires_utc": "2026-10-09T03:00:00Z",
                    "released_utc": None,
                }
            ).encode(),
        )
        clock = {"t": now}

        def _sleep(seconds):
            clock["t"] += dt.timedelta(seconds=seconds)

        with pytest.raises(DailyChainLockTimeout, match="other-box"):
            with daily_chain_lock(
                store,
                job="serve.daily[m]",
                trading_day="2026-10-08",
                wait=dt.timedelta(minutes=5),
                clock=lambda: clock["t"],
                sleep=_sleep,
            ):
                pytest.fail("the body must not run without the lease")

    def test_an_expired_lease_is_taken_over(self, tmp_path) -> None:
        """A box reclaimed while holding it cannot block the next starter."""
        store = LocalStore(tmp_path)
        store.put_bytes(
            DAILY_CHAIN_LOCK_KEY,
            json.dumps(
                {"holder": "dead-box", "expires_utc": "2026-10-09T01:00:00Z", "released_utc": None}
            ).encode(),
        )
        with daily_chain_lock(
            store,
            job="data.daily",
            trading_day="2026-10-08",
            clock=lambda: dt.datetime(2026, 10, 9, 1, 0, 1, tzinfo=dt.UTC),
        ) as holder:
            assert json.loads(store.get_bytes(DAILY_CHAIN_LOCK_KEY))["holder"] == holder

    def test_a_corrupt_lease_does_not_wedge_every_run(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(DAILY_CHAIN_LOCK_KEY, b"not json")
        with daily_chain_lock(store, job="data.daily", trading_day="2026-10-08"):
            pass

    def test_two_runs_never_overlap(self, tmp_path) -> None:
        """Three PROCESSES (the M, U and S boxes), not threads: `LocalStore`'s
        create-if-absent names its temp file by pid, like separate boxes do."""
        ctx = multiprocessing.get_context("spawn")
        workers = [
            ctx.Process(target=_hold_and_record, args=(str(tmp_path), f"serve.daily[{s}]"))
            for s in "mus"
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0
        spans = sorted(
            tuple(float(x) for x in path.read_text().split())
            for path in tmp_path.glob("span-*.txt")
        )
        assert len(spans) == 3
        for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
            assert end <= start, spans

    def test_the_lease_is_renewed_while_the_body_runs(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        with daily_chain_lock(
            store, job="data.daily", trading_day="2026-10-08", lease=dt.timedelta(seconds=0.3)
        ):
            first = json.loads(store.get_bytes(DAILY_CHAIN_LOCK_KEY))["expires_utc"]
            time.sleep(1.2)
            renewed = json.loads(store.get_bytes(DAILY_CHAIN_LOCK_KEY))["expires_utc"]
        assert renewed > first


def _data_args(store_root, day: dt.date, **over) -> argparse.Namespace:
    base = dict(
        job="data.daily",
        date=None,
        store=str(store_root),
        symbols=None,
        dry_run=False,
        run_mode="live",
        skip_if_ok=True,
        then_serve=False,
    )
    base.update(over)
    ns = argparse.Namespace(**base)
    ns.trading_day = day
    return ns


class _Compiled(Exception):
    """Raised by a stub source: the handler reached the compile path."""


class TestDataDailySkipIfOk:
    def test_a_done_day_files_a_no_op_and_never_compiles(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _ok(store, "data.daily", THURSDAY)
        canonical = store.get_bytes(manifest_key("data.daily", THURSDAY.isoformat()))
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)
        monkeypatch.setattr(track_a, "_source", lambda args, config: pytest.fail("compiled"))

        assert track_a.handle_data_daily(_data_args(tmp_path, THURSDAY)) == 0

        assert store.get_bytes(manifest_key("data.daily", THURSDAY.isoformat())) == canonical
        noop = calendar_day_discriminator(THURSDAY, suffix="already-ok")
        assert _status(store, "data.daily", THURSDAY, noop) == "ok"

    def test_a_failed_day_is_compiled_again(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _failed(store, "data.daily", THURSDAY)
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)

        def _source(args, config):
            raise _Compiled

        monkeypatch.setattr(track_a, "_source", _source)
        with pytest.raises(_Compiled):
            track_a.handle_data_daily(_data_args(tmp_path, THURSDAY))

    def test_a_late_friday_collection_compiles_friday_not_a_holiday_no_op(
        self, tmp_path, monkeypatch
    ) -> None:
        """An event after midnight ET on a Friday fires on a Saturday whose
        session is Friday's. Without `--skip-if-ok` the holiday guard files a
        no-op there; with it the day is still owed its compile."""
        monkeypatch.setattr(track_a, "_today", lambda: SATURDAY)

        def _source(args, config):
            raise _Compiled

        monkeypatch.setattr(track_a, "_source", _source)
        with pytest.raises(_Compiled):
            track_a.handle_data_daily(_data_args(tmp_path, FRIDAY))

    def test_without_the_flag_a_holiday_firing_is_unchanged(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        monkeypatch.setattr(track_a, "_today", lambda: SATURDAY)
        assert track_a.handle_data_daily(_data_args(tmp_path, FRIDAY, skip_if_ok=False)) == 0
        assert _status(store, "data.daily", FRIDAY, SATURDAY.isoformat()) == "ok"
        assert not store.exists(manifest_key("data.daily", FRIDAY.isoformat()))

    def test_the_lease_is_released_after_the_run(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _ok(store, "data.daily", THURSDAY)
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)
        track_a.handle_data_daily(_data_args(tmp_path, THURSDAY))
        assert json.loads(store.get_bytes(DAILY_CHAIN_LOCK_KEY))["released_utc"] is not None


def _fake_servers(calls: list[str]) -> dict[str, types.SimpleNamespace]:
    def _server(slot):
        def serve_daily(ctx, settings):
            calls.append(slot)
            return {}

        return types.SimpleNamespace(serve_daily=serve_daily)

    return {slot: _server(slot) for slot in DAILY_CHAIN_SERVE_ORDER}


def _serve_args(store_root, day: dt.date, slot: str, **over) -> argparse.Namespace:
    base = dict(
        job="serve.daily",
        date=day.isoformat(),
        store=str(store_root),
        dry_run=False,
        run_mode="live",
        slot=slot,
        skip_if_ok=True,
    )
    base.update(over)
    ns = argparse.Namespace(**base)
    ns.trading_day = day
    return ns


class TestServeDailySkipIfOk:
    def test_a_served_slot_files_a_no_op_and_does_not_serve(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _ok(store, "serve.daily", THURSDAY, discriminator="m")
        calls: list[str] = []
        monkeypatch.setattr(track_a, "daily_servers", lambda: _fake_servers(calls))
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)

        assert track_a.handle_serve_daily(_serve_args(tmp_path, THURSDAY, "m")) == 0

        assert calls == []
        assert (
            _status(store, "serve.daily", THURSDAY, f"m.{THURSDAY.isoformat()}-already-ok") == "ok"
        )

    def test_a_failed_slot_is_served_again(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _failed(store, "serve.daily", THURSDAY, discriminator="u")
        calls: list[str] = []
        monkeypatch.setattr(track_a, "daily_servers", lambda: _fake_servers(calls))
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)

        assert track_a.handle_serve_daily(_serve_args(tmp_path, THURSDAY, "u")) == 0

        assert calls == ["u"]
        assert _status(store, "serve.daily", THURSDAY, "u") == "ok"


class TestThenServe:
    def _chain(self, monkeypatch) -> list[dict]:
        seen: list[dict] = []

        def _fake_chain(trading_day, **kwargs):
            seen.append({"trading_day": trading_day, **kwargs})
            return dict.fromkeys(DAILY_CHAIN_SERVE_ORDER, 0)

        monkeypatch.setattr(track_a, "run_serve_chain", _fake_chain)
        return seen

    def test_an_ok_compile_is_followed_by_the_serve_chain(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _ok(store, "data.daily", THURSDAY)
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)
        seen = self._chain(monkeypatch)
        assert track_a.handle_data_daily(_data_args(tmp_path, THURSDAY, then_serve=True)) == 0
        assert [s["trading_day"] for s in seen] == [THURSDAY]

    def test_a_failed_compile_serves_nothing(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)
        seen = self._chain(monkeypatch)

        def _source(args, config):
            raise _Compiled

        monkeypatch.setattr(track_a, "_source", _source)
        with pytest.raises(_Compiled):
            track_a.handle_data_daily(_data_args(tmp_path, THURSDAY, then_serve=True))
        assert seen == []

    def test_a_failed_serve_slot_fails_the_chain(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _ok(store, "data.daily", THURSDAY)
        monkeypatch.setattr(track_a, "_today", lambda: THURSDAY)
        monkeypatch.setattr(
            track_a, "run_serve_chain", lambda trading_day, **kw: {"m": 0, "u": 1, "s": 0}
        )
        assert track_a.handle_data_daily(_data_args(tmp_path, THURSDAY, then_serve=True)) == 1


class TestRunServeChain:
    def test_the_slots_run_in_read_order_with_the_day_pinned(self) -> None:
        argvs: list[list[str]] = []
        codes = run_serve_chain(
            THURSDAY,
            store="s3://bucket/crucible",
            run_mode="live",
            main=lambda argv: argvs.append(argv) or 0,
            servers=dict.fromkeys("mus"),
        )
        assert codes == {"m": 0, "u": 0, "s": 0}
        assert [argv[2] for argv in argvs] == ["m", "u", "s"]
        for argv in argvs:
            assert argv[:2] == ["serve.daily", "--slot"]
            assert argv[argv.index("--date") + 1] == "2026-10-08"
            assert "--skip-if-ok" in argv
            assert argv[argv.index("--run-mode") + 1] == "live"

    def test_every_slot_runs_after_a_failure_and_the_failure_is_reported(self) -> None:
        def _main(argv):
            if argv[2] == "m":
                raise RuntimeError("fit of record could not be proven")
            return 0

        codes = run_serve_chain(
            THURSDAY, store=None, run_mode=None, main=_main, servers=dict.fromkeys("mus")
        )
        assert codes == {"m": 1, "u": 0, "s": 0}

    def test_a_new_daily_server_cannot_be_left_out_silently(self) -> None:
        with pytest.raises(RuntimeError, match="DAILY_CHAIN_SERVE_ORDER"):
            run_serve_chain(
                THURSDAY,
                store=None,
                run_mode=None,
                main=lambda argv: 0,
                servers=dict.fromkeys("musr"),
            )

    def test_the_declared_order_names_exactly_the_live_daily_servers(self) -> None:
        from crucible.slots import daily_servers

        assert set(daily_servers()) == set(DAILY_CHAIN_SERVE_ORDER)


class TestTheDispatchArgvParses:
    """The exact argv nous-ergon-ops' dispatches pass (`crucible-v2.yaml`)."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["data.daily", "--skip-if-ok", "--then-serve"],
            ["serve.daily", "--slot", "m", "--skip-if-ok"],
            ["serve.daily", "--slot", "u", "--skip-if-ok"],
            ["serve.daily", "--slot", "s", "--skip-if-ok"],
        ],
    )
    def test_it_parses(self, argv) -> None:
        args = build_parser().parse_args(argv)
        assert args.skip_if_ok is True

    def test_then_serve_is_data_daily_only(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["serve.daily", "--slot", "m", "--then-serve"])


def test_the_lease_key_lives_under_the_already_granted_runs_prefix() -> None:
    assert DAILY_CHAIN_LOCK_KEY.startswith("runs/_")
    assert daily_chain.DAILY_CHAIN_LOCK_KEY is DAILY_CHAIN_LOCK_KEY
