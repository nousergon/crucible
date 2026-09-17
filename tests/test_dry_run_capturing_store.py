"""The write-capturing store: `--dry-run` executes the real job body.

`alpha-engine-config-I11012`.

The defect this closes, measured: a laptop `--dry-run` of
`experiment.backfill --slot u --arm attractiveness --from 2025-11-17 --to
2026-09-09` reported "would produce 205 session(s)". The real dispatch died
in under two minutes on the FIRST session. No dry run in this CLI executed a
job body, because executing it writes — `crucible.store.read_only` made a dry
run safe by refusing at the first write — so no rehearsal could fail the way
its run fails, and one certified a command that was structurally broken.

The fix is a store whose writes are RECORDED rather than performed, so the
real body runs against real reads and reports the exact key set it would
write. These are the store-level properties; the per-job coverage matrix is
`tests/test_cli_and_alerts.py::TestDryRunNeverWrites`.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.store import (
    CAPTURED_URL_SCHEME,
    ETAG_ABSENT,
    CaptureLedger,
    LocalStore,
    PointerConflictError,
    S3Store,
    Store,
    active_capture_ledger,
    begin_capture,
    capture_ledger_of,
    capturing,
    end_capture,
    is_capturing,
    open_store,
    sha256_hex,
)


@pytest.fixture
def backing(tmp_path) -> LocalStore:
    return LocalStore(tmp_path / "store")


class TestNoWriteReachesTheBackend:
    """The whole point. A dry run writes nothing — structurally."""

    def test_put_bytes_records_and_the_directory_stays_empty(self, backing, tmp_path) -> None:
        store = capturing(backing)

        digest = store.put_bytes("board/current.json", b'{"schema_version": "board.v3"}')

        assert digest == sha256_hex(b'{"schema_version": "board.v3"}')
        assert sorted(LocalStore(tmp_path / "store").list_keys()) == []
        ledger = capture_ledger_of(store)
        assert ledger is not None
        assert ledger.keys == ("board/current.json",)
        (write,) = ledger.writes
        assert write.method == "put_bytes"
        assert write.size_bytes == 30
        assert write.schema_version == "board.v3"

    def test_compare_and_swap_records_and_the_directory_stays_empty(
        self, backing, tmp_path
    ) -> None:
        store = capturing(backing)

        store.compare_and_swap("champions/u/current.json", ETAG_ABSENT, b'{"arm": "x"}')

        assert sorted(LocalStore(tmp_path / "store").list_keys()) == []
        (write,) = capture_ledger_of(store).writes
        assert write.method == "compare_and_swap"
        assert write.schema_version is None  # the payload declares none

    def test_there_is_no_flag_that_re_enables_the_write(self, backing) -> None:
        """The constraint stated in the issue: impossible BY TYPE, not by
        branch. A real `Store` with writes disabled by a flag is one `if`
        away from a production write; the capturing override never calls the
        backend's write at all, so there is no `if` to get wrong.

        Asserted by construction rather than by prose: the overriding
        functions' bytecode names no `put_bytes`/`compare_and_swap` on the
        base class, and the only `super`-ish reference in the class is on the
        READ path."""
        store = capturing(backing)
        for name in Store.MUTATORS:
            override = type(store).__dict__[name]
            referenced = set(override.__code__.co_names)
            assert "super" not in referenced, (
                f"{name} reaches its base implementation; a dry-run write is then one "
                "resolution away from the real backend."
            )

    def test_a_mutator_with_no_capture_override_is_refused_at_class_build(
        self, backing, monkeypatch
    ) -> None:
        """A third mutator added to the interface without a capture override
        would write to the real backend under `--dry-run`. It is refused when
        the capturing class is built, which is the first moment it can be
        seen — the same shape as `Store.MUTATORS` existing at all."""
        from crucible import store as store_module

        monkeypatch.setattr(store_module, "_CAPTURING_CLASS_CACHE", {})
        monkeypatch.setattr(Store, "MUTATORS", (*Store.MUTATORS, "delete"))
        with pytest.raises(RuntimeError, match="delete"):
            capturing(backing)


class TestTheReadsAreReal:
    def test_a_read_falls_through_to_the_backend(self, backing) -> None:
        backing.put_bytes("features/2026-09-11.json", b"{}")
        store = capturing(backing)

        assert store.get_bytes("features/2026-09-11.json") == b"{}"
        assert store.exists("features/2026-09-11.json")

    def test_the_run_reads_back_what_it_would_have_written(self, backing) -> None:
        """A job that writes a key and then reads it back is the ordinary
        shape (the arm register, the board, every pointer). A rehearsal whose
        read raised `KeyError` where the real run succeeds would fail in a way
        the run does not — the mirror image of the defect being closed."""
        store = capturing(backing)
        store.put_bytes("arms/u/register.json", b'{"arms": []}')

        assert store.get_bytes("arms/u/register.json") == b'{"arms": []}'
        assert store.exists("arms/u/register.json")
        assert store.etag("arms/u/register.json") == sha256_hex(b'{"arms": []}')
        assert "arms/u/register.json" in list(store.list_keys("arms/"))

    def test_a_missing_key_still_raises(self, backing) -> None:
        """Absence stays a first-class fact. A capturing store that answered
        `b""` for an absent key would make every dry run of a job whose input
        is missing read as a job whose input is empty."""
        store = capturing(backing)
        with pytest.raises(KeyError):
            store.get_bytes("nothing/here.json")
        assert not store.exists("nothing/here.json")
        assert store.etag("nothing/here.json") == ETAG_ABSENT


class TestItFailsTheWayTheRunFails:
    def test_a_lost_pointer_race_conflicts_in_a_dry_run_too(self, backing) -> None:
        backing.put_bytes("champions/u/current.json", b'{"arm": "incumbent"}')
        store = capturing(backing)

        with pytest.raises(PointerConflictError):
            store.compare_and_swap("champions/u/current.json", ETAG_ABSENT, b'{"arm": "new"}')

    def test_a_key_the_backend_would_refuse_is_refused(self, backing) -> None:
        """`LocalStore._path` refuses an absolute or traversing key. A dry run
        that never reaches the write path never reaches that refusal either —
        so it would report `/etc/hosts` as a key it would write, and the real
        run would raise on it."""
        store = capturing(backing)
        with pytest.raises(ValueError, match="never traverse"):
            store.put_bytes("/etc/hosts", b"x")
        assert capture_ledger_of(store).keys == ()

    def test_a_local_object_lock_request_is_still_unimplemented(self, backing) -> None:
        """Recorded, not silently accepted: `LocalStore` has no Object Lock
        concept, and a capture that swallowed the request would make a dry run
        of a release publish green over a guarantee the backend cannot give.

        The capture records the REQUEST — `object_lock_mode` is on the
        `CapturedWrite` — which is what a rehearsal should report; the
        backend's own `NotImplementedError` belongs to the real write and is
        not re-raised here, because the real run against S3 (the only backend
        a locked publish uses) would not raise it either."""
        store = capturing(backing)
        store.put_bytes("releases/abc/wheel", b"x", object_lock_mode="COMPLIANCE")
        (write,) = capture_ledger_of(store).writes
        assert write.object_lock_mode == "COMPLIANCE"


class TestIsinstanceStaysTrue:
    def test_a_capturing_s3_store_is_still_an_s3_store(self) -> None:
        """`release_retention.py`, `release.py` and `release_lock_sweep.py`
        each do `isinstance(store, S3Store)` to refuse a laptop directory
        outright. A composition wrapper would fail every one of those checks
        under `--dry-run`, trading this bug for a wrong-backend-type one."""
        store = capturing(S3Store("some-bucket", "prefix"))
        assert isinstance(store, S3Store)
        assert isinstance(store, Store)
        assert store.bucket == "some-bucket"

    def test_a_capturing_local_store_is_still_a_local_store(self, backing) -> None:
        store = capturing(backing)
        assert isinstance(store, LocalStore)
        assert store.root == backing.root


class TestTheInvocationLedger:
    def test_begin_capture_is_re_entrant(self) -> None:
        """`crucible weekly --dry-run` re-enters `crucible.cli.main` once per
        stage in the same process. A nested `begin_capture` that started a
        fresh ledger would throw away every stage before it."""
        try:
            first = begin_capture()
            assert begin_capture() is first
            assert active_capture_ledger() is first
        finally:
            end_capture()
        assert active_capture_ledger() is None

    def test_every_store_opened_inside_one_capture_shares_the_ledger(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        try:
            ledger = begin_capture()
            one = open_store(str(tmp_path / "s"), dry_run=True)
            two = open_store(str(tmp_path / "s"), dry_run=True)
            one.put_bytes("a.json", b"{}")
            two.put_bytes("b.json", b"{}")
            assert ledger.keys == ("a.json", "b.json")
        finally:
            end_capture()

    def test_outside_a_capture_each_store_gets_its_own_ledger(self, backing) -> None:
        assert active_capture_ledger() is None
        one, two = capturing(backing), capturing(backing)
        one.put_bytes("a.json", b"{}")
        assert capture_ledger_of(one).keys == ("a.json",)
        assert capture_ledger_of(two).keys == ()

    def test_a_key_written_twice_appears_twice_in_writes_and_once_in_keys(self, backing) -> None:
        store = capturing(backing)
        store.put_bytes("board/current.json", b"{}")
        store.put_bytes("board/current.json", b'{"a": 1}')
        ledger = capture_ledger_of(store)
        assert len(ledger.writes) == 2
        assert ledger.keys == ("board/current.json",)

    def test_the_render_names_every_key_and_an_empty_one_says_so(self, backing) -> None:
        empty = CaptureLedger()
        assert "would write nothing" in empty.render()

        store = capturing(backing)
        store.put_bytes("board/current.json", b'{"schema_version": "board.v3"}')
        rendered = capture_ledger_of(store).render()
        assert "board/current.json" in rendered
        assert "board.v3" in rendered
        assert "30 bytes" in rendered


class TestOpenStoreResolvesACapturingStore:
    def test_dry_run_true_is_capturing_and_dry_run_false_is_not(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        assert is_capturing(open_store(str(tmp_path), dry_run=True))
        assert not is_capturing(open_store(str(tmp_path), dry_run=False))

    def test_settings_store_resolves_the_same_way(self, tmp_path) -> None:
        """Two entry points into the same decision that disagreed is what let
        a `--dry-run` reach a production bucket through one of them."""
        from crucible.config import Settings

        def settings(*, dry_run: bool) -> Settings:
            return Settings(
                store_uri=str(tmp_path),
                arctic_bucket="",
                strategy_dir=None,
                dry_run=dry_run,
            )

        assert is_capturing(settings(dry_run=True).store())
        assert not is_capturing(settings(dry_run=False).store())


class TestPresignedUrlOverACapturedKey:
    def test_a_captured_key_presigns_to_a_visibly_unopenable_url(self, backing) -> None:
        """Raising `KeyError` here would make the rehearsal fail where the
        real run succeeds; returning a `file://` or `https://` URL would put a
        dead link in front of whoever reads the rehearsal. The scheme says
        what it is."""
        store = capturing(backing)
        store.put_bytes("board/board.html", b"<html></html>")
        url = store.presigned_url("board/board.html", 3600)
        assert url == f"{CAPTURED_URL_SCHEME}board/board.html"

    def test_the_expiry_bound_is_still_enforced(self, backing) -> None:
        store = capturing(backing)
        store.put_bytes("board/board.html", b"<html></html>")
        with pytest.raises(ValueError, match="outside 1"):
            store.presigned_url("board/board.html", 99_999_999)


class TestTheTradingDayWalkStillHolds:
    def test_a_capturing_store_walks_its_captured_keys_too(self, backing) -> None:
        """`assert_keys_bind_to_trading_days` is inherited and lists through
        the overlay-aware `list_keys`, so a dry run that would write a
        non-session key is caught by the same walk the real store is."""
        from crucible.calendar import NonTradingDayKeyError

        store = capturing(backing)
        store.put_bytes("features/2026-09-19.json", b"{}")  # a Saturday
        with pytest.raises(NonTradingDayKeyError):
            store.assert_keys_bind_to_trading_days()


def test_a_dry_run_records_the_manifest_run_job_would_have_written(tmp_path) -> None:
    """`run_job(dry_run=True)` against a CAPTURING store takes the ordinary
    write path: the manifest is assembled, validated, and recorded.

    This is what makes the reported key set equal the set a real run writes —
    the manifest is one of those keys, and a dry run that skipped it would
    report a set no real run ever produces.
    """
    from crucible.keys import manifest_key
    from crucible.runner import run_job

    day = dt.date(2026, 9, 11)
    store = capturing(LocalStore(tmp_path / "store"))
    run_job(
        "report",
        lambda ctx: ctx.record_output("reports/2026-09-11.json", b'{"schema_version": "x.v1"}'),
        store=store,
        trading_day=day,
        run_mode="replay",
        dry_run=True,
    )

    assert set(capture_ledger_of(store).keys) == {
        "reports/2026-09-11.json",
        manifest_key("report", day.isoformat()),
    }
    assert sorted(LocalStore(tmp_path / "store").list_keys()) == []


def test_a_dry_run_against_a_plain_store_still_writes_no_manifest(tmp_path) -> None:
    """The unchanged path: a direct library/test `run_job(dry_run=True)`
    holding a plain backend. Refusing it would be wrong — its contract is
    that `run_job` makes exactly zero writes of its own, and that is still
    true. Every CLI job reaches the capturing path instead, because
    `--dry-run` resolves a capturing store."""
    from crucible.runner import run_job

    store = LocalStore(tmp_path / "store")
    run_job(
        "report",
        lambda ctx: None,
        store=store,
        trading_day=dt.date(2026, 9, 11),
        run_mode="replay",
        dry_run=True,
    )
    assert sorted(store.list_keys()) == []
