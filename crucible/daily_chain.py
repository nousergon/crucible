"""The daily chain: `data.daily`, then `serve.daily` m, u, s, one writer at a time.

`alpha-engine-config-I12020`. Measured 2026-10-07 and 2026-10-08:
nousergon-data's EOD collection failed and its heals did not converge until a
manual fix, so the ArcticDB append (D32) landed at 04:07Z and 03:18Z. The
`data-daily` schedule had already fired at 01:15Z with no retry and failed
(`MissingSourceError`, no rows for the day), the `serve-daily-*` schedules then
failed for missing features, and two paper sessions were lost with the day's
data sitting in the store.

The fix is that these jobs run when their input LANDS, not only at a clock
time. nous-ergon-ops `crucible-v2.yaml` now also dispatches
`data.daily --skip-if-ok --then-serve` on the collection state machine's own
SUCCEEDED event, and keeps the fixed-time schedules as backstops. That gives
one session several starters, which is safe only with the two rules this
module holds:

* **Never redo a day that is done** (:func:`already_ok`). With
  `--skip-if-ok`, a run whose day already carries an `ok` manifest at its
  canonical key writes one `ok` no-op manifest of its own — discriminated by
  the calendar day it ran on, so it can never overwrite the real one — and
  nothing else (rule 2: nothing to do is a complete, correct result). A
  `failed` or absent manifest is NOT done, so a late event repairs a day the
  backstop failed.
* **One writer at a time** (:func:`daily_chain_lock`). Every `data.daily` and
  `serve.daily` run holds one fleet-wide lease while it decides and writes.
  The decision is taken INSIDE the lease, so "not done yet" can never be read
  by two runs at once. `serve.daily --slot m` writes the money-path feed, and
  crucible#396's chain claim makes a race fail safe; this makes it not happen.

The chain itself (:func:`run_serve_chain`) re-enters `crucible.cli.main` once
per slot, in the order the slots read each other: M and U from `data.daily`'s
features, S from M's predictions and U's cut. Same in-process shape as the
weekly arc (`crucible.weekly.run_arc`).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from crucible.documents import UnreadableDocumentError, load_store_document
from crucible.keys import DAILY_CHAIN_LOCK_KEY
from crucible.store import ETAG_ABSENT, PointerConflictError, Store

__all__ = [
    "DAILY_CHAIN_LOCK_SCHEMA_VERSION",
    "DAILY_CHAIN_SERVE_ORDER",
    "LOCK_LEASE",
    "LOCK_WAIT",
    "DailyChainLockLost",
    "DailyChainLockTimeout",
    "already_ok",
    "daily_chain_lock",
    "run_serve_chain",
]

#: The lease document's declared shape, same convention as
#: `crucible.manifest.MONEY_PATH_CLAIM_SCHEMA_VERSION`.
DAILY_CHAIN_LOCK_SCHEMA_VERSION = "daily_chain_lock.v1"

#: How long a lease lives without a heartbeat. The holder renews it every
#: third of this (:data:`_RENEW_FRACTION`), so a live run never loses it, and a
#: box that dies holding it (a spot reclaim, a kernel panic) frees it within
#: this bound rather than blocking the next starter. Measured runs: `data.daily`
#: 2-3 min end to end, `serve.daily --slot m` 1m38s on its box, U and S seconds
#: (2026-10-05).
LOCK_LEASE = dt.timedelta(minutes=15)

#: How long a run waits for the lease before it fails loud. Longer than
#: :data:`LOCK_LEASE` so a dead holder's lease always expires inside one wait,
#: and long enough for a whole chain (one compile, three serves) ahead of it.
LOCK_WAIT = dt.timedelta(minutes=40)

#: Seconds between attempts while the lease is held by someone else.
LOCK_POLL_S = 20.0

_RENEW_FRACTION = 3

#: The order `--then-serve` runs the slots in. Not derived from
#: `crucible.slots.daily_servers()`'s dict order, because the order is a fact
#: about what each slot READS (S reads M's predictions and U's cut), not about
#: registration; :func:`run_serve_chain` asserts the two name the same slots,
#: so a fourth daily server cannot be silently left out of the chain.
DAILY_CHAIN_SERVE_ORDER: tuple[str, ...] = ("m", "u", "s")

_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class DailyChainLockTimeout(RuntimeError):
    """The lease stayed held by another run for the whole wait.

    Raised before `run_job`, so the run writes no manifest: the box exits
    non-zero and pages, and a dispatch with no manifest is graded by
    `crucible.alerts.evaluate_dispatch_absence`. Never a skip — a run that
    gave up waiting and exited 0 would read as a day that was served.
    """


class DailyChainLockLost(RuntimeError):
    """The lease was taken over while this run held it.

    Only possible if renewals stopped landing for a whole :data:`LOCK_LEASE`,
    which means another writer may have overlapped this one. Raised after the
    job body so its manifest stands, and the run still exits non-zero.
    """


def _utc(moment: dt.datetime) -> str:
    return moment.astimezone(dt.UTC).strftime(_FORMAT)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def default_holder(job: str) -> str:
    """A holder name an operator can find a box by: job, host, pid, nonce."""
    return f"{job}@{socket.gethostname()}:{os.getpid()}:{os.urandom(4).hex()}"


def _read_lease(store: Store) -> tuple[str, dict[str, Any] | None]:
    """``(token, document)`` at the lease key; ``(ETAG_ABSENT, None)`` if absent.

    A document that does not parse is returned as ``{}``: it has no expiry, so
    it is treated as EXPIRED and replaced by compare-and-swap on its own token.
    A corrupt lease must not wedge every daily run forever.
    """
    token = store.etag(DAILY_CHAIN_LOCK_KEY)
    if token == ETAG_ABSENT:
        return token, None
    try:
        return token, load_store_document(store, DAILY_CHAIN_LOCK_KEY)
    except (KeyError, UnreadableDocumentError):
        return token, {}


def _is_free(document: dict[str, Any] | None, now: dt.datetime) -> bool:
    if document is None or document.get("released_utc"):
        return True
    try:
        expires = dt.datetime.strptime(str(document["expires_utc"]), _FORMAT).replace(tzinfo=dt.UTC)
    except (KeyError, ValueError):
        return True
    return now >= expires


def _lease_body(
    *,
    holder: str,
    job: str,
    trading_day: str,
    acquired: dt.datetime,
    now: dt.datetime,
    lease: dt.timedelta,
) -> bytes:
    return json.dumps(
        {
            "schema_version": DAILY_CHAIN_LOCK_SCHEMA_VERSION,
            "holder": holder,
            "job": job,
            "trading_day": trading_day,
            "acquired_utc": _utc(acquired),
            "expires_utc": _utc(now + lease),
            "released_utc": None,
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


@contextlib.contextmanager
def daily_chain_lock(
    store: Store,
    *,
    job: str,
    trading_day: str,
    holder: str | None = None,
    lease: dt.timedelta = LOCK_LEASE,
    wait: dt.timedelta = LOCK_WAIT,
    poll_s: float = LOCK_POLL_S,
    clock: Callable[[], dt.datetime] = _now,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Hold the fleet-wide daily-chain lease for the body of the ``with``.

    Acquire is a compare-and-swap on the lease key's version: create it when
    absent, replace it when it is released or expired. Two runs racing for a
    free lease both read the same version and exactly one write lands (S3's
    `IfNoneMatch`/`IfMatch`); the loser re-reads and waits.

    A daemon thread renews the lease every ``lease / 3`` for as long as the
    body runs. Release writes `released_utc` by compare-and-swap on the
    holder's own last version, so a run can never release a lease somebody
    else now holds.

    Yields the holder name.
    """
    holder = holder or default_holder(job)
    deadline = clock() + wait
    acquired = clock()
    while True:
        token, document = _read_lease(store)
        now = clock()
        if _is_free(document, now):
            acquired = now
            body = _lease_body(
                holder=holder,
                job=job,
                trading_day=trading_day,
                acquired=acquired,
                now=now,
                lease=lease,
            )
            try:
                token = store.compare_and_swap(DAILY_CHAIN_LOCK_KEY, token, body)
            except PointerConflictError:
                continue
            break
        if now >= deadline:
            raise DailyChainLockTimeout(
                f"{job} for {trading_day} waited {wait} for the daily-chain lease "
                f"({DAILY_CHAIN_LOCK_KEY}) and it is still held by "
                f"{(document or {}).get('holder')!r} for {(document or {}).get('job')!r} "
                f"until {(document or {}).get('expires_utc')!r}. No manifest was written; "
                "re-dispatch once the holder has finished."
            )
        sleep(poll_s)

    state = {"token": token, "lost": None}
    guard = threading.Lock()
    stop = threading.Event()

    def _renew() -> None:
        while not stop.wait(lease.total_seconds() / _RENEW_FRACTION):
            with guard:
                if state["lost"]:
                    return
                body = _lease_body(
                    holder=holder,
                    job=job,
                    trading_day=trading_day,
                    acquired=acquired,
                    now=clock(),
                    lease=lease,
                )
                try:
                    state["token"] = store.compare_and_swap(
                        DAILY_CHAIN_LOCK_KEY, state["token"], body
                    )
                except PointerConflictError as exc:
                    state["lost"] = str(exc)
                    print(f"daily-chain lease LOST by {holder}: {exc}", file=sys.stderr)
                    return
                except Exception as exc:  # noqa: BLE001 - reported, retried next beat
                    print(
                        f"daily-chain lease renewal by {holder} failed, retrying: {exc}",
                        file=sys.stderr,
                    )

    renewer = threading.Thread(target=_renew, name="daily-chain-lease", daemon=True)
    renewer.start()
    failed = False
    try:
        yield holder
    except BaseException:
        failed = True
        raise
    finally:
        stop.set()
        renewer.join()
        with guard:
            lost = state["lost"]
            if not lost:
                released = json.loads(
                    _lease_body(
                        holder=holder,
                        job=job,
                        trading_day=trading_day,
                        acquired=acquired,
                        now=clock(),
                        lease=lease,
                    )
                )
                released["released_utc"] = _utc(clock())
                try:
                    store.compare_and_swap(
                        DAILY_CHAIN_LOCK_KEY,
                        state["token"],
                        json.dumps(released, indent=2, sort_keys=True).encode("utf-8"),
                    )
                except PointerConflictError as exc:
                    lost = str(exc)
        if lost and not failed:
            raise DailyChainLockLost(
                f"{job} for {trading_day} lost the daily-chain lease ({DAILY_CHAIN_LOCK_KEY}) "
                f"while it held it, so another writer may have overlapped it: {lost}"
            )


def already_ok(
    store: Store, job: str, trading_day: str, *, discriminator: str | None = None
) -> bool:
    """Whether ``job``'s canonical manifest for ``trading_day`` is a valid `ok`.

    Absent, `failed`, unreadable or non-conforming all read as NOT done, so
    the caller runs the job. That is the safe direction: a rerun overwrites
    with a fresh verdict, while a skip over a manifest nobody could validate
    would let a broken day stand as served.
    """
    from crucible.manifest import read_manifest  # noqa: PLC0415 - manifest imports keys only

    try:
        document = read_manifest(store, job, trading_day, discriminator=discriminator)
    except Exception:  # noqa: BLE001 - every unreadable shape means "not done"; see docstring
        return False
    return document.get("status") == "ok"


def run_serve_chain(
    trading_day: dt.date,
    *,
    store: str | None,
    run_mode: str | None,
    dry_run: bool = False,
    main: Callable[[list[str]], int] | None = None,
    servers: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Run `serve.daily --skip-if-ok` for every slot, in :data:`DAILY_CHAIN_SERVE_ORDER`.

    Every slot runs even if an earlier one failed: U does not read M, and each
    stage files its own manifest either way (a failed S is graded on its own
    manifest, and the 22:30 ET backstop re-runs it with `--skip-if-ok`).
    Returns ``{slot: exit code}``; a stage that RAISED is recorded as 1 after
    its traceback is printed, never swallowed — the caller turns any non-zero
    into its own non-zero exit.

    ``--date`` is passed explicitly so a chain that straddles a session close
    cannot serve one day and compile another.
    """
    if main is None:
        from crucible.cli import main as cli_main  # noqa: PLC0415 - cli imports this module

        main = cli_main
    if servers is None:
        from crucible.slots import daily_servers  # noqa: PLC0415

        servers = daily_servers()
    if set(servers) != set(DAILY_CHAIN_SERVE_ORDER):
        raise RuntimeError(
            f"the daily chain serves {list(DAILY_CHAIN_SERVE_ORDER)} but "
            f"`crucible.slots.daily_servers()` reads {sorted(servers)}; declare the new "
            "slot's place in DAILY_CHAIN_SERVE_ORDER rather than leave it unserved."
        )
    codes: dict[str, int] = {}
    for slot in DAILY_CHAIN_SERVE_ORDER:
        argv = ["serve.daily", "--slot", slot, "--date", trading_day.isoformat(), "--skip-if-ok"]
        if run_mode:
            argv += ["--run-mode", run_mode]
        if store:
            argv += ["--store", store]
        if dry_run:
            argv.append("--dry-run")
        try:
            codes[slot] = int(main(argv))
        except SystemExit as exc:
            codes[slot] = 0 if exc.code is None else exc.code if isinstance(exc.code, int) else 1
        except Exception:  # noqa: BLE001 - printed here, surfaced as the caller's exit
            import traceback  # noqa: PLC0415

            traceback.print_exc()
            codes[slot] = 1
    return codes
