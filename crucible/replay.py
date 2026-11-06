"""`replay.full_universe`: one trading day's `data.daily`, at FULL universe, in a scratch store.

alpha-engine-config-I12519 (S-31, delivery-policy.md ENV-SCALE-REPLAY). Staging
runs a SUBSET universe, so three green staging days prove nothing about a
change whose cost or correctness grows with the universe. nous-ergon-ops'
promote soak gate (`scripts/promote_soak_gate.py`, clause
`full_universe_replay_if_scale_sensitive`) therefore refuses a scale-sensitive
release unless staging's run store holds a green
``runs/replay.full_universe/{trading_day}/{digest}/run.json`` whose
``release_manifest`` is that digest. This job is the producer of that record.

The discriminator is the release digest, the run's natural axis: a replay is
evidence about one release, so two releases replayed on one day keep two
records, a re-run of the same release replaces its own, and the gate reads the
one key it can name from the candidate alone. A run with no
``$CRUCIBLE_RELEASE_MANIFEST`` is evidence about nothing and is refused before
it compiles.

Two stores, never one:

* the **run store** (``--store`` / ``$CRUCIBLE_STORE``) receives exactly one
  object, this job's own manifest, at the key the gate reads. The runner binds
  ``release_manifest`` from ``$CRUCIBLE_RELEASE_MANIFEST`` as for every job.
* the **replay store** (``--replay-store`` / ``$CRUCIBLE_REPLAY_STORE``) is where
  the full-universe ``data.daily`` writes everything else: its data, its
  features, its own manifest and its daily-chain lease. A full-universe compile
  into the run store would overwrite that environment's own subset record and
  artifacts for the day, so the replay store is REQUIRED and must not overlap
  the run store (equal to it, inside it, or containing it): refused by name.

The universe is not chosen here. ``data.daily`` reads ``$CRUCIBLE_UNIVERSE_URI``
as always; the crucible-v2 dispatcher's per-job override (nous-ergon-ops
``crucible-v2.yaml`` ``FullUniverseReplay``) points it at the full universe for this
job only, and sets ``$CRUCIBLE_REPLAY_STORE``.

``ok`` means the full-universe compile for the day exited 0 and left an ``ok``
manifest in the replay store. Anything else raises, so the run store gets a
``failed`` record naming why, and the gate reads it as red rather than missing.

Scope, stated so it is not over-read: this exercises ``data.daily`` (point-in-time
data access and feature computation). It does not run the weekly cut or a
backfill at full universe.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from crucible.keys import manifest_key
from crucible.runner import RunContext, resolve_release_manifest, run_job

__all__ = [
    "INNER_JOB",
    "REPLAY_JOB",
    "REPLAY_STORE_ENV",
    "ReplayFailedError",
    "ReplayStoreError",
    "ReplayUnboundError",
    "handle_replay_full_universe",
    "resolve_replay_store",
]

REPLAY_JOB = "replay.full_universe"
INNER_JOB = "data.daily"
REPLAY_STORE_ENV = "CRUCIBLE_REPLAY_STORE"


class ReplayStoreError(ValueError):
    """No replay store, or one that overlaps the run store."""


class ReplayUnboundError(ValueError):
    """No ``$CRUCIBLE_RELEASE_MANIFEST``: the replay would be bound to no release."""


class ReplayFailedError(RuntimeError):
    """The full-universe compile did not produce an ``ok`` manifest."""


def _norm(uri: str) -> str:
    if uri.startswith("s3://"):
        return uri.rstrip("/")
    return str(Path(uri).expanduser().resolve())


def resolve_replay_store(run_store_uri: str | None, flag: str | None) -> str:
    """The replay store URI, refusing an absent one or one that overlaps ``run_store_uri``."""
    uri = flag or os.environ.get(REPLAY_STORE_ENV, "")
    if not uri:
        raise ReplayStoreError(
            f"{REPLAY_JOB} needs a replay store (--replay-store or ${REPLAY_STORE_ENV}). "
            "Refused rather than compile the full universe into the run store, whose own "
            "daily record and artifacts for the day it would overwrite."
        )
    if not run_store_uri:
        raise ReplayStoreError(
            f"{REPLAY_JOB} cannot check the replay store against an unset run store"
        )
    a, b = _norm(run_store_uri), _norm(uri)
    if a == b or b.startswith(a + "/") or a.startswith(b + "/"):
        raise ReplayStoreError(
            f"replay store {uri!r} overlaps the run store {run_store_uri!r}; a full-universe "
            "compile there would overwrite that environment's own daily records. Use a "
            "separate prefix (the dispatcher's is `crucible-replay`)."
        )
    return uri


def handle_replay_full_universe(
    args: argparse.Namespace,
    *,
    main: Callable[[Sequence[str]], int] | None = None,
) -> int:
    """`crucible replay.full_universe [--date D] [--replay-store URI]`."""
    from crucible.config import settings as resolve_settings  # noqa: PLC0415
    from crucible.config import store_from_uri  # noqa: PLC0415
    from crucible.documents import load_document_bytes  # noqa: PLC0415

    if main is None:
        from crucible.cli import main as cli_main  # noqa: PLC0415 - cli imports this module

        main = cli_main
    dry_run = bool(getattr(args, "dry_run", False))
    # Same resolution as every track-A handler (`crucible.track_a._settings`).
    config = resolve_settings(store_uri=getattr(args, "store", None), dry_run=dry_run)
    store = config.store()
    replay_uri = resolve_replay_store(config.store_uri, getattr(args, "replay_store", None))
    digest = resolve_release_manifest()
    if digest is None:
        raise ReplayUnboundError(
            f"{REPLAY_JOB} needs $CRUCIBLE_RELEASE_MANIFEST (the release this replay is "
            "evidence about); an unbound replay could satisfy no promotion and is refused."
        )
    day = args.trading_day.isoformat()
    run_mode = getattr(args, "run_mode", None)
    argv = [INNER_JOB, "--date", day, "--store", replay_uri]
    if run_mode:
        argv += ["--run-mode", run_mode]
    if dry_run:
        argv.append("--dry-run")

    # The compile runs BEFORE this job's own `run_job`, not inside it, so two
    # manifests are never open in one process at once. A compile that RAISES
    # (cli.main re-raises a job's exception) is caught and re-raised inside
    # the body below, so it files a `failed` record at this digest's key
    # rather than leaving an earlier `ok` record for the day standing (Codex
    # finding 85ceac0e). Only a process that dies outright leaves no record.
    compile_error: Exception | None = None
    code: int | None = None
    try:
        code = main(argv)
    except Exception as exc:  # noqa: BLE001 - recorded as this job's failure below
        compile_error = exc

    def body(ctx: RunContext) -> None:
        if compile_error is not None:
            raise ReplayFailedError(
                f"full-universe {INNER_JOB} for {day} raised in {replay_uri}: "
                f"{type(compile_error).__name__}: {compile_error}"
            ) from compile_error
        if code != 0:
            raise ReplayFailedError(
                f"full-universe {INNER_JOB} for {day} exited {code} in {replay_uri}; "
                "its own manifest there says why"
            )
        if dry_run:
            return
        key = manifest_key(INNER_JOB, day)
        replay_store = store_from_uri(replay_uri)
        # ONE read: the bytes checked are the bytes hashed into the lineage
        # row, never a second read a concurrent same-day replay into this
        # store could have swapped in between (Codex finding e1fd6859). The
        # record must also be bound to THIS release: a same-day compile for
        # another release that overwrote it is that release's evidence.
        raw = replay_store.get_bytes(key)
        inner = load_document_bytes(key, raw)
        if inner.get("status") != "ok":
            raise ReplayFailedError(
                f"{replay_uri}/{key} reads status {inner.get('status')!r} "
                f"({inner.get('reason')!r}), not ok"
            )
        if inner.get("release_manifest") != digest:
            raise ReplayFailedError(
                f"{replay_uri}/{key} is bound to release {inner.get('release_manifest')!r}, "
                f"not this replay's {digest!r}: another compile replaced it"
            )
        # The inner manifest is the evidence; its key is relative to the replay
        # store printed below, never to this record's own store.
        ctx.record_input(key, raw)
        ctx.record_rows(
            rows_in=int(inner.get("rows_in") or 0), rows_out=int(inner.get("rows_out") or 0)
        )

    ctx = run_job(
        REPLAY_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        run_mode=run_mode,
        dry_run=dry_run,
        discriminator=digest,
    )
    print(
        json.dumps({"run_id": ctx.run_id, "replay_store": replay_uri, "inner_argv": argv}, indent=2)
    )
    return 0
