"""`python -m crucible.deploy` — the three steps `deploy.yml` drives.

Normative source: plan §4.11.

    publish   upload the wheel and release.json under releases/{sha}/
    capture   write the pointer's version token to a file, BEFORE the smoke
    flip      read the smoke MANIFEST and, only on `ok`, repoint current
    record    write the deploy's own run manifest, on BOTH paths

**Why `capture` is a step of its own.** The flip is a compare-and-swap, and a
compare-and-swap is only worth the name over the interval it actually covers.
Reading the version token immediately before the swap makes that interval
microseconds wide — it catches a second deploy that finished inside it and
nothing else, which is the one race `deploy.yml`'s `concurrency` group has
already serialised away. The interval that matters is **the whole smoke**:
that is the minutes-long window in which an operator rollback
(`crucible release.pin`, which swaps against whatever is there right now)
can land, and against a token read after the smoke that rollback is silently
undone by the flip with no error anywhere. The token is therefore captured
before the smoke starts and carried to the flip on disk, so the swap compares
against what the pointer held when this deploy began verifying.

**Why this is Python in the repository and not shell in the workflow.**
Everything here is a decision — is the smoke ok, did another deploy win the
pointer, what does the manifest say — and a decision expressed in workflow
YAML is a decision with no tests. The workflow is left with what only a
workflow can do: build, authenticate, and sequence. It is also what makes the
deploy step *code committed in the repository*, which is `pull-request-policy`
§4.2 form 1 rather than a command someone must remember to run after merging.

It is a separate module from `crucible.cli` because a deploy is not a job: it
runs in CI, under the deploy identity, and it is the only thing in the system
that moves the release pointer. Putting it behind `crucible <job>` would give
every runtime identity a subcommand it must never be able to execute.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from crucible.calendar import resolve_trading_day
from crucible.manifest import RUN_MANIFEST_SCHEMA_VERSION, manifest_key
from crucible.release import (
    POINTER_KEY,
    ReleaseProvenance,
    ReleaseRecord,
    assert_immutable_write,
    current_release,
    flip_on_smoke,
    provenance_key,
    read_pointer,
    release_json_key,
    release_object_lock_params,
    wheel_key,
    write_deploy_manifest,
)
from crucible.store import PointerConflictError, Store, open_store, sha256_hex

__all__ = ["main"]

_UNKNOWN_SHA = "0" * 40


def _publish(args: argparse.Namespace, store: Store) -> int:
    """Upload the immutable half. Promotes nothing.

    `release.json` is the DETERMINISTIC identity half — built in the
    workflow from the lockfile and wheel, and validated here against
    :class:`ReleaseRecord` — so a malformed record is refused before it is
    durable rather than discovered by whatever reads it next.
    `provenance.json` is this attempt's :class:`ReleaseProvenance` (the
    fields that move on every rebuild: `built_at`, `workflow_run_url`,
    `test_summary`) and is written unconditionally, never immutable-checked
    (alpha-engine-config-I9786): a re-run for an unchanged commit produces
    identity bytes the store already holds and a NEW provenance record,
    never a `ReleaseImmutabilityError`.

    Two things are checked that validating the dataclass does not:

    * **the record describes the wheel it is shipped with.** `wheel_sha256`
      is the only statement anyone downstream has about what these bytes
      are, and until it is checked against the bytes it is a claim the
      publisher made about its own artifact. Re-hashed here, and again by
      the smoke against what actually landed in the store (§4.11), so a
      corruption in the upload is caught by the gate rather than by whatever
      installs the wheel next week.
    * **the prefix is not already occupied by different IDENTITY bytes.**
      See :class:`crucible.release.ReleaseImmutabilityError`. This is the
      whole fix for I9786: `release.json` no longer carries the three fields
      that moved on every run, so two builds of the same commit really are
      byte-identical and a re-run is a clean no-op instead of a guaranteed
      failure.

    On S3, each identity key actually written is locked under S3 Object Lock
    GOVERNANCE mode (`crucible.release.RELEASE_OBJECT_LOCK_RETENTION`) on the
    SAME `put_bytes` call that writes it (alpha-engine-config-I9787) — see
    `crucible.release.release_object_lock_params`. `assert_immutable_write`
    defends at this writer only; the lock defends against a writer that skips
    it (a hand-rolled `aws s3 cp`, a second `workflow_dispatch` running code
    that predates this fix).
    """
    wheel = Path(args.wheel).read_bytes()
    record = ReleaseRecord(**json.loads(Path(args.release_json).read_text(encoding="utf-8")))
    provenance = ReleaseProvenance(
        **json.loads(Path(args.provenance_json).read_text(encoding="utf-8"))
    )
    if record.sha != args.sha:
        raise SystemExit(
            f"release.json is for {record.sha}, not {args.sha}. Publishing it under the "
            "wrong prefix would make the rollback target a build it does not describe."
        )
    if provenance.sha != args.sha:
        raise SystemExit(
            f"provenance.json is for {provenance.sha}, not {args.sha}. Recording it "
            "under the wrong prefix would attribute this attempt to a build it did not "
            "produce."
        )
    digest = sha256_hex(wheel)
    if digest != record.wheel_sha256:
        raise SystemExit(
            f"release.json for {args.sha} claims wheel_sha256={record.wheel_sha256}, but "
            f"{args.wheel} hashes to {digest}. The record's integrity claim is the only "
            "thing anyone downstream has about these bytes; publishing a record that "
            "does not describe its own artifact makes every later verification vacuous."
        )
    # Both identity keys are checked before either is written: a refusal
    # must not be able to leave a wheel from one build beside a release.json
    # from another.
    writes = [
        (key, payload)
        for key, payload in (
            (wheel_key(args.sha), wheel),
            (release_json_key(args.sha), record.to_json()),
        )
        if assert_immutable_write(store, key, payload)
    ]
    lock_mode, retain_until = release_object_lock_params(store)
    for key, payload in writes:
        store.put_bytes(
            key, payload, object_lock_mode=lock_mode, object_lock_retain_until=retain_until
        )
    # Unconditional and unlocked: keyed per attempt, so it never contends
    # with itself, and it is the durable trace that THIS attempt happened
    # even when the identity keys needed no write at all — which is exactly
    # the re-run-of-an-unchanged-commit case I9786 asks to be a no-op.
    store.put_bytes(
        provenance_key(args.sha, provenance.run_id, provenance.run_attempt), provenance.to_json()
    )
    if not writes:
        print(
            f"releases/{args.sha}/ already holds exactly this identity; nothing to "
            f"publish. Recorded this attempt's provenance at "
            f"{provenance_key(args.sha, provenance.run_id, provenance.run_attempt)}."
        )
        return 0
    print(f"published releases/{args.sha}/ ({len(wheel)} bytes)")
    return 0


def _capture(args: argparse.Namespace, store: Store) -> int:
    """Write the pointer's current version token to ``--out``, before the smoke.

    The opening half of the compare-and-swap, split from the closing half by
    a whole workflow step, because that is the interval the protection is
    supposed to cover (see the module docstring). The token goes to a file
    rather than a step output because :data:`crucible.store.ETAG_ABSENT`
    contains a NUL byte, and a version token that has to survive shell
    quoting is a version token that will one day arrive mangled and compare
    unequal to everything — which fails a deploy that should have passed.

    The sha is printed with it purely so an operator reading the log sees
    which deploy owns this token.
    """
    before, version = read_pointer(store, POINTER_KEY)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(version.encode("utf-8"))
    print(
        f"captured the releases/current version token before the smoke for {args.sha}; "
        f"the pointer is at {before or '(unset)'}"
    )
    return 0


def _flip(args: argparse.Namespace, store: Store) -> int:
    """Read the smoke manifest and flip only on `ok`.

    Exits non-zero when the smoke did not pass, so the deploy is FAILED and
    visible rather than green with an unmoved pointer — "nothing was promoted"
    and "nothing needed promoting" must not look the same on the surface.

    ``--expect-pointer-file`` is **required**, and holds the token `capture`
    read before the smoke began. It is not optional and does not fall back to
    reading the pointer here: a flip that swapped against a token it read
    itself would be a compare-and-swap over a window microseconds wide, which
    is the defect this argument exists to close. A missing file fails the
    deploy rather than degrading to the unprotected swap.
    """
    token_path = Path(args.expect_pointer_file)
    if not token_path.is_file():
        raise SystemExit(
            f"no pointer token at {token_path}. It is written by "
            "`python -m crucible.deploy capture` before the smoke runs, and the flip "
            "compares against it so an operator rollback landing mid-smoke fails this "
            "deploy instead of being silently undone. Without it there is nothing to "
            "compare against and the deploy is refused."
        )
    expect = token_path.read_bytes().decode("utf-8")
    trading_day = resolve_trading_day()
    key = manifest_key("smoke", trading_day.isoformat())
    if not store.exists(key):
        raise SystemExit(
            f"no smoke manifest at {key}. The gate is the smoke RUN; promoting without "
            "one would flip the pointer on a step that may never have executed."
        )
    smoke = json.loads(store.get_bytes(key).decode("utf-8"))
    before = current_release(store)
    try:
        flipped = flip_on_smoke(store, sha=args.sha, smoke_manifest=smoke, expect=expect)
    except PointerConflictError as exc:
        raise SystemExit(
            f"releases/current moved while the smoke for {args.sha} was running: it is "
            f"now {before or '(unset)'}, and this deploy read a different token before "
            "it started. Someone rolled the pointer back or another actor promoted a "
            "build; this deploy FAILS rather than overwriting them. Re-run the deploy "
            "if the current pointer is wrong, or leave it if the rollback was intended."
        ) from exc
    if not flipped:
        raise SystemExit(
            f"smoke for {args.sha} ended {smoke.get('status')!r}: "
            f"{smoke.get('reason')!r}. releases/current is untouched at "
            f"{before or '(unset)'}, and this deploy is FAILED."
        )
    print(f"releases/current: {before or '(unset)'} -> {args.sha}")
    return 0


def _record(args: argparse.Namespace, store: Store) -> int:
    """Write the deploy's own manifest, in the run-manifest schema.

    Runs with `if: always()`, so it must be correct on the failure path — that
    is the path where a deploy manifest is worth having. `status` is derived
    from the job's outcome and from whether the pointer actually moved, never
    declared: a deploy that reported itself ok while the pointer had not moved
    would be the degraded-SUCCEEDED this whole system refuses.
    """
    now = dt.datetime.now(dt.UTC)
    trading_day = resolve_trading_day(now)
    promoted = current_release(store)
    ok = args.outcome == "success" and promoted == args.sha
    reason = ""
    if not ok:
        reason = (
            f"deploy outcome={args.outcome!r}; releases/current is "
            f"{promoted or '(unset)'}, expected {args.sha}. See {args.run_url}"
        )
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": _run_id_from(args.sha, now),
        "job": "deploy",
        "trading_day": trading_day.isoformat(),
        "calendar_date": now.date().isoformat(),
        "status": "ok" if ok else "failed",
        "reason": reason,
        "started": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": args.sha if len(args.sha) == 40 else _UNKNOWN_SHA,
        "release_sha": args.sha if len(args.sha) == 40 else _UNKNOWN_SHA,
        "seed": int(trading_day.strftime("%Y%m%d")),
        "inputs": [],
        "outputs": [],
        "rows_in": 0,
        "rows_out": 0,
        "rows_rejected": [],
        "cost_usd": 0.0,
        "llm_calls": [],
        "resource": {
            # A GitHub-hosted runner. Declared rather than omitted: §9.2 class
            # 3 is required on every manifest, and a deploy running on metered
            # compute is exactly the thing a cost row wants to see.
            "instance_type": "github-hosted-ubuntu-latest",
            "spot": False,
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 0.0,
            "disk_free_mb": 0.0,
        },
        "metrics": [
            {
                "name": "release_pointer_matches_deployed_sha",
                "module": "crucible.deploy",
                "metric_type": "operational",
                "value": 1.0 if ok else 0.0,
                "unit": "boolean",
                "n_floor": 0,
                "status": "OK" if ok else "BREACH",
                "status_reason": reason or f"releases/current is {args.sha}.",
                "source_path": POINTER_KEY,
                "last_updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ],
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    key = write_deploy_manifest(store, manifest)
    print(f"wrote {key} ({manifest['status']})")
    # Exits 0 even for a failed deploy: this step RECORDS the outcome, and a
    # recorder that failed the job a second time would mask which step
    # actually broke.
    return 0


def _run_id_from(sha: str, now: dt.datetime) -> str:
    """A ULID-shaped id derived from the deploy's sha and instant.

    Derived rather than random so a re-record of the same deploy in the same
    second is idempotent, and so the id is reproducible by anyone holding the
    same two facts.
    """
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    value = (int(now.timestamp() * 1000) << 80) | (int(sha[:20] or "0", 16) & ((1 << 80) - 1))
    out = []
    for _ in range(26):
        out.append(alphabet[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible.deploy",
        description="The four steps deploy.yml drives: publish, capture, flip, record.",
    )
    sub = parser.add_subparsers(dest="step", required=True)

    for name, help_text in (
        ("publish", "Upload the wheel and release.json. Promotes nothing."),
        ("capture", "Write the releases/current version token, BEFORE the smoke."),
        ("flip", "Read the smoke manifest and repoint releases/current on `ok`."),
        ("record", "Write the deploy's own run manifest, on both paths."),
    ):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--sha", required=True)
        p.add_argument("--store", required=True)
        if name == "publish":
            p.add_argument("--wheel", required=True)
            p.add_argument("--release-json", required=True)
            p.add_argument("--provenance-json", required=True)
        if name == "capture":
            p.add_argument("--out", required=True)
        if name == "flip":
            # Required, never defaulted: see `_flip`. A default would let the
            # flip fall back to the unprotected read-then-swap the moment the
            # workflow lost its capture step, and nothing would say so.
            p.add_argument("--expect-pointer-file", required=True)
        if name == "record":
            p.add_argument("--outcome", required=True)
            p.add_argument("--run-url", default="")

    args = parser.parse_args(argv)
    store = open_store(args.store)
    return {"publish": _publish, "capture": _capture, "flip": _flip, "record": _record}[args.step](
        args, store
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
