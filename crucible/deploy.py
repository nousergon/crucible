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
from crucible.documents import load_store_document, read_store_document
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
    write_deploy_manifest,
)
from crucible.runmode import RUN_MODES, resolve_run_mode
from crucible.store import PointerConflictError, Store, open_store, sha256_hex

__all__ = ["main"]

_UNKNOWN_SHA = "0" * 40


def _required_smoke_extras() -> frozenset[str]:
    """The extra names a released wheel must be installed with before the
    flip may trust its smoke.

    Read from the installed distribution's own metadata (`Provides-Extra`,
    which the build writes from `pyproject.toml`'s
    `[project.optional-dependencies]`) rather than hardcoded or read off a
    `pyproject.toml` path relative to this file: `arcticdb` is declared
    there because ArcticDB has no Linux aarch64 wheel and the v2 box
    (r6i.large, x86_64) installs it with `pip install "...[arcticdb]"` — a
    smoke that never proved the same install on the same architecture
    proves nothing about a wheel the box can actually run
    (alpha-engine-config-I10069). Distribution metadata is the one source
    that is present both in the checkout `deploy.yml` runs from and in an
    installed wheel; a path walk up from this module is only true of the
    first.
    """
    from importlib.metadata import metadata  # noqa: PLC0415 - lazy, one call site

    return frozenset(metadata("crucible").get_all("Provides-Extra") or ())


def _smoked_extras(smoke_manifest: dict[str, Any]) -> frozenset[str]:
    """The extras `deploy.yml`'s install-proof step actually verified,
    as recorded on the smoke manifest's `smoke_ok` metric.

    `crucible.track_c.smoke_handler` writes this field from
    `$CRUCIBLE_SMOKED_EXTRAS`, which the install-proof step exports after
    installing the published wheel with every required extra and importing
    each one's module — on the same x86_64 runner the box uses. A manifest
    from before this field existed, or one whose metric is missing
    altogether, reads as having smoked nothing: absence must not read as
    coverage.
    """
    for metric in smoke_manifest.get("metrics", []):
        if metric.get("name") == "smoke_ok":
            extras = metric.get("smoked_extras")
            if isinstance(extras, list):
                return frozenset(extras)
    return frozenset()


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
    # `ReleaseRecord.__post_init__` / `ReleaseProvenance.__post_init__`
    # validate against their own schema on construction (alpha-engine-config
    # I9814) — the two calls below are the ONLY code path in this module
    # that turns the files on disk into these dataclasses, so a
    # schema-refused `release.json`/`provenance.json` fails HERE, before a
    # single byte reaches the store, rather than shipping through the CLI
    # the deploy workflow actually runs. `TypeError` covers a document
    # missing a required field or carrying one the dataclass does not
    # declare; `ValueError` covers the schema violations `__post_init__`
    # raises for a field present but non-conformant (bad pattern, empty
    # string, unknown `schema_version`). Both are re-raised as `SystemExit`
    # so the failure reads like every other refusal in this function rather
    # than as an uncaught traceback.
    try:
        record = ReleaseRecord(**json.loads(Path(args.release_json).read_text(encoding="utf-8")))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{args.release_json} does not conform to release.v3.json: {exc}") from exc
    try:
        provenance = ReleaseProvenance(
            **json.loads(Path(args.provenance_json).read_text(encoding="utf-8"))
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"{args.provenance_json} does not conform to release_provenance.v1.json: {exc}"
        ) from exc
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
    # alpha-engine-config-I9908: the record's OWN declared filename is what
    # gets published to, not a filename this function re-derives from the
    # sha — a `--wheel` file named anything else would publish under a key
    # `wheel_filename` never described, which is exactly the "wheel this
    # pipeline built is not the wheel a consumer can find" defect the issue
    # is closing.
    if Path(args.wheel).name != record.wheel_filename:
        raise SystemExit(
            f"--wheel is {Path(args.wheel).name}, but release.json for {args.sha} "
            f"declares wheel_filename={record.wheel_filename!r}. Publishing the file "
            "under a name release.json does not describe would make it undiscoverable "
            "to any reader that trusts the record — which is every reader."
        )
    # Both identity keys are checked before either is written: a refusal
    # must not be able to leave a wheel from one build beside a release.json
    # from another.
    writes = [
        (key, payload)
        for key, payload in (
            # `record.sha == args.sha` was asserted above, so the record's own
            # key IS this sha's key (alpha-engine-config-I9932: one pairing of
            # sha and wheel_filename, stated on the record).
            (record.wheel_key, wheel),
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
    # STRICT face of the one reader (`crucible.documents`): a smoke manifest
    # that is not an object stops the flip with the key named.
    smoke = load_store_document(store, key)
    before = current_release(store)
    # alpha-engine-config-I10069: same shape as the `release_sha` check
    # `flip_on_smoke` makes below — a smoke manifest that never proved the
    # box's required extras install and import is a smoke that passed
    # against a wheel the box cannot actually run.
    required_extras = _required_smoke_extras()
    missing_extras = sorted(required_extras - _smoked_extras(smoke))
    if missing_extras:
        raise SystemExit(
            f"the smoke manifest for {args.sha} does not record installing extra(s) "
            f"{missing_extras}. pyproject.toml declares them required and "
            "nous-ergon-ops/infrastructure/cloudformation/crucible-v2.yaml installs "
            "them on the box; a smoke that never proved the wheel installs and "
            "imports what the box needs would flip the pointer on a build the box "
            f"cannot run. releases/current is untouched at {before or '(unset)'}."
        )
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

    Reads BOTH the pointer and the smoke manifest through the GUARDED face
    (`read_store_document`), never `current_release`/`load_store_document`'s
    STRICT one (alpha-engine-config-I9945): this step is the one job
    designed to always record, under `if: always()`, so a corrupt
    `releases/current` OR a corrupt smoke manifest must become a
    `status: failed` manifest naming the fault — never a raise that leaves
    the deploy that observed the corruption with no manifest at all. A
    document that is simply ABSENT (no smoke ever ran because an earlier
    step failed first) is not a fault here — only a present-but-unreadable
    one is: that is the one case a raise would have propagated instead of
    being recorded.
    """
    now = dt.datetime.now(dt.UTC)
    trading_day = resolve_trading_day(now)
    pointer_read = read_store_document(store, POINTER_KEY)
    pointer_fault = pointer_read.require("sha", str)
    promoted = (
        pointer_read.document["sha"]
        if pointer_fault is None and pointer_read.document is not None
        else None
    )
    smoke_key = manifest_key("smoke", trading_day.isoformat())
    smoke_read = read_store_document(store, smoke_key)
    smoke_fault = (
        smoke_read.problem if smoke_read.document is None and not smoke_read.absent else None
    )
    ok = (
        args.outcome == "success"
        and pointer_fault is None
        and smoke_fault is None
        and promoted == args.sha
    )
    reason = ""
    if not ok:
        if pointer_fault is not None:
            reason = f"releases/current is unreadable: {pointer_fault}. See {args.run_url}"
        elif smoke_fault is not None:
            reason = f"{smoke_key} is unreadable: {smoke_fault}. See {args.run_url}"
        else:
            reason = (
                f"deploy outcome={args.outcome!r}; releases/current is "
                f"{promoted or '(unset)'}, expected {args.sha}. See {args.run_url}"
            )
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": _run_id_from(args.sha, now),
        "job": "deploy",
        # From the invocation, never from the date: `deploy.yml` declares it,
        # and a deploy replayed against a historical day must not read as a
        # live one. Resolved before any of the work above is recorded.
        "run_mode": resolve_run_mode(getattr(args, "run_mode", None)),
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
            # Same surface, same rules, same resolver as `crucible --run-mode`
            # (`crucible.runmode`): this step writes a run manifest, and
            # `run_manifest.v2` requires the field with no default. Omitted,
            # $CRUCIBLE_RUN_MODE decides; neither, and the step refuses.
            p.add_argument("--run-mode", choices=list(RUN_MODES), default=None)

    args = parser.parse_args(argv)
    store = open_store(args.store)
    return {"publish": _publish, "capture": _capture, "flip": _flip, "record": _record}[args.step](
        args, store
    )


if __name__ == "__main__":
    sys.exit(main())
