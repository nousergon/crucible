"""`python -m crucible.deploy` — the three steps `deploy.yml` drives.

Normative source: plan §4.11.

    publish   upload the wheel and release.json under releases/{sha}/
    flip      read the smoke MANIFEST and, only on `ok`, repoint current
    record    write the deploy's own run manifest, on BOTH paths

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
    ReleaseRecord,
    current_release,
    flip_on_smoke,
    read_pointer,
    release_json_key,
    wheel_key,
    write_deploy_manifest,
)
from crucible.store import Store, open_store

__all__ = ["main"]

_UNKNOWN_SHA = "0" * 40


def _publish(args: argparse.Namespace, store: Store) -> int:
    """Upload the immutable half. Promotes nothing.

    `release.json` is built in the workflow (where the lockfile and the test
    summary are) and validated here against :class:`ReleaseRecord`, so a
    malformed record is refused before it is durable rather than discovered
    by whatever reads it next.
    """
    wheel = Path(args.wheel).read_bytes()
    record = ReleaseRecord(**json.loads(Path(args.release_json).read_text(encoding="utf-8")))
    if record.sha != args.sha:
        raise SystemExit(
            f"release.json is for {record.sha}, not {args.sha}. Publishing it under the "
            "wrong prefix would make the rollback target a build it does not describe."
        )
    store.put_bytes(wheel_key(args.sha), wheel)
    store.put_bytes(release_json_key(args.sha), record.to_json())
    print(f"published releases/{args.sha}/ ({len(wheel)} bytes)")
    return 0


def _flip(args: argparse.Namespace, store: Store) -> int:
    """Read the smoke manifest and flip only on `ok`.

    Exits non-zero when the smoke did not pass, so the deploy is FAILED and
    visible rather than green with an unmoved pointer — "nothing was promoted"
    and "nothing needed promoting" must not look the same on the surface.
    """
    trading_day = resolve_trading_day()
    key = manifest_key("smoke", trading_day.isoformat())
    if not store.exists(key):
        raise SystemExit(
            f"no smoke manifest at {key}. The gate is the smoke RUN; promoting without "
            "one would flip the pointer on a step that may never have executed."
        )
    smoke = json.loads(store.get_bytes(key).decode("utf-8"))
    before, version = read_pointer(store, POINTER_KEY)
    flipped = flip_on_smoke(store, sha=args.sha, smoke_manifest=smoke, expect=version)
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
        description="The three steps deploy.yml drives: publish, flip, record.",
    )
    sub = parser.add_subparsers(dest="step", required=True)

    for name, help_text in (
        ("publish", "Upload the wheel and release.json. Promotes nothing."),
        ("flip", "Read the smoke manifest and repoint releases/current on `ok`."),
        ("record", "Write the deploy's own run manifest, on both paths."),
    ):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--sha", required=True)
        p.add_argument("--store", required=True)
        if name == "publish":
            p.add_argument("--wheel", required=True)
            p.add_argument("--release-json", required=True)
        if name == "record":
            p.add_argument("--outcome", required=True)
            p.add_argument("--run-url", default="")

    args = parser.parse_args(argv)
    store = open_store(args.store)
    return {"publish": _publish, "flip": _flip, "record": _record}[args.step](args, store)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
