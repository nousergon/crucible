"""`crucible fault.record` — the producer of `faults/{trading_day}/{fault}.json`.

Normative source: `alpha-engine-config-I10320` (there is no producer of the
key `crucible.gate._clause_fault_injection_against_scheduled_path` already
reads) and `-I10322` (a fault induced on a replay trading day must not
regress `arc_runs_ok`/`replays_ok`, and the record that reconciles the two
must not become a way to turn an arbitrary red clause green).

**This module reads the reader; it does not redesign it.** The key shape
(`crucible.keys.fault_injection_key`), the scripted-fault vocabulary
(`crucible.gate.SCRIPTED_FAULTS`) and the two fields the gate clause already
requires (`crucible.gate.FAULT_RECORD_MANIFEST_FIELD`/`FAULT_RECORD_BUS_FIELD`)
are all declared by `crucible.gate`, which landed first (`crucible-PR169`)
and found no producer. This module conforms to that contract.

**The whole design is the refusal, not the write** (`-I10322`):

* a record names the `run_id` it excuses and is REFUSED at write time unless
  a manifest with that exact `run_id` exists under
  `runs/{target_job}/{trading_day}/` and reads `status: failed` — a record
  that could excuse an arbitrary failure is worse than the conflict it
  resolves;
* it is written by the injection procedure itself, as a CLI job going
  through `crucible.runner.run_job` like every other job (AGENTS.md rule
  1) — never typed by hand after the fact. A human-authored record about an
  exercise a human ran is exactly the assertion-by-having-looked this clause
  exists to remove;
* `--bus-key`, when given, is refused unless it is shaped like a real
  alert-bus key (`crucible.keys.parse_bus_key`) AND the store actually holds
  it — a record naming a page that never fired is the same overclaim in
  miniature. Omitted, the record's `bus_key` is null — legitimate when the
  induction's live-sweep half has not produced a bus row yet
  (`alpha-engine-config-I10125`); `fault_injection_against_scheduled_path`
  then reads that fault UNMET, not refused.

**Fault 4 (`stale_release_pointer`) is out of scope for this module.**
`crucible.release.pin` already refuses at write time before that state can
ever be reached in production (verified against a `LocalStore` with an
unpublished sha), so it can never produce a FAILED manifest for this
producer to name under the shape above. Whether and how fault 4 should ever
be recorded is an open design question on the tracker, not something this
module decides or works around.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from crucible.documents import read_store_document
from crucible.gate import FAULT_RECORD_BUS_FIELD, FAULT_RECORD_MANIFEST_FIELD, SCRIPTED_FAULTS
from crucible.keys import fault_injection_key, is_manifest_key, manifest_prefix, parse_bus_key
from crucible.models import FaultRecordDocument
from crucible.runner import RunContext
from crucible.store import Store

__all__ = [
    "FAULT_RECORD_JOB",
    "FAULT_RECORD_SCHEMA_VERSION",
    "FaultRecordRefusedError",
    "record_fault",
]

#: The job name registered in `crucible.cli.JOBS`, `crucible/components.yaml`
#: and `crucible.models.JOB_VALUES` — a constant, imported by `cli.py`,
#: rather than a string literal restated at each of those three sites.
FAULT_RECORD_JOB = "fault.record"

FAULT_RECORD_SCHEMA_VERSION = "fault_record.v1"


class FaultRecordRefusedError(ValueError):
    """The exercise this record would describe did not happen the way the
    record claims.

    Always raised, never swallowed — a `ValueError`, so `crucible.runner
    .run_job`'s `finally` still writes this JOB's own manifest naming the
    refusal (`status: failed`, `reason` carrying this message) before the
    process exits non-zero. A record that could be filed anyway is a way to
    turn an arbitrary red clause green (`alpha-engine-config-I10322`).
    """


def _find_failed_manifest(store: Store, target_job: str, trading_day: str, run_id: str) -> str:
    """The manifest key under `runs/{target_job}/{trading_day}/` whose own
    `run_id` equals ``run_id`` and whose `status` is `failed`, or a refusal
    naming exactly why not.

    Lists the whole per-day prefix rather than assuming the bare
    (undiscriminated) key: `target_job` may be one of the jobs that writes a
    discriminated manifest (`experiment.run`/`experiment.grade`/
    `alerts.sweep`), and this command is handed a `run_id`, not a
    discriminator.
    """
    prefix = manifest_prefix(target_job, trading_day)
    try:
        keys = sorted(store.list_keys(prefix))
    except Exception as exc:
        raise FaultRecordRefusedError(
            f"could not list {prefix!r} to find the manifest run_id {run_id!r} claims to "
            f"excuse: {type(exc).__name__}: {exc}. A record cannot be filed against a "
            "listing we could not read."
        ) from exc
    matches: list[tuple[str, dict[str, Any]]] = []
    for key in keys:
        if not is_manifest_key(key):
            continue
        read = read_store_document(store, key)
        if read.problem is not None or read.absent:
            continue
        document = read.document or {}
        if document.get("run_id") == run_id:
            matches.append((key, document))
    if not matches:
        raise FaultRecordRefusedError(
            f"no manifest under {prefix!r} carries run_id {run_id!r}. A fault record "
            "names the run it excuses; this run does not exist on the store, so there is "
            "nothing for this record to be evidence of."
        )
    # `manifest_prefix` names AT MOST ONE writer per (job, trading_day, run_id)
    # in practice (a run_id is a fresh ULID per invocation), so a real
    # collision here is itself a finding worth surfacing rather than
    # silently picking one; the lexically-last key is deterministic either
    # way.
    key, document = sorted(matches)[-1]
    if document.get("status") != "failed":
        # A record naming a successful run is exactly the arbitrary-excuse
        # shape this producer's whole design exists to forbid (see module
        # docstring).
        raise FaultRecordRefusedError(
            f"{key} (run_id {run_id!r}) reads status {document.get('status')!r}, not "
            "'failed' — a fault record may only excuse a run that actually failed."
        )
    return key


def _validate_bus_key(store: Store, bus_key: str) -> None:
    if parse_bus_key(bus_key) is None:
        raise FaultRecordRefusedError(
            f"--bus-key {bus_key!r} is not shaped like an alert bus key "
            "(alerts/{trading_day}/{incident_id}.json)."
        )
    try:
        present = store.exists(bus_key)
    except Exception as exc:
        raise FaultRecordRefusedError(
            f"--bus-key {bus_key!r} could not be checked against the store: "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    if not present:
        raise FaultRecordRefusedError(
            f"--bus-key {bus_key!r} names a key the store does not hold — a fault "
            "record may only name a page that actually fired."
        )


def record_fault(
    ctx: RunContext,
    store: Store,
    *,
    fault_id: str,
    target_job: str,
    trading_day: str,
    run_id: str,
    bus_key: str | None,
) -> str:
    """Write `faults/{trading_day}/{fault_id}.json`, or refuse.

    Returns the key written. Raises :class:`FaultRecordRefusedError` for
    every refusal named in the module docstring. Called from inside
    `crucible.runner.run_job`'s job body (see `crucible.cli`'s
    `fault.record` handler) so a refusal is itself a fully-telemetered
    `status: failed` run of this job, never an exception that vanishes.

    Overwrites in place if a record already exists for ``(fault_id,
    trading_day)`` — re-inducing the same fault on the same day and
    re-filing is correct (`crucible.keys.fault_injection_key`'s own
    docstring): the record describes the day's state, and two records for
    one question is two answers to it.
    """
    if fault_id not in SCRIPTED_FAULTS:
        raise FaultRecordRefusedError(
            f"fault_id {fault_id!r} is not one of the {len(SCRIPTED_FAULTS)} scripted "
            f"faults plan §10.7 names ({sorted(SCRIPTED_FAULTS)}). A record for a fault "
            "the gate does not grade would never move any clause and is not evidence of "
            "anything."
        )
    found_manifest_key = _find_failed_manifest(store, target_job, trading_day, run_id)
    if bus_key:
        _validate_bus_key(store, bus_key)
    document: dict[str, Any] = {
        "schema_version": FAULT_RECORD_SCHEMA_VERSION,
        "fault_id": fault_id,
        "trading_day": trading_day,
        "run_id": run_id,
        FAULT_RECORD_MANIFEST_FIELD: found_manifest_key,
        FAULT_RECORD_BUS_FIELD: bus_key,
        "recorded_at_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        FaultRecordDocument.model_validate(document)
    except ValidationError as exc:
        # Refuse BEFORE writing anything — a producer that could emit a
        # non-conformant document would defeat the schema entirely (the same
        # discipline `crucible.data.universe.DeclaredUniverse.record` uses).
        detail = "\n".join(
            f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise FaultRecordRefusedError(
            f"fault_record.v1: document does not conform:\n{detail}"
        ) from exc
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    key = fault_injection_key(fault_id, trading_day)
    ctx.record_output(key, payload, schema_version=FAULT_RECORD_SCHEMA_VERSION)
    return key
