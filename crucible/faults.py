"""`crucible fault.record` — the producer of `faults/{trading_day}/{fault}.json`.

Normative source: `alpha-engine-config-I10320` (there is no producer of the
key `crucible.gate._clause_fault_injection_against_scheduled_path` already
reads), `-I10322` (a fault induced on a replay trading day must not regress
`arc_runs_ok`/`replays_ok`, and the record that reconciles the two must not
become a way to turn an arbitrary red clause green) and `-I10327` (the record
had ONE outcome kind and two of plan §10.7's four faults do not take it).

**This module reads the reader; it does not redesign it.** The key shape
(`crucible.keys.fault_injection_key`), the scripted-fault vocabulary
(`crucible.gate.SCRIPTED_FAULTS`) and the two field names the gate clause
already requires (`crucible.gate.FAULT_RECORD_MANIFEST_FIELD`/
`FAULT_RECORD_BUS_FIELD`) are all declared by `crucible.gate`, which landed
first (`crucible-PR169`) and found no producer. This module conforms to that
contract.

**The whole design is the refusal, not the write.** Three outcome kinds
(`crucible.models.FAULT_OUTCOME_VALUES`), each with its own refusal, and no
kind reachable by relaxing another's:

* **`induced`** — the fault fired and the job FAILED. Refused unless a
  manifest with that exact `run_id` exists under
  `runs/{target_job}/{trading_day}/` and reads `status: failed`, and refused
  without a `bus_key` the store actually holds. A record that could excuse an
  arbitrary failure is worse than the conflict it resolves, and an induced
  fault that paged nobody is half of §10.7's exercise.
* **`absorbed`** — the fault fired and the system HANDLED it. Refused unless
  the named manifest reads `status: ok` AND its `attempts[]` carries a retry
  whose `reason` is in `crucible.runner.TRANSIENT_CLASSIFIERS`. `bus_key` is
  **forbidden**: a page here would mean the retry did not work, so its
  absence is required rather than tolerated. A STRONGER result than
  `induced`, not a weaker one — which is why it is not reachable by dropping
  a requirement from `induced`, but by meeting a different one.
* **`unreachable`** — the state cannot be entered at all. Carries NO
  `run_id`, so it is structurally incapable of excusing any manifest, and it
  is the HARDEST record here to write: a probe registered in
  :data:`UNREACHABLE_PROBES` is EXECUTED per closed path against the real
  store, and the record is refused unless every probe observed what it
  required. A fault with no registered probe is refused outright rather than
  accepted on an attestation somebody typed.

`bus_key` is required for `induced`, forbidden for `absorbed`, absent for
`unreachable` — never optional, and **never borrowable from an unrelated
incident**: it must parse as a bus key AND be present on the store.
`-I10317`'s agent proposed naming an existing bus row to satisfy §10.7, which
is the rubber stamp the `run_id` refusal exists to prevent arriving through
the other field.

`-I10322`'s exclusion stays keyed on `run_id` and is narrowed to `induced`
records, so `absorbed` and `unreachable` excuse nothing at all.

Every refusal is a `ValueError` raised inside `crucible.runner.run_job`'s job
body (AGENTS.md rule 1), so a refusal is itself a fully-telemetered
`status: failed` run of THIS job — never a hand-written record about an
exercise a human ran, and never a bare traceback.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from crucible.documents import read_store_document
from crucible.gate import FAULT_RECORD_BUS_FIELD, FAULT_RECORD_MANIFEST_FIELD, SCRIPTED_FAULTS
from crucible.keys import fault_injection_key, is_manifest_key, manifest_prefix, parse_bus_key
from crucible.models import FAULT_OUTCOME_VALUES, FaultRecordDocument
from crucible.runner import TRANSIENT_CLASSIFIERS, RunContext
from crucible.store import Store

__all__ = [
    "FAULT_RECORD_JOB",
    "FAULT_RECORD_SCHEMA_VERSION",
    "TRANSIENT_RETRY_REASONS",
    "UNREACHABLE_PROBES",
    "ClosedPath",
    "FaultRecordRefusedError",
    "record_fault",
]

#: The job name registered in `crucible.cli.JOBS`, `crucible/components.yaml`
#: and `crucible.models.JOB_VALUES` — a constant, imported by `cli.py`,
#: rather than a string literal restated at each of those three sites.
FAULT_RECORD_JOB = "fault.record"

FAULT_RECORD_SCHEMA_VERSION = "fault_record.v1"

#: The declared transient class's `attempts[].reason` vocabulary, DERIVED from
#: `crucible.runner.TRANSIENT_CLASSIFIERS` rather than restated. An `absorbed`
#: record is refused unless the manifest's retry names one of these, and a
#: hand-kept copy here would let a retry outside the declared class be
#: recorded as an absorption the moment the two lists drifted — which is the
#: same defect in the same shape as widening the class at runtime.
TRANSIENT_RETRY_REASONS: frozenset[str] = frozenset(
    reason for reason, _types, _needles in TRANSIENT_CLASSIFIERS
)


class FaultRecordRefusedError(ValueError):
    """The exercise this record would describe did not happen the way the
    record claims.

    Always raised, never swallowed — a `ValueError`, so `crucible.runner
    .run_job`'s `finally` still writes this JOB's own manifest naming the
    refusal (`status: failed`, `reason` carrying this message) before the
    process exits non-zero. A record that could be filed anyway is a way to
    turn an arbitrary red clause green (`alpha-engine-config-I10322`).
    """


class ClosedPath:
    """One executed probe's result, on its way into a `ClosedPathRow`.

    A plain carrier rather than the pydantic row itself so a probe cannot
    file a row that skips validation: `record_fault` builds the document and
    validates the whole thing once, in one place.
    """

    __slots__ = ("expected", "mechanism", "observed", "path", "probe")

    def __init__(self, *, path: str, mechanism: str, probe: str, expected: str, observed: str):
        self.path = path
        self.mechanism = mechanism
        self.probe = probe
        self.expected = expected
        self.observed = observed

    def as_row(self, now: dt.datetime) -> dict[str, str]:
        return {
            "path": self.path,
            "mechanism": self.mechanism,
            "probe": self.probe,
            "expected": self.expected,
            "observed": self.observed,
            "checked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


#: A sha that cannot name a published release: forty zeroes is a legal sha
#: SHAPE (so `crucible.release._assert_sha` accepts it and the probe reaches
#: the refusal it is testing) and is not a reachable git object id. If it ever
#: WERE published the probe would observe no refusal and this record would be
#: refused — the safe direction.
_UNPUBLISHABLE_SHA = "0" * 40


def _probe_pin_refuses_an_unpublished_sha(store: Store) -> ClosedPath:
    """`release.pin` refuses at WRITE time, so `releases/current` can never
    come to name a sha whose artifacts are not there.

    Executed, not asserted: `pin` is actually called, against a
    :func:`crucible.store.read_only` view of the real store, so the probe
    CANNOT move the pointer even if the refusal it is testing were absent. A
    `DryRunWriteRefusedError` instead of a `StaleReleasePointerError` means
    `pin` reached the compare-and-swap for an unpublished sha — the path is
    open, and this record is refused rather than filed over the finding.
    """
    from crucible.release import StaleReleasePointerError, pin  # noqa: PLC0415 - one call site
    from crucible.store import DryRunWriteRefusedError, read_only  # noqa: PLC0415

    expected = "StaleReleasePointerError, raised before any write"
    try:
        pin(read_only(store), _UNPUBLISHABLE_SHA)
    except StaleReleasePointerError as exc:
        return ClosedPath(
            path="`releases/current` comes to name a sha whose artifacts are absent",
            mechanism="crucible.release.pin",
            probe="pin_refuses_an_unpublished_sha",
            expected=expected,
            observed=f"StaleReleasePointerError: {exc}",
        )
    except DryRunWriteRefusedError as exc:
        raise FaultRecordRefusedError(
            f"pin_refuses_an_unpublished_sha: `release.pin` reached a WRITE for the "
            f"unpublished sha {_UNPUBLISHABLE_SHA} ({type(exc).__name__}: {exc}). The path "
            "this record claims is closed is OPEN, which is a finding, not a record."
        ) from exc
    except Exception as exc:
        raise FaultRecordRefusedError(
            f"pin_refuses_an_unpublished_sha: expected {expected}, observed "
            f"{type(exc).__name__}: {exc}. A probe that did not observe what it required "
            "is not evidence a path is closed."
        ) from exc
    raise FaultRecordRefusedError(
        f"pin_refuses_an_unpublished_sha: `release.pin` did NOT refuse the unpublished "
        f"sha {_UNPUBLISHABLE_SHA}. `stale_release_pointer` is reachable, so it is not "
        "an `unreachable` fault — record it as `induced` against the run it breaks."
    )


def _probe_published_release_objects_are_retained(store: Store) -> ClosedPath:
    """Every published release object carries Object Lock retention, so the
    other route into a stale pointer — the artifacts being deleted from under
    a pointer that stays valid — is closed too.

    Read through `crucible.release_lock_sweep.release_lock_findings`, the
    existing read-only detector, rather than a second retention reader here.
    An UNMEASURABLE finding refuses the record: a `LocalStore` has no Object
    Lock concept at all and reports every key unmeasurable, so this record
    cannot be filed from a laptop store — which is the point. Machine-checked
    evidence has to come from the store the claim is about.
    """
    from crucible.release import RELEASE_OBJECT_LOCK_RETENTION  # noqa: PLC0415
    from crucible.release_lock_sweep import release_lock_findings  # noqa: PLC0415

    expected = (
        f"every published release object MET the declared "
        f"{RELEASE_OBJECT_LOCK_RETENTION.days}-day Object Lock retention"
    )
    try:
        findings = list(release_lock_findings(store))
    except Exception as exc:
        raise FaultRecordRefusedError(
            f"published_release_objects_are_retained: the retention readings could not be "
            f"taken: {type(exc).__name__}: {exc}. That is a statement about our access, "
            "and a record may not be filed over one."
        ) from exc
    if not findings:
        raise FaultRecordRefusedError(
            "published_release_objects_are_retained: not one published release object was "
            "found, so 'every published object is locked' is vacuously true and evidence "
            "of nothing."
        )
    unmeasurable = [f for f in findings if f.state == "UNMEASURABLE"]
    if unmeasurable:
        causes = ", ".join(sorted({f.cause for f in unmeasurable})[:3])
        raise FaultRecordRefusedError(
            f"published_release_objects_are_retained: {len(unmeasurable)} of "
            f"{len(findings)} release object(s) could not be read ({causes}). An "
            "unreadable retention is not a closed path."
        )
    unmet = [f for f in findings if f.state == "UNMET"]
    if unmet:
        raise FaultRecordRefusedError(
            f"published_release_objects_are_retained: {len(unmet)} of {len(findings)} "
            f"release object(s) are NOT retained to policy — {unmet[0].detail} The path "
            "is open."
        )
    return ClosedPath(
        path="a published release artifact is deleted from under a valid pointer",
        mechanism="crucible.release.release_object_lock_params",
        probe="published_release_objects_are_retained",
        expected=expected,
        observed=f"{len(findings)} published release object(s) read, all MET",
    )


#: Per fault, the probes that must ALL observe what they require before an
#: `unreachable` record may be filed for it. A fault absent from this mapping
#: cannot be recorded `unreachable` at all — which is the enforcement of
#: `alpha-engine-config-I10327`'s rule that `unreachable` is the harder record
#: to write: adding a fault here means writing a probe, not writing a
#: sentence.
#:
#: `stale_release_pointer` (fault 4) has two, because the state has two routes
#: in and closing one is not closing the state: the pointer coming to name an
#: unpublished sha (`pin` refuses at write time) and the artifacts being
#: deleted from under a valid pointer (Object Lock retention).
#:
#: **Fault 4's outcome is settled, corrected 2026-09-10.** This comment said
#: it "is still an open ruling on the tracker (`-I10126`); this module makes
#: the kind writable and writes no record." A record has existed since
#: 2026-09-09T18:09:03Z — `faults/2026-08-07/stale_release_pointer.json`,
#: `outcome: unreachable`, both probes above observed. `-I10126` framed the
#: choice as re-scoping the fault or planting a decoy pin plus a governance
#: bypass, and `-I10327` dissolved it by building a third kind neither option
#: had: machine-executed evidence rather than an assertion. Nothing was ruled
#: out of band, and a stale "awaiting a ruling" line reads exactly like one
#: that was.
UNREACHABLE_PROBES: dict[str, tuple[Callable[[Store], ClosedPath], ...]] = {
    "stale_release_pointer": (
        _probe_pin_refuses_an_unpublished_sha,
        _probe_published_release_objects_are_retained,
    ),
}


def _find_manifest(
    store: Store, target_job: str, trading_day: str, run_id: str
) -> tuple[str, dict[str, Any]]:
    """The manifest key and document under `runs/{target_job}/{trading_day}/`
    whose own `run_id` equals ``run_id``, or a refusal naming exactly why not.

    Lists the whole per-day prefix rather than assuming the bare
    (undiscriminated) key: `target_job` may be one of the jobs that writes a
    discriminated manifest (`experiment.run`/`experiment.grade`/
    `alerts.sweep`), and this command is handed a `run_id`, not a
    discriminator.

    Says nothing about `status` — the caller checks the status ITS outcome
    requires, because `induced` and `absorbed` require opposite ones and a
    shared helper that checked either would be a helper that checked neither.
    """
    prefix = manifest_prefix(target_job, trading_day)
    try:
        keys = sorted(store.list_keys(prefix))
    except Exception as exc:
        raise FaultRecordRefusedError(
            f"could not list {prefix!r} to find the manifest run_id {run_id!r} claims to "
            f"describe: {type(exc).__name__}: {exc}. A record cannot be filed against a "
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
            "names the run it describes; this run does not exist on the store, so there "
            "is nothing for this record to be evidence of."
        )
    # `manifest_prefix` names AT MOST ONE writer per (job, trading_day, run_id)
    # in practice (a run_id is a fresh ULID per invocation), so a real
    # collision here is itself a finding worth surfacing rather than
    # silently picking one; the lexically-last key is deterministic either
    # way.
    return sorted(matches)[-1]


def _induced_evidence(
    store: Store, target_job: str, trading_day: str, run_id: str
) -> tuple[str, None]:
    """The FAILED manifest an `induced` record excuses."""
    key, document = _find_manifest(store, target_job, trading_day, run_id)
    if document.get("status") != "failed":
        # A record naming a successful run is exactly the arbitrary-excuse
        # shape this producer's whole design exists to forbid.
        raise FaultRecordRefusedError(
            f"{key} (run_id {run_id!r}) reads status {document.get('status')!r}, not "
            "'failed' — an `induced` fault record may only excuse a run that actually "
            "failed. A run the system handled is `--outcome absorbed`, which requires "
            "the retry to be recorded and forbids a bus key."
        )
    return key, None


def _absorbed_evidence(
    store: Store, target_job: str, trading_day: str, run_id: str
) -> tuple[str, dict[str, Any]]:
    """The OK manifest an `absorbed` record describes, plus the `attempts[]`
    row recording the declared-transient-class retry that made it ok.

    Both halves, and the second is the one that matters: a manifest reading
    `ok` says only that the job succeeded, which every clean first-attempt run
    also says. What makes it an ABSORPTION is a retry inside the declared
    class, and that lives in `attempts[]` — so a record filed off an `ok`
    manifest with no retry is refused, not accepted as a fault the system
    survived so gracefully it left no trace.
    """
    key, document = _find_manifest(store, target_job, trading_day, run_id)
    status = document.get("status")
    if status != "ok":
        raise FaultRecordRefusedError(
            f"{key} (run_id {run_id!r}) reads status {status!r}, not 'ok' — an "
            "`absorbed` fault record claims the system HANDLED the fault, so the run it "
            "names has to have succeeded. A run that failed is `--outcome induced`, "
            "which requires the bus row the failure paged."
        )
    attempts = document.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise FaultRecordRefusedError(
            f"{key} records no `attempts[]`, so it cannot show a retry. A manifest "
            "without the attempt ladder predates the guarantee this record reads and is "
            "not evidence of an absorption."
        )
    retries = [
        a
        for a in attempts
        if isinstance(a, dict)
        and isinstance(a.get("n"), int)
        and a["n"] > 1
        and a.get("reason") in TRANSIENT_RETRY_REASONS
    ]
    if not retries:
        reasons = [a.get("reason") for a in attempts if isinstance(a, dict)]
        raise FaultRecordRefusedError(
            f"{key} records {len(attempts)} attempt(s) with reason(s) {reasons} and not "
            "one retry in the declared transient class "
            f"({sorted(TRANSIENT_RETRY_REASONS)}). An `ok` manifest with no retry is a "
            "clean first-attempt run: it is evidence the job worked, never evidence a "
            "fault was absorbed."
        )
    # The LAST retry, when a future MAX_ATTEMPTS allows more than one: the
    # attempt that actually carried the run to `ok` is the one this record is
    # about.
    return key, dict(max(retries, key=lambda a: int(a["n"])))


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


def _unreachable_evidence(store: Store, fault_id: str, now: dt.datetime) -> list[dict[str, str]]:
    """Run every probe declared for ``fault_id`` and return the rows.

    A fault with no declared probe is REFUSED. That refusal is the whole of
    `alpha-engine-config-I10327`'s "if the evidence cannot be machine-checked,
    leave the clause UNMEASURABLE rather than accept an attestation a human
    types": there is no code path here that accepts prose.
    """
    probes = UNREACHABLE_PROBES.get(fault_id)
    if not probes:
        raise FaultRecordRefusedError(
            f"no machine-checkable evidence is declared for {fault_id!r}, so it cannot be "
            f"recorded `unreachable`. Registered: {sorted(UNREACHABLE_PROBES)}. An "
            "`unreachable` record with no executed probe would be an attestation a human "
            "typed, which is what plan §10.7 exists to replace — leave the clause "
            "UNMEASURABLE instead."
        )
    return [probe(store).as_row(now) for probe in probes]


def record_fault(
    ctx: RunContext,
    store: Store,
    *,
    fault_id: str,
    outcome: str,
    target_job: str | None,
    trading_day: str,
    run_id: str | None,
    bus_key: str | None,
) -> str:
    """Write `faults/{trading_day}/{fault_id}.json`, or refuse.

    Returns the key written. Raises :class:`FaultRecordRefusedError` for
    every refusal named in the module docstring. Called from inside
    `crucible.runner.run_job`'s job body (see `crucible.cli`'s
    `fault.record` handler) so a refusal is itself a fully-telemetered
    `status: failed` run of this job, never an exception that vanishes.

    Overwrites in place if a record already exists for ``(fault_id,
    trading_day)`` — re-exercising the same fault on the same day and
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
    if outcome not in FAULT_OUTCOME_VALUES:
        raise FaultRecordRefusedError(
            f"outcome {outcome!r} is not one of {list(FAULT_OUTCOME_VALUES)}. The three "
            "kinds are a closed set for the same reason the runner has two statuses: a "
            "fourth is a design change that has to be visible in a diff."
        )

    manifest_key: str | None = None
    attempt: dict[str, Any] | None = None
    closed_paths: list[dict[str, str]] | None = None

    if outcome == "unreachable":
        for name, value in (("--run-id", run_id), ("--target-job", target_job)):
            if value:
                raise FaultRecordRefusedError(
                    f"an `unreachable` record names no run, so {name} {value!r} is "
                    "refused. Carrying one would make a record that claims a state "
                    "cannot be entered also capable of excusing a manifest — the exact "
                    "widening the run_id-keyed exclusion exists to prevent."
                )
        if bus_key:
            raise FaultRecordRefusedError(
                f"an `unreachable` record names no run, so there is nothing for a page to "
                f"be about; --bus-key {bus_key!r} is refused."
            )
        closed_paths = _unreachable_evidence(store, fault_id, ctx.started)
    else:
        if not run_id or not target_job:
            raise FaultRecordRefusedError(
                f"an `{outcome}` record describes a run, so both --target-job and "
                "--run-id are required; this one names "
                f"target_job={target_job!r}, run_id={run_id!r}."
            )
        if outcome == "induced":
            if not bus_key:
                raise FaultRecordRefusedError(
                    "an `induced` record requires --bus-key: the fault fired and the job "
                    "failed, so the sweep filed a page, and a manifest with no bus row is "
                    "half of plan §10.7's exercise. If the live-sweep half has not "
                    "produced the row yet, the record waits for it — the clause reads "
                    "UNMEASURABLE until then, which is the honest reading."
                )
            _validate_bus_key(store, bus_key)
            manifest_key, attempt = _induced_evidence(store, target_job, trading_day, run_id)
        else:
            if bus_key:
                raise FaultRecordRefusedError(
                    f"an `absorbed` record forbids --bus-key, and this one names "
                    f"{bus_key!r}. A page on an absorbed fault would mean the retry did "
                    "NOT work, so the bus row's ABSENCE is part of what this record "
                    "asserts — it is not a field to be borrowed from an unrelated "
                    "incident to make a clause read better."
                )
            manifest_key, attempt = _absorbed_evidence(store, target_job, trading_day, run_id)

    document: dict[str, Any] = {
        "schema_version": FAULT_RECORD_SCHEMA_VERSION,
        "fault_id": fault_id,
        "outcome": outcome,
        "trading_day": trading_day,
        "run_id": run_id or None,
        FAULT_RECORD_MANIFEST_FIELD: manifest_key,
        FAULT_RECORD_BUS_FIELD: bus_key or None,
        "attempt": attempt,
        "closed_paths": closed_paths,
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
