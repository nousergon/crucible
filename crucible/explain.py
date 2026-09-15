"""`crucible explain <run_id|key>` — the lineage walk, from manifests alone.

Normative source: plan §10 component 8; principle 1 (Transparency).

Principle 1 asks whether someone can reconstruct *why* an unattended run did
what it did, **from durable artifacts alone, without asking Brian**. The
manifests already make that possible; this command makes it a five-second
operation instead of an S3 archaeology session.

**It reads manifests and nothing else.** No index, no graph store, no
side-table that could drift from the runs it describes. The chain is
recovered by the one relation the manifest schema already carries: a run's
`inputs[].key` is some other run's `outputs[].key`. Walking that backwards
from a verdict reaches the arena cycle, the shadows, the feature layer, the
price panel and the data source — with each hop's `code_sha`, `seed`,
`cost_usd` and `llm_calls` attached, because "why" includes which code, on
which seed, at what cost.

**SOTA:** a lineage graph (OpenLineage, MLflow). **Delta:** a manifest walk
with no graph store — at one producer per key the edges are already in the
artifacts, and a second store of the same edges is a thing to keep in sync.

**An unresolvable hop is reported, never elided.** A key nobody claims as an
output is printed as `produced by: UNKNOWN`, with the key. A chain that
quietly dropped its unexplained hops would read as complete.

**A walk that crosses the money path verifies the hash chain**
(`alpha-engine-config-I10414`, plan §9.5). Every money-path manifest carries
`money_path_link.prev_sha256` — the digest of its predecessor's stored bytes,
written by `crucible.manifest.write_manifest`. :func:`verify_money_path_chain`
walks that and grades it `ok` or `failed`, the manifest's own two statuses and
no third. A break is a FINDING, not a log line: it names the record index, the
run, the digest the record claims and the digest the store actually holds, and
:meth:`ChainVerification.raise_if_broken` turns it into a non-zero exit for a
caller that must fail on it. The v1 NAV series was manually restated over four
sessions; a restatement is legitimate, an unnoticed one is not, and this is
what makes the difference machine-checkable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crucible.documents import load_store_document, read_manifests_under
from crucible.keys import EXPERIMENTS_ROOT, RUNS_ROOT, manifest_key
from crucible.manifest import (
    ManifestValidationError,
    MoneyPathChainError,
    digest_of_stored,
    money_path_manifests,
    money_path_writes,
    on_money_path,
    validate,
)

if TYPE_CHECKING:
    from crucible.store import Store

__all__ = [
    "ChainRecord",
    "ChainVerification",
    "Lineage",
    "ManifestLoad",
    "MoneyPathChainError",
    "NoSettledVerdictError",
    "explain",
    "load_manifests",
    "render",
    "select_newest_settled_verdict",
    "verify_money_path_chain",
]


class NoSettledVerdictError(RuntimeError):
    """No `verdict.json` exists anywhere under `experiments/` — nothing for
    a scheduled `explain --select-newest-verdict` refresh to walk.

    Raised inside the job body (`crucible.track_a.handle_explain`), not
    before it, so `crucible.runner.run_job` still files a `failed` manifest
    naming this — rule 1 (manifest or it did not happen): a scheduled run
    that found nothing to walk is a fact about the system, not a fact this
    command is entitled to leave unrecorded.
    """


def select_newest_settled_verdict(store: Store) -> str:
    """The key of the most recently SETTLED `verdict.json`, across every
    slot and arm — deterministic, for the scheduled `explain` arc stage
    (`alpha-engine-config-I10858`).

    "Newest" is the verdict document's own `trading_day` field, not the S3
    key's lexical order (an arm's hashed segment sorts arbitrarily relative
    to another arm's) and not object last-modified (`experiment.backfill`
    can write an old decision date's verdict long after a newer one already
    exists). A tie — two verdicts settling the same `trading_day` in
    different slots — breaks on the key itself, which is a stable total
    order and keeps the selection reproducible for the same store state
    rather than depending on iteration order.

    Every verdict this store holds is by construction SETTLED: a slot never
    writes `verdict.json` before its horizon has passed
    (`crucible.slots.grading`), so there is no separate "is it settled" test
    here — reading the key at all is the settlement fact.

    Raises :class:`NoSettledVerdictError` rather than returning `None` —
    §5 (fail loud): a caller silently walking nothing would file an `explain`
    manifest whose `inputs` never include a verdict, which is exactly the
    gap `explain_walks_a_verdict` exists to detect, and swallowing it here
    would hide that detector's own failure mode from itself.
    """
    best_key: str | None = None
    best_day: dt.date | None = None
    for key in sorted(store.list_keys(EXPERIMENTS_ROOT)):
        if not key.endswith("/verdict.json"):
            continue
        document = load_store_document(store, key)
        day = dt.date.fromisoformat(document["trading_day"])
        if best_day is None or day > best_day or (day == best_day and key > (best_key or "")):
            best_day = day
            best_key = key
    if best_key is None:
        raise NoSettledVerdictError(
            "no settled verdict.json exists anywhere under experiments/ in this store — "
            "nothing for a scheduled explain refresh to walk"
        )
    return best_key


#: How deep the walk goes before it stops. A cycle in the input/output graph
#: is impossible by construction (a key has one producer, and a run cannot
#: read its own output), so a limit reached is a bug worth surfacing rather
#: than a depth to raise.
MAX_DEPTH = 12


@dataclass
class Lineage:
    """One node of the chain: a run, the key it explains, and what fed it.

    ``collisions`` is the OTHER run ids that also claimed this key as an
    output, oldest first — empty for the overwhelming majority of keys,
    which have exactly one producer. It is never used to pick a different
    winner; :func:`_index` already decided that (the later run, by
    `finished`). It exists so the caller sees both run_ids rather than one,
    per this module's own docstring: a shared output key is the
    last-writer-wins shape that gave a cycle's verdict to its
    worst-informed author, and a walk that silently kept only the winner
    would hide the exact defect it is meant to surface.
    """

    key: str
    manifest: dict[str, Any] | None
    depth: int
    parents: list[Lineage] = field(default_factory=list)
    collisions: tuple[str, ...] = ()
    #: The money-path chain verdict, set on the ROOT node only and only when
    #: this walk crosses the money path (alpha-engine-config-I10414). `None`
    #: on every other node and on every walk that touches no money-path
    #: artifact — which is a different answer from "the chain verified", and
    #: `render` prints the two differently for exactly that reason.
    chain: ChainVerification | None = None
    #: Every manifest under `runs/` this walk's own :func:`load_manifests`
    #: call could not validate — `{key: reason}` — set on the ROOT node only,
    #: the same convention `chain` uses (`alpha-engine-config-I10626`). Not
    #: about THIS walk's hops: a store-wide count, so an operator sees the
    #: gap even on a walk that never touches the broken keys. Named rather
    #: than elided, in the same spirit as `produced by: UNKNOWN`.
    unreadable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        run = self.manifest or {}
        return {
            "key": self.key,
            "produced_by": run.get("run_id", "UNKNOWN"),
            "job": run.get("job"),
            "trading_day": run.get("trading_day"),
            "status": run.get("status"),
            "reason": run.get("reason"),
            "code_sha": run.get("code_sha"),
            "release_sha": run.get("release_sha"),
            "seed": run.get("seed"),
            "cost_usd": run.get("cost_usd"),
            "llm_calls": len(run.get("llm_calls") or []),
            "inputs": [i["key"] for i in run.get("inputs") or []],
            "also_claimed_by": list(self.collisions),
            "money_path_chain": self.chain.to_dict() if self.chain is not None else None,
            "unreadable_manifests": {
                "count": len(self.unreadable),
                "keys": dict(self.unreadable),
            },
            "parents": [p.to_dict() for p in self.parents],
        }


@dataclass(frozen=True)
class ChainRecord:
    """One verified (or refuted) record of the money-path hash chain."""

    index: int
    run_id: str
    key: str
    #: The digest of this record's OWN stored bytes, as the store holds them
    #: now. Its successor's `prev_sha256` is compared against this.
    sha256: str
    #: What this record CLAIMS its predecessor's digest is. `None` at index 0.
    prev_sha256: str | None
    #: What the predecessor's digest ACTUALLY is, read back from the store.
    #: `None` at index 0. Equal to `prev_sha256` on an intact link — and the
    #: pair is carried rather than a bare boolean because "the chain is
    #: broken" is not an actionable finding and "claimed X, store holds Y" is.
    actual_prev_sha256: str | None
    money_path_writes: tuple[str, ...]

    @property
    def intact(self) -> bool:
        return self.prev_sha256 == self.actual_prev_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "run_id": self.run_id,
            "manifest": self.key,
            "sha256": self.sha256,
            "prev_sha256_claimed": self.prev_sha256,
            "prev_sha256_actual": self.actual_prev_sha256,
            "money_path_writes": list(self.money_path_writes),
            "intact": self.intact,
        }


@dataclass(frozen=True)
class ChainVerification:
    """The money-path chain's verdict. `ok` or `failed`, and nothing else.

    The manifest's own two statuses (plan §4.2), deliberately: a chain with a
    `degraded` or `unverified` state would be the third status this whole
    system exists to make unrepresentable, and it is the state a tampered
    history would settle into.
    """

    status: str
    reason: str
    records: tuple[ChainRecord, ...]
    #: Money-path manifests written at or after the genesis record that carry
    #: NO link, oldest first. Each is a `failed` finding in its own right: a
    #: record can only be removed from a hash chain by removing its link, so
    #: an unlinked money-path manifest is either the tamper or the producer
    #: that skipped the writer. Never a warning, and never dropped — a chain
    #: that verified the records it still had would grade a deletion green.
    unlinked: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "records": [r.to_dict() for r in self.records],
            "unlinked": list(self.unlinked),
        }

    def raise_if_broken(self) -> None:
        """Raise :class:`crucible.manifest.MoneyPathChainError` unless `ok`.

        The hook for a caller whose correct response to a break is to STOP —
        the `crucible explain` CLI handler, and any phase-4 gate clause that
        reads chain verification. `explain` itself never raises: an operator
        holding a broken chain needs to see the walk that reaches it, and a
        command that refused to print the lineage at the moment the lineage
        became interesting would be the wrong trade.
        """
        if not self.ok:
            raise MoneyPathChainError(self.reason)


def verify_money_path_chain(store: Store) -> ChainVerification:
    """Walk the money-path hash chain in ``store`` and grade it.

    Every money-path manifest (`crucible.manifest.money_path_manifests`) is
    ordered by `(finished, run_id)` and checked three ways:

    1. **Indices are 0..N-1, contiguous.** A gap is a removed record. A chain
       that renumbered around one would report a deletion as health.
    2. **Each record's `prev_sha256` equals the digest of the predecessor's
       bytes AS THE STORE HOLDS THEM NOW** — read back and hashed here, never
       compared against a stored copy of itself and never against
       `Store.etag`, which is an opaque backend version token
       (`nous-ergon-ops-I1145`). This is the check that catches the mutation:
       edit any record and its successor's link stops matching.
    3. **No money-path manifest at or after the genesis lacks a link.** A
       record leaves a hash chain by having its link removed, so the absence
       is the finding.

    Returns `status: ok` over an empty chain, with a reason saying so: a store
    where nothing has touched the money path has an intact (empty) history,
    which is a true statement and a different one from "verified 40 records".
    """
    chain = money_path_manifests(store)
    linked = [(k, m) for k, m in chain if m.get("money_path_link") is not None]
    unlinked = [(k, m) for k, m in chain if m.get("money_path_link") is None]

    if not linked:
        if unlinked:
            return ChainVerification(
                status="failed",
                reason=(
                    f"{len(unlinked)} money-path manifest(s) exist and NOT ONE carries a "
                    f"`money_path_link`: {[k for k, _ in unlinked]}. Either every link was "
                    "removed, or these runs predate the chain and it has never been started "
                    "against this store — and those two are indistinguishable from here, "
                    "which is why this is `failed` rather than a quiet pass over a store "
                    "with no chain in it."
                ),
                records=(),
                unlinked=tuple(k for k, _ in unlinked),
            )
        return ChainVerification(
            status="ok",
            reason="no money-path artifact has been written in this store; the chain is empty.",
            records=(),
        )

    genesis_finished = linked[0][1]["finished"]
    orphans = tuple(k for k, m in unlinked if m["finished"] >= genesis_finished)

    records: list[ChainRecord] = []
    findings: list[str] = []
    previous: tuple[str, dict[str, Any], str] | None = None  # (key, manifest, digest)
    for position, (key, manifest) in enumerate(linked):
        link = manifest["money_path_link"]
        actual_prev = digest_of_stored(store, previous[0]) if previous is not None else None
        record = ChainRecord(
            index=link["index"],
            run_id=manifest["run_id"],
            key=key,
            sha256=digest_of_stored(store, key),
            prev_sha256=link["prev_sha256"],
            actual_prev_sha256=actual_prev,
            money_path_writes=money_path_writes(manifest),
        )
        records.append(record)
        if record.index != position:
            findings.append(
                f"record at chain position {position} declares index {record.index} "
                f"(run {record.run_id}, {key}). Indices are contiguous from 0; a gap is a "
                "record that was removed, and a chain renumbered around one would report "
                "the deletion as health."
            )
        if not record.intact:
            findings.append(
                f"CHAIN BROKEN at record {record.index} (run {record.run_id}, {key}): it "
                f"claims prev_sha256={record.prev_sha256!r}, and the predecessor it names "
                f"({link['prev_run_id']}, {previous[0] if previous else None}) actually "
                f"hashes to {record.actual_prev_sha256!r}. The predecessor's bytes are not "
                "the bytes this record was written against."
            )
        if previous is not None and link["prev_run_id"] != previous[1]["run_id"]:
            findings.append(
                f"record {record.index} (run {record.run_id}) names predecessor "
                f"{link['prev_run_id']!r}, but the record before it in the chain is run "
                f"{previous[1]['run_id']!r}. The digest and the run id must point at the "
                "same predecessor or the chain is describing two different histories."
            )
        previous = (key, manifest, record.sha256)

    for orphan in orphans:
        findings.append(
            f"{orphan} wrote a money-path artifact at or after the chain's genesis and "
            "carries no `money_path_link`. A record leaves a hash chain by having its link "
            "removed, so an unlinked money-path manifest is the tamper, or a producer that "
            "bypassed `crucible.manifest.write_manifest`."
        )

    if findings:
        return ChainVerification(
            status="failed",
            reason="money-path chain verification FAILED:\n  - " + "\n  - ".join(findings),
            records=tuple(records),
            unlinked=orphans,
        )
    return ChainVerification(
        status="ok",
        reason=f"{len(records)} money-path record(s) verified, indices 0..{len(records) - 1}.",
        records=tuple(records),
    )


@dataclass(frozen=True)
class ManifestLoad:
    """Every readable, conformant manifest under `runs/`, and every one that
    was not — SURFACE face (alpha-engine-config-I10626), not strict.

    `explain` publishes what it can, exactly like the console, the board and
    the ladder do over the same store: a corrupt or non-conforming manifest
    is a fault named on the output, never a reason the whole walk refuses to
    run. `unreadable` covers two distinct failure modes without collapsing
    them into one bucket-with-no-reason: a JSON-level fault from
    `crucible.documents.read_manifests_under` (corrupt body, listed-then-
    vanished, wrong shape) and a SCHEMA-level one from `crucible.manifest.
    validate` (a conformant JSON object that still fails `run_manifest.v2` —
    the all-zero `code_sha` placeholder, before `alpha-engine-config-I10626`'s
    migration, was exactly this).
    """

    manifests: tuple[dict[str, Any], ...] = ()
    unreadable: dict[str, str] = field(default_factory=dict)


def load_manifests(store: Store) -> ManifestLoad:
    """Every run manifest in the store: the ones that validate, and the ones
    that do not, named rather than raised over.

    `crucible.documents.read_manifests_under` already refuses to collapse a
    listing failure into "nothing here" (`raise_if_unlistable`) — that
    failure mode is about OUR ACCESS to the prefix, not about any one
    manifest, so it stays a raise; a per-manifest fault does not.
    """
    listing = read_manifests_under(store, RUNS_ROOT)
    listing.raise_if_unlistable()
    manifests: list[dict[str, Any]] = []
    unreadable: dict[str, str] = dict(listing.faults)
    for key, document in listing.documents:
        try:
            validate(document)
        except ManifestValidationError as exc:
            unreadable[key] = str(exc)
            continue
        manifests.append(document)
    return ManifestLoad(tuple(manifests), unreadable)


def _index(
    manifests: list[dict[str, Any]],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, tuple[str, ...]]]:
    """`{run_id: manifest}`, `{output key: manifest}`, and `{output key: other claimants}`.

    A key claimed as an output by two runs keeps the LATER one (by
    `finished`) as the answer `by_output` gives; that part was already true.
    What was missing is the third return value: every OTHER run_id that also
    claimed the key, oldest first, so `explain`/`render` can attach them to
    the node instead of the collision vanishing when the later manifest
    overwrites the earlier one in a plain dict.
    """
    by_run: dict[str, dict] = {}
    by_output: dict[str, dict] = {}
    claimants: dict[str, list[str]] = {}
    for manifest in sorted(manifests, key=lambda m: m["finished"]):
        by_run[manifest["run_id"]] = manifest
        for output in manifest.get("outputs") or []:
            key = output["key"]
            by_output[key] = manifest
            claimants.setdefault(key, []).append(manifest["run_id"])
    collisions = {
        key: tuple(run_ids[:-1]) for key, run_ids in claimants.items() if len(run_ids) > 1
    }
    return by_run, by_output, collisions


def explain(store: Store, target: str) -> Lineage:
    """Walk backwards from a run id or an artifact key.

    A `run_id` resolves to that run and then to everything it read; an
    artifact key resolves to the run that wrote it. Both are accepted
    because an operator holding a verdict has a key, and an operator holding
    a page has a run id.
    """
    load = load_manifests(store)
    manifests = list(load.manifests)
    if not manifests:
        suffix = (
            f" {len(load.unreadable)} manifest(s) ARE present but did not validate: "
            f"{', '.join(sorted(load.unreadable))}."
            if load.unreadable
            else ""
        )
        raise FileNotFoundError(
            "no CONFORMANT run manifests under `runs/` in this store. Nothing readable "
            "has run here, so there is no lineage — which is a different answer from "
            "'this verdict has no explanation'." + suffix
        )
    by_run, by_output, collisions = _index(manifests)

    if target in by_run:
        root_manifest: dict[str, Any] | None = by_run[target]
        root_key = manifest_key(
            root_manifest["job"],
            root_manifest["trading_day"],
            discriminator=root_manifest.get("discriminator"),
        )
    elif target in by_output:
        root_manifest = by_output[target]
        root_key = target
    else:
        raise KeyError(
            f"{target!r} is neither a run_id nor a key any run claims as an output. "
            f"The store holds {len(manifests)} manifest(s) claiming "
            f"{len(by_output)} output key(s); `crucible explain` never guesses a near "
            "match, because an explanation of the wrong run is worse than none."
        )

    root = _walk(root_key, root_manifest, by_output, collisions, depth=0, seen=set())
    root.unreadable = dict(load.unreadable)
    if _crosses_money_path(root):
        # Verified over the WHOLE store, not over the nodes this walk
        # reached. A chain is only evidence if the record before the one you
        # are looking at is checked too, and the walk that brought an operator
        # here has no reason to have visited it.
        root.chain = verify_money_path_chain(store)
    return root


def _crosses_money_path(node: Lineage) -> bool:
    """Whether any hop of this walk touches a money-path artifact.

    Three signals, because each alone has a blind spot: the node's own key; a
    money-path OUTPUT of the run at that node (the run that wrote the champion
    pointer is on the money path even when the walk arrived at it by another
    of its outputs); and a `money_path_link` already on the manifest (which
    catches a run whose money-path output key has since been retired from
    `MONEY_PATH_PREDICATES` — its record is still in the chain, and the chain
    must still verify).
    """
    if on_money_path(node.key):
        return True
    if node.manifest is not None and (
        money_path_writes(node.manifest) or node.manifest.get("money_path_link") is not None
    ):
        return True
    return any(_crosses_money_path(parent) for parent in node.parents)


def _walk(
    key: str,
    manifest: dict[str, Any] | None,
    by_output: dict[str, dict],
    collisions: dict[str, tuple[str, ...]],
    *,
    depth: int,
    seen: set[str],
) -> Lineage:
    node = Lineage(key=key, manifest=manifest, depth=depth, collisions=collisions.get(key, ()))
    if manifest is None or depth >= MAX_DEPTH:
        return node
    for entry in manifest.get("inputs") or []:
        input_key = entry["key"]
        if input_key in seen:
            continue
        seen.add(input_key)
        node.parents.append(
            _walk(
                input_key,
                by_output.get(input_key),
                by_output,
                collisions,
                depth=depth + 1,
                seen=seen,
            )
        )
    return node


def render(node: Lineage) -> str:
    """The chain, as an operator reads it. One line per hop, indented by depth.

    The money-path verdict goes FIRST when the walk crosses the money path —
    ahead of the lineage, not appended under it. A break printed below forty
    lines of hops is a warning wearing a finding's clothes, and the whole
    point of `alpha-engine-config-I10414` is that a tampered history is the
    first thing the reader learns, not the last.
    """
    lines: list[str] = []
    if node.chain is not None:
        if node.chain.ok:
            lines.append(f"MONEY-PATH CHAIN: ok — {node.chain.reason}")
        else:
            lines.append(f"MONEY-PATH CHAIN: FAILED — {node.chain.reason}")
        lines.append("")

    # Never elided, in the same spirit as `produced by: UNKNOWN` below — a
    # zero is printed rather than the line being omitted when there is
    # nothing to report (`alpha-engine-config-I10626`).
    lines.append(f"{len(node.unreadable)} manifest(s) in this store could not be validated.")
    for key, reason in sorted(node.unreadable.items()):
        lines.append(f"  - {key}: {reason}")
    lines.append("")

    def emit(current: Lineage) -> None:
        pad = "  " * current.depth
        run = current.manifest
        if run is None:
            lines.append(
                f"{pad}- {current.key}\n{pad}    produced by: UNKNOWN — no run in this "
                "store claims this key as an output"
            )
        else:
            lines.append(
                f"{pad}- {current.key}\n"
                f"{pad}    run {run['run_id']}  job={run['job']}  "
                f"trading_day={run['trading_day']}  status={run['status']}"
                + (f"  reason={run['reason']}" if run["reason"] else "")
                + f"\n{pad}    code_sha={run['code_sha'][:12]}  release_sha="
                f"{run['release_sha'][:12]}  seed={run['seed']}  "
                f"cost_usd={run['cost_usd']}  llm_calls={len(run.get('llm_calls') or [])}"
            )
            if current.collisions:
                lines.append(
                    f"{pad}    ALSO CLAIMED BY: {', '.join(current.collisions)} — "
                    "this key was written by more than one run; the run above is only "
                    "the LATEST by `finished`, and the others are not shown in this walk"
                )
        for parent in current.parents:
            emit(parent)

    emit(node)
    return "\n".join(lines)
