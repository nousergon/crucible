"""`crucible migrate.history` — import v1 champion lineage into the v2 layout.

Normative source: plan §9.8 row "Migration of history"; §4.4's closing
paragraph.

Without this the OOS clock and the trial ledger start at zero on cutover
day, and every arm looks newly registered — which would make the R
champion's provenance disappear at exactly the moment it matters. R's
champion has been `scanner_predictor_direct` since 2026-07-13 under
`promotion_source: operator_bootstrap`; **that flag is carried across**, so
the console can render "a pointer that has never moved on evidence" as the
finding it is, and the first evidence-won promotion is visible as such.

**Read-only against every v1 prefix.** Nothing here writes to a v1 key. The
old system is still running, and a migration that mutated its state would
be a cutover disguised as an import.

**Absence is reported by key, never fabricated.** Each source is declared in
:data:`SOURCES` with the key it reads and what it contributes; a source that
is not there produces a row saying so, naming the exact key, and the run
FAILS unless `--allow-missing` was passed. A migration that quietly imported
three of five sources would seed the OOS clock from a subset and nothing
would say which.

**One-shot and idempotent.** Registering an arm that is already in the
register appends nothing (`crucible.slots.arms.register_arms`), so a rerun
after fixing one absent source imports the rest without duplicating what
already landed.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from crucible.calendar import resolve_trading_day
from crucible.documents import load_document_bytes, load_store_document, read_manifests_under
from crucible.keys import RUNS_ROOT, champion_key, migration_key
from crucible.manifest import money_path_writes, validate
from crucible.release import release_json_key
from crucible.slots.arms import ArmSpec, read_register, register_arms, write_register
from crucible.store import PointerConflictError, sha256_hex

if TYPE_CHECKING:
    from crucible.store import Store

#: `alpha-engine-config-I10506`/`crucible.promote._PLACEHOLDER_CODE_SHA`:
#: same value, same reasoning — kept local rather than imported from
#: `crucible.models` (a private name there) so this module carries no
#: import-time dependency on it.
_PLACEHOLDER_CODE_SHA = "0" * 40

__all__ = [
    "SOURCES",
    "CodeShaMigrationReport",
    "MigrationSourceMissing",
    "V1Source",
    "read_v1_json",
    "run_migrate_code_sha",
    "run_migrate_history",
]


class MigrationSourceMissing(RuntimeError):
    """A declared v1 source is absent. Its exact key is in the message."""


@dataclass(frozen=True)
class V1Source:
    """One v1 artifact the migration reads, and what it contributes."""

    name: str
    key: str
    slot: str | None
    contributes: str

    def series_prefix(self) -> str:
        """The v1 listing prefix for a dated series: everything before the
        `{date}` placeholder in :attr:`key`. A v1 shape, owned by this
        migration and not by `crucible.keys` (which is the v2 grammar) —
        registered as such in `tests/test_key_construction_placement.py`."""
        if "{date}" not in self.key:
            raise ValueError(f"{self.name}: {self.key!r} is not a dated series")
        return self.key.split("{date}")[0]


#: The exhaustive source list, with the literal v1 keys as they appear in the
#: v1 code. They are literals HERE, in the migration, on purpose: a migration
#: reads a system that no longer changes, and pointing it at a configurable
#: prefix would let a rerun read somewhere else and report the same success.
SOURCES: tuple[V1Source, ...] = (
    V1Source(
        name="producer_champion",
        key="config/producer_champion.json",
        slot="r",
        contributes=(
            "R's champion pointer and its promotion_source — `operator_bootstrap` "
            "since 2026-07-13, which is the flag that makes a never-moved pointer "
            "render as a finding rather than as a settled result"
        ),
    ),
    V1Source(
        name="scanner_spec_champion",
        key="config/scanner_spec_champion.json",
        slot="u",
        contributes="U's champion pointer and the date it was last written.",
    ),
    V1Source(
        name="producer_leaderboard",
        key="research/producer_leaderboard/{date}.json",
        slot="r",
        contributes=(
            "R's per-date leaderboard history: the registration dates that start each "
            "arm's OOS clock, and the trial rows that seed the ledger's n_trials."
        ),
    ),
    V1Source(
        name="model_zoo_promotions",
        key="predictor/model_zoo/promotions/{date}.json",
        slot="m",
        contributes=(
            "M's promotion markers, including promoted_kind. Imported as lineage only "
            "— the M slot's arms arrive with track B, and an imported pointer with no "
            "arms to point at would be a champion with no register row."
        ),
    ),
)


def read_v1_json(store: Store, key: str) -> dict[str, Any] | None:
    """Read one v1 JSON artifact, or ``None`` when it is absent. Never raises on 404."""
    if not store.exists(key):
        return None
    # STRICT face of the one reader: a present v1 artifact that is not an
    # object stops the migration with the key named (rule 5).
    return load_store_document(store, key)


def _bootstrap_spec(
    *,
    slot: str,
    name: str,
    registered_at: str,
    promotion_source: str,
    ranker: str,
    params: dict[str, Any],
) -> ArmSpec:
    """A v1 champion as a v2 arm recipe, flagged with where it came from.

    `bootstrap=True` is the library's own field and is what
    `arena.arms.ArmRecord` carries into every artifact downstream. It is not
    a note: an operator-installed champion and an evidence-won one must not
    render alike, and `promoted_kind: champion-arch-refresh` — v1's way of
    blurring the two — does not exist in v2.
    """
    return ArmSpec(
        name=name,
        slot=slot,
        ranker=ranker,
        params=params,
        registered_at=registered_at,
        bootstrap=True,
        promotion_source=promotion_source,
        notes=(
            f"Imported from v1 by `crucible migrate.history`. promotion_source="
            f"{promotion_source!r}; registered_at is the v1 pointer's own date, so the "
            "OOS clock starts where the arm actually started, not at cutover."
        ),
        source_key="crucible.migrate:v1",
    )


def run_migrate_history(
    ctx: Any,
    *,
    v1_store: Store,
    slots: tuple[str, ...] = ("u", "r"),
    arm_recipes: dict[str, ArmSpec] | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Import v1 lineage into the v2 register and ledger.

    ``arm_recipes`` maps a v1 champion NAME to the v2 recipe that reproduces
    it. It is required rather than inferred: v1 named a producer, v2 names a
    ranker plus its parameters, and guessing the mapping would silently
    attach a v1 track record to a rule that is not the one that earned it.
    """
    recipes = arm_recipes or {}
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for source in SOURCES:
        if "{date}" in source.key:
            # A dated series: the migration takes whatever dates exist rather
            # than asserting a range, because a range asserted against a
            # system that has stopped writing is a range that will never fill.
            prefix = source.series_prefix()
            keys = sorted(k for k in v1_store.list_keys(prefix) if k.endswith(".json"))
            if not keys:
                missing.append(
                    {
                        "source": source.name,
                        "key": source.key,
                        "reason": "no dated artifacts under the prefix",
                    }
                )
                continue
            found.append({"source": source.name, "key": prefix, "n_artifacts": len(keys)})
            continue
        document = read_v1_json(v1_store, source.key)
        if document is None:
            missing.append({"source": source.name, "key": source.key, "reason": "absent"})
            continue
        found.append({"source": source.name, "key": source.key, "document": document})

    if missing and not allow_missing:
        detail = "\n".join(f"  - {m['source']}: {m['key']} ({m['reason']})" for m in missing)
        raise MigrationSourceMissing(
            f"{len(missing)} of {len(SOURCES)} declared v1 sources are absent:\n{detail}\n"
            "The migration seeds the OOS clock and the trial ledger; importing a subset "
            "would start some arms' clocks from real history and others' from cutover "
            "day, and nothing downstream would say which. Re-run with --allow-missing "
            "only once each absence above is understood and recorded."
        )

    imported: dict[str, list[str]] = {}
    for slot in slots:
        pointer = next(
            (
                f["document"]
                for f in found
                if f.get("document") is not None
                and next(s for s in SOURCES if s.name == f["source"]).slot == slot
                and "champion" in f["document"]
            ),
            None,
        )
        if pointer is None:
            imported[slot] = []
            continue
        champion_name = pointer["champion"]
        recipe = recipes.get(champion_name)
        if recipe is None:
            raise MigrationSourceMissing(
                f"v1 slot {slot!r} names champion {champion_name!r} and no v2 recipe was "
                "supplied for it. The mapping from a v1 producer name to a v2 ranker "
                "plus parameters is a judgement, not an inference: attaching a v1 track "
                "record to a rule that is not the one that earned it would launder "
                "provenance. Supply it in `arm_recipes`."
            )
        spec = _bootstrap_spec(
            slot=slot,
            name=recipe.name,
            registered_at=_date_of(pointer),
            promotion_source=pointer.get("promotion_source", "unknown"),
            ranker=recipe.ranker,
            params=recipe.params,
        )
        register = read_register(ctx.store, slot)
        register, _ = register_arms(register, [spec])
        write_register(ctx.store, slot, register)
        key = champion_key(slot)
        payload = json.dumps(
            {
                "schema_version": "champion.v1",
                "slot": slot,
                "champion": spec.arm_id,
                "promoted_at": pointer.get("promoted_at"),
                "promotion_source": pointer.get("promotion_source", "unknown"),
                "imported_from": next(s.key for s in SOURCES if s.slot == slot),
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        # §4.2's outputs contract and §10.8's lineage walk both need this
        # pointer, and `store.py` provides exactly one CAS primitive for the
        # thing this key is: a pointer more than one actor can write.
        # `ctx.record_output_cas` (not a bare `ctx.store.put_bytes`, and not
        # a bare `compare_and_swap` either) is the one call that both writes
        # conditionally AND enters this write into `outputs[]` — a bare PUT
        # here was last-writer-wins on the one pointer the store has a CAS
        # primitive for, and it never entered the manifest's lineage, so
        # `crucible explain champions/{slot}/current.json` raised `KeyError`
        # for every imported champion (alpha-engine-config-I9757 defect #7).
        # A migration is one-shot but not guaranteed single-attempt (the
        # runner retries a transient failure with a fresh context, and the
        # command itself is documented idempotent), so the expected version
        # is read fresh immediately before the swap rather than assumed
        # absent.
        ctx.record_output_cas(key, ctx.store.etag(key), payload, schema_version="champion.v1")
        imported[slot] = [spec.arm_id]

    result = {
        "schema_version": "migration.v1",
        "sources_found": [{k: v for k, v in f.items() if k != "document"} for f in found],
        "sources_missing": missing,
        "arms_imported": imported,
        "allow_missing": allow_missing,
    }
    ctx.record_metric(
        {
            "name": "arms_migrated",
            "module": "crucible.migrate",
            "metric_type": "count",
            "value": float(sum(len(v) for v in imported.values())),
            "unit": "arms",
            "n_floor": 0,
            # §2 row 4: "No skip flags. No fail-open. No degraded-SUCCEEDED."
            # `DEGRADED_BY_OPERATOR_CONSENT` was a third state spelled at the
            # metric level, inside a manifest whose own `status` said `ok` —
            # the exact shape the top-level status enum exists to forbid,
            # reintroduced one level down (alpha-engine-config-I9757 defect
            # #3). `FAIL` is the honest word: the migration did not import
            # everything it declared as SOURCES, `--allow-missing` is why the
            # RUN still succeeded rather than raising, and the operator's
            # consent to that gap is recorded in `status_reason` below, in
            # prose, where a free-text explanation belongs — not manufactured
            # as a status token this schema now has to know about.
            "status": "OK" if not missing else "FAIL",
            "status_reason": (
                f"{len(found)} of {len(SOURCES)} v1 sources read; "
                f"{sum(len(v) for v in imported.values())} arm(s) imported with their v1 "
                "registration dates and promotion_source"
                + (f"; ABSENT: {[m['key'] for m in missing]}" if missing else "")
            ),
            "source_path": "arms/*/register.jsonl",
            "last_updated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    ctx.record_output(
        migration_key(ctx.trading_day.isoformat(), ctx.run_id),
        json.dumps(result, indent=2, sort_keys=True).encode("utf-8"),
        schema_version="migration.v1",
    )
    return result


def _date_of(pointer: dict[str, Any]) -> str:
    """The v1 pointer's own promotion date, as an ISO date.

    Taken from the artifact rather than from today: the OOS clock starts
    when the arm actually started, and starting it at cutover would give
    every imported arm a fresh eligibility window it has not earned.
    """
    raw = pointer.get("promoted_at")
    if not raw:
        raise MigrationSourceMissing(
            "the v1 champion pointer carries no `promoted_at`, so there is no date to "
            "start its OOS clock from. Defaulting to today would reset a clock that has "
            "been running since 2026-07-13."
        )
    return str(raw)[:10]


# ── `crucible migrate.code_sha` (alpha-engine-config-I10626) ────────────────
#
# 89 of 164 production run manifests carry the all-zero `code_sha`
# placeholder, written before `crucible.runner.resolve_code_sha` and the
# schema-level refusal (alpha-engine-config-I10454) existed. No producer has
# written it since crucible-PR219 (2026-09-11), but `crucible.manifest.validate`
# validates ON READ, so `crucible explain` cannot complete a lineage walk
# against the store as it stands today.
#
# This is a one-off REPAIR, not a job: it patches documents another job
# already wrote, so it cannot honestly run through `crucible.runner.run_job`
# (whose manifest schema's `job` enum is closed, `crucible/models.py`, and is
# out of this change's ownership) and does not claim to. What it does instead:
#
# 1. Reads every manifest under `runs/` (`crucible.documents.read_manifests_under`,
#    the same tolerant reader `crucible.explain` will eventually use for I10626's
#    other half — never `crucible.manifest.validate`'s STRICT face, which is
#    exactly what raises on these 89 today).
# 2. For each manifest whose `code_sha` is the placeholder, DERIVES the real
#    value from the durable record of what `releases/current` pointed at when
#    the manifest's own `started` instant was reached: the most recent
#    successful `deploy` manifest at or before that instant
#    (`runs/deploy/{trading_day}/run.json`, `status: ok`, written directly by
#    `crucible.deploy._record` — see its own `code_sha`/`release_sha` fields).
#    Cross-checked against every OTHER manifest sharing the same `trading_day`:
#    they all read `releases/current` at close to the same time, so a
#    conflicting `code_sha` among them is treated as ambiguity, not resolved
#    by picking one.
# 3. REFUSES, naming the manifest and the reason, rather than guessing, when:
#    the manifest is on the money path (`crucible.manifest.money_path_writes`
#    / `money_path_link`) — a fabricated `code_sha` there is exactly the
#    false-provenance claim I10454 exists to prevent; the manifest is itself a
#    `deploy` record (its `code_sha` names the release BEING deployed, not one
#    read from the pointer, so this derivation does not apply); no successful
#    deploy record exists at or before the manifest's `started` instant; same-
#    trading-day peers disagree on `code_sha`; or the derived sha has no
#    `releases/{sha}/release.json` in the store to verify it against.
# 4. Rewrites ONLY `code_sha` on a refused-nowhere-else manifest, via
#    `Store.compare_and_swap` (never a bare PUT — a manifest is a "single
#    object more than one actor could touch" the moment a repair tool exists
#    for it), re-validated whole against `RunManifestV2` before the write. The
#    rewrite is recorded ON the manifest itself: an `inputs[]` entry naming the
#    deploy manifest the sha was read from (so the existing `inputs`/`outputs`
#    lineage a future `explain` walk already understands gains one more real
#    edge), and a `metrics[]` row naming the placeholder it replaced, the
#    source and this migration's own run id — `MetricRecordRow` is the one
#    open (`extra="allow"`) shape on the manifest, by design, for exactly this
#    kind of forward-compatible annotation.
# 5. Writes ONE summary document per attempt at `migrations/{trading_day}/{run_id}.json`
#    (`crucible.keys.migration_key`, the same shape `run_migrate_history`
#    already files under) — every key this run touched or refused, and why,
#    so the repair itself is explainable without re-deriving it from prose.
#
# **One-shot and idempotent.** A manifest whose `code_sha` is already real
# (rewritten by a prior attempt, or never broken) is not a candidate at all —
# re-running finds nothing left to do for it. A refusal is not remembered
# across runs: a later attempt, with more deploy history available, may
# resolve it, and CI does not need this module to carry state to say so.


@dataclass(frozen=True)
class CodeShaMigrationReport:
    """What one `crucible migrate.code_sha` attempt did, in full.

    ``rewritten`` and ``refused`` are exhaustive: every manifest this run
    found carrying the all-zero `code_sha` placeholder appears in exactly one
    of the two, named by its store key, never silently dropped
    (`alpha-engine-config-I10626`).
    """

    migration_run_id: str
    dry_run: bool
    rewritten: tuple[dict[str, Any], ...] = ()
    refused: tuple[dict[str, Any], ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {"rewritten": len(self.rewritten), "refused": len(self.refused)}

    def summary_line(self) -> str:
        suffix = " (dry run — nothing written)" if self.dry_run else ""
        return f"{len(self.rewritten)} rewritten, {len(self.refused)} refused{suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "migrate_code_sha.v1",
            "migration_run_id": self.migration_run_id,
            "dry_run": self.dry_run,
            "rewritten": list(self.rewritten),
            "refused": list(self.refused),
            "counts": self.counts,
        }


def _parse_utc_instant(raw: Any) -> dt.datetime | None:
    """``raw`` as a UTC instant, or ``None`` if it is not one.

    Never raises: every call site here is deciding whether to TRUST a
    timestamp read off a manifest nothing has validated yet, and an
    unparseable one is a reason to refuse that manifest, not to crash the
    whole migration attempt.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _successful_deploy_events(
    documents: list[tuple[str, dict[str, Any]]],
) -> list[tuple[dt.datetime, str, str]]:
    """Every successful `deploy` manifest as ``(finished_at, code_sha, key)``,
    oldest first.

    `finished`, not `started`: `crucible.deploy._record` writes both close to
    the same instant, but `finished` is when the flip it is REPORTING was
    already observed (its own `promoted == args.sha` check), so it is the
    later and more conservative bound for "the pointer had moved by this
    instant".
    """
    events: list[tuple[dt.datetime, str, str]] = []
    for key, document in documents:
        if document.get("job") != "deploy" or document.get("status") != "ok":
            continue
        sha = document.get("code_sha")
        if not isinstance(sha, str) or sha == _PLACEHOLDER_CODE_SHA:
            continue
        instant = _parse_utc_instant(document.get("finished"))
        if instant is None:
            continue
        events.append((instant, sha, key))
    events.sort(key=lambda event: event[0])
    return events


def _conflicting_same_day_peers(
    documents: list[tuple[str, dict[str, Any]]],
    *,
    trading_day: Any,
    exclude_key: str,
    candidate_sha: str,
) -> dict[str, list[str]]:
    """Every OTHER `code_sha` a same-`trading_day` peer manifest carries,
    mapped to the keys that carry it — empty when every peer agrees with
    ``candidate_sha`` (or there are no peers with a real value to compare).

    `deploy` manifests are excluded: they are the source of `candidate_sha`
    in the common case, and comparing a candidate against itself proves
    nothing.
    """
    conflicts: dict[str, list[str]] = {}
    for key, document in documents:
        if key == exclude_key or document.get("job") == "deploy":
            continue
        if document.get("trading_day") != trading_day:
            continue
        sha = document.get("code_sha")
        if not isinstance(sha, str) or sha == _PLACEHOLDER_CODE_SHA or sha == candidate_sha:
            continue
        conflicts.setdefault(sha, []).append(key)
    return conflicts


def _derive_code_sha(
    key: str,
    document: dict[str, Any],
    *,
    store: Store,
    deploy_events: list[tuple[dt.datetime, str, str]],
    documents: list[tuple[str, dict[str, Any]]],
) -> tuple[str, str] | tuple[None, str]:
    """The real `code_sha` for ``document`` and the key it was derived from,
    or ``(None, reason)``.

    Returns ``(sha, source_key)`` on success — ``source_key`` is the deploy
    manifest's own store key, returned directly (never through a dict) so
    every caller binds it as a plain tuple-unpack name, the shape
    `tests/test_no_inline_store_keys.py` accepts for a key that already came
    from a real `crucible.keys` call elsewhere. ``(None, reason)`` on a
    refusal, with ``reason`` naming exactly why.
    """
    if document.get("money_path_link") is not None or money_path_writes(document):
        # A fabricated code_sha here is the exact false-provenance claim
        # alpha-engine-config-I10454 refuses.
        return (
            None,
            "on the money path (money_path_link/outputs); a fabricated code_sha there is "
            "refused rather than guessed.",
        )
    if document.get("job") == "deploy":
        return (
            None,
            "job=deploy: its code_sha names the release BEING deployed, not one read from "
            "releases/current — this migration's derivation does not apply to it.",
        )
    started = _parse_utc_instant(document.get("started"))
    if started is None:
        return (
            None,
            f"no readable `started` UTC instant (got {document.get('started')!r}) to derive "
            "a release-as-of time from.",
        )
    candidates = [event for event in deploy_events if event[0] <= started]
    if not candidates:
        return (
            None,
            f"no successful deploy manifest is recorded at or before "
            f"started={started.isoformat()}; there is no durable record of which release "
            "was current.",
        )
    _instant, sha, source_key = candidates[-1]
    conflicts = _conflicting_same_day_peers(
        documents,
        trading_day=document.get("trading_day"),
        exclude_key=key,
        candidate_sha=sha,
    )
    if conflicts:
        named = ", ".join(f"{s} ({', '.join(keys)})" for s, keys in sorted(conflicts.items()))
        return (
            None,
            f"ambiguous: same-trading_day peers disagree with the derived code_sha {sha} "
            f"(from {source_key}): {named}.",
        )
    if not store.exists(release_json_key(sha)):
        return (
            None,
            f"derived code_sha {sha} (from {source_key}) has no releases/{sha}/release.json "
            "in the store; refusing to write an unverifiable sha.",
        )
    return sha, source_key


def run_migrate_code_sha(store: Store, *, dry_run: bool = False) -> CodeShaMigrationReport:
    """Derive and rewrite the all-zero `code_sha` placeholder, store-wide.

    See the module-level comment above for the full derivation and refusal
    rules. Safe to re-run: a manifest already carrying a real `code_sha` is
    not a candidate, so a rerun after fixing one absence in the deploy
    history only touches what is still broken.
    """
    now = dt.datetime.now(dt.UTC)
    migration_run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    listing = read_manifests_under(store, RUNS_ROOT)
    listing.raise_if_unlistable()
    documents = list(listing.documents)
    deploy_events = _successful_deploy_events(documents)

    rewritten: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []

    for key, document in documents:
        if document.get("code_sha") != _PLACEHOLDER_CODE_SHA:
            continue
        # `_derive_code_sha` returns `(sha, source_key)` on success or
        # `(None, reason)` on a refusal — one name, two meanings depending on
        # which branch below reads it, so it stays unlabeled here rather than
        # claiming a single purpose it does not have.
        sha, source_key_or_reason = _derive_code_sha(
            key, document, store=store, deploy_events=deploy_events, documents=documents
        )
        if sha is None:
            refused.append({"key": key, "reason": source_key_or_reason})
            continue
        source_key = source_key_or_reason
        if dry_run:
            rewritten.append(
                {
                    "key": key,
                    "old_code_sha": _PLACEHOLDER_CODE_SHA,
                    "new_code_sha": sha,
                    "source_key": source_key,
                    "dry_run": True,
                }
            )
            continue

        expected_version = store.etag(key)
        current_bytes = store.get_bytes(key)
        current_document = load_document_bytes(key, current_bytes)
        if current_document.get("code_sha") != _PLACEHOLDER_CODE_SHA:
            # Raced with (or already fixed by) another attempt between the
            # listing above and this write — idempotent, not an error: the
            # manifest is no longer a candidate, so this run simply reports
            # it as one it did not need to touch.
            refused.append(
                {
                    "key": key,
                    "reason": "code_sha changed since listing (another writer already fixed "
                    "it); nothing left for this attempt to do.",
                }
            )
            continue
        source_bytes = store.get_bytes(source_key)
        new_document = dict(current_document)
        new_document["code_sha"] = sha
        new_document["inputs"] = [
            *current_document.get("inputs", []),
            {
                "key": source_key,
                "sha256": sha256_hex(source_bytes),
                "schema_version": "run_manifest.v2",
            },
        ]
        new_document["metrics"] = [
            *current_document.get("metrics", []),
            {
                "name": "code_sha_migrated",
                "module": "crucible.migrate",
                "metric_type": "provenance",
                "n_floor": 0,
                "status": "OK",
                # Tracker: alpha-engine-config-I10454 (the schema-level
                # refusal this repairs the fallout of), -I10626 (this
                # migration's own issue) — cited here rather than in the
                # runtime string below, per test_no_stale_tracker_literals.py.
                "status_reason": (
                    f"code_sha derived from {source_key} (release {sha}); the original "
                    "manifest carried the all-zero placeholder."
                ),
                "source_path": source_key,
                "last_updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                # Extra, forward-compat fields (MetricRecordRow is
                # extra="allow" — see the module-level comment, point 4):
                # structured provenance a future reader can key off without
                # parsing status_reason's prose.
                "migrated_from_code_sha": _PLACEHOLDER_CODE_SHA,
                "migration_run_id": migration_run_id,
            },
        ]
        validate(new_document)
        payload = json.dumps(new_document, indent=2, sort_keys=True).encode("utf-8")
        try:
            store.compare_and_swap(key, expected_version, payload)
        except PointerConflictError as exc:
            refused.append(
                {
                    "key": key,
                    "reason": f"concurrent write detected (compare-and-swap conflict): {exc}. "
                    "Re-run this migration to re-evaluate.",
                }
            )
            continue
        rewritten.append(
            {
                "key": key,
                "old_code_sha": _PLACEHOLDER_CODE_SHA,
                "new_code_sha": sha,
                "source_key": source_key,
            }
        )

    report = CodeShaMigrationReport(
        migration_run_id=migration_run_id,
        dry_run=dry_run,
        rewritten=tuple(rewritten),
        refused=tuple(refused),
    )
    if not dry_run:
        trading_day = resolve_trading_day(now)
        store.put_bytes(
            migration_key(trading_day.isoformat(), migration_run_id),
            json.dumps(report.as_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
    return report
