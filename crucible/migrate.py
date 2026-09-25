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

import dataclasses
import datetime as dt
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crucible.calendar import resolve_trading_day
from crucible.carryover import V1_ZOO_CHAMPION_NAME, V1_ZOO_LEADERBOARD_KEY, read_production
from crucible.champion import PROMOTION_SOURCES, ChampionPointer
from crucible.documents import load_document_bytes, load_store_document, read_manifests_under
from crucible.keys import RUNS_ROOT, arm_register_key, champion_key, migration_key
from crucible.keys import manifest_key as run_manifest_key
from crucible.manifest import money_path_writes, validate
from crucible.release import release_json_key
from crucible.runner import resolve_code_sha
from crucible.slots.arms import ArmSpec, read_register, register_arms, write_register
from crucible.slots.producibility import UnproducibleChampionError, catalog_refusal
from crucible.store import ETAG_ABSENT, PointerConflictError, sha256_hex

if TYPE_CHECKING:
    from crucible.store import Store

#: `alpha-engine-config-I10506`/`crucible.promote._PLACEHOLDER_CODE_SHA`:
#: same value, same reasoning — kept local rather than imported from
#: `crucible.models` (a private name there) so this module carries no
#: import-time dependency on it.
_PLACEHOLDER_CODE_SHA = "0" * 40

__all__ = [
    "ARM_FILING_CORRECTIONS",
    "MIGRATABLE_SLOTS",
    "SOURCES",
    "V1_SERVED_MODEL",
    "ArmFilingCorrection",
    "ArmFilingMigrationReport",
    "CodeShaMigrationReport",
    "MigrationPointerConflict",
    "MigrationSourceMissing",
    "V1ModelTree",
    "V1Source",
    "admission_refusal",
    "read_v1_json",
    "run_migrate_arm_filed_on",
    "run_migrate_code_sha",
    "run_migrate_history",
    "served_model_recipe",
]


class MigrationSourceMissing(RuntimeError):
    """A declared v1 source is absent. Its exact key is in the message."""


class MigrationPointerConflict(RuntimeError):
    """`champions/{slot}/current.json` already holds a pointer this migration did
    not write, or one naming a different arm.

    A migration seeds a pointer that does not exist yet. Overwriting one that
    does would erase a promotion (evidence-won or an operator revert) with a v1
    import, which is the opposite of carrying provenance across.
    """


@dataclass(frozen=True)
class V1Source:
    """One v1 artifact the migration reads, and what it contributes."""

    name: str
    key: str
    slot: str | None
    contributes: str
    #: How this artifact names the v1 CHAMPION, or ``None`` when it carries no
    #: pointer at all (a dated history series). Declared per source rather
    #: than sniffed with `"champion" in document`: the zoo leaderboard HAS a
    #: `champion` key and it is a metrics block, not a name, so the sniff
    #: would have read `{"forward_days": 21, ...}` as an arm the moment M was
    #: admitted (measured 2026-09-17, `alpha-engine-config-I10961`).
    champion: Callable[[dict[str, Any]], str | None] | None = None
    #: The field of a champion-pointer document that carries the date the
    #: v1 champion was installed, or ``None`` when the v1 document carries no
    #: such date at all. Declared per source rather than assumed to be
    #: `promoted_at` everywhere: measured 2026-09-14, the U pointer
    #: (`config/scanner_spec_champion.json`) has `decided_on` (the date of the
    #: latest HOLD, which is not an installation date) and
    #: `last_promoted_on: null`, and no `promoted_at` — so reading
    #: `promoted_at` there failed every real U import.
    date_field: str | None = None
    #: How the champion NAME above resolves to a v2 recipe. ``"name"`` (U, R):
    #: v1 and v2 spell the arm the same way, and ``arm_recipes`` is keyed by
    #: recipe name. :data:`V1_SERVED_MODEL` (M): v1 names a served MODEL
    #: VERSION, no v2 recipe carries that name, and the recipe is the one
    #: :func:`served_model_recipe` identifies from the recipes' own
    #: `supersedes_v1` declarations (`alpha-engine-config-I10961`
    #: deliverable 2).
    resolves_by: str = "name"

    def series_prefix(self) -> str:
        """The v1 listing prefix for a dated series: everything before the
        `{date}` placeholder in :attr:`key`. A v1 shape, owned by this
        migration and not by `crucible.keys` (which is the v2 grammar) —
        registered as such in `tests/test_key_construction_placement.py`."""
        if "{date}" not in self.key:
            raise ValueError(f"{self.name}: {self.key!r} is not a dated series")
        return self.key.split("{date}")[0]


def _pointer_champion(document: dict[str, Any]) -> str | None:
    """A v1 champion POINTER document names its champion in `champion`."""
    champion = document.get("champion")
    if champion is None:
        return None
    if not isinstance(champion, str) or not champion:
        raise MigrationSourceMissing(
            f"a v1 champion pointer names `champion` as {champion!r}, which is not an arm "
            "name. Importing it would point a v2 slot at something that is not an arm."
        )
    return champion


def _zoo_champion_arch(document: dict[str, Any]) -> str | None:
    """v1's M champion, which is not a pointer document but a FIELD of the zoo
    leaderboard (`alpha-engine-config-I10961` deliverable 2).

    Returns `serving_champion.served_version` — the exact model v1 SERVES —
    and never `champion_arch.version_id`. The two differ, and measured
    2026-09-25 they did: `champion_arch` is the latest REFRESH of the serving
    architecture (`v3.0-meta-2026-09-23-d7f8f864`, a promotion baseline that
    was never served), while `served_version` is what the predictor loads
    (`v3.0-meta-2026-08-14-119e069b`). v1's lineage is the served one, and the
    served version is also the only identity a v2 recipe declares
    (`supersedes_v1`), so it is the name :func:`served_model_recipe` resolves.
    A v1 that re-serves a new version therefore DEFERS the import rather than
    attaching the new model's track record to a port of the old one.

    `champion_arch: null` is v1 declaring no serving architecture: no
    champion, reported as such by the caller. The leaderboard's own
    `champion` key is a metrics block and is deliberately not read here.
    """
    if document.get("champion_arch") is None:
        return None
    serving = document.get("serving_champion")
    served = serving.get("served_version") if isinstance(serving, dict) else None
    if not isinstance(served, str) or not served:
        raise MigrationSourceMissing(
            f"the v1 zoo leaderboard names its serving architecture "
            f"({V1_ZOO_CHAMPION_NAME}) and no `serving_champion.served_version`, so the "
            "model v1 actually serves cannot be named and its lineage cannot be carried."
        )
    return served


#: :attr:`V1Source.resolves_by` for a source whose champion is a served model
#: version rather than an arm name.
V1_SERVED_MODEL = "supersedes_v1"


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
        date_field="promoted_at",
        champion=_pointer_champion,
    ),
    V1Source(
        name="scanner_spec_champion",
        key="config/scanner_spec_champion.json",
        slot="u",
        contributes=(
            "U's champion pointer. It carries no installation date (its `decided_on` "
            "is the latest hold), so the imported arm's clock starts at the published "
            "recipe's own `registered_at` — the date the v2 register already holds."
        ),
        date_field=None,
        champion=_pointer_champion,
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
            "M's promotion markers, including promoted_kind. Lineage only: M's champion "
            "is read from the zoo leaderboard below, never from a promotion marker."
        ),
    ),
    V1Source(
        name="model_zoo_leaderboard",
        key=V1_ZOO_LEADERBOARD_KEY,
        slot="m",
        contributes=(
            "M's champion: the serving architecture `champion_arch`, resolving to the model "
            "v1 actually serves (`serving_champion.served_version`). Not a standalone "
            "pointer like R's and U's — it is a field of the leaderboard, and it carries no "
            "installation date, so the imported arm's clock starts at the published recipe's "
            "own `registered_at`, exactly as U's does. The served version maps to the v2 "
            "recipe that reproduces it through the recipes' `supersedes_v1`."
        ),
        date_field=None,
        champion=_zoo_champion_arch,
        resolves_by=V1_SERVED_MODEL,
    ),
)

#: Every slot this migration CAN import a champion for: one declared champion
#: source each. Not a list of slots it WILL import — that is resolved per run
#: by :func:`admit_slot`, against the state of the v2 store.
MIGRATABLE_SLOTS: tuple[str, ...] = tuple(
    dict.fromkeys(source.slot for source in SOURCES if source.champion is not None and source.slot)
)


def read_v1_json(store: Store, key: str) -> dict[str, Any] | None:
    """Read one v1 JSON artifact, or ``None`` when it is absent. Never raises on 404."""
    if not store.exists(key):
        return None
    # STRICT face of the one reader: a present v1 artifact that is not an
    # object stops the migration with the key named (rule 5).
    return load_store_document(store, key)


@dataclass(frozen=True)
class V1ModelTree:
    """The M slot's recipe tree, in the two halves :func:`served_model_recipe`
    needs (`alpha-engine-config-I10961` deliverable 2).

    ``declared`` is every recipe FILED — `crucible.slots.model.ModelRecipe`,
    registrable or not — because which recipe is the served MODEL and which
    are its legs is read off the `predictions[...]` edges between them, and a
    refused stacker must stay in that graph: dropping it would leave a leg
    looking like the whole model. ``registrable`` is what the release
    registers (`crucible.slots.model.RegisteredModelArm`, the wrapper
    `register_arms` folds), and ``refused`` names every recipe it does not,
    with the reason.
    """

    declared: tuple[Any, ...] = ()
    registrable: tuple[Any, ...] = ()
    refused: Mapping[str, str] = field(default_factory=dict)


def served_model_recipe(
    tree: V1ModelTree, *, slot: str, served_version: str
) -> tuple[Any | None, str | None]:
    """The registrable v2 recipe that reproduces v1's served model, or why none.

    The adapter `alpha-engine-config-I10961` deliverable 2 asks for. v1 names
    its M champion by the model VERSION it serves; a v2 M recipe is a
    `ModelRecipe`, whose id hashes a different spec shape than
    :func:`_bootstrap_spec` builds, so neither a name lookup nor an `ArmSpec`
    can reach it. What connects the two is the recipe's own declaration:
    every v2 port of that model carries `supersedes_v1: <served_version>`.

    **Several recipes carry it, and only one of them is the model.** v1's
    served `v3.0-meta` is a stack — two Layer-1 heads and a Layer-2 combine —
    and each head is ported as its own arm because the stacker reads it as
    `predictions[<head>]`. All three descend from the served version; the
    MODEL is the one that no other descendant consumes. That is read from the
    declared input edges, never from a name, so it holds for a single-model
    port (one descendant, consumed by nothing) and for any depth of stack.

    Refuses rather than guesses — each refusal is a deferral reason on the
    scheduled path and a raise on the asserted one, exactly like every other
    refusal in :func:`run_migrate_history`:

    * no recipe declares the served version (v1 re-served, or the port was
      never filed);
    * more or fewer than one descendant is consumed by none of the others
      (two unrelated ports of one model is a judgement nobody has made);
    * the model's recipe is refused at registration, or does not register
      under the id it hashes to — a pointer at it would name an arm that can
      never write a shadow, which `crucible.slots.producibility` exists to
      stop (`alpha-engine-config-I11085`).
    """
    descendants = [recipe for recipe in tree.declared if recipe.supersedes_v1 == served_version]
    if not descendants:
        return None, (
            f"v1 slot {slot!r} serves {served_version!r} and no v2 recipe declares "
            f"`supersedes_v1: {served_version}`, so nothing in the strategy tree reproduces "
            "the served model. Mapping it onto some other recipe would attach v1's track "
            "record to a model that did not earn it."
        )
    names = {recipe.name for recipe in descendants}
    consumed = {
        ref.ref
        for recipe in descendants
        for ref in recipe.inputs
        if ref.kind == "predictions" and ref.ref in names
    }
    models = sorted(recipe.name for recipe in descendants if recipe.name not in consumed)
    if len(models) != 1:
        return None, (
            f"v1 slot {slot!r} serves {served_version!r}, and {len(models)} of the v2 "
            f"recipes declaring it ({sorted(names)}) are consumed by no other: "
            f"{models}. The served model is the one that stacks on the rest; with "
            "anything but exactly one such recipe, which of them reproduces it is a "
            "judgement the tree has not recorded."
        )
    model = next(recipe for recipe in descendants if recipe.name == models[0])
    if model.name in tree.refused:
        return None, (
            f"v1 slot {slot!r} serves {served_version!r}, reproduced by v2 recipe "
            f"{model.name!r}, which the release refuses at registration: "
            f"{tree.refused[model.name]} A pointer at it would name an arm that never "
            "writes a shadow."
        )
    registered = next(
        (spec for spec in tree.registrable if spec.arm_id == model.arm_id),
        None,
    )
    if registered is None:
        return None, (
            f"v1 slot {slot!r} serves {served_version!r}, reproduced by v2 recipe "
            f"{model.name!r} ({model.arm_id}), and the release registers no arm under that "
            "id, so a pointer at it would never be scored."
        )
    return registered, None


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
        notes=_bootstrap_notes(promotion_source),
        source_key="crucible.migrate:v1",
    )


def _bootstrap_notes(promotion_source: str) -> str:
    return (
        f"Imported from v1 by `crucible migrate.history`. promotion_source="
        f"{promotion_source!r}; registered_at is the v1 pointer's own date, so the "
        "OOS clock starts where the arm actually started, not at cutover."
    )


def admission_refusal(store: Store, *, slot: str, arm_id: str) -> str | None:
    """Why ``slot`` may NOT be seated from v1 yet, or ``None`` when it may.

    The deferral that left M out of the first import was a DEFAULT ARGUMENT —
    `slots=("u","r")` — with no trigger: it was correct on 2026-09-14, expired
    when M's arms landed on 2026-09-17, and nothing anywhere went red or ran
    (`alpha-engine-config-I10961`). A default cannot expire; a predicate can,
    so the deferral is one, evaluated against the v2 store on every run.

    The predicate is PRODUCTION, not registration. Registration is the weaker
    fact and the one that already misled every earlier surface
    (`alpha-engine-config-I10964`): `m:v3meta_stack` is registered and cannot
    emit on any date while `-I10947` is unruled, and seating the M slot on it
    would point the trader's whole contract at an arm that produces nothing.
    An arm that has produced has necessarily arrived; an arm that has not is
    refused whether or not it is in the register, and the refusal says which,
    because "the slot's arms have not arrived" and "the arm is mute" are
    different problems with different owners.

    An unreadable production listing refuses admission too — an unknown is
    never an admission.
    """
    production = read_production(store, arm_id)
    if production.produced is None:
        return (
            f"whether {arm_id} has produced could not be read ({production.problem}); an "
            "unknown is never an admission"
        )
    if production.produced:
        return None
    registered = arm_id in set(read_register(store, slot).all_arms())
    where = arm_register_key(slot)
    return (
        f"{arm_id} has produced nothing under {production.key}; it is "
        + (f"registered in {where}" if registered else f"not in {where}")
        + ". Seating the slot on it would point the trader's contract at an arm that "
        "emits nothing"
    )


def run_migrate_history(
    ctx: Any,
    *,
    v1_store: Store,
    slots: tuple[str, ...] | None = None,
    arm_recipes: dict[str, ArmSpec] | None = None,
    model_tree: V1ModelTree | None = None,
    allow_missing: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import v1 lineage into the v2 register and ledger.

    ``arm_recipes`` maps a v1 champion NAME to the v2 recipe that reproduces
    it. It is required rather than inferred: v1 named a producer, v2 names a
    ranker plus its parameters, and guessing the mapping would silently
    attach a v1 track record to a rule that is not the one that earned it.

    ``model_tree`` is the M slot's recipe tree. v1's M champion is a served
    model VERSION rather than an arm name, so it is resolved through
    :func:`served_model_recipe` over the recipes' `supersedes_v1`
    declarations instead of through ``arm_recipes`` — the same refusal
    discipline, a different key. ``None`` is a caller that supplied no M tree,
    and M then defers naming exactly that.

    ``dry_run`` resolves every source, recipe and existing pointer exactly as
    a real run does and reports what it WOULD write, but writes nothing: not
    the register, not a pointer, not the migration record.

    ``slots`` defaults to :data:`MIGRATABLE_SLOTS` — every slot with a declared
    v1 champion source — and each is admitted or DEFERRED by
    :func:`admission_refusal` against the live v2 store. A deferred slot is
    reported with its reason in ``deferred`` and never silently omitted, and
    the run still succeeds: a slot whose arms have not arrived is a by-design
    state, and a stage that raised on it would kill every later stage of the
    arc (`crucible-PR317` / `alpha-engine-config-I10927`).

    Passing ``slots`` explicitly ASSERTS those slots: a named slot with no
    champion source or no recipe raises rather than defers, because the caller
    said it was there. The admission predicate still applies to every slot —
    an explicit request cannot seat a pointer on an arm that emits nothing.

    **One rule, two renderings** (`alpha-engine-config-I10961` deliverable 4):
    on the ASSERTED path every refusal raises; on the scheduled path — which
    asserts nothing, because the arc dispatches this job with no `--slots`
    equivalent at all — EVERY refusal is a recorded `deferred` reason and the
    run exits `ok`. Exhaustively: a slot whose arm has not produced, a declared
    v1 source that is absent, a champion name with no v2 recipe supplied, a
    recipe belonging to another slot, a recipe whose id the bootstrap cannot
    reproduce, a recipe the release refuses at registration so the arm can
    never produce (`alpha-engine-config-I11085`, raised as
    :class:`~crucible.slots.producibility.UnproducibleChampionError`), and a
    pointer already held by another writer. An arc stage that
    raised on any of them would kill every stage after it, which is the defect
    `crucible-PR317` closed one stage earlier — and every one of those states
    is reachable on an ordinary Saturday: `promote` writes the same pointer an
    hour later, and v1 is being decommissioned underneath the sources.
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

    asserted = slots is not None
    # ASSERTED only (`alpha-engine-config-I10961` deliverable 4). A caller that
    # named its slots said the sources were there, and a partial import under
    # that claim is the provenance laundering this refusal exists to stop. The
    # SCHEDULED path names nothing: it is an arc stage now, and an arc stage
    # that raises on a by-design state kills every stage after it
    # (`crucible-PR317` / `-I10927`). v1 is being decommissioned, so a source
    # that has stopped existing is the expected end state, not a Saturday
    # failure — and the hazard the refusal names is already prevented per slot
    # below, where a slot whose own champion source is absent is DEFERRED with
    # `no_v1_pointer` rather than imported from a cutover-day clock. Every
    # absence stays in `sources_missing` and in the metric's `status_reason`.
    if missing and not allow_missing and asserted:
        detail = "\n".join(f"  - {m['source']}: {m['key']} ({m['reason']})" for m in missing)
        raise MigrationSourceMissing(
            f"{len(missing)} of {len(SOURCES)} declared v1 sources are absent:\n{detail}\n"
            "The migration seeds the OOS clock and the trial ledger; importing a subset "
            "would start some arms' clocks from real history and others' from cutover "
            "day, and nothing downstream would say which. Re-run with --allow-missing "
            "only once each absence above is understood and recorded."
        )

    imported: dict[str, list[str]] = {}
    pointers: dict[str, str] = {}
    deferred: dict[str, str] = {}
    planned: list[tuple[Any, ...]] = []
    documents_by_source = {
        f["source"]: f["document"] for f in found if f.get("document") is not None
    }
    for slot in slots if slots is not None else MIGRATABLE_SLOTS:
        source = next((s for s in SOURCES if s.slot == slot and s.champion is not None), None)
        pointer = documents_by_source.get(source.name) if source is not None else None
        champion_name = (
            source.champion(pointer) if source is not None and pointer is not None else None
        )
        if champion_name is None:
            why = (
                f"no v1 champion source is declared for slot {slot!r}"
                if source is None
                else f"{source.key} is absent or declares no champion"
            )
            if asserted:
                raise MigrationSourceMissing(
                    f"v1 slot {slot!r} was named explicitly and {why}. A named slot is an "
                    "assertion that its champion is there; reporting it as deferred would "
                    "turn a caller's mistake into a quiet no-op."
                )
            imported[slot] = []
            pointers[slot] = "no_v1_pointer"
            deferred[slot] = why
            continue
        if source.resolves_by == V1_SERVED_MODEL:
            if model_tree is None:
                recipe = None
                why = (
                    f"v1 slot {slot!r} serves {champion_name!r} and no v2 {slot.upper()} "
                    "recipe tree was supplied to resolve it against. Supply it in "
                    "`model_tree`."
                )
            else:
                recipe, why = served_model_recipe(
                    model_tree, slot=slot, served_version=champion_name
                )
        else:
            recipe = recipes.get(champion_name)
            why = (
                f"v1 slot {slot!r} names champion {champion_name!r} and no v2 recipe was "
                "supplied for it. The mapping from a v1 producer name to a v2 ranker "
                "plus parameters is a judgement, not an inference: attaching a v1 track "
                "record to a rule that is not the one that earned it would launder "
                "provenance. Supply it in `arm_recipes`."
            )
        if recipe is None:
            if asserted:
                raise MigrationSourceMissing(why)
            imported[slot] = []
            pointers[slot] = "deferred"
            deferred[slot] = why
            continue
        if recipe.slot != slot:
            why = (
                f"v1 slot {slot!r} names champion {champion_name!r}, and the recipe "
                f"supplied under that name is a slot-{recipe.slot!r} arm. Importing it "
                f"would register `{slot}:{champion_name}` - an arm no published recipe "
                "defines - and point the slot at it, which would launder provenance."
            )
            if asserted:
                raise MigrationSourceMissing(why)
            imported[slot] = []
            pointers[slot] = "deferred"
            deferred[slot] = why
            continue
        v1_promotion_source = pointer.get("promotion_source", "unknown")
        registered_at, date_source = _date_of(pointer, source=source, recipe=recipe)
        ranked = isinstance(recipe, ArmSpec)
        if ranked:
            spec = _bootstrap_spec(
                slot=slot,
                name=recipe.name,
                registered_at=registered_at,
                promotion_source=v1_promotion_source,
                ranker=recipe.ranker,
                params=recipe.params,
            )
        else:
            # An M (`RegisteredModelArm`) recipe: its id is the RECIPE's hash
            # and is forwarded untouched, never re-derived through an
            # `ArmSpec` (that would hash `ranker`/`params` and name a second
            # arm). Only the provenance an import adds changes. `registered_at`
            # is the recipe's own here by construction: the zoo source declares
            # no date field, so `_date_of` returned it.
            spec = dataclasses.replace(
                recipe, bootstrap=True, notes=_bootstrap_notes(v1_promotion_source)
            )
        if spec.arm_id != recipe.arm_id:
            # `ArmSpec.spec` hashes slot/name/ranker/params/control/control_kind
            # and no provenance, so these agree unless the recipe is a control
            # arm - which can never hold a pointer (section 10.1).
            why = (
                f"v1 slot {slot!r} champion {champion_name!r} resolves to recipe "
                f"{recipe.arm_id}, but its import would register {spec.arm_id}; a "
                "control arm cannot be a champion, and a pointer at an id no recipe "
                "produces would never be scored."
            )
            if asserted:
                raise MigrationSourceMissing(why)
            imported[slot] = []
            pointers[slot] = "deferred"
            deferred[slot] = why
            continue
        # `alpha-engine-config-I11085`. A pointer is a SERVING decision, so the
        # arm it names must be one the release can produce - asked of the
        # recipe itself, with the predicate `experiment.run` applies before
        # producing. On 2026-09-14 this import seated R on
        # `scanner_predictor_direct`, whose recipe refuses BY NAME until track
        # B lands; nothing checked, and the 2026-09-19 arc died at
        # `experiment.run[r]` resolving it. Checked BEFORE the existing-pointer
        # branch as well, so a rerun over a pointer this migration already
        # seated on such an arm reports the refusal instead of `unchanged`.
        # `admission_refusal` below is the RUNTIME half (has it produced?);
        # this is the half knowable from the tree, and a seeded or backfilled
        # shadow cannot satisfy it.
        # An M recipe reaches here only from the release's REGISTRABLE half
        # (`served_model_recipe`), which is the M slot's own registration
        # refusal already applied; the catalogue partition is a U/R ranker's.
        refused_by_recipe = catalog_refusal(recipe) if ranked else None
        if refused_by_recipe is not None:
            why = (
                f"v1 slot {slot!r} names champion {champion_name!r}, which resolves to "
                f"{spec.arm_id} - an arm that cannot produce: {refused_by_recipe} Seating "
                "the slot on it would point the serving path at an arm that writes no "
                "shadow, and the first arc to resolve the pointer would fail at "
                f"`experiment.run[{slot}]`."
            )
            if asserted:
                raise UnproducibleChampionError(why)
            imported[slot] = []
            pointers[slot] = "deferred"
            deferred[slot] = why
            continue
        pointer_key = champion_key(slot)
        expected = ctx.store.etag(pointer_key)
        if expected != ETAG_ABSENT:
            existing = load_store_document(ctx.store, pointer_key)
            existing_evidence = existing.get("evidence") or {}
            if (
                existing.get("arm_id") == spec.arm_id
                and existing_evidence.get("status") == "migrated"
            ):
                # Idempotent rerun: this migration already seeded this exact
                # pointer. Rewriting it would only change `run_id`/`decided_at`
                # and make the pointer claim a run that decided nothing.
                imported[slot] = [spec.arm_id]
                pointers[slot] = "unchanged"
                continue
            conflict = (
                f"{pointer_key} already holds arm {existing.get('arm_id')!r} "
                f"(promotion_source {existing.get('promotion_source')!r}, evidence status "
                f"{existing_evidence.get('status')!r}); this migration would point it at "
                f"{spec.arm_id!r}. A v1 import seeds an ABSENT pointer and never "
                "overwrites a promotion or an operator revert."
            )
            if asserted:
                raise MigrationPointerConflict(conflict)
            # The scheduled path DEFERS the same fact. Once `promote` moves a
            # pointer on the arena's own evidence — which is the normal end
            # state, and `promote` is an arc stage an hour after this one —
            # every later run of this stage would otherwise raise and kill the
            # rest of the arc behind it (`crucible-PR317` / `-I10927`). The
            # refusal is unchanged in substance: nothing is written, and the
            # reason is on the manifest rather than in a traceback.
            imported[slot] = []
            pointers[slot] = "deferred"
            deferred[slot] = conflict
            continue
        # The pointer is ABSENT, so this run would SEAT the slot. That is the
        # moment the deferral predicate applies — and only that moment: a
        # pointer this migration already wrote is `unchanged` above, and never
        # re-litigated against today's production state.
        refusal = admission_refusal(ctx.store, slot=slot, arm_id=spec.arm_id)
        if refusal is not None:
            imported[slot] = []
            pointers[slot] = "deferred"
            deferred[slot] = refusal
            continue
        imported[slot] = [spec.arm_id]
        pointers[slot] = "would_write" if dry_run else "written"
        planned.append(
            (
                slot,
                spec,
                source,
                registered_at,
                date_source,
                v1_promotion_source,
                pointer_key,
                expected,
            )
        )

    # Every slot's sources, recipe, date and existing pointer are resolved
    # above BEFORE anything is written: a refusal on R must not leave U
    # already imported (measured in `tests/test_migrate_history_cli.py`: the
    # single-pass loop wrote `champions/u/current.json` and then raised on R).
    for (
        slot,
        spec,
        source,
        registered_at,
        date_source,
        v1_promotion_source,
        pointer_key,
        expected,
    ) in [] if dry_run else planned:
        register = read_register(ctx.store, slot)
        if spec.arm_id not in set(register.all_arms()):
            register, _ = register_arms(register, [spec], filed_on=ctx.trading_day.isoformat())
            write_register(ctx.store, slot, register)
        champion_pointer = ChampionPointer(
            slot=slot,
            arm_id=spec.arm_id,
            as_of=registered_at,
            decided_at=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            run_id=ctx.run_id,
            code_sha=resolve_code_sha(),
            promotion_source=_map_promotion_source(v1_promotion_source),
            manifest_key=run_manifest_key(ctx.job, ctx.trading_day.isoformat()),
            evidence={
                "status": "migrated",
                "reason": (
                    f"imported from v1 key {source.key!r} by `crucible migrate.history` "
                    f"(v1 promotion_source: {v1_promotion_source!r}; as_of from "
                    f"{date_source})"
                ),
                "moved": False,
            },
            # `kind="pit_parity"`/`status="UNKNOWN"` is the closed schema's only shape
            # for "this pointer carries no contamination attestation" (`ATTESTED_SLOTS`
            # is `("s",)` only, so U/R/M ignore this field on read) — `key` names the
            # v1 source so a reviewer can trace the import without decoding `evidence`,
            # and `UNKNOWN` is deliberate: a future S-slot import would be refused by
            # `read_champion` on the exact rule the trader applies to a real pointer.
            attestation={
                "kind": "pit_parity",
                "status": "UNKNOWN",
                "key": source.key,
                "reason": "v1 import carries no contamination attestation.",
            },
        )
        payload = json.dumps(champion_pointer.to_dict(), indent=2, sort_keys=True).encode("utf-8")
        # `ctx.record_output_cas` is the one call that both writes
        # conditionally AND enters this write into `outputs[]`
        # (`alpha-engine-config-I9757` defect #7). `expected` was read
        # immediately above, so a concurrent writer between that read and this
        # swap raises `PointerConflictError` rather than being overwritten.
        ctx.record_output_cas(pointer_key, expected, payload, schema_version="champion.v1")

    result = {
        "schema_version": "migration.v1",
        "sources_found": [{k: v for k, v in f.items() if k != "document"} for f in found],
        "sources_missing": missing,
        "arms_imported": imported,
        "pointers": pointers,
        "slots_considered": list(slots if slots is not None else MIGRATABLE_SLOTS),
        "deferred": deferred,
        "allow_missing": allow_missing,
        "dry_run": dry_run,
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
                + (
                    "; DEFERRED: "
                    + "; ".join(f"{slot}: {why}" for slot, why in sorted(deferred.items()))
                    if deferred
                    else ""
                )
                + (f"; ABSENT: {[m['key'] for m in missing]}" if missing else "")
            ),
            "source_path": "arms/*/register.jsonl",
            "last_updated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if not dry_run:
        ctx.record_output(
            migration_key(ctx.trading_day.isoformat(), ctx.run_id),
            json.dumps(result, indent=2, sort_keys=True).encode("utf-8"),
            schema_version="migration.v1",
        )
    return result


def _date_of(pointer: dict[str, Any], *, source: V1Source, recipe: Any) -> tuple[str, str]:
    """The imported arm's start date as an ISO date, and where it came from.

    Taken from the v1 artifact when the source declares a date field: the OOS
    clock starts when the arm actually started, and starting it at cutover
    would give every imported arm a fresh eligibility window it has not
    earned. A source that declares a field and lacks it FAILS — defaulting to
    today would reset a clock that has been running for months.

    A source that declares NO date field (U's pointer carries none) takes the
    published recipe's own `registered_at`: a reviewed date in the strategy
    tree, and the one the v2 register already holds for that arm. The
    returned provenance string lands in the pointer's `evidence.reason`, so
    which of the two was used is never implicit.
    """
    if source.date_field is None:
        # An M arm is a `RegisteredModelArm` wrapping the `ModelRecipe` that
        # carries the provenance; a U/R `ArmSpec` carries it itself.
        origin = getattr(recipe, "recipe", recipe).source_key or recipe.name
        return (
            recipe.registered_at,
            f"the recipe's registered_at ({origin}); {source.key} carries no installation date",
        )
    raw = pointer.get(source.date_field)
    if not raw:
        raise MigrationSourceMissing(
            f"the v1 champion pointer at {source.key!r} carries no `{source.date_field}`, "
            "so there is no date to start its OOS clock from. Defaulting to today would "
            "reset a clock that has been running since the arm was installed."
        )
    return str(raw)[:10], f"{source.key}:{source.date_field}"


def _map_promotion_source(v1_promotion_source: str) -> str:
    """The v1 pointer's `promotion_source`, coerced into
    :data:`~crucible.champion.PROMOTION_SOURCES` — the closed set
    `ChampionPointer` validates against.

    Passed through unchanged when it already names one of the three v2
    values (`operator_bootstrap` is the common case: R's champion has
    carried it since 2026-07-13, and the whole point of carrying it across
    is that a never-moved pointer keeps rendering as the finding it is).
    Anything else — an absent field, or a v1-only value such as
    `gate_engine` — is mapped to `operator_bootstrap` rather than
    `evidence`: a migration did not re-run the arena, so the imported
    pointer must never be mistaken for one the evidence won. The raw v1
    value is never discarded — it is carried in the pointer's own
    `evidence.reason`.
    """
    return v1_promotion_source if v1_promotion_source in PROMOTION_SOURCES else "operator_bootstrap"


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


# ---------------------------------------------------------------------------
# `crucible migrate.arm_filed_on` — the one-time repair of the seven rows
# whose `registered` event was dated from the recipe, not from the append.
# ---------------------------------------------------------------------------
#
# A DATE-ONLY repair of rows that already exist, not a mutation channel. It
# changes `ArmEvent.date` on a `registered` row and NOTHING else: no `arm_id`,
# no `kind`, no `reason`, no `record` — so no `spec_hash`, no `created_date`,
# no `control`, no `supersedes`. `ImmutableArmError` and every other guard in
# `nousergon_lib.arena.arms` are untouched, and this module gains no general
# "edit a register row" function: the exhaustive set of rows it may touch is
# the literal table below, so a rerun after the table is exhausted rewrites
# nothing and a row that is not in the table cannot be reached at all.
#
# The store is versioned, so the pre-repair object stays retrievable and the
# correction is reconstructible from the bucket alone; the compare-and-swap
# below means a concurrent append is detected rather than clobbered.
#
# **Why the filing days are literals here.** They cannot be derived from the
# store: the appends wrote their register rows but the only run manifest that
# survives them is `runs/experiment.new/2026-09-11/run.json`, which seven
# successive invocations overwrote at one key (measured 2026-09-17 — see
# `crucible.track_a.ARMS_APPENDED_METRIC` for the producer fix). The true
# filing day was recovered from S3 object-version metadata on
# `crucible/arms/{u,m}/register.jsonl`, and each row below carries the exact
# version id and `LastModified` that introduced it, so the derivation is
# reviewable in the diff rather than asserted. This mirrors `SOURCES` above:
# a migration reads a system that no longer changes, and a configurable
# source would let a rerun read somewhere else and report the same success.


@dataclass(frozen=True)
class ArmFilingCorrection:
    """One `registered` row whose event date is being corrected, with the
    S3 object version that is the evidence for the new value."""

    slot: str
    arm_id: str
    #: The date the row carries today — the recipe's `created_date`, copied
    #: into the event by the pre-fix `ArmRegister.register`. Asserted before
    #: the rewrite: a row carrying anything else is refused, never coerced.
    wrong_date: str
    #: The corrected value.
    filed_on: str
    #: The object version of `arms/{slot}/register.jsonl` whose last line IS
    #: this row — i.e. the version the append created.
    evidence_version_id: str
    #: That version's `LastModified`, verbatim.
    evidence_last_modified: str


#: The exhaustive set. Seven rows: four U (the scanner_cut ports) and three M
#: (the v3.0-meta heads), appended 2026-09-17T00:23:40Z-00:24:28Z UTC.
#:
#: `filed_on` is 2026-09-17, the calendar date of every one of those appends
#: and itself an NYSE session, so it is a legal trading-day value. It is
#: deliberately the LATER of the two defensible readings:
#: `crucible.calendar.resolve_trading_day` maps those instants to 2026-09-16
#: (they land at ~20:23 ET on the 16th, after that session's close), but the
#: 2026-09-16 arena cycle had already run when the appends happened, so
#: dating the rows 2026-09-16 would demand the arms on a day whose cycle
#: could not have scored them — the exact false gap this repair exists to
#: remove. The first cycle that can score them is the 2026-09-19 arc, keyed
#: to 2026-09-18, and 2026-09-17 < 2026-09-18 either way.
ARM_FILING_CORRECTIONS: tuple[ArmFilingCorrection, ...] = (
    ArmFilingCorrection(
        slot="u",
        arm_id="u:attractiveness:a1ecc956fea0",
        wrong_date="2026-07-27",
        filed_on="2026-09-17",
        evidence_version_id="_NDUEanrMme58GBoNtuIrwj7LRok_wu1",
        evidence_last_modified="2026-09-17T00:23:40+00:00",
    ),
    ArmFilingCorrection(
        slot="u",
        arm_id="u:attractiveness_hard3:57af7b69c3d5",
        wrong_date="2026-08-24",
        filed_on="2026-09-17",
        evidence_version_id="lfd6amUgcK9P7gLQz.kyOeAditNhWQbt",
        evidence_last_modified="2026-09-17T00:24:13+00:00",
    ),
    ArmFilingCorrection(
        slot="u",
        arm_id="u:attractiveness_mom121:0a7286997a8d",
        wrong_date="2026-08-17",
        filed_on="2026-09-17",
        evidence_version_id="JT1tOqfpAabw34EZPbowEA2BSw_HHRJW",
        evidence_last_modified="2026-09-17T00:24:16+00:00",
    ),
    ArmFilingCorrection(
        slot="u",
        arm_id="u:attractiveness_momzero:c46adedc32e2",
        wrong_date="2026-08-17",
        filed_on="2026-09-17",
        evidence_version_id="ILMPbMldqI0XQB_4ybb0OOIFkBn0Xh.B",
        evidence_last_modified="2026-09-17T00:24:19+00:00",
    ),
    ArmFilingCorrection(
        slot="m",
        arm_id="m:v3meta_momentum_head:ccfb2e608dd0",
        wrong_date="2026-09-14",
        filed_on="2026-09-17",
        evidence_version_id="PPCOBBIO2KaAZO1gu4gl79rjuCvRp14_",
        evidence_last_modified="2026-09-17T00:24:22+00:00",
    ),
    ArmFilingCorrection(
        slot="m",
        arm_id="m:v3meta_stack:1bb8d3646649",
        wrong_date="2026-09-14",
        filed_on="2026-09-17",
        evidence_version_id="qG9BDdrDNFwGI7IFpU.j1MMEQS.kVLGq",
        evidence_last_modified="2026-09-17T00:24:25+00:00",
    ),
    ArmFilingCorrection(
        slot="m",
        arm_id="m:v3meta_volatility_head:97645d421bea",
        wrong_date="2026-09-14",
        filed_on="2026-09-17",
        evidence_version_id="6TDZwG.18wBvGPKv8zBtAMKgQXLLzEEB",
        evidence_last_modified="2026-09-17T00:24:28+00:00",
    ),
)


@dataclass(frozen=True)
class ArmFilingMigrationReport:
    """What one `crucible migrate.arm_filed_on` attempt did, in full.

    Exhaustive over :data:`ARM_FILING_CORRECTIONS`: every declared row appears
    in exactly one of ``corrected`` or ``refused``, named by arm id, never
    silently dropped. `len(corrected) + len(refused) == len(...)` is asserted
    before the report is returned.
    """

    migration_run_id: str
    dry_run: bool
    corrected: tuple[dict[str, Any], ...] = ()
    refused: tuple[dict[str, Any], ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {"corrected": len(self.corrected), "refused": len(self.refused)}

    def summary_line(self) -> str:
        suffix = " (dry run — nothing written)" if self.dry_run else ""
        return f"{len(self.corrected)} corrected, {len(self.refused)} refused{suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "migrate_arm_filed_on.v1",
            "migration_run_id": self.migration_run_id,
            "dry_run": self.dry_run,
            "corrected": list(self.corrected),
            "refused": list(self.refused),
            "counts": self.counts,
        }


def _register_rows(raw: bytes, key: str) -> list[dict[str, Any]]:
    """The register's rows, parsed. Raises on anything unreadable.

    Not `read_register`: the fold would hand back `ArmEvent` objects and this
    repair rewrites LINES, so that every field it is not correcting survives
    byte-for-byte rather than being re-serialized from a reconstruction.
    """
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{key}:{lineno} parsed to {type(parsed).__name__}, not an object")
        rows.append(parsed)
    return rows


def _serialize_register_rows(rows: list[dict[str, Any]]) -> bytes:
    """The same bytes shape `crucible.slots.arms.write_register` produces."""
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode("utf-8")


def _apply_filing_correction(
    rows: list[dict[str, Any]], correction: ArmFilingCorrection
) -> tuple[bool, str]:
    """Correct ``correction``'s row IN ``rows``. Returns ``(changed, detail)``.

    Refuses rather than coerces: the row must exist, be the `registered`
    event for that arm id, and still carry ``wrong_date``. A row already
    carrying ``filed_on`` is reported as an idempotent no-op, which is what a
    rerun after a partial apply must look like.
    """
    from nousergon_lib.arena.arms import (  # noqa: PLC0415 - heavy import, one call site
        EVENT_REGISTERED,
    )

    matches = [
        row
        for row in rows
        if row.get("arm_id") == correction.arm_id and row.get("kind") == EVENT_REGISTERED
    ]
    if not matches:
        return False, (
            f"no `{EVENT_REGISTERED}` row for {correction.arm_id} in the register; this "
            "repair corrects a row that exists and never creates one."
        )
    if len(matches) > 1:
        return False, (
            f"{len(matches)} `{EVENT_REGISTERED}` rows for {correction.arm_id}; the register "
            "is already inconsistent and a date repair would hide which one is authoritative."
        )
    row = matches[0]
    current = row.get("date")
    if current == correction.filed_on:
        return False, (
            f"already carries date={correction.filed_on}; nothing left for this attempt to do."
        )
    if current != correction.wrong_date:
        return False, (
            f"carries date={current!r}, not the {correction.wrong_date!r} this repair was "
            "derived against; refusing to overwrite a value this session did not measure."
        )
    row["date"] = correction.filed_on
    return True, f"date {correction.wrong_date} -> {correction.filed_on}"


def run_migrate_arm_filed_on(store: Store, *, dry_run: bool = False) -> ArmFilingMigrationReport:
    """Correct the `registered` event dates listed in
    :data:`ARM_FILING_CORRECTIONS`. One-time, audited, date-only.

    See the module comment above this function's table for the derivation and
    for why the values are literals. Safe to re-run: a row already carrying
    its corrected date is refused as a no-op rather than rewritten.
    """
    now = dt.datetime.now(dt.UTC)
    migration_run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    corrected: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []

    by_slot: dict[str, list[ArmFilingCorrection]] = {}
    for correction in ARM_FILING_CORRECTIONS:
        by_slot.setdefault(correction.slot, []).append(correction)

    for slot, corrections in sorted(by_slot.items()):
        key = arm_register_key(slot)
        if not store.exists(key):
            for correction in corrections:
                refused.append(
                    {
                        "slot": slot,
                        "arm_id": correction.arm_id,
                        "key": key,
                        "reason": f"{key} does not exist; there is no row to correct.",
                    }
                )
            continue
        expected_version = store.etag(key)
        raw = store.get_bytes(key)
        rows = _register_rows(raw, key)
        applied: list[ArmFilingCorrection] = []
        for correction in corrections:
            changed, detail = _apply_filing_correction(rows, correction)
            if not changed:
                refused.append(
                    {
                        "slot": slot,
                        "arm_id": correction.arm_id,
                        "key": key,
                        "reason": detail,
                    }
                )
                continue
            applied.append(correction)
        if not applied:
            continue
        payload = _serialize_register_rows(rows)
        if not dry_run:
            try:
                store.compare_and_swap(key, expected_version, payload)
            except PointerConflictError as exc:
                for correction in applied:
                    refused.append(
                        {
                            "slot": slot,
                            "arm_id": correction.arm_id,
                            "key": key,
                            "reason": (
                                f"concurrent write detected (compare-and-swap conflict): {exc}. "
                                "Nothing was written for this slot; re-run to re-evaluate."
                            ),
                        }
                    )
                continue
        for correction in applied:
            corrected.append(
                {
                    "slot": slot,
                    "arm_id": correction.arm_id,
                    "key": key,
                    "old_date": correction.wrong_date,
                    "new_date": correction.filed_on,
                    "evidence_version_id": correction.evidence_version_id,
                    "evidence_last_modified": correction.evidence_last_modified,
                    "dry_run": dry_run,
                }
            )

    if len(corrected) + len(refused) != len(ARM_FILING_CORRECTIONS):
        raise RuntimeError(
            f"{len(corrected)} corrected + {len(refused)} refused != "
            f"{len(ARM_FILING_CORRECTIONS)} declared corrections; a declared row was "
            "neither applied nor accounted for, and a repair that loses track of a row "
            "is worse than one that refuses it."
        )

    report = ArmFilingMigrationReport(
        migration_run_id=migration_run_id,
        dry_run=dry_run,
        corrected=tuple(corrected),
        refused=tuple(refused),
    )
    if not dry_run:
        trading_day = resolve_trading_day(now)
        store.put_bytes(
            migration_key(trading_day.isoformat(), migration_run_id),
            json.dumps(report.as_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
    return report
