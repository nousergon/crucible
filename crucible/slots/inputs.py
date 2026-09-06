"""What an M recipe may declare as an input, and how each kind resolves.

Normative sources: `champion-challenger-policy.md` §3.1 (the recipe is the
immutable unit), plan §4.4, §4.12, §10.4, §10.8; the M0 contract discipline
in `AGENTS.md` (a new cross-repo artifact gets a versioned schema and a
producer/consumer contract test at birth).

Tracker: `alpha-engine-config-I9777`.

**The defect this module removes.** `sota_directional_combine` declared two
columns — `gbm_directional_score_zscore` and `sentiment_directional_score_zscore`
— that the feature layer has no producer for. `load_model_recipes` validated
*shape*, not *producibility*, so the arm registered cleanly and then died
deep inside `FeatureLayerSource.panel()` at grading time, with a message
about a missing parquet column rather than about an arm that could never
have been built. **Registered fine, could never be graded** is the shape of
every silent-until-Saturday defect this harness exists to make impossible.

**The ruling (2026-09-01).** A column that is itself another model's output
is not a feature — it is a *prediction*. Registering it as a `FeatureSpec`
whose expression is a model fit would put training inside the feature layer
and make the layer's version hash depend on a fit, which destroys exactly
the point-in-time guarantee that hash exists to provide. So a meta-learner
declares its base models as **stacked inputs on the predictions contract**,
which is the SOTA meta-learner shape.

**The grammar.** `spec.inputs` is an optional list of typed references:

    inputs:
      - predictions[gbm_directional]
      - features[residual_vol_20d_ratio]

`features[...]` is the same thing `spec.features` already declares and is
accepted so the two kinds read alike in one list; `spec.features` stays the
required field and is unchanged, so no existing arm's id moves.

**`inputs` is absent from the hashed spec when it is empty.** Adding a field
that always appeared would have re-hashed every recipe already registered
and orphaned its score series (policy §3.1). :meth:`ModelRecipe.spec` emits
the key only when the arm declares one, and a test pins an existing arm's id
across this change.

**Point-in-time by construction.** A prediction input resolves to
:func:`crucible.keys.arm_predictions_key` for the *same* trading day as the
row being built, on the panel's own date axis. There is no code path that reads
a later day: the key names the day, and :func:`read_arm_predictions` refuses
a payload whose own ``trading_day`` field disagrees with the key it was read
from. A base arm's opinion about a later session is therefore unreachable
rather than merely unused.

**Identity.** The stacked recipe hashes the base arm's *name*, not its id —
the id is not knowable from one file. The binding from that name to the arm
actually read is made twice, and both times by code rather than by a caller:
:func:`crucible.slots.model.design_panel` resolves every base id from the
loaded recipe set, and :func:`stack_prediction_columns` asserts that the id
it was handed carries the name that asked for it
(:func:`arm_name_from_id`). Before that assertion existed, a caller-supplied
`base_arm_ids` could stack any arm's cross-section under any declared base's
column; the lineage recorded the key honestly and the design matrix was
still wrong. The resolved key then carries the base arm's spec hash in its
segment, and every read is recorded as a manifest input, so `crucible
explain` walks a stacked verdict back to the precise base vintage that
produced each column (plan §10.8).

**Wiring is a table, not a call site.** :data:`INPUT_RESOLVERS` maps every
kind in :data:`INPUT_KINDS` to the function that materialises it onto a
panel. :func:`partition_producible` refuses at registration any declared
kind with no row, importing this module fails outright if the grammar admits
a kind nothing resolves, and :func:`resolve_declared_inputs` — reached from
the M slot's one panel-building seam — iterates the table rather than naming
kinds. A producer declared and never wired is the defect this module was
written to remove and then reproduced once; the table is what stops a third
occurrence, because there is no longer a call site to forget.

**The refusal is PER ARM, not per directory** (`alpha-engine-config-I9955`).
:func:`partition_producible` returns the arms that register beside the
:class:`InputRefusal` values for the arms that do not, because an exception
is slot-wide by construction: one arm that will not be buildable until phase
5 used to refuse every sibling in the same directory, so a slot with two
perfectly producible arms accumulated no evidence at all. The refusal is not
softened — a refused arm still does not register, it is recorded as a metric
on the loading job's manifest, and an EMPTY registered set raises
:class:`SlotUnservableError`, which pages through the existing failure
condition. Two conditions stay slot-wide because they are properties of the
graph rather than of a member: a dependency cycle
(:class:`InputCycleError`) and two recipes sharing a name.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from crucible.documents import load_document_bytes
from crucible.keys import arm_predictions_key

if TYPE_CHECKING:
    from crucible.slots.model import FeaturePanel
    from crucible.store import Store

__all__ = [
    "ARM_PREDICTIONS_SCHEMA_VERSION",
    "INPUT_KINDS",
    "INPUT_RESOLVERS",
    "ArmPredictionsContractError",
    "BasePredictionsUnavailableError",
    "InputCycleError",
    "InputRef",
    "InputRefusal",
    "SlotUnservableError",
    "UnproducibleInputError",
    "UnresolvedInputError",
    "arm_name_from_id",
    "arm_predictions_key",
    "parse_input_ref",
    "partition_producible",
    "prediction_column",
    "read_arm_predictions",
    "resolve_declared_inputs",
    "stack_prediction_columns",
    "write_arm_predictions",
]

ARM_PREDICTIONS_SCHEMA_VERSION = "arm_predictions.v1"

#: The closed set of input kinds. Closed on purpose: a kind resolved by name
#: at runtime is an input whose recipe does not describe where it comes from,
#: which is the property this whole module exists to restore.
INPUT_KINDS: tuple[str, ...] = ("features", "predictions")

_REF = re.compile(r"^(?P<kind>[a-z_]+)\[(?P<ref>[A-Za-z0-9_.:~-]+)\]$")

#: A prediction input becomes this design column. `_raw` because the producer
#: writes the arm's own alpha in its own units and applies no transform: a
#: standardisation done by the producer is a standardisation no consumer
#: declared, and `AGENTS.md`'s units rule exists because exactly that went
#: wrong once (`avg_volume_20d`).
_PREDICTION_COLUMN_TEMPLATE = "predicted_alpha_{name}_raw"


class UnproducibleInputError(ValueError):
    """An arm declares an input nothing in this harness can produce.

    Raised at LOAD, which is registration: plan §9.1 makes registration the
    gate, and an arm admitted to the register that cannot be graded is
    indistinguishable, on every surface, from one that simply has not been
    graded yet.
    """


class InputCycleError(ValueError):
    """A stacked arm depends, directly or transitively, on itself."""


class ArmPredictionsContractError(ValueError):
    """A predictions artifact that does not conform, or is not the one asked for."""


class BasePredictionsUnavailableError(ArmPredictionsContractError):
    """A base arm's predictions do not exist for every session the panel carries.

    Separate from a malformed document because the remedy is different and
    mechanical: run the base arm's producer over the named sessions. The
    message carries the count and the command, because a stacked arm on a
    504-session window needs 504 base-arm artifacts and "KeyError" repeated
    once per missing day is not an operator instruction.
    """


class UnresolvedInputError(UnproducibleInputError):
    """A declared input was never materialised onto the panel being trained on.

    The refusal that replaces `KeyError: feature 'predicted_alpha_x_raw' is
    not in this panel`. That message blamed the parquet layer for a column
    the parquet layer was never asked to produce; this one names the seam
    that was skipped and the producer that fills it.
    """


@dataclass(frozen=True)
class InputRef:
    """One typed input reference: a kind and the thing it names."""

    kind: str
    ref: str

    def __post_init__(self) -> None:
        if self.kind not in INPUT_KINDS:
            raise UnproducibleInputError(
                f"input kind {self.kind!r} is not one of {list(INPUT_KINDS)}. An input "
                "whose kind the harness does not know cannot be resolved to a producer, "
                "so the arm does not register."
            )

    @property
    def text(self) -> str:
        return f"{self.kind}[{self.ref}]"

    @property
    def column(self) -> str:
        """The design-matrix column this reference contributes."""
        if self.kind == "predictions":
            return prediction_column(self.ref)
        return self.ref


def prediction_column(arm_name: str) -> str:
    """The design column a `predictions[<arm-name>]` input contributes."""
    return _PREDICTION_COLUMN_TEMPLATE.format(name=arm_name)


def arm_name_from_id(arm_id: str) -> str:
    """`m:gbm_directional:ab12cd` -> `gbm_directional`.

    The declared name and the artifact actually read are bound here. Without
    it, `base_arm_ids` was a caller-supplied lookup nothing checked, so any
    arm's cross-section could be stacked under any declared base's column
    and the design matrix would be wrong while every surface stayed quiet.
    """
    parts = str(arm_id).split(":")
    if len(parts) != 3 or not all(parts):
        raise UnproducibleInputError(
            f"arm id {arm_id!r} is not `{{slot}}:{{name}}:{{spec_hash}}`. A stacked arm "
            "binds a declared base NAME to a resolved arm ID; an id whose name cannot be "
            "read is an id that cannot be checked against the name that asked for it."
        )
    return parts[1]


def parse_input_ref(text: str) -> InputRef:
    """`"predictions[gbm_directional]"` -> :class:`InputRef`.

    A bare column name is refused rather than guessed at. `spec.features`
    already declares feature columns; a second, untyped way to declare the
    same thing is how two spellings of one input end up meaning different
    things to two readers.
    """
    match = _REF.match(str(text).strip())
    if match is None:
        raise UnproducibleInputError(
            f"input {text!r} is not a typed reference. Every entry under `spec.inputs` "
            f"reads `<kind>[<name>]` with kind in {list(INPUT_KINDS)} — for example "
            "`predictions[gbm_directional]`. A bare name is refused rather than assumed "
            "to be a feature column: the whole point of this field is that an input says "
            "where it comes from."
        )
    return InputRef(kind=match.group("kind"), ref=match.group("ref"))


# ---------------------------------------------------------------------------
# Registration-time validation (deliverables 3 and 4 of I9777).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputRefusal:
    """One arm that cannot be graded, carried as a VALUE rather than thrown.

    The refusal itself is not the defect `alpha-engine-config-I9955` names —
    a declared input with no producer is exactly as unbuildable as it was
    before, and softening that would put `registers fine, dies at grading`
    back. What was wrong is the BLAST RADIUS: an exception ends the load, so
    one unbuildable arm took every sibling in the directory down with it, and
    the M slot accumulated no evidence at all while two perfectly producible
    arms sat in the same directory.

    An exception is slot-wide by construction; a value is per-arm. So the
    refusal becomes a first-class fact that is returned, recorded as a metric
    on the manifest of whatever job loaded the slot
    (:meth:`crucible.slots.model.SlotRecipes.refusal_metrics`), and — when
    NOTHING registers — raised as :class:`SlotUnservableError`, which is what
    makes the job's manifest `status: failed` and pages through the existing
    failure condition. No third page condition, and nothing quieter than a
    refused arm is today.

    ``unresolvable`` names the exact inputs that have no producer — the input
    reference text for a typed input, the bare column name for a
    `spec.features` entry the feature layer does not produce.
    """

    arm: str
    unresolvable: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not str(self.arm).strip():
            raise ValueError(
                "a refusal that does not name the arm it refuses is unactionable: the "
                "whole point of carrying it as a value is that a surface can say WHICH "
                "arm is unservable."
            )
        if not self.unresolvable:
            raise ValueError(
                f"the refusal of arm {self.arm!r} names no unresolvable input. A refusal "
                "whose cause is not in it is a red row nobody can act on, which is the "
                "defect this type exists to prevent rather than to reproduce."
            )
        if not str(self.reason).strip():
            raise ValueError(f"the refusal of arm {self.arm!r} carries no reason")


class SlotUnservableError(UnproducibleInputError):
    """EVERY arm in the slot is refused, so the slot can serve nothing.

    The honest reading of an empty registered set (plan §5.3, §7:
    `unservable` is a first-class status that PAGES). It is a subclass of
    :class:`UnproducibleInputError` because that is precisely what it is —
    the whole-slot case of it — and because a single-arm slot whose one arm
    is unproducible must keep raising exactly as it did before this change.

    It raises rather than returning a value on purpose: a raise reaches
    :func:`crucible.runner.run_job`'s `try/finally`, which writes
    `status: failed` with this message as the cause, and
    :func:`crucible.alerts.evaluate_failure` pages on that manifest. That is
    how `unservable` pages without inventing a third page condition.
    """

    def __init__(self, refusals: Sequence[InputRefusal]) -> None:
        self.refusals: tuple[InputRefusal, ...] = tuple(refusals)
        names = [r.arm for r in self.refusals]
        super().__init__(
            f"every arm in this slot is refused, so the slot is `unservable` and pages "
            f"(plan §5.3, §7). Refused arm(s) {names}: " + " ".join(r.reason for r in self.refusals)
        )


def _by_name(recipes: Sequence[Any]) -> dict[str, Any]:
    """`{name: recipe}`, refusing an ambiguous slot SLOT-WIDE.

    Two files sharing a name is a property of the directory, not of one
    member: a `predictions[...]` reference would resolve to whichever loaded
    last, so no arm in the slot can be trusted to have read the graph the
    operator wrote. Partitioning that would be dishonest, exactly as
    partitioning a cycle would be.
    """
    by_name = {r.name: r for r in recipes}
    if len(by_name) != len(list(recipes)):
        seen: set[str] = set()
        duplicated = sorted({r.name for r in recipes if r.name in seen or seen.add(r.name)})
        raise UnproducibleInputError(
            f"two recipes share the name(s) {duplicated}; a `predictions[...]` reference "
            "would resolve to whichever loaded last, so the graph is ambiguous."
        )
    return by_name


def _declared_edges(
    recipes: Sequence[Any], by_name: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    """`{arm name: base arm names that EXIST}`, one key per recipe.

    A base that does not exist is left out of the edge set and reported as a
    refusal instead; it cannot participate in a cycle, and putting it in the
    graph would make the cycle walk raise `KeyError` on a condition that has
    a precise message of its own.
    """
    return {
        recipe.name: tuple(
            ref.ref
            for ref in getattr(recipe, "inputs", ())
            if ref.kind == "predictions" and ref.ref in by_name
        )
        for recipe in recipes
    }


def _absent_base_reason(recipe: Any, absent: Sequence[str], by_name: Mapping[str, Any]) -> str:
    names = sorted({t[t.index("[") + 1 : -1] for t in absent})
    return (
        f"arm {recipe.name!r} declares input(s) {sorted(absent)}, but no recipe named "
        f"{names} exists in this slot; the slot declares {sorted(by_name)}. A stacked "
        "arm's base model is itself an arm — it is registered, scored and promoted like "
        "any other — so an input naming something that is not one has no producer and "
        "the arm does not register (plan §9.1). This is the refusal that replaces "
        "`registers fine, dies at grading`."
    )


def _unwired_kind_reason(recipe: Any, ref: InputRef) -> str:
    return (
        f"arm {recipe.name!r} declares input {ref.text!r}, but the harness has no producer "
        f"wired for kind {ref.kind!r}: `crucible.slots.inputs.INPUT_RESOLVERS` carries "
        f"{sorted(INPUT_RESOLVERS)}. The arm does NOT register. A kind that is declarable "
        "but not resolvable is exactly `registers fine, could never be graded` with a new "
        "name."
    )


def _missing_feature_reason(recipe: Any, missing: Sequence[str], produced: set[str]) -> str:
    return (
        f"arm {recipe.name!r} declares feature column(s) {sorted(missing)}, which the "
        f"feature layer does not produce; it produces {sorted(produced)}. "
        "The arm does NOT register. A column that is itself another model's output is not "
        "a feature — declare it as `predictions[<arm-name>]` under `spec.inputs` and "
        "register that model as an arm. Registering here and failing later at "
        "`FeatureLayerSource.panel()` is the failure mode this refusal replaces: an arm "
        "nobody can grade is indistinguishable, on every surface, from an arm nobody has "
        "graded yet."
    )


def partition_producible(
    recipes: Sequence[Any], *, feature_columns: Sequence[str]
) -> tuple[tuple[Any, ...], tuple[InputRefusal, ...]]:
    """Split ``recipes`` into what registers and what is refused, PER ARM.

    The whole content of `alpha-engine-config-I9955`. Everything the strict
    the old all-or-nothing `assert_inputs_producible` refused is refused here
    too, with the same message; what changes is that the refusal names one arm
    and the slot's other arms keep loading.

    Two conditions stay SLOT-WIDE because they are properties of the graph
    rather than of a member, and partitioning them would be dishonest:

    * two recipes sharing a name (:func:`_by_name`) — no arm in the slot can
      be trusted to have read the graph the operator wrote;
    * a dependency cycle (:func:`_assert_acyclic`) — a cycle belongs to the
      set of arms in it, and reporting it as "these three arms are each
      individually refused" loses the one fact an operator needs.

    Refusal is TRANSITIVE. An arm stacking on a refused arm cannot be graded
    either — its base has no producer — so it is refused too, to a fixed
    point. Registering it because its own declaration parses would move
    `registers fine, dies at grading` up one level instead of removing it.
    """
    by_name = _by_name(recipes)
    edges = _declared_edges(recipes, by_name)
    _assert_acyclic(edges)

    produced = set(feature_columns)
    refusals: dict[str, InputRefusal] = {}
    for recipe in recipes:
        causes: list[str] = []
        reasons: list[str] = []
        declared = tuple(getattr(recipe, "inputs", ()))

        unwired = [ref for ref in declared if ref.kind not in INPUT_RESOLVERS]
        for ref in unwired:
            causes.append(ref.text)
            reasons.append(_unwired_kind_reason(recipe, ref))

        wanted = {c: "spec.features" for c in recipe.features}
        for ref in declared:
            if ref.kind == "features":
                wanted[ref.ref] = "spec.inputs"
        missing = sorted(c for c in wanted if c not in produced)
        if missing:
            causes.extend(missing)
            reasons.append(_missing_feature_reason(recipe, missing, produced))

        absent = [r.text for r in declared if r.kind == "predictions" and r.ref not in by_name]
        if absent:
            causes.extend(absent)
            reasons.append(_absent_base_reason(recipe, absent, by_name))

        if causes:
            refusals[recipe.name] = InputRefusal(
                arm=recipe.name, unresolvable=tuple(causes), reason=" ".join(reasons)
            )

    _propagate_refusals(edges, refusals)
    registered = tuple(r for r in recipes if r.name not in refusals)
    return registered, tuple(refusals[name] for name in sorted(refusals))


def _propagate_refusals(
    edges: Mapping[str, tuple[str, ...]], refusals: dict[str, InputRefusal]
) -> None:
    """An arm stacking on a refused arm is refused too — to a fixed point.

    Mutates ``refusals`` in place. Without this, a partition would register
    an arm whose base cannot be produced: its own declaration parses, its
    base names a real recipe in the slot, and it still dies the first time
    anything tries to build its design matrix. That is the defect this whole
    module exists to remove, arriving one frame later.
    """
    changed = True
    while changed:
        changed = False
        for name, bases in edges.items():
            if name in refusals:
                continue
            bad = tuple(f"predictions[{b}]" for b in sorted(set(bases)) if b in refusals)
            if not bad:
                continue
            refusals[name] = InputRefusal(
                arm=name,
                unresolvable=bad,
                reason=(
                    f"arm {name!r} stacks on {list(bad)}, which is itself refused in this "
                    "slot, so this arm has no producer either. A stacked arm is exactly as "
                    "gradable as the least gradable arm it consumes; registering it because "
                    "its OWN declaration parses is `registers fine, dies at grading` one "
                    "level up."
                ),
            )
            changed = True


def _assert_acyclic(edges: Mapping[str, tuple[str, ...]]) -> None:
    """Depth-first three-colour walk; the raise names the cycle it found.

    Reporting the path rather than the fact is the difference between an
    operator editing the right file and an operator reading four of them.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(edges, WHITE)

    def visit(node: str, path: list[str]) -> None:
        colour[node] = GREY
        path.append(node)
        for base in edges[node]:
            if colour[base] == GREY:
                cycle = [*path[path.index(base) :], base]
                raise InputCycleError(
                    "stacked-arm input cycle: " + " -> ".join(cycle) + ". An arm cannot "
                    "consume its own prediction for the day it is producing, directly or "
                    "through any chain — the base arm's artifact for that trading day "
                    "does not exist until the stacked arm has already run."
                )
            if colour[base] == WHITE:
                visit(base, path)
        path.pop()
        colour[node] = BLACK

    for node in sorted(edges):
        if colour[node] == WHITE:
            visit(node, [])


# ---------------------------------------------------------------------------
# The predictions artifact — the versioned producer/consumer contract.
# ---------------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "arm_predictions.v1.json"


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"arm predictions schema missing at {SCHEMA_PATH}. It ships inside the "
            "package; a missing schema is a broken build, not a degraded run."
        )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(document: Mapping[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(dict(document)), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise ArmPredictionsContractError(
            f"predictions document does not conform to {ARM_PREDICTIONS_SCHEMA_VERSION}: {detail}"
        )


def write_arm_predictions(
    ctx: Any,
    *,
    arm_id: str,
    trading_day: str,
    feature_version: str,
    predicted_alpha: Mapping[str, float],
) -> str:
    """The PRODUCER half. Validates before it writes, and records the output.

    Validation on the write side as well as the read side is deliberate: a
    producer able to emit a non-conforming document defeats the schema, and
    the consumer would then be the first thing to notice — a week later, on
    someone else's arm.
    """
    document = {
        "schema_version": ARM_PREDICTIONS_SCHEMA_VERSION,
        "arm_id": arm_id,
        "trading_day": trading_day,
        "feature_version": feature_version,
        "predicted_alpha": {str(k): float(v) for k, v in predicted_alpha.items()},
    }
    _validate(document)
    key = arm_predictions_key(arm_id, trading_day)
    ctx.record_output(
        key,
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        schema_version=ARM_PREDICTIONS_SCHEMA_VERSION,
    )
    return key


def read_arm_predictions(
    store: Store,
    *,
    arm_id: str,
    trading_day: str,
    ctx: Any = None,
) -> dict[str, float]:
    """The CONSUMER half, and the point-in-time guard.

    The key names the trading day, and the document repeats it. Both are
    checked against what the caller asked for, so a document written for a
    later session — however it came to sit under this key — is refused rather
    than trained on. That is what makes "an input arm's prediction for a later
    day is unreachable" a property of the code rather than of the caller's
    discipline.

    Recorded as a manifest input when a ``ctx`` is given, which is what
    `crucible explain` walks: a stacked arm's verdict reaches back to the
    exact base-arm vintage that fed each of its columns (plan §10.8).
    """
    key = arm_predictions_key(arm_id, trading_day)
    payload = store.get_bytes(key)  # raises KeyError when absent; never None
    document = load_document_bytes(key, payload)  # the bytes recorded below decide
    _validate(document)
    if document["arm_id"] != arm_id:
        raise ArmPredictionsContractError(
            f"{key} carries arm_id {document['arm_id']!r} but sits under the key for "
            f"{arm_id!r}. A misfiled prediction cross-section trains a stacked arm on "
            "another model's opinion under this model's name."
        )
    if document["trading_day"] != trading_day:
        raise ArmPredictionsContractError(
            f"{key} carries trading_day {document['trading_day']!r}, not {trading_day!r}. "
            "A stacked arm reads its base arm's prediction for the SAME session it is "
            "scoring; a document from another session is look-ahead if it is later and a "
            "stale input if it is earlier, and neither is silently acceptable."
        )
    if ctx is not None:
        ctx.record_input(key, payload, schema_version=ARM_PREDICTIONS_SCHEMA_VERSION)
    return {str(k): float(v) for k, v in document["predicted_alpha"].items()}


def stack_prediction_columns(
    panel: FeaturePanel,
    *,
    store: Store,
    recipe: Any,
    base_arm_ids: Mapping[str, str],
    ctx: Any = None,
) -> FeaturePanel:
    """Add one column per `predictions[...]` input to ``panel``.

    Read per PANEL DATE, so the stacked column is point-in-time along the
    whole training axis rather than only at its anchor — a base arm's
    prediction for session *t* is the one that lands in row *t*.

    **The declared name is checked against the id it resolved to.**
    ``base_arm_ids`` is a lookup, and a lookup nothing verifies is a lookup
    that can be wrong: before this check, `base_arm_ids={"base":
    "m:totally_different_arm:deadbe"}` stacked an unrelated arm's opinion
    under the column `predicted_alpha_base_raw` without complaint. The
    lineage recorded the key honestly and the design matrix was still wrong.
    :func:`crucible.slots.model.design_panel` resolves the mapping from the
    loaded recipe set so a caller cannot supply one at all; this assertion is
    what makes that the only reachable outcome rather than the usual one.

    A base arm that did not score every name the panel carries is a refusal,
    not a hole: a stacked arm's design row is only meaningful when every base
    model expressed an opinion about that name, and a substituted zero is the
    2026-08-28 hard-zeroed-features condition arriving by a different door.
    """
    import numpy as np  # noqa: PLC0415

    refs = [r for r in getattr(recipe, "inputs", ()) if r.kind == "predictions"]
    if not refs:
        return panel

    blocks = dict(panel.features)
    for ref in refs:
        arm_id = _resolved_base_arm_id(recipe, ref, base_arm_ids)
        _assert_base_predictions_present(store, arm_id=arm_id, ref=ref, dates=panel.dates)
        block = np.full((len(panel.dates), len(panel.names)), np.nan, dtype="float64")
        for row, day in enumerate(panel.dates):
            scores = read_arm_predictions(store, arm_id=arm_id, trading_day=day, ctx=ctx)
            absent = [n for n in panel.names if n not in scores]
            if absent:
                raise ArmPredictionsContractError(
                    f"base arm {arm_id!r} scored {len(scores)} name(s) on {day} but the "
                    f"panel carries {len(panel.names)}; {len(absent)} missing, first "
                    f"five {absent[:5]}. A stacked arm needs its base model's opinion on "
                    "every name it scores — nothing is substituted, because a "
                    "substituted zero is a hard-zeroed feature column with a friendlier "
                    "name."
                )
            block[row, :] = np.array([scores[n] for n in panel.names], dtype="float64")
        blocks[ref.column] = block

    return _with_resolved(panel, features=blocks, refs=refs)


def _resolved_base_arm_id(recipe: Any, ref: InputRef, base_arm_ids: Mapping[str, str]) -> str:
    """The arm id for ``ref``, verified to belong to the name ``ref`` declares."""
    try:
        arm_id = base_arm_ids[ref.ref]
    except KeyError as exc:
        raise UnproducibleInputError(
            f"arm {getattr(recipe, 'name', '<unnamed>')!r} declares input {ref.text!r} but "
            f"no arm id was resolved for {ref.ref!r}; ids were resolved for "
            f"{sorted(base_arm_ids)}. Build the panel with "
            "`crucible.slots.model.design_panel`, which resolves every base id from the "
            "loaded recipe set."
        ) from exc
    actual = arm_name_from_id(arm_id)
    if actual != ref.ref:
        raise UnproducibleInputError(
            f"input {ref.text!r} resolved to arm id {arm_id!r}, whose name is {actual!r}. "
            f"The design column {ref.column!r} would then carry {actual!r}'s opinion under "
            f"{ref.ref!r}'s name — a wrong design matrix that every surface renders as a "
            "healthy one, because the lineage honestly records the key that was read."
        )
    return arm_id


def _assert_base_predictions_present(
    store: Store, *, arm_id: str, ref: InputRef, dates: Sequence[str]
) -> None:
    """Every session the panel carries has a base-arm artifact, or one refusal.

    Checked ahead of the read loop and reported as a COUNT with the command
    that fills it. Registration can establish that a base arm exists; only
    the store can say whether it has run, and a stacked arm on a 504-session
    window needs 504 artifacts. One `KeyError` per missing day, discovered
    one day at a time, is not an operator instruction.
    """
    missing = [d for d in dates if not store.exists(arm_predictions_key(arm_id, d))]
    if not missing:
        return
    raise BasePredictionsUnavailableError(
        f"base arm {arm_id!r} has no predictions artifact for {len(missing)} of "
        f"{len(dates)} panel session(s) (first {missing[0]}, last {missing[-1]}). A "
        f"stacked arm reads {ref.text!r} on every row of its training window, so a gap is "
        "a refusal rather than a hole — a substituted zero is the 2026-08-28 hard-zeroed "
        "condition by another door. Produce them with:\n"
        f"    crucible experiment.run --slot m --arm {ref.ref} --trading-day <session>"
    )


def _resolve_feature_inputs(
    panel: FeaturePanel,
    *,
    store: Store,  # noqa: ARG001 - resolver signature; features come from the layer
    recipe: Any,
    base_arm_ids: Mapping[str, str],  # noqa: ARG001 - resolver signature
    ctx: Any = None,  # noqa: ARG001 - resolver signature
) -> FeaturePanel:
    """`features[...]` inputs: already produced by the layer, asserted here.

    A resolver rather than a no-op so the table below is total over
    :data:`INPUT_KINDS`. A kind present in the grammar and absent from the
    table is what `predictions` was before this change — declarable,
    registerable, and resolved by nothing.
    """
    refs = [r for r in getattr(recipe, "inputs", ()) if r.kind == "features"]
    if not refs:
        return panel
    missing = sorted(r.column for r in refs if r.column not in panel.features)
    if missing:
        raise UnresolvedInputError(
            f"arm {getattr(recipe, 'name', '<unnamed>')!r} declares feature input(s) "
            f"{missing} which this panel does not carry; it carries "
            f"{sorted(panel.features)}. The feature layer produces these columns — the "
            "panel was built without asking for them."
        )
    return _with_resolved(panel, features=dict(panel.features), refs=refs)


def _with_resolved(
    panel: FeaturePanel, *, features: dict[str, Any], refs: Sequence[InputRef]
) -> FeaturePanel:
    """``panel`` with ``features``, and the resolved refs recorded on it.

    The provenance is what lets :func:`crucible.slots.model._design` tell
    "this panel was never built through the seam" from "the layer is missing
    a column", and report the first as the wiring defect it is.
    """
    from dataclasses import replace  # noqa: PLC0415

    resolved = tuple(getattr(panel, "resolved_inputs", ())) + tuple(r.text for r in refs)
    return replace(panel, features=features, resolved_inputs=resolved)


#: kind -> the function that materialises that kind's columns onto a panel.
#:
#: The table is the wiring. :func:`partition_producible` refuses at
#: REGISTRATION any declared kind absent from it, and the import-time check
#: below refuses to load a module whose grammar admits a kind nothing
#: resolves — so the shape of `alpha-engine-config-I9777` (a producer
#: declared and never wired, an arm registering fine and dying at training)
#: cannot reopen for a third kind without the process failing to start.
INPUT_RESOLVERS: dict[str, Any] = {
    "features": _resolve_feature_inputs,
    "predictions": stack_prediction_columns,
}


def _check_every_kind_is_wired(kinds: Iterable[str], resolvers: dict[str, Any]) -> None:
    """Refuse a grammar that admits an input kind nothing resolves.

    A function rather than a bare import-time `if` so
    `tests/test_slot_inputs_wiring.py` can show the refusal firing on a kind
    with no resolver; the module still calls it at import, so the process
    fails to start on exactly the shape it did before.
    """
    unwired = tuple(k for k in kinds if k not in resolvers)
    if unwired:
        raise RuntimeError(
            f"input kind(s) {list(unwired)} are declarable under `spec.inputs` but have no "
            "entry in INPUT_RESOLVERS, so an arm declaring one would register and then die "
            "when its design matrix was built. Wire the producer or remove the kind from "
            "INPUT_KINDS; there is no third option."
        )


_check_every_kind_is_wired(INPUT_KINDS, INPUT_RESOLVERS)


def resolve_declared_inputs(
    panel: FeaturePanel,
    *,
    store: Store,
    recipe: Any,
    base_arm_ids: Mapping[str, str],
    ctx: Any = None,
) -> FeaturePanel:
    """Materialise EVERY declared input onto ``panel``, one resolver per kind.

    Iterates :data:`INPUT_RESOLVERS` rather than naming kinds, so a fourth
    kind is wired by adding a row to the table and nothing else — the seam
    picks it up, and a row that is missing is refused at registration.
    """
    for kind in INPUT_KINDS:
        panel = INPUT_RESOLVERS[kind](
            panel, store=store, recipe=recipe, base_arm_ids=base_arm_ids, ctx=ctx
        )
    return panel
