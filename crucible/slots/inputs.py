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
``predictions/{arm}/{trading_day}.json`` — the *same* trading day as the row
being built, on the panel's own date axis. There is no code path that reads
a later day: the key names the day, and :func:`read_arm_predictions` refuses
a payload whose own ``trading_day`` field disagrees with the key it was read
from. A base arm's opinion about a later session is therefore unreachable
rather than merely unused.

**Identity.** The stacked recipe hashes the base arm's *name*, not its id —
the id is not knowable from one file. The binding to the exact base *recipe*
is made where it can be verified: the resolved key carries the base arm's
spec hash in its segment, and every read is recorded as a manifest input, so
`crucible explain` walks a stacked verdict back to the precise base vintage
that produced each column (plan §10.8).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from crucible.keys import arm_key_segment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crucible.slots.model import FeaturePanel
    from crucible.store import Store

__all__ = [
    "ARM_PREDICTIONS_SCHEMA_VERSION",
    "INPUT_KINDS",
    "ArmPredictionsContractError",
    "InputCycleError",
    "InputRef",
    "UnproducibleInputError",
    "arm_predictions_key",
    "assert_inputs_producible",
    "parse_input_ref",
    "prediction_column",
    "read_arm_predictions",
    "resolve_prediction_inputs",
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


def resolve_prediction_inputs(recipes: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """`{arm name: base arm names}` for every recipe, cycle-checked.

    Two refusals, both at registration:

    * a `predictions[<name>]` naming an arm that is not a recipe in the same
      slot — there is no producer, so the stacked arm can never be graded;
    * a dependency cycle, including an arm naming itself. A self-stacked arm
      would need its own prediction for the day it is computing, which is not
      a slow path but an impossible one, and a two-arm cycle is the same
      thing wearing a second file.
    """
    by_name = {r.name: r for r in recipes}
    if len(by_name) != len(list(recipes)):
        seen: set[str] = set()
        duplicated = sorted({r.name for r in recipes if r.name in seen or seen.add(r.name)})
        raise UnproducibleInputError(
            f"two recipes share the name(s) {duplicated}; a `predictions[...]` reference "
            "would resolve to whichever loaded last, so the graph is ambiguous."
        )

    edges: dict[str, tuple[str, ...]] = {}
    for recipe in recipes:
        bases: list[str] = []
        for ref in getattr(recipe, "inputs", ()):
            if ref.kind != "predictions":
                continue
            if ref.ref not in by_name:
                raise UnproducibleInputError(
                    f"arm {recipe.name!r} declares input {ref.text!r}, but no recipe named "
                    f"{ref.ref!r} exists in this slot; the slot declares "
                    f"{sorted(by_name)}. A stacked arm's base model is itself an arm — it "
                    "is registered, scored and promoted like any other — so an input "
                    "naming something that is not one has no producer and the arm does "
                    "not register (plan §9.1). This is the refusal that replaces "
                    "`registers fine, dies at grading` (alpha-engine-config-I9777)."
                )
            bases.append(ref.ref)
        edges[recipe.name] = tuple(bases)

    _assert_acyclic(edges)
    return edges


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


def assert_inputs_producible(recipes: Sequence[Any], *, feature_columns: Sequence[str]) -> None:
    """Every declared input of every recipe has a producer, or nothing registers.

    ``feature_columns`` is the feature layer's catalogue. Both declaration
    surfaces are checked — the legacy `spec.features` list and the typed
    `features[...]` entries under `spec.inputs` — because a check that read
    one of two channels is its own measured bug class.
    """
    produced = set(feature_columns)
    for recipe in recipes:
        wanted = {c: "spec.features" for c in recipe.features}
        for ref in getattr(recipe, "inputs", ()):
            if ref.kind == "features":
                wanted[ref.ref] = "spec.inputs"
        missing = sorted(c for c in wanted if c not in produced)
        if missing:
            raise UnproducibleInputError(
                f"arm {recipe.name!r} declares feature column(s) {missing}, which the "
                f"feature layer does not produce; it produces {sorted(produced)}. "
                "The arm does NOT register. A column that is itself another model's "
                "output is not a feature — declare it as `predictions[<arm-name>]` "
                "under `spec.inputs` and register that model as an arm "
                "(alpha-engine-config-I9777). Registering here and failing later at "
                "`FeatureLayerSource.panel()` is the failure mode this refusal replaces: "
                "an arm nobody can grade is indistinguishable, on every surface, from an "
                "arm nobody has graded yet."
            )
    resolve_prediction_inputs(recipes)


# ---------------------------------------------------------------------------
# The predictions artifact — the versioned producer/consumer contract.
# ---------------------------------------------------------------------------

#: `predictions/{arm}/{trading_day}.json`. Declared here rather than in
#: `crucible.keys` only because that module is being edited concurrently for
#: `alpha-engine-config-I9772`; it belongs there, and moving it is tracked.
_ARM_PREDICTIONS_PREFIX = "predictions"

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "arm_predictions.v1.json"


def arm_predictions_key(arm_id: str, trading_day: str) -> str:
    """What ONE arm predicted on ONE trading day.

    Per-arm, not per-slot: `predictions/{trading_day}.json` is the *champion's*
    serving feed, and a stacked arm reading that would depend on whichever arm
    holds the pointer — a base model that silently changes identity between
    two cycles, and a self-reference the moment the stacked arm won the slot.
    """
    return f"{_ARM_PREDICTIONS_PREFIX}/{arm_key_segment(arm_id)}/{trading_day}.json"


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
    document = json.loads(payload.decode("utf-8"))
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
        arm_id = base_arm_ids[ref.ref]
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

    from dataclasses import replace  # noqa: PLC0415

    return replace(panel, features=blocks)
