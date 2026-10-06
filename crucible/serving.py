"""The M champion's serving feed — the second half of the trader contract.

Normative sources: plan §3 ("**The contract is the only coupling.**"), §4.4,
§4.12; `crucible/AGENTS.md` — *"the trader reads one contract —
`champions/{slot}/current.json` plus `predictions/{trading_day}.json`"*;
`alpha-engine-config-I10129`.

**Why this module exists at all.** `crucible.keys.predictions_key` has named
that key since `crucible-PR207` and nothing wrote it. The harness could hold
a valid, attested M champion pointer and serve the trader nothing, with no
surface saying so — one contract out from `alpha-engine-config-I9957`'s bug
class, a producer declared and reached from nothing. A contract half of which
has no producer is not a contract; it is prose that happens to sit next to
code.

**The feed is a REPUBLICATION, never a computation.** `crucible.slots.cycle`
already states the rule this module obeys: *"the serving path resolves the
pointer — it never imports a ranking function directly."* So
:func:`publish_predictions_feed` reads `champions/m/current.json`, resolves
the arm it names to that arm's own `arm_predictions.v1` document for the same
trading day, and republishes it under the trader's key with provenance
attached. It never fits, never ranks and never re-derives. Two consequences,
both deliberate:

1. **The trader's numbers and the harness's graded numbers are the same
   numbers**, provably: `source_key` names the exact document, and a
   digest-level comparison is a `store.get_bytes` away.
2. **A pointer naming an arm that produced nothing this session is a
   FAILURE**, not an empty feed. It raises
   :class:`~crucible.slots.cycle.MissingArtifactError` — the same type
   `run_produce` raises for the same condition on R and U, rather than a
   second error class meaning the same thing.

**No champion is not a failure.** :func:`publish_predictions_feed` returns
``None`` when no pointer exists: the M slot has had no champion at any point
in this store's life, no feed is owed, and a producer that raised here would
make every promote run before the first M promotion fail. An *unusable*
pointer is a different matter — :class:`~crucible.champion.ChampionUnusableError`
from :func:`~crucible.champion.read_champion` propagates untouched, so a
champion whose producing run was not `ok` never gets a feed written for it.
That refusal is the reader's and is not re-implemented here.

**On the money path.** `predictions/{trading_day}.json` is what the trader
sizes orders from, so it is a money-path artifact under plan §9.5 and joins
the hash chain that `alpha-engine-config-I10414` / `crucible-PR240` builds;
that PR's `crucible.manifest.MONEY_PATH_PREDICATES` already enumerates
:func:`~crucible.keys.predictions_key`. `tests/test_predictions_feed.py`
guards the membership on PRESENCE — it asserts it once that module exposes
the predicate set and states the pending dependency otherwise — so the two
PRs can land in either order without one silently dropping the other's
guarantee.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from crucible.champion import ChampionPointer, read_champion
from crucible.documents import UnreadableDocumentError, load_document_bytes
from crucible.keys import arm_predictions_key, predictions_key
from crucible.models import PredictionsFeedDocument
from crucible.slots.inputs import ARM_PREDICTIONS_SCHEMA_VERSION, PREDICTION_STD_FIELDS
from crucible.store import Store

__all__ = [
    "PREDICTIONS_FEED_SCHEMA_VERSION",
    "PREDICTIONS_FEED_SLOT",
    "PredictionsFeed",
    "PredictionsFeedContractError",
    "publish_predictions_feed",
    "publish_promoted_feed",
    "read_predictions_feed",
    "servable_source",
]

PREDICTIONS_FEED_SCHEMA_VERSION = "predictions_feed.v1"

#: The feed is the M slot's contract specifically. R and U serve their
#: champions under their own keys (`signals_key`, `universe_members_key`),
#: and S serves a book rather than a cross-section.
PREDICTIONS_FEED_SLOT = "m"


class PredictionsFeedContractError(RuntimeError):
    """The feed exists at the trader's key and must not be served.

    Deliberately distinct from ``KeyError`` (no feed at all), for the reason
    :class:`~crucible.champion.ChampionUnusableError` is distinct from it:
    "the harness has not published today's feed yet" and "today's feed is
    malformed" have different fixes, and a trader that collapsed them would
    treat a corrupt document as a quiet day.
    """


@dataclass(frozen=True)
class PredictionsFeed:
    """One trading day's champion cross-section, as the trader reads it."""

    slot: str
    trading_day: str
    champion: str
    feature_version: str
    source_key: str
    predicted_alpha: dict[str, float]
    schema_version: str = PREDICTIONS_FEED_SCHEMA_VERSION
    #: `alpha-engine-config-I11791`: the champion's per-name std fields and
    #: their method, republished verbatim from the source document. Empty
    #: when the source carries none — the feed never computes one.
    uncertainty: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_alpha_std(self) -> dict[str, float] | None:
        """The TOTAL std the executor's conviction gate reads, or ``None``."""
        return self.uncertainty.get("predicted_alpha_std")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slot": self.slot,
            "trading_day": self.trading_day,
            "champion": self.champion,
            "feature_version": self.feature_version,
            "source_key": self.source_key,
            "predicted_alpha": dict(self.predicted_alpha),
            **self.uncertainty,
        }

    def to_bytes(self) -> bytes:
        """The exact bytes written to the store — sorted and indented, so a
        republication of an unchanged cross-section is byte-identical and a
        digest comparison means what it looks like it means."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True).encode("utf-8")


def _validate(payload: dict[str, Any]) -> PredictionsFeedDocument:
    try:
        return PredictionsFeedDocument.model_validate(payload)
    except ValidationError as exc:
        detail = "\n".join(
            f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise PredictionsFeedContractError(
            f"predictions feed does not conform to {PREDICTIONS_FEED_SCHEMA_VERSION}:\n{detail}"
        ) from exc


def publish_predictions_feed(
    store: Store,
    *,
    trading_day: str,
    slot: str = PREDICTIONS_FEED_SLOT,
    ctx: Any = None,
) -> str | None:
    """Republish the champion's cross-section at the trader's key; return it.

    Returns the key written, or ``None`` when the slot has no champion
    pointer and therefore owes no feed.

    Given a ``ctx``, the write goes through
    :meth:`~crucible.runner.RunContext.record_output`, so the feed enters the
    run manifest's ``outputs[]`` and `crucible explain` can name the run that
    served the trader — the same reason `write_arm_predictions` takes one.
    Without a ``ctx`` it is a plain store write, which is what a replay or an
    operator republication is.

    Raises :class:`~crucible.champion.ChampionUnusableError` when a pointer
    exists and the reader refuses it, and
    :class:`~crucible.slots.cycle.MissingArtifactError` when the pointer names
    an arm with no `arm_predictions.v1` document for ``trading_day``.

    **Validated on the way out as well as on the way in.** The source document
    is checked against its own schema by
    :func:`~crucible.slots.inputs.read_arm_predictions`'s reader, and the feed
    this function builds is checked against `predictions_feed.v1` before the
    PUT — a producer able to emit a non-conformant feed would defeat the
    schema, and the trader would be the thing that discovered it, at market
    open, with money.
    """
    try:
        pointer = read_champion(store, slot)
    except KeyError:
        # No champion, no feed owed. See the module docstring: raising here
        # would fail every promote run taken before the slot's first
        # promotion, which is every promote run the M slot has ever had.
        return None
    return _publish(store, arm_id=pointer.arm_id, trading_day=trading_day, slot=slot, ctx=ctx)


def publish_promoted_feed(
    store: Store,
    *,
    pointer: ChampionPointer,
    ctx: Any = None,
) -> str:
    """Republish the feed for a pointer `promote` has JUST written; return the key.

    `crucible promote --slot m` moves the pointer AFTER `experiment.run[m]`
    already published ``predictions/{as_of}.json`` for the PREVIOUS champion
    — so a promotion that did not also republish left the two halves of the
    trader contract naming two different arms, and the trader's resolver
    refuses exactly that disagreement (2026-10-05: the 2026-10-02 M promotion).

    Why this does not go through :func:`publish_predictions_feed`: that
    function resolves the pointer with :func:`~crucible.champion.read_champion`,
    whose producing-run gate requires the pointer's manifest to read ``ok``.
    For a pointer written by the promote run in flight, that manifest IS this
    run's, and `crucible.runner.run_job` writes it only when the job returns —
    so the gate cannot pass from inside the job that is satisfying it. It is
    not bypassed for the trader: the trader still resolves the pointer
    through `read_champion`, so a promote run that fails after this write
    leaves a pointer the reader refuses, and the feed is never served alone.
    Everything else — the source document's existence, schema, arm, session
    and non-empty cross-section, and the outgoing `predictions_feed.v1`
    schema — is the same code path :func:`publish_predictions_feed` runs.
    """
    if pointer.slot != PREDICTIONS_FEED_SLOT:
        raise ValueError(
            f"slot {pointer.slot!r} serves no predictions feed; only slot "
            f"{PREDICTIONS_FEED_SLOT!r} does (R and U serve under their own keys, S a book)"
        )
    return _publish(
        store,
        arm_id=pointer.arm_id,
        trading_day=pointer.as_of,
        slot=pointer.slot,
        ctx=ctx,
    )


def servable_source(store: Store, *, arm_id: str, trading_day: str) -> str:
    """The `arm_predictions.v1` key the feed WOULD republish for ``arm_id``.

    Raises exactly what a publication for ``arm_id`` on ``trading_day`` would
    raise — :class:`~crucible.slots.cycle.MissingArtifactError` when the arm
    wrote no cross-section for the session, and
    :class:`PredictionsFeedContractError` when it wrote one the serving path
    refuses — so a caller deciding whether an arm CAN serve a session asks the
    serving path itself rather than a second, drift-prone copy of its checks.
    `crucible.slots.model.grade` reads this as the ``servable_as_of`` serving
    precondition, which is what keeps `promote` from seating an arm that has
    nothing to serve (2026-10-05).
    """
    source_key, _ = _load_source(store, arm_id=arm_id, trading_day=trading_day)
    return source_key


def _load_source(
    store: Store, *, arm_id: str, trading_day: str, named_by: str | None = None
) -> tuple[str, dict[str, Any]]:
    # Imported here, not at module scope: `crucible.slots.cycle` imports the
    # grading stack, and `crucible.promote` — this module's caller — is on
    # the import path of the CLI's fastest job. A local import keeps the
    # error TYPE shared without making every promote run pay for the loop.
    from crucible.slots.cycle import MissingArtifactError

    source_key = arm_predictions_key(arm_id, trading_day)
    try:
        payload = store.get_bytes(source_key)
    except KeyError as exc:
        subject = f"{named_by} names {arm_id!r}, which" if named_by else f"arm {arm_id!r}"
        raise MissingArtifactError(
            f"{subject} wrote no prediction cross-section for {trading_day} "
            f"({source_key!r} is not in the store). The serving path resolves the "
            "pointer — it never imports a ranking function directly — so a pointer "
            "to an arm that did not produce means production has no feed today."
        ) from exc

    try:
        document = load_document_bytes(source_key, payload)
    except UnreadableDocumentError as exc:
        raise PredictionsFeedContractError(
            f"{source_key!r} is not readable JSON and is the document the trader's "
            f"feed for {trading_day} would republish: {exc}"
        ) from exc

    _assert_source_is_this_session(document, source_key, arm_id, trading_day)
    return source_key, document


def _publish(store: Store, *, arm_id: str, trading_day: str, slot: str, ctx: Any) -> str:
    source_key, document = _load_source(
        store,
        arm_id=arm_id,
        trading_day=trading_day,
        named_by=f"the champion pointer for slot {slot!r}",
    )

    feed = PredictionsFeed(
        slot=slot,
        trading_day=trading_day,
        champion=arm_id,
        feature_version=str(document["feature_version"]),
        source_key=source_key,
        predicted_alpha={str(k): float(v) for k, v in document["predicted_alpha"].items()},
        uncertainty=_uncertainty_fields(document),
    )
    _validate(feed.to_dict())
    key = predictions_key(trading_day)
    if ctx is not None:
        ctx.record_output(key, feed.to_bytes(), schema_version=PREDICTIONS_FEED_SCHEMA_VERSION)
    else:
        store.put_bytes(key, feed.to_bytes())
    return key


#: The source-document keys the feed republishes beside `predicted_alpha`.
_UNCERTAINTY_KEYS: tuple[str, ...] = (
    *PREDICTION_STD_FIELDS,
    "predicted_alpha_std_method",
    "predicted_alpha_std_note",
)


def _uncertainty_fields(document: dict[str, Any]) -> dict[str, Any]:
    """The source's std fields, verbatim, for whichever of them it carries.

    `alpha-engine-config-I11791`. A republication, like everything else here:
    the std the trader reads is the std the champion's produce run wrote and
    its grade calibrated, never one this module derives.
    """
    out: dict[str, Any] = {}
    for key in _UNCERTAINTY_KEYS:
        if key not in document:
            continue
        value = document[key]
        out[key] = (
            {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value
        )
    return out


def _assert_source_is_this_session(
    document: dict[str, Any], source_key: str, arm_id: str, trading_day: str
) -> None:
    """The point-in-time guard, restated on the serving path.

    `crucible.slots.inputs.read_arm_predictions` makes the same three checks
    for a STACKED ARM's read. They are made again here rather than that
    function being called, because its contract is a training input and this
    one is the money path: the two must be able to diverge (this one will
    grow a digest check when the money-path chain lands) without either
    quietly relaxing the other. A cross-section from another session is
    look-ahead if it is later and stale if it is earlier, and on the serving
    path neither is a degraded feed — both are a refusal.
    """
    version = document.get("schema_version")
    if version != ARM_PREDICTIONS_SCHEMA_VERSION:
        raise PredictionsFeedContractError(
            f"{source_key!r} declares schema_version {version!r}; the serving path "
            f"republishes {ARM_PREDICTIONS_SCHEMA_VERSION!r} only. A feed built from a "
            "document shape this producer does not understand would hand the trader "
            "fields nobody checked."
        )
    if document.get("arm_id") != arm_id:
        raise PredictionsFeedContractError(
            f"{source_key!r} carries arm_id {document.get('arm_id')!r} but sits under "
            f"the key for {arm_id!r}. A misfiled cross-section served here trades "
            "another model's opinion under the champion's name."
        )
    if document.get("trading_day") != trading_day:
        raise PredictionsFeedContractError(
            f"{source_key!r} carries trading_day {document.get('trading_day')!r}, not "
            f"{trading_day!r}. Serving it would size today's book on another session's "
            "cross-section — look-ahead if it is later, stale if it is earlier."
        )
    alpha = document.get("predicted_alpha")
    if not isinstance(alpha, dict) or not alpha:
        raise PredictionsFeedContractError(
            f"{source_key!r} carries no predicted_alpha names. An empty cross-section "
            "is not an empty opinion; it is a producer that failed and wrote anyway, "
            "and the trader must page rather than flatten the book on it."
        )


def read_predictions_feed(store: Store, trading_day: str) -> PredictionsFeed:
    """The CONSUMER half — what the trader calls.

    This function is the contract's reference implementation and is what the
    trader repo imports rather than re-deriving (fleet M0 discipline: a
    producer and a consumer at birth, and the consumer in the consuming
    repo's test suite). It is deliberately in the PUBLIC harness repo even
    though its only caller is private: a second implementation of the trader
    must be able to consume the feed from the schema and this reader alone.

    Raises :class:`KeyError` when no feed exists for ``trading_day``, and
    :class:`PredictionsFeedContractError` when one exists and is malformed.
    A trader must distinguish "the harness has not published yet" (wait, then
    page on the absence deadline) from "today's feed is corrupt" (page now,
    trade nothing).
    """
    key = predictions_key(trading_day)
    payload = store.get_bytes(key)  # KeyError when absent, by the store's contract
    try:
        raw = load_document_bytes(key, payload)
    except UnreadableDocumentError as exc:
        raise PredictionsFeedContractError(f"{key!r} is not readable JSON: {exc}") from exc
    document = _validate(raw)
    if document.trading_day != trading_day:
        raise PredictionsFeedContractError(
            f"{key!r} carries trading_day {document.trading_day!r}, not {trading_day!r}. "
            "A feed misfiled under another session's key is refused rather than served: "
            "the key and the body are two statements of the same fact and a trader that "
            "trusted the key alone would size today on another day's cross-section."
        )
    return PredictionsFeed(
        slot=document.slot,
        trading_day=document.trading_day,
        champion=document.champion,
        feature_version=document.feature_version,
        source_key=document.source_key,
        predicted_alpha=dict(document.predicted_alpha),
        schema_version=document.schema_version,
        uncertainty={
            key: (dict(value) if isinstance(value, dict) else value)
            for key in _UNCERTAINTY_KEYS
            if (value := getattr(document, key)) is not None
        },
    )
