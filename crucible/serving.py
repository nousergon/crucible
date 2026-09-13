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
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from crucible.champion import read_champion
from crucible.documents import UnreadableDocumentError, load_document_bytes
from crucible.keys import arm_predictions_key, predictions_key
from crucible.models import PredictionsFeedDocument
from crucible.slots.inputs import ARM_PREDICTIONS_SCHEMA_VERSION
from crucible.store import Store

__all__ = [
    "PREDICTIONS_FEED_SCHEMA_VERSION",
    "PREDICTIONS_FEED_SLOT",
    "PredictionsFeed",
    "PredictionsFeedContractError",
    "publish_predictions_feed",
    "read_predictions_feed",
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slot": self.slot,
            "trading_day": self.trading_day,
            "champion": self.champion,
            "feature_version": self.feature_version,
            "source_key": self.source_key,
            "predicted_alpha": dict(self.predicted_alpha),
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
) -> str | None:
    """Republish the champion's cross-section at the trader's key; return it.

    Returns the key written, or ``None`` when the slot has no champion
    pointer and therefore owes no feed.

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
    # Imported here, not at module scope: `crucible.slots.cycle` imports the
    # grading stack, and `crucible.promote` — this module's caller — is on
    # the import path of the CLI's fastest job. A local import keeps the
    # error TYPE shared without making every promote run pay for the loop.
    from crucible.slots.cycle import MissingArtifactError

    try:
        pointer = read_champion(store, slot)
    except KeyError:
        # No champion, no feed owed. See the module docstring: raising here
        # would fail every promote run taken before the slot's first
        # promotion, which is every promote run the M slot has ever had.
        return None

    source_key = arm_predictions_key(pointer.arm_id, trading_day)
    try:
        payload = store.get_bytes(source_key)
    except KeyError as exc:
        raise MissingArtifactError(
            f"the champion pointer for slot {slot!r} names {pointer.arm_id!r}, which "
            f"wrote no prediction cross-section for {trading_day} ({source_key!r} is "
            "not in the store). The serving path resolves the pointer — it never "
            "imports a ranking function directly — so a pointer to an arm that did "
            "not produce means production has no feed today."
        ) from exc

    try:
        document = load_document_bytes(source_key, payload)
    except UnreadableDocumentError as exc:
        raise PredictionsFeedContractError(
            f"{source_key!r} is not readable JSON and is the document the trader's "
            f"feed for {trading_day} would republish: {exc}"
        ) from exc

    _assert_source_is_this_session(document, source_key, pointer.arm_id, trading_day)

    feed = PredictionsFeed(
        slot=slot,
        trading_day=trading_day,
        champion=pointer.arm_id,
        feature_version=str(document["feature_version"]),
        source_key=source_key,
        predicted_alpha={str(k): float(v) for k, v in document["predicted_alpha"].items()},
    )
    _validate(feed.to_dict())
    key = predictions_key(trading_day)
    store.put_bytes(key, feed.to_bytes())
    return key


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
    )
