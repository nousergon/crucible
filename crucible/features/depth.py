"""Whether the live feature layer is the deepest one in the store.

Normative source: `alpha-engine-config-I10498` deliverable 2.

The feature layer is content-addressed: `feature_version()` hashes the whole
`registry.py::CATALOG` (see `crucible.features.registry`), so a catalog edit
that changes what a column means — intentionally or not — writes to a NEW
`features/{version}/` prefix rather than overwriting the layer an earlier
verdict was computed from. That is a deliberate safety property. It has an
exact, symmetric failure mode: every reader resolves `feature_version()`
first, so if the code moves on to a new version while the expensive backfill
stays behind at the old one, **nothing downstream can tell** — the layer
looks exactly like one that was never built, indistinguishable from the
inside.

`-I10498` is that failure, measured: `feature_version()` -> `v6df3c0a27b70`
(8 objects, 7 sessions) while `vf795db2b5049` (532 objects, 532 sessions)
sits one prefix over, unreachable by any consumer.

This module is a pure comparison over what :meth:`Store.list_keys` already
returns — it lists `features/`, counts the parquet objects filed under each
version prefix, and compares the live version's count against the deepest
version present. It never calls out to AWS on its own account and never
raises for an ordinary listing outcome; a genuine listing failure (a denied
credential, a store that cannot be reached) propagates to the caller exactly
as any other :class:`Store` read would, so the caller — `crucible.board`,
here — decides how a read failure renders, the same posture every other
board reader takes (`UNMEASURABLE`, never folded into a false green or a
crash).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from crucible.features.registry import feature_version
from crucible.store import Store

__all__ = [
    "FEATURES_PREFIX",
    "PARQUET_SUFFIX",
    "FeatureLayerDepthReading",
    "check_feature_layer_depth",
    "count_session_objects_by_version",
]

#: Where every version of the feature layer is filed, one sub-prefix per
#: `feature_version()` value: `features/{version}/{trading_day}.parquet`,
#: plus a `registry.json` sibling per version that this counts past (it is
#: not a session object).
FEATURES_PREFIX = "features/"

#: The one file extension that names a session's worth of data. `registry.json`
#: sits beside them at the same prefix depth and must never be counted as a
#: session — a version with zero real sessions and one registry document
#: would otherwise read as "1 session deep".
PARQUET_SUFFIX = ".parquet"

DepthState = Literal["RED", "GREEN"]


@dataclass(frozen=True)
class FeatureLayerDepthReading:
    """The result of one comparison. `state` is the board's whole verdict;
    every other field is the evidence a reader needs to act on it without
    re-deriving the listing.
    """

    state: DepthState
    detail: str
    live_version: str
    live_count: int
    deepest_version: str | None
    deepest_count: int
    counts: dict[str, int] = field(default_factory=dict)


def count_session_objects_by_version(
    store: Store, *, prefix: str = FEATURES_PREFIX
) -> dict[str, int]:
    """`{version: session object count}` for every version prefix present.

    A version absent from the store entirely — never backfilled, or reclaimed
    — is simply absent from this dict rather than mapped to `0`; the caller
    distinguishes "present with zero sessions" (which cannot happen: a prefix
    with no parquet objects has nothing under it to list) from "the prefix
    does not exist" by membership, not by value.
    """
    counts: dict[str, int] = {}
    for key in store.list_keys(prefix):
        rest = key[len(prefix) :] if key.startswith(prefix) else key
        parts = rest.split("/", 1)
        if len(parts) != 2:
            continue
        version, filename = parts
        if not filename.endswith(PARQUET_SUFFIX):
            continue
        counts[version] = counts.get(version, 0) + 1
    return counts


def check_feature_layer_depth(
    store: Store, *, live_version: str | None = None
) -> FeatureLayerDepthReading:
    """RED when the live version's prefix is shallower than the deepest
    version present, or is absent entirely. GREEN when the live version is
    at least as deep as every other version present, including the case
    where it is the only version present.

    ``live_version`` defaults to `feature_version()` (the code's own current
    catalog hash); a caller may pass a specific value for a fixture or a
    replay, but never for production use — the whole point is comparing
    against what the RUNNING code resolves.
    """
    version = live_version if live_version is not None else feature_version()
    counts = count_session_objects_by_version(store)

    if not counts:
        return FeatureLayerDepthReading(
            state="RED",
            detail=(
                f"no feature layer prefix exists under {FEATURES_PREFIX!r} at all — the "
                f"live feature_version() {version!r} names a prefix that was never built"
            ),
            live_version=version,
            live_count=0,
            deepest_version=None,
            deepest_count=0,
            counts=counts,
        )

    deepest_version, deepest_count = max(counts.items(), key=lambda kv: kv[1])
    live_count = counts.get(version, 0)

    if version not in counts:
        return FeatureLayerDepthReading(
            state="RED",
            detail=(
                f"the live feature_version() {version!r} names a prefix absent from the "
                f"store entirely; the deepest version present is {deepest_version!r} with "
                f"{deepest_count} session object(s) — a catalog edit moved the "
                "content-addressed hash and left a built backfill unreachable at its old "
                "prefix"
            ),
            live_version=version,
            live_count=0,
            deepest_version=deepest_version,
            deepest_count=deepest_count,
            counts=counts,
        )

    if live_count < deepest_count:
        return FeatureLayerDepthReading(
            state="RED",
            detail=(
                f"the live feature_version() {version!r} holds {live_count} session "
                f"object(s), fewer than {deepest_version!r}'s {deepest_count} — every "
                "reader resolves feature_version() first, so the system sees the "
                f"shallower {live_count}-session layer regardless of what the deeper "
                "backfill contains"
            ),
            live_version=version,
            live_count=live_count,
            deepest_version=deepest_version,
            deepest_count=deepest_count,
            counts=counts,
        )

    return FeatureLayerDepthReading(
        state="GREEN",
        detail=(
            f"the live feature_version() {version!r} holds {live_count} session "
            f"object(s), at least as many as any other version present (deepest: "
            f"{deepest_version!r} with {deepest_count})"
        ),
        live_version=version,
        live_count=live_count,
        deepest_version=deepest_version,
        deepest_count=deepest_count,
        counts=counts,
    )
