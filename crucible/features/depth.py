"""Whether the live feature layer is the deepest one in the store, and
whether what it holds measured anything.

Normative source: `alpha-engine-config-I10498` deliverable 2 (depth);
`alpha-engine-config-I10693` (completeness).

**Depth counts objects, never contents** — that is this module's own
documented scope, and it has an exact failure mode of its own:
`alpha-engine-config-I10688` measured `features/v6df3c0a27b70` at 536 session
objects, the deepest version present, so `check_feature_layer_depth` read
GREEN, while `residual_momentum_252d_skip21d_ratio` and its z-score were
`NaN` for 903 of 903 tickers on all 536 sessions — the producer's trailing
panel was 275 sessions and the column needs 313
(`crucible.features.compute._RESIDUAL_MOMENTUM_DEPTH_TRADING_DAYS`). The
producer bug is fixed (`crucible-PR266`: `min_panel_trading_days`,
`PanelDepthError`), but a layer compiled before that guard landed — every
object in the store at the time of measurement — still reads GREEN on depth
alone, and nothing about depth catches the *next* column shaped the same
way. `check_feature_layer_completeness` is the sibling reading that closes
that blindness: it reads the live version's most recent session and grades
each catalogue column on what fraction of its rows are null, RED only when a
column is null on **every** row — `catalog_column_depths()` already explains
an ordinary head-of-history null (a ticker younger than a column's declared
depth), so the only ratio no per-ticker depth can explain is every ticker at
once.

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

from crucible.features.compute import catalog_column_depths, read_features
from crucible.features.registry import feature_names, feature_version
from crucible.keys import features_key, features_prefix
from crucible.store import Store

__all__ = [
    "FEATURES_PREFIX",
    "PARQUET_SUFFIX",
    "NULL_RATIO_CEILING",
    "FeatureLayerDepthReading",
    "FeatureLayerCompletenessReading",
    "check_feature_layer_depth",
    "check_feature_layer_completeness",
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


#: The null ratio (fraction of rows null on one session) at or above which a
#: catalogue column is graded RED. Named and applied uniformly to every
#: catalogue column — never a per-column allowlist of "expected null" names
#: (`crucible/AGENTS.md` rule 4: no suppression collections). A column
#: legitimately null for the few tickers younger than its declared depth
#: (`catalog_column_depths()`) sits far below this ceiling — measured
#: 2026-09-13 at 4 of 903 tickers (0.44%) once the producer was fixed. Half
#: the universe null at once is not a head-of-history shape any per-ticker
#: depth explains; the `-I10688` shape (903 of 903) is the extreme of it.
#: Principle 7: a column that measured nothing for most of the universe is
#: unobserved, not green.
NULL_RATIO_CEILING = 0.5


@dataclass(frozen=True)
class FeatureLayerCompletenessReading:
    """The result of one column-completeness comparison over the live
    version's most recent session. `state` is the board's whole verdict;
    `null_ratios` carries every catalogue column's null fraction so a
    partial degradation is visible on the reading before it reaches
    :data:`NULL_RATIO_CEILING`.
    """

    state: DepthState
    detail: str
    live_version: str
    session: str | None
    null_ratios: dict[str, float] = field(default_factory=dict)
    dead_columns: tuple[str, ...] = field(default_factory=tuple)


def check_feature_layer_completeness(
    store: Store, *, live_version: str | None = None
) -> FeatureLayerCompletenessReading:
    """RED when any catalogue column is null on at least :data:`NULL_RATIO_CEILING`
    of the rows of the live
    version's most recent session parquet; GREEN otherwise.

    Depth (:func:`check_feature_layer_depth`) asks how many sessions exist.
    This asks whether the deepest catalogue column measured anything on the
    most recent one — the two are complementary, never a replacement for
    each other, and both render as separate `crucible board` rows.

    Reads two objects at most (a listing, then one session parquet); never
    calls out to AWS on its own account beyond that and never raises for an
    ordinary outcome. A genuine read failure (a denied credential, a store
    that cannot be reached, a corrupt parquet) propagates to the caller
    exactly as any other :class:`Store` read would, so the caller —
    `crucible.board` — decides how a read failure renders (`UNMEASURABLE`,
    never folded into a false green), the same posture
    :func:`check_feature_layer_depth` takes.
    """
    version = live_version if live_version is not None else feature_version()
    prefix = features_prefix(version)
    sessions = sorted(
        key[len(prefix) : -len(PARQUET_SUFFIX)]
        for key in store.list_keys(prefix)
        if key.endswith(PARQUET_SUFFIX)
    )

    if not sessions:
        return FeatureLayerCompletenessReading(
            state="RED",
            detail=(
                f"no session parquet exists under {prefix!r} — the live "
                f"feature_version() {version!r} has never been built, so no column's "
                "completeness can be read"
            ),
            live_version=version,
            session=None,
        )

    session = sessions[-1]
    key = features_key(version, session)
    frame = read_features(store.get_bytes(key))

    if len(frame) == 0:
        return FeatureLayerCompletenessReading(
            state="RED",
            detail=(f"{key!r} holds zero rows — a session with no tickers measured nothing"),
            live_version=version,
            session=session,
        )

    depths = catalog_column_depths()
    columns = [c for c in feature_names() if c in frame.columns]
    null_ratios = {col: float(frame[col].isna().mean()) for col in columns}
    dead_columns = tuple(
        sorted(col for col, ratio in null_ratios.items() if ratio >= NULL_RATIO_CEILING)
    )

    if dead_columns:
        named = ", ".join(
            f"{col!r} (declared depth {depths.get(col, 'unknown')} session(s))"
            for col in dead_columns
        )
        detail = (
            f"{key!r}: catalogue column(s) null on >= {NULL_RATIO_CEILING:.0%} of "
            f"{len(frame)} row(s) of the most recent session ({session}): {named} — a "
            "column that looks computed and measured nothing for most of the universe, "
            "the avg_volume_20d/residual_momentum class. "
            "catalog_column_depths() explains an ordinary null for one ticker younger "
            "than a column's declared depth; it never explains every ticker null at once"
        )
        state: DepthState = "RED"
    else:
        worst_col, worst_ratio = (
            max(null_ratios.items(), key=lambda kv: kv[1]) if null_ratios else (None, 0.0)
        )
        detail = (
            f"{key!r}: no catalogue column is null on >= {NULL_RATIO_CEILING:.0%} of rows "
            f"of the most recent session ({session}, {len(frame)} row(s)); worst null ratio "
            f"{worst_ratio:.2%} on {worst_col!r}"
        )
        state = "GREEN"

    return FeatureLayerCompletenessReading(
        state=state,
        detail=detail,
        live_version=version,
        session=session,
        null_ratios=null_ratios,
        dead_columns=dead_columns,
    )
