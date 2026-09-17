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

from crucible.documents import UnreadableDocumentError, read_store_document
from crucible.features.compute import catalog_column_depths, read_features
from crucible.features.registry import feature_names, feature_version
from crucible.keys import coverage_key, features_key, features_prefix
from crucible.store import Store

__all__ = [
    "FEATURES_PREFIX",
    "PARQUET_SUFFIX",
    "NULL_RATIO_CEILING",
    "FeatureLayerDepthReading",
    "FeatureLayerCompletenessReading",
    "check_feature_layer_depth",
    "sample_sessions",
    "sample_coverage_sentence",
    "COMPLETENESS_SAMPLE_SIZE",
    "check_feature_layer_completeness",
    "count_session_objects_by_version",
    "FeatureLayerProvenanceReading",
    "check_feature_layer_provenance",
    "PROVENANCE_SAMPLE_SIZE",
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


#: Sessions sampled across the live version, newest and oldest always among
#: them. Reading ONLY the newest session is what let
#: `alpha-engine-config-I10733`'s null band sit unreported: on 2026-09-14 a
#: degenerate `data/inst_ownership/2025Q3/latest.parquet` was live between
#: 19:32Z and 21:23Z, every feature session healed inside that window read it,
#: and `institutional_accumulation_raw` came out null for 903 of 903 tickers
#: on the ~50 sessions of 2025-11-17..2026-01-30. This reading was GREEN over
#: all of it, because the newest session (2026-09-15) was healed afterwards
#: and is fine. A layer is not its last day.
#:
#: 40 over a ~1,180-session layer is a stride near 29, so any contiguous band
#: of 30 sessions or more is certain to be sampled. The number is a
#: cost/coverage trade and the reading STATES it (below) rather than implying
#: the whole layer was read -- a sample reported as a sweep is the same
#: false-green shape one layer up.
COMPLETENESS_SAMPLE_SIZE = 40


@dataclass(frozen=True)
class FeatureLayerCompletenessReading:
    """The result of one column-completeness comparison over a sample of the
    live version's sessions. `state` is the board's whole verdict;
    `null_ratios` carries every catalogue column's null fraction on the WORST
    sampled session, so a partial degradation is visible on the reading
    before it reaches :data:`NULL_RATIO_CEILING`.

    `session` is the session the ratios and the verdict were taken from --
    the newest when everything is clean, the offending one when a column is
    dead somewhere in the layer. `sessions_read` and `sessions_total` are
    what makes the sample honest at the point of reading it.
    """

    state: DepthState
    detail: str
    live_version: str
    session: str | None
    null_ratios: dict[str, float] = field(default_factory=dict)
    dead_columns: tuple[str, ...] = field(default_factory=tuple)
    sessions_read: tuple[str, ...] = field(default_factory=tuple)
    sessions_total: int = 0
    dead_sessions: tuple[str, ...] = field(default_factory=tuple)


def sample_sessions(sessions: list[str], size: int = COMPLETENESS_SAMPLE_SIZE) -> list[str]:
    """`size` sessions spread evenly across `sessions`, newest and oldest
    included, in chronological order and without duplicates.

    Evenly spaced rather than random: a random sample makes the reading
    non-reproducible, so two consecutive board runs can disagree about a
    layer that did not change, and nobody can tell which run was unlucky.
    An even stride has a stated guarantee instead -- any band at least
    `len(sessions) / size` long is hit.
    """
    if size <= 0:
        raise ValueError(f"sample size must be positive; got {size}")
    if len(sessions) <= size:
        return list(sessions)
    step = (len(sessions) - 1) / (size - 1)
    picked = {round(i * step) for i in range(size)}
    picked.add(0)
    picked.add(len(sessions) - 1)
    return [sessions[i] for i in sorted(picked)]


def sample_coverage_sentence(sessions: list[str], sampled: list[str]) -> str:
    """The one sentence every sampled reading ends with: how much of the
    layer was actually read, and the exact length of band the sample could
    still miss.

    DERIVED from the sample actually taken, never from `len // size`: with
    1,181 sessions and 40 samples the stride is 30.3, so the widest gap is 31
    and `len // size` would claim 29 — an under-statement of the blind
    window, which is the one direction this sentence must never be wrong in.

    Shared by :func:`check_feature_layer_completeness` and
    :func:`check_feature_layer_provenance` rather than written twice: two
    readings over the same layer that state their coverage differently is
    how one of them quietly stops being true (`policy-shared-code`, second
    adoption).
    """
    positions = {session_name: index for index, session_name in enumerate(sessions)}
    picked_positions = [positions[candidate] for candidate in sampled]
    widest_gap = max(
        (b - a for a, b in zip(picked_positions, picked_positions[1:], strict=False)),
        default=1,
    )
    return (
        f"{len(sampled)} of {len(sessions)} session(s) sampled evenly across the layer "
        f"({sampled[0]}..{sampled[-1]}); any contiguous band of {widest_gap} session(s) "
        "or more is sampled, a shorter one can sit between two samples unseen"
    )


def check_feature_layer_completeness(
    store: Store, *, live_version: str | None = None
) -> FeatureLayerCompletenessReading:
    """RED when any catalogue column is null on at least :data:`NULL_RATIO_CEILING`
    of the rows of ANY sampled session of the live version; GREEN otherwise.

    Depth (:func:`check_feature_layer_depth`) asks how many sessions exist.
    This asks whether the deepest catalogue column measured anything on the
    most recent one — the two are complementary, never a replacement for
    each other, and both render as separate `crucible board` rows.

    Reads a listing plus :data:`COMPLETENESS_SAMPLE_SIZE` session parquets at
    most -- it used to read exactly one, the newest, and that is the blindness
    `alpha-engine-config-I10733` walked through. Never
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

    sampled = sample_sessions(sessions)
    depths = catalog_column_depths()
    catalogue = feature_names()

    empty_sessions: list[str] = []
    per_session: list[tuple[str, dict[str, float], tuple[str, ...], int]] = []
    for candidate in sampled:
        frame = read_features(store.get_bytes(features_key(version, candidate)))
        if len(frame) == 0:
            empty_sessions.append(candidate)
            continue
        columns = [c for c in catalogue if c in frame.columns]
        ratios = {col: float(frame[col].isna().mean()) for col in columns}
        dead = tuple(sorted(col for col, ratio in ratios.items() if ratio >= NULL_RATIO_CEILING))
        per_session.append((candidate, ratios, dead, len(frame)))

    if empty_sessions:
        # A session with no tickers measured nothing, wherever it sits in the
        # layer. Reported before the column verdict because a zero-row parquet
        # has no column ratios to weigh against anything.
        return FeatureLayerCompletenessReading(
            state="RED",
            detail=(
                f"{len(empty_sessions)} sampled session parquet(s) under {prefix!r} hold "
                f"zero rows: {', '.join(empty_sessions[:4])} — a session with no tickers "
                "measured nothing"
            ),
            live_version=version,
            session=empty_sessions[0],
            sessions_read=tuple(sampled),
            sessions_total=len(sessions),
        )

    coverage = sample_coverage_sentence(sessions, sampled)

    dead_readings = [reading for reading in per_session if reading[2]]
    if dead_readings:
        # The WORST sampled session is the one reported, not the newest: the
        # newest is exactly the session that read green over
        # `alpha-engine-config-I10733`'s band.
        worst_session, null_ratios, dead_columns, row_count = max(
            dead_readings, key=lambda reading: (len(reading[2]), max(reading[1].values()))
        )
        named = ", ".join(
            f"{col!r} (declared depth {depths.get(col, 'unknown')} session(s))"
            for col in dead_columns
        )
        dead_sessions = tuple(reading[0] for reading in dead_readings)
        detail = (
            f"{features_key(version, worst_session)!r}: catalogue column(s) null on >= "
            f"{NULL_RATIO_CEILING:.0%} of {row_count} row(s) of session {worst_session}: "
            f"{named} — a column that looks computed and measured nothing for most of "
            "the universe, the avg_volume_20d/residual_momentum class. "
            "catalog_column_depths() explains an ordinary null for one ticker younger "
            "than a column's declared depth; it never explains every ticker null at once. "
            f"{len(dead_sessions)} of the sampled session(s) read dead: "
            f"{', '.join(dead_sessions[:6])}. {coverage}"
        )
        return FeatureLayerCompletenessReading(
            state="RED",
            detail=detail,
            live_version=version,
            session=worst_session,
            null_ratios=null_ratios,
            dead_columns=dead_columns,
            sessions_read=tuple(sampled),
            sessions_total=len(sessions),
            dead_sessions=dead_sessions,
        )

    newest_session, null_ratios, _, row_count = per_session[-1]
    worst_col, worst_ratio, worst_where = None, 0.0, newest_session
    for candidate, ratios, _, _ in per_session:
        if not ratios:
            continue
        col, ratio = max(ratios.items(), key=lambda kv: kv[1])
        if ratio >= worst_ratio:
            worst_col, worst_ratio, worst_where = col, ratio, candidate
    detail = (
        f"no catalogue column is null on >= {NULL_RATIO_CEILING:.0%} of rows on any "
        f"sampled session of {version!r}; worst null ratio {worst_ratio:.2%} on "
        f"{worst_col!r} (session {worst_where}). {coverage}"
    )
    return FeatureLayerCompletenessReading(
        state="GREEN",
        detail=detail,
        live_version=version,
        session=newest_session,
        null_ratios=null_ratios,
        dead_columns=(),
        sessions_read=tuple(sampled),
        sessions_total=len(sessions),
    )


#: Sessions sampled for the provenance reading. The same stride argument as
#: :data:`COMPLETENESS_SAMPLE_SIZE`, and deliberately the same number: the two
#: readings grade the same layer from two sides, and a reader comparing them
#: should not have to hold two different coverage guarantees in mind. Over a
#: ~1,180-session layer the stride is near 29, so any contiguous band of 30
#: sessions or more is certain to be sampled -- the measured `-I10733` band is
#: 92 and could not hide.
PROVENANCE_SAMPLE_SIZE = COMPLETENESS_SAMPLE_SIZE


@dataclass(frozen=True)
class FeatureLayerProvenanceReading:
    """Which point-in-time source(s) actually compiled the live feature layer.

    `sources` maps each source name found to the sampled sessions that
    carry it, so a mixed layer names its own bands rather than reporting a
    bare count. `missing` is the sampled sessions with no coverage record at
    all -- unobserved, which is its own state and never folded into green.
    """

    state: DepthState
    detail: str
    live_version: str
    expected_source: str
    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing: tuple[str, ...] = field(default_factory=tuple)
    unreadable: tuple[str, ...] = field(default_factory=tuple)
    sessions_read: tuple[str, ...] = field(default_factory=tuple)
    sessions_total: int = 0


def check_feature_layer_provenance(
    store: Store,
    *,
    live_version: str | None = None,
    expected_source: str | None = None,
) -> FeatureLayerProvenanceReading:
    """RED when the live feature layer was not compiled, end to end, by the
    one production point-in-time source; GREEN when every sampled session
    names it.

    Depth asks how many sessions exist. Completeness asks whether their
    columns measured anything. This asks **what produced them** -- and it is
    the reading the other two structurally cannot give, because a session
    compiled from a shallower source is a well-formed parquet with plausible
    columns. It is only wrong relative to its neighbours.

    Measured 2026-09-17 (`alpha-engine-config-I10733`): `v553618c991dd` held
    1,180 sessions of which 92 (2025-07-09..2025-11-14) were compiled by
    `v1-snapshots`, each carrying 11 unmeasured columns and four null
    attractiveness pillars, inside a layer whose every other session used
    `edgar-filing-date`. The cause was a heal chunk that booted a release
    predating the EDGAR switch. `data/{day}/coverage.json` recorded
    `point_in_time.source` correctly for every one of those sessions from the
    day they were written -- the fact was never missing, only unread, which is
    the `-I10733` blindness one level up from the null band itself.

    Keys on the PROPERTY (the layer is single-sourced, by the declared
    production source) rather than on the MECHANISM that produced any one
    band, so a future third source is caught by the same predicate without
    an edit. `expected_source` defaults to
    :data:`crucible.data.point_in_time.PRODUCTION_FUNDAMENTALS_SOURCE`, the
    single declaration of which source production compiles from.

    Reads a listing plus at most :data:`PROVENANCE_SAMPLE_SIZE` small JSON
    objects. A genuine read failure propagates to the caller exactly as in
    the sibling readings, so `crucible.board` renders it `UNMEASURABLE`
    rather than folding it into a false green. A coverage record that is
    ABSENT is not a read failure -- it is a measured fact about that session
    and is reported as `missing`.
    """
    from crucible.data.point_in_time import (  # noqa: PLC0415 - avoids an import cycle
        PRODUCTION_FUNDAMENTALS_SOURCE,
    )

    wanted = expected_source if expected_source is not None else PRODUCTION_FUNDAMENTALS_SOURCE
    version = live_version if live_version is not None else feature_version()
    prefix = features_prefix(version)
    sessions = sorted(
        key[len(prefix) : -len(PARQUET_SUFFIX)]
        for key in store.list_keys(prefix)
        if key.endswith(PARQUET_SUFFIX)
    )

    if not sessions:
        return FeatureLayerProvenanceReading(
            state="RED",
            detail=(
                f"no session parquet exists under {prefix!r} — the live "
                f"feature_version() {version!r} has never been built, so no session's "
                "point-in-time provenance can be read"
            ),
            live_version=version,
            expected_source=wanted,
            sessions_total=0,
        )

    read = sample_sessions(sessions, PROVENANCE_SAMPLE_SIZE)
    found: dict[str, list[str]] = {}
    missing: list[str] = []
    unreadable: list[str] = []
    for day in read:
        key = coverage_key(day)
        # Through the ONE guarded reader (`crucible.documents`), never
        # `json.loads` over `get_bytes` here: three outcomes, not two.
        # ABSENT is a fact about that session (`missing`), UNREADABLE is a
        # fact about that artifact (`unreadable`), and an ACCESS problem is a
        # statement about us — which is the caller's to render, so it is
        # raised rather than graded (`alpha-engine-config-I9931`).
        outcome = read_store_document(store, key)
        if outcome.absent:
            missing.append(day)
            continue
        if outcome.problem is not None:
            if outcome.access_problem:
                raise UnreadableDocumentError(outcome.problem)
            unreadable.append(day)
            continue
        record = outcome.document or {}
        source = (record.get("point_in_time") or {}).get("source")
        found.setdefault(str(source), []).append(day)

    sources = {name: tuple(days) for name, days in sorted(found.items())}
    sampled = sample_coverage_sentence(sessions, read)

    wrong = {name: days for name, days in sources.items() if name != wanted}
    if wrong or missing or unreadable:
        parts: list[str] = []
        for name, days in sorted(wrong.items()):
            parts.append(
                f"{len(days)} sampled session(s) were compiled by {name!r} "
                f"(e.g. {', '.join(days[:6])})"
            )
        if unreadable:
            parts.append(
                f"{len(unreadable)} sampled session(s) carry a coverage record that "
                f"could not be parsed, so what compiled them cannot be read "
                f"(e.g. {', '.join(unreadable[:6])})"
            )
        if missing:
            parts.append(
                f"{len(missing)} sampled session(s) carry no "
                f"{coverage_key('{day}')!r} record at all, so what compiled them cannot "
                f"be read (e.g. {', '.join(missing[:6])})"
            )
        return FeatureLayerProvenanceReading(
            state="RED",
            detail=(
                f"the live feature_version() {version!r} was not compiled end to end by "
                f"the production point-in-time source {wanted!r}: "
                + "; ".join(parts)
                + ". A session compiled by a shallower source is a well-formed parquet "
                "with plausible columns — neither the depth nor the completeness reading "
                "can see it, because it is only wrong relative to its neighbours. " + sampled
            ),
            live_version=version,
            expected_source=wanted,
            sources=sources,
            missing=tuple(missing),
            unreadable=tuple(unreadable),
            sessions_read=tuple(read),
            sessions_total=len(sessions),
        )

    return FeatureLayerProvenanceReading(
        state="GREEN",
        detail=(
            f"every sampled session of the live feature_version() {version!r} was "
            f"compiled by the production point-in-time source {wanted!r}. " + sampled
        ),
        live_version=version,
        expected_source=wanted,
        sources=sources,
        sessions_read=tuple(read),
        sessions_total=len(sessions),
    )
