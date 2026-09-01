"""Every store key shape, in one place.

Normative source: plan §4.12 — the key of every artifact is an NYSE trading
day — and the M0 contract discipline: a key format restated at each call site
is a contract restated fifty times, and one of them has already drifted.

`crucible.store.Store.assert_keys_bind_to_trading_days` walks the store and
refuses any key whose date component is not a session. That walk is the
enforcement; this module is the single producer, so there is exactly one
thing for the walk to be right about.

**Arm ids are not key-safe as written.** `nousergon_lib.arena.derive_arm_id`
returns ``{slot}:{name}:{spec_hash}``, and a colon inside an S3 key is legal
but breaks every path-shaped tool that reads one (a local `LocalStore` path,
`aws s3 cp`, a URL). :func:`arm_key_segment` is the one translation, and
:func:`arm_id_from_segment` is its inverse — round-tripping is a test, so a
future change to the separator cannot orphan the artifacts already written
under the old one without failing.
"""

from __future__ import annotations

__all__ = [
    "ARM_SEGMENT_SEPARATOR",
    "arena_cycle_key",
    "arm_id_from_segment",
    "arm_key_segment",
    "arm_register_key",
    "arm_series_key",
    "champion_key",
    "coverage_key",
    "data_panel_key",
    "feature_registry_key",
    "features_key",
    "ledger_key",
    "shadow_key",
    "signals_key",
    "strategy_arm_key",
    "universe_members_key",
    "verdict_key",
]

#: `:` is legal in an S3 key and hostile in every path-shaped tool that
#: reads one. `~` is legal in both and appears in no arm name.
ARM_SEGMENT_SEPARATOR = "~"


def arm_key_segment(arm_id: str) -> str:
    """`u:momentum_sleeve:ab12cd` -> `u~momentum_sleeve~ab12cd`."""
    if not arm_id:
        raise ValueError("arm_id must be non-empty")
    if ARM_SEGMENT_SEPARATOR in arm_id:
        raise ValueError(
            f"arm_id {arm_id!r} already contains {ARM_SEGMENT_SEPARATOR!r}, which is "
            "this module's separator; the round trip would not be invertible and two "
            "distinct arms could collide onto one key."
        )
    return arm_id.replace(":", ARM_SEGMENT_SEPARATOR)


def arm_id_from_segment(segment: str) -> str:
    """The inverse of :func:`arm_key_segment`."""
    if not segment:
        raise ValueError("segment must be non-empty")
    return segment.replace(ARM_SEGMENT_SEPARATOR, ":")


# -- data layer -------------------------------------------------------------


def data_panel_key(trading_day: str) -> str:
    """The day's compiled price panel."""
    return f"data/{trading_day}/panel.parquet"


def coverage_key(trading_day: str) -> str:
    """The day's coverage record: what was read, from where, and how much."""
    return f"data/{trading_day}/coverage.json"


# -- feature layer ----------------------------------------------------------


def features_key(version: str, trading_day: str) -> str:
    """§10.4: one materialized, hashed layer between data and slots."""
    return f"features/{version}/{trading_day}.parquet"


def feature_registry_key(version: str) -> str:
    """The registry that names, units and dates every column of a version."""
    return f"features/{version}/registry.json"


# -- arms and slots ---------------------------------------------------------


def strategy_arm_key(slot: str, name: str) -> str:
    """An arm recipe inside the synced strategy tree (plan §4.11)."""
    return f"strategy/current/arms/{slot}/{name}.yaml"


def arm_register_key(slot: str) -> str:
    """The append-only arm event log for one slot, folded to state on read."""
    return f"arms/{slot}/register.jsonl"


def shadow_key(arm_id: str, trading_day: str) -> str:
    """What an arm SELECTED on a trading day, before any outcome is known.

    Written by `experiment.run` at the decision date. Separate from the
    verdict because the verdict cannot exist until the horizon settles, and
    conflating them is how a look-ahead gets written into a production
    artifact.
    """
    return f"experiments/{arm_key_segment(arm_id)}/{trading_day}/shadow.json"


def verdict_key(arm_id: str, trading_day: str) -> str:
    """What that selection turned out to be worth, once the horizon settled."""
    return f"experiments/{arm_key_segment(arm_id)}/{trading_day}/verdict.json"


def arm_series_key(slot: str, arm_id: str) -> str:
    """One arm's per-date score series, as produced by `experiment.grade`.

    The scores are already expressed against the SLOT's benchmark: the arena
    never applies a benchmark, because the correct one is a per-slot fact
    (policy §4). A series written against the wrong benchmark is therefore
    caught at the grader, not here.
    """
    return f"scores/{slot}/{arm_key_segment(arm_id)}/series.json"


def arena_cycle_key(slot: str, trading_day: str) -> str:
    """The §4.4 / policy §11 durable cycle artifact, one per slot per cycle."""
    return f"arena/{slot}/{trading_day}/arena_cycle.json"


def champion_key(slot: str) -> str:
    """The serving pointer. Dateless by design — it is a pointer, not a record."""
    return f"champions/{slot}/current.json"


# -- slot outputs the rest of the system consumes ---------------------------


def universe_members_key(trading_day: str) -> str:
    """The U champion's feed: which names reach the predictor."""
    return f"universe/{trading_day}/members.json"


def signals_key(trading_day: str) -> str:
    """The R champion's feed: how names were scored."""
    return f"signals/{trading_day}/signals.json"


# -- fleet ledger -----------------------------------------------------------


def ledger_key() -> str:
    """§9.1: the fleet trial ledger DSR's `n_trials` deflation reads.

    Dateless: it is one append-only log across every slot and every cycle,
    and a per-date ledger would make "everything ever tried" a join.
    """
    return "ledger/trials.jsonl"
