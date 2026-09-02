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

import re

from crucible.calendar import assert_trading_day

__all__ = [
    "ARM_SEGMENT_SEPARATOR",
    "arena_cycle_key",
    "arm_id_from_segment",
    "arm_key_segment",
    "arm_predictions_key",
    "arm_register_key",
    "arm_series_key",
    "attribution_key",
    "champion_key",
    "coverage_key",
    "cross_section_key",
    "cross_section_settled_key",
    "data_panel_key",
    "experiments_key",
    "feature_registry_key",
    "features_key",
    "ledger_key",
    "manifest_key",
    "manifest_prefix",
    "retirement_log_key",
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


def arm_predictions_key(arm_id: str, trading_day: str) -> str:
    """What ONE arm predicted on ONE trading day.

    Per-arm, not per-slot: `predictions/{trading_day}.json` is the *champion's*
    serving feed, and a stacked arm reading that would depend on whichever arm
    holds the pointer — a base model that silently changes identity between
    two cycles, and a self-reference the moment the stacked arm won the slot.
    """
    return f"predictions/{arm_key_segment(arm_id)}/{trading_day}.json"


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


#: A discriminator is a path segment, not free text: it must round-trip
#: through every path-shaped tool a key passes through (a `LocalStore` path,
#: `aws s3 cp`, a URL) the same way :func:`arm_key_segment` protects arm ids.
#: Slot letters (`u`/`r`/`m`/`s`) and ISO calendar dates both already satisfy
#: this, so no translation table is needed — only a refusal of anything that
#: would not.
_DISCRIMINATOR_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def manifest_key(job: str, trading_day: str, *, discriminator: str | None = None) -> str:
    """The store key a manifest is written under.

    The trading day is the key (§4.12). This function is the single place
    that shape is expressed, so the trading-day contract test has one thing
    to walk.

    ``discriminator`` distinguishes multiple manifests a single job
    legitimately writes for one trading day — the slot letter for
    `experiment.run`/`experiment.grade` (four slots, one job name, one
    trading day, four writers), or a firing's own `calendar_date` for a job
    like `alerts.sweep` that runs more than once between two trading-day
    rollovers. Omitted (the default) for every job that writes at most one
    manifest per trading day, which keeps the key shape unchanged for every
    existing caller (alpha-engine-config-I9781).

    The alternative weighed in I9781 — folding the slot into the job name
    itself (`experiment.run:r`) — was rejected: `job` is a closed enum the
    schema validates and `components.yaml` keys its one-row-per-job registry
    off it, so multiplying it by slot would multiply the registry too. A
    discriminator is an orthogonal path segment instead, so `job` keeps
    meaning "which of the thirteen CLI jobs wrote this" and nothing else.

    ``discriminator`` is never an arm id, so it does not go through
    :func:`arm_key_segment` — it is validated directly against a plain
    path-segment charset instead.
    """
    if not job:
        raise ValueError("job must be non-empty")
    if not trading_day:
        raise ValueError("trading_day must be non-empty")
    if discriminator is None:
        return f"runs/{job}/{trading_day}/run.json"
    if not _DISCRIMINATOR_RE.match(discriminator):
        raise ValueError(
            f"discriminator {discriminator!r} must be 1-64 characters of "
            "[A-Za-z0-9_.-] — it is a path segment, and this is the one place "
            "that is enforced so a future writer cannot orphan a manifest under "
            "a key no path-shaped tool can address."
        )
    return f"runs/{job}/{trading_day}/{discriminator}/run.json"


def manifest_prefix(job: str, trading_day: str) -> str:
    """The prefix under which every manifest for ``job`` on ``trading_day``
    lives, discriminated or not.

    `manifest_key(job, trading_day)` (bare) and
    `manifest_key(job, trading_day, discriminator=d)` (any ``d``) both start
    with this prefix, so a reader that does not know in advance whether a
    job writes one manifest or several per trading day — `crucible.alerts`,
    chiefly — lists this prefix rather than guessing a discriminator.
    """
    if not job:
        raise ValueError("job must be non-empty")
    if not trading_day:
        raise ValueError("trading_day must be non-empty")
    return f"runs/{job}/{trading_day}/"


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


def cross_section_key(arm_id: str, trading_day: str) -> str:
    """The full scored cross-section an arm produced on ``trading_day``.

    Lives beside `shadow.json` under the same `experiments/{arm}/{day}/` prefix.
    """
    return f"experiments/{arm_key_segment(arm_id)}/{trading_day}/cross_section.json"


def cross_section_settled_key(arm_id: str, trading_day: str) -> str:
    """The same cross-section, joined against the realized forward return
    once ``trading_day``'s horizon has settled. Never written before then."""
    return f"experiments/{arm_key_segment(arm_id)}/{trading_day}/cross_section_settled.json"


def retirement_log_key(slot: str) -> str:
    """The append-only retirement event log for ``slot``.

    Dateless by design: it is the log, not a per-cycle artifact, and the §4.12
    key walk skips keys with no date component rather than requiring a
    trading day of something that spans all of them.
    """
    return f"retirements/{slot}/events.jsonl"


def experiments_key(trading_day: str) -> str:
    """The generated `EXPERIMENTS` feed for one trading day.

    Plan §9.1: "`EXPERIMENTS.md` entry is generated from the register event,
    never hand-written." A negative result that only ever existed in a
    private doc someone remembered to update is not a record.
    """
    return f"experiments/{trading_day}/events.jsonl"


def arm_series_key(slot: str, arm_id: str) -> str:
    """One arm's per-date score series, as produced by `experiment.grade`.

    The scores are already expressed against the SLOT's benchmark: the arena
    never applies a benchmark, because the correct one is a per-slot fact
    (policy §4). A series written against the wrong benchmark is therefore
    caught at the grader, not here.
    """
    return f"scores/{slot}/{arm_key_segment(arm_id)}/series.json"


def arena_cycle_key(slot: str, trading_day: str) -> str:
    """The §4.4 / policy §11 durable cycle artifact, one per slot per cycle.

    The trading day is the key (§4.12); the slot is above it so that a
    slot's whole history lists under one prefix, which is what the console's
    "cycles since the pointer last moved" panel walks.
    """
    if not slot:
        raise ValueError("slot must be non-empty")
    assert_trading_day(trading_day, context=f"arena_cycle key for slot {slot!r}")
    return f"arena/{slot}/{trading_day}/arena_cycle.json"


def champion_key(slot: str) -> str:
    """The store key for ``slot``'s pointer. The trader's whole read surface.

    Dateless by design — it is a pointer, not a record.
    """
    if slot not in ("u", "r", "m", "s"):
        raise KeyError(f"unknown slot {slot!r}; the four slots are ['m', 'r', 's', 'u']")
    return f"champions/{slot}/current.json"


# -- slot outputs the rest of the system consumes ---------------------------


def attribution_key(trading_day: str) -> str:
    """The week's report card. Keyed by trading day like everything else."""
    return f"report/{trading_day}/attribution.json"


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
