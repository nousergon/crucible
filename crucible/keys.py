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
from dataclasses import dataclass
from typing import Any

from crucible.calendar import assert_trading_day

__all__ = [
    "ACCEPTANCE_REQUIRED_FIELDS",
    "ALERTS_ROOT",
    "ARM_SEGMENT_SEPARATOR",
    "AcceptanceReading",
    "BOARD_CURRENT_KEY",
    "BOARD_HTML_KEY",
    "CONSOLE_JSON_KEY",
    "CONSOLE_KEY",
    "DRIFT_INPUTS",
    "MANIFEST_BASENAME",
    "POINTER_KEY",
    "RELEASES_ROOT",
    "REVIEWER_PATTERN",
    "RUNS_ROOT",
    "TRADER_PIN_KEY",
    "TRIGGER_RE",
    "TRIGGER_UNKNOWN",
    "acceptance_reading_key",
    "arena_cycle_key",
    "arm_id_from_segment",
    "arm_key_segment",
    "arm_predictions_key",
    "arm_register_key",
    "arm_series_key",
    "attribution_key",
    "board_html_key",
    "board_key",
    "champion_key",
    "closing_record_key",
    "coverage_key",
    "cross_section_key",
    "cross_section_settled_key",
    "data_panel_key",
    "declared_universe_key",
    "drift_input_key",
    "drift_metrics_key",
    "experiments_key",
    "experiments_prefix",
    "feature_registry_key",
    "features_key",
    "features_prefix",
    "gate_key",
    "gate_prefix",
    "heal_key",
    "is_manifest_key",
    "ledger_key",
    "legacy_dead_lambdas_key",
    "legacy_weekly_executions_key",
    "manifest_key",
    "manifest_prefix",
    "migration_key",
    "morning_history_row_key",
    "morning_report_key",
    "morning_trigger_key",
    "morning_update_key",
    "parse_acceptance_reading",
    "parse_bus_key",
    "parse_manifest_key",
    "retirement_log_key",
    "review_key",
    "review_prefix",
    "runs_prefix",
    "shadow_key",
    "signals_key",
    "strategy_arm_key",
    "strategy_arms_prefix",
    "universe_members_key",
    "verdict_key",
]

#: `:` is legal in an S3 key and hostile in every path-shaped tool that
#: reads one. `~` is legal in both and appears in no arm name.
ARM_SEGMENT_SEPARATOR = "~"

#: The root namespace segment every run manifest lives under — the prefix
#: `manifest_key`, `manifest_prefix` and `runs_prefix` all in turn narrow. A
#: caller with no job and no trading day at all — an existential "has
#: ANYTHING ever run" scan (`crucible.explain.load_manifests`,
#: `crucible.llm.week_to_date_llm_spend`, `crucible.console.render.build_page`'s
#: week-cost/deploys loop, `crucible.alerts._week_summary`) — lists this
#: constant directly rather than hardcoding `"runs/"` at the call site.
RUNS_ROOT = "runs/"

#: The basename EVERY run manifest is written under, and the only object under
#: a manifest prefix that is a manifest.
#:
#: A manifest prefix is a namespace, not a manifest list: a job may legitimately
#: file its own evidence beside its manifest, and `report.morning` does exactly
#: that (`morning_report_key` -> `{manifest_prefix}/{calendar_date}/message.txt`,
#: deliberately under the job's own prefix so the delivery needs no second IAM
#: grant). A reader that `json.loads` every key it lists under a manifest prefix
#: therefore parses a plain-text message as a manifest and raises — which is
#: `alpha-engine-config-I9900`, observed live on 2026-09-03: one `message.txt`
#: took `crucible board` down for seven hours, publishing no board and no
#: ladder at all.
#:
#: Named here rather than as a `"run.json"` literal at each filter so
#: `manifest_key`, `parse_manifest_key` and every consumer that narrows a
#: listing agree by construction. Consumers should prefer
#: :func:`is_manifest_key`, which additionally checks the root and the arity.
MANIFEST_BASENAME = "run.json"

#: The root namespace segment every alert bus row lives under, narrowed by
#: `crucible.alerts.bus_key` (an architectural exception — see
#: `tests/test_key_construction_placement.py`). `crucible.alerts.pages_in_window`
#: lists this constant rather than hardcoding `"alerts/"` when it counts every
#: incident across the whole trailing window, not one group.
ALERTS_ROOT = "alerts/"


def parse_bus_key(key: str) -> tuple[str, str] | None:
    """The inverse of `crucible.alerts.bus_key`: ``(trading_day, incident_id)``.

    The CONSTRUCTION of a bus key stays in `crucible.alerts` (it is a function
    of that module's derived `incident_id`, see `bus_key`); the PARSE belongs
    here, with every other key shape, because more than one reader counts the
    bus — `crucible.alerts.pages_in_range` and `crucible.gate`'s phase-2
    ceiling clause — and each one that restated the shape as ``len(parts) !=
    3`` would be a second contract invisible to any grep for a key-shaped
    literal. That is the `parse_manifest_key` failure one namespace over
    (alpha-engine-config-I9879), where an arity restated as an integer
    silently dropped 100% of discriminated manifests.

    Returns ``None`` for any key that is not a bus row under this shape, so
    the caller decides whether an unrecognised key under `alerts/` is an error
    or simply not of interest.
    """
    if not key.startswith(ALERTS_ROOT) or not key.endswith(".json"):
        return None
    parts = key.split("/")
    if len(parts) != 3:
        return None
    _, trading_day, filename = parts
    return trading_day, filename[: -len(".json")]


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


def features_prefix(version: str) -> str:
    """The prefix under which every trading day's compiled feature layer for
    ``version`` lives.

    `features_key(version, trading_day)` for any ``trading_day`` starts with
    this prefix — a reader that needs to know which days a version has been
    compiled for (`crucible.slots.model.FeatureLayer._sessions`, listing the
    store rather than deriving a date range) lists this prefix instead of
    restating its shape.
    """
    if not version:
        raise ValueError(
            "version must be non-empty — a blank version would list every version's "
            "features under one empty-segment prefix, and `store.list_keys('features//')` "
            "returning nothing reads as 'no data' rather than the caller's own bug."
        )
    return f"features/{version}/"


# -- arms and slots ---------------------------------------------------------


def strategy_arm_key(slot: str, name: str) -> str:
    """An arm recipe inside the synced strategy tree (plan §4.11)."""
    return f"strategy/current/arms/{slot}/{name}.yaml"


def arm_register_key(slot: str) -> str:
    """The append-only arm event log for one slot, folded to state on read."""
    return f"arms/{slot}/register.jsonl"


def strategy_arms_prefix(slot: str) -> str:
    """The prefix under which every arm recipe for ``slot`` lives, synced
    into the store.

    `strategy_arm_key(slot, name)` for any ``name`` starts with this prefix —
    `crucible.slots.arms.load_arm_specs` lists it (rather than restating the
    shape) when reading from the spot-box synced tree instead of a local
    `CRUCIBLE_STRATEGY_DIR` checkout.
    """
    if not slot:
        raise ValueError(
            "slot must be non-empty — a blank slot would list every slot's arms under "
            "one empty-segment prefix, and `store.list_keys('strategy/current/arms//')` "
            "returning nothing reads as 'no data' rather than the caller's own bug."
        )
    return f"strategy/current/arms/{slot}/"


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
        return f"runs/{job}/{trading_day}/{MANIFEST_BASENAME}"
    if not _DISCRIMINATOR_RE.match(discriminator):
        raise ValueError(
            f"discriminator {discriminator!r} must be 1-64 characters of "
            "[A-Za-z0-9_.-] — it is a path segment, and this is the one place "
            "that is enforced so a future writer cannot orphan a manifest under "
            "a key no path-shaped tool can address."
        )
    return f"runs/{job}/{trading_day}/{discriminator}/{MANIFEST_BASENAME}"


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


def parse_manifest_key(key: str) -> tuple[str, str, str | None] | None:
    """The inverse of :func:`manifest_key`: ``(job, trading_day, discriminator)``.

    A reader that lists :data:`RUNS_ROOT` (or any manifest prefix) and needs
    to know what each returned key names — `crucible.console.render`'s
    weekly cost roll-up and `crucible.alerts._week_summary`, chiefly — parses
    through this function rather than restating the shape as a positional
    ``parts[i]`` or an arity check. Both `manifest_key(job, trading_day)`
    (four segments) and `manifest_key(job, trading_day, discriminator=d)`
    (five segments) are valid; a reader that checks ``len(parts) != 4`` alone
    silently drops every discriminated manifest — which is every manifest
    `experiment.run`/`experiment.grade` or `alerts.sweep` ever write
    (alpha-engine-config-I9879).

    Returns ``None`` for any key that is not a manifest under this module's
    shape (wrong root, wrong suffix, or an arity other than four or five) —
    the caller decides whether a non-manifest key under `runs/` is an error
    or simply not of interest, this function only decides what it does not
    understand.
    """
    if not key.startswith(RUNS_ROOT) or not key.endswith(f"/{MANIFEST_BASENAME}"):
        return None
    parts = key.split("/")
    if len(parts) == 4:
        _, job, trading_day, _ = parts
        return job, trading_day, None
    if len(parts) == 5:
        _, job, trading_day, discriminator, _ = parts
        return job, trading_day, discriminator
    return None


def is_manifest_key(key: str) -> bool:
    """Whether ``key`` names a run manifest this module could have written.

    The predicate every consumer that LISTS a manifest prefix — or
    :data:`RUNS_ROOT`, or :func:`runs_prefix` — narrows the listing with
    before it reads anything. A manifest prefix is a namespace and not a
    manifest list (see :data:`MANIFEST_BASENAME`), so "every key under this
    prefix is a manifest" is a false assumption that reads as true for as long
    as no job files evidence beside its own manifest — and one now does.

    Expressed as :func:`parse_manifest_key` rather than as a suffix test, so a
    consumer gets the root and the arity checked too: `runs/x/run.json` ends
    with the right basename and is not a manifest key, and a suffix test at
    the call site is the shape that let `len(parts) != 4` drop every
    discriminated manifest (`alpha-engine-config-I9879`).
    """
    return parse_manifest_key(key) is not None


def runs_prefix(job: str) -> str:
    """The prefix under which EVERY trading day's manifest for ``job`` lives —
    coarser than :func:`manifest_prefix`, which additionally fixes the day.

    `manifest_prefix(job, trading_day)` for any ``trading_day`` starts with
    this prefix. A reader asking an existential question across every day a
    job has ever run (`crucible.console.render._has_history`: "has this
    component EVER produced a manifest?") lists this prefix instead of
    restating `runs/{job}/` inline.
    """
    if not job:
        raise ValueError("job must be non-empty")
    return f"runs/{job}/"


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


def experiments_prefix(arm_id: str) -> str:
    """The prefix under which every dated artifact for ``arm_id`` lives —
    `shadow_key`, `verdict_key`, `cross_section_key` and
    `cross_section_settled_key` all start with it.

    A reader that walks every trading day an arm has an artifact for
    (`crucible.report`'s slot-alpha and rank-IC rows, `crucible.slots.cycle`'s
    grader and `_shadow_dates`) lists this prefix rather than restating the
    shape those four key functions already own.
    """
    return f"experiments/{arm_key_segment(arm_id)}/"


def retirement_log_key(slot: str) -> str:
    """The append-only retirement event log for ``slot``.

    Dateless by design: it is the log, not a per-cycle artifact, and the §4.12
    key walk skips keys with no date component rather than requiring a
    trading day of something that spans all of them.
    """
    return f"retirements/{slot}/events.jsonl"


#: A reviewer identity is a Claude session id — the ONE identity in this fleet
#: that separates one agent from another. Every agent dispatches as the same
#: single GitHub collaborator, so a login can never tell a reviewer from an
#: author; a session id can, and it is already written into every commit this
#: fleet produces (the `Claude-Session:` trailer). That is what lets the
#: AUTHOR side of the independence comparison be DERIVED from the commit under
#: review rather than asserted by the party asking to be graded
#: (alpha-engine-config-I9873).
#: Case-INSENSITIVE, matching `.github/scripts/adversarial_review.py`'s
#: `REVIEWER_PATTERN` exactly — flags included, asserted equal by
#: `tests/test_adversarial_review.py`. The first version of these two had
#: different flags while a comment claimed they matched, so `SESSION_01AB...`
#: passed the workflow and would have raised here the moment the store writer
#: landed: a contract restated in two places had already drifted inside the
#: change that introduced it.
REVIEWER_PATTERN = r"^session_[A-Za-z0-9]{8,}$"
_REVIEWER_RE = re.compile(REVIEWER_PATTERN, re.IGNORECASE)


def _reviewer_segment(reviewer: str) -> str:
    """``reviewer`` as one key segment, lower-cased, or raise.

    Lower-cased because every comparison downstream is, and the GitHub status
    context is too: two spellings of one session would otherwise be two
    reviewers to the store and one reviewer to the gate.

    Rule 5, fail loud: a reviewer id carrying a `/` would file one review
    document under a key nothing lists back, and a review that cannot be read
    back is indistinguishable from a review that never happened. Free text is
    refused for the same reason the clause exists at all — an identity field
    the dispatcher may fill with any string measures nothing.
    """
    if not _REVIEWER_RE.match(reviewer):
        raise ValueError(
            f"reviewer {reviewer!r} is not a Claude session id (`session_<alnum>`). "
            "Independence is measured between session identities, which is the only "
            "identity that differs between two agents in this fleet"
        )
    return reviewer.lower()


def review_prefix(phase: str) -> str:
    """Every independent adversarial review filed against ``phase``.

    A phase exit is reviewed once per ROUND, not once per trading day, and the
    gate cannot know in advance which session reviewed or on which session —
    so the clause LISTS this prefix rather than reading one known key.
    """
    return f"reviews/{phase}/"


def review_key(phase: str, trading_day: str, reviewer: str, verdict: str) -> str:
    """One reviewer's verdict on ``phase``, filed on ``trading_day``.

    Two segments carry identity rather than living only in the body, and each
    closes a different overwrite.

    **The reviewer**, because one key per ``(phase, trading_day)`` would make
    two reviewers on the same session a last-writer-wins race whose survivor is
    whichever execution finished second — the shape that gave a cycle's verdict
    to its worst-informed author.

    **The verdict**, because otherwise a reviewer could erase its own adverse
    finding by re-recording a `pass` on the same session: one write, no code
    change, and the phase gate goes green on a change nobody re-reviewed. With
    the verdict in the key the two are different objects and `put_bytes` cannot
    make one replace the other — under ANY caller, including an `aws s3 cp`
    that never imports `crucible.review`. The review-writer role carries
    `PutObject`/`GetObject` and no `DeleteObject`, so a recorded finding cannot
    be removed by the party it was recorded against either.

    How a fail is legitimately superseded is a question for the reader, not the
    key, and `crucible.gate._clause_independently_reviewed` states its rule:
    an independent `pass` naming a DIFFERENT head sha, filed no earlier than
    the fail.
    """
    if verdict not in ("pass", "fail"):
        raise ValueError(
            f"verdict {verdict!r} is not 'pass' or 'fail'. A third verdict would be a "
            "third key shape the gate clause has never agreed to read"
        )
    return f"{review_prefix(phase)}{trading_day}/{_reviewer_segment(reviewer)}/{verdict}.json"


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


def gate_key(gate: str, trading_day: str) -> str:
    """Where a gate reading is filed. Keyed by trading day like everything else."""
    return f"gates/{gate}/{trading_day}/gate.json"


def closing_record_key(phase: str) -> str:
    """The ONE durable record that a phase's gate was read MET at its exit.

    `alpha-engine-config-I9967` deliverable 2. Deliberately NOT keyed by
    trading day, and deliberately a sibling of the dated readings under
    :func:`gate_prefix` rather than one of them: a phase exits once, the
    record of that exit is written once and never overwritten
    (`RunContext.record_output_cas` against an absent key), and a second file
    per day would make "when did this phase exit" a question with as many
    answers as there are days the gate has been read since.

    Its basename is not `gate.json` and its depth is three segments rather
    than four, so `crucible.gate.last_read` — which lists this same prefix to
    find the most recent trading day a gate was read for — does not mistake
    it for a dated reading.
    """
    if not phase:
        raise ValueError(
            "phase must be non-empty — a blank phase would file every phase's closing "
            "record at one key, which is the single-answer-per-phase property this key "
            "shape exists to guarantee."
        )
    return f"gates/{phase}/closing.json"


def gate_prefix(gate: str) -> str:
    """The prefix under which every trading day's reading of ``gate`` lives.

    `gate_key(gate, trading_day)` for any ``trading_day`` starts with this
    prefix — `crucible.gate.last_read`, which lists the store to find the
    most recent trading day a gate was read for, lists this prefix instead
    of restating its shape.
    """
    if not gate:
        raise ValueError(
            "gate must be non-empty — a blank gate would list every gate's readings "
            "under one empty-segment prefix, and `store.list_keys('gates//')` "
            "returning nothing reads as 'no data' rather than the caller's own bug."
        )
    return f"gates/{gate}/"


def universe_members_key(trading_day: str) -> str:
    """The U champion's feed: which names reach the predictor."""
    return f"universe/{trading_day}/members.json"


def declared_universe_key(trading_day: str) -> str:
    """`universe/declared/{trading_day}/members.json` — the store copy of the
    universe a data job graded ``trading_day``'s coverage against
    (`crucible.data.universe`).

    Beside, not inside, the U slot's :func:`universe_members_key` feed: that
    one is what the U arms SELECT, this one is what the data layer must
    COVER, and a reader that conflated them would grade selection quality
    against the denominator it was selected from.
    """
    if not trading_day:
        raise ValueError("trading_day must be non-empty")
    return f"universe/declared/{trading_day}/members.json"


def signals_key(trading_day: str) -> str:
    """The R champion's feed: how names were scored."""
    return f"signals/{trading_day}/signals.json"


# -- drift ------------------------------------------------------------------


DRIFT_INPUTS = ("features", "predictions", "ic")


def drift_input_key(name: str, trading_day: str) -> str:
    """One of the three artifacts `crucible.track_c.drift_handler` reads
    before it will compute a drift metric at all — track A/B's features,
    predictions and realized IC for the cycle.

    ``name`` is one of :data:`DRIFT_INPUTS`; anything else raises rather
    than silently producing a fourth input key nothing writes.
    """
    if name not in DRIFT_INPUTS:
        raise ValueError(f"unknown drift input {name!r}; the three inputs are {DRIFT_INPUTS}")
    return f"drift/{trading_day}/input_{name}.json"


def drift_metrics_key(trading_day: str) -> str:
    """Where `crucible.track_c.drift_handler` files the cycle's three
    drift `MetricRecord`s, alongside the run manifest's own copy."""
    return f"drift/{trading_day}/metrics.json"


# -- one-shot / repair jobs --------------------------------------------------


def migration_key(trading_day: str, run_id: str) -> str:
    """Where `crucible migrate.history` files its own result, per attempt.

    Keyed by `run_id`, not by trading day alone: the migration is a one-shot
    the runbook says is safe to rerun (`--allow-missing` recovery), and a
    bare `{trading_day}/migration.json` would let a second attempt on the
    same day overwrite the first attempt's record of what it actually
    imported.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty — see manifest_key's discriminator for why.")
    return f"migrations/{trading_day}/{run_id}.json"


def heal_key(trading_day: str, run_id: str) -> str:
    """Where `crucible data.heal` files its own result, per attempt.

    Keyed by `run_id` for the same reason as :func:`migration_key`: a heal is
    rerun idempotently, and each attempt's own record — which sessions were
    repaired versus already present — is worth keeping distinct from the
    attempt before it.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty — see manifest_key's discriminator for why.")
    return f"heals/{trading_day}/{run_id}.json"


# -- fleet ledger -----------------------------------------------------------


def ledger_key() -> str:
    """§9.1: the fleet trial ledger DSR's `n_trials` deflation reads.

    Dateless: it is one append-only log across every slot and every cycle,
    and a per-date ledger would make "everything ever tried" a join.
    """
    return "ledger/trials.jsonl"


# -- the declared board (alpha-engine-config-I9837) --------------------------


#: The pointer the fleet-console adapter reads, and the served page. Here
#: rather than in `crucible/board.py` because this module's rule admits no
#: exception: every store key shape, in one place. `tests/test_key_construction_placement.py`
#: caught these two the moment it landed, and its message is the right one —
#: there is no "not moved yet" registry, because that is a suppression list.
BOARD_CURRENT_KEY = "board/current.json"

#: `console-policy` requires every view to serve the JSON an agent reads
#: alongside its HTML: a page whose numbers can only be scraped out of markup
#: is a page the next automated reader re-derives incorrectly.
BOARD_HTML_KEY = "board/index.html"

#: The fleet console's served page and its agent-readable JSON twin
#: (`console-policy`), previously two literals in `crucible/console/render.py`
#: that `tests/test_no_inline_store_keys.py` could not see until it learned
#: to resolve a module-level Name (alpha-engine-config-I9899, round 2).
CONSOLE_KEY = "console/index.html"
CONSOLE_JSON_KEY = "console/index.json"

#: Every published release lives under here — `releases/{sha}/...` (owned by
#: `crucible.release.release_prefix`, an architectural exception registered
#: in `tests/test_key_construction_placement.py`) and the `releases/current`
#: pointer. The listing root for the release-lock sweep, which used to
#: derive it from the pointer key with `rsplit` because this constant did not
#: exist (same round).
RELEASES_ROOT = "releases/"

#: The single mutable object in the whole release layout — the pointer every
#: job follows. Everything else under `releases/` is immutable and
#: content-addressed by the sha in its own prefix. Lived in
#: `crucible/release.py` until alpha-engine-config-I9899 round 2, when the
#: call-site guard learned to resolve a module-level Name and found it.
POINTER_KEY = f"{RELEASES_ROOT}current"

#: The trader's separate pin. Two pointers, never one: a trader that followed
#: `current` would be promoted by every merge, and release promotion to the
#: trader is an explicit off-market-hours action (plan §4.11).
TRADER_PIN_KEY = "trader/release_pin"


def board_key(trading_day: str) -> str:
    """Where one day's board is filed. Keyed by trading day like everything else."""
    return f"board/{trading_day}/board.json"


def board_html_key() -> str:
    return BOARD_HTML_KEY


# -- the v1 weekly pipeline, read by phase 0's exit gate --------------------


def legacy_weekly_executions_key(week_anchor: str) -> str:
    """Where the v1 weekly state machine's start count for one week is filed.

    ``week_anchor`` is a :func:`weekly_anchor` session, never a raw render day.

    **The gate reads this; it never counts.** A clause that called
    `states:ListExecutions` would reach live AWS, which makes the reading
    unreplayable, untestable without credentials, and impossible to grade at a
    past date — and phase 0's whole claim is about a cadence sustained over a
    completed week, which is a claim about the past.

    An absent document reads UNMET and names the key, so the operator's next
    action is in the output (`alpha-engine-config-I9860`).

    **The document at this key is versioned, and the key is not.** `v1` filed
    one integer, `executions_started`. Since Brian's 2026-09-04 ruling
    (`alpha-engine-config-I9756`, implemented under
    `alpha-engine-config-I9962`) the clause grades executions that PASSED
    `WeeklyRunDayGate`, which needs a per-execution record — so
    `nous-ergon-ops/scripts/legacy_weekly_executions_producer.py` files
    `schema_version: legacy-weekly-executions.v2` carrying name, start, stop,
    duration and status per execution. The key stays the same because the week
    it identifies is the same; a `v1` document still sitting at it reads
    UNMEASURABLE, never a pass. Producer and consumer are held together by
    `crucible/tests/test_legacy_weekly_executions_contract.py` and
    `nous-ergon-ops/tests/test_legacy_weekly_executions_producer.py`, which
    grade the same fixture from the two sides.
    """
    return f"legacy/weekly/{week_anchor}/executions.json"


def legacy_dead_lambdas_key(week_anchor: str) -> str:
    """Where the probe of the six deleted v1 Lambda functions is filed.

    ``week_anchor`` is a :func:`weekly_anchor` session, the same anchor
    :func:`legacy_weekly_executions_key` uses, so phase 0's clauses read one
    week's worth of v1 facts at one set of keys.

    **The gate reads this; it never probes.** `alpha-engine-config-I9756`'s
    second deliverable is that six zero-invocation v1 functions are gone, and
    the only surface that answers it is a live AWS read, which a gate may not
    make — the same separation `legacy_weekly_executions_key` states for
    itself. `nous-ergon-ops/scripts/legacy_dead_lambdas_producer.py` holds the
    identity and files the answer here
    (`schema_version: legacy-dead-lambdas.v1`).

    **The probe is per EXACT name, never a prefix.** Four live functions --
    `alpha-engine-research-eval-judge-process`, `-poll`, `-submit` and
    `-spot-dispatcher` -- share a prefix with the deleted
    `alpha-engine-research-eval-judge`, and every one of them would match a
    `startswith` or a substring search. A prefix-matching probe would report
    this deliverable UNMET forever against a system that satisfies it, so the
    document carries one entry per exact name and the consumer refuses a
    document that does not cover the full name set.
    """
    return f"legacy/lambdas/{week_anchor}/probe.json"


# -- the 6am PT morning report (alpha-engine-config-I9896) ------------------


def morning_report_key(trading_day: str, calendar_date: str) -> str:
    """The exact message `report.morning` delivered, filed beside its manifest.

    Filed at all because the delivery is the deliverable and Telegram is not a
    durable artifact: principle 1 asks whether someone can reconstruct what an
    unattended run did from durable artifacts alone, and "check Brian's phone"
    is not one. Under the job's OWN manifest prefix, so the identity that
    writes the manifest needs no second grant to write this — a report whose
    evidence needed a wider IAM scope than its manifest would be a reason to
    widen the scope.

    ``calendar_date`` is the FIRING, and it is the same discriminator the job's
    manifest carries. `crucible.morning.DELIVERY_CRON_UTC` fires every calendar day while
    `resolve_trading_day` collapses Saturday, Sunday and Monday onto Friday's
    close (§4.12), so three genuinely different deliveries would otherwise
    overwrite one another at one key and the store's answer to "what was Brian
    told on Sunday" would be decided by write ordering. Same shape
    `crucible.alerts.sweep` already carries for the same reason.
    """
    return f"{manifest_prefix('report.morning', trading_day)}{calendar_date}/message.txt"


def morning_update_key(trading_day: str, calendar_date: str) -> str:
    """Where `report.morning` files the FULL daily update it also posts to
    the tracker (`alpha-engine-config-I10123`).

    Since Brian's 2026-09-06 ruling the delivered Telegram message is a
    headline that LINKS to the full update, and the full update is a GitHub
    comment — this is the durable filed copy of exactly that markdown,
    beside `message.txt` under the job's OWN manifest prefix, so the writer
    needs no second grant to file it. `principle 1` (transparency): the
    comment lives in a different repository's tracker and could in
    principle be edited or deleted there; this key is the store's own
    record of what was posted, independent of the tracker.
    """
    return f"{manifest_prefix('report.morning', trading_day)}{calendar_date}/update.md"


def morning_history_row_key(trading_day: str, calendar_date: str) -> str:
    """One delivery's compact facts, beside its manifest
    (`alpha-engine-config-I10123` deliverable 7).

    The rolling tracker issue's BODY is a regenerated history index — a
    newest-first table of every trading day's phase states and acceptance
    figure — and rebuilding it from `update.md`'s Markdown on every delivery
    would make the index a re-parse of prose this module already owns
    structured data for. This key is that structured data instead: a small
    JSON object (`trading_day`, `delivered_pt`, `phases`, `acceptance_met`,
    `acceptance_total`, `comment_url`) the index reader lists and reads
    through the guarded parser, the same way it reads every other artifact
    under this job's manifest prefix — never a second, disagreeing
    computation of what the board said.
    """
    return f"{manifest_prefix('report.morning', trading_day)}{calendar_date}/history_row.json"


#: What a trigger name may be, before it becomes a key segment. Not a
#: convenience: `GITHUB_EVENT_NAME` is an environment string, and an
#: unvalidated one concatenated into an S3 key is a path the caller chooses.
TRIGGER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: The value recorded when the invocation said nothing about what started it
#: — a laptop run, or any dispatcher that sets neither variable. DECLARED,
#: like a `null` signal class in `components.yaml`, rather than omitted: an
#: absent trigger object is indistinguishable from a forgotten one, and this
#: value satisfies no predicate that asks for a scheduled firing.
TRIGGER_UNKNOWN = "unknown"


def morning_trigger_key(trading_day: str, calendar_date: str, trigger: str) -> str:
    """Evidence of WHAT started this delivery, filed beside its manifest.

    The trigger is in the KEY, not in a body: `alpha-engine-config-I9896` /
    `-I9914` / `-I9921` each close on "at least one delivery that no human
    dispatched", and the sweep that closes a tracker issue on evidence
    (`alpha-engine-config/scripts/cross_repo_close_reconciliation.py`)
    evaluates an S3 `Verified-when:` predicate whose ops are existence,
    object count, age and one JSON field. A key that is itself the answer is
    readable by the `exists` op with a wildcard and needs no body parse:

        Verified-when: s3://<store>/crucible/runs/report.morning/*/*/trigger.schedule exists

    Nothing about the manifest could have carried this. `run_manifest.v2`
    declares `additionalProperties: false`, and `run_mode` is live-vs-replay
    — a scheduled run and a dispatched one are both live, which is why the
    three issues could not close on any artifact that existed.

    ``trigger`` comes from the INVOCATION and is validated (`TRIGGER_RE`)
    rather than trusted: it is an environment string on its way into a key.
    """
    if not TRIGGER_RE.match(trigger):
        raise ValueError(
            f"{trigger!r} is not a trigger name. It becomes a key segment, so it is "
            f"constrained to {TRIGGER_RE.pattern} rather than trusted — an environment "
            "variable concatenated into an S3 key is a path the caller chooses."
        )
    return f"{manifest_prefix('report.morning', trading_day)}{calendar_date}/trigger.{trigger}"


# -- the acceptance suite's reading, filed as an artifact (I9902) -----------


def acceptance_reading_key(trading_day: str) -> str:
    """Where `main`'s plan §2 acceptance reading is filed.

    §12 rule 3 makes the acceptance count the ONLY progress figure, and today
    it exists solely in `tests/acceptance/ratchet.json` — a property of the
    repository, readable from a git checkout and from nowhere else. Every
    consumer outside a checkout therefore has no way to read the one number
    the plan names as progress. `alpha-engine-config-I9902` builds the
    producer; this is the key it writes to, declared here first so the
    consumer (`crucible.morning`) and the producer cannot disagree about the
    shape, and so the reader has something to name while it reads absent.

    **The producer contract**, and the document `crucible.morning` parses::

        {"met": int, "unmet": int, "unmeasurable": int,
         "commit": str, "measured_at": str,
         "met_clauses": [str],
         "unmet_clauses": [str], "unmeasurable_clauses": [str],
         "store_versioning": str}

    The three clause-id lists are OPTIONAL, and so is `store_versioning`
    (`alpha-engine-config-I9921`, extended by `alpha-engine-config-I9964`):
    the report and the board page name the failing clauses when the producer
    files them, and say "the artifact names no unmet clause ids" when it does
    not. Optional rather than required so a producer shipping the counts first
    is a partial producer rather than a broken one — but never inferred: an
    absent list means "not filed", never "there are none", and the consumers
    render those differently.

    **`met_clauses` is why the counts were not enough** (`I9964`). Phase 0's
    `v2_resources_tagged_and_versioned` deliverable is declared, in this
    gate's own deliverable table, to be graded by ONE §2 clause —
    `TestCost::test_every_v2_resource_is_tagged_for_cost_attribution`. A
    document carrying only `{"met": 22, "unmet": 2}` cannot answer whether
    THAT clause is one of the 22, and "not named in `unmet_clauses`" is not an
    answer either: an absent list means not filed. So the producer files the
    met ids as well, and the gate clause reads UNMEASURABLE — never MET —
    when they are absent.

    **`store_versioning` is the second half of that same deliverable**, and it
    rides on this document rather than on one of its own for a stated reason.
    The deliverable names two properties of the v2 resources: the cost tag,
    which the §2 clause above grades, and S3 versioning on the store. The
    acceptance job is the one v2 CI identity that may read either — it holds
    `cloudformation:ListStackResources`, `tag:GetResources`, `iam:ListRoleTags`
    and `s3:GetBucketVersioning`, and can write exactly this prefix. Two
    documents from the same job at the same instant would be two artifacts
    that can disagree about one deliverable, and the gate would then have to
    decide which one wins. The value is the bucket's `Status` verbatim
    (`Enabled`, `Suspended`) and the field is ABSENT rather than guessed when
    the read failed — a failed read is UNMEASURABLE, never `Suspended`.

    `unmeasurable` is its own integer and is never folded into `unmet`: "the
    clause says no" and "we could not ask" are different facts and the second
    one is about us (`alpha-engine-config-I9828` is exactly that distinction
    for the cost clause). `commit` is the sha the reading was taken at — plan
    §6 rule 2, a reading is always quoted with its store and commit — and a
    count with no commit beside it is a number nobody can reproduce.
    """
    if not trading_day:
        raise ValueError("trading_day must be non-empty")
    return f"report/acceptance/{trading_day}.json"


#: The fields a reading must carry before either surface will quote a figure
#: from it. Declared once, here, beside the key whose contract they are —
#: :func:`parse_acceptance_reading` is the only reader, and both consumers go
#: through it.
ACCEPTANCE_REQUIRED_FIELDS: tuple[str, ...] = ("met", "unmet", "unmeasurable", "commit")


@dataclass(frozen=True)
class AcceptanceReading:
    """One parsed §2 acceptance reading. Complete by construction.

    There is no partial instance of this type: :func:`parse_acceptance_reading`
    returns `None` rather than an object with a missing field, so no consumer
    can render three quarters of the only number the plan calls progress and
    have it look exactly like a measurement.
    """

    met: int
    unmet: int
    unmeasurable: int
    commit: str
    measured_at: str | None
    #: `None` means the producer filed no list — never "there are none". The
    #: two render differently on both surfaces.
    unmet_clauses: tuple[str, ...] | None
    unmeasurable_clauses: tuple[str, ...] | None
    #: The ids the suite reported MET, or `None` when the producer filed no
    #: list. `None` is never "no clause was met" and never "every clause not
    #: named unmet was met" — a gate clause asking about ONE id reads
    #: UNMEASURABLE against `None` (`alpha-engine-config-I9964`).
    met_clauses: tuple[str, ...] | None = None
    #: The store bucket's S3 versioning `Status` verbatim, or `None` when the
    #: producer did not file one. `None` is never `Suspended`.
    store_versioning: str | None = None

    @property
    def total(self) -> int:
        return self.met + self.unmet + self.unmeasurable


def _clause_ids(document: dict[str, Any], field_name: str) -> tuple[str, ...] | None:
    named = document.get(field_name)
    if not isinstance(named, list) or not named:
        return None
    return tuple(str(cid) for cid in named)


def parse_acceptance_reading(document: Any) -> AcceptanceReading | None:
    """The ONE reader of an acceptance artifact. `None` when it is not one.

    **One parse, two consumers** (`alpha-engine-config-I9921` adversarial
    review F2). `crucible.morning` renders this reading into the Telegram
    message and `crucible.board` renders it onto the page the message links
    to. Before this function existed each had its own completeness rule — the
    message required all four of :data:`ACCEPTANCE_REQUIRED_FIELDS` and the
    page required only the three integers, rendering a missing `commit` as
    the literal `UNKNOWN`. Measured on ONE document: the message said
    `acceptance count: not on any artifact` while the page it pointed at
    printed `21 met / 2 unmet / 1 unmeasurable ... at commit UNKNOWN`. A
    surface that says a number does not exist, linking to a surface printing
    it, is worse than either alone.

    The rule, stated once: all four required fields present, the three counts
    real `int`s (a `bool` is not a count, however much Python agrees it is an
    `int`), `commit` a non-empty string. Anything else is not a reading —
    `None`, and each surface says so in its own words. Absence and denial are
    NOT this function's business: the caller already told them apart when it
    read the key, and folding them in here would put plan §6 rule 1's
    forbidden conflation inside the shared parser.
    """
    if not isinstance(document, dict):
        return None
    counts: list[int] = []
    for field_name in ("met", "unmet", "unmeasurable"):
        value = document.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        counts.append(value)
    commit = document.get("commit")
    if not isinstance(commit, str) or not commit:
        return None
    measured_at = document.get("measured_at")
    versioning = document.get("store_versioning")
    return AcceptanceReading(
        met=counts[0],
        unmet=counts[1],
        unmeasurable=counts[2],
        commit=commit,
        measured_at=measured_at if isinstance(measured_at, str) and measured_at else None,
        unmet_clauses=_clause_ids(document, "unmet_clauses"),
        unmeasurable_clauses=_clause_ids(document, "unmeasurable_clauses"),
        met_clauses=_clause_ids(document, "met_clauses"),
        store_versioning=versioning if isinstance(versioning, str) and versioning else None,
    )
