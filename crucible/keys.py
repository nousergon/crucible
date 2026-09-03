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
    "ALERTS_ROOT",
    "ARM_SEGMENT_SEPARATOR",
    "BOARD_CURRENT_KEY",
    "BOARD_HTML_KEY",
    "DRIFT_INPUTS",
    "REVIEWER_PATTERN",
    "RUNS_ROOT",
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
    "coverage_key",
    "cross_section_key",
    "cross_section_settled_key",
    "data_panel_key",
    "drift_input_key",
    "drift_metrics_key",
    "experiments_key",
    "experiments_prefix",
    "feature_registry_key",
    "features_key",
    "features_prefix",
    "gate_key",
    "heal_key",
    "ledger_key",
    "legacy_weekly_executions_key",
    "manifest_key",
    "manifest_prefix",
    "migration_key",
    "morning_report_key",
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

#: The root namespace segment every alert bus row lives under, narrowed by
#: `crucible.alerts.bus_key` (an architectural exception — see
#: `tests/test_key_construction_placement.py`). `crucible.alerts.pages_in_window`
#: lists this constant rather than hardcoding `"alerts/"` when it counts every
#: incident across the whole trailing window, not one group.
ALERTS_ROOT = "alerts/"


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
    if not key.startswith(RUNS_ROOT) or not key.endswith("/run.json"):
        return None
    parts = key.split("/")
    if len(parts) == 4:
        _, job, trading_day, _ = parts
        return job, trading_day, None
    if len(parts) == 5:
        _, job, trading_day, discriminator, _ = parts
        return job, trading_day, discriminator
    return None


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


def universe_members_key(trading_day: str) -> str:
    """The U champion's feed: which names reach the predictor."""
    return f"universe/{trading_day}/members.json"


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
    past date — and phase 0's whole claim is about a cadence sustained over
    two consecutive weeks, which is a claim about the past.

    Nothing files this key yet, and that is the honest state of the
    measurement rather than a reason to relax the clause: the clause reads
    UNMET and names the key, so the operator's next action is in the output
    (`alpha-engine-config-I9860`). The shape belongs in `crucible.keys` and
    moves there once the branch that owns that file lands.
    """
    return f"legacy/weekly/{week_anchor}/executions.json"


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
    manifest carries. A 13:00 UTC cron fires every calendar day while
    `resolve_trading_day` collapses Saturday, Sunday and Monday onto Friday's
    close (§4.12), so three genuinely different deliveries would otherwise
    overwrite one another at one key and the store's answer to "what was Brian
    told on Sunday" would be decided by write ordering. Same shape
    `crucible.alerts.sweep` already carries for the same reason.
    """
    return f"{manifest_prefix('report.morning', trading_day)}{calendar_date}/message.txt"
