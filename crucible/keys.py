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

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from crucible.calendar import ISO_DATE_RE, assert_trading_day

__all__ = [
    "ACCEPTANCE_REQUIRED_FIELDS",
    "ALERTS_ROOT",
    "ARM_PREDICTIONS_PREFIX",
    "ARM_SEGMENT_SEPARATOR",
    "AcceptanceReading",
    "BOARD_CURRENT_KEY",
    "BOARD_HTML_KEY",
    "CONSOLE_JSON_KEY",
    "CONSOLE_KEY",
    "DATA_BUCKET_KEY_HELPERS",
    "DISPATCH_EXIT_SUFFIX",
    "DISPATCH_ROOT",
    "DRIFT_INPUTS",
    "EXPERIMENTS_ROOT",
    "FAULT_INJECTION_ROOT",
    "INTEGRATION_STORE_SUBPREFIX",
    "MANIFEST_BASENAME",
    "POINTER_KEY",
    "PREDICTIONS_PREFIX",
    "PUBLIC_JSON_KEY",
    "PUBLIC_KEY",
    "RANGE_BOUND_JOBS",
    "RELEASES_ROOT",
    "REVIEWER_PATTERN",
    "RUNS_ROOT",
    "TRADER_BROKER_STATEMENTS_PREFIX",
    "TRADER_EVIDENCE_KEY",
    "TRADER_EXECUTION_SHORTFALL_PREFIX",
    "TRADER_FIRE_DRILLS_PREFIX",
    "TRADER_FIRE_DRILL_SCHEDULE_PREFIX",
    "TRADER_KILL_SWITCH_EVENTS_PREFIX",
    "TRADER_KILL_SWITCH_KEY",
    "TRADER_PAPER_SMOKE_PREFIX",
    "TRADER_PIN_KEY",
    "TRADER_RECONCILIATION_PREFIX",
    "TRADER_SHADOW_BOOKS_PREFIX",
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
    "backfill_key",
    "board_html_key",
    "board_key",
    "calendar_day_discriminator",
    "champion_key",
    "closing_record_key",
    "constituents_key",
    "coverage_key",
    "cross_section_key",
    "cross_section_settled_key",
    "data_panel_key",
    "declared_universe_key",
    "dispatch_exit_key",
    "dispatch_key",
    "dispatch_prefix",
    "drift_input_key",
    "drift_metrics_key",
    "edgar_fundamentals_session_key",
    "execution_shortfall_key",
    "experiments_key",
    "experiments_prefix",
    "fault_injection_key",
    "feature_registry_key",
    "features_key",
    "features_prefix",
    "fundamental_snapshot_key",
    "gate_key",
    "gate_prefix",
    "heal_key",
    "holdout_unseal_key",
    "holdout_unseal_prefix",
    "iac_conformance_key",
    "inst_ownership_key",
    "integration_store_key",
    "is_manifest_key",
    "ledger_key",
    "legacy_dead_lambdas_key",
    "legacy_weekly_executions_key",
    "manifest_calendar_day",
    "manifest_key",
    "manifest_prefix",
    "manifest_ran_on",
    "migration_key",
    "morning_history_row_key",
    "morning_report_key",
    "morning_trigger_key",
    "morning_update_key",
    "parse_acceptance_reading",
    "parse_bus_key",
    "parse_dispatch_exit_key",
    "parse_dispatch_key",
    "parse_fault_injection_key",
    "parse_manifest_key",
    "predictions_key",
    "retirement_log_key",
    "review_key",
    "review_prefix",
    "runs_prefix",
    "session_inputs_key",
    "shadow_books_key",
    "shadow_key",
    "signals_key",
    "strategy_arm_key",
    "strategy_arms_prefix",
    "strategy_holdout_key",
    "strategy_slot_key",
    "strategy_slots_prefix",
    "trader_broker_statement_key",
    "trader_fire_drill_key",
    "trader_fire_drill_schedule_key",
    "trader_kill_switch_event_key",
    "trader_paper_smoke_key",
    "trader_reconciliation_key",
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

#: The root namespace segment every dated arm artifact lives under —
#: `experiments_prefix` narrows it to one arm. A caller with no arm at all —
#: an existential "every settled verdict, across every slot and arm" scan
#: (`crucible.explain.select_newest_settled_verdict`,
#: `alpha-engine-config-I10858`) — lists this constant directly rather than
#: hardcoding `"experiments/"` at the call site.
EXPERIMENTS_ROOT = "experiments/"

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

#: The one declaration of where the dedicated integration store sits relative
#: to the production store's root — `alpha-engine-config-I10706`. The
#: nightly integration tier (`.github/workflows/integration-nightly.yml`)
#: writes through `CRUCIBLE_INTEGRATION_STORE_URI`, a repository variable
#: whose value is the production root plus this sub-prefix
#: (`tests/integration/conftest.py::integration_store_uri` refuses any URI
#: that does not carry an `integration` path segment, for the same reason
#: this file forbids literal infrastructure identifiers: it cannot compare
#: against the production prefix by name). Every reader of the tier's
#: manifests FROM the production store — today, only
#: `crucible.gate._clause_integration_tier_current` — narrows through this
#: constant rather than restating `"integration/"` as a literal, so a future
#: change to the sub-prefix has exactly one place to change.
INTEGRATION_STORE_SUBPREFIX = "integration/"


def integration_store_key(relative_key: str) -> str:
    """Translate a key or prefix relative to the dedicated integration store
    (the shape every other key function in this module already produces —
    `runs/test.integration/{day}/run.json`, `runs/test.integration/`) into
    the real key under :data:`INTEGRATION_STORE_SUBPREFIX` in the production
    store the phase gate reads. The one call site,
    `crucible.gate._DedicatedSubtreeView`, translates every `Store` call at
    this boundary through here rather than restating the shape inline
    (`tests/test_no_inline_store_keys.py`).
    """
    return f"{INTEGRATION_STORE_SUBPREFIX}{relative_key}"


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


#: The root namespace segment every fault-injection record lives under
#: (plan §10.7). One record per scripted fault induced against the real
#: scheduled path, naming the manifest key and the bus row that fault
#: actually produced, so "we ran fault injection" is a reading rather than a
#: sentence in a session transcript.
#:
#: Deliberately a sibling of :data:`RUNS_ROOT` rather than nested under it,
#: for the reason :data:`DISPATCH_ROOT` is: a fault record is EVIDENCE ABOUT
#: a run, not a run, and a `runs/{job}/` listing must not be able to reach
#: it even by accident.
#:
#: **No producer files one today.** The two halves that would — a live sweep
#: reaching a safely-old trading day, and an authorized seam for the three
#: faults that have none — are open on the tracker, and the reader below is
#: what makes that absence visible on the phase ladder instead of only in an
#: issue. A shape declared by its reader is the honest state: the reader can
#: say exactly which key it looked for, and a producer arriving later has one
#: contract to write against rather than inventing a second.
FAULT_INJECTION_ROOT = "faults/"


def fault_injection_key(fault_id: str, trading_day: str) -> str:
    """`faults/{trading_day}/{fault_id}.json` — one record per induced fault.

    Keyed by trading day and not by wall clock: a fault is induced against a
    trading day's scheduled path, and the manifest and bus row it produced
    are both keyed that way (rule 3). Inducing the same fault again on the
    same day overwrites in place, which is correct — the record describes the
    day's state, and two records for one (fault, day) would be two answers to
    one question.
    """
    assert_trading_day(trading_day, context=f"fault_injection_key({fault_id!r})")
    if not _DISCRIMINATOR_RE.match(fault_id):
        raise ValueError(
            f"fault_id {fault_id!r} must be 1-64 characters of [A-Za-z0-9_.-] — it is a "
            "path segment, and this is the one place that is enforced."
        )
    return f"{FAULT_INJECTION_ROOT}{trading_day}/{fault_id}.json"


def parse_fault_injection_key(key: str) -> tuple[str, str] | None:
    """The inverse of :func:`fault_injection_key`: ``(trading_day, fault_id)``.

    Returns ``None`` for anything not under :data:`FAULT_INJECTION_ROOT` in
    this exact three-segment shape, so a caller listing the whole root
    decides what an unrecognised key means rather than this function
    guessing — the same contract :func:`parse_bus_key` keeps.
    """
    if not key.startswith(FAULT_INJECTION_ROOT) or not key.endswith(".json"):
        return None
    parts = key.split("/")
    if len(parts) != 3:
        return None
    _, trading_day, filename = parts
    return trading_day, filename[: -len(".json")]


#: `alpha-engine-config-I10134` deliverable 1/2: proof that a job was
#: EXPLICITLY dispatched, written by the v2 dispatcher before
#: `RunInstances` returns, so a later sweep can grade "requested but never
#: completed" — a state a *scheduled* job's `components.yaml` deadline
#: cannot represent for an on-demand job (`data.heal`'s row carries
#: `deadline: null`). Deliberately a sibling of `RUNS_ROOT` rather than
#: nested under it: a dispatch record is not a manifest and must never be
#: mistaken for one by a `runs/{job}/` listing — `parse_manifest_key`'s own
#: suffix check (`/run.json`) already excludes it, but a *second*, syntactic
#: separation is what makes that true by construction rather than by one
#: function agreeing to filter it out.
DISPATCH_ROOT = "runs/_dispatch/"


def dispatch_prefix(job: str) -> str:
    """Every dispatch record ever written for ``job``, across every attempt.

    `crucible.alerts.evaluate_dispatch_absence` lists this prefix — it has no
    trading day to key on ahead of time (that is resolved from the record's
    own `dispatched_at_utc`, the same way :func:`crucible.calendar.
    resolve_trading_day` resolves every other wall-clock firing), so it reads
    every dispatch for the job rather than guessing a day.
    """
    if not job:
        raise ValueError(
            "job must be non-empty — a blank job would list every job's dispatch "
            "records under one empty-segment prefix, and `store.list_keys("
            "'runs/_dispatch//')` returning nothing reads as 'no data' rather than "
            "the caller's own bug."
        )
    return f"{DISPATCH_ROOT}{job}/"


def dispatch_key(job: str, dispatch_id: str) -> str:
    """`runs/_dispatch/{job}/{dispatch_id}.json` — one record per attempt.

    ``dispatch_id`` is a path segment (the dispatcher's own random token, not
    a ULID library this Lambda would have to vendor), validated against the
    same charset a manifest discriminator is, for the same reason: it is
    about to become a key component no path-shaped tool may choke on.
    """
    if not _DISCRIMINATOR_RE.match(dispatch_id):
        raise ValueError(
            f"dispatch_id {dispatch_id!r} must be 1-64 characters of [A-Za-z0-9_.-] "
            "— it is a path segment, and this is the one place that is enforced."
        )
    return f"{dispatch_prefix(job)}{dispatch_id}.json"


#: The suffix separating a dispatch's TERMINAL record from its request record
#: (`alpha-engine-config-I11050`). Both live under
#: :func:`dispatch_prefix` and both end `.json`, so the two are told apart by
#: this suffix and by nothing else — which is why :func:`parse_dispatch_key`
#: refuses it explicitly below rather than leaving the distinction to a
#: caller. Without that refusal every exit record read back as a dispatch
#: record for a `dispatch_id` ending `.exit`, with no manifest anywhere near
#: it, and the detector meant to explain absences would have MANUFACTURED one
#: per box.
DISPATCH_EXIT_SUFFIX = ".exit.json"


def dispatch_exit_key(job: str, dispatch_id: str) -> str:
    """`runs/_dispatch/{job}/{dispatch_id}.exit.json` — what the box's own
    exit path recorded about how this dispatch ENDED.

    `alpha-engine-config-I11050`. The dispatch record is the request; this is
    the terminal document, written from the box's EXIT trap (or, when the box
    died before `crucible` was installed, by the bootstrap shell) before the
    instance goes away. It is the only durable account of an exit that leaves
    no manifest — the reclaimed path deliberately writes none — and it is
    what `crucible.alerts` reads so an absence page names a cause instead of
    telling a human to go and look at a box that no longer exists.
    """
    if not _DISCRIMINATOR_RE.match(dispatch_id):
        raise ValueError(
            f"dispatch_id {dispatch_id!r} must be 1-64 characters of [A-Za-z0-9_.-] "
            "— it is a path segment, and this is the one place that is enforced."
        )
    return f"{dispatch_prefix(job)}{dispatch_id}{DISPATCH_EXIT_SUFFIX}"


def parse_dispatch_key(key: str) -> tuple[str, str] | None:
    """The inverse of :func:`dispatch_key`: ``(job, dispatch_id)``.

    Returns ``None`` for anything not under :data:`DISPATCH_ROOT` in this
    exact three-segment-under-the-root shape, so a caller listing the whole
    root decides what an unrecognised key means rather than this function
    guessing — and ``None`` for an EXIT record
    (:data:`DISPATCH_EXIT_SUFFIX`), which shares the prefix and the `.json`
    suffix but is a different document about the same dispatch.
    """
    if not key.startswith(DISPATCH_ROOT) or not key.endswith(".json"):
        return None
    if key.endswith(DISPATCH_EXIT_SUFFIX):
        return None
    parts = key.split("/")
    if len(parts) != 4:
        return None
    _, _, job, filename = parts
    return job, filename[: -len(".json")]


def parse_dispatch_exit_key(key: str) -> tuple[str, str] | None:
    """The inverse of :func:`dispatch_exit_key`: ``(job, dispatch_id)``."""
    if not key.startswith(DISPATCH_ROOT) or not key.endswith(DISPATCH_EXIT_SUFFIX):
        return None
    parts = key.split("/")
    if len(parts) != 4:
        return None
    _, _, job, filename = parts
    return job, filename[: -len(DISPATCH_EXIT_SUFFIX)]


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


#: The trader's champion serving feed — the contract named in
#: `crucible.slots.__init__` and in this repo's `AGENTS.md`: *"the trader reads
#: one contract — `champions/{slot}/current.json` plus
#: `predictions/{trading_day}.json`"*. ONE object per trading day, at the root
#: of its own prefix.
PREDICTIONS_PREFIX = "predictions/"

#: What one ARM predicted, which is a different artifact answering a different
#: question, and therefore lives under a different prefix
#: (`alpha-engine-config-I9822`). It shared `predictions/` until 2026-09-11 and
#: was distinguishable from the serving feed only by counting path segments —
#: so any consumer doing `store.list_keys("predictions/")` saw both shapes and
#: had to discriminate by depth. Nothing did.
#:
#: **Moved while the prefix was EMPTY.** Measured 2026-09-11: zero objects
#: existed under `predictions/` in the production store, so this rename
#: orphaned nothing. The same change made after either shape had been written
#: would have stranded those objects at an address no code resolves — which is
#: `alpha-engine-config-I10498`, filed the same day for a feature layer that
#: was stranded exactly that way by a content-hash move. The cheapest moment to
#: separate two artifact shapes is before either exists.
ARM_PREDICTIONS_PREFIX = "arm_predictions/"


def predictions_key(trading_day: str) -> str:
    """The champion's serving feed for ONE trading day — what the trader reads.

    Written by :func:`crucible.serving.publish_predictions_feed`, which
    republishes the `arm_predictions.v1` document the M champion pointer
    resolves to — never an independently computed cross-section. Read by
    :func:`crucible.serving.read_predictions_feed`, the reference consumer
    the trader imports.

    Was a key builder with no producer from `crucible-PR207` until
    `alpha-engine-config-I10129` closed it: the contract named in
    `AGENTS.md` is `champions/{slot}/current.json` PLUS this key, and half
    of it had no writer. Declared here rather than at the writer because a
    contract with no single source for its key is how the prose and the code
    drift apart, and because :func:`arm_predictions_key` cannot be tested
    against a collision with a key that does not exist.
    """
    return f"{PREDICTIONS_PREFIX}{trading_day}.json"


def arm_predictions_key(arm_id: str, trading_day: str) -> str:
    """What ONE arm predicted on ONE trading day.

    Per-arm, not per-slot: :func:`predictions_key` is the *champion's* serving
    feed, and a stacked arm reading that would depend on whichever arm holds
    the pointer — a base model that silently changes identity between two
    cycles, and a self-reference the moment the stacked arm won the slot.

    Under `arm_predictions/`, never `predictions/`: see
    :data:`ARM_PREDICTIONS_PREFIX`.
    """
    return f"{ARM_PREDICTIONS_PREFIX}{arm_key_segment(arm_id)}/{trading_day}.json"


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


def strategy_slot_key(slot: str) -> str:
    """A slot's published portfolio-construction parameter set (plan §4.4).

    `alpha-engine-config-I10511`. Sibling of :func:`strategy_arm_key`, but
    ONE file per slot rather than a directory of named recipes: a slot's
    portfolio parameters (`crucible.portfolio.PortfolioParams`) are a single
    document, unlike an arm, of which a slot may register several. Today only
    `strategy/slots/s.yaml` exists in the repo tree — the S slot's parameter
    set — but the key is general over ``slot`` so a second slot gaining its
    own portfolio-construction parameters needs no new key shape.
    """
    if not slot:
        raise ValueError(
            "slot must be non-empty — a blank slot would resolve to a key indistinguishable "
            "from a real one and every reader would be one string away from reading the "
            "wrong slot's parameters."
        )
    return f"strategy/current/slots/{slot}.yaml"


def strategy_slots_prefix() -> str:
    """The prefix under which every slot's published parameter file lives.

    `strategy_slot_key(slot)` for any ``slot`` starts with this prefix — the
    publisher (`alpha-engine-config/scripts/publish_crucible_v2_strategy_tree.py`)
    lists it to find a slot file retired from the repo tree that the store
    still carries, the same verify-and-repair shape `strategy_arms_prefix`
    already gives the arm tree.

    Takes no argument, unlike `strategy_arms_prefix(slot)`: an arm tree is
    already namespaced by slot on disk (`strategy/arms/{slot}/*.yaml`, many
    files), while every slot's single parameter file lives flat under one
    directory (`strategy/slots/{slot}.yaml`) — there is one prefix for the
    whole namespace, not one per slot.
    """
    return "strategy/current/slots/"


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

    **The date segment is the trading day, for every job, including an
    on-demand run given `--trading-day`.** Which day a key encodes, why the
    segment does not become the calendar date, and how an on-demand run's own
    calendar day becomes answerable from `runs/` anyway, are decided and
    written down immediately below :func:`is_manifest_key` — see
    :func:`calendar_day_discriminator` and :func:`manifest_ran_on`
    (alpha-engine-config-I10999).
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


#: The jobs whose ONE manifest is bound to the END of a range rather than to
#: a `--date` (`alpha-engine-config-I11048`). Both handlers pass
#: `trading_day=end` to `crucible.runner.run_job`:
#: `crucible.track_a.handle_data_heal` and
#: `crucible.track_a.handle_experiment_backfill`.
#:
#: **Why a declared set and not a flag-precedence rule over argv.** Every
#: dispatch the fleet makes carries `--date` — `nous-ergon-ops`'s
#: `scripts/dispatch_crucible_v2_job.sh` injects it unconditionally and the
#: job's own `--from/--to` arrive after it as pass-through — so "prefer
#: whichever flag appears" answers the question with the wrong flag for
#: exactly the jobs whose manifest is not under it. The binding is a property
#: of the JOB, so it is declared as one.
#:
#: **And not a hand-typed list.** `tests/test_range_bound_jobs_contract.py`
#: parses `crucible/track_a.py` and derives the set of jobs whose `run_job`
#: call passes `trading_day=end`; a third range job added there, or one of
#: these two rebound to `--date`, fails that test rather than silently
#: re-opening this defect. A membership fact that only a human keeps true is
#: the shape every detector in this fleet has had to be repaired from.
RANGE_BOUND_JOBS: frozenset[str] = frozenset({"data.heal", "experiment.backfill"})


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


# ── Which day a manifest key encodes (alpha-engine-config-I10999) ───────────
#
# THE DECISION, at the construction site, with its rationale.
#
# A manifest carries two days: `trading_day` (what the run is ABOUT) and
# `calendar_date` (when the process actually ran). Only the first has ever
# been in the key, and for a job graded on the trading-day axis that is
# right. For an ON-DEMAND run given `--trading-day` it is not enough, and the
# measured cost is this: the seven arm registrations of 2026-09-17 ran
# `experiment.new --trading-day 2026-09-11`, so every manifest landed under
# `runs/experiment.new/2026-09-11/`, `runs/experiment.new/2026-09-17/` does
# not exist, and "what mutated the store on 2026-09-17" had no answer in
# `runs/` at all — recovering the real day took S3 object-version
# `LastModified` archaeology.
#
# **The decision: the date SEGMENT stays the trading day for every job, and
# an on-demand job additionally leads its discriminator with the calendar day
# it ran on.** The manifest is therefore discoverable under the day it is
# about (the prefix, unchanged) and under the day it ran (the discriminator,
# via `manifest_ran_on`), from `runs/` keys alone.
#
# **Why the segment does not become the calendar day.** `crucible.alerts`
# grades ABSENCE off a predictable manifest key per trading day: it asks
# whether `runs/{job}/{trading_day}/` has anything under it for each session
# in its catch-up window. Move the segment for a scheduled job and the
# absence checker reads nothing where the job in fact ran — it is blinded in
# exactly the direction that never pages. That is `alpha-engine-config-I10981`
# one step over, and it is why this is an ASYMMETRY rather than a uniform
# change:
#
# * a job with a `schedule`/`deadline` in `components.yaml` keeps its key
#   shape untouched, because something grades its absence at that key;
# * an on-demand job (`experiment.new`: `schedule: null`, `deadline: null`)
#   has no absence grading keyed off it, so its discriminator is free.
#
# The asymmetry is written HERE, where the key is built, rather than left
# implicit at the one call site that uses it. A second on-demand job wanting
# the same property calls `calendar_day_discriminator` and inherits the
# reasoning; a scheduled job that calls it is making a visible mistake with
# the reason it is a mistake three lines above.
#
# **Why the calendar day leads the discriminator.** `manifest_calendar_day`
# reads it back off the key by position, which also — for free — reads the
# BARE calendar-date discriminators three jobs already write for the same
# "when did this actually fire" purpose (`alerts.sweep` and `report.morning`
# via `ctx.calendar_date`, `data.daily`'s holiday no-op via the wall-clock
# date). One predicate answers for all four rather than one per writer.


def calendar_day_discriminator(calendar_date: dt.date, *, suffix: str | None = None) -> str:
    """The manifest discriminator for an ON-DEMAND run: the calendar day it
    ran on, optionally followed by whatever else distinguishes the writer.

    See the block above for why this exists and why it is only for a job
    with no `schedule` and no `deadline`.

    ``suffix`` is the discriminator the caller would otherwise have passed
    (`experiment.new`'s `{slot}-{run_id}`, which keeps seven registrations on
    one day from overwriting each other, `alpha-engine-config-I10967`). It is
    appended, never replaced: this adds an axis rather than taking one away.

    The result is validated by :func:`manifest_key` like any other
    discriminator, so an unusable ``suffix`` is refused there, in the one
    place that refusal lives.
    """
    day = calendar_date.isoformat()
    return f"{day}-{suffix}" if suffix else day


def manifest_calendar_day(key: str) -> dt.date | None:
    """The calendar day ``key`` says its run happened on, or ``None``.

    ``None`` means the KEY does not state one — a bare manifest key, or one
    whose discriminator is a slot letter or a release sha. It is not a claim
    that the run has no calendar date (every manifest body carries one); it
    is the honest answer that this key does not encode it, and
    :func:`manifest_ran_on` is where that is turned into a reading.

    Returns ``None`` for a key that is not a manifest key at all, for the
    same reason :func:`parse_manifest_key` does: what a caller does about a
    foreign key is the caller's decision.
    """
    parsed = parse_manifest_key(key)
    if parsed is None:
        return None
    _job, _trading_day, discriminator = parsed
    if discriminator is None:
        return None
    candidate = discriminator[:10]
    if not ISO_DATE_RE.fullmatch(candidate):
        return None
    if len(discriminator) > 10 and discriminator[10] != "-":
        # `2026-09-1x...` — a ten-character prefix that parses but is not a
        # whole segment. Reading it as the calendar day would invent a fact
        # out of a discriminator that never claimed one.
        return None
    try:
        return dt.date.fromisoformat(candidate)
    except ValueError:
        # `2026-13-45` satisfies the shape and is not a date. The key states
        # no calendar day, which is what `None` says — reading it as one
        # would invent a fact out of a malformed segment.
        return None


def manifest_ran_on(key: str, calendar_date: dt.date) -> bool:
    """Whether ``key`` names a run that happened on ``calendar_date``.

    This is the answer to "what mutated the store on day X", computed from
    `runs/` KEYS alone — no manifest bodies read, and no S3 object-version
    metadata (`alpha-engine-config-I10999`).

    Two readings, and the second is the one that keeps every pre-existing key
    answerable:

    * the key STATES a calendar day (:func:`manifest_calendar_day`) — it is
      compared directly;
    * the key states none — the run is taken to have happened on its own
      trading day, which is true by construction for every writer that does
      not carry a calendar discriminator: a job that ran on some OTHER day is
      exactly a job that was given `--trading-day`, and an on-demand job given
      `--trading-day` now carries the day it ran in its discriminator.

    The second reading is an inference and is deliberately not silent about
    it: a scheduled job backfilled with `--trading-day` (the one shape it
    still gets wrong) keeps its key unchanged BECAUSE moving it would blind
    `crucible.alerts`' absence grading — see the block above this function.
    """
    stated = manifest_calendar_day(key)
    if stated is not None:
        return stated == calendar_date
    parsed = parse_manifest_key(key)
    if parsed is None:
        return False
    _job, trading_day, _discriminator = parsed
    try:
        return dt.date.fromisoformat(trading_day) == calendar_date
    except ValueError:
        return False


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


def session_inputs_key(arm_id: str, trading_day: str) -> str:
    """The POINT-IN-TIME construction inputs one S arm saw on ``trading_day``.

    The S slot's analogue of `shadow_key`, and it exists for the same reason.
    An S arm's score is its realized book's return against the benchmark, net
    of cost — a number produced at GRADE time by walking the sessions again
    through `crucible.portfolio`. If the alpha vector, the eligibility mask
    and the caps that walk read were resolved at grade time too, the grade
    would be taken over whatever those upstream artifacts say TODAY, and a
    revision anywhere upstream would silently re-price the arm's whole
    history. So `experiment.run --slot s` writes what the session actually
    carried, once, on the session, and the grade reads it back.

    Lives beside `shadow.json` under the same `experiments/{arm}/{day}/`
    prefix, so `experiments_prefix` still walks every dated artifact an arm
    has.
    """
    return f"experiments/{arm_key_segment(arm_id)}/{trading_day}/session_inputs.json"


def experiments_prefix(arm_id: str) -> str:
    """The prefix under which every dated artifact for ``arm_id`` lives —
    `shadow_key`, `verdict_key`, `cross_section_key`,
    `cross_section_settled_key` and `session_inputs_key` all start with it.

    A reader that walks every trading day an arm has an artifact for
    (`crucible.report`'s slot-alpha and rank-IC rows, `crucible.slots.cycle`'s
    grader and `_shadow_dates`) lists this prefix rather than restating the
    shape those four key functions already own.
    """
    return f"experiments/{arm_key_segment(arm_id)}/"


def v1_carryover_key() -> str:
    """The v1 carry-over ledger, inside the synced strategy tree.

    `alpha-engine-config-I10716`. A sibling of :func:`strategy_holdout_key`:
    one row per v1 arm per v1 slot (`carried`/`excluded`/`pending`), AUTHORED in
    the private strategy tree (`alpha-engine-config/strategy/v1_carryover.yaml`)
    and published under `strategy/current/`, read by
    `crucible.carryover.parse_ledger` for the phase-3 clause
    `v1_arms_carried_or_excluded`.
    """
    return "strategy/current/v1_carryover.yaml"


def strategy_holdout_key() -> str:
    """The sealed holdout document, inside the synced strategy tree (plan §9.4).

    `alpha-engine-config-I10502`. A sibling of :func:`strategy_arm_key` and
    :func:`strategy_slot_key` and, like them, a document AUTHORED in the
    private strategy tree (`alpha-engine-config/strategy/holdout.json`) and
    published into the store under `strategy/current/` — which held-out
    sessions are reserved is strategy edge, not framework
    (`repository-tiering-policy` test 2).

    Dateless, like :func:`retirement_log_key` and every other pointer-shaped
    document: a holdout is not a per-session artifact, it is the standing
    reservation every session's grading is measured outside of. The §4.12
    walk skips keys with no date component; the seal's own
    ``sealed_trading_day`` carries the day it was sealed on, inside the
    document, where it belongs.

    Takes no argument: there is ONE holdout for the harness, not one per
    slot. A per-slot holdout would let a slot grade itself against a
    reservation of its own choosing, which is the property `-I10502` exists
    to remove.
    """
    return "strategy/current/holdout.json"


def holdout_unseal_prefix() -> str:
    """The prefix under which every holdout unseal audit record lives.

    `holdout_unseal_key(trading_day, ruling)` for any argument starts with
    this prefix — `crucible.holdout.unseal_records` (and the phase-3
    `sealed_holdout` clause, through it) lists this rather than restating the
    shape. An unseal nobody can enumerate is an unseal nobody audits.
    """
    return "holdout/unseal/"


def holdout_unseal_key(trading_day: str, ruling: str) -> str:
    """One unseal audit record: WHO ruled, on WHICH session, unsealing WHAT.

    `alpha-engine-config-I10502`. Keyed by trading day like everything else
    (§4.12), then by the ruling reference — so a second unseal under the same
    ruling on the same session is idempotent by construction (the same key,
    the same content-addressed bytes) while two different rulings on one day
    are two records rather than one silently overwriting the other.

    ``ruling`` is validated by `crucible.holdout.assert_ruling_reference`
    BEFORE it reaches here; this function refuses an empty one only, the same
    way every other key function in this module refuses an empty segment —
    the grammar of a ruling reference is the holdout module's to own, and
    restating it here would be the second declaration that drifts.
    """
    assert_trading_day(trading_day, context=f"holdout_unseal_key({ruling!r})")
    if not ruling:
        raise ValueError(
            "an unseal record needs a ruling reference — the whole point of the "
            "record is naming the human decision that authorised the read, and a "
            "blank segment would file it at a key indistinguishable from any other "
            "unauthorised unseal's."
        )
    return f"{holdout_unseal_prefix()}{trading_day}/{ruling}.json"


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


def iac_conformance_key(trading_day: str) -> str:
    """Where `crucible.iac_conformance` files both comparison readings for
    the cycle (account-vs-template and template-vs-declared-inventory),
    alongside the run manifest's own copy of each as a `MetricRecord`
    (`alpha-engine-config-I10418`)."""
    return f"iac/{trading_day}/conformance.json"


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


def backfill_key(trading_day: str, run_id: str) -> str:
    """Where `crucible experiment.backfill` files its own result, per attempt.

    Keyed by `run_id` for the same reason as :func:`heal_key`: a backfill is
    rerun idempotently (a chunk interrupted by a spot reclamation is resumed
    by running it again), and each attempt's own record — which sessions were
    produced, which were already present, which refused — is worth keeping
    distinct from the attempt before it.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty — see manifest_key's discriminator for why.")
    return f"backfills/{trading_day}/{run_id}.json"


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

#: The PUBLIC surface and its agent-readable twin (`alpha-engine-config-I10223`).
#: A SEPARATE prefix from `console/`, not a second file beside it, because the
#: prefix is what the serving boundary is drawn on: `public/*` is the only
#: prefix intended to leave the bucket, and a CloudFront origin-access policy
#: scoped to it cannot reach `console/`, `runs/`, `champions/` or `report/` by
#: any request an outside reader can make. A boundary that is a filename
#: convention inside one prefix is a boundary one typo wide.
#: `crucible.console.public` is the one producer.
PUBLIC_KEY = "public/index.html"
PUBLIC_JSON_KEY = "public/index.json"

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

#: The trader's consumer-evidence document (`alpha-engine-config-I10648`).
#: ONE rolling key, not a dated one: `crucible.gate` reads a single document
#: and takes `trading_days` from it, because the question the phase-4 clause
#: asks — "has the trader run a week on the v2 champion" — is cumulative, and a
#: dated artifact would make the gate reconstruct the count by listing, which
#: is a second implementation of a number the trader already knows.
#:
#: **The harness may not reach into the trader** (plan §3: two separate systems
#: coupled by two contract documents). So this key is not the harness inspecting
#: the trader — it is the artifact the trader AGREED to write, declared here so
#: both sides name the same string. `crucible-trader` writes it under its own
#: identity, which may write this key and `runs/trader.read/*` and nothing else
#: in the store.
#:
#: Under `trader/` beside :data:`TRADER_PIN_KEY`, and a module-level CONSTANT
#: rather than a `def trader_evidence_key()` for the same reason that one is:
#: the key is fixed, and a helper returning `f"trader/..."` would enter the
#: population `nous-ergon-ops`'s store-prefix lockstep guard derives
#: the spot-box runtime role's PutObject grants from — granting the box a
#: write on the one prefix it must not have.
TRADER_EVIDENCE_KEY = "trader/evidence.json"

#: The trader's per-session execution-shortfall artifact (`execution_shortfall.v1`,
#: `alpha-engine-config-I10652`) and its per-session shadow books
#: (`shadow_books.v1`, `alpha-engine-config-I10653`). Under `trader/` beside
#: :data:`TRADER_EVIDENCE_KEY` for the same reason: artifacts the trader AGREED
#: to write, written under its own identity, read by the harness
#: (`crucible.execution`). The shadow BOOK is deliberately not under
#: `experiments/` — :func:`shadow_key` there is an arm's grading SELECTION, a
#: different object, and two shapes sharing one word under one prefix is how
#: `predictions/` once held two artifacts distinguishable only by path depth.
TRADER_EXECUTION_SHORTFALL_PREFIX = "trader/execution_shortfall/"
TRADER_SHADOW_BOOKS_PREFIX = "trader/shadow_books/"

#: The trader's daily broker reconciliation (`broker_reconciliation.v1`,
#: `alpha-engine-config-I10651`) and the broker statement it reconciled
#: (`broker_statement.v1`), both written as OUTPUTS of the `trader.reconcile`
#: run manifest (`crucible.models.TRADER_JOB_VALUES`). Under `trader/` for the
#: reason :data:`TRADER_EVIDENCE_KEY` is: artifacts the trader agreed to write,
#: under its own identity. The statement is also tomorrow's reconciliation
#: ANCHOR — the trader reads the newest one strictly before the session — so it
#: is a dated key the trader lists, not a rolling document it overwrites.
#:
#: Declared here, single-source, because `crucible-trader-PR5` shipped both
#: shapes as its own literals pending this module; the trader's pin bump
#: replaces them with these helpers.
TRADER_RECONCILIATION_PREFIX = "trader/reconciliation/"
TRADER_BROKER_STATEMENTS_PREFIX = "trader/broker_statements/"

#: The trader's IB-paper smoke, kill switch and fire drill artifacts
#: (crucible-trader-PR7; `alpha-engine-config-I10649` deliverable 2 and
#: `-I10650` deliverables 1, 2 and 5). Declared here, single-source, because
#: PR7 shipped each shape as its own literal "only until `crucible.keys` owns
#: it"; the trader's pin bump replaces those literals with these names.
#:
#: * :data:`TRADER_PAPER_SMOKE_PREFIX` — `trader_paper_smoke.v1`, one per
#:   (session, release sha), an output of `trader.smoke`.
#: * :data:`TRADER_KILL_SWITCH_KEY` — `kill_switch.v1`, the ONE rolling halt
#:   document every trader session reads before it sends an order. Rolling on
#:   purpose: the halt is state, not history; the history is the events below.
#: * :data:`TRADER_KILL_SWITCH_EVENTS_PREFIX` — `kill_switch_outcome.v1`, one
#:   per fire, keyed by the run that fired it.
#: * :data:`TRADER_FIRE_DRILLS_PREFIX` — `fire_drill.v1`, one per drill, keyed by
#:   the `trader.fire_drill` run; read by
#:   `crucible.gate._clause_kill_switch_fire_drill_passed`.
TRADER_PAPER_SMOKE_PREFIX = "trader/paper_smoke/"
TRADER_KILL_SWITCH_KEY = "trader/kill_switch.json"
TRADER_KILL_SWITCH_EVENTS_PREFIX = "trader/kill_switch/events/"
TRADER_FIRE_DRILLS_PREFIX = "trader/fire_drills/"

#: `fire_drill_schedule.v1` — one SEALED drill schedule per document
#: (`alpha-engine-config-I10761`), written BEFORE the drill it seals by the
#: operator identity, never by the trader. Nested under
#: :data:`TRADER_FIRE_DRILLS_PREFIX` on purpose so the drill and its seal share
#: one read grant, and denied to the trader's WRITE by an explicit IAM Deny in
#: the operated stack: a trader that could write its own seal could seal a
#: drill a minute before firing it. Every reader of the drill prefix skips this
#: sub-prefix (a schedule is not a drill).
TRADER_FIRE_DRILL_SCHEDULE_PREFIX = f"{TRADER_FIRE_DRILLS_PREFIX}schedule/"

_RELEASE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_iso_day(trading_day: str) -> None:
    try:
        dt.date.fromisoformat(trading_day)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"trading_day {trading_day!r} is not an ISO calendar date; a key built from it "
            "would orphan the artifact under a path no reader lists (§4.12)"
        ) from exc


def _require_run_id(run_id: str) -> None:
    if not run_id or "/" in run_id:
        raise ValueError(
            f"run_id {run_id!r} must be one non-empty key segment; an empty or slashed id "
            "would collide with, or nest under, another run's artifact"
        )


def trader_paper_smoke_key(trading_day: str, sha: str) -> str:
    """One `trader_paper_smoke.v1` document: the smoke of release ``sha`` on a session."""
    _require_iso_day(trading_day)
    if not _RELEASE_SHA_RE.match(sha):
        raise ValueError(f"sha {sha!r} is not a 40-hex git sha")
    return f"trader/paper_smoke/{trading_day}/{sha}.json"


def trader_kill_switch_event_key(trading_day: str, run_id: str) -> str:
    """One `kill_switch_outcome.v1` document: what the book did after one fire."""
    _require_iso_day(trading_day)
    _require_run_id(run_id)
    return f"trader/kill_switch/events/{trading_day}/{run_id}.json"


def trader_fire_drill_key(trading_day: str, run_id: str) -> str:
    """One `fire_drill.v1` document, written by the `trader.fire_drill` run ``run_id``."""
    _require_iso_day(trading_day)
    _require_run_id(run_id)
    return f"trader/fire_drills/{trading_day}/{run_id}.json"


def trader_fire_drill_schedule_key(window_start: str, window_end: str, schedule_id: str) -> str:
    """One `fire_drill_schedule.v1` document: a drill sealed for ``[window_start, window_end]``.

    Both window bounds are in the key (so a listing of one window finds every
    seal for it, and the §4.12 walk sees two dates it can hold to the calendar);
    ``schedule_id`` is one key segment.
    """
    _require_iso_day(window_start)
    _require_iso_day(window_end)
    if window_end < window_start:
        raise ValueError(f"window {window_start}..{window_end} ends before it starts")
    if not schedule_id or "/" in schedule_id:
        raise ValueError(f"schedule_id {schedule_id!r} must be one non-empty key segment")
    return f"{TRADER_FIRE_DRILL_SCHEDULE_PREFIX}{window_start}_{window_end}/{schedule_id}.json"


def execution_shortfall_key(trading_day: str) -> str:
    """One session's `execution_shortfall.v1` document, written by the trader."""
    _require_iso_day(trading_day)
    return f"trader/execution_shortfall/{trading_day}.json"


def shadow_books_key(trading_day: str) -> str:
    """One session's `shadow_books.v1` document, written by the trader."""
    _require_iso_day(trading_day)
    return f"trader/shadow_books/{trading_day}.json"


def trader_reconciliation_key(trading_day: str) -> str:
    """One session's `broker_reconciliation.v1` result, written by `trader.reconcile`."""
    _require_iso_day(trading_day)
    return f"trader/reconciliation/{trading_day}.json"


def trader_broker_statement_key(trading_day: str) -> str:
    """One session's `broker_statement.v1` (the broker's positions and cash)."""
    _require_iso_day(trading_day)
    return f"trader/broker_statements/{trading_day}.json"


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


# -- v1 point-in-time snapshots in the DATA bucket (alpha-engine-config-I10721) --
#
# Read-only key shapes of the v1 producer's dated snapshots, read by
# `crucible.data.point_in_time`. Relative to the data bucket root, not the store.


#: Key helpers that name objects in the DATA bucket (`CRUCIBLE_ARCTIC_BUCKET`),
#: not the crucible store. v1's dated point-in-time snapshots are READ as
#: feature inputs (alpha-engine-config-I10721); nothing in crucible writes
#: them, so they are no store prefix for the runtime role to be granted. The
#: ops store-prefix grant guard (`tests/crossrepo/test_crucible_store_prefix_grants.py`)
#: reads this literal and skips these helpers; a name here that is not a
#: module-level function fails that guard.
DATA_BUCKET_KEY_HELPERS: tuple[str, ...] = (
    "fundamental_snapshot_key",
    "constituents_key",
    "inst_ownership_key",
    "edgar_fundamentals_session_key",
)


def fundamental_snapshot_key(label: dt.date) -> str:
    return f"features/{label.isoformat()}/fundamental.parquet"


def constituents_key(label: dt.date) -> str:
    return f"market_data/weekly/{label.isoformat()}/constituents.json"


def inst_ownership_key(year: int, quarter: int) -> str:
    return f"data/inst_ownership/{year}Q{quarter}/latest.parquet"


def edgar_fundamentals_session_key(label: dt.date) -> str:
    """One SEC EDGAR XBRL companyfacts-derived fundamentals session, keyed by
    the NYSE session label it is admissible for the day after
    (`alpha-engine-config-I10733`).

    A sibling of :func:`fundamental_snapshot_key` in a different producer's
    namespace, never a replacement of it in place: the v1 `features/{date}/`
    tree stays read-only for manifests already written against it (the same
    reason `crucible/schemas/run_manifest.v1.json` is frozen rather than
    edited), while `crucible.data.point_in_time.FilingDatePointInTimeSource`
    reads this new tree instead.
    """
    return f"fundamentals_pit/edgar/v1/sessions/{label.isoformat()}.parquet"
