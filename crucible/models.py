"""Typed models for the documents this package READS.

Normative source: `alpha-engine-config-I9847`; Brian, 2026-09-02 — *"I want to
make sure we are using pydantic rather than plain text to ensure we don't run
into any issues down the line."*

This module is the first boundary of that migration and the place its design
is written down. It is deliberately not a 24-module rewrite: `I9847`'s own
constraint is one boundary per PR, each with its contract test, because a
sweeping refactor cannot be adversarially reviewed and the reviews are the
only reason `crucible-PR38`'s defects were found.

── WHY, precisely ────────────────────────────────────────────────────────

The defect is not that untyped reads are ugly. It is **where a malformed
document surfaces**:

* Typed: a named field error at the boundary, naming the document, the row
  and the field.
* Untyped: a `KeyError` three functions in, indistinguishable in a log from
  the reader itself being broken. Measured — `tests/acceptance/check_reading.py`
  read `ratchet["met"]` straight off `json.loads`, and a missing field
  produced exactly that traceback in an adversarial review.

And the sharper half, which no amount of care at the call site fixes: an
untyped read **silently ignores a key it does not know**. A row carrying
`consol_surface:` is not a typo the reader complains about; it is a field
that does nothing, in a file whose whole purpose is to declare what is
observed. `extra="forbid"` is the point of this module at least as much as
the type annotations are.

── WHAT THIS MODULE IS NOT ───────────────────────────────────────────────

**Pydantic does not replace a published JSON Schema.** The versioned schemas
in `crucible/schemas/` are the cross-repo contract surface: other repos read
them, and plan §4.2 and the M0 contract discipline both depend on their being
readable without a Python import. Where an artifact has both, the schema is
**generated from the model** so there is one source of truth rather than two
that must agree — `tests/test_typed_boundaries.py` fails when the committed
file and the generated one differ, which is `I9847` deliverable 3.

**A model is not a validator of semantics.** Cross-field rules that carry a
measured incident in their message — "six rows read `weekly, Saturday` while
the scheduler dispatched exactly one of them" — stay written out, as
`model_validator`s here rather than as hand-rolled checks in the reader. They
move; they are not diluted.

── THE ORDER THE REST LANDS IN ───────────────────────────────────────────

Boundaries are taken in descending order of *how far a malformed document
travels before anything notices*, not in file order. `components.yaml` is
first because it is the declaration that decides what is watched at all: a
row silently missing a field makes a component unobserved, and unobserved
reads exactly like healthy. The remaining boundaries are filed as children of
`alpha-engine-config-I9847`.

── THE SECOND BOUNDARY: THE RUN MANIFEST ─────────────────────────────────

`RunManifestV2` (`alpha-engine-config-I10045` row 1) is regenerated the same
way as `components_registry.v1.json`: `run_manifest.v2.json` IS
`RunManifestV2.model_json_schema()` (`tests/test_manifest_schema.py` fails
when the committed file and the generated one differ). The two
run-manifest-status cross-field rules (`status: ok` implies empty `reason`,
`status: failed` implies non-empty `reason`) stay enforced by
`RunManifestV2._status_and_reason_agree` as the source of truth — **and are
also emitted into the published schema's `allOf`**, by
`_run_manifest_v2_json_schema_extra` mirroring the validator exactly, so a
consumer with no Python import still gets the rule rather than a weaker
contract (PR123 review finding 3: the first draft dropped the rule from the
published schema entirely, the same choice `ComponentRow`'s dispatch/deadline
rules made for `components_registry.v1.json` — but those rules read a
sibling field's *presence*, where this one reads a sibling field's *value*
against a two-branch enum, which is exactly what `allOf`/`if`/`then` can
state declaratively without duplicating the model's control flow). The model
stays the enforcement point on every path that imports Python; the schema
carries the same rule for the one that does not.

`run_manifest.v1.json` is untouched — FROZEN by `alpha-engine-config-I9918`,
read by `crucible.manifest.load_schema_for("run_manifest.v1")` exactly as
before, and carries no model. A v1 document is still checked against the raw
v1 schema file; only a `run_manifest.v2` document is validated through
`RunManifestV2`.

`MetricRecordRow` is NOT `krepis.metrics.MetricRecord` subclassed — measured
before writing it: live and test manifests write `metric_type` values
(`"operational"`, `"coverage"`, `"gauge"`, `"drift"`, `"control"`,
`"decision"`, `"repair"`) outside `krepis.metrics.MetricTypeLiteral`, which is
scoped to the System Report Card v2, a different producer set. Subclassing
and widening `status` alone would have silently narrowed `metric_type` and
broken every one of those manifests. `status` still needs widening for the
same reason in the other direction: crucible's own producers
(`crucible.data`/`alerts`/`deploy`/`slots.cycle`/`track_c`'s `OK`/`FAIL`/
`BREACH`, and `crucible.promote`'s forwarded arena vocabulary) write states
`krepis.metrics.StatusLiteral` does not carry, which is why
`METRIC_STATUS_VALUES` extends `typing.get_args(StatusLiteral)` rather than
restating either vocabulary — `tests/test_manifest_schema.py`'s
`TestTheMetricStatusVocabularyIsDerivedNotRestated` (pre-existing, unchanged)
still holds because that derivation happens here, not in a second copy.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal, get_args

from krepis.metrics import StatusLiteral
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ARM_RECIPE_REQUIRED_FIELDS",
    "ArenaCycleDocument",
    "ArmRecipeDocument",
    "ArtifactRef",
    "AttemptRow",
    "ComponentRow",
    "ComponentsDocument",
    "DeadlineRow",
    "LlmCallRow",
    "LlmCallSiteRow",
    "LlmCallsiteRegistryDocument",
    "METRIC_STATUS_VALUES",
    "MetricRecordRow",
    "RegistryDefaults",
    "RejectedRow",
    "ResourceRow",
    "RunManifestV2",
    "SignalsRow",
]


class _Strict(BaseModel):
    """Every model here forbids what it does not declare.

    `extra="forbid"` is the half of this migration that a careful call site
    cannot substitute for: a key the reader does not know is an edit somebody
    made and nothing performed.
    """

    model_config = ConfigDict(extra="forbid")


class SignalsRow(_Strict):
    """§9.2's five signal classes, always all five.

    A class the job legitimately does not emit is `null` — DECLARED, not
    omitted, because an omitted class is indistinguishable from a forgotten
    one. That is why every field is required and nullable rather than
    optional with a default: a default would let an omission read as a
    declaration.
    """

    execution: str | None
    cost: str | None
    resource: str | None
    lineage: str | None
    outcome: str | None


class DeadlineRow(_Strict):
    """The structured deadline `crucible.components.Deadline` is built from.

    Two anchors, exhaustively; a third is a design change visible in a diff,
    for the same reason the two page conditions are a closed set.
    """

    anchor: Literal["close_plus", "next_calendar_day_at"]
    offset_hours: float | None = None
    at: dt.time | None = None

    @model_validator(mode="after")
    def _the_anchor_carries_the_field_it_needs(self) -> DeadlineRow:
        if self.anchor == "close_plus" and self.offset_hours is None:
            raise ValueError("a close_plus deadline needs offset_hours")
        if self.anchor == "next_calendar_day_at" and self.at is None:
            raise ValueError("a next_calendar_day_at deadline needs `at`")
        return self


class ComponentRow(_Strict):
    """One CLI job's observability declaration.

    A component with no row is unobserved, not healthy: its absence cannot
    page (§4.6 reads `deadline` from this file), its logs have no declared
    location or retention, and the console has no surface to render it on.
    """

    description: str = Field(min_length=1)
    lifecycle: Literal["ACTIVE", "DISABLED", "RETIRED"] = "ACTIVE"
    signals: SignalsRow
    log_location: str = Field(min_length=1)
    #: CALENDAR days — one of §4.12's exhaustive exceptions, because retention
    #: is an AWS property billed by calendar time.
    log_retention_days: Annotated[int, Field(ge=1)] | Literal["forever"]
    alert_channel: str = Field(min_length=1)
    console_surface: str = Field(min_length=1)
    artifact_retention: str = Field(min_length=1)
    #: Prose. `dispatch` below is the wiring, and they are separate fields
    #: because six rows read "weekly, Saturday" while exactly one of them was
    #: dispatched by anything.
    schedule: str | None
    #: WHO starts a scheduled row. Required and nullable: `null` is
    #: "on-demand", an assertion, and it must not be spellable by leaving the
    #: key out.
    dispatch: Literal["arc", "scheduler", "github-actions"] | None
    dispatch_workflow: str | None = None
    absence_watched_by: str = "alerts.sweep"
    deadline: DeadlineRow | None

    @model_validator(mode="after")
    def _a_scheduled_row_names_who_starts_it(self) -> ComponentRow:
        """Every scheduled row names its starter; no other row does.

        The failure this prevents is one that shipped: six rows read
        `schedule: weekly, Saturday` while the scheduler dispatched exactly
        one of them, so five components were deadlined, watched for absence,
        and started by nobody.
        """
        scheduled = self.schedule is not None
        if scheduled and self.dispatch is None:
            raise ValueError(
                "is scheduled but its dispatch is null. Something has to start it, "
                "and a row naming no starter is a job whose absence pages every "
                "cycle for work nobody was going to run."
            )
        if not scheduled and self.dispatch is not None:
            raise ValueError(
                f"is on-demand but declares dispatch {self.dispatch!r}; an "
                "unscheduled job is started by a person or another job, and naming "
                "a starter here would claim a cadence it does not have."
            )
        return self

    @model_validator(mode="after")
    def _an_on_demand_row_declares_no_deadline(self) -> ComponentRow:
        if self.schedule is None and self.deadline is not None:
            raise ValueError(
                "is on-demand but declares a deadline; its absence is not a fact "
                "about the system and the deadline would page for nothing."
            )
        return self


class RegistryDefaults(_Strict):
    """§4.6: exactly two page conditions, and one channel either reaches."""

    page_channel: str = Field(min_length=1)
    quiet_channel: str = Field(min_length=1)


class ComponentsDocument(_Strict):
    """`crucible/components.yaml`, whole.

    Read by this package AND, across the repo boundary, by
    `nous-ergon-ops/tests/crossrepo/test_crucible_dispatch_lockstep.py`, which
    is why it is a versioned artifact with a published schema rather than a
    private config file: the M0 contract discipline applies to every
    cross-repo artifact, and this is one.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"$id": "components_registry.v1"})

    version: Literal[1]
    defaults: RegistryDefaults
    components: dict[str, ComponentRow] = Field(min_length=1)


# ── The run manifest (`alpha-engine-config-I10045` row 1) ─────────────────

#: `job` is a closed set restated here, matched to `crucible/components.yaml`
#: plus `deploy` (the one permitted difference — `deploy.yml` writes a
#: manifest in this schema so §4.5's page shows deploys beside runs, and it
#: is not a `crucible.cli.JOBS` entry). Restated rather than derived from
#: `crucible.components.load_registry` to avoid a `models -> components ->
#: models` import cycle (`components.py` already imports `ComponentsDocument`
#: from this module); `tests/test_components_registry.py`'s
#: `test_the_manifest_schema_job_enum_matches_the_registry` is the drift
#: guard, in both directions, exactly as it was before this migration.
JOB_VALUES: tuple[str, ...] = (
    "data.daily",
    "data.weekly",
    "data.heal",
    "experiment.new",
    "experiment.run",
    "experiment.grade",
    "promote",
    "report",
    "explain",
    "migrate.history",
    "release.pin",
    "release.lock",
    "smoke",
    "alerts.sweep",
    "heartbeat",
    "drift",
    "console",
    "board",
    "deploy",
    "weekly",
    "gate",
    "gate.close",
    "report.morning",
)

#: The exhaustive `attempts[].reason` vocabulary: `initial` for the first
#: attempt, otherwise one of `crucible.manifest.TRANSIENT_RETRY_REASONS`.
#: Restated rather than imported: `crucible.manifest` imports this module, so
#: the reverse import would be circular. `tests/test_manifest_schema.py`
#: pins the two lists equal.
ATTEMPT_REASON_VALUES: tuple[str, ...] = (
    "initial",
    "spot_interruption",
    "provider_5xx",
    "provider_timeout",
    "s3_throttling",
)

#: `metricRecord.status`, DERIVED rather than restated (the defect this
#: guards against: `main` was red for eleven minutes on 2026-09-01 because
#: two branches each restated a *different* half of this vocabulary). The
#: krepis half is `krepis.metrics.derive_status`'s own return type, forwarded
#: verbatim by `crucible.report`'s five attribution rows; the extra half is
#: crucible's own producers' coverage-and-ceiling vocabulary
#: (`crucible.data`/`alerts`/`deploy`/`slots.cycle`/`track_c`) plus
#: `crucible.promote`'s forwarded `nousergon_lib.arena.engine` decision
#: vocabulary, plus `crucible.drift`'s `UNREPORTED`. Neither half is
#: sufficient alone; a value must be added to whichever half is its source of
#: truth, never spelled fresh here.
_METRIC_STATUS_EXTRAS: tuple[str, ...] = (
    "OK",
    "FAIL",
    "BREACH",
    "unservable",
    "bootstrap",
    "unmeasurable",
    "measured",
    "decided",
    "held",
    "UNREPORTED",
)
METRIC_STATUS_VALUES: tuple[str, ...] = get_args(StatusLiteral) + _METRIC_STATUS_EXTRAS

_ISO_DATE_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
_UTC_TIMESTAMP_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

IsoDate = Annotated[str, Field(pattern=_ISO_DATE_PATTERN, json_schema_extra={"format": "date"})]
UtcTimestamp = Annotated[
    str,
    Field(
        pattern=_UTC_TIMESTAMP_PATTERN,
        description="RFC 3339, UTC, `Z` suffix. A local-time timestamp in an "
        "artifact keyed by trading day is a date bug waiting to happen.",
    ),
]
GitSha = Annotated[str, Field(pattern=_GIT_SHA_PATTERN)]
Sha256 = Annotated[str, Field(pattern=_SHA256_PATTERN)]


class ArtifactRef(_Strict):
    """§9.2 class 4: one artifact a job read or wrote, content-hashed.

    A store key rather than a URI, so the same manifest is portable between
    the S3 and local-dir store backends; `schema_version` is required
    because an un-versioned cross-module artifact is the M0 contract
    violation.
    """

    key: str = Field(
        min_length=1,
        description=(
            "Store key, not a URI. The store backend (S3 or local dir) resolves it, so the same "
            "manifest is portable between them."
        ),
    )
    sha256: Sha256
    schema_version: str = Field(
        min_length=1,
        description=(
            "The version the artifact was read or written under. Required — an un-versioned "
            "cross-module artifact is the M0 contract violation."
        ),
    )


class LlmCallRow(_Strict):
    """§9.2 class 2: one logical LLM call.

    `model_requested`/`model_served` are separate because a routed call that
    silently served a different model is the failure this field exists to
    make visible (principle 8: never a provider model id at the call site,
    always a capability class or registry group). `route_degraded` and
    `fallback_used` are both required booleans and are separate facts —
    resolve-time versus call-time — because on the router-edge route the
    fallback chain is walked by the proxy AFTER resolution, so a route can
    resolve healthy while the call is actually served by a fallback
    (`alpha-engine-config-I9969`, `I10006`).
    """

    callsite_id: str = Field(
        min_length=1,
        description=(
            "Key into LLM_CALLSITE_REGISTRY. Registry coverage of v2 call sites is 100%, and that "
            "is a test."
        ),
    )
    model_requested: str = Field(
        min_length=1,
        description=(
            "What the caller asked the router for — a capability class or registry group, never a "
            "provider model id (principle 8)."
        ),
    )
    model_served: str = Field(
        min_length=1,
        description=(
            "What the router actually served. Required and separate from `model_requested`: a "
            "routed call that silently served a different model is the failure this field exists "
            "to make visible."
        ),
    )
    route_degraded: bool = Field(
        description=(
            "Whether RESOLUTION had already fallen past the group's primary entry when this call "
            "was routed (model-router-policy R12: serving from a fallback is an alert, not a log "
            "line). Required and boolean, so a fallback-served call is distinguishable from a "
            "primary-served one in the durable record rather than only in a log; `false` is an "
            "assertion that the primary served, never an absence. Added by I9969 WITHOUT a version "
            "bump, unlike `run_mode`: the field is on `llmCall`, `crucible.llm.call` is the only "
            "writer of one, and it could not complete a call at all before this change (the "
            "registry it admits against is empty and the spec it resolved came from an SSM path "
            "that does not exist) — so no manifest in the store carries an `llm_calls` row this "
            "could retroactively invalidate."
        )
    )
    #: `null` is the router reporting no deployment — an ANSWER, not an
    #: absence — so the key is required and the value is nullable
    #: (`alpha-engine-config-I10006`, `I9995`).
    fallback_used: bool = Field(
        description=(
            "Whether a fallback entry in the group's chain actually SERVED this call — the "
            "call-time fact, from `krepis.llm.LLMResult.fallback_used`. Distinct from "
            "`route_degraded`, which is the resolve-time fact about the route object: on the "
            "router-edge route the chain is walked by the proxy AFTER resolution, so resolution "
            "can report a healthy route while the proxy serves from a fallback, and "
            "`route_degraded` alone would then read `false` beside a call the primary never "
            "answered. Both are recorded because they answer different questions and neither can "
            "be derived from the other; a run whose arm was served by a fallback and whose "
            "manifest says the primary served is a run that grades a model nobody selected. Added "
            "by I10006 WITHOUT a version bump, on the same reasoning `route_degraded` records: the "
            "field is on `llmCall`, `crucible.llm.call` is its only writer, and "
            "`crucible/llm_callsites.yaml` still declares `callsites: {}` so the door has never "
            "completed a call — no manifest in the store carries an `llm_calls` row this could "
            "retroactively invalidate."
        )
    )
    served_deployment: str | None = Field(
        description=(
            "The deployment name the router reported for this call — `{group}-{mid}` on the router "
            "edge, the upstream model id elsewhere — from "
            "`krepis.llm.LLMResult.served_deployment`. `null` when the route reported none, which "
            "is an absence of the router's own answer and is NOT the same as an absent field. "
            "Recorded beside `model_served` rather than folded into it: `model_served` is the "
            "resolved upstream id price cards key on, and two deployments can share one upstream "
            "model, so the comparison that decides `fallback_used` happens at the deployment layer "
            "and an artifact reader cannot reconstruct it from `model_served` alone (krepis, "
            "I9995). Added by I10006 without a version bump, same reasoning as `fallback_used`."
        )
    )
    tokens_in: Annotated[int, Field(ge=0)]
    tokens_out: Annotated[int, Field(ge=0)]
    cache_read: Annotated[int, Field(ge=0)]
    cache_write: Annotated[int, Field(ge=0)]
    usd: Annotated[float, Field(ge=0)]


class ResourceRow(_Strict):
    """§9.2 class 3. Present on every manifest including laptop runs, where
    `spot` is false and `instance_type` is `local`."""

    instance_type: str = Field(
        min_length=1, description="EC2 instance type, `lambda`, or `local` for a laptop run."
    )
    spot: bool
    #: I5727: spot-to-on-demand fallback is a COUNTABLE metric, so it is
    #: required rather than an optional annotation.
    escalated_to_on_demand: bool = Field(
        description=(
            "I5727: spot-to-on-demand fallback emits a COUNTABLE metric. Required, so a fleet-wide "
            "escalation rate is always computable rather than sampled from whichever runs happened "
            "to annotate it."
        )
    )
    interruptions: Annotated[int, Field(ge=0)]
    mem_peak_mb: Annotated[float, Field(ge=0)]
    disk_free_mb: Annotated[float, Field(ge=0)]


class RejectedRow(_Strict):
    """Rejections WITH REASON, never a bare count — a count with no reason
    cannot be acted on, and is how 901 of 903 tickers silently failed a
    liquidity gate for months."""

    reason: str = Field(min_length=1, max_length=200)
    count: Annotated[int, Field(ge=1)]


class AttemptRow(_Strict):
    """§11 risk 2: one attempt this run made, including the declared
    transient-class retry. A run that executed once records one attempt, so a
    retried run cannot be mistaken for a clean first-attempt run."""

    n: Annotated[int, Field(ge=1)]
    reason: Literal[ATTEMPT_REASON_VALUES] = Field(
        description=(
            "Why THIS attempt happened. `initial` for the first; otherwise the declared transient "
            "class that caused the retry. The class grows only by PR with the failure named."
        )
    )  # type: ignore[valid-type]


class MetricRecordRow(BaseModel):
    """§9.2 class 5. `MetricRecord`-shaped (`krepis.metrics.MetricRecord`),
    open on extra fields for forward compatibility with a newer producer, but
    strict about the two that make a number readable.

    Deliberately its OWN model rather than `krepis.metrics.MetricRecord`
    subclassed — see this module's docstring: crucible's producers write
    `metric_type` values (`operational`, `coverage`, `drift`, ...) outside
    `krepis.metrics.MetricTypeLiteral`, which is scoped to the System Report
    Card v2. Reusing that model directly would have silently rejected them.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "description": (
                "MetricRecord-shaped. Open (`additionalProperties: true`) to match "
                "krepis.metrics.MetricRecord's forward-compat contract, but the required core and "
                "the two constrained fields below are enforced here."
            )
        },
    )

    name: str = Field(min_length=1)
    module: str = Field(min_length=1)
    metric_type: str = Field(min_length=1)
    value: float | None = None
    #: Required whenever `value` is set — see the validator below. A numeric
    #: field with no declared unit is the defect that emitted
    #: `avg_volume_20d` as a ratio and consumed it as raw shares.
    unit: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Required whenever `value` is set — see the dependency below. A numeric field with no "
            "declared unit is the defect that emitted avg_volume_20d as a ratio and consumed it as "
            "raw shares."
        ),
    )
    n_floor: Annotated[int, Field(ge=0)]
    status: Literal[METRIC_STATUS_VALUES] = Field(  # type: ignore[valid-type]
        description=(
            "Closed set (I9757, defect #3): a run's own `status` above is `ok`/`failed` by schema "
            "so a third state cannot be spelled there, but §2 row 4 ('no degraded-SUCCEEDED') is a "
            "claim about every level a human reads, and a metric nested inside an `ok` manifest is "
            "one of them — `migrate.history --allow-missing` once wrote "
            "`DEGRADED_BY_OPERATOR_CONSENT` here, inside a manifest whose own `status` said `ok`, "
            "which reintroduces exactly the third state the top-level enum exists to forbid. "
            "Enumerated rather than left open: `OK`/`FAIL`/`BREACH` are the coverage-and-ceiling "
            "vocabulary shared by `crucible.data`, `crucible.alerts`, `crucible.deploy`, "
            "`crucible.slots.cycle` and `crucible.track_c`; "
            "`unservable`/`bootstrap`/`unmeasurable`/`measured`/`decided`/`held` are "
            "`nousergon_lib.arena.engine`'s own champion/challenger decision vocabulary, forwarded "
            "verbatim by `crucible.promote`'s `pointer_moved` metric rather than re-encoded into a "
            "second vocabulary that could drift from the first. `GREEN`/`WATCH`/`RED` and the four "
            "`N/A-*` states are `krepis.metrics.StatusLiteral` — the vocabulary "
            "`krepis.metrics.derive_status` returns and `MetricRecord` carries — forwarded "
            "verbatim by `crucible.report`'s five attribution rows for the same reason the arena's "
            "vocabulary is: re-encoding them into a second set is how two spellings of one state "
            "drift apart. They were MISSING when the enum was closed, because the audit that built "
            "it read the call sites that existed on that branch and `crucible.report` was on "
            "another one: two PRs green apart and red together, which is what `main` looked like "
            "for eleven minutes on 2026-09-01. `tests/test_manifest_schema.py` now derives the "
            "requirement from `typing.get_args(StatusLiteral)` rather than restating it. A "
            "producer needing a value outside this set grows the enum by PR, with the new state "
            "named here, same discipline as `TRANSIENT_CLASSIFIERS` — never by writing a word this "
            "schema does not know and letting `additionalProperties` wave it through. `UNREPORTED` "
            "is `crucible.drift`'s no-value status (`Band.status`, plan §10 component 5): a drift "
            "row whose input history has not settled yet renders `UNREPORTED` with its reason, "
            "never a green, and `crucible.console.classify` already counts a metric in that state "
            "toward the transparency gap. Added 2026-09-05 when the first arc to reach `drift` "
            "(weekly@2026-08-07) showed the row could not be filed at all — the word was in the "
            "console's vocabulary and not in this one."
        )
    )
    status_reason: str = Field(
        min_length=1, description="One operator-readable sentence, never generic."
    )
    source_path: str = Field(min_length=1)
    last_updated_utc: UtcTimestamp
    #: §4.12: a horizon is an INTEGER COUNT OF TRADING DAYS — 21/63/126/252.
    #: There is no string form, so `"1 month"` fails validation. Absent when
    #: the metric has no forward horizon.
    horizon_trading_days: Annotated[int, Field(ge=1)] | None = Field(
        default=None,
        description=(
            "§4.12: a horizon is an INTEGER COUNT OF TRADING DAYS — 21 / 63 / 126 / 252. There is "
            "no string form, so `1 month` cannot be expressed and fails validation. Absent when "
            "the metric has no forward horizon."
        ),
    )

    @model_validator(mode="after")
    def _unit_required_when_value_present(self) -> MetricRecordRow:
        if self.value is not None and not self.unit:
            raise ValueError(
                f"{self.name!r} sets value={self.value!r} with no unit. A numeric "
                "value with no declared unit is not a measurement a consumer can "
                "safely render or compare."
            )
        return self


def _run_manifest_v2_json_schema_extra(schema: dict[str, object]) -> None:
    """Sets `run_manifest.v2.json`'s document-level metadata and mirrors
    `RunManifestV2._status_and_reason_agree` into the schema's `allOf`
    (`alpha-engine-config-I10045` row 1, PR123 review finding 3).

    The `allOf` here must stay byte-for-byte what the validator enforces: it
    exists so a consumer with no Python import — the case the published
    schema is FOR — gets the same status<->reason rule a
    `crucible.manifest.validate` caller gets. It is not the source of truth;
    the model_validator is, and this function is the one place that copies
    it, so a future change to one and not the other is a diff in one
    function rather than a silent drift between two independent call sites.
    """
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/nousergon/crucible/schemas/run_manifest.v2.json"
    schema["title"] = "Crucible run manifest, v2"
    schema["description"] = (
        "The record every job writes at runs/{job}/{trading_day}/run.json. This "
        "schema is the enforcement surface for the plan's central guarantee (§4.2): "
        "a run is `ok` or it is `failed`; there is no third state, and it is "
        "excluded here rather than by policy. It is also the observability contract "
        "(§9.2): one document carries all five signal classes — execution, "
        "cost/tokens, resource, data lineage, and outcome vs baseline — under one "
        "run_id. v2 adds one REQUIRED field, `run_mode`, and is otherwise "
        "byte-for-byte v1's contract. v1 declared `additionalProperties: false` and "
        "no live/replay field, so no producer could say whether a run was a live "
        "Saturday or a replay of a historical one, and phase 2's exit gate (plan §6 "
        "row 2 / §6.1) was permanently UNMEASURABLE — I9918. It "
        "is a NEW VERSION rather than a v1 addition because a required field added "
        "to v1 would retroactively invalidate every manifest already in the store: "
        "`run_manifest.v1.json` stays in this package, frozen, and every object "
        "written under it stays readable at its own declared version. Nothing "
        "backfills those objects — a manufactured `run_mode` on a run nobody "
        "observed is exactly the false liveness claim this field exists to prevent."
    )
    schema["allOf"] = [
        {
            "description": (
                "A failed run states why. Enforced in the schema so no writer can omit it."
            ),
            "if": {"properties": {"status": {"const": "failed"}}, "required": ["status"]},
            "then": {"properties": {"reason": {"minLength": 1}}},
        },
        {
            "description": (
                "An ok run has nothing to explain; a non-empty reason on success is a "
                "degraded-SUCCEEDED in disguise."
            ),
            "if": {"properties": {"status": {"const": "ok"}}, "required": ["status"]},
            "then": {"properties": {"reason": {"const": ""}}},
        },
    ]


class RunManifestV2(_Strict):
    """`run_manifest.v2`, whole — the record every job writes at
    `runs/{job}/{trading_day}/run.json` (plan §4.2, §9.2).

    Read by `crucible.manifest.validate`/`read_manifest`, and by every module
    the run manifest lists as a boundary consumer
    (`alpha-engine-config-I10045` row 1: `crucible.gate`, `.alerts`,
    `.runner`, `.deploy`, `.release`, `.report`, `.explain`, `.console`).
    Those consumers keep reading the plain `dict` `read_manifest` returns —
    this model is the validator at the boundary, not a new return type, so
    converting one boundary does not require editing every consumer in the
    same PR (`I9847`'s own constraint).

    `run_manifest.v1.json` stays FROZEN and carries no model
    (`alpha-engine-config-I9918`): a v1 document is still checked against the
    raw v1 schema file by `crucible.manifest.validate`, never through this
    class, because a required field added here would retroactively
    invalidate every v1 manifest already in the store.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=_run_manifest_v2_json_schema_extra,
    )

    schema_version: Literal["run_manifest.v2"] = Field(
        description=(
            "Version of THIS schema. A consumer that cannot read the version refuses the document "
            "rather than guessing."
        )
    )
    run_id: str = Field(
        pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$",
        description=(
            "The correlation identity (§9.2). Appears on every log line, alert, cost row and S3 "
            "object's metadata for this run. ULID: lexically sortable by creation time."
        ),
    )
    job: Literal[JOB_VALUES] = Field(  # type: ignore[valid-type]
        description=(
            "The CLI job that produced this manifest. Closed set: a job absent from "
            "crucible/components.yaml has no declared log location, alert channel or retention, "
            "and is therefore unobserved."
        )
    )
    #: Whether this run was a LIVE execution or a REPLAY of a historical
    #: trading day. Required, closed vocabulary, deliberately NO DEFAULT: a
    #: default is what makes a replay indistinguishable from a live run the
    #: first time a producer forgets to set it (`alpha-engine-config-I9918`).
    #: Set from the invocation (`crucible.runmode.resolve_run_mode`), never
    #: derived from `trading_day`/`calendar_date`.
    run_mode: Literal["live", "replay"] = Field(
        description=(
            "Whether this run was a LIVE execution or a REPLAY of a historical trading day. "
            "REQUIRED, closed vocabulary, and deliberately NO DEFAULT: a default is what makes a "
            "replay indistinguishable from a live run the first time a producer forgets to set it, "
            "so absence is a schema violation rather than a silent `live`. The value comes from "
            "the INVOCATION — `crucible --run-mode` or $CRUCIBLE_RUN_MODE, resolved by "
            "`crucible.runmode.resolve_run_mode`, which refuses when neither says. It is NEVER "
            "derived from `trading_day` or `calendar_date`: plan §6.1 replays past Saturdays on an "
            "accelerated schedule, so a date-derived reading would call exactly those replays "
            "live, which is the `green on replays` failure phase 2's clause list was withheld for "
            "(I9918). Read by `crucible.gate._clause_live_saturdays_first_attempt_ok`."
        )
    )
    #: THE KEY (§4.12). Never the wall-clock date.
    trading_day: IsoDate = Field(
        description=(
            "THE KEY (§4.12). An NYSE trading day from krepis.trading_calendar. A run launched on "
            "a non-trading day binds to the last completed trading day — a Saturday weekly run is "
            "keyed to Friday's close. Never the wall-clock date."
        )
    )
    #: Present only for a job that legitimately writes more than one
    #: manifest for the same `job`+`trading_day` (`alpha-engine-config-
    #: I9781`). Absent for every other job.
    discriminator: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]{1,64}$",
        description=(
            "Present only for a job that legitimately writes more than one manifest for the same "
            "`job`+`trading_day` — the slot letter (u/r/m/s) for "
            "`experiment.run`/`experiment.grade`, or the firing's own `calendar_date` for "
            "`alerts.sweep`, which runs every calendar day and can therefore fire more than once "
            "against one trading day. Absent for every other job. Never encodes a fact `job` or "
            "`trading_day` already carries (I9781); see `crucible.manifest.manifest_key`."
        ),
    )
    #: Wall-clock date the run actually executed on, for PROVENANCE ONLY.
    #: Never a key, never an input to a promotion/retirement/freshness/
    #: grading decision.
    calendar_date: IsoDate = Field(
        description=(
            "Wall-clock date the run actually executed on, recorded for provenance ONLY. Never "
            "used as a key, never an input to a promotion, retirement, freshness or grading "
            "decision."
        )
    )
    #: Exhaustive. There is deliberately no `partial`, `skipped`, `degraded`
    #: or `unknown`.
    status: Literal["ok", "failed"] = Field(
        description=(
            "Exhaustive. `ok` means a complete manifest with every output written; `failed` means "
            "the run did not produce its deliverable and pages. There is deliberately no "
            "`partial`, `skipped`, `degraded` or `unknown` — the v1 system's `cycle_verdict: "
            "unknown` (I9729) and its 32-of-33 no-op skip flags (I9721) are unrepresentable here."
        )
    )
    #: Empty when `status` is `ok`; a specific, operator-readable cause when
    #: `failed` — enforced by `_status_and_reason_agree` below AND, for a
    #: consumer with no Python import, by the published schema's `allOf`
    #: (see `_run_manifest_v2_json_schema_extra`).
    reason: str = Field(
        max_length=2000,
        description=(
            "Mandatory. Empty string when `status` is `ok`; a specific, operator-readable cause "
            "when `failed` — the conditional below forbids an empty reason on a failure, because a "
            "failure with no cause is indistinguishable from every other failure."
        ),
    )
    started: UtcTimestamp
    #: Written in the runner's `finally` block, so it is present even when
    #: the job raised.
    finished: UtcTimestamp = Field(
        description=(
            "Written in the runner's `finally` block, so it is present even when the job raised."
        )
    )
    code_sha: GitSha = Field(
        description=(
            "Commit sha of the crucible tree that ran. Half of `explain`'s answer to 'why did it "
            "do that'."
        )
    )
    release_sha: GitSha = Field(
        description=(
            "Sha of the immutable release artifact this process was installed from. Differs from "
            "code_sha only when running from a working tree, where it is the same value."
        )
    )
    #: Required, including for jobs that are deterministic today: an
    #: unrecorded seed makes a replay diff unattributable.
    seed: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "The RNG seed. Required, including for jobs that are deterministic today: an "
                "unrecorded seed makes a replay diff unattributable."
            ),
        ),
    ]
    inputs: list[ArtifactRef] = Field(
        description=(
            "§9.2 class 4, upstream half. Every artifact read, content-hashed, with the schema "
            "version it was read under. An empty list is legal and means the job read nothing."
        )
    )
    outputs: list[ArtifactRef] = Field(
        description=(
            "§9.2 class 4, downstream half. Content hashes are what make a rerun idempotent: "
            "identical bytes are a no-op."
        )
    )
    rows_in: Annotated[int, Field(ge=0)]
    rows_out: Annotated[int, Field(ge=0)]
    rows_rejected: list[RejectedRow] = Field(
        description=(
            "Rejections WITH REASON, never a bare count. A count with no reason cannot be acted "
            "on, and is how 901 of 903 tickers silently failed a liquidity gate for months."
        )
    )
    #: Must be >= the sum of `llm_calls[].usd`; the runner asserts that,
    #: since a schema cannot.
    cost_usd: Annotated[
        float,
        Field(
            ge=0,
            description=(
                "Total spend attributable to this run: LLM plus metered infrastructure. Must be >= "
                "the sum of llm_calls[].usd; the runner asserts that, since a schema cannot."
            ),
        ),
    ]
    #: Empty for every non-research job — LLM access is confined to research
    #: by standing rule.
    llm_calls: list[LlmCallRow] = Field(
        description=(
            "§9.2 class 2. One entry per logical call. Empty for every non-research job — LLM "
            "access is confined to research by standing rule, and an empty list is the evidence of "
            "that, not an absence of instrumentation."
        )
    )
    resource: ResourceRow = Field(
        description=(
            "§9.2 class 3. Present on every manifest including laptop runs, where `spot` is false "
            "and the instance type is `local`."
        )
    )
    metrics: list[MetricRecordRow] = Field(
        description=(
            "§9.2 class 5. MetricRecord-shaped (krepis.metrics.MetricRecord), permissive on extra "
            "fields for forward compatibility with a newer producer, but strict about the two that "
            "make a number readable: `unit` whenever `value` is set, and `horizon_trading_days` as "
            "an integer count of TRADING days."
        )
    )
    #: Never empty: a run that executed once records one attempt.
    attempts: list[AttemptRow] = Field(
        min_length=1,
        description=(
            "§11 risk 2. Every attempt this run made, including the transient-class retry. Never "
            "empty: a run that executed once records one attempt, so a retried run cannot be "
            "mistaken for a clean first-attempt run."
        ),
    )

    @model_validator(mode="after")
    def _status_and_reason_agree(self) -> RunManifestV2:
        """The plan's central guarantee (§4.2): a failed run states why, and
        an ok run has nothing to explain.

        This is the cross-field rule this module's docstring names. It stays
        the model_validator (the enforcement point for every Python-import
        caller) AND is mirrored into the published schema's `allOf` by
        `_run_manifest_v2_json_schema_extra`, so the two cannot drift apart
        silently: a change here that is not also made there is caught by
        `tests/test_manifest_schema.py`'s plain-`jsonschema` rejection test,
        which validates against the committed file with no model import.
        """
        if self.status == "failed" and self.reason == "":
            raise ValueError(
                "status is `failed` but `reason` is empty. A failed run states why; "
                "an empty reason is indistinguishable from every other failure."
            )
        if self.status == "ok" and self.reason != "":
            raise ValueError(
                f"status is `ok` but reason={self.reason!r}, not empty. An ok run "
                "has nothing to explain, and a non-empty reason on success is a "
                "degraded-SUCCEEDED in disguise."
            )
        return self


# ── I10045 row 2: the arm register ─────────────────────────────────────────
# Additive only. New boundaries land as new classes appended below this
# marker so a rebase across concurrent rows stays a pure append-append.

#: §9.1 pre-registration, restated here (not imported from
#: `crucible.slots.arms`) so the model has no import-time dependency on the
#: reader it types. `crucible.slots.arms.REQUIRED_ARM_FIELDS` stays the
#: public name other modules import; `tests/test_public_surface.py` pins
#: that the two tuples are equal so they cannot drift silently.
ARM_RECIPE_REQUIRED_FIELDS: tuple[str, ...] = ("name", "slot", "ranker", "params", "registered_at")


class ArmRecipeDocument(_Strict):
    """One filed U/R arm recipe, `arms/{slot}/{name}.yaml`.

    Normative source: `champion-challenger-policy.md` §3, §3.1, §4; plan
    §4.4; §9.1 pre-registration. `crucible.slots.arms._parse` used to build
    this by hand: a missing field raised a `KeyError` on the field it forgot
    to check for, not the one that was actually missing, and an unknown
    top-level key registered cleanly and did nothing — the same silent-typo
    failure `alpha-engine-config-I9944` named for `components.yaml`.

    Three checks stay in the READER rather than moving onto this model,
    because each needs something a document-shape model cannot carry:

    * `ranker` naming a real ranking callable — checked against
      `crucible.slots.rankers.get_ranker`'s live registry, and raised as
      `KeyError` (`"unknown ranker"`), a distinct failure mode from a
      malformed document.
    * `params.llm_callsite` naming a registered LLM call site — checked
      against `crucible.llm.LLM_CALLSITE_REGISTRY`, imported lazily because
      it is a heavy module with one caller.
    * `registered_at` naming an actual NYSE trading day — asserted on
      `ArmSpec.__post_init__` via `crucible.calendar.assert_trading_day`,
      which needs the trading calendar rather than the document alone. It is
      a validity check against an external oracle, not a rule relating this
      document's fields to each other, so it is not the kind of cross-field
      rule binding constraint 3 asks to move onto the model.

    `params` is required and non-empty, matching the reader's existing
    falsy-field check (`not document.get("params")`) exactly: this PR types
    the boundary, it does not decide whether a ranker with zero parameters
    should be legal — that is a separate, un-filed question about the
    reader's own semantics.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"$id": "arm_recipe.v1"})

    #: Defaulted, not required (unlike `RunManifestV2.schema_version`): this is a
    #: NEW published contract for a document class that has never declared a
    #: version, so every already-filed recipe omits the key. Requiring it would
    #: fail every recipe filed before this PR on its next load.
    schema_version: Literal["arm_recipe.v1"] = Field(
        default="arm_recipe.v1",
        description="Version of THIS schema. Defaulted for recipes filed before this "
        "boundary existed; a future incompatible version makes this field required.",
    )
    name: str = Field(min_length=1, description="the arm's name, unique within its slot")
    slot: str = Field(min_length=1, description="the slot this recipe belongs to (u/r/m/s)")
    ranker: str = Field(
        min_length=1,
        description="the ranking callable's registered name; existence is checked at "
        "load time against the live ranker registry, not by this schema",
    )
    params: dict[str, Any] = Field(
        description="the ranker's own open argument mapping, hashed as-is into the "
        "arm's spec (`ArmSpec.spec`). Membership of this KEY is closed by the "
        "top-level vocabulary; its CONTENTS are an open mapping. Required and "
        "non-empty, matching the reader's existing behaviour."
    )
    registered_at: IsoDate = Field(
        description="the NYSE trading day the arm's out-of-sample clock starts on. "
        "Every ladder rung is counted in trading weeks from this date; validity as "
        "an actual session is asserted at construction (`crucible.calendar."
        "assert_trading_day`), not by this schema."
    )
    supersedes: str | None = Field(
        default=None, description="the arm id this recipe replaces, if any (§3.1 lineage)"
    )
    control: bool = Field(
        default=False,
        description="whether this is a control arm. Filed recipes normally leave this "
        "false; the harness sets it on the two control arms it generates itself "
        "(`crucible.slots.arms.control_specs`).",
    )
    control_kind: str | None = Field(default=None, description="the control's kind, if any")
    bootstrap: bool = Field(
        default=False, description="whether this arm bootstraps its own promotion, per policy"
    )
    promotion_source: str = Field(
        default="",
        description="where this recipe was promoted from, for provenance only — "
        "never hashed into the spec (`ArmSpec.spec` excludes it, per §3.1)",
    )
    notes: str = Field(
        default="",
        description="a free-text clarifying comment, provenance only, never hashed "
        "into the spec — an edited note must not orphan the arm's score series",
    )

    @model_validator(mode="before")
    @classmethod
    def _the_five_pre_registration_fields_are_present(cls, data: object) -> object:
        """§9.1: an arm declares slot, recipe and registration date before its
        first score. Restated as a `mode="before"` check (rather than relying
        on pydantic's own per-field "field required" errors) so a recipe
        missing several fields at once names ALL of them in one message, and
        so `params: {}` — present but empty, hence falsy — is refused with
        the same text as `params` being absent entirely, matching the
        reader's pre-existing behaviour exactly.
        """
        if not isinstance(data, dict):
            return data
        missing = [f for f in ARM_RECIPE_REQUIRED_FIELDS if not data.get(f)]
        if missing:
            raise ValueError(
                f"arm recipe is missing required field(s) {missing}. §9.1 "
                "pre-registration: an arm declares its slot, recipe and registration date "
                "before its first score, and a recipe that leaves one blank produces a "
                "verdict that cannot answer for itself. Note that `metric`, `horizon` and "
                "`benchmark` are deliberately NOT arm fields — they are the SLOT's, so "
                "every arm is scored on the same axis (policy §4)."
            )
        return data


# ── I10045 row 3: the arena cycle artifact ─────────────────────────────────
# Additive only, appended after row 2's marker for the same rebase reason.


class ArenaCycleDocument(BaseModel):
    """The `arena_cycle` artifact this package reads back, `champion-
    challenger-policy.md` §11.

    **Unlike every other model in this module, `extra` is NOT forbidden.**
    The published contract for this document is
    `nousergon_lib.contracts.arena_cycle.schema.json` — the LIBRARY's schema,
    validated by `crucible.arena_io.validate_arena_cycle` on every read and
    write, unchanged by this PR — not a schema this module owns or
    generates. A field the library adds tomorrow is a valid `arena_cycle`
    today's crucible does not yet know the name of; `extra="forbid"` here
    would make THIS type reject documents the real contract accepts, which
    is the CloudTrail-payload shape binding constraint 2 carves out (a shape
    a third party writes), not the `components.yaml` shape (a shape that is
    ours). `crucible.slots.arena_config_for` already returns the library's
    own `ArenaConfig` for exactly this reason (repo AGENTS.md, "the arena is
    called, never re-implemented") — this model does not re-implement the
    library's `ArenaCycle` either; it exists only so
    `crucible.arena_io.read_arena_cycle` hands its one caller named, typed
    top-level access instead of a raw dict indexed by hand, while the
    library schema stays what decides conformance. Nested substructures
    (`ladders`, `ranking`, `decision`, `retirements`) are left as `dict`/
    `list[dict]` rather than re-modelled field-by-field: the library schema
    already fully validates their shape via `validate_arena_cycle`, and a
    crucible-local re-typing of library-owned internals is precisely the
    "second copy that drifts" this migration exists to avoid.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(description="the library contract's own version, currently 1")
    slot: str = Field(min_length=1)
    slot_kind: str = Field(min_length=1)
    benchmark: str = Field(
        min_length=1,
        description="the population an arm is graded against, e.g. 'population'",
    )
    as_of: str = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        description="the trading day this cycle was scored as of",
    )
    scored_arms: list[str] = Field(default_factory=list)
    active_arms: list[str] = Field(default_factory=list)
    ladders: list[dict[str, Any]] = Field(
        default_factory=list,
        description="per-arm score ladders; shape owned and validated by the library schema",
    )
    ranking: dict[str, Any] | None = Field(
        default=None, description="the Condorcet ranking, if the cycle produced one"
    )
    decision: dict[str, Any] = Field(description="the pointer decision for this cycle")
    retirements: list[dict[str, Any]] = Field(default_factory=list)


# ── I10045 row 4: the LLM call-site registry ───────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class LlmCallSiteRow(_Strict):
    """One row of ``callsites:`` in `llm_callsites.yaml`, plan §4.8.

    `crucible.llm.load_registry` used to build this by hand: `missing = [f
    for f in (...) if f not in row]` named every absent field in one message,
    but a WRONGLY TYPED field (a string `max_usd_per_call`) fell through to
    `float(row["max_usd_per_call"])`, which either coerces silently or raises
    a bare `ValueError` naming neither the call site nor the field.
    """

    purpose: str = Field(min_length=1, description="what this call site asks a model for")
    capability_class: str = Field(
        min_length=1,
        description="the router GROUP or capability class asked for — never a provider "
        "model id, a base url or an SDK client (principle 8). Membership against the "
        "live router+allowlist union is checked at load time, not by this schema, because "
        "the router's own groups are not knowable from this document alone.",
    )
    max_usd_per_call: float = Field(gt=0, description="this site's own ceiling, under the cap")
    owner: str = Field(min_length=1, description="the module that holds the call")


class LlmCallsiteRegistryDocument(_Strict):
    """``llm_callsites.yaml`` — plan §4.8, `crucible.llm`'s module docstring.

    Two checks stay in the reader (`crucible.llm._require_capability_class`,
    `crucible.llm.capability_group`) rather than moving onto this model,
    because each is a membership check against `krepis.router`'s LIVE tier
    groups plus this same document's own `capability_classes` allowlist — a
    document-shape model cannot know the router's groups without importing
    it, and this module (like `ArmRecipeDocument`'s) stays free of a
    dependency on the thing it types.
    """

    schema_version: Literal["llm_callsite_registry.v1"] = Field(
        description="Version of THIS schema. The real file already declares it "
        "(unlike the arm recipe boundary, this is not a migration default)."
    )
    capability_classes: list[str] = Field(
        description="the allowlist `crucible.llm.call` admits against, beyond the router's "
        "bare tiers. An empty registry still declares this key, as `[]` — its absence is a "
        "broken build, not an empty allowlist."
    )
    callsites: dict[str, LlmCallSiteRow] = Field(
        description="every LLM call site this package can reach a model from, by id. "
        "Empty is the correct state before any LLM arm exists, and is written `{}`."
    )

    @model_validator(mode="after")
    def _capability_classes_are_non_empty_strings(self) -> LlmCallsiteRegistryDocument:
        bad = [c for c in self.capability_classes if not isinstance(c, str) or not c]
        if bad:
            raise ValueError(
                f"`capability_classes` must be a list of non-empty strings; found {bad!r}"
            )
        return self
