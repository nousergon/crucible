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
from typing import Annotated, Any, ClassVar, Literal, get_args

from krepis.metrics import StatusLiteral
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

__all__ = [
    "ARM_RECIPE_REQUIRED_FIELDS",
    "ArenaCycleDocument",
    "ArmRecipeDocument",
    "ArtifactRef",
    "AttemptRow",
    "BoardDeclarationRow",
    "ChampionAttestation",
    "ClosingReadingClauseRow",
    "ChampionEvidence",
    "ChampionPointerDocument",
    "CloudTrailRecord",
    "CloudTrailSessionContext",
    "CloudTrailSessionIssuer",
    "CloudTrailUserIdentity",
    "ClosedPathRow",
    "ComponentRow",
    "ComponentsDocument",
    "DeadlineRow",
    "DeclaredUniverseDocument",
    "EXPERIMENT_EVENT_ROW_ADAPTER",
    "FAULT_OUTCOME_VALUES",
    "EligibilityHoldEventRow",
    "ExperimentEventRow",
    "FaultRecordDocument",
    "FeatureRegistryDocument",
    "FeatureRow",
    "GitHubCommit",
    "GitHubCommitDetail",
    "GitHubCommitIdentity",
    "GitHubUser",
    "LlmCallRow",
    "LlmCallSiteRow",
    "LlmCallsiteRegistryDocument",
    "METRIC_STATUS_VALUES",
    "MetricRecordRow",
    "NegativeResultEventRow",
    "NoComparisonEventRow",
    "PhaseClosingReadingDocument",
    "PhaseLadderDocument",
    "PhaseLadderRow",
    "PromotionEventRow",
    "RegistryDefaults",
    "RejectedRow",
    "ReleaseProvenanceDocument",
    "ReleaseRecordDocument",
    "ResourceRow",
    "RetirementEventRow",
    "RetirementLogRow",
    "ReviewDocument",
    "RunManifestV2",
    "SignalsRow",
    "TrialRow",
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
    #: WHICH trading days this row is due on. Required and closed, for the
    #: same reason `anchor` is: without it the deadline answers "by when"
    #: while nothing answers "on what days", and `evaluate_absence` graded
    #: every scheduled row on every trading day. Eight rows read
    #: `schedule: weekly, Saturday` and were paged absent on the four
    #: weekdays they were never going to run (measured 2026-09-08: seven of
    #: the ABSENCE page's twenty-two members). `schedule` above is PROSE and
    #: is not read by anything; this is the machine field.
    cadence: Literal["daily", "weekly"]
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
    "fault.record",
    "fault.probe",
    "iac.conformance",
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

#: The exhaustive `fault_record.outcome` vocabulary — THREE kinds, because
#: plan §10.7's four scripted faults do not all end the same way and a
#: producer that accepts only one shape can record only the faults that
#: happen to take it (`alpha-engine-config-I10327`).
#:
#: * `induced` — the fault fired and the job FAILED. The shape
#:   `alpha-engine-config-I10320`/`-I10322` shipped: it names the `run_id` of
#:   a manifest reading `status: failed`, and it is the ONLY kind whose
#:   `run_id` excuses that manifest from `arc_runs_ok`/`replays_ok`.
#: * `absorbed` — the fault fired and the system HANDLED it: the manifest
#:   reads `ok` and its `attempts[]` records the declared-transient-class
#:   retry that made it so. A STRONGER result than `induced`, not a weaker
#:   one, and the outcome fault 1 (`spot_terminated_mid_job`) is designed to
#:   produce — a spot interruption is the first row of
#:   `crucible.runner.TRANSIENT_CLASSIFIERS`, so the runner retries it on a
#:   fresh instance and the run succeeds. `bus_key` is FORBIDDEN here: a page
#:   would mean the retry did not work.
#: * `unreachable` — the state cannot be entered at all, evidenced by a
#:   machine-executed probe per closed path and NO `run_id`, so it is
#:   structurally incapable of excusing any manifest.
#:
#: An outcome outside this set cannot be recorded, for the same reason
#: `crucible.runner`'s two statuses are a closed set: a fourth kind is a
#: design change that must be visible in a diff.
FAULT_OUTCOME_VALUES: tuple[str, ...] = ("induced", "absorbed", "unreachable")

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
#: `alpha-engine-config-I10454`: the all-zero placeholder `code_sha` a
#: dispatched box used to write, refused at the schema level as well as by
#: `crucible.runner.resolve_code_sha` — a producer that skips the runner
#: (or a future one that reintroduces the same default) is still stopped
#: here. pydantic-core's regex engine has no look-around
#: (`SchemaError: look-around ... is not supported`, measured against
#: pydantic-core 2.46.5), so this cannot be one `pattern`; it is enforced by
#: :meth:`RunManifestV2._code_sha_is_not_the_placeholder` AND mirrored into
#: the published schema's `not` by `_run_manifest_v2_json_schema_extra`,
#: same shape as the status/reason cross-field rule just above it.
_PLACEHOLDER_GIT_SHA = "0" * 40

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
        # See `RunManifestV2._code_sha_is_not_the_placeholder` (alpha-engine-config-I10454)
        # for why this is a mirrored `not` clause rather than folded into
        # `code_sha`'s own `pattern`.
        {
            "description": (
                "code_sha is never the all-zero placeholder: it validates the same pattern "
                "as a real commit sha and answers nothing. Mirrors "
                "`RunManifestV2._code_sha_is_not_the_placeholder`, which pydantic-core's lack "
                "of regex look-around keeps out of `code_sha`'s own `pattern`."
            ),
            "not": {"properties": {"code_sha": {"const": "0" * 40}}, "required": ["code_sha"]},
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
    #: `alpha-engine-config-I10125`: present only when the run's own
    #: evaluation logic reasoned from an OPERATOR-OVERRIDDEN instant rather
    #: than the real wall clock at `started`/`finished`. Absent for every
    #: natural run.
    now_override_utc: UtcTimestamp | None = Field(
        default=None,
        description=(
            "I10125: present only when the run's own evaluation logic "
            "reasoned from an OPERATOR-OVERRIDDEN instant rather than the real wall clock at "
            "`started`/`finished`. Today only `alerts.sweep --now` sets it, to sweep a historical "
            "trading day's catch-up and ceiling windows without landing inside "
            "`crucible.gate._clause_pages_within_ceiling`'s live grading window, which is "
            "anchored to the real wall clock and unaffected by this field. Absent for every "
            "natural run — never encodes a fact `started`/`finished` already carry, and its "
            "presence is what makes a hand-chosen historical sweep structurally distinguishable "
            "from one that actually happened live."
        ),
    )
    #: `alpha-engine-config-I10343`: present only when this run's LLM routing
    #: was deliberately redirected to a fault-injection capability class.
    #: Absent for every natural run.
    fault_capability_class: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
        description=(
            "I10343: present only when an operator deliberately redirected this run's LLM "
            "routing to a capability class contracted NEVER to serve, to induce plan §10.7 "
            "fault 3 (the router returns an error) against the real dispatched path. Set from "
            "`crucible fault.probe --fault-capability-class`, never by a job body, and only "
            "ever one of `crucible.llm.FAULT_INJECTION_CAPABILITY_CLASSES`. Absent for every "
            "natural run — its presence is what makes an ARRANGED transport failure "
            "structurally distinguishable from a real one, which a fault record naming this "
            "manifest as `induced` evidence depends on."
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
    #: The all-zero placeholder is refused, not just discouraged
    #: (alpha-engine-config-I10454): a producer that cannot measure this for
    #: real must not write it, and `crucible.runner.resolve_code_sha` raises
    #: before any manifest write is attempted rather than defaulting to a
    #: value that validated and answered nothing.
    code_sha: GitSha = Field(
        description=(
            "Commit sha of the crucible tree that ran. Half of `explain`'s answer to 'why did it "
            "do that'. The all-zero placeholder is refused: a producer that cannot measure this "
            "for real must not write it."
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

    @model_validator(mode="after")
    def _code_sha_is_not_the_placeholder(self) -> RunManifestV2:
        """`alpha-engine-config-I10454`: the all-zero sha validated against
        `code_sha`'s `pattern` (forty lowercase hex characters, same as any
        real commit) and answered nothing — every v2 manifest a dispatched
        box wrote carried it, silently, because nothing refused it. Kept as
        a model_validator rather than folded into `code_sha`'s `pattern`
        because pydantic-core's regex engine has no look-around support
        (measured against pydantic-core 2.46.5); mirrored into the
        published schema's `not` by `_run_manifest_v2_json_schema_extra`
        for the same reason `_status_and_reason_agree` is mirrored into its
        `allOf` — a consumer with no Python import gets the same refusal.
        """
        if self.code_sha == _PLACEHOLDER_GIT_SHA:
            raise ValueError(
                f"code_sha is the all-zero placeholder ({_PLACEHOLDER_GIT_SHA!r}). It "
                "validates the same pattern as a real commit sha and answers nothing — "
                "half of `explain`'s answer to 'why did it do that' would be silently "
                "absent. A producer that cannot measure code_sha for real must refuse to "
                "write a manifest at all (see `crucible.runner.resolve_code_sha`), never "
                "substitute this value."
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


# ── I10045 row 5: the release pointer + provenance ─────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class ReleaseRecordDocument(_Strict):
    """`release.json`, at `releases/{sha}/release.json` — plan §4.11.

    `crucible.release.ReleaseRecord` (a frozen dataclass; unchanged by this
    PR — its `.wheel_key` property and `.to_json()` method are domain
    behaviour this module does not carry, the same reason
    `crucible.slots.arms.ArmSpec` stayed a dataclass beside row 2's
    `ArmRecipeDocument`) validates its own constructed payload against THIS
    model in `__post_init__`, via `crucible.release._validate_release_artifact`
    — replacing that function's previous hand-rolled `jsonschema`
    `Draft202012Validator` machinery with this model, while keeping its
    public signature, its "does not conform" message text, and both of its
    existing direct tests (`tests/test_release.py`) unchanged.

    Deterministic across every rebuild of the same commit
    (`alpha-engine-config-I9786`) — nothing here can differ between two
    builds of the same sha, which is what makes
    `crucible.release.assert_immutable_write`'s byte comparison a
    correctness check rather than a false-alarm generator. `extra="forbid"`:
    a field a reader does not understand is a field the producer expected it
    to act on.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/nousergon/crucible/schemas/release.v3.json",
            "title": "Crucible release identity, v3",
            "description": (
                "What release.json carries at releases/{sha}/release.json. "
                "Deterministic across every rebuild of the same commit "
                "(I9786) -- nothing in this document can differ "
                "between two builds of the same sha, which is what makes "
                "assert_immutable_write's byte comparison a correctness check rather "
                "than a false-alarm generator. v3 replaces v2 (I9908): "
                "adds wheel_filename, the PEP-440-legal name the wheel was actually "
                "published under. Every release published before I9908 was named "
                "crucible-{sha}-py3-none-any.whl, which pip refuses outright (a 40-hex "
                "git sha is not a PEP 440 version) -- no wheel this pipeline ever "
                "published was installable. v3's wheel_filename lets a bash bootstrap on "
                "a spot box download the exact object by name without reimplementing "
                "crucible.release.wheel_filename_for's version derivation. "
                "additionalProperties: false because a field a reader does not "
                "understand is a field the producer expected it to act on."
            ),
        },
    )

    schema_version: Literal["release.v3"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    sha: GitSha = Field(
        description="The commit this release is built from. Content-addresses the release prefix."
    )
    lockfile_sha256: Sha256 = Field(
        description="The resolved dependency tree's hash. The wheel does not pin its "
        "own transitive tree, so two wheels from one commit against two lockfiles are "
        "two different artifacts, and only this says which."
    )
    wheel_sha256: Sha256 = Field(
        description="The wheel's own hash, checked against the bytes actually uploaded "
        "by both the publisher and the smoke."
    )
    wheel_filename: str = Field(
        pattern=r"^crucible-.+-py3-none-any\.whl$",
        description="The exact object name the wheel is published under, at "
        "releases/{sha}/{wheel_filename}. PEP-440-legal "
        "(crucible-{version}-py3-none-any.whl), so pip can install it by name -- "
        "unlike every release published before I9908, which pip "
        "refused under both the download name and the store's own filename.",
    )
    python_requires: str = Field(
        min_length=1, description="The interpreter constraint this wheel was built against."
    )
    extra: dict[str, Any] = Field(
        description="Reserved for future identity-bearing fields. Empty on every "
        "release today -- anything that would vary per run belongs in "
        "release_provenance.v1, not here."
    )


class ReleaseProvenanceDocument(_Strict):
    """`releases/{sha}/provenance/{run_id}-{run_attempt}.json` — plan §4.11.

    `crucible.release.ReleaseProvenance` validates against this model the
    same way `ReleaseRecord` validates against `ReleaseRecordDocument` —
    see that model's docstring. Immutable-checked (alpha-engine-config-I9817)
    only WITHIN one attempt's own key: a second, distinct run_id/run_attempt
    for an already-published sha is EXPECTED to differ here and is written as
    its own durable record; two writes naming the SAME run_id/run_attempt
    with differing bytes are refused.
    """

    # Immutability within one attempt's own key is enforced by
    # `crucible.release.assert_immutable_write` at the write site, and
    # `built_at`'s type below by `UtcTimestamp` -- see the class docstring
    # above for the citation (issue number kept out of these description
    # strings on purpose: they reach `model_json_schema()`, which
    # `tests/test_no_stale_tracker_literals.py` scans as a potential
    # message-fragment surface, not prose).
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/nousergon/crucible/schemas/release_provenance.v1.json",
            "title": "Crucible release provenance, v1",
            "description": (
                "One publish ATTEMPT for a release, at "
                "releases/{sha}/provenance/{run_id}-{run_attempt}.json. Split out of "
                "release.json by I9786: these three fields move on "
                "every rebuild of the same commit, so keeping them in the immutable "
                "identity record made a re-run of an unchanged commit an unconditional "
                "ReleaseImmutabilityError. Immutable-checked only WITHIN one attempt's "
                "own key: a second, distinct run_id/run_attempt for an "
                "already-published sha is EXPECTED to differ here and is written as "
                "its own durable record; two writes naming the SAME run_id/run_attempt "
                "with differing bytes are refused."
            ),
        },
    )

    schema_version: Literal["release_provenance.v1"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    sha: GitSha = Field(
        description="The commit this attempt built. Must match the release.json this "
        "attempt accompanies."
    )
    run_id: str = Field(
        min_length=1,
        description="The CI run that produced this attempt (GitHub's GITHUB_RUN_ID). "
        "Part of this document's own key.",
    )
    run_attempt: str = Field(
        min_length=1,
        description="Distinguishes a 're-run failed jobs' retry that reuses the same "
        "run_id. Part of this document's own key.",
    )
    # built_at Gotcha (I9786): sourced from github.event.repository.updated_at
    # in deploy.yml, which is repository metadata, not the build instant --
    # deploy.yml should source this from github.run_started_at instead
    # (residual tracked alpha-engine-config-I9817, out of scope for the
    # workflow file in this change). Typed as UtcTimestamp here so a producer
    # that DOES source it correctly cannot regress the shape while that
    # value-correctness fix lands separately.
    built_at: UtcTimestamp = Field(
        description="When this attempt ran. Gotcha (I9786): sourced from "
        "github.event.repository.updated_at in deploy.yml, which is repository "
        "metadata, not the build instant -- deploy.yml should source this from "
        "github.run_started_at instead."
    )
    workflow_run_url: str = Field(
        description="The specific workflow run this attempt is. Different on every "
        "attempt by construction."
    )
    test_summary: str = Field(
        description="The foundation-test line this attempt printed. Free text; not "
        "parsed by anything downstream."
    )


# ── I10045 row 6: the champion pointer ──────────────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class ChampionEvidence(BaseModel):
    """`evidence` on the champion pointer — why this arm, plan §3/§9.1.

    Deliberately NOT `_Strict`: the published schema's `evidence` object
    carries no `additionalProperties: false`, because "a decision's evidence
    is slot-specific" (schema description) — an operator-revert evidence
    blob and an evidence-promotion blob share none of these named fields.
    `extra="allow"` matches that open shape exactly; forbidding extras here
    would refuse every operator-revert pointer ever written.
    """

    model_config = ConfigDict(extra="allow")

    incumbent: str | None = None
    status: str | None = None
    reason: str | None = None
    moved: bool | None = None
    paired_dates: Annotated[int, Field(ge=0)] | None = None
    window_start: str | None = None
    window_end: str | None = None
    mean_diff: float | None = None
    confidence_sequence: dict[str, Any] | None = None
    promote_min_weeks: Annotated[int, Field(ge=1)] | None = None
    #: The slot's serving evidence bar, mirroring
    #: `nousergon_lib.arena.engine.ArenaConfig.promote_evidence`
    #: (`alpha-engine-config-I10547`). Optional because every pointer written
    #: before 2026-09-12 predates the field, and an operator-revert pointer
    #: carries no promotion evidence at all — never because a promotion may
    #: omit it.
    promote_evidence: Literal["anytime_valid", "point"] | None = None
    paired_dates_required: Annotated[int, Field(ge=1)] | None = None
    operator: str | None = None
    eligible_arms: list[str] | None = None


class ChampionAttestation(_Strict):
    """The contamination attestation, plan §9.1. Mandatory in EFFECT for the
    S slot (the reader refuses an S champion whose status is not PASS) and
    null elsewhere — not `required` at the top level because a null
    attestation on U/R/M is correct, and the refusal belongs to the reader,
    which can see which slot it is reading.
    """

    kind: Literal["pit_parity"]
    status: Literal["PASS", "FAIL", "PARTIAL", "UNKNOWN"]
    key: str | None = None
    reason: str | None = None


def _champion_pointer_json_schema_extra(schema: dict[str, object]) -> None:
    """Sets `champion_pointer.v1.json`'s document-level metadata and mirrors
    `ChampionPointerDocument._code_sha_is_not_the_placeholder`
    (`alpha-engine-config-I10506`) into the schema's `not`, same shape and
    same reason `_run_manifest_v2_json_schema_extra` mirrors
    `RunManifestV2._code_sha_is_not_the_placeholder` — pydantic-core's regex
    engine has no look-around, so the refusal cannot be folded into
    `code_sha`'s own `pattern`, and a consumer with no Python import needs
    the same refusal restated here rather than relying on the
    model_validator alone.
    """
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/nousergon/crucible/schemas/champion_pointer.v1.json"
    schema["title"] = "Crucible champion pointer, v1"
    schema["description"] = (
        "The single artifact coupling the harness to the trader (plan §3). "
        "Written at champions/{slot}/current.json by `crucible promote`, read "
        "by the trader before it sizes anything. Versioned because a second "
        "implementation of the trader must be able to consume it from this "
        "document alone; additionalProperties: false because a field this "
        "reader does not understand is a field the producer expected it to "
        "act on."
    )
    schema["not"] = {
        "description": (
            "code_sha is never the all-zero placeholder: it validates the same "
            "pattern as a real commit sha and answers nothing. Mirrors "
            "`ChampionPointerDocument._code_sha_is_not_the_placeholder`."
        ),
        "properties": {"code_sha": {"const": "0" * 40}},
        "required": ["code_sha"],
    }


class ChampionPointerDocument(_Strict):
    """`champions/{slot}/current.json` — the one contract the trader reads,
    plan §3/§4.4/§9.1; `champion-challenger-policy.md` §11.

    `crucible.champion.ChampionPointer` (a frozen dataclass; unchanged by
    this PR — its `to_dict()` is domain behaviour this module does not
    carry, the same reason `crucible.release.ReleaseRecord` stayed a
    dataclass beside row 5's `ReleaseRecordDocument`) validates a payload
    against this model in `from_dict`, replacing that method's previous
    hand-rolled `jsonschema.Draft202012Validator` call, while keeping the
    same `ChampionUnusableError` type and message text.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=_champion_pointer_json_schema_extra,
    )

    schema_version: Literal["champion_pointer.v1"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    slot: Literal["u", "r", "m", "s"] = Field(
        description="Which decision this pointer governs. Closed set: the four slots "
        "of champion-challenger-policy.md §2."
    )
    arm_id: str = Field(
        min_length=1,
        description="The serving arm. Its id encodes its own spec hash (policy §3.1), "
        "so an edited recipe cannot reuse it and the trader can always tell which "
        "recipe it is running.",
    )
    as_of: str = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        description="The NYSE trading day whose cycle produced this decision (§4.12). "
        "Never a wall-clock date.",
    )
    decided_at: str = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
        description="UTC instant the pointer was written, for provenance only.",
    )
    run_id: str = Field(
        pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$",
        description="The run that wrote this pointer. Correlates the decision with "
        "its manifest, its logs and its cost row (§9.2).",
    )
    # See `_code_sha_is_not_the_placeholder` below (alpha-engine-config-I10506,
    # same reasoning as `RunManifestV2.code_sha`/-I10454) for why the all-zero
    # placeholder is refused rather than accepted here — kept out of the
    # `description` string itself (`tests/test_no_stale_tracker_literals.py`).
    code_sha: GitSha = Field(
        description="The commit that decided. A producer that cannot measure it "
        "for real refuses to write a pointer at all, rather than substituting "
        "the all-zero placeholder."
    )
    promotion_source: Literal["evidence", "operator_bootstrap", "bootstrap"] = Field(
        description="How this pointer came to be. `evidence` = the anytime-valid "
        "sequence supported the lead; `operator_bootstrap` = a human placed it; "
        "`bootstrap` = the engine's §9.1 cold start. Carried so that a pointer which "
        "has never moved on evidence renders as the finding it is (policy §11) "
        "rather than as a stable system."
    )
    manifest_key: str = Field(
        min_length=1,
        description="The run manifest of the job that wrote this pointer. The reader "
        "re-derives that run's status from it and refuses a champion produced by a "
        "run that did not finish `ok`.",
    )
    evidence: ChampionEvidence = Field(
        description="Why this arm. For an evidence promotion: the incumbent it "
        "passed, the paired window it passed on, and the confidence-sequence bound. "
        "For an operator revert: who reverted and why. Free-form beyond the named "
        "fields because a decision's evidence is slot-specific, but never empty -- "
        "an unexplained pointer cannot be reviewed (principles.md §2.1)."
    )
    attestation: ChampionAttestation | None = Field(
        default=None,
        description="The contamination attestation (plan §9.1). Mandatory in EFFECT "
        "for the S slot -- the reader refuses an S champion whose status is not PASS "
        "-- and null elsewhere. Not `required` in the schema because a null "
        "attestation on U/R/M is correct; the refusal is on the reader, where it can "
        "see which slot it is reading.",
    )

    @model_validator(mode="after")
    def _code_sha_is_not_the_placeholder(self) -> ChampionPointerDocument:
        """`alpha-engine-config-I10506`: same defect, second producer.
        `crucible.promote` defaulted an unresolved `code_sha` to the all-zero
        placeholder on every real invocation (`crucible.cli._promote` never
        passed one), so every champion pointer this harness ever wrote
        carried it -- half of `explain`'s answer to "why did this promote"
        silently absent. Mirrored into the published schema's `not` by
        `_champion_pointer_json_schema_extra`, same shape and same reason as
        `RunManifestV2._code_sha_is_not_the_placeholder`.
        """
        if self.code_sha == _PLACEHOLDER_GIT_SHA:
            raise ValueError(
                f"code_sha is the all-zero placeholder ({_PLACEHOLDER_GIT_SHA!r}). It "
                "validates the same pattern as a real commit sha and answers nothing — "
                "half of 'why did this promote' would be silently absent. A producer "
                "that cannot measure code_sha for real must refuse to write a pointer "
                "at all, never substitute this value."
            )
        return self


# ── I10045 row 7: the feature registry ─────────────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.

_FEATURE_NAME_PATTERN = r"^[a-z0-9]+(_[a-z0-9]+)*(_raw|_ratio|_pct|_zscore|_log_return)$"

#: Suffix -> the ONE unit that suffix may declare. Mirrors
#: `crucible.features.registry._NORMALIZED_UNIT_BY_SUFFIX` exactly; restated
#: rather than imported so this module carries no import-time dependency on
#: the reader it types (the `ArmRecipeDocument`/row 2 precedent).
_NORMALIZED_UNIT_BY_SUFFIX: dict[str, str] = {
    "_ratio": "ratio",
    "_pct": "pct",
    "_zscore": "zscore",
    "_log_return": "log_return",
}


def _feature_row_json_schema_extra(schema: dict[str, object]) -> None:
    """Mirrors `FeatureRow._suffix_and_unit_agree` into the published
    schema's `allOf`, one `if`/`then` per normalized suffix plus the `_raw`
    exclusion — byte-for-byte what `feature_registry.v1.json` carried before
    this PR, so a consumer with no Python import still gets the rule
    (`RunManifestV2`/row 1 precedent: `_run_manifest_v2_json_schema_extra`).
    """
    schema["allOf"] = [
        {
            "if": {"properties": {"name": {"pattern": "_ratio$"}}, "required": ["name"]},
            "then": {"properties": {"unit": {"const": "ratio"}}},
        },
        {
            "if": {"properties": {"name": {"pattern": "_pct$"}}, "required": ["name"]},
            "then": {"properties": {"unit": {"const": "pct"}}},
        },
        {
            "if": {"properties": {"name": {"pattern": "_zscore$"}}, "required": ["name"]},
            "then": {"properties": {"unit": {"const": "zscore"}}},
        },
        {
            "if": {"properties": {"name": {"pattern": "_log_return$"}}, "required": ["name"]},
            "then": {"properties": {"unit": {"const": "log_return"}}},
        },
        {
            "if": {"properties": {"name": {"pattern": "_raw$"}}, "required": ["name"]},
            "then": {
                "properties": {
                    "unit": {
                        "description": (
                            "A `_raw` column may not claim a NORMALIZED unit: the name "
                            "promises the consumer an unnormalized value and the declared "
                            "unit would say otherwise. That is the avg_volume_20d defect "
                            "in the other direction."
                        ),
                        "not": {"enum": ["ratio", "pct", "zscore", "log_return"]},
                    }
                }
            },
        },
    ]


class FeatureRow(_Strict):
    """One column of `features/{version}/registry.json`'s `features` array —
    plan §10 component 4.

    `crucible.features.registry.FeatureSpec` (a frozen dataclass; unchanged
    by this PR — its `to_dict()` is domain behaviour this module does not
    carry) keeps its own `__post_init__` cross-field checks, exercised on
    every `CATALOG` entry at IMPORT time. This model validates the same
    rules a second time, at the DOCUMENT boundary — `crucible.features.
    registry.validate_registry_payload` — the way `RunManifestV2._status_and_
    reason_agree` and its schema `allOf` twin both enforce one rule rather
    than the model deferring to `FeatureSpec`'s check.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_feature_row_json_schema_extra)

    name: str = Field(
        pattern=_FEATURE_NAME_PATTERN,
        description="The column name. The units suffix is MANDATORY and is enforced by "
        "the pattern: avg_volume_20d was emitted as a normalized ratio and consumed as "
        "raw shares, and 901 of 903 tickers silently failed the scanner liquidity gate "
        "for months. There is no grandfather list -- this layer has no history to "
        "grandfather.",
    )
    unit: Literal["USD", "beta", "indicator", "log_return", "pct", "ratio", "zscore"] = Field(
        description="The concrete unit, from a CLOSED vocabulary. Pinned by the suffix "
        "for every NORMALIZED suffix (see the allOf below: ratio, pct, zscore, "
        "log_return); the _raw set is open in MEANING but still enumerated rather than "
        "a free string, so a variant spelling of a normalized word (Ratio, RATIO, "
        "'ratio ') cannot pass the allOf not/enum check below by evading exact-string "
        'matching (I9815 -- avg_volume_20d_raw declaring unit: "Ratio" validated '
        "before this enum existed, the exact defect the suffix-unit agreement check "
        "exists to catch). Extend deliberately, by adding both the enum member here "
        "and a FeatureSpec in registry.py::CATALOG that uses it -- never widen this to "
        "a free string again.",
    )
    expression: str = Field(
        min_length=1,
        description="How the column is computed, as the registry states it. Lineage "
        "is a field, not a comment.",
    )
    description: str = Field(
        min_length=1,
        description="What the column means and why it is shaped that way. Empty is "
        "refused: a column nobody can read the intent of rots without anyone noticing "
        "it stopped being computed correctly.",
    )
    inputs: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        description="The panel or feature columns this feature reads. At least one, "
        "always: a column with no lineage cannot be traced back to the data that "
        "produced it (principle 1), which is what explain answers 'the signal "
        "degraded or the feature changed' from.",
    )
    window_trading_days: Annotated[int, Field(ge=1)] | None = Field(
        description="A count of SESSIONS (§4.12), never calendar days. Null for a "
        "point-in-time column reading only the current row. Zero and negative are "
        "refused -- a window is at least one session."
    )
    cross_sectional: bool = Field(
        description="True when the column is computed across one day's cross-section "
        "(a z-score or a rank) rather than along one ticker's history."
    )
    #: ADDITIVE OPTIONAL (I10114): not required, because a document written
    #: before this field existed still validates against `feature_registry.v1`
    #: unchanged. `bool | None = None` is the row-1 (`RunManifestV2`)
    #: accepted-difference shape: pydantic necessarily renders an optional
    #: field as `anyOf: [boolean, null]` rather than "boolean, or absent" --
    #: the two are semantically distinct in JSON Schema and this migration's
    #: own design note (row 1) already names forcing a non-nullable optional
    #: as requiring a hand-edit of the generated output, which it exists to
    #: prohibit.
    market_wide: bool | None = Field(
        default=None,
        description="True when the column's VALUE is one value repeated identically "
        "across every ticker on a day, by construction (e.g. "
        "market_return_1d_log_return); false when it varies across the "
        "cross-section. Declared, never inferred from variance at runtime. ADDITIVE "
        "OPTIONAL (I10114): not in required because a document written before this "
        "field existed (schema_version: feature_registry.v1, same as today) lacks it "
        "and must keep validating against this schema -- the same additive-optional "
        "shape as window_trading_days joining an earlier revision. The current "
        "producer (crucible.features.registry.FeatureSpec.to_dict) always emits it "
        "for every catalogue column; a reader of an OLDER document falls back to "
        "treating an absent market_wide as unknown rather than assuming a value.",
    )

    @model_validator(mode="after")
    def _suffix_and_unit_agree(self) -> FeatureRow:
        """The `avg_volume_20d` defect, both directions — mirrors
        `FeatureSpec.__post_init__` exactly, and is also emitted into the
        published schema's `allOf` by `_feature_row_json_schema_extra`."""
        matched = [
            suffix
            for suffix in ("_raw", "_ratio", "_pct", "_zscore", "_log_return")
            if self.name.endswith(suffix)
        ]
        suffix = max(matched, key=len)
        if suffix in _NORMALIZED_UNIT_BY_SUFFIX:
            expected_unit = _NORMALIZED_UNIT_BY_SUFFIX[suffix]
            if self.unit != expected_unit:
                raise ValueError(
                    f"feature {self.name!r} carries suffix {suffix!r} but declares "
                    f"unit={self.unit!r}; suffix {suffix!r} means unit={expected_unit!r} "
                    "and nothing else. A suffix that disagrees with the declared unit is "
                    "the `avg_volume_20d` defect: emitted as a ratio, consumed as raw "
                    "shares, and 901 of 903 tickers silently failed the liquidity gate "
                    "for months."
                )
        elif self.unit in _NORMALIZED_UNIT_BY_SUFFIX.values():
            raise ValueError(
                f"feature {self.name!r} carries suffix '_raw' but declares "
                f"unit={self.unit!r}, which is a NORMALIZED unit. '_raw' means "
                "unnormalized — a raw column claiming a normalized unit is the same "
                "defect in the other direction: the name promises the consumer an "
                "unnormalized value and the declared unit says otherwise."
            )
        return self


class FeatureRegistryDocument(_Strict):
    """`features/{version}/registry.json` — the feature layer's producer/
    consumer contract, plan §10 component 4.

    `crucible.features.registry.validate_registry_payload` routes through
    this model instead of a hand-rolled `jsonschema.Draft202012Validator`;
    `crucible.features.registry.load_registry_schema` is unchanged (it only
    reads whichever file is committed, and this PR's only change to that
    file is regenerating it from this model).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/nousergon/crucible/schemas/feature_registry.v1.json",
            "title": "Crucible feature registry, v1",
            "description": (
                "The document written to features/{version}/registry.json beside "
                "every day's feature parquet (plan §10 component 4). It is the "
                "PRODUCER/CONSUMER CONTRACT of the feature layer: the producer "
                "(crucible.features) may not emit a payload this schema refuses, and "
                "the consumer (crucible.slots.model.FeatureLayerSource) resolves a "
                "recipe's declared columns against exactly the names listed here. "
                "Interfaces reconciled by I9772 / -I9765; the schema exists so the "
                "next disagreement between the two tracks is a validation failure "
                "rather than two correct readings of an undeclared interface."
            ),
        },
    )

    schema_version: Literal["feature_registry.v1"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    feature_version: str = Field(
        pattern=r"^v[0-9a-f]{12}$",
        description="The layer version, DERIVED by hashing this catalogue "
        "(crucible.features.feature_version) and never hand-written. A hand-written "
        "string lets an edited recipe overwrite the layer an earlier verdict was "
        "computed from, and nothing would show it; the pattern makes a hand-written "
        "'v1' a validation failure rather than a convention someone remembers "
        "(I9772, disagreement 2).",
    )
    features: list[FeatureRow] = Field(
        min_length=1,
        description="Every column this version produces, in catalogue order. A "
        "consumer naming a column absent from this list must fail at load, never "
        "train on a silently substituted zero (the 2026-08-28 "
        "seven-hard-zeroed-features condition).",
    )


# ── I10045 row 8: the phase ladder + closing reading ───────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.

_TRACKER_ISSUE_PATTERN = r"^alpha-engine-config-I[0-9]+$"
_PHASE_ID_PATTERN = r"^phase[0-5]$"


class PhaseLadderRow(_Strict):
    """One phase row of `gates/ladder.json`'s `phases` array, plan §6."""

    decision_id: str = Field(
        pattern=_TRACKER_ISSUE_PATTERN,
        description="`console-policy` §2.1: the identifier IS the tracker ref, so a "
        "`git-host` claim about the same issue merges onto this row.",
    )
    phase: str = Field(pattern=_PHASE_ID_PATTERN)
    number: Annotated[int, Field(ge=0, le=5)]
    title: str = Field(min_length=1)
    tracker: str = Field(pattern=_TRACKER_ISSUE_PATTERN)
    tracker_url: str
    gate: str | None = Field(
        description="The registered gate name, or null when no gate has been written "
        "for this phase yet."
    )
    state: Literal["MET", "UNMET", "UNMEASURED", "UNMEASURABLE", "OUT_OF_ORDER"] = Field(
        description="crucible.gate.LADDER_STATES: MET | UNMET | UNMEASURED | "
        "UNMEASURABLE | OUT_OF_ORDER. Closed set."
    )
    console_state: Literal["HEALTHY", "DEGRADED", "UNREPORTED", "FAILED"] = Field(
        description="crucible.gate.LADDER_CONSOLE_STATE's rendering, in "
        "observability-policy §8.3's vocabulary."
    )
    gate_state: Literal["MET", "UNMET", "UNMEASURED", "UNMEASURABLE"]
    detail: str
    clauses_met: Annotated[int, Field(ge=0)] | None = Field(
        description="null when the phase has never been graded at all (no registered gate)."
    )
    clauses_total: Annotated[int, Field(ge=0)] | None
    clauses_unmeasurable: Annotated[int, Field(ge=0)] | None = Field(
        description="Count of this phase's clauses that read UNMEASURABLE (a store "
        "access failure, distinct from a clause that was read and found unmet). null "
        "exactly where clauses_total is null (no registered gate)."
    )
    met_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(
        description="null when nothing was measured (no clauses). Never 0.0 for an "
        "unmeasured reading -- 0.0 is a real measurement of zero clauses met out of a "
        "nonzero total (principle 7, I9824)."
    )
    read_on: str | None = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        description="The last trading day this gate was read, or null when never.",
    )
    blocked_by: str | None = Field(
        description="The earlier unmet phase id blocking this one, when state is "
        "OUT_OF_ORDER; null otherwise.",
    )
    generated_utc: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class PhaseLadderDocument(_Strict):
    """`gates/ladder.json` — plan §6's phase ladder.

    `crucible.gate.Ladder`/`PhaseRow` (frozen dataclasses with `.to_dict()`/
    `.render()`; unchanged by this PR) validate through this model in
    `crucible.gate.validate_ladder_document`, replacing that function's
    previous hand-rolled `jsonschema.Draft202012Validator` call.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/nousergon/crucible/schemas/phase_ladder.v1.json",
            "title": "Crucible plan §6 phase ladder, v1",
            "description": (
                "The record at gates/ladder.json, produced by "
                "crucible.gate.build_ladder/ladder_payload and published by exactly ONE "
                "writer: crucible gate --publish / crucible gate.close "
                "(crucible/track_f.py::gate_handler). crucible console was a second "
                "publisher until I10575 and re-evaluated every gate under its own "
                "environment, republishing a false phase0 UNMEASURABLE ladder over the "
                "gate publisher's phase2 reading; it now READS this key and embeds it. "
                "Versioned because the fleet console reads this key through an "
                "s3-records adapter, a cross-repo consumer; additionalProperties: false "
                "because a field this reader does not understand is a field the producer "
                "expected it to act on (I9825)."
            ),
        },
    )

    schema_version: Literal["phase_ladder.v1"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    trading_day: str = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        description="The NYSE trading day the ladder was built for (§4.12). Never a "
        "wall-clock date.",
    )
    generated_utc: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
    current_phase: str = Field(
        description="The lowest phase whose gate is not met, or `complete`. Not the "
        "highest phase with work in it."
    )
    phases_total: Annotated[int, Field(ge=0)]
    phases_met: Annotated[int, Field(ge=0)]
    unmeasured: Annotated[int, Field(ge=0)] = Field(
        description="Count of rows whose gate_state is UNMEASURED. Published rather "
        "than left for a reader to derive."
    )
    out_of_order: list[str] = Field(
        description="Phase ids currently graded ahead of an earlier unmet phase."
    )
    phases: list[PhaseLadderRow]


class ClosingReadingClauseRow(_Strict):
    """One clause verdict inside a `phase_closing_reading.v1` block."""

    name: str = Field(min_length=1)
    met: bool
    unmeasurable: bool
    detail: str


class PhaseClosingReadingDocument(_Strict):
    """The `phase_closing_reading.v1` block pasted into a phase-closing
    comment, plan §6 rule 2.

    `crucible.gate.closing_reading` builds the dict this validates via
    `crucible.gate.validate_closing_reading_document`, replacing that
    function's previous hand-rolled `jsonschema.Draft202012Validator` call.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://nousergon.ai/crucible/schemas/phase_closing_reading.v1.json",
            "title": "phase_closing_reading.v1",
            "description": (
                "The gate reading that justifies CLOSING a phase issue, pasted into "
                "the closing comment on the tracker (plan §6 rule 2; I9967 deliverable "
                "3). It is a TRANSCRIPT of one crucible gate --gate phaseN reading, not "
                "a claim: it carries the store it was read from, the commit of the "
                "code that read it, the trading day it was keyed to, the durable gate "
                "artifact the same run wrote, and every clause with its own verdict. A "
                "reader who doubts it can fetch gate_artifact from store and compare. "
                "Its consumer is alpha-engine-config's phase-tracker consistency "
                "sweep, which refuses a CLOSED phase issue that carries no such block "
                "or one that does not read MET -- so the fields below are a contract, "
                "not a rendering convenience, and additionalProperties is false in "
                "both directions."
            ),
        },
    )

    schema_version: Literal["phase_closing_reading.v1"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    phase: str = Field(
        pattern=r"^phase[0-9]+$",
        description="The plan §6 rung this reading closes, as crucible.gate.Phase.id.",
    )
    tracker: str = Field(
        pattern=_TRACKER_ISSUE_PATTERN,
        description="The phase issue this block belongs on, derived from "
        "crucible.gate.PHASES -- never typed. A block pasted onto a different issue "
        "is detectable because this field names the one it was rendered for.",
    )
    tracker_url: str = Field(min_length=1)
    gate: str = Field(min_length=1, description="The registered gate name that was read.")
    gate_state: Literal["MET", "UNMET", "UNMEASURABLE"] = Field(
        description="MET, UNMET or UNMEASURABLE -- crucible.gate.gate_state_for, the "
        "same function the ladder row uses, so a block and the ladder beside it cannot "
        "disagree. OUT_OF_ORDER is deliberately absent: it is a fact about the LADDER "
        "(a later phase graded ahead of an earlier one), not about this gate's own "
        "clauses, and the sweep that reads this block checks phase ordering from the "
        "tracker states it already holds.",
    )
    clauses_met: Annotated[int, Field(ge=0)]
    clauses_total: Annotated[int, Field(ge=0)]
    clauses_unmeasurable: Annotated[int, Field(ge=0)]
    met_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(
        description="Met clauses over total, or null when nothing was measured or any "
        "clause was unmeasurable -- GateResult.met_ratio, not re-derived. null, never "
        "0.0: zero is a measurement and absence is not."
    )
    coverage: str | None = Field(
        description="How much of the phase issue's declared deliverable list this "
        "gate grades, or null when the gate declares no deliverable list. Carried "
        "onto the block on purpose: a phase whose gate grades a SUBSET of its "
        "deliverables reads MET, and the closing comment is the last surface where "
        "that can still be seen."
    )
    trading_day: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    generated_utc: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
    store: str = Field(
        min_length=1,
        description="The store URI this reading was taken against. Provenance, and "
        "the half that makes the block falsifiable: without it gate_artifact names a "
        "key in no particular bucket.",
    )
    commit: str = Field(
        pattern=r"^[0-9a-f]{12,40}$",
        description="The commit of the crucible tree whose clause list produced this "
        "reading. At least 12 lowercase hex characters. A reading with no commit "
        "cannot be re-run against the same clause definitions, which is the whole "
        "point of recording it.",
    )
    gate_artifact: str = Field(
        min_length=1,
        description="The durable, never-overwritten key the same run wrote under "
        "gates/{gate}/{trading_day}/gate.json. The block is a copy; this is the "
        "original.",
    )
    clauses: list[ClosingReadingClauseRow]


# ── I10045 row 9: the trial ledger ─────────────────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class TrialRow(BaseModel):
    """One row of `trials/ledger.jsonl` — plan §9.1 row 2, the DSR
    multiplicity denominator.

    **Unlike most models in this module, `extra` is NOT forbidden.**
    `crucible.ledger`'s own module docstring states the design directly:
    "Additive on `trial.v1`: the row has no JSON Schema, every reader
    ignores unknown keys, and an OLDER row that predates the field carries
    no `lineage` key at all". This is the `ArenaCycleDocument` (row 3)
    shape, not the `components.yaml` shape: there is no published contract
    file for this document, by design, so there is nothing to generate a
    schema from and no byte-identity test.

    Used only by `crucible.ledger.trial_rows` — the WRITER — to validate a
    row's own core fields before it is appended, the producer-validates-on-
    the-way-out pattern every other boundary in this migration also takes.
    `crucible.ledger.read_trials`/`append_trials`/`n_trials` are UNCHANGED:
    they still read and compare plain dicts, because the append-only race
    guard in `append_trials` (`current != existing`, both built from
    `read_trials`) is exactly the kind of load-bearing, already-correct
    logic this migration's own row 2/5/6 precedent says to leave alone
    rather than touch for its own sake.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["trial.v1"] = "trial.v1"
    slot: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    as_of: IsoDate
    control: bool
    active: bool
    benchmark: str = Field(min_length=1)
    n_dates_scored: Annotated[int, Field(ge=0)]
    first_date: str | None = None
    last_date: str | None = None
    mean_score_ratio: float | None = None
    lineage: dict[str, list[str]] = Field(default_factory=dict)
    run_id: str = Field(min_length=1)
    arena_cycle_key: str = Field(min_length=1)
    written_at_utc: str = Field(pattern=_UTC_TIMESTAMP_PATTERN)


# ── I10045 row 10: the declared universe ───────────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class DeclaredUniverseDocument(_Strict):
    """`declared_universe.v1`, written at `declared/{trading_day}/universe.json`
    by `crucible.data.universe.DeclaredUniverse.record`.

    The document this repo WRITES (an OWN artifact, `extra="forbid"`) —
    distinct from the document it READS to build a `DeclaredUniverse` in
    the first place (`crucible.data.universe.load_declared_universe`'s
    membership/pointer document), which is a THIRD-PARTY artifact (the
    fleet's live constituents artifact, written by the v1 trading path) and
    stays untyped by this PR, the same carve-out row 12 states explicitly
    for the CloudTrail partition payload: model what we read only when the
    shape is ours to declare, never forbid extras on a document someone
    else's system writes. No schema existed for this write-side document
    before this PR; it is new, per the parent issue's own instruction for a
    "no schema today" row.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$id": "declared_universe.v1",
            "title": "Crucible declared universe, v1",
            "description": (
                "The resolved membership a data job's coverage floor was measured "
                "against, written beside the run so crucible explain can name the "
                "exact denominator. Plan §12 rule 5."
            ),
        },
    )

    schema_version: Literal["declared_universe.v1"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    trading_day: IsoDate
    source_uri: str = Field(
        min_length=1,
        description="Where the membership came from: a URI or path, or argv:--symbols.",
    )
    source_sha256: Sha256 = Field(
        description="sha256 of the source document's bytes (or of the argv literal)."
    )
    origin: str = Field(
        min_length=1,
        description="Settings.origins-style provenance: argument / environ:....",
    )
    count: Annotated[int, Field(ge=0)]
    symbols: list[Annotated[str, Field(min_length=1)]]

    @model_validator(mode="after")
    def _count_matches_the_symbol_list(self) -> DeclaredUniverseDocument:
        """A measured-incident cross-field rule: `count` and `len(symbols)`
        disagreeing is exactly the "denominator nobody can trust" shape this
        whole module exists to prevent, and it is cheap enough to catch at
        the boundary that there is no reason not to.
        """
        if self.count != len(self.symbols):
            raise ValueError(
                f"count={self.count} does not match len(symbols)={len(self.symbols)}. "
                "A declared universe whose own count disagrees with its own list is "
                "not a denominator anyone can trust."
            )
        return self


# ── I10045 row 11: the board declaration ───────────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class BoardDeclarationRow(_Strict):
    """One row body of `board.yaml`'s `objectives`/`cutover` sections,
    `alpha-engine-config-I9837`.

    Types the RAW document shape only — presence, absence and type of the
    body's own keys. `crucible.board.Declaration` (a frozen dataclass with
    its own `__post_init__`; UNCHANGED by this PR) keeps every cross-field
    rule this migration would otherwise move onto the model: `reader` XOR
    `planned_because`, a non-empty `artifact`/`means_when_red`, and `source`
    membership in `crucible.board.SOURCES`. `Declaration` is constructed
    from EVERY code path already (`crucible.board._declaration` is not the
    only caller `tests/test_board.py` constructs it directly to prove
    validation is structural, not per-caller — the same property row 5/6's
    `TestValidationIsStructuralNotPerCaller` proves for `ReleaseRecord`/
    `ChampionPointer`), so duplicating those rules here would be two
    enforcement points for one contract.

    `reader` membership in `crucible.board.READERS` also stays in the
    reader (`crucible.board._declaration`), not here, for the same reason
    row 2's `ArmRecipeDocument` leaves `get_ranker` membership to
    `crucible.slots.arms._parse`: it is a live-registry membership check,
    not a document-shape fact, and importing `crucible.board` here would
    invert this module's own import direction (`crucible.board` imports
    `crucible.models`, not the reverse).
    """

    statement: str
    surface: str
    reader: str | None = None
    artifact: str
    means_when_red: str
    section: str = ""
    clause_class: str = ""
    planned_because: str = ""


# ── I10045 row 12: the CloudTrail partition payload ────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class CloudTrailSessionIssuer(BaseModel):
    """`userIdentity.sessionContext.sessionIssuer` — the assumed ROLE, not
    the session. `extra="allow"` throughout this boundary: AWS owns this
    shape, not us, and forbidding a field AWS adds tomorrow is the defect
    binding constraint 2 names for a third-party payload."""

    model_config = ConfigDict(extra="allow")

    userName: str | None = None


class CloudTrailSessionContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessionIssuer: CloudTrailSessionIssuer | None = None


class CloudTrailUserIdentity(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "Unknown"
    userName: str | None = None
    arn: str | None = None
    sessionContext: CloudTrailSessionContext | None = None


class CloudTrailRecord(BaseModel):
    """One CloudTrail event record — plan §2 row 1 / §11 risk 8, the
    autonomy gate's own "0 human mutating calls" measurement.

    `crucible.autonomy._principal` used to walk `record.get("userIdentity",
    {}) or {}` three levels deep by hand; a typo'd key at any level (`Sessi
    onIssuer`, `userame`) would resolve to `{}` and silently read as "no
    issuer" rather than raising, which is invisible in exactly the way this
    whole migration exists to prevent EXCEPT that this document is a
    third-party shape: `extra="allow"` at every level, because CloudTrail's
    schema varies by event source and this reader must not refuse a field
    it does not yet know the name of. Scoped to the KEPT records only
    (`crucible.autonomy.count_operator_actions`'s `read.records`, already
    filtered down from the full scanned archive by `_is_candidate`) — the
    per-object hot path (`_touches`, `_is_candidate`) stays on raw dicts,
    unchanged, for the memory/throughput reason
    `crucible.autonomy.iter_archive_records`'s own docstring measures:
    ~181k records/day, 6-7 GB resident if every one were held or validated.
    """

    model_config = ConfigDict(extra="allow")

    userIdentity: CloudTrailUserIdentity = Field(default_factory=CloudTrailUserIdentity)
    eventTime: str = ""
    eventName: str = ""
    eventSource: str = ""
    requestID: str = ""
    readOnly: bool | None = None


# ── I10045 row 13: the review artifact ─────────────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class GitHubCommitIdentity(BaseModel):
    """`commit.author`/`commit.committer` on one `GET /pulls/{n}/commits`
    row — the git identity, not the GitHub account. `extra="allow"`: GitHub
    owns this shape, not us (the CloudTrail-record carve-out, row 12)."""

    model_config = ConfigDict(extra="allow")

    email: str | None = None


class GitHubCommitDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""
    author: GitHubCommitIdentity | None = None
    committer: GitHubCommitIdentity | None = None


class GitHubUser(BaseModel):
    """The top-level `author`/`committer` on one commit row — the GitHub
    ACCOUNT, distinct from `GitHubCommitDetail`'s git identity. `None` for a
    commit GitHub cannot associate with an account, which
    `crucible.review.author_identities` must still read without raising."""

    model_config = ConfigDict(extra="allow")

    login: str | None = None


class GitHubCommit(BaseModel):
    """One row of `GET /pulls/{n}/commits` — plan §11 risk 1,
    `crucible.review.author_identities`'s independence derivation.

    `author_identities` used to walk `commit.get("commit") or {}`, then
    `(payload.get("author") or {}).get("email")`, by hand, for both the git
    identity and the GitHub account, plus the `Claude-Session:` trailer scan
    over the message. A typo'd key at any level resolved silently to `{}`/
    `None` rather than surfacing — the same defect class row 12 fixes for
    CloudTrail. `extra="allow"` throughout: a real commits-API response
    carries dozens of fields this reader never looks at (`sha`, `url`,
    `stats`, `files`, `parents`...), and none of them should be refused.
    """

    model_config = ConfigDict(extra="allow")

    commit: GitHubCommitDetail = Field(default_factory=GitHubCommitDetail)
    author: GitHubUser | None = None
    committer: GitHubUser | None = None


class ReviewDocument(_Strict):
    """`review.v1` — `crucible.review.review_document`'s output, at
    `reviews/{phase}/{trading_day}/{reviewer}/{verdict}.json`, plan §11
    risk 1.

    A FINAL, defense-in-depth check inside `review_document` — added AFTER
    that function's own pre-existing checks (a non-'pass'/'fail' `verdict`
    and a malformed `head_sha` each already raise `ReviewError` with their
    own tested message text, `tests/test_review.py::
    test_a_third_verdict_is_refused`/`test_a_review_that_names_no_commit_
    is_refused`, both unchanged and both firing before this model is ever
    reached). This model additionally catches what those two checks do
    not: an empty/blank `phase`, a non-`session_*` `reviewer`, a non-list
    or empty `authors`, a wrong-typed `pr_number`, or an unknown extra key.

    `crucible.gate._review_problem` — the READ side, cited alongside this
    module in the parent issue — is DELIBERATELY UNCHANGED. It already
    performs the equivalent checks by hand (`schema_version`, `authors`
    shape, `head_sha` pattern) with specific, already-tested message text
    (`tests/test_gate_independent_review.py`:
    `"non-empty list of identities"`, `"40-hex commit sha"`) AND a control-
    flow shape no other boundary in this migration shares: every branch
    returns a `(problem, review)` tuple rather than raising, because a gate
    clause must render UNMEASURABLE, never throw. Routing that function's
    checks through a raise-based model without changing its tested
    behaviour is a real refactor for the read side alone, and this row
    scopes to the WRITE side plus `author_identities`'s commits-API
    boundary; the read side's own migration is a separate, better-isolated
    follow-up if one is wanted.
    """

    schema_version: Literal["review.v1"]
    phase: str = Field(min_length=1)
    verdict: Literal["pass", "fail"]
    reviewer: str = Field(pattern=r"^session_[A-Za-z0-9]{8,}$")
    authors: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    pr_number: int
    head_sha: GitSha
    summary: str
    reviewed_at: str = Field(min_length=1)


# ── I10045 row 14: promote.py's event logs ─────────────────────────────────
# Additive only, appended after the prior rows' markers for the same
# rebase reason.


class RetirementLogRow(_Strict):
    """One row of `retirements/{slot}.jsonl`, written by
    `crucible.promote._append_retirement_events`, policy §6.1: "a
    retirement list containing only retirements cannot be audited" — so
    every active arm gets a row, survivors included, not only the retired
    ones.

    `as_of`/`slot` are this writer's own; the rest is
    `nousergon_lib.arena.RetirementVerdict.to_dict()` verbatim — the
    library's shape, restated here as required fields (not re-imported,
    per the row-2/row-3 precedent: this module carries no import-time
    dependency on the library type it mirrors). `event_id` is stamped
    AFTER this validation, by `crucible.promote._append_events`, and is
    therefore not part of this model — validating it here would validate a
    field that does not exist yet at the point this model is used.
    """

    as_of: IsoDate
    slot: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    retire: bool
    reason: str
    age_weeks: Annotated[int, Field(ge=0)]
    pairwise_losses: Annotated[int, Field(ge=0)]
    is_champion: bool


class _ExperimentEventBase(_Strict):
    slot: str = Field(min_length=1)
    as_of: IsoDate


class PromotionEventRow(_ExperimentEventBase):
    """One `experiments/{as_of}.jsonl` row: a challenger moved the pointer."""

    kind: Literal["promotion"]
    arm_id: str = Field(min_length=1)
    incumbent: str | None
    status: str
    reason: str
    window: dict[str, Any]


class NegativeResultEventRow(_ExperimentEventBase):
    """A challenger that was measured and did not win — plan §9.1's
    negative result, durable rather than a private doc somebody remembers
    to edit."""

    kind: Literal["negative_result"]
    arm_id: str = Field(min_length=1)
    incumbent: str | None
    status: str
    reason: str
    window: dict[str, Any]
    confidence_sequence: dict[str, Any] | None


class EligibilityHoldEventRow(_ExperimentEventBase):
    kind: Literal["eligibility_hold"]
    reason: str
    promote_min_weeks: Annotated[int, Field(ge=1)]
    #: Which bar the hold was measured against (`alpha-engine-config-I10547`).
    #: Optional so rows already on the feed still validate; the writer always
    #: sets it.
    promote_evidence: Literal["anytime_valid", "point"] | None = None
    paired_dates_required: Annotated[int, Field(ge=1)]


class RetirementEventRow(_ExperimentEventBase):
    """The RETIRED subset of a cycle's verdicts, also mirrored onto the
    experiments feed so "why did nothing happen this week" is answerable
    from one place."""

    kind: Literal["retirement"]
    arm_id: str = Field(min_length=1)
    reason: str


class NoComparisonEventRow(_ExperimentEventBase):
    """A cycle with no comparisons at all (a single-arm slot, an
    unservable slot) still emits its shape, because a feed silent on a
    cycle is indistinguishable from a cycle that never ran."""

    kind: Literal["no_comparison"]
    status: str
    reason: str


ExperimentEventRow = Annotated[
    PromotionEventRow
    | NegativeResultEventRow
    | EligibilityHoldEventRow
    | RetirementEventRow
    | NoComparisonEventRow,
    Field(discriminator="kind"),
]

#: The one adapter for the closed `kind` union above. A `TypeAdapter`
#: rather than five separate `model_validate` call sites, so a row with an
#: unrecognized `kind` is refused BY THE UNION'S OWN DISCRIMINATOR, naming
#: the five legal values, instead of failing against whichever branch a
#: caller happened to try first.
EXPERIMENT_EVENT_ROW_ADAPTER: TypeAdapter[ExperimentEventRow] = TypeAdapter(ExperimentEventRow)


# ── The fault-injection record (`alpha-engine-config-I10320`/`-I10322`) ────
# Additive only, appended after the prior rows for the same rebase reason.


class ClosedPathRow(_Strict):
    """One machine-executed probe showing one path into an `unreachable`
    fault's state is closed (`alpha-engine-config-I10327`).

    **This row is the reason `unreachable` is the HARDER record to write,
    not the easier one.** A fault whose state cannot arise has no failed run
    to name, so the `run_id` refusal that guards `induced` cannot guard it;
    without positive evidence, `unreachable` would degrade into an
    attestation a human types, which is precisely the
    assertion-by-having-looked plan §10.7 exists to remove. So the producer
    RUNS a probe per closed path against the real store and records what it
    observed: `crucible.faults` refuses the record unless every declared
    probe actually observed its `expected` outcome.

    `expected` and `observed` are separate fields on purpose. A single
    "result" field collapses "we required a refusal and got one" into "it
    said no", and a reader cannot then tell a probe that verified something
    from a probe whose failure was recorded as prose.
    """

    path: str = Field(
        min_length=1,
        description="The path into the fault's state that this probe shows is closed.",
    )
    mechanism: str = Field(
        min_length=1,
        description="The code that closes it — a dotted symbol, so a reader can go and "
        "read the refusal rather than trust this sentence.",
    )
    probe: str = Field(
        min_length=1,
        description="The machine check `crucible.faults` executed. A named probe, "
        "registered in `crucible.faults.UNREACHABLE_PROBES` — never free text a "
        "caller supplies, which would make the evidence an assertion again.",
    )
    expected: str = Field(
        min_length=1,
        description="What the probe REQUIRED to observe for the path to count as closed.",
    )
    observed: str = Field(
        min_length=1,
        description="What the probe actually observed. Recorded verbatim: a record "
        "whose observation is missing is not evidence of anything.",
    )
    checked_at_utc: UtcTimestamp = Field(
        description="RFC 3339, UTC, `Z` suffix. When the probe ran — a probe result "
        "with no instant cannot be told from one copied out of an older record."
    )


class FaultRecordDocument(_Strict):
    """`fault_record.v1`, written at `faults/{trading_day}/{fault_id}.json`
    by `crucible.faults.record_fault` (`crucible fault.record`).

    The document this repo WRITES (an OWN artifact, `extra="forbid"`), per
    plan §10.7 and `crucible.gate._clause_fault_injection_against_scheduled_
    path`, which declared the key shape and the two field names
    (`manifest_key`, `bus_key`) first and found no producer
    (`alpha-engine-config-I10320`). This model conforms to that contract
    rather than restating it — `fault_id` is deliberately a pattern-
    constrained string, not a `Literal` over `crucible.gate.SCRIPTED_FAULTS`,
    because `crucible.gate` already imports this module and a reverse import
    would be circular; `crucible.faults.record_fault` is where membership in
    `SCRIPTED_FAULTS` is actually enforced, at write time.

    **`outcome` decides which other fields are legal, and none of them is
    optional** (`alpha-engine-config-I10327`). The producer shipped with ONE
    shape — a `run_id` naming a manifest reading `status: failed` — and two of
    plan §10.7's four faults never produce one: fault 1 is ABSORBED by the
    declared transient class (the manifest reads `ok`) and fault 4's state
    cannot be entered at all. A vocabulary of one kind cannot record them.

    +----------------+-----------+-----------+---------------+
    | field          | induced   | absorbed  | unreachable   |
    +================+===========+===========+===============+
    | `run_id`       | required  | required  | **null**      |
    | `manifest_key` | required  | required  | **null**      |
    | `bus_key`      | required  | **null**  | **null**      |
    | `attempt`      | **null**  | required  | **null**      |
    | `closed_paths` | **null**  | **null**  | required      |
    +----------------+-----------+-----------+---------------+

    **`bus_key` is required for `induced`, FORBIDDEN for `absorbed`, absent
    for `unreachable` — never optional, and never borrowable from an
    unrelated incident.** It was nullable-for-any-outcome when this shipped,
    and `alpha-engine-config-I10317`'s agent proposed satisfying §10.7 by
    naming an existing bus row: that is the rubber stamp the `run_id` refusal
    exists to prevent, arriving through the other field. A page on an
    `absorbed` record would mean the retry did NOT work, so its absence is
    required rather than tolerated; and an `induced` fault that paged nobody
    is half the §10.7 exercise, so its presence is required rather than
    deferred.

    **`run_id` is the whole design constraint from `-I10322`, and only
    `induced` carries the excusal.** A fault record names the run it excuses;
    `crucible.gate._clause_arc_runs_ok` (and `_clause_replays_ok`, which reads
    the same predicate) excludes a failed manifest from those clauses ONLY
    when an **`induced`** record's `run_id` matches that manifest's own
    `run_id` — never the trading day alone, so a genuine failure on a day a
    fault was once induced still fails the clause. `absorbed` names an `ok`
    manifest and `unreachable` names none, so neither can excuse anything: the
    exclusion is keyed on `run_id` and narrowed by `outcome`, and widening
    either would make a fault record a way to turn an arbitrary red clause
    green.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$id": "fault_record.v1",
            "title": "Crucible fault-injection record, v1",
            "description": (
                "One durable record per scripted fault exercised against the real "
                "scheduled path (plan §10.7). `outcome` says how the fault ended and "
                "decides which evidence fields are legal: an induced fault names the "
                "failed manifest and the page it produced, an absorbed one names the "
                "ok manifest and the transient-class retry that made it ok, and an "
                "unreachable one names neither and carries a machine-executed probe "
                "per closed path — so 'we ran fault injection' is a reading rather "
                "than a sentence in a session transcript."
            ),
        },
    )

    schema_version: Literal["fault_record.v1"] = Field(
        description="Version of THIS schema. A consumer that cannot read the version "
        "refuses the document rather than guessing."
    )
    fault_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]{1,64}$",
        description="One of `crucible.gate.SCRIPTED_FAULTS`, enforced at write time by "
        "`crucible.faults.record_fault` rather than here (see class docstring).",
    )
    outcome: Literal[FAULT_OUTCOME_VALUES] = Field(  # type: ignore[valid-type]
        description="How the fault ended: `induced` (the job failed), `absorbed` (the "
        "declared transient class handled it and the run succeeded) or `unreachable` "
        "(the state cannot be entered). Decides which other fields are legal — see the "
        "class docstring's table."
    )
    trading_day: IsoDate
    run_id: str | None = Field(
        pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$",
        description="The run_id of the manifest this record describes. ULID. Required "
        "for `induced` and `absorbed`; NULL for `unreachable`, which has no run and "
        "must therefore be structurally incapable of excusing one. Required and "
        "nullable rather than optional: an omitted field is indistinguishable from a "
        "forgotten one.",
    )
    manifest_key: str | None = Field(
        min_length=1,
        description="The store key of the manifest named by run_id, which "
        "`crucible.faults.record_fault` looked up and verified reads the status this "
        "`outcome` requires before this record could be written. NULL for "
        "`unreachable`.",
    )
    bus_key: str | None = Field(
        min_length=1,
        description="The alert bus row this fault produced (alerts/{day}/{incident}.json). "
        "REQUIRED for `induced` — a fault that paged nobody is half of §10.7's exercise. "
        "FORBIDDEN for `absorbed` and `unreachable`: a page on an absorbed fault would "
        "mean the retry did not work, and there is no run to page about at all when the "
        "state is unreachable.",
    )
    attempt: AttemptRow | None = Field(
        description="For `absorbed`, the `attempts[]` row from the named manifest that "
        "records the declared-transient-class retry — copied out of the manifest the "
        "producer verified, so the record says WHICH transient class absorbed the fault "
        "rather than merely that one did. NULL for every other outcome.",
    )
    closed_paths: list[ClosedPathRow] | None = Field(
        description="For `unreachable`, one machine-executed probe per closed path, "
        "never empty. NULL for every other outcome — a record that both excuses a run "
        "and claims the state is unreachable is claiming two contradictory things.",
    )
    recorded_at_utc: UtcTimestamp = Field(
        description="RFC 3339, UTC, `Z` suffix. When the injection procedure filed this "
        "record, distinct from trading_day — a fault is exercised against a trading "
        "day's scheduled path, but the record is filed at the wall-clock instant of the "
        "exercise."
    )

    #: Per outcome, the fields that MUST be non-null and the fields that MUST
    #: be null. Declared as data rather than written out as a chain of `if`s
    #: so the table in the class docstring and the enforcement cannot drift,
    #: and so `tests/test_typed_boundary_fault_record.py` can walk every cell
    #: of it instead of asserting the handful somebody remembered.
    _REQUIRED_BY_OUTCOME: ClassVar[dict[str, tuple[str, ...]]] = {
        "induced": ("run_id", "manifest_key", "bus_key"),
        "absorbed": ("run_id", "manifest_key", "attempt"),
        "unreachable": ("closed_paths",),
    }
    _EVIDENCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "run_id",
        "manifest_key",
        "bus_key",
        "attempt",
        "closed_paths",
    )

    @model_validator(mode="after")
    def _the_outcome_carries_exactly_its_own_evidence(self) -> FaultRecordDocument:
        required = self._REQUIRED_BY_OUTCOME[self.outcome]
        for field in self._EVIDENCE_FIELDS:
            value = getattr(self, field)
            if field in required and value is None:
                raise ValueError(
                    f"a {self.outcome!r} fault record requires `{field}` and this one "
                    f"has none. Required for {self.outcome!r}: {list(required)}."
                )
            if field not in required and value is not None:
                raise ValueError(
                    f"a {self.outcome!r} fault record must not carry `{field}`, and this "
                    f"one names {value!r}. Only {list(required)} are legal evidence for "
                    f"{self.outcome!r}; a field borrowed from another outcome's shape is "
                    "how a record stops being evidence of what it claims."
                )
        if self.closed_paths is not None and not self.closed_paths:
            raise ValueError(
                "an `unreachable` fault record with an EMPTY `closed_paths` claims a "
                "state cannot be entered and names not one closed path — the shape this "
                "outcome exists to forbid."
            )
        if self.attempt is not None and self.attempt.reason == "initial":
            raise ValueError(
                "an `absorbed` fault record's `attempt` must be the RETRY that absorbed "
                "the fault, not the initial attempt: `reason: initial` is every run's "
                "first attempt and says nothing about a fault being handled."
            )
        if self.attempt is not None and self.attempt.n < 2:
            raise ValueError(
                f"an `absorbed` fault record's `attempt` is the retry, so its `n` is at "
                f"least 2; this one reads {self.attempt.n}."
            )
        return self
