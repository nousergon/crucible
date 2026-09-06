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

`RunManifestV2` (`alpha-engine-config-I10045` row 1) is a partial exception to
"the schema is generated from the model, full stop": `run_manifest.v2.json`
IS regenerated from `RunManifestV2.model_json_schema()` — same one-source-of-
truth test as `components_registry.v1.json` — but the two run-manifest-status
cross-field rules (`status: ok` implies empty `reason`, `status: failed`
implies non-empty `reason`) are enforced ONLY by
`_RunManifestV2._status_and_reason_agree` and are deliberately **not**
re-encoded as an `allOf` in the generated schema, for the same reason
`ComponentRow`'s dispatch/deadline rules never appeared in
`components_registry.v1.json`: a cross-field rule is the model's job, and a
schema carrying it twice is the two-sources-of-truth shape this migration
exists to remove. A consumer validating a manifest against the published
schema alone (no Python import) gets the structural contract; a consumer
going through `crucible.manifest.validate`/`read_manifest` gets both.

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
from typing import Annotated, Literal, get_args

from krepis.metrics import StatusLiteral
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ArtifactRef",
    "AttemptRow",
    "ComponentRow",
    "ComponentsDocument",
    "DeadlineRow",
    "LlmCallRow",
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

IsoDate = Annotated[str, Field(pattern=_ISO_DATE_PATTERN)]
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

    key: str = Field(min_length=1)
    sha256: Sha256
    schema_version: str = Field(min_length=1)


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

    callsite_id: str = Field(min_length=1)
    model_requested: str = Field(min_length=1)
    model_served: str = Field(min_length=1)
    route_degraded: bool
    fallback_used: bool
    #: `null` is the router reporting no deployment — an ANSWER, not an
    #: absence — so the key is required and the value is nullable
    #: (`alpha-engine-config-I10006`, `I9995`).
    served_deployment: str | None
    tokens_in: Annotated[int, Field(ge=0)]
    tokens_out: Annotated[int, Field(ge=0)]
    cache_read: Annotated[int, Field(ge=0)]
    cache_write: Annotated[int, Field(ge=0)]
    usd: Annotated[float, Field(ge=0)]


class ResourceRow(_Strict):
    """§9.2 class 3. Present on every manifest including laptop runs, where
    `spot` is false and `instance_type` is `local`."""

    instance_type: str = Field(min_length=1)
    spot: bool
    #: I5727: spot-to-on-demand fallback is a COUNTABLE metric, so it is
    #: required rather than an optional annotation.
    escalated_to_on_demand: bool
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
    reason: Literal[ATTEMPT_REASON_VALUES]  # type: ignore[valid-type]


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

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    module: str = Field(min_length=1)
    metric_type: str = Field(min_length=1)
    value: float | None = None
    #: Required whenever `value` is set — see the validator below. A numeric
    #: field with no declared unit is the defect that emitted
    #: `avg_volume_20d` as a ratio and consumed it as raw shares.
    unit: str | None = None
    n_floor: Annotated[int, Field(ge=0)]
    status: Literal[METRIC_STATUS_VALUES]  # type: ignore[valid-type]
    status_reason: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    last_updated_utc: UtcTimestamp
    #: §4.12: a horizon is an INTEGER COUNT OF TRADING DAYS — 21/63/126/252.
    #: There is no string form, so `"1 month"` fails validation. Absent when
    #: the metric has no forward horizon.
    horizon_trading_days: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _unit_required_when_value_present(self) -> MetricRecordRow:
        if self.value is not None and not self.unit:
            raise ValueError(
                f"{self.name!r} sets value={self.value!r} with no unit. A numeric "
                "value with no declared unit is not a measurement a consumer can "
                "safely render or compare."
            )
        return self


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
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/nousergon/crucible/schemas/run_manifest.v2.json",
            "title": "Crucible run manifest, v2",
            "description": (
                "The record every job writes at runs/{job}/{trading_day}/run.json. "
                "Generated from crucible.models.RunManifestV2; v1 declared no "
                "live/replay field and stays frozen at "
                "crucible/schemas/run_manifest.v1.json. "
                "The status<->reason cross-field rule is enforced by the model's "
                "`_status_and_reason_agree`, not restated here as an `allOf` -- "
                "see crucible/models.py's module docstring."
            ),
        },
    )

    schema_version: Literal["run_manifest.v2"]
    run_id: str = Field(
        pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$",
        description="ULID: lexically sortable by creation time. The correlation "
        "identity (§9.2), on every log line, alert, cost row and S3 object's "
        "metadata for this run.",
    )
    job: Literal[JOB_VALUES]  # type: ignore[valid-type]
    #: Whether this run was a LIVE execution or a REPLAY of a historical
    #: trading day. Required, closed vocabulary, deliberately NO DEFAULT: a
    #: default is what makes a replay indistinguishable from a live run the
    #: first time a producer forgets to set it (`alpha-engine-config-I9918`).
    #: Set from the invocation (`crucible.runmode.resolve_run_mode`), never
    #: derived from `trading_day`/`calendar_date`.
    run_mode: Literal["live", "replay"]
    #: THE KEY (§4.12). Never the wall-clock date.
    trading_day: IsoDate
    #: Present only for a job that legitimately writes more than one
    #: manifest for the same `job`+`trading_day` (`alpha-engine-config-
    #: I9781`). Absent for every other job.
    discriminator: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]{1,64}$"
    )
    #: Wall-clock date the run actually executed on, for PROVENANCE ONLY.
    #: Never a key, never an input to a promotion/retirement/freshness/
    #: grading decision.
    calendar_date: IsoDate
    #: Exhaustive. There is deliberately no `partial`, `skipped`, `degraded`
    #: or `unknown`.
    status: Literal["ok", "failed"]
    #: Empty when `status` is `ok`; a specific, operator-readable cause when
    #: `failed` — enforced by `_status_and_reason_agree` below, not by the
    #: published schema (see this module's docstring).
    reason: str = Field(max_length=2000)
    started: UtcTimestamp
    #: Written in the runner's `finally` block, so it is present even when
    #: the job raised.
    finished: UtcTimestamp
    code_sha: GitSha
    release_sha: GitSha
    #: Required, including for jobs that are deterministic today: an
    #: unrecorded seed makes a replay diff unattributable.
    seed: Annotated[int, Field(ge=0)]
    inputs: list[ArtifactRef]
    outputs: list[ArtifactRef]
    rows_in: Annotated[int, Field(ge=0)]
    rows_out: Annotated[int, Field(ge=0)]
    rows_rejected: list[RejectedRow]
    #: Must be >= the sum of `llm_calls[].usd`; the runner asserts that,
    #: since a schema cannot.
    cost_usd: Annotated[float, Field(ge=0)]
    #: Empty for every non-research job — LLM access is confined to research
    #: by standing rule.
    llm_calls: list[LlmCallRow]
    resource: ResourceRow
    metrics: list[MetricRecordRow]
    #: Never empty: a run that executed once records one attempt.
    attempts: list[AttemptRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _status_and_reason_agree(self) -> RunManifestV2:
        """The plan's central guarantee (§4.2): a failed run states why, and
        an ok run has nothing to explain.

        This is the cross-field rule this module's docstring names — it
        stays a model_validator and is not re-encoded as the schema's
        `allOf`, the same choice `ComponentRow`'s dispatch/deadline rules
        made for `components_registry.v1.json`.
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
