r"""IaC conformance: two independent comparisons, never one.

Normative source: `alpha-engine-config-I10418`, scope corrected 2026-09-10
(the issue's own comment is normative over its original body — see that
comment for the full argument). Plan §4.11 (Infra row) and the amended §13
row: *"One template, two roles, weekly drift diff — asserted. Plus §9.8
conformance clause: tagged estate == template, measured."*

**Why two comparisons and not one.** As filed, this clause was:

    (a) **account vs template** — the set of live AWS resources tagged
    `system=crucible-v2` equals the set the template declares. A resource in
    the account and not in the template is a finding; so is the reverse (a
    stack update that silently failed or rolled back).

That alone is exactly the checker that would have rendered GREEN while the
plan's own §9.8 objective ("two IAM roles only") had already gone stale: (a)
only ever compares the account to *whatever the template currently says*, so
a template that drifted from the plan's intent — 15 IAM roles where the plan
said two — passes (a) by construction, because the template agrees with
itself. Measured 2026-09-10: `nous-ergon-ops/infrastructure/cloudformation/
crucible-v2.yaml` declares 15 IAM roles (`DeployRole`, `RuntimeRole`,
`BoardRole`, `LockstepRole`, `GateCloseRole`, `ReviewWriterRole`,
`MorningReportRole`, `AcceptanceReaderRole`, `DlpConfigPublisherRole`,
`StrategyPublisherRole`, `StackCheckRole`, `LegacyWeeklyProducerRole`,
`DispatcherRole`, `SchedulerRole`, `FisExecutionRole`) and ~34 resources —
ten of the fifteen are per-workflow OIDC deploy roles, one identity per
authority boundary per `identity-access-policy` and Brian's 2026-09-06
ruling, so the drift from "two roles" is *correct*, not a defect. But nothing
had ever measured that it disagreed with the plan's stated objective, which
is precisely the v1 counterexample this whole clause exists to prevent:
~16-18 IAM roles live in CloudFormation and 42 more as raw JSON outside it,
an estate that became unauditable one convenient exception at a time because
an assertion nothing read was treated as a fact.

So the second comparison:

    (b) **template vs the plan's own declared inventory** — the plan states a
    resource and role inventory; the template is checked against THAT,
    independently of what the live account holds.

**Non-circularity is the whole design constraint on (b).** The "declared
inventory" cannot live inside the template file itself (a `Metadata` block,
say) — a template author who adds a role edits the same file the comparison
reads on both sides, so drift becomes structurally undetectable, exactly the
shape that let "two roles" survive fifteen. It also cannot be re-derived from
the live stack — that collapses back into (a). So it is read from an
independent source an operator sets deliberately, by hand, when the plan's
own inventory changes: an SSM parameter (`declared_inventory_from_ssm`),
never written by `aws cloudformation deploy` and never read by it. Updating
the template does not update the parameter; the two can only agree because
someone looked at both and made them agree, which is the property this
clause exists to keep true.

**Findings page only on the SECOND consecutive week** (issue's revised
closes-when). A stack mid-`UPDATE_IN_PROGRESS` reads a plausible partial
result once and heals itself the next Saturday; requiring persistence is
what keeps that from being indistinguishable from real drift. See
`iac_conformance_handler` and `_previous_week_had_findings`.

Both readings are always written to the store (`iac_conformance_key`) and
onto the manifest's own `metrics`, whether or not they persist far enough to
page — the board is where the sibling track renders them (this module does
not touch `crucible/board.py`); a resource created out-of-band in a fixture
is `only_in_account`, a plan inventory edited to disagree with the template
is `roles_only_in_declared`/`roles_only_in_template`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

from krepis.trading_calendar import subtract_trading_days

from crucible.calendar import TRADING_DAYS_PER_WEEK
from crucible.documents import read_store_document
from crucible.keys import iac_conformance_key, manifest_key
from crucible.runner import RunContext, run_job
from crucible.store import Store
from crucible.tags import (
    TAG_KEY,
    TAG_VALUE,
    StackNotAppliedError,
    _tagged_identifiers,  # noqa: PLC2701 - established cross-module read; see crucible.autonomy
    audit_stack_tags,
)

__all__ = [
    "IAC_CONFORMANCE_JOB",
    "IAC_CONFORMANCE_MODULE",
    "AccessDeniedReadingTheAccountError",
    "AccountTemplateAudit",
    "DeclaredInventory",
    "DeclaredInventoryUnreadableError",
    "DeclaredInventoryUnsetError",
    "IacConformanceDrift",
    "TemplateDeclaredAudit",
    "TemplateInventory",
    "TemplateUnreadableError",
    "audit_account_vs_template",
    "audit_template_vs_declared",
    "declared_inventory_from_ssm",
    "iac_conformance_handler",
    "template_inventory_from_stack",
]

IAC_CONFORMANCE_JOB = "iac.conformance"
IAC_CONFORMANCE_MODULE = "crucible.iac_conformance"

#: The two comparison names, as recorded on `MetricRecordRow.name`. A closed
#: pair — a third comparison would be a design change visible in a diff, the
#: same discipline `components.py`'s two deadline anchors and `alerts.py`'s
#: two page conditions are held to.
COMPARISON_ACCOUNT_VS_TEMPLATE = "iac_account_vs_template"
COMPARISON_TEMPLATE_VS_DECLARED = "iac_template_vs_declared"
COMPARISON_PERSISTED = "iac_conformance_persisted"


#: The AWS error codes that mean "this identity may not make that call".
#: A closed set, checked by code rather than by matching on a message: the
#: message is prose AWS may reword, the code is the contract. Everything
#: outside this set is a real error and propagates, because the whole point
#: of the distinction below is that an access failure is a statement about
#: OUR GRANTS and any other failure is a statement about the estate.
ACCESS_DENIED_CODES = frozenset({"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"})


def _is_access_denied(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    return response.get("Error", {}).get("Code", "") in ACCESS_DENIED_CODES


class AccessDeniedReadingTheAccountError(RuntimeError):
    """Comparison (a) could not read the account: a grant is missing.

    The sibling of :class:`TemplateUnreadableError` and
    :class:`DeclaredInventoryUnreadableError`, and it exists because
    comparison (a) was the ONE of the three that had no such sibling.
    Comparisons (b) and the cost-allocation read both already say "that is a
    statement about our access, not about <the subject>" and render
    UNMEASURABLE; (a) let a raw `ClientError` out of the handler, and
    `iac.conformance` is an ARC STAGE — so a missing grant did not degrade
    one metric, it raised `ArcStageFailed` and took the whole weekly arc down
    with it, twice, on two consecutive rehearsals of 2026-07-31:

      2026-09-10, first rehearsal:
        AccessDenied ... cloudformation:ListStackResources
      2026-09-11, after that grant landed:
        AccessDenied ... iam:ListRoleTags on resource: role admin

    Each cost a whole arc, and on a real Saturday would have cost
    `live_saturdays_first_attempt_ok` — a phase-2 exit clause that grades the
    FIRST attempt and cannot be re-run for credit. The grants are fixed in
    `nous-ergon-ops` where they belong; this class is the reason a THIRD
    missing grant costs one UNMEASURABLE metric instead of a Saturday.

    UNMEASURABLE is never a pass. The metric still reads red on the board and
    the clause reading it is still unmet — what changes is only that the
    stages after this one get to run.
    """


class TemplateUnreadableError(RuntimeError):
    """The deployed template body could not be read or carries no Resources.

    Raised rather than reporting zero resources: an empty template read is
    indistinguishable, from the caller's side, from a template that genuinely
    declares nothing — and a stack this system depends on always declares
    something once applied (`crucible.tags.StackNotAppliedError` is the
    sibling refusal for the stack itself not existing yet).
    """


class DeclaredInventoryUnsetError(RuntimeError):
    """No declared-inventory source is configured.

    Raised rather than defaulting to "zero roles declared" (which would make
    the template read as drifted against nothing it ever agreed to) or to
    "whatever the template currently has" (which collapses comparison (b)
    into comparing the template to itself — the exact circularity the module
    docstring names as the reason this value lives outside the template).
    """


class DeclaredInventoryUnreadableError(RuntimeError):
    """The configured declared-inventory parameter could not be read or parsed."""


class IacConformanceDrift(RuntimeError):
    """A conformance finding persisted two consecutive weekly cycles.

    Raised by the job body, never caught here: `crucible.runner.run_job`'s
    manifest guarantee turns this into a `failed` manifest carrying the
    finding, which is what makes `crucible.alerts.evaluate_failure` — one of
    the two declared page conditions — the thing that pages, rather than a
    third page condition this module would otherwise have had to invent.
    """


# ── comparison (a): account vs template ──────────────────────────────────


@dataclass(frozen=True)
class AccountTemplateAudit:
    """The symmetric difference between what is tagged live and what the
    currently-deployed stack's taggable resources are.

    `only_in_account` is a resource carrying `system=crucible-v2` that the
    stack does not manage — created out-of-band, or orphaned by a stack
    replacement. `only_in_template` is a resource the stack declares (and
    which is taggable) that the tag scan did not find — an untagged resource,
    or one a rolled-back `UPDATE_ROLLBACK_COMPLETE` never actually created.
    Both are findings; neither is the same question as
    `crucible.tags.audit_stack_tags` alone answers, which only ever reads
    resources the stack CURRENTLY lists and so cannot see the first case.
    """

    stack: str
    only_in_account: tuple[str, ...]
    only_in_template: tuple[tuple[str, str, str], ...]  # (logical_id, type, physical_id)
    template_resource_count: int
    account_tagged_count: int

    @property
    def met(self) -> bool:
        return not self.only_in_account and not self.only_in_template

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack": self.stack,
            "met": self.met,
            "template_resource_count": self.template_resource_count,
            "account_tagged_count": self.account_tagged_count,
            "only_in_account": list(self.only_in_account),
            "only_in_template": [
                {"logical_id": lid, "type": kind, "physical_id": pid}
                for lid, kind, pid in self.only_in_template
            ],
        }

    def detail(self) -> str:
        if self.met:
            return (
                f"{self.account_tagged_count} tagged resource(s) in the account match the "
                f"{self.template_resource_count} taggable resource(s) {self.stack!r} declares"
            )
        parts = []
        if self.only_in_account:
            parts.append(f"{len(self.only_in_account)} tagged but not stack-managed")
        if self.only_in_template:
            parts.append(f"{len(self.only_in_template)} stack-declared but not tagged live")
        return "; ".join(parts)


def _tagged_iam_role_names(iam: Any) -> set[str]:
    """Every IAM role account-wide carrying `system=crucible-v2`.

    `resourcegroupstaggingapi` does not cover IAM at all (`crucible.tags`'
    module docstring, measured 2026-09-01), so a role created out-of-band —
    the exact case `only_in_account` exists to catch — would be invisible to
    the tagging-API scan alone. `iam:ListRoles` is account-wide and
    unavoidably a per-role `iam:ListRoleTags` call each; that is the
    documented cost of this comparison, paid once a week.
    """
    if iam is None:
        return set()
    names: set[str] = set()
    paginator = iam.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page.get("Roles", []):
            role_name = role.get("RoleName")
            if not role_name:
                continue
            tags = iam.list_role_tags(RoleName=role_name).get("Tags", [])
            if any(t.get("Key") == TAG_KEY and t.get("Value") == TAG_VALUE for t in tags):
                names.add(role_name)
    return names


def audit_account_vs_template(
    *, stack: str, cfn: Any, tagging: Any, iam: Any = None
) -> AccountTemplateAudit:
    """Comparison (a). Raises `StackNotAppliedError` if the stack does not
    exist yet — the same UNMEASURABLE-never-a-pass refusal
    `crucible.tags.audit_stack_tags` already gives, reused rather than
    reimplemented (a stack with no resources is vacuous truth either way).

    Raises :class:`AccessDeniedReadingTheAccountError` when any of its four
    reads is denied. Declared, so the handler can render this comparison
    UNMEASURABLE the way it already renders comparison (b) — see that class
    for why an arc stage must never let a raw `ClientError` out.
    """
    try:
        stack_audit = audit_stack_tags(stack=stack, cfn=cfn, tagging=tagging, iam=iam)
        stack_ids = {physical for _, _, physical in stack_audit.resources}

        tagged_ids = set(_tagged_identifiers(tagging))
        tagged_ids |= _tagged_iam_role_names(iam)
    except StackNotAppliedError:
        # The stack genuinely not existing is a different answer from being
        # unable to look, and it has its own UNMEASURABLE path already.
        raise
    except Exception as exc:  # noqa: BLE001 - narrowed on the next line, re-raised otherwise
        if not _is_access_denied(exc):
            raise
        raise AccessDeniedReadingTheAccountError(
            f"comparison (a) could not read the account for stack {stack!r}: "
            f"{type(exc).__name__}: {exc}. That is a statement about our grants, not "
            "about whether the estate conforms"
        ) from exc

    only_in_account = tuple(sorted(pid for pid in tagged_ids if pid and pid not in stack_ids))
    return AccountTemplateAudit(
        stack=stack,
        only_in_account=only_in_account,
        only_in_template=stack_audit.untagged,
        template_resource_count=len(stack_audit.resources),
        account_tagged_count=len(tagged_ids),
    )


# ── comparison (b): template vs the plan's declared inventory ───────────


@dataclass(frozen=True)
class TemplateInventory:
    """What the currently-deployed template's `Resources` block declares."""

    resource_count: int
    iam_role_logical_ids: tuple[str, ...]

    @property
    def iam_role_count(self) -> int:
        return len(self.iam_role_logical_ids)


@dataclass(frozen=True)
class DeclaredInventory:
    """What the plan states the estate should be, read from a source
    independent of the template (see the module docstring)."""

    resource_count: int
    iam_role_logical_ids: tuple[str, ...]
    source: str

    @property
    def iam_role_count(self) -> int:
        return len(self.iam_role_logical_ids)


@dataclass(frozen=True)
class TemplateDeclaredAudit:
    template: TemplateInventory
    declared: DeclaredInventory
    roles_only_in_template: tuple[str, ...]
    roles_only_in_declared: tuple[str, ...]

    @property
    def met(self) -> bool:
        return (
            self.template.resource_count == self.declared.resource_count
            and not self.roles_only_in_template
            and not self.roles_only_in_declared
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "met": self.met,
            "template_resource_count": self.template.resource_count,
            "declared_resource_count": self.declared.resource_count,
            "template_iam_role_count": self.template.iam_role_count,
            "declared_iam_role_count": self.declared.iam_role_count,
            "roles_only_in_template": list(self.roles_only_in_template),
            "roles_only_in_declared": list(self.roles_only_in_declared),
            "declared_source": self.declared.source,
        }

    def detail(self) -> str:
        if self.met:
            return (
                f"template ({self.template.resource_count} resources, "
                f"{self.template.iam_role_count} IAM roles) matches the plan's declared "
                f"inventory at {self.declared.source}"
            )
        parts = []
        if self.template.resource_count != self.declared.resource_count:
            parts.append(
                f"resource count {self.template.resource_count} != declared "
                f"{self.declared.resource_count}"
            )
        if self.roles_only_in_template:
            parts.append(f"roles only in template: {', '.join(self.roles_only_in_template)}")
        if self.roles_only_in_declared:
            parts.append(
                f"roles only in declared inventory: {', '.join(self.roles_only_in_declared)}"
            )
        return "; ".join(parts)


def _multi_constructor(loader: Any, tag_suffix: str, node: Any) -> Any:
    """Tolerate every CloudFormation short-form tag (`!Ref`, `!Sub`, `!GetAtt`,
    ...) without resolving it. This module only ever reads `Resources.<id>
    .Type`, a plain scalar never expressed with an intrinsic function, so
    resolving the tagged nodes correctly is unnecessary — only *not choking
    on them* is.
    """
    import yaml  # noqa: PLC0415 - lazy, this is the one call site

    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


def _parse_cfn_yaml(text: str) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 - lazy, keeps yaml off the import path for callers that never parse

    loader_cls = type("_CfnLoader", (yaml.SafeLoader,), {})
    loader_cls.add_multi_constructor("!", _multi_constructor)
    document = yaml.load(text, Loader=loader_cls)  # noqa: S506 - the loader above is a SafeLoader subclass
    if not isinstance(document, dict):
        raise TemplateUnreadableError(
            "the template body did not parse to a mapping; it may not be valid CloudFormation YAML"
        )
    return document


def template_inventory_from_stack(*, stack: str, cfn: Any) -> TemplateInventory:
    """Reads the ACTUAL currently-applied template via `GetTemplate` —
    deliberately not the stack's current resource list
    (`list_stack_resources`, which `audit_account_vs_template` already
    reads): a resource the template declares but which failed to create is
    exactly what comparison (a) exists to catch, and reading the same
    live-resource source for both comparisons would make them agree by
    construction on the one disagreement (a) is built to surface.
    """
    try:
        response = cfn.get_template(StackName=stack)
    except Exception as exc:  # noqa: BLE001 - narrowed immediately below, same shape as crucible.tags
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        message = str(exc)
        if code == "ValidationError" and "does not exist" in message:
            raise StackNotAppliedError(
                f"stack {stack!r} does not exist, so its template cannot be read. Comparison (b) "
                "is UNMEASURABLE until the operator has applied the stack"
            ) from exc
        raise TemplateUnreadableError(
            f"get_template({stack!r}) failed: {type(exc).__name__}: {exc}. That is a statement "
            "about our access, not about what the template declares"
        ) from exc
    body = response.get("TemplateBody")
    if body is None:
        raise TemplateUnreadableError(f"get_template({stack!r}) returned no TemplateBody")
    document = body if isinstance(body, dict) else _parse_cfn_yaml(body)
    resources = document.get("Resources") or {}
    if not resources:
        raise TemplateUnreadableError(
            f"the template for {stack!r} declares no Resources — an empty template would make "
            "this comparison vacuously true, which it must never be"
        )
    iam_roles = tuple(
        sorted(
            logical_id
            for logical_id, definition in resources.items()
            if isinstance(definition, dict) and definition.get("Type") == "AWS::IAM::Role"
        )
    )
    return TemplateInventory(resource_count=len(resources), iam_role_logical_ids=iam_roles)


def declared_inventory_from_ssm(*, parameter_name: str, ssm: Any) -> DeclaredInventory:
    """The plan's own declared inventory, read from an SSM parameter an
    operator sets by hand when the plan's stated inventory changes — never
    written by `aws cloudformation deploy`. See the module docstring
    ("Non-circularity is the whole design constraint on (b)") for why this
    cannot be read from the template or the live stack.

    Parameter value is a JSON object: `{"resource_count": int, "iam_roles":
    [str, ...]}`. Set with, e.g.:

        aws ssm put-parameter --name /crucible-v2/iac/declared-inventory \\
          --type String --overwrite --value '{"resource_count": 34, \\
          "iam_roles": ["DeployRole", "RuntimeRole", "BoardRole", \\
          "LockstepRole", "GateCloseRole", "ReviewWriterRole", \\
          "MorningReportRole", "AcceptanceReaderRole", \\
          "DlpConfigPublisherRole", "StrategyPublisherRole", \\
          "StackCheckRole", "LegacyWeeklyProducerRole", "DispatcherRole", \\
          "SchedulerRole", "FisExecutionRole"]}'
    """
    if not parameter_name:
        raise DeclaredInventoryUnsetError(
            "no declared-inventory parameter is configured "
            "(CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM). Comparison (b) cannot run without the "
            "plan's own declared inventory to check the template against — treating an unset "
            "value as 'no roles declared' would make every template read as drifted, and "
            "treating it as 'whatever the template has' would compare the template to itself, "
            "exactly the circularity this comparison exists to avoid."
        )
    try:
        response = ssm.get_parameter(Name=parameter_name)
    except Exception as exc:  # noqa: BLE001 - re-raised as a declared type immediately below
        raise DeclaredInventoryUnreadableError(
            f"ssm:GetParameter({parameter_name!r}) failed: {type(exc).__name__}: {exc}. That is "
            "a statement about our access, not about what the plan declares"
        ) from exc
    raw = (response.get("Parameter") or {}).get("Value")
    if not raw:
        raise DeclaredInventoryUnreadableError(f"{parameter_name!r} returned no value")
    try:
        payload = json.loads(raw)
        resource_count = int(payload["resource_count"])
        iam_roles = tuple(sorted(str(name) for name in payload["iam_roles"]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DeclaredInventoryUnreadableError(
            f"{parameter_name!r} is not a valid declared-inventory document "
            f"({{'resource_count': int, 'iam_roles': [str, ...]}}): {type(exc).__name__}: {exc}"
        ) from exc
    return DeclaredInventory(
        resource_count=resource_count,
        iam_role_logical_ids=iam_roles,
        source=f"ssm:{parameter_name}",
    )


def audit_template_vs_declared(
    template: TemplateInventory, declared: DeclaredInventory
) -> TemplateDeclaredAudit:
    """Comparison (b). Pure — the two readings are already resolved by the
    caller, so this function alone is what a fixture proves the comparison
    logic against, with no AWS client at all."""
    template_roles = set(template.iam_role_logical_ids)
    declared_roles = set(declared.iam_role_logical_ids)
    return TemplateDeclaredAudit(
        template=template,
        declared=declared,
        roles_only_in_template=tuple(sorted(template_roles - declared_roles)),
        roles_only_in_declared=tuple(sorted(declared_roles - template_roles)),
    )


# ── the weekly job: both readings, persistence-gated paging ─────────────


def _metric(
    name: str,
    *,
    status: str,
    status_reason: str,
    now: dt.datetime,
    source_path: str,
    value: float | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "module": IAC_CONFORMANCE_MODULE,
        "metric_type": "compliance",
        "value": value,
        "unit": unit,
        "n_floor": 1,
        "status": status,
        "status_reason": status_reason[:2000],
        "source_path": source_path,
        "last_updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _previous_trading_day_one_week_back(trading_day: dt.date) -> dt.date:
    return subtract_trading_days(trading_day, TRADING_DAYS_PER_WEEK)


def _previous_week_had_findings(store: Store, trading_day: dt.date) -> bool:
    """Whether last week's `iac.conformance` manifest recorded a BREACH on
    either comparison.

    **Absent and unreadable are both treated as "not confirmed persisted",
    deliberately** (the swallow rule 5 requires naming): the first Saturday
    this job ever ran has no prior manifest, and a corrupt or inaccessible
    prior manifest is a fact about read access, not about whether a finding
    is real. Reading either as "persisted" would page on a finding this
    process cannot actually confirm was there last week — the SAFE direction
    is to treat both as "not yet two weeks", which under-pages by at most one
    week rather than paging on an unconfirmed premise. The read outcome
    (absent vs. unreadable) is recorded on this week's own
    `iac_conformance_persisted` metric's `status_reason`, so the swallow is
    never silent.
    """
    previous_day = _previous_trading_day_one_week_back(trading_day)
    key = manifest_key(IAC_CONFORMANCE_JOB, previous_day.isoformat())
    read = read_store_document(store, key)
    if read.absent or read.document is None:
        return False
    metrics = read.document.get("metrics") or []
    return any(
        m.get("name") in (COMPARISON_ACCOUNT_VS_TEMPLATE, COMPARISON_TEMPLATE_VS_DECLARED)
        and m.get("status") == "BREACH"
        for m in metrics
        if isinstance(m, dict)
    )


def _cfn_client() -> Any:
    import boto3  # noqa: PLC0415 - lazy on purpose, mirrors crucible.autonomy._cfn_client

    return boto3.client("cloudformation")


def _tagging_client() -> Any:
    import boto3  # noqa: PLC0415 - lazy on purpose

    return boto3.client("resourcegroupstaggingapi")


def _iam_client() -> Any:
    import boto3  # noqa: PLC0415 - lazy on purpose

    return boto3.client("iam")


def _ssm_client() -> Any:
    import boto3  # noqa: PLC0415 - lazy on purpose

    return boto3.client("ssm")


def iac_conformance_handler(args: argparse.Namespace) -> int:
    """The weekly `iac.conformance` job. `dispatch: arc` in `components.yaml`
    is what puts it on the same schedule as every other weekly stage — see
    `crucible.weekly.arc_stages`, which derives its sequence from that row
    rather than a hand-listed stage table.
    """
    from crucible.config import settings  # noqa: PLC0415 - avoid an import cycle at module load

    dry_run = bool(getattr(args, "dry_run", False))
    cfg = settings(store_uri=getattr(args, "store", None), dry_run=dry_run)
    store = cfg.store()

    def body(ctx: RunContext) -> None:
        stack = cfg.stack_name
        now = ctx.started
        metrics: list[dict[str, Any]] = []
        findings: list[str] = []
        readings: dict[str, Any] = {}

        try:
            account_audit = audit_account_vs_template(
                stack=stack, cfn=_cfn_client(), tagging=_tagging_client(), iam=_iam_client()
            )
        except (StackNotAppliedError, AccessDeniedReadingTheAccountError) as exc:
            metrics.append(
                _metric(
                    COMPARISON_ACCOUNT_VS_TEMPLATE,
                    status="unmeasurable",
                    status_reason=str(exc),
                    now=now,
                    source_path=f"{IAC_CONFORMANCE_MODULE}.audit_account_vs_template",
                )
            )
        else:
            metrics.append(
                _metric(
                    COMPARISON_ACCOUNT_VS_TEMPLATE,
                    status="OK" if account_audit.met else "BREACH",
                    status_reason=account_audit.detail(),
                    now=now,
                    source_path=f"{IAC_CONFORMANCE_MODULE}.audit_account_vs_template",
                    value=float(
                        len(account_audit.only_in_account) + len(account_audit.only_in_template)
                    ),
                    unit="count",
                )
            )
            readings["account_vs_template"] = account_audit.to_dict()
            if not account_audit.met:
                findings.append(COMPARISON_ACCOUNT_VS_TEMPLATE)

        try:
            template_inv = template_inventory_from_stack(stack=stack, cfn=_cfn_client())
            declared_inv = declared_inventory_from_ssm(
                parameter_name=cfg.iac_declared_inventory_param, ssm=_ssm_client()
            )
        except (
            StackNotAppliedError,
            TemplateUnreadableError,
            DeclaredInventoryUnsetError,
            DeclaredInventoryUnreadableError,
        ) as exc:
            metrics.append(
                _metric(
                    COMPARISON_TEMPLATE_VS_DECLARED,
                    status="unmeasurable",
                    status_reason=str(exc),
                    now=now,
                    source_path=f"{IAC_CONFORMANCE_MODULE}.audit_template_vs_declared",
                )
            )
        else:
            template_audit = audit_template_vs_declared(template_inv, declared_inv)
            metrics.append(
                _metric(
                    COMPARISON_TEMPLATE_VS_DECLARED,
                    status="OK" if template_audit.met else "BREACH",
                    status_reason=template_audit.detail(),
                    now=now,
                    source_path=f"{IAC_CONFORMANCE_MODULE}.audit_template_vs_declared",
                    value=float(
                        len(template_audit.roles_only_in_template)
                        + len(template_audit.roles_only_in_declared)
                    ),
                    unit="count",
                )
            )
            readings["template_vs_declared"] = template_audit.to_dict()
            if not template_audit.met:
                findings.append(COMPARISON_TEMPLATE_VS_DECLARED)

        prev_had_findings = _previous_week_had_findings(store, ctx.trading_day)
        this_week_has_findings = bool(findings)
        persisted = this_week_has_findings and prev_had_findings
        prev_day = _previous_trading_day_one_week_back(ctx.trading_day)
        prev_key = manifest_key(IAC_CONFORMANCE_JOB, prev_day.isoformat())
        persistence_reason = (
            f"findings this week: {findings or 'none'}; prior week had findings: "
            f"{prev_had_findings} (read from {prev_key}, absent/unreadable treated as "
            "not-yet-persisted, per this function's docstring)"
        )
        metrics.append(
            _metric(
                COMPARISON_PERSISTED,
                status="BREACH" if persisted else "OK",
                status_reason=persistence_reason,
                now=now,
                source_path=f"{IAC_CONFORMANCE_MODULE}.iac_conformance_handler",
                value=1.0 if persisted else 0.0,
                unit="bool",
            )
        )

        for metric in metrics:
            ctx.record_metric(metric)

        if readings and not dry_run:
            payload = json.dumps(readings, indent=2, sort_keys=True).encode("utf-8")
            ctx.record_output(
                iac_conformance_key(ctx.trading_day.isoformat()),
                payload,
                schema_version="iac_conformance.v1",
            )
        elif readings and dry_run:
            print(json.dumps(readings, indent=2, sort_keys=True))

        if persisted:
            raise IacConformanceDrift(
                f"IaC conformance finding(s) persisted two consecutive weekly cycles: "
                f"{', '.join(findings)}. See {IAC_CONFORMANCE_JOB}'s own metrics for detail."
            )

    run_job(
        IAC_CONFORMANCE_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    return 0
