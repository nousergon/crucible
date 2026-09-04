"""Whether every v2 AWS resource carries the cost-attribution tag.

Normative source: plan §2 row 3 and §11 risk 7 — "`system=crucible-v2` tag
from phase 0 and a monthly cost row on the console with the target as its
ceiling". Without the tag the ≤ $40/mo ceiling has no denominator: the number
on the console would be the whole account's bill, and the ceiling would be
asserted rather than measured.

**The check is a DIFFERENCE, not a count.** Asking the tagging API "how many
resources carry the tag" answers a question nobody has: a stack that lost half
its resources still returns a positive number. This reads the stack's own
resource list and subtracts the tagged set, so the answer names the resources
that are missing the tag — which is the thing an operator acts on.

**An absent stack is UNMEASURABLE, never a pass.** Before the operator applies
the template there is nothing to tag, and an empty difference over an empty
stack is vacuous truth (principle 7). :class:`StackNotAppliedError` is raised
so the acceptance clause fails with the apply command as its remedy.

**Tags live behind more than one API, and reading one of them is the bug this
module was nearly an instance of.** The first revision resolved every resource
through `resourcegroupstaggingapi`, which **does not cover IAM at all**: run
live against the applied stack on 2026-09-01 it reported five correctly-tagged
roles as untagged, while `aws iam list-role-tags` returned
`system=crucible-v2` for every one of them. A checker that reads one of
several channels and reports the rest as absent is the fleet's own recurring
defect (`bugclass_a_checker_that_read_one_of_four_channels_260901`), and it
fails in the direction that looks like diligence. :data:`TAG_READERS` is the
per-type resolution table, and a type with no entry falls through to the
tagging API — so adding a service the API does not cover is a row here, not a
silent false positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "TAG_KEY",
    "TAG_READERS",
    "TAG_VALUE",
    "UNTAGGABLE_TYPES",
    "StackNotAppliedError",
    "TagAudit",
    "audit_stack_tags",
]

TAG_KEY = "system"
TAG_VALUE = "crucible-v2"

#: CloudFormation resource types that carry no tags at all, with the reason.
#: Declared rather than inferred from an empty tag list, because "this
#: resource has no tags" and "this resource cannot have tags" are the same
#: observation and only one of them is a finding.
UNTAGGABLE_TYPES: dict[str, str] = {
    # EventBridge Scheduler exposes NO `Tags` property on
    # `AWS::Scheduler::Schedule`, and `scheduler:ListTagsForResource` accepts
    # only a schedule-GROUP arn — measured 2026-09-01: passing a schedule arn
    # returns `ValidationException ... Member must satisfy ...schedule-group/`.
    # The group is the tagging surface, so the stack carries its own
    # `AWS::Scheduler::ScheduleGroup`, tagged, rather than putting schedules in
    # the account-wide `default` group and tagging that — which would label
    # every other system's schedules `crucible-v2` and corrupt the very
    # denominator this audit exists to establish.
    "AWS::Scheduler::Schedule": (
        "schedules carry no tags of their own; their schedule GROUP does, and it "
        "is a resource of this stack"
    ),
    # An instance profile is a container for a role; the role carries the tag
    # and the profile has no tag surface in IAM.
    "AWS::IAM::InstanceProfile": "IAM instance profiles have no tag surface",
    # Bucket policies, role policies and the like are properties of a tagged
    # parent, not resources with an identity of their own.
    "AWS::S3::BucketPolicy": "a bucket policy is a property of the tagged bucket",
    "AWS::IAM::Policy": "an inline policy is a property of the tagged role",
    "AWS::Lambda::Permission": "a permission is a property of the tagged function",
    "AWS::SNS::TopicPolicy": "a topic policy is a property of the tagged topic",
    # `AWS::SNS::Subscription` has no `Tags` property in CloudFormation and no
    # tagging API of its own; a subscription's cost is the topic's, and the
    # topic is tagged. Measured 2026-09-04: `nous-ergon-ops-PR1036` added the
    # pages topic's email and Lambda legs and the acceptance tag clause went
    # UNMET on both, taking the phase-0 `v2_resources_tagged_and_versioned`
    # clause with it — the audit had a rule for a permission and none for a
    # subscription, and the acceptance suite is the surface that noticed.
    "AWS::SNS::Subscription": "a subscription is a property of the tagged topic",
}


class StackNotAppliedError(RuntimeError):
    """The stack this audit reads does not exist yet."""


@dataclass(frozen=True)
class TagAudit:
    """What the stack contains, what is tagged, and what is not."""

    stack: str
    resources: tuple[tuple[str, str, str], ...]  # (logical id, type, physical id)
    untagged: tuple[tuple[str, str, str], ...]
    skipped: tuple[tuple[str, str], ...]  # (logical id, why it carries no tag)

    @property
    def met(self) -> bool:
        return not self.untagged

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack": self.stack,
            "tag": f"{TAG_KEY}={TAG_VALUE}",
            "resources": len(self.resources),
            "untagged": [
                {"logical_id": lid, "type": kind, "physical_id": pid}
                for lid, kind, pid in self.untagged
            ],
            "skipped": [{"logical_id": lid, "reason": why} for lid, why in self.skipped],
            "met": self.met,
        }

    def detail(self) -> str:
        if self.met:
            return (
                f"{len(self.resources)} taggable resources in {self.stack}, all carrying "
                f"{TAG_KEY}={TAG_VALUE} ({len(self.skipped)} untaggable by type)"
            )
        names = ", ".join(f"{lid} ({kind})" for lid, kind, _ in self.untagged[:5])
        return f"{len(self.untagged)} of {len(self.resources)} resources untagged: {names}"


def _iam_role_tagged(clients: dict[str, Any], physical_id: str) -> bool:
    """Whether an IAM role carries the tag, read from IAM itself.

    `resourcegroupstaggingapi` does not cover IAM, and it does not say so — it
    simply omits the resource, which is indistinguishable from an untagged one.
    """
    iam = clients.get("iam")
    if iam is None:
        raise ValueError(
            "auditing an AWS::IAM::Role needs an `iam` client: the tagging API does "
            "not cover IAM, so falling back to it would report every correctly "
            "tagged role as untagged."
        )
    tags = iam.list_role_tags(RoleName=physical_id).get("Tags", [])
    return any(t["Key"] == TAG_KEY and t["Value"] == TAG_VALUE for t in tags)


#: Per-type tag resolution. A type absent from this table resolves through
#: `resourcegroupstaggingapi`, which is right for most services and wrong for
#: the ones it does not cover.
TAG_READERS: dict[str, Any] = {
    "AWS::IAM::Role": _iam_role_tagged,
}


def _stack_resources(cfn: Any, stack: str) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    try:
        paginator = cfn.get_paginator("list_stack_resources")
        for page in paginator.paginate(StackName=stack):
            resources.extend(page.get("StackResourceSummaries", []))
    except Exception as exc:  # noqa: BLE001 - narrowed immediately below
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        message = str(exc)
        if code == "ValidationError" and "does not exist" in message:
            raise StackNotAppliedError(
                f"stack {stack!r} does not exist. The cost-attribution tag is applied by "
                "the CloudFormation deploy, so this clause is UNMEASURABLE until the "
                "operator has run it (nous-ergon-ops, `Crucible v2 Stack` workflow)."
            ) from exc
        raise
    if not resources:
        raise StackNotAppliedError(
            f"stack {stack!r} lists no resources. An empty stack would make this audit "
            "vacuously true, which is the one answer it must never give."
        )
    return resources


def _tagged_identifiers(tagging: Any) -> set[str]:
    """Every ARN carrying the tag, plus the last ARN segment of each.

    The trailing segment is included because a CloudFormation physical id is a
    name for most types and a full ARN for a few, so matching on either is
    what lets one comparison cover both without a per-type table that would go
    stale the first time a resource type is added.
    """
    found: set[str] = set()
    paginator = tagging.get_paginator("get_resources")
    for page in paginator.paginate(TagFilters=[{"Key": TAG_KEY, "Values": [TAG_VALUE]}]):
        for entry in page.get("ResourceTagMappingList", []):
            arn = entry["ResourceARN"]
            found.add(arn)
            found.add(arn.rsplit("/", 1)[-1])
            found.add(arn.rsplit(":", 1)[-1])
    return found


def audit_stack_tags(*, stack: str, cfn: Any, tagging: Any, iam: Any = None) -> TagAudit:
    """Which of ``stack``'s taggable resources lack `system=crucible-v2`."""
    summaries = _stack_resources(cfn, stack)
    tagged = _tagged_identifiers(tagging)
    clients = {"iam": iam}

    resources: list[tuple[str, str, str]] = []
    untagged: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    for summary in summaries:
        logical = summary["LogicalResourceId"]
        kind = summary["ResourceType"]
        physical = summary.get("PhysicalResourceId", "")
        if kind in UNTAGGABLE_TYPES:
            skipped.append((logical, UNTAGGABLE_TYPES[kind]))
            continue
        resources.append((logical, kind, physical))
        reader = TAG_READERS.get(kind)
        if reader is not None:
            if not reader(clients, physical):
                untagged.append((logical, kind, physical))
            continue
        if physical not in tagged and physical.rsplit("/", 1)[-1] not in tagged:
            untagged.append((logical, kind, physical))
    return TagAudit(
        stack=stack,
        resources=tuple(resources),
        untagged=tuple(untagged),
        skipped=tuple(skipped),
    )
