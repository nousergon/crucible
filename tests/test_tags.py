"""Every v2 resource carries `system=crucible-v2`, or the ceiling has no denominator.

Normative source: plan §2 row 3 and §11 risk 7. The check is a DIFFERENCE
between what the stack contains and what the tagging API returns, not a count
— a stack that lost half its resources still returns a positive count.
"""

from __future__ import annotations

import pytest

from crucible.tags import (
    TAG_KEY,
    TAG_VALUE,
    UNTAGGABLE_TYPES,
    CostAllocationTagUnreadableError,
    StackNotAppliedError,
    audit_stack_tags,
    cost_allocation_tag_status,
)


class _FakeIam:
    """IAM's own tag surface. `resourcegroupstaggingapi` does not cover IAM."""

    def __init__(self, tagged_roles: set[str]) -> None:
        self._tagged = tagged_roles

    def list_role_tags(self, *, RoleName: str) -> dict:  # noqa: N803 - boto3's shape
        if RoleName in self._tagged:
            return {"Tags": [{"Key": TAG_KEY, "Value": TAG_VALUE}]}
        return {"Tags": []}


STACK = "crucible-v2"
ROLE = "arn:aws:iam::123456789012:role/test-runtime"
FUNCTION = "arn:aws:lambda:us-east-1:123456789012:function:test-dispatcher"


class _FakeCfn:
    def __init__(self, resources: list[dict] | Exception) -> None:
        self._resources = resources

    def get_paginator(self, name: str):
        assert name == "list_stack_resources"
        resources = self._resources

        class _Paginator:
            def paginate(self, *, StackName: str):  # noqa: N803 - boto3's shape
                if isinstance(resources, Exception):
                    raise resources
                yield {"StackResourceSummaries": resources}

        return _Paginator()


class _FakeTagging:
    def __init__(self, arns: list[str]) -> None:
        self._arns = arns

    def get_paginator(self, name: str):
        assert name == "get_resources"
        arns = self._arns

        class _Paginator:
            def paginate(self, *, TagFilters):  # noqa: N803 - boto3's shape
                assert TagFilters == [{"Key": TAG_KEY, "Values": [TAG_VALUE]}]
                yield {"ResourceTagMappingList": [{"ResourceARN": a} for a in arns]}

        return _Paginator()


def _summary(logical: str, kind: str, physical: str) -> dict:
    return {"LogicalResourceId": logical, "ResourceType": kind, "PhysicalResourceId": physical}


class TestUnmeasurableIsNeverAPass:
    def test_an_absent_stack_raises_with_the_remedy(self) -> None:
        """Before the operator applies the template there is nothing to tag,
        and an empty difference over an empty stack is vacuous truth."""
        error = RuntimeError("Stack with id crucible-v2 does not exist")
        error.response = {"Error": {"Code": "ValidationError"}}  # type: ignore[attr-defined]
        with pytest.raises(StackNotAppliedError, match="Crucible v2 Stack"):
            audit_stack_tags(
                stack=STACK, cfn=_FakeCfn(error), tagging=_FakeTagging([]), iam=_FakeIam(set())
            )

    def test_a_stack_with_no_resources_raises_rather_than_passing(self) -> None:
        with pytest.raises(StackNotAppliedError, match="vacuously true"):
            audit_stack_tags(
                stack=STACK, cfn=_FakeCfn([]), tagging=_FakeTagging([]), iam=_FakeIam(set())
            )

    def test_an_unrelated_client_error_is_not_swallowed_as_not_applied(self) -> None:
        """An AccessDenied reported as 'the stack is not applied' would send
        an operator to run a deploy that was never the problem."""
        error = RuntimeError("User is not authorized to perform cloudformation:ListStackResources")
        error.response = {"Error": {"Code": "AccessDenied"}}  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="not authorized"):
            audit_stack_tags(
                stack=STACK, cfn=_FakeCfn(error), tagging=_FakeTagging([]), iam=_FakeIam(set())
            )


class TestTheDifference:
    def test_a_fully_tagged_stack_is_met(self) -> None:
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("RuntimeRole", "AWS::IAM::Role", "test-runtime"),
                    _summary("Dispatcher", "AWS::Lambda::Function", "test-dispatcher"),
                ]
            ),
            tagging=_FakeTagging([ROLE, FUNCTION]),
            iam=_FakeIam({"test-runtime"}),
        )
        assert audit.met
        assert len(audit.resources) == 2
        assert not audit.untagged

    def test_one_untagged_resource_fails_and_is_named(self) -> None:
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("RuntimeRole", "AWS::IAM::Role", "test-runtime"),
                    _summary("Dispatcher", "AWS::Lambda::Function", "test-dispatcher"),
                ]
            ),
            tagging=_FakeTagging([ROLE]),
            iam=_FakeIam({"test-runtime"}),
        )
        assert not audit.met
        assert [lid for lid, _, _ in audit.untagged] == ["Dispatcher"]
        assert "Dispatcher" in audit.detail()

    def test_a_count_would_have_passed_where_the_difference_fails(self) -> None:
        """The reason this is a difference. Two tagged resources exist, so any
        'how many carry the tag' check reads healthy while a resource of the
        stack is untagged."""
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("RuntimeRole", "AWS::IAM::Role", "test-runtime"),
                    _summary("Dispatcher", "AWS::Lambda::Function", "test-dispatcher"),
                    _summary("Bucket", "AWS::S3::Bucket", "test-releases-bucket"),
                ]
            ),
            tagging=_FakeTagging([ROLE, FUNCTION]),
            iam=_FakeIam({"test-runtime"}),
        )
        assert not audit.met
        assert [lid for lid, _, _ in audit.untagged] == ["Bucket"]

    def test_an_untaggable_type_is_skipped_with_its_reason(self) -> None:
        """'This resource has no tags' and 'this resource cannot have tags'
        are the same observation, and only one of them is a finding."""
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("RuntimeRole", "AWS::IAM::Role", "test-runtime"),
                    _summary("Profile", "AWS::IAM::InstanceProfile", "crucible-v2-profile"),
                ]
            ),
            tagging=_FakeTagging([ROLE]),
            iam=_FakeIam({"test-runtime"}),
        )
        assert audit.met
        assert audit.skipped == (("Profile", UNTAGGABLE_TYPES["AWS::IAM::InstanceProfile"]),)
        assert "untaggable by type" in audit.detail()

    def test_a_physical_id_that_is_a_bare_name_matches_its_arn(self) -> None:
        """CloudFormation gives a name for most types and an ARN for a few;
        one comparison covers both rather than a per-type table that goes
        stale the first time a resource type is added."""
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn([_summary("Topic", "AWS::SNS::Topic", "test-pages-topic")]),
            tagging=_FakeTagging(["arn:aws:sns:us-east-1:123456789012:test-pages-topic"]),
            iam=_FakeIam(set()),
        )
        assert audit.met

    def test_the_reading_serializes_with_what_is_missing(self) -> None:
        document = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn([_summary("Bucket", "AWS::S3::Bucket", "test-releases-bucket")]),
            tagging=_FakeTagging([]),
            iam=_FakeIam(set()),
        ).to_dict()
        assert document["met"] is False
        assert document["tag"] == f"{TAG_KEY}={TAG_VALUE}"
        assert document["untagged"][0]["logical_id"] == "Bucket"


class TestEveryChannelIsRead:
    """The false positive this class exists for.

    Run live against the applied stack on 2026-09-01, the first revision of
    this module reported five correctly-tagged IAM roles as untagged —
    `resourcegroupstaggingapi` does not cover IAM and does not say so, it
    simply omits the resource, which is indistinguishable from an untagged one.
    `aws iam list-role-tags` returned `system=crucible-v2` for every one.
    """

    def test_a_role_tagged_in_iam_is_not_reported_untagged(self) -> None:
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn([_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")]),
            # The tagging API returns NOTHING for IAM — exactly as it does live.
            tagging=_FakeTagging([]),
            iam=_FakeIam({"test-runtime"}),
        )
        assert audit.met, audit.detail()

    def test_a_role_genuinely_untagged_is_still_caught(self) -> None:
        """The fix must not turn the check off. Reading the right channel and
        finding nothing is a finding; reading the wrong one is not."""
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn([_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")]),
            tagging=_FakeTagging([]),
            iam=_FakeIam(set()),
        )
        assert not audit.met
        assert [lid for lid, _, _ in audit.untagged] == ["RuntimeRole"]

    def test_auditing_a_role_without_an_iam_client_raises(self) -> None:
        """Never a silent fallback to the API that cannot see IAM. That
        fallback IS the defect, and a caller that forgot the client would get
        the same confident wrong answer as before."""
        with pytest.raises(ValueError, match="does not cover IAM"):
            audit_stack_tags(
                stack=STACK,
                cfn=_FakeCfn([_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")]),
                tagging=_FakeTagging([]),
            )

    def test_a_scheduler_schedule_is_untaggable_by_type(self) -> None:
        """`AWS::Scheduler::Schedule` exposes no `Tags` property, and
        `scheduler:ListTagsForResource` accepts only a schedule-GROUP arn —
        measured: a schedule arn returns `ValidationException`. The group is
        the tagging surface and is a resource of the stack."""
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [_summary("WeeklySchedule", "AWS::Scheduler::Schedule", "test-weekly-schedule")]
            ),
            tagging=_FakeTagging([]),
            iam=_FakeIam(set()),
        )
        assert audit.met
        assert audit.skipped[0][0] == "WeeklySchedule"
        assert "GROUP" in UNTAGGABLE_TYPES["AWS::Scheduler::Schedule"]

    def test_an_sns_subscription_is_untaggable_by_type(self) -> None:
        """`AWS::SNS::Subscription` has no `Tags` property and no tagging API;
        the topic carries the tag. Measured 2026-09-04: the pages topic's two
        new legs (`nous-ergon-ops-PR1036`) read UNTAGGED under this audit and
        took the phase-0 clause down with them. A subscription is skipped
        with its reason; the TOPIC is still graded."""
        topic = "arn:aws:sns:us-east-1:000000000000:test-pages-topic"
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("PagesTopic", "AWS::SNS::Topic", topic),
                    _summary("PagesEmailSubscription", "AWS::SNS::Subscription", f"{topic}:1"),
                ]
            ),
            tagging=_FakeTagging([topic]),
            iam=_FakeIam(set()),
        )
        assert audit.met, audit.untagged
        assert audit.skipped == (
            ("PagesEmailSubscription", UNTAGGABLE_TYPES["AWS::SNS::Subscription"]),
        )
        untagged_topic = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn([_summary("PagesTopic", "AWS::SNS::Topic", topic)]),
            tagging=_FakeTagging([]),
            iam=_FakeIam(set()),
        )
        assert not untagged_topic.met


class _FakeCe:
    """`ce:ListCostAllocationTags`'s own shape — distinct from Cost Explorer's
    `get_cost_and_usage`, and asserted on the request to catch a caller
    filtering on the wrong key."""

    def __init__(self, tags: list[dict] | Exception) -> None:
        self._tags = tags

    def list_cost_allocation_tags(self, *, TagKeys: list[str]):  # noqa: N803
        assert TagKeys == [TAG_KEY]
        if isinstance(self._tags, Exception):
            raise self._tags
        return {"CostAllocationTags": self._tags}


class TestCostAllocationTagStatus:
    """`alpha-engine-config-I10076`: the `system` key was `Inactive` in
    Billing while every resource carried it, and Cost Explorer's tag filter
    read a genuine `$0.00` forever. This is the reading that names the
    activation state `audit_stack_tags` cannot answer."""

    def test_active_is_reported_with_its_date(self) -> None:
        status = cost_allocation_tag_status(
            _FakeCe([{"TagKey": TAG_KEY, "Status": "Active", "LastUpdatedDate": "2026-09-06"}])
        )
        assert status.active
        assert status.status == "Active"
        assert status.last_updated_date == "2026-09-06"

    def test_inactive_is_reported_not_active(self) -> None:
        status = cost_allocation_tag_status(
            _FakeCe([{"TagKey": TAG_KEY, "Status": "Inactive", "LastUpdatedDate": "2026-09-01"}])
        )
        assert not status.active
        assert status.status == "Inactive"

    def test_a_key_billing_has_never_seen_is_absent_not_inactive(self) -> None:
        """`Inactive` and `absent` share a remedy (activate it) but are
        different observations — a key present with a status and a key never
        returned at all are not the same finding."""
        status = cost_allocation_tag_status(_FakeCe([]))
        assert not status.active
        assert status.status == "absent"
        assert status.last_updated_date is None

    def test_a_denied_call_raises_naming_the_action_not_inactive(self) -> None:
        """The whole reason this raises rather than returning `Inactive`: a
        denied `ce:ListCostAllocationTags` and a genuinely deactivated key are
        different findings with different remedies."""
        error = RuntimeError("User is not authorized to perform ce:ListCostAllocationTags")
        with pytest.raises(CostAllocationTagUnreadableError, match="ce:ListCostAllocationTags"):
            cost_allocation_tag_status(_FakeCe(error))

    def test_serializes_with_status_and_date(self) -> None:
        status = cost_allocation_tag_status(
            _FakeCe([{"TagKey": TAG_KEY, "Status": "Active", "LastUpdatedDate": "2026-09-06"}])
        )
        assert status.to_dict() == {"status": "Active", "last_updated_date": "2026-09-06"}
