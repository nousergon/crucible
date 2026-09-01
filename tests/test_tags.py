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
    StackNotAppliedError,
    audit_stack_tags,
)

STACK = "crucible-v2"
ROLE = "arn:aws:iam::711398986525:role/crucible-v2-runtime"
FUNCTION = "arn:aws:lambda:us-east-1:711398986525:function:crucible-v2-dispatcher"


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
            audit_stack_tags(stack=STACK, cfn=_FakeCfn(error), tagging=_FakeTagging([]))

    def test_a_stack_with_no_resources_raises_rather_than_passing(self) -> None:
        with pytest.raises(StackNotAppliedError, match="vacuously true"):
            audit_stack_tags(stack=STACK, cfn=_FakeCfn([]), tagging=_FakeTagging([]))

    def test_an_unrelated_client_error_is_not_swallowed_as_not_applied(self) -> None:
        """An AccessDenied reported as 'the stack is not applied' would send
        an operator to run a deploy that was never the problem."""
        error = RuntimeError("User is not authorized to perform cloudformation:ListStackResources")
        error.response = {"Error": {"Code": "AccessDenied"}}  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="not authorized"):
            audit_stack_tags(stack=STACK, cfn=_FakeCfn(error), tagging=_FakeTagging([]))


class TestTheDifference:
    def test_a_fully_tagged_stack_is_met(self) -> None:
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("RuntimeRole", "AWS::IAM::Role", "crucible-v2-runtime"),
                    _summary("Dispatcher", "AWS::Lambda::Function", "crucible-v2-dispatcher"),
                ]
            ),
            tagging=_FakeTagging([ROLE, FUNCTION]),
        )
        assert audit.met
        assert len(audit.resources) == 2
        assert not audit.untagged

    def test_one_untagged_resource_fails_and_is_named(self) -> None:
        audit = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("RuntimeRole", "AWS::IAM::Role", "crucible-v2-runtime"),
                    _summary("Dispatcher", "AWS::Lambda::Function", "crucible-v2-dispatcher"),
                ]
            ),
            tagging=_FakeTagging([ROLE]),
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
                    _summary("RuntimeRole", "AWS::IAM::Role", "crucible-v2-runtime"),
                    _summary("Dispatcher", "AWS::Lambda::Function", "crucible-v2-dispatcher"),
                    _summary("Bucket", "AWS::S3::Bucket", "crucible-v2-releases"),
                ]
            ),
            tagging=_FakeTagging([ROLE, FUNCTION]),
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
                    _summary("RuntimeRole", "AWS::IAM::Role", "crucible-v2-runtime"),
                    _summary("Profile", "AWS::IAM::InstanceProfile", "crucible-v2-profile"),
                ]
            ),
            tagging=_FakeTagging([ROLE]),
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
            cfn=_FakeCfn([_summary("Topic", "AWS::SNS::Topic", "crucible-v2-pages")]),
            tagging=_FakeTagging(["arn:aws:sns:us-east-1:711398986525:crucible-v2-pages"]),
        )
        assert audit.met

    def test_the_reading_serializes_with_what_is_missing(self) -> None:
        document = audit_stack_tags(
            stack=STACK,
            cfn=_FakeCfn([_summary("Bucket", "AWS::S3::Bucket", "crucible-v2-releases")]),
            tagging=_FakeTagging([]),
        ).to_dict()
        assert document["met"] is False
        assert document["tag"] == f"{TAG_KEY}={TAG_VALUE}"
        assert document["untagged"][0]["logical_id"] == "Bucket"
