"""Two independent IaC conformance comparisons (`alpha-engine-config-I10418`).

(a) account vs template — live tagged resources vs the deployed stack's
declared taggable resources, both directions.
(b) template vs the plan's declared inventory — the fixture in
`TestComparisonBCatchesAPlanTemplateDisagreement` is the proof this repo's
own bug (`§9.8` said "two roles" while the template declared 15) would have
been CAUGHT: comparison (a) alone renders GREEN against a template that
agrees with itself, and only (b) can see the plan disagree with it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from crucible.documents import read_store_document
from crucible.iac_conformance import (
    COMPARISON_ACCOUNT_VS_TEMPLATE,
    COMPARISON_PERSISTED,
    COMPARISON_TEMPLATE_VS_DECLARED,
    IAC_CONFORMANCE_JOB,
    AccessDeniedReadingTheAccountError,
    DeclaredInventory,
    DeclaredInventoryUnreadableError,
    DeclaredInventoryUnsetError,
    IacConformanceDrift,
    TemplateInventory,
    TemplateUnreadableError,
    audit_account_vs_template,
    audit_template_vs_declared,
    declared_inventory_from_ssm,
    iac_conformance_handler,
    template_inventory_from_stack,
)
from crucible.keys import manifest_key
from crucible.store import LocalStore
from crucible.tags import TAG_KEY, TAG_VALUE, StackNotAppliedError

STACK = "crucible-v2"
FRIDAY = dt.date(2026, 8, 28)  # a trading day, matches tests/test_weekly.py
PRIOR_FRIDAY = dt.date(2026, 8, 21)  # exactly one trading week earlier


def _summary(logical: str, kind: str, physical: str) -> dict:
    return {"LogicalResourceId": logical, "ResourceType": kind, "PhysicalResourceId": physical}


class _FakeCfn:
    def __init__(
        self, resources: list[dict] | Exception, template: dict | str | None = None
    ) -> None:
        self._resources = resources
        self._template = template

    def get_paginator(self, name: str):
        assert name == "list_stack_resources"
        resources = self._resources

        class _Paginator:
            def paginate(self, *, StackName: str):  # noqa: N803 - boto3's shape
                if isinstance(resources, Exception):
                    raise resources
                yield {"StackResourceSummaries": resources}

        return _Paginator()

    def get_template(self, *, StackName: str):  # noqa: N803 - boto3's shape
        if isinstance(self._template, Exception):
            raise self._template
        return {"TemplateBody": self._template}


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


class _FakeIam:
    """Covers both `list_role_tags` (crucible.tags' per-role read) and
    `list_roles` (this module's account-wide scan, which `crucible.tags`
    never needs)."""

    def __init__(self, roles: dict[str, bool]) -> None:
        # role_name -> tagged?
        self._roles = roles

    def list_role_tags(self, *, RoleName: str) -> dict:  # noqa: N803 - boto3's shape
        if self._roles.get(RoleName):
            return {"Tags": [{"Key": TAG_KEY, "Value": TAG_VALUE}]}
        return {"Tags": []}

    def get_paginator(self, name: str):
        assert name == "list_roles"
        roles = self._roles

        class _Paginator:
            def paginate(self):
                yield {"Roles": [{"RoleName": name} for name in roles]}

        return _Paginator()


class _DenyingIam(_FakeIam):
    """`list_role_tags` denied on the first role outside this stack's prefix.

    The live shape, measured 2026-09-11 on the v2 box: the grant was
    scoped `role/crucible-v2-*` while `_tagged_iam_role_names` reads tags on
    every role `list_roles` returns, so the very first unowned role — `admin`
    — was denied.
    """

    def __init__(self, roles: dict[str, bool], *, denied: str) -> None:
        super().__init__(roles)
        self._denied = denied

    def list_role_tags(self, *, RoleName: str) -> dict:  # noqa: N803 - boto3's shape
        if RoleName == self._denied:
            raise _client_error(
                "AccessDenied",
                f"User: <the v2 runtime role, on the v2 box> is not authorized to "
                f"perform: iam:ListRoleTags on resource: role {RoleName}",
            )
        return super().list_role_tags(RoleName=RoleName)


def _client_error(code: str, message: str) -> Exception:
    """A botocore-shaped error without importing botocore: the production code
    reads `exc.response['Error']['Code']` and nothing else about the type, and
    a test that imported the real class would be asserting botocore's
    constructor rather than our own branch."""

    class _ClientError(Exception):
        def __init__(self) -> None:
            super().__init__(f"An error occurred ({code}): {message}")
            self.response = {"Error": {"Code": code, "Message": message}}

    return _ClientError()


class _FakeSsm:
    def __init__(self, value: str | Exception | None) -> None:
        self._value = value

    def get_parameter(self, *, Name: str) -> dict:  # noqa: N803 - boto3's shape
        if isinstance(self._value, Exception):
            raise self._value
        if self._value is None:
            return {"Parameter": {}}
        return {"Parameter": {"Value": self._value}}


TEMPLATE_TWO_ROLES = {
    "Resources": {
        "DeployRole": {"Type": "AWS::IAM::Role"},
        "RuntimeRole": {"Type": "AWS::IAM::Role"},
        "Bucket": {"Type": "AWS::S3::Bucket"},
    }
}

TEMPLATE_FIFTEEN_ROLES = {
    "Resources": {
        # The original two the plan declared, PLUS thirteen more — this is
        # the actual shape of the bug the issue exists for: the template
        # grew past the plan's stated objective without breaking it.
        "DeployRole": {"Type": "AWS::IAM::Role"},
        "RuntimeRole": {"Type": "AWS::IAM::Role"},
        **{f"ExtraRole{i}": {"Type": "AWS::IAM::Role"} for i in range(13)},
        "Bucket": {"Type": "AWS::S3::Bucket"},
        "Dispatcher": {
            "Type": "AWS::Lambda::Function",
            "Properties": {"Handler": {"Fn::Sub": "${SomeVar}.handler"}},
        },
    }
}


# ── comparison (a): account vs template ──────────────────────────────────


class TestAccountVsTemplate:
    def test_a_fully_matched_estate_is_met(self) -> None:
        audit = audit_account_vs_template(
            stack=STACK,
            cfn=_FakeCfn([_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")]),
            tagging=_FakeTagging([]),
            iam=_FakeIam({"test-runtime": True}),
        )
        assert audit.met
        assert not audit.only_in_account
        assert not audit.only_in_template

    def test_a_resource_tagged_but_not_stack_managed_is_only_in_account(self) -> None:
        """An out-of-band resource carrying the tag: the direction
        `crucible.tags.audit_stack_tags` alone cannot see, because it only
        ever iterates the stack's OWN resource list."""
        audit = audit_account_vs_template(
            stack=STACK,
            cfn=_FakeCfn([_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")]),
            tagging=_FakeTagging(["arn:aws:sns:us-east-1:123456789012:rogue-topic"]),
            iam=_FakeIam({"test-runtime": True}),
        )
        assert not audit.met
        assert "rogue-topic" in audit.only_in_account
        assert not audit.only_in_template

    def test_an_out_of_band_iam_role_is_also_only_in_account(self) -> None:
        """IAM is invisible to the tagging API — this is the account-wide
        `iam:ListRoles` scan's own reason to exist."""
        audit = audit_account_vs_template(
            stack=STACK,
            cfn=_FakeCfn([_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")]),
            tagging=_FakeTagging([]),
            iam=_FakeIam({"test-runtime": True, "rogue-role": True}),
        )
        assert not audit.met
        assert "rogue-role" in audit.only_in_account

    def test_a_stack_declared_resource_that_is_not_tagged_live_is_only_in_template(self) -> None:
        """A stack update that silently failed or rolled back — the stack
        still declares the resource, but the tag scan cannot find it."""
        audit = audit_account_vs_template(
            stack=STACK,
            cfn=_FakeCfn(
                [
                    _summary("RuntimeRole", "AWS::IAM::Role", "test-runtime"),
                    _summary("Dispatcher", "AWS::Lambda::Function", "test-dispatcher"),
                ]
            ),
            tagging=_FakeTagging([]),
            iam=_FakeIam({"test-runtime": True}),
        )
        assert not audit.met
        assert [lid for lid, _, _ in audit.only_in_template] == ["Dispatcher"]

    def test_an_absent_stack_is_unmeasurable_never_a_pass(self) -> None:
        error = RuntimeError("Stack with id crucible-v2 does not exist")
        error.response = {"Error": {"Code": "ValidationError"}}  # type: ignore[attr-defined]
        with pytest.raises(StackNotAppliedError):
            audit_account_vs_template(
                stack=STACK, cfn=_FakeCfn(error), tagging=_FakeTagging([]), iam=_FakeIam({})
            )


# ── comparison (b): template vs the plan's declared inventory ───────────


class TestAnAccessFailureIsNeverAnArcFailure:
    """The class defect behind two lost weekly arcs.

    `iac.conformance` is a `dispatch: arc` stage, and `crucible.weekly.run_arc`
    stops the arc on the first stage that raises. Comparison (b) and the
    cost-allocation read both already render an access failure UNMEASURABLE
    and say so in the same words ("a statement about our access"). Comparison
    (a) did not — it let a raw `ClientError` out — so a missing grant did not
    cost a metric, it cost the whole Saturday:

      2026-09-10 (first v2-box rehearsal): cloudformation:ListStackResources
      2026-09-11 (the next one): iam:ListRoleTags on role `admin`

    -- two consecutive rehearsals of 2026-07-31, one denial further along each
    time. The grants are fixed where they live (`nous-ergon-ops`
    `infrastructure/cloudformation/crucible-v2.yaml`); these tests are what
    make a THIRD missing grant cost one red metric instead of an arc.
    """

    def test_a_denied_role_tag_read_is_declared_not_raw(self) -> None:
        with pytest.raises(AccessDeniedReadingTheAccountError) as caught:
            audit_account_vs_template(
                stack=STACK,
                cfn=_FakeCfn([_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")]),
                tagging=_FakeTagging([]),
                iam=_DenyingIam({"test-runtime": True, "admin": False}, denied="admin"),
            )
        assert "statement about our grants" in str(caught.value)

    def test_a_denied_stack_resource_read_is_declared_too(self) -> None:
        """The FIRST of the two denials, from the same call path one layer up."""
        with pytest.raises(AccessDeniedReadingTheAccountError):
            audit_account_vs_template(
                stack=STACK,
                cfn=_FakeCfn(_client_error("AccessDenied", "not authorized: ListStackResources")),
                tagging=_FakeTagging([]),
                iam=_FakeIam({}),
            )

    def test_a_non_access_error_still_propagates(self) -> None:
        """Fail loud. Only AccessDenied is a statement about our grants; every
        other failure is a statement about the estate and must not be laundered
        into UNMEASURABLE."""
        boom = _client_error("ThrottlingException", "slow down")
        with pytest.raises(Exception) as caught:
            audit_account_vs_template(
                stack=STACK,
                cfn=_FakeCfn(boom),
                tagging=_FakeTagging([]),
                iam=_FakeIam({}),
            )
        assert not isinstance(caught.value, AccessDeniedReadingTheAccountError)

    def test_the_stack_not_existing_keeps_its_own_answer(self) -> None:
        """ "Cannot look" and "nothing to look at" are different answers and
        must not collapse into one."""
        missing = _client_error("ValidationError", f"Stack with id {STACK} does not exist")
        with pytest.raises(StackNotAppliedError):
            audit_account_vs_template(
                stack=STACK,
                cfn=_FakeCfn(missing),
                tagging=_FakeTagging([]),
                iam=_FakeIam({}),
            )


class TestTemplateInventoryFromStack:
    def test_counts_resources_and_names_iam_roles(self) -> None:
        inv = template_inventory_from_stack(stack=STACK, cfn=_FakeCfn([], TEMPLATE_TWO_ROLES))
        assert inv.resource_count == 3
        assert inv.iam_role_logical_ids == ("DeployRole", "RuntimeRole")

    def test_tolerates_cfn_intrinsic_tags(self) -> None:
        """`!Sub`/`!GetAtt`/etc. must not make the loader choke — only the
        plain `Type` scalars this module reads matter."""
        inv = template_inventory_from_stack(stack=STACK, cfn=_FakeCfn([], TEMPLATE_FIFTEEN_ROLES))
        assert inv.resource_count == 17
        assert inv.iam_role_count == 15

    def test_a_yaml_string_template_body_parses_the_same_as_a_dict(self) -> None:
        text = (
            "Resources:\n"
            "  DeployRole:\n"
            "    Type: AWS::IAM::Role\n"
            "  Bucket:\n"
            "    Type: AWS::S3::Bucket\n"
        )
        inv = template_inventory_from_stack(stack=STACK, cfn=_FakeCfn([], text))
        assert inv.resource_count == 2
        assert inv.iam_role_logical_ids == ("DeployRole",)

    def test_an_empty_resources_section_raises_rather_than_reading_as_zero(self) -> None:
        with pytest.raises(TemplateUnreadableError, match="Resources"):
            template_inventory_from_stack(stack=STACK, cfn=_FakeCfn([], {"Resources": {}}))

    def test_an_absent_stack_is_unmeasurable(self) -> None:
        error = RuntimeError("Stack with id crucible-v2 does not exist")
        error.response = {"Error": {"Code": "ValidationError"}}  # type: ignore[attr-defined]
        with pytest.raises(StackNotAppliedError):
            template_inventory_from_stack(stack=STACK, cfn=_FakeCfn([], error))


class TestDeclaredInventoryFromSsm:
    def test_an_unset_parameter_name_refuses_rather_than_guessing(self) -> None:
        with pytest.raises(DeclaredInventoryUnsetError):
            declared_inventory_from_ssm(parameter_name="", ssm=_FakeSsm(None))

    def test_a_valid_parameter_parses(self) -> None:
        value = json.dumps({"resource_count": 34, "iam_roles": ["DeployRole", "RuntimeRole"]})
        inv = declared_inventory_from_ssm(
            parameter_name="/crucible-v2/iac/declared-inventory", ssm=_FakeSsm(value)
        )
        assert inv.resource_count == 34
        assert inv.iam_role_logical_ids == ("DeployRole", "RuntimeRole")
        assert inv.source == "ssm:/crucible-v2/iac/declared-inventory"

    def test_malformed_json_is_unreadable_not_zero(self) -> None:
        with pytest.raises(DeclaredInventoryUnreadableError):
            declared_inventory_from_ssm(parameter_name="/x", ssm=_FakeSsm("not json"))

    def test_a_denied_read_is_unreadable_not_zero(self) -> None:
        with pytest.raises(DeclaredInventoryUnreadableError):
            declared_inventory_from_ssm(
                parameter_name="/x", ssm=_FakeSsm(RuntimeError("AccessDeniedException"))
            )

    def test_a_missing_value_is_unreadable(self) -> None:
        with pytest.raises(DeclaredInventoryUnreadableError):
            declared_inventory_from_ssm(parameter_name="/x", ssm=_FakeSsm(None))


class TestComparisonBCatchesAPlanTemplateDisagreement:
    """The fixture proving (b) does what (a) structurally cannot: this repo's
    own measured bug, reproduced. The plan's declared inventory says "two IAM
    roles only"; the template has grown to fifteen. Comparison (a), run
    against ONLY this template, is vacuously agreeable with it — the template
    always agrees with the live stack it was used to create. Only (b), which
    reads the declared inventory from an INDEPENDENT source, can see the
    disagreement."""

    def test_a_template_that_outgrew_the_declared_two_roles_is_not_met(self) -> None:
        template = template_inventory_from_stack(
            stack=STACK, cfn=_FakeCfn([], TEMPLATE_FIFTEEN_ROLES)
        )
        declared = DeclaredInventory(
            resource_count=3, iam_role_logical_ids=("DeployRole", "RuntimeRole"), source="ssm:/x"
        )
        audit = audit_template_vs_declared(template, declared)
        assert not audit.met
        assert audit.template.iam_role_count == 15
        assert audit.declared.iam_role_count == 2
        # DeployRole/RuntimeRole are in both sets, so only the thirteen
        # ExtraRoles are "only in template" — the exact shape of the real
        # bug: the template grew PAST the plan's stated objective without
        # ever contradicting it on the two roles the plan did name.
        assert len(audit.roles_only_in_template) == 13
        assert not audit.roles_only_in_declared

    def test_an_agreeing_template_and_declared_inventory_is_met(self) -> None:
        template = template_inventory_from_stack(stack=STACK, cfn=_FakeCfn([], TEMPLATE_TWO_ROLES))
        declared = DeclaredInventory(
            resource_count=3, iam_role_logical_ids=("DeployRole", "RuntimeRole"), source="ssm:/x"
        )
        audit = audit_template_vs_declared(template, declared)
        assert audit.met

    def test_a_role_the_plan_declares_but_the_template_dropped_is_the_other_direction(self) -> None:
        """Symmetric: the plan's inventory naming a role the template no
        longer has is exactly as much a finding as the reverse."""
        template = TemplateInventory(resource_count=2, iam_role_logical_ids=("DeployRole",))
        declared = DeclaredInventory(
            resource_count=2, iam_role_logical_ids=("DeployRole", "RetiredRole"), source="ssm:/x"
        )
        audit = audit_template_vs_declared(template, declared)
        assert not audit.met
        assert audit.roles_only_in_declared == ("RetiredRole",)
        assert not audit.roles_only_in_template


# ── the weekly job: both readings, persistence-gated paging ─────────────


def _args(tmp_path, *, trading_day: dt.date, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        job=IAC_CONFORMANCE_JOB,
        store=str(tmp_path),
        dry_run=dry_run,
        date=trading_day.isoformat(),
        trading_day=trading_day,
        run_mode="replay",
    )


def _patch_clients(monkeypatch, *, cfn, tagging, iam, ssm) -> None:
    import crucible.iac_conformance as mod

    monkeypatch.setattr(mod, "_cfn_client", lambda: cfn)
    monkeypatch.setattr(mod, "_tagging_client", lambda: tagging)
    monkeypatch.setattr(mod, "_iam_client", lambda: iam)
    monkeypatch.setattr(mod, "_ssm_client", lambda: ssm)


class TestTheWeeklyJob:
    def test_a_first_observed_finding_does_not_page(self, tmp_path, monkeypatch) -> None:
        """Rule 2: a job that found a discrepancy still produced a complete,
        correct READING — that is `ok`, not `failed`. Only a SECOND
        consecutive week of the same picture raises."""
        monkeypatch.setenv(
            "CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM", "/crucible-v2/iac/declared-inventory"
        )
        _patch_clients(
            monkeypatch,
            cfn=_FakeCfn(
                [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
            ),
            tagging=_FakeTagging(["arn:aws:sns:us-east-1:123456789012:rogue-topic"]),
            iam=_FakeIam({"test-runtime": True}),
            ssm=_FakeSsm(
                json.dumps({"resource_count": 3, "iam_roles": ["DeployRole", "RuntimeRole"]})
            ),
        )
        code = iac_conformance_handler(_args(tmp_path, trading_day=FRIDAY))
        assert code == 0
        store = LocalStore(tmp_path)
        read = read_store_document(store, manifest_key(IAC_CONFORMANCE_JOB, FRIDAY.isoformat()))
        assert read.document is not None
        assert read.document["status"] == "ok"
        names = {m["name"]: m["status"] for m in read.document["metrics"]}
        assert names[COMPARISON_ACCOUNT_VS_TEMPLATE] == "BREACH"
        assert names[COMPARISON_TEMPLATE_VS_DECLARED] == "OK"
        assert names[COMPARISON_PERSISTED] == "OK"

    def test_a_missing_grant_costs_one_metric_not_the_whole_arc(
        self, tmp_path, monkeypatch
    ) -> None:
        """The property that makes this an arc-safe stage.

        The handler exits 0 with comparison (a) UNMEASURABLE, so
        `crucible.weekly.run_arc` — which stops on the first stage that raises
        — runs the stages after it. UNMEASURABLE is never a pass: the metric is
        not `OK`, it names the denied action, and the clause reading it stays
        unmet until the grant lands.
        """
        monkeypatch.setenv(
            "CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM", "/crucible-v2/iac/declared-inventory"
        )
        _patch_clients(
            monkeypatch,
            cfn=_FakeCfn(
                [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
            ),
            tagging=_FakeTagging([]),
            iam=_DenyingIam({"test-runtime": True, "admin": False}, denied="admin"),
            ssm=_FakeSsm(
                json.dumps({"resource_count": 3, "iam_roles": ["DeployRole", "RuntimeRole"]})
            ),
        )
        code = iac_conformance_handler(_args(tmp_path, trading_day=FRIDAY))
        assert code == 0, "a missing grant must not exit non-zero and stop the arc"
        store = LocalStore(tmp_path)
        read = read_store_document(store, manifest_key(IAC_CONFORMANCE_JOB, FRIDAY.isoformat()))
        assert read.document is not None
        assert read.document["status"] == "ok"
        by_name = {m["name"]: m for m in read.document["metrics"]}
        account = by_name[COMPARISON_ACCOUNT_VS_TEMPLATE]
        assert account["status"] == "unmeasurable"
        assert account["status"] != "OK"
        assert "iam:ListRoleTags" in account["status_reason"], (
            "the metric must name the denied action, or the operator cannot fix it"
        )
        # Comparison (b) reads a different client and is unaffected — a denial
        # in (a) must not be reported as though it darkened both.
        assert by_name[COMPARISON_TEMPLATE_VS_DECLARED]["status"] == "OK"

    def test_a_finding_persisting_a_second_week_pages(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(
            "CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM", "/crucible-v2/iac/declared-inventory"
        )
        cfn = _FakeCfn(
            [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
        )
        tagging = _FakeTagging(["arn:aws:sns:us-east-1:123456789012:rogue-topic"])
        iam = _FakeIam({"test-runtime": True})
        ssm = _FakeSsm(
            json.dumps({"resource_count": 3, "iam_roles": ["DeployRole", "RuntimeRole"]})
        )
        _patch_clients(monkeypatch, cfn=cfn, tagging=tagging, iam=iam, ssm=ssm)

        # Week 1: first observation, does not raise.
        iac_conformance_handler(_args(tmp_path, trading_day=PRIOR_FRIDAY))

        # Week 2: the SAME account-level drift is still there.
        with pytest.raises(IacConformanceDrift):
            iac_conformance_handler(_args(tmp_path, trading_day=FRIDAY))

        store = LocalStore(tmp_path)
        read = read_store_document(store, manifest_key(IAC_CONFORMANCE_JOB, FRIDAY.isoformat()))
        assert read.document is not None
        assert read.document["status"] == "failed"
        names = {m["name"]: m["status"] for m in read.document["metrics"]}
        assert names[COMPARISON_PERSISTED] == "BREACH"

    def test_a_clean_week_after_a_dirty_one_does_not_page(self, tmp_path, monkeypatch) -> None:
        """The healed case: week 1 has a finding, week 2 does not — the
        drift resolved itself (a completed stack update) and must not page."""
        monkeypatch.setenv(
            "CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM", "/crucible-v2/iac/declared-inventory"
        )
        declared = json.dumps({"resource_count": 3, "iam_roles": ["DeployRole", "RuntimeRole"]})

        _patch_clients(
            monkeypatch,
            cfn=_FakeCfn(
                [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
            ),
            tagging=_FakeTagging(["arn:aws:sns:us-east-1:123456789012:rogue-topic"]),
            iam=_FakeIam({"test-runtime": True}),
            ssm=_FakeSsm(declared),
        )
        iac_conformance_handler(_args(tmp_path, trading_day=PRIOR_FRIDAY))

        # Week 2: the rogue resource is gone; the estate now matches.
        _patch_clients(
            monkeypatch,
            cfn=_FakeCfn(
                [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
            ),
            tagging=_FakeTagging([]),
            iam=_FakeIam({"test-runtime": True}),
            ssm=_FakeSsm(declared),
        )
        code = iac_conformance_handler(_args(tmp_path, trading_day=FRIDAY))
        assert code == 0
        store = LocalStore(tmp_path)
        read = read_store_document(store, manifest_key(IAC_CONFORMANCE_JOB, FRIDAY.isoformat()))
        assert read.document["status"] == "ok"

    def test_an_unset_declared_inventory_param_is_unmeasurable_not_a_finding(
        self, tmp_path, monkeypatch
    ) -> None:
        """The operator has not set the SSM parameter yet — comparison (b)
        must read UNMEASURABLE, never OK (nothing to compare) or BREACH
        (nothing to disagree with)."""
        monkeypatch.delenv("CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM", raising=False)
        _patch_clients(
            monkeypatch,
            cfn=_FakeCfn(
                [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
            ),
            tagging=_FakeTagging([]),
            iam=_FakeIam({"test-runtime": True}),
            ssm=_FakeSsm(None),
        )
        code = iac_conformance_handler(_args(tmp_path, trading_day=FRIDAY))
        assert code == 0
        store = LocalStore(tmp_path)
        read = read_store_document(store, manifest_key(IAC_CONFORMANCE_JOB, FRIDAY.isoformat()))
        names = {m["name"]: m["status"] for m in read.document["metrics"]}
        assert names[COMPARISON_TEMPLATE_VS_DECLARED] == "unmeasurable"

    def test_dry_run_computes_readings_but_writes_no_output_key(
        self, tmp_path, monkeypatch
    ) -> None:
        """`--dry-run` suppresses the store WRITE, never the AWS reads
        themselves — the comparisons still run against the (fake, here)
        clients, and the manifest carries no `iac/{day}/conformance.json`
        output entry."""
        from crucible.keys import iac_conformance_key

        monkeypatch.setenv("CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM", "/x")
        _patch_clients(
            monkeypatch,
            cfn=_FakeCfn(
                [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
            ),
            tagging=_FakeTagging([]),
            iam=_FakeIam({"test-runtime": True}),
            ssm=_FakeSsm(
                json.dumps({"resource_count": 3, "iam_roles": ["DeployRole", "RuntimeRole"]})
            ),
        )
        code = iac_conformance_handler(_args(tmp_path, trading_day=FRIDAY, dry_run=True))
        assert code == 0
        store = LocalStore(tmp_path)
        assert not store.exists(iac_conformance_key(FRIDAY.isoformat()))
        assert not store.exists(manifest_key(IAC_CONFORMANCE_JOB, FRIDAY.isoformat()))

    def test_an_absent_previous_manifest_is_not_treated_as_persisted(
        self, tmp_path, monkeypatch
    ) -> None:
        """The very first Saturday this job ever runs: no prior manifest
        exists at all, so even a real finding must not page."""
        monkeypatch.setenv("CRUCIBLE_IAC_DECLARED_INVENTORY_PARAM", "/x")
        _patch_clients(
            monkeypatch,
            cfn=_FakeCfn(
                [_summary("RuntimeRole", "AWS::IAM::Role", "test-runtime")], TEMPLATE_TWO_ROLES
            ),
            tagging=_FakeTagging(["arn:aws:sns:us-east-1:123456789012:rogue-topic"]),
            iam=_FakeIam({"test-runtime": True}),
            ssm=_FakeSsm(
                json.dumps({"resource_count": 3, "iam_roles": ["DeployRole", "RuntimeRole"]})
            ),
        )
        code = iac_conformance_handler(_args(tmp_path, trading_day=FRIDAY))
        assert code == 0
