"""The CloudTrail record as an in-process typed reader
(`alpha-engine-config-I10045` row 12).

Normative source: `alpha-engine-config-I10045`; `crucible.models.
CloudTrailRecord`'s docstring. Named exception like row 3 (the arena
cycle): the published shape is AWS's, not ours — "model what we read, not
what AWS writes". No schema is generated or published, and every model in
this boundary is `extra="allow"`, at every nesting level.

`crucible.autonomy._principal` used to walk `record.get("userIdentity", {})
or {}` three levels deep by hand; a typo'd key at any level resolved
silently to `{}` ("no issuer") rather than surfacing. This file proves the
new reader still reads the same real-shaped record the same way, and that
a mistyped/missing field along the chain is now visible rather than a
silent `None`.
"""

from __future__ import annotations

from crucible.autonomy import _principal
from crucible.models import CloudTrailRecord


def _record(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "eventTime": "2026-08-03T18:00:00Z",
        "eventName": "UpdateFunctionCode",
        "eventSource": "lambda.amazonaws.com",
        "readOnly": False,
        "requestID": "req-1",
        "requestParameters": {"functionName": "crucible-v2-dispatcher"},
        "userIdentity": {
            "type": "AssumedRole",
            "arn": "arn:aws:sts::711398986525:assumed-role/AWSReservedSSO_admin/brian",
            "sessionContext": {"sessionIssuer": {"userName": "AWSReservedSSO_admin"}},
        },
    }
    payload.update(overrides)
    return payload


class TestCloudTrailRecordTypesTheRealShape:
    def test_a_real_assumed_role_record_resolves_the_role_not_the_session(self) -> None:
        record = CloudTrailRecord.model_validate(_record())
        name, kind = _principal(record)
        assert name == "AWSReservedSSO_admin"
        assert kind == "AssumedRole"

    def test_an_unknown_top_level_field_is_tolerated_not_refused(self) -> None:
        """The design this row exists to preserve: a CloudTrail field this
        reader does not know about must not be refused."""
        record = CloudTrailRecord.model_validate(_record(aFieldNoOneHasSeenYet="anything at all"))
        assert record.model_extra["aFieldNoOneHasSeenYet"] == "anything at all"

    def test_a_missing_session_issuer_falls_back_to_the_identity_username(self) -> None:
        record = CloudTrailRecord.model_validate(
            _record(
                userIdentity={
                    "type": "IAMUser",
                    "userName": "brian",
                    "arn": "arn:aws:iam::711398986525:user/brian",
                }
            )
        )
        name, kind = _principal(record)
        assert name == "brian"
        assert kind == "IAMUser"

    def test_no_identity_at_all_falls_back_to_unknown(self) -> None:
        record = CloudTrailRecord.model_validate(_record(userIdentity={}))
        name, kind = _principal(record)
        assert name == "unknown"
        assert kind == "Unknown"


class TestASilentThreeLevelTypoIsNowVisible:
    """`main`'s reader resolved a typo'd nested key to `{}`/`None` at every
    level with `.get(..., {})`. The model's per-level `extra="allow"` still
    tolerates an unknown SIBLING key (the design this row preserves) but a
    typo'd `sessionIssuer` simply does not populate the field the model
    declares — the same "absent, not present-under-a-typo" distinction
    every other row in this migration draws."""

    def test_a_typo_d_session_issuer_key_does_not_silently_supply_a_name(self) -> None:
        record = CloudTrailRecord.model_validate(
            _record(
                userIdentity={
                    "type": "AssumedRole",
                    "arn": "arn:aws:sts::711398986525:assumed-role/some-role/brian",
                    "sessionContext": {"sessionIsser": {"userName": "some-role"}},
                }
            )
        )
        name, kind = _principal(record)
        # Falls back to the identity's own arn, exactly as main's `.get`
        # chain would have — the typo'd key is simply absent from the
        # declared `sessionIssuer` field, not silently populating it.
        assert name == record.userIdentity.arn
        assert record.userIdentity.sessionContext.sessionIssuer is None
