"""Two typed document boundaries from `alpha-engine-config-I9847` wave 2:
the release pointer (`crucible.models.ReleasePointerDocument`) and the v2
dispatcher's absence-input record (`crucible.models.DispatchRecordDocument`).

Neither carries a published JSON Schema — the same shape row 9 (the trial
ledger) and the LLM callsite registry's callsite rows take, not row 1's
(`RunManifestV2`) or row 2's (`ArmRecipeDocument`) — so there is no
byte-identity test here. What this file proves is the reader-boundary claim
itself: a malformed document fails with the field named, at the read, rather
than three frames later at the first place the caller dereferenced it.

Integration coverage for both — a real document written to a `LocalStore`
and read back through the actual consumer — lives beside each consumer:
`tests/test_release.py::TestPointer` (`crucible.release.read_pointer`),
`tests/test_deploy.py::TestTheSmokeGate` (`crucible.track_c`'s smoke),
`tests/test_alerts_dispatch_absence.py` (`crucible.alerts.
evaluate_dispatch_absence`), `tests/test_alerts_min_active_arms.py` and
`tests/test_report.py` (the arena-cycle reuse), and
`tests/test_console.py` (the champion-pointer reuse).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.models import DispatchRecordDocument, ReleasePointerDocument

SHA = "a" * 40


class TestReleasePointerDocument:
    def test_a_conforming_pointer_validates(self) -> None:
        pointer = ReleasePointerDocument.model_validate(
            {"sha": SHA, "target": "current", "pinned_at": "2026-06-01T00:00:00Z"}
        )
        assert pointer.sha == SHA
        assert pointer.target == "current"

    def test_the_trader_pin_target_is_accepted(self) -> None:
        pointer = ReleasePointerDocument.model_validate(
            {"sha": SHA, "target": "trader", "pinned_at": "2026-06-01T00:00:00Z"}
        )
        assert pointer.target == "trader"

    @pytest.mark.parametrize(
        "payload",
        [
            {"target": "current", "pinned_at": "2026-06-01T00:00:00Z"},  # sha missing
            {"sha": "not-a-sha", "target": "current", "pinned_at": "2026-06-01T00:00:00Z"},
            {"sha": SHA, "target": "not-a-target", "pinned_at": "2026-06-01T00:00:00Z"},
            {"sha": SHA, "target": "current", "pinned_at": "not-a-timestamp"},
            {"sha": SHA, "target": "current"},  # pinned_at missing
        ],
    )
    def test_a_malformed_pointer_refuses_named(self, payload: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            ReleasePointerDocument.model_validate(payload)

    def test_an_extra_key_refuses(self) -> None:
        """`extra="forbid"`: this document has exactly one job."""
        with pytest.raises(ValidationError, match="extra"):
            ReleasePointerDocument.model_validate(
                {
                    "sha": SHA,
                    "target": "current",
                    "pinned_at": "2026-06-01T00:00:00Z",
                    "note": "oops",
                }
            )


class TestDispatchRecordDocument:
    def test_a_conforming_record_validates(self) -> None:
        record = DispatchRecordDocument.model_validate(
            {
                "schema_version": "dispatch_record.v1",
                "job": "data.heal",
                "args": "--from 2025-01-21 --to 2025-01-21",
                "instance_id": "i-0cb52a780eb7eb90c",
                "requested_by": "test-dispatcher",
                "dispatched_at_utc": "2026-08-28T04:03:00Z",
            }
        )
        assert record.job == "data.heal"
        assert record.instance_id == "i-0cb52a780eb7eb90c"

    def test_every_field_is_optional_or_defaulted(self) -> None:
        """A partially-written dispatch record must never be the reason the
        absence page itself fails to fire."""
        record = DispatchRecordDocument.model_validate({})
        assert record.job is None
        assert record.args == ""
        assert record.instance_id is None
        assert record.dispatched_at_utc is None

    def test_an_unknown_field_is_allowed(self) -> None:
        """Unlike `components.yaml`, this document is written by
        infrastructure outside this repository — `extra="forbid"` would
        refuse a record the moment the dispatcher adds a field this reader
        has no opinion about."""
        record = DispatchRecordDocument.model_validate({"job": "data.heal", "new_field": 1})
        assert record.job == "data.heal"

    def test_a_field_present_with_the_wrong_type_refuses_named(self) -> None:
        """The gap this migration closes: `args` written as a list (not the
        string every real dispatcher writes) used to reach
        `args_synthetic_marker` unnamed; it now refuses at the boundary."""
        with pytest.raises(ValidationError, match="args"):
            DispatchRecordDocument.model_validate({"args": ["--from", "x"]})
