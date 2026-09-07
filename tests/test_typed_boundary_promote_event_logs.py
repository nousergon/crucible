"""`promote.py`'s two event logs as typed boundaries
(`alpha-engine-config-I10045` row 14, last row).

Normative source: `alpha-engine-config-I10045`; `crucible.models.
RetirementLogRow`/`ExperimentEventRow`'s docstrings. Ruled 2026-09-06: this
row is `crucible.promote`'s two OWN append-only JSONL logs —
`retirements/{slot}.jsonl` (`_append_retirement_events`) and
`experiments/{as_of}.jsonl` (`_append_experiment_events`) — distinct from
the row-2 arm register (`arms/{slot}/register.jsonl`, `ArmEvent`/
`ArmRegister`-shaped, library-owned, already carved out).

Both logs are validated on the WRITE side only, before
`crucible.promote._append_events` stamps each row's `event_id` — the same
producer-validates-on-the-way-out shape row 9's `trial_rows` takes.
`_read_register_events`/`_append_events`/`_append_lines`/`_event_id` (the
generic JSONL infrastructure) are UNCHANGED.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.models import (
    EXPERIMENT_EVENT_ROW_ADAPTER,
    EligibilityHoldEventRow,
    NoComparisonEventRow,
    RetirementLogRow,
)


def _retirement_row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "as_of": "2026-08-28",
        "slot": "u",
        "arm_id": "u:momentum_sleeve:abc123",
        "retire": False,
        "reason": "champion",
        "age_weeks": 10,
        "pairwise_losses": 0,
        "is_champion": True,
    }
    payload.update(overrides)
    return payload


class TestRetirementLogRow:
    def test_a_well_formed_row_validates(self) -> None:
        RetirementLogRow.model_validate(_retirement_row())

    def test_a_missing_field_names_it(self) -> None:
        row = _retirement_row()
        del row["pairwise_losses"]
        with pytest.raises(ValidationError, match="pairwise_losses"):
            RetirementLogRow.model_validate(row)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bogus_field"):
            RetirementLogRow.model_validate(_retirement_row(bogus_field="nope"))

    def test_a_wrongly_typed_age_weeks_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="age_weeks"):
            RetirementLogRow.model_validate(_retirement_row(age_weeks="ten"))


class TestExperimentEventRowUnion:
    def test_each_kind_validates_against_its_own_shape(self) -> None:
        no_comparison = EXPERIMENT_EVENT_ROW_ADAPTER.validate_python(
            {
                "kind": "no_comparison",
                "slot": "u",
                "as_of": "2026-08-28",
                "status": "held",
                "reason": "single-arm slot",
            }
        )
        assert isinstance(no_comparison, NoComparisonEventRow)

        hold = EXPERIMENT_EVENT_ROW_ADAPTER.validate_python(
            {
                "kind": "eligibility_hold",
                "slot": "u",
                "as_of": "2026-08-28",
                "reason": "not enough paired dates",
                "promote_min_weeks": 4,
                "paired_dates_required": 20,
            }
        )
        assert isinstance(hold, EligibilityHoldEventRow)

    def test_an_unrecognized_kind_is_refused_naming_the_legal_values(self) -> None:
        with pytest.raises(ValidationError, match="kind"):
            EXPERIMENT_EVENT_ROW_ADAPTER.validate_python(
                {"kind": "made_up_kind", "slot": "u", "as_of": "2026-08-28"}
            )

    def test_a_promotion_row_missing_its_arm_id_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="arm_id"):
            EXPERIMENT_EVENT_ROW_ADAPTER.validate_python(
                {
                    "kind": "promotion",
                    "slot": "u",
                    "as_of": "2026-08-28",
                    "incumbent": None,
                    "status": "decided",
                    "reason": "sole eligible arm",
                    "window": {},
                }
            )

    def test_a_negative_result_row_carrying_an_eligibility_hold_field_is_refused(self) -> None:
        """The discriminated union's whole point: a row cannot mix fields
        from two `kind` variants."""
        with pytest.raises(ValidationError):
            EXPERIMENT_EVENT_ROW_ADAPTER.validate_python(
                {
                    "kind": "negative_result",
                    "slot": "u",
                    "as_of": "2026-08-28",
                    "arm_id": "u:a:1",
                    "incumbent": "u:b:2",
                    "status": "decided",
                    "reason": "lost the pair",
                    "window": {},
                    "confidence_sequence": None,
                    "promote_min_weeks": 4,
                }
            )
