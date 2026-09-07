"""The trial ledger row as a producer-validated boundary
(`alpha-engine-config-I10045` row 9).

Normative source: `alpha-engine-config-I10045`; `crucible.models.TrialRow`'s
docstring. **Named exception, like row 3 (the arena cycle):** the module
docstring `crucible.ledger` already carries states the design directly —
"the row has no JSON Schema, every reader ignores unknown keys" — so there
is no `crucible/schemas/trial*.json` and no byte-identity test.
`crucible.ledger.trial_rows` (the WRITER) validates each row's core fields
through `TrialRow` before returning them; `read_trials`/`append_trials`/
`n_trials` are unchanged and still work with plain dicts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from crucible.ledger import trial_rows
from crucible.models import TrialRow


def _cycle(**overrides: object) -> SimpleNamespace:
    defaults = dict(
        scored_arms=frozenset({"u:momentum_sleeve:abc123"}),
        active_arms=frozenset({"u:momentum_sleeve:abc123"}),
        benchmark="population",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _series(**overrides: object) -> SimpleNamespace:
    defaults = dict(
        scores={"2026-08-24": 0.01, "2026-08-25": 0.02},
        lineage={"feature_version": ("v1abc",)},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestTrialRowsValidatesEachRowOnTheWayOut:
    def test_a_well_formed_row_validates_and_round_trips(self) -> None:
        cycle = _cycle()
        [row] = trial_rows(
            cycle,
            slot="u",
            as_of="2026-08-28",
            run_id="01JG0000000000000000000000",
            control_ids=set(),
            series_by_arm={"u:momentum_sleeve:abc123": _series()},
            arena_cycle_key="arena_cycle/u/2026-08-28.json",
        )
        # Still a plain dict — read_trials/append_trials are unchanged.
        assert isinstance(row, dict)
        assert row["slot"] == "u"
        assert row["lineage"] == {"feature_version": ["v1abc"]}
        # And it validates directly against the model too.
        TrialRow.model_validate(row)

    def test_extra_keys_on_a_row_are_tolerated_by_the_model(self) -> None:
        """The design this row exists to preserve: an older or newer row
        carrying a key this reader does not know must not be refused."""
        cycle = _cycle()
        [row] = trial_rows(
            cycle,
            slot="u",
            as_of="2026-08-28",
            run_id="01JG0000000000000000000000",
            control_ids=set(),
            series_by_arm={"u:momentum_sleeve:abc123": _series()},
            arena_cycle_key="arena_cycle/u/2026-08-28.json",
        )
        row["a_future_field_this_reader_does_not_know"] = "anything"
        TrialRow.model_validate(row)  # must not raise

    def test_a_malformed_as_of_is_refused_at_the_writer(self) -> None:
        cycle = _cycle()
        with pytest.raises(ValueError, match="does not conform"):
            trial_rows(
                cycle,
                slot="u",
                as_of="not-a-date",
                run_id="01JG0000000000000000000000",
                control_ids=set(),
                series_by_arm={"u:momentum_sleeve:abc123": _series()},
                arena_cycle_key="arena_cycle/u/2026-08-28.json",
            )

    def test_an_empty_slot_is_refused_at_the_writer(self) -> None:
        cycle = _cycle()
        with pytest.raises(ValueError, match="does not conform"):
            trial_rows(
                cycle,
                slot="",
                as_of="2026-08-28",
                run_id="01JG0000000000000000000000",
                control_ids=set(),
                series_by_arm={"u:momentum_sleeve:abc123": _series()},
                arena_cycle_key="arena_cycle/u/2026-08-28.json",
            )
