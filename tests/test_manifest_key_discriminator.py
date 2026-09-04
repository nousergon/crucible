"""Contract test for `manifest_key`'s discriminator (alpha-engine-config-I9781).

Born alongside the fix, per the fleet's M0 discipline: a new key shape gets a
producer/consumer contract test at birth, not retrofitted once something
downstream already depends on the shape informally.

The bug this closes: `manifest_key(job, trading_day)` carried the job and the
trading day but not the slot, so `experiment.run --slot u` and `--slot r`
wrote the identical object — four slots, one manifest, last writer wins. Same
for `alerts.sweep`, which fires every calendar day and can collide with
itself across a weekend.
"""

from __future__ import annotations

import pytest

from crucible.manifest import manifest_key, manifest_prefix

TRADING_DAY = "2026-08-28"


class TestBackwardCompatibility:
    """Every existing caller passes no discriminator; the key shape for
    those callers must not move under them."""

    def test_no_discriminator_is_the_pre_i9781_shape(self) -> None:
        assert manifest_key("data.daily", TRADING_DAY) == "runs/data.daily/2026-08-28/run.json"

    def test_discriminator_none_is_identical_to_omitting_it(self) -> None:
        assert manifest_key("data.daily", TRADING_DAY) == manifest_key(
            "data.daily", TRADING_DAY, discriminator=None
        )


class TestDiscriminator:
    def test_two_slots_produce_two_distinct_keys(self) -> None:
        u = manifest_key("experiment.run", TRADING_DAY, discriminator="u")
        r = manifest_key("experiment.run", TRADING_DAY, discriminator="r")
        assert u != r
        assert u == "runs/experiment.run/2026-08-28/u/run.json"
        assert r == "runs/experiment.run/2026-08-28/r/run.json"

    def test_a_calendar_date_discriminator_separates_weekend_sweep_firings(self) -> None:
        friday = manifest_key("alerts.sweep", "2026-08-28", discriminator="2026-08-28")
        saturday = manifest_key("alerts.sweep", "2026-08-28", discriminator="2026-08-29")
        sunday = manifest_key("alerts.sweep", "2026-08-28", discriminator="2026-08-30")
        assert len({friday, saturday, sunday}) == 3

    def test_job_and_trading_day_still_validated_first(self) -> None:
        with pytest.raises(ValueError, match="job"):
            manifest_key("", TRADING_DAY, discriminator="u")
        with pytest.raises(ValueError, match="trading_day"):
            manifest_key("experiment.run", "", discriminator="u")

    @pytest.mark.parametrize("bad", ["", "u/r", "u r", "a" * 65])
    def test_a_discriminator_that_is_not_a_clean_path_segment_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="discriminator"):
            manifest_key("experiment.run", TRADING_DAY, discriminator=bad)

    def test_a_slash_in_the_discriminator_cannot_forge_extra_path_segments(self) -> None:
        with pytest.raises(ValueError):
            manifest_key("experiment.run", TRADING_DAY, discriminator="u/../../../etc")


class TestManifestPrefix:
    def test_prefix_is_the_common_root_of_bare_and_discriminated_keys(self) -> None:
        prefix = manifest_prefix("experiment.run", TRADING_DAY)
        assert prefix == "runs/experiment.run/2026-08-28/"
        assert manifest_key("experiment.run", TRADING_DAY).startswith(prefix)
        assert manifest_key("experiment.run", TRADING_DAY, discriminator="u").startswith(prefix)

    def test_prefix_does_not_leak_into_a_different_trading_day(self) -> None:
        prefix = manifest_prefix("experiment.run", TRADING_DAY)
        other_day_key = manifest_key("experiment.run", "2026-08-29", discriminator="u")
        assert not other_day_key.startswith(prefix)
