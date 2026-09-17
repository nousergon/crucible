"""An on-demand run's own calendar day is answerable from `runs/` alone.

`alpha-engine-config-I10999`.

Measured 2026-09-17: the seven arm registrations of that day ran
`experiment.new --trading-day 2026-09-11`, so every manifest landed under
`crucible/runs/experiment.new/2026-09-11/` carrying `calendar_date:
2026-09-17`. `runs/experiment.new/2026-09-17/` did not exist, and "what
mutated the store on 2026-09-17" had no answer in `runs/` at all — recovering
the true day took S3 object-version `LastModified` archaeology.

The decision and its rationale live at the key-construction site
(`crucible.keys.calendar_day_discriminator` and the block above it). These
are its tests, including the one that pins the ASYMMETRY: a job whose absence
`crucible.alerts` grades keeps its key shape exactly.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.keys import (
    calendar_day_discriminator,
    manifest_calendar_day,
    manifest_key,
    manifest_prefix,
    manifest_ran_on,
)

#: The measured case. A Friday session, registered on the following Thursday.
TRADING_DAY = dt.date(2026, 9, 11)
RAN_ON = dt.date(2026, 9, 17)


class TestTheDiscriminator:
    def test_it_leads_with_the_calendar_day(self) -> None:
        assert calendar_day_discriminator(RAN_ON) == "2026-09-17"

    def test_a_suffix_is_appended_never_replaced(self) -> None:
        """`experiment.new`'s `{slot}-{run_id}` is what keeps seven
        registrations on one day from overwriting each other
        (`alpha-engine-config-I10967`). This adds an axis; it must not take
        that one away."""
        assert (
            calendar_day_discriminator(RAN_ON, suffix="u-01M2R98M9FMCVPPFQRXHV23FKF")
            == "2026-09-17-u-01M2R98M9FMCVPPFQRXHV23FKF"
        )

    def test_the_key_it_builds_is_a_legal_manifest_key(self) -> None:
        key = manifest_key(
            "experiment.new",
            TRADING_DAY.isoformat(),
            discriminator=calendar_day_discriminator(RAN_ON, suffix="u-0" * 1),
        )
        assert key == "runs/experiment.new/2026-09-11/2026-09-17-u-0/run.json"

    def test_an_unusable_suffix_is_refused_where_that_refusal_lives(self) -> None:
        """Not re-validated here: `manifest_key` is the one place a
        discriminator is checked against the path-segment charset, and a
        second copy of that check is the one that drifts."""
        with pytest.raises(ValueError, match="path segment"):
            manifest_key(
                "experiment.new",
                TRADING_DAY.isoformat(),
                discriminator=calendar_day_discriminator(RAN_ON, suffix="slot/u"),
            )


class TestReadingTheCalendarDayBackOffTheKey:
    def test_the_measured_case_is_now_answerable(self) -> None:
        key = manifest_key(
            "experiment.new",
            TRADING_DAY.isoformat(),
            discriminator=calendar_day_discriminator(RAN_ON, suffix="u-ABC"),
        )
        assert manifest_calendar_day(key) == RAN_ON

    def test_a_bare_calendar_date_discriminator_reads_too(self) -> None:
        """`alerts.sweep` and `report.morning` already discriminate by
        `ctx.calendar_date`, and `data.daily`'s holiday no-op by the
        wall-clock date, all for this same "when did this actually fire"
        purpose. One predicate answers for all of them rather than one per
        writer."""
        key = manifest_key("alerts.sweep", "2026-09-11", discriminator="2026-09-12")
        assert manifest_calendar_day(key) == dt.date(2026, 9, 12)

    def test_a_slot_discriminator_states_no_calendar_day(self) -> None:
        key = manifest_key("experiment.run", "2026-09-11", discriminator="u")
        assert manifest_calendar_day(key) is None

    def test_a_bare_manifest_key_states_no_calendar_day(self) -> None:
        assert manifest_calendar_day(manifest_key("data.daily", "2026-09-11")) is None

    def test_a_date_shaped_prefix_that_is_not_a_whole_segment_is_not_read(self) -> None:
        """`2026-09-11x` is not a calendar day followed by a suffix. Reading
        its first ten characters as one would invent a fact."""
        key = manifest_key("experiment.new", "2026-09-11", discriminator="2026-09-11x")
        assert manifest_calendar_day(key) is None

    def test_a_date_shaped_segment_that_is_not_a_date_is_not_read(self) -> None:
        key = manifest_key("experiment.new", "2026-09-11", discriminator="2026-13-45-u")
        assert manifest_calendar_day(key) is None

    def test_a_foreign_key_is_not_a_manifest_key(self) -> None:
        assert manifest_calendar_day("features/2026-09-11.json") is None
        assert manifest_calendar_day("runs/experiment.new/2026-09-11/evidence.json") is None


class TestWhatMutatedTheStoreOnADay:
    """The question the issue says `runs/` could not answer."""

    def _keys(self) -> list[str]:
        return [
            # the measured case: registered on the 17th, about the 11th
            manifest_key(
                "experiment.new",
                TRADING_DAY.isoformat(),
                discriminator=calendar_day_discriminator(RAN_ON, suffix="u-ABC"),
            ),
            manifest_key(
                "experiment.new",
                TRADING_DAY.isoformat(),
                discriminator=calendar_day_discriminator(RAN_ON, suffix="r-DEF"),
            ),
            # an ordinary scheduled run, keyed by the session it is about and
            # stating no calendar day
            manifest_key("data.daily", "2026-09-17"),
            manifest_key("experiment.run", "2026-09-11", discriminator="u"),
        ]

    def test_the_two_registrations_are_discoverable_under_the_day_they_ran(self) -> None:
        answered = [k for k in self._keys() if manifest_ran_on(k, RAN_ON)]
        assert answered == [
            "runs/experiment.new/2026-09-11/2026-09-17-u-ABC/run.json",
            "runs/experiment.new/2026-09-11/2026-09-17-r-DEF/run.json",
            "runs/data.daily/2026-09-17/run.json",
        ]

    def test_they_are_still_discoverable_under_the_day_they_are_about(self) -> None:
        """Both, not either: the trading-day prefix is unchanged, which is
        what keeps every trading-day-axis reader working."""
        prefix = manifest_prefix("experiment.new", TRADING_DAY.isoformat())
        assert all(key.startswith(prefix) for key in self._keys()[:2])

    def test_a_key_stating_no_calendar_day_answers_for_its_trading_day(self) -> None:
        assert manifest_ran_on(
            manifest_key("experiment.run", "2026-09-11", discriminator="u"), TRADING_DAY
        )

    def test_a_foreign_key_answers_for_no_day(self) -> None:
        assert not manifest_ran_on("features/2026-09-11.json", TRADING_DAY)


class TestTheAsymmetryIsReal:
    def test_a_scheduled_jobs_key_shape_is_untouched(self) -> None:
        """The binding constraint: `crucible.alerts` grades absence off a
        predictable manifest key per trading day. Change the key for a job
        with a `schedule`/`deadline` and the absence checker reads nothing
        where the job in fact ran — blinded in the direction that never pages.

        Derived from `components.yaml` itself rather than from a list written
        here, so a job that GAINS a schedule is covered the day it does.
        """
        from crucible.alerts import scheduled_components

        scheduled = scheduled_components()
        assert scheduled, "components.yaml declares no scheduled job — this has gone blind"
        for job in scheduled:
            assert manifest_key(job, "2026-09-11") == f"runs/{job}/2026-09-11/run.json"

    def test_experiment_new_is_the_on_demand_case_components_yaml_declares(self) -> None:
        """The premise the whole asymmetry rests on, asserted against
        `components.yaml` rather than quoted from the issue."""
        from crucible.alerts import scheduled_components

        assert "experiment.new" not in scheduled_components()


class TestEndToEndThroughTheCli:
    def test_experiment_new_with_a_trading_day_files_under_the_day_it_ran(
        self, tmp_path, monkeypatch, strategy_dir
    ) -> None:
        """The measured case, run for real: `--date` three sessions in the
        past (the flag the issue calls `--trading-day`; `crucible.cli`
        resolves `--date` onto `args.trading_day`), on a calendar day of its
        own.

        The clock is pinned rather than read: a test whose subject moves with
        the wall clock stops testing the same thing, and this one is entirely
        about which day is which.
        """
        import crucible.runner as runner_module
        from crucible.cli import main
        from crucible.store import LocalStore

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)

        real_datetime = dt.datetime
        pinned = dt.datetime(2026, 9, 17, 0, 23, 41, tzinfo=dt.UTC)

        class _PinnedDatetime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return pinned if tz is None else pinned.astimezone(tz)

        monkeypatch.setattr(runner_module.dt, "datetime", _PinnedDatetime)

        store_root = tmp_path / "store"
        assert (
            main(
                [
                    "experiment.new",
                    "--slot",
                    "u",
                    "--arm",
                    "momentum_sleeve",
                    "--date",
                    TRADING_DAY.isoformat(),
                    "--store",
                    str(store_root),
                    "--strategy-dir",
                    str(strategy_dir),
                    "--run-mode",
                    "replay",
                ]
            )
            == 0
        )

        manifests = [
            key for key in LocalStore(store_root).list_keys("runs/") if key.endswith("/run.json")
        ]
        assert len(manifests) == 1
        (key,) = manifests
        # Under the day it is ABOUT ...
        assert key.startswith(manifest_prefix("experiment.new", TRADING_DAY.isoformat()))
        # ... and answerable for the day it RAN on, from the key alone.
        assert manifest_calendar_day(key) == RAN_ON
        assert manifest_ran_on(key, RAN_ON)
        assert not manifest_ran_on(key, TRADING_DAY)
