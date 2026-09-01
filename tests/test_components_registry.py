"""`crucible/components.yaml` is the observability registry, and it is derived.

Normative source: plan §9.2 — "Both declared in one `components.yaml` — the
registry coverage is derived from that file, never hand-listed."

The failure this prevents: a job ships, runs, and has no declared log
location, alert channel, console surface or retention. It is then not
healthy, it is unobserved — and nothing anywhere says so, because the list of
things to check was written by hand from the jobs someone remembered.

So coverage is computed from the CLI's own job table against the registry, in
both directions. Written before `components.yaml` and seen failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from crucible.cli import JOBS
from crucible.manifest import SCHEMA_PATH

COMPONENTS_PATH = Path(__file__).resolve().parents[1] / "crucible" / "components.yaml"

REQUIRED_KEYS = {
    "description",
    "signals",
    "log_location",
    "log_retention_days",
    "alert_channel",
    "console_surface",
    "artifact_retention",
    "schedule",
    "deadline",
}

# §9.2: five signal classes, every component. `outcome` is legitimately
# absent from a job that grades nothing, but it is then absent BY DECLARATION
# (an explicit `null`), never by omission.
SIGNAL_CLASSES = {"execution", "cost", "resource", "lineage", "outcome"}


@pytest.fixture(scope="module")
def components() -> dict:
    return yaml.safe_load(COMPONENTS_PATH.read_text(encoding="utf-8"))


class TestCoverage:
    def test_every_cli_job_has_a_registry_row(self, components: dict) -> None:
        """Derived, not hand-listed: the expectation comes from the CLI's own
        job table, so a new job cannot be added without a row."""
        missing = sorted(set(JOBS) - set(components["components"]))
        assert not missing, (
            f"jobs with no components.yaml row: {missing}. A component with no "
            "declared log location, alert channel or retention is unobserved, and "
            "'no data' must never render as green (principle 7)."
        )

    def test_every_registry_row_is_a_real_job(self, components: dict) -> None:
        """The other direction. A row for a job that no longer exists makes
        the registry look complete while covering something that never runs —
        and its absence deadline can never fire, so it is a permanently
        silent monitor."""
        extra = sorted(set(components["components"]) - set(JOBS))
        assert not extra, f"components.yaml rows with no CLI job: {extra}"


class TestRowShape:
    def test_every_row_declares_every_required_key(self, components: dict) -> None:
        for name, row in components["components"].items():
            assert REQUIRED_KEYS <= set(row), (
                f"{name} is missing {sorted(REQUIRED_KEYS - set(row))}"
            )

    def test_every_row_declares_all_five_signal_classes(self, components: dict) -> None:
        """Explicitly, including the ones it does not emit. An omitted class
        is indistinguishable from a forgotten one."""
        for name, row in components["components"].items():
            assert set(row["signals"]) == SIGNAL_CLASSES, (
                f"{name} declares {sorted(row['signals'])}; all five §9.2 classes "
                "must appear, with `null` for one the job legitimately does not emit."
            )

    def test_a_scheduled_job_declares_a_deadline(self, components: dict) -> None:
        """§4.6: the ABSENCE page reads a deadline table. A scheduled job with
        no deadline can never be reported missing — the exact blindness that
        let a weekly pipeline not really run for two weeks."""
        for name, row in components["components"].items():
            if row["schedule"] is not None:
                assert row["deadline"], f"{name} is scheduled but declares no deadline"

    def test_an_unscheduled_job_declares_no_deadline(self, components: dict) -> None:
        """The converse, so a deadline is never quietly attached to an
        on-demand command whose absence means nothing."""
        for name, row in components["components"].items():
            if row["schedule"] is None:
                assert row["deadline"] is None, (
                    f"{name} is on-demand but declares a deadline; its absence is not "
                    "a fact about the system and would page for nothing."
                )

    def test_retention_is_declared_in_the_unit_the_source_bills_in(self, components: dict) -> None:
        """§4.12's exhaustive exception list: log retention is CALENDAR days,
        because that is an AWS property. Manifests are `forever`. Both are
        declared rather than inferred."""
        for name, row in components["components"].items():
            assert isinstance(row["log_retention_days"], int), name
            assert row["artifact_retention"] in ("forever", "90d", "365d"), (
                f"{name} declares artifact_retention={row['artifact_retention']!r}"
            )


class TestVocabulary:
    def test_disabled_and_retired_are_declared_never_inferred(self, components: dict) -> None:
        """§9.2: the console classifies a job from its manifests into a closed
        vocabulary, but DISABLED and RETIRED are declared here. A job inferred
        to be retired because it stopped producing is indistinguishable from
        one that broke."""
        for name, row in components["components"].items():
            assert row.get("lifecycle", "ACTIVE") in ("ACTIVE", "DISABLED", "RETIRED"), name

    def test_the_manifest_schema_job_enum_matches_the_registry(self, components: dict) -> None:
        """Third direction: the schema's `job` enum, the CLI's table and the
        registry are one closed set stated in three files, so any two
        disagreeing is caught rather than discovered.

        `deploy` is in the schema enum and is not a CLI job — the deploy
        workflow writes a manifest in the same schema so §4.5's page shows
        deploys beside runs (§4.11). It is the only permitted difference, and
        naming it here is what keeps it from becoming a general exemption."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        enum = set(schema["properties"]["job"]["enum"])
        assert enum - set(components["components"]) == {"deploy"}


class TestDeadlinesAreMachineReadable:
    """§4.6 reads the deadline table; a prose deadline needs its parse written
    twice, and a contract restated twice has already drifted."""

    def test_every_deadline_is_structured_not_prose(self, components: dict) -> None:
        for name, row in components["components"].items():
            if row["deadline"] is None:
                continue
            assert isinstance(row["deadline"], dict), (
                f"{name}'s deadline is {row['deadline']!r}. A deadline the alerter has "
                "to parse out of English is a second declaration of the same contract."
            )

    def test_every_anchor_is_in_the_closed_set(self, components: dict) -> None:
        from crucible.components import ANCHORS

        for name, row in components["components"].items():
            if row["deadline"]:
                assert row["deadline"]["anchor"] in ANCHORS, name

    def test_the_sentence_is_rendered_from_the_structure(self) -> None:
        """Not stored beside it. The prose is a projection of the data, so
        the two cannot disagree."""
        import datetime as dt

        from crucible.components import load_registry

        deadline = load_registry()["data.daily"].deadline
        assert deadline is not None
        assert deadline.describe(dt.date(2026, 8, 28)) == (
            "3h after the close of trading day 2026-08-28"
        )

    def test_a_deadline_resolves_against_the_trading_calendar(self) -> None:
        """A Monday holiday moves the deadline rather than producing a page
        that has to be dismissed."""
        import datetime as dt

        from crucible.calendar import NonTradingDayKeyError
        from crucible.components import load_registry

        deadline = load_registry()["data.daily"].deadline
        assert deadline is not None
        due = deadline.due_at(dt.date(2026, 8, 28))
        assert due == dt.datetime(2026, 8, 28, 23, 0, tzinfo=dt.UTC)  # 19:00 ET
        with pytest.raises(NonTradingDayKeyError):
            deadline.due_at(dt.date(2026, 8, 29))  # a Saturday


class TestNothingWatchesItself:
    """Detection blindness outranks the defects it hides."""

    def test_no_component_is_its_own_absence_watcher(self, components: dict) -> None:
        for name, row in components["components"].items():
            assert row.get("absence_watched_by", "alerts.sweep") != name, (
                f"{name} declares itself its own absence watcher. A component that "
                "never ran cannot report itself missing, so the row would be unwatched "
                "while reading as covered."
            )

    def test_every_machine_watcher_is_a_real_component(self, components: dict) -> None:
        """A watcher naming a job that does not exist is a row watched by
        nothing, indistinguishable from one that is watched."""
        for name, row in components["components"].items():
            watcher = row.get("absence_watched_by", "alerts.sweep")
            assert watcher == "operator" or watcher in components["components"], (
                f"{name} is watched by {watcher!r}, which is neither `operator` nor a "
                "registered component."
            )

    def test_exactly_one_row_is_watched_by_a_human_and_it_is_the_heartbeat(
        self, components: dict
    ) -> None:
        """Naming it is what keeps it from being mistaken for coverage that
        does not exist. A second human-watched row is a gap that has been
        declared rather than closed."""
        human = {
            name
            for name, row in components["components"].items()
            if row.get("absence_watched_by") == "operator"
        }
        assert human == {"heartbeat"}


class TestSignalClassesAreDeclaredForEveryRow:
    def test_the_five_classes_survive_the_track_c_additions(self, components: dict) -> None:
        for name in ("alerts.sweep", "heartbeat", "drift", "console"):
            assert set(components["components"][name]["signals"]) == SIGNAL_CLASSES, name

    def test_the_registry_parses_into_typed_rows(self) -> None:
        """The loader is the only reader, so a row the loader refuses is a
        row that never reaches the alerter."""
        from crucible.components import load_registry

        registry = load_registry()
        assert set(registry) == set(JOBS)
        assert all(c.lifecycle == "ACTIVE" for c in registry.values())
