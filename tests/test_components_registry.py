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
from pydantic import ValidationError

from crucible.cli import JOBS
from crucible.manifest import SCHEMA_PATH
from crucible.models import TRADER_JOB_VALUES, WORKFLOW_JOB_VALUES, derived_log_location

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
    "dispatch",
    "absence_watched_by",
    "deadline",
}

# §9.2: five signal classes, every component. `outcome` is legitimately
# absent from a job that grades nothing, but it is then absent BY DECLARATION
# (an explicit `null`), never by omission.
SIGNAL_CLASSES = {"execution", "cost", "resource", "lineage", "outcome"}


def _scheduling_disagreements(components: dict) -> list[str]:
    """Every CLI job whose `JobSpec.scheduled` disagrees with its registry
    row's `schedule` (`alpha-engine-config-I11041`)."""
    rows = components["components"]
    return sorted(
        name
        for name, spec in JOBS.items()
        if spec.scheduled is not (rows[name].get("schedule") is not None)
    )


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

    def test_the_cli_and_the_registry_agree_about_what_is_scheduled(self, components: dict) -> None:
        """`alpha-engine-config-I11041`. `JobSpec.scheduled` exists to be
        checked against the registry, and nothing checked it. The first run
        of this test found three jobs the registry schedules that the CLI
        called unscheduled: `promote` and `explain` (arc stages) and `gate`
        (daily since `alpha-engine-config-I10508`)."""
        assert _scheduling_disagreements(components) == []

    def test_the_scheduling_cross_check_names_a_disagreeing_job(self, components: dict) -> None:
        """The guard above, made to fail on a parsed copy with one row
        flipped."""
        import copy

        flipped = copy.deepcopy(components)
        flipped["components"]["report"]["schedule"] = None
        assert _scheduling_disagreements(flipped) == ["report"]

    def test_every_registry_row_is_a_real_job(self, components: dict) -> None:
        """The other direction. A row for a job that no longer exists makes
        the registry look complete while covering something that never runs —
        and its absence deadline can never fire, so it is a permanently
        silent monitor."""
        extra = sorted(
            set(components["components"])
            - set(JOBS)
            - set(TRADER_JOB_VALUES)
            - set(WORKFLOW_JOB_VALUES)
        )
        assert not extra, (
            "components.yaml rows with no CLI job, declared trader job or declared "
            f"workflow job: {extra}"
        )

    def test_every_declared_trader_job_has_a_registry_row(self, components: dict) -> None:
        """`crucible.models.TRADER_JOB_VALUES` jobs are written by the trader,
        not the CLI, and are observed exactly like CLI jobs: a trader job with
        no row would write manifests no failure page reads."""
        missing = sorted(set(TRADER_JOB_VALUES) - set(components["components"]))
        assert not missing, f"trader jobs with no components.yaml row: {missing}"

    def test_a_trader_job_is_never_also_a_cli_job(self) -> None:
        """Disjoint by construction. A name in both would let the harness run a
        trader job body, which plan §3 forbids."""
        assert not set(TRADER_JOB_VALUES) & set(JOBS)

    def test_every_declared_workflow_job_has_a_registry_row(self, components: dict) -> None:
        """The trader half's mirror, for the jobs a GitHub Actions workflow
        writes (`crucible.models.WORKFLOW_JOB_VALUES` — `deploy`).

        `alpha-engine-config-I10969`: `deploy` was in the manifest schema's
        `job` enum and in no registry row, and the difference was asserted as
        `enum - registry == {"deploy"}`. An exemption spelled as a literal in
        a test is a suppression collection with one member; the one-member
        case is the one that reads as harmless, and it froze the gap for the
        single most consequential mutation in the system."""
        missing = sorted(set(WORKFLOW_JOB_VALUES) - set(components["components"]))
        assert not missing, f"workflow jobs with no components.yaml row: {missing}"

    def test_a_workflow_job_is_never_also_a_cli_job(self) -> None:
        """Disjoint by construction, and this one is a security property, not
        only a hygiene one: `crucible/deploy.py`'s module docstring states that
        putting the deploy behind `crucible <job>` would give every runtime
        identity a subcommand it must never be able to execute."""
        assert not set(WORKFLOW_JOB_VALUES) & set(JOBS)
        assert not set(WORKFLOW_JOB_VALUES) & set(TRADER_JOB_VALUES)


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

        There is no permitted difference. `deploy` used to be one — it is in
        the enum, it is not a CLI job, and the deploy workflow writes a
        manifest in the same schema so §4.5's page shows deploys beside runs
        (§4.11). Naming it here did keep it from becoming a general
        exemption, and it also froze the gap: `deploy` had no registry row at
        all, so the one component that moves `releases/current` declared no
        log location, no alert channel, no console surface and no absence
        watcher (`alpha-engine-config-I10969`). It has a row now, and it is
        admitted as a job through `crucible.models.WORKFLOW_JOB_VALUES` —
        checked in both directions, like the trader half."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        enum = set(schema["properties"]["job"]["enum"])
        registry = set(components["components"])
        assert enum - registry == set(), (
            "the manifest schema's job enum and the registry are one closed set. "
            f"In the enum and unregistered: {sorted(enum - registry)}. `deploy` was "
            "the one asserted exception until it was given a row of its own; it is "
            "admitted as a job by crucible.models.WORKFLOW_JOB_VALUES, "
            "in both directions, not by a literal here."
        )
        # The other direction (alpha-engine-config-I9907). Measured before
        # this assertion existed: deleting one job from the schema enum left
        # this file at 18 passed, 0 failed — the "three files, any two
        # disagreeing is caught" claim in AGENTS.md rule 1 held for two of the
        # three pairs. A job present in the CLI table and the registry but
        # absent from the enum ships fine and fails at runtime on its first
        # manifest, with schema validation as the only (indirect) catch.
        assert set(JOBS) - enum == set(), (
            f"CLI jobs missing from the manifest schema's job enum: {sorted(set(JOBS) - enum)}"
        )
        assert registry - enum == set(), (
            f"registry rows missing from the manifest schema's job enum: {sorted(registry - enum)}"
        )


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
            "3h after the close of trading day 2026-08-28 (every trading day)"
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
            assert row["absence_watched_by"] != name, (
                f"{name} declares itself its own absence watcher. A component that "
                "never ran cannot report itself missing, so the row would be unwatched "
                "while reading as covered."
            )

    def test_every_machine_watcher_is_a_real_component(self, components: dict) -> None:
        """A watcher naming a job that does not exist is a row watched by
        nothing, indistinguishable from one that is watched."""
        for name, row in components["components"].items():
            watcher = row["absence_watched_by"]
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
            if row["absence_watched_by"] == "operator"
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
        assert set(registry) == set(JOBS) | set(TRADER_JOB_VALUES) | set(WORKFLOW_JOB_VALUES)
        assert all(c.lifecycle == "ACTIVE" for c in registry.values())


class TestTheLogLocationIsDerivedNotAsserted:
    """`alpha-engine-config-I10971`.

    20 of 33 rows named a CloudWatch group that does not exist in the account,
    and `crucible/board.py` and `crucible/console/render.py` render that
    string to an operator as the place to look when a component misbehaves.
    For most of them no dispatch path could ever create one: the box shell
    derives the group from the DISPATCHED job and the arc dispatches as one
    job, so every stage lands in `/crucible/weekly`; and a `github-actions`
    row's logs live in its Actions run, with no path to CloudWatch at all.

    The fix is to make the declaration honest and then to make the honest
    value the only spellable one — so these tests derive the expected value
    from the row rather than restating a table of 34 strings.
    """

    def test_every_clock_started_row_declares_what_its_dispatch_produces(
        self, components: dict
    ) -> None:
        for name, row in components["components"].items():
            expected = derived_log_location(name, row["dispatch"], row.get("dispatch_workflow"))
            if expected is None:
                continue
            assert row["log_location"] == expected, (
                f"{name}: dispatch {row['dispatch']!r} puts its output in "
                f"{expected!r}, and the row declares {row['log_location']!r}"
            )

    def test_every_on_demand_row_names_a_location_a_dispatch_could_create(
        self, components: dict
    ) -> None:
        """The on-demand half, where the registry does not say whether a
        person will dispatch the row to a box or a workflow will run it. The
        scheme is open; the locator is not."""
        workflows = {
            path.name for path in (COMPONENTS_PATH.parents[1] / ".github" / "workflows").iterdir()
        }
        for name, row in components["components"].items():
            if derived_log_location(name, row["dispatch"], row.get("dispatch_workflow")):
                continue
            scheme, _, locator = row["log_location"].partition(":")
            assert scheme in ("cloudwatch", "github-actions"), f"{name}: {row['log_location']}"
            if scheme == "cloudwatch":
                assert locator == f"/crucible/{name}", (
                    f"{name}: a box derives the group from the job name, so a dispatch "
                    f"of this row can only ever create '/crucible/{name}'"
                )
            else:
                assert locator in workflows, (
                    f"{name}: names workflow {locator!r}, which is not in "
                    ".github/workflows/ — a location an operator would be sent to "
                    "and find nothing"
                )

    def test_no_row_claims_a_per_stage_group_the_arc_can_never_create(
        self, components: dict
    ) -> None:
        """The guard firing on the exact shape this issue was filed over. Named
        separately from the derivation test because it is the one an operator
        would recognise: `promote` declared `/crucible/promote`, and the arc
        cannot create it."""
        arc_rows = {n for n, r in components["components"].items() if r["dispatch"] == "arc"}
        assert arc_rows, "no arc-dispatched rows — this guard has stopped measuring anything"
        for name in arc_rows:
            assert components["components"][name]["log_location"] == "cloudwatch:/crucible/weekly"

    def test_the_loader_refuses_a_log_location_the_dispatch_cannot_produce(self) -> None:
        """The self-test: the guard is shown firing (`AGENTS.md` test
        discipline — a detector nobody has made fail is a detector nobody
        knows works)."""
        from crucible.models import ComponentsDocument

        document = yaml.safe_load(COMPONENTS_PATH.read_text(encoding="utf-8"))
        document["components"]["promote"]["log_location"] = "cloudwatch:/crucible/promote"
        with pytest.raises(ValidationError) as excinfo:
            ComponentsDocument.model_validate(document)
        message = str(excinfo.value)
        assert "promote" in message
        assert "/crucible/weekly" in message, (
            "the refusal names the location the row's dispatch really produces"
        )

    def test_the_loader_refuses_a_location_with_no_scheme(self) -> None:
        from crucible.models import ComponentsDocument

        document = yaml.safe_load(COMPONENTS_PATH.read_text(encoding="utf-8"))
        document["components"]["data.daily"]["log_location"] = "/crucible/data.daily"
        with pytest.raises(ValidationError) as excinfo:
            ComponentsDocument.model_validate(document)
        assert "log_location" in str(excinfo.value)


class TestAbsenceWatchedByIsDeclaredNotDefaulted:
    """`alpha-engine-config-I10970`.

    The field that answers *what would notice this component is missing* was
    defaulted to `alerts.sweep`, so leaving the key out was spellable and read
    as coverage. `holdout` left it out, and `crucible/board.py` and
    `crucible/console/render.py` both told an operator it was watched — a
    coverage claim manufactured by a default rather than declared by anyone.
    `observability-policy` §2.2: coverage is derived, never assumed.
    """

    def test_every_row_declares_its_watcher(self, components: dict) -> None:
        missing = sorted(
            name
            for name, row in components["components"].items()
            if "absence_watched_by" not in row
        )
        assert not missing, f"rows with no declared absence watcher: {missing}"

    def test_the_loader_refuses_a_row_that_omits_the_watcher(self) -> None:
        """The self-test. Modelled on `tests/test_typed_boundaries.py`'s
        `log_location` case: delete the field from a parsed document and
        assert the refusal NAMES it."""
        from crucible.models import ComponentsDocument

        document = yaml.safe_load(COMPONENTS_PATH.read_text(encoding="utf-8"))
        del document["components"]["holdout"]["absence_watched_by"]
        with pytest.raises(ValidationError) as excinfo:
            ComponentsDocument.model_validate(document)
        message = str(excinfo.value)
        assert "absence_watched_by" in message, "the failure names the FIELD that is missing"
        assert "holdout" in message, "the failure names the ROW that is missing it"
