"""The fourteen-state classifier and the page it renders.

Normative source: `observability-policy.md` §8.3/§8.4; plan §4.9, §9.2.

The classifier's whole job is to never render the absence of evidence as
green, so most of these tests are about what it does when there is nothing
to read.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.components import Component, Deadline, load_registry
from crucible.console import STATES, build_page, classify, render_html
from crucible.console.render import write_page
from crucible.manifest import manifest_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
DURING_SESSION = dt.datetime(2026, 8, 28, 20, 30, tzinfo=dt.UTC)  # 16:30 ET Friday


def _component(**over) -> Component:
    base = dict(
        name="data.daily",
        description="d",
        lifecycle="ACTIVE",
        signals={k: None for k in ("execution", "cost", "resource", "lineage", "outcome")},
        log_location="/crucible/data.daily",
        log_retention_days=90,
        alert_channel="telegram:crucible-pages",
        console_surface="crucible/runs",
        artifact_retention="forever",
        schedule="weekdays after close",
        deadline=Deadline(anchor="close_plus", offset_hours=3),
    )
    base.update(over)
    return Component(**base)


def _manifest(status="ok", reason="", attempts=1):
    return {
        "run_id": "01JG0000000000000000000001",
        "trading_day": FRIDAY.isoformat(),
        "status": status,
        "reason": reason,
        "cost_usd": 0.5,
        "attempts": [{"n": i + 1, "reason": "initial"} for i in range(attempts)],
    }


class TestVocabulary:
    def test_there_are_exactly_fourteen(self) -> None:
        assert len(STATES) == 14
        assert len(set(STATES)) == 14

    def test_the_fifteenth_state_escape_hatches_are_absent(self) -> None:
        """UNKNOWN, OTHER, PENDING and N/A are the fall-through the
        vocabulary exists to remove."""
        assert not ({"UNKNOWN", "OTHER", "PENDING", "N/A"} & set(STATES))

    def test_a_state_outside_the_vocabulary_is_refused(self) -> None:
        from crucible.console.classify import Classification

        with pytest.raises(ValueError, match="closed vocabulary"):
            Classification("x", "PROBABLY_FINE", "because")

    def test_a_classification_without_a_reason_is_refused(self) -> None:
        """A dot that cannot say how it knows is not trustworthy."""
        from crucible.console.classify import Classification

        with pytest.raises(ValueError, match="no reason"):
            Classification("x", "HEALTHY", "  ")


class TestClassifier:
    def test_an_ok_manifest_is_healthy(self) -> None:
        c = classify(_component(), _manifest(), now=SATURDAY_NIGHT)
        assert c.state == "HEALTHY"

    def test_a_retried_success_says_so(self) -> None:
        c = classify(_component(), _manifest(attempts=2), now=SATURDAY_NIGHT)
        assert c.state == "HEALTHY"
        assert "retry" in c.reason

    def test_a_failed_manifest_is_failed_and_carries_its_cause(self) -> None:
        c = classify(_component(), _manifest("failed", "RuntimeError: boom"), now=SATURDAY_NIGHT)
        assert c.state == "FAILED"
        assert "boom" in c.reason

    def test_a_status_outside_the_closed_set_is_unreported_not_a_guess(self) -> None:
        """A producer whose outcome we cannot read is a finding, in either
        direction — never green and never a fabricated failure."""
        c = classify(_component(), _manifest("degraded"), now=SATURDAY_NIGHT)
        assert c.state == "UNREPORTED"

    def test_a_scheduled_job_past_its_deadline_with_no_run_is_missed(self) -> None:
        c = classify(_component(), None, now=SATURDAY_NIGHT, history=True)
        assert c.state == "MISSED"
        assert "upstream of the component" in c.reason

    def test_a_job_that_has_never_run_is_never_ran_not_missed(self) -> None:
        """Collapsing them reports a defect as a decision, or the reverse."""
        c = classify(_component(), None, now=SATURDAY_NIGHT, history=False)
        assert c.state == "NEVER_RAN"

    def test_before_the_deadline_it_is_running_not_missed(self) -> None:
        c = classify(_component(), None, now=DURING_SESSION, history=True)
        assert c.state == "RUNNING"

    def test_an_on_demand_job_is_armed_not_missed(self) -> None:
        """Its silence is not a fact about the system, and its trigger — the
        CLI job table — still carries it."""
        c = classify(
            _component(name="explain", schedule=None, deadline=None), None, now=SATURDAY_NIGHT
        )
        assert c.state == "ARMED"

    def test_disabled_is_declared_never_inferred(self) -> None:
        c = classify(_component(lifecycle="DISABLED"), None, now=SATURDAY_NIGHT)
        assert c.state == "DISABLED"
        assert "not a defect" in c.reason

    def test_retired_is_declared_and_the_row_persists(self) -> None:
        c = classify(_component(lifecycle="RETIRED"), None, now=SATURDAY_NIGHT)
        assert c.state == "RETIRED"

    def test_a_disabled_component_is_not_evaluated_against_its_deadline(self) -> None:
        """Declared lifecycle is read BEFORE evidence and before deadlines,
        or a component deliberately taken off would page for being off."""
        state = classify(_component(lifecycle="DISABLED"), None, now=SATURDAY_NIGHT).state
        assert state != "MISSED"


class TestPage:
    def test_the_population_is_the_whole_registry(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        page = build_page(store, now=SATURDAY_NIGHT)
        assert page.population == len(load_registry())
        assert {r["component"] for r in page.rows} == set(load_registry())

    def test_every_row_resolves_to_a_member_of_the_vocabulary(self, tmp_path) -> None:
        """The totality invariant: no default branch, no `else: HEALTHY`."""
        page = build_page(LocalStore(tmp_path), now=SATURDAY_NIGHT)
        assert all(r["state"] in STATES for r in page.rows)

    def test_an_empty_store_reports_no_healthy_rows(self, tmp_path) -> None:
        """The single most important property: nothing ran, so nothing is
        green."""
        page = build_page(LocalStore(tmp_path), now=SATURDAY_NIGHT)
        assert not [r for r in page.rows if r["state"] == "HEALTHY"]

    def test_the_transparency_gap_count_is_published(self, tmp_path) -> None:
        """§8.4: a fleet that reports no gaps because it never computes this
        number has the largest gap of all."""
        page = build_page(LocalStore(tmp_path), now=SATURDAY_NIGHT)
        assert page.unreported == 0
        assert "transparency gap" in render_html(page)

    def test_deploys_appear_beside_runs(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            manifest_key("deploy", FRIDAY.isoformat()),
            json.dumps(
                {
                    "job": "deploy",
                    "trading_day": FRIDAY.isoformat(),
                    "status": "ok",
                    "reason": "",
                    "release_sha": "a" * 40,
                    "run_id": "01JG0000000000000000000009",
                    "cost_usd": 0.0,
                }
            ).encode(),
        )
        page = build_page(store, now=SATURDAY_NIGHT)
        assert [d["status"] for d in page.deploys] == ["ok"]

    def test_the_weeks_cost_is_summed_from_the_manifests(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        for job, cost in (("data.daily", 0.25), ("report", 1.5)):
            store.put_bytes(
                manifest_key(job, FRIDAY.isoformat()),
                json.dumps(
                    {
                        "job": job,
                        "trading_day": FRIDAY.isoformat(),
                        "status": "ok",
                        "reason": "",
                        "cost_usd": cost,
                        "run_id": "01JG0000000000000000000001",
                    }
                ).encode(),
            )
        assert build_page(store, now=SATURDAY_NIGHT).week_cost_usd == 1.75

    def test_a_missing_attribution_artifact_says_so_rather_than_showing_zeros(
        self, tmp_path
    ) -> None:
        html = render_html(build_page(LocalStore(tmp_path), now=SATURDAY_NIGHT))
        assert "No attribution artifact" in html

    def test_the_page_serves_json_beside_the_html(self, tmp_path) -> None:
        """console-policy: every view serves the JSON an agent reads. A page
        whose numbers can only be scraped out of HTML is re-derived
        incorrectly by the next automated reader."""
        store = LocalStore(tmp_path)
        html_key, json_key = write_page(store, build_page(store, now=SATURDAY_NIGHT))
        assert store.exists(html_key) and store.exists(json_key)
        assert json.loads(store.get_bytes(json_key))["population"] == len(load_registry())

    def test_the_render_escapes_what_it_puts_in_the_page(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            manifest_key("data.daily", FRIDAY.isoformat()),
            json.dumps(
                {
                    "job": "data.daily",
                    "trading_day": FRIDAY.isoformat(),
                    "status": "failed",
                    "reason": "<script>alert(1)</script>",
                    "run_id": "01JG0000000000000000000001",
                    "cost_usd": 0.0,
                }
            ).encode(),
        )
        html = render_html(build_page(store, now=SATURDAY_NIGHT))
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
