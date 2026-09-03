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
from crucible.console.render import (
    ATTRIBUTION_STATUSES,
    STATUS_COLORS,
    _read_representative_manifest,
    _state_class,
    write_page,
)
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


def _manifest(status="ok", reason="", attempts=1, metrics=None):
    return {
        "run_id": "01JG0000000000000000000001",
        "trading_day": FRIDAY.isoformat(),
        "status": status,
        "reason": reason,
        "cost_usd": 0.5,
        "attempts": [{"n": i + 1, "reason": "initial"} for i in range(attempts)],
        "metrics": metrics or [],
    }


def _metric(name: str, status: str) -> dict:
    return {"name": name, "status": status, "value": None, "unit": "ratio"}


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

    def test_an_ok_run_with_every_metric_unreported_is_not_healthy(self) -> None:
        """alpha-engine-config-I9757, C5. A manifest can carry `status: ok`
        and declare metrics that all carry no value — reproduced by the
        drift job over structurally-present, semantically-empty inputs
        (`crucible.drift.drift_metrics` now refuses to let THAT combination
        reach a manifest, by raising before the caller can write `status:
        ok` — see `tests/test_drift.py`). This is the systemic backstop:
        any other metric-emitting component that has not been given that
        producer-side guard must still not render green here.
        """
        manifest = _manifest(
            metrics=[
                _metric("feature_psi_max_ratio", "UNREPORTED"),
                _metric("ic_decay_ratio", "UNREPORTED"),
            ]
        )
        c = classify(_component(), manifest, now=SATURDAY_NIGHT)
        assert c.state == "UNREPORTED"
        assert "feature_psi_max_ratio" in c.reason and "ic_decay_ratio" in c.reason

    def test_an_ok_run_with_some_metrics_unreported_is_degraded_not_healthy(self) -> None:
        """The exact C5 reproduction: two of three drift MetricRecords carry
        no value (`feature_psi_max_ratio`, `ic_decay_ratio`) while the third
        (`prediction_psi_ratio`) is a real 0.0. The run legitimately
        measured something, so it is not the total-blindness case above —
        but it must not read as a clean HEALTHY either, or the blind
        metrics are invisible on the one row that owns them."""
        manifest = _manifest(
            metrics=[
                _metric("feature_psi_max_ratio", "UNREPORTED"),
                _metric("prediction_psi_ratio", "OK"),
                _metric("ic_decay_ratio", "UNREPORTED"),
            ]
        )
        c = classify(_component(), manifest, now=SATURDAY_NIGHT)
        assert c.state == "DEGRADED"
        assert "feature_psi_max_ratio" in c.reason and "ic_decay_ratio" in c.reason
        assert "prediction_psi_ratio" not in c.reason

    def test_an_ok_run_with_no_unreported_metrics_is_still_healthy(self) -> None:
        """The new branches must not fire when there is nothing to flag."""
        manifest = _manifest(metrics=[_metric("prediction_psi_ratio", "OK")])
        assert classify(_component(), manifest, now=SATURDAY_NIGHT).state == "HEALTHY"


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

    def test_a_failed_slot_manifest_is_not_masked_by_a_healthy_sibling(self, tmp_path) -> None:
        """alpha-engine-config-I9781: `experiment.run` now writes one
        manifest per slot. A row still renders one classification per job,
        so the console reads the failed slot's manifest rather than
        whichever discriminated key sorts last."""
        store = LocalStore(tmp_path)
        for slot, status, reason in (("u", "ok", ""), ("r", "failed", "boom"), ("s", "ok", "")):
            store.put_bytes(
                manifest_key("experiment.run", FRIDAY.isoformat(), discriminator=slot),
                json.dumps(
                    {
                        "job": "experiment.run",
                        "trading_day": FRIDAY.isoformat(),
                        "status": status,
                        "reason": reason,
                        "cost_usd": 0.0,
                        "run_id": "01JG0000000000000000000001",
                        "discriminator": slot,
                    }
                ).encode(),
            )
        manifest = _read_representative_manifest(store, "experiment.run", FRIDAY.isoformat())
        assert manifest["status"] == "failed"
        assert manifest["discriminator"] == "r"

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

    def test_the_weeks_cost_includes_discriminated_manifests(self, tmp_path) -> None:
        """alpha-engine-config-I9879: the roll-up used to filter on
        `len(parts) != 4`, which is the BARE manifest shape
        (`runs/{job}/{trading_day}/run.json`, 4 segments). A discriminated
        manifest (`runs/{job}/{trading_day}/{discriminator}/run.json`, 5
        segments) — what `experiment.run`/`experiment.grade` and
        `alerts.sweep` actually write — was silently dropped. Shown RED
        against pre-fix `main` in the PR body; this asserts both the bare
        and the discriminated manifest for the SAME job/day are counted.
        """
        store = LocalStore(tmp_path)
        store.put_bytes(
            manifest_key("data.daily", FRIDAY.isoformat()),
            json.dumps(
                {
                    "job": "data.daily",
                    "trading_day": FRIDAY.isoformat(),
                    "status": "ok",
                    "reason": "",
                    "cost_usd": 0.25,
                    "run_id": "01JG0000000000000000000001",
                }
            ).encode(),
        )
        store.put_bytes(
            manifest_key("experiment.run", FRIDAY.isoformat(), discriminator="r"),
            json.dumps(
                {
                    "job": "experiment.run",
                    "trading_day": FRIDAY.isoformat(),
                    "status": "ok",
                    "reason": "",
                    "cost_usd": 1.5,
                    "run_id": "01JG0000000000000000000002",
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
        # Every key `write_page` writes, not the first two: the phase-ladder
        # artifact joined the page and its JSON, and a two-name unpack is how
        # a third output turns a passing test into a ValueError.
        keys = write_page(store, build_page(store, now=SATURDAY_NIGHT))
        json_key = keys[1]
        assert all(store.exists(k) for k in keys)
        assert json.loads(store.get_bytes(json_key))["population"] == len(load_registry())

    def test_the_transparency_gap_count_sees_metric_statuses_not_only_component_states(
        self, tmp_path
    ) -> None:
        """alpha-engine-config-I9757, C5. Before this fix `page.unreported`
        was `sum(1 for r in rows if r["state"] == "UNREPORTED")` — a
        component classified DEGRADED (partial metric blindness) or even
        HEALTHY (a future producer without the classify.py backstop)
        contributed zero, so the objective-0 count could read 0 while real
        metrics were silently unmeasured. It must count those metrics too.
        """
        store = LocalStore(tmp_path)
        registry = load_registry()
        name = next(iter(registry))
        store.put_bytes(
            manifest_key(name, FRIDAY.isoformat()),
            json.dumps(
                {
                    "job": name,
                    "trading_day": FRIDAY.isoformat(),
                    "status": "ok",
                    "reason": "",
                    "cost_usd": 0.0,
                    "run_id": "01JG0000000000000000000001",
                    "metrics": [
                        {"name": "a_ratio", "status": "UNREPORTED", "value": None, "unit": "ratio"},
                        {"name": "b_ratio", "status": "OK", "value": 0.01, "unit": "ratio"},
                        {"name": "c_ratio", "status": "UNREPORTED", "value": None, "unit": "ratio"},
                    ],
                }
            ).encode(),
        )
        page = build_page(store, registry=registry, now=SATURDAY_NIGHT)
        row = next(r for r in page.rows if r["component"] == name)
        assert row["state"] == "DEGRADED"  # not counted as UNREPORTED itself
        assert page.unreported == 2  # the two blind metrics are counted anyway

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

    def test_the_attribution_tables_own_statuses_render_with_a_stylesheet_rule(
        self, tmp_path
    ) -> None:
        """alpha-engine-config-I9757, C14. `render.py` used to emit
        `class="state s-{status}"` with CSS rules hand-listed against the
        fourteen COMPONENT states only. The attribution table's own
        vocabulary — `OK`, `RED`, `GREEN`, `BREACH`, `N/A-NOT-RUN`,
        `N/A-NOT-IMPL` — had no rule at all, so a fully-unmeasured report
        card rendered visually identical to a green one. Reproduced here
        with all five real attribution statuses in one table.
        """
        store = LocalStore(tmp_path)
        store.put_bytes(
            f"report/{FRIDAY.isoformat()}/attribution.json",
            json.dumps(
                {
                    "rows": [
                        {
                            "name": "data_freshness",
                            "value": None,
                            "unit": None,
                            "status": "N/A-NOT-RUN",
                            "status_reason": "not run this week",
                        },
                        {
                            "name": "signal_ic",
                            "value": None,
                            "unit": None,
                            "status": "N/A-NOT-IMPL",
                            "status_reason": "track A not built",
                        },
                        {
                            "name": "prediction_ic",
                            "value": 0.04,
                            "unit": "ratio",
                            "status": "OK",
                            "status_reason": "within band",
                        },
                        {
                            "name": "portfolio_alpha",
                            "value": 0.30,
                            "unit": "ratio",
                            "status": "BREACH",
                            "status_reason": "over band",
                        },
                        {
                            "name": "execution_shortfall",
                            "value": None,
                            "unit": None,
                            "status": "RED",
                            "status_reason": "critical",
                        },
                        {
                            "name": "contamination_attestation",
                            "value": 1.0,
                            "unit": "ratio",
                            "status": "GREEN",
                            "status_reason": "clean",
                        },
                        # The three `krepis.metrics.derive_status` states the
                        # hand-written `ATTRIBUTION_STATUSES` tuple omitted.
                        # `derive_status` returns all of them and
                        # `crucible.report` therefore writes them, so before
                        # the tuple was derived a WATCH row would have raised
                        # in `_state_class` and taken the console down.
                        {
                            "name": "signal_ic",
                            "value": 0.02,
                            "unit": "ratio",
                            "status": "WATCH",
                            "status_reason": "between the floor and half of it",
                        },
                        {
                            "name": "prediction_ic",
                            "value": None,
                            "unit": None,
                            "status": "N/A-LOW-N",
                            "status_reason": "2 of 6 sessions settled",
                        },
                        {
                            "name": "universe_coverage",
                            "value": None,
                            "unit": None,
                            "status": "N/A-MISSING-INPUT",
                            "status_reason": "features/v3/2026-08-28.parquet is absent",
                        },
                    ]
                }
            ).encode(),
        )
        html = render_html(build_page(store, now=SATURDAY_NIGHT))
        for status in ATTRIBUTION_STATUSES:
            assert f".s-{status} {{" in html, f"no stylesheet rule rendered for {status!r}"
            assert f'class="state s-{status}"' in html, f"{status!r} did not render at all"

    def test_status_colors_covers_every_declared_status(self) -> None:
        """The completeness guard `render.py` asserts at import time,
        exercised here so a broken guard fails a test rather than only an
        import. `STATUS_COLORS` is the single source the stylesheet is
        generated from — it must carry a rule for every component state
        AND every attribution status, or one of them renders with no color
        (C14)."""
        assert set(STATES) <= set(STATUS_COLORS)
        assert set(ATTRIBUTION_STATUSES) <= set(STATUS_COLORS)

    def test_a_status_with_no_registered_color_is_refused_not_rendered_plain(self) -> None:
        """The mapping cannot silently go stale: a status reaching the
        template with no entry in `STATUS_COLORS` raises rather than
        rendering with no CSS rule — the exact silent failure C14 was."""
        with pytest.raises(KeyError, match="no entry in STATUS_COLORS"):
            _state_class("SOMETHING_NOBODY_REGISTERED")
