"""A deliberate probe must never be indistinguishable from a real failure.

Normative source: the class measured 2026-09-09 against the real production
store (the v2 store URI, read from the environment — never a literal here):

* `alerts/2026-08-07/failure.data.weekly.json` — `sent: true`,
  `destination: operator_chat` — was produced by
  `runs/data.weekly/2026-08-07/run.json`, a `run_mode: replay` run whose
  reason names a ticker that exists only in a fault-injection dispatch. A
  human was paged at incident severity by an exercise.
* `runs/fault.probe/2026-09-11/run.json` was dispatched with
  `--date 2026-09-11`, so `evaluate_dispatch_absence` — which resolved the
  expected manifest prefix from the dispatch's WALL CLOCK — looked under a
  different trading day and would have paged an ABSENCE for a run whose
  artifact exists.

Each test below is one of those two, plus the property that prevents the
obvious wrong fix (recognising the sentinel ticker or the `fault.probe` job
name) from being reintroduced.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import (
    Page,
    bus_row,
    cause_key,
    evaluate_dispatch_absence,
    group_pages,
)
from crucible.keys import dispatch_key, manifest_key
from crucible.store import LocalStore
from crucible.synthetic import (
    args_synthetic_marker,
    manifest_synthetic_marker,
    synthetic_marker,
)

DAY = dt.date(2026, 8, 7)

#: The real dispatch that produced the observed page, verbatim.
FAULT_TWO_REASON = (
    "MissingSourceError: ArcticDB universe library dropped 1 of 4 requested "
    "symbol(s) for the window ending 2026-08-07: ['ZZZCRUCIBLEFAULTINJECT']."
)


class TestTheMarkerIsDerivedNotMatched:
    """The marker comes from what the INVOCATION declared, nowhere else."""

    def test_replay_is_marked(self) -> None:
        assert synthetic_marker(run_mode="replay", fault_capability_class=None) == "replay"

    def test_fault_class_is_marked(self) -> None:
        assert (
            synthetic_marker(run_mode="live", fault_capability_class="chaos_probe")
            == "fault-injected: chaos_probe"
        )

    def test_both_are_named_separately(self) -> None:
        assert (
            synthetic_marker(run_mode="replay", fault_capability_class="chaos_probe")
            == "replay; fault-injected: chaos_probe"
        )

    def test_a_live_run_is_not_synthetic(self) -> None:
        assert synthetic_marker(run_mode="live", fault_capability_class=None) is None

    def test_the_sentinel_ticker_alone_marks_nothing(self) -> None:
        """The WRONG fix, asserted against.

        A `live` run whose reason names the fault-injection sentinel is a real
        production failure — somebody has put that ticker in the real
        universe — and it must page like one. A marker derived from the reason
        text would mislabel it, and would also miss the next fault, which will
        be induced with a different sentinel on a different job.
        """
        manifest = {"run_mode": "live", "reason": FAULT_TWO_REASON}
        assert manifest_synthetic_marker(manifest) is None

    def test_the_job_name_alone_marks_nothing(self) -> None:
        manifest = {"job": "fault.probe", "run_mode": "live"}
        assert manifest_synthetic_marker(manifest) is None

    def test_an_unreadable_manifest_is_treated_as_real(self) -> None:
        for document in (None, "failed", ["failed"], 3):
            assert manifest_synthetic_marker(document) is None


class TestTheObservedPage:
    """`runs/data.weekly/2026-08-07/run.json` — the manifest that paged."""

    def _page(self) -> Page:
        manifest = {
            "run_mode": "replay",
            "status": "failed",
            "reason": FAULT_TWO_REASON,
            "run_id": "01M23N1HSW5DPJ9S4M0E1W0ZPK",
        }
        return Page(
            condition="failure",
            job="data.weekly",
            trading_day=DAY,
            reason=FAULT_TWO_REASON,
            run_id="01M23N1HSW5DPJ9S4M0E1W0ZPK",
            synthetic=manifest_synthetic_marker(manifest),
        )

    def test_the_rendered_page_leads_with_the_banner(self) -> None:
        rendered = group_pages([self._page()])[0].render()
        assert rendered.startswith("[crucible-v2] SYNTHETIC (replay) FAILURE on 2026-08-07")

    def test_the_body_says_nothing_is_known_to_be_wrong(self) -> None:
        rendered = group_pages([self._page()])[0].render()
        assert "deliberate exercise" in rendered

    def test_the_bus_row_carries_it_machine_readably(self) -> None:
        group = group_pages([self._page()])[0]
        row = bus_row(
            group,
            alert_id="01M23N1HSW5DPJ9S4M0E1W0ZPK",
            sent=True,
            destination="operator_chat",
            first_observed_utc="2026-08-08T01:00:00Z",
            last_observed_utc="2026-08-08T01:00:00Z",
        )
        assert row["synthetic"] == "replay"
        assert row["members"][0]["synthetic"] == "replay"

    def test_a_real_failure_carries_a_null_marker_not_a_missing_field(self) -> None:
        """`no data` is never green: the field is always present."""
        page = Page(
            condition="failure",
            job="data.weekly",
            trading_day=DAY,
            reason="ConnectionError: the source was down",
            run_id="01M23N1HSW5DPJ9S4M0E1W0ZPK",
        )
        group = group_pages([page])[0]
        row = bus_row(
            group,
            alert_id=None,
            sent=True,
            destination="operator_chat",
            first_observed_utc="2026-08-08T01:00:00Z",
            last_observed_utc="2026-08-08T01:00:00Z",
        )
        assert "synthetic" in row
        assert row["synthetic"] is None
        assert group.render().startswith("[crucible-v2] FAILURE")


class TestSyntheticAndRealNeverShareAPage:
    """The worst outcome of this change, asserted against."""

    def test_two_incidents_two_groups(self) -> None:
        # Same trading day, same CAUSE_MATCHERS bucket — everything that
        # would have grouped them before.
        induced = Page(
            condition="failure",
            job="data.weekly",
            trading_day=DAY,
            reason="data source unavailable: " + FAULT_TWO_REASON,
            run_id="01M23N1HSW5DPJ9S4M0E1W0ZPK",
            synthetic="replay",
        )
        real = Page(
            condition="failure",
            job="data.daily",
            trading_day=DAY,
            reason="data source unavailable: the vendor returned 503",
            run_id="01M23N1HSW5DPJ9S4M0E1W0ZPL",
        )
        groups = group_pages([real, induced])
        assert len(groups) == 2
        assert {g.synthetic for g in groups} == {None, "replay"}
        assert cause_key(induced).startswith("synthetic.")
        assert not cause_key(real).startswith("synthetic.")

    def test_the_two_incidents_get_two_bus_keys(self) -> None:
        from crucible.alerts import bus_key

        induced = Page(
            condition="failure",
            job="data.weekly",
            trading_day=DAY,
            reason="data source unavailable",
            run_id="01M23N1HSW5DPJ9S4M0E1W0ZPK",
            synthetic="replay",
        )
        real = Page(
            condition="failure",
            job="data.weekly",
            trading_day=DAY,
            reason="data source unavailable",
            run_id="01M23N1HSW5DPJ9S4M0E1W0ZPL",
        )
        keys = {bus_key(g) for g in group_pages([induced, real])}
        assert len(keys) == 2


class TestDispatchArgsAreTheMarkerWhenThereIsNoManifest:
    def test_the_real_fault_probe_dispatch(self) -> None:
        args = "--date 2026-09-11 --run-mode replay --fault-capability-class chaos_probe"
        assert args_synthetic_marker(args) == "replay; fault-injected: chaos_probe"

    def test_equals_form(self) -> None:
        assert args_synthetic_marker("--run-mode=replay") == "replay"

    def test_a_live_dispatch_is_not_marked(self) -> None:
        assert args_synthetic_marker("--date 2026-09-11 --run-mode live") is None

    def test_a_substring_is_not_a_flag(self) -> None:
        """`--reason 'the --run-mode replay arc'` is not a declaration."""
        assert args_synthetic_marker("--reason 'a note about --run-mode replay runs'") is None

    def test_a_flag_with_no_value_is_not_a_declaration(self) -> None:
        assert args_synthetic_marker("--run-mode") is None


DISPATCHED_AT = dt.datetime(2026, 9, 9, 18, 55, 17, tzinfo=dt.UTC)
PAST_HORIZON = DISPATCHED_AT + dt.timedelta(hours=10)


def _write_dispatch(store: LocalStore, *, job: str, args: str, dispatch_id: str) -> None:
    store.put_bytes(
        dispatch_key(job, dispatch_id),
        json.dumps(
            {
                "schema_version": "dispatch_record.v1",
                "job": job,
                "args": args,
                "instance_id": "i-0022daf169d1f7412",
                "requested_by": "operator:fault-3",
                "dispatched_at_utc": DISPATCHED_AT.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ).encode(),
    )


def _no_reason(instance_id: str) -> None:
    return None


class TestTheDispatchAbsenceReadsTheDispatchsOwnDate:
    """The un-gradeable-probe half: the detector must look where the manifest
    actually lands, or it grades a prefix nobody writes to."""

    ARGS = "--date 2026-09-11 --run-mode replay --fault-capability-class chaos_probe"

    def test_a_manifest_under_the_dispatchs_own_date_clears_the_absence(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store, job="fault.probe", args=self.ARGS, dispatch_id="01probe")
        store.put_bytes(
            manifest_key("fault.probe", "2026-09-11"),
            json.dumps({"status": "failed", "run_mode": "replay"}).encode(),
        )
        pages = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
        )
        assert pages == []

    def test_a_missing_manifest_pages_against_the_dispatchs_own_date(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store, job="fault.probe", args=self.ARGS, dispatch_id="01probe")
        pages = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
        )
        assert len(pages) == 1
        assert pages[0].trading_day == dt.date(2026, 9, 11)
        assert "runs/fault.probe/2026-09-11" in pages[0].reason

    def test_that_absence_page_is_marked_synthetic(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store, job="fault.probe", args=self.ARGS, dispatch_id="01probe")
        pages = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
        )
        assert pages[0].synthetic == "replay; fault-injected: chaos_probe"
        assert group_pages(pages)[0].render().startswith("[crucible-v2] SYNTHETIC")

    def test_a_dispatch_with_no_date_still_uses_the_wall_clock(self, tmp_path) -> None:
        """The behaviour `alpha-engine-config-I10134` shipped, unchanged."""
        store = LocalStore(tmp_path)
        _write_dispatch(
            store,
            job="data.heal",
            args="--from 2025-01-21 --to 2025-01-21 --run-mode live",
            dispatch_id="01heal",
        )
        pages = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
        )
        assert len(pages) == 1
        assert pages[0].trading_day != dt.date(2026, 9, 11)
        assert pages[0].synthetic is None


class TestAUsageRefusalIsNotAHarnessDeath:
    """`crucible.cli.USAGE_EXIT_CODE`: a refused invocation exits 2, which the
    box wrapper reports as MALFORMED DISPATCH ("no run manifest was written by
    this attempt") rather than as "the harness died before writing one"."""

    def test_a_bad_fault_capability_class_exits_two(self) -> None:
        from crucible.cli import USAGE_EXIT_CODE, main

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "fault.probe",
                    "--date",
                    "2026-09-11",
                    "--run-mode",
                    "replay",
                    "--fault-capability-class",
                    "gpt5-tier",
                ]
            )
        # argparse's own `choices` claims this one and already exits 2; the
        # assertion stands as the regression guard for the pair — the flag's
        # choices and `crucible.llm.FAULT_INJECTION_CAPABILITY_CLASSES` must
        # not drift into a value argparse admits and `main` then refuses at
        # exit 1.
        assert excinfo.value.code == USAGE_EXIT_CODE

    def test_a_missing_run_mode_exits_two(self, monkeypatch, capsys) -> None:
        """`alpha-engine-config-I10517`: `main` no longer lets a `UsageError`
        propagate as an uncaught `SystemExit` — Python's own top-level
        handling only prints an uncaught `SystemExit`'s message when `.code`
        is a STRING, and `UsageError.code` is always the int
        `USAGE_EXIT_CODE`, so the message never reached stderr on the
        installed console script. `main` now catches it, prints the message,
        and RETURNS the code — still exit 2 at the `sys.exit(main())`
        console-script boundary, but with the message on stderr this time."""
        from crucible.cli import USAGE_EXIT_CODE, main
        from crucible.runmode import RUN_MODE_ENV

        monkeypatch.delenv(RUN_MODE_ENV, raising=False)
        assert main(["data.weekly", "--date", "2026-09-11"]) == USAGE_EXIT_CODE
        assert "live" in capsys.readouterr().err

    def test_a_bad_date_exits_two(self, capsys) -> None:
        from crucible.cli import USAGE_EXIT_CODE, main

        assert main(["data.weekly", "--date", "not-a-date", "--run-mode", "live"]) == (
            USAGE_EXIT_CODE
        )
        assert "YYYY-MM-DD" in capsys.readouterr().err
