"""Every plan §6 phase is graded BY CLAUSE, and no clause reads green on absence.

Normative source: plan §6 (one gate per phase, "a gate reads; it never runs"),
§12 rule 8 (instruments before product), and `alpha-engine-config-I9913`.

The defect these tests exist for: `GATES` registered `phase0` and `phase1`
only, so four of six ladder rows rendered with no instrument behind them.
Blank and "no data yet" render identically, so a reader could not tell whether
the instrument for phase 3 EXISTED or merely had nothing to read.

Two properties are asserted here and they pull in opposite directions, which
is the point:

1. **Structural** — every phase carries a registered clause list, and every
   registered gate is read by a phase. Neither direction alone is enough: the
   first lets a gate be registered that nothing renders, the second lets a
   phase go blank.
2. **Per clause** — each clause has a MET, an UNMET and an UNMEASURABLE
   reading, and the UNMEASURABLE case is the EMPTY store. A clause with no
   exercised unmeasurable path is one that will report a number when it should
   report that it could not look.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

import crucible.alerts as alerts_module
import crucible.autonomy as autonomy_module
import crucible.cost as cost_module
import crucible.gate as gate_module
import crucible.llm as llm_module
from crucible.gate import (
    GATES,
    MANIFEST_RUN_MODE_FIELD,
    MANIFEST_RUN_MODE_LIVE,
    PHASE2_LIVE_SATURDAYS,
    PHASE2_MAX_PAGES,
    PHASE2_MAX_TAGGED_USD,
    PHASE4_MAX_TOTAL_USD,
    PHASES,
    evaluate,
    weekly_anchor,
)
from crucible.keys import (
    champion_key,
    manifest_key,
    parse_bus_key,
    runs_prefix,
    strategy_arm_key,
    verdict_key,
)
from crucible.manifest import PREDECESSOR_SCHEMA_VERSION, RUN_MANIFEST_SCHEMA_VERSION
from crucible.slots import SLOTS
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SHA = "a" * 40


def _window(weeks: int) -> list[dt.date]:
    return [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(weeks))]


PHASE2_WINDOW = _window(PHASE2_LIVE_SATURDAYS)

#: A render day that is NOT a weekly anchor, and the window `_window` builds
#: off it. Weekly work binds to a Friday close while the ladder renders DAILY,
#: so a clause that keys on the render weekday and one that anchors are
#: indistinguishable when the fixture renders on a Friday — the shape that hid
#: `alpha-engine-config-I9904` from this suite (PR68 adversarial review, F1).
WEDNESDAY = dt.date(2026, 9, 9)
PHASE2_RENDER_WINDOW = [
    WEDNESDAY - dt.timedelta(weeks=n) for n in reversed(range(PHASE2_LIVE_SATURDAYS))
]

#: Where a weekly run's manifest is actually filed for those render days: the
#: Friday close strictly before each, resolved through the trading calendar.
PHASE2_ANCHORS = [weekly_anchor(day) for day in PHASE2_RENDER_WINDOW]


@pytest.fixture
def store(tmp_path) -> LocalStore:
    """An EMPTY store. Every unmeasurable case below reads this one."""
    return LocalStore(tmp_path)


def _put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------


class TestNoPlanPhaseCanRenderBlank:
    def test_every_plan_phase_carries_a_gate(self) -> None:
        """The guard that closes `alpha-engine-config-I9913` for good.

        Adding a phase to `PHASES` without writing its clause list is now a
        red CI run rather than a silent blank row on the board.
        """
        blank = [p.id for p in PHASES if p.gate is None]
        assert not blank, f"{blank} would render UNMEASURED — blank, not red"

    def test_every_registered_gate_is_read_by_a_phase(self) -> None:
        """The other direction. A gate no phase names is a clause list nobody
        renders — measured, and invisible."""
        assert set(GATES) == {p.gate for p in PHASES}

    @pytest.mark.parametrize("gate", ["phase2", "phase3", "phase4", "phase5"])
    def test_an_empty_store_grades_every_clause_and_meets_none(
        self, store: LocalStore, gate: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Against an empty store: clauses exist, none is met, and at least one
        says it could not be read rather than reporting a zero.

        The AWS readers are stubbed to RAISE — the reading identity has no
        `ce:GetCostAndUsage` and CI has no CloudTrail archive, and a test that
        depended on which of those two is true would pass for the wrong
        reason on a laptop that happens to hold credentials.
        """
        monkeypatch.setattr(gate_module, "_ce_client", _raising_client)
        monkeypatch.setattr(gate_module, "_s3_client", _raising_client)
        result = evaluate(store, gate=gate, trading_day=FRIDAY)
        assert result.clauses, f"{gate} was graded by zero clauses"
        assert not result.met
        assert [c for c in result.clauses if c.unmeasurable], (
            f"{gate} reported a reading for every clause against an EMPTY store"
        )
        for clause in result.clauses:
            assert not clause.met
            assert clause.detail.strip(), f"{gate}/{clause.name} gave no reason"


def _raising_client() -> object:
    raise RuntimeError("no credentials are configured for this reading identity")


# ---------------------------------------------------------------------------
# phase 2 — live Saturdays
# ---------------------------------------------------------------------------


def _schema_without_run_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build whose run-manifest schema declares no live/replay field.

    That was the real state until `run_manifest.v2` landed
    (alpha-engine-config-I9918), and the guard reading it is kept — a clause
    that answered from `crucible.gate`'s own constant rather than from the
    CONTRACT would grade every manifest as malformed on a build whose schema
    had lost the field. Patched rather than deleted so the unmeasurable branch
    stays exercised now that the real schema satisfies it.
    """
    monkeypatch.setattr(
        gate_module,
        "_manifest_property_names",
        lambda: frozenset({"status", "attempts"}),
    )


def _weekly(day: dt.date, *, mode: str = MANIFEST_RUN_MODE_LIVE, attempts: int = 1) -> dict:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": "ok",
        "reason": "",
        MANIFEST_RUN_MODE_FIELD: mode,
        "attempts": [{"n": n + 1} for n in range(attempts)],
    }


class TestLiveSaturdaysAreReadFromTheManifestNeverTheDate:
    """Every case here renders on a WEDNESDAY and files on the Friday anchors.

    The render day is deliberately not the anchor. `_window` steps back in raw
    calendar weeks, so a Friday render day makes the window days and the Friday
    anchors identical — a clause keyed on the render weekday would then pass
    every case in this class while being unsatisfiable on the four other
    weekdays the ladder renders on (PR68 adversarial review, F1;
    `alpha-engine-config-I9904`).
    """

    def test_the_fixture_days_and_the_render_window_are_different_keys(self) -> None:
        """The precondition every other case in this class depends on. If this
        ever passes trivially again, the class has stopped testing anything."""
        assert not set(PHASE2_ANCHORS) & set(PHASE2_RENDER_WINDOW)

    def test_unmeasurable_when_the_manifest_cannot_say_live_or_replay(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reading on a build whose schema declares no live/replay field:
        the clause refuses to answer rather than inferring liveness from the
        trading day — which the accelerated replay schedule would satisfy."""
        _schema_without_run_mode(monkeypatch)
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert MANIFEST_RUN_MODE_FIELD in clause.detail

    def test_the_shipped_schema_makes_this_clause_measurable(self, store: LocalStore) -> None:
        """The half `alpha-engine-config-I9918` actually closed: with NO
        monkeypatch, against the schema this build ships, the clause grades
        the store instead of reporting that it could not look."""
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.unmeasurable and not clause.met
        assert "never ran" in clause.detail

    def test_met_when_both_saturdays_are_live_and_first_attempt_ok(self, store: LocalStore) -> None:
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert clause.met and not clause.unmeasurable

    def test_the_same_two_saturdays_read_met_on_every_render_weekday(
        self, store: LocalStore
    ) -> None:
        """The regression that closes the class rather than the instance.

        Two flawless live Saturdays are filed at their Friday closes and left
        alone; only the day somebody rendered the ladder moves. A clause
        windowed on the render weekday reads MET on the Friday and `never ran`
        on the other four — the board renders DAILY, so that is a contract
        unsatisfiable four days in five.
        """
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day))
        for render in (
            dt.date(2026, 9, 7),
            dt.date(2026, 9, 9),
            dt.date(2026, 9, 10),
            dt.date(2026, 9, 11),
        ):
            window = [
                render - dt.timedelta(weeks=n) for n in reversed(range(PHASE2_LIVE_SATURDAYS))
            ]
            clause = gate_module._clause_live_saturdays_first_attempt_ok(store, window)
            assert clause.met, f"{render.isoformat()}: {clause.detail}"

    def test_a_replay_is_unmet_not_met(self, store: LocalStore) -> None:
        """The whole reason the clause list was withheld. A perfect replay
        Saturday is UNMET here, and says which day was not live."""
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day, mode="replay"))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "not live" in clause.detail

    def test_a_manifest_predating_the_field_cannot_be_counted_live(self, store: LocalStore) -> None:
        """The grandfathering half. A weekly manifest written under
        `run_manifest.v1` is not malformed and is not condemned — it simply
        cannot establish liveness, so it reads UNMET with that said in terms.
        Counting it live on the strength of its date is the one thing this
        clause exists not to do.
        """
        for day in PHASE2_ANCHORS:
            document = _weekly(day)
            del document[MANIFEST_RUN_MODE_FIELD]
            document["schema_version"] = PREDECESSOR_SCHEMA_VERSION
            _put(store, manifest_key("weekly", day.isoformat()), document)
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "predate the live/replay field" in clause.detail
        assert PREDECESSOR_SCHEMA_VERSION in clause.detail

    def test_a_retried_run_is_not_a_first_attempt_ok(self, store: LocalStore) -> None:
        for day in PHASE2_ANCHORS:
            _put(store, manifest_key("weekly", day.isoformat()), _weekly(day, attempts=2))
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met
        assert "retried" in clause.detail

    def test_an_absent_manifest_is_unmet_by_the_anchor_key_not_the_render_day(
        self, store: LocalStore
    ) -> None:
        clause = gate_module._clause_live_saturdays_first_attempt_ok(store, PHASE2_RENDER_WINDOW)
        assert not clause.met
        assert "never ran" in clause.detail
        for anchor in PHASE2_ANCHORS:
            assert manifest_key("weekly", anchor.isoformat()) in clause.detail
        for rendered in PHASE2_RENDER_WINDOW:
            assert manifest_key("weekly", rendered.isoformat()) not in clause.detail


class TestReplaysReuseThePhaseOnePredicate:
    def test_the_clause_is_phase_ones_reading_renamed_over_five_saturdays(
        self, store: LocalStore
    ) -> None:
        """Not a restatement. The requirement quotes phase 1's own sentence and
        the window is the five replay Saturdays, so phase 2 and phase 1 cannot
        disagree about what a good replay is."""
        clause = gate_module._clause_replays_ok(store, PHASE2_RENDER_WINDOW, {})
        assert clause.name == "replays_ok"
        assert not clause.met
        # Anchored to weekly closes, not to the render weekday
        # (`alpha-engine-config-I9904`): a Wednesday render reads the five
        # Fridays strictly before it, and names none of the Wednesdays.
        anchors = gate_module.weekly_window(WEDNESDAY, gate_module.PHASE2_REPLAY_SATURDAYS)
        assert len(anchors) == gate_module.PHASE2_REPLAY_SATURDAYS
        assert all(day.weekday() == 4 for day in anchors)
        assert f"{anchors[0].isoformat()}..{anchors[-1].isoformat()}" in clause.requirement
        for anchor in anchors:
            assert manifest_key("data.weekly", anchor.isoformat()) in clause.evidence
        assert manifest_key("data.weekly", WEDNESDAY.isoformat()) not in clause.evidence

    def test_the_replay_window_is_the_same_on_every_render_weekday(self, store: LocalStore) -> None:
        clauses = [
            gate_module._clause_replays_ok(
                store,
                [WEDNESDAY + dt.timedelta(days=offset)],
                {},
            )
            for offset in (-2, -1, 0, 1, 2)  # Mon..Fri of the render week
        ]
        assert len({c.requirement for c in clauses}) == 1
        assert len({c.evidence for c in clauses}) == 1


# ---------------------------------------------------------------------------
# phase 2 — autonomy, pages, cost
# ---------------------------------------------------------------------------


def _archive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure an archive location so the reader gets past the "no trail"
    branch. A test-only value: no bucket name is written into this tree."""
    monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://a-test-archive/trail")


class _Counted:
    def __init__(self, count: int) -> None:
        self.count = count
        self.actions = []
        self.objects_read = 3
        self.records_scanned = 40


class TestHumanMutatingCallsAreCountedOrNotAnswered:
    def test_unmeasurable_when_no_archive_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Today's reading, and `0` and `no trail` must never be the same
        answer. There is no default archive on purpose: a guessed bucket name
        produces a `NoSuchBucket` that reads like a permissions problem."""
        monkeypatch.delenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", raising=False)
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "no CloudTrail archive is configured" in clause.detail

    def test_unmeasurable_when_the_archive_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _archive(monkeypatch)
        monkeypatch.setattr(gate_module, "_s3_client", _raising_client)
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert "RuntimeError" in clause.detail

    def test_met_on_a_clean_archive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _archive(monkeypatch)
        monkeypatch.setattr(gate_module, "_s3_client", lambda: object())
        monkeypatch.setattr(autonomy_module, "count_operator_actions", lambda *a, **k: _Counted(0))
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert clause.met and not clause.unmeasurable

    def test_unmet_when_a_human_mutated_something(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _archive(monkeypatch)
        monkeypatch.setattr(gate_module, "_s3_client", lambda: object())
        monkeypatch.setattr(autonomy_module, "count_operator_actions", lambda *a, **k: _Counted(3))
        clause = gate_module._clause_zero_human_mutating_calls(PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable
        assert "3 human mutating call" in clause.detail


class TestPagesAreCountedOnlyOnceSomethingHasSwept:
    def _sweep(self, store: LocalStore) -> None:
        _put(store, manifest_key("alerts.sweep", FRIDAY.isoformat()), {"status": "ok"})

    def test_an_empty_bus_with_no_sweep_is_unmeasurable_not_a_clean_month(
        self, store: LocalStore
    ) -> None:
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.unmeasurable and not clause.met
        assert runs_prefix("alerts.sweep") in clause.detail

    def test_met_when_the_sweep_ran_and_the_bus_is_under_the_ceiling(
        self, store: LocalStore
    ) -> None:
        self._sweep(store)
        _put(store, f"alerts/{FRIDAY.isoformat()}/one.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.met and not clause.unmeasurable

    def test_unmet_over_the_ceiling(self, store: LocalStore) -> None:
        self._sweep(store)
        for n in range(PHASE2_MAX_PAGES + 1):
            _put(store, f"alerts/{FRIDAY.isoformat()}/incident{n}.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert not clause.met and not clause.unmeasurable

    def test_the_gate_and_the_alerts_module_count_the_same_bus(self, store: LocalStore) -> None:
        """One implementation of "pages over a span", not two.

        The clause used to carry its own copy of the count, and the copies
        disagreed at the FIRST day of the window (`start <= day` against the
        module's `start < day`), so the gate and the
        `pages_per_20_trading_days` metric could report different numbers for
        the same bus (PR68 adversarial review, F4). An incident filed on
        exactly `window[0]` is the case that separates them.
        """
        self._sweep(store)
        edge = PHASE2_WINDOW[0]
        _put(store, f"alerts/{edge.isoformat()}/edge.json", {"severity": "page"})
        counted = alerts_module.pages_in_range(store, start=edge, end=PHASE2_WINDOW[-1])
        assert counted == [f"alerts/{edge.isoformat()}/edge.json"]
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.evidence == tuple(counted)
        assert f"{len(counted)} paged incident(s)" in clause.detail

    def test_a_key_that_is_not_a_bus_row_is_not_counted(self, store: LocalStore) -> None:
        """Parsed through `crucible.keys.parse_bus_key`, never by a positional
        index or an arity restated as an integer outside that module — the
        class where `len(parts) != 4` silently dropped every discriminated
        manifest (`alpha-engine-config-I9879`)."""
        assert parse_bus_key(f"alerts/{FRIDAY.isoformat()}/one.json") == (
            FRIDAY.isoformat(),
            "one",
        )
        assert parse_bus_key(f"alerts/{FRIDAY.isoformat()}/nested/one.json") is None
        assert parse_bus_key(f"alerts/{FRIDAY.isoformat()}/one.txt") is None
        assert parse_bus_key(f"runs/weekly/{FRIDAY.isoformat()}/run.json") is None
        self._sweep(store)
        _put(store, f"alerts/{FRIDAY.isoformat()}/nested/one.json", {"severity": "page"})
        clause = gate_module._clause_pages_within_ceiling(store, PHASE2_WINDOW)
        assert clause.met and clause.evidence == ()
        assert f"ceiling {PHASE2_MAX_PAGES}" in clause.detail


class _CostClient:
    """A Cost Explorer stand-in speaking the real response shape.

    ``amount`` answers the MONTHLY month-to-date request; ``daily`` answers the
    DAILY trailing-window request with one period per day in the interval,
    cycling the list. The default daily figure is a quiet `$1.00`/day so the
    pre-existing month-to-date cases keep grading the month-to-date half.
    """

    def __init__(self, amount: str, daily: list[str] | None = None) -> None:
        self.amount = amount
        self.daily = daily or ["1.00"]
        self.requests: list[dict] = []

    def get_cost_and_usage(self, **request) -> dict:
        self.requests.append(request)
        if request["Granularity"] == "DAILY":
            start = dt.date.fromisoformat(request["TimePeriod"]["Start"])
            end = dt.date.fromisoformat(request["TimePeriod"]["End"])
            days = (end - start).days
            return {
                "ResultsByTime": [
                    {"Total": {"UnblendedCost": {"Amount": self.daily[n % len(self.daily)]}}}
                    for n in range(days)
                ]
            }
        return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": self.amount}}}]}


class TestACostCeilingIsNeverMetByAnUnreadableApi:
    def test_unmeasurable_when_cost_explorer_is_denied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`ce:GetCostAndUsage` is not granted to the reading identity today.
        A denial must not read as `$0.00 — under the ceiling`."""
        monkeypatch.setattr(gate_module, "_ce_client", _raising_client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW, name="aws_cost_within_ceiling", ceiling_usd=40.0, tagged=True
        )
        assert clause.unmeasurable and not clause.met

    def test_a_tag_filtered_zero_is_unmeasurable_not_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An untagged estate and a free month return the same `$0.00`, so the
        figure measures the FILTER rather than the spend."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("0"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met

    def test_an_untagged_zero_is_unmeasurable_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same trap one level up, and the half that shipped untracked.

        For a live AWS estate an ACCOUNT total of exactly `$0.00` means Cost
        Explorer answered with nothing chargeable — a broken reading, not a
        free month — so phase 4's row must not go green on it (PR68
        adversarial review, F3).
        """
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("0"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.unmeasurable and not clause.met
        assert "no spend recorded under this filter" in clause.detail.lower()

    def test_met_under_the_tagged_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read on the 28th. `month_to_date_usd` asks for [1st, today) —
        yesterday's close is the last complete day Cost Explorer has — so 27
        days are in; a quiet $1.00/day estate totals $30.00 over the trailing
        thirty, under $40."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("12.50"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.met and not clause.unmeasurable
        assert "$12.50 month-to-date" in clause.detail
        assert "27 of 31 days" in clause.detail
        assert "trailing 30 complete days (2026-07-29..2026-08-28) $30.00" in clause.detail
        assert "ungraded: trailing 7-day mean $1.00/day" in clause.detail

    def test_a_month_boundary_lump_does_not_read_a_compliant_month_as_over(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR80 review, B1. This account posts `$15.59` on the 1st and
        `$1.83`/day after, closing at `$69.90` under a `$70` ceiling. A
        pro-rata line read that month OVER on days 1–26. Over thirty days the
        lump is one day in thirty: $15.59 + 29 x $1.83 = $68.66, under."""
        client = _CostClient("15.59", daily=["15.59"] + ["1.83"] * 29)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 2)],  # one complete day in: the lump
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "$68.66" in clause.detail
        assert clause.detail.endswith("under")

    def test_a_weekly_batch_estate_over_budget_is_unmet_not_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR80 round 2, B1. The estate's real shape: eight of eleven
        registry rows are Saturday-weekly and the Saturday spot run is the
        expensive compute. `$60` every seventh day and `$0.20` otherwise is
        ~`$245`/month; a trailing-week MEDIAN read it as a `$0.20`/day estate
        (MET, "projected $66") until month-to-date itself crossed the ceiling
        on the 10th. Thirty days' total carries four or five batch days."""
        client = _CostClient("60.40", daily=["60"] + ["0.20"] * 6)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 3)],  # two complete days in: one batch, one quiet
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable, clause.detail
        assert clause.detail.endswith("OVER")
        # 30 days = 5 batch days ($300) + 25 quiet days ($5.00); the ungraded
        # leading indicator over the last 7 sees one batch day.
        assert "$305.00" in clause.detail
        assert "trailing 7-day mean $8.74/day" in clause.detail

    def test_four_free_days_in_the_window_cannot_make_it_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A median over a window with four `$0.00` days degenerates to zero
        and a projection to month-to-date — a clause that cannot fail. A
        total cannot be pulled down by quiet days: the three `$30` days are
        still in it."""
        client = _CostClient("30.00", daily=["0"] * 4 + ["30"] * 3)
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 3)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met, clause.detail

    def test_over_budget_with_one_quiet_day_is_unmet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _CostClient("6.10", daily=["3.00"] * 29 + ["0.10"])
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 3)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable
        assert "$87.10" in clause.detail
        assert clause.detail.endswith("OVER")

    def test_a_month_already_over_the_ceiling_is_unmet_whatever_the_trailing_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _CostClient("70.01", daily=["0.01"])
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 30)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable
        assert "already over" in clause.detail
        # The fast path: no trailing read was needed, and none was made.
        assert all(r["Granularity"] == "MONTHLY" for r in client.requests)

    def test_the_trailing_window_is_thirty_complete_days_and_may_cross_the_month(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The window is the ceiling's own period and a property of the
        estate, not of the calendar month it is read in — on the 2nd it
        reaches back into August, so the clause is measurable on day 1 and
        there is no "young month" branch."""
        client = _CostClient("1.50")
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 2)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.met and not clause.unmeasurable
        (daily,) = [r for r in client.requests if r["Granularity"] == "DAILY"]
        assert daily["TimePeriod"] == {"Start": "2026-08-03", "End": "2026-09-02"}
        assert "2026-08-03..2026-09-02" in clause.detail

    def test_the_tag_filter_reaches_the_trailing_read_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _CostClient("1.50")
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert len(client.requests) == 2
        assert all("Filter" in r for r in client.requests)

    def test_an_unreadable_trailing_window_is_unmeasurable_not_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Month-to-date readable and under, the daily read denied: the clause
        cannot grade the month and says so."""

        class _DailyDenied(_CostClient):
            def get_cost_and_usage(self, **request) -> dict:
                if request["Granularity"] == "DAILY":
                    raise RuntimeError("AccessDenied: ce:GetCostAndUsage DAILY")
                return super().get_cost_and_usage(**request)

        monkeypatch.setattr(gate_module, "_ce_client", lambda: _DailyDenied("1.50"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.unmeasurable and not clause.met
        assert "30-day total could not be read" in clause.detail
        assert "$1.50 month-to-date" in clause.detail

    def test_fewer_daily_periods_than_days_is_unmeasurable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Short(_CostClient):
            def get_cost_and_usage(self, **request) -> dict:
                if request["Granularity"] == "DAILY":
                    return {
                        "ResultsByTime": [
                            {"Total": {"UnblendedCost": {"Amount": "1.00"}}} for _ in range(3)
                        ]
                    }
                return super().get_cost_and_usage(**request)

        monkeypatch.setattr(gate_module, "_ce_client", lambda: _Short("1.50"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert clause.unmeasurable and not clause.met
        assert "3 daily period(s)" in clause.detail

    def test_a_free_trailing_month_is_unmeasurable_not_a_zero_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Thirty days of exactly `$0.00` is what the tagged scope returns on
        an untagged estate — the `$0.00` trap on the trailing window."""
        client = _CostClient("1.50", daily=["0"])
        monkeypatch.setattr(gate_module, "_ce_client", lambda: client)
        clause = gate_module._clause_aws_cost_within_ceiling(
            [dt.date(2026, 9, 10)],
            name="aws_cost_within_ceiling",
            ceiling_usd=PHASE2_MAX_TAGGED_USD,
            tagged=True,
        )
        assert clause.unmeasurable and not clause.met
        assert "read exactly $0.00" in clause.detail

    def test_unmet_over_the_account_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Phase 4's reading: same reader, no tag filter, the other ceiling."""
        monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostClient("230.21"))
        clause = gate_module._clause_aws_cost_within_ceiling(
            PHASE2_WINDOW,
            name="aws_total_within_ceiling",
            ceiling_usd=PHASE4_MAX_TOTAL_USD,
            tagged=False,
        )
        assert not clause.met and not clause.unmeasurable
        assert "230.21" in clause.detail


# ---------------------------------------------------------------------------
# phase 3
# ---------------------------------------------------------------------------


class TestASlotHoldsItsPointerOnEvidenceOrSaysNothingLooked:
    def test_neither_artifact_is_unmeasurable_not_unmet(self, store: LocalStore) -> None:
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.unmeasurable and not clause.met
        assert champion_key("r") in clause.detail

    def test_met_on_an_evidence_won_promotion_whose_run_reads_ok(self, store: LocalStore) -> None:
        producing = manifest_key("promote", FRIDAY.isoformat())
        _put(
            store,
            champion_key("r"),
            {
                "arm_id": "r:momentum:ab12cd",
                "promotion_source": "evidence",
                "manifest_key": producing,
            },
        )
        _put(store, producing, {"status": "ok", "reason": "", "metrics": []})
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.met and not clause.unmeasurable

    @pytest.mark.parametrize(
        "render_day",
        [
            dt.date(2026, 8, 31),
            dt.date(2026, 9, 1),
            dt.date(2026, 9, 2),
            dt.date(2026, 9, 3),
            dt.date(2026, 9, 4),
        ],
        ids=["mon", "tue", "wed", "thu", "fri"],
    )
    def test_the_promote_manifest_is_found_on_every_render_weekday(
        self, store: LocalStore, render_day: dt.date
    ) -> None:
        """PR80 review, B2 — `alpha-engine-config-I9904` in phase 3. Promote
        runs on Saturday and files at the Friday close; over the raw render
        window the clause found `runs/promote/2026-08-28/run.json` on a
        Friday render only. Phase 3 is now in `WEEKLY_ANCHORED_GATES`."""
        producing = manifest_key("promote", FRIDAY.isoformat())
        _put(
            store,
            champion_key("r"),
            {
                "arm_id": "r:momentum:ab12cd",
                "promotion_source": "evidence",
                "manifest_key": producing,
            },
        )
        _put(store, producing, {"status": "ok", "reason": "", "metrics": []})
        result = evaluate(store, gate="phase3", trading_day=render_day)
        assert result.window[-1] == FRIDAY
        clause = next(
            c for c in result.clauses if c.name == "r_promotion_or_verdict_backed_non_promotion"
        )
        assert clause.met and not clause.unmeasurable, clause.detail

    def test_a_bootstrap_pointer_is_not_a_promotion_the_system_won(self, store: LocalStore) -> None:
        _put(
            store,
            champion_key("r"),
            {
                "arm_id": "r:momentum:ab12cd",
                "promotion_source": "bootstrap",
                "manifest_key": manifest_key("promote", FRIDAY.isoformat()),
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert not clause.met and not clause.unmeasurable
        assert "not `evidence`" in clause.detail

    def test_met_on_a_verdict_backed_non_promotion(self, store: LocalStore) -> None:
        """A pointer that did NOT move, with a stated reason, is the other way
        phase 3's clause is satisfied — the plan grades the DECISION, not the
        change."""
        from crucible.keys import arena_cycle_key

        _put(
            store,
            manifest_key("promote", FRIDAY.isoformat()),
            {
                "status": "ok",
                "reason": "",
                "metrics": [
                    {
                        "name": "pointer_moved",
                        "source_path": arena_cycle_key("r", FRIDAY.isoformat()),
                        "value": 0,
                        "status_reason": "the challenger lost 3 of 4 paired weeks",
                    }
                ],
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.met and not clause.unmeasurable

    def test_a_non_promotion_with_no_stated_reason_is_unmet(self, store: LocalStore) -> None:
        from crucible.keys import arena_cycle_key

        _put(
            store,
            manifest_key("promote", FRIDAY.isoformat()),
            {
                "status": "ok",
                "reason": "",
                "metrics": [
                    {
                        "name": "pointer_moved",
                        "source_path": arena_cycle_key("r", FRIDAY.isoformat()),
                        "value": 0,
                        "status_reason": "",
                    }
                ],
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert not clause.met and not clause.unmeasurable
        assert "status_reason" in clause.detail

    def test_a_filed_manifest_that_names_no_verdict_is_not_reported_as_absent(
        self, store: LocalStore
    ) -> None:
        """A promote run that was filed, read `ok`, and said nothing about this
        slot is a PRODUCER gap, not "nothing looked".

        Reporting it as "no promote run manifest was filed" states something
        false about the store — the manifest is right there — and principle 1
        asks that someone reconstruct why from the artifact alone (PR68
        adversarial review, F2).
        """
        from crucible.keys import arena_cycle_key

        promote = manifest_key("promote", FRIDAY.isoformat())
        _put(
            store,
            promote,
            {
                "status": "ok",
                "reason": "",
                "metrics": [
                    {
                        "name": "pointer_moved",
                        "source_path": arena_cycle_key("m", FRIDAY.isoformat()),
                        "value": 0,
                        "status_reason": "the challenger lost 3 of 4 paired weeks",
                    }
                ],
            },
        )
        clause = gate_module._clause_slot_promotion_or_non_promotion(store, "r", _window(4))
        assert clause.unmeasurable and not clause.met
        assert promote in clause.detail
        assert "pointer_moved" in clause.detail
        assert arena_cycle_key("r", FRIDAY.isoformat()) in clause.detail
        assert "no promote run manifest was filed" not in clause.detail

    def test_one_clause_per_registered_slot(self, store: LocalStore) -> None:
        clauses = gate_module._phase3(store, _window(4), {}, trading_day=FRIDAY)
        assert [c.name for c in clauses] == [
            f"{slot}_promotion_or_verdict_backed_non_promotion" for slot in sorted(SLOTS)
        ]


# ---------------------------------------------------------------------------
# phase 4
# ---------------------------------------------------------------------------


TRADER_EVIDENCE = "consumers/trader/v2_champion_week.json"


class TestTheTraderIsGradedThroughItsContractOrNotAtAll:
    def test_unmeasurable_while_the_contract_declares_no_artifact(self, store: LocalStore) -> None:
        """The harness may not reach into the trader, so an undeclared
        artifact is a missing CONTRACT, not a missing file."""
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert clause.unmeasurable and not clause.met
        assert "declares no consumer-evidence artifact" in clause.detail

    def test_unmet_once_declared_and_absent(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "TRADER_EVIDENCE_KEY", TRADER_EVIDENCE)
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert not clause.met and not clause.unmeasurable
        assert "is absent" in clause.detail

    def test_met_on_a_full_week(self, store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gate_module, "TRADER_EVIDENCE_KEY", TRADER_EVIDENCE)
        _put(store, TRADER_EVIDENCE, {"trading_days": 5})
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert clause.met and not clause.unmeasurable

    def test_unmet_on_a_short_week(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "TRADER_EVIDENCE_KEY", TRADER_EVIDENCE)
        _put(store, TRADER_EVIDENCE, {"trading_days": 2})
        clause = gate_module._clause_trader_week_on_v2_champion(store, _window(2))
        assert not clause.met and not clause.unmeasurable


class TestDecommissionedMeansZeroNotWithinCadence:
    def test_the_phase_four_reading_is_the_phase_zero_reader_at_a_zero_ceiling(
        self, store: LocalStore
    ) -> None:
        """One reader, two ceilings. A second copy of this count is how phase 0
        and phase 4 would come to disagree about what a v1 start is."""
        clause = gate_module._clause_old_weekly_within_cadence(
            store, _window(2), name="old_sf_execution_count_zero", maximum=0
        )
        assert clause.name == "old_sf_execution_count_zero"
        assert not clause.met
        assert "at most 0" in clause.requirement

    def test_one_start_is_unmet_at_the_phase_four_ceiling(self, store: LocalStore) -> None:
        from crucible.gate import legacy_weekly_executions_key, weekly_anchor

        for day in _window(2):
            _put(
                store,
                legacy_weekly_executions_key(weekly_anchor(day).isoformat()),
                {"executions_started": 1},
            )
        clause = gate_module._clause_old_weekly_within_cadence(
            store, _window(2), name="old_sf_execution_count_zero", maximum=0
        )
        assert not clause.met
        assert "ceiling 0" in clause.detail

    def test_zero_starts_is_met(self, store: LocalStore) -> None:
        from crucible.gate import legacy_weekly_executions_key, weekly_anchor

        for day in _window(2):
            _put(
                store,
                legacy_weekly_executions_key(weekly_anchor(day).isoformat()),
                {"executions_started": 0},
            )
        clause = gate_module._clause_old_weekly_within_cadence(
            store, _window(2), name="old_sf_execution_count_zero", maximum=0
        )
        assert clause.met


# ---------------------------------------------------------------------------
# phase 5
# ---------------------------------------------------------------------------


class _CallSite:
    """Enough of a registered call site for the membership test."""


class TestAnEmptyArmSetIsNeverAPass:
    def test_unmeasurable_while_no_llm_call_site_is_registered(self, store: LocalStore) -> None:
        """The empty-set trap, stated as a reading. A property over an empty
        set is vacuously true, and that is how a v1 row went green."""
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.unmeasurable and not clause.met
        assert "empty" in clause.detail

    def test_unmeasurable_while_an_arm_recipe_cannot_name_a_call_site(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second precondition, named separately so the reason is
        actionable: call sites exist, but the register cannot say which arms
        use one. The field exists on `main` since alpha-engine-config-I9920,
        so this is the reading a build that LOST it would give — forced by
        setting the constant back to `None`."""
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", None)
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.unmeasurable and not clause.met
        assert "not " in clause.detail

    def test_the_gate_reads_the_same_params_key_the_loader_validates(self) -> None:
        """One key, two modules: the gate restates it by value so it never
        imports the slot machinery; this is what keeps the two from drifting."""
        from crucible.slots.arms import LLM_CALLSITE_PARAM

        assert gate_module.LLM_ARM_CALLSITE_FIELD == LLM_CALLSITE_PARAM == "llm_callsite"

    def test_unmeasurable_when_no_active_arm_declares_one(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", "callsite_id")
        self._register(store, "r", [("plain", {})])
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.unmeasurable and not clause.met

    def test_unmet_when_an_llm_arm_has_no_verdict_in_the_cycle(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", "callsite_id")
        self._register(store, "r", [("llm", {"callsite_id": "research.rank"})])
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert not clause.met and not clause.unmeasurable
        assert "no verdict" in clause.detail

    def test_met_when_every_llm_arm_has_one(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        monkeypatch.setattr(gate_module, "LLM_ARM_CALLSITE_FIELD", "callsite_id")
        arm_id = self._register(store, "r", [("llm", {"callsite_id": "research.rank"})])
        _put(store, verdict_key(arm_id, FRIDAY.isoformat()), {"status": "ok"})
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert clause.met and not clause.unmeasurable

    def test_an_active_arm_with_no_synced_recipe_is_unmet_not_skipped(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The register says an arm is active; the strategy tree cannot say
        what it is. Silently treating it as a non-LLM arm is how the set this
        property quantifies over would lose exactly the arms it is for."""
        monkeypatch.setattr(llm_module, "LLM_CALLSITE_REGISTRY", {"research.rank": _CallSite()})
        self._register(store, "r", [("llm", {"callsite_id": "research.rank"})], sync=False)
        clause = gate_module._clause_every_llm_arm_has_a_verdict(store, _window(1))
        assert not clause.met and not clause.unmeasurable
        assert "no arm recipes" in clause.detail or "no recipe" in clause.detail

    def test_the_m_recipe_has_no_place_for_a_call_site(self) -> None:
        """What justifies `LLM_ARM_RECIPE_SLOTS` omitting the M slot: the
        `ModelRecipe` contract carries no `params` and no call-site field, so
        no M arm can be an LLM arm by declaration. If either appears, this
        fails and the tuple is widened deliberately."""
        import dataclasses

        from crucible.slots.model import ModelRecipe

        names = {f.name for f in dataclasses.fields(ModelRecipe)}
        assert "params" not in names
        assert gate_module.LLM_ARM_CALLSITE_FIELD not in names
        assert set(gate_module.LLM_ARM_RECIPE_SLOTS) == {"u", "r"}

    @staticmethod
    def _register(
        store: LocalStore, slot: str, arms: list[tuple[str, dict]], *, sync: bool = True
    ) -> str:
        """Register each arm the way production does — a REAL `ArmSpec`, its
        id the hash of its spec, folded into the register (which carries only
        the `spec_hash`) — and sync its recipe into the store's strategy tree,
        which is where the gate reads `params.llm_callsite` from.

        The call-site field is `LLM_ARM_CALLSITE_FIELD` on the spec's params;
        a fixture key of `callsite_id` is rewritten to it so the cases above
        read as "this arm names that site".
        """
        from nousergon_lib.arena.arms import ArmRegister

        from crucible.slots import get_slot
        from crucible.slots.arms import ArmSpec, control_specs, write_register

        register = ArmRegister()
        # Production registers the slot's two controls beside every filed arm
        # (`crucible.slots.cycle`, both call sites), and controls are never in
        # the strategy tree — so the fixture carries them too, or the join
        # below is proven only on a register production never writes.
        for control in control_specs(get_slot(slot)):
            register, _ = register.register(
                slot=slot,
                name=control.name,
                spec=control.spec,
                created_date=control.registered_at,
            )
        arm_id = ""
        for name, params in arms:
            field = gate_module.LLM_ARM_CALLSITE_FIELD
            params = {(field if k == "callsite_id" and field else k): v for k, v in params.items()}
            spec = ArmSpec(
                name=name,
                slot=slot,
                ranker="momentum_sleeve",
                params={"top_n": 8, **params},
                registered_at="2026-01-02",
            )
            register, _ = register.register(
                slot=slot, name=name, spec=spec.spec, created_date=spec.registered_at
            )
            arm_id = spec.arm_id
            if sync:
                lines = [
                    f"name: {name}",
                    f"slot: {slot}",
                    "ranker: momentum_sleeve",
                    "registered_at: '2026-01-02'",
                    "params:",
                ] + [f"  {k}: {json.dumps(v)}" for k, v in spec.params.items()]
                store.put_bytes(
                    strategy_arm_key(slot, name), ("\n".join(lines) + "\n").encode("utf-8")
                )
        write_register(store, slot, register)
        return arm_id


# ---------------------------------------------------------------------------
# the cost reader itself
# ---------------------------------------------------------------------------


class TestCostExplorerReadsRaiseRatherThanReturnZero:
    def test_a_denied_call_raises_rather_than_returning_zero(self) -> None:
        class _Denied:
            def get_cost_and_usage(self, **_):
                raise RuntimeError("AccessDeniedException")

        with pytest.raises(cost_module.CostUnreadableError) as caught:
            cost_module.month_to_date_usd(_Denied(), today=FRIDAY, tagged=True)
        assert "AccessDeniedException" in str(caught.value)

    def test_an_empty_response_is_not_a_spend_of_zero(self) -> None:
        class _Empty:
            def get_cost_and_usage(self, **_):
                return {"ResultsByTime": []}

        with pytest.raises(cost_module.CostUnreadableError):
            cost_module.month_to_date_usd(_Empty(), today=FRIDAY, tagged=False)

    def test_the_first_of_the_month_asks_for_a_non_empty_interval(self) -> None:
        """Cost Explorer's `End` is exclusive and must be strictly after
        `Start`, so a naive month-to-date reads `$0.00` on exactly one day a
        month."""
        captured: dict = {}

        class _Capturing:
            def get_cost_and_usage(self, **request):
                captured.update(request)
                return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": "1.00"}}}]}

        reading = cost_module.month_to_date_usd(
            _Capturing(), today=dt.date(2026, 9, 1), tagged=False
        )
        assert captured["TimePeriod"] == {"Start": "2026-09-01", "End": "2026-09-02"}
        assert reading.amount_usd == 1.0

    def test_the_tag_filter_is_the_only_difference_between_the_two_scopes(self) -> None:
        captured: list[dict] = []

        class _Capturing:
            def get_cost_and_usage(self, **request):
                captured.append(request)
                return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": "2.00"}}}]}

        client = _Capturing()
        cost_module.month_to_date_usd(client, today=FRIDAY, tagged=True)
        cost_module.month_to_date_usd(client, today=FRIDAY, tagged=False)
        assert "Filter" in captured[0]
        assert "Filter" not in captured[1]
        assert captured[0]["TimePeriod"] == captured[1]["TimePeriod"]
