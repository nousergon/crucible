"""A first champion is won on evidence, in `experiment.grade`.

`alpha-engine-config-I9759` gave a slot with no champion a first champion
won on evidence rather than seated on the library's §9.1 cold start, by
substituting §10.1's NULL control as the baseline incumbent before calling
`nousergon_lib.arena.engine.run_cycle`. That substitution lived inside
`crucible.promote.run_promotion`, which called `run_cycle` itself.

`alpha-engine-config-I10679` removed promote's `run_cycle` call — promote
now acts on the cycle `experiment.grade` already computed — and the
substitution went with it, leaving no slot able to win a first champion:
grade produced a raw `bootstrap` decision and `promote` refused it.

`alpha-engine-config-I10687` moves the substitution to the one writer that
computes the cycle, `crucible.slots.cycle.run_grade`. These tests pin the
WIN, the exemption that makes it possible, the record that makes it
readable, and the regression that the baseline is never itself promoted.
"""

from __future__ import annotations

import json

from conftest import sessions_ending

from crucible.champion import (
    CHAMPION_SCHEMA_VERSION,
    ChampionPointer,
    read_champion_etag,
    write_champion,
)
from crucible.config import Settings
from crucible.data import run_daily
from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.keys import arena_cycle_key
from crucible.runner import run_job
from crucible.slots import get_slot, universe
from crucible.slots.arms import control_specs, load_arm_specs
from crucible.slots.cycle import (
    BASELINE_CONTROL_KIND,
    INCUMBENT_SOURCE_FIELD,
    baseline_control_arm,
)

HORIZON = 21
DECISION_DATES = 6


def _control_ids(slot: str) -> dict[str, str]:
    """Kind -> registered arm id, exactly as `run_grade` addresses them."""
    return {c.control_kind: c.arm_id for c in control_specs(get_slot(slot))}


class TestBaselineControlArm:
    """The baseline is matched on the control's KIND, never on the
    register's `control` flag — the flag says *whether* an arm is a control
    and this needs *which kind*. The PLANTED control must never be the
    baseline: it ranks on the realized forward return, so an arm that merely
    beat it would still be a look-ahead-relative measurement."""

    def test_the_null_control_is_the_baseline(self) -> None:
        spec = get_slot("u")
        ids = _control_ids("u")
        series_by_arm = dict.fromkeys([ids["planted"], ids["null"], "u:real:0123456789ab"], None)
        assert baseline_control_arm(spec, series_by_arm) == ids["null"]

    def test_the_planted_control_is_never_the_baseline(self) -> None:
        spec = get_slot("u")
        ids = _control_ids("u")
        series_by_arm = dict.fromkeys([ids["planted"], "u:real:0123456789ab"], None)
        assert baseline_control_arm(spec, series_by_arm) is None

    def test_an_unscored_null_control_is_no_baseline(self) -> None:
        """A slot with nothing scored yet is not an error, it is an empty
        slot: `None` here, and the engine answers it with its own stated
        reason rather than this function inventing a substitute."""
        assert baseline_control_arm(get_slot("u"), {}) is None

    def test_the_baseline_kind_is_the_null_control(self) -> None:
        assert BASELINE_CONTROL_KIND == "null"


def _pointer(slot: str, arm_id: str, *, as_of: str) -> ChampionPointer:
    return ChampionPointer(
        schema_version=CHAMPION_SCHEMA_VERSION,
        slot=slot,
        arm_id=arm_id,
        as_of=as_of,
        decided_at=f"{as_of}T02:00:00Z",
        run_id="01JG0000000000000000000000",
        code_sha="a" * 40,
        promotion_source="evidence",
        manifest_key=f"runs/promote/{as_of}/run.json",
        evidence={"status": "decided", "moved": True, "paired_dates": 40},
        attestation=None,
    )


class TestRunGradeSubstitutesTheBaseline:
    """The integration seam, reusing `tests/test_cycle_incumbent.py`'s shape:
    produce a few U-slot decision days, grade with and without a seated
    champion, and assert on the cycle the job itself emitted."""

    def _grade(self, store, source, strategy_dir, cycle_date, tmp_path, *, seed_champion: bool):
        settings = Settings(
            store_uri=str(tmp_path / "store"),
            arctic_bucket="unused-in-this-test",
            strategy_dir=strategy_dir,
            origins={"store_uri": "test", "strategy_dir": "test"},
        )
        sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
        decision_days = sessions[:DECISION_DATES]

        for day in decision_days + [cycle_date]:
            run_job(
                "data.daily",
                lambda c, day=day: run_daily(
                    c,
                    point_in_time=UnavailablePointInTimeSource(
                        reason="synthetic fixture market carries no fundamentals"
                    ),
                    source=source,
                    expected_symbols=source.symbols(),
                ),
                store=store,
                trading_day=day,
            )
        for day in decision_days:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=day,
            )
        if seed_champion:
            seeded = load_arm_specs("u", strategy_dir=strategy_dir)[0].arm_id
            write_champion(
                store,
                _pointer("u", seeded, as_of=decision_days[-1].isoformat()),
                expected=read_champion_etag(store, "u"),
            )

        result: dict = {}

        def job(ctx):
            result.update(universe.grade(ctx, settings=settings))

        run_job("experiment.grade", job, store=store, trading_day=cycle_date)
        cycle = json.loads(store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())))
        return result, cycle

    def test_a_cold_slot_grades_against_the_null_control_not_a_cold_start(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """RED before `alpha-engine-config-I10687`: `run_grade` called
        `run_cycle` with `incumbent=None`, so the library took §9.1 and the
        decision read `bootstrap` — the one pointer kind §6 rejects by name.
        """
        result, cycle = self._grade(
            store, source, strategy_dir, cycle_date, tmp_path, seed_champion=False
        )
        null_control = _control_ids("u")["null"]
        assert cycle["decision"]["incumbent"] == null_control
        assert cycle["decision"]["status"] != "bootstrap"
        assert result["pointer"]["incumbent"] == null_control

    def test_the_cycle_records_that_the_incumbent_was_a_substituted_baseline(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """Transparency (`principles.md` §2.1): a reader of the artifact can
        tell a slot that GRADED AGAINST NOISE from one that graded against a
        seated champion, without reconstructing it from the arm id."""
        result, cycle = self._grade(
            store, source, strategy_dir, cycle_date, tmp_path, seed_champion=False
        )
        record = cycle[INCUMBENT_SOURCE_FIELD]
        assert record["source"] == "baseline_control"
        assert record["arm_id"] == _control_ids("u")["null"]
        assert record["baseline_control_kind"] == BASELINE_CONTROL_KIND
        assert record["reason"]
        assert result[INCUMBENT_SOURCE_FIELD] == record

    def test_the_baseline_is_exempt_from_the_control_veto(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """It has to be ELIGIBLE to be the incumbent: the engine forces the
        pointer off an incumbent that fails a precondition, which would turn
        a cold slot's first cycle into an unbarred promotion of whatever
        ranked first. The PLANTED control is never exempt."""
        _, cycle = self._grade(
            store, source, strategy_dir, cycle_date, tmp_path, seed_champion=False
        )
        ids = _control_ids("u")
        ineligible = cycle["decision"]["ineligible"]
        assert ids["null"] not in ineligible
        assert ids["planted"] in ineligible
        assert any(c["name"] == "not_a_control_arm" for c in ineligible[ids["planted"]])

    def test_the_baseline_is_never_itself_promoted(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """Regression for §10.1 under the exemption. A challenger is never
        compared to itself, so `moved` is false whenever the baseline
        "wins"; and the baseline stays out of the promotable pool, so no
        later job can seat it either."""
        result, cycle = self._grade(
            store, source, strategy_dir, cycle_date, tmp_path, seed_champion=False
        )
        null_control = _control_ids("u")["null"]
        decision = cycle["decision"]
        if decision["champion"] == null_control:
            assert decision["moved"] is False, (
                "the baseline holding the pointer is a NON-promotion; a `moved` "
                "decision naming a control arm is a look-ahead one write away "
                "from the trader contract"
            )
        assert null_control not in result["promotable_arms"]
        assert null_control not in cycle["promotable_arms"]

    def test_a_seated_champion_is_still_the_incumbent_and_the_null_is_vetoed(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """The substitution applies ONLY when there is no champion
        (`alpha-engine-config-I10686` made the real read work; this must not
        displace it), and with a champion seated the null control is a
        control again — vetoed like any other."""
        seeded = load_arm_specs("u", strategy_dir=strategy_dir)[0].arm_id
        _, cycle = self._grade(
            store, source, strategy_dir, cycle_date, tmp_path, seed_champion=True
        )
        ids = _control_ids("u")
        assert cycle["decision"]["incumbent"] == seeded
        assert cycle[INCUMBENT_SOURCE_FIELD]["source"] == "champion_pointer"
        assert cycle[INCUMBENT_SOURCE_FIELD]["baseline_control_kind"] is None
        assert ids["null"] in cycle["decision"]["ineligible"]
