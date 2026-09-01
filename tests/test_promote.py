"""The pointer decision, the eligibility age, and the retirement record.

Normative sources: `champion-challenger-policy.md` §5.0/§5.2/§5.3/§6, plan
§4.4 and §4.12 (Brian ruling 2026-09-01: `promote_min_weeks` = 4 paired
weeks = 20 paired TRADING days).

Every test here builds synthetic per-date series and asserts on the decision
the library engine reaches plus the one thing crucible adds on top of it —
the eligibility age. Nothing in this file re-implements a statistic: if a
test could pass against a crucible-local copy of the confidence sequence,
it is testing the wrong thing.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace

import pytest
from nousergon_lib.arena import ArmRegister, ArmSeries, ServingPrecondition, TrainingStatus
from nousergon_lib.arena.engine import TrainingIntegrityError

from crucible.calendar import TRADING_DAYS_PER_WEEK, is_trading_day
from crucible.promote import (
    PROMOTION_SOURCES,
    PromotionRefused,
    apply_eligibility_age,
    paired_days_required,
    run_promotion,
)
from crucible.slots import get_slot
from crucible.store import LocalStore

# --------------------------------------------------------------------------
# Synthetic trading-day series builders.
# --------------------------------------------------------------------------


def trading_days(n: int, end: dt.date = dt.date(2026, 8, 28)) -> list[str]:
    """``n`` NYSE sessions ending at ``end``, oldest first.

    Real sessions, not `end - timedelta(i)`: a synthetic series keyed on a
    Saturday would be refused by the §4.12 walk, and a test that could only
    pass on fabricated dates would not be exercising the contract.
    """
    out: list[str] = []
    day = end
    while len(out) < n:
        if is_trading_day(day):
            out.append(day.isoformat())
        day -= dt.timedelta(days=1)
    return list(reversed(out))


def series(arm_id: str, dates: list[str], value: float) -> ArmSeries:
    return ArmSeries(arm_id=arm_id, scores={d: value for d in dates})


def narrow(spec, clip: float = 0.01):
    """The slot's own spec with a smaller declared difference clip.

    The confidence sequence's radius scales with the DECLARED clip, so a test
    that wants a supported lead at 19 and at 20 paired dates has to declare a
    scale the lead is large against. Narrowing the clip rather than inflating
    the scores keeps the eligibility test about the AGE: both windows support
    the lead, and only the age separates them.
    """
    return replace(spec, diff_clip=clip)


def register_with(slot: str, names: list[str], created: str) -> tuple[ArmRegister, dict[str, str]]:
    reg = ArmRegister()
    ids: dict[str, str] = {}
    for name in names:
        reg, record = reg.register(slot=slot, name=name, spec={"name": name}, created_date=created)
        ids[name] = record.arm_id
    return reg, ids


# --------------------------------------------------------------------------
# The eligibility age — Brian's 2026-09-01 ruling, in trading days.
# --------------------------------------------------------------------------


class TestEligibilityAge:
    def test_four_paired_weeks_is_twenty_paired_trading_days(self) -> None:
        for slot in ("u", "r", "m", "s"):
            spec = get_slot(slot)
            assert spec.promote_min_weeks == 4
            assert paired_days_required(spec) == 4 * TRADING_DAYS_PER_WEEK == 20

    def test_nineteen_paired_days_is_ineligible_even_with_a_supported_lead(self) -> None:
        spec = narrow(get_slot("m"))
        dates = trading_days(19)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        decision = apply_eligibility_age(
            spec=spec,
            register=reg,
            decision=_decide(spec, reg, ids, dates, champ=0.0, chal=0.01),
        )
        assert decision.champion == ids["champ"]
        assert decision.moved is False
        assert decision.status == "held"
        assert "promote_min_weeks" in decision.reason
        assert "19" in decision.reason

    def test_twenty_paired_days_with_a_supported_lead_promotes(self) -> None:
        spec = narrow(get_slot("m"))
        dates = trading_days(20)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        decision = apply_eligibility_age(
            spec=spec,
            register=reg,
            decision=_decide(spec, reg, ids, dates, champ=0.0, chal=0.01),
        )
        assert decision.champion == ids["chal"]
        assert decision.moved is True
        assert decision.status == "decided"

    def test_the_age_never_holds_a_pointer_the_sequence_did_not_move(self) -> None:
        """The age can delay a promotion; it can never cause one, and it can
        never change a HOLD into anything else (policy §5.0 delta record)."""
        spec = narrow(get_slot("m"))
        dates = trading_days(40)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        held = _decide(spec, reg, ids, dates, champ=0.01, chal=0.0)
        after = apply_eligibility_age(spec=spec, register=reg, decision=held)
        assert after.champion == held.champion
        assert after.moved == held.moved
        assert after.reason == held.reason


# --------------------------------------------------------------------------
# The pointer moves in BOTH directions (policy §5.2).
# --------------------------------------------------------------------------


class TestPointerMovesBothDirections:
    def test_pointer_moves_to_the_challenger_then_back(self) -> None:
        spec = narrow(get_slot("m"))
        dates = trading_days(40)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])

        forward = apply_eligibility_age(
            spec=spec,
            register=reg,
            decision=_decide(spec, reg, ids, dates, champ=0.0, chal=0.01, incumbent="champ"),
        )
        assert forward.champion == ids["chal"]
        assert forward.moved is True

        back = apply_eligibility_age(
            spec=spec,
            register=reg,
            decision=_decide(spec, reg, ids, dates, champ=0.01, chal=0.0, incumbent="chal"),
        )
        assert back.champion == ids["champ"], (
            "policy §5.2: no hysteresis, no cooldown — the pointer returns as soon "
            "as the cumulative window supports the other arm"
        )
        assert back.moved is True


# --------------------------------------------------------------------------
# Serving preconditions (policy §5.3).
# --------------------------------------------------------------------------


class TestServingPreconditions:
    def test_a_vetoed_arm_leading_the_ladder_does_not_serve(self) -> None:
        spec = get_slot("m")
        dates = trading_days(40)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        series_by_arm = {
            ids["champ"]: series(ids["champ"], dates, 0.0),
            ids["chal"]: series(ids["chal"], dates, 0.045),
        }
        cycle = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["champ"],
            preconditions={
                ids["chal"]: (
                    ServingPrecondition(
                        name="behavioural_veto",
                        passed=False,
                        reason="prediction spread collapsed",
                    ),
                )
            },
        ).cycle
        assert cycle.decision.champion == ids["champ"]
        assert ids["chal"] in cycle.decision.ineligible


# --------------------------------------------------------------------------
# Training integrity (policy §3) — a slot failure, never a miss.
# --------------------------------------------------------------------------


class TestTrainingIntegrity:
    def test_one_unsound_fit_fails_the_whole_slot(self) -> None:
        spec = get_slot("m")
        dates = trading_days(25)
        reg, ids = register_with("m", ["a", "b"], dates[0])
        with pytest.raises(TrainingIntegrityError):
            run_promotion(
                spec=spec,
                as_of=dates[-1],
                register=reg,
                series_by_arm={
                    ids["a"]: series(ids["a"], dates, 0.01),
                    ids["b"]: series(ids["b"], dates, 0.0),
                },
                incumbent=ids["a"],
                training={
                    ids["a"]: TrainingStatus(ids["a"], True),
                    ids["b"]: TrainingStatus(ids["b"], False, "seven features hard-zeroed"),
                },
            )

    def test_an_unasserted_fit_is_as_fatal_as_a_failed_one(self) -> None:
        spec = get_slot("m")
        dates = trading_days(25)
        reg, ids = register_with("m", ["a", "b"], dates[0])
        with pytest.raises(TrainingIntegrityError):
            run_promotion(
                spec=spec,
                as_of=dates[-1],
                register=reg,
                series_by_arm={
                    ids["a"]: series(ids["a"], dates, 0.01),
                    ids["b"]: series(ids["b"], dates, 0.0),
                },
                incumbent=ids["a"],
                training={ids["a"]: TrainingStatus(ids["a"], True)},
            )


# --------------------------------------------------------------------------
# Retirement (policy §6) — every active arm gets a verdict, survivors included.
# --------------------------------------------------------------------------


class TestRetirement:
    def test_never_retires_the_champion_and_never_below_the_floor(self) -> None:
        spec = get_slot("m")
        dates = trading_days(40)
        names = ["a", "b", "c"]
        reg, ids = register_with("m", names, dates[0])
        scores = {"a": 0.05, "b": 0.0, "c": -0.05}
        result = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm={ids[n]: series(ids[n], dates, scores[n]) for n in names},
            incumbent=ids["a"],
        )
        retired = [v for v in result.cycle.retirements if v.retire]
        assert retired == [], "policy §6.1: min_active_arms=3 — a three-arm pool can retire nobody"
        assert {v.arm_id for v in result.cycle.retirements} == set(ids.values()), (
            "policy §6.1: every ACTIVE arm receives a recorded verdict each cycle, "
            "survivors included — a retirement list containing only retirements "
            "cannot be audited"
        )
        assert all(v.reason for v in result.cycle.retirements)

    def test_retirement_events_are_appended_never_rewritten(self, tmp_path) -> None:
        spec = get_slot("m")
        dates = trading_days(40)
        names = ["a", "b", "c"]
        reg, ids = register_with("m", names, dates[0])
        store = LocalStore(tmp_path)
        args = dict(
            spec=spec,
            register=reg,
            series_by_arm={
                ids[n]: series(ids[n], dates, v)
                for n, v in zip(names, (0.05, 0.0, -0.05), strict=True)
            },
            incumbent=ids["a"],
            store=store,
        )
        run_promotion(as_of=dates[-2], **args)
        first = store.get_bytes("retirements/m/events.jsonl").decode()
        run_promotion(as_of=dates[-1], **args)
        second = store.get_bytes("retirements/m/events.jsonl").decode()
        assert second.startswith(first), "the retirement log is append-only"
        assert len(second.splitlines()) == 2 * len(names)


# --------------------------------------------------------------------------
# The champion pointer artifact.
# --------------------------------------------------------------------------


class TestChampionPointer:
    def test_promotion_source_is_declared_and_closed(self) -> None:
        assert PROMOTION_SOURCES == ("evidence", "operator_bootstrap", "bootstrap")

    def test_the_pointer_records_its_evidence(self, tmp_path) -> None:
        from crucible.champion import champion_key

        spec = get_slot("m")
        dates = trading_days(40)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        store = LocalStore(tmp_path)
        run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm={
                ids["champ"]: series(ids["champ"], dates, 0.0),
                ids["chal"]: series(ids["chal"], dates, 0.045),
            },
            incumbent=ids["champ"],
            store=store,
            manifest_key=f"runs/promote/{dates[-1]}/run.json",
        )
        pointer = json.loads(store.get_bytes(champion_key("m")))
        assert pointer["arm_id"] == ids["chal"]
        assert pointer["promotion_source"] == "evidence"
        assert pointer["evidence"]["confidence_sequence"]["lower"] > 0
        assert pointer["evidence"]["paired_dates"] >= 20

    def test_revert_records_the_operator_as_the_source(self, tmp_path) -> None:
        from crucible.champion import champion_key
        from crucible.promote import revert_champion

        spec = get_slot("m")
        dates = trading_days(40)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        store = LocalStore(tmp_path)
        revert_champion(
            spec=spec,
            register=reg,
            store=store,
            arm_id=ids["champ"],
            as_of=dates[-1],
            operator="cipher813",
            reason="challenger degraded live",
        )
        pointer = json.loads(store.get_bytes(champion_key("m")))
        assert pointer["arm_id"] == ids["champ"]
        assert pointer["promotion_source"] == "operator_bootstrap"
        assert pointer["evidence"]["operator"] == "cipher813"

    def test_revert_to_an_unregistered_arm_is_refused(self, tmp_path) -> None:
        from crucible.promote import revert_champion

        spec = get_slot("m")
        dates = trading_days(40)
        reg, _ = register_with("m", ["champ"], dates[0])
        with pytest.raises(PromotionRefused):
            revert_champion(
                spec=spec,
                register=reg,
                store=LocalStore(tmp_path),
                arm_id="m:ghost:deadbeef",
                as_of=dates[-1],
                operator="cipher813",
                reason="typo",
            )


# --------------------------------------------------------------------------
# Negative results reach the EXPERIMENTS feed (plan §9.1).
# --------------------------------------------------------------------------


class TestExperimentsFeed:
    def test_a_held_pointer_appends_a_negative_result(self, tmp_path) -> None:
        spec = get_slot("m")
        dates = trading_days(40)
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        store = LocalStore(tmp_path)
        run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm={
                ids["champ"]: series(ids["champ"], dates, 0.045),
                ids["chal"]: series(ids["chal"], dates, 0.0),
            },
            incumbent=ids["champ"],
            store=store,
        )
        rows = [
            json.loads(line)
            for line in store.get_bytes(f"experiments/{dates[-1]}/events.jsonl")
            .decode()
            .splitlines()
        ]
        kinds = {r["kind"] for r in rows}
        assert "negative_result" in kinds, (
            "plan §9.1: a challenger that did not win is a recorded negative result, "
            "generated from the cycle rather than hand-written into a private doc"
        )


# --------------------------------------------------------------------------
# Helper: one library pointer decision, with no crucible statistic in sight.
# --------------------------------------------------------------------------


def _decide(spec, register, ids, dates, *, champ, chal, incumbent="champ"):
    """One library pointer decision. ``champ``/``chal`` are the per-date
    scores of the arms named "champ" and "chal", whichever of them is the
    incumbent this cycle — so a reversal is written by swapping the scores,
    not by swapping which arm holds which series."""
    from nousergon_lib.arena.engine import decide_pointer

    return decide_pointer(
        config=spec.arena,
        as_of=dates[-1],
        incumbent=ids[incumbent],
        series_by_arm={
            ids["champ"]: series(ids["champ"], dates, champ),
            ids["chal"]: series(ids["chal"], dates, chal),
        },
    )
