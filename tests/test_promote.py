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
from crucible.slots import get_slot, is_control_arm
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


def register_with(
    slot: str,
    names: list[str],
    created: str,
    *,
    control_names: frozenset[str] = frozenset(),
) -> tuple[ArmRegister, dict[str, str]]:
    """Register ``names`` under ``slot``, each as ``control=False`` unless it
    appears in ``control_names`` (`alpha-engine-config-I10044`: once
    `is_control_arm` is register-backed, a fixture that wants the register's
    own record to say "control" has to register it that way rather than
    relying on the arm's NAME happening to collide with
    `SlotSpec.control_arms` — the fallback the register-backed path now
    overrides).
    """
    reg = ArmRegister()
    ids: dict[str, str] = {}
    for name in names:
        reg, record = reg.register(
            slot=slot,
            name=name,
            spec={"name": name},
            created_date=created,
            control=name in control_names,
        )
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


# --------------------------------------------------------------------------
# F3: the retirement reaches the REGISTER, not only the event log.
# --------------------------------------------------------------------------


def seed_register(store: LocalStore, slot: str, register: ArmRegister) -> None:
    """Put ``register``'s event log where `load_slot_inputs` reads it.

    The retirement is appended to the same document the next cycle folds, so
    a test that kept the register only in memory would be asserting against
    a store the production path never has.
    """
    from crucible.keys import arm_register_key

    store.put_bytes(
        arm_register_key(slot),
        b"".join(json.dumps(e, sort_keys=True).encode() + b"\n" for e in register.to_dicts()),
    )


def seed_series(store: LocalStore, slot: str, series_by_arm: dict) -> None:
    """Put every arm's score series where `load_slot_inputs` reads it.

    Written for every arm, retired ones included: policy §3 keeps a retired
    arm scored for its trailing window, and the loader refuses a registered
    arm with no series rather than silently presenting a smaller cohort.
    """
    from crucible.promote import arm_series_key

    for arm_id, arm_series in series_by_arm.items():
        store.put_bytes(
            arm_series_key(slot, arm_id),
            json.dumps({"arm_id": arm_id, "scores": arm_series.scores, "misses": []}).encode(),
        )


def four_arm_slot(store: LocalStore, dates: list[str]):
    """A cap-2, grace-1 slot where one arm is beaten by three.

    The engine's Condorcet rule (policy §6.1/§6.2) then returns a RETIRE
    verdict, which is the precondition for the defect: before this fix the
    verdict was written to `retirements/m/events.jsonl` and to the
    EXPERIMENTS feed, and to nothing any later cycle reads.
    """
    spec = replace(get_slot("m"), cap=2, grace_weeks=1, min_active_arms=2, diff_clip=0.01)
    names = ["a", "b", "c", "loser"]
    reg, ids = register_with("m", names, dates[0])
    seed_register(store, "m", reg)
    scores = {"a": 0.05, "b": 0.04, "c": 0.03, "loser": -0.05}
    series_by_arm = {ids[n]: series(ids[n], dates, scores[n]) for n in names}
    seed_series(store, "m", series_by_arm)
    return spec, reg, ids, series_by_arm


class TestRetirementReachesTheRegister:
    """F3 (`alpha-engine-config-I9757`).

    Reproduced against the unfixed code with cap=2, grace_weeks=1 and one arm
    beaten by three: the engine said RETIRE, the keys written were the arena
    cycle, the retirement event log and the experiments log — and
    `arms/m/register.jsonl` was not among them. `ArmRegister.retire()` was
    called nowhere in `crucible/` or `tests/`. Because `load_slot_inputs`
    rebuilds the register from that log, the retired arm was fully active the
    next cycle, was re-retired every week forever, `retired_trailing_cycles`
    never started counting, and `revert_champion`'s `retired_date is not
    None` guard could never fire.
    """

    def test_a_retired_arm_is_absent_from_active_arms_next_cycle(self, tmp_path) -> None:
        from crucible.promote import load_slot_inputs

        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, series_by_arm = four_arm_slot(store, dates)

        result = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["a"],
            store=store,
        )
        retired = {v.arm_id for v in result.cycle.retirements if v.retire}
        assert retired, "the fixture must actually produce a retirement to be a test of one"
        assert "arms/m/register.jsonl" in result.keys_written

        next_cycle = load_slot_inputs(store, "m").register
        for arm in retired:
            assert arm not in next_cycle.active_arms(), (
                "policy §6: a retirement the register never records is re-decided "
                "every cycle forever and removes the arm from nothing"
            )
            state = next_cycle.state(arm)
            assert state.retired_date == dates[-1], "§6.3: retired_date is a queryable fact"
            assert state.retired_reason

    def test_re_running_the_same_cycle_does_not_double_retire(self, tmp_path) -> None:
        """The runner retries a transient failure by re-entering the whole
        job. A second `retired` event for one arm does not merely duplicate a
        line — `ArmRegister._fold` raises `ImmutableArmError` on it, so the
        slot would be unloadable from the next cycle onward."""
        from crucible.promote import load_slot_inputs

        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, series_by_arm = four_arm_slot(store, dates)
        args = dict(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["a"],
            store=store,
        )
        run_promotion(**args)
        first = store.get_bytes("arms/m/register.jsonl").decode()
        run_promotion(**args)  # the re-entered pass, holding the SAME stale register
        second = store.get_bytes("arms/m/register.jsonl").decode()

        assert second == first, "a re-entered pass appends nothing it already appended"
        load_slot_inputs(store, "m")  # folds without raising

    def test_a_retired_arm_is_still_scored_for_its_trailing_window(self, tmp_path) -> None:
        """Policy §3/§6.3: retirement removes an arm from the ACTIVE pool and
        from the cap; it does not stop it being measured, or "we retired the
        wrong one" stops being detectable."""
        from crucible.promote import load_slot_inputs

        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, series_by_arm = four_arm_slot(store, dates)
        run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["a"],
            store=store,
        )
        after = load_slot_inputs(store, "m").register
        retired = [a for a in after.all_arms() if not after.state(a).active]
        assert retired
        scored = after.scored_arms(dates[-1], spec.retired_trailing_cycles)
        assert set(retired) <= set(scored)


# --------------------------------------------------------------------------
# F4: a control arm never reaches `champions/{slot}/current.json`.
# --------------------------------------------------------------------------


class TestControlArmsNeverServe:
    """F4 (`alpha-engine-config-I9757`).

    Reproduced against the unfixed code by registering `control_planted_m`
    with the best series in the slot: the decision came back `decided
    moved=True`, the champion was the planted control, and the pointer was
    written as `m:control_planted_m:7e8059f49558`. The planted control reads
    next-period returns by construction (plan §10.1), so this put a
    look-ahead arm on the single contract the trader reads.

    Two defects stacked. `_write_pointer_if_moved` never called
    `promotable_arms` at all — its only call site was a reported field on the
    cycle artifact, not an exclusion — and `promotable_arms` matched the
    `ControlArm.arm_id` literal against a registered `derive_arm_id` hash
    form, so it could not have matched even where it was called.
    """

    def _slot_with_a_winning_control(self, dates: list[str]):
        """The planted control OUT-SCORES both real arms, and the better real
        arm has a supported lead of its own.

        Both halves matter. Without the first, the fixture never puts the
        control in front and the exclusion is never exercised — measured:
        with the control on 0.045 and `real_b` on 0.01 the engine held on
        `real_a` either way, and the test passed against the unfixed code.
        Without the second, removing the control would leave nothing to
        promote, and "the pointer is not the control" would be satisfied by a
        slot that promoted nobody.
        """
        spec = get_slot("m")
        control = spec.control_arms[0].arm_id  # control_planted_m
        reg, ids = register_with(
            "m", ["real_a", "real_b", control], dates[0], control_names=frozenset({control})
        )
        series_by_arm = {
            ids["real_a"]: series(ids["real_a"], dates, 0.0),
            ids["real_b"]: series(ids["real_b"], dates, 0.04),
            ids[control]: series(ids[control], dates, 0.045),
        }
        return spec, reg, ids, control, series_by_arm

    def test_the_promotion_path_never_writes_a_control_as_champion(self, tmp_path) -> None:
        from crucible.champion import champion_key

        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, control, series_by_arm = self._slot_with_a_winning_control(dates)
        seed_register(store, "m", reg)

        result = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["real_a"],
            store=store,
        )

        assert result.decision.champion != ids[control]
        assert result.decision.champion == ids["real_b"], (
            "the control is barred from SERVING only; the best real arm still wins"
        )
        pointer = json.loads(store.get_bytes(champion_key("m")))
        assert pointer["arm_id"] == ids["real_b"]
        assert pointer["arm_id"] != ids[control]

    def test_the_control_is_vetoed_at_the_decision_and_the_reason_is_recorded(
        self, tmp_path
    ) -> None:
        """Excluded at the decision rather than at the write, so the artifact
        says WHY the control did not serve. A writer that silently declined
        would leave every artifact recording a promotion that did not
        happen — policy §7.2's dominant bug class, self-inflicted."""
        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, control, series_by_arm = self._slot_with_a_winning_control(dates)
        seed_register(store, "m", reg)

        cycle = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["real_a"],
            store=store,
        ).cycle

        assert ids[control] in cycle.decision.ineligible
        vetoes = cycle.decision.ineligible[ids[control]]
        assert any(v.name == "not_a_control_arm" and not v.passed for v in vetoes)
        assert any("next-period returns" in v.reason for v in vetoes)

    def test_the_control_is_still_scored_and_laddered(self, tmp_path) -> None:
        """§10.1: a control's whole purpose is being graded beside the real
        arms. Barring it from serving must not bar it from measurement, or
        the grader-integrity check it exists for stops running."""
        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, control, series_by_arm = self._slot_with_a_winning_control(dates)
        seed_register(store, "m", reg)

        cycle = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["real_a"],
            store=store,
        ).cycle

        assert ids[control] in cycle.scored_arms
        assert ids[control] in {ladder.arm_id for ladder in cycle.ladders}

    def test_a_caller_supplied_precondition_is_kept_alongside_the_control_veto(
        self, tmp_path
    ) -> None:
        """An arm can fail more than one gate. Replacing the caller's checks
        with the control veto would hide a behavioural veto (§5.3) behind a
        control flag."""
        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, control, series_by_arm = self._slot_with_a_winning_control(dates)
        seed_register(store, "m", reg)

        cycle = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["real_a"],
            store=store,
            preconditions={
                ids[control]: (
                    ServingPrecondition(name="behavioural_veto", passed=False, reason="collapsed"),
                )
            },
        ).cycle

        names = {v.name for v in cycle.decision.ineligible[ids[control]]}
        assert names == {"behavioural_veto", "not_a_control_arm"}

    def test_the_writer_refuses_a_control_even_if_the_decision_names_one(self, tmp_path) -> None:
        """The guard on the write itself. It is unreachable while the
        decision-layer exclusion holds, and it is what makes the failure loud
        rather than a pointer silently left on the previous arm if that
        exclusion is ever removed."""
        from crucible.promote import _write_pointer_if_moved
        from crucible.store import ETAG_ABSENT

        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, control, series_by_arm = self._slot_with_a_winning_control(dates)
        seed_register(store, "m", reg)

        cycle = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["real_a"],
        ).cycle
        forged = replace(
            cycle,
            decision=replace(cycle.decision, champion=ids[control], moved=True, status="decided"),
        )
        with pytest.raises(PromotionRefused, match="control arm"):
            _write_pointer_if_moved(
                store=store,
                spec=spec,
                cycle=forged,
                expected=ETAG_ABSENT,
                manifest_key=None,
                run_id=None,
                code_sha=None,
                attestation=None,
                now=None,
                register=reg,
            )


# --------------------------------------------------------------------------
# `alpha-engine-config-I10044`: `_with_control_vetoes` and
# `_write_pointer_if_moved` thread the already-loaded register into
# `is_control_arm`, mirroring `tests/test_slots.py::TestIsControlArmIsRegisterBacked`
# at both promote.py call sites.
# --------------------------------------------------------------------------


class TestIsControlArmIsRegisterBackedAtPromoteCallSites:
    """A filed, non-control recipe whose generated name COLLIDES with a
    slot's control-arm name (``control_planted_m``) must not be treated as a
    control once it is registered with ``control=False`` — at either call
    site. Before threading the register through, both fell back to the NAME
    match and would have vetoed / refused a real arm that only happened to
    share a control's name.
    """

    def _slot_with_a_filed_name_collision(self, dates: list[str]):
        """A real, filed arm named identically to slot m's planted control,
        registered ``control=False`` — plus one uncontested real arm so the
        collider has something to be compared against."""
        spec = get_slot("m")
        collider_name = spec.control_arms[0].arm_id  # "control_planted_m"
        reg, ids = register_with("m", ["real_a", collider_name], dates[0])
        series_by_arm = {
            ids["real_a"]: series(ids["real_a"], dates, 0.0),
            ids[collider_name]: series(ids[collider_name], dates, 0.045),
        }
        return spec, reg, ids, collider_name, series_by_arm

    def test_with_control_vetoes_does_not_veto_a_name_collision_once_registered(
        self, tmp_path
    ) -> None:
        from crucible.promote import _with_control_vetoes

        dates = trading_days(40)
        spec, reg, ids, collider_name, series_by_arm = self._slot_with_a_filed_name_collision(
            dates
        )
        collider_id = ids[collider_name]

        assert is_control_arm(spec, collider_id), (
            "sanity: the NAME-ONLY fallback (no register) still matches the colliding name"
        )
        vetoed = _with_control_vetoes(spec, series_by_arm, None, reg)
        assert vetoed is None or collider_id not in vetoed, (
            "the record says control=False; a name collision must not be vetoed once "
            "the arm is registered, once the register is threaded through"
        )

    def test_write_pointer_if_moved_does_not_refuse_a_name_collision_as_champion(
        self, tmp_path
    ) -> None:
        """Without the register, this champion's NAME matches a control and
        the write would raise `PromotionRefused` against a real, promotable
        arm — the false-positive twin of `TestControlArmsNeverServe`'s
        guard, which is checking the opposite direction (a REAL control must
        still be refused)."""
        from crucible.promote import _write_pointer_if_moved
        from crucible.store import ETAG_ABSENT

        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec, reg, ids, collider_name, series_by_arm = self._slot_with_a_filed_name_collision(
            dates
        )
        collider_id = ids[collider_name]
        seed_register(store, "m", reg)

        cycle = run_promotion(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm=series_by_arm,
            incumbent=ids["real_a"],
        ).cycle
        forged = replace(
            cycle,
            decision=replace(cycle.decision, champion=collider_id, moved=True, status="decided"),
        )

        pointer = _write_pointer_if_moved(
            store=store,
            spec=spec,
            cycle=forged,
            expected=ETAG_ABSENT,
            manifest_key=None,
            run_id=None,
            code_sha=None,
            attestation=None,
            now=None,
            register=reg,
        )
        assert pointer is not None
        assert pointer.arm_id == collider_id


# --------------------------------------------------------------------------
# F6 on the promotion path, and the re-entrancy of the append helper.
# --------------------------------------------------------------------------


class TestPointerWriteIsConditional:
    def test_a_cycle_decided_against_a_moved_pointer_fails_loudly(self, tmp_path) -> None:
        """The pointer moved between the read the decision rests on and the
        write. Publishing anyway would clobber whatever moved it; retrying
        would publish a verdict reached against inputs nobody recorded. So
        the run fails and the next cycle decides from a premise that is
        true."""
        from crucible.champion import ChampionPointer, read_champion_etag, write_champion
        from crucible.store import PointerConflictError

        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec = narrow(get_slot("m"))
        reg, ids = register_with("m", ["champ", "chal"], dates[0])
        seed_register(store, "m", reg)
        stale = read_champion_etag(store, "m")

        # Another writer publishes between our read and our write.
        write_champion(
            store,
            ChampionPointer(
                slot="m",
                arm_id=ids["champ"],
                as_of=dates[-1],
                decided_at="2026-08-28T12:00:00Z",
                run_id="0" * 26,
                code_sha="0" * 40,
                promotion_source="operator_bootstrap",
                manifest_key=f"runs/promote/{dates[-1]}/run.json",
                evidence={"operator": "cipher813", "reason": "concurrent revert"},
            ),
            expected=read_champion_etag(store, "m"),
        )

        with pytest.raises(PointerConflictError):
            run_promotion(
                spec=spec,
                as_of=dates[-1],
                register=reg,
                series_by_arm={
                    ids["champ"]: series(ids["champ"], dates, 0.0),
                    ids["chal"]: series(ids["chal"], dates, 0.01),
                },
                incumbent=ids["champ"],
                store=store,
                pointer_etag=stale,
            )


class TestAppendIsIdempotent:
    """The reviewer's unverified suspicion, confirmed: `_append_events` is a
    read-modify-write, and `crucible.runner.run_job` retries a declared
    transient class by re-running the job body from the top. A first pass
    that wrote the retirement log and then hit an S3 throttle on the
    experiments write left both to be appended again on the retry, doubling
    every row of the first. "Single writer per slot per cycle" is a claim
    about concurrency and says nothing about re-entrancy.
    """

    def test_re_entering_a_cycle_does_not_duplicate_its_events(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec = narrow(get_slot("m"))
        names = ["a", "b", "c"]
        reg, ids = register_with("m", names, dates[0])
        seed_register(store, "m", reg)
        args = dict(
            spec=spec,
            as_of=dates[-1],
            register=reg,
            series_by_arm={
                ids[n]: series(ids[n], dates, v)
                for n, v in zip(names, (0.05, 0.0, -0.05), strict=True)
            },
            incumbent=ids["a"],
            store=store,
        )
        run_promotion(**args)
        first_retire = store.get_bytes("retirements/m/events.jsonl").decode().splitlines()
        first_exp = store.get_bytes(f"experiments/{dates[-1]}/events.jsonl").decode().splitlines()

        run_promotion(**args)  # the retry, same cycle, same as_of
        assert (
            store.get_bytes("retirements/m/events.jsonl").decode().splitlines() == first_retire
        ), "a re-entered pass must not double the retirement verdicts"
        assert (
            store.get_bytes(f"experiments/{dates[-1]}/events.jsonl").decode().splitlines()
            == first_exp
        ), "a re-entered pass must not double the EXPERIMENTS feed"

    def test_a_later_cycle_still_appends(self, tmp_path) -> None:
        """Idempotence must not become a write that never happens again: a
        DIFFERENT cycle's rows differ in `as_of` and are appended."""
        store = LocalStore(tmp_path)
        dates = trading_days(40)
        spec = narrow(get_slot("m"))
        names = ["a", "b", "c"]
        reg, ids = register_with("m", names, dates[0])
        seed_register(store, "m", reg)
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
        run_promotion(as_of=dates[-1], **args)
        log = store.get_bytes("retirements/m/events.jsonl").decode().splitlines()
        assert len(log) == 2 * len(names)


class TestOneProducerPerKeyShape:
    """F11 (`alpha-engine-config-I9757`): `arm_register_key` was DECLARED
    twice — `crucible/keys.py` and `crucible/promote.py` — both returning
    `arms/{slot}/register.jsonl`. `arena_io.py`'s own docstring names the
    pattern: "a second copy of it is how two slots end up writing artifacts
    the console renders differently". Identity, not string equality: two
    functions returning the same string today is exactly the state that
    precedes the drift.
    """

    def test_the_register_key_has_a_single_producer(self) -> None:
        import crucible.keys as keys
        import crucible.promote as promote

        assert promote.arm_register_key is keys.arm_register_key
        assert promote.arm_register_key.__module__ == "crucible.keys"
        assert keys.arm_register_key("m") == "arms/m/register.jsonl"

    def test_promote_declares_no_key_shape_keys_already_owns(self) -> None:
        """The scan, not the instance: a second copy re-added under any name
        fails here rather than at the console."""
        import inspect

        import crucible.keys as keys
        import crucible.promote as promote

        def single_slot_arg(fn) -> bool:
            """A plain function taking exactly `slot` — the shape of a key
            producer. Classes and exceptions are excluded before `signature`
            is asked, because several of them have none."""
            return inspect.isfunction(fn) and set(inspect.signature(fn).parameters) == {"slot"}

        owned_values = {
            fn("m")
            for fn in vars(keys).values()
            if getattr(fn, "__module__", "") == "crucible.keys" and single_slot_arg(fn)
        }
        assert owned_values, "the scan found no slot-keyed producers; it is not reading keys.py"
        for name, fn in vars(promote).items():
            if getattr(fn, "__module__", "") != "crucible.promote":
                continue
            if not single_slot_arg(fn):
                continue
            assert fn("m") not in owned_values, (
                f"crucible.promote.{name} re-declares a key shape crucible.keys already "
                "produces; keys.py is the single producer (plan §4.12)"
            )


class TestArmSeriesKeyUsesTheSharedSeparator:
    """`alpha-engine-config-I9784`: `arm_series_key` built `scores/{slot}/{arm_id}/series.json`
    from the RAW arm id, so it carried colons while `shadow_key`/`verdict_key` both routed
    through `arm_key_segment` (`crucible/keys.py`'s own docstring: "the one translation").
    Same class as F11's duplicate `arm_register_key` (`crucible-PR11`), and deliberately left
    out of that PR's scope because it changes a key shape.
    """

    def test_the_series_key_has_a_single_producer(self) -> None:
        import crucible.keys as keys
        import crucible.promote as promote

        assert promote.arm_series_key is keys.arm_series_key
        assert promote.arm_series_key.__module__ == "crucible.keys"

    def test_a_colon_bearing_arm_id_round_trips_through_every_arm_scoped_key(self) -> None:
        """The round trip the module's own docstring requires: a future change to
        `ARM_SEGMENT_SEPARATOR` cannot orphan `scores/` artifacts silently while leaving
        `experiments/` ones intact, because all three keys are asserted against the same
        segment here.
        """
        from crucible.keys import (
            arm_id_from_segment,
            arm_key_segment,
            arm_series_key,
            shadow_key,
            verdict_key,
        )

        arm_id = "m:momentum_sleeve:ab12cd"
        segment = arm_key_segment(arm_id)
        assert arm_id_from_segment(segment) == arm_id

        for key in (
            arm_series_key("m", arm_id),
            shadow_key(arm_id, "2026-08-28"),
            verdict_key(arm_id, "2026-08-28"),
        ):
            assert ":" not in key, f"a colon-bearing arm id leaked into {key!r}"
            assert segment in key, f"{key!r} does not route through arm_key_segment"
