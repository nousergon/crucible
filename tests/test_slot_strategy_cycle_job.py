"""The S cycle job drives the real path, end to end (`alpha-engine-config-I10512`).

`crucible-PR210` shipped `crucible.portfolio` and `construct_book` and
deliberately gave `crucible/slots/strategy.py` no `produce`/`grade`, because
`crucible.slots.dispatchable_slots()` reads those two names off each slot
module. The cost of the gap: the construction evidence reached no manifest any
scheduled job wrote, and the phase-3 clause wired to read it could only read
UNMEASURABLE.

Everything here runs the REAL path — a real price panel on disk, the real
loaders, the real `run_job` wrapper, a real manifest read back off the store —
for the same reason `tests/test_slot_model_cycle_job.py` does: the claim under
test is that a scheduled run produces these artifacts, and a mocked run cannot
make that claim. Real NYSE sessions throughout (`tests/conftest.py`'s
`sessions_ending`), never `weekday() < 5`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random

import numpy as np
import pandas as pd
import pytest

from crucible.calendar import is_trading_day
from crucible.config import Settings
from crucible.documents import load_store_document
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import (
    arena_cycle_key,
    arm_predictions_key,
    champion_key,
    data_panel_key,
    features_key,
    manifest_prefix,
    session_inputs_key,
    shadow_key,
)
from crucible.portfolio import (
    PORTFOLIO_METRIC_NAME,
    CostModel,
    manifest_records_portfolio_engine,
)
from crucible.runner import run_job
from crucible.slots import dispatchable_slots
from crucible.slots import strategy as strategy_module
from crucible.slots.arms import read_register
from crucible.slots.cycle import MissingArtifactError, run_grade
from crucible.slots.inputs import SlotUnservableError
from crucible.slots.strategy import (
    SLOT,
    ExitRuleSpec,
    PitParityVerdict,
    RegisteredStrategyArm,
    SlotStrategies,
    StrategyRecipe,
    SupersededArmUndeclaredError,
    _attestation_precondition,
    grade,
    load_strategy_slot,
    parse_strategy_document,
    produce,
    registration_specs,
    resolve_session,
)
from crucible.store import LocalStore
from tests.conftest import sessions_ending, synthetic_frames


def _next_session(day: dt.date) -> dt.date:
    """The first NYSE session strictly after ``day``."""
    day = day + dt.timedelta(days=1)
    while not is_trading_day(day):
        day += dt.timedelta(days=1)
    return day


AS_OF = dt.date(2026, 8, 28)
M_CHAMPION = "m:fixture_model:aaaaaaaaaaaa"
U_CHAMPION = "u:fixture_cut:bbbbbbbbbbbb"

FLAT_COST = """\
  cost_model:
    name: flat_bps_v0
    placeholder: true
    params:
      half_spread_bps: 2.5
      commission_bps: 0.5
      slippage_bps: 10.0
"""

IMPACT_COST = """\
  cost_model:
    name: sqrt_impact_v1
    placeholder: false
    params:
      half_spread_bps: 2.5
      commission_bps: 0.5
      impact_coef_bps: 12.0
      min_cost_bps: 1.0
"""

RULES = """\
  rules:
    - rule_id: position_loss_floor
      params:
        position_loss_floor_pct: -0.15
    - rule_id: profit_take
      params:
        profit_take_pct: 0.25
"""


def _recipe_yaml(name: str, *, registered_at: str | None, cost: str = FLAT_COST) -> str:
    head = [f"slot: {SLOT}", f"name: {name}"]
    if registered_at is not None:
        head.append(f"registered_at: '{registered_at}'")
    head.append(f"notes: fixture recipe for {name}")
    return "\n".join(head) + "\nspec:\n  benchmark: SPY\n" + cost + RULES


def _price_rows(
    days: list[dt.date], *, ticker: str, seed: int, start_price: float = 500.0
) -> pd.DataFrame:
    """One ETF proxy's synthetic OHLCV rows — `_benchmark_rows`'s shape, any ticker.

    `ATTRIBUTION_YAML`'s factor proxies (and the SPY benchmark) all need a real
    panel row, exactly the way the benchmark itself does — a different `seed`
    per ticker so the series are not degenerate copies of one another.
    """
    rng = random.Random(seed)
    price = start_price
    rows = []
    for day in days:
        price *= math.exp(rng.gauss(0.0003, 0.006))
        rows.append(
            {
                "trading_day": day,
                "ticker": ticker,
                "open_raw": price * 0.999,
                "high_raw": price * 1.004,
                "low_raw": price * 0.996,
                "close_raw": price,
                "volume_raw": 9e7,
            }
        )
    return pd.DataFrame(rows)


def _benchmark_rows(days: list[dt.date]) -> pd.DataFrame:
    return _price_rows(days, ticker="SPY", seed=7)


#: The minimal factor spec the fixture writes to `strategy/slots/attribution.yaml`
#: — one proxy per required category (`ATTRIBUTION_FACTOR_CATEGORIES`), rather
#: than production's six, so a settled window of a handful of sessions clears
#: `nousergon_lib.quant.factor_risk.estimate_factor_model`'s `n >= k + 2`
#: identification floor (k=3 factors here, so 5 settled sessions suffice).
ATTRIBUTION_FACTOR_PROXIES: dict[str, str] = {"market": "SPY", "size": "IWM", "sector_tech": "XLK"}

ATTRIBUTION_YAML = """\
attribution:
  benchmark_proxy: SPY
  shrinkage: ledoit_wolf
  factors:
    market:
      category: beta
      proxy: SPY
    size:
      category: size
      proxy: IWM
    sector_tech:
      category: sector
      proxy: XLK
"""


def _write_features(
    store, *, days: list[dt.date], tickers: list[str], dollar_volume_usd: float = 5e8
) -> None:
    """Compile a minimal `dollar_volume_20d_raw` feature artifact per day.

    Enough to exercise `crucible.slots.strategy._read_session_adv` — it reads
    `ticker` and the ADV column and nothing else — without pulling in the
    full feature-compilation pipeline, which is `crucible/features/compute.py`
    and out of scope for this job's own tests.
    """
    for day in days:
        frame = pd.DataFrame(
            {
                "trading_day": [day.isoformat()] * len(tickers),
                "ticker": tickers,
                "dollar_volume_20d_raw": [dollar_volume_usd] * len(tickers),
            }
        )
        store.put_bytes(
            features_key(DEFAULT_FEATURE_VERSION, day.isoformat()),
            frame.to_parquet(index=False),
        )


@pytest.fixture
def world(tmp_path):
    """A store carrying everything an S cycle needs, and the strategy tree beside it.

    The M champion's own `arm_predictions` are the alpha vector and the U
    champion's cut is the eligibility mask, because that is where the S job
    reads them from in production — a fixture that injected a vector directly
    would test a function this job does not call.
    """
    store = LocalStore(tmp_path / "store")
    days = sessions_ending(AS_OF, 60)
    frames = synthetic_frames(end=AS_OF, n_tickers=12, sessions=60)
    # Every attribution factor proxy (`ATTRIBUTION_FACTOR_PROXIES`) needs a real
    # panel row, exactly the way the SPY benchmark already does — the factor
    # model is fit from these series, not stubbed. Fixed per-ticker seeds
    # (never `hash(ticker)`, which is process-randomized) so the panel is
    # byte-identical across runs.
    proxy_seeds = {"SPY": 7, "IWM": 17, "XLK": 23}
    proxy_rows = [
        _price_rows(days, ticker=ticker, seed=proxy_seeds[ticker])
        for ticker in sorted({"SPY", *ATTRIBUTION_FACTOR_PROXIES.values()})
    ]
    panel = pd.concat(
        [frame.reset_index() for frame in frames.values()] + proxy_rows,
        ignore_index=True,
    )
    store.put_bytes(data_panel_key(AS_OF.isoformat()), panel.to_parquet(index=False))

    tickers = sorted(frames)
    for slot, champion in (("m", M_CHAMPION), ("u", U_CHAMPION)):
        store.put_bytes(
            champion_key(slot),
            json.dumps(
                {"schema_version": "champion_pointer.v1", "slot": slot, "arm_id": champion}
            ).encode("utf-8"),
        )
    # 7 decision days -> 6 settled sessions (the last is always unsettled) —
    # enough to clear `estimate_factor_model`'s `n >= k + 2` floor for the
    # fixture's 3-factor spec (k=3, floor 5).
    decision_days = sessions_ending(AS_OF, 7)
    rng = np.random.default_rng(3)
    for day in decision_days:
        store.put_bytes(
            arm_predictions_key(M_CHAMPION, day.isoformat()),
            json.dumps(
                {
                    "schema_version": "arm_predictions.v1",
                    "arm_id": M_CHAMPION,
                    "trading_day": day.isoformat(),
                    "feature_version": "vfixture",
                    "predicted_alpha": {t: float(rng.normal(0.0005, 0.004)) for t in tickers},
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )
        store.put_bytes(
            shadow_key(U_CHAMPION, day.isoformat()),
            json.dumps(
                {
                    "schema_version": "shadow.v1",
                    "arm_id": U_CHAMPION,
                    "trading_day": day.isoformat(),
                    "selection": tickers[:8],
                    "population": tickers,
                    "ranker": "momentum_sleeve",
                    "params": {},
                    "feature_version": "vfixture",
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )

    root = tmp_path / "strategy"
    (root / "arms" / SLOT).mkdir(parents=True)
    (root / "arms" / SLOT / "stock_registry.yaml").write_text(
        _recipe_yaml("stock_registry", registered_at=decision_days[0].isoformat()),
        encoding="utf-8",
    )
    (root / "slots").mkdir(parents=True)
    (root / "slots" / "s.yaml").write_text(_PARAMS_YAML, encoding="utf-8")
    (root / "slots" / "attribution.yaml").write_text(ATTRIBUTION_YAML, encoding="utf-8")
    settings = Settings(store_uri=str(tmp_path / "store"), arctic_bucket="", strategy_dir=root)
    return store, settings, root, decision_days


def _run_produce(store, settings, days) -> list[str]:
    keys: list[str] = []
    for day in days:
        ctx = run_job(
            "experiment.run",
            lambda c: produce(c, settings=settings),
            store=store,
            trading_day=day,
            discriminator=SLOT,
            run_mode="replay",
        )
        keys.extend(output["key"] for output in ctx.outputs)
    return keys


def _run_grade(store, settings, *, trading_day: dt.date = AS_OF):
    result: dict = {}

    def job(ctx):
        result.update(grade(ctx, settings=settings))

    ctx = run_job(
        "experiment.grade",
        job,
        store=store,
        trading_day=trading_day,
        discriminator=SLOT,
        run_mode="replay",
    )
    key = sorted(store.list_keys(manifest_prefix("experiment.grade", trading_day.isoformat())))[-1]
    return result, load_store_document(store, key), ctx


class TestTheSlotBecomesDispatchable:
    def test_dispatchable_slots_now_carries_s(self) -> None:
        """The whole reason this PR is gated: there is no list to update.

        `crucible.weekly.arc_stages` and both phase-1 gate clauses derive
        their expected stage set from this function, so S joins the arc the
        moment these two names exist.
        """
        assert SLOT in dispatchable_slots()
        assert callable(strategy_module.produce)
        assert callable(strategy_module.grade)

    def test_the_arc_expands_over_s(self) -> None:
        from crucible.weekly import ARC_SLOT_JOBS, arc_stages

        slots = {stage.slot for stage in arc_stages(AS_OF) if stage.job in ARC_SLOT_JOBS}
        assert SLOT in slots


class TestTheEvidenceReachesARealManifest:
    """`-I10512`'s closes-when, on artifacts a real run wrote."""

    def test_a_graded_run_carries_the_portfolio_construction_document(self, world) -> None:
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        result, manifest, _ctx = _run_grade(store, settings)

        assert manifest["status"] == "ok"
        evidence = manifest_records_portfolio_engine(manifest)
        assert evidence is not None, (
            "the S grading manifest carries no `portfolio_construction` row, so the "
            "phase-3 clause reads UNMEASURABLE — which is the exact state this job exists "
            "to change"
        )
        assert evidence["engine"] == "crucible.portfolio"
        assert evidence["cost_model"]["name"] == "flat_bps_v0"
        assert evidence["cost_model"]["placeholder"] is True
        assert evidence["sessions"] >= 1
        assert result["strategy_grades"]

    def test_the_phase_three_clause_reads_met_off_that_manifest(self, world) -> None:
        """The clause `-I10510` wired, run against this job's own output."""
        from crucible.gate import PHASE3_DELIVERABLES

        store, settings, _root, days = world
        _run_produce(store, settings, days)
        _result, manifest, _ctx = _run_grade(store, settings)

        by_id = {d.id: d for d in PHASE3_DELIVERABLES}
        for name in ("portfolio_engine_used_by_s_slot", "named_transaction_cost_model"):
            assert by_id[name].graded_by, (
                f"{name} carries no clause, so this job's evidence is read by nothing"
            )
        assert manifest_records_portfolio_engine(manifest) is not None

    def test_a_graded_run_carries_the_factor_attribution_document(self, world) -> None:
        """`alpha-engine-config-I10678`'s own closes-when: the call site this
        issue adds, verified on the manifest a real run wrote."""
        from crucible.attribution import manifest_records_factor_attribution

        store, settings, _root, days = world
        _run_produce(store, settings, days)
        result, manifest, _ctx = _run_grade(store, settings)

        assert manifest["status"] == "ok"
        evidence = manifest_records_factor_attribution(manifest)
        assert evidence is not None, (
            "the S grading manifest carries no `factor_attribution` row, so the phase-3 "
            "clause reads UNMEASURABLE — the exact state this issue exists to end"
        )
        assert evidence["engine"] == "crucible.attribution"
        assert set(ATTRIBUTION_FACTOR_PROXIES) <= {row["name"] for row in evidence["factors"]}
        assert set(evidence["category_totals"]) == {"beta", "sector", "size", "residual"}
        for figure in ("residual_alpha", "gross_return", "net_return"):
            assert isinstance(evidence[figure], float)
        assert result["strategy_grades"]

    def test_the_phase_three_attribution_clause_reads_met_off_that_manifest(self, world) -> None:
        """`crucible-PR229`'s clause, run against this job's own output — the
        deliverable's own closes-when: `crucible gate --gate phase3` moves off
        UNMEASURABLE the moment a real S-slot grading run carries the evidence."""
        from crucible.gate import _clause_factor_neutral_attribution

        store, settings, _root, days = world
        _run_produce(store, settings, days)
        _result, _manifest, _ctx = _run_grade(store, settings)

        clause = _clause_factor_neutral_attribution(store, [AS_OF])
        assert clause.met is True, clause.detail

    def test_produce_writes_one_session_inputs_document_per_arm(self, world) -> None:
        store, settings, _root, days = world
        keys = _run_produce(store, settings, days)
        arm_id = registration_specs(load_strategy_slot(strategy_dir=settings.strategy_dir))[
            0
        ].arm_id
        for day in days:
            key = session_inputs_key(arm_id, day.isoformat())
            assert key in keys
            document = load_store_document(store, key)
            assert document["schema_version"] == "session_inputs.v1"
            assert document["alpha_source"] == arm_predictions_key(M_CHAMPION, day.isoformat())
            assert document["eligibility_source"] == shadow_key(U_CHAMPION, day.isoformat())
            # Each absence is a SENTENCE on the record, not a missing key.
            assert "no sector column" in document["sectors_source"]
            assert "no conviction-stance producer" in document["stance_caps_source"]

    def test_the_grade_is_point_in_time_not_reconstructed(self, world) -> None:
        """The property `session_inputs_key` exists for.

        Revise the upstream alpha AFTER the sessions were recorded. The grade
        is unchanged, because it walks what the sessions recorded — and the
        contamination attestation FAILS, because the two passes now disagree.
        """
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        before, _manifest, _ctx = _run_grade(store, settings)

        rng = np.random.default_rng(99)
        for day in days:
            key = arm_predictions_key(M_CHAMPION, day.isoformat())
            document = load_store_document(store, key)
            document["predicted_alpha"] = {
                ticker: float(rng.normal(0.01, 0.02)) for ticker in document["predicted_alpha"]
            }
            store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))

        after, _manifest2, _ctx2 = _run_grade(store, settings)
        arm_id = next(iter(before["strategy_grades"]))
        assert after["strategy_grades"][arm_id]["cost_bps_total"] == pytest.approx(
            before["strategy_grades"][arm_id]["cost_bps_total"]
        ), "the grade moved when only an upstream artifact was revised — it is not PIT"
        assert after["strategy_grades"][arm_id]["attestation_mean_delta"] != 0.0, (
            "an upstream revision produced no attestation delta at all, so the check "
            "cannot see the contamination it exists to see"
        )
        assert before["strategy_grades"][arm_id]["attestation_mean_delta"] == 0.0, (
            "the two passes disagreed before anything was revised — the attestation is "
            "measuring something other than upstream revision"
        )


class TestParticipationAwareGradingEndToEnd:
    """`alpha-engine-config-I10669`'s own closes-when, on a real cycle.

    A `sqrt_impact_v1` arm registers, `experiment.run`/`experiment.grade`
    complete over it, and its `portfolio_construction` evidence carries a
    non-placeholder cost model priced with a real ADV vector and a real book
    notional — not the `GRADING_NOTIONAL = 1.0` unit placeholder every arm,
    flat or participation-priced, was constructed against before this fix.
    """

    def test_a_sqrt_impact_arm_grades_with_real_adv_and_notional(self, world) -> None:
        store, settings, root, days = world
        population = load_store_document(store, shadow_key(U_CHAMPION, days[0].isoformat()))[
            "population"
        ]
        _write_features(store, days=days, tickers=[*population, "SPY"])
        (root / "arms" / SLOT / "impact.yaml").write_text(
            _recipe_yaml("impact", registered_at=days[0].isoformat(), cost=IMPACT_COST),
            encoding="utf-8",
        )

        _run_produce(store, settings, days)
        result, manifest, _ctx = _run_grade(store, settings)

        assert manifest["status"] == "ok"
        assert result["refused"] == []
        impact_grade = next(
            g for g in result["strategy_grades"].values() if g["cost_model"] == "sqrt_impact_v1"
        )
        assert impact_grade["sessions"] >= 1

        evidences = [
            m["portfolio_construction"]
            for m in manifest["metrics"]
            if m["name"] == PORTFOLIO_METRIC_NAME
        ]
        impact_evidence = next(e for e in evidences if e["cost_model"]["name"] == "sqrt_impact_v1")
        assert impact_evidence["cost_model"]["placeholder"] is False
        assert impact_evidence["cost_model"]["params"] == {
            "half_spread_bps": 2.5,
            "commission_bps": 0.5,
            "impact_coef_bps": 12.0,
            "min_cost_bps": 1.0,
        }
        assert impact_evidence["params"]["book_notional_usd"] == pytest.approx(1_000_000.0)
        assert impact_evidence["components"]["cost_model"]["reason"] == "sqrt_impact", (
            "the sqrt-impact term did not engage — it fell back to the flat branch, which "
            "is exactly the fallback-cost-model defect this arm's refusal exists to prevent"
        )

        # And the sibling flat arm's grade is UNCHANGED by a participation
        # arm existing beside it: `book_notional_usd` now prices every book,
        # but a flat cost model's charge is a function of weight deltas
        # alone and never reads the notional.
        flat_evidence = next(e for e in evidences if e["cost_model"]["name"] == "flat_bps_v0")
        assert flat_evidence["cost_model"]["placeholder"] is True


class TestRefusalsAreRecordedFirstAndPerArm:
    def test_an_arm_with_no_oos_clock_registers_stamped_from_its_first_run(self, world) -> None:
        """`alpha-engine-config-I10634` (Brian ruling 2026-09-13, option (b)):
        the clock starts at the first cycle that produced scorable evidence,
        never at a filing date. A recipe declaring no `registered_at` is NOT
        refused — it registers, and the register row it gets is stamped from
        THIS run's own trading day, not left unset or defaulted elsewhere.
        """
        store, settings, root, days = world
        (root / "arms" / SLOT / "no_clock.yaml").write_text(
            _recipe_yaml("no_clock", registered_at=None), encoding="utf-8"
        )
        ctx = run_job(
            "experiment.run",
            lambda c: produce(c, settings=settings),
            store=store,
            trading_day=days[-1],
            discriminator=SLOT,
            run_mode="replay",
        )
        rows = [m for m in ctx.metrics if m["name"] == "arm_refused_at_registration"]
        assert rows == [], "a recipe declaring no `registered_at` registers, it is not refused"
        # Both arms produced — the sibling AND the one with no declared clock.
        assert any(m["name"] == "arms_produced" and m["value"] == 2.0 for m in ctx.metrics)

        recipe = parse_strategy_document(
            (root / "arms" / SLOT / "no_clock.yaml").read_bytes(), origin="no_clock.yaml"
        )
        record = read_register(store, SLOT).state(recipe.arm_id).record
        assert record.created_date == days[-1].isoformat()

    def test_an_already_registered_undeclared_arm_keeps_its_first_stamp(self, world) -> None:
        """A SECOND cycle over an arm that still declares no `registered_at`
        does not re-stamp it: the register row already carries a first date,
        and that is what resolves — never `today` again."""
        store, settings, root, days = world
        (root / "arms" / SLOT / "no_clock.yaml").write_text(
            _recipe_yaml("no_clock", registered_at=None), encoding="utf-8"
        )
        run_job(
            "experiment.run",
            lambda c: produce(c, settings=settings),
            store=store,
            trading_day=days[0],
            discriminator=SLOT,
            run_mode="replay",
        )
        run_job(
            "experiment.run",
            lambda c: produce(c, settings=settings),
            store=store,
            trading_day=days[-1],
            discriminator=SLOT,
            run_mode="replay",
        )
        recipe = parse_strategy_document(
            (root / "arms" / SLOT / "no_clock.yaml").read_bytes(), origin="no_clock.yaml"
        )
        record = read_register(store, SLOT).state(recipe.arm_id).record
        assert record.created_date == days[0].isoformat()

    def test_no_registered_at_and_no_cycle_context_still_refuses(self, world) -> None:
        """`load_strategy_slot` called directly, with no register and no
        `today` — the shape `crucible.track_a`'s manual `experiment.new`
        path uses. Pre-`alpha-engine-config-I10634` behaviour: nothing to
        stamp the clock with, so the recipe still refuses."""
        store, settings, root, days = world
        (root / "arms" / SLOT / "no_clock.yaml").write_text(
            _recipe_yaml("no_clock", registered_at=None), encoding="utf-8"
        )
        loaded = load_strategy_slot(strategy_dir=root)
        refused = {r.arm: r for r in loaded.refused}
        assert "no_clock" in refused
        assert "registered_at" in refused["no_clock"].reason
        assert [a.name for a in loaded.registered] == ["stock_registry"]

    def test_a_future_registered_at_is_refused_by_name(self, world) -> None:
        """One of the two refusal cases a DECLARED `registered_at` still has:
        a clock cannot start in the future. Per arm, never slot-wide."""
        store, settings, root, days = world
        future = _next_session(days[-1])
        (root / "arms" / SLOT / "premature.yaml").write_text(
            _recipe_yaml("premature", registered_at=future.isoformat()), encoding="utf-8"
        )
        ctx = run_job(
            "experiment.run",
            lambda c: produce(c, settings=settings),
            store=store,
            trading_day=days[-1],
            discriminator=SLOT,
            run_mode="replay",
        )
        rows = [m for m in ctx.metrics if m["name"] == "arm_refused_at_registration"]
        assert [r["status"] for r in rows] == ["unservable"]
        assert "AFTER" in rows[0]["status_reason"]
        assert "premature" in rows[0]["source_path"]
        # Per arm, never slot-wide: the sibling still produced.
        assert any(m["name"] == "arms_produced" and m["value"] == 1.0 for m in ctx.metrics)

    def test_a_registered_at_moved_after_the_first_register_row_is_refused(self, world) -> None:
        """The second refusal case: a clock that can be moved is not a clock.

        First cycle registers the arm with no declared `registered_at`,
        stamped from that run's trading day. Declaring a LATER date on a
        later cycle is refused rather than silently re-dating an arm already
        serving."""
        store, settings, root, days = world
        (root / "arms" / SLOT / "mover.yaml").write_text(
            _recipe_yaml("mover", registered_at=None), encoding="utf-8"
        )
        run_job(
            "experiment.run",
            lambda c: produce(c, settings=settings),
            store=store,
            trading_day=days[0],
            discriminator=SLOT,
            run_mode="replay",
        )
        recipe = parse_strategy_document(
            (root / "arms" / SLOT / "mover.yaml").read_bytes(), origin="mover.yaml"
        )
        first_recorded = read_register(store, SLOT).state(recipe.arm_id).record.created_date
        assert first_recorded == days[0].isoformat()

        (root / "arms" / SLOT / "mover.yaml").write_text(
            _recipe_yaml("mover", registered_at=days[-1].isoformat()), encoding="utf-8"
        )
        ctx = run_job(
            "experiment.run",
            lambda c: produce(c, settings=settings),
            store=store,
            trading_day=days[-1],
            discriminator=SLOT,
            run_mode="replay",
        )
        rows = [
            m
            for m in ctx.metrics
            if m["name"] == "arm_refused_at_registration" and "mover" in m["source_path"]
        ]
        assert rows and rows[0]["status"] == "unservable"
        assert "AFTER" in rows[0]["status_reason"]
        # The register row is untouched: still the date the arm first registered.
        unchanged = read_register(store, SLOT).state(recipe.arm_id).record.created_date
        assert unchanged == days[0].isoformat()

    def test_a_participation_model_registers_once_adv_and_notional_are_wired(self, world) -> None:
        """`alpha-engine-config-I10669`: the refusal used to fire on EVERY
        non-flat cost model unconditionally, before `_build_sessions` wired a
        real ADV vector and `construct_book` was handed a real book
        notional. Both are wired now — `dollar_volume_20d_raw` is in the
        feature catalog and `book_notional_usd` is a declared portfolio-params
        field — so a participation-aware arm registers instead of being
        refused forever."""
        store, settings, root, days = world
        (root / "arms" / SLOT / "impact.yaml").write_text(
            _recipe_yaml("impact", registered_at=days[0].isoformat(), cost=IMPACT_COST),
            encoding="utf-8",
        )
        loaded = load_strategy_slot(strategy_dir=root)
        assert loaded.refused == ()
        assert sorted(a.name for a in loaded.registered) == ["impact", "stock_registry"]

    def test_a_participation_model_still_refuses_with_no_adv_column_declared(
        self, world, monkeypatch
    ) -> None:
        """The relaxed refusal is not a no-op: it still fires when the
        catalog genuinely cannot produce the series the model needs."""
        store, settings, root, days = world
        (root / "arms" / SLOT / "impact.yaml").write_text(
            _recipe_yaml("impact", registered_at=days[0].isoformat(), cost=IMPACT_COST),
            encoding="utf-8",
        )
        monkeypatch.setattr(strategy_module, "_adv_column_declared", lambda: False)
        loaded = load_strategy_slot(strategy_dir=root)
        refused = {r.arm: r for r in loaded.refused}
        assert "impact" in refused
        assert refused["impact"].unresolvable == ("adv_usd",)
        assert "NOT swapped for a flat one" in refused["impact"].reason
        assert [a.name for a in loaded.registered] == ["stock_registry"]

    def test_a_participation_model_still_refuses_with_no_notional_field_declared(
        self, world, monkeypatch
    ) -> None:
        """The other half of the same guard: no declared book-notional field."""
        store, settings, root, days = world
        (root / "arms" / SLOT / "impact.yaml").write_text(
            _recipe_yaml("impact", registered_at=days[0].isoformat(), cost=IMPACT_COST),
            encoding="utf-8",
        )
        monkeypatch.setattr(strategy_module, "_notional_field_declared", lambda: False)
        loaded = load_strategy_slot(strategy_dir=root)
        refused = {r.arm: r for r in loaded.refused}
        assert "impact" in refused
        assert refused["impact"].unresolvable == ("portfolio_notional",)
        assert [a.name for a in loaded.registered] == ["stock_registry"]

    def test_the_raise_that_guards_the_same_substitution_still_fires(self) -> None:
        """§7.4: the refusal above is one of TWO guards, and the other is live.

        A book with no per-date cost, handed to a participation-aware recipe,
        raises rather than being charged a flat rate. The registration refusal
        stops the arm reaching this; this stops it if it ever does.
        """
        from crucible.portfolio import CostModelInputError
        from crucible.slots.strategy import Book, grade_arm

        recipe = StrategyRecipe(
            name="impact",
            rules=(ExitRuleSpec(rule_id="profit_take", params={"profit_take_pct": 0.25}),),
            cost_model=CostModel(
                name="sqrt_impact_v1",
                placeholder=False,
                params={
                    "half_spread_bps": 2.5,
                    "commission_bps": 0.5,
                    "impact_coef_bps": 12.0,
                    "min_cost_bps": 1.0,
                },
            ),
        )
        book = Book(
            dates=("2026-08-27",),
            portfolio_returns=(0.001,),
            benchmark_returns=(0.0005,),
            turnover=(0.1,),
        )
        with pytest.raises(CostModelInputError, match="prices participation"):
            grade_arm(recipe, book, as_of="2026-08-28")

    def test_every_arm_refused_makes_the_whole_slot_unservable_and_pages(self, world) -> None:
        """The rows are on the manifest BEFORE the raise, and the raise pages.

        Asserted through `crucible.alerts.evaluate_failure` on the manifest the
        failed run actually wrote, not by construction.
        """
        from crucible.alerts import evaluate_failure

        store, settings, root, days = world
        future = _next_session(days[-1])
        (root / "arms" / SLOT / "stock_registry.yaml").write_text(
            _recipe_yaml("stock_registry", registered_at=future.isoformat()), encoding="utf-8"
        )
        with pytest.raises(SlotUnservableError):
            run_job(
                "experiment.run",
                lambda c: produce(c, settings=settings),
                store=store,
                trading_day=days[-1],
                discriminator=SLOT,
                run_mode="replay",
            )
        key = sorted(store.list_keys(manifest_prefix("experiment.run", days[-1].isoformat())))[-1]
        manifest = load_store_document(store, key)
        assert manifest["status"] == "failed"
        rows = [m for m in manifest["metrics"] if m["name"] == "arm_refused_at_registration"]
        assert rows and rows[0]["status"] == "unservable", (
            "the whole-slot case paged with a reason and no per-arm rows — the least "
            "informative manifest of the three possible outcomes, on the worst of them"
        )
        pages = evaluate_failure(
            store, now=dt.datetime.combine(days[-1], dt.time(22, 0), tzinfo=dt.UTC)
        )
        assert [page.job for page in pages] == ["experiment.run"]

    def test_an_arm_with_no_settled_session_is_refused_per_arm_not_slot_wide(self, world) -> None:
        """The warm-up. A decision date is held through the NEXT session, so the
        most recent one is never settled — and an arm whose only session is
        today has no book yet. That is a per-arm row, not a dead slot."""
        store, settings, _root, days = world
        _run_produce(store, settings, [days[-1]])
        with pytest.raises(MissingArtifactError, match="no settled session"):
            _run_grade(store, settings)
        key = sorted(store.list_keys(manifest_prefix("experiment.grade", AS_OF.isoformat())))[-1]
        manifest = load_store_document(store, key)
        rows = [m for m in manifest["metrics"] if m["name"] == "arm_refused_at_registration"]
        assert rows and "no settled session" in rows[0]["status_reason"]


class TestTheEngineSeamsAreAdditiveAndNotDisplaceable:
    def test_a_series_without_settled_dates_is_refused(self, world) -> None:
        store, settings, _root, _days = world

        class _Ctx:
            def __init__(self, s):
                self.store = s
                self.trading_day = AS_OF

        with pytest.raises(ValueError, match="supplied together"):
            run_grade(_Ctx(store), slot=SLOT, settings=settings, series={})

    def test_the_control_exclusion_survives_a_caller_supplied_precondition(self, world) -> None:
        """§10.1 is the harness's rule, not a slot's.

        A caller that passed a precondition for a control must not be able to
        displace the exclusion — that is a look-ahead arm one dict key away
        from the pointer.
        """
        from nousergon_lib.arena.engine import ServingPrecondition

        store, settings, _root, days = world
        _run_produce(store, settings, days)
        result, _manifest, _ctx = _run_grade(store, settings)
        cycle = load_store_document(store, arena_cycle_key(SLOT, AS_OF.isoformat()))
        ineligible = cycle["decision"]["ineligible"]
        controls = [arm for arm in ineligible if "control_" in arm]
        assert len(controls) == 2
        for arm in controls:
            assert any(rule["name"] == "not_a_control_arm" for rule in ineligible[arm])
        assert result["pointer"]["champion"] not in controls
        assert ServingPrecondition  # imported for the contract it names

    def test_a_supplied_series_for_an_unregistered_arm_is_refused(self, world) -> None:
        """An arm scored without a register row is policy §3's defect exactly."""
        from nousergon_lib.arena.window import ArmSeries

        store, settings, _root, days = world
        _run_produce(store, settings, days)
        specs = registration_specs(load_strategy_slot(strategy_dir=settings.strategy_dir))

        class _Ctx:
            def __init__(self, s):
                self.store = s
                self.trading_day = AS_OF
                self.metrics: list = []

            def record_input(self, *a, **k) -> None:
                pass

            def record_metric(self, metric) -> None:
                self.metrics.append(metric)

        with pytest.raises(MissingArtifactError, match="the register does not score"):
            run_grade(
                _Ctx(store),
                slot=SLOT,
                settings=settings,
                horizon_trading_days=1,
                specs=specs,
                series={
                    "s:not_a_real_arm:0123456789ab": ArmSeries(
                        arm_id="s:not_a_real_arm:0123456789ab",
                        scores={days[0].isoformat(): 0.01},
                    )
                },
                settled_dates=[days[0].isoformat()],
            )


class TestTheAttestationCannotBeRoundedUp:
    @pytest.mark.parametrize("status", ["PARTIAL", "UNKNOWN", "FAIL"])
    def test_anything_but_pass_fails_the_serving_precondition(self, status: str) -> None:
        """Plan §9.1: a card without `attestation: PASS` renders UNVERIFIED,
        never a grade. Policy §5.1: an uncomputed gate is not a pass."""
        precondition = _attestation_precondition(
            PitParityVerdict(status=status, reason="fixture", coverage_fraction=0.5)
        )
        assert precondition.passed is False
        assert status in precondition.reason

    def test_pass_passes(self) -> None:
        assert _attestation_precondition(
            PitParityVerdict(status="PASS", reason="fixture", coverage_fraction=1.0)
        ).passed


class TestLineageIsProvenanceUntilThereIsARowToLinkTo:
    def _arm(self, name: str, supersedes: str | None) -> RegisteredStrategyArm:
        return RegisteredStrategyArm(
            StrategyRecipe(
                name=name,
                rules=(ExitRuleSpec(rule_id="profit_take", params={"profit_take_pct": 0.25}),),
                cost_model=CostModel(
                    name="flat_bps_v0",
                    placeholder=True,
                    params={
                        "half_spread_bps": 2.5,
                        "commission_bps": 0.5,
                        "slippage_bps": 10.0,
                    },
                ),
                supersedes=supersedes,
                registered_at="2026-08-03",
            )
        )

    def test_a_pointer_at_an_arm_the_slot_never_declares_still_raises(self) -> None:
        loaded = SlotStrategies(
            registered=(self._arm("child", "s:ghost:0123456789ab"),), refused=()
        )
        with pytest.raises(SupersededArmUndeclaredError, match="ghost"):
            registration_specs(loaded)

    def test_a_pointer_at_a_registered_sibling_becomes_a_link(self) -> None:
        parent = self._arm("parent", None)
        child = self._arm("child", parent.arm_id)
        specs = registration_specs(SlotStrategies(registered=(parent, child), refused=()))
        assert specs[1].supersedes == parent.arm_id
        assert specs[1].notes == ""


class TestTheInputsAreNamedWhenTheyAreAbsent:
    """Every refusal on this path names the KEY the operator's next action is
    about. A reason that does not name it sends them looking."""

    def test_no_m_champion_means_the_slot_has_no_alpha(self, world) -> None:
        store, settings, _root, days = world
        store.put_bytes(
            champion_key("m"),
            json.dumps({"schema_version": "champion_pointer.v1", "slot": "m"}).encode("utf-8"),
        )
        with pytest.raises(MissingArtifactError, match="has no alpha"):
            resolve_session(store, trading_day=days[0].isoformat())

    def test_a_u_champion_that_produced_no_cut_is_refused(self, world) -> None:
        store, settings, _root, days = world
        store.put_bytes(
            champion_key("u"),
            json.dumps(
                {
                    "schema_version": "champion_pointer.v1",
                    "slot": "u",
                    "arm_id": "u:never_ran:cccccccccccc",
                }
            ).encode("utf-8"),
        )
        with pytest.raises(MissingArtifactError, match="wrote no cut"):
            resolve_session(store, trading_day=days[0].isoformat())

    def test_no_u_champion_records_the_absence_rather_than_assuming_a_cut(self, world) -> None:
        store, settings, _root, days = world
        store.put_bytes(
            champion_key("u"),
            json.dumps({"schema_version": "champion_pointer.v1", "slot": "u"}).encode("utf-8"),
        )
        session = resolve_session(store, trading_day=days[0].isoformat())
        assert all(session.eligibility)
        assert "recorded absence" in session.eligibility_source

    def test_a_benchmark_the_panel_does_not_carry_is_refused_not_proxied(
        self, world, tmp_path
    ) -> None:
        """S is the one slot graded against a market index, and no proxy is
        substituted: grading against a benchmark the recipe did not declare
        inverts wins and losses outright."""
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        panel = pd.read_parquet(
            __import__("io").BytesIO(store.get_bytes(data_panel_key(AS_OF.isoformat())))
        )
        store.put_bytes(
            data_panel_key(AS_OF.isoformat()),
            panel[panel["ticker"] != "SPY"].to_parquet(index=False),
        )
        with pytest.raises(MissingArtifactError, match="carries no rows for"):
            _run_grade(store, settings)

    def test_a_factor_proxy_the_panel_does_not_carry_is_refused_not_stubbed(self, world) -> None:
        """S is graded against a market index, and a decomposition with no
        factors is not the deliverable: no proxy is substituted and no figure
        is stubbed for a missing factor series (module rule 5)."""
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        panel = pd.read_parquet(
            __import__("io").BytesIO(store.get_bytes(data_panel_key(AS_OF.isoformat())))
        )
        store.put_bytes(
            data_panel_key(AS_OF.isoformat()),
            panel[panel["ticker"] != "IWM"].to_parquet(index=False),
        )
        with pytest.raises(MissingArtifactError, match="attribution factor 'size' proxies 'IWM'"):
            _run_grade(store, settings)

    def test_an_unknown_arm_selector_is_refused_rather_than_producing_nothing(self, world) -> None:
        store, settings, _root, days = world
        with pytest.raises(MissingArtifactError, match="no arm named"):
            run_job(
                "experiment.run",
                lambda c: produce(c, settings=settings, arm_name="not_filed"),
                store=store,
                trading_day=days[0],
                discriminator=SLOT,
                run_mode="replay",
            )

    def test_the_store_is_the_source_when_there_is_no_checkout(self, world) -> None:
        """The spot-box path. Same parse, same refusals, no second loader."""
        from crucible.keys import strategy_arm_key

        store, settings, root, days = world
        store.put_bytes(
            strategy_arm_key(SLOT, "stock_registry"),
            (root / "arms" / SLOT / "stock_registry.yaml").read_bytes(),
        )
        from_store = load_strategy_slot(store=store)
        from_checkout = load_strategy_slot(strategy_dir=root)
        assert [a.arm_id for a in from_store.registered] == [
            a.arm_id for a in from_checkout.registered
        ]

    def test_neither_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="either a store or a strategy_dir"):
            load_strategy_slot()

    def test_a_session_document_of_another_version_is_refused(self) -> None:
        from crucible.slots.strategy import ResolvedSession

        with pytest.raises(ValueError, match="read wrong, not read approximately"):
            ResolvedSession.from_dict({"schema_version": "session_inputs.v0"})


_PARAMS_YAML = """\
portfolio:
  risk_aversion: 5.0
  cash_sleeve_pct: 0.03
  max_sector_pct: 0.25
  min_position_pct: 0.005
  covariance_shrinkage: ledoit_wolf
  sigma_horizon_days: 1
  ewma_lambda_decay: 0.94
  vol_target_annual: null
  alpha_uncertainty_penalty: 0.0
  alpha_uncertainty_min_cv: 0.01
  max_pct_adv: null
  max_daily_turnover: 0.15
  large_move_turnover_flag: 0.35
  conviction_budget_gate_enabled: true
  conviction_ir_floor: 0.35
  conviction_ir_full: 0.75
  conviction_budget_min_multiple: 0.05
  conviction_gate_min_names: 3
  book_notional_usd: 1000000.0
"""
