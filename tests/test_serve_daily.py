"""`serve.daily --slot m` — the champion's feed on every trading day.

`alpha-engine-config-I12047` (Brian's ruling: a daily M job). The trader's
session binds to the last closed trading day and requires
`predictions/{that day}.json`, refusing an older feed by design; the only
producer of that key was `experiment.run[m]` on the weekly arc, so only the
session after the arc could be served.

The job scores the CURRENT champion on the session's features from the
champion's fit of record — re-derived through the fitting path's own calls
and PROVEN against the cross-section that fit published — and publishes
through the one serving path. Every test drives the real path: a real feature
layer on disk, the real recipe loader, the real `run_job` wrapper and a real
manifest. Dates are fixed literals off a pinned session axis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pytest

import crucible.track_a as track_a
from crucible.calendar import is_trading_day
from crucible.champion import (
    CHAMPION_SCHEMA_VERSION,
    ChampionPointer,
    read_champion,
    read_champion_etag,
    write_champion,
)
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import (
    arm_predictions_key,
    cross_section_key,
    features_key,
    manifest_key,
    predictions_key,
    shadow_key,
)
from crucible.models import PredictionsFeedDocument
from crucible.runner import run_job
from crucible.serving import read_predictions_feed
from crucible.slots import daily_servers
from crucible.slots.cycle import MissingArtifactError
from crucible.slots.model import (
    DAILY_FEED_METRIC,
    DAILY_SERVE_JOB,
    SLOT,
    FeatureLayerSource,
    FitReconstructionError,
    design_panel,
    load_model_recipes,
    predict_cross_section,
    produce,
    serve_daily,
    train_arm,
)
from crucible.store import LocalStore

BASE_COLUMN = "momentum_20d_zscore"
SECOND_COLUMN = "volatility_20d_ratio"
_NAMES = tuple(f"N{i:02d}" for i in range(12))


def _sessions(start: dt.date, count: int) -> tuple[str, ...]:
    days: list[str] = []
    day = start
    while len(days) < count:
        if is_trading_day(day):
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    return tuple(days)


SESSIONS = _sessions(dt.date(2026, 6, 1), 60)
#: The weekly arc's fit day (a Wednesday on this axis; the weekday does not
#: matter to the job, only the session count after it).
FIT_DAY = SESSIONS[45]
#: Two sessions after the fit: the trader's next session needs this feed.
SERVE_DAY = SESSIONS[47]
#: Six sessions after the fit — past the recipe's `refit_cadence_trading_days: 5`.
TOO_LATE = SESSIONS[51]
SATURDAY = "2026-08-08"


@pytest.fixture
def store(tmp_path) -> LocalStore:
    import pandas as pd

    backing = LocalStore(tmp_path / "store")
    rng = np.random.default_rng(20261006)
    for i, day in enumerate(SESSIONS):
        frame = pd.DataFrame(
            {
                "ticker": list(_NAMES),
                "close_raw": 100.0 + i * 0.5 + rng.normal(0.0, 1.0, len(_NAMES)),
                BASE_COLUMN: rng.normal(0.0, 1.0, len(_NAMES)),
                SECOND_COLUMN: rng.normal(1.0, 0.3, len(_NAMES)),
            }
        )
        backing.put_bytes(features_key(DEFAULT_FEATURE_VERSION, day), frame.to_parquet(index=False))
    return backing


def _write_recipe(directory, name, *, features, inputs=()):
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["slot: m", f"name: {name}", "spec:", f"  features: [{', '.join(features)}]"]
    if inputs:
        lines.append("  inputs:")
        lines += [f"    - {entry}" for entry in inputs]
    lines += [
        "  estimator: {kind: ridge, alpha: 1.0}",
        "  label_horizon_trading_days: 2",
        "  refit_cadence_trading_days: 5",
        "  training_window: {kind: expanding, min_trading_days: 10}",
        "  cpcv: {n_groups: 4, k_test: 1, embargo_trading_days: 1}",
        f"registered_at: '{SESSIONS[0]}'",
    ]
    (directory / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Settings:
    def __init__(self, strategy_dir):
        self.strategy_dir = strategy_dir


@pytest.fixture
def settings(tmp_path):
    arms = tmp_path / "strategy" / "arms" / SLOT
    _write_recipe(arms, "base", features=[BASE_COLUMN])
    _write_recipe(arms, "stacked", features=[SECOND_COLUMN], inputs=["predictions[base]"])
    return _Settings(tmp_path / "strategy")


def _recipes(settings):
    return {r.name: r for r in load_model_recipes(settings.strategy_dir / "arms" / SLOT).registered}


def _fit_on_the_arc(store, settings, day: str = FIT_DAY, *, arm: str | None = "base") -> None:
    """`experiment.run --slot m` on the arc day: the fit of record."""
    run_job(
        "experiment.run",
        lambda c: produce(c, settings=settings, arm_name=arm),
        store=store,
        trading_day=dt.date.fromisoformat(day),
        run_mode="replay",
        discriminator=SLOT,
    )


def _seat(store, arm_id: str) -> None:
    promote_manifest = f"runs/promote/{FIT_DAY}/m/run.json"
    store.put_bytes(promote_manifest, json.dumps({"status": "ok", "job": "promote"}).encode())
    write_champion(
        store,
        ChampionPointer(
            schema_version=CHAMPION_SCHEMA_VERSION,
            slot=SLOT,
            arm_id=arm_id,
            as_of=FIT_DAY,
            decided_at=f"{FIT_DAY}T19:00:00Z",
            run_id="01JG0000000000000000000000",
            code_sha="a" * 40,
            promotion_source="evidence",
            manifest_key=promote_manifest,
            evidence={"status": "decided", "moved": True, "paired_dates": 40},
            attestation=None,
        ),
        expected=read_champion_etag(store, SLOT),
    )


def _serve(store, settings, day: str = SERVE_DAY) -> dict:
    result: dict = {}
    run_job(
        DAILY_SERVE_JOB,
        lambda c: result.update(serve_daily(c, settings=settings)),
        store=store,
        trading_day=dt.date.fromisoformat(day),
        run_mode="replay",
        discriminator=SLOT,
    )
    return result


def _manifest(store, day: str = SERVE_DAY) -> dict:
    key = manifest_key(DAILY_SERVE_JOB, day, discriminator=SLOT)
    return json.loads(store.get_bytes(key).decode("utf-8"))


def _trader_resolves(store, day: str):
    """The trader's contract check (`crucible_trader.contract.resolve`), the
    two halves it compares: the pointer through `read_champion`, the feed
    through `read_predictions_feed`, and `champion == pointer.arm_id`."""
    pointer = read_champion(store, SLOT)
    feed = read_predictions_feed(store, day)
    assert feed.champion == pointer.arm_id, (feed.champion, pointer.arm_id)
    assert feed.trading_day == day
    return pointer, feed


class TestTheTraderContractRoundTrips:
    def test_a_non_arc_session_gets_a_feed_the_trader_accepts(self, store, settings) -> None:
        _fit_on_the_arc(store, settings)
        base = _recipes(settings)["base"]
        _seat(store, base.arm_id)
        assert not store.exists(predictions_key(SERVE_DAY))

        result = _serve(store, settings)

        pointer, feed = _trader_resolves(store, SERVE_DAY)
        assert result["champion"] == pointer.arm_id == base.arm_id
        assert result["champion_feed"] == predictions_key(SERVE_DAY)
        assert result["scored_from_fit_of"] == {base.arm_id: FIT_DAY}
        assert feed.source_key == arm_predictions_key(base.arm_id, SERVE_DAY)
        assert set(feed.predicted_alpha) == set(_NAMES)

    def test_the_feed_is_byte_compatible_with_predictions_feed_v1_and_carries_the_std(
        self, store, settings
    ) -> None:
        _fit_on_the_arc(store, settings)
        _seat(store, _recipes(settings)["base"].arm_id)
        _serve(store, settings)

        raw = json.loads(store.get_bytes(predictions_key(SERVE_DAY)).decode("utf-8"))
        PredictionsFeedDocument.model_validate(raw)
        assert raw["schema_version"] == "predictions_feed.v1"
        assert "predicted_alpha_std_method" in raw
        assert set(raw["predicted_alpha_std"]) == set(raw["predicted_alpha"])
        assert all(v > 0 for v in raw["predicted_alpha_std"].values())

    def test_the_weights_are_the_arc_fit_never_a_refit(self, store, settings) -> None:
        """The served cross-section is the FIT DAY's weights applied to the
        serve day's features — and differs from a fit as of the serve day."""
        _fit_on_the_arc(store, settings)
        base = _recipes(settings)["base"]
        _seat(store, base.arm_id)
        _serve(store, settings)
        served = read_predictions_feed(store, SERVE_DAY).predicted_alpha

        source = FeatureLayerSource(store=store)
        serve_panel = design_panel(base, source=source, trading_day=SERVE_DAY)

        def fit_as_of(day: str):
            panel = design_panel(base, source=source, trading_day=day, lookback_trading_days=12)
            return train_arm(base, panel, as_of=day)

        carried = predict_cross_section(fit_as_of(FIT_DAY), serve_panel, trading_day=SERVE_DAY)
        refit = predict_cross_section(fit_as_of(SERVE_DAY), serve_panel, trading_day=SERVE_DAY)
        assert served == pytest.approx(carried, rel=0, abs=0)
        assert served != pytest.approx(refit)

    def test_it_writes_no_graded_artifact_and_an_ok_manifest(self, store, settings) -> None:
        """No shadow, no cross-section: the arm's graded series is untouched,
        and a served session can never become a fit of record."""
        _fit_on_the_arc(store, settings)
        base = _recipes(settings)["base"]
        _seat(store, base.arm_id)
        _serve(store, settings)

        assert not store.exists(shadow_key(base.arm_id, SERVE_DAY))
        assert not store.exists(cross_section_key(base.arm_id, SERVE_DAY))
        manifest = _manifest(store)
        assert manifest["status"] == "ok", manifest["reason"]
        (metric,) = [m for m in manifest["metrics"] if m["name"] == DAILY_FEED_METRIC]
        assert metric["value"] == 1.0
        outputs = {o["key"] for o in manifest["outputs"]}
        assert predictions_key(SERVE_DAY) in outputs
        assert arm_predictions_key(base.arm_id, SERVE_DAY) in outputs

    def test_a_rerun_serves_the_written_cross_section_as_it_stands(self, store, settings) -> None:
        _fit_on_the_arc(store, settings)
        base = _recipes(settings)["base"]
        _seat(store, base.arm_id)
        _serve(store, settings)
        first = store.get_bytes(predictions_key(SERVE_DAY))

        again = _serve(store, settings)

        assert again["scored_from_fit_of"] == {}
        assert again["reused"] == [base.arm_id]
        assert store.get_bytes(predictions_key(SERVE_DAY)) == first

    def test_a_stacked_champion_scores_its_base_first(self, store, settings) -> None:
        """`stacked` reads `predictions[base]` for the session it scores, so
        the base is scored for the serve day too, from ITS fit of record."""
        recipes = _recipes(settings)
        # The base's history over the stack's training window, then the arc.
        for day in SESSIONS[32:45]:
            _fit_on_the_arc(store, settings, day, arm="base")
        _fit_on_the_arc(store, settings, arm=None)
        _seat(store, recipes["stacked"].arm_id)

        result = _serve(store, settings)

        _trader_resolves(store, SERVE_DAY)
        assert result["scored_from_fit_of"] == {
            recipes["base"].arm_id: FIT_DAY,
            recipes["stacked"].arm_id: FIT_DAY,
        }
        assert store.exists(arm_predictions_key(recipes["base"].arm_id, SERVE_DAY))


class TestNoChampionOwesNoFeed:
    def test_an_absent_pointer_writes_no_feed_and_an_ok_manifest(self, store, settings) -> None:
        _fit_on_the_arc(store, settings)
        result = _serve(store, settings)

        assert result["champion"] is None and result["champion_feed"] is None
        assert not store.exists(predictions_key(SERVE_DAY))
        manifest = _manifest(store)
        assert manifest["status"] == "ok"
        (metric,) = [m for m in manifest["metrics"] if m["name"] == DAILY_FEED_METRIC]
        assert metric["value"] == 0.0
        assert "no champion" in metric["status_reason"]


class TestEveryOtherFailureIsLoudAndServesNothing:
    def _assert_failed_without_a_feed(self, store, day: str = SERVE_DAY) -> None:
        assert not store.exists(predictions_key(day))
        assert _manifest(store, day)["status"] == "failed"

    def test_a_champion_that_was_never_fitted_is_refused(self, store, settings) -> None:
        _seat(store, _recipes(settings)["base"].arm_id)
        with pytest.raises(MissingArtifactError, match="never been fitted"):
            _serve(store, settings)
        self._assert_failed_without_a_feed(store)

    def test_a_champion_no_recipe_declares_is_refused(self, store, settings) -> None:
        _fit_on_the_arc(store, settings)
        _seat(store, "m:not_in_the_release:0123456789ab")
        with pytest.raises(MissingArtifactError, match="no recipe in the release"):
            _serve(store, settings)
        self._assert_failed_without_a_feed(store)

    def test_a_fit_older_than_the_declared_refit_cadence_is_refused(self, store, settings) -> None:
        """A missed weekly arc must page, not quietly serve a staler fit."""
        _fit_on_the_arc(store, settings)
        _seat(store, _recipes(settings)["base"].arm_id)
        with pytest.raises(MissingArtifactError, match="refit_cadence_trading_days=5"):
            _serve(store, settings, TOO_LATE)
        self._assert_failed_without_a_feed(store, TOO_LATE)

    def test_a_session_whose_features_are_missing_is_refused(
        self, store, settings, tmp_path
    ) -> None:
        """`data.daily` did not compile the session: nothing is scored from
        another day's features, and the refusal names the job that compiles it."""
        _fit_on_the_arc(store, settings)
        _seat(store, _recipes(settings)["base"].arm_id)
        (tmp_path / "store" / features_key(DEFAULT_FEATURE_VERSION, SERVE_DAY)).unlink()
        with pytest.raises(MissingArtifactError, match="data.daily --date"):
            _serve(store, settings)
        self._assert_failed_without_a_feed(store)

    def test_a_fit_that_does_not_reproduce_its_published_cross_section_is_refused(
        self, store, settings
    ) -> None:
        """The inputs under the fit changed after it was graded (a healed
        feature day inside its window): the weights that would be served are
        not the weights that were graded."""
        _fit_on_the_arc(store, settings)
        base = _recipes(settings)["base"]
        _seat(store, base.arm_id)
        key = arm_predictions_key(base.arm_id, FIT_DAY)
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        document["predicted_alpha"][_NAMES[0]] += 0.01
        store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))

        with pytest.raises(FitReconstructionError, match="does not reproduce"):
            _serve(store, settings)
        self._assert_failed_without_a_feed(store)

    def test_a_non_trading_day_is_refused(self, store, settings) -> None:
        _fit_on_the_arc(store, settings)
        _seat(store, _recipes(settings)["base"].arm_id)
        with pytest.raises(ValueError, match="not an NYSE trading day|not a trading day"):
            serve_daily(_NonTradingDayCtx(store), settings=settings)
        assert not store.exists(predictions_key(SATURDAY))


class _NonTradingDayCtx:
    """The one input the refusal reads: the run's own trading day."""

    def __init__(self, store) -> None:
        self.store = store
        self.trading_day = dt.date.fromisoformat(SATURDAY)


class TestTheJobSurface:
    def test_m_u_and_s_are_the_slots_with_a_daily_serving_entry_point(self) -> None:
        """M's feed (I12047), then U's cut and S's construction inputs, which
        need it (I12021). R has none: nothing reads an R feed between arcs."""
        assert set(daily_servers()) == {"m", "u", "s"}

    def _args(self, tmp_path, **over) -> argparse.Namespace:
        base = dict(
            job=DAILY_SERVE_JOB,
            slot="m",
            date=None,
            store=str(tmp_path),
            dry_run=False,
            run_mode="replay",
        )
        base.update(over)
        ns = argparse.Namespace(**base)
        ns.trading_day = dt.date(2026, 9, 4)
        return ns

    def test_a_slot_with_no_daily_entry_point_is_refused_by_name(self, tmp_path) -> None:
        with pytest.raises(SystemExit, match="no daily serving entry point"):
            track_a.handle_serve_daily(self._args(tmp_path, slot="r"))

    def test_a_holiday_firing_files_its_own_manifest_and_serves_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """Labor Day 2026-09-07: `--date` absent resolves to Friday 09-04,
        whose real manifest must not be overwritten."""
        monkeypatch.setattr(track_a, "_today", lambda: dt.date(2026, 9, 7))
        monkeypatch.setenv("CRUCIBLE_STORE", str(tmp_path))
        assert track_a.handle_serve_daily(self._args(tmp_path)) == 0
        store = LocalStore(tmp_path)
        assert not store.exists(manifest_key(DAILY_SERVE_JOB, "2026-09-04", discriminator="m"))
        holiday = manifest_key(DAILY_SERVE_JOB, "2026-09-04", discriminator="m.2026-09-07")
        assert json.loads(store.get_bytes(holiday))["status"] == "ok"
        assert not store.exists(predictions_key("2026-09-04"))
