"""`crucible experiment.backfill` — a base arm's prediction history, on demand.

`alpha-engine-config-I10696`. A stacked M arm reads
`predictions[<base>]` on every row of its training window, so it refuses at
registration until the base has an `arm_predictions` artifact for every
session of that window. A weekly arc adds ONE base session per week, so a
504-session window is 504 weeks away — a stacked arm could never register
from weekly runs alone. This job produces the missing sessions the only way
that is honest: the SAME per-arm produce path `experiment.run` calls, once
per session, point-in-time.

Every test drives the real path — a real feature layer on disk, the real
loader, the real `crucible.runner.run_job` wrapper and a real manifest in a
real store. A test that asserted a call site would pass over a job whose
rows reached nothing.

Dates are fixed literals off a pinned session axis, never `today` arithmetic
(AGENTS.md test discipline).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from crucible.backfill import (
    BackfillProducedNothingError,
    UnknownArmError,
    run_backfill,
)
from crucible.calendar import is_trading_day
from crucible.data.heal import LAPTOP_SESSION_ALLOWANCE, NotInRegionError
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import (
    arm_predictions_key,
    backfill_key,
    cross_section_key,
    data_panel_key,
    manifest_key,
    shadow_key,
)
from crucible.runner import run_job
from crucible.slots.model import (
    FEATURE_COMPLETENESS_METRIC,
    SLOT,
    load_model_recipes,
    produce,
    produce_history,
    registration_specs,
)
from crucible.store import LocalStore

BASE_COLUMN = "momentum_20d_zscore"
SECOND_COLUMN = "volatility_20d_ratio"

_START = dt.date(2026, 6, 1)
_SESSIONS = 60
_NAMES = (
    "AAA",
    "BBB",
    "CCC",
    "DDD",
    "EEE",
    "FFF",
    "GGG",
    "HHH",
    "III",
    "JJJ",
    "KKK",
    "LLL",
)


def _sessions() -> tuple[str, ...]:
    days: list[str] = []
    day = _START
    while len(days) < _SESSIONS:
        if is_trading_day(day):
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    return tuple(days)


SESSIONS = _sessions()

#: The window a stacked arm on `min_trading_days: 10` + a 2-session label
#: horizon needs its base to have already predicted, and the anchor it is
#: then producible on.
WARMUP_FROM = SESSIONS[32]
WARMUP_TO = SESSIONS[45]
RUN_DAY = SESSIONS[45]


@pytest.fixture
def store(tmp_path):
    return _seeded_store(tmp_path / "store")


def _seeded_store(root):
    import pandas as pd

    backing = LocalStore(root)
    rng = np.random.default_rng(20260914)
    rows = []
    for i, day in enumerate(SESSIONS):
        closes = 100.0 + i * 0.5 + rng.normal(0.0, 1.0, len(_NAMES))
        frame = pd.DataFrame(
            {
                "ticker": list(_NAMES),
                "close_raw": closes,
                BASE_COLUMN: rng.normal(0.0, 1.0, len(_NAMES)),
                SECOND_COLUMN: rng.normal(1.0, 0.3, len(_NAMES)),
            }
        )
        backing.put_bytes(
            f"features/{DEFAULT_FEATURE_VERSION}/{day}.parquet", frame.to_parquet(index=False)
        )
        rows.extend(
            {"trading_day": dt.date.fromisoformat(day), "ticker": t, "close_raw": float(c)}
            for t, c in zip(_NAMES, closes, strict=True)
        )
    panel = pd.DataFrame(rows)
    for day in SESSIONS:
        backing.put_bytes(data_panel_key(day), panel.to_parquet(index=False))
    return backing


def _write_recipe(directory, name, *, features, inputs=(), min_days=10):
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["slot: m", f"name: {name}", "spec:", f"  features: [{', '.join(features)}]"]
    if inputs:
        lines.append("  inputs:")
        lines += [f"    - {entry}" for entry in inputs]
    lines += [
        "  estimator: {kind: ridge, alpha: 1.0}",
        "  label_horizon_trading_days: 2",
        "  refit_cadence_trading_days: 5",
        f"  training_window: {{kind: expanding, min_trading_days: {min_days}}}",
        "  cpcv: {n_groups: 4, k_test: 1, embargo_trading_days: 1}",
        f"registered_at: '{SESSIONS[0]}'",
    ]
    (directory / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Settings:
    def __init__(self, strategy_dir=None):
        self.strategy_dir = strategy_dir


@pytest.fixture
def strategy(tmp_path):
    arms = tmp_path / "strategy" / "arms" / SLOT
    _write_recipe(arms, "base", features=[BASE_COLUMN])
    _write_recipe(arms, "stacked", features=[SECOND_COLUMN], inputs=["predictions[base]"])
    return _Settings(tmp_path / "strategy")


def _specs(settings):
    loaded = load_model_recipes(settings.strategy_dir / "arms" / SLOT)
    return registration_specs(loaded)


def _backfill(store, settings, *, arm="base", start=WARMUP_FROM, end=WARMUP_TO, **kwargs):
    result: dict = {}
    ctx = run_job(
        "experiment.backfill",
        lambda c: result.update(
            run_backfill(
                c,
                produce=produce_history,
                specs=_specs(settings),
                settings=settings,
                slot=SLOT,
                arm=arm,
                start=dt.date.fromisoformat(start),
                end=dt.date.fromisoformat(end),
                i_am_in_region=kwargs.pop("i_am_in_region", True),
                **kwargs,
            )
        ),
        store=store,
        trading_day=dt.date.fromisoformat(end),
        run_mode="replay",
        discriminator=f"{SLOT}.{arm}",
    )
    return ctx, result


def _manifest(store, arm, day):
    key = manifest_key("experiment.backfill", day, discriminator=f"{SLOT}.{arm}")
    return json.loads(store.get_bytes(key).decode("utf-8"))


class TestTheRangeIsProducedThroughTheRealProducePath:
    def test_every_session_in_the_range_gets_the_three_per_session_artifacts(
        self, store, strategy
    ) -> None:
        _, result = _backfill(store, strategy)
        arm_id = result["arm_id"]
        assert len(result["sessions"]) == len(result["produced"]) > 10
        for day in result["sessions"]:
            assert store.exists(arm_predictions_key(arm_id, day))
            assert store.exists(shadow_key(arm_id, day))
            assert store.exists(cross_section_key(arm_id, day))

    def test_the_backfilled_history_is_what_lets_a_stacked_arm_produce(
        self, store, strategy
    ) -> None:
        """The issue's own closes-when, end to end: before the backfill the
        stacked arm refuses on its base's missing history; after it, the
        slot produces both arms on the same session."""
        _backfill(store, strategy)
        produced: dict = {}
        run_job(
            "experiment.run",
            lambda c: produced.update(produce(c, settings=strategy)),
            store=store,
            trading_day=dt.date.fromisoformat(RUN_DAY),
            run_mode="replay",
            discriminator=SLOT,
        )
        assert len(produced["arms"]) == 2, produced


class TestOneManifestForTheWholeRange:
    def test_the_manifest_is_keyed_to_to_and_counts_sessions(self, store, strategy) -> None:
        _, result = _backfill(store, strategy)
        document = _manifest(store, "base", WARMUP_TO)
        assert document["status"] == "ok", document["reason"]
        assert document["job"] == "experiment.backfill"
        assert document["trading_day"] == WARMUP_TO
        assert document["run_mode"] == "replay"
        assert document["rows_in"] == len(result["sessions"])
        assert document["rows_out"] == len(result["sessions"])
        assert document["rows_rejected"] == []

    def test_the_per_session_feature_completeness_records_are_on_it(self, store, strategy) -> None:
        _, result = _backfill(store, strategy)
        document = _manifest(store, "base", WARMUP_TO)
        rows = [m for m in document["metrics"] if m["name"] == FEATURE_COMPLETENESS_METRIC]
        assert len(rows) >= len(result["sessions"])

    def test_the_result_document_is_written_and_named_on_the_manifest(
        self, store, strategy
    ) -> None:
        ctx, result = _backfill(store, strategy)
        key = backfill_key(WARMUP_TO, ctx.run_id)
        assert store.exists(key)
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        assert document["schema_version"] == "backfill.v1"
        assert document["arm"] == "base"
        assert document["produced"] == result["produced"]
        assert key in {o["key"] for o in _manifest(store, "base", WARMUP_TO)["outputs"]}


class TestIdempotence:
    def test_a_session_whose_artifacts_exist_is_skipped(self, store, strategy) -> None:
        _backfill(store, strategy)
        _, second = _backfill(store, strategy)
        assert second["produced"] == []
        assert second["already_present"] == second["sessions"]

    def test_force_reproduces_every_session(self, store, strategy) -> None:
        _backfill(store, strategy)
        _, second = _backfill(store, strategy, force=True)
        assert second["already_present"] == []
        assert second["produced"] == second["sessions"]


class TestItRefusesRatherThanGuesses:
    def test_a_bulk_range_off_ec2_refuses_and_names_this_job(
        self, store, strategy, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "crucible.backfill.in_region", lambda: (False, "IMDSv2 did not answer: a laptop")
        )
        with pytest.raises(NotInRegionError) as excinfo:
            _backfill(store, strategy, i_am_in_region=False)
        message = str(excinfo.value)
        assert "crucible experiment.backfill" in message
        assert f"--from {WARMUP_FROM} --to {WARMUP_TO}" in message
        assert str(LAPTOP_SESSION_ALLOWANCE) in message

    def test_an_arm_the_slot_does_not_register_is_refused(self, store, strategy) -> None:
        with pytest.raises(UnknownArmError) as excinfo:
            _backfill(store, strategy, arm="nowhere")
        assert "nowhere" in str(excinfo.value)
        assert "base" in str(excinfo.value)

    def test_a_non_trading_day_bound_is_refused(self, store, strategy) -> None:
        with pytest.raises(ValueError, match="2026-07-03"):
            _backfill(store, strategy, start="2026-07-03", end=WARMUP_TO)

    def test_a_range_where_every_session_refuses_fails_the_job(self, store, strategy) -> None:
        """`stacked` with NO base history: nothing can be produced, so the
        job fails rather than reporting a completed backfill of nothing."""
        with pytest.raises(BackfillProducedNothingError):
            _backfill(store, strategy, arm="stacked", start=SESSIONS[40], end=SESSIONS[45])
        document = _manifest(store, "stacked", SESSIONS[45])
        assert document["status"] == "failed"
        assert document["rows_rejected"], document


class TestAPartiallySatisfiableRangeIsNamedNotSwallowed:
    def test_sessions_that_refuse_are_counted_in_rows_rejected(self, store, strategy) -> None:
        """The base is warmed over 32..45, so `stacked` can produce on the
        late sessions of 40..45 and cannot on the early ones. The refusals
        are rows_rejected on the one manifest, and the run is `ok`."""
        _backfill(store, strategy)
        _, result = _backfill(store, strategy, arm="stacked", start=SESSIONS[40], end=SESSIONS[45])
        assert result["produced"], result
        assert result["refused"], result
        document = _manifest(store, "stacked", SESSIONS[45])
        assert document["status"] == "ok", document["reason"]
        assert len(document["rows_rejected"]) == len(result["refused"])
        assert document["rows_out"] == len(result["produced"])


class TestTheRefusalTextNamesThisJob:
    def test_a_missing_base_history_sends_the_operator_to_experiment_backfill(
        self, store, strategy
    ) -> None:
        from crucible.slots.inputs import BasePredictionsUnavailableError
        from crucible.slots.model import FeatureLayerSource, design_panel

        loaded = load_model_recipes(strategy.strategy_dir / "arms" / SLOT)
        by_name = {r.name: r for r in loaded.registered}
        with pytest.raises(BasePredictionsUnavailableError) as excinfo:
            design_panel(
                by_name["stacked"],
                source=FeatureLayerSource(store=store),
                trading_day=RUN_DAY,
                recipes=loaded.registered,
                lookback_trading_days=12,
                store=store,
            )
        message = str(excinfo.value)
        assert "crucible experiment.backfill --slot m --arm base" in message
        assert "--run-mode replay" in message
        assert "experiment.run --slot m --arm base --trading-day" not in message


class TestTheCliHandler:
    """The production entry point, driven through `crucible.cli.main`.

    A test that called `run_backfill` only would pass over a handler that
    resolved the wrong store, the wrong slot module or the wrong specs —
    which is every way this job can be wired wrong and none of the ways its
    body can.
    """

    def _argv(self, store_root, strategy, **flags):
        argv = [
            "experiment.backfill",
            "--slot",
            SLOT,
            "--arm",
            "base",
            "--from",
            WARMUP_FROM,
            "--to",
            WARMUP_TO,
            "--store",
            str(store_root),
            "--strategy-dir",
            str(strategy.strategy_dir),
            "--run-mode",
            "replay",
        ]
        return argv + list(flags.get("extra", []))

    def test_the_handler_backfills_the_range_and_writes_the_manifest(
        self, store, strategy, monkeypatch
    ) -> None:
        from crucible.cli import main

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        assert main(self._argv(store.root, strategy, extra=["--i-am-in-region"])) == 0
        document = _manifest(store, "base", WARMUP_TO)
        assert document["status"] == "ok", document["reason"]
        assert document["rows_out"] == document["rows_in"] > 10

    def test_a_dry_run_executes_the_body_and_writes_nothing(
        self, store, strategy, monkeypatch, capsys
    ) -> None:
        """`alpha-engine-config-I11012`. PR323's pre-flight still runs and
        still prints — the arm RESOLVED (not merely named) and the history
        entry point named — and then the body EXECUTES, per session, against
        a store that records its writes instead of performing them.

        The sentence this used to assert, "NOT rehearsed: the per-session
        produce call itself", is gone because the thing it named is no longer
        true. That call is exactly where the measured failure lived.
        """
        from crucible.cli import main
        from crucible.store import begin_capture, end_capture

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        before = sorted(store.list_keys())
        # The test opens the ledger, so `main` sees one already active, adds
        # to it and leaves it — the same re-entrancy `crucible weekly` uses.
        ledger = begin_capture()
        try:
            assert main(self._argv(store.root, strategy, extra=["--dry-run"])) == 0
        finally:
            end_capture()
        printed = capsys.readouterr().out
        assert "experiment.backfill --slot m --arm base (m:base:" in printed
        assert "produce_history" in printed
        assert "does not enter the serving path" in printed
        assert "NOT rehearsed" not in printed
        assert sorted(store.list_keys()) == before
        # It got as far as producing sessions, which is the whole point.
        assert len(ledger.keys) > 1
        assert any(key.startswith("runs/experiment.backfill/") for key in ledger.keys)

    def test_the_reported_key_set_is_the_set_the_real_run_writes(
        self, store, strategy, monkeypatch
    ) -> None:
        """The Closes-when property (`alpha-engine-config-I11012`): the keys a
        dry run REPORTS are the keys a real run of the same command WRITES.

        Run in this order on purpose — the rehearsal first, against the store
        the real run has not touched yet, because a rehearsal run second
        would see its own outputs already present and skip every session as
        `already_present`.
        """
        from crucible.cli import main
        from crucible.store import begin_capture, end_capture

        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        before = set(store.list_keys())

        ledger = begin_capture()
        try:
            assert main(self._argv(store.root, strategy, extra=["--dry-run"])) == 0
        finally:
            end_capture()
        assert set(store.list_keys()) == before  # the rehearsal wrote nothing

        assert main(self._argv(store.root, strategy, extra=["--i-am-in-region"])) == 0
        really_written = set(store.list_keys()) - before

        # The backfill's own result document is keyed by `run_id`, which is
        # fresh per run by construction, so the two runs name different keys
        # under that ONE prefix and are compared by prefix there. Every other
        # key is compared exactly.
        # Derived from the key builder, so a rename cannot leave this
        # normalisation matching nothing and the comparison vacuously exact.
        result_prefix = backfill_key(WARMUP_TO, "run").rsplit("/", 1)[0] + "/"

        def _normalise(keys):
            return {
                (result_prefix + "<run_id>.json" if key.startswith(result_prefix) else key)
                for key in keys
            }

        assert any(key.startswith(result_prefix) for key in ledger.keys)

        assert _normalise(ledger.keys) == _normalise(really_written)


#: The prefix the backfill's own result document lands under, derived from
#: the key builder so a rename cannot leave this test excluding nothing.
_BACKFILLS_PREFIX = backfill_key(SESSIONS[0], "run").split("/", 1)[0] + "/"


class TestABackfilledSessionCarriesTheSameArtifactsAsALiveOne:
    """`alpha-engine-config-I10709`, deliverable 1 — the CLASS: two producers
    of the same session artifacts, free to diverge.

    They cannot diverge today, and this test is what keeps that true: this
    module owns no fitting code and reaches the slot only through the
    ``produce`` argument, so a backfilled session is written by the same call
    `experiment.run --slot m --arm <a>` makes. The test compares the KEY SETS
    two independently seeded stores carry after one session — the backfill's
    own manifest and result document excepted, because one job writing one
    manifest for a whole range is this job's declared shape.

    The three §5.3 veto inputs are deliberately NOT in either set: they are
    not per-session artifacts at all. `serving_metrics` derives them at GRADE
    time off the arm's settled walk-forward
    (`crucible.slots.model.grade_arm`), which is why I10709's read-side half
    lives in that function and not here.
    """

    SESSION = SESSIONS[45]

    def _live(self, store, strategy):
        run_job(
            "experiment.run",
            lambda c: produce(c, settings=strategy, arm_name="base"),
            store=store,
            trading_day=dt.date.fromisoformat(self.SESSION),
            run_mode="replay",
            discriminator=SLOT,
        )

    def _written(self, store, seeded):
        return {
            key
            for key in store.list_keys("")
            if key not in seeded and not key.startswith(("runs/", _BACKFILLS_PREFIX))
        }

    def test_the_key_sets_are_identical_for_the_same_arm_and_date(
        self, store, strategy, tmp_path
    ) -> None:
        seeded = set(store.list_keys(""))
        _backfill(store, strategy, start=self.SESSION, end=self.SESSION)
        backfilled = self._written(store, seeded)

        live_store = _seeded_store(tmp_path / "live")
        live_seeded = set(live_store.list_keys(""))
        self._live(live_store, strategy)
        live = self._written(live_store, live_seeded)

        assert backfilled, "the backfill wrote nothing outside the seed"
        assert backfilled == live, {
            "only_backfilled": sorted(backfilled - live),
            "only_live": sorted(live - backfilled),
        }

    def test_the_set_is_the_three_per_session_artifacts_plus_the_register(
        self, store, strategy
    ) -> None:
        """Named, so a key silently dropped from BOTH producers still fails:
        an equality between two empty-ish sets proves nothing."""
        seeded = set(store.list_keys(""))
        _, result = _backfill(store, strategy, start=self.SESSION, end=self.SESSION)
        arm_id = result["arm_id"]
        assert self._written(store, seeded) >= {
            arm_predictions_key(arm_id, self.SESSION),
            shadow_key(arm_id, self.SESSION),
            cross_section_key(arm_id, self.SESSION),
        }


class TestTheBackfillNeverEntersTheServingPath:
    """`alpha-engine-config-I11005` — the M half.

    Measured live 2026-09-17 on the U slot: a single-arm backfill of a
    NON-champion arm failed in two minutes on the first of 205 sessions,
    because the backfill ran the slot's production `produce`, whose serving
    half asserts that the arm the CHAMPION pointer names produced this cycle.
    Correct for production — a pointer to an arm that did not produce means
    production has no feed today — and fatal for a historical backfill, which
    feeds nothing: no challenger could ever accumulate the history it needs
    to become champion.

    M's serving half is `crucible.serving.publish_predictions_feed` rather
    than `cycle._serve_champion_feed`, and it answers an ABSENT pointer with
    `None` — so the M backfill was not blocked on 2026-09-17, when
    `champions/m/current.json` did not exist. Both shapes are covered here,
    because the absent pointer is a fact about today and the
    present-but-unproduced pointer is what M gets the day it wins its first
    champion — which is the very thing the blocked backfill exists to enable.
    """

    SESSION = SESSIONS[45]

    def _seat_champion(self, store, arm_id: str) -> None:
        """A USABLE M pointer naming ``arm_id``: an `ok` producing manifest
        (`crucible.champion._assert_producing_run_ok`) and no attestation,
        since `ATTESTED_SLOTS` is S alone."""
        from crucible.champion import (
            CHAMPION_SCHEMA_VERSION,
            ChampionPointer,
            read_champion_etag,
            write_champion,
        )

        promote_manifest = f"runs/promote/{self.SESSION}/run.json"
        store.put_bytes(
            promote_manifest,
            json.dumps({"status": "ok", "job": "promote"}).encode("utf-8"),
        )
        write_champion(
            store,
            ChampionPointer(
                schema_version=CHAMPION_SCHEMA_VERSION,
                slot=SLOT,
                arm_id=arm_id,
                as_of=self.SESSION,
                decided_at=f"{self.SESSION}T02:00:00Z",
                run_id="01JG0000000000000000000000",
                code_sha="a" * 40,
                promotion_source="evidence",
                manifest_key=promote_manifest,
                evidence={"status": "decided", "moved": True, "paired_dates": 40},
                attestation=None,
            ),
            expected=read_champion_etag(store, SLOT),
        )

    def test_a_non_champion_arm_backfills_while_the_pointer_names_another_arm(
        self, store, strategy
    ) -> None:
        """The blocker, in the M shape: a seated champion that produces
        nothing on the backfilled session must not stop the session."""
        self._seat_champion(store, "m:momentum_sleeve:2a5526115a4c")
        _, result = _backfill(store, strategy, start=self.SESSION, end=self.SESSION)
        assert result["produced"] == [self.SESSION], result
        assert store.exists(arm_predictions_key(result["arm_id"], self.SESSION))

    def test_a_serving_cycle_with_an_unproduced_champion_still_refuses(
        self, store, strategy
    ) -> None:
        """The other half, unchanged: `experiment.run` is a serving cycle and
        still fails when the pointer resolves to nothing produced — and the
        refusal still names the pointer's slot and the arm."""
        from crucible.slots.cycle import MissingArtifactError

        self._seat_champion(store, "m:momentum_sleeve:2a5526115a4c")
        with pytest.raises(MissingArtifactError) as excinfo:
            run_job(
                "experiment.run",
                lambda c: produce(c, settings=strategy, arm_name="base"),
                store=store,
                trading_day=dt.date.fromisoformat(self.SESSION),
                run_mode="replay",
                discriminator=SLOT,
            )
        message = str(excinfo.value)
        assert "'m'" in message
        assert "m:momentum_sleeve:2a5526115a4c" in message
        assert "The serving path resolves the pointer" in message

    def test_an_absent_pointer_owes_no_feed_on_either_path(self, store, strategy) -> None:
        """Today's M store: `champions/m/current.json` does not exist. The
        serving cycle reports `champion_feed: None` — a true statement about
        a slot that has never promoted — and the history path reports
        `served: False`, which is a DIFFERENT statement and deliberately so."""
        from crucible.keys import champion_key

        assert not store.exists(champion_key(SLOT))
        served: dict = {}
        run_job(
            "experiment.run",
            lambda c: served.update(produce(c, settings=strategy, arm_name="base")),
            store=store,
            trading_day=dt.date.fromisoformat(self.SESSION),
            run_mode="replay",
            discriminator=SLOT,
        )
        assert served["champion_feed"] is None

        history: dict = {}
        run_job(
            "experiment.backfill",
            lambda c: history.update(produce_history(c, settings=strategy, arm_name="base")),
            store=store,
            trading_day=dt.date.fromisoformat(self.SESSION),
            run_mode="replay",
            discriminator=f"{SLOT}.base.history",
        )
        assert history["served"] is False
        assert "champion_feed" not in history

    def test_the_history_path_publishes_no_feed_even_with_a_produced_champion(
        self, store, strategy
    ) -> None:
        """The strongest form: the champion IS the arm being backfilled, so
        the serving path would have succeeded. The history path still writes
        no trader feed — a backfill of a past session must never republish
        one under today's contract key."""
        from crucible.keys import predictions_key

        specs = {spec.name: spec.arm_id for spec in _specs(strategy)}
        self._seat_champion(store, specs["base"])
        _backfill(store, strategy, start=self.SESSION, end=self.SESSION)
        assert not store.exists(predictions_key(self.SESSION))
