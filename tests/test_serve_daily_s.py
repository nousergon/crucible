"""`serve.daily --slot u` and `--slot s` — the S half of the nightly chain.

`alpha-engine-config-I12021`, defect A, under Brian's 2026-10-05 ruling
("Daily producer"). The daily shadow books construct every live S challenger's
book from `experiments/{arm}/{day}/session_inputs.json` for EACH decision day,
and the only writer of that document was `experiment.run --slot s` on the
weekly arc — so measured on 2026-10-05, every book for 2026-10-01 failed on a
missing document. The S inputs need the M champion's predictions for the day
(`serve.daily --slot m`, crucible-PR386) and the U champion's cut for the day,
whose only writer was also the weekly arc.

Everything here drives the real path: the S cycle job's own world (a real
panel, real M predictions, a real strategy tree), the real `run_job` wrapper,
the real resolver the trader's shadow books and the S grade both call, and a
real `experiment.grade`. Dates are the S world's fixed sessions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pytest

import crucible.track_a as track_a
import tests.test_slot_strategy_cycle_job as cycle_job
from crucible.config import Settings
from crucible.data import run_daily
from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.documents import load_store_document
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import (
    arm_predictions_key,
    arm_register_key,
    champion_key,
    cross_section_key,
    features_key,
    manifest_key,
    session_inputs_key,
    shadow_key,
    universe_members_key,
)
from crucible.runner import run_job
from crucible.slots import daily_servers, universe
from crucible.slots.arms import load_arm_specs, read_register
from crucible.slots.cycle import DAILY_FEED_METRIC, MissingArtifactError
from crucible.slots.inputs import resolve_strategy_sessions
from crucible.slots.strategy import (
    DAILY_INPUTS_METRIC,
    DAILY_SERVE_JOB,
    UPSTREAM_CHAMPION_ABSENT_METRIC,
    load_strategy_slot,
    produce,
    registration_specs,
    serve_daily,
)
from crucible.store import LocalStore
from tests.conftest import sessions_ending
from tests.test_slot_strategy_cycle_job import AS_OF, M_CHAMPION, U_CHAMPION, _recipe_yaml

#: The S cycle job's own world, re-exported so the daily chain runs over the
#: SAME store the weekly path's tests grade.
world = cycle_job.world


def _arm(settings):
    return registration_specs(load_strategy_slot(strategy_dir=settings.strategy_dir))[0]


def _remove(settings: Settings, key: str) -> None:
    (Path(settings.store_uri) / key).unlink()


def _as_a_daily_u_session(store, settings, day: dt.date) -> None:
    """Turn ``day`` into a session the weekly arc did NOT produce.

    The world writes the U champion's `shadow.json` on every session, which is
    only true on an arc day. On every other session the one U artifact is the
    champion feed `serve.daily --slot u` publishes — the republication of the
    same selection — so that is what replaces it here.
    """
    shadow = load_store_document(store, shadow_key(U_CHAMPION, day.isoformat()))
    _remove(settings, shadow_key(U_CHAMPION, day.isoformat()))
    store.put_bytes(
        universe_members_key(day.isoformat()),
        json.dumps(
            {
                "schema_version": "feed.v1",
                "slot": "u",
                "trading_day": day.isoformat(),
                "champion": U_CHAMPION,
                "members": list(shadow["selection"]),
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
    )


def _arc(store, settings, day: dt.date) -> None:
    """The weekly arc's `experiment.run[s]` on ``day``: registers the arms."""
    run_job(
        "experiment.run",
        lambda c: produce(c, settings=settings),
        store=store,
        trading_day=day,
        discriminator="s",
        run_mode="replay",
    )


def _serve(store, settings, day: dt.date) -> dict:
    result: dict = {}
    run_job(
        DAILY_SERVE_JOB,
        lambda c: result.update(serve_daily(c, settings=settings)),
        store=store,
        trading_day=day,
        run_mode="replay",
        discriminator="s",
    )
    return result


def _manifest(store, day: dt.date, slot: str = "s") -> dict:
    return load_store_document(
        store, manifest_key(DAILY_SERVE_JOB, day.isoformat(), discriminator=slot)
    )


@pytest.fixture
def chain(world):
    """The arc ran on the first session; every later session is a daily one."""
    store, settings, root, days = world
    _arc(store, settings, days[0])
    for day in days[1:]:
        _as_a_daily_u_session(store, settings, day)
    return store, settings, root, days


class TestEverySessionGetsItsInputs:
    def test_each_daily_session_records_the_challengers_inputs_from_the_daily_feeds(
        self, chain
    ) -> None:
        store, settings, _root, days = chain
        arm = _arm(settings)
        for day in days[1:]:
            result = _serve(store, settings, day)
            assert result["arms"] == [arm.arm_id]
            document = load_store_document(store, session_inputs_key(arm.arm_id, day.isoformat()))
            assert document["trading_day"] == day.isoformat()
            assert document["alpha_source"] == arm_predictions_key(M_CHAMPION, day.isoformat())
            assert document["eligibility_source"] == universe_members_key(day.isoformat())
            assert sum(document["eligibility"]) == 8, "the U cut is the feed's 8 members"
            manifest = _manifest(store, day)
            assert manifest["status"] == "ok"
            (metric,) = [m for m in manifest["metrics"] if m["name"] == DAILY_INPUTS_METRIC]
            assert metric["value"] == 1.0

    def test_controls_are_a_ruled_non_book_and_get_no_document(self, chain) -> None:
        store, settings, _root, days = chain
        register = read_register(store, "s")
        controls = [a for a in register.active_arms() if "control" in a]
        assert len(controls) == 2, "the arc registered both S controls"
        result = _serve(store, settings, days[1])
        assert sorted(result["controls"]) == sorted(controls)
        for control in controls:
            assert not store.exists(session_inputs_key(control, days[1].isoformat()))

    def test_it_never_writes_the_register(self, chain) -> None:
        store, settings, _root, days = chain
        before = store.get_bytes(arm_register_key("s"))
        _serve(store, settings, days[1])
        assert store.get_bytes(arm_register_key("s")) == before


class TestTheConsumersReadEveryDailySession:
    def test_the_shadow_book_resolver_constructs_every_session(self, chain) -> None:
        """The call `crucible_trader.shadow_inputs` makes, over the whole run."""
        store, settings, _root, days = chain
        for day in days[1:]:
            _serve(store, settings, day)
        arm = _arm(settings)
        decided = [d.isoformat() for d in days[:-1]]
        inputs = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=decided,
            as_of=AS_OF.isoformat(),
        )
        assert inputs.settled is True
        assert inputs.session_inputs_keys == tuple(
            session_inputs_key(arm.arm_id, d) for d in decided
        )

    def test_without_the_daily_producer_a_non_arc_session_has_no_book(self, chain) -> None:
        """The defect, reproduced: the arc alone leaves day two unconstructible."""
        store, settings, _root, days = chain
        arm = _arm(settings)
        with pytest.raises(MissingArtifactError, match="recorded no construction inputs"):
            resolve_strategy_sessions(
                store,
                arm_id=arm.arm_id,
                benchmark=arm.recipe.benchmark,
                decision_days=[days[1].isoformat()],
                as_of=days[2].isoformat(),
            )

    def test_the_grade_attests_every_daily_session_in_full(self, chain) -> None:
        """The §9.1 attestation re-resolves every recorded session from TODAY's
        upstream artifacts. On a daily session the U cut is the feed, so a
        resolver that knew only the arc's shadow would cover one session in
        seven and fail the arm's serving precondition."""
        store, settings, _root, days = chain
        for day in days[1:]:
            _serve(store, settings, day)
        result, manifest, _ctx = cycle_job._run_grade(store, settings)
        assert manifest["status"] == "ok"
        graded = result["strategy_grades"][_arm(settings).arm_id]
        assert graded["attestation_coverage"] == 1.0
        assert graded["attestation_mean_delta"] == 0.0


class TestRecordedInputsAreImmutable:
    def test_a_rerun_leaves_the_recorded_session_as_it_stands(self, chain) -> None:
        store, settings, _root, days = chain
        arm = _arm(settings)
        _serve(store, settings, days[1])
        key = session_inputs_key(arm.arm_id, days[1].isoformat())
        recorded = store.get_bytes(key)
        alpha = arm_predictions_key(M_CHAMPION, days[1].isoformat())
        revised = load_store_document(store, alpha)
        revised["predicted_alpha"] = {t: 0.5 for t in revised["predicted_alpha"]}
        store.put_bytes(alpha, json.dumps(revised).encode("utf-8"))

        result = _serve(store, settings, days[1])
        assert result["arms"] == []
        assert result["reused"] == [arm.arm_id]
        assert store.get_bytes(key) == recorded

    def test_the_arc_days_inputs_are_served_as_the_arc_recorded_them(self, chain) -> None:
        store, settings, _root, days = chain
        arm = _arm(settings)
        recorded = store.get_bytes(session_inputs_key(arm.arm_id, days[0].isoformat()))
        result = _serve(store, settings, days[0])
        assert result["reused"] == [arm.arm_id]
        assert store.get_bytes(session_inputs_key(arm.arm_id, days[0].isoformat())) == recorded


class TestWhatIsOwedAndWhatFails:
    def test_no_m_champion_is_the_one_declared_outcome(self, chain) -> None:
        store, settings, _root, days = chain
        _remove(settings, champion_key("m"))
        result = _serve(store, settings, days[1])
        assert result["declared_outcome"] == UPSTREAM_CHAMPION_ABSENT_METRIC
        manifest = _manifest(store, days[1])
        assert manifest["status"] == "ok"
        assert not store.exists(session_inputs_key(_arm(settings).arm_id, days[1].isoformat()))

    def test_no_m_predictions_for_the_session_fails_and_records_nothing(self, chain) -> None:
        store, settings, _root, days = chain
        _remove(settings, arm_predictions_key(M_CHAMPION, days[1].isoformat()))
        with pytest.raises(MissingArtifactError, match="serve.daily --slot m"):
            _serve(store, settings, days[1])
        assert _manifest(store, days[1])["status"] == "failed"
        assert not store.exists(session_inputs_key(_arm(settings).arm_id, days[1].isoformat()))

    def test_no_u_cut_for_the_session_fails_naming_the_u_job(self, chain) -> None:
        store, settings, _root, days = chain
        _remove(settings, universe_members_key(days[1].isoformat()))
        with pytest.raises(MissingArtifactError, match="serve.daily --slot u"):
            _serve(store, settings, days[1])

    def test_a_u_feed_for_another_champion_is_not_this_sessions_cut(self, chain) -> None:
        store, settings, _root, days = chain
        key = universe_members_key(days[1].isoformat())
        feed = load_store_document(store, key)
        feed["champion"] = "u:someone_else:cccccccccccc"
        store.put_bytes(key, json.dumps(feed).encode("utf-8"))
        with pytest.raises(MissingArtifactError, match="not this session's eligibility mask"):
            _serve(store, settings, days[1])

    def test_a_live_challenger_with_no_recipe_fails_after_the_others_are_recorded(
        self, chain
    ) -> None:
        store, settings, root, days = chain
        extra = root / "arms" / "s" / "stock_registry_wide.yaml"
        extra.write_text(
            _recipe_yaml("stock_registry_wide", registered_at=days[0].isoformat()),
            encoding="utf-8",
        )
        _arc(store, settings, days[0])  # the arc registers it
        extra.unlink()  # ...and its recipe then leaves the release
        with pytest.raises(MissingArtifactError, match="stock_registry_wide"):
            _serve(store, settings, days[1])
        assert store.exists(session_inputs_key(_arm(settings).arm_id, days[1].isoformat()))

    def test_a_recipe_the_arc_has_not_registered_is_not_owed_inputs(self, chain) -> None:
        store, settings, root, days = chain
        (root / "arms" / "s" / "stock_registry_new.yaml").write_text(
            _recipe_yaml("stock_registry_new", registered_at=days[0].isoformat()),
            encoding="utf-8",
        )
        result = _serve(store, settings, days[1])
        by_name = {
            s.name: s.arm_id
            for s in registration_specs(load_strategy_slot(strategy_dir=settings.strategy_dir))
        }
        assert result["arms"] == [by_name["stock_registry"]]
        assert not store.exists(
            session_inputs_key(by_name["stock_registry_new"], days[1].isoformat())
        )
        assert _manifest(store, days[1])["status"] == "ok"


# ---------------------------------------------------------------------------
# The U half: the champion's cut on every session.
# ---------------------------------------------------------------------------


@pytest.fixture
def u_settings(tmp_path, strategy_dir) -> Settings:
    return Settings(
        store_uri=str(tmp_path / "store"),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )


@pytest.fixture
def u_day(store, source, cycle_date) -> dt.date:
    """One compiled session: its feature layer and price panel, by the real job."""
    (day,) = sessions_ending(cycle_date, 1)
    run_job(
        "data.daily",
        lambda c: run_daily(
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
    return day


def _seat_u(store, arm_id: str) -> None:
    store.put_bytes(
        champion_key("u"),
        json.dumps({"schema_version": "champion_pointer.v1", "slot": "u", "arm_id": arm_id}).encode(
            "utf-8"
        ),
    )


def _serve_u(store, settings, day: dt.date) -> dict:
    result: dict = {}
    run_job(
        DAILY_SERVE_JOB,
        lambda c: result.update(universe.serve_daily(c, settings=settings)),
        store=store,
        trading_day=day,
        run_mode="replay",
        discriminator="u",
    )
    return result


class TestTheUChampionsCut:
    def test_it_publishes_the_champions_selection_and_no_graded_artifact(
        self, store, u_settings, u_day
    ) -> None:
        specs = {s.name: s for s in load_arm_specs("u", strategy_dir=u_settings.strategy_dir)}
        champion = specs["momentum_sleeve"]
        _seat_u(store, champion.arm_id)
        result = _serve_u(store, u_settings, u_day)

        feed = load_store_document(store, universe_members_key(u_day.isoformat()))
        assert feed["champion"] == champion.arm_id == result["champion"]
        assert feed["trading_day"] == u_day.isoformat()
        assert len(feed["members"]) == 8, "momentum_sleeve's declared top_n"
        for spec in specs.values():
            assert not store.exists(shadow_key(spec.arm_id, u_day.isoformat()))
            assert not store.exists(cross_section_key(spec.arm_id, u_day.isoformat()))
        assert not store.exists(arm_register_key("u"))
        manifest = _manifest(store, u_day, "u")
        assert manifest["status"] == "ok"
        assert features_key(DEFAULT_FEATURE_VERSION, u_day.isoformat()) in {
            i["key"] for i in manifest["inputs"]
        }
        (metric,) = [m for m in manifest["metrics"] if m["name"] == DAILY_FEED_METRIC]
        assert metric["value"] == 1.0

    def test_the_cut_is_the_one_the_arc_would_publish(self, store, u_settings, u_day) -> None:
        specs = {s.name: s for s in load_arm_specs("u", strategy_dir=u_settings.strategy_dir)}
        _seat_u(store, specs["momentum_sleeve"].arm_id)
        _serve_u(store, u_settings, u_day)
        daily = load_store_document(store, universe_members_key(u_day.isoformat()))
        _remove(u_settings, universe_members_key(u_day.isoformat()))
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=u_settings),
            store=store,
            trading_day=u_day,
            discriminator="u",
            run_mode="replay",
        )
        assert load_store_document(store, universe_members_key(u_day.isoformat())) == daily

    def test_no_champion_owes_no_feed(self, store, u_settings, u_day) -> None:
        result = _serve_u(store, u_settings, u_day)
        assert result["champion"] is None
        assert not store.exists(universe_members_key(u_day.isoformat()))
        (metric,) = [
            m for m in _manifest(store, u_day, "u")["metrics"] if m["name"] == DAILY_FEED_METRIC
        ]
        assert metric["value"] == 0.0

    def test_a_published_feed_is_served_as_it_stands(self, store, u_settings, u_day) -> None:
        specs = {s.name: s for s in load_arm_specs("u", strategy_dir=u_settings.strategy_dir)}
        _seat_u(store, specs["momentum_sleeve"].arm_id)
        key = universe_members_key(u_day.isoformat())
        written = json.dumps(
            {
                "schema_version": "feed.v1",
                "slot": "u",
                "trading_day": u_day.isoformat(),
                "champion": specs["momentum_sleeve"].arm_id,
                "members": ["AAA"],
            }
        ).encode("utf-8")
        store.put_bytes(key, written)
        assert _serve_u(store, u_settings, u_day)["reused"] is True
        assert store.get_bytes(key) == written

    def test_a_misfiled_feed_is_refused_and_never_overwritten(
        self, store, u_settings, u_day
    ) -> None:
        specs = {s.name: s for s in load_arm_specs("u", strategy_dir=u_settings.strategy_dir)}
        _seat_u(store, specs["momentum_sleeve"].arm_id)
        key = universe_members_key(u_day.isoformat())
        written = json.dumps(
            {"schema_version": "feed.v1", "slot": "u", "trading_day": "2026-08-27", "members": []}
        ).encode("utf-8")
        store.put_bytes(key, written)
        with pytest.raises(MissingArtifactError, match="another session's key"):
            _serve_u(store, u_settings, u_day)
        assert store.get_bytes(key) == written

    def test_a_pointer_no_recipe_declares_is_refused(self, store, u_settings, u_day) -> None:
        _seat_u(store, "u:not_a_recipe:dddddddddddd")
        with pytest.raises(MissingArtifactError, match="no recipe in the release"):
            _serve_u(store, u_settings, u_day)
        assert not store.exists(universe_members_key(u_day.isoformat()))

    def test_a_session_whose_features_are_absent_is_refused(self, store, u_settings, u_day) -> None:
        specs = {s.name: s for s in load_arm_specs("u", strategy_dir=u_settings.strategy_dir)}
        _seat_u(store, specs["momentum_sleeve"].arm_id)
        _remove(u_settings, features_key(DEFAULT_FEATURE_VERSION, u_day.isoformat()))
        with pytest.raises(MissingArtifactError, match="feature layer is absent"):
            _serve_u(store, u_settings, u_day)


class TestTheJobSurface:
    def test_u_and_s_join_m_under_the_one_job(self) -> None:
        assert {"u", "s"} <= set(daily_servers())

    def test_the_cli_runs_the_s_slot_and_files_its_own_manifest(self, chain) -> None:
        store, settings, root, days = chain
        args = argparse.Namespace(
            job=DAILY_SERVE_JOB,
            slot="s",
            date=days[1],
            store=settings.store_uri,
            strategy_dir=str(root),
            dry_run=False,
            run_mode="replay",
        )
        args.trading_day = days[1]
        assert track_a.handle_serve_daily(args) == 0
        assert _manifest(LocalStore(Path(settings.store_uri)), days[1])["status"] == "ok"
        assert store.exists(session_inputs_key(_arm(settings).arm_id, days[1].isoformat()))
