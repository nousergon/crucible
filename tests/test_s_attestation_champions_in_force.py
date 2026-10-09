"""The §9.1 attestation re-resolves each S session under the champions IN FORCE
on that session, not today's.

`alpha-engine-config-I12055`. `crucible.slots.strategy._attest` used to
re-resolve every recorded session through `resolve_session`, which reads
TODAY's `champions/m/current.json` and `champions/u/current.json`. With daily
S sessions only the champion of the day is scored — `serve.daily --slot m`
writes that champion's predictions, `--slot u` that champion's cut — so after
the next M or U promotion the new champion has no document for the earlier
sessions, re-resolution fails on every one of them, coverage drops, and every
S arm fails its serving precondition until the window ages out.

The champions in force are read off the keys the session RECORDED
(`alpha_source`, `eligibility_source`), so `session_inputs.v1` is unchanged;
the attestation still re-reads those keys' CURRENT bytes, so an upstream
revision stays a non-zero delta.

Everything drives the real chain: the S world, the real `serve.daily --slot s`
path and a real `experiment.grade`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tests.test_serve_daily_s as daily
import tests.test_slot_strategy_cycle_job as cycle_job
from crucible.documents import load_store_document
from crucible.keys import (
    arm_predictions_key,
    champion_key,
    session_inputs_key,
    shadow_key,
    universe_members_key,
)
from crucible.slots.strategy import (
    ResolvedSession,
    resolve_recorded_session,
    resolve_session,
)
from tests.test_slot_strategy_cycle_job import M_CHAMPION, U_CHAMPION

world = cycle_job.world
chain = daily.chain

#: The arms a later promotion seats. Neither has written anything for the
#: sessions the S arm recorded, which is exactly the daily-session state.
NEW_M = "m:fixture_promoted:cccccccccccc"
NEW_U = "u:fixture_promoted:dddddddddddd"


def _seat(store, slot: str, arm_id: str) -> None:
    store.put_bytes(
        champion_key(slot),
        json.dumps(
            {"schema_version": "champion_pointer.v1", "slot": slot, "arm_id": arm_id}
        ).encode("utf-8"),
    )


def _served(chain):
    store, settings, _root, days = chain
    for day in days[1:]:
        daily._serve(store, settings, day)
    return store, settings, days


def _attestation(store, settings) -> dict:
    result, manifest, _ctx = cycle_job._run_grade(store, settings)
    assert manifest["status"] == "ok"
    return result["strategy_grades"][daily._arm(settings).arm_id]


class TestAPromotionLeavesThePastAttestable:
    def test_after_an_m_promotion_every_session_attests_in_full(self, chain) -> None:
        """The defect, reproduced: the new M champion has no predictions for
        any recorded session, so today's pointer re-resolves none of them."""
        store, settings, days = _served(chain)
        _seat(store, "m", NEW_M)
        for day in days:
            assert not store.exists(arm_predictions_key(NEW_M, day.isoformat()))
        graded = _attestation(store, settings)
        assert graded["attestation_coverage"] == 1.0
        assert graded["attestation_mean_delta"] == 0.0
        assert graded["attestation"] == "PASS"

    def test_after_a_u_promotion_every_session_attests_in_full(self, chain) -> None:
        """The arc session recorded the old U champion's shadow and the daily
        ones its feed; the new U champion wrote neither."""
        store, settings, _days = _served(chain)
        _seat(store, "u", NEW_U)
        graded = _attestation(store, settings)
        assert graded["attestation_coverage"] == 1.0
        assert graded["attestation_mean_delta"] == 0.0

    def test_a_revised_upstream_document_still_moves_the_attestation(self, chain) -> None:
        """Re-resolution re-reads the recorded keys' CURRENT bytes: revising
        the previous champion's predictions after the promotion is still a
        contamination the attestation sees."""
        store, settings, days = _served(chain)
        _seat(store, "m", NEW_M)
        rng = np.random.default_rng(99)
        for day in days:
            key = arm_predictions_key(M_CHAMPION, day.isoformat())
            document = load_store_document(store, key)
            document["predicted_alpha"] = {
                ticker: float(rng.normal(0.01, 0.02)) for ticker in document["predicted_alpha"]
            }
            store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))
        graded = _attestation(store, settings)
        assert graded["attestation_coverage"] == 1.0
        assert graded["attestation_mean_delta"] != 0.0


class TestTheRecordedSourcesAreReadBack:
    def _recorded(self, store, settings, day) -> ResolvedSession:
        key = session_inputs_key(daily._arm(settings).arm_id, day.isoformat())
        return ResolvedSession.from_dict(load_store_document(store, key))

    def test_an_unrevised_session_re_resolves_to_what_it_recorded(self, chain) -> None:
        store, settings, days = _served(chain)
        _seat(store, "m", NEW_M)
        _seat(store, "u", NEW_U)
        for day in days[:-1]:
            recorded = self._recorded(store, settings, day)
            assert resolve_recorded_session(store, recorded) == recorded

    def test_it_reads_the_keys_the_session_named(self, chain) -> None:
        store, settings, days = _served(chain)
        arc, daily_day = days[0], days[1]
        assert self._recorded(store, settings, arc).eligibility_source == shadow_key(
            U_CHAMPION, arc.isoformat()
        )
        assert self._recorded(
            store, settings, daily_day
        ).eligibility_source == universe_members_key(daily_day.isoformat())

    def test_a_recorded_absence_of_a_u_champion_stays_an_absence(self, chain) -> None:
        """Seating a U champion later does not cut a session that had none."""
        store, settings, days = _served(chain)
        (store.root / champion_key("u")).unlink()
        recorded = resolve_session(store, trading_day=days[1].isoformat())
        assert recorded.eligibility_source.startswith("absent")
        _seat(store, "u", NEW_U)
        assert resolve_recorded_session(store, recorded) == recorded

    def test_an_alpha_source_that_names_no_arm_predictions_is_refused(self, chain) -> None:
        store, settings, days = _served(chain)
        recorded = self._recorded(store, settings, days[1])
        forged = ResolvedSession.from_dict(
            {**recorded.to_dict(), "alpha_source": f"predictions/{days[1].isoformat()}.json"}
        )
        with pytest.raises(ValueError, match="names no arm's predictions"):
            resolve_recorded_session(store, forged)

    def test_an_eligibility_source_it_cannot_read_is_refused(self, chain) -> None:
        store, settings, days = _served(chain)
        recorded = self._recorded(store, settings, days[1])
        forged = ResolvedSession.from_dict(
            {**recorded.to_dict(), "eligibility_source": "universe/elsewhere.json"}
        )
        with pytest.raises(ValueError, match="names no U cut"):
            resolve_recorded_session(store, forged)
