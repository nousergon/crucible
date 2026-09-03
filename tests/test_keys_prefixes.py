"""The `*_prefix` helpers added by `alpha-engine-config-I9852`, and the guard
`drift_input_key` carries.

Normative source: `crucible/keys.py`'s module docstring — every store key
shape, in one place — and `AGENTS.md`'s Test discipline: "give every guard a
self-test that shows it firing: a detector nobody has made fail is a
detector nobody knows works."

**What a prefix function is FOR, and what breaks if the invariant slips.**
`experiments_prefix(arm_id)`, `features_prefix(version)`,
`strategy_arms_prefix(slot)` and `runs_prefix(job)` exist so a caller can
`store.list_keys(prefix)` and get back every key their sibling `*_key`
function would produce for that arm/version/slot/job, across every trading
day. If a prefix function's shape ever diverges from its sibling key
function's shape — a typo, a forgotten escape, an extra segment — the
`list_keys` call does not error. It returns fewer keys, or none, and that
silently reads as "no data" rather than as the bug it is. So the assertion
that matters here is not "the prefix looks right" but the literal contract:
every key the sibling function can produce STARTS WITH the prefix.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from crucible.keys import (
    DRIFT_INPUTS,
    cross_section_key,
    cross_section_settled_key,
    drift_input_key,
    drift_metrics_key,
    experiments_prefix,
    features_key,
    features_prefix,
    manifest_key,
    runs_prefix,
    shadow_key,
    strategy_arm_key,
    strategy_arms_prefix,
    verdict_key,
)
from crucible.store import LocalStore
from crucible.track_c import drift_handler

FRIDAY = dt.date(2026, 8, 28)


class TestExperimentsPrefix:
    """Both arm-id forms: one that needs `arm_key_segment` escaping (a colon)
    and one that does not, so the prefix invariant is checked against both
    code paths inside `arm_key_segment`, not just the one that happens not to
    exercise the replace."""

    @pytest.mark.parametrize(
        "arm_id",
        ["u:momentum_sleeve:ab12cd", "already_flat_no_colons"],
    )
    def test_every_dated_key_starts_with_the_prefix(self, arm_id: str) -> None:
        prefix = experiments_prefix(arm_id)
        for key in (
            shadow_key(arm_id, FRIDAY.isoformat()),
            verdict_key(arm_id, FRIDAY.isoformat()),
            cross_section_key(arm_id, FRIDAY.isoformat()),
            cross_section_settled_key(arm_id, FRIDAY.isoformat()),
        ):
            assert key.startswith(prefix), f"{key!r} does not start with {prefix!r}"


class TestFeaturesPrefix:
    @pytest.mark.parametrize("version", ["v1", "2026-08-28-v3"])
    def test_the_key_starts_with_the_prefix(self, version: str) -> None:
        prefix = features_prefix(version)
        assert features_key(version, FRIDAY.isoformat()).startswith(prefix)

    def test_an_empty_version_raises_rather_than_listing_every_version(self) -> None:
        """Fail loud (`AGENTS.md` rule 5): a blank version would list under
        `features//`, and `list_keys` returning nothing reads as 'no data',
        not as the caller's own bug."""
        with pytest.raises(ValueError):
            features_prefix("")


class TestStrategyArmsPrefix:
    @pytest.mark.parametrize("slot", ["r", "m"])
    def test_the_key_starts_with_the_prefix(self, slot: str) -> None:
        prefix = strategy_arms_prefix(slot)
        assert strategy_arm_key(slot, "momentum_sleeve").startswith(prefix)

    def test_an_empty_slot_raises_rather_than_listing_every_slot(self) -> None:
        with pytest.raises(ValueError):
            strategy_arms_prefix("")


class TestRunsPrefix:
    def test_the_key_starts_with_the_prefix(self) -> None:
        prefix = runs_prefix("report")
        assert manifest_key("report", FRIDAY.isoformat()).startswith(prefix)
        assert manifest_key("report", FRIDAY.isoformat(), discriminator="r").startswith(prefix)

    def test_an_empty_job_raises(self) -> None:
        with pytest.raises(ValueError):
            runs_prefix("")


class TestDriftKeys:
    def test_the_three_shapes_are_literal(self) -> None:
        assert drift_input_key("features", "2026-08-28") == "drift/2026-08-28/input_features.json"
        assert (
            drift_input_key("predictions", "2026-08-28")
            == "drift/2026-08-28/input_predictions.json"
        )
        assert drift_input_key("ic", "2026-08-28") == "drift/2026-08-28/input_ic.json"
        assert drift_metrics_key("2026-08-28") == "drift/2026-08-28/metrics.json"

    def test_drift_inputs_is_exhaustive_and_matches_the_key_function(self) -> None:
        assert DRIFT_INPUTS == ("features", "predictions", "ic")
        for name in DRIFT_INPUTS:
            assert drift_input_key(name, "2026-08-28")  # does not raise

    def test_an_unknown_drift_input_name_raises(self) -> None:
        """A detector nobody has made fire is a detector nobody knows works
        (`AGENTS.md` Test discipline) — this is what stops a typo like
        `input_feature.json` for `input_features.json` from silently
        producing a fourth, orphaned key shape."""
        with pytest.raises(ValueError, match="unknown drift input"):
            drift_input_key("feature", "2026-08-28")


class TestDriftHandlerKeyShape:
    """`drift_handler` (`crucible.track_c`) has no test elsewhere. Without
    this, a typo in one of the three input key names — `input_feature.json`
    for `input_features.json`, say — would leave the handler raising
    `FileNotFoundError` for artifacts that were, in fact, present at their
    correctly-spelled keys, and CI would still be green because nothing
    asserts the handler reads the SAME keys `crucible.keys` writes them
    under.
    """

    def _write_inputs(self, store: LocalStore, day: str) -> None:
        store.put_bytes(
            drift_input_key("features", day),
            json.dumps({"psi_by_feature": {"mom_21d": 0.05}}).encode("utf-8"),
        )
        store.put_bytes(
            drift_input_key("predictions", day),
            json.dumps({"psi": 0.03}).encode("utf-8"),
        )
        store.put_bytes(
            drift_input_key("ic", day),
            json.dumps({"decay_by_horizon": {"21": 0.1}}).encode("utf-8"),
        )

    def test_it_reads_the_keys_crucible_keys_writes_and_writes_its_own_output_there(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path / "store")
        day = FRIDAY.isoformat()
        self._write_inputs(store, day)

        drift_handler(argparse.Namespace(trading_day=FRIDAY, store=str(tmp_path / "store")))

        manifest = json.loads(store.get_bytes(manifest_key("drift", day)).decode("utf-8"))
        assert manifest["status"] == "ok", manifest.get("reason")
        assert store.exists(drift_metrics_key(day))
        records = json.loads(store.get_bytes(drift_metrics_key(day)).decode("utf-8"))
        assert len(records) == 3

    def test_a_missing_input_fails_the_manifest_rather_than_scoring_from_nothing(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path / "store")
        day = FRIDAY.isoformat()
        # Only two of three inputs present.
        store.put_bytes(
            drift_input_key("features", day),
            json.dumps({"psi_by_feature": {"mom_21d": 0.05}}).encode("utf-8"),
        )
        store.put_bytes(
            drift_input_key("predictions", day),
            json.dumps({"psi": 0.03}).encode("utf-8"),
        )

        # `crucible.runner.run_job` writes the failed manifest in a
        # try/finally and then lets the exception propagate (`AGENTS.md`
        # rule 1: "the exception continues to propagate so the process exits
        # non-zero") — so the handler call itself raises here.
        with pytest.raises(FileNotFoundError, match="ic"):
            drift_handler(argparse.Namespace(trading_day=FRIDAY, store=str(tmp_path / "store")))

        manifest = json.loads(store.get_bytes(manifest_key("drift", day)).decode("utf-8"))
        assert manifest["status"] == "failed"
        assert "ic" in manifest["reason"]
