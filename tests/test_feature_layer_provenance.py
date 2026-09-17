"""`check_feature_layer_provenance`: which point-in-time source actually
compiled the live feature layer (`alpha-engine-config-I10733`).

The gap this closes, measured 2026-09-17 against the live feature layer
(catalog `v553618c991dd`, in the v2 store): it
held 1,180 sessions and 92 of them (2025-07-09..2025-11-14) were compiled by
`v1-snapshots` while every neighbouring session used `edgar-filing-date`.
Each of the 92 carries 11 unmeasured columns and four null attractiveness
pillars. `data/{day}/coverage.json` recorded `point_in_time.source`
correctly for all of them from the moment they were written — nothing read
it, so a heal chunk that booted a release predating the EDGAR switch wrote a
shallow-source band into an EDGAR-source layer and no reading went red.

Style mirrors `tests/test_feature_layer_depth.py`: a `LocalStore` over
`tmp_path`, per this repo's test convention.
"""

from __future__ import annotations

import json

import pytest

from crucible.data.point_in_time import PRODUCTION_FUNDAMENTALS_SOURCE
from crucible.documents import UnreadableDocumentError
from crucible.features.depth import (
    FEATURES_PREFIX,
    PROVENANCE_SAMPLE_SIZE,
    check_feature_layer_completeness,
    check_feature_layer_provenance,
    sample_coverage_sentence,
    sample_sessions,
)
from crucible.keys import coverage_key
from crucible.store import LocalStore

VERSION = "vprovenance01"


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def _sessions(count: int) -> list[str]:
    """`count` distinct, chronologically sortable session labels."""
    return [f"2025-{1 + index // 28:02d}-{1 + index % 28:02d}" for index in range(count)]


def _build(store: LocalStore, days: list[str], sources: dict[str, str]) -> None:
    """A feature layer over `days`, each session's coverage record naming
    `sources[day]`. A day absent from `sources` gets NO coverage record —
    the unobserved case, which is a measured fact and not a read failure.
    """
    for day in days:
        store.put_bytes(f"{FEATURES_PREFIX}{VERSION}/{day}.parquet", b"parquet-bytes")
        if day in sources:
            store.put_bytes(
                coverage_key(day),
                json.dumps(
                    {
                        "schema_version": "coverage.v1",
                        "trading_day": day,
                        "point_in_time": {"source": sources[day], "snapshot_id": "x"},
                    }
                ).encode("utf-8"),
            )


class TestSingleSourcedLayer:
    def test_a_layer_compiled_wholly_by_the_production_source_is_green(self, store) -> None:
        days = _sessions(120)
        _build(store, days, dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE))

        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert reading.state == "GREEN"
        assert reading.expected_source == PRODUCTION_FUNDAMENTALS_SOURCE
        assert set(reading.sources) == {PRODUCTION_FUNDAMENTALS_SOURCE}
        assert reading.missing == ()
        assert reading.sessions_total == 120

    def test_the_reading_states_the_band_it_could_still_miss(self, store) -> None:
        """A sample reported as a sweep is the same false-green shape one
        layer up, so the sentence is asserted, not assumed."""
        days = _sessions(120)
        _build(store, days, dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE))

        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert "sampled evenly across the layer" in reading.detail
        assert "can sit between two samples unseen" in reading.detail
        assert f"{len(reading.sessions_read)} of 120 session(s)" in reading.detail


class TestMixedSourceLayer:
    def test_a_band_from_another_source_reds_and_names_itself(self, store) -> None:
        """The measured `-I10733` shape: a contiguous band compiled by the
        shallower source inside an otherwise EDGAR-sourced layer."""
        days = _sessions(120)
        sources = dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE)
        band = days[40:80]
        for day in band:
            sources[day] = "v1-snapshots"
        _build(store, days, sources)

        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert reading.state == "RED"
        assert "v1-snapshots" in reading.sources
        assert set(reading.sources["v1-snapshots"]) <= set(band)
        assert "v1-snapshots" in reading.detail
        assert reading.sources["v1-snapshots"][0] in reading.detail

    def test_a_layer_wholly_from_the_wrong_source_reds(self, store) -> None:
        """A uniformly shallow layer is internally CONSISTENT, so a predicate
        keyed only on 'more than one source' would read it green. This keys
        on the production source by name instead."""
        days = _sessions(60)
        _build(store, days, dict.fromkeys(days, "v1-snapshots"))

        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert reading.state == "RED"
        assert set(reading.sources) == {"v1-snapshots"}

    def test_a_band_shorter_than_the_stride_can_be_missed_and_the_reading_says_so(
        self, store
    ) -> None:
        """The honest limit of a sample, asserted rather than left implied."""
        days = _sessions(400)
        sources = dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE)
        sampled = set(sample_sessions(days, PROVENANCE_SAMPLE_SIZE))
        hidden = next(day for day in days if day not in sampled)
        sources[hidden] = "v1-snapshots"
        _build(store, days, sources)

        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert reading.state == "GREEN"
        assert "can sit between two samples unseen" in reading.detail


class TestUnobserved:
    def test_a_session_with_no_coverage_record_reds_as_unread(self, store) -> None:
        """Absent provenance is never folded into green: principle 7."""
        days = _sessions(60)
        sources = dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE)
        for day in days[10:40]:
            del sources[day]
        _build(store, days, sources)

        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert reading.state == "RED"
        assert reading.missing
        assert "carry no" in reading.detail

    def test_an_unbuilt_version_reds_rather_than_reading_empty_as_clean(self, store) -> None:
        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert reading.state == "RED"
        assert reading.sessions_total == 0
        assert "has never been built" in reading.detail


class TestContract:
    def test_the_expected_source_defaults_to_the_one_declaration(self, store) -> None:
        """Not a source-name literal in the detector — the single declaration
        beside the implementations, so a production switch moves one line."""
        days = _sessions(30)
        _build(store, days, dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE))

        default = check_feature_layer_provenance(store, live_version=VERSION)
        explicit = check_feature_layer_provenance(
            store, live_version=VERSION, expected_source=PRODUCTION_FUNDAMENTALS_SOURCE
        )

        assert default.expected_source == explicit.expected_source
        assert default.state == explicit.state == "GREEN"

    def test_an_access_failure_propagates_rather_than_reading_green(self, store) -> None:
        """An access problem is a statement about US, not about which source
        compiled the layer. It is raised so `crucible.board` renders
        UNMEASURABLE — never graded into a verdict."""
        days = _sessions(30)
        _build(store, days, dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE))

        class Broken(LocalStore):
            def get_bytes(self, key: str) -> bytes:
                raise PermissionError("denied")

        with pytest.raises(UnreadableDocumentError):
            check_feature_layer_provenance(Broken(store.root), live_version=VERSION)

    def test_a_corrupt_coverage_record_reds_rather_than_raising(self, store) -> None:
        """A truncated artifact is a fact about the STORE's contents, not
        about our access — the two outcomes the guarded reader keeps apart."""
        days = _sessions(60)
        _build(store, days, dict.fromkeys(days, PRODUCTION_FUNDAMENTALS_SOURCE))
        for day in days[10:40]:
            store.put_bytes(coverage_key(day), b'{"point_in_time": {"sour')

        reading = check_feature_layer_provenance(store, live_version=VERSION)

        assert reading.state == "RED"
        assert reading.unreadable
        assert "could not be parsed" in reading.detail

    def test_both_sampled_readings_state_their_coverage_identically(self, store) -> None:
        """Two readings over the same layer that describe their coverage
        differently is how one of them quietly stops being true."""
        days = _sessions(120)
        sampled = sample_sessions(days, PROVENANCE_SAMPLE_SIZE)

        sentence = sample_coverage_sentence(days, sampled)

        assert sentence.startswith(f"{len(sampled)} of {len(days)} session(s)")
        assert check_feature_layer_completeness.__module__ == (
            check_feature_layer_provenance.__module__
        )
