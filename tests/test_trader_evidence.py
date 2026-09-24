"""The trader's consumer evidence — the one artifact the phase-4 gate reads.

Normative source: plan §3 (the harness and the trader are separate systems
coupled by contract documents, and the harness may not reach into the trader),
§9.5 ("champion-consumption contract test"), §4.12 (trading days);
`alpha-engine-config-I10648`.

**Why this file is in the PUBLIC harness repo.** The PRODUCER is
`nousergon/crucible-trader`, which is private. The CONSUMER is
`crucible.gate._clause_trader_week_on_v2_champion`, here. A contract whose
schema lived only with the producer would be a contract the consumer cannot
check and a second implementation cannot read; so the schema, the key and the
consumer's reading of both are here, and the trader imports them rather than
re-deriving them.

Before `-I10648` the clause read UNMEASURABLE for every store, which meant
phase 4's own headline deliverable could never be false. A clause that cannot
produce a negative result grades nothing (plan §10.1) — so the negative cases
below are the point of this file, not decoration on it.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

import crucible.gate as gate_module
from crucible.keys import TRADER_EVIDENCE_KEY, TRADER_PIN_KEY
from crucible.models import SERVED_SESSION_MODES, TraderEvidenceDocument

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "crucible" / "schemas" / "trader_evidence.v2.json"
)

SCHEMA_VERSION = "trader_evidence.v2"
CHAMPION = "m:ridge_21d:0123456789ab"
WEEK = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-12"]


def _document(**overrides: object) -> dict:
    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "slot": "m",
        "champion": CHAMPION,
        "trading_days": len(WEEK),
        "days_served": list(WEEK),
        "calendar_date": "2026-09-13",
    }
    payload.update(overrides)
    if "session_modes" not in overrides:
        payload["session_modes"] = {day: "shadow" for day in payload["days_served"]}
    return payload


class TestTheKeyIsOneStringBothSidesName:
    def test_the_gate_resolves_the_declared_key_rather_than_a_literal(self) -> None:
        assert gate_module.TRADER_EVIDENCE_KEY == TRADER_EVIDENCE_KEY

    def test_it_sits_under_the_trader_prefix_beside_the_release_pin(self) -> None:
        """One top-level prefix for everything about the trader, and a fixed
        key rather than a `def trader_evidence_key()` helper. The helper form
        would enter the population `nous-ergon-ops`'s store-prefix lockstep
        guard derives the spot-box runtime role's PutObject grants from,
        granting that box a write on the one prefix it must not have."""
        assert TRADER_EVIDENCE_KEY.startswith("trader/")
        assert TRADER_PIN_KEY.startswith("trader/")
        assert TRADER_EVIDENCE_KEY != TRADER_PIN_KEY

    def test_the_key_is_rolling_not_dated(self) -> None:
        """The clause asks a cumulative question. A dated key would make the
        gate reconstruct the count by listing — a second implementation of a
        number the trader already knows."""
        assert "{" not in TRADER_EVIDENCE_KEY


class TestTheDocumentAcceptsAnHonestRecord:
    def test_a_full_week_validates(self) -> None:
        document = TraderEvidenceDocument.model_validate(_document())

        assert document.trading_days == 5
        assert document.champion == CHAMPION

    def test_a_trader_that_has_served_nothing_yet_is_representable(self) -> None:
        """Zero is a true statement and a different one from "no document".
        A trader that could not file an empty record would have to choose
        between silence and a lie on its first day."""
        document = TraderEvidenceDocument.model_validate(_document(trading_days=0, days_served=[]))

        assert document.trading_days == 0


class TestTheCountCannotBeForged:
    def test_a_count_larger_than_its_days_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="days_served carries 4 day"):
            TraderEvidenceDocument.model_validate(_document(trading_days=5, days_served=WEEK[:4]))

    def test_a_count_smaller_than_its_days_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must be supported by the days"):
            TraderEvidenceDocument.model_validate(_document(trading_days=3))

    def test_a_repeated_day_is_refused(self) -> None:
        """A session served twice is one day, and counting it twice is how a
        four-day week reads as five."""
        repeated = [*WEEK[:4], WEEK[3]]
        with pytest.raises(ValidationError, match="repeats a trading day"):
            TraderEvidenceDocument.model_validate(_document(trading_days=5, days_served=repeated))

    def test_an_out_of_order_list_is_refused(self) -> None:
        shuffled = [WEEK[1], WEEK[0], *WEEK[2:]]
        with pytest.raises(ValidationError, match="not strictly increasing"):
            TraderEvidenceDocument.model_validate(_document(days_served=shuffled))

    def test_a_day_that_is_shaped_like_a_date_but_is_not_one_is_refused(self) -> None:
        """A shape-only check would accept 2026-02-31."""
        with pytest.raises(ValidationError, match="not a calendar date"):
            TraderEvidenceDocument.model_validate(
                _document(trading_days=1, days_served=["2026-02-31"])
            )


class TestEveryServedDayRecordsItsSessionMode:
    """`alpha-engine-config-I11545`, Brian's ruling 2 of 2026-09-24: a shadow
    session counts as a served day. The mode is recorded per day so the count
    can treat the two alike while the reading still tells them apart."""

    def test_both_modes_are_served_modes_by_the_ruling(self) -> None:
        assert SERVED_SESSION_MODES == ("shadow", "live")

    def test_a_mixed_week_validates_and_names_its_shadow_days(self) -> None:
        modes = {day: "shadow" for day in WEEK[:3]} | {day: "live" for day in WEEK[3:]}
        document = TraderEvidenceDocument.model_validate(_document(session_modes=modes))

        assert document.shadow_days() == WEEK[:3]

    def test_a_served_day_with_no_mode_is_refused(self) -> None:
        modes = {day: "shadow" for day in WEEK[:4]}
        with pytest.raises(ValidationError, match="no mode for"):
            TraderEvidenceDocument.model_validate(_document(session_modes=modes))

    def test_a_mode_for_a_day_that_was_not_served_is_refused(self) -> None:
        modes = {day: "shadow" for day in [*WEEK, "2026-09-15"]}
        with pytest.raises(ValidationError, match="which was not served"):
            TraderEvidenceDocument.model_validate(_document(session_modes=modes))

    def test_an_unknown_mode_is_refused(self) -> None:
        """A third mode would be counted by nobody's ruling."""
        modes = {day: "shadow" for day in WEEK} | {WEEK[0]: "dry_run"}
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(_document(session_modes=modes))

    def test_the_session_modes_field_is_required(self) -> None:
        document = _document()
        del document["session_modes"]
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(document)


class TestTheDocumentRefusesWhatItCannotActOn:
    def test_an_unknown_top_level_field_is_refused(self) -> None:
        """A field this reader does not understand is a field the producer
        expected it to act on."""
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(_document(pnl_usd=1234.0))

    def test_another_slot_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(_document(slot="r"))

    def test_an_unknown_schema_version_is_refused_not_guessed(self) -> None:
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(_document(schema_version="trader_evidence.v3"))

    def test_a_v1_document_is_refused_not_read_as_modeless(self) -> None:
        """v1 carried no session mode. Reading one as if every day were live
        (or shadow) would be the guess this field exists to remove."""
        v1 = _document(schema_version="trader_evidence.v1")
        del v1["session_modes"]
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(v1)

    def test_an_empty_champion_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(_document(champion=""))

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            TraderEvidenceDocument.model_validate(_document(trading_days=-1, days_served=[]))


class TestTheCommittedSchemaIsTheContract:
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(TraderEvidenceDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        assert SCHEMA_PATH.read_text(encoding="utf-8") == generated, (
            f"{SCHEMA_PATH.name} has drifted from TraderEvidenceDocument. The schema is "
            "generated, never hand-edited: regenerate it rather than editing the file."
        )

    def test_a_real_document_validates_against_the_committed_schema(self) -> None:
        Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
            _document()
        )

    def test_a_consumer_with_no_python_import_is_told_the_cross_field_rule(self) -> None:
        """JSON Schema cannot state `trading_days == len(days_served)`, so the
        published file says so in `$comment`. Without that, a second
        implementation reading the schema alone would be free to file a count
        its own day list does not support — the exact forgery this document
        exists to prevent."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        assert "trading_days MUST equal" in schema["$comment"]
        assert "strictly increasing" in schema["$comment"]
        assert "session_modes MUST key exactly" in schema["$comment"]
        assert schema["additionalProperties"] is False

    def test_the_schema_declares_its_own_id_and_version(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        assert schema["$id"].endswith("trader_evidence.v2.json")
        assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
