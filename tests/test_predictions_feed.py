"""The M champion's serving feed — producer, consumer, and the one schema
between them (`alpha-engine-config-I10129`).

Normative source: plan §3, §4.4, §4.12; `crucible/AGENTS.md` — *"the trader
reads one contract — `champions/{slot}/current.json` plus
`predictions/{trading_day}.json`"*.

M0 discipline: a versioned schema plus a producer/consumer contract test at
birth. The producer is `crucible.serving.publish_predictions_feed`; the
consumer is `crucible.serving.read_predictions_feed`, which is the reference
implementation the trader repo imports rather than re-deriving. This file is
the contract between them, and it is in the PUBLIC harness repo on purpose:
a second implementation of the trader must be able to consume the feed from
`predictions_feed.v1.json` and this reader alone.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from jsonschema import Draft202012Validator

from crucible.champion import (
    CHAMPION_SCHEMA_VERSION,
    ChampionPointer,
    ChampionUnusableError,
    read_champion_etag,
    write_champion,
)
from crucible.keys import arm_predictions_key, predictions_key
from crucible.models import PredictionsFeedDocument
from crucible.serving import (
    PREDICTIONS_FEED_SCHEMA_VERSION,
    PredictionsFeed,
    PredictionsFeedContractError,
    publish_predictions_feed,
    read_predictions_feed,
)
from crucible.slots.cycle import MissingArtifactError
from crucible.slots.inputs import ARM_PREDICTIONS_SCHEMA_VERSION
from crucible.store import LocalStore

DAY = "2026-08-28"
OTHER_DAY = "2026-08-27"
ARM = "m:ridge_21d:0123456789ab"

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crucible"
    / "schemas"
    / "predictions_feed.v1.json"
)


def _manifest(status: str = "ok") -> bytes:
    return json.dumps(
        {"status": status, "job": "experiment.run", "trading_day": DAY, "reason": ""}
    ).encode("utf-8")


def _pointer(arm_id: str = ARM, slot: str = "m") -> ChampionPointer:
    return ChampionPointer(
        schema_version=CHAMPION_SCHEMA_VERSION,
        slot=slot,
        arm_id=arm_id,
        as_of=DAY,
        decided_at="2026-08-29T02:00:00Z",
        run_id="01JG0000000000000000000000",
        code_sha="a" * 40,
        promotion_source="evidence",
        manifest_key=f"runs/promote/{DAY}/run.json",
        evidence={"status": "decided", "moved": True, "paired_dates": 40},
    )


def _arm_predictions(
    *, arm_id: str = ARM, trading_day: str = DAY, alpha: dict[str, float] | None = None
) -> bytes:
    return json.dumps(
        {
            "schema_version": ARM_PREDICTIONS_SCHEMA_VERSION,
            "arm_id": arm_id,
            "trading_day": trading_day,
            "feature_version": "features.v3",
            "predicted_alpha": alpha if alpha is not None else {"AAA": 0.031, "BBB": -0.012},
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def _seat_champion(store: LocalStore, pointer: ChampionPointer | None = None) -> ChampionPointer:
    pointer = pointer or _pointer()
    write_champion(store, pointer, expected=read_champion_etag(store, pointer.slot))
    store.put_bytes(pointer.manifest_key, _manifest())
    return pointer


class TestTheProducerRepublishesAndNeverRecomputes:
    def test_the_feed_lands_at_the_key_the_trader_contract_names(self, store) -> None:
        _seat_champion(store)
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions())
        assert publish_predictions_feed(store, trading_day=DAY) == f"predictions/{DAY}.json"

    def test_the_feed_carries_exactly_the_champions_numbers(self, store) -> None:
        """The republication property: the trader's numbers and the graded
        numbers are the SAME numbers, not two computations that agree."""
        _seat_champion(store)
        alpha = {"AAA": 0.031, "BBB": -0.012, "CCC": 0.004}
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions(alpha=alpha))
        publish_predictions_feed(store, trading_day=DAY)
        assert read_predictions_feed(store, DAY).predicted_alpha == alpha

    def test_the_feed_names_the_document_it_republished(self, store) -> None:
        """`source_key` is the hop `explain` walks from the trader's read back
        to the arm's own artifact. A feed that carried the numbers and not
        their provenance would be unreconcilable against the grade."""
        _seat_champion(store)
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions())
        publish_predictions_feed(store, trading_day=DAY)
        feed = read_predictions_feed(store, DAY)
        assert feed.source_key == arm_predictions_key(ARM, DAY)
        assert feed.champion == ARM
        assert feed.feature_version == "features.v3"

    def test_republishing_an_unchanged_cross_section_is_byte_identical(self, store) -> None:
        """A digest comparison between two days' feeds must mean what it looks
        like it means, so the writer sorts and indents deterministically."""
        _seat_champion(store)
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions())
        publish_predictions_feed(store, trading_day=DAY)
        first = store.get_bytes(predictions_key(DAY))
        publish_predictions_feed(store, trading_day=DAY)
        assert store.get_bytes(predictions_key(DAY)) == first


class TestNoChampionIsNotAFailure:
    def test_an_empty_store_owes_no_feed_and_writes_none(self, store) -> None:
        """The M slot has had no champion at any point in the production
        store's life. A producer that raised here would fail every promote
        and produce run taken before the slot's first promotion."""
        assert publish_predictions_feed(store, trading_day=DAY) is None
        assert not store.exists(predictions_key(DAY))


class TestAnUnusableChampionNeverGetsAFeed:
    def test_a_pointer_whose_producing_run_failed_is_refused(self, store) -> None:
        """The refusal is `read_champion`'s and is not re-implemented here —
        this test exists to prove the producer did not bypass it, which is
        the only way a fail-open could enter on this path."""
        pointer = _pointer()
        write_champion(store, pointer, expected=read_champion_etag(store, "m"))
        store.put_bytes(pointer.manifest_key, _manifest("failed"))
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions())
        with pytest.raises(ChampionUnusableError):
            publish_predictions_feed(store, trading_day=DAY)
        assert not store.exists(predictions_key(DAY))

    def test_a_pointer_whose_manifest_is_absent_is_refused(self, store) -> None:
        pointer = _pointer()
        write_champion(store, pointer, expected=read_champion_etag(store, "m"))
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions())
        with pytest.raises(ChampionUnusableError):
            publish_predictions_feed(store, trading_day=DAY)


class TestAChampionThatProducedNothingIsAFailureNotAnEmptyFeed:
    def test_a_missing_cross_section_raises_the_same_error_run_produce_raises(self, store) -> None:
        _seat_champion(store)
        with pytest.raises(MissingArtifactError, match="no prediction cross-section"):
            publish_predictions_feed(store, trading_day=DAY)
        assert not store.exists(predictions_key(DAY))

    def test_an_empty_cross_section_is_refused(self, store) -> None:
        _seat_champion(store)
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions(alpha={}))
        with pytest.raises(PredictionsFeedContractError, match="predicted_alpha"):
            publish_predictions_feed(store, trading_day=DAY)


class TestThePointInTimeGuardOnTheServingPath:
    def test_a_cross_section_from_another_session_is_refused(self, store) -> None:
        """Look-ahead if it is later, stale if it is earlier. Neither is a
        degraded feed on the money path."""
        _seat_champion(store)
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions(trading_day=OTHER_DAY))
        with pytest.raises(PredictionsFeedContractError, match="trading_day"):
            publish_predictions_feed(store, trading_day=DAY)

    def test_a_misfiled_arms_cross_section_is_refused(self, store) -> None:
        _seat_champion(store)
        store.put_bytes(
            arm_predictions_key(ARM, DAY), _arm_predictions(arm_id="m:someone_else:ffff")
        )
        with pytest.raises(PredictionsFeedContractError, match="arm_id"):
            publish_predictions_feed(store, trading_day=DAY)

    def test_an_unknown_source_schema_version_is_refused_not_guessed(self, store) -> None:
        _seat_champion(store)
        payload = json.loads(_arm_predictions())
        payload["schema_version"] = "arm_predictions.v99"
        store.put_bytes(arm_predictions_key(ARM, DAY), json.dumps(payload).encode("utf-8"))
        with pytest.raises(PredictionsFeedContractError, match="schema_version"):
            publish_predictions_feed(store, trading_day=DAY)


class TestTheConsumerHalf:
    def test_an_absent_feed_raises_key_error_not_a_contract_error(self, store) -> None:
        """A trader must distinguish "the harness has not published yet"
        (wait, page on the absence deadline) from "today's feed is corrupt"
        (page now, trade nothing)."""
        with pytest.raises(KeyError):
            read_predictions_feed(store, DAY)

    def test_a_feed_misfiled_under_another_days_key_is_refused(self, store) -> None:
        feed = PredictionsFeed(
            slot="m",
            trading_day=OTHER_DAY,
            champion=ARM,
            feature_version="features.v3",
            source_key=arm_predictions_key(ARM, OTHER_DAY),
            predicted_alpha={"AAA": 0.1},
        )
        store.put_bytes(predictions_key(DAY), feed.to_bytes())
        with pytest.raises(PredictionsFeedContractError, match="trading_day"):
            read_predictions_feed(store, DAY)

    def test_unreadable_bytes_are_refused_by_name(self, store) -> None:
        store.put_bytes(predictions_key(DAY), b"not json at all")
        with pytest.raises(PredictionsFeedContractError):
            read_predictions_feed(store, DAY)

    def test_an_unknown_top_level_field_is_refused(self, store) -> None:
        payload = PredictionsFeed(
            slot="m",
            trading_day=DAY,
            champion=ARM,
            feature_version="features.v3",
            source_key=arm_predictions_key(ARM, DAY),
            predicted_alpha={"AAA": 0.1},
        ).to_dict()
        payload["sizing_hint"] = "max"
        store.put_bytes(predictions_key(DAY), json.dumps(payload).encode("utf-8"))
        with pytest.raises(PredictionsFeedContractError, match="sizing_hint"):
            read_predictions_feed(store, DAY)


class TestTheCommittedSchemaIsGeneratedFromTheModel:
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(PredictionsFeedDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        assert SCHEMA_PATH.read_text(encoding="utf-8") == generated, (
            f"{SCHEMA_PATH.name} has drifted from PredictionsFeedDocument. The schema is "
            "GENERATED, never hand-edited: regenerate it in the same commit as the model."
        )

    def test_a_real_produced_feed_validates_against_the_committed_schema(self, store) -> None:
        """The PRODUCER side of the contract, exercised through the real
        writer rather than a hand-built document."""
        _seat_champion(store)
        store.put_bytes(arm_predictions_key(ARM, DAY), _arm_predictions())
        publish_predictions_feed(store, trading_day=DAY)
        Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
            json.loads(store.get_bytes(predictions_key(DAY)))
        )

    def test_a_consumer_with_no_python_import_gets_the_non_empty_rule(self) -> None:
        """`predicted_alpha` is non-empty in the PUBLISHED schema, not only in
        the model: the trader may be a second implementation that never
        imports this package."""
        from jsonschema import ValidationError as JsonSchemaValidationError

        payload = PredictionsFeed(
            slot="m",
            trading_day=DAY,
            champion=ARM,
            feature_version="features.v3",
            source_key=arm_predictions_key(ARM, DAY),
            predicted_alpha={},
        ).to_dict()
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
                payload
            )

    def test_the_feed_is_its_own_schema_and_never_a_feed_v1_variant(self) -> None:
        """`alpha-engine-config-I10129` deliverable 3. R's and U's `feed.v1`
        carries `members`, a selection; M's payload is a cross-section.
        One `schema_version` string meaning two document shapes would push
        the discrimination onto every consumer AFTER it claimed to have
        validated the document."""
        assert PREDICTIONS_FEED_SCHEMA_VERSION == "predictions_feed.v1"
        assert "members" not in PredictionsFeedDocument.model_fields


class TestTheFeedIsOnTheMoneyPath:
    """Plan §9.5: the trader sizes orders from this document, so it is a
    money-path artifact and joins the hash chain `crucible-PR240` builds
    (`alpha-engine-config-I10414`).

    Guarded on PRESENCE so the two PRs can land in either order — and
    guarded with an ASSERTION, never a skip: a skipped test measures nothing
    and reads as a pass on the summary line (plan §11.1). The assertion is
    true in both worlds and arms itself, with no edit, the moment
    `crucible.manifest` declares the predicate set.
    """

    def test_the_feed_key_is_a_money_path_artifact_once_the_chain_lands(self) -> None:
        import crucible.manifest as manifest_module

        predicates = getattr(manifest_module, "MONEY_PATH_PREDICATES", ())
        key = predictions_key(DAY)
        assert not predicates or any(predicate(key) for predicate in predicates), (
            f"{key} is what the trader sizes orders from and is absent from "
            "crucible.manifest.MONEY_PATH_PREDICATES. A money-path artifact outside "
            "the chain is a restatement nobody can detect (plan §9.5)."
        )


class TestTheProducerIsReachedFromTheMProducePath:
    """The defect this feed closes is a CONTRACT HALF WITH NO PRODUCER.
    Shipping a producer nothing calls would move the same defect one function
    along, so this is the guard that makes the wiring un-skippable rather than
    a note in a PR body.

    Measured 2026-09-13 on `crucible` `main`: `crucible.slots.model` exposes
    no `produce` entry point at all, so `dispatchable_slots()` returns
    `{u, r}` and the M produce path does not exist yet — it is
    `crucible-PR149`, held as a draft behind phase 2. The moment that PR
    lands and M becomes dispatchable, this fails until the feed publication
    is wired into the produce path, in exactly the module that PR creates it
    in. An assertion rather than a conditional skip, for the reason above.
    """

    def test_m_is_not_dispatchable_without_the_feed_being_published(self) -> None:
        import inspect

        import crucible.slots.model as model_module
        from crucible.slots import dispatchable_slots

        dispatchable = dispatchable_slots()
        source = inspect.getsource(model_module)
        assert "m" not in dispatchable or "publish_predictions_feed" in source, (
            "the M slot is dispatchable and its produce path never publishes "
            "predictions/{trading_day}.json. The trader contract is "
            "`champions/m/current.json` PLUS that feed (crucible/AGENTS.md); a "
            "dispatchable M slot with no feed writer is a valid champion pointer the "
            "trader cannot act on, with nothing saying so."
        )
