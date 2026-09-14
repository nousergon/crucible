"""The money-path hash chain, and `explain`'s verification of it.

`alpha-engine-config-I10414`; plan §9.5 (Tamper-evident money-path
artifacts) and §10.8 (`explain`). Written before the implementation (fleet
TDD rule) and seen failing.

The v1 NAV series was manually restated over four sessions after sitting
built-but-never-run for five days. A restatement is legitimate; an unnoticed
one is not, and S3 versioning records only that *a change happened*. The
chain is what makes "the history now present is the history that was
written" machine-checkable — so every test here asserts that something is
DETECTED, not that a happy path serialises. A chain that only ever verifies
intact histories has not been shown to constrain anything.

**The one thing that must never regress** is the digest's provenance. The
incident is `nous-ergon-ops-I1145`: an S3 ETag written into a field named
`sha256` agreed with every local test — the local backend's token happens to
be a content hash — and was wrong on S3 the moment an object went multipart.
`TestTheDigestIsContentNotAVersionToken` fails if any chain code path so
much as calls `Store.etag`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.explain import (
    ChainVerification,
    MoneyPathChainError,
    explain,
    render,
    verify_money_path_chain,
)
from crucible.keys import (
    TRADER_EVIDENCE_KEY,
    champion_key,
    execution_shortfall_key,
    holdout_unseal_key,
    manifest_key,
    predictions_key,
    shadow_books_key,
    strategy_holdout_key,
    trader_broker_statement_key,
    trader_reconciliation_key,
)
from crucible.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    money_path_writes,
    on_money_path,
    validate,
    write_manifest,
)
from crucible.models import MoneyPathLink
from crucible.store import LocalStore, sha256_hex

TRADING_DAY = "2026-08-28"
CALENDAR_DAY = "2026-08-31"


def _manifest(
    run_id: str,
    *,
    job: str = "promote",
    outputs: list[str] | None = None,
    finished: str = "2026-08-31T13:04:11Z",
    trading_day: str = TRADING_DAY,
) -> dict[str, Any]:
    """A conformant v2 manifest at the floor, plus whatever outputs a test needs."""
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "job": job,
        "run_mode": "live",
        "trading_day": trading_day,
        "calendar_date": CALENDAR_DAY,
        "status": "ok",
        "reason": "",
        "started": "2026-08-31T13:00:00Z",
        "finished": finished,
        "code_sha": "a" * 40,
        "release_sha": "b" * 40,
        "seed": 7,
        "inputs": [],
        "outputs": [
            {"key": k, "sha256": sha256_hex(k.encode()), "schema_version": "v1"}
            for k in (outputs or [])
        ],
        "rows_in": 0,
        "rows_out": 0,
        "rows_rejected": [],
        "cost_usd": 0.0,
        "llm_calls": [],
        "resource": {
            "instance_type": "local",
            "spot": False,
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 1.0,
            "disk_free_mb": 1.0,
        },
        "metrics": [],
        "attempts": [{"n": 1, "reason": "initial"}],
    }


def _write(store: LocalStore, manifest: dict[str, Any]) -> str:
    """Write ``manifest`` through the single writer; return its manifest key."""
    key = manifest_key(
        manifest["job"], manifest["trading_day"], discriminator=manifest.get("discriminator")
    )
    write_manifest(store, key, manifest)
    return key


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "store")


# ── membership ─────────────────────────────────────────────────────────────


class TestWhatIsOnTheMoneyPath:
    """`MONEY_PATH_PREDICATES` is a POSITIVE membership rule, pinned here so
    widening or narrowing it is a deliberate edit with a test diff, not a
    silent change to what the chain makes a claim about."""

    @pytest.mark.parametrize(
        "key",
        [
            champion_key("u"),
            champion_key("r"),
            champion_key("m"),
            champion_key("s"),
            predictions_key(TRADING_DAY),
            strategy_holdout_key(),
            holdout_unseal_key(TRADING_DAY, "I10414"),
            # Phase 4, `alpha-engine-config-I10651`: the broker reconciliation
            # result (plan §9.5 by name) and the per-session fills record.
            trader_reconciliation_key(TRADING_DAY),
            execution_shortfall_key(TRADING_DAY),
        ],
    )
    def test_the_money_path_members(self, key: str) -> None:
        assert on_money_path(key)

    @pytest.mark.parametrize(
        "key",
        [
            # `alpha-engine-config-I9822`: what ONE arm predicted is a
            # different artifact answering a different question. It shared
            # `predictions/` until 2026-09-11 and a prefix literal here would
            # have chained every arm's output along with the serving feed.
            "arm_predictions/m__baseline__abc/2026-08-28.json",
            manifest_key("promote", TRADING_DAY),
            "report/2026-08-28/attribution.json",
            "board/current.json",
            "releases/current",
            # SIMULATED paper books: evidence beside a promotion, never money
            # (`alpha-engine-config-I10653` deliverable 5).
            shadow_books_key(TRADING_DAY),
            # The reconciliation's INPUT, content-hashed into that run's
            # manifest already; the result is what §9.5 chains.
            trader_broker_statement_key(TRADING_DAY),
            TRADER_EVIDENCE_KEY,
            # Not a day: the predicate rebuilds the key from the basename and
            # must refuse rather than let the key helper's date check raise.
            "trader/reconciliation/notaday.json",
            "trader/reconciliation/2026-08-28.json.bak",
        ],
    )
    def test_what_is_not_on_the_money_path(self, key: str) -> None:
        assert not on_money_path(key)

    def test_a_rogue_serving_feed_key_is_still_money_path(self) -> None:
        """`predictions/notaday.json` matches. Deliberately: whether the date
        component is a real NYSE session is `Store.assert_keys_bind_to_trading_days`'
        question, and a membership rule that EXEMPTED a malformed key would
        exempt exactly the object an attacker would write."""
        assert on_money_path("predictions/notaday.json")

    def test_reading_a_money_path_artifact_does_not_join_the_chain(self, store) -> None:
        """Outputs only. A run that READ the champion pointer changed nothing
        about where money goes; a chain that grew a record per reader would
        make its own length meaningless."""
        reader = _manifest("01JG000000000000000000000R")
        reader["inputs"] = [
            {
                "key": champion_key("m"),
                "sha256": sha256_hex(b"x"),
                "schema_version": "champion_pointer.v1",
            }
        ]
        assert money_path_writes(reader) == ()
        _write(store, reader)
        assert verify_money_path_chain(store).records == ()


# ── the link the writer produces ───────────────────────────────────────────


class TestTheWriterEmitsTheLink:
    def test_a_non_money_path_run_carries_no_link(self, store) -> None:
        """Absent, not null. `not on the money path` and `link dropped` must
        not be the same document."""
        written = write_manifest(
            store,
            manifest_key("report", TRADING_DAY),
            _manifest("01JG00000000000000000000RP", job="report"),
        )
        assert "money_path_link" not in written

    def test_the_first_money_path_run_is_the_genesis(self, store) -> None:
        written = write_manifest(
            store,
            manifest_key("promote", TRADING_DAY),
            _manifest("01JG0000000000000000000001", outputs=[champion_key("m")]),
        )
        link = written["money_path_link"]
        assert link["index"] == 0
        assert link["prev_sha256"] is None
        assert link["prev_run_id"] is None
        assert link["money_path_writes"] == [champion_key("m")]

    def test_the_second_run_names_the_first_by_digest_of_its_stored_bytes(self, store) -> None:
        first_key = _write(
            store,
            _manifest(
                "01JG0000000000000000000001",
                outputs=[champion_key("m")],
                finished="2026-08-31T13:04:11Z",
            ),
        )
        second = write_manifest(
            store,
            manifest_key("holdout", TRADING_DAY),
            _manifest(
                "01JG0000000000000000000002",
                job="holdout",
                outputs=[strategy_holdout_key()],
                finished="2026-08-31T14:04:11Z",
            ),
        )
        link = second["money_path_link"]
        assert link["index"] == 1
        assert link["prev_run_id"] == "01JG0000000000000000000001"
        # The digest is over exactly what the store holds, byte for byte.
        assert link["prev_sha256"] == sha256_hex(store.get_bytes(first_key))

    def test_the_link_is_inside_the_bytes_that_were_validated(self, store) -> None:
        """A link attached after validation would be an unvalidated field on
        the one document the whole system's trust rests on."""
        key = _write(store, _manifest("01JG0000000000000000000001", outputs=[champion_key("m")]))
        stored = json.loads(store.get_bytes(key))
        assert "money_path_link" in stored
        validate(stored)


# ── the model's own rules ──────────────────────────────────────────────────


class TestTheLinkModelRefusesHalfALink:
    def test_a_genesis_naming_a_predecessor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="genesis"):
            MoneyPathLink(
                index=0,
                prev_sha256="c" * 64,
                prev_run_id="01JG0000000000000000000001",
                money_path_writes=[champion_key("m")],
            )

    def test_a_linkless_record_after_the_genesis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unverifiable hop"):
            MoneyPathLink(
                index=3, prev_sha256=None, prev_run_id=None, money_path_writes=[champion_key("m")]
            )

    def test_a_digest_without_a_run_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="half a link"):
            MoneyPathLink(
                index=1,
                prev_sha256="c" * 64,
                prev_run_id=None,
                money_path_writes=[champion_key("m")],
            )

    def test_an_empty_writes_list_is_refused(self) -> None:
        with pytest.raises(ValueError):
            MoneyPathLink(index=0, prev_sha256=None, prev_run_id=None, money_path_writes=[])

    def test_a_link_attesting_to_a_key_the_run_did_not_write_is_refused(self) -> None:
        manifest = _manifest("01JG0000000000000000000001", outputs=[champion_key("m")])
        manifest["money_path_link"] = {
            "index": 0,
            "prev_sha256": None,
            "prev_run_id": None,
            "money_path_writes": [champion_key("m"), champion_key("s")],
        }
        with pytest.raises(
            ManifestValidationError, match="does not list among its outputs|outputs"
        ):
            validate(manifest)

    def test_the_published_schema_alone_refuses_a_linkless_record(self) -> None:
        """Plain `jsonschema` against the committed file, no model import —
        the case the published schema is FOR (PR123 review finding 3's
        shape)."""
        import pathlib

        from jsonschema import Draft202012Validator, ValidationError

        schema = json.loads(
            (
                pathlib.Path(__file__).resolve().parents[1]
                / "crucible"
                / "schemas"
                / "run_manifest.v2.json"
            ).read_text(encoding="utf-8")
        )
        doc = _manifest("01JG0000000000000000000001", outputs=[champion_key("m")])
        doc["money_path_link"] = {
            "index": 4,
            "prev_sha256": None,
            "prev_run_id": None,
            "money_path_writes": [champion_key("m")],
        }
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(doc)


# ── verification: the contract test the issue names ────────────────────────


class TestTheChainDetectsTamperingAndAcceptsHonestAppends:
    """Issue `alpha-engine-config-I10414`, deliverable 4, verbatim: *a mutated
    historical record is detected; an appended record with a correct
    `prev_digest` is accepted*."""

    def _three_record_chain(self, store: LocalStore) -> list[str]:
        return [
            _write(
                store,
                _manifest(
                    "01JG0000000000000000000001",
                    outputs=[champion_key("m")],
                    finished="2026-08-31T13:00:11Z",
                ),
            ),
            _write(
                store,
                _manifest(
                    "01JG0000000000000000000002",
                    job="holdout",
                    outputs=[strategy_holdout_key()],
                    finished="2026-08-31T14:00:11Z",
                ),
            ),
            _write(
                store,
                _manifest(
                    "01JG0000000000000000000003",
                    job="experiment.run",
                    outputs=[predictions_key(TRADING_DAY)],
                    finished="2026-08-31T15:00:11Z",
                ),
            ),
        ]

    def test_an_honest_chain_verifies(self, store) -> None:
        self._three_record_chain(store)
        verdict = verify_money_path_chain(store)
        assert verdict.status == "ok", verdict.reason
        assert [r.index for r in verdict.records] == [0, 1, 2]
        assert all(r.intact for r in verdict.records)

    def test_an_appended_record_with_a_correct_prev_digest_is_accepted(self, store) -> None:
        self._three_record_chain(store)
        _write(
            store,
            _manifest(
                "01JG0000000000000000000004",
                job="promote",
                trading_day="2026-08-27",
                outputs=[champion_key("s")],
                finished="2026-08-31T16:00:11Z",
            ),
        )
        verdict = verify_money_path_chain(store)
        assert verdict.status == "ok", verdict.reason
        assert len(verdict.records) == 4

    def test_a_mutated_historical_record_is_detected(self, store) -> None:
        keys = self._three_record_chain(store)
        # The restatement: record 1's cost is edited in place, exactly the
        # shape of the v1 NAV series being manually restated. The record is
        # still a perfectly valid v2 manifest — that is the point.
        tampered = json.loads(store.get_bytes(keys[1]))
        tampered["cost_usd"] = 41.99
        validate(tampered)
        store.put_bytes(keys[1], json.dumps(tampered, indent=2, sort_keys=True).encode("utf-8"))

        verdict = verify_money_path_chain(store)
        assert verdict.status == "failed"
        assert "CHAIN BROKEN at record 2" in verdict.reason
        # Both digests named, per the issue: "with the record index and both
        # digests named". A bare "chain is broken" is not actionable.
        broken = [r for r in verdict.records if not r.intact]
        assert len(broken) == 1
        assert broken[0].prev_sha256 in verdict.reason
        assert broken[0].actual_prev_sha256 in verdict.reason
        assert broken[0].prev_sha256 != broken[0].actual_prev_sha256

    def test_a_removed_link_is_detected(self, store) -> None:
        """A record leaves a hash chain by having its link removed. If that
        read as "not on the chain" the deletion would grade green."""
        keys = self._three_record_chain(store)
        stripped = json.loads(store.get_bytes(keys[2]))
        del stripped["money_path_link"]
        validate(stripped)
        store.put_bytes(keys[2], json.dumps(stripped, indent=2, sort_keys=True).encode("utf-8"))

        verdict = verify_money_path_chain(store)
        assert verdict.status == "failed"
        assert keys[2] in verdict.unlinked
        assert "carries no `money_path_link`" in verdict.reason

    def test_a_deleted_middle_record_is_detected_as_an_index_gap(self, store) -> None:
        keys = self._three_record_chain(store)
        (store.root / keys[1]).unlink()
        verdict = verify_money_path_chain(store)
        assert verdict.status == "failed"
        assert "declares index 2" in verdict.reason

    def test_a_record_whose_run_id_pointer_disagrees_with_its_digest_is_detected(
        self, store
    ) -> None:
        keys = self._three_record_chain(store)
        doc = json.loads(store.get_bytes(keys[2]))
        doc["money_path_link"]["prev_run_id"] = "01JG000000000000000000000Z"
        validate(doc)
        store.put_bytes(keys[2], json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))
        verdict = verify_money_path_chain(store)
        assert verdict.status == "failed"
        assert "two different histories" in verdict.reason

    def test_a_store_with_money_path_manifests_and_no_chain_at_all_fails(self, store) -> None:
        """Not a quiet pass. "every link was removed" and "these runs predate
        the chain" are indistinguishable from inside the store, and grading
        the pair `ok` would make the first one invisible."""
        manifest = _manifest("01JG0000000000000000000001", outputs=[champion_key("m")])
        store.put_bytes(
            manifest_key("promote", TRADING_DAY),
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        verdict = verify_money_path_chain(store)
        assert verdict.status == "failed"
        assert "NOT ONE carries" in verdict.reason

    def test_an_empty_store_is_ok_and_says_the_chain_is_empty(self, store) -> None:
        verdict = verify_money_path_chain(store)
        assert verdict.status == "ok"
        assert "empty" in verdict.reason
        assert verdict.records == ()

    def test_two_statuses_only(self, store) -> None:
        """Plan §4.2 / `AGENTS.md` rule 2. No `degraded`, no `unverified` —
        the state a tampered history would otherwise settle into."""
        self._three_record_chain(store)
        assert verify_money_path_chain(store).status in ("ok", "failed")


# ── the digest's provenance: the I1145 guard ───────────────────────────────


class TestTheDigestIsContentNotAVersionToken:
    """`nous-ergon-ops-I1145`. An S3 ETag in a field named `sha256` agreed
    with every local test — `LocalStore.etag` returns a content hash — and was
    wrong on S3 the moment an object went multipart. `Store.etag`'s own
    docstring calls itself "an opaque version token"; the local agreement is
    a coincidence of the backend, not a property of the field.

    So this does not test that the value is right on LocalStore, which proves
    nothing. It tests that no chain code path CALLS `etag` at all.
    """

    def test_no_chain_code_path_calls_store_etag(self, store, monkeypatch) -> None:
        def refuse(self, key: str) -> str:
            raise AssertionError(
                f"the money-path chain called Store.etag({key!r}). The digest is sha256 "
                "over the stored bytes; a backend version token here is nous-ergon-ops-I1145 "
                "reintroduced, and it would agree with this very test suite while being "
                "wrong on a multipart S3 object."
            )

        monkeypatch.setattr(LocalStore, "etag", refuse)
        first = _write(store, _manifest("01JG0000000000000000000001", outputs=[champion_key("m")]))
        write_manifest(
            store,
            manifest_key("holdout", TRADING_DAY),
            _manifest(
                "01JG0000000000000000000002",
                job="holdout",
                outputs=[strategy_holdout_key()],
                finished="2026-08-31T14:04:11Z",
            ),
        )
        assert verify_money_path_chain(store).status == "ok"
        assert first  # the write happened without an etag read

    def test_the_stored_digest_is_the_sha256_of_the_stored_bytes(self, store) -> None:
        first = _write(store, _manifest("01JG0000000000000000000001", outputs=[champion_key("m")]))
        second = write_manifest(
            store,
            manifest_key("holdout", TRADING_DAY),
            _manifest(
                "01JG0000000000000000000002",
                job="holdout",
                outputs=[strategy_holdout_key()],
                finished="2026-08-31T14:04:11Z",
            ),
        )
        import hashlib

        expected = hashlib.sha256((store.root / first).read_bytes()).hexdigest()
        assert second["money_path_link"]["prev_sha256"] == expected


# ── explain integration ────────────────────────────────────────────────────


class TestExplainVerifiesTheChain:
    def _chain(self, store: LocalStore) -> list[str]:
        return [
            _write(
                store,
                _manifest(
                    "01JG0000000000000000000001",
                    outputs=[champion_key("m")],
                    finished="2026-08-31T13:00:11Z",
                ),
            ),
            _write(
                store,
                _manifest(
                    "01JG0000000000000000000002",
                    job="experiment.run",
                    outputs=[predictions_key(TRADING_DAY)],
                    finished="2026-08-31T14:00:11Z",
                ),
            ),
        ]

    def test_a_walk_that_crosses_the_money_path_carries_the_verdict(self, store) -> None:
        self._chain(store)
        node = explain(store, predictions_key(TRADING_DAY))
        assert isinstance(node.chain, ChainVerification)
        assert node.chain.status == "ok"
        assert node.to_dict()["money_path_chain"]["status"] == "ok"

    def test_a_walk_that_does_not_cross_the_money_path_carries_no_verdict(self, store) -> None:
        """`None`, not `ok`. "this walk touched no money-path artifact" and
        "the chain verified" are different answers, and rendering the second
        for the first would be a green light nobody earned."""
        self._chain(store)
        _write(
            store,
            _manifest(
                "01JG000000000000000000000B",
                job="report",
                outputs=["report/2026-08-28/attribution.json"],
                finished="2026-08-31T15:00:11Z",
            ),
        )
        node = explain(store, "report/2026-08-28/attribution.json")
        assert node.chain is None
        assert node.to_dict()["money_path_chain"] is None

    def test_a_break_is_the_first_thing_render_prints(self, store) -> None:
        """Not appended under forty lines of hops. A break printed last is a
        warning wearing a finding's clothes."""
        keys = self._chain(store)
        tampered = json.loads(store.get_bytes(keys[0]))
        tampered["seed"] = 999
        store.put_bytes(keys[0], json.dumps(tampered, indent=2, sort_keys=True).encode("utf-8"))

        node = explain(store, predictions_key(TRADING_DAY))
        assert node.chain is not None and node.chain.status == "failed"
        text = render(node)
        assert text.splitlines()[0].startswith("MONEY-PATH CHAIN: FAILED")

    def test_a_break_is_a_named_failure_a_caller_can_exit_non_zero_on(self, store) -> None:
        keys = self._chain(store)
        tampered = json.loads(store.get_bytes(keys[0]))
        tampered["seed"] = 999
        store.put_bytes(keys[0], json.dumps(tampered, indent=2, sort_keys=True).encode("utf-8"))

        verdict = verify_money_path_chain(store)
        with pytest.raises(MoneyPathChainError, match="CHAIN BROKEN"):
            verdict.raise_if_broken()

    def test_an_intact_chain_raises_nothing(self, store) -> None:
        self._chain(store)
        verify_money_path_chain(store).raise_if_broken()

    def test_explain_itself_never_raises_on_a_break(self, store) -> None:
        """An operator holding a broken chain needs to SEE the walk that
        reaches it; a command that refused to print the lineage at the moment
        the lineage became interesting would be the wrong trade. The refusal
        belongs to the caller, via `raise_if_broken`."""
        keys = self._chain(store)
        tampered = json.loads(store.get_bytes(keys[0]))
        tampered["seed"] = 999
        store.put_bytes(keys[0], json.dumps(tampered, indent=2, sort_keys=True).encode("utf-8"))
        node = explain(store, predictions_key(TRADING_DAY))
        assert "produced by" in render(node) or "run 01JG" in render(node)
