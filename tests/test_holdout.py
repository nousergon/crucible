"""The sealed holdout: the seal, the reader that withholds it, the ruled unseal.

`alpha-engine-config-I10502`, phase-3 deliverable `sealed_holdout`.

**Every test here was seen failing before the code that makes it pass**
(AGENTS.md "Test discipline"), and the balance of the module is deliberate:
plan §11 risk 1 is "v2 passes its gates because its tests were written to
pass", and the holdout is the principal defence against it, so the REFUSALS
are the subject — an unseal with no ruling, a malformed ruling, a payload
edited after sealing, a document that is not sealed at all. A suite that only
showed a valid holdout being read would have demonstrated nothing about what
the mechanism constrains.

Fixed date literals throughout, never `today` arithmetic. Every store is a
`LocalStore` under `tmp_path`: nothing here touches the live v2 store.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path

import pytest

import crucible
import crucible.gate as gate_module
from crucible.cli import USAGE_EXIT_CODE, main
from crucible.holdout import (
    RULING_REFERENCE_PATTERN,
    SEALED_HOLDOUT_SCHEMA_VERSION,
    UNSEAL_RECORD_SCHEMA_VERSION,
    HoldoutAbsentError,
    HoldoutError,
    HoldoutSealBrokenError,
    HoldoutUnsealedError,
    SealedHoldout,
    UnsealRulingRequiredError,
    assert_ruling_reference,
    holdout_key_readers,
    payload_digest,
    read_sealed_holdout,
    seal_document,
    unseal,
    unseal_record_bytes,
    unseal_records,
)
from crucible.keys import holdout_unseal_key, holdout_unseal_prefix, strategy_holdout_key
from crucible.store import LocalStore

#: A closed weekday two sessions before `FRIDAY`'s week — a fixed literal, so
#: what this suite tests does not move with the clock.
FRIDAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 28, 21, 5, tzinfo=dt.UTC)
#: This suite's own tracker, as a plain `int` assembled into the reference at
#: use — `tests/test_no_stale_tracker_literals.py` (alpha-engine-config-I9839)
#: refuses a hardcoded `I<N>` string anywhere in the package, and the same
#: shape is what `crucible.cli._EPIC_ISSUE` already uses.
_RULING_ISSUE = 10502
RULING = f"alpha-engine-config-I{_RULING_ISSUE}"

PAYLOAD = {
    "reserved_sessions": ["2026-07-01", "2026-07-02"],
    "reason_class": "final_evaluation",
}


def _sealed(**overrides) -> dict:
    document = seal_document(
        PAYLOAD,
        sealed_trading_day=FRIDAY.isoformat(),
        sealed_by="brian",
        purpose="final out-of-sample evaluation of the S slot",
    )
    document.update(overrides)
    return document


def _candidate(tmp_path: Path) -> Path:
    """A plaintext candidate payload OUTSIDE the store root — a file written
    into the store root would be listed as a store key and every
    "wrote nothing" assertion below would read it as a write."""
    path = tmp_path / "authoring" / "candidate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    return path


def _publish(store: LocalStore, document: dict) -> None:
    store.put_bytes(
        strategy_holdout_key(), json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    )


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


class TestTheSealRoundTrips:
    def test_a_sealed_document_reads_back_as_its_own_seal(self, store: LocalStore) -> None:
        document = _sealed()
        _publish(store, document)

        sealed = read_sealed_holdout(store)

        assert isinstance(sealed, SealedHoldout)
        assert sealed.digest == payload_digest(PAYLOAD)
        assert sealed.sealed_trading_day == FRIDAY.isoformat()
        assert sealed.sealed_by == "brian"
        assert "final out-of-sample" in sealed.purpose
        assert document["schema_version"] == SEALED_HOLDOUT_SCHEMA_VERSION

    def test_the_reader_hands_back_no_payload_anywhere(self, store: LocalStore) -> None:
        """The mechanism itself: a grading path holding a `SealedHoldout`
        cannot leak the reservation, however it logs or serialises it."""
        _publish(store, _sealed())

        sealed = read_sealed_holdout(store)

        assert not hasattr(sealed, "payload")
        rendered = f"{sealed.render()} {sealed.to_dict()}"
        for session in PAYLOAD["reserved_sessions"]:
            assert session not in rendered

    def test_formatting_is_not_part_of_the_holdouts_identity(self, store: LocalStore) -> None:
        """The digest is over CANONICAL JSON, so a re-indent of the published
        file is not a re-seal — otherwise every reformat would read as
        tampering and the real signal would be trained away."""
        document = _sealed()
        store.put_bytes(
            strategy_holdout_key(),
            json.dumps(document, indent=4, sort_keys=False).encode("utf-8"),
        )

        assert read_sealed_holdout(store).digest == payload_digest(PAYLOAD)

    def test_key_order_inside_the_payload_is_not_part_of_it_either(self) -> None:
        reordered = dict(reversed(list(PAYLOAD.items())))
        assert payload_digest(reordered) == payload_digest(PAYLOAD)

    def test_sealing_an_empty_payload_is_refused(self) -> None:
        with pytest.raises(HoldoutError, match="reserves nothing"):
            seal_document(
                {}, sealed_trading_day=FRIDAY.isoformat(), sealed_by="brian", purpose="nothing"
            )


class TestTheReaderRefuses:
    def test_an_absent_document_raises_absent_not_empty(self, store: LocalStore) -> None:
        with pytest.raises(HoldoutAbsentError, match=strategy_holdout_key()):
            read_sealed_holdout(store)

    def test_a_plain_unsealed_document_is_refused(self, store: LocalStore) -> None:
        """A holdout published without a seal is not a holdout that happens to
        be readable — it is a document at the holdout's key, and reading it as
        the reservation would grade against data nobody sealed."""
        store.put_bytes(strategy_holdout_key(), json.dumps(PAYLOAD).encode("utf-8"))

        with pytest.raises(HoldoutUnsealedError, match=SEALED_HOLDOUT_SCHEMA_VERSION):
            read_sealed_holdout(store)

    def test_a_payload_edited_after_sealing_is_refused_by_name(self, store: LocalStore) -> None:
        document = _sealed()
        document["payload"]["reserved_sessions"].append("2026-07-06")
        _publish(store, document)

        with pytest.raises(HoldoutSealBrokenError, match="edited after it was sealed"):
            read_sealed_holdout(store)

    def test_a_seal_naming_an_unknown_algorithm_is_refused(self, store: LocalStore) -> None:
        document = _sealed()
        document["seal"]["algorithm"] = "md5"
        _publish(store, document)

        with pytest.raises(HoldoutUnsealedError, match="algorithm"):
            read_sealed_holdout(store)

    def test_an_unparseable_document_is_a_problem_not_an_absence(self, store: LocalStore) -> None:
        store.put_bytes(strategy_holdout_key(), b"{not json")

        with pytest.raises(HoldoutUnsealedError):
            read_sealed_holdout(store)


class TestTheRulingIsRequired:
    @pytest.mark.parametrize(
        "ruling",
        [None, "", "I10502", "10502", "crucible-I10502", "alpha-engine-config-I0", "see I10502"],
    )
    def test_anything_that_is_not_a_fleet_ruling_reference_is_refused(self, ruling) -> None:
        with pytest.raises(UnsealRulingRequiredError):
            assert_ruling_reference(ruling, action="test")

    def test_the_refusal_names_the_flag_and_the_reserved_matter(self) -> None:
        with pytest.raises(UnsealRulingRequiredError) as caught:
            assert_ruling_reference(None, action="`crucible holdout --unseal`")
        message = str(caught.value)
        assert "--ruling" in message
        assert "RESERVED" in message
        assert "no proceed-anyway form" in message

    def test_a_valid_reference_is_returned_unchanged(self) -> None:
        assert assert_ruling_reference(RULING, action="test") == RULING

    def test_the_pattern_is_anchored_at_both_ends(self) -> None:
        assert RULING_REFERENCE_PATTERN.startswith("^")
        assert RULING_REFERENCE_PATTERN.endswith("$")

    def test_unseal_refuses_before_it_reads_the_store(self, store: LocalStore) -> None:
        """No document is published here at all. A refusal that depended on the
        holdout being readable would be a different answer on a bad day."""
        with pytest.raises(UnsealRulingRequiredError):
            unseal(
                store,
                ruling=None,
                trading_day=FRIDAY.isoformat(),
                operator="brian",
                reason="why",
                now_utc=NOW,
            )

    def test_unseal_refuses_without_a_reason(self, store: LocalStore) -> None:
        _publish(store, _sealed())
        with pytest.raises(HoldoutError, match="reason"):
            unseal(
                store,
                ruling=RULING,
                trading_day=FRIDAY.isoformat(),
                operator="brian",
                reason="",
                now_utc=NOW,
            )


class TestAnUnsealIsRecorded:
    def test_a_ruled_unseal_releases_the_payload_and_builds_the_record(
        self, store: LocalStore
    ) -> None:
        _publish(store, _sealed())

        payload, record = unseal(
            store,
            ruling=RULING,
            trading_day=FRIDAY.isoformat(),
            operator="brian",
            reason="phase-3 final evaluation",
            now_utc=NOW,
        )

        assert payload == PAYLOAD
        assert record["schema_version"] == UNSEAL_RECORD_SCHEMA_VERSION
        assert record["ruling"] == RULING
        assert record["holdout_digest"] == payload_digest(PAYLOAD)
        assert record["trading_day"] == FRIDAY.isoformat()
        assert record["unsealed_at_utc"].startswith("2026-08-28T21:05")

    def test_the_record_reads_back_off_the_prefix(self, store: LocalStore) -> None:
        _publish(store, _sealed())
        _, record = unseal(
            store,
            ruling=RULING,
            trading_day=FRIDAY.isoformat(),
            operator="brian",
            reason="phase-3 final evaluation",
            now_utc=NOW,
        )
        store.put_bytes(holdout_unseal_key(FRIDAY.isoformat(), RULING), unseal_record_bytes(record))

        records, problems = unseal_records(store)

        assert problems == []
        assert [r.ruling for r in records] == [RULING]
        assert records[0].store_key.startswith(holdout_unseal_prefix())

    def test_a_malformed_record_is_a_problem_never_a_dropped_row(self, store: LocalStore) -> None:
        """The prefix exists so an unseal cannot happen unobserved; a reader
        that skipped what it could not parse would reintroduce exactly the
        blindness the prefix removes."""
        store.put_bytes(
            holdout_unseal_key(FRIDAY.isoformat(), RULING),
            json.dumps({"schema_version": UNSEAL_RECORD_SCHEMA_VERSION}).encode("utf-8"),
        )

        records, problems = unseal_records(store)

        assert records == []
        assert len(problems) == 1


class TestTheCli:
    @staticmethod
    def _argv(*extra: str, store_path: Path) -> list[str]:
        return [
            "holdout",
            "--date",
            FRIDAY.isoformat(),
            "--store",
            str(store_path),
            "--run-mode",
            "replay",
            *extra,
        ]

    def test_unseal_without_a_ruling_exits_non_zero_and_writes_nothing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        _publish(LocalStore(tmp_path), _sealed())
        before = sorted(LocalStore(tmp_path).list_keys())

        code = main(self._argv("--unseal", "--reason", "because", store_path=tmp_path))

        assert code == USAGE_EXIT_CODE
        assert "--ruling" in capsys.readouterr().err
        assert sorted(LocalStore(tmp_path).list_keys()) == before

    def test_unseal_with_a_malformed_ruling_is_refused(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        _publish(LocalStore(tmp_path), _sealed())
        before = sorted(LocalStore(tmp_path).list_keys())

        code = main(
            self._argv("--unseal", "--ruling", "I10502", "--reason", "because", store_path=tmp_path)
        )

        assert code == USAGE_EXIT_CODE
        assert "not a ruling reference" in capsys.readouterr().err
        assert sorted(LocalStore(tmp_path).list_keys()) == before

    def test_unseal_without_a_reason_is_refused(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        _publish(LocalStore(tmp_path), _sealed())
        before = sorted(LocalStore(tmp_path).list_keys())

        code = main(self._argv("--unseal", "--ruling", RULING, store_path=tmp_path))

        assert code == USAGE_EXIT_CODE
        assert "--reason" in capsys.readouterr().err
        assert sorted(LocalStore(tmp_path).list_keys()) == before

    def test_a_ruled_unseal_files_the_record_and_its_manifest(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        _publish(LocalStore(tmp_path), _sealed())

        code = main(
            self._argv(
                "--unseal",
                "--ruling",
                RULING,
                "--reason",
                "phase-3 final evaluation",
                "--operator",
                "brian",
                store_path=tmp_path,
            )
        )

        assert code == 0
        store = LocalStore(tmp_path)
        record = json.loads(store.get_bytes(holdout_unseal_key(FRIDAY.isoformat(), RULING)))
        assert record["ruling"] == RULING
        assert record["operator"] == "brian"
        manifest = json.loads(store.get_bytes(f"runs/holdout/{FRIDAY.isoformat()}/run.json"))
        assert manifest["status"] == "ok"
        assert [row["name"] for row in manifest["metrics"]] == ["holdout_unseal"]
        assert any(RULING in out["key"] for out in manifest["outputs"])

    def test_a_dry_run_unseal_writes_nothing_at_all(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        _publish(LocalStore(tmp_path), _sealed())
        before = sorted(LocalStore(tmp_path).list_keys())

        code = main(
            self._argv(
                "--unseal",
                "--ruling",
                RULING,
                "--reason",
                "rehearsal",
                "--dry-run",
                store_path=tmp_path,
            )
        )

        assert code == 0
        assert sorted(LocalStore(tmp_path).list_keys()) == before

    def test_the_read_form_prints_the_seal_and_writes_no_manifest(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        _publish(LocalStore(tmp_path), _sealed())
        before = sorted(LocalStore(tmp_path).list_keys())

        code = main(self._argv(store_path=tmp_path))

        out = capsys.readouterr().out
        assert code == 0
        assert "holdout SEALED sha256:" in out
        assert "no unseal has ever been filed" in out
        for session in PAYLOAD["reserved_sessions"]:
            assert session not in out
        assert sorted(LocalStore(tmp_path).list_keys()) == before

    def test_the_read_form_exits_non_zero_on_a_fresh_store(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)

        code = main(self._argv(store_path=tmp_path))

        assert code == 1
        assert strategy_holdout_key() in capsys.readouterr().out
        assert sorted(LocalStore(tmp_path).list_keys()) == []

    def test_seal_and_unseal_together_are_refused(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        candidate = _candidate(tmp_path)

        code = main(
            self._argv(
                "--seal",
                str(candidate),
                "--unseal",
                "--ruling",
                RULING,
                "--reason",
                "x",
                store_path=tmp_path / "store",
            )
        )

        assert code == USAGE_EXIT_CODE
        assert "names no single intent" in capsys.readouterr().err

    def test_the_authoring_form_prints_a_sealed_document_and_touches_no_store(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        candidate = _candidate(tmp_path)

        code = main(
            self._argv(
                "--seal",
                str(candidate),
                "--purpose",
                "final out-of-sample evaluation",
                store_path=tmp_path / "store",
            )
        )

        assert code == 0
        document = json.loads(capsys.readouterr().out)
        assert document["seal"]["digest"] == payload_digest(PAYLOAD)
        assert sorted(LocalStore(tmp_path / "store").list_keys()) == []

    def test_the_authoring_form_requires_a_purpose(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        candidate = _candidate(tmp_path)

        code = main(self._argv("--seal", str(candidate), store_path=tmp_path / "store"))

        assert code == USAGE_EXIT_CODE
        assert "--purpose" in capsys.readouterr().err


class TestThePhase3Clause:
    @staticmethod
    def _clause(store: LocalStore):
        return gate_module._clause_sealed_holdout(store, [FRIDAY])

    def test_absent_reads_unmeasurable_and_never_met(self, store: LocalStore) -> None:
        clause = self._clause(store)

        assert clause.unmeasurable is True
        assert clause.met is False
        assert strategy_holdout_key() in clause.detail

    def test_a_sealed_holdout_reads_met(self, store: LocalStore) -> None:
        _publish(store, _sealed())

        clause = self._clause(store)

        assert clause.met is True
        assert clause.unmeasurable is False
        assert "no unseal" not in clause.detail
        for session in PAYLOAD["reserved_sessions"]:
            assert session not in clause.detail

    def test_a_tampered_holdout_reads_unmet_not_unmeasurable(self, store: LocalStore) -> None:
        """A document that exists and does not satisfy its own seal is a real
        reading of a real document — the opposite answer from 'nothing is
        published', and a clause conflating the two would grade a broken seal
        as a gap in the data."""
        document = _sealed()
        document["payload"]["reserved_sessions"] = ["2026-07-07"]
        _publish(store, document)

        clause = self._clause(store)

        assert clause.met is False
        assert clause.unmeasurable is False

    def test_an_unseal_record_that_names_no_ruling_reads_unmet(self, store: LocalStore) -> None:
        _publish(store, _sealed())
        store.put_bytes(
            holdout_unseal_key(FRIDAY.isoformat(), RULING),
            json.dumps(
                {
                    "schema_version": UNSEAL_RECORD_SCHEMA_VERSION,
                    "trading_day": FRIDAY.isoformat(),
                    "operator": "brian",
                    "reason": "unauthorised",
                    "holdout_digest": payload_digest(PAYLOAD),
                    "unsealed_at_utc": NOW.isoformat(timespec="seconds"),
                }
            ).encode("utf-8"),
        )

        clause = self._clause(store)

        assert clause.met is False
        assert clause.unmeasurable is False
        assert "nobody authorised" in clause.detail

    def test_a_properly_ruled_unseal_keeps_the_clause_met(self, store: LocalStore) -> None:
        _publish(store, _sealed())
        _, record = unseal(
            store,
            ruling=RULING,
            trading_day=FRIDAY.isoformat(),
            operator="brian",
            reason="phase-3 final evaluation",
            now_utc=NOW,
        )
        store.put_bytes(holdout_unseal_key(FRIDAY.isoformat(), RULING), unseal_record_bytes(record))

        clause = self._clause(store)

        assert clause.met is True
        assert RULING in clause.detail

    def test_the_clause_is_registered_in_the_phase3_deliverable_table(self) -> None:
        graded = {
            deliverable.graded_by
            for deliverable in gate_module.PHASE3_DELIVERABLES
            if deliverable.graded_by
        }
        assert "sealed_holdout" in graded


class TestTheMechanismHasOnlyOneDoor:
    def test_only_the_holdout_module_names_the_holdout_key(self) -> None:
        """The deliverable's own words: the grading path reads the holdout
        "only through the sealed reader". That is only true while nothing else
        in the package opens a second door, so this walks the tree rather than
        trusting the convention."""
        package = Path(crucible.__file__).parent
        allowed = set(holdout_key_readers())
        offenders = []
        for path in sorted(package.rglob("*.py")):
            relative = str(path.relative_to(package.parent))
            if relative in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "strategy_holdout_key":
                        offenders.append(f"{relative}:{node.lineno}")
        assert offenders == [], offenders

    def test_the_guard_fires_on_a_second_door(self, tmp_path) -> None:
        """A guard nobody has seen fail is a guard nobody knows works."""
        source = "from crucible.keys import strategy_holdout_key\nx = strategy_holdout_key()\n"
        tree = ast.parse(source)
        hits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "strategy_holdout_key"
        ]
        assert hits == [2]

    def test_the_module_declares_no_parallel_reserved_event_list(self) -> None:
        """`alpha-engine-config-I10416` §11 row 9 already declares ONE config
        surface for reserved actions — `Settings.autonomy_reserved_events`. A
        second list here would be a second thing to widen, in a repository that
        goes public, out of sight of the operator who audits the first."""
        source = (Path(crucible.__file__).parent / "holdout.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstrings = {
            # `clean=False`: the cleaned form strips indentation and would not
            # compare equal to the raw `ast.Constant` value below, so every
            # docstring would count as a string literal.
            ast.get_docstring(node, clean=False)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        }
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in docstrings
        ]
        assert not [text for text in literals if text.startswith("PutRolePolicy")]
        assert "autonomy_reserved_events" not in "".join(
            text for text in literals if "reserved_events" in text
        )
