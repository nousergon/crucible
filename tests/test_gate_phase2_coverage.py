"""The four phase-2 deliverables that had no clause, and the readings they give.

`alpha-engine-config-I10309` built the instrument that counts how much of a
phase issue its gate actually grades. Run against phase 2 the moment it landed,
it read: *grades 2 of 6 deliverables; not gate-readable:
two_page_conditions_on_real_channel, transient_retry_class_in_runner,
fault_injection_against_scheduled_path, runbook_in_readme.* All four existed
and were built; none was graded by anything. Under Brian's 2026-09-09 ruling
("all of phase 2 should be fully validated by the next weekly SF"), a
deliverable asserted by a human having looked is not validated.

Every test here is written the way `AGENTS.md`'s test discipline demands: each
clause is shown REFUSING, against a store that is missing exactly the artifact
it reads, before it is shown accepting. That is policy §7.4's requirement and
it is also the only way to tell a clause that grades something from a clause
that returns MET over an empty listing — the defect this whole file exists to
keep out.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import re
from pathlib import Path

import pytest

import crucible.gate as gate_module
from crucible.alerts import (
    DESTINATION_MUTED,
    DESTINATION_SNS_ONLY,
    NON_OPERATOR_DESTINATIONS,
    PAGE_CONDITIONS,
)
from crucible.components import load_registry
from crucible.gate import (
    FAULT_RECORD_BUS_FIELD,
    FAULT_RECORD_MANIFEST_FIELD,
    MANIFEST_ATTEMPT_INITIAL,
    RUNBOOK_PROCEDURES,
    SCRIPTED_FAULTS,
)
from crucible.keys import (
    FAULT_INJECTION_ROOT,
    fault_injection_key,
    manifest_key,
)
from crucible.runner import TRANSIENT_CLASSIFIERS
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 9, 4)
DAY = FRIDAY.isoformat()
OPERATOR_CHANNEL = "operator_chat"

#: The fault suite the plan's four scripted faults are exercised in. Parsed,
#: never imported: importing it would run its fixtures.
FAULT_SUITE = Path(__file__).resolve().parent / "faults" / "test_four_scripted_faults.py"


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def _put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


def _bus_key(condition: str, discriminator: str) -> str:
    return f"alerts/{DAY}/{condition}.{discriminator}.json"


def _row(
    store: LocalStore,
    condition: str,
    *,
    sent: bool = True,
    destination: str | None = OPERATOR_CHANNEL,
    discriminator: str = "one",
) -> str:
    key = _bus_key(condition, discriminator)
    document: dict = {
        "schema_version": "alert_bus.v2",
        "condition": condition,
        "sent": sent,
        "trading_day": DAY,
        "members": [{"job": "weekly", "run_id": None, "reason": "x"}],
    }
    if destination is not None:
        document["destination"] = destination
    _put(store, key, document)
    return key


def _manifest(
    store: LocalStore, job: str, *, attempts: list[dict] | None = None, day: str = DAY
) -> str:
    key = manifest_key(job, day)
    _put(
        store,
        key,
        {
            "status": "ok",
            "reason": "",
            "attempts": attempts or [{"n": 1, "reason": MANIFEST_ATTEMPT_INITIAL}],
        },
    )
    return key


# ── deliverable 2: both conditions route to the real channel ───────────────


class TestTwoPageConditionsOnRealChannel:
    """The clause reads `destination`, which nothing else reads."""

    def test_an_empty_bus_is_unmeasurable_not_unmet(self, store: LocalStore) -> None:
        """§7.4: the clause refuses to answer when the artifact is absent.

        A condition that has had nothing to page about has left no delivery
        record, and "never fired" is not "not deliverable" — the reading a
        clause returning MET over an empty listing would destroy.
        """
        clause = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert clause.unmeasurable and not clause.met
        for condition in PAGE_CONDITIONS:
            assert f"{condition}: no bus row exists" in clause.detail

    def test_met_when_every_condition_reached_an_operator_channel(self, store: LocalStore) -> None:
        for condition in PAGE_CONDITIONS:
            _row(store, condition)
        clause = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert OPERATOR_CHANNEL in clause.detail

    def test_the_sns_leg_alone_is_a_real_channel(self, store: LocalStore) -> None:
        """`sns_only` is a publish to the pages topic whose Telegram leg was
        not reached — one declared leg delivered, not a non-delivery."""
        for condition in PAGE_CONDITIONS:
            _row(store, condition, destination=DESTINATION_SNS_ONLY)
        clause = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert clause.met, clause.detail

    def test_a_page_routed_to_the_muted_topic_is_unmet(self, store: LocalStore) -> None:
        """The reading that makes this clause distinct from `pages_commissioned`.

        A `legacy=True` send lands on the v1 overlap topic and can still record
        `sent: true`; the lifecycle clause counts it, and nobody on the
        operator channel heard it.
        """
        assert DESTINATION_MUTED in NON_OPERATOR_DESTINATIONS
        for condition in PAGE_CONDITIONS:
            _row(store, condition, destination=DESTINATION_MUTED)
        clause = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert not clause.met and not clause.unmeasurable
        assert "non-operator-destined" in clause.detail

    def test_a_row_that_never_left_is_unmet(self, store: LocalStore) -> None:
        for condition in PAGE_CONDITIONS:
            _row(store, condition, sent=False)
        clause = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert not clause.met and not clause.unmeasurable

    def test_one_condition_short_is_not_met(self, store: LocalStore) -> None:
        _row(store, PAGE_CONDITIONS[0])
        clause = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert not clause.met
        assert f"{PAGE_CONDITIONS[1]}: no bus row exists" in clause.detail

    def test_a_row_with_no_destination_is_a_named_fault_not_a_pass(self, store: LocalStore) -> None:
        for condition in PAGE_CONDITIONS:
            _row(store, condition, destination=None)
        clause = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert not clause.met
        assert "malformed bus row" in clause.detail

    def test_this_clause_and_pages_commissioned_come_apart(self, store: LocalStore) -> None:
        """Neither clause can stand in for the other, shown in one store.

        Every condition is commissioned (fired, `sent: true`, member job now
        `ok`) and every one of them went to the muted topic.
        """
        _manifest(store, "alerts.sweep")
        _manifest(store, "weekly")
        for condition in PAGE_CONDITIONS:
            _row(store, condition, destination=DESTINATION_MUTED)
        commissioned = gate_module._clause_pages_commissioned(store)
        channel = gate_module._clause_two_page_conditions_on_real_channel(store)
        assert commissioned.met, commissioned.detail
        assert not channel.met, channel.detail


# ── deliverable 3: the transient-retry class, read from a manifest ──────────


class TestTransientRetryClassInRunner:
    def test_the_schema_can_record_every_class_the_runner_declares(self) -> None:
        """The contract check the clause makes before reading anything — and
        the one that separates its two negative readings."""
        declared = {reason for reason, _types, _needles in TRANSIENT_CLASSIFIERS}
        assert declared == set(gate_module._manifest_retry_reasons())

    def test_no_retry_anywhere_is_unmeasurable_and_says_which_reading_it_is(
        self, store: LocalStore
    ) -> None:
        """§7.4, and the distinction the issue was filed on: "nothing retried"
        must not render the same as "retry is not implemented"."""
        _manifest(store, "weekly")
        clause = gate_module._clause_transient_retry_class_in_runner(store, load_registry())
        assert clause.unmeasurable and not clause.met
        assert "nothing transient has occurred" in clause.detail
        assert "retry is not implemented" in clause.detail

    def test_met_when_a_manifest_records_a_declared_retry(self, store: LocalStore) -> None:
        reason = TRANSIENT_CLASSIFIERS[0][0]
        key = _manifest(
            store,
            "weekly",
            attempts=[{"n": 1, "reason": MANIFEST_ATTEMPT_INITIAL}, {"n": 2, "reason": reason}],
        )
        clause = gate_module._clause_transient_retry_class_in_runner(store, load_registry())
        assert clause.met and not clause.unmeasurable, clause.detail
        assert key in clause.evidence
        assert reason in clause.detail

    def test_a_retry_outside_the_declared_class_is_unmet(self, store: LocalStore) -> None:
        """A recorded retry whose reason the runner does not declare is a
        retry nobody authorised — evidence, and a finding, never a pass."""
        _manifest(
            store,
            "weekly",
            attempts=[
                {"n": 1, "reason": MANIFEST_ATTEMPT_INITIAL},
                {"n": 2, "reason": "because_it_felt_transient"},
            ],
        )
        clause = gate_module._clause_transient_retry_class_in_runner(store, load_registry())
        assert not clause.met and not clause.unmeasurable
        assert "outside the declared class" in clause.detail

    def test_an_unrecordable_class_is_unmeasurable_naming_it(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other negative reading, driven for real: a class the runner
        declares and the schema cannot record leaves no artifact, so its
        absence from the store says nothing at all."""
        monkeypatch.setattr(
            gate_module, "_manifest_retry_reasons", frozenset({"provider_5xx"}).copy
        )
        _manifest(store, "weekly")
        clause = gate_module._clause_transient_retry_class_in_runner(store, load_registry())
        assert clause.unmeasurable and not clause.met
        assert "cannot record" in clause.detail

    def test_no_retry_vocabulary_at_all_is_unmeasurable(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "_manifest_retry_reasons", frozenset().copy)
        clause = gate_module._clause_transient_retry_class_in_runner(store, load_registry())
        assert clause.unmeasurable and not clause.met
        assert "no retry vocabulary" in clause.detail


# ── deliverable 4: fault injection against the scheduled path ───────────────


def _fault_record(store: LocalStore, fault: str, *, manifest: str, bus: str) -> str:
    key = fault_injection_key(fault, DAY)
    _put(store, key, {FAULT_RECORD_MANIFEST_FIELD: manifest, FAULT_RECORD_BUS_FIELD: bus})
    return key


def _induced(store: LocalStore, fault: str, discriminator: str) -> str:
    manifest = _manifest(store, "weekly")
    bus = _row(store, "failure", discriminator=discriminator)
    return _fault_record(store, fault, manifest=manifest, bus=bus)


class TestFaultInjectionAgainstScheduledPath:
    def test_an_empty_store_is_unmeasurable_naming_every_missing_key(
        self, store: LocalStore
    ) -> None:
        """§7.4. No producer files a fault record today; the clause says
        exactly which key it looked for, per fault."""
        clause = gate_module._clause_fault_injection_against_scheduled_path(store)
        assert clause.unmeasurable and not clause.met
        for fault in SCRIPTED_FAULTS:
            assert f"{FAULT_INJECTION_ROOT}<trading-day>/{fault}.json" in clause.detail

    def test_met_when_every_scripted_fault_has_a_record_whose_keys_exist(
        self, store: LocalStore
    ) -> None:
        for index, fault in enumerate(SCRIPTED_FAULTS):
            _induced(store, fault, discriminator=f"f{index}")
        clause = gate_module._clause_fault_injection_against_scheduled_path(store)
        assert clause.met and not clause.unmeasurable, clause.detail

    def test_one_fault_short_is_unmeasurable_not_met(self, store: LocalStore) -> None:
        for index, fault in enumerate(SCRIPTED_FAULTS[:-1]):
            _induced(store, fault, discriminator=f"f{index}")
        clause = gate_module._clause_fault_injection_against_scheduled_path(store)
        assert clause.unmeasurable and not clause.met
        assert SCRIPTED_FAULTS[-1] in clause.detail

    def test_a_record_naming_a_manifest_the_store_does_not_hold_is_unmet(
        self, store: LocalStore
    ) -> None:
        """A claim the store contradicts is evidence, read positively — never
        the same reading as no record at all."""
        for index, fault in enumerate(SCRIPTED_FAULTS):
            _induced(store, fault, discriminator=f"f{index}")
        _fault_record(
            store,
            SCRIPTED_FAULTS[0],
            manifest=manifest_key("weekly", "2026-08-07"),
            bus=_bus_key("failure", "f0"),
        )
        clause = gate_module._clause_fault_injection_against_scheduled_path(store)
        assert not clause.met and not clause.unmeasurable
        assert "which the store does not hold" in clause.detail

    def test_a_record_naming_a_key_of_the_wrong_shape_is_unmet(self, store: LocalStore) -> None:
        for index, fault in enumerate(SCRIPTED_FAULTS):
            _induced(store, fault, discriminator=f"f{index}")
        _fault_record(
            store,
            SCRIPTED_FAULTS[1],
            manifest=manifest_key("weekly", DAY),
            bus=manifest_key("weekly", DAY),
        )
        clause = gate_module._clause_fault_injection_against_scheduled_path(store)
        assert not clause.met and not clause.unmeasurable
        assert "is not an alert bus key" in clause.detail

    def test_a_record_missing_a_field_is_unmet(self, store: LocalStore) -> None:
        for index, fault in enumerate(SCRIPTED_FAULTS):
            _induced(store, fault, discriminator=f"f{index}")
        _put(store, fault_injection_key(SCRIPTED_FAULTS[2], DAY), {FAULT_RECORD_BUS_FIELD: ""})
        clause = gate_module._clause_fault_injection_against_scheduled_path(store)
        assert not clause.met
        assert f"names no `{FAULT_RECORD_MANIFEST_FIELD}`" in clause.detail

    def test_the_declared_fault_list_matches_the_suite_that_exercises_them(self) -> None:
        """`SCRIPTED_FAULTS` is the plan's list, not a scan of the suite — so
        this is the guard that stops the two drifting apart silently."""
        tree = ast.parse(FAULT_SUITE.read_text(encoding="utf-8"))
        cases = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.startswith("TestFault")
        ]
        assert len(cases) == len(SCRIPTED_FAULTS), (
            f"{FAULT_SUITE.name} defines {cases}, which is not the "
            f"{len(SCRIPTED_FAULTS)} scripted faults the gate grades"
        )


# ── deliverable 6: the runbook, and every command it publishes ──────────────


def _readme(tmp_path: Path, body: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "README.md"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setattr(gate_module, "README_PATH", path)
    return path


def _runbook(**bodies: str) -> str:
    sections = "\n".join(f"### {verb}\n\n{body}\n" for verb, body in bodies.items())
    return f"# crucible\n\n## Runbook\n\n{sections}\n## Development\n"


class TestRunbookInReadme:
    def test_the_real_readme_is_met(self) -> None:
        clause = gate_module._clause_runbook_in_readme()
        assert clause.met and not clause.unmeasurable, clause.detail

    def test_an_absent_readme_is_unmet_with_the_path_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wheel install ships no README. UNMET, never a pass — this reader
        knows exactly which file it wanted and that it is not there."""
        missing = tmp_path / "nope" / "README.md"
        monkeypatch.setattr(gate_module, "README_PATH", missing)
        clause = gate_module._clause_runbook_in_readme()
        assert not clause.met and not clause.unmeasurable
        assert str(missing) in clause.detail

    def test_a_readme_with_no_runbook_section_is_unmet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _readme(tmp_path, "# crucible\n\nnothing here\n", monkeypatch)
        clause = gate_module._clause_runbook_in_readme()
        assert not clause.met

    def test_a_missing_procedure_is_unmet_naming_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§7.4 for this clause: drop one of the five and it refuses."""
        body = gate_module.README_PATH.read_text(encoding="utf-8")
        victim, _reserved = RUNBOOK_PROCEDURES[0]
        _readme(tmp_path, body.replace(f"### {victim}", "### something else"), monkeypatch)
        clause = gate_module._clause_runbook_in_readme()
        assert not clause.met
        assert victim in clause.detail

    def test_a_command_that_no_longer_parses_is_unmet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parser is the oracle. A renamed flag fails here rather than in
        an operator's terminal at the moment they reached for the runbook."""
        body = gate_module.README_PATH.read_text(encoding="utf-8")
        _readme(
            tmp_path,
            body.replace("crucible weekly --date", "crucible weekly --not-a-real-flag --date"),
            monkeypatch,
        )
        clause = gate_module._clause_runbook_in_readme()
        assert not clause.met
        assert "does not parse" in clause.detail

    def test_a_command_naming_a_job_the_cli_does_not_carry_is_unmet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = gate_module.README_PATH.read_text(encoding="utf-8")
        _readme(tmp_path, body.replace("crucible weekly ", "crucible weakly "), monkeypatch)
        clause = gate_module._clause_runbook_in_readme()
        assert not clause.met
        assert "which the CLI does not carry" in clause.detail

    def test_a_procedure_with_no_command_at_all_is_unmet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prose describing a procedure is not the procedure."""
        bodies = {
            verb: ("reserved" if reserved else "words, and no command block")
            for verb, reserved in RUNBOOK_PROCEDURES
        }
        _readme(
            tmp_path, _runbook(**{k.replace(" ", "_"): v for k, v in bodies.items()}), monkeypatch
        )
        clause = gate_module._clause_runbook_in_readme()
        assert not clause.met

    def test_a_reserved_procedure_publishing_a_command_is_unmet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reserved half, refusing. `unseal` is a human ruling (plan §9.4);
        a runbook that hands an operator a command for it is a defect, and a
        clause that graded "has a command that parses" would reward it."""
        reserved = [verb for verb, is_reserved in RUNBOOK_PROCEDURES if is_reserved]
        assert reserved, "no procedure is declared reserved — this test grades nothing"
        body = gate_module.README_PATH.read_text(encoding="utf-8")
        injected = body.replace(
            f"### {reserved[0]}\n",
            f"### {reserved[0]}\n\n```\ncrucible weekly --date 2026-08-07 --run-mode replay\n```\n",
            1,
        )
        _readme(tmp_path, injected, monkeypatch)
        clause = gate_module._clause_runbook_in_readme()
        assert not clause.met
        assert "reserved" in clause.detail


# ── coverage: the reading the issue was filed on ───────────────────────────


class TestPhase2GradesEveryDeliverable:
    def test_the_coverage_line_names_no_ungraded_deliverable(self, store: LocalStore) -> None:
        """`alpha-engine-config-I10314`'s own closes-when, asserted here rather
        than left to a live read: every phase-2 deliverable names a clause."""
        note = gate_module.coverage_note(
            "phase2",
            [
                c.name
                for c in gate_module.evaluate(store, gate="phase2", trading_day=FRIDAY).clauses
            ],
        )
        assert "not gate-readable" not in note
        assert re.search(r"grades all \d+ of \d+ ", note), note
