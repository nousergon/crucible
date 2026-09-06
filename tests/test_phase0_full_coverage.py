"""Phase 0 grades all five of its deliverables, and each clause can FAIL.

Normative source: Brian's ruling of 2026-09-04 on `alpha-engine-config-I9964`
— option (a), *"lets complete phase 0 now so after the weekly sf phase 0 will
be fully met"*. Three of `alpha-engine-config-I9756`'s five deliverables were
declared "not gate-readable", each on the same argument: the fact lives in
AWS and a gate does not call AWS. That is true about the READ and a
non-sequitur about the CLAUSE — `old_weekly_within_cadence` had been reading
a live AWS fact through a producer since `alpha-engine-config-I9860`. So
phase 0 was on course to exit on gate evidence covering 40% of itself, in a
rebuild whose plan §11 risk 1 is *"v2 passes its gates because its tests were
written to pass"*.

**Every one of the three deliverables is already TRUE**, measured
2026-09-04: all six functions absent from `lambda:ListFunctions`; the
2026-09-03 weekly execution's input carrying
`"sns_topic_arn": "...:alpha-engine-alerts-muted"`; `get-bucket-versioning`
returning `Enabled` and the `crucible-v2` stack carrying `system=crucible-v2`.
So all three clauses read MET on their first run — **and a clause that always
returns MET is indistinguishable from a correct one until the day it
matters**. Plan §11 row 1 / policy §7.4 therefore governs this file: every
clause below is shown FAILING against a planted negative, and each
demonstration asserts the two readings DIFFER rather than asserting the
negative case alone. `_mutated` exists for the other half of that rule — a
fixture edit that matched nothing is exactly how a demonstration silently
proves nothing.

**And the third answer.** An absent or unreadable document is UNMEASURABLE,
never MET and never UNMET: "we could not ask" is a fact about our access, not
about the system (`crucible/gate.py`'s module docstring; principle 7). A
permissions failure rendering as a green clause is the defect class this arc
has now found five times, so each clause is also shown reading `[?]` — with
its reason — against an absence and against a denial.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from typing import Any

import pytest

from crucible.gate import (
    LEGACY_DEAD_LAMBDA_NAMES,
    LEGACY_DEAD_LAMBDAS_SCHEMA_VERSION,
    LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
    MUTED_ALERTS_TOPIC_NAME,
    PHASE0_DELIVERABLES,
    V2_STORE_VERSIONING_ENABLED,
    V2_TAG_ACCEPTANCE_CLAUSE_ID,
    evaluate,
    expected_legacy_weekly_window,
    legacy_dead_lambdas_key,
    legacy_weekly_executions_key,
    weekly_anchor,
)
from crucible.keys import acceptance_reading_key
from crucible.store import LocalStore

#: A Friday close. Fixed literal, never `today` arithmetic — a test whose
#: subject moves with the clock stops testing the same thing.
FRIDAY = dt.date(2026, 8, 28)
ANCHOR = weekly_anchor(FRIDAY)

#: A live sibling of one of the six, and the whole reason the contract is per
#: EXACT name. `alpha-engine-research-eval-judge-process` EXISTS (measured
#: 2026-09-04, along with `-poll`, `-submit` and `-spot-dispatcher`) and every
#: one of them starts with `alpha-engine-research-eval-judge`, which is
#: deleted. A prefix, substring or `startswith` probe reports four survivors
#: and this deliverable UNMET forever against a system that satisfies it.
LIVE_SIBLING = "alpha-engine-research-eval-judge-process"

MUTED_TOPIC_ARN = f"arn:aws:sns:us-east-1:acct:{MUTED_ALERTS_TOPIC_NAME}"
PAGING_TOPIC_ARN = "arn:aws:sns:us-east-1:acct:alpha-engine-alerts"


def _put(store: LocalStore, key: str, document: Any) -> None:
    store.put_bytes(key, json.dumps(document, sort_keys=True).encode("utf-8"))


def _mutated(before: Any, after: Any, what: str) -> Any:
    """``after``, having asserted it is not ``before``.

    The §7.4 trap in miniature: a demonstration built on a fixture edit that
    matched nothing reads exactly like one built on an edit that worked. Every
    negative case below is constructed through here, so a planted failure that
    did not actually change the document fails the test that plants it rather
    than quietly grading the positive case twice.
    """
    assert after != before, f"the {what} fixture mutation changed nothing"
    return after


def _executions(*, topic: str | None = MUTED_TOPIC_ARN, with_field: bool = True) -> dict:
    day = ANCHOR.isoformat()
    executions = [
        {
            "name": "uuid_skip",
            "start": f"{day}T09:00:49+00:00",
            "stop": f"{day}T09:00:52+00:00",
            "duration_seconds": 3.0,
            "status": "SUCCEEDED",
        },
        {
            "name": "uuid_run",
            "start": f"{day}T09:00:49+00:00",
            "stop": f"{day}T14:02:40+00:00",
            "duration_seconds": 18111.0,
            "status": "SUCCEEDED",
        },
    ]
    if with_field:
        for execution in executions:
            execution["sns_topic_arn"] = topic
    return {
        "schema_version": LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION,
        "executions_started": len(executions),
        "executions": executions,
        "source": "a filed record, not a live API call",
    }


def _probe(*, alive: tuple[str, ...] = (), extra: tuple[str, ...] = ()) -> dict:
    names = tuple(LEGACY_DEAD_LAMBDA_NAMES) + extra
    return {
        "schema_version": LEGACY_DEAD_LAMBDAS_SCHEMA_VERSION,
        "probed_at": "2026-08-29T15:00:00+00:00",
        "region": "us-east-1",
        "functions": [{"name": name, "present": name in alive} for name in names],
        "source": "a filed record, not a live API call",
    }


def _reading(*, met: bool = True, versioning: str | None = V2_STORE_VERSIONING_ENABLED) -> dict:
    met_ids = ["TestAlerting::test_there_are_exactly_two_page_conditions"]
    unmet_ids: list[str] = []
    (met_ids if met else unmet_ids).append(V2_TAG_ACCEPTANCE_CLAUSE_ID)
    document: dict[str, Any] = {
        "met": len(met_ids),
        "unmet": len(unmet_ids),
        "unmeasurable": 0,
        "commit": "0" * 40,
        "measured_at": "2026-08-28T12:00:00+00:00",
        "met_clauses": sorted(met_ids),
        "unmet_clauses": sorted(unmet_ids),
        "unmeasurable_clauses": [],
    }
    if versioning is not None:
        document["store_versioning"] = versioning
    return document


def _seed(tmp_path, *, executions=None, probe=None, reading=None) -> LocalStore:
    """A store in which all three new clauses read MET, unless overridden."""
    store = LocalStore(tmp_path)
    _put(store, legacy_weekly_executions_key(ANCHOR.isoformat()), executions or _executions())
    _put(store, legacy_dead_lambdas_key(ANCHOR.isoformat()), probe or _probe())
    _put(store, acceptance_reading_key(FRIDAY.isoformat()), reading or _reading())
    return store


def _clause(store: LocalStore, name: str):
    result = evaluate(store, gate="phase0", trading_day=FRIDAY)
    return next(c for c in result.clauses if c.name == name)


class _CostExplorer:
    """`list_cost_allocation_tags` speaking the real response shape.

    `alpha-engine-config-I10076` deliverable 4: the tag clause reads Billing's
    activation state live, so every test in this module that expects the
    clause MET needs a Cost Explorer that says `Active` -- the module's autouse
    fixture below supplies one, and the class at the bottom swaps in the
    other answers.
    """

    def __init__(self, status: str | None = "Active", *, fail: Exception | None = None) -> None:
        self.status = status
        self.fail = fail

    def list_cost_allocation_tags(self, **_request: Any) -> dict[str, Any]:
        if self.fail is not None:
            raise self.fail
        if self.status is None:
            return {"CostAllocationTags": []}
        return {
            "CostAllocationTags": [
                {
                    "TagKey": "system",
                    "Status": self.status,
                    "LastUpdatedDate": "2026-09-06T14:56:34Z",
                }
            ]
        }


@pytest.fixture(autouse=True)
def _billing_has_the_key_active(monkeypatch: pytest.MonkeyPatch) -> None:
    import crucible.gate as gate_module  # noqa: PLC0415 - local to the fixture

    monkeypatch.setattr(gate_module, "_ce_client", lambda: _CostExplorer("Active"))


class _AccessDenied(LocalStore):
    """A store that refuses one key the way S3 refuses an unauthorised read."""

    def __init__(self, root, denied: str) -> None:
        super().__init__(root)
        self._denied = denied

    def exists(self, key: str) -> bool:
        if key == self._denied:
            raise PermissionError("AccessDenied")
        return super().exists(key)


class TestCoverageIsTheAcceptanceTest:
    """`alpha-engine-config-I9964`'s own closes-when, asserted rather than
    observed: the `coverage:` line that read `grades 2 of 5 ...; not
    gate-readable: dead_lambdas_deleted, old_alerts_muted,
    v2_resources_tagged_and_versioned`."""

    def test_the_coverage_line_reads_five_of_five(self, tmp_path) -> None:
        result = evaluate(_seed(tmp_path), gate="phase0", trading_day=FRIDAY)
        assert "5 of 5" in result.coverage, result.coverage
        assert "not gate-readable" not in result.coverage, result.coverage

    def test_no_deliverable_is_ungraded(self) -> None:
        assert [d.id for d in PHASE0_DELIVERABLES if d.graded_by is None] == []

    def test_the_seeded_store_meets_every_clause(self, tmp_path) -> None:
        result = evaluate(_seed(tmp_path), gate="phase0", trading_day=FRIDAY)
        assert result.met, result.render()
        assert len(result.clauses) == len(PHASE0_DELIVERABLES)


class TestDeadLambdasDeleted:
    def test_a_surviving_function_flips_the_reading(self, tmp_path) -> None:
        """§7.4. All six are absent today, so this clause reads MET on its
        first run and would read MET forever if the reader were broken. The
        planted survivor is what shows it is not."""
        clean = _probe()
        planted = _mutated(clean, _probe(alive=("alpha-engine-ec2-lifecycle",)), "survivor")
        met = _clause(_seed(tmp_path / "a", probe=clean), "dead_lambdas_deleted")
        unmet = _clause(_seed(tmp_path / "b", probe=planted), "dead_lambdas_deleted")
        assert met.met and not unmet.met, "the two readings did not differ"
        assert not unmet.unmeasurable, "a function that IS there is a finding, not a denial"
        assert "alpha-engine-ec2-lifecycle" in unmet.detail

    def test_a_live_sibling_sharing_the_prefix_does_not_flip_it(self, tmp_path) -> None:
        """The trap this contract exists for. A probe that also reports the
        LIVE `...-eval-judge-process` as present must still read MET: that is
        a different function, and the deliverable names the bare one."""
        probe = _mutated(_probe(), _probe(alive=(LIVE_SIBLING,), extra=(LIVE_SIBLING,)), "sibling")
        clause = _clause(_seed(tmp_path, probe=probe), "dead_lambdas_deleted")
        assert clause.met, clause.detail

    @pytest.mark.parametrize(
        ("probe", "expected"),
        [
            (None, "no filed probe"),
            ({"schema_version": "legacy-dead-lambdas.v0"}, "schema_version"),
            ({"schema_version": LEGACY_DEAD_LAMBDAS_SCHEMA_VERSION, "functions": []}, "not a list"),
        ],
        ids=["absent", "an unrecognised version", "an empty function list"],
    )
    def test_a_document_that_cannot_answer_is_unmeasurable(self, tmp_path, probe, expected) -> None:
        store = LocalStore(tmp_path)
        _put(store, legacy_weekly_executions_key(ANCHOR.isoformat()), _executions())
        _put(store, acceptance_reading_key(FRIDAY.isoformat()), _reading())
        if probe is not None:
            _put(store, legacy_dead_lambdas_key(ANCHOR.isoformat()), probe)
        clause = _clause(store, "dead_lambdas_deleted")
        assert clause.unmeasurable and not clause.met
        assert expected in clause.detail

    def test_a_probe_covering_only_five_of_the_six_is_unmeasurable(self, tmp_path) -> None:
        """A narrower question answered under the wider name. Five absent
        functions is not "the six are deleted", and MET here would publish the
        narrower answer as the deliverable."""
        full = _probe()
        narrow = copy.deepcopy(full)
        narrow["functions"] = [
            f for f in narrow["functions"] if f["name"] != "alpha-engine-ec2-lifecycle"
        ]
        clause = _clause(
            _seed(tmp_path, probe=_mutated(full, narrow, "narrowed probe")), "dead_lambdas_deleted"
        )
        assert clause.unmeasurable and not clause.met
        assert "narrower question" in clause.detail

    def test_a_non_boolean_present_is_unmeasurable_not_absent(self, tmp_path) -> None:
        """A probe that answered `null` did not answer. Reading it as falsey —
        the one-character version of this bug — would turn a failed AWS call
        into a deleted function."""
        full = _probe()
        broken = copy.deepcopy(full)
        broken["functions"][0]["present"] = None
        clause = _clause(
            _seed(tmp_path, probe=_mutated(full, broken, "null present")), "dead_lambdas_deleted"
        )
        assert clause.unmeasurable and not clause.met

    def test_a_denied_read_is_unmeasurable(self, tmp_path) -> None:
        _seed(tmp_path)
        store = _AccessDenied(tmp_path, legacy_dead_lambdas_key(ANCHOR.isoformat()))
        clause = _clause(store, "dead_lambdas_deleted")
        assert clause.unmeasurable and not clause.met
        assert "about our access" in clause.detail


class TestOldAlertsMuted:
    def test_a_paging_topic_flips_the_reading(self, tmp_path) -> None:
        """§7.4. The live weekly input already names the muted topic, so this
        clause reads MET today; the planted paging ARN is what shows the
        reader can say no."""
        muted = _executions()
        paging = _mutated(muted, _executions(topic=PAGING_TOPIC_ARN), "paging topic")
        met = _clause(_seed(tmp_path / "a", executions=muted), "old_alerts_muted")
        unmet = _clause(_seed(tmp_path / "b", executions=paging), "old_alerts_muted")
        assert met.met and not unmet.met, "the two readings did not differ"
        assert not unmet.unmeasurable
        assert "alpha-engine-alerts" in unmet.detail

    def test_a_topic_whose_name_merely_contains_the_muted_one_does_not_pass(self, tmp_path) -> None:
        """The match is on the ARN's last segment, whole. A topic named
        `alpha-engine-alerts-muted-shadow` is a different topic."""
        muted = _executions()
        lookalike = _mutated(
            muted,
            _executions(topic=f"{MUTED_TOPIC_ARN}-shadow"),
            "look-alike topic",
        )
        clause = _clause(_seed(tmp_path, executions=lookalike), "old_alerts_muted")
        assert not clause.met and not clause.unmeasurable

    def test_an_input_declaring_no_topic_is_unmet_not_met(self, tmp_path) -> None:
        """`null` is a fact about the input — the execution declared no topic
        — and it is not evidence that alerts were muted."""
        muted = _executions()
        none_declared = _mutated(muted, _executions(topic=None), "null topic")
        clause = _clause(_seed(tmp_path, executions=none_declared), "old_alerts_muted")
        assert not clause.met and not clause.unmeasurable
        assert "declared no topic" in clause.detail

    def test_a_week_with_no_executions_is_unmeasurable_not_met(self, tmp_path) -> None:
        """Routing cannot be read from an absence. "No execution named a
        paging topic" and "no execution ran" are the same reading and only one
        of them is the deliverable — the cadence clause grades the other."""
        full = _executions()
        empty = copy.deepcopy(full)
        empty["executions"] = []
        empty["executions_started"] = 0
        clause = _clause(
            _seed(tmp_path, executions=_mutated(full, empty, "empty week")), "old_alerts_muted"
        )
        assert clause.unmeasurable and not clause.met

    def test_a_pre_I9964_document_is_unmeasurable_for_routing_and_fine_for_cadence(
        self, tmp_path
    ) -> None:
        """The reason the field was added WITHOUT a `schema_version` bump. A
        week filed between `alpha-engine-config-I9962` and this change answers
        the cadence question correctly and says nothing about routing —
        bumping the version would have made phase 0's only currently-graded
        clause read UNMEASURABLE on every week already in the store."""
        with_field = _executions()
        without = _mutated(with_field, _executions(with_field=False), "dropped topic field")
        store = _seed(tmp_path, executions=without)
        routing = _clause(store, "old_alerts_muted")
        cadence = _clause(store, "old_weekly_within_cadence")
        assert routing.unmeasurable and not routing.met
        assert cadence.met, cadence.detail

    def test_a_denied_read_is_unmeasurable(self, tmp_path) -> None:
        _seed(tmp_path)
        store = _AccessDenied(tmp_path, legacy_weekly_executions_key(ANCHOR.isoformat()))
        clause = _clause(store, "old_alerts_muted")
        assert clause.unmeasurable and not clause.met
        assert "about our access" in clause.detail

    def test_a_declared_window_disagreeing_with_the_anchor_is_unmeasurable(self, tmp_path) -> None:
        """`alpha-engine-config-I9992`. The routing clause reads the SAME
        document the cadence clause does
        (`tests/test_legacy_weekly_executions_contract.py::TestTheDeclaredWindowGatesTheClause`
        holds the cadence side), so a wrong window must gate it here too — a
        producer bug that mis-collected the week would otherwise still let a
        confidently wrong routing count through."""
        good = _executions()
        wrong_window = copy.deepcopy(good)
        wrong_window["window"] = {"start": "2000-01-01", "end": "2000-01-02"}
        clause = _clause(
            _seed(tmp_path, executions=_mutated(good, wrong_window, "wrong window")),
            "old_alerts_muted",
        )
        assert not clause.met
        assert clause.unmeasurable, clause.detail
        expected_start, expected_end = expected_legacy_weekly_window(ANCHOR)
        assert expected_start in clause.detail and expected_end in clause.detail

    def test_a_correctly_declared_window_reads_exactly_as_no_window_at_all(self, tmp_path) -> None:
        good = _executions()
        start, end = expected_legacy_weekly_window(ANCHOR)
        with_window = copy.deepcopy(good)
        with_window["window"] = {"start": start, "end": end}
        clause = _clause(
            _seed(tmp_path, executions=_mutated(good, with_window, "correct window")),
            "old_alerts_muted",
        )
        assert clause.met, clause.detail
        assert not clause.unmeasurable


class TestV2ResourcesTaggedAndVersioned:
    def test_the_tag_clause_going_unmet_flips_the_reading(self, tmp_path) -> None:
        """§7.4, half one. The §2 cost-tag clause is in the ratchet's `met`
        set today, so this reads MET; the planted unmet id is what shows the
        gate clause reads the NAME rather than the aggregate."""
        good = _reading()
        bad = _mutated(good, _reading(met=False), "tag clause unmet")
        met = _clause(_seed(tmp_path / "a", reading=good), "v2_resources_tagged_and_versioned")
        unmet = _clause(_seed(tmp_path / "b", reading=bad), "v2_resources_tagged_and_versioned")
        assert met.met and not unmet.met, "the two readings did not differ"
        assert not unmet.unmeasurable
        assert "not met" in unmet.detail

    def test_suspended_versioning_flips_the_reading(self, tmp_path) -> None:
        """§7.4, half two. `get-bucket-versioning` returns `Enabled` today."""
        good = _reading()
        bad = _mutated(good, _reading(versioning="Suspended"), "suspended versioning")
        met = _clause(_seed(tmp_path / "a", reading=good), "v2_resources_tagged_and_versioned")
        unmet = _clause(_seed(tmp_path / "b", reading=bad), "v2_resources_tagged_and_versioned")
        assert met.met and not unmet.met, "the two readings did not differ"
        assert "Suspended" in unmet.detail

    def test_an_aggregate_only_reading_is_unmeasurable(self, tmp_path) -> None:
        """The exact gap `alpha-engine-config-I9964` names: the document as it
        was filed before this change — `{"commit":…, "met":22, "unmet":2}` —
        cannot say whether the ONE clause this deliverable is graded by is
        among the 22. Not met, and not a finding: unmeasurable."""
        full = _reading()
        aggregate = copy.deepcopy(full)
        for field in ("met_clauses", "unmet_clauses", "unmeasurable_clauses", "store_versioning"):
            aggregate.pop(field, None)
        clause = _clause(
            _seed(tmp_path, reading=_mutated(full, aggregate, "aggregate-only reading")),
            "v2_resources_tagged_and_versioned",
        )
        assert clause.unmeasurable and not clause.met
        assert "met_clauses" in clause.detail

    def test_a_missing_versioning_field_is_unmeasurable_not_suspended(self, tmp_path) -> None:
        """`ci.yml` omits the field when `get-bucket-versioning` failed. That
        is a statement about the producer's access, and grading it as
        `Suspended` would publish a permissions failure as a finding about the
        bucket."""
        full = _reading()
        no_versioning = _mutated(full, _reading(versioning=None), "dropped versioning")
        clause = _clause(
            _seed(tmp_path, reading=no_versioning), "v2_resources_tagged_and_versioned"
        )
        assert clause.unmeasurable and not clause.met
        assert "store_versioning" in clause.detail

    def test_no_reading_at_all_is_unmeasurable(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _put(store, legacy_weekly_executions_key(ANCHOR.isoformat()), _executions())
        _put(store, legacy_dead_lambdas_key(ANCHOR.isoformat()), _probe())
        clause = _clause(store, "v2_resources_tagged_and_versioned")
        assert clause.unmeasurable and not clause.met

    def test_the_most_recent_reading_in_the_span_is_the_one_graded(self, tmp_path) -> None:
        """CI files this document on merges, not on a schedule, so a clause
        reading only the render day would flap with the merge calendar. The
        newest reading wins, and it is the one whose verdict shows."""
        store = _seed(tmp_path, reading=_reading(met=False))
        older = FRIDAY - dt.timedelta(days=1)
        _put(store, acceptance_reading_key(older.isoformat()), _reading())
        clause = _clause(store, "v2_resources_tagged_and_versioned")
        assert not clause.met, "an older green reading must not override the newest one"
        assert FRIDAY.isoformat() in clause.detail

    def test_a_reading_dated_after_the_render_day_does_not_satisfy_it(self, tmp_path) -> None:
        """A document that did not exist when the gate was read cannot satisfy
        the exit — the same rule phase 1's review clause carries."""
        store = _seed(tmp_path, reading=_reading(met=False))
        later = FRIDAY + dt.timedelta(days=4)
        _put(store, acceptance_reading_key(later.isoformat()), _reading())
        clause = _clause(store, "v2_resources_tagged_and_versioned")
        assert not clause.met

    def test_a_denied_read_is_unmeasurable(self, tmp_path) -> None:
        _seed(tmp_path)
        store = _AccessDenied(tmp_path, acceptance_reading_key(FRIDAY.isoformat()))
        clause = _clause(store, "v2_resources_tagged_and_versioned")
        assert clause.unmeasurable and not clause.met
        assert "about our access" in clause.detail


class TestTheTagKeyMustBeActiveInBilling:
    """`alpha-engine-config-I10076` deliverable 4. Measured 2026-09-06: every
    resource carried `system=crucible-v2`, the store was versioned, this clause
    read MET -- and Billing had never activated `system` as a cost-allocation
    tag, so every tag-filtered dollar read `$0.00`. A tagged estate under an
    inactive key has no denominator; the clause now reads the key's state."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch, ce: _CostExplorer) -> None:
        import crucible.gate as gate_module  # noqa: PLC0415 - local to the test

        monkeypatch.setattr(gate_module, "_ce_client", lambda: ce)

    def test_an_inactive_key_is_unmet_naming_the_activation_command(
        self, tmp_path, monkeypatch
    ) -> None:
        self._patch(monkeypatch, _CostExplorer("Inactive"))
        clause = _clause(_seed(tmp_path, reading=_reading()), "v2_resources_tagged_and_versioned")
        assert not clause.met and not clause.unmeasurable
        assert "Inactive as a cost-allocation tag" in clause.detail
        assert "update-cost-allocation-tags-status" in clause.detail

    def test_a_key_billing_has_never_seen_is_unmet_too(self, tmp_path, monkeypatch) -> None:
        self._patch(monkeypatch, _CostExplorer(None))
        clause = _clause(_seed(tmp_path, reading=_reading()), "v2_resources_tagged_and_versioned")
        assert not clause.met and not clause.unmeasurable
        assert "absent as a cost-allocation tag" in clause.detail

    def test_a_denied_activation_read_is_unmeasurable_naming_the_action(
        self, tmp_path, monkeypatch
    ) -> None:
        self._patch(monkeypatch, _CostExplorer(fail=PermissionError("AccessDenied")))
        clause = _clause(_seed(tmp_path, reading=_reading()), "v2_resources_tagged_and_versioned")
        assert clause.unmeasurable and not clause.met
        assert "ce:ListCostAllocationTags" in clause.detail
        assert "ce:ListCostAllocationTags" in clause.evidence

    def test_an_active_key_reads_met_and_says_since_when(self, tmp_path) -> None:
        clause = _clause(_seed(tmp_path, reading=_reading()), "v2_resources_tagged_and_versioned")
        assert clause.met
        assert "Active as a cost-allocation tag since 2026-09-06" in clause.detail

    def test_the_activation_read_happens_only_after_the_filed_reading_passes(
        self, tmp_path, monkeypatch
    ) -> None:
        """An unmet acceptance reading is reported as such; Billing is not
        consulted for a deliverable the filed document already fails."""
        self._patch(monkeypatch, _CostExplorer(fail=AssertionError("must not be called")))
        clause = _clause(
            _seed(tmp_path, reading=_reading(met=False)), "v2_resources_tagged_and_versioned"
        )
        assert not clause.met and not clause.unmeasurable
        assert "not met" in clause.detail
