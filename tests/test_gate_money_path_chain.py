"""The phase-4 gate reads money-path chain verification
(`alpha-engine-config-I10627`, the last `closes-when` bullet of `-I10414`).

`-I10414` built the hash chain over every money-path manifest and the verifier
that grades it. Nothing read the verdict, so the tamper-evidence plan §9.5
makes a hard precondition for phase 6 (real capital) was a control no gate
graded — and phase 6's entry condition (5) is "the money-path artifacts
hash-chained", a condition with no clause behind it.

**`crucible-PR240` has landed** (verified 2026-09-13, merged onto this branch
along with `origin/main`): `crucible.explain.verify_money_path_chain` exists,
and `_clause_money_path_chain_verified` reads it with no edit to
`crucible/gate.py` — the lazy import in the clause's own `try` block is what
makes that true. `TestTheClauseReadsTheRealVerifier` asserts that against the
real, empty tree; the readings below still drive a stand-in matching
`ChainVerification`'s declared shape via `monkeypatch`, which remains valid
now that the real attribute exists to patch.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

import crucible.explain as explain_module
from crucible.gate import _clause_money_path_chain_verified, _phase4
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)


@dataclass(frozen=True)
class _Record:
    """`crucible.explain.ChainRecord`'s shape, as `crucible-PR240` declares it."""

    index: int
    run_id: str
    key: str
    sha256: str
    prev_sha256: str | None
    actual_prev_sha256: str | None

    @property
    def intact(self) -> bool:
        return self.prev_sha256 == self.actual_prev_sha256


@dataclass(frozen=True)
class _Verification:
    """`crucible.explain.ChainVerification`'s shape."""

    status: str
    reason: str
    records: tuple[_Record, ...] = ()
    unlinked: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _install(monkeypatch: pytest.MonkeyPatch, verification: _Verification) -> None:
    monkeypatch.setattr(
        explain_module, "verify_money_path_chain", lambda store: verification, raising=False
    )


def _chain(*, links: int, break_at: int | None = None) -> tuple[_Record, ...]:
    records = []
    for index in range(links):
        claimed = None if index == 0 else f"{index - 1:064x}"
        actual = claimed
        if break_at is not None and index == break_at:
            actual = "f" * 64
        records.append(
            _Record(
                index=index,
                run_id=f"run{index}",
                key=f"runs/promote/2026-08-2{index}/run.json",
                sha256=f"{index:064x}",
                prev_sha256=claimed,
                actual_prev_sha256=actual,
            )
        )
    return tuple(records)


@pytest.fixture
def store(tmp_path: object) -> LocalStore:
    return LocalStore(tmp_path)  # type: ignore[arg-type]


class TestTheClauseReadsTheRealVerifier:
    def test_the_verifier_now_exists_and_the_clause_reads_it_with_no_edit(
        self, store: LocalStore
    ) -> None:
        """`crucible-PR240` has landed (this is the notification the removed
        stand-in test used to raise): `crucible.explain.verify_money_path_chain`
        exists, and `_clause_money_path_chain_verified` reads it with no change
        to `crucible/gate.py` — the lazy import in the clause's `try` block is
        exactly what makes that true.

        Measured against the real, empty tree: nothing has written a
        money-path manifest to `store`, so the real verifier grades it `ok`
        with an "empty" reason (`verify_money_path_chain`'s own contract), and
        the clause reads that as UNMET rather than inheriting `ok` — see
        `TestTheThreeReadingsAreDistinct.test_an_empty_chain_is_unmet_rather_than_inheriting_ok`
        for the same decision against a stand-in.
        """
        assert hasattr(explain_module, "verify_money_path_chain")
        clause = _clause_money_path_chain_verified(store)
        assert not clause.met
        assert not clause.unmeasurable, clause.detail
        assert "empty" in clause.detail

    def test_the_clause_is_registered_on_phase_4(self, store: LocalStore) -> None:
        names = [c.name for c in _phase4(store, [FRIDAY], {}, trading_day=FRIDAY)]
        assert "money_path_chain_verified" in names


class TestTheThreeReadingsAreDistinct:
    def test_an_intact_chain_is_met_and_names_the_record_count(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, _Verification("ok", "40 records verified", _chain(links=3)))
        clause = _clause_money_path_chain_verified(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "3 money-path record(s) verified" in clause.detail
        assert len(clause.evidence) == 3

    def test_a_broken_chain_is_unmet_and_names_the_index_and_both_digests(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator reading the gate must not have to re-run anything to
        know what broke — "the chain is broken" is not an actionable finding,
        "record 2 claims X and the store holds Y" is."""
        records = _chain(links=4, break_at=2)
        _install(monkeypatch, _Verification("failed", "link 2 does not verify", records))
        clause = _clause_money_path_chain_verified(store)
        assert not clause.met
        assert not clause.unmeasurable, "a chain we READ and refuted is a finding, not a gap"
        assert "record 2" in clause.detail
        assert records[2].prev_sha256 in clause.detail
        assert "f" * 64 in clause.detail

    def test_an_unlinked_manifest_is_named_even_with_every_link_intact(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            _Verification(
                "failed",
                "1 money-path manifest carries no link",
                _chain(links=2),
                unlinked=("runs/promote/2026-08-29/run.json",),
            ),
        )
        clause = _clause_money_path_chain_verified(store)
        assert not clause.met
        assert "runs/promote/2026-08-29/run.json" in clause.detail

    def test_an_empty_chain_is_unmet_rather_than_inheriting_ok(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`verify_money_path_chain` grades an empty chain `ok`, with a reason
        saying it is empty. True, and not evidence the control works — by phase
        4 the money path IS written, so nothing in it is the finding. The
        clause decides this explicitly instead of reading `.ok`."""
        _install(monkeypatch, _Verification("ok", "the chain is empty"))
        clause = _clause_money_path_chain_verified(store)
        assert not clause.met, clause.detail
        assert not clause.unmeasurable
        assert "empty" in clause.detail

    def test_a_store_that_cannot_be_walked_is_unmeasurable(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The containment wrapper's job, asserted at this clause rather than
        assumed: a verifier that raises is a reading we could not take, and it
        must not darken the other phase-4 clauses."""

        def _raise(_: object) -> _Verification:
            raise PermissionError("AccessDenied")

        monkeypatch.setattr(explain_module, "verify_money_path_chain", _raise, raising=False)
        clause = _clause_money_path_chain_verified(store)
        assert clause.unmeasurable and not clause.met
        assert "AccessDenied" in clause.detail
