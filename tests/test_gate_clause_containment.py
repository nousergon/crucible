"""No gate clause may raise into its caller (`alpha-engine-config-I10328`).

**Measured 2026-09-09 on the v2 box.** `crucible-PR172` gave
`_clause_zero_human_mutating_calls` a `DescribeStacks` call for the stack's
`LastUpdatedTime`. Client CONSTRUCTION — `boto3.client("cloudformation")` —
raises `NoRegionError` where no region is configured, and the box shell exports
none, so the construction sat outside the try block the read itself had:

```
runs/weekly/2026-08-07/run.json -> status: failed
ArcStageFailed: weekly arc stage console raised for 2026-08-07:
  NoRegionError: You must specify a region.
```

The `console` arc stage renders `gates/ladder.json`, which evaluates every
phase gate, so ONE unguarded line in ONE clause failed every `weekly` arc on
the box — and thereby regressed phase 1's `arc_runs_ok`, turning a gate READ
into a system FAILURE.

This is `alpha-engine-config-I9869`'s class recurring: "phase-1 gate clauses
still read manifests unguarded — one malformed run.json darkens the ladder",
fixed then for manifest reads, clause by clause. The next clause to reach a new
AWS service reintroduced it, which is the evidence that per-clause guarding is
a habit rather than a rule. So the containment lives in ONE place —
`crucible.gate._contained`, applied to every `_clause_*` function by
`_contain_clause_exceptions` — and this module tests the CLASS: not "the stack
read is guarded now", but "a clause cannot raise, whatever it reaches for".

Nothing here relies on a region being configured. The region export lands in
`nous-ergon-ops`, and the whole point is that the clause is correct where none
is.
"""

from __future__ import annotations

import datetime as dt

import pytest

import crucible.autonomy as autonomy_module
import crucible.gate as gate_module
from crucible.gate import (
    CLAUSE_FUNCTION_PREFIX,
    Clause,
    ClauseMisconfiguredError,
    _contained,
)
from crucible.store import LocalStore, S3Store

DAY = dt.date(2026, 8, 28)

#: The release pointer's flip instant, long enough before `DAY` that the window
#: guards clear and the STACK read is the thing under test. Without a readable
#: pointer, `_last_system_change` refuses before it ever reaches the stack, and
#: a test using a `LocalStore` here would pass for the wrong reason — measured:
#: it did, until this fixture replaced it.
POINTER_FLIPPED_AT = dt.datetime(2026, 8, 10, 9, 0, tzinfo=dt.UTC)


class _HeadOnlyS3:
    """An S3 client answering `head_object` (the pointer's flip instant) and
    nothing else. A real `S3Store` with a substituted client, so the key the
    head is taken against is part of what is exercised."""

    def head_object(self, **_: object) -> dict:
        return {"LastModified": POINTER_FLIPPED_AT}

    def get_paginator(self, name: str) -> object:
        raise AssertionError(f"no listing is expected in these tests, got {name!r}")


def _store_with_a_readable_pointer() -> S3Store:
    return S3Store("a-test-store", "crucible", client=_HeadOnlyS3())


class TestEveryClauseInTheModuleIsContained:
    """The rule, asserted over the whole module rather than over the clauses
    somebody remembered. A clause added tomorrow is covered by this without
    anybody editing this file — which is the difference between a rule and a
    habit."""

    @staticmethod
    def _clause_functions() -> dict[str, object]:
        return {
            name: value
            for name, value in vars(gate_module).items()
            if name.startswith(CLAUSE_FUNCTION_PREFIX) and callable(value)
        }

    def test_the_module_has_clauses_to_contain(self) -> None:
        """A containment test that matched nothing would be a guard grading an
        empty set."""
        assert len(self._clause_functions()) > 20

    def test_every_clause_function_is_wrapped(self) -> None:
        unwrapped = sorted(
            name
            for name, fn in self._clause_functions().items()
            if not getattr(fn, "_contained", False)
        )
        assert not unwrapped, (
            "these clause functions can raise into the ladder: "
            f"{unwrapped}. Containment is applied by walking this module's globals for "
            f"the {CLAUSE_FUNCTION_PREFIX!r} prefix, so a clause defined BELOW that pass "
            "would escape it."
        )

    def test_the_containment_pass_refuses_to_match_nothing(self) -> None:
        """The pass runs at import, above `GATES` and below every clause. If it
        ever ran too early it would wrap nothing and every clause would be able
        to raise again — so it raises rather than passing silently."""
        with pytest.raises(RuntimeError, match="wrapped 0 functions"):
            gate_module._contain_clause_exceptions()


class TestAClauseThatRaisesReadsUnmeasurable:
    def test_an_arbitrary_exception_becomes_an_unmeasurable_clause(self) -> None:
        def _clause_a_thing_we_cannot_read(store) -> Clause:
            raise RuntimeError("the archive fell over")

        clause = _contained(_clause_a_thing_we_cannot_read)(LocalStore("/tmp"))
        assert clause.unmeasurable and not clause.met
        assert clause.name == "a_thing_we_cannot_read"
        assert "RuntimeError: the archive fell over" in clause.detail

    def test_the_clause_name_survives_the_wrapper(self) -> None:
        """The reading has to be attributable to a clause, or the ladder shows
        a row nobody can trace (principle 1). Derived from the function's own
        name, so it cannot be spelled differently from the clause it replaces.
        """

        def _clause_pages_within_ceiling(store) -> Clause:
            raise ValueError("nope")

        assert _contained(_clause_pages_within_ceiling)(None).name == "pages_within_ceiling"

    def test_an_unmeasurable_clause_is_never_counted_as_met(self) -> None:
        """`met=False` always: `met_ratio` counts `met`, and a contained failure
        painted green would be *no data* rendered as a pass."""

        def _clause_whatever(store) -> Clause:
            raise RuntimeError("x")

        clause = _contained(_clause_whatever)(None)
        assert clause.met is False

    def test_a_keyboard_interrupt_is_not_contained(self) -> None:
        """`Exception`, not `BaseException`: a KeyboardInterrupt or a spot
        reclamation must still stop the process, or a job would keep grading
        clauses on an instance that is going away."""

        def _clause_interrupted(store) -> Clause:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _contained(_clause_interrupted)(None)

    def test_a_misconfigured_clause_still_raises(self) -> None:
        """The one carve-out, and it is about the CALLER being wrong rather than
        the environment: a clause called with arguments that cannot describe any
        system is a bug in this module's own clause list, and rendering it as an
        UNMEASURABLE row would put our defect behind a message that reads like
        an AWS problem, every week, forever."""

        def _clause_asked_the_impossible(store) -> Clause:
            raise ClauseMisconfiguredError("minimum 1 exceeds maximum 0")

        with pytest.raises(ClauseMisconfiguredError):
            _contained(_clause_asked_the_impossible)(None)

    def test_a_clause_that_returns_normally_is_untouched(self) -> None:
        expected = Clause("a_name", "a requirement", True, "a detail")

        def _clause_a_name(store) -> Clause:
            return expected

        assert _contained(_clause_a_name)(None) is expected


class TestTheStackReadThatCausedThis:
    """The instance, kept alongside the class. `alpha-engine-config-I10328`:
    the failure was at CLIENT CONSTRUCTION, outside the try block guarding the
    call — so a test that only made `describe_stacks` fail would have passed
    against the broken code."""

    @staticmethod
    def _no_region(monkeypatch: pytest.MonkeyPatch) -> None:
        """`_cfn_client` raising the way boto3 does with no region configured.

        Substituted rather than reproduced by unsetting environment variables:
        a laptop, a CI runner and the box each carry a different region
        posture, and a test whose subject is "whatever this machine happens to
        have configured" is testing the machine.
        """

        def _raise() -> object:
            from botocore.exceptions import NoRegionError

            raise NoRegionError()

        monkeypatch.setattr(autonomy_module, "_cfn_client", _raise)

    def test_the_stack_read_reports_unreadable_rather_than_no_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._no_region(monkeypatch)
        with pytest.raises(gate_module.LastChangeUnreadableError, match="NoRegionError"):
            gate_module._stack_last_updated()

    def test_the_autonomy_clause_reads_unmeasurable_with_no_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The clause a box actually renders. UNMEASURABLE, not a raise and not
        a window starting at zero.

        The pointer is READABLE here on purpose: `_last_system_change` reads it
        first and refuses on a `LocalStore` before the stack read is reached at
        all, so a `LocalStore` version of this test passes whether the stack
        read is guarded or not."""
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://a-test-archive/trail")
        self._no_region(monkeypatch)
        clause = gate_module._clause_zero_human_mutating_calls(
            _store_with_a_readable_pointer(), [DAY]
        )
        assert clause.unmeasurable and not clause.met
        assert "NoRegionError" in clause.detail

    def test_a_whole_gate_still_renders_when_one_clause_cannot_read_its_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reading that was lost: `gates/ladder.json` evaluates every phase
        gate, so one clause raising took the weekly arc's `console` stage down
        with it. Every other clause must still produce a row."""
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://a-test-archive/trail")
        self._no_region(monkeypatch)
        result = gate_module.evaluate(
            _store_with_a_readable_pointer(), gate="phase2", trading_day=DAY
        )
        assert len(result.clauses) > 5
        by_name = {c.name: c for c in result.clauses}
        assert by_name["zero_human_mutating_calls"].unmeasurable
        assert "NoRegionError" in by_name["zero_human_mutating_calls"].detail
