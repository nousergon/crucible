"""`crucible.keys.RANGE_BOUND_JOBS` is DERIVED, not typed from memory.

`alpha-engine-config-I11048`. Which trading day a dispatch's manifest is
keyed under is a property of the JOB — a range job passes `trading_day=end`
to `crucible.runner.run_job` and binds its one manifest to the end of the
range — and `crucible.alerts._dispatch_target_trading_day` grades every
dispatch against that property.

A membership fact kept true only by a human remembering it is the shape
every detector in this fleet has had to be repaired from, so this test reads
the handlers and derives the set. A third range job added to `track_a.py`, or
one of the two rebound to `--date`, fails HERE rather than silently
re-opening the defect: three of the four absence pages on 2026-09-18 named
runs that had succeeded.
"""

from __future__ import annotations

import ast
import pathlib

from crucible.keys import RANGE_BOUND_JOBS

TRACK_A = pathlib.Path(__file__).resolve().parents[1] / "crucible" / "track_a.py"


def _jobs_binding_their_manifest_to_a_range_end(source: str) -> set[str]:
    """Every `run_job("<job>", ..., trading_day=end, ...)` in ``source``.

    Positional first argument must be a string literal — every `run_job` call
    in `track_a` names its job that way, and a call that did not would be a
    job whose name this file cannot derive, which is a finding rather than
    something to guess at (see the assertion below).
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if not (isinstance(callee, ast.Name) and callee.id == "run_job"):
            continue
        binds_range_end = any(
            keyword.arg == "trading_day"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "end"
            for keyword in node.keywords
        )
        if not binds_range_end:
            continue
        assert node.args and isinstance(node.args[0], ast.Constant), (
            "a run_job call binding trading_day=end names its job through something "
            "other than a string literal; this test cannot derive the set and the "
            "derivation must be repaired rather than the assertion relaxed"
        )
        found.add(node.args[0].value)
    return found


def test_the_declared_set_matches_the_handlers_that_bind_to_a_range_end() -> None:
    derived = _jobs_binding_their_manifest_to_a_range_end(TRACK_A.read_text(encoding="utf-8"))
    assert derived == set(RANGE_BOUND_JOBS), (
        "crucible.keys.RANGE_BOUND_JOBS disagrees with crucible/track_a.py. Declared: "
        f"{sorted(RANGE_BOUND_JOBS)}; handlers passing trading_day=end: {sorted(derived)}. "
        "The declared set is what the absence detector grades a dispatch against — a "
        "range job missing from it is graded against a key nothing writes."
    )


def test_the_derivation_itself_finds_something() -> None:
    """A derivation that silently found nothing would make the test above
    pass by making both sides empty — which is exactly how a guard reads
    green over the gap it exists for."""
    derived = _jobs_binding_their_manifest_to_a_range_end(TRACK_A.read_text(encoding="utf-8"))
    assert derived, "no run_job call in track_a.py binds trading_day=end; the walk is broken"
