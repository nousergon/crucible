"""`crucible gate --every-gate` — every registered gate in one process.

`alpha-engine-config-I12793`. `gate-close.yml`'s publish job spawned one
`crucible gate --gate <phase>` process per gate, and each rebuilt the ladder by
re-evaluating every OTHER gate: six gates cost 36 evaluations, and on
2026-11-06 the job was cancelled at its 30-minute timeout. `--every-gate`
evaluates each gate once and hands the readings to every ladder.
"""

from __future__ import annotations

import json
from collections import Counter
from unittest import mock

import pytest

from crucible import track_f
from crucible.cli import USAGE_EXIT_CODE, main
from crucible.gate import GATE_DELIVERABLES, GATES, Clause
from crucible.store import LocalStore

DAY = "2026-08-28"


def _fake_gates(*, met: dict[str, bool] | None = None, raises: tuple[str, ...] = ()):
    """Every registered gate replaced by a one-clause stub, so the test reads
    no real clause and counts exactly how often each gate is evaluated."""
    met = met or {}
    calls: Counter[str] = Counter()
    gates = {}
    for name, (weeks, _fn) in GATES.items():

        def clauses(*_a, _name=name, **_k) -> list[Clause]:
            calls[_name] += 1
            if _name in raises:
                raise RuntimeError(f"synthetic store outage reading {_name}")
            ok = met.get(_name, False)
            return [Clause("c", "req", ok, "met" if ok else "unmet", ())]

        gates[name] = (weeks, clauses)
    return gates, calls


def _run(tmp_path, *extra: str) -> int:
    return main(["gate", "--store", str(tmp_path), "--date", DAY, *extra])


class TestEveryGateEvaluatesEachGateOnce:
    def test_each_gate_is_evaluated_exactly_once_and_every_reading_is_published(
        self, tmp_path
    ) -> None:
        gates, calls = _fake_gates()
        with (
            mock.patch.dict(GATES, gates),
            mock.patch.dict(GATE_DELIVERABLES, {name: () for name in gates}),
        ):
            code = _run(tmp_path, "--every-gate", "--publish")

        assert code == 1  # every stub reads UNMET: the ordinary day
        assert set(calls) == set(GATES)
        assert all(n == 1 for n in calls.values()), calls
        store = LocalStore(tmp_path)
        for name in GATES:
            assert store.exists(f"gates/{name}/{DAY}/gate.json"), name
        ladder = json.loads(store.get_bytes("gates/ladder.json"))
        assert {row["gate"] for row in ladder["phases"] if row.get("gate")} >= set(GATES)

    def test_the_one_gate_form_still_reads_one_gate_and_its_ladder(self, tmp_path) -> None:
        gates, calls = _fake_gates()
        with (
            mock.patch.dict(GATES, gates),
            mock.patch.dict(GATE_DELIVERABLES, {name: () for name in gates}),
        ):
            code = _run(tmp_path, "--gate", "phase1", "--publish")

        assert code == 1
        assert calls["phase1"] == 1
        store = LocalStore(tmp_path)
        assert store.exists(f"gates/phase1/{DAY}/gate.json")
        assert not store.exists(f"gates/phase0/{DAY}/gate.json")

    def test_exits_zero_only_when_every_gate_is_met(self, tmp_path) -> None:
        gates, _ = _fake_gates(met={name: True for name in GATES})
        with (
            mock.patch.dict(GATES, gates),
            mock.patch.dict(GATE_DELIVERABLES, {name: () for name in gates}),
        ):
            assert _run(tmp_path, "--every-gate") == 0


class TestAFailedGateDoesNotStopTheOthers:
    def test_a_raising_gate_is_named_and_the_rest_still_publish(self, tmp_path, capsys) -> None:
        names = sorted(GATES)
        broken = names[0]
        gates, _ = _fake_gates(raises=(broken,))
        with (
            mock.patch.dict(GATES, gates),
            mock.patch.dict(GATE_DELIVERABLES, {name: () for name in gates}),
        ):
            code = _run(tmp_path, "--every-gate", "--publish")

        assert code == track_f.EVERY_GATE_MEASUREMENT_FAILED_EXIT
        assert code > 1  # the workflow's "failed to MEASURE" threshold
        assert f"could not be measured: {broken}" in capsys.readouterr().err
        store = LocalStore(tmp_path)
        assert not store.exists(f"gates/{broken}/{DAY}/gate.json")
        for name in names[1:]:
            assert store.exists(f"gates/{name}/{DAY}/gate.json"), name

    def test_an_empty_registry_is_refused_not_read_as_a_pass(self, capsys) -> None:
        import argparse

        with mock.patch.object(track_f, "gate_names", return_value=[]):
            code = track_f.gate_handler(argparse.Namespace(every_gate=True))
        assert code == track_f.EVERY_GATE_MEASUREMENT_FAILED_EXIT
        assert "would publish nothing" in capsys.readouterr().err


class TestUsage:
    @pytest.mark.parametrize(
        "extra",
        [
            (),
            ("--gate", "phase1", "--every-gate"),
            ("--every-gate", "--weeks", "2"),
            ("--every-gate", "--closing-comment"),
        ],
    )
    def test_refused_as_a_usage_error(self, tmp_path, extra) -> None:
        assert _run(tmp_path, *extra) == USAGE_EXIT_CODE
