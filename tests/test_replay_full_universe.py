"""`replay.full_universe` (alpha-engine-config-I12519): the producer of the record
nous-ergon-ops' promote soak gate reads for its scale clause."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import crucible.cli as cli
from crucible.keys import manifest_key
from crucible.replay import (
    REPLAY_STORE_ENV,
    ReplayStoreError,
    ReplayUnboundError,
    resolve_replay_store,
)
from crucible.runner import RELEASE_MANIFEST_ENV, RunContext, run_job
from crucible.store import LocalStore

DAY = "2026-11-02"
DIGEST = "ab" * 32
#: The exact key nous-ergon-ops `scripts/promote_soak_gate.py` reads for digest D:
#: `run_key("replay.full_universe", day, D)`.
GATE_KEY = f"runs/replay.full_universe/{DAY}/{DIGEST}/run.json"


Compile = Callable[[Sequence[str]], int]


def _fake_compile(status: str = "ok", code: int = 0) -> tuple[Compile, list[list[str]]]:
    calls: list[list[str]] = []

    def inner(argv: Sequence[str]) -> int:
        argv = list(argv)
        calls.append(argv)
        store = LocalStore(Path(argv[argv.index("--store") + 1]))

        def body(ctx: RunContext) -> None:
            ctx.record_rows(rows_in=5200, rows_out=5100)
            if status != "ok":
                raise RuntimeError("compile broke")

        try:
            run_job(
                "data.daily",
                body,
                store=store,
                trading_day=dt.date.fromisoformat(DAY),
                run_mode="replay",
            )
        except RuntimeError:
            return 1
        return code

    return inner, calls


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.delenv(REPLAY_STORE_ENV, raising=False)
    monkeypatch.setenv(RELEASE_MANIFEST_ENV, DIGEST)
    return tmp_path / "run", tmp_path / "replay"


def _replay(run: Path, replay: Path | None, monkeypatch: pytest.MonkeyPatch, inner: Compile) -> int:
    outer = cli.main
    monkeypatch.setattr(cli, "main", inner)
    argv = ["replay.full_universe", "--date", DAY, "--run-mode", "replay", "--store", str(run)]
    if replay is not None:
        argv += ["--replay-store", str(replay)]
    return outer(argv)


def test_ok_writes_the_gates_key_bound_to_the_digest_and_nothing_else_in_the_run_store(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, replay = stores
    inner, calls = _fake_compile()
    assert _replay(run, replay, monkeypatch, inner) == 0
    rec = json.loads((run / GATE_KEY).read_text())
    assert rec["job"] == "replay.full_universe"
    assert rec["status"] == "ok"
    assert rec["release_manifest"] == DIGEST
    assert rec["rows_out"] == 5100
    assert [i["key"] for i in rec["inputs"]] == [manifest_key("data.daily", DAY)]
    # The compile went to the replay store, with the day pinned.
    assert calls == [["data.daily", "--date", DAY, "--store", str(replay), "--run-mode", "replay"]]
    assert (replay / manifest_key("data.daily", DAY)).exists()
    # The run store holds this job's record and NO data.daily record or artifact.
    written = sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file())
    assert not [k for k in written if "data.daily" in k or k.startswith(("data/", "features/"))]
    assert GATE_KEY in written


def test_a_failed_compile_files_a_failed_record_the_gate_reads_as_red(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, replay = stores
    inner, _ = _fake_compile(status="failed")
    with pytest.raises(RuntimeError, match="exited 1"):
        _replay(run, replay, monkeypatch, inner)
    rec = json.loads((run / GATE_KEY).read_text())
    assert rec["status"] == "failed"
    assert "exited 1" in rec["reason"]


def test_a_compile_that_raises_files_a_failed_record_over_an_earlier_ok(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex finding 85ceac0e: `cli.main` re-raises a job's exception, so a
    compile that raises must still file this digest's `failed` record rather
    than leave an earlier `ok` one for the day standing."""
    run, replay = stores
    real = cli.main
    ok, _ = _fake_compile()
    assert _replay(run, replay, monkeypatch, ok) == 0
    monkeypatch.setattr(cli, "main", real)  # `_replay` left the fake installed

    def raising(argv: Sequence[str]) -> int:
        raise RuntimeError("compile broke")

    with pytest.raises(RuntimeError, match="compile broke"):
        _replay(run, replay, monkeypatch, raising)
    rec = json.loads((run / GATE_KEY).read_text())
    assert rec["status"] == "failed"
    assert "compile broke" in rec["reason"]


def test_an_inner_record_bound_to_another_release_is_not_this_replays_evidence(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex finding e1fd6859: a same-day compile for another release that
    replaced the inner record is never read as this replay's evidence."""
    run, replay = stores
    inner, _ = _fake_compile()

    def swapped(argv: Sequence[str]) -> int:
        with monkeypatch.context() as m:
            m.setenv(RELEASE_MANIFEST_ENV, "cd" * 32)
            return inner(argv)

    with pytest.raises(RuntimeError, match="another compile replaced it"):
        _replay(run, replay, monkeypatch, swapped)
    assert json.loads((run / GATE_KEY).read_text())["status"] == "failed"


def test_the_env_var_supplies_the_replay_store(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, replay = stores
    monkeypatch.setenv(REPLAY_STORE_ENV, str(replay))
    inner, calls = _fake_compile()
    assert _replay(run, None, monkeypatch, inner) == 0
    assert calls[0][calls[0].index("--store") + 1] == str(replay)


def test_no_replay_store_is_refused_before_any_compile(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = stores
    inner, calls = _fake_compile()
    with pytest.raises(ReplayStoreError, match="needs a replay store"):
        _replay(run, None, monkeypatch, inner)
    assert calls == []
    assert not (run / GATE_KEY).exists()


@pytest.mark.parametrize(
    ("run_uri", "replay_uri"),
    [
        ("s3://b/crucible", "s3://b/crucible"),
        ("s3://b/crucible", "s3://b/crucible/"),
        ("s3://b/crucible", "s3://b/crucible/replay"),
        ("s3://b/crucible/x", "s3://b/crucible"),
    ],
)
def test_an_overlapping_replay_store_is_refused(run_uri: str, replay_uri: str) -> None:
    with pytest.raises(ReplayStoreError, match="overlaps"):
        resolve_replay_store(run_uri, replay_uri)


def test_the_dispatchers_sibling_prefix_does_not_overlap() -> None:
    # `crucible-replay` shares a leading string with `crucible` but not a path segment.
    assert (
        resolve_replay_store("s3://b/crucible", "s3://b/crucible-replay")
        == "s3://b/crucible-replay"
    )


def test_an_unbound_replay_is_refused_before_any_compile(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, replay = stores
    monkeypatch.delenv(RELEASE_MANIFEST_ENV)
    inner, calls = _fake_compile()
    with pytest.raises(ReplayUnboundError):
        _replay(run, replay, monkeypatch, inner)
    assert calls == []
    assert not run.exists() or not any(p.is_file() for p in run.rglob("*"))


def test_dry_run_writes_nothing_in_either_store(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, replay = stores
    seen: list[list[str]] = []

    def inner(argv: Sequence[str]) -> int:
        seen.append(list(argv))
        return 0

    outer = cli.main
    monkeypatch.setattr(cli, "main", inner)
    argv = ["replay.full_universe", "--date", DAY, "--run-mode", "replay", "--dry-run"]
    assert outer([*argv, "--store", str(run), "--replay-store", str(replay)]) == 0
    assert seen and seen[0][-1] == "--dry-run"
    for root in (run, replay):
        assert not root.exists() or not any(p.is_file() for p in root.rglob("*")), root
