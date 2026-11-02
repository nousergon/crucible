"""The v2 weekly refuses a universe pointer that names another week, and
this repo never becomes a second writer of that pointer.

nous-ergon-ops-I2018 / alpha-engine-config-I13065. `market_data/latest_weekly.json`
has ONE writer: `nousergon-data`'s weekly Phase 1
(`weekly_collector._write_manifest`, the `ne-data-collection-weekly` schedule).
The v2 `data.weekly` reads it on an independent schedule, with nothing
ordering the two. On 2026-10-24 the pointer was rewritten 334 s before the v2
weekly started, and on 2026-10-31 about ten minutes before; the dates matched
only because the collector happened to finish first. A late or failed
collector would have left last week's pointer in place and the weekly would
have graded last week's universe, reporting nothing wrong.

Two things are pinned here:

1. **The reader verifies.** `data.weekly` passes its own trading day, and a
   pointer dated anything else (or undated) raises
   `StaleUniversePointerError` naming both dates, before the pointer's target
   is fetched and before the store, the price source or the point-in-time
   source is opened. `data.daily` is unchanged: between Saturdays it
   legitimately reads last Friday's pointer.
2. **No second writer.** No string literal in the `crucible` package names
   the pointer key, so no code path here can write it. A literal naming it
   appearing anywhere outside prose fails this module.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest

from crucible.cli import main
from crucible.config import settings
from crucible.data.universe import (
    MalformedUniverseError,
    StaleUniversePointerError,
    load_declared_universe,
)
from crucible.track_a import _declared_universe

REPO_ROOT = Path(__file__).resolve().parent.parent
CRUCIBLE_SRC = REPO_ROOT / "crucible"
POINTER_KEY_FRAGMENT = "latest_weekly"

THIS_WEEK = "2026-10-30"
LAST_WEEK = "2026-10-23"


def _layout(root: Path, *, pointer_date: str | None, target_week: str = LAST_WEEK) -> str:
    """The live layout: a dated `constituents.json` and a pointer beside it."""
    target = root / "market_data" / "weekly" / target_week / "constituents.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"date": target_week, "tickers": ["AAPL", "NVDA"]}), "utf-8")
    pointer: dict[str, str] = {"s3_prefix": f"market_data/weekly/{target_week}/"}
    if pointer_date is not None:
        pointer["date"] = pointer_date
    path = root / "market_data" / "latest_weekly.json"
    path.write_text(json.dumps(pointer), "utf-8")
    return str(path)


class TestTheLoaderChecksThePointersWeek:
    def test_a_pointer_dated_a_week_earlier_is_refused_by_name(self, tmp_path) -> None:
        pointer = _layout(tmp_path, pointer_date=LAST_WEEK)
        with pytest.raises(StaleUniversePointerError) as caught:
            load_declared_universe(pointer, origin="argument", expected_date=THIS_WEEK)
        assert (caught.value.pointer_date, caught.value.expected_date) == (LAST_WEEK, THIS_WEEK)
        assert LAST_WEEK in str(caught.value) and THIS_WEEK in str(caught.value)
        assert pointer in str(caught.value)
        # A stale pointer must fail the run the way a malformed one does.
        assert isinstance(caught.value, MalformedUniverseError)

    def test_the_target_is_never_fetched_when_the_pointer_is_stale(self, tmp_path) -> None:
        pointer = _layout(tmp_path, pointer_date=LAST_WEEK)
        fetched: list[str] = []

        def read(uri: str) -> bytes:
            fetched.append(uri)
            return Path(uri).read_bytes()

        with pytest.raises(StaleUniversePointerError):
            load_declared_universe(pointer, origin="argument", read=read, expected_date=THIS_WEEK)
        assert fetched == [pointer]

    def test_an_undated_pointer_is_refused_when_a_week_is_expected(self, tmp_path) -> None:
        pointer = _layout(tmp_path, pointer_date=None)
        with pytest.raises(StaleUniversePointerError, match="None"):
            load_declared_universe(pointer, origin="argument", expected_date=THIS_WEEK)

    def test_this_weeks_pointer_is_followed(self, tmp_path) -> None:
        pointer = _layout(tmp_path, pointer_date=THIS_WEEK, target_week=THIS_WEEK)
        declared = load_declared_universe(pointer, origin="argument", expected_date=THIS_WEEK)
        assert declared.symbols == ("AAPL", "NVDA")
        assert declared.source_uri.endswith(f"market_data/weekly/{THIS_WEEK}/constituents.json")

    def test_without_an_expected_week_the_daily_behaviour_is_unchanged(self, tmp_path) -> None:
        """`data.daily` on a Wednesday reads last Friday's pointer, correctly."""
        pointer = _layout(tmp_path, pointer_date=LAST_WEEK)
        declared = load_declared_universe(pointer, origin="argument")
        assert declared.symbols == ("AAPL", "NVDA")

    def test_a_membership_document_carries_no_week_and_is_unaffected(self, tmp_path) -> None:
        doc = tmp_path / "members.json"
        doc.write_text(json.dumps({"tickers": ["ZZZ"]}), "utf-8")
        declared = load_declared_universe(str(doc), origin="argument", expected_date=THIS_WEEK)
        assert declared.symbols == ("ZZZ",)


class TestDataWeeklyRefusesBeforeOpeningAnything:
    def test_resolution_passes_the_week_through(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_UNIVERSE_URI", _layout(tmp_path, pointer_date=LAST_WEEK))
        with pytest.raises(StaleUniversePointerError):
            _declared_universe(
                argparse.Namespace(symbols=None), settings(), expected_date=THIS_WEEK
            )

    def test_the_cli_job_refuses_a_last_week_pointer(self, tmp_path, monkeypatch) -> None:
        """The real entry point the arc's first stage calls. The refusal comes
        before the store, the price source or the point-in-time source is
        opened: no bucket is configured here, and none is reached."""
        monkeypatch.setenv("CRUCIBLE_UNIVERSE_URI", _layout(tmp_path, pointer_date=LAST_WEEK))
        monkeypatch.delenv("CRUCIBLE_ARCTIC_BUCKET", raising=False)
        store = tmp_path / "store"
        with pytest.raises(StaleUniversePointerError):
            main(
                [
                    "data.weekly",
                    "--date",
                    THIS_WEEK,
                    "--run-mode",
                    "replay",
                    "--store",
                    str(store),
                ]
            )
        assert not (store / "runs" / "data.weekly").exists()

    def test_data_daily_still_reads_last_fridays_pointer(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_UNIVERSE_URI", _layout(tmp_path, pointer_date=LAST_WEEK))
        declared = _declared_universe(argparse.Namespace(symbols=None), settings())
        assert declared is not None and declared.symbols == ("AAPL", "NVDA")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def pointer_key_literals(source: str) -> list[int]:
    """Line numbers of non-docstring string literals naming the pointer key."""
    tree = ast.parse(source)
    prose = _docstring_nodes(tree)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in prose
        and POINTER_KEY_FRAGMENT in node.value
    )


class TestNoSecondWriter:
    def test_the_package_never_names_the_pointer_key(self) -> None:
        """A writer needs the key; a package that cannot spell it cannot write
        it. The reader is configured by `CRUCIBLE_UNIVERSE_URI` alone."""
        offenders = {
            str(path.relative_to(REPO_ROOT)): lines
            for path in sorted(CRUCIBLE_SRC.rglob("*.py"))
            if (lines := pointer_key_literals(path.read_text(encoding="utf-8")))
        }
        assert offenders == {}, (
            "the weekly universe pointer has one writer (nousergon-data weekly Phase 1); "
            f"these crucible modules name its key outside prose: {offenders}"
        )

    def test_the_scan_sees_a_second_writer(self) -> None:
        """The guard above is only worth its green if it goes red on the
        shape it exists to catch."""
        writer = (
            '"""Prose may mention latest_weekly.json freely."""\n'
            "def advance(store, day):\n"
            '    store.put_bytes(f"market_data/latest_weekly.json", day.encode())\n'
        )
        assert pointer_key_literals(writer) == [3]
