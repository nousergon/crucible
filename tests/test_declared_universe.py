"""The declared universe a scheduled data job grades coverage against.

Measured 2026-09-04 on the first v2 spot box that reached the runner: every
scheduled `data.daily` failed with `UndeclaredUniverseError`, because the
dispatcher passes no `--symbols` and nothing else supplied the denominator.
`crucible.data.universe` is the fix; every test here was seen failing before
it existed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from crucible.config import settings
from crucible.data import run_daily
from crucible.data.daily import UndeclaredUniverseError
from crucible.data.universe import (
    DECLARED_UNIVERSE_SCHEMA_VERSION,
    MalformedUniverseError,
    load_declared_universe,
    universe_from_argv,
)
from crucible.keys import declared_universe_key
from crucible.manifest import read_manifest
from crucible.runner import run_job
from crucible.track_a import _declared_universe, _expected_symbols


def _write(path: Path, document: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


class TestTheDocumentShapes:
    def test_a_membership_document_resolves(self, tmp_path) -> None:
        uri = _write(tmp_path / "m" / "members.json", {"tickers": ["bbb", "AAA", " ccc "]})
        declared = load_declared_universe(uri, origin="argument")
        assert declared.symbols == ("AAA", "BBB", "CCC")
        assert declared.source_uri == uri
        assert len(declared.source_sha256) == 64
        assert declared.origin == "argument"

    def test_a_pointer_is_followed_exactly_once(self, tmp_path) -> None:
        """The live constituents artifact is a pointer (`s3_prefix`) beside a
        dated directory holding `constituents.json`."""
        _write(
            tmp_path / "market_data" / "weekly" / "2026-09-03" / "constituents.json",
            {"date": "2026-09-03", "tickers": ["NVDA", "AAPL"]},
        )
        pointer = _write(
            tmp_path / "market_data" / "latest_weekly.json",
            {"date": "2026-09-03", "s3_prefix": "market_data/weekly/2026-09-03/"},
        )
        declared = load_declared_universe(pointer, origin="environ:CRUCIBLE_UNIVERSE_URI")
        assert declared.symbols == ("AAPL", "NVDA")
        assert declared.source_uri.endswith("market_data/weekly/2026-09-03/constituents.json")

    def test_a_pointer_to_a_pointer_is_refused(self, tmp_path) -> None:
        _write(
            tmp_path / "market_data" / "weekly" / "x" / "constituents.json",
            {"s3_prefix": "market_data/weekly/y/"},
        )
        pointer = _write(
            tmp_path / "market_data" / "latest_weekly.json",
            {"s3_prefix": "market_data/weekly/x/"},
        )
        with pytest.raises(MalformedUniverseError, match="pointer to a pointer"):
            load_declared_universe(pointer, origin="argument")

    def test_an_empty_membership_is_refused_not_read_as_no_universe(self, tmp_path) -> None:
        """An empty list must not degrade to `None`: `run_daily` would then
        raise the 'nothing declared' error and the fix it names — pass
        --symbols — is the wrong fix for a document that exists and is empty."""
        uri = _write(tmp_path / "m" / "members.json", {"tickers": []})
        with pytest.raises(MalformedUniverseError, match="non-empty list"):
            load_declared_universe(uri, origin="argument")

    def test_a_document_that_is_neither_shape_is_refused(self, tmp_path) -> None:
        uri = _write(tmp_path / "m" / "members.json", {"date": "2026-09-03"})
        with pytest.raises(MalformedUniverseError, match="neither a membership document"):
            load_declared_universe(uri, origin="argument")

    def test_non_json_is_refused_by_name(self, tmp_path) -> None:
        path = tmp_path / "m" / "members.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(MalformedUniverseError, match="not JSON"):
            load_declared_universe(str(path), origin="argument")

    def test_duplicate_spellings_are_refused(self, tmp_path) -> None:
        uri = _write(tmp_path / "m" / "members.json", {"tickers": ["aapl", "AAPL"]})
        with pytest.raises(MalformedUniverseError, match="duplicate"):
            load_declared_universe(uri, origin="argument")

    def test_an_empty_uri_resolves_nothing(self) -> None:
        with pytest.raises(MalformedUniverseError, match="empty universe URI"):
            load_declared_universe("", origin="default")

    def test_the_fetch_seam_is_used_for_both_hops(self) -> None:
        """An s3:// URI never reaches boto3 in this test; the seam sees the
        pointer and then the sibling it names, in the same bucket."""
        seen: list[str] = []

        def fake_read(uri: str) -> bytes:
            seen.append(uri)
            if uri.endswith("latest_weekly.json"):
                return json.dumps({"s3_prefix": "market_data/weekly/2026-09-03/"}).encode()
            return json.dumps({"tickers": ["MSFT"]}).encode()

        declared = load_declared_universe(
            "s3://some-bucket/market_data/latest_weekly.json",
            origin="environ:CRUCIBLE_UNIVERSE_URI",
            read=fake_read,
        )
        assert seen == [
            "s3://some-bucket/market_data/latest_weekly.json",
            "s3://some-bucket/market_data/weekly/2026-09-03/constituents.json",
        ]
        assert declared.symbols == ("MSFT",)


class TestTheArgvForm:
    def test_symbols_are_normalised_the_same_way(self) -> None:
        declared = universe_from_argv("bbb, AAA ,,ccc")
        assert declared.symbols == ("AAA", "BBB", "CCC")
        assert declared.source_uri == "argv:--symbols"
        assert declared.origin == "argument"

    def test_an_empty_argv_is_refused(self) -> None:
        with pytest.raises(MalformedUniverseError):
            universe_from_argv(" , ")


class TestResolutionOrder:
    def test_argv_wins_over_the_environment(self, tmp_path, monkeypatch) -> None:
        uri = _write(tmp_path / "m" / "members.json", {"tickers": ["ZZZ"]})
        monkeypatch.setenv("CRUCIBLE_UNIVERSE_URI", uri)
        config = settings()
        args = argparse.Namespace(symbols="AAA")
        assert _declared_universe(args, config).symbols == ("AAA",)

    def test_the_environment_is_used_when_argv_is_absent(self, tmp_path, monkeypatch) -> None:
        uri = _write(tmp_path / "m" / "members.json", {"tickers": ["ZZZ"]})
        monkeypatch.setenv("CRUCIBLE_UNIVERSE_URI", uri)
        config = settings()
        assert config.origins["universe_uri"] == "environ:CRUCIBLE_UNIVERSE_URI"
        declared = _declared_universe(argparse.Namespace(symbols=None), config)
        assert declared.symbols == ("ZZZ",)
        assert declared.origin == "environ:CRUCIBLE_UNIVERSE_URI"

    def test_nothing_declared_is_none_and_the_job_still_refuses(
        self, monkeypatch, store, source, cycle_date
    ) -> None:
        """The scheduled-box failure of 2026-09-04, reproduced: no argv, no
        environment. The refusal is `run_daily`'s own, and it still fires."""
        monkeypatch.delenv("CRUCIBLE_UNIVERSE_URI", raising=False)
        config = settings()
        assert config.universe_uri == ""
        declared = _declared_universe(argparse.Namespace(symbols=None), config)
        assert declared is None
        with pytest.raises(UndeclaredUniverseError, match="no --symbols"):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c, source=source, expected_symbols=_expected_symbols(declared, c)
                ),
                store=store,
                trading_day=cycle_date,
            )

    def test_a_malformed_document_fails_before_the_source_is_read(
        self, tmp_path, monkeypatch
    ) -> None:
        uri = _write(tmp_path / "m" / "members.json", {"tickers": "AAPL"})
        monkeypatch.setenv("CRUCIBLE_UNIVERSE_URI", uri)
        with pytest.raises(MalformedUniverseError):
            _declared_universe(argparse.Namespace(symbols=None), settings())


class TestTheRunRecordsWhatItGradedAgainst:
    def test_the_universe_is_copied_beside_the_run_and_listed_as_an_output(
        self, tmp_path, store, frames, cycle_date
    ) -> None:
        from crucible.data import FramePriceSource

        uri = _write(tmp_path / "m" / "members.json", {"tickers": sorted(frames)})
        declared = load_declared_universe(uri, origin="environ:CRUCIBLE_UNIVERSE_URI")
        ctx = run_job(
            "data.daily",
            lambda c: run_daily(
                c,
                source=FramePriceSource(frames),
                expected_symbols=_expected_symbols(declared, c),
            ),
            store=store,
            trading_day=cycle_date,
        )
        key = declared_universe_key(cycle_date.isoformat())
        assert key == f"universe/declared/{cycle_date.isoformat()}/members.json"
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        assert manifest["status"] == "ok"
        recorded = [o for o in manifest["outputs"] if o["key"] == key]
        assert recorded, "the declared universe must be an output of the run that used it"
        assert recorded[0]["schema_version"] == DECLARED_UNIVERSE_SCHEMA_VERSION
        copy = json.loads(store.get_bytes(key))
        assert copy["symbols"] == sorted(frames)
        assert copy["source_uri"] == uri
        assert copy["source_sha256"] == declared.source_sha256
        assert copy["origin"] == "environ:CRUCIBLE_UNIVERSE_URI"
        assert copy["count"] == len(frames)
        assert ctx.run_id

    def test_the_declared_key_is_not_the_champion_feed(self, cycle_date) -> None:
        from crucible.keys import universe_members_key

        day = cycle_date.isoformat()
        assert declared_universe_key(day) != universe_members_key(day)
