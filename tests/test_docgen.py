"""Unit tests for `crucible.features.docgen` (generator for FEATURE_CATALOG.md).

Offline: every document lives under `tmp_path`; the repository's own catalogue
document is never written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.features import docgen
from crucible.features.docgen import MarkerError, render_document

BEGIN = "<!-- BEGIN GENERATED CATALOG TABLE -->"
END = "<!-- END GENERATED CATALOG TABLE -->"
EMPTY_TABLE_HEADER = (
    "| Column | Unit | Window (sessions) | Cross-sectional | Expression | Inputs |\n"
    "|---|---|---|---|---|---|"
)


def _doc(body: str = "stale row") -> str:
    return f"# Title\n\nintro prose\n\n{BEGIN}\n{body}\n{END}\n\ntrailing prose\n"


class TestRenderDocument:
    def test_replaces_only_the_region_between_the_markers(self) -> None:
        out = render_document(_doc(), catalog=())
        assert out == (
            f"# Title\n\nintro prose\n\n{BEGIN}\n{EMPTY_TABLE_HEADER}\n{END}\n\ntrailing prose\n"
        )

    def test_is_idempotent(self) -> None:
        once = render_document(_doc(), catalog=())
        assert render_document(once, catalog=()) == once

    def test_boundary_adjacent_markers_with_no_body(self) -> None:
        out = render_document(f"{BEGIN}{END}", catalog=())
        assert out == f"{BEGIN}\n{EMPTY_TABLE_HEADER}\n{END}"

    # Mutation guarded: dropping the `end < begin` test (or the `< 0` tests)
    # would make these return a mangled document instead of raising.
    @pytest.mark.parametrize(
        "text",
        [
            "no markers at all",
            f"{BEGIN}\nonly a begin marker\n",
            f"only an end marker\n{END}\n",
            f"{END}\nreversed\n{BEGIN}\n",
        ],
    )
    def test_missing_or_misordered_markers_raise(self, text: str) -> None:
        with pytest.raises(MarkerError, match="declared region"):
            render_document(text, catalog=())

    def test_marker_error_is_a_value_error(self) -> None:
        assert issubclass(MarkerError, ValueError)


class TestMain:
    def test_write_mode_regenerates_the_table_and_keeps_prose(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = tmp_path / "FEATURE_CATALOG.md"
        doc.write_text(_doc(), encoding="utf-8")
        monkeypatch.setattr(docgen, "DOC_PATH", doc)
        assert docgen.main([]) == 0
        text = doc.read_text(encoding="utf-8")
        assert text.startswith("# Title\n\nintro prose\n\n" + BEGIN + "\n| Column |")
        assert text.endswith(END + "\n\ntrailing prose\n")
        assert "stale row" not in text

    def test_check_passes_after_a_write_and_does_not_modify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = tmp_path / "FEATURE_CATALOG.md"
        doc.write_text(_doc(), encoding="utf-8")
        monkeypatch.setattr(docgen, "DOC_PATH", doc)
        docgen.main([])
        written = doc.read_text(encoding="utf-8")
        assert docgen.main(["--check"]) == 0
        assert doc.read_text(encoding="utf-8") == written

    # Mutation guarded: making --check return 0 (or write the file) on drift.
    def test_check_fails_on_a_stale_document_without_writing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doc = tmp_path / "FEATURE_CATALOG.md"
        stale = _doc()
        doc.write_text(stale, encoding="utf-8")
        monkeypatch.setattr(docgen, "DOC_PATH", doc)
        assert docgen.main(["--check"]) == 1
        assert doc.read_text(encoding="utf-8") == stale
        assert "is stale" in capsys.readouterr().err

    def test_a_document_without_markers_raises_and_is_left_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = tmp_path / "FEATURE_CATALOG.md"
        doc.write_text("no markers", encoding="utf-8")
        monkeypatch.setattr(docgen, "DOC_PATH", doc)
        with pytest.raises(MarkerError):
            docgen.main([])
        assert doc.read_text(encoding="utf-8") == "no markers"

    def test_a_missing_document_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docgen, "DOC_PATH", tmp_path / "absent.md")
        with pytest.raises(FileNotFoundError):
            docgen.main(["--check"])
