"""Regenerate the catalogue table in `docs/FEATURE_CATALOG.md`.

    uv run python -m crucible.features.docgen

The fleet feature-store rule asks for a documentation row per column. A
hand-maintained table is that contract restated in a second place, and a
contract restated in a second place has already drifted once in this fleet.
So the rows are RENDERED from `CATALOG` and the prose around them is written
by hand, separated by the two markers this module rewrites between.

`tests/test_feature_registry_contract.py` asserts the file on disk equals
what this module would write, so forgetting to run it is a red test rather
than a document that quietly stops describing the layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

from crucible.features.registry import (
    CATALOG,
    CATALOG_TABLE_BEGIN,
    CATALOG_TABLE_END,
    FeatureSpec,
    render_catalog_markdown,
)

__all__ = ["DOC_PATH", "main", "render_document"]

#: The repository's catalogue document. Resolved from this file rather than
#: from the working directory, so the generator and the test that checks it
#: cannot disagree about which file they mean.
DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "FEATURE_CATALOG.md"


class MarkerError(ValueError):
    """The document is missing the markers the table lives between.

    Raised rather than appending a second table: a generator that repairs a
    document it cannot parse is how two tables end up in one file, each
    describing a different catalogue.
    """


def render_document(current: str, catalog: tuple[FeatureSpec, ...] = CATALOG) -> str:
    """``current`` with the generated region replaced. Prose is preserved."""
    begin = current.find(CATALOG_TABLE_BEGIN)
    end = current.find(CATALOG_TABLE_END)
    if begin < 0 or end < 0 or end < begin:
        raise MarkerError(
            f"{DOC_PATH} does not carry {CATALOG_TABLE_BEGIN!r} ... {CATALOG_TABLE_END!r} "
            "in that order; the generated table has no declared region to replace."
        )
    head = current[: begin + len(CATALOG_TABLE_BEGIN)]
    tail = current[end:]
    return f"{head}\n{render_catalog_markdown(catalog)}\n{tail}"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check" in argv
    current = DOC_PATH.read_text(encoding="utf-8")
    rendered = render_document(current)
    if check_only:
        if current != rendered:
            print(
                f"{DOC_PATH} is stale: the catalogue has columns the table does not "
                "describe, or describes columns the catalogue no longer produces. "
                "Regenerate with `uv run python -m crucible.features.docgen`.",
                file=sys.stderr,
            )
            return 1
        return 0
    DOC_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
