"""The CATALOG pin: a committed check that a catalog edit is never silent.

Normative source: `alpha-engine-config-I10498` deliverable 3, the second
adoption of the feature-store units-suffix idea (`AGENTS.md`, and the
`registry.py` module docstring): make the cost of a change visible at EDIT
TIME, in the diff, rather than discoverable only after an expensive backfill
goes stale.

`crucible/features/registry.py::feature_version()` hashes the whole
`CATALOG` — every name, unit, window, expression id and input column — so
ANY edit to `CATALOG` moves the version string and re-addresses the whole
feature layer to a new, empty `features/{version}/` prefix
(`crucible.features.depth` is the detector that watches for exactly this
after the fact). `crucible/features/CATALOG_VERSION` is a committed pin of
the version the catalog is EXPECTED to hash to; this test fails the moment
`feature_version()` disagrees with it, which is the moment a PR touches
`CATALOG` without also touching the pin — a diff nobody can merge without
looking at both lines at once.

This is deliberately not a check that `CATALOG` never changes. It is a check
that a change to `CATALOG` cannot land silently: the fix for a failing run of
this test is (a) update `CATALOG_VERSION` to the new `feature_version()`
value IN THE SAME PR, and (b) record the backfill/copy consequence in the PR
body and in `changelog.d/` (or, absent a `changelog.d/` convention in this
repo, in the PR body alone) — never to "make the test pass" by any other
means, which is exactly the suppression `AGENTS.md` rule 4 forbids.
"""

from __future__ import annotations

from pathlib import Path

from crucible.features.registry import CATALOG, feature_version

CATALOG_VERSION_PATH = (
    Path(__file__).resolve().parent.parent / "crucible" / "features" / "CATALOG_VERSION"
)


def _read_pin() -> str:
    return CATALOG_VERSION_PATH.read_text().strip()


def test_the_pin_file_exists_and_is_not_blank() -> None:
    assert CATALOG_VERSION_PATH.is_file(), (
        f"{CATALOG_VERSION_PATH} is missing. The pin file is what makes a CATALOG edit "
        "visible at edit time; without it there is nothing for this test to compare "
        "feature_version() against."
    )
    assert _read_pin(), f"{CATALOG_VERSION_PATH} is blank"


def test_the_live_catalog_hashes_to_the_pinned_version() -> None:
    live = feature_version(CATALOG)
    pinned = _read_pin()
    assert live == pinned, (
        f"crucible.features.registry.feature_version() now resolves to {live!r}, but "
        f"{CATALOG_VERSION_PATH} still pins {pinned!r}. CATALOG changed.\n\n"
        "The content-addressed feature layer RE-ADDRESSES ITSELF on any catalog edit: "
        f"every reader resolves feature_version() first, so the store's backfill filed "
        f"under {pinned!r} becomes UNREACHABLE the moment this pin goes stale — the "
        "exact failure measured against production (a 532-session backfill left behind "
        "at the old address while the live code saw 7 sessions).\n\n"
        f"Update {CATALOG_VERSION_PATH.name} to {live!r} in THIS SAME PR, and record the "
        "backfill/copy consequence (which prefix the old data is stranded at, and "
        "whether/how it is being carried forward to the new prefix) in the PR body and "
        "in changelog.d/ — never edit this test or the pin without doing both."
    )


def test_the_pin_matches_the_default_feature_version_export() -> None:
    """`crucible.features.DEFAULT_FEATURE_VERSION` is the same derivation
    (`feature_version(CATALOG)`) exported for consumers. Pinned separately so
    a future refactor that makes the two diverge is caught here rather than
    only in production behavior.
    """
    from crucible.features import DEFAULT_FEATURE_VERSION

    assert DEFAULT_FEATURE_VERSION == _read_pin()
