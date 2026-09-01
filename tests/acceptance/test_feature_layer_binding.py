"""The M slot reads track A's feature layer through the registry, not a path.

Normative source: plan §10 component 4 — "without it R and M recompute
features from different code, and 'the signal degraded' cannot be separated
from 'the feature changed'".

**This test fails until `crucible/features/` lands, and that is the design.**
The repository carries no suppression collection at all (§11.1): no xfail, no
skip, no marker. A cross-track dependency is therefore expressed the same way
every plan clause is — an honest failure naming the requirement and the
owning track, which goes green the moment both PRs are on `main` with no
marker for anyone to remember to remove.

**It lives in `tests/acceptance/` for exactly that reason.** This directory is
the repository's declared home for clauses written before the code that
satisfies them: `ci.yml` runs it in full on every commit and publishes the
count, and does not block on it. A cross-track clause in the BLOCKING job
would red `main` for tracks A and C over work neither of them has started —
a public signal the groom loop reads — and the alternatives (an xfail, or a
path exclusion in `ci.yml`) are both suppressions wearing different clothes.
Reusing the mechanism that already exists is neither. See `README.md` here.

What track B commits to here is the CONSUMER side: the M slot resolves its
feature columns through `crucible.features`' registry interface and records
the layer's version and hash as a manifest input, so R and M provably read
the same artifact. The producer is track A's.
"""

from __future__ import annotations

from typing import Any, NoReturn

import pytest

CLAUSE = "plan §10 component 4 (feature layer) × track B M slot"
REQUIREMENT = (
    "crucible.features exposes a registry interface returning a versioned, hashed "
    "feature panel for a trading day; crucible.slots.model.FeatureLayerSource reads "
    "the M recipe's declared columns through it, and the M run manifest records the "
    "same features/{version}/{trading_day} input hash the R slot records."
)


def _unmet(exc: BaseException | None = None) -> NoReturn:
    detail = f" Blocked on: {exc}" if exc is not None else ""
    pytest.fail(
        f"UNMET — {CLAUSE}\n"
        f"  Required: {REQUIREMENT}\n"
        f"  Status:   track A has not landed crucible/features yet "
        f"(crucible v2 phase 1, alpha-engine-config-I9757).{detail}",
        pytrace=False,
    )


def _features_module() -> Any:
    try:
        import crucible.features as features
    except ImportError as exc:
        _unmet(exc)
    return features


class TestFeatureLayerBinding:
    def test_the_m_slot_resolves_its_columns_through_the_registry(self) -> None:
        from crucible.slots.model import FeatureLayerSource

        features = _features_module()
        registry = getattr(features, "CATALOG", None)
        if registry is None:
            _unmet(AttributeError("crucible.features exposes no CATALOG registry"))

        source = FeatureLayerSource(registry=registry, version="v1")
        try:
            panel = source.panel(trading_day="2026-08-28", columns=("mom_21d_ratio",))
        except NotImplementedError as exc:
            _unmet(exc)
        assert panel.feature_version == "v1"
        assert "mom_21d_ratio" in panel.features

    def test_every_recipe_column_carries_a_units_suffix(self) -> None:
        """The fleet's units-suffix contract (`AGENTS.md`): a bare column name
        is how `avg_volume_20d` was emitted as a ratio and consumed as raw
        shares, silently failing 901 of 903 tickers for months."""
        from crucible.slots.model import assert_units_suffixes

        features = _features_module()
        registry = getattr(features, "CATALOG", None)
        if registry is None:
            _unmet(AttributeError("crucible.features exposes no CATALOG registry"))
        assert_units_suffixes(tuple(registry))
