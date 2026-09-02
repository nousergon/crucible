"""The M slot reads track A's feature layer through the registry, not a path.

Normative source: plan §10 component 4 — "without it R and M recompute
features from different code, and 'the signal degraded' cannot be separated
from 'the feature changed'".

**This clause failed until `crucible/features/` landed, and that was the
design.** The repository carries no suppression collection at all (§11.1): no
xfail, no skip, no marker. A cross-track dependency is expressed the same way
every plan clause is — an honest failure naming the requirement and the
owning track, which went green the moment both producer and consumer were on
`main`, with no marker for anyone to remember to remove. It went green on
2026-09-01 with `alpha-engine-config-I9772` / `-I9765`, once the two tracks'
interfaces were reconciled; the `_unmet` path below stays, because a clause
that could not report the producer's absence would be dark, not green.

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

from crucible.gate import PHASES

#: The producer this clause waits on landed under phase 1
#: (alpha-engine-config-I9757 per `crucible.gate.PHASES`). Derived rather
#: than hardcoded so a phase renumbering cannot leave this message stale
#: (alpha-engine-config-I9839) — this is the same class of defect the
#: `_unmet` in `test_plan_section_2_objectives.py` was fixed for.
_PHASE1 = next(p for p in PHASES if p.id == "phase1")

CLAUSE = "plan §10 component 4 (feature layer) × track B M slot"
REQUIREMENT = (
    "crucible.features exposes a registry interface returning a versioned, hashed "
    "feature panel for a trading day; crucible.slots.model.FeatureLayerSource reads "
    "the M recipe's declared columns through it, and the M run manifest records the "
    "same features/{version}/{trading_day} input hash the R slot records."
)


def _unmet(exc: BaseException | None = None) -> NoReturn:
    """Fail naming WHICH of the two conditions holds, not one of them.

    The message this replaced said "track A has not landed
    `crucible/features` yet" unconditionally — and the first real failure of
    this clause was not that at all: the producer had landed, and the two
    tracks had reconciled nothing (`alpha-engine-config-I9772`). An interface
    MISMATCH wearing an unimplemented-producer message sends the reader to
    the wrong track's backlog, which is how three disagreements sat on `main`
    reading as one unstarted dependency.

    So the status is MEASURED here rather than asserted: the module either
    imports or it does not, and the message says which, with what it found.
    """
    try:  # noqa: SIM105 - the import IS the measurement
        import crucible.features as _features
    except ImportError:
        status = (
            "the producer is ABSENT — `import crucible.features` fails. This clause is "
            f"waiting on track A (crucible v2 phase {_PHASE1.number}, {_PHASE1.tracker}), "
            "and nothing in track B can clear it."
        )
    else:
        status = (
            "the producer is PRESENT — `crucible.features` imports and exposes "
            f"{sorted(n for n in dir(_features) if not n.startswith('_'))}. This is "
            "therefore an INTERFACE MISMATCH between a landed producer and a landed "
            "consumer, NOT an unimplemented producer: read the failure below against "
            "`crucible/schemas/feature_registry.v1.json`, which is the declared "
            "contract both sides are held to, and fix the side that departs from it "
            "(see this function's docstring for the issue that first found this)."
        )
    detail = f"\n  Blocked on: {exc!r}" if exc is not None else ""
    pytest.fail(
        f"UNMET — {CLAUSE}\n  Required: {REQUIREMENT}\n  Status:   {status}{detail}",
        pytrace=False,
    )


def _features_module() -> Any:
    try:
        import crucible.features as features
    except ImportError as exc:
        _unmet(exc)
    return features


#: The columns the M arm `residual_momentum` declares in
#: `alpha-engine-config/strategy/arms/m/residual_momentum.yaml`. Named here
#: rather than a placeholder so the clause asserts the real demand: these are
#: the columns an M arm will not train without.
M_ARM_COLUMNS: tuple[str, ...] = (
    "residual_momentum_252d_skip21d_zscore",
    "residual_vol_20d_ratio",
    "momentum_change_21d_zscore",
    "beta_60d_raw",
)


class TestFeatureLayerBinding:
    def test_the_m_slot_resolves_its_columns_through_the_registry(
        self, store: Any, source: Any, cycle_date: Any
    ) -> None:
        from crucible.data import run_daily
        from crucible.runner import run_job
        from crucible.slots.model import FeatureLayerSource

        features = _features_module()
        registry = getattr(features, "CATALOG", None)
        if registry is None:
            _unmet(AttributeError("crucible.features exposes no CATALOG registry"))

        # Track A's producer writes the artifact; nothing here recomputes it.
        run_job(
            "data.daily",
            lambda ctx: run_daily(ctx, source=source, expected_symbols=source.symbols()),
            store=store,
            trading_day=cycle_date,
        )

        # No consumer names a version. It is DERIVED from the catalogue, so an
        # edited recipe writes to a new prefix instead of overwriting the layer
        # an earlier verdict was computed from (alpha-engine-config-I9772).
        layer = FeatureLayerSource(store=store, registry=registry)
        try:
            panel = layer.panel(trading_day=cycle_date.isoformat(), columns=M_ARM_COLUMNS)
        except NotImplementedError as exc:
            _unmet(exc)

        assert panel.feature_version == features.DEFAULT_FEATURE_VERSION
        for column in M_ARM_COLUMNS:
            assert column in panel.features
            assert panel.column(column).shape == (len(panel.dates), len(panel.names))

    def test_a_column_the_layer_does_not_produce_raises_at_load(
        self, store: Any, source: Any, cycle_date: Any
    ) -> None:
        """The 2026-08-28 condition, refused rather than trained on.

        Seven features were hard-zeroed that week and every surface said
        healthy. A recipe naming a column the layer does not produce must fail
        where it is loaded, never arrive as a silently substituted zero.
        """
        from crucible.data import run_daily
        from crucible.runner import run_job
        from crucible.slots.model import FeatureLayerSource

        features = _features_module()
        run_job(
            "data.daily",
            lambda ctx: run_daily(ctx, source=source, expected_symbols=source.symbols()),
            store=store,
            trading_day=cycle_date,
        )
        layer = FeatureLayerSource(store=store, registry=features.CATALOG)
        with pytest.raises(KeyError, match="not produced by feature layer version"):
            layer.panel(
                trading_day=cycle_date.isoformat(),
                columns=("a_column_the_layer_never_produced_ratio",),
            )

    def test_every_recipe_column_carries_a_units_suffix(self) -> None:
        """The fleet's units-suffix contract (`AGENTS.md`): a bare column name
        is how `avg_volume_20d` was emitted as a ratio and consumed as raw
        shares, silently failing 901 of 903 tickers for months.

        The registry is a tuple of `FeatureSpec`s carrying units and lineage,
        not a tuple of names — so the check reads names through the accessor
        the producer exposes rather than assuming the element type
        (alpha-engine-config-I9772).
        """
        from crucible.slots.model import assert_units_suffixes

        features = _features_module()
        registry = getattr(features, "CATALOG", None)
        if registry is None:
            _unmet(AttributeError("crucible.features exposes no CATALOG registry"))
        assert_units_suffixes(features.feature_names(registry))
