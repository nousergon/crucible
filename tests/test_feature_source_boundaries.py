"""Declared source boundaries — `alpha-engine-config-I10721`.

The measured failure these tests pin, 2026-09-17 against the live layer
`features/v553618c991dd/` (1,180 sessions, 2022-01-03..2026-09-16):

`check_feature_layer_completeness` read RED and named session **2022-01-03**,
the first session of history, where no fundamentals snapshot can be admissible
by construction. It named that session every time, because the worst-session
tie-break maximises the number of dead columns and a head-of-history boundary
has every source-derived column dead at once — no downstream defect can
exceed it. Sitting inside the same RED, invisible, were 92 sessions
(2025-07-09..2025-11-14) whose fundamentals were dead because an EDGAR re-heal
chunk died four minutes into its range. The hole survived for three days
because the row that would have lit was already lit, permanently and
correctly, for a reason nobody could act on.

`TestTheDefectIsNotHiddenBehindTheBoundary` is the load-bearing case: it
builds a layer with BOTH bands and asserts the reading names the defect one.
Deleting the boundary declaration makes it fail.
"""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import pytest

from crucible.data.point_in_time import FUNDAMENTAL_FIELD_COLUMNS, INSTITUTIONAL_COLUMNS
from crucible.features.boundaries import (
    DECLARED_SOURCE_BOUNDARIES,
    boundary_for_column,
    explain_dead_columns,
)
from crucible.features.compute import PILLAR_COMPONENTS
from crucible.features.depth import (
    FEATURES_PREFIX,
    check_feature_layer_completeness,
)
from crucible.features.registry import feature_names
from crucible.store import LocalStore

LIVE_VERSION = "vtestboundaries"

#: The sixteen fundamental columns and the three institutional ones that the
#: live layer reads dead across `alpha-engine-config-I10721`'s band.
_FUNDAMENTAL_DEAD = (
    "roe_ratio",
    "debt_to_equity_div2_ratio",
    "gross_margin_ratio",
    "current_ratio_div3_ratio",
    "pe_div30_ratio",
    "pb_div5_ratio",
    "fcf_yield_ratio",
    "revenue_growth_3y_ratio",
    "eps_growth_3y_ratio",
    "capex_growth_5y_ratio",
    "payout_ratio",
    "sustainable_growth_rate_ratio",
    "quality_pillar_pct",
    "value_pillar_pct",
    "growth_pillar_pct",
    "stewardship_pillar_pct",
)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _session_frame(day: str, null_columns: tuple[str, ...]) -> pd.DataFrame:
    tickers = tuple(f"T{i:03d}" for i in range(20))
    data: dict[str, list[object]] = {
        "trading_day": [day] * len(tickers),
        "ticker": list(tickers),
    }
    for i, col in enumerate(feature_names()):
        data[col] = [None] * len(tickers) if col in null_columns else [float(i + 1)] * len(tickers)
    return pd.DataFrame(data)


def _put(store: LocalStore, day: str, null_columns: tuple[str, ...] = ()) -> None:
    buf = io.BytesIO()
    _session_frame(day, null_columns).to_parquet(buf)
    store.put_bytes(f"{FEATURES_PREFIX}{LIVE_VERSION}/{day}.parquet", buf.getvalue())


class TestTheDeclarationItself:
    def test_every_vendor_fundamental_field_is_bound_by_the_fundamental_boundary(self) -> None:
        """A field added to `FUNDAMENTAL_FIELD_COLUMNS` that is NOT declared
        here would read as a defect on every pre-2022-01-04 session. That is
        the safe direction, and this test exists so the divergence is loud
        rather than discovered from a red board.
        """
        for column in FUNDAMENTAL_FIELD_COLUMNS.values():
            boundary = boundary_for_column(column)
            assert boundary is not None, column
            assert boundary.first_measurable_session == dt.date(2022, 1, 4)

    def test_every_institutional_column_is_bound_by_the_13f_deadline(self) -> None:
        for column in INSTITUTIONAL_COLUMNS.values():
            boundary = boundary_for_column(column)
            assert boundary is not None, column
            assert boundary.first_measurable_session == dt.date(2022, 11, 15)

    def test_a_pillar_inherits_the_latest_boundary_among_its_components(self) -> None:
        """`stewardship_pillar_pct` reads `payout_ratio` (2022-01-04) and
        `institutional_accumulation_raw` (2022-11-15). It is measurable only
        from the later, and a boundary that took the earlier would excuse the
        pillar over ten months it genuinely could not be computed for — the
        exact band `alpha-engine-config-I10721` measured as by-design.
        """
        boundary = boundary_for_column("stewardship_pillar_pct")
        assert boundary is not None
        assert boundary.first_measurable_session == dt.date(2022, 11, 15)

    def test_a_price_only_pillar_carries_no_boundary_at_all(self) -> None:
        """`momentum_pillar_pct` and `defensiveness_pillar_pct` are built from
        the price panel, which has history everywhere the layer does. Dead on
        any session is a defect on every session.
        """
        price_only = (
            "momentum_pillar_pct",
            "defensiveness_pillar_pct",
            "momentum_12_1_pillar_pct",
        )
        for pillar in price_only:
            assert pillar in PILLAR_COMPONENTS
            assert boundary_for_column(pillar) is None

    def test_no_boundary_is_open_ended(self) -> None:
        """Every declared boundary names a real date and a reason with its
        evidence — a boundary without them is a suppression list
        (`crucible/AGENTS.md` rule 4) wearing a detector's clothes.
        """
        assert DECLARED_SOURCE_BOUNDARIES
        for boundary in DECLARED_SOURCE_BOUNDARIES:
            assert isinstance(boundary.first_measurable_session, dt.date)
            assert len(boundary.reason) > 80
            assert "measured" in boundary.evidence


class TestExplainDeadColumns:
    def test_a_column_dead_before_its_boundary_is_explained(self) -> None:
        unexplained, explained, applied = explain_dead_columns("2022-01-03", ("roe_ratio",))
        assert unexplained == ()
        assert explained == ("roe_ratio",)
        assert [b.group for b in applied] == ["fundamental"]

    def test_the_same_column_dead_ON_its_boundary_is_a_defect(self) -> None:
        """The boundary is the FIRST measurable session, so it explains
        nothing on that session itself. An off-by-one the other way would
        excuse a real dead session at the head of every band.
        """
        unexplained, explained, _ = explain_dead_columns("2022-01-04", ("roe_ratio",))
        assert unexplained == ("roe_ratio",)
        assert explained == ()

    def test_a_column_dead_long_after_its_boundary_is_a_defect(self) -> None:
        """`alpha-engine-config-I10721`'s band, in one call."""
        unexplained, explained, _ = explain_dead_columns("2025-09-15", _FUNDAMENTAL_DEAD)
        assert unexplained == tuple(sorted(_FUNDAMENTAL_DEAD))
        assert explained == ()

    def test_a_column_with_no_boundary_is_never_explained(self) -> None:
        unexplained, _, _ = explain_dead_columns("2022-01-03", ("momentum_pillar_pct",))
        assert unexplained == ("momentum_pillar_pct",)

    def test_an_unparseable_session_raises_rather_than_reading_as_explained(self) -> None:
        with pytest.raises(ValueError):
            explain_dead_columns("not-a-date", ("roe_ratio",))


class TestTheDefectIsNotHiddenBehindTheBoundary:
    """The `alpha-engine-config-I10721` shape, end to end."""

    def test_the_reading_names_the_defect_band_not_the_boundary_band(self, store) -> None:
        # The by-design band: every source-derived column dead, before both
        # boundaries. This is the session the old tie-break always picked.
        _put(store, "2022-01-03", _FUNDAMENTAL_DEAD + ("institutional_accumulation_raw",))
        # The defect band: the same fundamentals dead, 3.5 years later.
        _put(store, "2025-09-15", _FUNDAMENTAL_DEAD)
        _put(store, "2026-09-16")

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "RED"
        assert reading.session == "2025-09-15"
        assert reading.dead_sessions == ("2025-09-15",)
        assert reading.boundary_sessions == ("2022-01-03",)
        assert "2025-09-15" in reading.detail
        assert "boundaries, not defects" in reading.detail

    def test_a_layer_whose_only_dead_band_is_declared_reads_green_and_says_so(self, store) -> None:
        """GREEN, never silent: the boundary band is named in the detail, so
        "no data" is stated rather than rendered as health (principle 7).
        """
        _put(store, "2022-01-03", _FUNDAMENTAL_DEAD + ("institutional_accumulation_raw",))
        _put(store, "2022-06-01", ("institutional_accumulation_raw", "stewardship_pillar_pct"))
        _put(store, "2026-09-16")

        reading = check_feature_layer_completeness(store, live_version=LIVE_VERSION)

        assert reading.state == "GREEN"
        assert reading.dead_sessions == ()
        assert reading.boundary_sessions == ("2022-01-03", "2022-06-01")
        assert "boundaries, not defects" in reading.detail
        assert "2022-01-03" in reading.detail

    def test_mutation_without_the_boundary_the_reading_names_the_unfixable_band(
        self, store
    ) -> None:
        """The load-bearing mutation: with boundaries removed from the
        classification, the same two-band layer reports 2022-01-03 — a true,
        permanent, unactionable RED — and the 2025 defect appears only as one
        date in a list. That is the state the live board was in on
        2026-09-17, and it is what this module changes.
        """
        _put(store, "2022-01-03", _FUNDAMENTAL_DEAD + ("institutional_accumulation_raw",))
        _put(store, "2025-09-15", _FUNDAMENTAL_DEAD)
        _put(store, "2026-09-16")

        import crucible.features.depth as depth_module

        original = depth_module.explain_dead_columns
        try:
            depth_module.explain_dead_columns = lambda session, dead: (
                tuple(sorted(dead)),
                (),
                (),
            )
            mutated = check_feature_layer_completeness(store, live_version=LIVE_VERSION)
        finally:
            depth_module.explain_dead_columns = original

        assert mutated.state == "RED"
        assert mutated.session == "2022-01-03"
        assert mutated.boundary_sessions == ()
