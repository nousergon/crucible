"""`crucible data.daily` — compile one trading day's inputs.

Normative source: plan §4.3, §9.7, §4.12.

One job, three writes and one refusal:

* ``data/{trading_day}/panel.parquet`` — the day's compiled price panel,
  covering the trailing window the feature layer needs, not just the day —
  and every DEPARTED name's history over that window, up to its own last
  session (`alpha-engine-config-I12287`, below);
* ``data/{trading_day}/coverage.json`` — what was read, from which source,
  under which snapshot, with the per-gate coverage figures;
* ``features/{version}/{trading_day}.parquet`` plus its registry — the
  materialized feature layer (§10.4), because R and M recomputing features
  from different code is exactly what makes "the signal degraded"
  inseparable from "the feature changed".

The refusal: **a missing source is `status: failed`, never a zero-fill.**
That is not a policy this module states, it is the only behaviour available
to it — `PriceSource` raises, and `crucible.runner.run_job` writes the
failed manifest and re-raises. There is no branch here that could write a
partial panel and return.

Coverage is a MetricRecord with a declared baseline, not a log line: §9.2
class 5, so `report` reduces it and the console renders it. A day whose
coverage falls below the floor FAILS — a panel covering a third of the
universe is a well-formed artifact containing nothing, and every gate
downstream passes on it.

**`expected_symbols` is mandatory.** It is the coverage ratio's
denominator, and a run given none used to write the metric `status: "OK"`
with `value: None` — the floor above never got a chance to fire, on the
only path that actually runs in production, since nothing in this repo
resolves a universe automatically. That is defect #2 of the 2026-09-01
review: the 901-of-903 bug class the floor exists to catch, restored by a
default argument. A run with no declared universe now raises
:class:`UndeclaredUniverseError` before it reads a source.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crucible.calendar import assert_trading_day
from crucible.data.point_in_time import GROUP_COVERAGE_FLOOR_RATIO, PointInTimeSource
from crucible.data.sources import MissingSourceError, PriceSource
from crucible.documents import load_store_document
from crucible.features import (
    build_features,
    catalog_column_depths,
    min_panel_trading_days,
    registry_payload,
)
from crucible.keys import (
    coverage_key,
    data_panel_key,
    declared_universe_prefix,
    feature_registry_key,
    features_key,
)
from crucible.slots import attribution_factor_symbols, declared_benchmark_symbols

if TYPE_CHECKING:
    import pandas as pd

    from crucible.runner import RunContext

__all__ = [
    "CALENDAR_DAYS_PER_TRADING_DAY",
    "COVERAGE_FLOOR_RATIO",
    "DEFAULT_LOOKBACK_DAYS",
    "MIN_PANEL_TRADING_DAYS",
    "CoverageError",
    "PanelDepthError",
    "UndeclaredUniverseError",
    "run_daily",
    "write_panel",
]

#: Sessions of history the feature layer needs before EVERY catalogue column
#: is computable, read from the producer that declares it
#: (`crucible.features.min_panel_trading_days`) rather than restated here. 313
#: today, and it moves on its own the day a window constant does.
MIN_PANEL_TRADING_DAYS = min_panel_trading_days()

#: Calendar days per trading day, used ONLY to size the vendor request — the
#: panel the vendor returns is then measured in SESSIONS against
#: :data:`MIN_PANEL_TRADING_DAYS` below, so this factor being generous costs a
#: slightly larger fetch and being wrong costs nothing silently. Measured over
#: the NYSE calendar krepis covers (2016-01-01 onward): the worst calendar span
#: of 313 consecutive sessions is 460 days, a ratio of 1.470; 1.55 carries the
#: margin.
CALENDAR_DAYS_PER_TRADING_DAY = 1.55

#: The trailing window requested from the price source, in CALENDAR days
#: because that is the unit every vendor range takes.
#:
#: **Was a hard-coded 400** whose comment read "the longest feature horizon is
#: 252 trading days" — a true statement about the deepest DECLARED
#: `window_trading_days` and a false one about the panel, because
#: `residual_momentum_252d_skip21d_ratio` composes that window over a residual
#: return stream that is itself 61 sessions deep. 400 calendar days is 275
#: sessions; the catalogue needs 313. Measured 2026-09-13
#: (`alpha-engine-config-I10688`): that column was null for 903 of 903 tickers
#: on every one of the 536 sessions the layer had been compiled for, so the M
#: arm that ranks on it (`residual_momentum`) could never fit on any date. A
#: literal cannot notice a window constant changing under it; this cannot be
#: wrong without `min_panel_trading_days` being wrong.
DEFAULT_LOOKBACK_DAYS = math.ceil(MIN_PANEL_TRADING_DAYS * CALENDAR_DAYS_PER_TRADING_DAY)

#: A day covering less of the expected universe than this FAILS. Not a
#: warning: 901 of 903 tickers silently failing a gate for months is the bug
#: class this floor exists to make loud, and a threshold nothing enforces is
#: a number on a dashboard.
COVERAGE_FLOOR_RATIO = 0.90


class CoverageError(RuntimeError):
    """The panel was readable but too thin to compute on."""


class PanelDepthError(RuntimeError):
    """The panel covered the universe but too FEW SESSIONS to compute on.

    The orthogonal half of :class:`CoverageError`, and the one nothing
    measured until `alpha-engine-config-I10688`. Coverage asks how much of the
    universe closed on the day; depth asks how far back the trailing window
    reaches. A panel can be 1.00 on coverage and still be shallower than the
    catalogue's deepest column needs — and the artifact it produces is not
    thinner, it carries that column NULL FOR EVERY TICKER, on every day,
    forever. Measured: 903 of 903 tickers, 536 of 536 sessions, one column
    that looked computed and measured nothing.

    Raised rather than recorded: this is the `avg_volume_20d` class exactly
    (a column consumed as a measurement that was never one), and the feature
    layer's own contract says a column is null because a TICKER lacks history,
    never because the producer was not asked for enough of it.
    """


class UndeclaredUniverseError(RuntimeError):
    """The run declared no expected universe, so the coverage floor could not fire.

    This is defect #2 of the 2026-09-01 adversarial review, restored: with
    `expected_symbols=None` the ratio has no denominator, the metric used to
    be written `status: "OK"` with `value: None`, and `COVERAGE_FLOOR_RATIO`
    never got a chance to raise — on the ONLY path that actually runs in
    production, since nothing in this repo resolves a universe automatically
    and `--symbols` is hand-typed. A run that cannot state what it expected
    to see is not a run whose coverage was "not applicable" — it is a run
    that cannot detect the 901-of-903 bug class at all, which is exactly the
    shape that let it run for months. Principle 7: no data is never
    rendered as green, so this layer refuses to render it as anything.
    """


def write_panel(ctx: RunContext, panel: pd.DataFrame, key: str) -> bytes:
    """Serialize ``panel`` to parquet, write it, and record it as an output."""
    import io

    buffer = io.BytesIO()
    # `index=False` and a fixed compression so the same panel serializes to
    # the same bytes: the replay gate asserts a rerun reproduces the verdict,
    # and a content hash that moved because pyarrow chose a different codec
    # would make every replay diff unattributable.
    panel.to_parquet(buffer, index=False, compression="snappy")
    payload = buffer.getvalue()
    ctx.record_output(key, payload, schema_version="panel.v1")
    return payload


def run_daily(
    ctx: RunContext,
    *,
    source: PriceSource,
    point_in_time: PointInTimeSource,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    expected_symbols: list[str] | None = None,
    coverage_floor: float = COVERAGE_FLOOR_RATIO,
    strategy_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Compile the day. Called through `crucible.runner.run_job`, never directly.

    ``expected_symbols`` is the denominator of the coverage ratio, and it is
    now MANDATORY — a run supplying none raises :class:`UndeclaredUniverseError`
    before touching the source. There is no code path in this repo that
    resolves a universe automatically (the U slot's champion feed is
    downstream of this job, not upstream of it), so the only alternative to
    requiring it here was letting the floor stay unreachable on the path
    that actually runs — which is defect #2 of the 2026-09-01 review,
    restored. A ratio computed over whatever arrived would always read 1.0;
    a ratio with no declared denominator is not "not applicable", it is a
    run that cannot detect a partial universe at all.

    ``strategy_dir``, when given, is forwarded to
    `crucible.slots.attribution_factor_symbols` exactly the way
    `crucible.slots.strategy.grade` forwards `settings.strategy_dir` to
    `load_attribution_params_from_store` — a developer editing
    `alpha-engine-config/strategy/` locally expects the edit to reach this
    fetch too, not just the grading read.

    There is deliberately no ``feature_version`` parameter. `crucible-PR27`
    added a `--feature-version` override that flowed only into the S3 KEY
    (`features_key`, `feature_registry_key`) while `registry_payload()`
    always recomputed the version from the catalogue for the document BODY
    — so `--feature-version v1` wrote `features/v1/registry.json` whose own
    `feature_version` field read the derived hash, silently reachable
    through the override the schema was added to close
    (`alpha-engine-config-I9816`). The fix removes the parameter rather than
    reconciling the two: the version is `feature_version(catalog)` and
    nothing else, computed once below and used for the key, the registry
    body and the parquet's own `attrs["feature_version"]` — one value, three
    places it appears, structurally unable to disagree.
    """
    trading_day = ctx.trading_day
    assert_trading_day(trading_day, context=f"data.daily --date {trading_day}")

    if not expected_symbols:
        raise UndeclaredUniverseError(
            f"data.daily --date {trading_day} was given no --symbols. The coverage "
            f"floor ({coverage_floor:.2f}) cannot fire without a declared universe to "
            "measure coverage against, and a run that silently skipped the floor is "
            "the 901-of-903 bug class COVERAGE_FLOOR_RATIO exists to make loud. Supply "
            "the expected universe explicitly: `crucible data.daily --symbols "
            "AAA,BBB,...`."
        )

    panel = source.load_panel(
        end=trading_day,
        lookback_days=lookback_days,
        symbols=expected_symbols,
    )

    # `PriceSource.load_panel` names, rather than fails on, an expected symbol
    # whose entire stored history falls outside this window — 2025-26 listings
    # read against a 2024 heal window is the measured case
    # (`alpha-engine-config-I10127`). Recorded unconditionally (0 is still a
    # value, never a missing metric) so the absence of listings-in-window is
    # never mistaken for "not measured" — the same discipline `crucible.data`
    # already applies to `universe_coverage_ratio` below.
    unlisted_in_window = sorted(getattr(panel, "attrs", {}).get("unlisted_in_window", []))
    ctx.record_metric(
        {
            "name": "symbols_unlisted_in_window",
            "module": "crucible.data.daily",
            "metric_type": "coverage",
            "value": float(len(unlisted_in_window)),
            "unit": "symbols",
            "n_floor": 0,
            "status": "OK",
            "status_reason": (
                f"{len(unlisted_in_window)} of {len(set(expected_symbols))} expected "
                f"symbol(s) have no stored history overlapping the window ending "
                f"{trading_day}"
                + (
                    f": {unlisted_in_window[:20]}{'…' if len(unlisted_in_window) > 20 else ''}"
                    if unlisted_in_window
                    else ""
                )
            ),
            "source_path": coverage_key(trading_day.isoformat()),
            "last_updated_utc": _utc_now(),
        }
    )

    day_rows = panel[panel["trading_day"] == trading_day]
    if day_rows.empty:
        raise MissingSourceError(
            f"source {source.name!r} returned a panel with NO rows for {trading_day} "
            f"itself (it carried {len(panel)} rows over the trailing window). The day "
            "is an NYSE session, so an absent close is an outage or an unsettled "
            "feed — not an empty market, and not something to carry forward."
        )

    observed = sorted(str(t) for t in day_rows["ticker"].unique())
    expected = sorted(set(expected_symbols))
    coverage_ratio: float = len(observed) / len(expected)
    absent = sorted(set(expected) - set(observed))
    coverage_reason = (
        f"{len(observed)} of {len(expected)} expected tickers closed on {trading_day}"
        + (f"; absent: {absent[:20]}{'…' if len(absent) > 20 else ''}" if absent else "")
    )
    if absent:
        ctx.record_rejected("no close on the trading day", len(absent))

    panel_sessions = int(panel["trading_day"].nunique())
    ctx.record_metric(
        {
            "name": "panel_depth_trading_days",
            "module": "crucible.data.daily",
            "metric_type": "coverage",
            "value": float(panel_sessions),
            "unit": "trading_days",
            "n_floor": 1,
            "status": "OK" if panel_sessions >= MIN_PANEL_TRADING_DAYS else "FAIL",
            "status_reason": (
                f"the trailing panel for {trading_day} carries {panel_sessions} session(s); "
                f"the feature catalogue's deepest column needs {MIN_PANEL_TRADING_DAYS}"
            ),
            "source_path": data_panel_key(trading_day.isoformat()),
            "last_updated_utc": _utc_now(),
            "baseline": float(MIN_PANEL_TRADING_DAYS),
        }
    )
    if panel_sessions < MIN_PANEL_TRADING_DAYS:
        deepest = max(
            (name for name, depth in catalog_column_depths().items() if depth > panel_sessions),
            key=lambda name: catalog_column_depths()[name],
            default="",
        )
        raise PanelDepthError(
            f"the trailing panel for {trading_day} carries {panel_sessions} session(s), "
            f"below the {MIN_PANEL_TRADING_DAYS} the feature catalogue's deepest column "
            f"needs (lookback_days={lookback_days} calendar days). Column {deepest!r} "
            "would be NULL for every ticker on this day, and so on every other day the "
            "same window was used — a column that looks computed and measured nothing. "
            "Raise `lookback_days`; the derived default "
            f"is {DEFAULT_LOOKBACK_DAYS}."
        )

    ctx.record_rows(rows_in=int(len(panel)), rows_out=int(len(panel)))
    ctx.record_metric(
        {
            "name": "universe_coverage_ratio",
            "module": "crucible.data.daily",
            "metric_type": "coverage",
            "value": coverage_ratio,
            "unit": "ratio",
            "n_floor": 1,
            "status": "OK" if coverage_ratio >= coverage_floor else "FAIL",
            "status_reason": coverage_reason,
            "source_path": coverage_key(trading_day.isoformat()),
            "last_updated_utc": _utc_now(),
            "baseline": coverage_floor,
        }
    )

    if coverage_ratio < coverage_floor:
        raise CoverageError(
            f"universe coverage {coverage_ratio:.3f} is below the floor {coverage_floor:.2f} "
            f"on {trading_day}: {coverage_reason}. A thin panel is a FAILED day, not a "
            "degraded one — every downstream gate passes on a well-formed artifact "
            "containing a third of the market."
        )

    # Two declared sources of panel symbols beyond the universe itself, both
    # fetched here — as real symbols, through the same source, never a
    # zero-fill — so a compile-time gap is red at compile time rather than
    # discovered as a grading refusal on a box:
    #
    # * S is graded against SPY (`crucible.slots.SLOTS["s"].benchmark`), and
    #   `crucible.slots.strategy.grade_arm` refuses a book whose benchmark
    #   has no panel row (`alpha-engine-config-I10635`).
    # * The factor-neutral attribution spec
    #   (`strategy/slots/attribution.yaml`, when one is declared) names ETF
    #   proxies for the beta/sector/size factors, and
    #   `crucible.attribution.compute_factor_attribution` (called from
    #   `crucible.slots.strategy.grade`) raises `MissingArtifactError` the
    #   moment one has no panel row (`alpha-engine-config-I10683`).
    #
    # Symbols already part of the declared universe are not refetched.
    extra_panel_symbols = sorted(
        (
            declared_benchmark_symbols()
            | attribution_factor_symbols(store=ctx.store, strategy_dir=strategy_dir)
        )
        - set(expected_symbols)
    )
    if extra_panel_symbols:
        import pandas as pd  # noqa: PLC0415 - only needed on the extra-symbol merge path

        extra_panel = source.load_panel(
            end=trading_day,
            lookback_days=lookback_days,
            symbols=extra_panel_symbols,
        )
        extra_day_rows = extra_panel[extra_panel["trading_day"] == trading_day]
        extra_observed_today = {str(t) for t in extra_day_rows["ticker"].unique()}
        extra_missing_today = sorted(set(extra_panel_symbols) - extra_observed_today)
        if extra_missing_today:
            raise MissingSourceError(
                f"symbol(s) {extra_missing_today} — declared by a slot's "
                "`SlotSpec.benchmark` or by the factor-attribution spec's "
                f"`AttributionFactorParams.factors` — have no close on {trading_day}, "
                "the day this panel covers. Each is read for the settle session "
                "exactly like any ticker's; a panel missing one is not a degraded "
                "read, it is the grading refusal this fetch exists to prevent."
            )
        panel = pd.concat([panel, extra_panel], ignore_index=True)
        panel = panel.sort_values(["trading_day", "ticker"]).reset_index(drop=True)

    # The feature layer is built over the declared universe and its declared
    # extras ONLY, never over the departed names added below: the
    # cross-section is what M scores and U selects from, and a name that left
    # the universe is not a candidate. Its history changes no feature byte.
    feature_panel = panel
    panel, departed = _with_departed_history(
        ctx,
        panel,
        source=source,
        lookback_days=lookback_days,
        already=set(expected) | set(extra_panel_symbols),
    )

    panel_key = data_panel_key(trading_day.isoformat())
    write_panel(ctx, panel, panel_key)

    # One derivation, three consumers: `registry` is the exact document
    # written to `feature_registry_key`, and `registry["feature_version"]`
    # is the exact string used to build BOTH keys below. There is no second
    # source that could name a different version — see the docstring above.
    # The session's non-price inputs, resolved for the DECLARED universe only
    # (benchmark and attribution proxies carry no fundamentals), with one
    # metric per input group so a pillar that is null on a session says why.
    point_in_time_inputs = point_in_time.load(trading_day=trading_day, symbols=expected)
    for reading in point_in_time_inputs.readings:
        reason = f"{reading.state}: {reading.detail}"
        if reading.unmeasured_fields:
            reason += f"; unmeasured field(s): {reading.unmeasured_fields}"
        ctx.record_metric(
            {
                "name": f"point_in_time_{reading.group}_coverage_ratio",
                "module": "crucible.data.daily",
                "metric_type": "coverage",
                "value": reading.covered / reading.expected if reading.expected else 0.0,
                "unit": "ratio",
                "n_floor": 0,
                "status": reading.metric_status,
                "status_reason": reason,
                "source_path": reading.snapshot or point_in_time_inputs.snapshot_id,
                "last_updated_utc": _utc_now(),
                "baseline": GROUP_COVERAGE_FLOOR_RATIO,
            }
        )

    features, catalog = build_features(feature_panel, point_in_time=point_in_time_inputs)
    registry = registry_payload(catalog)
    feature_version = registry["feature_version"]

    feature_key = features_key(feature_version, trading_day.isoformat())
    write_panel(ctx, features, feature_key)

    registry_key = feature_registry_key(feature_version)
    ctx.record_output(
        registry_key,
        json.dumps(registry, indent=2, sort_keys=True).encode("utf-8"),
        schema_version="feature_registry.v1",
    )

    coverage = {
        "schema_version": "coverage.v1",
        "trading_day": trading_day.isoformat(),
        "source": source.name,
        "data_snapshot_id": source.snapshot_id(),
        "lookback_days": lookback_days,
        "rows_total": int(len(panel)),
        "rows_on_day": int(len(day_rows)),
        "tickers_on_day": observed,
        "expected_tickers": expected,
        "coverage_ratio": coverage_ratio,
        "coverage_floor": coverage_floor,
        "panel_key": panel_key,
        "departed_tickers": departed["tickers"],
        "departed_without_history": departed["without_history"],
        "features_key": feature_key,
        "feature_version": feature_version,
        "point_in_time": point_in_time_inputs.to_dict(),
    }
    ctx.record_output(
        coverage_key(trading_day.isoformat()),
        json.dumps(coverage, indent=2, sort_keys=True).encode("utf-8"),
        schema_version="coverage.v1",
    )
    return coverage


def _declared_in_window(store: Any, *, window_start: dt.date, end: dt.date) -> set[str]:
    """Every symbol a declared universe RECORDED for a session inside
    ``(window_start, end)`` named — ``end`` itself excluded, since that day's
    declaration is the run's own ``expected_symbols``.

    Read from the store copies `crucible.data.universe.DeclaredUniverse.record`
    writes beside every data run, never from a live document: the membership
    the panel recovers must be what was declared ON those sessions, and a
    record for a later session is never read, so a recompile cannot depend on
    when it ran.
    """
    prefix = declared_universe_prefix()
    members: set[str] = set()
    for key in sorted(store.list_keys(prefix)):
        day_text = key[len(prefix) :].split("/", 1)[0]
        try:
            day = dt.date.fromisoformat(day_text)
        except ValueError as exc:
            raise ValueError(
                f"declared-universe key {key!r} does not carry an ISO trading day under "
                f"{prefix!r}; every key there is written by `declared_universe_key`"
            ) from exc
        if not window_start < day < end:
            continue
        document = load_store_document(store, key)
        members.update(str(symbol) for symbol in document["symbols"])
    return members


def _with_departed_history(
    ctx: RunContext,
    panel: pd.DataFrame,
    *,
    source: PriceSource,
    lookback_days: int,
    already: set[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """``panel`` plus the window's history for every name that LEFT the universe.

    `alpha-engine-config-I12287`: the 2026-10-09 panel carried no row at all
    for OLN, PSKY and WBD — not just none after they stopped trading — so
    every grade over a window they were part of read a survivors-only
    universe. Two things dropped them, and both are closed here:

    * this job asked the source for today's declared universe and nothing
      else, so a name absent from it was never requested. The departed set
      is now every name an earlier session's RECORDED declaration named
      (:func:`_declared_in_window`), plus
    * every name the source RETAINS after pruning it from the live listing
      whose history reaches the window (`PriceSource.retained_symbols`) — the
      data layer moves a delisted name's whole history into a retention
      library, so a live-library read finds nothing for it at all.

    Departed names never enter the coverage denominator (they are not
    expected to close today) or the feature cross-section (see the caller).
    A departed name the source holds no history for anywhere is NAMED in the
    coverage record and in a metric, never invented — the S grade reads that
    list to tell a name with genuinely no data from a panel that dropped one.
    """
    trading_day = ctx.trading_day
    window_start = trading_day - dt.timedelta(days=lookback_days)
    candidates = _declared_in_window(ctx.store, window_start=window_start, end=trading_day)
    candidates |= set(source.retained_symbols(end=trading_day, lookback_days=lookback_days))
    departed = source.load_departed(
        end=trading_day,
        lookback_days=lookback_days,
        symbols=sorted(candidates - already),
    )
    tickers = sorted(str(t) for t in departed.panel["ticker"].unique())
    without = sorted(departed.without_history)
    for name, names, reason in (
        (
            "departed_symbols_in_panel",
            tickers,
            "name(s) that left the declared universe carry history in the panel ending",
        ),
        (
            "departed_symbols_without_history",
            without,
            "name(s) that left the declared universe have NO stored history in any "
            "library for the window ending",
        ),
    ):
        # Recorded unconditionally (0 is a value, never a missing metric), the
        # same discipline `symbols_unlisted_in_window` follows. OK, not FAIL:
        # a departed name is outside the coverage denominator, and the S
        # grade refuses any HELD name without a return series on its own.
        ctx.record_metric(
            {
                "name": name,
                "module": "crucible.data.daily",
                "metric_type": "coverage",
                "value": float(len(names)),
                "unit": "symbols",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"{len(names)} {reason} {trading_day}"
                    + (f": {names[:20]}{'…' if len(names) > 20 else ''}" if names else "")
                ),
                "source_path": coverage_key(trading_day.isoformat()),
                "last_updated_utc": _utc_now(),
            }
        )
    if tickers:
        import pandas as pd  # noqa: PLC0415 - only needed on the departed merge path

        panel = pd.concat([panel, departed.panel], ignore_index=True)
        panel = panel.sort_values(["trading_day", "ticker"]).reset_index(drop=True)
    return panel, {"tickers": tickers, "without_history": without}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
