"""`crucible data.daily` — compile one trading day's inputs.

Normative source: plan §4.3, §9.7, §4.12.

One job, three writes and one refusal:

* ``data/{trading_day}/panel.parquet`` — the day's compiled price panel,
  covering the trailing window the feature layer needs, not just the day;
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
from typing import TYPE_CHECKING, Any

from crucible.calendar import assert_trading_day
from crucible.data.sources import MissingSourceError, PriceSource
from crucible.features import build_features, registry_payload
from crucible.keys import (
    coverage_key,
    data_panel_key,
    feature_registry_key,
    features_key,
)

if TYPE_CHECKING:
    import pandas as pd

    from crucible.runner import RunContext

__all__ = [
    "COVERAGE_FLOOR_RATIO",
    "DEFAULT_LOOKBACK_DAYS",
    "CoverageError",
    "UndeclaredUniverseError",
    "run_daily",
    "write_panel",
]

#: The trailing window the feature layer needs. 400 calendar days is a little
#: over 252 sessions plus slack: the longest feature horizon is 252 trading
#: days, and a window that only just covers it produces a first row of NaN on
#: every holiday-heavy year.
DEFAULT_LOOKBACK_DAYS = 400

#: A day covering less of the expected universe than this FAILS. Not a
#: warning: 901 of 903 tickers silently failing a gate for months is the bug
#: class this floor exists to make loud, and a threshold nothing enforces is
#: a number on a dashboard.
COVERAGE_FLOOR_RATIO = 0.90


class CoverageError(RuntimeError):
    """The panel was readable but too thin to compute on."""


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
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    expected_symbols: list[str] | None = None,
    coverage_floor: float = COVERAGE_FLOOR_RATIO,
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

    panel_key = data_panel_key(trading_day.isoformat())
    write_panel(ctx, panel, panel_key)

    # One derivation, three consumers: `registry` is the exact document
    # written to `feature_registry_key`, and `registry["feature_version"]`
    # is the exact string used to build BOTH keys below. There is no second
    # source that could name a different version — see the docstring above.
    features, catalog = build_features(panel)
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
        "features_key": feature_key,
        "feature_version": feature_version,
    }
    ctx.record_output(
        coverage_key(trading_day.isoformat()),
        json.dumps(coverage, indent=2, sort_keys=True).encode("utf-8"),
        schema_version="coverage.v1",
    )
    return coverage


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
