"""Point-in-time non-price inputs: fundamentals, sector and 13F accumulation.

Normative source: plan §4.3 (sources behind an adapter), §9.7 (a restatement
is versioned, never overwritten), and `alpha-engine-config-I10721`.

The price panel is not enough for the attractiveness pillars
(`crucible.slots.rankers.ATTRACTIVENESS_PILLAR_COLUMNS`): four of the seven are
built from vendor fundamentals, and all seven are ranked WITHIN SECTOR. This
module is the one place the feature layer reads those inputs, and the one rule
it enforces is that **a value is what was known on the trading day it is used
for** — never a later snapshot, never today's value stamped onto history.

The contract
============

:class:`PointInTimeSource.load` returns a :class:`PointInTimeInputs` for one
session: one row per requested symbol, the columns of
:data:`POINT_IN_TIME_COLUMNS`, and one :class:`GroupReading` per input group
saying whether that group MEASURED anything on the session and why not if it
did not. A group that did not measure leaves its columns null for every
ticker. It never substitutes a neutral value (v1's collectors wrote `0.0` for
an absent field, which is how `fcf_yield` read one distinct value across 903
names for months, `alpha-engine-config-I8255`).

Knowledge time, per group
-------------------------

A snapshot is admissible for session ``S`` only when its KNOWLEDGE DATE is
strictly before ``S``. The knowledge date is established per group from what
the artifact itself records, never assumed from a folder name:

* **Fundamentals** (`features/{date}/fundamental.parquet`): the folder label.
  Measured 2026-09-14 over all 130 snapshots: every object was last written
  on or before its label date in New York time (128 on the day, 2 the evening
  before), so "strictly after the label" never admits a value written after
  the session it is used for.
* **Sector** (`market_data/weekly/{date}/constituents.json`): the document's
  own ``fetched_at``, converted to the New York calendar date. The folder
  label is NOT a knowledge date here — measured on the same day, every folder
  from 2026-06-25 on was fetched the NEXT business morning, so reading the
  label would admit a sector map on a session before it existed.
* **13F accumulation** (`data/inst_ownership/{YYYYQn}/latest.parquet`): the
  quarter end plus the 45-day Form 13F filing deadline — the regulatory date
  by which every filing in the quarter is public on EDGAR. It is a PUBLIC
  availability date, not the date this fleet ingested the bulk file (the
  2026Q1 table was written on 2026-09-13); that is the point-in-time
  convention every institutional PIT store uses, and it is written here so it
  is a declared choice rather than an accident.

What makes a group unmeasured
-----------------------------

1. No admissible snapshot exists yet (``predates_source``) — a fact about
   history, recorded `unmeasurable`, not a failure.
2. The newest admissible snapshot is older than the group's staleness bound
   (``stale``) — an outage, recorded `FAIL`.
3. The snapshot covers fewer than :data:`GROUP_COVERAGE_FLOOR_RATIO` of the
   requested symbols (``below_coverage``) — recorded `FAIL`. A sector map
   covering the S&P 500 but not the 400 (every snapshot before 2026-04-30)
   would otherwise rank half the universe within sector and silently drop the
   rest.
4. A FIELD whose non-null cross-section carries fewer distinct values than
   :func:`distinctness_floor` is unmeasured on that session even when its
   group is measured — recorded `FAIL`, per field. Non-null is not the same as
   informative: measured 2026-09-14, `fcf_yield` carries ONE distinct value on
   every v1 snapshot before 2026-08-19, `gross_margin` two, `roe` 14-19 and
   `capex_growth_5y` one.

Sector before the first full-universe snapshot: a flagged backfill
------------------------------------------------------------------

No free dated GICS history exists, and the first constituents snapshot that
covers the declared universe was fetched for 2026-05-01. Brian's ruling
(a) on `alpha-engine-config-I10733`, 2026-09-14: a session BEFORE that first
adequate snapshot's knowledge date resolves the EARLIEST adequate snapshot's
sector map instead of reading unmeasured. It is a declared, named source
mode, not a silent fallback. The reading carries
``source_mode="earliest_snapshot_backfill"`` (every other reading carries
``"point_in_time"``), its ``knowledge_date`` is the snapshot's real fetch date
(which is AFTER the session, stated rather than hidden), and
``known_look_ahead`` lists, as data, every entry of
:data:`KNOWN_GICS_RECLASSIFICATIONS` that took effect between the session and
that knowledge date: those tickers carry their post-change sector on a
session that pre-dates the change. The feature layer stamps the mode onto
every row (`sector_earliest_snapshot_backfill_raw`), and `experiment.grade` carries it into
each arm's series lineage, its `scores/` document and the `arena_cycle`
artifact, so a reader can see which part of a score rests on backfilled
sectors. A session at or after the first adequate snapshot's knowledge date
is resolved strictly point-in-time exactly as before.

**Measured depth (2026-09-14).** Sector measured point-in-time from 2026-05-01;
backfilled (flagged) before it. Every
fundamental field the pillars read is measured from 2026-08-20 (the first
session after the 2026-08-19 snapshot). 13F accumulation is measured from
2026-05-18 (2026Q1: 2026-03-31 + 45 days). So the four attractiveness arms
have point-in-time pillars from **2026-08-20** onward; before it, the pillars
are null and the rankers refuse by name.

A second fundamentals source, keyed by filing date
===================================================

:class:`FilingDatePointInTimeSource` reads a different producer's tree —
SEC EDGAR XBRL `companyfacts`, at
`fundamentals_pit/edgar/v1/sessions/{date}.parquet`
(`crucible.keys.edgar_fundamentals_session_key`) — for the fundamental group
only; sector and 13F are the same reads :class:`SnapshotPointInTimeSource`
already makes. The knowledge rule is identical: the file labelled ``L`` uses
only EDGAR facts with `filed` <= ``L`` and ``L``'s own split-adjusted close,
and it is admissible for session ``S`` only when ``L < S``. Every row also
declares its own `schema_version`, `knowledge_date` and `latest_filed`, and
:class:`FilingDatePointInTimeSource` refuses (rather than silently reads) a
snapshot whose self-declaration is inconsistent with its own key — a
look-ahead in the producer stops the compile, never gets averaged into a
pillar.

Why these snapshots and not ArcticDB
====================================

`nousergon-data` also writes fundamentals into the ArcticDB `universe`
library, but as a SCALAR broadcast over the whole frame being written
(`features/feature_engineer.py`, ``df["pe_ratio"] = ...``): a backfill stamps
the collection day's value onto every historical row. That is look-ahead by
construction, so it is not a point-in-time source and is not read here.

**SOTA:** a filing-date-indexed fundamentals store (SEC EDGAR XBRL
`companyfacts`, whose every fact carries its `filed` date) now EXISTS as
:class:`FilingDatePointInTimeSource`, reaching back as far as the producer's
backfill (planned from 2022-01-03). **Delta:** sector still has no free
dated GICS history and is measured only from 2026-05-01 (constituents
`fetched_at`), and 13F is still bound by its 45-day quarterly filing
deadline — so the fundamental pillar's own depth grows with the EDGAR
backfill while sector and institutional remain the shallower bound on the
seven attractiveness pillars until a dated GICS source and a filing-date 13F
feed exist too.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from crucible.data.sources import MissingSourceError
from crucible.documents import read_document
from crucible.keys import (
    constituents_key,
    edgar_fundamentals_session_key,
    fundamental_snapshot_key,
    inst_ownership_key,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "EARLIEST_SNAPSHOT_BACKFILL_MODE",
    "EDGAR_SESSION_SCHEMA_VERSION",
    "FUNDAMENTAL_FIELD_COLUMNS",
    "FUNDAMENTAL_DISTINCTNESS_FLOOR",
    "GROUP_COVERAGE_FLOOR_RATIO",
    "INSTITUTIONAL_COLUMNS",
    "KNOWN_GICS_RECLASSIFICATIONS",
    "MAX_FUNDAMENTAL_STALENESS_SESSIONS",
    "MAX_SECTOR_STALENESS_SESSIONS",
    "POINT_IN_TIME_COLUMNS",
    "POINT_IN_TIME_MODE",
    "SECTOR_COLUMN",
    "THIRTEEN_F_FILING_LAG_DAYS",
    "FilingDatePointInTimeSource",
    "GicsReclassification",
    "GroupReading",
    "MappingSnapshotReader",
    "PointInTimeInputs",
    "PointInTimeSource",
    "SnapshotPointInTimeSource",
    "UnavailablePointInTimeSource",
    "constituents_key",
    "distinctness_floor",
    "edgar_fundamentals_session_key",
    "fundamental_snapshot_key",
    "inst_ownership_key",
    "quarter_knowledge_date",
]

GroupName = Literal["fundamental", "sector", "institutional"]
GroupState = Literal["measured", "predates_source", "stale", "below_coverage", "not_supplied"]
SourceMode = Literal["point_in_time", "earliest_snapshot_backfill"]

#: The mode of a reading resolved strictly from what was known before the session.
POINT_IN_TIME_MODE: SourceMode = "point_in_time"
#: The mode of a sector reading resolved from the earliest adequate constituents
#: snapshot for a session that pre-dates it (Brian's ruling (a),
#: `alpha-engine-config-I10733`). Carries a known look-ahead; never silent.
EARLIEST_SNAPSHOT_BACKFILL_MODE: SourceMode = "earliest_snapshot_backfill"


@dataclass(frozen=True)
class GicsReclassification:
    """One public GICS change that a backfilled sector map cannot see.

    ``effective_session`` is the first NYSE session the new classification
    applied to. A backfilled map for a session BEFORE it carries each
    ticker's ``to_sector`` where the truth was ``from_sector``.
    ``tickers`` are the symbols as the constituents snapshots key them today
    (Fleetcor is `CPAY`, Fiserv is `FI`); the list is the S&P 500 members
    named in the public notices and is NOT exhaustive for the S&P 400, nor
    for single-company reclassifications, which no free source enumerates.
    """

    effective_session: dt.date
    change: str
    from_sector: str
    to_sector: str
    tickers: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_session": self.effective_session.isoformat(),
            "change": self.change,
            "from_sector": self.from_sector,
            "to_sector": self.to_sector,
            "tickers": list(self.tickers),
            "source": self.source,
        }


_GICS_2023_NOTICE = (
    "S&P DJI / MSCI GICS structure change announced 2022-12, effective after the close of "
    "2023-03-17 (S&P DJI Indexology, '2023 GICS Changes: S&P 500 Impact Analysis')"
)

#: Every GICS STRUCTURE change effective between the start of the v2 history
#: (2022-01-03) and the first adequate constituents snapshot (2026-05-01).
#: There is exactly one: the March 2023 change. The next structure review's
#: consultation closes 2026-10-30, after the snapshots begin, so it is a
#: point-in-time event the snapshots themselves record.
KNOWN_GICS_RECLASSIFICATIONS: tuple[GicsReclassification, ...] = (
    GicsReclassification(
        effective_session=dt.date(2023, 3, 20),
        change="Data Processing & Outsourced Services discontinued; Transaction & Payment "
        "Processing Services created under Financials",
        from_sector="Information Technology",
        to_sector="Financials",
        tickers=("V", "MA", "PYPL", "FIS", "FI", "GPN", "CPAY", "JKHY"),
        source=_GICS_2023_NOTICE,
    ),
    GicsReclassification(
        effective_session=dt.date(2023, 3, 20),
        change="Data Processing & Outsourced Services remainder moved to Commercial & "
        "Professional Services",
        from_sector="Information Technology",
        to_sector="Industrials",
        tickers=("ADP", "PAYX", "BR"),
        source=_GICS_2023_NOTICE,
    ),
    GicsReclassification(
        effective_session=dt.date(2023, 3, 20),
        change="General Merchandise Stores moved to Consumer Staples Merchandise Retail",
        from_sector="Consumer Discretionary",
        to_sector="Consumer Staples",
        tickers=("DG", "DLTR", "TGT"),
        source=_GICS_2023_NOTICE,
    ),
)

#: v1 fundamental field -> feature column. The v1 values are carried AS STORED,
#: including `nousergon-data/collectors/fundamentals.py`'s normalisations, and
#: the column name says so: `pe_ratio` there is trailing P/E divided by 30 and
#: clipped to [-3, 3], so a column named `pe_ratio` here would be read as a P/E
#: and be off by thirty — the `avg_volume_20d` defect in a new place.
FUNDAMENTAL_FIELD_COLUMNS: dict[str, str] = {
    "roe": "roe_ratio",
    "debt_to_equity": "debt_to_equity_div2_ratio",
    "gross_margin": "gross_margin_ratio",
    "current_ratio": "current_ratio_div3_ratio",
    "pe_ratio": "pe_div30_ratio",
    "pb_ratio": "pb_div5_ratio",
    "fcf_yield": "fcf_yield_ratio",
    "revenue_growth_3y": "revenue_growth_3y_ratio",
    "eps_growth_3y": "eps_growth_3y_ratio",
    "capex_growth_5y": "capex_growth_5y_ratio",
    "payout_ratio": "payout_ratio",
}

SECTOR_COLUMN = "sector_raw"

#: The 13F inputs, raw fund counts. `crucible.features.compute` derives the
#: catalogue column `institutional_accumulation_raw` from them.
INSTITUTIONAL_COLUMNS: dict[str, str] = {
    "n_funds_increasing": "n_funds_increasing_raw",
    "n_funds_decreasing": "n_funds_decreasing_raw",
}

POINT_IN_TIME_COLUMNS: tuple[str, ...] = (
    SECTOR_COLUMN,
    *FUNDAMENTAL_FIELD_COLUMNS.values(),
    *INSTITUTIONAL_COLUMNS.values(),
)

#: v1's staleness bound on a fundamentals snapshot
#: (`crucible-research/scoring/factor_scoring.py::_MAX_FUNDAMENTAL_STALENESS_TD`).
MAX_FUNDAMENTAL_STALENESS_SESSIONS = 10
#: The constituents snapshot is written most sessions and at least weekly; the
#: same bound, so a sector map two weeks old is an outage rather than a value.
MAX_SECTOR_STALENESS_SESSIONS = 10

#: Form 13F is due 45 calendar days after the quarter ends (17 CFR 240.13f-1).
THIRTEEN_F_FILING_LAG_DAYS = 45

#: v1's cross-sectional distinct-count floor
#: (`crucible-research/scoring/universe_board.py::_FUNDAMENTAL_DISTINCTNESS_FLOOR`).
FUNDAMENTAL_DISTINCTNESS_FLOOR = 100

#: The share of requested symbols a group snapshot must cover before any of
#: its values are used. The same 0.90 `crucible.data.daily.COVERAGE_FLOOR_RATIO`
#: holds the price panel to, restated rather than imported because `daily`
#: imports this module.
GROUP_COVERAGE_FLOOR_RATIO = 0.90

#: The `schema_version` every row of an EDGAR filing-date session must carry
#: (`alpha-engine-config-I10733`). A row at any other version, or with the
#: column absent, is a producer contract violation and stops the compile
#: (`FilingDatePointInTimeSource._validate_fundamental_snapshot`) rather than
#: being silently admitted — the same "refuse an outage, never guess" rule
#: `SnapshotPointInTimeSource._require_listing` already keeps for an empty
#: source.
EDGAR_SESSION_SCHEMA_VERSION = 1

_ISO = r"(\d{4}-\d{2}-\d{2})"
_FUNDAMENTAL_KEY_RE = re.compile(rf"^features/{_ISO}/fundamental\.parquet$")
_EDGAR_SESSION_KEY_RE = re.compile(rf"^fundamentals_pit/edgar/v1/sessions/{_ISO}\.parquet$")
_CONSTITUENTS_KEY_RE = re.compile(rf"^market_data/weekly/{_ISO}/constituents\.json$")
_INST_KEY_RE = re.compile(r"^data/inst_ownership/(\d{4})Q([1-4])/latest\.parquet$")

_NEW_YORK = "America/New_York"


def quarter_knowledge_date(year: int, quarter: int) -> dt.date:
    """The 13F filing deadline for a calendar quarter: quarter end + 45 days."""
    month = quarter * 3
    next_month_first = dt.date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    quarter_end = next_month_first - dt.timedelta(days=1)
    return quarter_end + dt.timedelta(days=THIRTEEN_F_FILING_LAG_DAYS)


def distinctness_floor(covered: int) -> int:
    """Distinct non-null values a field needs over ``covered`` names.

    v1's 100, capped at half the covered cross-section so a small but real
    universe is not declared degenerate by construction (v1 exempted any
    universe under 200 names outright, which let a 25-name all-zero snapshot
    through — the 2026-03-27 v1 snapshot is exactly that).
    """
    return max(2, min(FUNDAMENTAL_DISTINCTNESS_FLOOR, math.ceil(covered / 2)))


class SnapshotReader(Protocol):
    """The two read operations this module needs. `crucible.store.Store` has both."""

    def list_keys(self, prefix: str = "") -> Iterator[str]: ...

    def get_bytes(self, key: str) -> bytes: ...


class MappingSnapshotReader:
    """An in-memory :class:`SnapshotReader` over ``{key: bytes}``.

    The fixture backend. Not a mock: :class:`SnapshotPointInTimeSource` runs its
    real resolution over it, so a test exercises the production rules.
    """

    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self._objects = dict(objects)

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        return iter(sorted(k for k in self._objects if k.startswith(prefix)))

    def get_bytes(self, key: str) -> bytes:
        try:
            return self._objects[key]
        except KeyError as exc:
            raise FileNotFoundError(key) from exc


@dataclass(frozen=True)
class GroupReading:
    """What one input group measured on one session, and why not if it did not."""

    group: GroupName
    state: GroupState
    snapshot: str | None
    knowledge_date: str | None
    covered: int
    expected: int
    detail: str
    #: column -> reason, for fields unmeasured while the group itself measured.
    unmeasured_fields: dict[str, str] = field(default_factory=dict)
    #: How the value was resolved. Only the sector group can read anything but
    #: :data:`POINT_IN_TIME_MODE` (`alpha-engine-config-I10733`, ruling (a)).
    source_mode: SourceMode = POINT_IN_TIME_MODE
    #: The public reclassifications the resolved map is known to get wrong on
    #: this session. Empty for a point-in-time reading, by construction.
    known_look_ahead: tuple[GicsReclassification, ...] = ()

    @property
    def metric_status(self) -> str:
        if self.state == "measured":
            return "FAIL" if self.unmeasured_fields else "OK"
        if self.state in ("predates_source", "not_supplied"):
            return "unmeasurable"
        return "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "state": self.state,
            "snapshot": self.snapshot,
            "knowledge_date": self.knowledge_date,
            "covered": self.covered,
            "expected": self.expected,
            "detail": self.detail,
            "unmeasured_fields": dict(sorted(self.unmeasured_fields.items())),
            "source_mode": self.source_mode,
            "known_look_ahead": [r.to_dict() for r in self.known_look_ahead],
        }


@dataclass(frozen=True)
class PointInTimeInputs:
    """One session's point-in-time inputs, and the readings that qualify them."""

    trading_day: dt.date
    frame: pd.DataFrame
    readings: tuple[GroupReading, ...]
    source: str
    snapshot_id: str

    def unmeasured_columns(self) -> frozenset[str]:
        """Every point-in-time column that measured nothing on this session."""
        out: set[str] = set()
        for reading in self.readings:
            if reading.state != "measured":
                out.update(_GROUP_COLUMNS[reading.group])
            out.update(reading.unmeasured_fields)
        return frozenset(out)

    @property
    def sector_source_mode(self) -> SourceMode | None:
        """The sector reading's mode, or None when sector measured nothing.

        This is what `crucible.features.compute` stamps onto every feature row
        as `sector_earliest_snapshot_backfill_raw`: a session whose sector is null has no
        sector-derived value to qualify.
        """
        sector = next(r for r in self.readings if r.group == "sector")
        return sector.source_mode if sector.state == "measured" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_source_mode": self.sector_source_mode,
            "source": self.source,
            "snapshot_id": self.snapshot_id,
            "trading_day": self.trading_day.isoformat(),
            "readings": [r.to_dict() for r in self.readings],
            "unmeasured_columns": sorted(self.unmeasured_columns()),
        }


_GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "fundamental": tuple(FUNDAMENTAL_FIELD_COLUMNS.values()),
    "sector": (SECTOR_COLUMN,),
    "institutional": tuple(INSTITUTIONAL_COLUMNS.values()),
}


def _empty_frame(symbols: Sequence[str]) -> pd.DataFrame:
    import pandas as pd

    frame = pd.DataFrame({"ticker": sorted(set(symbols))})
    frame[SECTOR_COLUMN] = pd.Series([None] * len(frame), dtype="object")
    for column in (*FUNDAMENTAL_FIELD_COLUMNS.values(), *INSTITUTIONAL_COLUMNS.values()):
        frame[column] = float("nan")
    return frame


class PointInTimeSource(ABC):
    """One adapter: the non-price inputs known before a trading day."""

    name: str = "abstract"

    @abstractmethod
    def load(self, *, trading_day: dt.date, symbols: Sequence[str]) -> PointInTimeInputs:
        """The inputs admissible for ``trading_day``, one row per symbol."""

    @abstractmethod
    def snapshot_id(self) -> str:
        """Which store was read — recorded beside the panel's own snapshot id."""


class UnavailablePointInTimeSource(PointInTimeSource):
    """A run that declares, by name and with a reason, that it has no such inputs.

    Every group reads ``not_supplied`` and every point-in-time column is null,
    so a pillar built on them is null and a ranker reading it refuses by name.
    Production never constructs this (`crucible.track_a` builds
    :class:`SnapshotPointInTimeSource`); it exists for a fixture market that has
    no fundamentals, where an empty snapshot store would be read as an outage.
    """

    name = "unavailable"

    def __init__(self, *, reason: str) -> None:
        if not reason.strip():
            raise ValueError("an unavailable point-in-time source must say why")
        self.reason = reason

    def load(self, *, trading_day: dt.date, symbols: Sequence[str]) -> PointInTimeInputs:
        expected = len(set(symbols))
        readings = tuple(
            GroupReading(
                group=group,
                state="not_supplied",
                snapshot=None,
                knowledge_date=None,
                covered=0,
                expected=expected,
                detail=f"declared unavailable: {self.reason}",
            )
            for group in ("fundamental", "sector", "institutional")
        )
        return PointInTimeInputs(
            trading_day=trading_day,
            frame=_empty_frame(symbols),
            readings=readings,
            source=self.name,
            snapshot_id=self.snapshot_id(),
        )

    def snapshot_id(self) -> str:
        return f"unavailable:{self.reason}"


def _sessions_between(start: dt.date, end: dt.date) -> int:
    """NYSE sessions in ``(start, end]``."""
    from crucible.calendar import count_trading_days  # noqa: PLC0415 - krepis is heavy

    return int(count_trading_days(start, end))


class SnapshotPointInTimeSource(PointInTimeSource):
    """The production source: v1's dated snapshots, read-only, in the data bucket.

    ``reader`` is any :class:`SnapshotReader` rooted at the data bucket — in
    production a read-only `crucible.store.S3Store` built from
    ``CRUCIBLE_ARCTIC_BUCKET`` (`crucible.track_a`), so no bucket name appears
    in this package. Listings and parsed snapshots are cached per instance:
    `data.heal` recompiles hundreds of sessions through one source, and
    re-listing the prefix per session would be the whole cost of the heal.
    """

    name = "v1-snapshots"

    def __init__(self, reader: SnapshotReader, *, label: str) -> None:
        self._reader = reader
        self._label = label
        self._listing: dict[str, list[str]] = {}
        self._parquet_cache: dict[str, Any] = {}
        self._json_cache: dict[str, Any] = {}
        self._sector_names_cache: dict[str, frozenset[str]] = {}

    def snapshot_id(self) -> str:
        return f"v1-snapshots:{self._label}"

    # -- reads ---------------------------------------------------------------

    def _keys(self, prefix: str) -> list[str]:
        if prefix not in self._listing:
            self._listing[prefix] = sorted(self._reader.list_keys(prefix))
        return self._listing[prefix]

    def _parquet(self, key: str) -> pd.DataFrame:
        if key not in self._parquet_cache:
            import pandas as pd

            try:
                payload = self._reader.get_bytes(key)
            except Exception as exc:
                raise MissingSourceError(
                    f"point-in-time snapshot {key!r} is listed but could not be read: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._parquet_cache[key] = pd.read_parquet(io.BytesIO(payload))
        return self._parquet_cache[key]

    def _json(self, key: str) -> dict[str, Any]:
        if key not in self._json_cache:
            read = read_document(key, lambda: self._reader.get_bytes(key))
            if read.document is None:
                raise MissingSourceError(
                    f"point-in-time snapshot {key!r} is listed but could not be read as an "
                    f"object: {read.problem}"
                )
            self._json_cache[key] = read.document
        return self._json_cache[key]

    # -- resolution ------------------------------------------------------------

    def load(self, *, trading_day: dt.date, symbols: Sequence[str]) -> PointInTimeInputs:
        wanted = sorted(set(symbols))
        if not wanted:
            raise ValueError("point-in-time inputs need a declared set of symbols")
        frame = _empty_frame(wanted).set_index("ticker")
        readings = (
            self._load_fundamentals(trading_day, wanted, frame),
            self._load_sector(trading_day, wanted, frame),
            self._load_institutional(trading_day, wanted, frame),
        )
        return PointInTimeInputs(
            trading_day=trading_day,
            frame=frame.reset_index(),
            readings=readings,
            source=self.name,
            snapshot_id=self.snapshot_id(),
        )

    def _require_listing(self, prefix: str, pattern: re.Pattern[str]) -> list[re.Match[str]]:
        matches = [m for key in self._keys(prefix) if (m := pattern.match(key))]
        if not matches:
            raise MissingSourceError(
                f"point-in-time source {self.snapshot_id()!r} holds no object matching "
                f"{pattern.pattern!r} under {prefix!r}. An empty source is a wrong bucket "
                "or an outage, not a market with no fundamentals, and reading it as the "
                "second would null every pillar on every session without a word."
            )
        return matches

    # -- fundamental-group resolution parameters, overridable per source ----
    #
    # `SnapshotPointInTimeSource` and `FilingDatePointInTimeSource` share ONE
    # resolution path — listing prefix, key regex and key function are the
    # only things that differ between v1's dated snapshots and the EDGAR
    # filing-date tree, and the knowledge-time rule (`label < day`), the
    # staleness bound, the coverage floor and the per-field distinctness
    # floor are identical for both (`alpha-engine-config-I10733`). A
    # subclass wanting a fourth source implements these four hooks rather
    # than re-copying `_load_fundamentals`.

    def _fundamental_source_prefix(self) -> str:
        return "features/"

    def _fundamental_key_pattern(self) -> re.Pattern[str]:
        return _FUNDAMENTAL_KEY_RE

    def _fundamental_key(self, label: dt.date) -> str:
        return fundamental_snapshot_key(label)

    def _validate_fundamental_snapshot(
        self, key: str, label: dt.date, snapshot: pd.DataFrame
    ) -> None:
        """A hook for a subclass to refuse a snapshot beyond the base shape
        checks (`ticker` present) `_load_fundamentals` already makes. A no-op
        for v1's dated snapshots, which carry no `schema_version` or
        `knowledge_date` self-declaration to check.
        """

    def _load_fundamentals(
        self, day: dt.date, wanted: list[str], frame: pd.DataFrame
    ) -> GroupReading:
        prefix = self._fundamental_source_prefix()
        pattern = self._fundamental_key_pattern()
        labels = sorted(
            dt.date.fromisoformat(m.group(1)) for m in self._require_listing(prefix, pattern)
        )
        admissible = [label for label in labels if label < day]
        if not admissible:
            return GroupReading(
                group="fundamental",
                state="predates_source",
                snapshot=None,
                knowledge_date=None,
                covered=0,
                expected=len(wanted),
                detail=(
                    f"no fundamentals snapshot is labelled before {day}; the earliest is "
                    f"{labels[0]}"
                ),
            )
        label = admissible[-1]
        key = self._fundamental_key(label)
        age = _sessions_between(label, day)
        if age > MAX_FUNDAMENTAL_STALENESS_SESSIONS:
            return GroupReading(
                group="fundamental",
                state="stale",
                snapshot=key,
                knowledge_date=label.isoformat(),
                covered=0,
                expected=len(wanted),
                detail=(
                    f"newest admissible fundamentals snapshot {key} is {age} sessions before "
                    f"{day}, beyond the {MAX_FUNDAMENTAL_STALENESS_SESSIONS}-session bound"
                ),
            )
        snapshot = self._parquet(key)
        self._validate_fundamental_snapshot(key, label, snapshot)
        missing = [c for c in ("ticker", *FUNDAMENTAL_FIELD_COLUMNS) if c not in snapshot.columns]
        # A snapshot written before a field existed (the growth and payout
        # fields arrived 2026-05-21) does not carry it: that field is
        # unmeasured on the session, by name, rather than the snapshot refused.
        if "ticker" in missing:
            raise MissingSourceError(f"{key!r} carries no `ticker` column")
        block = snapshot.drop_duplicates("ticker", keep="last").set_index("ticker")
        covered_names = sorted(set(block.index) & set(wanted))
        reading = _coverage_reading("fundamental", key, label, covered_names, wanted)
        if reading is not None:
            return reading
        floor = distinctness_floor(len(covered_names))
        unmeasured: dict[str, str] = {}
        for v1_field, column in FUNDAMENTAL_FIELD_COLUMNS.items():
            if v1_field not in block.columns:
                unmeasured[column] = f"{key} does not carry `{v1_field}`"
                continue
            values = block.loc[covered_names, v1_field].astype(float)
            distinct = int(values.dropna().nunique())
            if distinct < floor:
                unmeasured[column] = (
                    f"`{v1_field}` carries {distinct} distinct value(s) over "
                    f"{len(covered_names)} names in {key}, below the floor of {floor}: "
                    "non-null, and not informative"
                )
                continue
            frame.loc[covered_names, column] = values.to_numpy()
        return GroupReading(
            group="fundamental",
            state="measured",
            snapshot=key,
            knowledge_date=label.isoformat(),
            covered=len(covered_names),
            expected=len(wanted),
            detail=f"{key}: {len(covered_names)} of {len(wanted)} names",
            unmeasured_fields=unmeasured,
        )

    def _load_sector(self, day: dt.date, wanted: list[str], frame: pd.DataFrame) -> GroupReading:
        labels = sorted(
            dt.date.fromisoformat(m.group(1))
            for m in self._require_listing("market_data/weekly/", _CONSTITUENTS_KEY_RE)
        )
        # A document labelled D was fetched no earlier than the evening before
        # D, so only labels up to a week past `day` can possibly be admissible;
        # the knowledge date itself is read from each candidate document.
        candidates = [label for label in labels if label <= day + dt.timedelta(days=7)]
        chosen: tuple[str, dt.date, dict[str, Any]] | None = None
        for label in reversed(candidates):
            key = constituents_key(label)
            knowledge_date, document = self._sector_knowledge_date(key)
            if knowledge_date < day:
                chosen = (key, knowledge_date, document)
                break
        backfill = self._earliest_adequate_sector_snapshot(labels, wanted)
        if backfill is not None and backfill[1] >= day:
            # Brian's ruling (a), `alpha-engine-config-I10733`: before the first
            # snapshot that covers the universe became known, use that snapshot's
            # map as a NAMED mode. `chosen` may exist here (an S&P-500-only map,
            # which reads below_coverage) — it is superseded by the flagged
            # backfill because the ruling covers every session before the first
            # adequate map, not only sessions before any map at all.
            return self._backfilled_sector(day, wanted, frame, *backfill)
        if chosen is None:
            return GroupReading(
                group="sector",
                state="predates_source",
                snapshot=None,
                knowledge_date=None,
                covered=0,
                expected=len(wanted),
                detail=f"no constituents snapshot was fetched before {day}",
            )
        key, knowledge_date, document = chosen
        age = _sessions_between(knowledge_date, day)
        if age > MAX_SECTOR_STALENESS_SESSIONS:
            return GroupReading(
                group="sector",
                state="stale",
                snapshot=key,
                knowledge_date=knowledge_date.isoformat(),
                covered=0,
                expected=len(wanted),
                detail=(
                    f"newest admissible constituents snapshot {key} was fetched {age} sessions "
                    f"before {day}, beyond the {MAX_SECTOR_STALENESS_SESSIONS}-session bound"
                ),
            )
        sector_map = document.get("sector_map")
        if not isinstance(sector_map, dict):
            raise MissingSourceError(f"{key!r} carries no `sector_map` object")
        covered_names = sorted(
            t for t in wanted if isinstance(sector_map.get(t), str) and sector_map[t].strip()
        )
        reading = _coverage_reading("sector", key, knowledge_date, covered_names, wanted)
        if reading is not None:
            return reading
        frame.loc[covered_names, SECTOR_COLUMN] = [sector_map[t] for t in covered_names]
        return GroupReading(
            group="sector",
            state="measured",
            snapshot=key,
            knowledge_date=knowledge_date.isoformat(),
            covered=len(covered_names),
            expected=len(wanted),
            detail=f"{key} (fetched {document['fetched_at']}): {len(covered_names)} names",
        )

    def _sector_knowledge_date(self, key: str) -> tuple[dt.date, dict[str, Any]]:
        import pandas as pd

        document = self._json(key)
        fetched = document.get("fetched_at")
        if not isinstance(fetched, str) or not fetched:
            raise MissingSourceError(
                f"{key!r} records no `fetched_at`, so when it became known cannot be "
                "established; a sector map of unknown knowledge time cannot be admitted "
                "to any session"
            )
        known = pd.Timestamp(fetched)
        if known.tzinfo is None:
            raise MissingSourceError(f"{key!r} `fetched_at`={fetched!r} carries no timezone")
        return known.tz_convert(_NEW_YORK).date(), document

    def _earliest_adequate_sector_snapshot(
        self, labels: list[dt.date], wanted: list[str]
    ) -> tuple[str, dt.date, dict[str, Any]] | None:
        """The earliest-FETCHED constituents snapshot covering the coverage floor.

        "Adequate" is the same :data:`GROUP_COVERAGE_FLOOR_RATIO` a
        point-in-time reading must meet: the earliest map in the store is an
        S&P-500-only document that would rank half the universe within sector,
        so it is not the map the ruling means. Ordered by knowledge date, not
        folder label (the two disagree, see the module docstring).
        """
        wanted_set = frozenset(wanted)
        best: tuple[str, dt.date, dict[str, Any]] | None = None
        for label in labels:
            # A document labelled L is fetched no earlier than the evening
            # before L, so once labels pass the best knowledge date by a week
            # no later label can be known earlier; stop reading.
            if best is not None and label > best[1] + dt.timedelta(days=7):
                break
            key = constituents_key(label)
            knowledge_date, document = self._sector_knowledge_date(key)
            if best is not None and knowledge_date >= best[1]:
                continue
            covered = len(self._sector_names(key, document) & wanted_set)
            if covered / len(wanted) >= GROUP_COVERAGE_FLOOR_RATIO:
                best = (key, knowledge_date, document)
        return best

    def _sector_names(self, key: str, document: dict[str, Any]) -> frozenset[str]:
        """The tickers a map labels, cached: a heal asks this once per session."""
        if key not in self._sector_names_cache:
            sector_map = document.get("sector_map")
            if not isinstance(sector_map, dict):
                raise MissingSourceError(f"{key!r} carries no `sector_map` object")
            self._sector_names_cache[key] = frozenset(
                t for t, v in sector_map.items() if isinstance(v, str) and v.strip()
            )
        return self._sector_names_cache[key]

    def _backfilled_sector(
        self,
        day: dt.date,
        wanted: list[str],
        frame: pd.DataFrame,
        key: str,
        knowledge_date: dt.date,
        document: dict[str, Any],
    ) -> GroupReading:
        sector_map = document["sector_map"]
        covered_names = sorted(
            t for t in wanted if isinstance(sector_map.get(t), str) and sector_map[t].strip()
        )
        frame.loc[covered_names, SECTOR_COLUMN] = [sector_map[t] for t in covered_names]
        look_ahead = tuple(
            r for r in KNOWN_GICS_RECLASSIFICATIONS if day < r.effective_session <= knowledge_date
        )
        named = "; ".join(
            f"{r.from_sector}->{r.to_sector} effective {r.effective_session} "
            f"({', '.join(r.tickers)})"
            for r in look_ahead
        )
        return GroupReading(
            group="sector",
            state="measured",
            snapshot=key,
            knowledge_date=knowledge_date.isoformat(),
            covered=len(covered_names),
            expected=len(wanted),
            detail=(
                f"{EARLIEST_SNAPSHOT_BACKFILL_MODE}: no universe-covering sector map was "
                f"fetched before {day}; {key} (fetched {document['fetched_at']}, AFTER the "
                f"session) supplies {len(covered_names)} names. Known look-ahead: "
                + (named or "no enumerated GICS structure change falls in the gap")
                + ". Single-company reclassifications are not enumerated."
            ),
            source_mode=EARLIEST_SNAPSHOT_BACKFILL_MODE,
            known_look_ahead=look_ahead,
        )

    def _load_institutional(
        self, day: dt.date, wanted: list[str], frame: pd.DataFrame
    ) -> GroupReading:
        quarters = sorted(
            (int(m.group(1)), int(m.group(2)))
            for m in self._require_listing("data/inst_ownership/", _INST_KEY_RE)
        )
        admissible = [q for q in quarters if quarter_knowledge_date(*q) < day]
        if not admissible:
            return GroupReading(
                group="institutional",
                state="predates_source",
                snapshot=None,
                knowledge_date=None,
                covered=0,
                expected=len(wanted),
                detail=(
                    f"no 13F quarter's filing deadline precedes {day}; the earliest held is "
                    f"{quarters[0][0]}Q{quarters[0][1]}"
                ),
            )
        year, quarter = admissible[-1]
        key = inst_ownership_key(year, quarter)
        known = quarter_knowledge_date(year, quarter)
        # One missing quarter is tolerated (the SEC bulk data set for the latest
        # quarter publishes weeks after its deadline); a second is an outage.
        skipped = _quarter_offset(year, quarter, 2)
        if quarter_knowledge_date(*skipped) < day:
            return GroupReading(
                group="institutional",
                state="stale",
                snapshot=key,
                knowledge_date=known.isoformat(),
                covered=0,
                expected=len(wanted),
                detail=(
                    f"newest 13F quarter held is {year}Q{quarter}; the filing deadline of "
                    f"{skipped[0]}Q{skipped[1]} ({quarter_knowledge_date(*skipped)}) has also "
                    f"passed by {day}, so two quarters are missing"
                ),
            )
        table = self._parquet(key)
        needed = ["ticker", *INSTITUTIONAL_COLUMNS]
        absent = [c for c in needed if c not in table.columns]
        if absent:
            raise MissingSourceError(f"{key!r} is missing column(s) {absent}")
        block = table.drop_duplicates("ticker", keep="last").set_index("ticker")
        covered_names = sorted(set(block.index) & set(wanted))
        reading = _coverage_reading("institutional", key, known, covered_names, wanted)
        if reading is not None:
            return reading
        floor = distinctness_floor(len(covered_names))
        unmeasured: dict[str, str] = {}
        for v1_field, column in INSTITUTIONAL_COLUMNS.items():
            values = block.loc[covered_names, v1_field].astype(float)
            distinct = int(values.dropna().nunique())
            if distinct < floor:
                unmeasured[column] = (
                    f"`{v1_field}` carries {distinct} distinct value(s) over "
                    f"{len(covered_names)} names in {key}, below the floor of {floor}"
                )
                continue
            frame.loc[covered_names, column] = values.to_numpy()
        return GroupReading(
            group="institutional",
            state="measured",
            snapshot=key,
            knowledge_date=known.isoformat(),
            covered=len(covered_names),
            expected=len(wanted),
            detail=f"{key} (public from {known}): {len(covered_names)} of {len(wanted)} names",
            unmeasured_fields=unmeasured,
        )


class FilingDatePointInTimeSource(SnapshotPointInTimeSource):
    """The SOTA fundamentals source: SEC EDGAR XBRL `companyfacts`, keyed by
    the filing date every fact carries (`alpha-engine-config-I10733`).

    Sector and institutional (13F) are UNCHANGED from
    :class:`SnapshotPointInTimeSource` — this class narrows only the
    fundamental group's resolution to a different producer's key shape
    (:func:`crucible.keys.edgar_fundamentals_session_key`), by way of the
    four ``_fundamental_*`` hooks that method defines. The knowledge rule is
    identical to v1's: the file labelled ``L`` uses only facts with EDGAR
    ``filed`` <= ``L`` and ``L``'s own close, and it is admissible for
    session ``S`` only when ``L < S``.

    A row that fails the producer's own declared contract stops the compile
    rather than being read as if it were valid: a `schema_version` other
    than :data:`EDGAR_SESSION_SCHEMA_VERSION`, a `knowledge_date` that does
    not equal the file's own label, or a `latest_filed` that is null or
    later than the label (a look-ahead) all raise :class:`MissingSourceError`
    naming the key and the reason — never a `stale`/`below_coverage` reading,
    which would let a corrupt producer write pass as an ordinary outage.
    """

    name = "edgar-filing-date"

    def snapshot_id(self) -> str:
        return f"edgar-filing-date:{self._label}"

    def _fundamental_source_prefix(self) -> str:
        return "fundamentals_pit/edgar/v1/sessions/"

    def _fundamental_key_pattern(self) -> re.Pattern[str]:
        return _EDGAR_SESSION_KEY_RE

    def _fundamental_key(self, label: dt.date) -> str:
        return edgar_fundamentals_session_key(label)

    def _validate_fundamental_snapshot(
        self, key: str, label: dt.date, snapshot: pd.DataFrame
    ) -> None:
        if "schema_version" not in snapshot.columns:
            raise MissingSourceError(f"{key!r} carries no `schema_version` column")
        versions = snapshot["schema_version"]
        if versions.isna().any() or set(versions.dropna().unique().tolist()) != {
            EDGAR_SESSION_SCHEMA_VERSION
        }:
            raise MissingSourceError(
                f"{key!r} `schema_version` is not exactly [{EDGAR_SESSION_SCHEMA_VERSION}] "
                f"for every row (found {sorted(versions.dropna().unique().tolist())!r}, "
                f"{int(versions.isna().sum())} null)"
            )
        if "knowledge_date" not in snapshot.columns:
            raise MissingSourceError(f"{key!r} carries no `knowledge_date` column")
        knowledge = snapshot["knowledge_date"]
        label_iso = label.isoformat()
        if knowledge.isna().any() or set(knowledge.dropna().unique().tolist()) != {label_iso}:
            raise MissingSourceError(
                f"{key!r} `knowledge_date` does not equal its own label {label_iso} for "
                f"every row (found {sorted(knowledge.dropna().unique().tolist())!r})"
            )
        if "latest_filed" not in snapshot.columns:
            raise MissingSourceError(f"{key!r} carries no `latest_filed` column")
        filed = snapshot["latest_filed"]
        if filed.isna().any():
            raise MissingSourceError(
                f"{key!r} carries a null `latest_filed` for at least one row — a filing "
                "date of unknown knowledge time cannot be admitted to any session"
            )
        look_ahead = filed[filed > label_iso]
        if not look_ahead.empty:
            raise MissingSourceError(
                f"{key!r} carries `latest_filed` > its own label {label_iso} "
                f"(e.g. {look_ahead.iloc[0]!r}) — a look-ahead in the producer, refused "
                "rather than silently admitted"
            )


def _quarter_offset(year: int, quarter: int, offset: int) -> tuple[int, int]:
    index = year * 4 + (quarter - 1) + offset
    return index // 4, index % 4 + 1


def _coverage_reading(
    group: GroupName,
    key: str,
    knowledge_date: dt.date,
    covered_names: list[str],
    wanted: list[str],
) -> GroupReading | None:
    ratio = len(covered_names) / len(wanted)
    if ratio >= GROUP_COVERAGE_FLOOR_RATIO:
        return None
    return GroupReading(
        group=group,
        state="below_coverage",
        snapshot=key,
        knowledge_date=knowledge_date.isoformat(),
        covered=len(covered_names),
        expected=len(wanted),
        detail=(
            f"{key} covers {len(covered_names)} of {len(wanted)} requested names "
            f"({ratio:.3f}), below the {GROUP_COVERAGE_FLOOR_RATIO:.2f} floor; none of its "
            "values are used on this session"
        ),
    )
