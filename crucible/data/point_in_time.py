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

**Measured depth (2026-09-14).** Sector measured from 2026-05-01. Every
fundamental field the pillars read is measured from 2026-08-20 (the first
session after the 2026-08-19 snapshot). 13F accumulation is measured from
2026-05-18 (2026Q1: 2026-03-31 + 45 days). So the four attractiveness arms
have point-in-time pillars from **2026-08-20** onward; before it, the pillars
are null and the rankers refuse by name.

Why these snapshots and not ArcticDB
====================================

`nousergon-data` also writes fundamentals into the ArcticDB `universe`
library, but as a SCALAR broadcast over the whole frame being written
(`features/feature_engineer.py`, ``df["pe_ratio"] = ...``): a backfill stamps
the collection day's value onto every historical row. That is look-ahead by
construction, so it is not a point-in-time source and is not read here.

**SOTA:** a filing-date-indexed fundamentals store (SEC EDGAR XBRL
`companyfacts`, whose every fact carries its `filed` date, plus 13F by filing
date and a dated GICS history) — point-in-time back to 2009. **Delta:** v1's
dated snapshots give 17 fully measured sessions today, growing one per
session; the deeper source is filed as a follow-up in the PR that introduced
this module.
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
from crucible.keys import constituents_key, fundamental_snapshot_key, inst_ownership_key

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "FUNDAMENTAL_FIELD_COLUMNS",
    "FUNDAMENTAL_DISTINCTNESS_FLOOR",
    "GROUP_COVERAGE_FLOOR_RATIO",
    "INSTITUTIONAL_COLUMNS",
    "MAX_FUNDAMENTAL_STALENESS_SESSIONS",
    "MAX_SECTOR_STALENESS_SESSIONS",
    "POINT_IN_TIME_COLUMNS",
    "SECTOR_COLUMN",
    "THIRTEEN_F_FILING_LAG_DAYS",
    "GroupReading",
    "MappingSnapshotReader",
    "PointInTimeInputs",
    "PointInTimeSource",
    "SnapshotPointInTimeSource",
    "UnavailablePointInTimeSource",
    "constituents_key",
    "distinctness_floor",
    "fundamental_snapshot_key",
    "inst_ownership_key",
    "quarter_knowledge_date",
]

GroupName = Literal["fundamental", "sector", "institutional"]
GroupState = Literal["measured", "predates_source", "stale", "below_coverage", "not_supplied"]

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

_ISO = r"(\d{4}-\d{2}-\d{2})"
_FUNDAMENTAL_KEY_RE = re.compile(rf"^features/{_ISO}/fundamental\.parquet$")
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

    def to_dict(self) -> dict[str, Any]:
        return {
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

    def _load_fundamentals(
        self, day: dt.date, wanted: list[str], frame: pd.DataFrame
    ) -> GroupReading:
        labels = sorted(
            dt.date.fromisoformat(m.group(1))
            for m in self._require_listing("features/", _FUNDAMENTAL_KEY_RE)
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
        key = fundamental_snapshot_key(label)
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
        import pandas as pd

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
            knowledge_date = known.tz_convert(_NEW_YORK).date()
            if knowledge_date < day:
                chosen = (key, knowledge_date, document)
                break
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
