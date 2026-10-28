"""The Think Tank rating as a point-in-time input (`alpha-engine-config-I12395`).

Brian, 2026-10-28: the Think Tank arm competes in the R champion/challenger
every week until it is shown to be inferior. Its ranker reads
`thinktank_rating_ratio`, which nothing produced, so every weekly run refused
it by name. These tests pin the producer's knowledge-time rule and the ways a
ratings document is refused rather than served.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.data.point_in_time import (
    MAX_THINKTANK_STALENESS_SESSIONS,
    THINKTANK_COVERED_COLUMN,
    THINKTANK_RATING_COLUMN,
    MissingSourceError,
)
from crucible.keys import thinktank_ratings_key
from tests.support.thinktank import ratings_document
from tests.test_point_in_time_attractiveness import END, _objects, _source, _tickers


def _reading(inputs):
    return next(r for r in inputs.readings if r.group == "thinktank")


def _with(tickers, **docs):
    objects = {k: v for k, v in _objects(tickers).items() if not k.startswith("thinktank/")}
    objects.update(docs)
    return objects


class TestKnowledgeTime:
    def test_a_document_is_admitted_only_after_the_day_it_was_written(self) -> None:
        tickers = _tickers()
        label = dt.date(2026, 9, 9)
        # Labelled 09-09, written the afternoon of 09-10: unknown on 09-10.
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    tickers[:10], label, "2026-09-10T18:05:00+00:00"
                )
            },
        )
        on_write = _source(objects).load(trading_day=dt.date(2026, 9, 10), symbols=tickers)
        after = _source(objects).load(trading_day=END, symbols=tickers)
        assert _reading(on_write).state == "predates_source"
        assert on_write.frame[THINKTANK_RATING_COLUMN].isna().all()
        assert _reading(after).state == "measured"
        assert _reading(after).knowledge_date == "2026-09-10"

    def test_the_knowledge_date_is_the_new_york_date_of_the_write(self) -> None:
        """00:55Z on 09-11 is 20:55 on 09-10 in New York: admissible on 09-11."""
        tickers = _tickers()
        label = dt.date(2026, 9, 10)
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    tickers[:10], label, "2026-09-11T00:55:00+00:00"
                )
            },
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        assert _reading(inputs).state == "measured"
        assert _reading(inputs).snapshot == thinktank_ratings_key(label)

    def test_a_late_write_falls_back_to_the_previous_document(self) -> None:
        tickers = _tickers()
        early, late = dt.date(2026, 9, 8), dt.date(2026, 9, 9)
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(early): ratings_document(
                    tickers[:10], early, "2026-09-09T14:30:00+00:00", seed=1
                ),
                # Written on the session itself: not yet known when it is decided.
                thinktank_ratings_key(late): ratings_document(
                    tickers[:10], late, "2026-09-11T14:30:00+00:00", seed=2
                ),
            },
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        assert _reading(inputs).snapshot == thinktank_ratings_key(early)

    def test_a_document_without_a_timezone_is_refused(self) -> None:
        tickers = _tickers()
        label = dt.date(2026, 9, 9)
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    tickers[:10], label, "2026-09-10T14:30:00"
                )
            },
        )
        with pytest.raises(MissingSourceError, match="no timezone"):
            _source(objects).load(trading_day=END, symbols=tickers)


class TestValues:
    def test_the_rating_is_scaled_and_only_covered_names_carry_it(self) -> None:
        tickers = _tickers()
        label = dt.date(2026, 9, 9)
        ratings = {t: 10 + 7 * i for i, t in enumerate(tickers[:12])}
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    [], label, "2026-09-10T14:30:00+00:00", ratings=ratings
                )
            },
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        frame = inputs.frame.set_index("ticker")
        assert frame.loc[tickers[:12], THINKTANK_RATING_COLUMN].tolist() == [
            ratings[t] / 100.0 for t in tickers[:12]
        ]
        assert frame.loc[tickers[:12], THINKTANK_COVERED_COLUMN].eq(1.0).all()
        assert frame.loc[tickers[12:], THINKTANK_RATING_COLUMN].isna().all()
        assert frame.loc[tickers[12:], THINKTANK_COVERED_COLUMN].eq(0.0).all()
        reading = _reading(inputs)
        assert (reading.covered, reading.expected) == (12, 12)

    def test_a_degenerate_cross_section_is_unmeasured_by_name(self) -> None:
        tickers = _tickers()
        label = dt.date(2026, 9, 9)
        ratings = dict.fromkeys(tickers[:20], 50)
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    [], label, "2026-09-10T14:30:00+00:00", ratings=ratings
                )
            },
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        assert THINKTANK_RATING_COLUMN in _reading(inputs).unmeasured_fields
        assert inputs.frame[THINKTANK_RATING_COLUMN].isna().all()
        assert THINKTANK_RATING_COLUMN in inputs.unmeasured_columns()

    def test_a_document_whose_names_do_not_resolve_is_below_coverage(self) -> None:
        """Coverage is graded against the document's OWN names: a symbol-format
        change that strands most of them is an outage, never a smaller cut."""
        tickers = _tickers()
        label = dt.date(2026, 9, 9)
        ratings = {f"X{i}": 10 + i for i in range(20)} | {t: 40 for t in tickers[:5]}
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    [], label, "2026-09-10T14:30:00+00:00", ratings=ratings
                )
            },
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        assert _reading(inputs).state == "below_coverage"
        assert inputs.frame[THINKTANK_COVERED_COLUMN].isna().all()

    def test_an_out_of_range_rating_is_not_a_rating(self) -> None:
        tickers = _tickers()
        label = dt.date(2026, 9, 9)
        ratings = {t: 10 + 2 * i for i, t in enumerate(tickers[:30])}
        ratings[tickers[0]] = 140
        ratings[tickers[1]] = True
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    [], label, "2026-09-10T14:30:00+00:00", ratings=ratings
                )
            },
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        frame = inputs.frame.set_index("ticker")
        assert frame.loc[tickers[:2], THINKTANK_RATING_COLUMN].isna().all()
        assert frame.loc[tickers[:2], THINKTANK_COVERED_COLUMN].eq(0.0).all()
        assert _reading(inputs).covered == 28


class TestRefusals:
    def test_a_stale_document_is_an_outage(self) -> None:
        tickers = _tickers()
        label = dt.date(2026, 8, 28)
        objects = _with(
            tickers,
            **{
                thinktank_ratings_key(label): ratings_document(
                    tickers[:10], label, "2026-08-31T14:30:00+00:00"
                )
            },
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        reading = _reading(inputs)
        assert reading.state == "stale"
        assert f"{MAX_THINKTANK_STALENESS_SESSIONS}-session bound" in reading.detail
        assert inputs.frame[THINKTANK_RATING_COLUMN].isna().all()

    def test_an_empty_ratings_prefix_is_a_wrong_source_not_an_empty_market(self) -> None:
        tickers = _tickers()
        with pytest.raises(MissingSourceError, match="thinktank/ratings"):
            _source(_with(tickers)).load(trading_day=END, symbols=tickers)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("schema_version", 2, "schema_version"),
            ("trading_day", "2026-09-08", "trading_day"),
            ("rows", [], "rows"),
        ],
    )
    def test_a_malformed_document_is_refused(self, field, value, match) -> None:
        import json

        tickers = _tickers()
        label = dt.date(2026, 9, 9)
        document = json.loads(ratings_document(tickers[:10], label, "2026-09-10T14:30:00+00:00"))
        document[field] = value
        objects = _with(tickers, **{thinktank_ratings_key(label): json.dumps(document).encode()})
        with pytest.raises(MissingSourceError, match=match):
            _source(objects).load(trading_day=END, symbols=tickers)
