"""The PUBLIC surface: what it shows, and — mostly — what it refuses to show.

Normative sources: `alpha-engine-config-I10223`, `alpha-engine-config-I10215`
(no alpha on a public surface), `repository-tiering-policy.md`.

The interesting assertions here are all refusals. A test that shows the page
rendering a champion pointer has shown nothing about the property this module
exists for; the property is that a store FULL of return figures,
infrastructure identifiers and private links produces a page carrying none of
them, and that is what :class:`TestNothingLeaks` asserts against artifacts
deliberately stuffed with all three.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.console.public import (
    FORBIDDEN_SOURCE_FIELDS,
    PUBLIC_JSON_KEY,
    PUBLIC_KEY,
    VERDICT_LEGEND,
    VERDICT_STATE,
    _verdict_label,
    build_public_page,
    render_public_html,
    write_public_page,
)
from crucible.console.render import STATUS_COLORS
from crucible.gate import LADDER_KEY, PHASES
from crucible.keys import (
    BOARD_CURRENT_KEY,
    arena_cycle_key,
    attribution_key,
    champion_key,
    morning_report_key,
)
from crucible.slots import SLOTS
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
DAY = FRIDAY.isoformat()
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)

#: Every figure planted in the fixture store below. None of these strings may
#: appear in either published output. Distinctive on purpose: `0.0731` is not
#: a number that turns up by accident in a date, a count or a schema version.
PLANTED_FIGURES = (
    "0.0731",
    "1.9982",
    "-0.4413",
    "88.7712",
    "https://github.com/nousergon/alpha-engine-config/issues/9999",
    "/crucible/console",
    "01JG0000000000000000000001",
)


#: The phase this fixture's ladder row describes, read off `crucible.gate.PHASES`
#: rather than spelled as a number and a tracker id. A renumbered phase must not
#: leave this fixture asserting against an issue that no longer names it
#: (`tests/test_no_stale_tracker_literals.py`, alpha-engine-config-I9839).
_PHASE = PHASES[2]


def _ladder() -> dict:
    return {
        "schema_version": "phase_ladder.v1",
        "trading_day": DAY,
        "generated_utc": "2026-08-29T02:00:00Z",
        "current_phase": _PHASE.id,
        "phases_met": 2,
        "phases_total": 6,
        "out_of_order": [],
        "unmeasured": 0,
        "phases": [
            {
                "phase": _PHASE.id,
                "number": _PHASE.number,
                "title": _PHASE.title,
                "state": "UNMET",
                "console_state": "DEGRADED",
                "gate": _PHASE.id,
                "gate_state": "UNMET",
                "clauses_met": 7,
                "clauses_total": 10,
                "clauses_unmeasurable": 0,
                "met_ratio": 0.7,
                "blocked_by": None,
                # Every field below is the kind this page must not republish.
                "detail": "arm r:baseline scored 0.0731 against the population",
                "decision_id": _PHASE.tracker,
                "tracker": _PHASE.tracker,
                "tracker_url": "https://github.com/nousergon/alpha-engine-config/issues/9999",
                "read_on": DAY,
                "generated_utc": "2026-08-29T02:00:00Z",
            }
        ],
    }


def _board() -> dict:
    return {
        "trading_day": DAY,
        "generated_at": "2026-08-29T02:10:00Z",
        "rows": [
            {
                "id": "standing:human_touch_count",
                "source": "standing",
                "title": "Human mutating calls since the last human change",
                "state": "MET",
                "detail": "0 calls; last excess 88.7712 hours ago",
                "means_when_red": "somebody is operating this by hand",
                "artifact": "gates/phase2/2026-08-28/gate.json",
                "surface": "crucible/index",
                "last_read": DAY,
            },
            {
                "id": "phase:phase2",
                "source": "phase",
                "title": "Phase 2",
                "state": "UNMET",
                "detail": "7 of 10",
                "means_when_red": "x",
                "artifact": LADDER_KEY,
                "surface": "crucible/index",
            },
        ],
    }


def _cycle(slot: str, status: str = "held") -> dict:
    return {
        "schema_version": 1,
        "slot": slot,
        "slot_kind": "research",
        "benchmark": "population",
        "as_of": DAY,
        "scored_arms": [f"{slot}:baseline:abc", f"{slot}:challenger:def"],
        "active_arms": [f"{slot}:baseline:abc", f"{slot}:challenger:def"],
        "ladders": [{"arm_id": f"{slot}:baseline:abc", "score": 0.0731, "n": 63}],
        "ranking": {"order": [f"{slot}:baseline:abc"], "score": 1.9982},
        "decision": {
            "slot": slot,
            "as_of": DAY,
            "incumbent": f"{slot}:baseline:abc",
            "champion": f"{slot}:baseline:abc",
            "moved": False,
            "status": status,
            "reason": "incumbent held: challenger excess return -0.4413 did not clear",
            "comparisons": [],
            "ineligible": {},
        },
        "retirements": [],
    }


def _pointer(slot: str) -> dict:
    return {
        "schema_version": "champion_pointer.v1",
        "slot": slot,
        "arm_id": f"{slot}:baseline:abc",
        "as_of": DAY,
        "decided_at": "2026-08-29T02:00:00Z",
    }


def _attribution() -> dict:
    return {
        "trading_day": DAY,
        "rows": [
            {
                "name": "portfolio_excess_return_s_ratio",
                "value": 0.0731,
                "unit": "ratio",
                "status": "OK",
                "status_reason": "excess return 0.0731 over the benchmark",
                "source_path": "/crucible/console",
            },
            {
                "name": "signal_rank_ic_r",
                "value": -0.4413,
                "unit": "ic",
                "status": "BREACH",
                "status_reason": "rank ic -0.4413 below the floor",
                "source_path": "/crucible/console",
            },
        ],
    }


_HEADLINE = (
    "<b>CRUCIBLE V2 — 2026-08-28</b>\n"
    "Phase 0  MET          5/5\n"
    "Phase 2  UNMET        7/10\n"
    "acceptance: 18 of 24 clauses\n"
    "moved since 2026-08-27: 1\n"
    "pending operator action: apply the crucible-v2 stack from /crucible/console\n"
    '<a href="https://github.com/nousergon/alpha-engine-config/issues/9999">Full update</a>'
    ' · <a href="https://example.invalid/board">Board</a>'
)


@pytest.fixture
def store(tmp_path) -> LocalStore:
    """A store carrying every artifact the public page reads, each one stuffed
    with a figure, a link or an infrastructure identifier the page must drop."""
    s = LocalStore(tmp_path)
    s.put_bytes(LADDER_KEY, json.dumps(_ladder()).encode())
    s.put_bytes(BOARD_CURRENT_KEY, json.dumps(_board()).encode())
    s.put_bytes(attribution_key(DAY), json.dumps(_attribution()).encode())
    for slot in SLOTS:
        s.put_bytes(arena_cycle_key(slot, DAY), json.dumps(_cycle(slot)).encode())
        s.put_bytes(champion_key(slot), json.dumps(_pointer(slot)).encode())
    s.put_bytes(morning_report_key(DAY, "2026-08-29"), _HEADLINE.encode())
    return s


def _page(store: LocalStore):
    return build_public_page(store, now=SATURDAY_NIGHT)


class TestNothingLeaks:
    """The property the module exists for. Every assertion is a refusal."""

    @pytest.mark.parametrize("figure", PLANTED_FIGURES)
    def test_no_planted_figure_reaches_either_output(self, store, figure) -> None:
        page = _page(store)
        rendered = render_public_html(page)
        assert figure not in rendered
        assert figure not in page.to_json().decode()

    def test_no_forbidden_field_name_is_a_key_in_the_published_json(self, store) -> None:
        # Structural, not textual: a future edit that reaches for `row["value"]`
        # puts the KEY `value` somewhere in this document, and that is what
        # fails here — before anyone has to notice the number it carried.
        document = json.loads(_page(store).to_json())

        def keys(node) -> set[str]:
            if isinstance(node, dict):
                return set(node) | {k for v in node.values() for k in keys(v)}
            if isinstance(node, list):
                return {k for v in node for k in keys(v)}
            return set()

        assert keys(document) & FORBIDDEN_SOURCE_FIELDS == set()

    def test_the_page_carries_no_link_at_all(self, store) -> None:
        # Not "no private link": no link. The page is self-contained, so there
        # is no allowlist to get wrong and no third party between this
        # system's claims and its reader.
        assert "<a " not in render_public_html(_page(store))
        assert "http" not in render_public_html(_page(store))

    def test_a_headline_whose_link_filter_fails_is_withheld_not_trimmed(
        self, store, tmp_path
    ) -> None:
        # The drop filter matches `<a ` and `https?://`; a scheme-less link
        # (`www.…`, which Telegram autolinks) passes it and is caught by the
        # refusal instead. That is the backstop firing, and it is visible on
        # the page rather than silently trimmed.
        store.put_bytes(
            morning_report_key(DAY, "2026-08-29"),
            b"CRUCIBLE V2 2026-08-28 see www.nousergon.ai/board",
        )
        page = _page(store)
        assert page.headline == []
        assert "still carries a URL" in (page.headline_fault or "")
        assert "nousergon.ai/board" not in render_public_html(page)

    def test_a_headline_that_is_only_links_is_withheld_and_says_why(self, store) -> None:
        store.put_bytes(
            morning_report_key(DAY, "2026-08-29"),
            b'<a href="https://github.com/nousergon/x/issues/1">Full update</a>',
        )
        page = _page(store)
        assert page.headline == []
        assert "no publishable line" in (page.headline_fault or "")
        assert "github.com" not in render_public_html(page)

    def test_the_operator_action_line_is_dropped(self, store) -> None:
        page = _page(store)
        assert page.headline
        assert not any("pending operator action" in line for line in page.headline)

    def test_a_graded_layers_status_is_published_and_its_value_is_not(self, store) -> None:
        page = _page(store)
        assert {row["layer"] for row in page.integrity} == {
            "portfolio_excess_return_s_ratio",
            "signal_rank_ic_r",
        }
        assert {row["status"] for row in page.integrity} == {"OK", "BREACH"}
        assert all(set(row) == {"layer", "status"} for row in page.integrity)

    def test_only_standing_board_rows_are_published(self, store) -> None:
        page = _page(store)
        assert [row["id"] for row in page.slo] == ["standing:human_touch_count"]


class TestWhatItShows:
    def test_the_ladder_renders_state_and_clause_counts(self, store) -> None:
        page = _page(store)
        assert page.phases == [
            {
                "number": _PHASE.number,
                "title": _PHASE.title,
                "state": "UNMET",
                "clauses_met": 7,
                "clauses_total": 10,
                "artifact": LADDER_KEY,
            }
        ]
        assert "7/10" in render_public_html(page)

    def test_every_slot_gets_a_row_even_with_no_cycle(self, tmp_path) -> None:
        page = build_public_page(LocalStore(tmp_path), now=SATURDAY_NIGHT)
        assert [row["slot"] for row in page.slots] == list(SLOTS)
        assert all(row["verdict"] is None for row in page.slots)
        assert all(row["verdict_state"] == "UNREPORTED" for row in page.slots)

    def test_a_slot_row_carries_the_serving_arm_and_the_verdict(self, store) -> None:
        row = next(r for r in _page(store).slots if r["slot"] == "r")
        assert row["serving_arm"] == "r:baseline:abc"
        assert row["verdict"] == "held"
        assert row["moved"] is False
        assert row["active_arms"] == 2
        assert row["scored_arms"] == 2

    def test_the_json_twin_is_written_beside_the_html(self, store) -> None:
        keys = write_public_page(store, _page(store))
        assert keys == (PUBLIC_KEY, PUBLIC_JSON_KEY)
        assert b"<title>Crucible" in store.get_bytes(PUBLIC_KEY)
        assert json.loads(store.get_bytes(PUBLIC_JSON_KEY))["trading_day"] == DAY

    def test_the_page_says_out_loud_that_alpha_is_withheld_on_purpose(self, store) -> None:
        # Absence read as "there is none" is the misreading this line prevents.
        rendered = render_public_html(_page(store))
        assert "no return, no alpha, no score" in rendered


class TestAbsenceIsRendered:
    def test_an_empty_store_publishes_a_page_rather_than_raising(self, tmp_path) -> None:
        page = build_public_page(LocalStore(tmp_path), now=SATURDAY_NIGHT)
        rendered = render_public_html(page)
        assert "Crucible" in rendered
        assert page.phases == []
        assert LADDER_KEY in (page.phase_ladder_fault or "")
        assert {entry["key"] for entry in page.faults} >= {LADDER_KEY, BOARD_CURRENT_KEY}

    def test_an_unreadable_artifact_is_named_on_the_page(self, store) -> None:
        store.put_bytes(BOARD_CURRENT_KEY, b"{not json")
        page = _page(store)
        assert page.slo == []
        assert BOARD_CURRENT_KEY in render_public_html(page)

    def test_a_board_with_no_standing_rows_says_so_rather_than_rendering_empty(self, store) -> None:
        document = _board()
        document["rows"] = [row for row in document["rows"] if row["source"] != "standing"]
        store.put_bytes(BOARD_CURRENT_KEY, json.dumps(document).encode())
        page = _page(store)
        assert page.slo == []
        assert "predates" in (page.slo_fault or "")

    def test_a_stale_ladder_is_refused_not_rendered_as_todays(self, store) -> None:
        stale = _ladder()
        stale["trading_day"] = "2026-08-21"
        store.put_bytes(LADDER_KEY, json.dumps(stale).encode())
        page = _page(store)
        assert page.phases == []
        assert "stale" in (page.phase_ladder_fault or "")

    def test_a_cycle_with_no_decision_object_is_a_fault_not_a_blank_verdict(self, store) -> None:
        broken = _cycle("r")
        del broken["decision"]
        store.put_bytes(arena_cycle_key("r", DAY), json.dumps(broken).encode())
        page = _page(store)
        row = next(r for r in page.slots if r["slot"] == "r")
        assert row["verdict"] is None
        assert "carries no `decision` object" in (row["fault"] or "")

    def test_a_fault_is_recorded_once_per_key(self, store) -> None:
        store.put_bytes(champion_key("r"), b"nope")
        page = _page(store)
        keys = [entry["key"] for entry in page.faults]
        assert keys.count(champion_key("r")) == 1


class TestVerdictVocabularyIsClosed:
    def test_every_contract_status_has_a_public_wording_and_a_colour(self) -> None:
        # Read off the LIBRARY's own schema rather than restated here: a sixth
        # status added there must fail this test, which is the whole point of
        # asking the contract instead of a local tuple.
        from crucible.arena_io import load_arena_cycle_schema

        statuses = set(
            load_arena_cycle_schema()["properties"]["decision"]["properties"]["status"]["enum"]
        )
        assert statuses == set(VERDICT_LEGEND)
        assert statuses == set(VERDICT_STATE)
        assert set(VERDICT_STATE.values()) <= set(STATUS_COLORS)

    def test_an_unknown_verdict_is_refused_rather_than_rendered_blank(self) -> None:
        with pytest.raises(KeyError, match="no entry in VERDICT_LEGEND"):
            _verdict_label("promoted-sideways")


class TestReadFailuresAreReportedNotSwallowed:
    """The store itself failing, not the documents being wrong.

    Every branch here is a deliberate broad `except` in `crucible.console
    .public`, and a swallow nobody has made fire is a swallow nobody knows
    reports anything.
    """

    class _Exploding(LocalStore):
        """A store whose LISTING raises — the shape of an access failure."""

        def list_keys(self, prefix: str = ""):
            raise PermissionError(f"AccessDenied listing {prefix}")

    class _UnreadableText(LocalStore):
        """A store that lists a key and then cannot hand over its bytes."""

        def get_bytes(self, key: str) -> bytes:
            if key.endswith(".txt"):
                raise OSError(f"connection reset reading {key}")
            return super().get_bytes(key)

    def test_a_prefix_that_cannot_be_listed_is_named_not_reported_as_absent(self, tmp_path) -> None:
        page = build_public_page(self._Exploding(tmp_path), now=SATURDAY_NIGHT)
        assert page.headline == []
        fault = page.headline_fault or ""
        assert "could not be listed" in fault
        assert "PermissionError" in fault, "the fault must name the cause, not just the key"
        assert any("could not be listed" in entry["fault"] for entry in page.faults)

    def test_a_headline_that_cannot_be_read_is_named_not_reported_as_absent(self, tmp_path) -> None:
        store = self._UnreadableText(tmp_path)
        store.put_bytes(morning_report_key(DAY, "2026-08-29"), _HEADLINE.encode())
        page = build_public_page(store, now=SATURDAY_NIGHT)
        assert page.headline == []
        assert "could not be read" in (page.headline_fault or "")
        assert "OSError" in (page.headline_fault or "")

    def test_an_attribution_document_of_the_wrong_shape_is_a_named_fault(self, store) -> None:
        store.put_bytes(attribution_key(DAY), json.dumps({"rows": "five"}).encode())
        page = _page(store)
        assert page.integrity == []
        assert attribution_key(DAY) in (page.integrity_fault or "")

    def test_a_ladder_whose_phases_carry_a_non_object_skips_it_rather_than_raising(
        self, store
    ) -> None:
        from crucible.console.public import _phase_rows

        assert _phase_rows({"phases": ["not a row", {"number": 1}]}) == [
            {
                "number": 1,
                "title": None,
                "state": None,
                "clauses_met": None,
                "clauses_total": None,
                "artifact": LADDER_KEY,
            }
        ]
