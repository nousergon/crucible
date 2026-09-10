"""A deliberate exercise does not wake a human and does not spend the budget.

Brian's ruling, 2026-09-09 — `alpha-engine-config-I10366`, option (b):
"route synthetic pages to the muted topic and exclude them from
`pages_within_ceiling`", *executed after 2026-09-19*.

Three properties are asserted here and each one is load-bearing:

1. **The activation is structural.** Phase 2's two live first-attempt
   Saturdays are 2026-09-12 and 2026-09-19, and four `crucible.gate` clauses
   are graded across that window. Before
   :data:`crucible.synthetic.SYNTHETIC_ROUTING_ACTIVE_FROM` the code must
   behave EXACTLY as it did before this change — not "close enough", and not
   "unless somebody flips a label". The pre-date tests below assert that a
   store containing synthetic rows grades **identically** to a store whose
   same rows carry no `synthetic` field at all: if the pre-date path ignored
   the field in any way other than completely, those two readings would
   differ.
2. **Suppression is a DELIVERY decision, never a recording one**
   (`observability-policy.md` §7.2a). Post-activation, a synthetic page still
   reaches a transport, still lands on a durable topic, and still writes its
   bus row with `synthetic`, `condition`, `rendered` and `members` intact.
   The only thing that changes is WHICH topic and WHETHER it counts.
3. **The exclusion reads the row's own `synthetic` field** — the one
   `crucible.synthetic` derives from `run_mode` and `fault_capability_class`
   on the invocation (crucible-PR188) — never a second derivation from a job
   name, a reason string or a destination.

`SYNTHETIC_ROUTING_ACTIVE_FROM` is monkeypatched rather than the clock: the
real :func:`crucible.synthetic.synthetic_routing_active` is what every test
below exercises, so a change to its comparison is caught here.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest

from crucible import gate, synthetic
from crucible.alerts import (
    DESTINATION_MUTED,
    Page,
    bus_row,
    emit,
    group_pages,
    pages_in_range,
    pages_in_window,
)
from crucible.store import LocalStore
from crucible.synthetic import SYNTHETIC_ROUTING_ACTIVE_FROM, synthetic_routing_active

#: The two live first-attempt Saturdays phase 2 is graded on. Neither may see
#: any of this change.
LIVE_SATURDAYS = (dt.date(2026, 9, 12), dt.date(2026, 9, 19))

DAY = dt.date(2026, 9, 4)

#: Fixture topics. Never the real names — `tests/test_no_infra_literals.py`
#: forbids a topic name anywhere in this tree, and the resolver only requires
#: that the ARN's last segment equals the configured NAME.
PAGES_NAME = "fixture-pages"
MUTED_NAME = "fixture-pages-muted"
PAGES_ARN = f"arn:aws:sns:us-east-1:fixture:{PAGES_NAME}"
MUTED_ARN = f"arn:aws:sns:us-east-1:fixture:{MUTED_NAME}"


@pytest.fixture(autouse=True)
def _topics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUCIBLE_PAGES_TOPIC", PAGES_NAME)
    monkeypatch.setenv("CRUCIBLE_MUTED_TOPIC", MUTED_NAME)
    monkeypatch.setenv("CRUCIBLE_PAGES_TOPIC_ARN", PAGES_ARN)
    monkeypatch.setenv("CRUCIBLE_MUTED_TOPIC_ARN", MUTED_ARN)


def _activate(monkeypatch: pytest.MonkeyPatch, *, active: bool) -> None:
    """Put the module either side of its own activation date.

    The DATE moves, not the clock: `synthetic_routing_active()` reads the
    constant at call time, so this exercises the real comparison.
    """
    monkeypatch.setattr(
        synthetic,
        "SYNTHETIC_ROUTING_ACTIVE_FROM",
        dt.date(2000, 1, 1) if active else dt.date(2999, 1, 1),
    )


class _Transport:
    """Records every publish and reports no destination of its own.

    Reporting none is the point: `crucible.alerts._transport_outcome` then
    falls through to the destination the ROUTING implies, which is what a real
    SNS-only publish to the muted topic does and what the bus row must record.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, message: str, **kwargs: Any) -> Any:
        self.calls.append({"message": message, **kwargs})
        return _Ok()


class _Ok:
    any_ok = True
    dedup_skipped = False
    muted = False
    telegram_destination = None
    destination = None


def _synthetic_group():
    return group_pages(
        [
            Page(
                condition="failure",
                job="data.weekly",
                trading_day=DAY,
                reason="MissingSourceError: the injected fault",
                run_id="01M23N1HSW5DPJ9S4M0E1W0ZPK",
                synthetic="replay; fault-injected: chaos_probe",
            )
        ]
    )


def _real_group():
    return group_pages(
        [
            Page(
                condition="failure",
                job="data.daily",
                trading_day=DAY,
                reason="ConnectionError: the vendor was down",
                run_id="01M23N1HSW5DPJ9S4M0E1W0ZPL",
            )
        ]
    )


# ── 1. The activation itself ───────────────────────────────────────────────


class TestTheActivationDate:
    def test_the_constant_is_the_day_after_the_second_live_saturday(self) -> None:
        assert SYNTHETIC_ROUTING_ACTIVE_FROM == dt.date(2026, 9, 20)
        assert SYNTHETIC_ROUTING_ACTIVE_FROM > LIVE_SATURDAYS[-1]

    @pytest.mark.parametrize("day", LIVE_SATURDAYS)
    def test_neither_live_saturday_is_active(self, day: dt.date) -> None:
        assert synthetic_routing_active(on=day) is False

    def test_the_boundary_is_inclusive_on_the_activation_day(self) -> None:
        assert synthetic_routing_active(on=dt.date(2026, 9, 19)) is False
        assert synthetic_routing_active(on=dt.date(2026, 9, 20)) is True

    def test_a_datetime_is_normalised_to_utc(self) -> None:
        eve = dt.datetime(2026, 9, 20, 3, 0, tzinfo=dt.timezone(dt.timedelta(hours=10)))
        # 2026-09-19T17:00Z — still the 19th in UTC, so still inactive.
        assert synthetic_routing_active(on=eve) is False


# ── 2. Routing: the delivery decision, and only that ───────────────────────


class TestRouting:
    def test_before_activation_a_synthetic_page_goes_to_the_pages_topic(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _activate(monkeypatch, active=False)
        transport = _Transport()
        emit(
            LocalStore(tmp_path),
            _synthetic_group(),
            sweep_run_id="0" * 26,
            transport=transport,
            now=dt.datetime(2026, 9, 12, 12, 0, tzinfo=dt.UTC),
        )
        assert [c["sns_topic_arn"] for c in transport.calls] == [PAGES_ARN]

    def test_after_activation_a_synthetic_page_goes_to_the_muted_topic(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _activate(monkeypatch, active=True)
        transport = _Transport()
        emit(
            LocalStore(tmp_path),
            _synthetic_group(),
            sweep_run_id="0" * 26,
            transport=transport,
            now=dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC),
        )
        assert [c["sns_topic_arn"] for c in transport.calls] == [MUTED_ARN]

    def test_a_real_page_is_never_re_routed(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole risk of this change, asserted against."""
        _activate(monkeypatch, active=True)
        transport = _Transport()
        emit(
            LocalStore(tmp_path),
            _real_group(),
            sweep_run_id="0" * 26,
            transport=transport,
            now=dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC),
        )
        assert [c["sns_topic_arn"] for c in transport.calls] == [PAGES_ARN]

    def test_the_source_still_names_it_synthetic(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """crucible-PR188's marking is not regressed by the routing."""
        _activate(monkeypatch, active=True)
        transport = _Transport()
        emit(
            LocalStore(tmp_path),
            _synthetic_group(),
            sweep_run_id="0" * 26,
            transport=transport,
            now=dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC),
        )
        assert transport.calls[0]["source"] == "crucible-v2/synthetic/failure"
        assert transport.calls[0]["message"].startswith("[crucible-v2] SYNTHETIC")


class TestSuppressionIsADeliveryDecisionNeverARecordingOne:
    """`observability-policy.md` §7.2a, asserted as a property of the store."""

    def _emitted_row(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        _activate(monkeypatch, active=True)
        store = LocalStore(tmp_path)
        transport = _Transport()
        keys = emit(
            store,
            _synthetic_group(),
            sweep_run_id="0" * 26,
            transport=transport,
            now=dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC),
        )
        assert transport.calls, "a muted page is still PUBLISHED — routing is not silence"
        return json.loads(store.get_bytes(keys[0]).decode())

    def test_the_bus_row_is_still_written(self, tmp_path, monkeypatch) -> None:
        row = self._emitted_row(tmp_path, monkeypatch)
        assert row["condition"] == "failure"
        assert row["synthetic"] == "replay; fault-injected: chaos_probe"

    def test_the_rendered_page_is_still_recorded(self, tmp_path, monkeypatch) -> None:
        row = self._emitted_row(tmp_path, monkeypatch)
        assert "SYNTHETIC" in row["rendered"]
        assert row["members"][0]["job"] == "data.weekly"

    def test_the_row_records_the_muted_destination_honestly(self, tmp_path, monkeypatch) -> None:
        """Not "sent to the operator" and not a missing field: `muted`."""
        row = self._emitted_row(tmp_path, monkeypatch)
        assert row["destination"] == DESTINATION_MUTED


# ── 3. The ceiling exclusion, read off PR188's field ───────────────────────


def _write_row(store: LocalStore, *, day: dt.date, job: str, marker: str | None) -> str:
    page = Page(
        condition="failure",
        job=job,
        trading_day=day,
        reason="the reason",
        run_id="01M23N1HSW5DPJ9S4M0E1W0ZP" + job[-1].upper(),
        synthetic=marker,
    )
    group = group_pages([page])[0]
    from crucible.alerts import bus_key

    key = bus_key(group)
    row = bus_row(
        group,
        alert_id=None,
        sent=True,
        destination="operator_chat",
        first_observed_utc=f"{day.isoformat()}T01:00:00Z",
        last_observed_utc=f"{day.isoformat()}T01:00:00Z",
    )
    store.put_bytes(key, json.dumps(row, indent=2, sort_keys=True).encode())
    return key


class TestTheExclusionReadsTheRowsOwnField:
    def test_after_activation_a_synthetic_row_leaves_the_count(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _activate(monkeypatch, active=True)
        store = LocalStore(tmp_path)
        _write_row(store, day=DAY, job="data.weekly", marker="replay")
        real = _write_row(store, day=DAY, job="data.daily", marker=None)
        assert pages_in_range(store, start=DAY, end=DAY) == [real]

    def test_before_activation_it_counts_exactly_as_today(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _activate(monkeypatch, active=False)
        store = LocalStore(tmp_path)
        one = _write_row(store, day=DAY, job="data.weekly", marker="replay")
        two = _write_row(store, day=DAY, job="data.daily", marker=None)
        assert pages_in_range(store, start=DAY, end=DAY) == sorted([one, two])

    def test_a_row_that_cannot_be_read_stays_in_the_ceiling(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The safe direction: unreadable is REAL, never quietly excluded."""
        _activate(monkeypatch, active=True)
        store = LocalStore(tmp_path)
        key = _write_row(store, day=DAY, job="data.weekly", marker="replay")
        store.put_bytes(key, b"{ not json")
        assert pages_in_range(store, start=DAY, end=DAY) == [key]

    def test_the_window_metric_moves_with_the_gate(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One ceiling, one reading — `pages_per_20_trading_days` and the
        clause may never disagree about what a synthetic row is."""
        _activate(monkeypatch, active=True)
        store = LocalStore(tmp_path)
        _write_row(store, day=DAY, job="data.weekly", marker="replay")
        _write_row(store, day=DAY, job="data.daily", marker=None)
        now = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC)
        assert pages_in_window(store, now=now) == 1
        assert pages_in_window(store, now=now, exclude_synthetic=False) == 2


# ── 4. Byte-identical before the date ──────────────────────────────────────


def _sweep_manifest(store: LocalStore) -> None:
    from crucible.keys import manifest_key

    store.put_bytes(
        manifest_key("alerts.sweep", DAY.isoformat()),
        json.dumps({"status": "ok", "job": "alerts.sweep"}).encode(),
    )


def _clause_tuple(clause) -> tuple:
    return (clause.name, clause.requirement, clause.met, clause.detail, clause.evidence)


class TestThePreDateReadingIsIdenticalToIgnoringTheFieldEntirely:
    """The guarantee the phase-2 window needs, asserted as an equality.

    Two stores holding byte-identical rows under byte-identical KEYS, with
    exactly one difference: the second has the `synthetic` field deleted from
    the row document. Before the activation date the two must grade
    IDENTICALLY on every field of the clause — name, requirement, met, detail
    and evidence. If any part of the pre-date path consulted the field, these
    would differ.

    The field is stripped rather than the row re-derived from an unmarked
    `Page`, deliberately: an unmarked page also gets a different `cause_key`
    (crucible-PR188's `synthetic.` prefix), and comparing against THAT would
    grade PR188's change instead of this one. Keys are held constant so the
    only variable is the field these clauses now read.
    """

    def _stores(self, tmp_path) -> tuple[LocalStore, LocalStore]:
        marked, plain = LocalStore(tmp_path / "a"), LocalStore(tmp_path / "b")
        for store in (marked, plain):
            _sweep_manifest(store)
            key = _write_row(store, day=DAY, job="data.weekly", marker="replay")
            _write_row(store, day=DAY, job="data.daily", marker=None)
            if store is plain:
                row = json.loads(store.get_bytes(key).decode())
                del row["synthetic"]
                for member in row["members"]:
                    member.pop("synthetic", None)
                store.put_bytes(key, json.dumps(row, indent=2, sort_keys=True).encode())
        return marked, plain

    def test_pages_within_ceiling(self, tmp_path, monkeypatch) -> None:
        _activate(monkeypatch, active=False)
        marked, plain = self._stores(tmp_path)
        window = [DAY]
        assert _clause_tuple(gate._clause_pages_within_ceiling(marked, window)) == _clause_tuple(
            gate._clause_pages_within_ceiling(plain, window)
        )

    def test_pages_commissioned(self, tmp_path, monkeypatch) -> None:
        _activate(monkeypatch, active=False)
        marked, plain = self._stores(tmp_path)
        assert _clause_tuple(gate._clause_pages_commissioned(marked)) == _clause_tuple(
            gate._clause_pages_commissioned(plain)
        )

    def test_and_after_the_date_they_diverge(self, tmp_path, monkeypatch) -> None:
        """The change is real — the equality above is a date, not a no-op."""
        _activate(monkeypatch, active=True)
        marked, plain = self._stores(tmp_path)
        window = [DAY]
        assert _clause_tuple(gate._clause_pages_within_ceiling(marked, window)) != _clause_tuple(
            gate._clause_pages_within_ceiling(plain, window)
        )


class TestASyntheticRowCommissionsNothing:
    """`alpha-engine-config-I10366`'s Delta: (b) must not turn a red clause
    green by moving rows out of its view."""

    def test_a_condition_whose_only_row_is_synthetic_is_not_commissioned(
        self, tmp_path, monkeypatch
    ) -> None:
        _activate(monkeypatch, active=True)
        store = LocalStore(tmp_path)
        _sweep_manifest(store)
        _write_row(store, day=DAY, job="data.weekly", marker="replay")
        clause = gate._clause_pages_commissioned(store)
        assert clause.met is False
        assert "has never fired on the real system" in clause.detail
        assert "commission nothing" in clause.detail

    def test_the_synthetic_row_is_named_not_hidden(self, tmp_path, monkeypatch) -> None:
        """ "Never fired" and "fired only in a drill" are different readings."""
        _activate(monkeypatch, active=True)
        store = LocalStore(tmp_path)
        _sweep_manifest(store)
        _write_row(store, day=DAY, job="data.weekly", marker="replay")
        detail = gate._clause_pages_commissioned(store).detail
        assert "synthetic row(s)" in detail
        assert "alerts/" in detail
