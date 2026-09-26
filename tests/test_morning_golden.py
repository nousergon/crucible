"""Byte-identity of `report.morning`'s two documents over a captured real reading.

`alpha-engine-config-I10953` binding constraint 2: moving `crucible/morning.py`
onto `nousergon_lib.gates.report` must change NOTHING that reaches Brian — the
Telegram headline (`message.txt`) and the tracker comment (`update.md`) for a
fixed input are byte-identical before and after. This module is that proof, and
it is KEPT: the goldens were rendered by the pre-migration module, so any later
change to either document fails here and has to be made on purpose, by
regenerating the golden in the same diff that explains why.

**The input is a real reading, projected.** `reading_2026-09-24.json` is the
production store as `report.morning` would have read it for trading day
2026-09-24, captured read-only on 2026-09-25: `board/current.json`, the
previous session's board, the board run's manifest, `gates/ladder.json` and
`report/acceptance/2026-09-24.json`. The two boards are projected onto the
fields the renderers read (`id`, `source`, `state`, `detail`,
`component_state`, clause `name`/`met`, and the top-level counts and stamps) —
the live document is ~1 MB of provenance and per-row prose neither renderer
touches — and the run manifest onto `status`/`code_sha`/`reason`. The ladder
and acceptance documents are carried as read. Nothing in it is an
infrastructure identifier.

**Two scenarios.** `delivered` is the morning as it happened at the cron's own
instant. `degraded` is the same reading three days later with the previous
board absent and the acceptance read DENIED — so the stale headline, the
"cannot say" diff and the denied acceptance line, the three paths
`nousergon_lib.gates.report` now renders, are pinned too, not only the happy
one.

The store URI is fixed rather than the test's `tmp_path`, and the board link
goes to a console URL rather than a presigned `file://` path: both are
properties of where the test ran, and a golden that recorded them could never
be compared.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib

import pytest
from botocore.exceptions import ClientError

from crucible.gate import LADDER_KEY
from crucible.keys import (
    BOARD_CURRENT_KEY,
    acceptance_reading_key,
    board_key,
    manifest_key,
)
from crucible.morning import read_inputs, render_full_update, render_message
from crucible.store import LocalStore

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "morning"
READING = GOLDEN / "reading_2026-09-24.json"
DAY = dt.date(2026, 9, 24)
PREVIOUS = dt.date(2026, 9, 23)
#: `DELIVERY_CRON_UTC`'s instant on the morning after DAY.
DELIVERED_AT = dt.datetime(2026, 9, 25, 10, 0, tzinfo=dt.UTC)
STORE_URI = "s3://store/crucible"
CONSOLE_URL = "https://console.example"
UPDATE_URL = "https://github.com/nousergon/alpha-engine-config/issues/1#issuecomment-42"
HISTORY_URL = "https://github.com/nousergon/alpha-engine-config/issues/1"


class _DenyingStore(LocalStore):
    """Raises like an S3 caller denied `s3:ListBucket` on the named keys."""

    def __init__(self, root: pathlib.Path, *, denied: frozenset[str]) -> None:
        super().__init__(root)
        self._denied = denied

    def get_bytes(self, key: str) -> bytes:
        if key in self._denied:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": key}}, "GetObject")
        return super().get_bytes(key)


def _seed(root: pathlib.Path, *, previous: bool, denied: frozenset[str]) -> LocalStore:
    reading = json.loads(READING.read_text(encoding="utf-8"))
    store = _DenyingStore(root, denied=denied)
    documents = {
        BOARD_CURRENT_KEY: reading["current"],
        manifest_key("board", DAY.isoformat()): reading["board_run"],
        LADDER_KEY: reading["ladder"],
        acceptance_reading_key(DAY.isoformat()): reading["acceptance"],
    }
    if previous:
        documents[board_key(PREVIOUS.isoformat())] = reading["previous"]
    for key, document in documents.items():
        store.put_bytes(key, json.dumps(document, sort_keys=True).encode("utf-8"))
    return store


SCENARIOS: dict[str, dict] = {
    "delivered": {"now": DELIVERED_AT, "previous": True, "denied": frozenset()},
    "degraded": {
        "now": DELIVERED_AT + dt.timedelta(days=3),
        "previous": False,
        "denied": frozenset({acceptance_reading_key(DAY.isoformat())}),
    },
}


def render(scenario: str, root: pathlib.Path) -> tuple[str, str]:
    """`(message.txt, update.md)` for ``scenario``, exactly as the handler
    renders them — `read_inputs`, then the two renderers."""
    spec = SCENARIOS[scenario]
    store = _seed(root, previous=spec["previous"], denied=spec["denied"])
    inputs = read_inputs(store, trading_day=DAY, now=spec["now"], console_url=CONSOLE_URL)
    inputs = dataclasses.replace(inputs, board_uri=STORE_URI)
    update = render_full_update(inputs, now=spec["now"])
    message = render_message(
        inputs, now=spec["now"], update_url=UPDATE_URL, history_url=HISTORY_URL
    )
    return message, update


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_headline_is_byte_identical_to_the_golden(scenario: str, tmp_path) -> None:
    message, _ = render(scenario, tmp_path)
    golden = (GOLDEN / f"{scenario}.message.txt").read_bytes()
    assert message.encode("utf-8") == golden


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_full_update_is_byte_identical_to_the_golden(scenario: str, tmp_path) -> None:
    _, update = render(scenario, tmp_path)
    golden = (GOLDEN / f"{scenario}.update.md").read_bytes()
    assert update.encode("utf-8") == golden


def test_the_scenarios_exercise_the_paths_they_claim_to(tmp_path) -> None:
    """A golden over a reading that never reaches a path pins nothing about
    it. The degraded scenario must actually render stale, cannot-say and
    denied; the delivered one must render a real diff and a real count."""
    delivered_message, delivered_update = render("delivered", tmp_path / "a")
    degraded_message, degraded_update = render("degraded", tmp_path / "b")
    assert "STALE BOARD" not in delivered_update
    assert "cannot say" not in delivered_update
    assert "acceptance count: 22 met" in delivered_message
    assert degraded_update.startswith("**STALE BOARD: ")
    assert f"cannot say — the {PREVIOUS} board is absent at" in degraded_update
    assert "acceptance count: unreadable (AccessDenied)" in degraded_message
