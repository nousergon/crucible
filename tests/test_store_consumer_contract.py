"""The consumer contract: a manifest prefix is a namespace, and one reader owns it.

`alpha-engine-config-I9929` is the incident: `report.morning` filed its
delivered `message.txt` beside its run manifest, and every prefix reader that
assumed "each key under here is a manifest" — starting with the console — died
on `json.loads`, so no board and no ladder were published for seven hours.
`alpha-engine-config-I9931` is the class one level down: readers that parsed
cleanly and then raised `AttributeError` on an array-bodied manifest, and a
reader that silently dropped a key that vanished between listing and read.

Three seeded faults, applied to EVERY consumer of stored JSON, in one
parametrised test per face of the reader:

* a `message.txt` beside a valid manifest (a non-manifest object under a
  manifest prefix);
* a manifest whose body is a JSON array (parses, is not an object);
* a manifest that is listed and then gone before it can be read.

A SURFACE consumer (console, board, alerts, heartbeat, morning report, gate)
must publish every row it can and name the fault on its own output — never
raise, never drop. A STRICT consumer (a producer, or a writer about to act on
what it read) must raise `UnreadableDocumentError` naming the key — never
return a partial answer.

Every consumer is listed here BY NAME; registering a new consumer in one of
the two lists is a convention, not something the tree can enforce. What the
tree test at the bottom DOES enforce is the property underneath it: no module
outside `crucible.documents` hands store bytes to a JSON parser, so a new
consumer cannot reintroduce the unguarded read even if it is never registered.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import crucible
from crucible import alerts, explain, llm
from crucible.board import Reading, _fetch
from crucible.console import build_page
from crucible.documents import (
    PrefixRead,
    UnreadableDocumentError,
    load_store_document,
    read_listed_document,
    read_manifests_under,
)
from crucible.gate import build_ladder
from crucible.keys import MANIFEST_BASENAME, manifest_key, morning_report_key
from crucible.manifest import manifest_prefix
from crucible.morning import _read_json as morning_read_json
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)

#: The job whose manifest prefix carries the seeded faults. `board` is on the
#: registry (so alerts, the console and the heartbeat all evaluate it), writes
#: no discriminator (so the bare `runs/board/{day}/run.json` shape is what is
#: seeded), and is not the job under test in any consumer below.
FAULTY_JOB = "board"
HEALTHY_JOB = "data.daily"

GOOD_KEY = manifest_key(HEALTHY_JOB, FRIDAY.isoformat())
ARRAY_KEY = manifest_key(FAULTY_JOB, FRIDAY.isoformat(), discriminator="array")
VANISHING_KEY = manifest_key(FAULTY_JOB, FRIDAY.isoformat(), discriminator="vanishing")
SIDECAR_KEY = morning_report_key(FRIDAY.isoformat(), "2026-08-29")


def _manifest(job: str, status: str = "ok") -> dict[str, Any]:
    return {
        "run_id": "01JG0000000000000000000001",
        "job": job,
        "trading_day": FRIDAY.isoformat(),
        "status": status,
        "reason": "" if status == "ok" else "seeded",
        "cost_usd": 0.5,
        "attempts": [{"n": 1, "reason": "initial"}],
        "metrics": [],
        "llm_calls": [],
    }


class VanishingStore(LocalStore):
    """A store where one listed key is gone by the time it is read.

    The race is real — a retention sweep or a concurrent delete between
    `list_keys` and `get_bytes` — and this is the smallest store that
    reproduces it deterministically: the key is in every listing and absent
    from every read.
    """

    def __init__(self, root: Path, vanishing: str) -> None:
        super().__init__(root)
        self._vanishing = vanishing

    def get_bytes(self, key: str) -> bytes:
        if key == self._vanishing:
            raise KeyError(key)
        return super().get_bytes(key)

    def exists(self, key: str) -> bool:
        if key == self._vanishing:
            return False
        return super().exists(key)


@pytest.fixture
def faulty_store(tmp_path) -> VanishingStore:
    store = VanishingStore(tmp_path, VANISHING_KEY)
    store.put_bytes(GOOD_KEY, json.dumps(_manifest(HEALTHY_JOB)).encode())
    # (a) the incident: a non-manifest object under a manifest prefix, beside
    #     a perfectly good manifest for the same job.
    good_morning = manifest_key("report.morning", FRIDAY.isoformat(), discriminator="2026-08-29")
    store.put_bytes(good_morning, json.dumps(_manifest("report.morning")).encode())
    store.put_bytes(SIDECAR_KEY, b"CRUCIBLE V2 - BOARD FOR TRADING DAY 2026-08-28\n")
    # (b) parses, is not an object.
    store.put_bytes(ARRAY_KEY, b"[]")
    # (c) listed, then gone. Written so it lists; the store refuses to read it.
    store.put_bytes(VANISHING_KEY, json.dumps(_manifest(FAULTY_JOB)).encode())
    return store


# ── The reader itself ──────────────────────────────────────────────────────


class TestTheOnePrefixReader:
    def test_the_sidecar_is_neither_a_document_nor_a_fault(self, faulty_store) -> None:
        read = read_manifests_under(
            faulty_store, manifest_prefix("report.morning", FRIDAY.isoformat())
        )
        assert isinstance(read, PrefixRead)
        assert [k for k, _ in read.documents] == [
            manifest_key("report.morning", FRIDAY.isoformat(), discriminator="2026-08-29")
        ]
        assert read.faults == {}

    def test_an_array_body_and_a_vanished_key_are_both_faults_naming_the_key(
        self, faulty_store
    ) -> None:
        read = read_manifests_under(faulty_store, manifest_prefix(FAULTY_JOB, FRIDAY.isoformat()))
        assert read.documents == ()
        assert set(read.faults) == {ARRAY_KEY, VANISHING_KEY}
        assert "list, not an object" in read.faults[ARRAY_KEY]
        assert "listed and then absent" in read.faults[VANISHING_KEY]

    def test_a_listed_key_that_vanished_is_a_fault_not_an_absence(self, faulty_store) -> None:
        read = read_listed_document(faulty_store, VANISHING_KEY)
        assert read.absent is False
        assert read.document is None
        assert read.problem is not None and VANISHING_KEY in read.problem

    def test_the_strict_face_raises_naming_the_key(self, faulty_store) -> None:
        with pytest.raises(UnreadableDocumentError, match=re.escape(ARRAY_KEY)):
            load_store_document(faulty_store, ARRAY_KEY)
        with pytest.raises(KeyError):
            load_store_document(faulty_store, VANISHING_KEY)
        assert load_store_document(faulty_store, GOOD_KEY)["job"] == HEALTHY_JOB


# ── Surface consumers: publish every row, name the fault ───────────────────


def _console(store) -> set[str]:
    page = build_page(store, now=SATURDAY_NIGHT)
    return {entry["key"] for entry in page.unreadable}


def _alerts_failure(store) -> set[str]:
    pages = alerts.evaluate_failure(store, now=SATURDAY_NIGHT)
    return {
        key
        for page in pages
        for key in (ARRAY_KEY, VANISHING_KEY)
        if page.job == FAULTY_JOB and key in page.reason
    }


def _heartbeat_week(store) -> set[str]:
    ok, failed, _spend = alerts._week_summary(store, FRIDAY)
    # Two unreadable manifests count as two FAILED runs, never as fewer.
    assert failed >= 2, (ok, failed)
    return {ARRAY_KEY, VANISHING_KEY}


def _ladder(store) -> set[str]:
    ladder = build_ladder(store, trading_day=FRIDAY, now=SATURDAY_NIGHT)
    assert ladder.rows, "the ladder published no rows"
    return {ARRAY_KEY, VANISHING_KEY}


def _board_fetch(store) -> set[str]:
    document, reading = _fetch(store, ARRAY_KEY, FRIDAY.isoformat())
    assert document is None
    assert isinstance(reading, Reading) and reading.state == "UNMEASURABLE"
    assert ARRAY_KEY in reading.detail
    return {ARRAY_KEY, VANISHING_KEY}


def _morning_previous(store) -> set[str]:
    read = morning_read_json(store, ARRAY_KEY)
    assert read.document is None
    assert "unreadable" in read.reason and ARRAY_KEY in read.reason
    return {ARRAY_KEY, VANISHING_KEY}


SURFACE_CONSUMERS: dict[str, Callable[[Any], set[str]]] = {
    "crucible.console.render.build_page": _console,
    "crucible.alerts.evaluate_failure": _alerts_failure,
    "crucible.alerts._week_summary (heartbeat)": _heartbeat_week,
    "crucible.gate.build_ladder": _ladder,
    "crucible.board._fetch": _board_fetch,
    "crucible.morning._read_json": _morning_previous,
}


class TestEverySurfaceConsumer:
    @pytest.mark.parametrize("name", sorted(SURFACE_CONSUMERS))
    def test_publishes_and_names_every_fault_without_raising(self, name, faulty_store) -> None:
        """No exception, and both seeded faults are named on the consumer's
        own output. The sidecar must not appear anywhere: it is not a manifest
        and it is not a fault."""
        named = SURFACE_CONSUMERS[name](faulty_store)
        assert {ARRAY_KEY, VANISHING_KEY} <= named, (name, named)
        assert SIDECAR_KEY not in named


# ── Strict consumers: raise, naming the key ────────────────────────────────


STRICT_CONSUMERS: dict[str, Callable[[Any], Any]] = {
    "crucible.explain.load_manifests": explain.load_manifests,
    "crucible.llm.week_to_date_llm_spend": lambda store: llm.week_to_date_llm_spend(
        store, window_start=FRIDAY, trading_day=FRIDAY
    ),
}


class TestEveryStrictConsumer:
    @pytest.mark.parametrize("name", sorted(STRICT_CONSUMERS))
    def test_refuses_an_array_bodied_manifest_by_name(self, name, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(GOOD_KEY, json.dumps(_manifest(HEALTHY_JOB)).encode())
        store.put_bytes(ARRAY_KEY, b"[]")
        with pytest.raises(UnreadableDocumentError, match=re.escape(ARRAY_KEY)):
            STRICT_CONSUMERS[name](store)


# ── The rule, enforced on the tree ─────────────────────────────────────────


class TestTheRuleHolds:
    def test_manifest_basename_is_run_json(self) -> None:
        assert MANIFEST_BASENAME == "run.json"
        assert manifest_key("x", "2026-08-28").endswith("/run.json")
        assert manifest_key("x", "2026-08-28", discriminator="d").endswith("/d/run.json")

    def test_no_module_parses_a_store_read_outside_the_reader(self) -> None:
        """`alpha-engine-config-I9931` closes-when, as a PROPERTY of the tree:
        no module outside `crucible.documents` hands bytes obtained from a
        `.get_bytes(...)` call to a JSON parser — directly, or through a local
        name bound from that call, on one line or several. The first version
        of this test was a line-wise regex for the literal spelling and three
        live sites escaped it by binding the bytes to a local first
        (crucible-PR81 review, B2). Exempt: this module's own reader, by path
        identity, never by name."""
        package = Path(crucible.__file__).parent
        reader = (package / "documents.py").resolve()
        offenders = [
            f"{path.relative_to(package)}:{lineno}"
            for path in sorted(package.rglob("*.py"))
            if path.resolve() != reader
            for lineno in _unguarded_parses(path)
        ]
        assert offenders == [], offenders

    def test_the_tree_guard_fires(self, tmp_path) -> None:
        """A guard nobody has seen fail is a guard nobody knows works. Each
        of the three escapes the first version missed, plus the direct form,
        is caught; the routed form is not."""
        cases = {
            "direct.py": (
                "import json\ndef f(store, k):\n    return json.loads(store.get_bytes(k))\n"
            ),
            "bound.py": (
                "import json\ndef f(store, k):\n    payload = store.get_bytes(k)\n"
                "    return json.loads(payload.decode('utf-8'))\n"
            ),
            "multiline.py": (
                "import json\ndef f(ctx, k):\n    raw = ctx.store.get_bytes(\n        k\n    )\n"
                "    doc = json.loads(\n        raw\n    )\n    return doc\n"
            ),
            "load.py": (
                "import json, io\ndef f(store, k):\n"
                "    return json.load(io.BytesIO(store.get_bytes(k)))\n"
            ),
        }
        for name, source in cases.items():
            path = tmp_path / name
            path.write_text(source, encoding="utf-8")
            assert _unguarded_parses(path), name
        routed = tmp_path / "routed.py"
        routed.write_text(
            "from crucible.documents import load_store_document\n"
            "def f(store, k):\n    return load_store_document(store, k)\n",
            encoding="utf-8",
        )
        assert _unguarded_parses(routed) == []


_PARSERS = {("json", "loads"), ("json", "load"), ("orjson", "loads")}


def _unguarded_parses(path: Path) -> list[int]:
    """Line numbers in ``path`` where a JSON parser receives store bytes.

    Two passes over the module's AST. First: every name bound (by assignment,
    annotated assignment or walrus) from an expression whose subtree contains
    an ``X.get_bytes(...)`` call — that is "bytes from the store", whatever the
    store is called (`store`, `ctx.store`, `self.store`). Second: every call to
    `json.loads` / `json.load` / `orjson.loads` whose argument subtree contains
    either such a call directly or one of those bound names. Both passes are
    over nodes, so a call split across lines is one call.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def has_get_bytes(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get_bytes"
            for n in ast.walk(node)
        )

    bound: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.NamedExpr):
            targets, value = [node.target], node.value
        if value is not None and has_get_bytes(value):
            for target in targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)

    hits: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Name) and (owner.id, node.func.attr) in _PARSERS):
            continue
        for arg in node.args:
            if has_get_bytes(arg) or any(
                isinstance(n, ast.Name) and n.id in bound for n in ast.walk(arg)
            ):
                hits.append(node.lineno)
                break
    return hits
