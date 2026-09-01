"""The champion pointer: the one contract the trader reads.

Normative sources: plan §3 ("**The contract is the only coupling.**"), §4.4,
§9.1; `champion-challenger-policy.md` §11.

The harness and the trader are separate systems (plan §3), and everything the
trader knows about a slot arrives through `champions/{slot}/current.json`.
That makes this module a **product contract**, not an internal helper: a
second implementation of the trader must be able to consume it from the
schema alone.

**The refusals live on the READER, and that is the whole point.** A producer
can only refuse to write; it cannot un-write a pointer whose producing run
died a second later, and it cannot know that the attestation it wrote was
later superseded. So :func:`read_champion` re-derives both gates from durable
artifacts every time it is called:

1. **The producing manifest must be `status: ok`.** A pointer whose run then
   failed is the fleet's dominant bug class — a record asserting an action
   that never happened (policy §7.2).
2. **An S-slot champion needs a `pit_parity` attestation of `PASS`.** Plan
   §9.1: "a card without `attestation: PASS` renders UNVERIFIED, never a
   grade". A pointer is a card the trader acts on with money, so `PARTIAL`,
   `UNKNOWN` and a missing attestation are all refusals rather than degraded
   grades — the exact place a fail-open would be invisible.

Both raise :class:`ChampionUnusableError`. Neither returns ``None``: a
consumer that cannot distinguish "no champion" from "a champion I must not
serve" would trade the second one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from crucible.store import Store

__all__ = [
    "ATTESTED_SLOTS",
    "CHAMPION_SCHEMA_VERSION",
    "PROMOTION_SOURCES",
    "ChampionPointer",
    "ChampionUnusableError",
    "champion_key",
    "read_champion",
    "write_champion",
]

CHAMPION_SCHEMA_VERSION = "champion_pointer.v1"

SCHEMA_PATH = Path(__file__).parent / "schemas" / "champion_pointer.v1.json"

#: How this pointer came to be, exhaustively.
#:
#: * ``evidence`` — the anytime-valid sequence supported the lead (policy §5).
#: * ``operator_bootstrap`` — a human placed it. Policy §11: "a pointer that
#:   has never moved on evidence is a finding, not a stable system", and R's
#:   champion has carried this flag unnoticed since 2026-07-13. Carrying it in
#:   the artifact is what makes the first evidence-won promotion visible.
#: * ``bootstrap`` — the engine's §9.1 cold start: no incumbent existed, an
#:   arm had to serve, and the highest Copeland arm was taken.
PROMOTION_SOURCES: tuple[str, ...] = ("evidence", "operator_bootstrap", "bootstrap")

#: Slots whose champion is unusable without a contamination attestation.
#: Only S today: its grade is a backtest of a market position, and the
#: look-ahead delta is the thing a backtest cannot self-report (plan §9.1).
ATTESTED_SLOTS: tuple[str, ...] = ("s",)

#: The one attestation status that lets a card render as a grade.
ATTESTATION_PASS = "PASS"


class ChampionUnusableError(RuntimeError):
    """The stored champion exists and must not be served.

    Deliberately distinct from ``KeyError`` (no pointer at all): "there is no
    champion yet" and "there is a champion I am refusing" are different
    operational situations with different fixes, and a consumer that
    collapsed them would fall back to trading nothing in a case that should
    page.
    """


def champion_key(slot: str) -> str:
    """The store key for ``slot``'s pointer. The trader's whole read surface."""
    if slot not in ("u", "r", "m", "s"):
        raise KeyError(f"unknown slot {slot!r}; the four slots are ['m', 'r', 's', 'u']")
    return f"champions/{slot}/current.json"


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"champion pointer schema missing at {SCHEMA_PATH}. It ships inside the "
            "package; a missing schema means a broken build, not a degraded read."
        )
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@dataclass(frozen=True)
class ChampionPointer:
    """Which arm serves a slot, and the evidence that put it there."""

    slot: str
    arm_id: str
    as_of: str
    decided_at: str
    run_id: str
    code_sha: str
    promotion_source: str
    manifest_key: str
    evidence: dict[str, Any] = field(default_factory=dict)
    attestation: dict[str, Any] | None = None
    schema_version: str = CHAMPION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.promotion_source not in PROMOTION_SOURCES:
            raise ValueError(
                f"promotion_source {self.promotion_source!r} is not one of "
                f"{PROMOTION_SOURCES}. The set is closed so that an operator-placed "
                "pointer can never be mistaken for one the evidence won — which is "
                "exactly what R's 2026-07-13 bootstrap did for seven weeks."
            )
        if self.schema_version != CHAMPION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version!r} is not {CHAMPION_SCHEMA_VERSION!r}"
            )
        if not self.manifest_key:
            raise ValueError(
                "manifest_key is mandatory: the reader re-derives the producing run's "
                "status from it, and a pointer with no producing run cannot be checked"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slot": self.slot,
            "arm_id": self.arm_id,
            "as_of": self.as_of,
            "decided_at": self.decided_at,
            "run_id": self.run_id,
            "code_sha": self.code_sha,
            "promotion_source": self.promotion_source,
            "manifest_key": self.manifest_key,
            "evidence": self.evidence,
            "attestation": self.attestation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChampionPointer:
        version = payload.get("schema_version")
        if version != CHAMPION_SCHEMA_VERSION:
            raise ChampionUnusableError(
                f"champion pointer declares schema_version {version!r}; this reader "
                f"speaks {CHAMPION_SCHEMA_VERSION!r} only. A consumer that guessed at "
                "an unknown version would trade on fields it does not understand."
            )
        errors = sorted(_validator().iter_errors(payload), key=lambda e: list(e.absolute_path))
        if errors:
            detail = "\n".join(
                f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                for e in errors
            )
            raise ChampionUnusableError(
                f"champion pointer does not conform to {CHAMPION_SCHEMA_VERSION}:\n{detail}"
            )
        return cls(
            slot=payload["slot"],
            arm_id=payload["arm_id"],
            as_of=payload["as_of"],
            decided_at=payload["decided_at"],
            run_id=payload["run_id"],
            code_sha=payload["code_sha"],
            promotion_source=payload["promotion_source"],
            manifest_key=payload["manifest_key"],
            evidence=payload.get("evidence") or {},
            attestation=payload.get("attestation"),
        )


def write_champion(store: Store, pointer: ChampionPointer) -> str:
    """Write ``pointer`` at its slot's key. Returns the content hash.

    Validated on the way out as well as on the way in: a writer that could
    emit a non-conformant pointer would defeat the schema, and the trader
    would discover it at market open.
    """
    payload = pointer.to_dict()
    errors = sorted(_validator().iter_errors(payload), key=lambda e: list(e.absolute_path))
    if errors:
        detail = "\n".join(
            f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise ValueError(f"refusing to write a non-conformant champion pointer:\n{detail}")
    return store.put_bytes(
        champion_key(pointer.slot),
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
    )


def read_champion(store: Store, slot: str) -> ChampionPointer:
    """The arm the trader may serve for ``slot``.

    Raises :class:`KeyError` when no pointer exists and
    :class:`ChampionUnusableError` when one exists but fails either gate.
    """
    raw = store.get_bytes(champion_key(slot))  # KeyError when absent, by contract
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChampionUnusableError(
            f"champion pointer for slot {slot!r} is not readable JSON: {exc}"
        ) from exc

    pointer = ChampionPointer.from_dict(payload)
    _assert_producing_run_ok(store, pointer)
    _assert_attested(pointer)
    return pointer


def _assert_producing_run_ok(store: Store, pointer: ChampionPointer) -> None:
    try:
        manifest = json.loads(store.get_bytes(pointer.manifest_key))
    except KeyError as exc:
        raise ChampionUnusableError(
            f"champion {pointer.arm_id} names manifest {pointer.manifest_key!r}, which is "
            "not in the store. A pointer whose producing run cannot be found is "
            "indistinguishable from one no run ever wrote."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ChampionUnusableError(
            f"manifest {pointer.manifest_key!r} for champion {pointer.arm_id} is not "
            f"readable JSON: {exc}"
        ) from exc

    status = manifest.get("status")
    if status != "ok":
        raise ChampionUnusableError(
            f"champion {pointer.arm_id} was written by run {pointer.run_id}, whose "
            f"manifest {pointer.manifest_key!r} carries status {status!r}, not 'ok' "
            f"(reason: {manifest.get('reason') or 'none recorded'}). Serving it would "
            "be a record asserting an action that never completed "
            "(champion-challenger-policy.md §7.2)."
        )


def _assert_attested(pointer: ChampionPointer) -> None:
    if pointer.slot not in ATTESTED_SLOTS:
        return
    attestation = pointer.attestation or {}
    status = attestation.get("status")
    if status != ATTESTATION_PASS:
        raise ChampionUnusableError(
            f"champion {pointer.arm_id} renders UNVERIFIED: slot {pointer.slot!r} requires a "
            f"pit_parity attestation of {ATTESTATION_PASS!r} and carries {status!r} "
            f"({attestation.get('reason') or 'no reason recorded'}). Plan §9.1: a card "
            "without an attestation PASS renders UNVERIFIED, never a grade — and a "
            "pointer is a card the trader acts on with money."
        )
