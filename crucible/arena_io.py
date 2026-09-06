"""Reading and writing the `arena_cycle` artifact.

Normative sources: `champion-challenger-policy.md` §11, plan §4.4.

    "Each slot emits one `arena_cycle` artifact per cycle, conforming to
    `nousergon_lib/contracts/arena_cycle.schema.json` … A missing
    `arena_cycle` for a scheduled cycle is the §4.6 absence page."

One writer for all four slots, deliberately. The key shape and the schema
validation are the same fact for U, R, M and S, and a second copy of it is
how two slots end up writing artifacts the console renders differently.

**Validated against the LIBRARY's schema, not a local restatement.** The
contract ships inside `nousergon_lib.contracts`; validating against a
crucible copy would pass forever while the library's schema moved.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator
from nousergon_lib.arena import ArenaCycle

from crucible.documents import load_store_document
from crucible.keys import arena_cycle_key
from crucible.models import ArenaCycleDocument
from crucible.store import Store

__all__ = [
    "ArenaCycleValidationError",
    "arena_cycle_key",
    "load_arena_cycle_schema",
    "read_arena_cycle",
    "validate_arena_cycle",
    "write_arena_cycle",
]


class ArenaCycleValidationError(ValueError):
    """An `arena_cycle` document that does not conform to the library contract."""


@lru_cache(maxsize=1)
def load_arena_cycle_schema() -> dict[str, Any]:
    """The library's own `arena_cycle` schema."""
    text = (
        resources.files("nousergon_lib.contracts")
        .joinpath("arena_cycle.schema.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = load_arena_cycle_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_arena_cycle(payload: dict[str, Any]) -> None:
    """Raise :class:`ArenaCycleValidationError` unless ``payload`` conforms.

    Every error is reported, not just the first: a producer fixing one field
    per run against a validator that reports one error per run is how a
    half-conformant artifact ships over four cycles.
    """
    errors = sorted(_validator().iter_errors(payload), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    detail = "\n".join(
        f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    )
    raise ArenaCycleValidationError(
        "arena_cycle does not conform to nousergon_lib/contracts/arena_cycle.schema.json:\n"
        + detail
    )


def write_arena_cycle(store: Store, cycle: ArenaCycle) -> str:
    """Write ``cycle`` and return its key. Validates BEFORE the write.

    A non-conformant artifact is a bug in the producer, and it must surface
    where the producer runs rather than on the console at the end of a
    Saturday.
    """
    payload = cycle.to_dict()
    validate_arena_cycle(payload)
    key = arena_cycle_key(cycle.slot, cycle.as_of)
    store.put_bytes(key, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
    return key


def read_arena_cycle(store: Store, slot: str, trading_day: str) -> ArenaCycleDocument:
    """Read and validate one cycle artifact.

    Validated on read as well as on write: an artifact written by an older
    release is still refused if it does not conform, rather than being
    partially understood. `alpha-engine-config-I10045` row 3: the caller now
    gets `ArenaCycleDocument` — named, typed top-level access — rather than
    the raw dict a hand-indexed reader would need to know the shape of by
    heart. The library's schema, not this model, still decides conformance;
    see `ArenaCycleDocument`'s docstring for why `extra` is not forbidden
    here the way it is on every other boundary in this migration.
    """
    payload = load_store_document(store, arena_cycle_key(slot, trading_day))
    validate_arena_cycle(payload)
    return ArenaCycleDocument.model_validate(payload)
