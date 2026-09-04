"""The declared universe a data job measures its coverage against.

`run_daily` and `run_weekly` REFUSE to run without ``expected_symbols``: the
coverage floor cannot fire with no denominator, and a denominator derived
from whatever the source returned always reads 1.0 (the 901-of-903 class).
That refusal is correct and stays. What it left open, measured 2026-09-04 on
the first spot box that got as far as the runner, is that nothing on a
SCHEDULED invocation ever supplied the denominator: the dispatcher passes
``{"job": "data.daily"}``, the weekly arc's `Stage.argv` carries no
``--symbols``, and a 903-name list is not something a schedule can type. So
every scheduled data job — and the first stage of every weekly arc — failed
by construction with `UndeclaredUniverseError`.

**Two options were weighed** (plan §12 rule 5):

(a) **Read the universe from a declared DOCUMENT**, resolved from
    ``CRUCIBLE_UNIVERSE_URI`` when ``--symbols`` is absent, copied into the
    store beside the run and recorded as an output so `explain` can name the
    exact membership every panel was graded against.
(b) Enumerate the price source's own symbol set at run time. Zero
    maintenance — and exactly the denominator-equals-source shape the
    refusal exists to forbid: a name whose data stopped arriving would leave
    the denominator with the numerator and the floor would never fire.

(a) is built here. The document today is the fleet's live constituents
artifact — the same membership the v1 trading path declares its own daily
append against, which is the right denominator for a parallel run. **SOTA**
is point-in-time constituents from a provider (plan §9.6, phase 5). **Delta:**
until then this reads a document another system writes; the run records that
document's digest, so the dependency is visible on every manifest rather than
implicit.

Two document shapes are accepted, and nothing else:

* a **membership document** — a JSON object with a non-empty ``tickers``
  list of strings;
* a **pointer** — a JSON object with ``s3_prefix`` (and no ``tickers``),
  which names the prefix whose ``constituents.json`` is the membership
  document. Followed exactly once; a pointer to a pointer is refused.

A local path is accepted for the same shapes, so a laptop replay and a test
exercise the identical code path as a box (the ``s3://`` branch differs only
in how bytes are fetched).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.keys import declared_universe_key
from crucible.runner import RunContext
from crucible.store import sha256_hex

__all__ = [
    "DECLARED_UNIVERSE_SCHEMA_VERSION",
    "DeclaredUniverse",
    "MalformedUniverseError",
    "load_declared_universe",
    "universe_from_argv",
]

#: The schema version stamped on the store copy a data job writes beside its
#: run. Bumped when the document's shape changes, never for a membership change.
DECLARED_UNIVERSE_SCHEMA_VERSION = "declared_universe.v1"

#: The membership list's field name in the source document, and the file a
#: pointer's prefix is joined with. Both are the constituents artifact's own
#: contract, restated here once.
_TICKERS_FIELD = "tickers"
_POINTER_FIELD = "s3_prefix"
_POINTER_TARGET = "constituents.json"


class MalformedUniverseError(ValueError):
    """The declared-universe document was read and is not a universe.

    Distinct from `UndeclaredUniverseError` (nothing was declared) and from a
    fetch failure (the document could not be read): this one was read and
    says nothing usable. A run must fail on it, never fall back to the
    source's own symbol set — that fallback is option (b) above, by the back
    door.
    """


@dataclass(frozen=True)
class DeclaredUniverse:
    """A resolved universe, with the provenance a manifest needs."""

    symbols: tuple[str, ...]
    #: Where the membership came from: a URI or path, or ``argv:--symbols``.
    source_uri: str
    #: sha256 of the source document's bytes (or of the argv literal).
    source_sha256: str
    #: `Settings.origins`-style provenance: ``argument`` / ``environ:...``.
    origin: str

    def record(self, ctx: RunContext) -> str:
        """Copy the membership into the store beside the run and record it
        as an output, so `explain` names the exact denominator.

        Returns the store key written.
        """
        key = declared_universe_key(ctx.trading_day.isoformat())
        document = {
            "schema_version": DECLARED_UNIVERSE_SCHEMA_VERSION,
            "trading_day": ctx.trading_day.isoformat(),
            "source_uri": self.source_uri,
            "source_sha256": self.source_sha256,
            "origin": self.origin,
            "count": len(self.symbols),
            "symbols": list(self.symbols),
        }
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
        ctx.record_output(key, payload, schema_version=DECLARED_UNIVERSE_SCHEMA_VERSION)
        return key


def _normalise(raw: list[Any], *, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise MalformedUniverseError(
            f"{where}: `{_TICKERS_FIELD}` must be a non-empty list of symbols; got "
            f"{type(raw).__name__} of length {len(raw) if isinstance(raw, list) else 'n/a'}"
        )
    cleaned = {s.strip().upper() for s in raw if isinstance(s, str) and s.strip()}
    if len(cleaned) != len({s for s in raw if isinstance(s, str) and s.strip()}):
        # Two spellings of one symbol would count twice in the denominator
        # and once in the numerator, so the ratio could never reach 1.0.
        raise MalformedUniverseError(f"{where}: duplicate symbols after normalisation")
    non_strings = [s for s in raw if not isinstance(s, str) or not s.strip()]
    if non_strings:
        raise MalformedUniverseError(
            f"{where}: {len(non_strings)} entr(y/ies) in `{_TICKERS_FIELD}` are not "
            f"non-empty strings: {non_strings[:5]!r}"
        )
    return tuple(sorted(cleaned))


def universe_from_argv(raw: str) -> DeclaredUniverse:
    """The ``--symbols AAA,BBB`` form, with the same normalisation and the
    same provenance shape as a document."""
    # A stray comma on a command line is not a malformed document; a
    # non-string entry in a document is. Only the argv form drops empties.
    symbols = _normalise([s for s in raw.split(",") if s.strip()], where="--symbols")
    return DeclaredUniverse(
        symbols=symbols,
        source_uri="argv:--symbols",
        source_sha256=sha256_hex(raw.encode("utf-8")),
        origin="argument",
    )


def _read_s3(uri: str) -> bytes:
    import boto3  # noqa: PLC0415 - lazy: the laptop replay path never needs it

    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise MalformedUniverseError(f"{uri}: an s3:// universe URI needs a bucket and a key")
    return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()


def _read(uri: str) -> bytes:
    if uri.startswith("s3://"):
        return _read_s3(uri)
    return Path(uri).expanduser().read_bytes()


def _sibling(uri: str, prefix: str) -> str:
    """The pointer's target: ``prefix`` + ``constituents.json``, in the same
    bucket (or the same local root) as the pointer itself."""
    target = f"{prefix.rstrip('/')}/{_POINTER_TARGET}"
    if uri.startswith("s3://"):
        bucket = uri.removeprefix("s3://").partition("/")[0]
        return f"s3://{bucket}/{target}"
    # Local: the pointer's `s3_prefix` is relative to the bucket root, which
    # for a local fixture is the directory holding the pointer's own
    # top-level prefix. The pointer lives at `<root>/<prefix-head>/<name>`,
    # so the root is two levels up from the file.
    root = Path(uri).expanduser().resolve().parent.parent
    return str(root / target)


def load_declared_universe(
    uri: str,
    *,
    origin: str,
    read: Callable[[str], bytes] | None = None,
) -> DeclaredUniverse:
    """Resolve ``uri`` to a universe, following a pointer at most once.

    ``read`` is the fetch seam a test supplies; production reads S3 or a
    local path by scheme. Every failure raises — a universe that could not
    be read is a run that cannot grade coverage, and `run_daily` must see it
    as such rather than as ``None``.
    """
    fetch = read or _read
    if not uri:
        raise MalformedUniverseError("an empty universe URI resolves nothing")
    payload = fetch(uri)
    document = _parse(payload, where=uri)
    if _TICKERS_FIELD not in document:
        prefix = document.get(_POINTER_FIELD)
        if not isinstance(prefix, str) or not prefix:
            raise MalformedUniverseError(
                f"{uri}: neither a membership document (`{_TICKERS_FIELD}`) nor a pointer "
                f"(`{_POINTER_FIELD}`)"
            )
        target = _sibling(uri, prefix)
        payload = fetch(target)
        document = _parse(payload, where=target)
        if _TICKERS_FIELD not in document:
            raise MalformedUniverseError(
                f"{target}: the pointer's target is not a membership document; a pointer "
                "to a pointer is refused"
            )
        uri = target
    symbols = _normalise(document[_TICKERS_FIELD], where=uri)
    return DeclaredUniverse(
        symbols=symbols,
        source_uri=uri,
        source_sha256=sha256_hex(payload),
        origin=origin,
    )


def _parse(payload: bytes, *, where: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except ValueError as exc:
        raise MalformedUniverseError(f"{where}: not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise MalformedUniverseError(
            f"{where}: expected a JSON object, got {type(document).__name__}"
        )
    return document
