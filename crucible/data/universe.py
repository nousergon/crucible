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

**The pointer has one writer, and it is not this repo.**
``market_data/latest_weekly.json`` is written only by `nousergon-data`'s
weekly Phase 1 (``weekly_collector._write_manifest``, run by the
``ne-data-collection-weekly`` schedule at 09:00Z on Saturdays). Nothing in
this package names the key, so nothing here can be a second writer;
``tests/test_weekly_universe_pointer.py`` fails the day something does. What
this repo was missing is ORDERING: the v2 weekly's ``data.weekly`` starts on
its own schedule (11:00Z) and used to follow whatever the pointer named. On
2026-10-24 and 2026-10-31 the collector rewrote it 334 s and ~10 min before
the weekly started (nous-ergon-ops-I2018); had it run late or failed, the
weekly would have graded last week's universe silently. So a caller that
knows which week it is grading passes ``expected_date``, and a pointer whose
``date`` disagrees is refused by name (`StaleUniversePointerError`) rather
than followed. `data.daily` passes nothing: between Saturdays it
legitimately reads last Friday's pointer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

from crucible.keys import declared_universe_key
from crucible.models import DeclaredUniverseDocument
from crucible.runner import RunContext
from crucible.store import sha256_hex

__all__ = [
    "DECLARED_UNIVERSE_SCHEMA_VERSION",
    "DanglingUniversePointerError",
    "DeclaredUniverse",
    "MalformedUniverseError",
    "StaleUniversePointerError",
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
#: The pointer's own week label, written by its sole writer beside
#: ``s3_prefix`` (`nousergon-data` ``weekly_collector._write_manifest``).
_POINTER_DATE_FIELD = "date"
_POINTER_TARGET = "constituents.json"


class MalformedUniverseError(ValueError):
    """The declared-universe document was read and is not a universe.

    Distinct from `UndeclaredUniverseError` (nothing was declared) and from a
    fetch failure (the document could not be read): this one was read and
    says nothing usable. A run must fail on it, never fall back to the
    source's own symbol set — that fallback is option (b) above, by the back
    door.
    """


class DanglingUniversePointerError(FileNotFoundError):
    """A pointer was read and names a target that does not exist.

    A data-pointer fault, not a crash and not a malformed document: the
    producer advanced the pointer to a prefix it never wrote
    ``constituents.json`` into (alpha-engine-config-I12410 — a weekly Phase 1
    whose constituents collector failed). It used to surface as a raw
    ``botocore`` ``NoSuchKey`` naming only the target, which read as a crash
    in this repo; it carries BOTH keys so the fault is attributable to the
    pointer's writer from the message alone. A ``FileNotFoundError`` so any
    caller already treating a missing document as missing still does.
    """

    def __init__(self, pointer_uri: str, target_uri: str) -> None:
        self.pointer_uri = pointer_uri
        self.target_uri = target_uri
        super().__init__(
            f"universe pointer {pointer_uri} names {target_uri}, which does not exist: "
            "the pointer's writer advanced it to a prefix without a membership "
            "document (a data-pointer fault upstream, not a fault in this run)"
        )


class StaleUniversePointerError(MalformedUniverseError):
    """A pointer was read and names a different week than the caller is grading.

    nous-ergon-ops-I2018 / alpha-engine-config-I13065. The weekly pointer is
    written by `nousergon-data`'s weekly Phase 1 on its own schedule, and the
    v2 ``data.weekly`` reads it on another, with nothing ordering the two. A
    pointer still dated last week means the collector has not advanced it
    (late, failed, or refused its own advance); following it would grade this
    week against last week's membership and report nothing wrong. Carries
    both dates and the pointer URI so the message alone says which writer to
    look at. A `MalformedUniverseError` because the run must fail on it the
    same way, never fall back.
    """

    def __init__(self, pointer_uri: str, pointer_date: object, expected_date: str) -> None:
        self.pointer_uri = pointer_uri
        self.pointer_date = pointer_date
        self.expected_date = expected_date
        super().__init__(
            f"universe pointer {pointer_uri} is dated {pointer_date!r} but this run grades "
            f"the week ending {expected_date}: its writer (nousergon-data weekly Phase 1) "
            "has not advanced it for this week, so following it would grade last week's "
            "universe. Refused rather than followed."
        )


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
        # `alpha-engine-config-I10045` row 10: validated against
        # `crucible.models.DeclaredUniverseDocument` before it is written — a
        # producer that could emit a `count`/`symbols` mismatch would defeat
        # the whole point of a declared denominator.
        DeclaredUniverseDocument.model_validate(document)
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
        ctx.record_output(key, payload, schema_version=DECLARED_UNIVERSE_SCHEMA_VERSION)
        return key


def _normalise(raw: object, *, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise MalformedUniverseError(
            f"{where}: `{_TICKERS_FIELD}` must be a non-empty list of symbols; got "
            f"{type(cast(object, raw)).__name__} of length "
            f"{len(cast(list[object], raw)) if isinstance(raw, list) else 'n/a'}"
        )
    items = cast(list[object], raw)
    cleaned = {s.strip().upper() for s in items if isinstance(s, str) and s.strip()}
    if len(cleaned) != len({s for s in items if isinstance(s, str) and s.strip()}):
        # Two spellings of one symbol would count twice in the denominator
        # and once in the numerator, so the ratio could never reach 1.0.
        raise MalformedUniverseError(f"{where}: duplicate symbols after normalisation")
    non_strings = [s for s in items if not isinstance(s, str) or not s.strip()]
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


class _StreamingBody(Protocol):
    def read(self) -> bytes: ...


class _GetObjectOutput(TypedDict):
    Body: _StreamingBody


class _S3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> _GetObjectOutput: ...


class _Boto3(Protocol):
    def client(self, service_name: str) -> _S3Client: ...


def _read_s3(uri: str) -> bytes:
    import boto3  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415 - lazy; dependency has no stubs
    from botocore.exceptions import (  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415 - lazy; dependency has no stubs
        ClientError,
    )

    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise MalformedUniverseError(f"{uri}: an s3:// universe URI needs a bucket and a key")
    try:
        s3 = cast(_Boto3, boto3)
        return s3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        # A missing object reads as missing — the same `FileNotFoundError` the
        # local branch raises — so a pointer's dangling target is recognisable
        # by one type whichever branch fetched it. Anything else (denied,
        # throttled) is not "missing" and propagates unchanged.
        # botocore has no installed type information; this is its documented
        # JSON-like error response boundary.
        response = cast(dict[str, object], exc.response)
        error = response.get("Error")
        code = cast(dict[str, object], error).get("Code") if isinstance(error, dict) else None
        if code in ("NoSuchKey", "404"):
            raise FileNotFoundError(uri) from exc
        raise


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
    expected_date: str | None = None,
) -> DeclaredUniverse:
    """Resolve ``uri`` to a universe, following a pointer at most once.

    ``read`` is the fetch seam a test supplies; production reads S3 or a
    local path by scheme. Every failure raises — a universe that could not
    be read is a run that cannot grade coverage, and `run_daily` must see it
    as such rather than as ``None``.

    ``expected_date`` (ISO ``YYYY-MM-DD``) is the week the caller grades. When
    given and ``uri`` is a POINTER, the pointer's ``date`` must equal it, or
    `StaleUniversePointerError` is raised before the target is fetched; a
    pointer with no ``date`` is refused the same way, since it cannot show it
    is this week's. A membership document (no pointer) carries no week to
    compare and is unaffected. ``None`` keeps the undated behaviour.
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
        if expected_date is not None:
            pointer_date = document.get(_POINTER_DATE_FIELD)
            if pointer_date != expected_date:
                raise StaleUniversePointerError(uri, pointer_date, expected_date)
        target = _sibling(uri, prefix)
        try:
            payload = fetch(target)
        except FileNotFoundError as exc:
            raise DanglingUniversePointerError(uri, target) from exc
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


def _parse(payload: bytes, *, where: str) -> dict[str, object]:
    try:
        # ``json.loads`` is untyped; narrow its result before exposing it.
        document = cast(object, json.loads(payload))
    except ValueError as exc:
        raise MalformedUniverseError(f"{where}: not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise MalformedUniverseError(
            f"{where}: expected a JSON object, got {type(document).__name__}"
        )
    return cast(dict[str, object], document)
