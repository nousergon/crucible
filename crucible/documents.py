"""The ONE guarded JSON reader every surface parses a stored artifact with.

Normative source: `alpha-engine-config-I9869` (which built this shape inside
`crucible.gate`) and `alpha-engine-config-I9900` (which is why it now lives
here rather than there).

A surface — the gate, the console page, the board — exists to *publish a
reading*. An exception raised while parsing one artifact does not fail one
row of that reading: it propagates out of the builder and publishes NOTHING,
which is the absence-instead-of-red failure every one of those surfaces was
built to refuse. I9900 is that failure observed live: one `message.txt`
sitting beside a run manifest took the whole board down, so no board and no
ladder were published at all.

So the contract is three outcomes, never two — present-and-readable, ABSENT,
and UNREADABLE — plus a fourth bit saying whether an unreadable outcome is a
statement about the artifact or about **our access** to it. Collapsing any of
those pairs names the wrong remedy: "no file filed", "the file is corrupt"
and "we were denied" call for three different actions and only one of them is
about the system being measured.

This module depends only on `crucible.store` and `crucible.keys` (for the
manifest-key predicate), neither of which imports anything above it, so any
module may import it without creating a cycle. `crucible.gate` imports
these three names rather than defining a second copy, and
`crucible.console.render` imports them rather than defining a third — a
contract restated at each surface is a contract one of the surfaces has
already drifted from.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.keys import is_manifest_key
from crucible.store import Store

__all__ = [
    "DocumentRead",
    "PrefixRead",
    "UnreadableDocumentError",
    "load_document_bytes",
    "load_store_document",
    "read_document",
    "read_listed_document",
    "read_manifests_under",
    "read_path_document",
    "read_store_document",
]


class UnreadableDocumentError(ValueError):
    """A stored document that is present and cannot be read as an object.

    Raised by :func:`load_store_document`, the STRICT face of this reader,
    for a producer or writer whose correct response to a corrupt input is to
    stop (AGENTS.md rule 5) rather than to publish a fault row. The message is
    the same sentence :func:`read_document` would have returned as
    ``problem``, so the two faces of the reader never disagree about what was
    wrong — only about who handles it. A `ValueError`, so the call sites that
    already caught `json.JSONDecodeError` (itself a `ValueError`) still catch
    this without knowing which face they went through.
    """


@dataclass(frozen=True)
class DocumentRead:
    """One document a surface read, or the reason it could not be read.

    Three outcomes, never two: present-and-readable, ABSENT, and
    UNREADABLE. Collapsing the last two is the defect this exists to prevent
    — "no filed count" and "the file is corrupt" call for different actions,
    and only one of them is about the system rather than about us.

    This is the ONE reader for external documents (artifacts written by
    producers outside this repository) and first-party ones alike
    (the run manifest, at whichever version wrote it, and the arena artifacts
    `crucible.runner` writes and
    validates at write time). A validated-at-write-time schema is not a
    validated-at-READ-time guarantee — a truncated `run.json` (an interrupted
    write, a partial multipart upload, a hand-edited artifact) is unreadable
    regardless of who wrote it, and reading it unguarded was exactly the gap
    `alpha-engine-config-I9869` closed in the gate and
    `alpha-engine-config-I9900` closed on the console: `json.loads` plus
    direct field indexing, raising straight out of the surface.
    """

    document: dict[str, Any] | None
    absent: bool
    problem: str | None
    #: True when ``problem`` is a statement about OUR ACCESS (the reader
    #: raised: PermissionError, AccessDenied) rather than about the content
    #: (unparseable JSON, wrong shape). The distinction is what lets a caller
    #: report UNMEASURABLE instead of UNMET — round 2 of
    #: `alpha-engine-config-I9869`: the two were collapsed, so a store outage
    #: read identically to a broken build.
    access_problem: bool = False

    def require(self, field: str, kind: type) -> str | None:
        """``None`` when ``field`` is present and of type ``kind``, else why not.

        The fourth failure mode I9900 names: a document that parses to an
        object and then carries a required field of the wrong type. Reading it
        unguarded moves the crash one line down from `json.loads` to the first
        `.get(...)` the caller does arithmetic or a comparison on, which is
        the same absence-instead-of-red outcome with a less obvious traceback.

        Returns a sentence, not a bool, because every caller renders the fault
        into a row detail and a bare False names nothing.
        """
        if self.document is None:
            return self.problem
        if field not in self.document:
            return f"required field {field!r} is missing"
        value = self.document[field]
        if not isinstance(value, kind):
            return (
                f"required field {field!r} is {type(value).__name__}, not {kind.__name__} "
                f"({value!r})"
            )
        return None


def read_document(source: str, reader: Callable[[], bytes | None]) -> DocumentRead:
    """Read one document — first-party or written by a producer outside this
    repository, this module makes no distinction.

    ``reader`` returns the raw bytes, or ``None`` when the source is absent.

    **An exception raised here does not fail one row — it takes the whole
    surface down with it.** That is not a hypothetical: `alpha-engine-config-I9900`
    is a `JSONDecodeError` on a plain-text artifact that stopped `crucible
    board` publishing anything at all for seven hours. So every failure mode
    below becomes a returned fault naming the source and the reason.

    The broad `except` is deliberate and is not a swallow: the failure mode
    caught is "a document cannot be parsed", and the recording surface is the
    returned :class:`DocumentRead`, which every caller renders into an unmet
    clause or an unreadable row. Nothing is discarded and nothing degrades
    silently.
    """
    try:
        raw = reader()
    except Exception as exc:
        return DocumentRead(
            None,
            False,
            f"{source} could not be read: {type(exc).__name__}: {exc}. That is a "
            "statement about our access, not about the system being measured",
            access_problem=True,
        )
    if raw is None:
        return DocumentRead(None, True, None)
    document, problem = _parse_object(source, raw)
    if problem is not None:
        return DocumentRead(None, False, problem)
    return DocumentRead(document, False, None)


def _parse_object(source: str, raw: bytes | str) -> tuple[dict[str, Any] | None, str | None]:
    """``(document, None)`` or ``(None, why)``. The ONE place bytes become an object.

    Both faces of the reader — the guarded :func:`read_document` and the
    strict :func:`load_store_document` — parse through here, so "what counts
    as unreadable" (not JSON, literal `null`, an array, a string) has exactly
    one definition. `alpha-engine-config-I9931` is the shape this prevents: a
    manifest whose body was an array parsed cleanly under `json.loads` and
    then raised `AttributeError` on the first `.get()`, out of the sweep.
    """
    try:
        document = json.loads(raw)
    except Exception as exc:
        return None, f"{source} is present but is not readable JSON: {type(exc).__name__}: {exc}"
    if document is None:
        return None, (
            f"{source} is present and its body is literal `null` — present-but-null is "
            "unreadable, not absent, and reporting it as absent names the wrong remedy"
        )
    if not isinstance(document, dict):
        return None, f"{source} parsed to {type(document).__name__}, not an object with fields"
    return document, None


def read_store_document(store: Store, key: str) -> DocumentRead:
    """:func:`read_document` over a store key."""

    def reader() -> bytes | None:
        return store.get_bytes(key) if store.exists(key) else None

    return read_document(key, reader)


def read_listed_document(store: Store, key: str) -> DocumentRead:
    """:func:`read_document` over a key the caller obtained FROM A LISTING.

    Differs from :func:`read_store_document` in exactly one outcome: a key
    that was listed and is then gone at read time is a **fault**, never
    ``absent``. A reader that has just seen the key cannot honestly report
    "nothing filed"; what it observed is a listing/read race — a concurrent
    delete, an eventually-consistent listing, a retention sweep — and that is
    a fact about the store worth a row of its own. `alpha-engine-config-I9931`
    item 2 measured the console recording this race in one loop and silently
    dropping it in the other (`ManifestRead(None, None, {})`).
    """

    def reader() -> bytes:
        return store.get_bytes(key)

    try:
        raw = reader()
    except KeyError:
        return DocumentRead(
            None,
            False,
            f"{key} was listed and then absent before it could be read — a listing/read "
            "race (concurrent delete, eventually-consistent listing or a retention sweep), "
            "recorded rather than dropped",
        )
    except Exception as exc:
        return DocumentRead(
            None,
            False,
            f"{key} could not be read: {type(exc).__name__}: {exc}. That is a "
            "statement about our access, not about the system being measured",
            access_problem=True,
        )
    document, problem = _parse_object(key, raw)
    if problem is not None:
        return DocumentRead(None, False, problem)
    return DocumentRead(document, False, None)


def load_store_document(store: Store, key: str) -> dict[str, Any]:
    """The STRICT face: the object at ``key``, or a raise that names why not.

    For producers and writers, where AGENTS.md rule 5 wants a corrupt input to
    stop the job — with the cause in the manifest — rather than to become a
    fault row on a page that may not exist. Absence propagates as the store's
    own `KeyError` (the documented contract of `Store.get_bytes`); a present
    document that is not an object raises :class:`UnreadableDocumentError`
    carrying the same sentence the guarded face would have published.

    This exists so that "every read of stored JSON goes through one reader"
    is true for the whole package and not only for the surfaces
    (`alpha-engine-config-I9931` closes-when: no `json.loads(store.get_bytes(`
    outside this module).
    """
    return load_document_bytes(key, store.get_bytes(key))


def load_document_bytes(source: str, raw: bytes | str) -> dict[str, Any]:
    """The STRICT face over bytes the caller ALREADY HOLDS.

    For the caller that has fetched an object once — to record it as lineage,
    to compare-and-swap against its version, to hash it — and must decide on
    exactly those bytes. Re-fetching through :func:`load_store_document` would
    read the object a second time, and on the one object built to move under
    concurrent writers (`releases/current`, CAS-flipped by `crucible.deploy`)
    the lineage and the decision could then describe two different versions
    (crucible-PR81 review, B1). Same parser, same sentence, same exception as
    the store face; only the fetch is the caller's.
    """
    document, problem = _parse_object(source, raw)
    if problem is not None:
        raise UnreadableDocumentError(problem)
    assert document is not None  # _parse_object returns exactly one of the pair
    return document


@dataclass(frozen=True)
class PrefixRead:
    """Every manifest under one prefix, and every key there that could not be read.

    ``documents`` is ``(key, document)`` in listing order for each key that
    :func:`crucible.keys.is_manifest_key` accepts and that read as an object.
    ``faults`` is ``{key: why}`` for each manifest key that did not — corrupt,
    wrong shape, vanished between list and read, or denied. Non-manifest keys
    under the prefix (a job's own evidence filed beside its manifest, such as
    `report.morning`'s `message.txt`) appear in neither: a manifest prefix is a
    namespace, not a manifest list (`alpha-engine-config-I9900`).

    ``listing_problem`` is set, and both other fields are empty, when the
    PREFIX ITSELF could not be listed. That is not "no manifests are there";
    it is "we could not ask", and the two must not collapse — the same
    distinction `crucible.gate._list_store_keys` draws
    (`alpha-engine-config-I9960`). A caller that ignores it and reads the
    empty `documents` as an answer is claiming a fact it does not have.
    """

    documents: tuple[tuple[str, dict[str, Any]], ...] = ()
    faults: dict[str, str] = field(default_factory=dict)
    listing_problem: str | None = None

    def raise_if_unlistable(self) -> None:
        """Fail loud for the caller that has no honest way to carry it."""
        if self.listing_problem is not None:
            raise UnreadableDocumentError(self.listing_problem)


def read_manifests_under(store: Store, prefix: str) -> PrefixRead:
    """List ``prefix``, keep manifest keys only, read each through the guard.

    The one implementation of "read the manifests under this prefix" that
    every surface and sweep shares, so a new consumer cannot reintroduce the
    `json.loads`-every-listed-key shape that took the board down for seven
    hours (`alpha-engine-config-I9900`, `-I9929`). Listing order is preserved
    (sorted, as `Store.list_keys` documents) so callers that pick "the last
    manifest" keep their deterministic choice.
    """
    documents: list[tuple[str, dict[str, Any]]] = []
    faults: dict[str, str] = {}
    try:
        listed = sorted(store.list_keys(prefix))
    except Exception as exc:
        # NOT a swallow, and not an empty result: the failure mode is a
        # listing denial reading as "no manifests here", which is an ABSENCE
        # page about a job that may well have delivered. It is carried out as
        # `listing_problem`, and the recording surface is the caller's — the
        # sweep pages what it DID observe and then fails with this in its own
        # manifest (`alpha-engine-config-I9960`); `raise_if_unlistable` is
        # there for every caller with nowhere honest to put it.
        return PrefixRead(
            listing_problem=(
                f"listing {prefix!r} could not be read: {type(exc).__name__}: {exc}. That "
                "is a statement about our access, not about what is there"
            )
        )
    for key in listed:
        if not is_manifest_key(key):
            continue
        read = read_listed_document(store, key)
        if read.problem is not None:
            faults[key] = read.problem
            continue
        assert read.document is not None  # a listed key reads as an object or as a fault
        documents.append((key, read.document))
    return PrefixRead(tuple(documents), faults)


def read_path_document(path: Path) -> DocumentRead:
    """:func:`read_document` over a file in the checkout."""

    def reader() -> bytes | None:
        return path.read_bytes() if path.is_file() else None

    return read_document(str(path), reader)
