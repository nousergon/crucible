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

This module is deliberately dependency-free apart from `crucible.store`, so
any module may import it without creating a cycle. `crucible.gate` imports
these three names rather than defining a second copy, and
`crucible.console.render` imports them rather than defining a third — a
contract restated at each surface is a contract one of the surfaces has
already drifted from.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.store import Store

__all__ = [
    "DocumentRead",
    "read_document",
    "read_path_document",
    "read_store_document",
]


@dataclass(frozen=True)
class DocumentRead:
    """One document a surface read, or the reason it could not be read.

    Three outcomes, never two: present-and-readable, ABSENT, and
    UNREADABLE. Collapsing the last two is the defect this exists to prevent
    — "no filed count" and "the file is corrupt" call for different actions,
    and only one of them is about the system rather than about us.

    This is the ONE reader for external documents (artifacts written by
    producers outside this repository) and first-party ones alike
    (`run_manifest.v1` and the arena artifacts `crucible.runner` writes and
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
    try:
        document = json.loads(raw)
    except Exception as exc:
        return DocumentRead(
            None,
            False,
            f"{source} is present but is not readable JSON: {type(exc).__name__}: {exc}",
        )
    if document is None:
        return DocumentRead(
            None,
            False,
            f"{source} is present and its body is literal `null` — present-but-null is "
            "unreadable, not absent, and reporting it as absent names the wrong remedy",
        )
    if not isinstance(document, dict):
        return DocumentRead(
            None, False, f"{source} parsed to {type(document).__name__}, not an object with fields"
        )
    return DocumentRead(document, False, None)


def read_store_document(store: Store, key: str) -> DocumentRead:
    """:func:`read_document` over a store key."""

    def reader() -> bytes | None:
        return store.get_bytes(key) if store.exists(key) else None

    return read_document(key, reader)


def read_path_document(path: Path) -> DocumentRead:
    """:func:`read_document` over a file in the checkout."""

    def reader() -> bytes | None:
        return path.read_bytes() if path.is_file() else None

    return read_document(str(path), reader)
