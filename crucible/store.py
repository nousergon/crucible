"""The artifact store: one interface, an S3 backend and a local-dir backend.

Normative source: `crucible_v2_rebuild_plan_260901.md` §4.11 and §4.12.

Two backends behind one interface, because §2's "easy to run experiments"
objective is *"`crucible experiment.run <arm>` from a laptop returns a
verdict in one command, no AWS resources beyond S3 read/write"* — and a
developer who cannot run the whole thing against a directory will not run it
at all. The manifest records store *keys*, never URIs, so the same manifest
is portable between backends.

**Keys are trading days.** Any date component of any key is an NYSE session
(§4.12). :meth:`Store.assert_keys_bind_to_trading_days` is the walk that
enforces it, and it is exercised in `tests/test_trading_day_contract.py`
against the local backend today so the S3 backend inherits it rather than
needing a second test written after the fact.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from crucible.calendar import (
    ISO_DATE_RE,
    NEAR_MISS_DATE_RE,
    NonTradingDayKeyError,
    is_trading_day,
)

__all__ = ["LocalStore", "S3Store", "Store", "sha256_hex"]


def sha256_hex(payload: bytes) -> str:
    """The content hash recorded in `inputs[]` and `outputs[]`.

    Content addressing is what makes a rerun idempotent: a rerun producing
    identical bytes is a no-op, which is the whole basis of the runbook's
    `rerun` verb.
    """
    return hashlib.sha256(payload).hexdigest()


class Store(ABC):
    """The artifact interface every job reads and writes through.

    Deliberately small. A job needs to put bytes, get bytes, ask whether a
    key exists and list a prefix; anything larger is a backend feature
    leaking into the callers and pinning us to one provider (principle 8).
    """

    @abstractmethod
    def put_bytes(self, key: str, payload: bytes) -> str:
        """Write ``payload`` at ``key``; return its sha256 hex digest."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Read ``key``. Raises :class:`KeyError` when it does not exist.

        Never returns ``None`` for a missing key: a caller that cannot tell
        "absent" from "empty" is the shape of a silent no-op.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether ``key`` is present. Absence is a first-class fact here —
        it is one of the two page conditions (§4.6)."""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> Iterator[str]:
        """Every key under ``prefix``, in no guaranteed order."""

    def assert_keys_bind_to_trading_days(self, prefix: str = "") -> None:
        """Walk the store and refuse any key whose date is not a session.

        The §4.12 contract test, as a method rather than a script, so it runs
        against whichever backend the caller has.

        Three cases, and the third is the one that matters:

        * a key with a well-formed ISO date → the date must be a session;
        * a key with no date at all (``champions/r/current.json``,
          ``releases/current``) → legal, skipped;
        * a key with a **date-shaped but malformed** component
          (``2026-8-29``) → an error, NOT a dateless key. A strict regex
          alone would read the malformed one as dateless and pass it, which
          is precisely how a walk reports clean over a broken writer.

        Reports every offender in one raise. An operator fixing one bad key
        per test run is how a multi-day backfill defect takes a week to clear.
        """
        offenders: list[str] = []
        for key in self.list_keys(prefix):
            for candidate in NEAR_MISS_DATE_RE.findall(key):
                if not ISO_DATE_RE.fullmatch(candidate):
                    offenders.append(f"{key}: {candidate!r} is not a well-formed ISO-8601 date")
                    continue
                if not is_trading_day(dt.date.fromisoformat(candidate)):
                    offenders.append(f"{key}: {candidate} is not an NYSE trading day")
        if offenders:
            raise NonTradingDayKeyError(
                "store keys must bind to NYSE trading days (plan §4.12); "
                f"{len(offenders)} offending key(s):\n"
                + "\n".join(f"  - {o}" for o in sorted(offenders))
            )


class LocalStore(Store):
    """A directory on disk. The laptop backend, and the test backend.

    Not a mock: the trading-day walk, the content hashing and the key shapes
    are the same code the S3 backend runs, so a test against this backend
    tests the real contract.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(
                f"refusing key {key!r}: store keys are relative and never traverse. "
                "An absolute or traversing key would write outside the store root."
            )
        return self.root / key

    def put_bytes(self, key: str, payload: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return sha256_hex(payload)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(f"{key!r} is not present in {self.root}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                yield key


class S3Store(Store):
    """The production backend: one bucket, one prefix, no other AWS resource.

    Track C fills this in. It is stubbed rather than absent so the interface
    is fixed before three tracks build against it, and so the trading-day
    walk it inherits is written once.
    """

    def __init__(self, bucket: str, prefix: str = "", *, client: Any = None) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self._client = client

    def put_bytes(self, key: str, payload: bytes) -> str:
        raise NotImplementedError(
            "S3Store is track C's (crucible-v2 phase 1, alpha-engine-config-I9757). "
            "It must return the same sha256 LocalStore does and use a conditional "
            "PUT for pointer objects (releases/current, champions/{slot}/current.json)."
        )

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError(
            "S3Store is track C's (alpha-engine-config-I9757). A missing key raises "
            "KeyError, matching LocalStore — never returns None."
        )

    def exists(self, key: str) -> bool:
        raise NotImplementedError(
            "S3Store is track C's (alpha-engine-config-I9757). Absence is a page "
            "condition (§4.6), so this must distinguish absent from unreadable."
        )

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        raise NotImplementedError(
            "S3Store is track C's (alpha-engine-config-I9757). Must paginate — a "
            "truncated listing read as complete is the `--limit N` bug class."
        )
