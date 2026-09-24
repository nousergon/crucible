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
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.calendar import (
    ISO_DATE_RE,
    NEAR_MISS_DATE_RE,
    NonTradingDayKeyError,
    is_trading_day,
)

__all__ = [
    "CAPTURED_URL_SCHEME",
    "ETAG_ABSENT",
    "PRESIGN_MAX_S",
    "DEFAULT_READ_ONLY_REASON",
    "CaptureLedger",
    "CapturedWrite",
    "DryRunWriteRefusedError",
    "LocalStore",
    "PointerConflictError",
    "S3Store",
    "Store",
    "active_capture_ledger",
    "begin_capture",
    "capture_ledger_of",
    "capturing",
    "end_capture",
    "is_capturing",
    "open_store",
    "parse_store_scheme",
    "read_only",
    "resolve_store_uri",
    "sha256_hex",
]

#: The longest lifetime an S3 SigV4 presigned GET may declare: seven days.
#: A larger ``ExpiresIn`` is not clamped by S3 — the URL is simply rejected on
#: use, which turns a caller's arithmetic error into a link that looks right
#: and 403s, so :meth:`Store.presigned_url` refuses it at construction instead.
PRESIGN_MAX_S = 7 * 24 * 3600

#: The version token that means "this key does not exist yet". Passed as
#: ``expected`` to :meth:`Store.compare_and_swap` to express "create it,
#: and fail if someone beat me to it" — the create half of the same
#: primitive, so a pointer's first write and its every later write go
#: through one code path rather than two with different race properties.
ETAG_ABSENT = "\x00absent"

#: What a :func:`read_only` refusal leads with when its caller named no other
#: reason. `--dry-run` was the only read-only caller until
#: `alpha-engine-config-I10576`, so it stays the default rather than becoming
#: a generic wording that tells an operator nothing.
DEFAULT_READ_ONLY_REASON = "--dry-run"


class PointerConflictError(RuntimeError):
    """A conditional write lost. The caller re-reads and decides.

    Never retried inside the store. A pointer flip that silently retried
    would overwrite whatever the winner just published, which is exactly
    the last-writer-wins failure the conditional PUT exists to prevent.
    """


class DryRunWriteRefusedError(RuntimeError):
    """A :data:`Store.MUTATORS` method was called on a store resolved with
    ``dry_run=True`` (alpha-engine-config-I9922, finding N1).

    `run_job(dry_run=True)` alone only stops `run_job` from writing its OWN
    manifest — it hands `fn` the same store either way, so a job body that
    calls `ctx.record_output`, `ctx.record_output_cas`, or `store.put_bytes`
    directly still reached the real backend under `--dry-run` (reproduced:
    a body calling `record_output("board/current.json", ...)` under
    `dry_run=True` left the artifact in the store with no manifest, while
    the runner printed "no outputs recorded"). `open_store(..., dry_run=True)`
    and `Settings.store(dry_run=True)` are the two places every handler
    resolves a store, and both now return a :func:`read_only` wrapper: every
    read behaves identically, and every write raises this, loudly, before it
    reaches the backend — a job whose body never checked `args.dry_run` at
    all (`gate`, `report`, `weekly`, `smoke`, `release.pin`, `alerts.sweep`,
    `heartbeat`, `drift`, `console`) now fails loudly on the write it
    attempts rather than silently succeeding.
    """


_READ_ONLY_CLASS_CACHE: dict[type, type] = {}


def _read_only_class(base: type) -> type:
    """A subclass of ``base`` whose every :data:`Store.MUTATORS` method
    refuses. Built once per concrete backend class and cached.

    A dynamic **subclass**, not a composition wrapper, deliberately:
    `release_retention.py`, `release.py` and `release_lock_sweep.py` each do
    `isinstance(store, S3Store)` to refuse a laptop-directory store outright,
    and a wrapper that only delegated to an inner store would fail every one
    of those checks under `--dry-run` — trading the write-leak bug this class
    fixes for a silent wrong-backend-type bug instead. A real subclass keeps
    `isinstance` (and every other attribute — `.bucket`, `.client`, `.root`)
    true and working; only the two declared `MUTATORS` are overridden.
    """
    cached = _READ_ONLY_CLASS_CACHE.get(base)
    if cached is not None:
        return cached

    def _refuse(self: Any, key: str = "<unknown key>", *args: Any, **kwargs: Any) -> Any:
        # The REASON is carried on the instance, not baked into this closure:
        # the class is cached per backend, and `--dry-run` is no longer the
        # only caller that resolves a store read-only
        # (`alpha-engine-config-I10576` made a non-`--publish` `crucible gate`
        # read-only too). A refusal naming `--dry-run` on a run that never
        # passed it sends the operator looking for a flag they did not set.
        why = getattr(self, "_read_only_reason", None) or DEFAULT_READ_ONLY_REASON
        raise DryRunWriteRefusedError(
            f"{why}: refusing to write {key!r} — this store was resolved read-only "
            "(crucible.store.read_only, reached via open_store(..., dry_run=True), "
            "crucible.config.Settings.store(dry_run=True), or a handler that wraps "
            "its own store). A read must not reach the backend; see "
            "DryRunWriteRefusedError's own docstring."
        )

    namespace = {name: _refuse for name in Store.MUTATORS}
    cls = type(f"ReadOnly{base.__name__}", (base,), namespace)
    _READ_ONLY_CLASS_CACHE[base] = cls
    return cls


def read_only(store: Store, *, reason: str | None = None) -> Store:
    """``store``, wrapped so every :data:`Store.MUTATORS` call raises
    :class:`DryRunWriteRefusedError` instead of reaching the backend.

    Every :data:`Store.READERS` method, and every other attribute, behaves
    exactly as it does on ``store`` — this IS that object's state, under a
    subclass with two methods overridden, not a copy or a second connection.

    ``reason`` is the prefix the refusal message leads with, naming WHICH
    read-only invocation refused (`alpha-engine-config-I10576`). It defaults
    to ``--dry-run`` because that was the only caller for as long as there was
    only one; a second caller passing its own reason is what keeps the message
    from naming a flag the operator never typed.
    """
    cls = _read_only_class(type(store))
    wrapped = object.__new__(cls)
    wrapped.__dict__.update(store.__dict__)
    if reason:
        wrapped._read_only_reason = reason
    return wrapped


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

    #: Every method on this interface that WRITES. Declared, so a guard can be
    #: derived from it rather than from a literal list kept in step by hand —
    #: `tests/test_board.py::_refusing_store` installs a refusal per entry, and
    #: a third mutator added without a line here would leave that guard
    #: silently blind to it.
    #:
    #: `compare_and_swap` is the reason this exists: it writes WITHOUT going
    #: through `put_bytes` (a temp file plus `os.replace` locally, a
    #: conditional PUT on S3), so a "did it write?" check patching `put_bytes`
    #: alone was beaten by inserting one call.
    MUTATORS: tuple[str, ...] = ("put_bytes", "compare_and_swap")

    #: Every method on this interface that only READS. Declared as the
    #: complement so the two together must PARTITION the interface, which is
    #: what makes `MUTATORS` checkable without guessing from a signature.
    #:
    #: An earlier guard derived the write set from `"payload" in
    #: signature.parameters`. That is a heuristic on a parameter NAME: adding
    #: `delete(self, key)` — which takes no payload by definition — or
    #: `append_bytes(self, key, data)` left the derivation returning the same
    #: two names, the guard green, and `_refusing_store` installing no refusal
    #: for either. A partition cannot miss a method that way: a new abstract
    #: method belongs to one list or the other, and belonging to neither is
    #: the failure.
    READERS: tuple[str, ...] = (
        "get_bytes",
        "exists",
        "list_keys",
        "etag",
        "presigned_url",
        "assert_keys_bind_to_trading_days",
    )

    @abstractmethod
    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        object_lock_mode: str | None = None,
        object_lock_retain_until: dt.datetime | None = None,
    ) -> str:
        """Write ``payload`` at ``key``; return its sha256 hex digest.

        ``object_lock_mode`` / ``object_lock_retain_until`` request S3 Object
        Lock retention on THIS write, atomically with the bytes
        (alpha-engine-config-I9787) — never as a follow-up call, which leaves
        a window in which the object exists published and unlocked. Omitted
        (both ``None``) means "no retention requested", the ordinary case for
        every key that is not a locked release artifact.

        A backend that cannot honour a real request refuses rather than
        silently accepting and discarding it: see :class:`LocalStore`, which
        has no Object Lock concept and would otherwise make the test suite
        green over a guarantee the local backend does not provide.
        """

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

    @abstractmethod
    def etag(self, key: str) -> str:
        """An opaque version token for ``key``, or :data:`ETAG_ABSENT`.

        Absence is a token rather than ``None`` so a caller cannot forget to
        handle it: passing the token straight back to
        :meth:`compare_and_swap` is the create case, and there is no third
        thing to write.
        """

    @abstractmethod
    def compare_and_swap(self, key: str, expected: str, payload: bytes) -> str:
        """Write ``payload`` at ``key`` only if its version is ``expected``.

        Returns the new version token. Raises :class:`PointerConflictError`
        when the current version differs — which is the whole point: the
        release pointer and the champion pointers are single objects written
        by more than one actor, and last-writer-wins gives the verdict to
        whichever writer finished last rather than to the one that checked.
        """

    @abstractmethod
    def presigned_url(self, key: str, expires_s: int) -> str:
        """A URL that reads ``key`` without the reader holding a credential.

        The one reason this exists: `alpha-engine-config-I9921`. The morning
        report is a Telegram message, and a message that says "the board is
        red" without a way to open the board makes the reader ask an agent —
        which is the same defect the board's `means_when_red` column exists to
        remove one layer down.

        **A missing key raises `KeyError`.** S3 will happily presign a key
        that does not exist and hand back a URL that 403s on use, which puts a
        broken link on the operator's phone under a report claiming the render
        succeeded. Absence is a first-class fact in this interface everywhere
        else (:meth:`get_bytes`, :meth:`exists`); it is one here too.

        **``expires_s`` is validated, never clamped** — see
        :data:`PRESIGN_MAX_S`. Silently shortening a caller's requested
        lifetime would produce a link that expires at a time nobody stated.

        This is a READER: it takes no lock, writes nothing and is declared in
        :data:`READERS` accordingly.
        """

    def _assert_writable_key(self, key: str) -> None:
        """Raise whatever this backend would raise for ``key`` on a WRITE,
        without writing.

        Exists for :func:`capturing`, which never reaches the backend's own
        write path and would otherwise report a key the real run refuses —
        a rehearsal that cannot fail the way the run fails, which is the
        whole defect the capturing store exists to remove
        (alpha-engine-config-I11012).

        Private on purpose. :data:`MUTATORS` and :data:`READERS` must
        PARTITION this interface's public surface
        (`tests/test_board.py::test_the_mutator_declaration_partitions_the_
        interface`), and this is neither: it writes nothing and reads
        nothing, it only refuses. A public name here would have to be
        classified as one or the other, and both classifications would be
        false.

        The base implementation refuses nothing because :class:`S3Store` has
        no key refusal today; :class:`LocalStore` does, and overrides.
        """
        return None

    @staticmethod
    def _validated_expiry(expires_s: int) -> int:
        """``expires_s``, or a refusal naming the bound it broke."""
        if not isinstance(expires_s, int) or isinstance(expires_s, bool):
            raise TypeError(
                f"expires_s must be an int number of seconds, not {type(expires_s).__name__}"
            )
        if expires_s < 1 or expires_s > PRESIGN_MAX_S:
            raise ValueError(
                f"expires_s={expires_s} is outside 1..{PRESIGN_MAX_S} seconds. S3 does not "
                "clamp an over-long lifetime — it rejects the URL on use, so a link built "
                "from it looks correct and 403s."
            )
        return expires_s

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

    def _assert_writable_key(self, key: str) -> None:
        """The same refusal :meth:`put_bytes` takes, reached without writing.

        `_path` is where an absolute or traversing key is refused, and a
        capturing dry run never calls it — so without this override a dry run
        would report `"/etc/passwd"` as a key it would write and the real run
        would raise on it.
        """
        self._path(key)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(
                f"refusing key {key!r}: store keys are relative and never traverse. "
                "An absolute or traversing key would write outside the store root."
            )
        return self.root / key

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        object_lock_mode: str | None = None,
        object_lock_retain_until: dt.datetime | None = None,
    ) -> str:
        if object_lock_mode is not None or object_lock_retain_until is not None:
            # Historical citation, not a phase pointer: alpha-engine-config-I9787 is
            # where this gap was first found. Kept in this comment rather than the
            # raised message per alpha-engine-config-I9839.
            #
            # NOTE (alpha-engine-config-I9817): this guard is NOT reachable from
            # `crucible.release.publish_release` / `crucible.deploy._publish`
            # today. Both resolve their lock params through
            # `crucible.release.release_object_lock_params`, which returns
            # `(None, None)` for any non-`S3Store` — so a LocalStore-backed
            # release publish never reaches this branch at all; it silently
            # publishes with no retention claim, which is correct (a laptop
            # publish never claims retention), but means this raise defends
            # only a caller that constructs `object_lock_mode`/
            # `object_lock_retain_until` directly rather than through that
            # function — a hand-rolled writer, or a future local backend that
            # gains a partial Object Lock concept. It is not, itself, what
            # stops the test suite from going green over an unenforced
            # guarantee on the release path; `release_object_lock_params`'s
            # `(None, None)` branch is.
            raise NotImplementedError(
                f"LocalStore has no Object Lock concept and cannot honour "
                f"object_lock_mode={object_lock_mode!r} for {key!r}. Accepting and "
                "silently discarding it would make the test suite green over a "
                "guarantee the local backend does not provide. Use S3Store for a "
                "release publish that needs retention."
            )
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

    def etag(self, key: str) -> str:
        """The object's sha256, or :data:`ETAG_ABSENT`.

        Content-derived rather than a mtime or a counter, so it survives a
        copy of the store directory — a laptop backend whose version tokens
        changed when the tree was moved would refuse every conditional write
        after a `cp -r`.
        """
        path = self._path(key)
        if not path.is_file():
            return ETAG_ABSENT
        return sha256_hex(path.read_bytes())

    def compare_and_swap(self, key: str, expected: str, payload: bytes) -> str:
        """Check-then-write under an exclusive create, then rename.

        **Weaker than S3's conditional PUT, and deliberately named so.** Two
        processes racing on the same directory could both observe the same
        `expected` before either writes. The exclusive temp file narrows the
        window to the check itself rather than removing it, which is honest
        for the backend whose stated purpose is "a developer runs the whole
        thing against a directory": the production pointer lives in S3, where
        :class:`S3Store` makes this atomic at the service.
        """
        current = self.etag(key)
        if current != expected:
            raise PointerConflictError(
                f"{key!r} is at version {current[:12]!r}, not the expected "
                f"{expected[:12]!r}. Re-read it and decide — a retry here would "
                "overwrite whatever the winning writer just published."
            )
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "xb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
        return sha256_hex(payload)

    def presigned_url(self, key: str, expires_s: int) -> str:
        """A `file://` URL for the object on disk.

        There is no signature and nothing expires — a local directory has no
        credential to sign with — so ``expires_s`` is validated and then
        discarded. Validated rather than ignored: a caller whose arithmetic
        would be refused by the S3 backend must be refused by this one too,
        or the whole test suite passes over a bound production enforces.

        Deliberately NOT a rendered fake signature. A local URL that looked
        presigned would make a test asserting "the message carries a
        presigned URL" pass against a string nobody can use, which is the
        shape of a green test over an absent capability.
        """
        self._validated_expiry(expires_s)
        path = self._path(key)
        if not path.is_file():
            raise KeyError(
                f"{key!r} is not present in {self.root}, so a URL to it would be a link "
                "to nothing on the operator's phone."
            )
        return path.resolve().as_uri()

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        # Walk only the directory the prefix can live under, not the whole
        # root: every key that starts with `a/b/c` sits below `root/a/b`, and
        # the `startswith` filter below still decides membership, so the
        # result is identical. A prefix that is absolute or traverses is never
        # a key's prefix (`_path` refuses both), so it keeps the full walk and
        # yields nothing, exactly as before.
        head = prefix.rpartition("/")[0]
        base = self.root
        if head and not head.startswith("/") and ".." not in head.split("/"):
            base = self.root / head
            if not base.is_dir():
                return
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                yield key


class S3Store(Store):
    """The production backend: one bucket, one prefix, no other AWS resource.

    Three things it does that a naive wrapper does not, each because the
    naive version has already cost the fleet something:

    * **It paginates.** A truncated listing read as complete is the
      `--limit N` bug class — five sweeps once read a 2049-item backlog under
      a limit and one published report was 7 of 9 false.
    * **It distinguishes absent from unreadable.** Absence is one of the two
      page conditions (§4.6); an `AccessDenied` rendered as "not there yet"
      would page for a missing artifact that exists and is simply unreachable,
      and the operator would go looking for the wrong thing.
    * **It swaps conditionally.** `IfMatch` / `IfNoneMatch` on the pointer
      objects, so the release flip is a compare-and-swap at the service rather
      than a read-then-write with a window in the middle.

    ``boto3`` is imported lazily, inside the client property. The laptop path,
    the tests and `crucible --help` all import this module; none of them
    should need an AWS SDK on the import path, and a Lambda cold start should
    not pay for one it may never call.
    """

    def __init__(self, bucket: str, prefix: str = "", *, client: Any = None) -> None:
        if not bucket:
            raise ValueError("S3Store needs a bucket; an empty one would write nowhere")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # noqa: PLC0415 - lazy on purpose; see the class docstring

            self._client = boto3.client("s3")
        return self._client

    def _s3_key(self, key: str) -> str:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"refusing key {key!r}: store keys are relative and never traverse.")
        return f"{self.prefix}/{key}" if self.prefix else key

    def _strip(self, s3_key: str) -> str:
        if self.prefix and s3_key.startswith(f"{self.prefix}/"):
            return s3_key[len(self.prefix) + 1 :]
        return s3_key

    @staticmethod
    def _error_code(exc: Any) -> str:
        return str(exc.response.get("Error", {}).get("Code", ""))

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        object_lock_mode: str | None = None,
        object_lock_retain_until: dt.datetime | None = None,
    ) -> str:
        lock_kwargs: dict[str, Any] = {}
        if object_lock_mode is not None:
            lock_kwargs["ObjectLockMode"] = object_lock_mode
        if object_lock_retain_until is not None:
            lock_kwargs["ObjectLockRetainUntilDate"] = object_lock_retain_until
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._s3_key(key),
            Body=payload,
            ContentType=(
                "application/json" if key.endswith(".json") else "application/octet-stream"
            ),
            Metadata={"sha256": sha256_hex(payload)},
            **lock_kwargs,
        )
        return sha256_hex(payload)

    def get_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._s3_key(key))["Body"].read()
        except ClientError as exc:
            if self._error_code(exc) in ("NoSuchKey", "404"):
                raise KeyError(
                    f"{key!r} is not present in s3://{self.bucket}/{self.prefix}"
                ) from exc
            raise

    def exists(self, key: str) -> bool:
        """True/False for present/absent; RAISES for unreadable.

        The third case is the one that matters. An `AccessDenied` returned as
        `False` is an absence page for an artifact that is there, and the
        operator spends the morning looking for a producer that ran fine.
        """
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._s3_key(key))
            return True
        except ClientError as exc:
            if self._error_code(exc) in ("404", "NoSuchKey"):
                return False
            raise

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        scope = self._s3_key(prefix) if prefix else self.prefix
        for page in paginator.paginate(Bucket=self.bucket, Prefix=scope):
            for obj in page.get("Contents", []):
                yield self._strip(obj["Key"])

    def presigned_url(self, key: str, expires_s: int) -> str:
        """A SigV4 presigned GET, signed with whatever credential this process holds.

        **The stated lifetime is an upper bound, not a guarantee.** A URL
        signed with TEMPORARY credentials — every assumed role, which is what
        every crucible job runs as — stops working when the underlying session
        token expires, whichever comes first. So a 7-day presign taken by a
        GitHub Actions OIDC role survives that role's session and no longer;
        the caller renders that caveat beside the link
        (`crucible.morning.BOARD_URL_CAVEAT`) rather than quoting an expiry
        the credential cannot honour. The durable fix is a page on the fleet
        console, filed separately — a presigned URL is the surface that needs
        no new identity, not the surface that should exist forever.

        No bucket, account or ARN literal reaches this file: the bucket comes
        from the store URI the job was given (`crucible/AGENTS.md`).
        """
        self._validated_expiry(expires_s)
        if not self.exists(key):
            raise KeyError(
                f"{key!r} is not present in s3://{self.bucket}/{self.prefix}, so a URL to "
                "it would be a link that 403s under a report claiming a successful render."
            )
        return str(
            self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": self._s3_key(key)},
                ExpiresIn=expires_s,
            )
        )

    def etag(self, key: str) -> str:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            head = self.client.head_object(Bucket=self.bucket, Key=self._s3_key(key))
        except ClientError as exc:
            if self._error_code(exc) in ("404", "NoSuchKey"):
                return ETAG_ABSENT
            raise
        return str(head["ETag"]).strip('"')

    def compare_and_swap(self, key: str, expected: str, payload: bytes) -> str:
        """`IfNoneMatch='*'` to create, `IfMatch=<etag>` to replace.

        The same primitive `krepis.locks` uses for the universe-writer lock,
        applied to a pointer rather than a lock. S3 returns
        `PreconditionFailed` on both conditions; older SDK mocks surface the
        bare `412`, and both are accepted for the same reason krepis accepts
        both.
        """
        from botocore.exceptions import ClientError  # noqa: PLC0415

        condition = {"IfNoneMatch": "*"} if expected == ETAG_ABSENT else {"IfMatch": expected}
        try:
            resp = self.client.put_object(
                Bucket=self.bucket,
                Key=self._s3_key(key),
                Body=payload,
                ContentType="application/json",
                Metadata={"sha256": sha256_hex(payload)},
                **condition,
            )
        except ClientError as exc:
            conflicts = ("PreconditionFailed", "412", "ConditionalRequestConflict", "409")
            if self._error_code(exc) in conflicts:
                raise PointerConflictError(
                    f"conditional write of {key!r} lost: expected version "
                    f"{expected[:12]!r}. Re-read the pointer and decide."
                ) from exc
            raise
        return str(resp.get("ETag", "")).strip('"')


# ── The write-capturing store (alpha-engine-config-I11012) ──────────────────

#: The scheme :meth:`presigned_url` returns for a key that exists only as a
#: CAPTURED write. Deliberately not a `file://` or `https://` URL: a dry run
#: that handed back something that looked openable would put a dead link in
#: front of whoever is reading the rehearsal.
CAPTURED_URL_SCHEME = "crucible-dry-run://captured/"


@dataclass(frozen=True)
class CapturedWrite:
    """One write a dry run would have made, recorded instead of performed.

    ``schema_version`` is read out of the payload itself when the payload is
    a JSON object declaring one, and is ``None`` otherwise. It is NOT taken
    from `RunContext.record_output`'s `schema_version` argument: that value
    never reaches the store (see `crucible.runner.RunContext.record_output`,
    which hands `put_bytes` the bytes alone), and inventing a default here
    would put a version on a parquet blob that declares none.
    """

    key: str
    method: str
    size_bytes: int
    sha256: str
    schema_version: str | None
    object_lock_mode: str | None = None


class CaptureLedger:
    """The ordered record of what a dry run would have written.

    One ledger spans a whole invocation, not one store: a job that resolves
    the store once and a `weekly` arc that runs twelve stages in one process
    both answer "which keys would this command write" from the same object.
    """

    def __init__(self) -> None:
        self._writes: list[CapturedWrite] = []

    def record(self, write: CapturedWrite) -> None:
        self._writes.append(write)

    @property
    def writes(self) -> tuple[CapturedWrite, ...]:
        """Every captured write, in the order the job made it. A key written
        twice appears twice — a job that overwrites its own output inside one
        run is a fact about that job, not noise to deduplicate away."""
        return tuple(self._writes)

    @property
    def keys(self) -> tuple[str, ...]:
        """The distinct keys, first-write order. This is the set the
        Closes-when property compares against a real run's written keys."""
        seen: dict[str, None] = {}
        for write in self._writes:
            seen.setdefault(write.key, None)
        return tuple(seen)

    def render(self) -> str:
        """The operator-facing report. One line per write, key first."""
        if not self._writes:
            return "dry_run: no store writes captured — this command would write nothing."
        lines = [f"dry_run: {len(self._writes)} store write(s) captured, none performed:"]
        for write in self._writes:
            schema = write.schema_version or "-"
            lock = f" object_lock={write.object_lock_mode}" if write.object_lock_mode else ""
            lines.append(
                f"  {write.key}  ({write.method}, {write.size_bytes} bytes, schema {schema}){lock}"
            )
        return "\n".join(lines)


#: The ledger every :func:`capturing` store built during THIS invocation
#: records into, when one invocation has declared itself
#: (:func:`begin_capture`). ``None`` outside one, in which case each capturing
#: store gets a private ledger — a library caller wrapping a store by hand
#: still gets a complete record of its own.
_ACTIVE_LEDGER: CaptureLedger | None = None


def begin_capture() -> CaptureLedger:
    """Declare this invocation's capture ledger and return it.

    Re-entrant by returning the ledger already active rather than replacing
    it: `crucible weekly --dry-run` re-enters `crucible.cli.main` once per
    stage in the same process, and a nested `begin_capture` that started a
    fresh ledger would throw away every stage before it.

    The caller that actually STARTED the ledger — the one that saw
    :func:`active_capture_ledger` return ``None`` — is the one that calls
    :func:`end_capture`.
    """
    global _ACTIVE_LEDGER
    if _ACTIVE_LEDGER is None:
        _ACTIVE_LEDGER = CaptureLedger()
    return _ACTIVE_LEDGER


def active_capture_ledger() -> CaptureLedger | None:
    """This invocation's ledger, or ``None`` outside a declared capture."""
    return _ACTIVE_LEDGER


def end_capture() -> None:
    """Clear the invocation ledger. Idempotent."""
    global _ACTIVE_LEDGER
    _ACTIVE_LEDGER = None


def _schema_version_of(payload: bytes) -> str | None:
    """The `schema_version` a JSON payload declares, or ``None``.

    ``None`` is a real answer, not a swallowed failure: parquet blobs, the
    strategy YAML tree and the release pointer all legitimately declare no
    schema version at the top level, and a payload this cannot parse is a
    payload with no declaration to report. Nothing downstream treats ``None``
    as a pass — it is printed as `-` and compared as `None`.
    """
    if payload[:1] != b"{":
        return None
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    version = document.get("schema_version")
    return version if isinstance(version, str) else None


_CAPTURING_CLASS_CACHE: dict[type, type] = {}


def _capturing_class(base: type) -> type:
    """A subclass of ``base`` that RECORDS every :data:`Store.MUTATORS` call
    and serves its own recorded bytes back to the readers.

    A dynamic **subclass**, for the same measured reason
    :func:`_read_only_class` is one: `release_retention.py`, `release.py` and
    `release_lock_sweep.py` each do `isinstance(store, S3Store)`, and a
    composition wrapper would fail every one of those checks under
    `--dry-run`.

    **There is no flag on this class.** Its `put_bytes` does not call
    `super().put_bytes`; there is no branch on which a production write could
    be reached, because the code that reaches the backend is not present in
    the override at all. That is the difference between "a real Store with
    writes disabled" — one `if` away from a production write — and this.

    **The reads are real, and they see the run's own writes.** A job that
    writes something and then reads it back is the ordinary shape (the arm
    register, the board, every pointer), and a rehearsal whose read raised
    `KeyError` where the real run succeeds would fail in a way the run does
    not — the mirror image of the defect this closes. So `get_bytes`,
    `exists`, `etag`, `list_keys` and `presigned_url` consult the captured
    overlay first and fall through to the real backend, which is untouched.

    **`compare_and_swap` still conflicts.** It checks ``expected`` against the
    overlay-aware version exactly as the backend would and raises
    :class:`PointerConflictError` on a mismatch, so a dry run of a job racing
    a pointer it did not re-read fails the way the run fails.
    """
    cached = _CAPTURING_CLASS_CACHE.get(base)
    if cached is not None:
        return cached

    def _capture(
        self: Any,
        key: str,
        payload: bytes,
        *,
        method: str,
        object_lock_mode: str | None = None,
    ) -> None:
        # The backend's own refusal, reached without writing: a key the real
        # run would reject must not be reported as a key it would write.
        self._assert_writable_key(key)
        self._captured_bytes[key] = payload
        self._capture_ledger.record(
            CapturedWrite(
                key=key,
                method=method,
                size_bytes=len(payload),
                sha256=sha256_hex(payload),
                schema_version=_schema_version_of(payload),
                object_lock_mode=object_lock_mode,
            )
        )

    def put_bytes(
        self: Any,
        key: str,
        payload: bytes,
        *,
        object_lock_mode: str | None = None,
        object_lock_retain_until: dt.datetime | None = None,
    ) -> str:
        _capture(self, key, payload, method="put_bytes", object_lock_mode=object_lock_mode)
        # The content digest, which is what every backend's `put_bytes`
        # returns and what `record_output` re-derives for itself anyway.
        return sha256_hex(payload)

    def compare_and_swap(self: Any, key: str, expected: str, payload: bytes) -> str:
        current = self.etag(key)
        if current != expected:
            raise PointerConflictError(
                f"{key!r} is at version {current[:12]!r}, not the expected "
                f"{expected[:12]!r}. This is a DRY RUN: the conflict is real — the "
                "pointer in the store really is at another version — and the real run "
                "would take it too."
            )
        _capture(self, key, payload, method="compare_and_swap")
        return sha256_hex(payload)

    def get_bytes(self: Any, key: str) -> bytes:
        staged = self._captured_bytes.get(key)
        if staged is not None:
            return staged
        return base.get_bytes(self, key)

    def exists(self: Any, key: str) -> bool:
        return key in self._captured_bytes or base.exists(self, key)

    def etag(self: Any, key: str) -> str:
        staged = self._captured_bytes.get(key)
        if staged is not None:
            # The content digest, not the backend's own token shape. Within
            # one dry run it is only ever compared to itself — it is handed
            # to `compare_and_swap` above and nowhere else — and a dry run
            # cannot know what ETag S3 would have minted for bytes it never
            # sent.
            return sha256_hex(staged)
        return base.etag(self, key)

    def list_keys(self: Any, prefix: str = "") -> Iterator[str]:
        seen: set[str] = set()
        for key in base.list_keys(self, prefix):
            seen.add(key)
            yield key
        for key in sorted(self._captured_bytes):
            if key.startswith(prefix) and key not in seen:
                yield key

    def presigned_url(self: Any, key: str, expires_s: int) -> str:
        if key in self._captured_bytes:
            # Validated exactly as the backend validates it, so a caller
            # whose arithmetic production would refuse is refused here too.
            self._validated_expiry(expires_s)
            return f"{CAPTURED_URL_SCHEME}{key}"
        return base.presigned_url(self, key, expires_s)

    namespace = {
        "put_bytes": put_bytes,
        "compare_and_swap": compare_and_swap,
        "get_bytes": get_bytes,
        "exists": exists,
        "etag": etag,
        "list_keys": list_keys,
        "presigned_url": presigned_url,
    }
    missing = set(Store.MUTATORS) - set(namespace)
    if missing:
        # A third mutator added to the interface without a capture override
        # here would reach the real backend under `--dry-run`. Refused at
        # class construction, which is the first moment it can be seen.
        raise RuntimeError(
            f"crucible.store.capturing has no override for {sorted(missing)}, which "
            "Store.MUTATORS declares. A mutator with no capture override writes to the "
            "real backend under --dry-run."
        )
    cls = type(f"Capturing{base.__name__}", (base,), namespace)
    _CAPTURING_CLASS_CACHE[base] = cls
    return cls


def capturing(store: Store, *, ledger: CaptureLedger | None = None) -> Store:
    """``store``, with its writes RECORDED instead of performed.

    This is what `--dry-run` resolves (alpha-engine-config-I11012). It
    subsumes :func:`read_only` for that caller: `read_only` made a dry run
    safe by making it stop at the first write, which meant no job body ever
    ran far enough for the rehearsal to fail the way the run fails. The
    measured cost of that: a laptop dry run of `experiment.backfill --slot u
    --arm attractiveness` reported "would produce 205 session(s)" for a
    command that died on the FIRST session in production two minutes in.

    `read_only` is NOT retired — `crucible gate` without `--publish`,
    `migrate.history`'s v1 source and `crucible.faults`' unpublishable pin
    are real runs that must not write, where a refusal is the correct answer
    and a captured write would be a fiction.

    ``ledger`` defaults to this invocation's ledger when one is active
    (:func:`begin_capture`) and to a fresh private one otherwise.
    """
    cls = _capturing_class(type(store))
    wrapped = object.__new__(cls)
    wrapped.__dict__.update(store.__dict__)
    wrapped._captured_bytes = {}
    wrapped._capture_ledger = ledger or active_capture_ledger() or CaptureLedger()
    return wrapped


def is_capturing(store: Any) -> bool:
    """Whether ``store`` records its writes rather than performing them.

    The predicate `crucible.runner.run_job` asks before letting a dry run
    assemble and record its own manifest.
    """
    return isinstance(getattr(store, "_capture_ledger", None), CaptureLedger)


def capture_ledger_of(store: Any) -> CaptureLedger | None:
    """``store``'s ledger, or ``None`` when it is not a capturing store."""
    ledger = getattr(store, "_capture_ledger", None)
    return ledger if isinstance(ledger, CaptureLedger) else None


def resolve_store_uri(uri: str | None) -> str:
    """The store URI a caller actually gets, `--store` or `$CRUCIBLE_STORE`.

    Extracted from :func:`open_store` so a caller that must RECORD which store
    it read — `crucible gate --closing-comment`, whose block is worthless
    without the store its `gate_artifact` key sits in
    (`alpha-engine-config-I9967`) — resolves it through the same one place
    rather than restating the `or os.environ.get(...)` fallback and drifting
    from it the first time the default moves.
    """
    target = uri or os.environ.get("CRUCIBLE_STORE")
    if not target:
        raise ValueError(
            "no store: pass --store s3://bucket/prefix or a directory path, or set "
            "CRUCIBLE_STORE. There is no default production bucket on purpose — a job "
            "that wrote to production because a flag was missing is noticed once."
        )
    return target


def parse_store_scheme(uri: str) -> tuple[str, str]:
    """Classify ``uri``'s scheme: ``("s3", rest)`` or ``("local", uri)``.

    The one place `s3://` vs an unsupported scheme vs a local directory is
    told apart. `open_store` (this module) and `crucible.config.store_from_uri`
    each validated an independent copy of this same three-branch check —
    `store_from_uri` had it correctly from the start; `open_store` gained a
    textually separate copy in `alpha-engine-config-I9817` to fix a measured
    defect (`--store file://./store` silently wrote into a literal `file:`
    directory). `alpha-engine-config-I10519` collapses both copies into this
    one function; each caller still applies its own bucket-parsing and
    local-path construction on top; see `open_store` and `store_from_uri`,
    which differ slightly in exactly that step (`~`-expansion and
    slash-stripping) and stay that way rather than being silently unified
    along with the scheme check.

    For ``"s3"``, ``rest`` is the URI's substring after ``s3://``,
    unstripped and unpartitioned. RAISES for any other ``://`` scheme:
    reading an unknown scheme as a directory name would silently write a
    production run into a folder named after the scheme and report success.
    """
    if uri.startswith("s3://"):
        return "s3", uri[len("s3://") :]
    if "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ValueError(
            f"unsupported store scheme {scheme!r} in {uri!r}. The supported "
            "backends are `s3://bucket/prefix` and a local directory path; an "
            "unknown scheme is a typo, and reading it as a directory name would "
            "write a production run into a folder named after the scheme and "
            "report success."
        )
    return "local", uri


def open_store(uri: str | None, *, dry_run: bool = False) -> Store:
    """`s3://bucket/prefix` or a directory path, resolved to a backend.

    One factory so `--store` means the same thing to every job, and so the
    default lives in exactly one place. The default is the environment's
    `CRUCIBLE_STORE`; there is deliberately no hardcoded production bucket
    fallback — a job that silently wrote to production because a flag was
    missing is the kind of default that is only noticed once.

    ``dry_run=True`` returns the backend wrapped by :func:`capturing`
    (alpha-engine-config-I11012, superseding the :func:`read_only` wrap of
    -I9922) — every CLI handler resolves its store through this function (or
    `crucible.config.Settings.store`, which wraps the same way) with
    `dry_run=bool(args.dry_run)`, so `--dry-run` is true of every job
    regardless of whether that job's own handler body checks the flag.

    The wrap CHANGED rather than moved. `read_only` made a dry run safe by
    refusing at the first write, so no job body ever ran past it and no
    rehearsal could fail the way its run fails; `capturing` runs the real
    body against real reads and records the key set it would have written.
    A dry run still performs zero writes — that is structural, not a flag:
    the capturing subclass has no code path to the backend's write at all.
    """
    target = resolve_store_uri(uri)
    kind, rest = parse_store_scheme(target)
    if kind == "s3":
        bucket, _, prefix = rest.partition("/")
        store: Store = S3Store(bucket, prefix)
    else:
        store = LocalStore(target)
    return capturing(store) if dry_run else store
