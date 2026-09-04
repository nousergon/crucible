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
import os
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

__all__ = [
    "ETAG_ABSENT",
    "PRESIGN_MAX_S",
    "DryRunWriteRefusedError",
    "LocalStore",
    "PointerConflictError",
    "S3Store",
    "Store",
    "open_store",
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
        raise DryRunWriteRefusedError(
            f"--dry-run: refusing to write {key!r} — this store was resolved read-only "
            "(crucible.store.open_store(..., dry_run=True) / "
            "crucible.config.Settings.store(dry_run=True)). A dry run must not reach "
            "the backend; see DryRunWriteRefusedError's own docstring."
        )

    namespace = {name: _refuse for name in Store.MUTATORS}
    cls = type(f"ReadOnly{base.__name__}", (base,), namespace)
    _READ_ONLY_CLASS_CACHE[base] = cls
    return cls


def read_only(store: Store) -> Store:
    """``store``, wrapped so every :data:`Store.MUTATORS` call raises
    :class:`DryRunWriteRefusedError` instead of reaching the backend.

    Every :data:`Store.READERS` method, and every other attribute, behaves
    exactly as it does on ``store`` — this IS that object's state, under a
    subclass with two methods overridden, not a copy or a second connection.
    """
    cls = _read_only_class(type(store))
    wrapped = object.__new__(cls)
    wrapped.__dict__.update(store.__dict__)
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
        for path in sorted(self.root.rglob("*")):
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


def open_store(uri: str | None, *, dry_run: bool = False) -> Store:
    """`s3://bucket/prefix` or a directory path, resolved to a backend.

    One factory so `--store` means the same thing to every job, and so the
    default lives in exactly one place. The default is the environment's
    `CRUCIBLE_STORE`; there is deliberately no hardcoded production bucket
    fallback — a job that silently wrote to production because a flag was
    missing is the kind of default that is only noticed once.

    ``dry_run=True`` returns the backend wrapped by :func:`read_only`
    (alpha-engine-config-I9922) — every CLI handler resolves its store
    through this function (or `crucible.config.Settings.store`, which wraps
    the same way) with `dry_run=bool(args.dry_run)`, so `--dry-run` is now
    true of every job regardless of whether that job's own handler body
    checks the flag.
    """
    target = resolve_store_uri(uri)
    if target.startswith("s3://"):
        rest = target[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        store: Store = S3Store(bucket, prefix)
    else:
        store = LocalStore(target)
    return read_only(store) if dry_run else store
