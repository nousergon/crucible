"""Where the store, the strategy tree and the price source come from.

Normative source: plan §4.11 ("`alpha-engine-config/strategy/` is config, not
code"), §4.12, and principle 8.

**No literal bucket, prefix or path appears anywhere else in the package.**
The CloudFormation template that creates the v2 bucket lands with track C;
until it does, the bucket is an assumption recorded on
`alpha-engine-config-I9757` (the store bucket named there, prefix `crucible/`)
and resolved only through `CRUCIBLE_STORE_URI` / `--store` — never a literal
in this package, which goes PUBLIC at phase-1 exit (`crucible/AGENTS.md`
Visibility). It lives HERE, in one declared adapter, so the track-C change is
a one-line edit rather than a grep across the tree.

Three things are resolved, and each has exactly one resolution order —
explicit argument, then environment variable, then the declared default:

* the **store** (`--store` / ``CRUCIBLE_STORE``), an ``s3://bucket/prefix``
  URI or a directory path;
* the **strategy tree** (``--strategy-dir`` / ``CRUCIBLE_STRATEGY_DIR``), a
  checkout of ``alpha-engine-config/strategy/`` when running from a laptop;
  absent, arms are read from the store under ``strategy/current/``, which is
  what a spot instance sees;
* the **ArcticDB bucket** (``CRUCIBLE_ARCTIC_BUCKET``), the production price
  source's backing bucket;
* the **CloudTrail archive** (``CRUCIBLE_CLOUDTRAIL_ARCHIVE``), an
  ``s3://bucket/prefix`` URI the autonomy gate reads human-originated mutating
  calls from (§11 risk 8);
* the **stack name** (``CRUCIBLE_STACK``), the CloudFormation stack whose
  resources must all carry `system=crucible-v2` (§11 risk 7).

A resolution that fell through to a default says so in
:attr:`Settings.origins`, so `explain` can report *why* a run read what it
read (principle 1) instead of only what it read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.llm import DEFAULT_LLM_CAP_USD, DEFAULT_LLM_CAP_USD_MEASURED
from crucible.store import LocalStore, S3Store, Store, read_only

__all__ = [
    "DEFAULT_ARCTIC_BUCKET",
    "DEFAULT_CLOUDTRAIL_ARCHIVE",
    "DEFAULT_CONSOLE_URL",
    "DEFAULT_STACK_NAME",
    "DEFAULT_LLM_CAP_USD",
    "DEFAULT_LLM_CAP_USD_MEASURED",
    "DEFAULT_STORE_URI",
    "STRATEGY_PREFIX",
    "Settings",
    "settings",
    "store_from_uri",
]

#: **There is no default store, deliberately.** An earlier revision defaulted
#: to a production research bucket, which contradicted `store.open_store`'s own
#: refusal — "a job that wrote to production because a flag was missing is
#: noticed once" — and meant `crucible experiment.grade --slot r --dry-run`
#: with no `--store` wrote verdicts, the arm register, the arena cycle and
#: ledger rows into that bucket. The resolution now falls through to `None`,
#: and :meth:`Settings.store` refuses with the same message `open_store` uses,
#: so the two entry points agree.
DEFAULT_STORE_URI: str | None = None

#: The ArcticDB store is a v1 asset v2 READS and never writes to from a
#: laptop (the standing in-region rule). **No default, deliberately**
#: (`alpha-engine-config-I9906` finding 3): the bucket name is an
#: infrastructure identifier `crucible/AGENTS.md` (Visibility) forbids in this
#: repo, which goes PUBLIC at phase-1 exit. Set `CRUCIBLE_ARCTIC_BUCKET`;
#: `ArcticPriceSource.__init__` raises on an empty bucket rather than reaching
#: S3 with one — same shape as `DEFAULT_CLOUDTRAIL_ARCHIVE` below.
DEFAULT_ARCTIC_BUCKET = ""

#: The strategy tree's home inside the store. `current` is a pointer prefix,
#: not a mutable directory: a strategy change is a sync of a new tree, and
#: the run manifest records that tree's hash as an input.
STRATEGY_PREFIX = "strategy/current"

#: The CloudFormation stack the tag audit reads (§11 risk 7). One name, here,
#: so the audit carries no literal and a second account is one variable.
DEFAULT_STACK_NAME = "crucible-v2"

#: The CloudTrail archive the autonomy gate reads (§11 risk 8). Empty by
#: default and NOT a guess: the account had no trail at all when this was
#: written, and a plausible-looking bucket name here would have produced a
#: `NoSuchBucket` that reads like a permissions problem rather than the honest
#: answer, which is that the archive does not exist yet. `crucible.autonomy`
#: raises `ArchiveMissingError` on an empty value.
DEFAULT_CLOUDTRAIL_ARCHIVE = ""

#: The fleet console's base URL (`policy-console`), where the board's rows are
#: rendered as Decision entities at a STABLE address — the durable replacement
#: for the presigned board link the morning report otherwise carries
#: (`alpha-engine-config-I9926`). Empty by default, deliberately: a hostname is
#: an infrastructure identifier this tree carries none of (it goes public at
#: phase-1 exit), and an unset value means the report falls back to the
#: presigned page and SAYS so, rather than linking a console that may not have
#: the board yet. Set `CRUCIBLE_CONSOLE_URL`, scheme and host, no trailing slash.
DEFAULT_CONSOLE_URL = ""


@dataclass(frozen=True)
class Settings:
    """One resolved configuration, with the provenance of every value."""

    #: `None` when nothing resolved it. Not an error at resolution time — a
    #: `crucible report --help` or an LLM-cap lookup needs no store — but
    #: :meth:`store` refuses, so the failure lands where the write would.
    store_uri: str | None
    arctic_bucket: str
    strategy_dir: Path | None
    cloudtrail_archive: str = DEFAULT_CLOUDTRAIL_ARCHIVE
    stack_name: str = DEFAULT_STACK_NAME
    #: `alpha-engine-config-I9926` — see :data:`DEFAULT_CONSOLE_URL`. Read by
    #: `crucible.morning`; nothing else in this tree links out.
    console_url: str = DEFAULT_CONSOLE_URL
    origins: dict[str, str] = field(default_factory=dict)
    #: The per-weekly-run LLM spend ceiling, in USD (plan §2 row 3). Declared
    #: HERE, in config, rather than at a call site: a ceiling that lives beside
    #: the code it bounds is one that moves whenever that code is edited.
    #: `crucible.llm.SpendCap` refuses the call that would cross it, before the
    #: provider is reached, and the run fails rather than overspending.
    llm_cap_usd: float = DEFAULT_LLM_CAP_USD
    #: `alpha-engine-config-I9778`/`alpha-engine-config-I9823`: whether
    #: :data:`~crucible.llm.DEFAULT_LLM_CAP_USD` traces back to a phase-5
    #: cost-sink measurement — nothing else. Tracks
    #: `crucible.llm.DEFAULT_LLM_CAP_USD_MEASURED` exactly, and ONLY that: an
    #: operator override (`--llm-cap-usd` or `CRUCIBLE_LLM_CAP_USD`) does not
    #: flip it, on either value. The review that opened I9823 found the prior
    #: shape — `origins["llm_cap_usd"] != "default" or
    #: DEFAULT_LLM_CAP_USD_MEASURED` — let a caller re-declare the identical
    #: $5.00 default through the env var and have it read back as "measured";
    #: an assertion is not a measurement, and conflating the two made the
    #: flag flippable by anyone who could set an env var. Provenance of an
    #: operator override is already recorded, verbatim, in
    #: ``origins["llm_cap_usd"]`` (``"argument"`` / ``"environ:..."`` /
    #: ``"default"``) — nothing here needed a second, weaker channel to carry
    #: the same fact.
    llm_cap_usd_measured: bool = False
    #: alpha-engine-config-I9922. Set from `--dry-run` at resolution
    #: (`track_a._settings`), so every track-A handler's store is read-only
    #: without each of the eight handlers threading the flag individually —
    #: one field on the object every one of them already carries.
    dry_run: bool = False

    def store(self) -> Store:
        """The resolved store, or a refusal naming how to resolve one.

        Deliberately the same refusal `crucible.store.open_store` gives. Two
        entry points into the same decision that disagreed is what let a
        `--dry-run` reach a production bucket through one of them.

        Wrapped by :func:`crucible.store.read_only` when :attr:`dry_run` is
        set — the same wrapping `open_store(..., dry_run=True)` does, so a
        caller cannot get a real write out of `--dry-run` by resolving its
        store through this method instead of that function.
        """
        if not self.store_uri:
            raise ValueError(
                "no store was resolved. Pass `--store s3://bucket/prefix` or a "
                "directory path, or set CRUCIBLE_STORE. There is deliberately no "
                "hardcoded production bucket fallback — a job that wrote to "
                "production because a flag was missing is noticed once."
            )
        resolved = store_from_uri(self.store_uri)
        return read_only(resolved) if self.dry_run else resolved

    def cloudtrail_bucket_prefix(self) -> tuple[str, str]:
        """The archive URI split into bucket and prefix, or ("", "") if unset."""
        if not self.cloudtrail_archive:
            return "", ""
        rest = self.cloudtrail_archive.removeprefix("s3://").strip("/")
        bucket, _, prefix = rest.partition("/")
        return bucket, prefix

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_uri": self.store_uri,
            "arctic_bucket": self.arctic_bucket,
            "cloudtrail_archive": self.cloudtrail_archive,
            "stack_name": self.stack_name,
            "console_url": self.console_url,
            "strategy_dir": str(self.strategy_dir) if self.strategy_dir else None,
            "llm_cap_usd": self.llm_cap_usd,
            "llm_cap_usd_measured": self.llm_cap_usd_measured,
            "origins": dict(self.origins),
        }


def _resolve(explicit: str | None, variable: str, default: str | None) -> tuple[Any, str]:
    if explicit:
        return explicit, "argument"
    from_environment = os.environ.get(variable)
    if from_environment:
        return from_environment, f"environ:{variable}"
    return default, "default" if default is not None else "unresolved"


def settings(
    *,
    store_uri: str | None = None,
    arctic_bucket: str | None = None,
    strategy_dir: str | os.PathLike[str] | None = None,
    llm_cap_usd: float | None = None,
    cloudtrail_archive: str | None = None,
    stack_name: str | None = None,
    console_url: str | None = None,
    dry_run: bool = False,
) -> Settings:
    """Resolve configuration once, and record where each value came from."""
    origins: dict[str, str] = {}
    resolved_store, origins["store_uri"] = _resolve(store_uri, "CRUCIBLE_STORE", DEFAULT_STORE_URI)
    resolved_arctic, origins["arctic_bucket"] = _resolve(
        arctic_bucket, "CRUCIBLE_ARCTIC_BUCKET", DEFAULT_ARCTIC_BUCKET
    )
    resolved_cap, origins["llm_cap_usd"] = _resolve(
        None if llm_cap_usd is None else str(llm_cap_usd),
        "CRUCIBLE_LLM_CAP_USD",
        str(DEFAULT_LLM_CAP_USD),
    )
    resolved_archive, origins["cloudtrail_archive"] = _resolve(
        cloudtrail_archive, "CRUCIBLE_CLOUDTRAIL_ARCHIVE", DEFAULT_CLOUDTRAIL_ARCHIVE
    )
    resolved_stack, origins["stack_name"] = _resolve(
        stack_name, "CRUCIBLE_STACK", DEFAULT_STACK_NAME
    )
    resolved_console, origins["console_url"] = _resolve(
        console_url, "CRUCIBLE_CONSOLE_URL", DEFAULT_CONSOLE_URL
    )
    raw_dir = strategy_dir or os.environ.get("CRUCIBLE_STRATEGY_DIR")
    if raw_dir:
        origins["strategy_dir"] = "argument" if strategy_dir else "environ:CRUCIBLE_STRATEGY_DIR"
        resolved_dir: Path | None = Path(raw_dir).expanduser()
    else:
        origins["strategy_dir"] = f"store:{STRATEGY_PREFIX}"
        resolved_dir = None
    # `alpha-engine-config-I9823`: MEASURED tracks DEFAULT_LLM_CAP_USD_MEASURED
    # ONLY — a code-level fact nothing at runtime can flip. An operator
    # override (argument or CRUCIBLE_LLM_CAP_USD) is an assertion, not a
    # measurement, and asserting the identical $5.00 default through the env
    # var must not read back as "measured" — that was exactly the tamper
    # vector the prior `origins[...] != "default" or ...` shape left open.
    # The override's own provenance is still recorded, verbatim, in
    # origins["llm_cap_usd"].
    cap_measured = DEFAULT_LLM_CAP_USD_MEASURED
    return Settings(
        store_uri=resolved_store,
        arctic_bucket=resolved_arctic,
        strategy_dir=resolved_dir,
        cloudtrail_archive=resolved_archive,
        stack_name=resolved_stack,
        console_url=resolved_console.rstrip("/") if resolved_console else DEFAULT_CONSOLE_URL,
        llm_cap_usd=_positive_cap(resolved_cap, origins["llm_cap_usd"]),
        llm_cap_usd_measured=cap_measured,
        origins=origins,
        dry_run=dry_run,
    )


def _positive_cap(raw: str, origin: str) -> float:
    """The cap as a positive float, or a refusal naming where it came from.

    A malformed or non-positive cap RAISES rather than falling back to the
    default. A ceiling silently replaced by another number is a ceiling nobody
    is actually running under, and the environment variable is exactly where
    that typo happens.
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"the LLM cap {raw!r} (from {origin}) is not a number. It is USD per weekly "
            "run; a cap that cannot be parsed is not a cap."
        ) from exc
    if value <= 0:
        raise ValueError(
            f"the LLM cap {value} (from {origin}) must be positive. A run budget of zero "
            "is expressed by registering no call sites, not by a ceiling no call clears."
        )
    return value


def store_from_uri(uri: str) -> Store:
    """``s3://bucket/prefix`` to an :class:`S3Store`; anything else to a directory.

    A URI carrying a scheme this function does not know RAISES. Reading an
    unknown scheme as a relative directory would silently write a production
    run's artifacts into a folder named after the scheme, beside the source
    tree, and report success.
    """
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :].strip("/")
        if not rest:
            raise ValueError(f"{uri!r} names no bucket")
        bucket, _, prefix = rest.partition("/")
        return S3Store(bucket=bucket, prefix=prefix)
    if "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ValueError(
            f"unsupported store scheme {scheme!r} in {uri!r}. The supported backends "
            "are `s3://bucket/prefix` and a local directory path; an unknown scheme "
            "is a typo, and reading it as a directory name would write a production "
            "run into a folder named after the scheme and report success."
        )
    return LocalStore(Path(uri).expanduser())
