"""Where the store, the strategy tree and the price source come from.

Normative source: plan §4.11 ("`alpha-engine-config/strategy/` is config, not
code"), §4.12, and principle 8.

**No literal bucket, prefix or path appears anywhere else in the package.**
The CloudFormation template that creates the v2 bucket lands with track C;
until it does, the bucket is an assumption recorded on
`alpha-engine-config-I9757` (`alpha-engine-research`, prefix `crucible/`) and
it lives HERE, in one declared adapter, so the track-C change is a one-line
edit rather than a grep across the tree.

Three things are resolved, and each has exactly one resolution order —
explicit argument, then environment variable, then the declared default:

* the **store** (`--store` / ``CRUCIBLE_STORE``), an ``s3://bucket/prefix``
  URI or a directory path;
* the **strategy tree** (``--strategy-dir`` / ``CRUCIBLE_STRATEGY_DIR``), a
  checkout of ``alpha-engine-config/strategy/`` when running from a laptop;
  absent, arms are read from the store under ``strategy/current/``, which is
  what a spot instance sees;
* the **ArcticDB bucket** (``CRUCIBLE_ARCTIC_BUCKET``), the production price
  source's backing bucket.

A resolution that fell through to a default says so in
:attr:`Settings.origins`, so `explain` can report *why* a run read what it
read (principle 1) instead of only what it read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.llm import DEFAULT_LLM_CAP_USD
from crucible.store import LocalStore, S3Store, Store

__all__ = [
    "DEFAULT_ARCTIC_BUCKET",
    "DEFAULT_LLM_CAP_USD",
    "DEFAULT_STORE_URI",
    "STRATEGY_PREFIX",
    "Settings",
    "settings",
    "store_from_uri",
]

#: Recorded as an ASSUMPTION on alpha-engine-config-I9757, not as a fact: the
#: v2 bucket arrives with track C's CloudFormation template. Until then v2
#: artifacts live under an existing research bucket at a v2-only prefix, so
#: nothing v1 writes and nothing v1 reads can collide with them.
DEFAULT_STORE_URI = "s3://alpha-engine-research/crucible"

#: The ArcticDB store is a v1 asset v2 READS and never writes to from a
#: laptop (the standing in-region rule). Named here so the read path carries
#: no literal either.
DEFAULT_ARCTIC_BUCKET = "alpha-engine-data"

#: The strategy tree's home inside the store. `current` is a pointer prefix,
#: not a mutable directory: a strategy change is a sync of a new tree, and
#: the run manifest records that tree's hash as an input.
STRATEGY_PREFIX = "strategy/current"


@dataclass(frozen=True)
class Settings:
    """One resolved configuration, with the provenance of every value."""

    store_uri: str
    arctic_bucket: str
    strategy_dir: Path | None
    origins: dict[str, str] = field(default_factory=dict)
    #: The per-weekly-run LLM spend ceiling, in USD (plan §2 row 3). Declared
    #: HERE, in config, rather than at a call site: a ceiling that lives beside
    #: the code it bounds is one that moves whenever that code is edited.
    #: `crucible.llm.SpendCap` refuses the call that would cross it, before the
    #: provider is reached, and the run fails rather than overspending.
    llm_cap_usd: float = DEFAULT_LLM_CAP_USD

    def store(self) -> Store:
        return store_from_uri(self.store_uri)

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_uri": self.store_uri,
            "arctic_bucket": self.arctic_bucket,
            "strategy_dir": str(self.strategy_dir) if self.strategy_dir else None,
            "llm_cap_usd": self.llm_cap_usd,
            "origins": dict(self.origins),
        }


def _resolve(explicit: str | None, variable: str, default: str) -> tuple[str, str]:
    if explicit:
        return explicit, "argument"
    from_environment = os.environ.get(variable)
    if from_environment:
        return from_environment, f"environ:{variable}"
    return default, "default"


def settings(
    *,
    store_uri: str | None = None,
    arctic_bucket: str | None = None,
    strategy_dir: str | os.PathLike[str] | None = None,
    llm_cap_usd: float | None = None,
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
    raw_dir = strategy_dir or os.environ.get("CRUCIBLE_STRATEGY_DIR")
    if raw_dir:
        origins["strategy_dir"] = "argument" if strategy_dir else "environ:CRUCIBLE_STRATEGY_DIR"
        resolved_dir: Path | None = Path(raw_dir).expanduser()
    else:
        origins["strategy_dir"] = f"store:{STRATEGY_PREFIX}"
        resolved_dir = None
    return Settings(
        store_uri=resolved_store,
        arctic_bucket=resolved_arctic,
        strategy_dir=resolved_dir,
        llm_cap_usd=_positive_cap(resolved_cap, origins["llm_cap_usd"]),
        origins=origins,
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
