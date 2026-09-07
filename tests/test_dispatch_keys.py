"""`crucible.keys.dispatch_key` / `dispatch_prefix` / `parse_dispatch_key`.

alpha-engine-config-I10134 deliverable 1: the store key shape a dispatch
record is written under and read back through. Same round-trip and
prefix-contract discipline `test_keys_prefixes.py` holds every other `*_key`
/ `*_prefix` pair to.
"""

from __future__ import annotations

import pytest

from crucible.keys import (
    DISPATCH_ROOT,
    dispatch_key,
    dispatch_prefix,
    is_manifest_key,
    parse_dispatch_key,
    parse_manifest_key,
)


def test_dispatch_key_shape() -> None:
    assert dispatch_key("data.heal", "01jg000000000000000000abcd") == (
        "runs/_dispatch/data.heal/01jg000000000000000000abcd.json"
    )


def test_every_dispatch_key_starts_with_its_prefix() -> None:
    for job, dispatch_id in [("data.heal", "abc"), ("experiment.run", "def-123")]:
        assert dispatch_key(job, dispatch_id).startswith(dispatch_prefix(job))


def test_dispatch_prefix_starts_with_the_root() -> None:
    assert dispatch_prefix("data.heal").startswith(DISPATCH_ROOT)


def test_parse_dispatch_key_is_the_inverse() -> None:
    key = dispatch_key("data.heal", "abc123")
    assert parse_dispatch_key(key) == ("data.heal", "abc123")


@pytest.mark.parametrize(
    "key",
    [
        "runs/data.heal/2026-08-28/run.json",  # a real manifest, not a dispatch record
        "runs/_dispatch/data.heal/abc.txt",  # wrong suffix
        "runs/_dispatch/abc.json",  # missing the job segment
        "alerts/2026-08-28/absence.foo.json",  # a different namespace entirely
    ],
)
def test_parse_dispatch_key_returns_none_for_anything_else(key: str) -> None:
    assert parse_dispatch_key(key) is None


def test_a_dispatch_record_is_never_mistaken_for_a_manifest() -> None:
    """The two-namespace separation (DISPATCH_ROOT is a SIBLING of
    RUNS_ROOT's job segments, not nested under any job's own prefix) is what
    keeps a manifest-prefix listing (`crucible.documents.read_manifests_under`,
    every consumer that filters with `is_manifest_key`) from ever counting a
    dispatch record as a manifest, or vice versa."""
    key = dispatch_key("data.heal", "abc123")
    assert not is_manifest_key(key)
    assert parse_manifest_key(key) is None


def test_a_blank_job_is_refused_rather_than_listing_every_jobs_dispatches() -> None:
    with pytest.raises(ValueError, match="job must be non-empty"):
        dispatch_prefix("")


def test_a_hostile_dispatch_id_is_refused() -> None:
    with pytest.raises(ValueError, match="path segment"):
        dispatch_key("data.heal", "not/a/segment")
