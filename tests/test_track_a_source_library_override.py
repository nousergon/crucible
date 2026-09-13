"""`crucible.track_a._source` threads `--arctic-library` to `ArcticPriceSource`.

Normative source: `alpha-engine-config-I10457`. Additive: production never
passes `--arctic-library`, so `getattr(args, "arctic_library", None)` is
`None` there and `ArcticPriceSource.library` stays unset — byte-for-byte the
production behaviour before this change. The integration tier is the only
caller that passes a value, resolved through `crucible.required.require_env`
in `tests/integration/conftest.py::integration_arctic_library` (already
RAISE-on-absent there).
"""

from __future__ import annotations

import argparse

from crucible.config import settings as resolve_settings
from crucible.track_a import _source


def test_no_arctic_library_flag_leaves_the_source_unoverridden() -> None:
    args = argparse.Namespace()
    config = resolve_settings(arctic_bucket="test-bucket")
    source = _source(args, config)
    assert source.library is None
    assert source.bucket == "test-bucket"


def test_arctic_library_flag_threads_through_to_the_source() -> None:
    args = argparse.Namespace(arctic_library="crucible-integration")
    config = resolve_settings(arctic_bucket="test-bucket")
    source = _source(args, config)
    assert source.library == "crucible-integration"


def test_an_empty_arctic_library_flag_is_treated_as_unset() -> None:
    """`argparse`'s own default is `None`, but a caller that explicitly
    passes an empty string must not silently open a library named `''`."""
    args = argparse.Namespace(arctic_library="")
    config = resolve_settings(arctic_bucket="test-bucket")
    source = _source(args, config)
    assert source.library is None
