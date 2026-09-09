r"""`outputs[].sha256` is the CONTENT digest, on every store backend.

Normative source: `crucible/AGENTS.md` rule 1 (manifest or it did not happen)
and `run_manifest.v2`, whose `sha256` is `^[0-9a-f]{64}$`.

## The defect this file exists to make impossible

`RunContext.record_output_cas` recorded the return value of
`Store.compare_and_swap` in the `sha256` field. That method's own docstring
says it returns "the new version token", and the two backends answer it
differently:

| backend | `put_bytes` returns | `compare_and_swap` returns |
|---|---|---|
| `LocalStore` | `sha256_hex(payload)` | `sha256_hex(payload)` |
| `S3Store` | `sha256_hex(payload)` | the object's **ETag** |

So on the laptop backend the version token IS the content digest, by
coincidence of that backend's design, and every local test agreed with the
wrong answer. On S3 the ETag is an md5 for a single-part write and
`<md5>-<parts>` for a multipart one -- neither is a sha256, by contract, ever.

`run_job` validates the manifest in its `finally`, so the failure landed
AFTER the work was done and the object written. Measured 2026-09-09: the
`Gate close` run at 01:03:47Z filed BOTH `gates/phase0/closing.json` and
`gates/phase1/closing.json` by compare-and-swap and then died on

```
ManifestValidationError: run manifest does not conform to run_manifest.v2:
  - outputs/0/sha256: String should match pattern '^[0-9a-f]{64}$'
  - outputs/1/sha256: String should match pattern '^[0-9a-f]{64}$'
```

never reaching the step that posts the closing reading to the phase tracker
issues. **The phase-exit loop failed on the first day it ever succeeded**, and
`alpha-engine-config-I9756` and `-I9757` stayed open, with no comment saying
why, over gates reading 5/5 and 6/6.

CAS is the rare write -- a champion pointer, a release pointer, a phase
closing record -- so this sat latent from the day it was written until the
first phase actually exited.

## Why the assertions are shaped this way

The S3 path is exercised through the `fake_s3` double, whose ETag is an md5
hex, exactly as S3's is. A double that returned a sha256 would reproduce the
`LocalStore` coincidence and this file would pass over the live defect, so
the md5 is load-bearing and is asserted to be one.

The check is the SCHEMA's, not a hand-written regex: `outputs[].sha256` is
whatever `run_manifest.v2` says it is, and a second spelling of the pattern
here could drift from it.
"""

from __future__ import annotations

import datetime as dt
import re

from crucible.runner import run_job
from crucible.store import ETAG_ABSENT, LocalStore, S3Store, sha256_hex

FRIDAY = dt.date(2026, 9, 4)
PAYLOAD = b'{"phase": "phase0", "gate_state": "MET"}\n'
KEY = "gates/phase0/closing.json"


def _manifest(store, job: str = "gate.close") -> dict:
    import json

    key = next(k for k in store.list_keys() if k.startswith(f"runs/{job}/") and k.endswith(".json"))
    return json.loads(store.get_bytes(key))


class TestTheDoubleIsNotTheCoincidence:
    def test_the_fake_s3_etag_is_not_a_sha256(self, fake_s3) -> None:
        """If this ever becomes a 64-hex value the rest of this file is
        vacuous: it would be re-testing `LocalStore`'s coincidence twice."""
        store = S3Store("bucket", client=fake_s3)
        store.compare_and_swap(KEY, ETAG_ABSENT, PAYLOAD)
        etag = store.etag(KEY)
        assert not re.fullmatch(r"[0-9a-f]{64}", etag), (
            f"the fake's ETag {etag!r} is a sha256, so a CAS write recording the version "
            "token in `sha256` would validate here and fail in production — which is "
            "exactly the blindness this file exists to remove"
        )
        assert etag != sha256_hex(PAYLOAD)


class TestCasOutputRecordsTheContentDigest:
    def test_on_s3_the_manifest_validates(self, fake_s3) -> None:
        """The production failure, reproduced end to end. Fails without the
        fix with `outputs/0/sha256: String should match pattern`."""
        store = S3Store("bucket", client=fake_s3)
        run_job(
            "gate.close",
            lambda c: c.record_output_cas(KEY, ETAG_ABSENT, PAYLOAD, schema_version="v1"),
            store=store,
            trading_day=FRIDAY,
            run_mode="live",
        )
        outputs = _manifest(store)["outputs"]
        assert [o["sha256"] for o in outputs] == [sha256_hex(PAYLOAD)]

    def test_the_version_token_is_still_returned_to_the_caller(self, fake_s3) -> None:
        """The token is what a caller passes as the next write's `expected`,
        so recording the digest must not change what comes back."""
        store = S3Store("bucket", client=fake_s3)
        returned: list[str] = []
        run_job(
            "gate.close",
            lambda c: returned.append(c.record_output_cas(KEY, ETAG_ABSENT, PAYLOAD)),
            store=store,
            trading_day=FRIDAY,
            run_mode="live",
        )
        assert returned == [store.etag(KEY)]
        assert returned[0] != sha256_hex(PAYLOAD)

    def test_on_the_local_backend_too(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_job(
            "gate.close",
            lambda c: c.record_output_cas(KEY, ETAG_ABSENT, PAYLOAD),
            store=store,
            trading_day=FRIDAY,
            run_mode="live",
        )
        assert [o["sha256"] for o in _manifest(store)["outputs"]] == [sha256_hex(PAYLOAD)]


class TestPlainOutputRecordsTheContentDigest:
    """`record_output` reads the same field and must not depend on what a
    backend's `put_bytes` happens to return either."""

    def test_on_s3(self, fake_s3) -> None:
        store = S3Store("bucket", client=fake_s3)
        run_job(
            "gate.close",
            lambda c: c.record_output("gates/ladder.json", PAYLOAD),
            store=store,
            trading_day=FRIDAY,
            run_mode="live",
        )
        assert [o["sha256"] for o in _manifest(store)["outputs"]] == [sha256_hex(PAYLOAD)]
