# Run manifest store layout, version 2

This contract changes storage layout, not the `run_manifest.v2` document
schema.

For a job, trading day, and optional discriminator, `manifest_key(...)`
continues to name the **latest alias**:

```text
runs/{job}/{trading_day}/run.json
runs/{job}/{trading_day}/{discriminator}/run.json
```

Each invocation is also stored at an immutable key:

```text
runs/{job}/{trading_day}/history/{run_id}/run.json
runs/{job}/{trading_day}/{discriminator}/history/{run_id}/run.json
```

The producer validates the final document, creates the immutable object with
create-if-absent semantics, and only then publishes the same bytes to the
latest alias with compare-and-swap. A duplicate immutable key is an error.
Latest-alias conflicts are retried a bounded number of times; exhaustion may
leave an immutable invocation that is not the alias, but never deletes or
rewrites it.

Ordinary prefix readers enumerate latest aliases only. This preserves legacy
counts and verdicts and prevents the history copy from becoming a second
observation. History-aware consumers explicitly opt in, prefer the immutable
copy when an alias names the same `run_id`, and index each invocation once.
Stores containing only the legacy latest keys remain readable. Mixed stores
may therefore be migrated incrementally without rewriting historical objects.
