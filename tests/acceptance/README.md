# Acceptance tests — these fail today, on purpose

Every test in this directory is a clause of the plan's §2 objective table, or
of a §6 phase gate, written as an executable assertion **before** the code
that satisfies it. Plan §6 phase 0's own exit gate is "v2 acceptance tests
(§2) written as failing pytest".

**They are not skipped, not xfailed, not marked.** This repository carries no
suppression collection at all (plan §11.1), and `tests/test_no_suppressions.py`
enforces that over the whole tree. An `xfail` here would be the exact defect
§11 risk 1 names: a system that passes its gates because its tests were
written to pass.

**Each test performs the real check.** Where the implementation does not exist
yet it raises `NotImplementedError`, and the test converts that into a failure
naming the clause and the owning track. When a track lands the code, the test
goes green on its own — nothing has to be un-marked, and nobody has to
remember that a gate was waiting.

## How to read the count

`uv run pytest -q tests/acceptance` prints the phase gate:

```
N failed, M passed  →  M of N+M plan clauses currently satisfied
```

That number is the gate. It is expected to be `0 passed` at the end of the
foundation commit and to climb as tracks A, B and C land. A run of this
directory that reports **no tests at all** is not a pass — it is an
unobserved gate, and CI treats it as a failure.

## Cross-track clauses belong here too

A clause one track can only satisfy with another track's code is the same
shape as a §2 objective: written first, failing honestly, going green when
the code lands. `test_feature_layer_binding.py` is the first —
track B's M slot consuming track A's feature layer (plan §10 component 4).

It is here rather than in the blocking job because a cross-track clause that
reds `main` reds it for the tracks that have not started the work, which is a
public signal the groom loop reads. The two alternatives — an `xfail`, or a
path exclusion in `ci.yml` — are both suppressions wearing different clothes.
This directory already runs in full and publishes its count, so nothing had
to be invented and nothing is hidden.

## What is deliberately not here

Anything measured against live AWS: the four-consecutive-Saturday window, the
CloudTrail human-mutating-call count and the monthly bill are read from
artifacts and from the CloudTrail S3 archive, not from a unit test. Their
clauses appear below as assertions over the artifacts those measurements
produce, so the gate reads from the same durable record an operator would.
