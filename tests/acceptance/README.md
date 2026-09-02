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

## How to read the count — it is progress, not the gate

`uv run pytest -q tests/acceptance` prints how much of the plan is satisfied:

```
N failed, M passed  →  M of N+M plan clauses currently satisfied
```

**That number is progress. It is not a phase gate, and quoting it as one is a
known defect.** Phase 1 (`alpha-engine-config-I9757`) was closed
**2026-09-02T01:31Z**, the moment its build PRs merged, with its exit gate never
measured. Four different figures had been quoted for that gate — 15/9, 18/6,
19/4, "21 of 23" — none agreeing, all hand-typed into issue comments. Measured
2026-09-02 on `main` at `794a1b0`, the disagreement resolved into **two
instruments**: this suite read **21 of 24**, while the phase-1 gate read **NOT
MET, 1 of 5 clauses** (`pointer_flipped_on_smoke` only) against the real store
`s3://alpha-engine-crucible-v2/crucible`. Zero replay Saturdays had run. The
issue was reopened.

Note the store: the gate reads **1 of 5** against the production store and
**0 of 5** against an empty one. A gate reading without its store named is not
a reading — that is the same defect one level down.

**The gate is `crucible gate --gate <phase> --store <store>`**, which reads the
artifacts the phase actually produced and never runs the thing it grades
(`crucible/gate.py`). A phase exits on that command's output, pasted; nothing
here can substitute for it, because every clause in this directory is
satisfiable by code alone, and a gate that code alone can satisfy is not
measuring the system.

What this count is good for: it is the red board. It is expected to read
`0 passed` at the end of the foundation commit and to climb as tracks land,
and it is the only progress figure worth quoting in a report — not merged PRs,
not findings, not commits. A run of this directory that reports **no tests at
all** is not a pass; it is an unobserved board, and CI treats it as a failure.

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

Anything measured against live AWS: the live-gate window (2 live Saturdays plus
5 replayed, per plan §6.1 — not the 4 that §2 row 1 still names), the
CloudTrail human-mutating-call count and the monthly bill are read from
artifacts and from the CloudTrail S3 archive, not from a unit test. Their
clauses appear below as assertions over the artifacts those measurements
produce, so the gate reads from the same durable record an operator would.
