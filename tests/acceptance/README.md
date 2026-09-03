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
**2026-09-02T01:31:16Z**, the moment its build PRs merged, with its exit gate
never measured. Four readings had circulated beforehand — from three surfaces,
over two totals, and not all of them measurements:

| Reading | Where it came from |
|---|---|
| 9 of 23 clauses failing (14 met) | `I9757` comment, 2026-09-01T20:23Z, on `6eb52f0` |
| 18 met / 6 unmet | same issue — a **prediction** of what two unmerged PRs would produce |
| 19 met / 4 unmet | same issue — measured on a **local** merge of three PRs, not on `main` |
| "21 of 23" | carried in conversation; **no artifact** |

Measured 2026-09-02 on `main`, the disagreement resolved into **two
instruments**: this suite read **21 of 24**, while the phase-1 gate read
**NOT MET, 1 of 5 clauses** — `pointer_flipped_on_smoke` only — against the
store `s3://alpha-engine-crucible-v2/crucible`
(`gates/phase1/2026-09-01/gate.json`, `met_ratio: 0.2`). Zero replay Saturdays
had run. The issue was reopened.

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

## Three outcomes, not two

A clause has one of three outcomes, never a fourth: **MET**, **UNMET** (the
clause read fine and the property does not hold), and **UNMEASURABLE** (the
read itself failed — no credentials, no region, `AccessDenied`, an
unreachable endpoint). Only a handful of clauses read live infrastructure at
all, and only those can ever be unmeasurable; every other clause is MET or
UNMET, same as before.

**UNMEASURABLE is never a pass** (principle 7 — "no data is never rendered
as green"), and it fails the run exactly like UNMET does. It is reported
separately because it is not a statement about the system: rendering it
identically to UNMET made a permanently-uncredentialed CI job
indistinguishable from a real, closeable plan-clause gap, and inflated the
gate's own denominator with something no amount of application code could
ever fix (alpha-engine-config-I9828).

A clause raises `_unmeasurable(...)` (defined beside `_unmet` in
`test_plan_section_2_objectives.py`) rather than `_unmet(...)` when the
failure is in the READ, not the property — structurally, that means the
`try` block around the live call, never the `assert` that follows it once
the read succeeded. `_unmeasurable` tags the JUnit failure message
`UNMEASURABLE — `, which `check_reading.py` (`UNMEASURABLE_MARKER`) reads to
sort the clause into its own bucket.

`ratchet.json`'s `unmeasurable` map is a **subset of `unmet`'s keys**, not a
fourth top-level bucket — `crucible/gate.py`'s phase-0 clause reads this same
file and requires `met | unmet` to equal every clause the suite defines
(that two-bucket contract predates this issue and is owned by a different
track, alpha-engine-config-I9869), so an unmeasurable clause id is listed in
both `unmet` (for that coarse view) and `unmeasurable` (naming why the read
failed, for the finer one). `push: [main]`'s step summary reports
`N met / M unmet / K unmeasurable`, with `M` counting only the plain-unmet
remainder (`unmet` minus `unmeasurable`); a clause moving between met,
plain-unmet and unmeasurable without the ratchet moving with it fails the
run.

**Currently unmeasurable, and why:**

| Clause | Why | Tracked |
|---|---|---|
| `TestCost::test_every_v2_resource_is_tagged_for_cost_attribution` | Reads live AWS (`cloudformation:ListStackResources`, `resourcegroupstaggingapi`, `iam:ListRoleTags`) to audit the `system=crucible-v2` tag. The `acceptance` CI job carries no AWS credentials at all (`permissions: {contents: read}`, no OIDC); `ne-laptop-agent` lacks the three read actions on the `crucible-v2` stack (`AccessDenied`). Neither environment can read it. | alpha-engine-config-I9895 |

Every other clause in this directory either performs no live-infrastructure
read (it constructs its own fakes, as `TestAutonomy`'s CloudTrail clause
does) or is not yet built (`NotImplementedError` via `_unmet`/`_attempt`),
and neither of those is unmeasurable — a clause that has not been written
yet is UNMET, not UNMEASURABLE: the code, not the caller's credentials, is
what is missing.

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
