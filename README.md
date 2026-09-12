# crucible

[![CI](https://github.com/nousergon/crucible/actions/workflows/ci.yml/badge.svg)](https://github.com/nousergon/crucible/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnousergon%2Fcrucible%2Fbadges%2Fcoverage.json)](https://github.com/nousergon/crucible/actions/workflows/coverage-badge.yml)

An experiment harness for systematic strategies. It runs champion/challenger
experiments on a trading-day axis, writes one machine-readable manifest per
run, and grades every arm every cycle against a control it can fail.

Crucible is the *harness*, not the trader. The trader reads one contract —
`champions/{slot}/current.json` and `predictions/{trading_day}.json` — and
lives in its own repository. The harness must complete every acceptance test
with the trader switched off, and the trader must run a week on a frozen
champion with the harness switched off. That mutual independence is what
makes the harness the durable product.

## The one-command promise

Every job is one command, with identical behaviour on a laptop, in a Lambda
and on a spot instance. The scheduler does not know anything the CLI does not.

```
crucible experiment.run --slot r --arm <name-or-arm-id> --date 2026-09-01
```

Exit 0 means a verdict exists. There is no third outcome: a run either writes
a complete manifest with `status: ok`, or it is `failed` with a mandatory
`reason` and it pages. `partial`, `skipped`, `degraded` and `unknown` are not
representable — the manifest schema forbids them.

## Jobs

| Job | What it does |
|---|---|
| `data.daily` | Compile one trading day of market and fundamental data |
| `data.weekly` | Weekly refresh and coverage pass |
| `data.heal` | Repair a named gap in the store, idempotently |
| `experiment.new` | Register an immutable arm from a recipe spec |
| `experiment.run` | Score one arm for one trading day |
| `experiment.grade` | Run one slot's arena cycle: ladder, pairings, pointer |
| `promote` | Move a slot's champion pointer, evidence-gated |
| `report` | Reduce the week's manifests into the attribution table |
| `explain` | Walk a run_id or verdict back to everything that produced it |
| `migrate.history` | Import v1 arm history, flagged with its provenance |
| `release.pin` | Repoint a release, or pin the trader to one |
| `smoke` | A real end-to-end run that gates a release flip |

## Two rules that are not negotiable

**Every key is a trading day.** Artifact keys, manifest fields, ladder rungs,
windows and horizons are NYSE trading days from `krepis.trading_calendar`. A
run launched on a Saturday binds to Friday's close. `calendar_date` is
recorded beside it for provenance and is never used as a key. Horizons are
`21 / 63 / 126 / 252` trading days — never "1 month". A contract test walks
the store and fails on any non-trading-day key.

**Every run writes a manifest, including the ones that die.** The runner
writes `runs/{job}/{trading_day}/run.json` in a `try/finally`, so an
exception produces `status: failed` with its reason, its spend and its
lineage before the exception is re-raised. A run that produced no manifest is
an absence, and absence is one of the two conditions that page.

## Runbook

Six verbs. Each is one command; none needs a console. Any job that must run
in-region (`heal`, a replay week) is dispatched to a box rather than run from
a laptop. The dispatcher's function name is deployment configuration, not part
of the framework, so it is read from the environment rather than published
here (`alpha-engine-config-I10156`):

```
aws lambda invoke --function-name "$CRUCIBLE_DISPATCHER_FUNCTION" --payload '{"job": "<job>", "args": "<cli args>"}' out.json
```

### add an experiment

An arm is one recipe file in the strategy tree, registered before it can
score anything:

```
crucible experiment.new --slot r --arm <name> --run-mode live
```

Its id is the hash of its spec, so an edited recipe is a NEW arm carrying
`supersedes` and cannot inherit the old one's score series. The recipe
schema, the ranker registry, how to supersede or retire an arm, and every
refusal the loader raises are in
[docs/NEW_EXPERIMENT.md](docs/NEW_EXPERIMENT.md).

### rerun

Re-execute one stage for one trading day. Any job takes `--run-mode` (`live`
or `replay`; no default — an invocation naming neither is refused) and
`--date`; a non-trading day is refused rather than silently resolved.
Grading one slot's arena cycle for a past day:

```
crucible experiment.grade --slot r --date 2026-08-07 --run-mode replay
```

Idempotent by construction: outputs are keyed by trading day, so a rerun
that produces identical bytes overwrites in place rather than duplicating.
`experiment.grade` additionally refuses a cycle with nothing settled inside
its 21-session horizon.

### replay

Re-execute a past trading day (or a past week, via `crucible weekly`)
against pinned code with `--run-mode replay`:

```
crucible weekly --date 2026-08-07 --run-mode replay
```

**Replayed weeks must run sequentially, oldest first, never in parallel or
out of order.** The register, series and champion pointers are whole-object
rewrites keyed by slot, not append-only logs — a later week's replay
depends on the state a prior week's replay left behind, and running two out
of order (or concurrently) races those rewrites. Measured 2026-09-05/06:
five replay Saturdays run in series on the v2 box path.

### roll back

`crucible release.pin <prior-sha>` repoints `releases/current` (or, with
`--target trader`, the trader's pointer) to a prior release. No rebuild, no
redeploy — every release stays in S3.

To revert a slot's champion pointer directly, by operator authority rather
than by re-running the arena cycle:

```
crucible promote --slot r --revert-to <arm-id> --reason "<why>"
```

Recorded as `promotion_source=operator_bootstrap`, never as evidence;
`--reason` is mandatory — an unexplained override is the one pointer
movement nobody can reconstruct later.

### heal

`crucible data.heal --gap <name> --from YYYY-MM-DD --to YYYY-MM-DD` repairs
every NYSE session in `[--from, --to]`, idempotently: each session is
recompiled and reports `repaired` or `already_present`, and a session
already correct is rewritten with identical bytes. Both bounds must be
trading days — a Sunday, for example, raises `NonTradingDayKeyError` rather
than silently snapping to the nearest session.

```
crucible data.heal --from 2026-06-22 --to 2026-07-31 --gap <name> --i-am-in-region
```

**In region, or it refuses.** The job reads EC2 instance metadata; off EC2 it
allows at most 3 sessions as a diagnostic and otherwise refuses, naming the
in-region command to run instead. `--i-am-in-region` is the only override —
no environment variable, no config key. Measured 2026-09-05/06: 54 sessions
healed on a box via this path.

### unseal

Reserved: a human ruling, never an automated action (plan §9.4), and the
unseal is recorded permanently against the holdout the arms then see.

There is **no unsealing job and no automated path to the holdout's contents**.
The sealed document at `strategy/current/holdout.json` is read through
`crucible.holdout.read_sealed_holdout`, which returns the seal — the digest,
the day it was sealed, who sealed it, what it is reserved for — and drops the
payload entirely, so nothing that grades, renders or logs a reading can leak
the reservation.

Releasing the payload is an operator action that **refuses without a ruling
reference** (`alpha-engine-config-I<N>`) and a stated reason: the refusal is a
usage error, it writes nothing, and there is no proceed-anyway form of it. A
release that is ruled writes a `holdout_unseal.v1` audit record naming the
ruling, the operator, the reason and the exact digest released, and the
phase-3 `sealed_holdout` gate clause reads both the seal and every filed
record. Run the harness's own `--help` for the flag surface rather than
copying a command out of this section — unsealing is not a procedure anyone
should be pasting from a runbook under pressure.

Do not build a scheduled or automated caller for it; this section documents a
reservation, not an operations step.

## Development

```
uv sync --frozen
uv run pytest -q
uv run ruff check
```

`tests/acceptance/` holds the plan's acceptance tests. **They fail today, on
purpose.** They are the definition of done for phases 1 and 2, written before
the code that satisfies them. They are not skipped, not xfailed and not
marked — this repository carries no suppression collections at all, and
`tests/test_no_suppressions.py` enforces that.

## Documentation

- [docs/NEW_EXPERIMENT.md](docs/NEW_EXPERIMENT.md) — the runbook for adding, superseding and retiring an arm
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to propose a change, run the suite, what review to expect
- [SECURITY.md](SECURITY.md) — how to report a vulnerability
- `crucible/components.yaml` — the observability registry: every job's signals, log location, alert channel and deadline
- `tests/acceptance/README.md` — the phase-1/2 acceptance clauses, and what "done" means for this harness

Design rationale, the rebuild plan and strategy configuration are maintained
in the operating org's private repositories; this repository is the harness
itself.

## Licence

AGPL-3.0-or-later. See [LICENSE](LICENSE).
