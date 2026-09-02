# crucible

[![CI](https://github.com/nousergon/crucible/actions/workflows/ci.yml/badge.svg)](https://github.com/nousergon/crucible/actions/workflows/ci.yml)

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
crucible experiment.run --slot r --arm <arm-id> --date 2026-09-01
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

Five verbs. Each is one command; none needs a console.

### rerun
> **Stub.** Re-execute a job for a trading day whose manifest is `failed` or
> absent. Idempotent by construction: outputs are keyed by trading day and
> content hash, so a rerun that produces identical bytes is a no-op.

### replay
> **Stub.** Re-execute a past trading day against a pinned release and a
> pinned data snapshot, and diff the resulting manifest against the original.
> Deterministic arms must reproduce exactly; LLM arms are gated on live
> cycles only and are excluded from replay verdicts.

### roll back
> **Stub.** `crucible release.pin <prior-sha>` repoints `releases/current`.
> No rebuild, no redeploy. Every release stays in S3.

### heal
> **Stub.** `crucible data.heal` repairs a named gap — a missing trading day,
> a rejected-row class, a stale snapshot — and writes what it repaired into
> the manifest's `rows_in`/`rows_out`/`rows_rejected` fields.

### unseal
> **Stub.** Releasing a holdout period for evaluation. Reserved: a human
> ruling, never an automated action, and the unseal event is recorded
> permanently against the arms that then see the data.

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

- [CONTRIBUTING.md](CONTRIBUTING.md) — how to propose a change, run the suite, what review to expect
- [SECURITY.md](SECURITY.md) — how to report a vulnerability
- `crucible/components.yaml` — the observability registry: every job's signals, log location, alert channel and deadline
- `tests/acceptance/README.md` — the phase-1/2 acceptance clauses, and what "done" means for this harness

Design rationale, the rebuild plan and strategy configuration are maintained
in the operating org's private repositories; this repository is the harness
itself.

## Licence

AGPL-3.0-or-later. See [LICENSE](LICENSE).
