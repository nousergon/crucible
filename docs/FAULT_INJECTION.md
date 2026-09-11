# Fault injection

Plan §10.7 names four scripted faults and requires each to be induced against
the **real scheduled path** — not a unit test, not a mock — and recorded.
`crucible gate --gate phase2` reads those records through
`fault_injection_against_scheduled_path`, and a fault with no conforming
record leaves the clause unmet no matter how many times the exercise was run.

All four are recorded as of 2026-09-11. This page is how to run one again,
what a valid record requires, and the traps that cost real exercises.

## The four faults, and what each one takes

| Fault | Outcome kind | How it is induced | Record |
|---|---|---|---|
| `spot_terminated_mid_job` | `absorbed` | AWS FIS, `aws:ec2:send-spot-instance-interruptions` | `faults/2026-08-11/spot_terminated_mid_job.json` |
| `data_source_withheld` | `induced` | the vendor seam refuses | `faults/2026-08-07/data_source_withheld.json` |
| `router_returns_500` | `induced` | the `chaos_probe` capability class | `faults/2026-09-09/router_returns_500.json` |
| `stale_release_pointer` | `unreachable` | not inducible; two closed paths are executed instead | `faults/2026-08-07/stale_release_pointer.json` |

The operator scripts live in `nous-ergon-ops/scripts/`:
`run_fault1_spot_interruption.sh`, `run_fault3_router_500.sh`, and
`dispatch_crucible_v2_job.sh` for anything else.

## The three outcome kinds are not three severities

`crucible.faults` refuses each one differently, and no kind is reachable by
relaxing another's requirement.

- **`induced`** — the fault fired and the job FAILED. Refused unless a manifest
  with that exact `run_id` exists under `runs/{target_job}/{trading_day}/` and
  reads `status: failed`, **and** a `bus_key` the store holds. A failed job that
  paged nobody is half of §10.7's exercise.
- **`absorbed`** — the fault fired and the system HANDLED it. Refused unless the
  manifest reads `status: ok` **and** its `attempts[]` carries a retry whose
  reason is in `crucible.runner.TRANSIENT_CLASSIFIERS`. `bus_key` is
  **forbidden**: a page here would mean the retry did not work. This is a
  STRONGER result than `induced`, which is why it is not reachable by dropping a
  requirement from it.
- **`unreachable`** — the state cannot be entered at all. Carries no `run_id`,
  so it is structurally incapable of excusing any manifest, and every probe in
  `UNREACHABLE_PROBES` is EXECUTED against the real store before the record is
  accepted. A fault with no registered probe is refused outright rather than
  accepted on an attestation somebody typed.

## Traps, each one paid for

**A bus key is named for the SIGNAL CLASS, never the job.** `fault.probe`'s own
failure produced `alerts/2026-09-09/failure.synthetic.router_unavailable.json`.
A search for `failure.fault.probe.json` matches nothing, ever — it waited out a
1800s deadline over a row written 40 seconds in and reported "no bus row" over a
correct induction. Find the row by the `run_id` it carries (`alert_id` for a
single-member incident, `members[].run_id` always), which is also the stronger
test: a name match would accept a row about a different run of the same job.

**A manifest key carries no discriminator.** `runs/{job}/{trading_day}/run.json`
is overwritten by a second run for the same job and day, so waiting for the key
to EXIST returns the previous run's result instantly. Capture the `run_id` at
the key *before* dispatching and require a different one; that is what
`nous-ergon-ops/scripts/lib/crucible_manifest_wait.sh` does.

**An experiment completing is not the job finishing.** FIS's
`durationBeforeInterruption` is `PT2M`: when the experiment reports complete,
the interruption NOTICE has been delivered and the retry has not yet run. A
manifest read at that moment is missing, and reporting it as "no manifest"
discards a successful induction.

**A true `absorbed` reads `resource.interruptions: 0`.** Nothing increments that
counter on the path the box actually takes — the IMDS watcher signals the
crucible process directly. Asserting `>= 1` refuses every correct run
(`alpha-engine-config-I10463`).

**Pick the trading day deliberately, in both directions.** A fault whose page
must exist has to land inside `crucible.alerts.CATCH_UP_TRADING_DAYS` or no bus
row is written and the record cannot be filed. A fault whose page must NOT count
has to land outside the next gate reading's `pages_within_ceiling` window. The
synthetic marker does not rescue the second case before
`SYNTHETIC_ROUTING_ACTIVE_FROM`.

**Preconditions surface in series.** Fault 3 took four attempts, and each one
looked like the last: registry path, loopback route, 4xx upstream, client
library. A capability with N preconditions and one probe reports N−1 false
"fixed"s if it is run once per fix. Enumerate what a path needs from the source
before re-running it, not from the last error message.

## Filing a record by hand

The scripts do this as their last step, and it is worth knowing separately
because a script can lose the record after a correct induction:

```
CRUCIBLE_STORE=s3://<bucket>/<prefix> crucible fault.record --fault router_returns_500 --outcome induced --target-job fault.probe --run-id <ULID> --bus-key alerts/<trading_day>/<condition>.<incident_id>.json --date <trading_day> --run-mode replay
```

Every refusal is a `ValueError` raised inside the job body, so a refused record
is itself a fully telemetered `status: failed` run of `fault.record` — never a
bare traceback, and never a hand-written file.

## Reading the clause back

```
crucible gate --gate phase2 --run-mode replay --dry-run
```

`fault_injection_against_scheduled_path` names all four faults, each with its
outcome, its manifest and (for `induced`) its bus row. Anything missing is named
by fault, so the line says which exercise to run rather than that the clause is
red.
