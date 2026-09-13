# Integration tier — real S3, real ArcticDB, nightly and pre-gate only

`alpha-engine-config-I10419`. The third test tier, sitting beside `tests/`
(unit, mocked, the `fail_under = 93` ratchet) and `tests/acceptance/` (plan
§2 clauses). Where those two run on every PR, this one **never does** — it
runs **nightly** and **before any phase gate**, against a dedicated S3 prefix
and a dedicated ArcticDB library, never the production ones.

| Tier | What | When |
|---|---|---|
| `tests/` | unit, mocked | every PR |
| `tests/acceptance/` | plan §2 clauses | every PR (collection) + `push: [main]` |
| `tests/integration/` | one case per CLI job, real S3 + real ArcticDB | nightly, pre-gate |

## Why not per-PR

The one real-dependency control before this tier was the deploy smoke
(`crucible smoke`, plan §4.11 "Verify, then flip") — a real run against live
S3/ArcticDB gating the `releases/current` flip. It covers **one job of
twenty-four**. Everything else real happens at a phase gate, by procedure,
and a procedure produces no artifact when nobody runs it. Per-PR was
rejected on purpose (`I10419`): these cases cost minutes and money against
real infrastructure, and a slow per-PR tier gets skipped in its first week.
`pyproject.toml`'s `addopts` carries `--ignore=tests/integration` for exactly
this reason — every existing `pytest` invocation in `ci.yml` and `deploy.yml`
already excludes this directory without either workflow file changing.

## The jobs, enumerated from `crucible/cli.py::JOBS`

**This count is itself a finding, and it moves.** The rebuild plan and this
repo's own history cite "thirteen jobs" (`crucible_v2_rebuild_plan_260901.md`
§4.1, `README.md`); `crucible.cli.JOBS` carried twenty-four as of this
tier's introduction (2026-09-10) and twenty-five as of `test.integration`'s
own registration (`alpha-engine-config-I10459`) — re-count from the code,
never from a number written down here. Enumerate live, never from memory or
from the plan text:

```
uv run python -c "from crucible.cli import JOBS; print(len(JOBS)); [print(k) for k in JOBS]"
```

### Exercised for real (24 of 25)

`experiment.new`, `data.daily`, `data.weekly`, `data.heal`, `experiment.run`,
`weekly`, `experiment.grade`, `promote`, `explain`, `release.pin`,
`release.lock`, `smoke`, `alerts.sweep`, `heartbeat`, `drift`, `console`,
`board`, `gate`, `gate.close`, `report`, `fault.record`, `fault.probe`,
`migrate.history`, `test.integration` — each invoked through the real
`crucible.cli.main` entry point (not the handler function directly — this is
the wire a spot instance actually dispatches), against the dedicated store,
producing a real `run.json`.

`data.daily`, `data.weekly`, `data.heal` and `experiment.run` are newly
exercised here (`alpha-engine-config-I10457`): `ArcticPriceSource(library=
...)` — additive, `library=None` (the default, used by every production
call site) is byte-for-byte unchanged — routes the three data jobs through
`open_arctic(bucket).get_library(name, create_if_missing=True)` +
`nousergon_lib.arcticdb._load_arctic_frames` instead of the production-
hard-wired `open_universe_lib`/`load_universe_ohlcv`, against real synthetic
OHLCV rows this tier seeds into `CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY` itself
(`conftest.py::integration_arctic_symbols`). `experiment.run` was never
blocked by ArcticDB at all — `crucible.slots.cycle.run_produce` reads the
feature layer `data.daily` writes into the STORE, never ArcticDB directly —
its non-degenerate exercise was simply downstream of `data.daily` having
nothing real to read.

`weekly`, `experiment.grade` and `promote` are newly exercised here
(`alpha-engine-config-I10633`), closing the two residual gaps I10457 left
open:

* `crucible.weekly.Stage.argv`/`run_arc` now accept an `arctic_library`
  parameter, appended as `--arctic-library <name>` onto every
  `ARCTIC_LIBRARY_JOBS` stage's own argv (`crucible/weekly.py`) — additive
  and production-inert, the identical shape `--dry-run` already used.
  `test_weekly` exercises the arc's own `data.weekly` stage through this
  threading for real, against the dedicated library; the arc's other stages
  (the R slot's own arm registration is a separate, sibling-owned concern,
  `alpha-engine-config-I10628`) are stubbed there and exercised standalone,
  for real, by this module's other cases.
* `conftest.py::SETTLED_TRADING_DAY` (`2026-10-07`) is a second FIXED
  literal, exactly `DEFAULT_HORIZON_TRADING_DAYS` (21) NYSE sessions after
  `INTEGRATION_TRADING_DAY` — seeded rather than hand-written (the issue's
  own alternative (b)) so the real `data.daily`/`experiment.run` producer
  path settles a shadow rather than fabricating a verdict shape by hand.
  `integration_arctic_symbols` now seeds one continuous synthetic OHLCV
  series through `SETTLED_TRADING_DAY`, and `test_experiment_grade` grades
  the shadow `test_experiment_run` produced at `INTEGRATION_TRADING_DAY`
  against a panel compiled AT `SETTLED_TRADING_DAY`, asserting the result
  carries a genuinely non-empty `settled_dates`. `test_promote` runs
  immediately after, against the same graded cycle.

`test.integration` is this tier's own job (`alpha-engine-config-I10459`) —
see "The summary artifact" below.

### Not exercised, and why (1 of 25)

**`report.morning`** — `crucible.morning._operator_chat()` resolves
`krepis.alerts.DESTINATION_OPERATOR_CHAT` (a real Telegram channel) with no
override, and `crucible.morning._find_or_create_rolling_issue()` posts to a
real GitHub tracker repo with no override either. Running this job for real,
nightly, would deliver synthetic integration content to Brian's real
operator channel and the real tracker every night — that is a product
decision (a reserved matter, `principles.md` §3.2: "is this decision
reserved or delegated"), not one this tier makes unilaterally. Excluded
pending a ruling and, if approved, a dedicated-destination override in
`crucible/morning.py`. Filed as `alpha-engine-config-I10458` (needs Brian's
ruling first).

## The dedicated environment, resolved from required env — never a literal

Mirrors `crucible.required.require_env`'s existing shape (`CRUCIBLE_MUTED_
TOPIC`, `CRUCIBLE_PAGES_TOPIC`, `CRUCIBLE_LIQUIDITY_FLOOR_USD`): every value
below RAISES when unset, naming the variable, rather than guessing a
plausible-looking default that could resolve to production.

| Variable | What | Guard |
|---|---|---|
| `CRUCIBLE_INTEGRATION_STORE_URI` | dedicated S3 prefix | `conftest.py` refuses a URI whose path does not carry an `integration` segment |
| `CRUCIBLE_INTEGRATION_ARCTIC_BUCKET` | the ArcticDB bucket (may be the production bucket — see below) | — |
| `CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY` | the dedicated library NAME within that bucket | `conftest.py` refuses a name matching any of `nousergon_lib.arcticdb`'s production library constants (`universe`, `macro`, `preliminary`) |
| `CRUCIBLE_INTEGRATION_PAGES_TOPIC` | the SNS topic `alerts.sweep`/`heartbeat` publish to for real | passed to the jobs as `CRUCIBLE_PAGES_TOPIC` — never the production topic |
| `CRUCIBLE_INTEGRATION_MUTED_TOPIC` | the muted-topic clause reads | passed as `CRUCIBLE_MUTED_TOPIC` |

`data.daily`/`data.weekly`/`data.heal` additionally pass
`--arctic-library "$CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY"`
(`alpha-engine-config-I10457`) — the CLI flag `ArcticPriceSource(library=
...)` threads through; there is deliberately no `--arctic-bucket` flag, only
`CRUCIBLE_ARCTIC_BUCKET` (set from `CRUCIBLE_INTEGRATION_ARCTIC_BUCKET` by
`conftest.py::_arctic_bucket_env`, session-scoped autouse), since the bucket
is not this tier's isolation unit — see below.

**Why the ArcticDB bucket is not itself "dedicated".** ArcticDB has no
per-bucket-per-tenant isolation cheaper than a second bucket; the *library*
is the isolation unit this fleet already uses (`open_preliminary_lib`'s own
docstring: "structurally separate from `UNIVERSE_LIB`... physical-library
split"). `CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY` is what keeps this tier's
writes out of `universe`/`macro`/`preliminary`; the bucket may be shared
because a library is the real boundary ArcticDB enforces.

## Trading day

A fixed literal (`INTEGRATION_TRADING_DAY` in `conftest.py`), never wall-clock
— consistent with `AGENTS.md`'s "Use fixed date literals, never `today`
arithmetic" (Test discipline). Verified a real trading day at collection
time via `crucible.calendar.is_trading_day`, not merely asserted in a
comment.

## The summary artifact — a real registered CLI job

**`test.integration`** (`alpha-engine-config-I10459`, `crucible.
integration_summary`) replaces the hand-rolled `store.put_bytes` write this
workflow used to carry: a thin handler that shells out to
`pytest tests/integration --tb=short` and reports pass/fail through
`crucible.runner.run_job`, writing a real, schema-validated
`run_manifest.v2` document at `runs/test.integration/{trading_day}/
run.json` in the dedicated store — like every other job (rule 1, manifest or
it did not happen). No schedule, no deadline: `components.yaml`'s row
declares both `null` deliberately, since this job stays workflow-triggered
by `.github/workflows/integration-nightly.yml`'s own `schedule`/
`workflow_call`/`workflow_dispatch` triggers rather than gaining a second,
independent starter for the same nightly run.

The pass/fail exit code and a tail of the process's combined stdout/stderr
land in the manifest's `metrics[]` (`tests_integration_exit_code`) — a
failure's `reason` carries the same tail, so a reader does not have to open
the workflow's own log to learn what broke.

**This is why the gate's staleness refusal is not implemented in this PR.**
`crucible/gate.py` is owned by a sibling track this session; reading
`runs/test.integration/{trading_day}/run.json` and refusing a reading older
than N trading days is a small, well-specified addition now that the key
exists — `alpha-engine-config-I10460` names the exact key and the staleness
window (recommend: the gate's own window, same as every other clause) for
whoever picks it up.

## Running it

Never on a laptop — see `crucible/AGENTS.md` and the hard constraint this
tier was built under: ArcticDB carries an explicit S3 Deny that blocks even
`ne-admin` (`alpha-engine-config-I9771`), so this tier runs where the jobs
themselves run: in-region, on `.github/workflows/integration-nightly.yml`'s
schedule or via its `workflow_call`/`workflow_dispatch` triggers.

```
uv run pytest -q tests/integration --tb=short
```
