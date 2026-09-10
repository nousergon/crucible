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

## The twenty-four jobs, enumerated from `crucible/cli.py::JOBS`

**This count is itself a finding.** The rebuild plan and this repo's own
history cite "thirteen jobs" (`crucible_v2_rebuild_plan_260901.md` §4.1,
`README.md`); `crucible.cli.JOBS` carries **twenty-four** as of this tier's
introduction (2026-09-10) — eleven track-C/F/fault/morning/release-lock jobs
landed after the plan's own count was written and nothing updated the
citation. Enumerate live, never from memory or from the plan text:

```
uv run python -c "from crucible.cli import JOBS; print(len(JOBS)); [print(k) for k in JOBS]"
```

### Exercised for real (16 of 24)

`experiment.new`, `explain`, `release.pin`, `release.lock`, `smoke`,
`alerts.sweep`, `heartbeat`, `drift`, `console`, `board`, `gate`,
`gate.close`, `report`, `fault.record`, `fault.probe`, `migrate.history` —
each invoked through the real `crucible.cli.main` entry point (not the
handler function directly — this is the wire a spot instance actually
dispatches), against the dedicated store, producing a real `run.json`.

### Not exercised, and why (8 of 24)

**`data.daily`, `data.weekly`, `data.heal`** — `crucible.data.sources.
ArcticPriceSource` and the `nousergon_lib.arcticdb` helpers it calls
(`open_universe_lib`, `load_universe_ohlcv`) are hard-wired to the single
production `universe` library on `CRUCIBLE_ARCTIC_BUCKET`; there is no
library-selection parameter anywhere in that call path (verified against
installed `nousergon-lib==0.124.110`, 2026-09-10). This tier cannot point
these three jobs at a dedicated, isolated library without either widening
`nousergon_lib.arcticdb`'s API (a separate repo, out of this session's
scope) or adding a `library` override to `ArcticPriceSource` in this repo —
a real, correct, additive fix, but a production-source change this session
left unmade rather than rushed in unreviewed alongside a new test tier. The
real-ArcticDB dependency this tier's hard constraints require is instead
proven generically by `test_arctic_connectivity.py`: a write/read round trip
against the dedicated library, using the low-level `open_arctic(bucket).
get_library(name, create_if_missing=True)` path, which carries no
library-name constant.

**`experiment.run`, `experiment.grade`, `promote`, `weekly`** — each slot's
`produce`/`grade` scores arms against the feature layer `data.daily`/
`data.weekly` materialize; a non-degenerate exercise of these four is
downstream of the same ArcticDB gap and blocked by it. `weekly` runs the
declared weekly arc, which calls several of the above.

**`report.morning`** — `crucible.morning._operator_chat()` resolves
`krepis.alerts.DESTINATION_OPERATOR_CHAT` (a real Telegram channel) with no
override, and `crucible.morning._find_or_create_rolling_issue()` posts to a
real GitHub tracker repo with no override either. Running this job for real,
nightly, would deliver synthetic integration content to Brian's real
operator channel and the real tracker every night — that is a product
decision (a reserved matter, `principles.md` §3.2: "is this decision
reserved or delegated"), not one this tier makes unilaterally. Excluded
pending a ruling and, if approved, a dedicated-destination override in
`crucible/morning.py`.

Both gaps above are filed: `alpha-engine-config-I10457` (ArcticDB library
override, unblocks 7 of the 8 excluded jobs) and `alpha-engine-config-I10458`
(report.morning dedicated destination, needs Brian's ruling first).

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

## The summary artifact, and its DELIBERATE delta from `run.json`

**Not implemented as a registered CLI job.** Doing so correctly needs a new
entry in `crucible.cli.JOBS`/`HANDLERS`, a `components.yaml` row, and a new
member in `run_manifest.v2.json`'s closed `job` enum — three files this
session's ownership does not include and that the IaC/weekly-job-runner
track may also be touching concurrently. Filed as
`alpha-engine-config-I10459` (promote this to a real `crucible
integration.nightly` job once the enum and registry are free to edit) — the
correct SOTA shape, staged rather than rushed.

Until then, `.github/workflows/integration-nightly.yml` writes a plain JSON
summary directly to `integration/summary/{trading_day}.json` in the
**dedicated** store after the suite runs:

```json
{
  "schema_version": "integration_summary.v1",
  "trading_day": "YYYY-MM-DD",
  "generated_at": "2026-09-10T06:00:00Z",
  "commit": "<sha>",
  "outcome": "ok" | "failed",
  "jobs_exercised": ["experiment.new", "explain", ...],
  "jobs_not_exercised": {"data.daily": "<the reason above>", ...},
  "junit_summary": {"passed": N, "failed": M}
}
```

**This is why the gate's staleness refusal is not implemented in this PR.**
`crucible/gate.py` is owned by a sibling track this session; reading
`integration/summary/{trading_day}.json` and refusing a reading older than N
trading days is a small, well-specified addition once this key exists —
`alpha-engine-config-I10460` names the exact key, the `integration_
summary.v1` shape above, and the staleness window (recommend: the gate's own
window, same as every other clause) for whoever picks it up.

## Running it

Never on a laptop — see `crucible/AGENTS.md` and the hard constraint this
tier was built under: ArcticDB carries an explicit S3 Deny that blocks even
`ne-admin` (`alpha-engine-config-I9771`), so this tier runs where the jobs
themselves run: in-region, on `.github/workflows/integration-nightly.yml`'s
schedule or via its `workflow_call`/`workflow_dispatch` triggers.

```
uv run pytest -q tests/integration --tb=short
```
