# Runbook — adding an experiment

An experiment is an **arm**: one recipe file, scored every cycle against
every other arm in its slot and against a control it can fail. This runbook
is the whole path from an idea to a scored arm, and the refusals you will
meet on the way.

Two facts decide the shape of everything below:

- **An arm is a file, and its id is the hash of its spec.** Editing a
  recipe does not change an arm — it creates a *new* one. The old arm keeps
  its score series, and the new one starts its own.
- **Registration is the gate.** An arm that writes output without a row in
  its slot's register is a defect, not an experiment. There is no path from
  a recipe to a score that skips `crucible experiment.new`.

## 0. Which slot

| Slot | What an arm decides | Recipe schema |
|---|---|---|
| `u` | Which names are in the universe | `ranker` + `params` |
| `r` | How the cross-section is ranked | `ranker` + `params` |
| `m` | What predicts the label | `spec.features`, `spec.estimator`, `spec.cpcv`, … |
| `s` | How a ranking becomes positions | `spec.rules`, `spec.cost_model` |

U and R recipes are `ArmSpec`s and are what `experiment.new` registers.
**M and S recipes are a different document**, read by
`crucible.slots.load_model_recipes` and `crucible.slots.load_strategy_recipes`;
`experiment.new --slot m|s` refuses by name, and says which loader reads the
slot. The rest of this runbook is the U/R path.

## 1. Make sure the ranker exists

`ranker` names a callable in `crucible.slots.rankers.RANKERS`, checked at
load time. An unknown name raises by name and lists the registered ones — a
recipe naming a ranker that does not exist would hash to the id of a recipe
nothing can run.

```
python -c "from crucible.slots.rankers import RANKERS; print(sorted(RANKERS))"
```

Adding a ranking rule is a **code change in this repository**: a
`RankerSpec` in `crucible/slots/rankers.py` declaring its `reads` (the
feature columns it consumes) and its `params` (the argument names a recipe
may set). Every column it reads must already exist in the feature catalog —
see [FEATURE_CATALOG.md](FEATURE_CATALOG.md). Tuned *values* never land
here; they live in the recipe.

**Two live arms may not share a ranking callable.** A comparison of a rule
with itself always reads as a tie and consumes a slot in a pool with a cap,
so the loader refuses the pair as `inapplicable` before either produces
anything. A new arm on an existing ranker is a new *parameterisation* of it,
which is a different arm id but the same callable — that is the case the
guard refuses. Differentiate on the rule, not on the numbers alone.

## 2. Write the recipe

One YAML file per arm, at `arms/{slot}/{name}.yaml` in the strategy tree —
the private directory `--strategy-dir` / `$CRUCIBLE_STRATEGY_DIR` points at,
published to the store under `strategy/current/arms/{slot}/`. The recipe is
where tuned values live; this repository holds only the registry they are
checked against.

```yaml
# arms/r/example_composite.yaml — illustrative values, not a filed arm.
name: example_composite
slot: r
ranker: quant_composite
registered_at: '2026-09-04'
params:
  top_n: 15
  momentum_weight: 0.5
  trend_weight: 0.3
  reversal_weight: 0.2
notes: >-
  What this arm believes, in one paragraph: which effect it is betting on and
  why these weights express it.
```

The top-level vocabulary is closed (`crucible.models.ArmRecipeDocument`,
`extra="forbid"`): an unrecognised key is refused, never accepted and
ignored.

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Unique within the slot. The file name should match it. |
| `slot` | yes | `u` or `r` for this schema. |
| `ranker` | yes | A key of `RANKERS`. |
| `params` | yes, non-empty | The ranker's own arguments. Hashed into the id. |
| `registered_at` | yes | **An NYSE session.** See below. |
| `supersedes` | no | The arm id this recipe replaces (§3). |
| `notes` | no | Provenance only, never hashed. |
| `promotion_source` | no | Provenance only, never hashed. |
| `bootstrap` | no | Whether the arm bootstraps its own promotion. |
| `control`, `control_kind` | no | Left false: the harness generates its own controls. |
| `schema_version` | no | Defaults to `arm_recipe.v1`. |

**`registered_at` is a trading day, and it is asserted.** It starts the
arm's out-of-sample clock, and every ladder rung, `promote_min_weeks` rung
and grace rung is counted from it in trading weeks. A Saturday is refused
rather than resolved to the Friday — three recipes once registered with a
Saturday date and the error would have compounded silently for the life of
each arm.

**`metric`, `horizon` and `benchmark` are the slot's and are not settable
per arm.** Every arm in a slot is scored on one axis or the comparison means
nothing.

**An LLM arm declares its call site.** `params.llm_callsite` must be a key
of `crucible.llm.LLM_CALLSITE_REGISTRY`; an unregistered one is refused, so
"which arms are LLM arms" is answerable from the register rather than from a
string heuristic over ranker names. Register the call site in the same
change as the code that calls it.

## 3. Changing an arm that already exists

Do not edit a filed recipe in place unless you intend a new arm — you will
get one. The hashed spec is `{slot, name, ranker, params, control,
control_kind}`; `notes`, `promotion_source` and the recipe's source path are
deliberately outside it, so clarifying a comment does **not** orphan a score
series.

To replace an arm, file a new recipe naming the old arm's id:

```yaml
name: example_composite_v2
slot: r
ranker: mom_12_1_sleeve   # a different callable — see §1
registered_at: '2026-09-04'
supersedes: r:example_composite:9f2c1b…
params: {...}
```

`supersedes` must name an id **already in the register**: a lineage pointer
to nothing is worse than none, because it reads as history that was checked.

To retire an arm, remove its file from the strategy tree. The publisher
deletes the object it no longer backs, so the arm stops registering;
retirement inside a running cycle is the arena's cap-with-grace rule, not an
operator action.

## 4. Check it before it is filed

The loaders that validate a recipe are the ones production uses — there is
no second copy of the rules to drift from them. Against a checkout:

```
uv run crucible experiment.new --slot r --arm example_composite --dry-run --run-mode live --strategy-dir <tree>
```

`--dry-run` resolves the same inputs a real run would, prints
`would_register` / `already_present`, and writes nothing — not the register,
not a manifest. The store is opened read-only for the whole invocation.

## 5. Register it

```
uv run crucible experiment.new --slot r --arm example_composite --run-mode live --strategy-dir <tree>
```

This appends to `arms/{slot}/register.jsonl` — an append-only event log,
rewritten as one object because S3 has no append — and prints `registered`
and `already_present`. An arm already registered is not registered twice.
`--arm` is required here: registering whichever recipes happen to be on disk
is not a deliberate act. It takes a bare name or a full arm id.

Without `--strategy-dir` (a spot box has no checkout) the recipes are read
from the tree synced into the store under `strategy/current/arms/{slot}/`.
Which source was used is recorded on each spec and is reported by
`crucible explain`, so "why did this box register these arms" is answerable
after the fact.

## 6. Score it, grade it, promote it

```
uv run crucible experiment.run   --slot r --date 2026-09-04 --run-mode live   # every arm; --arm narrows
uv run crucible experiment.grade --slot r --date 2026-09-04 --run-mode live   # the arena cycle
```

`experiment.run` scores every registered arm — an arm that skipped a cycle
records a miss, and a miss is data; `--arm` narrows the run to one arm and
is a debugging affordance, not the normal path. Both jobs run weekly on the
trading calendar. `experiment.grade` calls the arena engine — ladder, paired
windows, the confidence sequence, Condorcet ranking, the pointer decision
and cap-with-grace retirement all live in `nousergon_lib.arena` and are
never re-implemented per slot. It refuses a cycle with nothing settled
inside its horizon.

Promotion is evidence-gated and belongs to `crucible promote`. A new arm
cannot win before its out-of-sample clock has run; the controls are scored
every cycle and are excluded from both the pointer and the cap.

## 7. Refusals you will meet

| What you see | What it means | What to do |
|---|---|---|
| `unknown ranker '…'` | `ranker` names nothing in `RANKERS` | Fix the name, or add the `RankerSpec` (§1) |
| `arm recipe is missing required field(s) [...]` | §9.1 pre-registration | Add every field named; `params: {}` counts as missing |
| `NonTradingDayKeyError` on `registered_at` | The date is not an NYSE session | Use a real session — it is not resolved for you |
| `arms share a ranking callable …` | Two arms are not two arms | Differentiate on the rule, not only on parameters |
| `arms … derive the same id` | Two files, one recipe | Delete the copy |
| `supersedes=… is not in the register` | Lineage points at nothing | Name a registered id, or drop the field |
| `slot 'm' does not use ArmSpec recipes` | Wrong schema for the slot | M and S are read by their own loaders (§0) |
| `params.llm_callsite names …, which is not a key of LLM_CALLSITE_REGISTRY` | Unregistered call site | Register the site with the code that calls it |
| `no arm recipes found for slot '…'` | An empty slot produces zero comparisons | Point `--strategy-dir` at the tree, or file an arm |
| `no arm named '…' in slot '…'` | The selector matched nothing | Check the name; a typo would otherwise exit 0 |

Every one of these is a refusal at load, before anything is written. That is
deliberate: a recipe that registers and then quietly fails to score is
indistinguishable from an arm that ran and selected nothing.
