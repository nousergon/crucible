# Contributing

## Setup

```
uv sync --frozen
uv run pytest -q
uv run ruff check && uv run ruff format --check
```

Python is pinned to 3.12 (`.python-version`). Dependencies are resolved from
the committed `uv.lock`; `--frozen` is not optional, and a change to
`pyproject.toml` must land with the regenerated lockfile in the same commit.

## What a change must satisfy

1. **A test was seen failing before the code that makes it pass.** Every test
   in this repository has that history.
2. **No suppression collections.** No `xfail`, no `pytest.skip`, no
   `_KNOWN_*` or `_GRANDFATHERED_*` list. A failing test is fixed or the
   feature is cut. `tests/test_no_suppressions.py` enforces this over the
   whole tree.
3. **Fail loud.** The default is `raise`. A bare `except: pass`, a silent
   `return None`, or a `continue` that hides a contract violation is a
   defect. Any deliberate swallow carries an inline comment naming the
   failure mode swallowed and the surface that records it.
4. **Trading days, not calendar days.** Any new key, window or horizon is in
   trading days. A calendar unit in a horizon fails schema validation.
5. **Manifest or it did not happen.** A new job writes a run manifest through
   `crucible.runner`, and gets a row in `crucible/components.yaml` declaring
   its signals, log location, alert channel, console surface and retention.

## Pull requests

One PR per repository, opened as a draft until it is merge-ready and green.
A PR must be deployable by the merge button alone — never embed a "run this
after merging" step in the body.

## What does not belong here

Tuned parameters, production feature recipes, exit and risk logic, prompt
templates, infrastructure identifiers and account numbers. This repository is
the harness; strategy content lives in the private config repository and is
loaded at runtime.
