"""Closed-vocabulary key refusal, shared by every recipe loader.

`alpha-engine-config-I9944`: `crucible.slots.model.load_model_recipes` read
the keys it knew from a recipe's `spec:` and silently accepted-and-ignored
everything else. Two live consequences measured 2026-09-03: both M recipes
carried a hand-written `feature_version: v1` that nothing read
(`alpha-engine-config-I9801`), and an M or S recipe author writing
`llm_callsite:` under `spec:` — the natural place to put it — would register
an arm the phase-5 LLM-arm gate silently never counts
(`alpha-engine-config-I9920`, which refuses exactly this for `ArmSpec`
recipes; this module closes the same hole for M and S).

Every recipe loader in `crucible.slots` refuses an unrecognised key BY NAME,
at the same layer the loader already refuses a MISSING required key — a key
the loader does not read is a guarantee it cannot honour, not a harmless
extra.

Leaf module: no imports from `crucible.slots` or its siblings, so every
loader (`crucible.slots.arms`, `.model`, `.strategy`) can import it without
risking a cycle.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["refuse_unknown_keys"]

#: Keys named in the refusal message with their own sentence, because each
#: has a documented reason it is not a recipe field rather than a generic
#: typo. The tracker citation for each lives only in the comment beside it
#: here, never inside the string reaching a raised message —
#: `tests/test_no_stale_tracker_literals.py` refuses a hardcoded
#: `alpha-engine-config-I<N>` anywhere reachable from a raise, since a
#: historical issue number baked into runtime text cannot be derived and
#: goes stale exactly the way `crucible.gate.PHASES` exists to prevent for
#: phase issues.
_NAMED_REASONS: dict[str, str] = {
    # `alpha-engine-config-I9801`: a hand-written literal inside the hashed
    # M spec, hashed into the arm id and read by nothing.
    "feature_version": (
        "`feature_version` is deliberately absent from the M recipe: it was a "
        "hand-written literal inside the hashed spec that nothing read"
    ),
    # `alpha-engine-config-I9920`: declared on a U/R recipe's `params` only.
    "llm_callsite": (
        "`llm_callsite` is declared on a U/R recipe's `params`, never on an M or S "
        "recipe's `spec` — an M/S arm is not dispatched through the LLM call-site "
        "registry, and the phase-5 gate would silently never count it as an LLM arm"
    ),
}


def refuse_unknown_keys(
    *,
    path: Path | str,
    keys: set[str],
    vocabulary: frozenset[str],
    level: str,
    slot_label: str,
) -> None:
    """Raise naming every key in ``keys`` outside ``vocabulary``, or do nothing.

    ``level`` is the document region in the message ("spec", "top-level
    recipe"); ``slot_label`` names the recipe kind ("M", "S", "U/R").
    """
    unknown = sorted(set(keys) - set(vocabulary))
    if not unknown:
        return
    named = [_NAMED_REASONS[k] for k in unknown if k in _NAMED_REASONS]
    trailer = " " + "; ".join(named) + "." if named else ""
    raise ValueError(
        f"{path}: {level} carries unknown key(s) {unknown} — the {slot_label} recipe "
        f"declares exactly {sorted(vocabulary)}; a key the loader does not read is a "
        f"guarantee it cannot honour.{trailer}"
    )
