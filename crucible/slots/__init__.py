"""The slot registry: four slots, one engine.

Normative source: plan §4.4; binding source `champion-challenger-policy.md`.

| Slot | Decision | Champion feeds |
|---|---|---|
| **U** | which predicted names the book may hold | `universe/{trading_day}/members.json` |
| **R** | how names are scored into signals | `signals/{trading_day}/signals.json` |
| **M** | which trained recipe emits `predicted_alpha` | `predictions/{trading_day}.json` |
| **S** | exit / risk rules | `strategies/current` |

**U gates the book, not the predictor** (`alpha-engine-config-I12691`). M
scores every name the feature layer carries for the session; U's champion cut
is read by exactly one consumer, S's alpha resolution
(`crucible.slots.strategy`), as the eligibility mask the book is constructed
inside. The daily chain serves M before U (`DAILY_CHAIN_SERVE_ORDER`), so M
cannot read the session's cut even by accident. There is no uncontested width
literal between U and M, which is the v1 residual (`predictor_universe_cut =
attractiveness_top_20`) this replaces. `tests/test_u_feed_reach.py` holds the
wiring and this paragraph to each other.

**This module configures the arena; it never re-implements it.** The ladder,
the paired windows, the anytime-valid confidence sequence, the Condorcet
ranking, the pointer decision and the cap-with-grace retirement rule all live
in `nousergon_lib.arena` and are CALLED. A slot that re-implements policy
§§3–6 is a defect (policy §10), and `arena_config_for` returning the
library's own `ArenaConfig` — rather than a look-alike — is what keeps that
honest: a crucible-local copy would pass every value assertion and diverge
silently the first time the library gained a field.

**The promotion bar is the library's, configured here per slot**
(`alpha-engine-config-I9763`, `-I10504`, `-I10547`). Two parameters say when a
challenger may take the pointer, and both live on `ArenaConfig`:
`promote_min_weeks` (Brian's ruling 2026-09-01: 4 paired weeks by default) and
`promote_evidence` (the anytime-valid sequence by default). Crucible declares
their per-slot VALUES the same way it declares `cap`, `grace_weeks` and
`alpha` — as fields fed into :attr:`SlotSpec.arena` — and never re-implements
the rule that reads them. That is the whole of `-I10504`'s invariant: what it
forbade was a SECOND place holding the value while the library used its own
default, which is what made the two able to drift. A field passed straight
into the config cannot: `spec.promote_min_weeks is spec.arena.promote_min_weeks`
by construction, and `tests/test_slots.py` asserts it for every slot.

**U serves on a point-estimate lead at 2 paired weeks** — Brian's ruling
2026-09-12 (`alpha-engine-config-I10546`): every scanner challenger is
promotable, and `universe_cut` promotes the point-estimate leader after 2
paired weeks. R and S keep the 4-week anytime-valid bar. The asymmetry is
deliberate: a universe cut is re-decided weekly and is cheap to reverse, and
requiring anytime-valid support of a cut's edge promoted nothing for months.

**Every `point` slot promotes only an arm that beats the champion AND every
other challenger** — Brian's ruling 2026-10-03 (`alpha-engine-config#11849`):
"if after minimum two weeks an arm outperforms the champion and all other
challengers then it gets promoted. Otherwise we compare the common window of
weeks for each arm in making our comparison." That is the library's
`promote_against="every_arm"`, declared on U and M: each head-to-head rests on
that pair's own longest common window, and every one of them is on the
`arena_cycle` record (`decision.rivals`) every cycle. The `anytime_valid`
slots, R and S, are unchanged.

**M serves on a point-estimate lead after 2 paired weeks, against every
arm** (`alpha-engine-config-I10689`, ratified with a directive on
`alpha-engine-config#11849`, 2026-10-03). At small n the anytime-valid
interval is wide for the same reason it was inert for `universe_cut`, and
Brian ruled he does not "want to go with no champion for an unknown number of
weeks". His directive then set M's age to 2 paired weeks — it was 4 — and
required the promoted arm to outperform the champion and every other
challenger. The §5.3 behavioural veto and input completeness are unchanged
and still outrank any lead. See the per-slot delta record in
`evaluation-policy.md` § Per-slot promotion deltas.

**Strategy content is not here.** An arm is an immutable recipe living in the
private config repository, loaded at runtime; its id is the hash of its spec.
This module holds the slot *shape* only, which is why it is publishable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypedDict, cast

from nousergon_lib.arena import ArmRegister
from nousergon_lib.arena.engine import (
    EVIDENCE_ANYTIME_VALID,
    EVIDENCE_POINT,
    PROMOTE_AGAINST_EVERY_ARM,
    PROMOTE_AGAINST_INCUMBENT,
    STATISTIC_INFORMATION_RATIO,
    STATISTIC_MEAN_DIFF,
    ArenaConfig,
)

__all__ = [
    "COST_MODELS",
    "EVIDENCE_ANYTIME_VALID",
    "EVIDENCE_POINT",
    "ESTIMATOR_KINDS",
    "PROMOTE_AGAINST_EVERY_ARM",
    "PROMOTE_AGAINST_INCUMBENT",
    "EXIT_RULES",
    "PORTFOLIO_PARAM_FIELDS",
    "REQUIRED_ARM_FIELDS",
    "REQUIRED_RECIPE_FIELDS",
    "SLOTS",
    "STATISTIC_INFORMATION_RATIO",
    "STATISTIC_MEAN_DIFF",
    "WIDTH_EXPERIMENTAL",
    "WIDTH_RANKING_ONLY",
    "WIDTH_SEMANTICS",
    "WidthContractError",
    "assert_width_contract",
    "selection_widths",
    "UNITS_SUFFIXES",
    "ControlArm",
    "EstimatorSpec",
    "SlotRecipes",
    "SlotSpec",
    "arena_config_for",
    "arm_name",
    "attribution_factor_symbols",
    "declared_benchmark_symbols",
    "daily_servers",
    "owed_slot_manifests",
    "dispatchable_slots",
    "history_producer",
    "get_slot",
    "is_control_arm",
    "load_arm_specs",
    "load_portfolio_params",
    "load_model_recipes",
    "load_strategy_recipes",
    "promotable_arms",
]

#: §10.1. Two controls per slot, every cycle: one with a KNOWN injected edge
#: and one that is pure noise. If the grader does not rank planted >
#: real-or-null > null with the expected margin, the grader is broken and the
#: cycle's verdicts are void. This is the only way to know the harness itself
#: works — the audit's central finding was a grading loop that ran for months
#: while measuring nothing.
ControlKind = Literal["planted", "null"]
_CONTROL_KINDS: tuple[str, ...] = ("planted", "null")


@dataclass(frozen=True)
class ControlArm:
    """A synthetic arm scored beside the real ones and never served.

    ``control`` is a field rather than an implied property so it appears
    explicitly in the register row, where the exclusion rules read it. A
    control arm whose flag was forgotten is a look-ahead arm sitting in the
    promotion pool, so the constructor refuses ``control=False`` outright.
    """

    arm_id: str
    kind: ControlKind
    control: bool = True

    def __post_init__(self) -> None:
        if self.kind not in _CONTROL_KINDS:
            raise ValueError(
                f"control kind must be one of {_CONTROL_KINDS}; got {self.kind!r}. "
                "A control that is neither a planted edge nor pure noise cannot "
                "establish that the grader ranks them in the expected order."
            )
        if not self.control:
            raise ValueError(
                f"control arm {self.arm_id!r} must carry control=True. The flag is "
                "what excludes it from the pointer and from the cap; a planted-edge "
                "arm without it is a look-ahead arm eligible for promotion."
            )


#: Width semantics of a selection family (`alpha-engine-config-I12565`, audit
#: CR-05). ``experimental``: the arms carry different selection widths ON
#: PURPOSE and the width is part of what is being tested, so the pointer must
#: rank on the information ratio — a raw mean selects for concentration, not
#: skill (IR ~= IC x sqrt(breadth)). ``ranking_only``: only the ranking is
#: under test, so every arm must select the SAME number of names and the
#: comparison is count-matched by construction.
WIDTH_EXPERIMENTAL = "experimental"
WIDTH_RANKING_ONLY = "ranking_only"
WIDTH_SEMANTICS: tuple[str, ...] = (WIDTH_EXPERIMENTAL, WIDTH_RANKING_ONLY)


class WidthContractError(ValueError):
    """A selection family whose widths and declared statistic disagree."""


@dataclass(frozen=True)
class SlotSpec:
    """One slot's shape and its arena parameters.

    Frozen: this is read in many places and written in one, and a mutable
    spec is a per-slot parameter that can differ between two readers inside
    the same process.
    """

    slot: str
    slot_kind: str
    benchmark: str
    #: The `crucible.slots.<module>` that carries this slot's `produce` and
    #: `grade` entry points once it has them. A declared fact about where
    #: the code lives, read by :func:`dispatchable_slots`; it says nothing
    #: about whether the entry points EXIST yet -- that is read off the
    #: module itself, never asserted here.
    module: str
    #: §4.4: cap 5 (a RETIREMENT criterion, never an admission gate), grace
    #: 4 weeks, floor 3 active arms, retired arms scored 8 trailing cycles.
    cap: int = 5
    grace_weeks: int = 4
    min_active_arms: int = 3
    retired_trailing_cycles: int = 8
    diff_clip: float = 0.05
    alpha: float = 0.05
    #: §5.0 / Brian ruling 2026-09-01: paired WEEKS a challenger must share
    #: with the incumbent before it may take the pointer. Declared here and
    #: enforced by the library — `crucible.promote` carries no age rule of
    #: its own (`alpha-engine-config-I10547`).
    promote_min_weeks: int = 4
    #: The evidence a lead must clear to SERVE: the anytime-valid confidence
    #: sequence, or the point estimate. Brian ruling 2026-09-12
    #: (`alpha-engine-config-I10546`) puts the U slot on `point`.
    promote_evidence: str = EVIDENCE_ANYTIME_VALID
    #: WHO a promotable challenger must beat: the incumbent alone (the library
    #: default), or — Brian ruling 2026-10-03, `alpha-engine-config#11849` —
    #: the incumbent AND every other eligible challenger head to head, each
    #: pair on its own longest common window (`every_arm`). Declared here and
    #: decided by the library, same as the two bars above.
    promote_against: str = PROMOTE_AGAINST_INCUMBENT
    #: WHICH statistic the pointer ranks on, passed to the library's
    #: `ArenaConfig.promote_statistic` (`alpha-engine-config-I12565`). The
    #: default is the library's own (`mean_diff`); a family whose widths are
    #: experimental must declare `information_ratio`, and the library then
    #: requires `promote_evidence=point`.
    promote_statistic: str = STATISTIC_MEAN_DIFF
    #: Whether arm widths (`params.top_n`) are part of the experiment
    #: (:data:`WIDTH_EXPERIMENTAL`) or held equal (:data:`WIDTH_RANKING_ONLY`).
    #: ``None`` = undeclared, which is legal ONLY while every arm the slot
    #: grades selects the same number of names: :func:`assert_width_contract`
    #: refuses a family whose widths vary under no declaration, because the
    #: default mean statistic would then promote the narrowest arm for its
    #: concentration (CR-05).
    width_semantics: str | None = None
    #: The fraction of a session's panel a STACKED arm's base model must have
    #: expressed an opinion on before that session may be built. Brian's
    #: ruling 2026-09-17 (`alpha-engine-config-I10947`, option (a)): a stacked
    #: arm scores the INTERSECTION of the panel and the names its base
    #: actually scored, publishes the coverage, and refuses only BELOW this
    #: floor. Nothing is ever substituted -- a name the base has no opinion on
    #: is an EXCLUDED name, never a zero.
    #:
    #: Declared here, on the slot, because it is a property of what the slot
    #: grades rather than of one arm's recipe: two arms held to two different
    #: coverage floors are graded over two different universes while every
    #: surface renders one number. `crucible.slots.inputs.base_coverage_floor`
    #: is the ONLY reader; a literal at a call site would be a second
    #: declaration of a value that already has one.
    stacked_base_coverage_floor: float = 0.90
    control_arms: tuple[ControlArm, ...] = field(default_factory=tuple)

    @property
    def arena(self) -> ArenaConfig:
        """The library config for this slot.

        Constructed on demand rather than stored, so the library's own
        validation (`ArenaConfigError` on a selection slot benchmarked
        against SPY, on `min_active_arms > cap`, and so on) runs against
        these values every time they are read.
        """
        return ArenaConfig(
            slot=self.slot,
            slot_kind=self.slot_kind,
            benchmark=self.benchmark,
            alpha=self.alpha,
            diff_clip=self.diff_clip,
            cap=self.cap,
            grace_weeks=self.grace_weeks,
            min_active_arms=self.min_active_arms,
            retired_trailing_cycles=self.retired_trailing_cycles,
            promote_min_weeks=self.promote_min_weeks,
            promote_evidence=self.promote_evidence,
            promote_against=self.promote_against,
            promote_statistic=self.promote_statistic,
        )

    def __post_init__(self) -> None:
        if self.width_semantics is not None and self.width_semantics not in WIDTH_SEMANTICS:
            raise WidthContractError(
                f"slot {self.slot!r}: width_semantics is one of {WIDTH_SEMANTICS} or None, "
                f"not {self.width_semantics!r}"
            )
        if (
            self.width_semantics == WIDTH_EXPERIMENTAL
            and self.promote_statistic != STATISTIC_INFORMATION_RATIO
        ):
            raise WidthContractError(
                f"slot {self.slot!r} declares width_semantics={WIDTH_EXPERIMENTAL!r} with "
                f"promote_statistic={self.promote_statistic!r}. When width is part of the "
                "experiment, a mean paired difference rewards concentration rather than "
                f"skill; the family must rank on {STATISTIC_INFORMATION_RATIO!r} "
                "(audit CR-05)."
            )


class WidthContract(TypedDict):
    """What :func:`assert_width_contract` validated, stamped on the cycle."""

    width_semantics: str | None
    widths: dict[str, int]
    promote_statistic: str
    promote_evidence: str


def selection_widths(specs: Iterable[object]) -> dict[str, int]:
    """Each non-control recipe's declared selection width (``params.top_n``).

    A recipe that declares no ``top_n`` has no selection width (the S slot's
    book is solved, not cut) and is left out: the contract below governs the
    width of a SELECTION, and there is none to govern.
    """
    widths: dict[str, int] = {}
    for spec in specs:
        if getattr(spec, "control", False):
            continue
        raw: object = getattr(spec, "params", None) or {}
        if not isinstance(raw, Mapping):
            continue
        params = cast("Mapping[str, object]", raw)
        top_n = params.get("top_n")
        if top_n is None:
            continue
        if not isinstance(top_n, (int, float, str)):
            raise TypeError(f"params.top_n must be a number, not {type(top_n).__name__}")
        name = str(getattr(spec, "arm_id", None) or getattr(spec, "name", spec))
        widths[name] = int(top_n)
    return widths


def assert_width_contract(slot_spec: SlotSpec, specs: Iterable[object]) -> WidthContract:
    """Validate a family's declared width semantics; return what was validated.

    The CR-05 carryover gate (`alpha-engine-config-I12565`). Refuses:

    * widths that vary under NO declaration — the omitted contract;
    * widths that vary under ``ranking_only`` — a ranking-only comparison is
      count-matched, so unequal widths are a recipe error.

    ``experimental`` with varying widths passes, and ranks on the information
    ratio (enforced at :class:`SlotSpec` construction). The returned record is
    stamped on the cycle artifact so the primary statistic, evidence rule and
    widths travel with the decision they produced.
    """
    widths = selection_widths(specs)
    distinct = sorted(set(widths.values()))
    if len(distinct) > 1:
        detail = ", ".join(f"{name}={width}" for name, width in sorted(widths.items()))
        if slot_spec.width_semantics is None:
            raise WidthContractError(
                f"slot {slot_spec.slot!r} grades arms of different widths ({detail}) and "
                "declares no width_semantics. Under the default "
                f"{slot_spec.promote_statistic!r} statistic the narrowest arm wins on "
                "concentration, not skill. Declare the family "
                f"{WIDTH_EXPERIMENTAL!r} (rank on {STATISTIC_INFORMATION_RATIO!r}, "
                f"evidence {EVIDENCE_POINT!r}) or {WIDTH_RANKING_ONLY!r} (equal widths) — a "
                "ruling on the slot's evidence rule, not a default "
                "(audit CR-05)."
            )
        if slot_spec.width_semantics == WIDTH_RANKING_ONLY:
            raise WidthContractError(
                f"slot {slot_spec.slot!r} is declared {WIDTH_RANKING_ONLY!r}, so every arm "
                f"must select the same number of names; got {detail}."
            )
    return {
        "width_semantics": slot_spec.width_semantics,
        "widths": dict(sorted(widths.items())),
        "promote_statistic": slot_spec.promote_statistic,
        "promote_evidence": slot_spec.promote_evidence,
    }


def _controls(slot: str) -> tuple[ControlArm, ...]:
    return (
        ControlArm(arm_id=f"control_planted_{slot}", kind="planted"),
        ControlArm(arm_id=f"control_null_{slot}", kind="null"),
    )


SLOTS: dict[str, SlotSpec] = {
    "u": SlotSpec(
        slot="u",
        slot_kind="universe_cut",
        module="universe",
        # A selection stage is graded against the population it drew from,
        # count-matched. Never SPY.
        benchmark="population",
        # Brian ruling 2026-09-12, `alpha-engine-config-I10546`/`-I10547`:
        # the universe cut promotes the point-estimate leader after 2 paired
        # weeks. R and S keep the 4-week anytime-valid bar.
        promote_min_weeks=2,
        promote_evidence=EVIDENCE_POINT,
        # Brian ruling 2026-10-03, `alpha-engine-config#11849`: the leader
        # must beat the champion AND every other challenger, each on that
        # pair's common window. Applied to every `point` slot; the
        # `anytime_valid` slots (R, S) are unchanged pending his word.
        promote_against=PROMOTE_AGAINST_EVERY_ARM,
        control_arms=_controls("u"),
    ),
    "r": SlotSpec(
        slot="r",
        slot_kind="selection_producer",
        module="research",
        benchmark="population",
        # Brian ruling 2026-10-23 (decision card, option (a);
        # `alpha-engine-config-I12565`, audit CR-05): R's arms select different
        # numbers of names (`no_agent_quant` 15, the rest 10), and that width is
        # part of what R tests. So R ranks on the information ratio, not the mean
        # paired difference, which would hand the pointer to the narrowest arm
        # for its concentration. The library requires `point` evidence with IR,
        # which moves R off the 4-week anytime-valid bar. This mirrors v1
        # research. `promote_min_weeks` and `promote_against` are unchanged.
        width_semantics=WIDTH_EXPERIMENTAL,
        promote_statistic=STATISTIC_INFORMATION_RATIO,
        promote_evidence=EVIDENCE_POINT,
        control_arms=_controls("r"),
    ),
    "m": SlotSpec(
        slot="m",
        slot_kind="model",
        module="model",
        # CPCV OOS IC on canonical 21 trading-day labels; the population is
        # the scored cross-section, not an index.
        benchmark="population",
        # `promote_evidence: point` — Brian ruling 2026-09-13
        # (`alpha-engine-config-I10689`), ratified 2026-10-03 on
        # `alpha-engine-config#11849`: "I don't want to go with no champion
        # for an unknown number of weeks."
        #
        # `promote_min_weeks=2` and `promote_against=every_arm` — his
        # directive of the same ruling, verbatim: "All arms should be
        # compared each week, performance tracked, and if after minimum two
        # weeks an arm outperforms the champion and all other challengers
        # then it gets promoted. Otherwise we compare the common window of
        # weeks for each arm in making our comparison." Was 4 weeks against
        # the incumbent alone.
        promote_min_weeks=2,
        promote_evidence=EVIDENCE_POINT,
        promote_against=PROMOTE_AGAINST_EVERY_ARM,
        # `alpha-engine-config-I10947`: measured over 2024-05-01..2026-06-04,
        # `residual_momentum`'s scored set ran from 887/903 names (98.2%) on
        # the oldest session to 905/908 (99.7%) on the newest -- it drops the
        # rows whose features are null, which young listings always have. The
        # floor sits below the measured worst so a normal history of fresh
        # listings produces, and far above a collapse: at 0.90 a base that
        # lost a tenth of the panel refuses rather than publishing a book
        # over a universe nobody chose.
        stacked_base_coverage_floor=0.90,
        control_arms=_controls("m"),
    ),
    "s": SlotSpec(
        slot="s",
        slot_kind="strategy",
        module="strategy",
        # S is not a selection stage: market-relative canonical alpha net of
        # the cost model, against SPY, is the correct axis here. The
        # population rule above must not be over-applied into a second defect.
        benchmark="SPY",
        control_arms=_controls("s"),
    ),
}


def dispatchable_slots() -> dict[str, ModuleType]:
    """The slots the CLI can actually run, in `SLOTS` order.

    A slot is dispatchable when the module its spec names exposes both
    ``produce`` and ``grade`` -- the entry points `experiment.run` and
    `experiment.grade` call. Read off the modules, never listed: the plan
    brings M and S onto the CLI at phase 3 (§6 row 3), and until their
    entry points exist a weekly arc that expanded over all four slots failed
    by construction at ``experiment.run[m]`` while the phase-1 gate, deriving
    the same set, could never read MET. Measured 2026-09-04 with
    ``crucible weekly --date 2026-08-28 --run-mode replay --dry-run``.

    So the arc grows the moment a slot's entry points land, with no list to
    update and no gate clause to re-derive -- both read this.
    """
    import importlib  # noqa: PLC0415 - lazy: the submodules import from this package

    found: dict[str, ModuleType] = {}
    for slot, spec in SLOTS.items():
        module = importlib.import_module(f"crucible.slots.{spec.module}")
        if callable(getattr(module, "produce", None)) and callable(getattr(module, "grade", None)):
            found[slot] = module
    if not found:
        raise RuntimeError(
            "no slot module exposes both `produce` and `grade`; the weekly arc would "
            "run no experiment at all and report `ok`"
        )
    return found


def daily_servers() -> dict[str, ModuleType]:
    """The slots `serve.daily` can run: those whose module exposes ``serve_daily``.

    `alpha-engine-config-I12047`. Read off the modules, never listed, for the
    reason :func:`dispatchable_slots` is: the job grows the day a slot's
    daily entry point lands, with no second list to update. Members today:
    M (the champion's predictions feed), U (the champion's cut) and S (every
    live challenger's construction inputs, `alpha-engine-config-I12021`),
    under one job name, discriminated by slot. R has none: nothing downstream
    reads an R feed on a session the arc did not produce.
    """
    import importlib  # noqa: PLC0415 - lazy: the submodules import from this package

    found: dict[str, ModuleType] = {}
    for slot, spec in SLOTS.items():
        module = importlib.import_module(f"crucible.slots.{spec.module}")
        if callable(getattr(module, "serve_daily", None)):
            found[slot] = module
    return found


def owed_slot_manifests(job: str) -> frozenset[str] | None:
    """The slots a SCHEDULED job owes one manifest EACH per trading day, or
    ``None`` for a job whose one manifest per day is owed whatever it carries.

    `alpha-engine-config-I12054`. `serve.daily` runs once per slot under one
    job name, discriminated by slot, and each slot has its own schedule — so
    its absence is a per-slot question. `crucible.alerts.evaluate_absence`
    used to answer it with "is ANY manifest under the day's prefix", and M's
    manifest then cleared a day on which `serve-daily-s` never fired.

    Read off :func:`daily_servers`, the same declaration the CLI's `--slot`
    choices are, never a list in the alerter: the day a slot's daily entry
    point lands, its absence is graded with no second list to update.

    `experiment.run`/`experiment.grade` answer ``None`` on purpose: they are
    `dispatch: arc` rows, and which slots the arc that ran actually planned is
    the arc's own record (`crucible.alerts._arc_declared_members`), not this
    function's to restate.
    """
    from crucible.slots.cycle import DAILY_SERVE_JOB  # noqa: PLC0415 - lazy, as above

    if job == DAILY_SERVE_JOB:
        return frozenset(daily_servers())
    return None


def history_producer(module: ModuleType) -> Any:
    """``module.produce_history`` — the per-session entry point a BACKFILL uses.

    `alpha-engine-config-I11005`. A slot module's ``produce`` is a production
    SERVING cycle: for U and R it resolves the champion pointer and writes the
    slot's feed, and for M it republishes the champion's predictions under the
    trader's key. Both refuse when the pointer names an arm that produced no
    artifact this cycle, which is correct for production — a pointer to an arm
    that did not produce means production has no feed today — and wrong for a
    historical backfill, which feeds nothing and serves nothing. Running it
    anyway made every NON-champion arm unbackfillable, so no challenger could
    accumulate the history it needs to become champion: measured live
    2026-09-17 on `experiment.backfill --slot u --arm attractiveness`, which
    failed in two minutes on the first of 205 sessions.

    **Raises rather than falling back to ``produce``.** A silent fallback is
    the whole defect wearing a resolver's clothes: it would put the serving
    path back under the backfill for exactly the module that forgot to
    declare its history entry point, and nothing would say so. S declares one
    explicitly even though it equals ``produce``, because S has no serving
    half — a fact worth stating, not inferring.
    """
    producer = getattr(module, "produce_history", None)
    if not callable(producer):
        raise RuntimeError(
            f"slot module {module.__name__!r} declares no `produce_history`, so a "
            "backfill has no entry point that skips the serving path. Falling back to "
            "`produce` would re-enter it and refuse every non-champion arm."
        )
    return producer


def get_slot(slot: str) -> SlotSpec:
    """The spec for ``slot``. Raises on an unknown slot — never returns None."""
    try:
        return SLOTS[slot]
    except KeyError as exc:
        raise KeyError(
            f"unknown slot {slot!r}; the four slots are {sorted(SLOTS)}. A slot that "
            "is not registered has no benchmark and no arena config, so a cycle run "
            "for it would produce an unlabelled number."
        ) from exc


def arena_config_for(slot: str) -> ArenaConfig:
    """The `nousergon_lib.arena` config for ``slot``."""
    return get_slot(slot).arena


def declared_benchmark_symbols() -> frozenset[str]:
    """Every non-population benchmark symbol a slot declares.

    ``"population"`` (U, R, M) means "graded against the universe it was
    drawn from" — it names no ticker and needs no panel row. Every other
    value (today: only S's ``"SPY"``) is a real symbol the panel compiler
    must carry a row for on every session it covers, or
    `crucible.slots.strategy.grade_arm` refuses the book
    (`alpha-engine-config-I10635`).

    Derived from :data:`SLOTS` rather than duplicated at the compiler: a
    second literal naming the same symbol is a second place it can be
    changed, which is the exact units-mismatch shape this repository exists
    to catch (`avg_volume_20d`, 901/903 tickers silently failing the
    liquidity gate for months).
    """
    return frozenset(spec.benchmark for spec in SLOTS.values() if spec.benchmark != "population")


def attribution_factor_symbols(
    *, store: Any | None = None, strategy_dir: Path | str | None = None
) -> frozenset[str]:
    """Every ticker the factor-neutral attribution spec proxies with, if one is declared.

    Sibling of :func:`declared_benchmark_symbols` — a SECOND source of
    symbols the compiled panel must carry a row for
    (`alpha-engine-config-I10683`). `strategy/slots/attribution.yaml` names
    ETF proxies (`SPY`, `IWM`, `XLK`, `XLV`, `XLF`, `XLE` today) for the
    beta/sector/size factors `crucible.attribution.compute_factor_attribution`
    regresses against; none of those five non-`SPY` tickers is any slot's own
    `benchmark`, so `declared_benchmark_symbols` alone never carried them, and
    `crucible.slots.strategy.grade`'s call into `compute_factor_attribution`
    raises `MissingArtifactError` the moment one has no panel row.

    Derived from `AttributionFactorParams.factors` — never a second hand-kept
    literal list — the same discipline `declared_benchmark_symbols` applies to
    `SLOTS`: a units-mismatch-shaped defect (`avg_volume_20d`, 901/903 tickers
    silently failing a gate for months) is exactly what a hand-kept second
    list would risk here too. Both a factor's raw ``proxy`` and, for a spread
    factor, its ``short_proxy`` (`alpha-engine-config-I10592`) are included:
    `crucible.attribution.factor_return_series` reads both legs.

    Returns an empty set when **no** attribution spec is declared at all —
    absence of `strategy/slots/attribution.yaml` (or its store key) is a
    legitimate state for an environment that has not adopted factor
    attribution (this repo's own unit fixtures, chiefly), unlike a slot's
    `benchmark`, which every slot always declares. Once a spec IS declared,
    `load_attribution_params_from_store` still raises on anything malformed
    — this function adds no swallow beyond the "was one declared at all"
    check.
    """
    if strategy_dir is not None:
        spec_path = Path(strategy_dir) / "slots" / "attribution.yaml"
        if not spec_path.exists():
            return frozenset()
    elif store is not None:
        from crucible.keys import strategy_slot_key  # noqa: PLC0415 - avoids a cycle

        if not store.exists(strategy_slot_key("attribution")):
            return frozenset()
    else:
        raise ValueError("attribution_factor_symbols needs either a store or a strategy_dir")

    from crucible.attribution import (
        load_attribution_params_from_store,  # noqa: PLC0415 - avoids a cycle
    )

    params = load_attribution_params_from_store(store=store, strategy_dir=strategy_dir)
    symbols: set[str] = set()
    for fdef in params.factors.values():
        symbols.add(fdef.proxy)
        if fdef.short_proxy is not None:
            symbols.add(fdef.short_proxy)
    return frozenset(symbols)


def arm_name(arm_id: str) -> str:
    """The NAME component of a registered arm id, or a bare name unchanged.

    `nousergon_lib.arena.derive_arm_id` returns ``{slot}:{name}:{spec_hash}``
    and forbids ``:`` inside either the slot or the name, so a registered id
    has exactly three colon-separated parts and the middle one is the name.
    A string with no colon at all is a bare name — what a recipe file and a
    :class:`ControlArm` carry before registration — and is returned as it is.

    **Anything else raises.** This function is what
    :func:`is_control_arm` and :func:`promotable_arms` bind on, so an id
    shape neither branch understands must not be quietly reported as "not a
    control": that is the F4 failure mode inverted, and a look-ahead arm
    would reach the pointer through it.
    """
    if not arm_id:
        raise ValueError("arm_id must be non-empty")
    if ":" not in arm_id:
        return arm_id
    parts = arm_id.split(":")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"arm_id {arm_id!r} is neither a bare name nor a "
            "'{slot}:{name}:{spec_hash}' id from nousergon_lib.arena.derive_arm_id. "
            "Refusing rather than guessing: the control-arm exclusion reads the NAME "
            "component, and an id this function cannot parse would be reported as a "
            "non-control and become eligible to serve."
        )
    return parts[1]


def is_control_arm(spec: SlotSpec, arm_id: str, register: ArmRegister | None = None) -> bool:
    """Whether ``arm_id`` is one of ``spec``'s control arms.

    **Register-backed when a register is given and carries the arm**
    (`alpha-engine-config-I9943`). `crucible.slots.arms.register_arms` now
    forwards `ArmSpec.control` onto the registered `ArmRecord.control`
    (`crucible-PR102`), so once an arm is registered the FLAG is the fact —
    read from `register.state(arm_id).record.control` — and the name match
    below is only the FALLBACK for a bare (unregistered) id or for a row
    written before the flag existed. The flag is authoritative even when it
    disagrees with the name: a filed, non-control recipe whose generated
    name happens to collide with a control's is not excluded once it is
    registered with `control=False`, and a control is excluded even if
    `spec.control_arms` is later emptied, because its own record still says
    `control=True`.

    **Matched on the NAME component when falling back, never on the whole
    string.** A :class:`ControlArm` carries the bare name
    (``control_planted_m``) because that is what
    `crucible.slots.arms.control_specs` passes to `derive_arm_id` as the
    recipe's name; the arm the register, the series and the pointer all
    speak about is ``m:control_planted_m:7e8059f49558``. An equality test
    against the literal therefore matched nothing that had ever been
    registered — the exclusion existed and could not fire
    (`alpha-engine-config-I9757`, F4).
    """
    if register is not None and arm_id in register:
        return register.state(arm_id).record.control
    return arm_name(arm_id) in {c.arm_id for c in spec.control_arms}


def promotable_arms(
    spec: SlotSpec, arm_ids: list[str], register: ArmRegister | None = None
) -> list[str]:
    """``arm_ids`` minus the slot's control arms, order preserved.

    §10.1: controls are scored every cycle and are excluded from the pointer
    and from the cap. Excluded from the pointer because the planted arm reads
    next-period returns and promoting it would be a look-ahead in production;
    excluded from the cap because the cap is a retirement criterion over
    competing arms, and two controls counting against it would mean every
    slot starts two arms into its own retirement pressure — a cap of 5 that
    behaves as 3.

    Accepts registered ids and bare names alike (:func:`arm_name`), because
    both shapes exist: a recipe is loaded by name and scored by id, and a
    filter that only understood one of them is a filter that never fired on
    the path that matters.

    ``register``, when given, makes the exclusion register-backed rather than
    name-matched — see :func:`is_control_arm` (`alpha-engine-config-I9943`).
    """
    return [a for a in arm_ids if not is_control_arm(spec, a, register)]


# ---------------------------------------------------------------------------
# The public surface for consumers who validate strategy content, never
# re-implement it (`alpha-engine-config-I9766`; plan §4.11, §9.1). Imported
# down here, after the classes above, because `crucible.slots.arms` imports
# `ControlArm` and `SlotSpec` back from this package — a top-of-file import
# would be circular.
# ---------------------------------------------------------------------------

from crucible.portfolio import (  # noqa: E402
    COST_MODELS,
    PORTFOLIO_PARAM_FIELDS,
    load_portfolio_params,
)
from crucible.slots.arms import REQUIRED_ARM_FIELDS, load_arm_specs  # noqa: E402
from crucible.slots.model import _ESTIMATORS as _MODEL_ESTIMATOR_KINDS  # noqa: E402
from crucible.slots.model import (  # noqa: E402
    REQUIRED_RECIPE_FIELDS,
    UNITS_SUFFIXES,
    EstimatorSpec,
    SlotRecipes,
    load_model_recipes,
)
from crucible.slots.strategy import EXIT_RULES, load_strategy_recipes  # noqa: E402

#: The estimator kinds `EstimatorSpec` accepts (`crucible.slots.model._ESTIMATORS`).
#: Re-exported here, rather than left private, so a consumer validating a
#: recipe file (`alpha-engine-config-I9766`) has a stable name to import
#: instead of reaching into a module-private tuple. `test_public_surface.py`
#: pins this alias to the module's own tuple by identity of *value*, so the
#: two cannot silently drift apart.
ESTIMATOR_KINDS: tuple[str, ...] = _MODEL_ESTIMATOR_KINDS
