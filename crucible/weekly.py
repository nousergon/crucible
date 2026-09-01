"""The weekly arc: one declared sequence, derived from the registry.

Normative source: plan §4.1 ("control flow lives in Python and is tested with
pytest, not in a state machine"), §4.6, §9.2, and the §6.1 phase-1 gate.

`components.yaml` declares six jobs as `schedule: weekly, Saturday`, each with
a deadline and each watched for absence. Something has to run them. Before
this module the EventBridge schedule dispatched exactly one of them
(`data.weekly`) and nothing chained the rest, so five declared components
would have been absent every Saturday — pages for work no scheduler was ever
going to start. A declared cadence is not evidence of a cadence.

**The sequence is DERIVED, never listed.** A row joins the arc by declaring
`dispatch: arc`, and its position is its own deadline. A hand-written stage
list is the bug class §11.1 is about: it is written from the jobs someone
remembered, and it goes stale in the direction of running fewer of them.

**Every stage runs through `crucible.cli.main`** — the exact argv an operator
would type. Not the handler, not a copy of the dispatch table: a parallel path
can be healthy while the real one is broken, and the arc exists precisely to
be the thing the phase gate measures.

**A failed stage fails the arc, immediately.** No `continue`, no partial
success, no "the rest still ran": the downstream stages read what the failed
one was supposed to write, and running them anyway produces a report card
built on an absent input — a well-formed artifact containing nothing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from crucible.components import Component, load_registry
from crucible.slots import SLOTS

__all__ = ["ARC_SLOT_JOBS", "Stage", "arc_stages", "run_arc"]

#: The arc jobs that are run once per slot rather than once. Derived from the
#: CLI's own `--slot` requirement in `tests/test_weekly.py`, so a new
#: slot-scoped job cannot join the arc and silently run for one slot.
ARC_SLOT_JOBS: frozenset[str] = frozenset({"experiment.run", "experiment.grade"})


@dataclass(frozen=True)
class Stage:
    """One invocation of the arc: a job, its slot if it has one, its argv."""

    job: str
    slot: str | None
    due_at: dt.datetime

    def argv(self, *, trading_day: dt.date, store: str | None) -> list[str]:
        argv = [self.job, "--date", trading_day.isoformat()]
        if store:
            argv += ["--store", store]
        if self.slot:
            argv += ["--slot", self.slot]
        return argv

    @property
    def label(self) -> str:
        return f"{self.job}[{self.slot}]" if self.slot else self.job


def _arc_rows(registry: dict[str, Component]) -> list[Component]:
    rows = [c for c in registry.values() if c.dispatch == "arc" and c.lifecycle == "ACTIVE"]
    if not rows:
        raise ValueError(
            "no component declares `dispatch: arc`, so the weekly arc would run "
            "nothing and report `ok`. An empty sequence is a broken registry, not "
            "a quiet week."
        )
    return rows


def arc_stages(trading_day: dt.date, registry: dict[str, Component] | None = None) -> list[Stage]:
    """The arc for ``trading_day``, in the order the deadlines already imply.

    Ordering is `(deadline, job name)`. The deadline table is the dependency
    order — `data.weekly` at 09:00 is what `experiment.run` at 12:00 reads,
    and `report` at 16:00 reduces both — so deriving the order from it means
    there is one declaration of the arc's shape rather than two that can
    disagree. The name is the tie-break, so two stages sharing a deadline
    have a deterministic order rather than a dict-insertion one.

    A slot-scoped job expands into one stage per slot, in `SLOTS` order (u
    then r then m then s), which is the data dependency: the universe cut
    feeds the signal, the signal feeds the model, the model feeds the
    strategy.
    """
    registry = registry or load_registry()
    stages: list[Stage] = []
    for row in sorted(_arc_rows(registry), key=lambda c: (c.deadline.due_at(trading_day), c.name)):
        due = row.deadline.due_at(trading_day)
        if row.name in ARC_SLOT_JOBS:
            stages.extend(Stage(row.name, slot, due) for slot in SLOTS)
        else:
            stages.append(Stage(row.name, None, due))
    return stages


class ArcStageFailed(RuntimeError):
    """A stage exited non-zero. Carries which one, so the arc's own manifest
    `reason` names a job rather than a traceback in a module nobody opens."""


def run_arc(
    trading_day: dt.date,
    *,
    store: str | None,
    registry: dict[str, Component] | None = None,
    main: object | None = None,
) -> list[Stage]:
    """Run every stage for ``trading_day``. Raises on the first failure.

    ``main`` is injectable for tests only; it defaults to `crucible.cli.main`,
    imported lazily because `cli` imports this module's handler. A test that
    passed a fake would be testing its fake, so `tests/test_weekly.py` also
    asserts the real default is `crucible.cli.main` itself.
    """
    if main is None:
        from crucible.cli import main as cli_main  # noqa: PLC0415 - cycle; see docstring

        main = cli_main
    ran: list[Stage] = []
    for stage in arc_stages(trading_day, registry):
        try:
            code = main(stage.argv(trading_day=trading_day, store=store))  # type: ignore[operator]
        except BaseException as exc:  # noqa: BLE001 - re-raised on the next line
            # NOT a swallow: re-raised immediately, chained to the original.
            # A stage that RAISES rather than returning non-zero would
            # otherwise leave the arc's own manifest reason naming the
            # underlying error with no clue which of twelve stages produced
            # it — and the stage name is the first thing an operator needs at
            # 09:00 on a Saturday. The stage's own manifest already carries
            # the full cause.
            raise ArcStageFailed(
                f"weekly arc stage {stage.label} raised for {trading_day.isoformat()}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if code != 0:
            raise ArcStageFailed(
                f"weekly arc stage {stage.label} exited {code} for {trading_day.isoformat()}; "
                "the stages after it read what it was to write, so the arc stops here "
                "rather than producing a report card over an absent input."
            )
        ran.append(stage)
    return ran
