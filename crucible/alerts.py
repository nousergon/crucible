"""Two page conditions, one heartbeat, and nothing else.

Normative source: plan §4.6, §9.3.

    1. **Absence** — a scheduled job's manifest does not exist by its deadline.
    2. **Failure** — a manifest exists with `status: failed`.

That is the exhaustive list. Everything else is a row on a page nobody is
paged to. The v1 system carried 185 alert rules and 153 alarms and still
missed three consecutive Saturday failures, because the signal was buried in
the ones that fire for nothing.

**The deadline table is declarative and lives in `components.yaml`**, ≈6 rows
rather than 185, and it is read from there rather than restated here — a
contract restated in two places has already drifted.

**Deadlines are trading-calendar-relative** (§4.12): "present by *T* after the
close of trading day *d*". A Monday holiday therefore raises no page, and
nobody has to mute anything to get that.

**Alert count is itself a metric with a ceiling.** Exceeding it is a defect in
this module, never a reason to add a suppression — and this repository carries
no suppression collection at all (§11.1).

**The heartbeat is not a third page condition.** It is the answer to "is the
alerting path itself alive": a component emitting nothing is unobserved, not
healthy, and that includes the thing whose job is to notice silence.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "PAGE_CONDITIONS",
    "Page",
    "dedup_key",
    "evaluate_absence",
    "evaluate_failure",
    "heartbeat",
    "send",
]

#: Exhaustive. Adding a third member is a design change, and it is visible in
#: a diff as one — which is the point of writing it down as a closed set
#: rather than as prose in a runbook.
PageCondition = Literal["absence", "failure"]
PAGE_CONDITIONS: tuple[str, ...] = ("absence", "failure")


@dataclass(frozen=True)
class Page:
    """One page. Carries its correlation identity, or it is unactionable.

    ``run_id`` is None only for an ABSENCE page — there was no run, so there
    is no id. That is the one legitimate case, and it is why the field is
    optional rather than why it is often missing.
    """

    condition: PageCondition
    job: str
    trading_day: dt.date
    reason: str
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.condition not in PAGE_CONDITIONS:
            raise ValueError(
                f"{self.condition!r} is not a page condition. There are exactly two "
                f"({PAGE_CONDITIONS}); a third is a design change, not a config value."
            )
        if not self.reason.strip():
            raise ValueError(
                f"a {self.condition} page for {self.job} on {self.trading_day} carries "
                "no reason. A page an operator cannot act on is noise with an alarm "
                "attached."
            )
        if self.condition == "failure" and self.run_id is None:
            raise ValueError(
                "a failure page must carry the run_id of the failed run — the manifest "
                "exists, so its correlation identity does. Only an ABSENCE page "
                "legitimately has none."
            )


def dedup_key(condition: str, job: str, trading_day: dt.date) -> str:
    """One key per (job, trading day). §4.6.

    Not per attempt and not per condition-instance: a job that fails, is
    retried by the declared transient class and fails again is ONE incident,
    and paging twice for it is how a two-page-per-month ceiling is blown by a
    single bad Saturday.
    """
    return f"{condition}:{job}:{trading_day.isoformat()}"


def evaluate_absence(*args: Any, **kwargs: Any) -> list[Page]:
    """Page for every scheduled job whose manifest is missing past its deadline.

    Track C. It reads the deadline table from `components.yaml` and resolves
    each deadline against the trading calendar, so a market holiday moves the
    deadline rather than producing a page that has to be dismissed.
    """
    raise NotImplementedError(
        "evaluate_absence is track C's (crucible-v2 phase 1, alpha-engine-config-I9757). "
        "It must read deadlines from crucible/components.yaml — never a second copy — "
        "and resolve them against the trading calendar."
    )


def evaluate_failure(*args: Any, **kwargs: Any) -> list[Page]:
    """Page for every manifest written with `status: failed`.

    Track C. The manifest already carries the cause; this reads it rather
    than re-deriving one, so the page and the artifact never disagree.
    """
    raise NotImplementedError(
        "evaluate_failure is track C's (alpha-engine-config-I9757). The page's reason "
        "is the manifest's `reason` verbatim — a page and its artifact disagreeing is "
        "worse than either alone."
    )


def heartbeat(*args: Any, **kwargs: Any) -> None:
    """Emit proof that the alerting path itself ran.

    Track C. Not a third page condition: it is the number that distinguishes
    "nothing was wrong this week" from "the thing that notices was dead this
    week". Its ABSENCE is what surfaces, on the console, never as a page.
    """
    raise NotImplementedError(
        "heartbeat is track C's (alpha-engine-config-I9757). It is a console row, not "
        "a page — 'no data' must never render as green (principle 7)."
    )


def send(page: Page, *args: Any, **kwargs: Any) -> None:
    """Deliver one page through `krepis.alerts`.

    Track C. Delivery failure RAISES: an alert that could not be sent and was
    logged instead is an outage nobody hears about, and the send path is the
    last place a silent swallow belongs.
    """
    raise NotImplementedError(
        "send is track C's (alpha-engine-config-I9757). It delivers via krepis.alerts "
        "and RAISES on delivery failure — a swallowed send is a silent outage."
    )
