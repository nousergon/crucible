"""No aggregate hides a member (`alpha-engine-config-I10417`).

Normative source: plan §2, new objective row added 2026-09-10.

Two measured v1 failures, one class: the evaluator's report card disagreed
with itself by 37.8 points (two graders writing one field name), and a
weighted tile scored RED/F was silently excluded from the headline grade.
Both are reductions that dropped a member — the defect lives in the
*reducer*, not the emitters, so v2's per-component observability contract
(§9.2) does not prevent it by itself.

This module is the ONE place the rule is enforced, shared by every reducer
that folds several graded things into one (`crucible.report`'s attribution
grade, `crucible.gate`'s `GateResult`, `crucible.board`'s declared-row
readings, `crucible.gate`'s phase ladder) — rather than four reimplementations
that could each get the ranking or the empty-input case slightly wrong.

Every reduced figure in this tree must:

1. name its members and each member's status in the same artifact as
   ``members[]``, each with ``id``/``value``/``status``;
2. propagate a member that is `null`/`UNVERIFIED`/`failed` to the parent —
   never dropped from the denominator, and the parent may not be greener
   than its worst member unless the artifact declares a reason code;
3. be reconstructible: recomputing the parent from ``members[]`` alone
   reproduces it.

:func:`worst_member` is (2) and the reconstruction half of (3): a caller
recomputes the parent status by calling it again over the SAME ``members``
the artifact carries, rather than trusting a cached field. :class:`MemberRow`
and :func:`member_dicts` are (1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["MemberRow", "member_dicts", "worst_member"]


@dataclass(frozen=True)
class MemberRow:
    """One member of a reduction: what it is, what it measured, and its status.

    ``value`` is whatever scalar or structure the member carries — a metric
    value, a boolean (a gate clause's ``met``), or `None` when the member has
    no scalar of its own (a board row, whose only fact is its state). ``status``
    is the member's own closed-vocabulary state, in whatever vocabulary the
    owning reducer uses (`krepis.metrics.StatusLiteral`, a gate clause's
    MET/UNMET/UNMEASURABLE, a board state) — this module never invents a
    second one.
    """

    id: str
    value: Any
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "value": self.value, "status": self.status}


def member_dicts(members: Sequence[MemberRow]) -> list[dict[str, Any]]:
    """``members``, rendered as the ``members[]`` an artifact carries."""
    return [m.to_dict() for m in members]


def worst_member(members: Sequence[MemberRow], *, rank: Mapping[str, int]) -> MemberRow:
    """The member whose status ranks WORST under ``rank`` (higher = worse).

    This is the whole enforcement of "never greener than the worst member":
    a caller derives the parent's status from this member's status (or from
    its rank), never by averaging, never by counting only the members that
    happened to report, and never by excluding a red member from the count.

    Two refusals, both loud rather than silently wrong:

    * **empty ``members``** raises. A reduction over nothing has no worst
      member to be no-greener-than, and rendering a parent status anyway is
      principle 7's "no data painted green" wearing a reducer's clothes.
    * **a status absent from ``rank``** raises, for every member carrying
      it — not just the first. A status this reduction does not know how to
      place must never be silently treated as the best one, which is what
      Python's `max` would do if an unranked status defaulted to a very low
      rank. Because clauses are the majority caller and typically differ by
      only one or two statuses, this is reported as the full offending set
      rather than one example, so a caller fixing its rank table does not
      have to run the check again to find the second problem.
    """
    if not members:
        raise ValueError(
            "worst_member called over zero members: a reduction with no members has "
            "nothing to be no-greener-than, and reporting a parent status over it "
            "would be principle 7's 'no data' rendered as green"
        )
    unranked = sorted({m.status for m in members if m.status not in rank})
    if unranked:
        raise ValueError(
            f"member status(es) {unranked} carry no declared rank in this reduction's "
            f"rank table {sorted(rank)} — an unranked status must never be treated as "
            "better than every ranked one"
        )
    return max(members, key=lambda m: rank[m.status])
