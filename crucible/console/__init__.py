"""The console page: one static render, from the manifests.

Normative source: plan §4.9, §9.2.

One page — the attribution table, the last N run manifests with their
fourteen-state classification, the champion per slot, the week's cost and the
deploys. Served from the manifests already in the store.

**No `GetMetricData` polling.** The v1 audit measured 517k CloudWatch
retrievals a month for a surface nobody could answer a question from. Every
number here is read from an artifact that had to exist anyway, so the page
costs a listing and nothing else, and every figure on it is reconstructible
by someone who was not in the session (principle 1).

**Static, rendered to S3.** `console-policy` owns how the fleet console is
organised; until v2 is onboarded onto it, a static render is what §4.9 says
is acceptable — and it is a page, not a dashboard framework, so onboarding
later replaces one function.
"""

from __future__ import annotations

from crucible.console.classify import STATES, Classification, classify
from crucible.console.render import ConsolePage, build_page, render_html

__all__ = [
    "STATES",
    "Classification",
    "ConsolePage",
    "build_page",
    "classify",
    "render_html",
]
