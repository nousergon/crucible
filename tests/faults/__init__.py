"""Scripted fault injection. §10 component 7 / plan §6 phase-2 gate.

"Flawless from day 1" is proven on the failure path, not the success path.
The old rehearsal pipeline failed 7 of 7 because nobody had made it fail on
purpose first.

Four faults, each asserting the same four properties:

    1. the run ends `status: failed`
    2. with the RIGHT `reason` — naming the fault, not a generic message
    3. carrying full telemetry (all five signal classes, the correlation id)
    4. producing EXACTLY ONE page

Property 4 is the one a fault-injection suite usually omits, and it is the
one that matters: a system that pages five times for one outage is as
unactionable as one that pages zero times, and only a captured transport can
tell the difference.
"""
