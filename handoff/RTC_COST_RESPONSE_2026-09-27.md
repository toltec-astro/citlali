# RTC cost/source response handoff — 2026-09-27

The [assessment](../validation/rtc_cost_response_2026-09-27/ASSESSMENT.md) completes
the authorized bounded local POINT152430/a2000 nw11/12 effort. A six trajectories,
B three, total 45 applications plus one RTC capture; no replacement trajectory.
Development endpoint: cold M0, warm fully re-solved observed-entry ALS rank 10,
five applications, m4 applied and m5 recorded. Original freeze remains unchanged.

RTC accounts for 15.88–18.77% of actual wall time. It earns measured antialias
filtering but leaves the 11 Hz line for PTC; transient completeness/false rejection
and donor efficacy are not established. Pre-RTC compact transfer gain 0.996307,
shoulder 0.999117, no changed state/center/map support. At T5 compact peak gains
0.99195/0.99197; broad aperture recovery 0.83945 with retained morphology cost.

B's two serialization changes are byte exact in all three gates. Network-11 RTC
array-publication median drops 1.607 to 1.125 s, command CPU 3.81%; command exit
72.676 to 73.197 s with overlapping ranges, hence no demonstrated overall speedup.
B response evidence is inherited. C is deliberately unselected.

Next action: owner decision on exact detector applicability/admission of the
historical 11.0062558446 Hz / 1 Hz / approximately 3 s fixed notch, followed by one
fixed-design proposal. No threshold sweep or automatic execution. Keep PTC/FRUIT
optimization parked. Reproduction, exact source bindings, warnings, comparator
incident, reviews and cleanup are in the [record](../validation/rtc_cost_response_2026-09-27/REPRODUCTION.md).
All work stays local; no push, merge, Unity or production activation.
