# Compromised detectors: owner direction and required diagnostic

## Owner decision

On 2026-10-28, Grant adopted this direction for the general pipeline:

> For the general pipeline, I recommend **excluding compromised detectors locally while allowing the observation to continue when the remaining data support the requested products**. We should develop the rejection rule from disturbance evidence and downstream consequences—not impose an arbitrary minimum uninterrupted duration.

Grant additionally requires an output diagnostic identifying how many detectors
were dropped for this reason. Preventing disturbance imprints in the final maps
is the motivating scientific priority. The small numerical fraction represented
by one detector is not itself a new noise, coverage or signal-recovery criterion.

This is an owner decision recorded on the local development branch, not a claim
of canonical integration, implemented behavior or production activation. No
detectors have been newly excluded and no reduction has been run under it.

## Policy meaning and limits

Exclusion is local to the affected observation and detector identity. It does
not permanently mark an instrument channel bad, discard its entire network,
or modify the supplied APT. Preserve original inputs and measured facts.

A compromised detector must not contribute through its own science samples,
atmospheric estimation or donor operations. Resolve its actual dependencies;
where it already influenced an estimate used by another detector, recompute
the affected processing from the bound original inputs under the frozen
exclusion decision. Removing only its final map contribution is insufficient.
Preserve paired x/r validity and the explicit downstream missing-value policy.

Continue other valid data through the existing project interface, provided
their required assessments and product-specific admission conditions hold.
For the first collaborator milestone, POINT and SCIENCE still require all
three arrays. An absent array, untrusted calibration or unmet required product
must not be hidden behind a successful partial-completion label. Do not invent
a minimum detector fraction or relax existing criteria to obtain completion.

Learn records the disturbances, support and assessment outcomes. The scientific
owner's Consider step decides exclusion under an identified rule and freezes
the detector membership. Apply uses that decision consistently. No silent
trial-and-error rejection, automatic threshold relaxation or map-appearance
selection is implied.

**The owner subsequently approved the first concrete predicate and its
implementation:** at least two distinct admitted level-shift events must affect
two distinct science processing segments under existing whole-segment
exclusions; numerical output must remain; both final conditioned-coordinate
assessments must be unavailable specifically with `fixed_grid_unavailable`
on nonempty support under the existing strict-grid method. Original spectra
must be available and input/arithmetic errors must not be hidden. The policy
identity is `repeated-level-shifts-unassessable-remainder-v1`. A short
stretch, an unavailable assessment, a single event flag, or a low source S/N
does not by itself identify a compromised detector under this decision.
Disturbance evidence and downstream consequences must support the rule.
This direction does not authorize an invented numerical threshold.

## Required compact output diagnostic

This diagnostic is mandatory even with `verbose=false`. It is a small detector
summary, not a detector-by-sample tracking product. Required diagnostic write
failures must propagate as required-output failures.

Report once per observation, with observation totals and one row per requested
array. Include these defined fields:

| Field | Meaning |
| --- | --- |
| Policy identity and evaluation state | Exact selected rule/version; `not_evaluated`, `incomplete` or `complete`, separately from the reduction's outcome |
| Input detector count | Unique detector occurrences in the requested input domain |
| Previously unavailable count | Detectors with zero eligible input to this decision for separately recorded upstream reasons |
| Evaluated detector count | Detectors actually considered by this disturbance-exclusion policy |
| Dropped for disturbances count | Unique previously eligible detectors newly excluded by this policy for the observation |
| Retained after this decision count | Evaluated detectors not excluded by this policy; this is not a claim of final map contribution |
| Final contributor count | Unique detectors actually contributing admitted samples to the required science products, when that processing completes |
| Per-reason counts | Counts for the implemented disturbance rules, with overlap explicitly labeled if a detector has multiple reasons |

Maintain a compact detail list for every newly excluded detector: array,
network and channel locator; authoritative observation/APT occurrence identity;
decision reason; and a reference to the supporting disturbance/assessment
evidence. Channel number alone is insufficient identity. This detail must be
available without rerunning a reduction or enabling verbose mode after the fact.

Count each excluded detector once per observation, not once per x/r coordinate,
processing segment, event, or downstream stage. Observation totals must equal
the sum of the disjoint array counts. Distinguish additional downstream losses
from this policy's exclusions. Report the input population alongside the count
so users can assess the scale of the loss.

For a complete decision, evaluated = newly dropped + retained. Previous
unavailability must not inflate the newly dropped count. If some inputs cannot
be evaluated, report them explicitly rather than force these populations into
a false complete partition. Final contributors require their own realized
accounting and need not equal retained-after-decision counts.

A zero is legitimate only after that quantity has been evaluated. On an early
failure, preserve the available diagnostic and mark unevaluated quantities
unavailable; do not report zero dropped as if the policy completed. Previously
saved reductions without this diagnostic remain historical results without
that measurement; do not retrofit invented counts.

The ordinary console summary should state the total newly dropped and the
three array counts, with a path to the compact detail file. The persistent
observation summary and detail file must agree. Numerical values in this
document are definitions, not outputs from an unperformed trial.

## Consequence for the 152418 proposal

The owner direction supersedes the earlier recommendation in
[the saved-evidence diagnosis](../handoff/POINT_152418_DIAGNOSIS_AND_DECISION_2026-10-28.md)
to prioritize short-stretch assessment. That method remains prior work for a
possible later recovery comparison; it is not selected or enabled here.

Channel 92/network 11 is the first motivating case: repeated disturbances,
whole-segment exclusions and an unavailable final assessment. The next bounded
design must state which of these facts justify an observation-local exclusion,
which merely record uncertainty, and how exclusion affects other detectors'
estimates. A channel-specific hard-coded exception is not the general rule.

Implementation follows the bounded work order at
`validation/configured_point_2026-10-28/DISTURBANCE_WORK_ORDER.md`. RTC owns
classification and original-input application; the project adapter owns the
compact diagnostic. Its first implementation measures RTC decision populations;
final map-contributor accounting is explicitly unavailable rather than inferred
from retained counts. That additional diagnostic remains future consumer work.
Completion and independent review are recorded separately from this owner decision.

Grant alone triggers reductions from the shell. Existing failed and successful
reductions remain unchanged.
