# EL-F13: test a pre-action benefit signal on retained data

Date: 2026-09-05

Decision candidate: `SCI-FRUIT-EL-F13-SCAN-AGREEMENT-FEASIBILITY-R0.1`

Status: owner-review proposal. No new method is approved, implemented or run.

## Recommended decision

Approve one bounded local feasibility test using the already retained EL-F12
historical-control products. It would ask whether a contribution helps its
scan agree with two reference maps made from other scans, before deciding
whether to request another intervention experiment. **It authorizes zero
Citlali runs and zero interventions.**

The next experiment should address benefit, not simply lower EL-F12's response
cutoff or relax its failed protections. EL-F12 selected every eligible record
through a support/conditioning warning, while neither alternative passed the
scientific screen. Its a1400 absolute-improvement target was also unattainable
on the observed historical-control baseline. Those results remain negative.

## Concrete proposed test

For every new eligible detector/scan at historical-control boundaries 0–5:

1. Reconstruct its scan from the recorded processed samples. Form two reference
   maps from the other scans, using fixed even/odd scan groups. Exclude all
   currently eligible detector UIDs from both reference groups.
2. Compare that scan at three fixed map coefficients for the candidate: 0,
   0.5 and 1. Coefficient 0 represents deletion with processed samples held
   fixed. The other coefficients are the two possible retention probes.
3. A retention probe passes the proposed agreement test only if it improves
   agreement with **both** reference maps by at least 10% and 0.1 mJy/beam,
   with sufficient declared overlap, no lost comparison pixel and a numerical
   margin. Missing evidence produces an unavailable result, not permission
   to retain a contribution.

This uses current-iteration samples, geometry and candidate identities. It
does not use source truth, injection coordinates, later maps, an outcome from
another arm, an explicit UID list, or EL-F12's selected/risk/response values to
make the prediction. The two reference groups share feedback and processing
history; they are not independent replications or sky truth.

The [design](EL_F13_RETAINED_SCAN_AGREEMENT_DESIGN_R0.1.md) fixes the domain,
thresholds, arithmetic, tests, input separation, reporting and resource limits.
These are new proposed choices, not calibrated values established by EL-F12.

## What the result could establish

The useful result is a complete feasibility record: does the proposed signal
have enough usable support, pass its construction tests, and avoid recommending
the first rescues that already failed EL-F12's protections? That last comparison
is an exposed development challenge. The two failed alternatives at the first
action share one observation and opportunity; they are not two independent
negative examples. Later failures cannot be assigned to individual decisions.

Synthetic positive examples are required so that an always-veto rule cannot
look useful. A deliberately misleading common reference is also required to
show why agreement is not truth. Passing these checks can support preparation
of a new intervention proposal only. It does not establish empirical benefit,
an operational selector, a policy, or eligibility for independent replication.

The recorded samples cannot reproduce the changes to cleaning, masks, weights
and feedback caused by a future pre-cleaning action. Those consequences would
still need a separately approved end-to-end experiment. No choice between
Half and Hold, joint action policy or action duration is made by this test.

## Boundaries of approval

Approval covers an isolated analysis helper, its focused tests, exact helper
registration, and six historical uninjected boundary evaluations using the
bound retained files. It includes one report of both fixed retention probes
and comparison with the already published EL-F12 outcomes after predictions
are frozen. No tuning, new observation, raw-data reduction, restart, native
Citlali change, or additional outcome search is included.

Limits are one CPU thread, two hours aggregate including tests and repairs,
4 GiB peak RSS per process and 4 GiB new output. Original spools stream from
their verified compressed copies. Existing products stay intact. Routine
implementation repairs follow the standing direction; scientific changes
return to the owner.

The next actual intervention proposal would still have to bind an attainable
benefit endpoint under the historical recurrence and preserve source recovery,
morphology, leakage, support, useful-exclusion, convergence and resource
protections. This proposal changes no EL-F12 verdict or end-to-end gate.
Independent-pointing replication, full Gate D, qualification, Stage B,
production and Unity remain separate decisions.

## Owner decision

Recommended: approve this exact feasibility-only package against
[its manifest](EL_F13_BUNDLE_MANIFEST_R0.1.md). Alternatively, revise this new
scientific design or stop this development direction. Approval does not
automatically authorize the experiment that might follow it.

The request to continue has authorized preparation of this concrete proposal.
It has not selected or approved the new scan-agreement method described here.
