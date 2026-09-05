# EL-F13: retained scan-agreement feasibility design

Date: 2026-09-05. Decision: `SCI-FRUIT-EL-F13-SCAN-AGREEMENT-FEASIBILITY-R0.1`.
Status: proposed method; implementation and measurements await owner approval.

## Program adherence and prior-work recovery

This continues the existing empirical lane under the
[program charter](../../../../README.md),
[accepted inputs](ACCEPTED_INPUT_BINDINGS_R0.1.md), and
[gate architecture](EMPIRICAL_LANE_GATE_ARCHITECTURE_R0.1.md).
The [recovery and input record](EL_F13_INPUT_AND_PRIOR_WORK_R0.1.md) identifies
the adopted boundaries, negative evidence and genuinely new question. This is
implementation-informed empirical preparation, not a new scientific contract
or Stage B derivation. The packet stays outside the Stage B author channel.

## Question, population and clock

Does a pre-action scan-agreement measure supply a usable, directional signal
for retaining a contribution, beyond knowing that its deletion has an effect?
The hypothesis is deliberately limited to a predictor with current processed
samples held fixed. It is not a forecast of the full adaptive trajectory.

Use only EL-F12 H/uninjected, observation 123424, arrays a1100/a1400/a2000,
completed absolute boundaries k=0,...,5. H followed the mandatory historical
recurrence with alpha 1 and ordinary hard actions. At each boundary enumerate
the complete existing census of newly eligible scan-local factor-zero
`map_pixel_outlier_detector_dominance` proposals from `mapdiag:`. The key is
observation, array, UID and zero-based scan. Preserve excluded dispositions.
Keep all eligible keys; never select them by EL-F12's recorded score or flag.
Limit 16 eligible keys per boundary, with no truncation if exceeded.

At boundary k the predictor may read only that boundary's bound H/uninjected
spool, occurrence ledger, checkpoint and same-grid maps. Candidate identity
and eligibility are decoded from state; its EL-F12 selection, response and
risk fields are discarded before the predictor is called. No incoming state
is altered. There is no action at k or k+1 and no hypothetical policy rollout.
Predictions are evaluated independently on H's actual history, not on a
counterfactual history that incorporates earlier predictions.

## Scan construction and reference groups

Read the entire original spool in recorded order. Validate its decompressed
hash, 104-byte schema, endianness, scan markers, occurrence identities, kernel
indices, multipliers, counts and EOF. Reconstruct signed numerator N,
denominator C, quadratic Q, absolute term sums and occurrence/unique-detector
counts per scan and array. Use the recorded signed JINC bank, squared bank,
pixel origin and clipped footprint. Preserve occurrence order within scans
and recorded scan order when combining them. No normalized-map summation,
nearest-pixel approximation, RTC/PTC replay, or noise-pass duplication.

First sum every scan back to the recorded full-array N/C/Q and reproduce the
retained signal and formal coefficient planes under the existing finalizer.
Require the EL-F12 accounting and forward-error checks before forming any new
partition. A construction mismatch invalidates the job; it is not a scientific
score. H's stored multipliers must all equal 1.

For key t in scan s, form its target-scan totals from all ordinary admitted
occurrences in s. The current eligible UID set in that array is U. Reference
group 0 contains the other even-indexed scans and group 1 the other odd-indexed
scans; exclude s from both. Indices are the recorded zero-based science-scan
identities, never file order or an outcome-based split. Require the recorded
science-scan identities 0–11 exactly. Remove every UID in U
from both groups, across all their scans. Do not remove other detectors or
scan regions after examining their maps. Require at least two distinct
contributing scans in each group for that array, otherwise unavailable.

These references are separately formed but statistically dependent: earlier
feedback, shared cleaning and common contamination remain. There is no
independence assumption, error bar, p-value or effective-replication count.

## The three fixed coefficient probes

Partition target-scan occurrences into t and its complement B. For
f in {0, 0.5, 1}, accumulate directly in original occurrence order:

    N_s(f) = N_B + f N_t
    C_s(f) = C_B + f C_t
    Q_s(f) = Q_B + f² Q_t

These equations define the contributions; direct accumulation avoids treating
a normalized map as an additive input. Counts and absolute sums follow the
actual f-weighted terms; retain original occurrence counts separately. Apply
f once. Reference maps always use ordinary coefficient 1 for admitted terms.
No scalar learned-penalty, cleaning, source model or kernel fit is altered.

For every constructed map, use the existing EL-F12 finalizer: finite N/C/Q,
abs(C)>1e-8, positive Q, coefficient weight C²/max(Q,1e-30), its provisional
coverage cut, then the unchanged full science coverage cut 0.1. This coefficient
support is descriptive, not a new physical inverse-variance claim. Also require
the inherited numerical denominator/Q margins, binary64 unit roundoff 2^-53
and finalization/identity safety factors 16/64. Empty or unconditioned required
support is unavailable. Preserve the exact WCS, unit and pixels; no smoothing,
background subtraction, fit, alignment, clipping or scale adjustment is allowed.

Coefficient 0 is a fixed-processed-sample deletion comparator, not the actual
historical pre-RTC/PTC action. Coefficients 0.5 and 1 likewise do not reproduce
the Half/Hold trajectories. The missing operator changes are a material
limitation of the proposed predictor.

## Comparison domain and proposed agreement condition

Let F be t's nonzero absolute-coefficient footprint on the completed full
historical map's ordinary science support. Freeze V as the intersection of
F and the conditioned science supports of M_s(0), R_0 and R_1, also requiring
at least two contributing scan identities per pixel in each reference. Use the same V
for both references and both retention probes. Require at least 256 pixels
and |V|/|F| >= 0.90. Empty F is `no_contribution`; inadequate overlap is
`unavailable_overlap`. Report every pixel outside V and why it was lost.

For f=0.5 and 1 separately, require that M_s(f) is conditioned and supported
on every pixel of V. A lost V pixel makes that probe unavailable. Report all
support changes on the union, including changes outside V. The 90% threshold
is a proposed limit on the observability of this predictor, not permission
for a later intervention to lose even one H comparison-domain pixel.

For g in {0,1}, compute the equal-area RMS agreement error in mJy/beam:

    E_g(f) = sqrt(mean over V of (M_s(f) - R_g)²).

Retain signed difference maps, positive/negative peaks and sums, and both
errors; do not average the two reference decisions or combine the two probes.
A retention probe passes only when both references satisfy:

    E_g(f) <= 0.90 E_g(0)
    E_g(0) - E_g(f) >= 0.1 mJy/beam.

Evaluate these inequalities conservatively with propagated roundoff intervals:
use the upper endpoint for E_g(f) and lower endpoint for E_g(0). The proposed
fixed guard treats recorded doubles as the inputs; it is not sky uncertainty.
For each pixel let n be its contributing pixel-term count, s its contributing
scan count, u=2^-53 and gamma(j)=j*u/(1-j*u). Stop if j*u>=1. Set

    b_X = 16 gamma(4*n + 4*s + 16) A_X, for X=N,C,Q,

where A_X is the sum of absolute contributing terms, conservatively rounded
upward. Require abs(C)-b_C>1e-8 and Q-b_Q>0. For M=N/C use

    b_M = (b_N + abs(M)*b_C)/(abs(C)-b_C) + 16*u*abs(M).

With v=|V|, bound the RMS error by

    b_E = RMS_V(b_M + b_R)
          + 16 gamma(4*v + 16) RMS_V(abs(M) + abs(R)).

Use stable scaled RMS evaluation, outward-rounded interval endpoints
[max(0,E-b_E), E+b_E], and retain all terms. Test these arithmetic guards
against higher-precision analytic constructions, threshold boundaries and
signed cancellation before data access. This is an explicit proposed safety
construction. Repair ordinary coding bugs under the standing direction. If the
mathematical guard itself is insufficient, return to the owner rather than
adjusting its factor during scoring. Equality of errors gives no gain. If E_g(0)'s upper
bound is below 0.1, mark that reference `insufficient_room`; never reduce the
floor to produce a pass.

These 10%, 0.1 mJy/beam, 256-pixel and 90% choices are proposed heuristics.
They have not been measured or tuned on this new score. No threshold sweep,
alternative partition, source mask or region search is included. EL-F12's
support-risk flag supplies neither a positive nor a negative vote here.
It remains reported historical evidence outside the predictor.

This yields two probe dispositions per eligible key, not a deployable action
choice. There is no winner rule, joint intervention composition or duration.
Multiple eligible keys must all be measured and reported. Synthetic multi-key
coverage tests reference exclusion and accounting only; a future action policy
would require its own joint-effect and precedence design.

## Required tests and exposed-result comparison

Before processing any retained spool, pass the following fixed construction
tests. Use analytic maps and exact, declared coefficients, not random searches.
The tests exercise all three arrays, signed/clipped/subpixel JINC placement,
noninteger weights, f² propagation, scan splitting and source gradients.

| Test | Required behavior |
| --- | --- |
| Clean target, biased complement; both references at the known clean sky | Retention can pass; reject an always-veto implementation |
| Corrupted target, clean complement and references | Neither retention probe passes; influence alone does not establish benefit |
| References disagree about a probe | A favorable average cannot pass it |
| Zero effect, exact tie or baseline below the absolute-improvement floor | No positive result; explicit no-gain/insufficient-room status |
| Missing reference scan, insufficient overlap, unconditioned denominator, lost V pixel, corrupt spool or nonfinite input | Explicit unavailable/invalid result, never silent omission |
| Two eligible UIDs in different scans | Both are excluded from both reference groups; every key remains in the census |
| Consistent UID renumbering, changed file-discovery order, poisoned unused EL-F12 scores and future/injection metadata | Same identities up to renumbering and identical decisions; forbidden data are never dependencies |
| Both references share a deliberately wrong sky feature | Retain the expected misleading agreement as a demonstrated limitation, not a successful truth test |

For the two clean-reference analytic examples, include constant planes with
equal target/complement coefficients: sky/reference 0, complement 10 and target
0 for beneficial retention; complement 0 and target 10 for useful deletion.
Replicate over 32×32 valid pixels in each reference, then exercise the same
relations with the declared JINC geometry tests. The common-contamination
example has true sky 0, complement 0 and target/reference 10: a passing score
there demonstrates non-identifiability of truth from agreement alone. It must
never be advertised as a safeguard. The exact deterministic fixture bytes and
expected arithmetic are registered before real-data scoring.

Process boundaries 0–5 in order. Freeze and hash all six complete prediction
records, including empty censuses, before the reporting process may join them
to the bound EL-F12 Half/Hold outcome tables. The authors have already seen
these outcomes; this separation controls data dependencies, not outcome
blinding or qualification. Neither injected products nor later historical
maps can be opened by a boundary predictor.

The primary exposed negative challenge is the first-action contrast: H and
both alternatives are bitwise identical through 1; their first action is at 2
and both alternatives then fail protections. Report whether each coefficient
probe at boundary 1 would permit its corresponding rescue. Count this as one
shared opportunity, not two independent cases. Boundary 2 is reported in full
but receives no isolated per-key harm label: prior interventions confound later
outcomes. No false-action rate, beneficial empirical case or causal action
effect is inferred from these two boundaries.

## Terminal record and hard limits

Always retain the full census, reference scan membership, excluded UID set,
N/C/Q accounting, support/gain/loss masks, agreement and signed-difference maps,
numerical intervals, both probe dispositions, access log, tests and resource
receipts. No empirical object is deleted because it is unfavorable.

Report separately construction validity, usable coverage and the exposed
negative challenge. If either first-action probe passes, record
`negative_challenge_failed` for that probe; if it is unavailable, record
`negative_challenge_unassessed`, not a correct veto. A valid no-gain result is
`negative_challenge_avoided`. Even avoiding both supplies only limited
feasibility evidence, never empirical benefit or permission to intervene.
If no real opportunity is scoreable, conclude `insufficient_evidence`.

Approval covers only an isolated local analysis helper and tests, registration,
six boundary evaluations and one paired report. Exact helper source, executable
if needed, dependencies, fixtures, numerical intervals and input hashes must
be frozen before scoring. Do not alter or invoke the Citlali executable or its
run controller. Zero reductions, replays, restarts and new observations.

One CPU thread; 2 hours aggregate from helper implementation/testing through
analysis and retention; 4 GiB RSS per process; 4 GiB new output. Stream gzip
inputs without retaining decompressed copies. New output, only after approval:
`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f13-scan-agreement-feasibility-r0.1`.
Routine parser/analysis repairs may repeat failed analysis within those limits,
with old attempts retained and no scientific change. No tuning or favorable
retry. A scope, input, method, gate, threshold or resource change stops for the
owner. A failed or inconclusive test is a complete legitimate outcome.

Before any later intervention, return a concrete new owner-review design that
addresses attainable benefit under the historical control, joint actions,
duration, ordinary caps/reason precedence, all original science protections,
exact paired inputs and independent-pointing replication before policy. No
endpoint or protected population is selected by this feasibility approval.
