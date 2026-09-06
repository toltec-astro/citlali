# EL-F14: fixed synthetic design for separating sky and shared time variation

Date: 2026-09-06. Status: proposed; no implementation or execution.

Decision: `SCI-FRUIT-EL-F14-SKY-TIME-FEASIBILITY-R0.1`.

## Question and scope

Can simultaneous donor measurements constrain sky contrasts tightly enough
to certify that retaining some of a suspect contribution improves those
contrasts? Test one explicit additive model, including its ambiguities. This
is a deterministic synthetic feasibility test, with no population inference,
random draws, real observation, actual hard-action execution or tuning.

Use one artificial array, one scan, three time indices (0, 1, 2), three sky
cells (A, B, C) and three detector UIDs (11, 22, 33). UID 33 is the sole probe
candidate. Its eligibility is supplied as pre-action state, not inferred from
truth. The complete eligible set is {33}, except in S11. An eventual real
candidate census is outside this test. All rows represent already available
measurements before a hypothetical subsequent action. There is no recurrence
or later iteration in the toy model.

## Inputs and separation of prediction from truth

The predictor receives only rows (array, scan, UID, time, sky cell, measured
value), candidate UID, complete eligible UID set, the fixed cell domain and
the declared error bound epsilon. All weights are one; all sky sampling and
map deposition are one-cell indicator operators. Signal uses arbitrary toy
units, never mJy/beam. Time indices have exact common-time meaning by fixture
construction. No candidate truth, nuisance realization, expected label,
future outcome or real UID is a predictor input.

The generator and report may access the declared truth, but the predictor
interface and its process may not. Materialize separate input and truth
files after approval, register both plus helper hashes before evaluation,
freeze all 17 predictor outputs, then join truth and expected outcomes in the
report. This is dependency control on deliberately authored examples, not
blinding, independence or discovery from a representative population.

Missing or non-finite required data, duplicate (array, scan, UID, time) rows,
unknown cells, negative epsilon or a candidate absent from the eligible set
produce `invalid_input`. Scientific missing-support and geometry results are
distinct from malformed input. Omitted rows in S09 are intentional absence,
not non-finite observations. No sample imputation or time interpolation.

## Proposed sky constraint

For donor rows only, assume

    x_i = theta[cell_i] + a[time_i] + e_i,   |e_i| <= epsilon.

Here theta is the common sky, a is an unrestricted additive value shared by
all donors at that time, and epsilon is a known synthetic bound. No
probabilistic noise independence or fitted error bound is assumed. Candidate
measurements can have arbitrary error; they do not constrain theta. Remove
every eligible UID from the donor set before making constraints, including
eligible UIDs other than the probe candidate.

For every unordered pair of different donor UIDs at the same array, scan and
time, require

    |theta[cell_i] - theta[cell_j] - (x_i - x_j)| <= 2*epsilon.

Keep same-cell constraints too: they can expose inconsistent measurements.
These pairwise constraints are equivalent to the existence of a shared a and
bounded donor residuals at each time. Different-time pairs are never formed.
The graph whose three cells are vertices and whose different-cell donor
pairs are edges must be connected. Otherwise return `unavailable_geometry`
for both probes, even if a particular functional might still be identifiable.

Only contrasts are identified. Set theta[C]=0 as a coordinate convention and
remove the mean from every map and truth before comparison. No estimate of
absolute sky level is claimed. With connected geometry, finite epsilon bounds
the resulting two-dimensional feasible set F. If F is empty, return
`model_mismatch`. Do not enlarge epsilon or discard inconsistent pairs.

Use exact rational arithmetic throughout these tiny examples. Intersect every
pair of nonparallel constraint boundaries in the two coordinates, retaining
all feasible intersections. They contain the extrema of a linear functional
on this bounded polytope, including point and line degeneracies at epsilon=0.
Check connectivity first; an empty vertex set then means an empty F. No
floating tolerance, optimizer choice or regularization supplies missing sky
information. Unexpected arithmetic or certificate failures are execution
failures, never scientific abstentions.

## Three fixed map probes and benefit interval

For f in {0, 1/2, 1}, give candidate rows coefficient f and all other rows
coefficient 1, including other eligible UIDs. At each cell accumulate directly

    N_f = sum(coefficient_i * x_i)
    C_f = sum(coefficient_i)
    Q_f = sum(coefficient_i^2)
    M_f = N_f / C_f.

Record N, C, Q, row count and unique contributing UID count for every cell
and probe. Positive coefficients count as contributing. All three cells must
have C_f>0 and Q_f>0 at all three f values. If any lacks support, both probes
are `unavailable_support`; never shrink the fixed domain. Validation order is
input, support, geometry, feasibility, benefit. Report all observed support
facts even when an earlier validation prevents scoring.

The f=0 map is the deletion proxy; f=1/2 and f=1 are fixed retention probes.
These coefficient probes hold samples fixed. They do not reproduce the
historical pre-cleaning hard action or changes to RTC/PTC, masks or feedback.
This toy uses positive unit deposition, not native signed JINC interpolation.
No inference about native support/conditioning follows from it.

Let H subtract the three-cell mean. Clean fully admitted map response in this
specific one-cell model is theta. Define

    L_f(theta) = ||H(M_f - theta)||^2 / 3
    B_f(theta) = L_0(theta) - L_f(theta).

B is affine in theta because the quadratic truth terms cancel. Evaluate its
minimum and maximum over all feasible vertices to obtain [B_low, B_high].
This is a deterministic range under the model, not a confidence interval.
Classify each of f=1/2 and f=1 using the fixed delta=1/4 toy units squared:

- `conditional_benefit` if B_low >= delta;
- `no_certified_benefit` if B_high < delta;
- `inconclusive` otherwise.

Report both intervals and labels. Never choose a winner, combine probes,
normalize by a vanishing baseline error or adjust delta after seeing outputs.
Mean removal defines a new toy contrast estimand. It cannot replace absolute
flux, source recovery, morphology or leakage protections in an intervention.

## Exact fixtures

Base sky truth is theta=(6,0,0) in A/B/C order, epsilon=0, with shared nuisance
a=(6,-6,0) in time order. The following nine predictor rows are the complete
S01 input; scan and array are both zero. Each listed row has unit weight.

| Time | UID | Cell | Measured value |
| --- | --- | --- | --- |
| 0 | 11 | A | 12 |
| 0 | 22 | B | 6 |
| 0 | 33 | C | 6 |
| 1 | 11 | B | -6 |
| 1 | 22 | C | -6 |
| 1 | 33 | A | 0 |
| 2 | 11 | C | 0 |
| 2 | 22 | A | 6 |
| 2 | 33 | B | 0 |

Each fixture starts afresh from S01 and applies only its listed changes. Row
identities below are (time, UID). Unless specified, evaluation truth remains
(6,0,0). Recomputing a value means theta[cell]+a[time], with no random noise.

| ID | Exact change | Required outcome |
| --- | --- | --- |
| S01 | None | Both probes conditional_benefit; exact B_half=126/25, B_full=6 |
| S02 | Candidate (0,33)=-12 and (1,33)=18 | Both no_certified_benefit; B_half=-234/25, B_full=-18 |
| S03 | Candidate (0,33)=15 and (1,33)=-9 | Half conditional_benefit with B=144/25; full no_certified_benefit with B=0 |
| S04 | Set a=(0,0,0); recompute all nine values | Both no_certified_benefit, B=0 |
| S05 | epsilon=1/4; add 1/4 to each UID 11 value and subtract 1/4 from each UID 22 value | Nonempty F; both computed intervals enclose the corresponding true B; report labels without an additional required sign |
| S06 | epsilon=12; observations unchanged | Both inconclusive; F admits worlds with opposite benefit signs |
| S07 | Donor cells at t=1 become A/B for UIDs 11/22; at t=2 become C/C; recompute donor values | unavailable_geometry; all map probes retain support |
| S08 | Add 1 to (0,11) | model_mismatch from inconsistent cycle, with no discarded pair |
| S09 | Remove donor rows (1,22) and (2,11) | unavailable_support; deletion has no C support; no intersection-domain repair |
| S10 | Replace time of (1,22) with missing/null | invalid_input; no inferred synchronization |
| S11 | Eligible set becomes {22,33}; rows unchanged | unavailable_geometry after excluding both eligible UIDs from inference; UID 22 remains in map probes |
| S12 | Multiply measured value (0,11) by 2 | model_mismatch for this specific gain-error example; no general gain-error detector claim |
| S13 | Predictor input exactly S01; provide two evaluation worlds: W1=S01, W2 theta=(12,0,-6) with added sky-shaped artifact h=(-6,0,6) on every row | Predictor identical to S01. Full-retention true B is +6 in W1 and -18 in W2; W2 violates the donor model. Mandatory demonstration that unconditional safety is not identified |

S01–S04 expected numbers are algebraic fixture specifications, not measured
results from an executed helper. S13 is one predictor evaluation with two
evaluation-only truth records, not two independent samples. All S13 rows in
W2 obey x=theta[cell]+h[cell]+a[time], which gives the same observed values
as W1. The reporter must show this equality and the opposite true signs.

Four additional predictor evaluations use S01 with these fixed transformations:

1. Replace UIDs 11/22/33 by 111/222/333 everywhere, including eligibility.
2. Reverse all nine input rows.
3. Add 100 to all measured values and to all evaluation sky cells.
4. Repeat S01 in a predictor process with evaluation truth and future-outcome
   canary files outside its permitted input directory; the harness records
   attempted accesses and fails any access to those files.

The first three must preserve exact benefit intervals, labels and support
counts after undoing identity/order transformations. The fourth must reproduce
S01 with zero forbidden reads. Implement the small helper with no file reads
after loading its allowed input; test its loader through an explicit read
allowlist. This is a dependency audit, not a security claim. Unknown truth or
outcome fields in the predictor schema must be rejected in focused unit tests.
Changing truth alone must never change a frozen prediction.

## Gates, execution and complete reporting

There are exactly 17 primary predictor evaluations. Focused construction
tests and routine repair reruns may repeat these fixed cases within the
resource limit; they may not add a new scientific fixture or parameter sweep.
Register the materialized fixtures, helper, test entry points, environment,
run command and output destination before the first primary evaluation.
Use a new isolated local output directory, never an old reduction directory.

Implementation gates require exact reproduction of all mandated outcomes,
correct min/max bounds on the defined feasible sets, no lost required cell,
truth separation and complete case accounting. An always-abstain helper fails
S01 and S03. S13's failure of an unconditional claim is mandatory evidence,
not an exception to omit from the success count. Report each case separately;
do not quote an accuracy, benefit rate, false-action rate or p-value.

Use one CPU thread with a 30-minute aggregate budget including focused tests
and routine repairs, 1 GiB peak RSS per process and 100 MiB new output. A
supervisor records elapsed time and RSS and terminates on a resource breach;
report any observed overshoot. No parallel worker or Citlali build/run is
included. Preserve all failed attempts. A method/input/gate/scope change or
exhausted budget stops execution for an owner decision under the standing
routine-repair direction.

The final report records every input/helper hash, exact interval, true loss,
support tuple, missing or infeasible result, integrity check, resource use
and failed attempt. If all gates pass, the strongest permitted conclusion is
conditional identifiability in the declared synthetic model. Field
applicability remains unestablished and an unconditional safeguard remains
unidentified. No automatic real-data score, recurrence experiment, independent
pointing, policy, qualification or Stage B follows.
