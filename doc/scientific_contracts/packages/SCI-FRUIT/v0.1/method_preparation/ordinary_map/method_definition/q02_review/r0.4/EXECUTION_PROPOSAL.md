# Next bounded execution and T2 design decision

2026-09-09. Owner review under existing Q02/Q05/Q06. **Neither decision below
has been approved.** The [current disposition](README.md) and its program/
prior-work recovery apply. A/B and all frozen scientific authorities remain
unchanged. No numerical coefficient family is adopted as a policy.

## Decision 1: run the synthetic T1 screen on its own

Recommend authorizing implementation, verification and one complete local T1
campaign, using exactly the T1 input/execution envelope, source/noise laws,
estimands, limits and interval prescription in
[FIRST_SCREEN_BINDINGS.md r0.3](../r0.3/FIRST_SCREEN_BINDINGS.md), with the
clarifications below. This explicitly splits the earlier proposed T1+T2 first
execution unit. It does not authorize real-data signal access or T2 mapping.

| Item | Exact proposed scope |
| --- | --- |
| Input authority | Only the abstract, arbitrary-unit synthetic rows in r0.3; no CAL/PTC product or physical-sky population admission. |
| Cases | W00, W01a/b, W02, W03a/b, W04a/b, W05 in their exact declared order. One 9x9 grid, 16 detectors and the declared visit/source/quality laws. |
| Weight laws | U=1 and the one empirical centered-scatter N law. Training has 256 prior samples, minimum 64, outside the declared fixed guard. No cap, fallback or detector pruning. |
| Trial count | 1,024 per case, paired null/signal and U/N: 36,864 map evaluations, plus the listed deterministic checks. No additional stochastic cases or outcome-dependent extension. |
| Randomness | The exact r0.3 PCG64/SeedSequence identities and draw ordering. Record the actual NumPy version and complete regeneration identity. |
| Mapping | Containing half-open unit cells, no interpolation/extrapolation. Per-array occurrence-mean-one coefficient normalization. The exact frozen MAP threshold order statistic with c=0.1 explicitly admitted for this synthetic test only; science rows require S_sci, normalization rows require S_norm. Retain both row sets. |
| Main tolerances | At most 5% excess uniform RMS; each arm's amplitude bias within 2%; centroid within 0.02w; width within 2%; gross-excursion difference at most 0.01; required support preserved. Exact comparisons and multiplicity handling remain r0.3. |
| Resources | One worker and one numerical-library thread, at most 2 hours, 4 GiB peak aggregate process-tree memory and 2 GiB output. Local /Users/gwilson/tolteca/bin/python. No Unity. |
| Run identity | Implement only after approval. Complete the declared arithmetic/failure checks, record environment and harness digest before the stochastic run, and preserve source/settings/output hashes. This does not require another vote for routine implementation details or authorized defect repair. |

There is no adjustable mode, grid, real observation, stopping criterion or
extra coefficient family hidden in this permission. The approved scientific
formulas are used conditionally as arithmetic; this is not Citlali conformity
or numerical upstream adoption. Historical JINC products are not rerun or
replaced, and no FRUIT operation enters this campaign.

### Completion and failure rules

The campaign reports each case as pass, fail or inconclusive under its exact
criteria. An expected scientific failure in a stress case does not justify
dropping later predeclared rows or changing the setup. A missing required
science pixel makes that row's decision unavailable; it does not disappear
from the denominator. Unexpected arithmetic, analytic-benchmark or required
output failure blocks scientific interpretation. Resource exceedance stops
with partial products and an explicit incomplete campaign.

For the equal/independent Gaussian U analytic check, use the two one-sided
Student-t bounds specified in r0.3 at 0.05/256 each; the analytic expected
squared RMS must lie inside their interval. For deterministic arithmetic,
require exact support/placement and equal-arm identities; for noninteger
floating arithmetic use rtol=1e-12 and atol=1e-14 in the declared test units.
These are proposed verification tolerances, not scientific loss limits.
Width means the positive square root of the stated second central moment.
The intended bound count is 215: per case, one noise contrast, four amplitude
tails, eight centroid tails, eight width tails and two excursion-proportion
bounds (23 times nine = 207), plus two analytic-check tails for each of W00,
W01a, W01b and W02 (eight). Other reported diagnostics remain descriptive;
support and cost are direct gates. Retain every metric definition and bound;
any changed decision set returns for review rather than consuming unused
allocation. No ad hoc reallocation is permitted.

Retain all small maps, input regeneration identities, coefficients, training
scatter, native/common supports, per-case metrics/failures, elapsed time and
peak memory. Keep coefficient-estimation cost separate from mapping. A routine
implementation bug may be repaired within the approved method, gates,
population, inputs, bounds and scope, retaining both attempts. A change to any
of those decisions returns to the owner.

T1's outcome can check the method and expose regimes where uniform weights
lose information. It cannot decide whether actual TolTEC detector noise is in
an acceptable regime, establish physical flux recovery or support a pointing
policy recommendation. No FRUIT convergence claim is available from T1.

## Decision 2: revise T2 training on paper before execution

The current per-detector/per-chunk first-half rule fails the feasibility check
in [the input report](INPUT_PREFLIGHT.md). Recommend authorizing a bounded
paper revision with **one pooled bootstrap training set per detector over the
observation**, while preserving the first-half/evaluation separation, source
guard, current flags and minimum 64. This is a different coefficient identity
from the proposed per-chunk family; it is not a routine bug fix.

For detector d, propose J_d as the union of its eligible source-excluded
first-half samples across all twelve recovered chunks. Evaluation remains
the eligible second-half samples. Compute:

```text
mean_d  = sum(J_d values) / |J_d|
v_d     = sum((J_d values - mean_d)^2) / |J_d|
w_i     = 1 / v_d(i)
gamma_i = w_i / mean(w over the exact evaluation population in array a(i))
```

Every training sample must already exist before the first weighted-map action.
This is causal for the offline bootstrap action; it is not an online estimator
available at each individual evaluation sample's acquisition time. No map,
injected truth or held-out outcome chooses J_d or the guard. Each realization
would estimate its own generation, which then remains fixed for its maps.

Pooling may make training more usable and retain the detector comparison, but
it loses chunk-specific adaptation to changing noise. That tradeoff must be
approved explicitly. Pooled feasibility has **not** been evaluated, and its
success is not promised. No lower threshold, fallback U, mixed estimator or
detector pruning is proposed. Required unavailable training still stops T2.

Approval of this decision would authorize only finishing the revised T2 paper
definition and its input/geometry feasibility checks. Before any training
scatter, real-data map or injection is computed, that successor must bind:

- exact experimental input/adapter permission, including recovered segments,
  quantity/beam/reference/frame limits and finite-value/retention predicates;
- exact grid, science/evaluation regions, support policy and coefficient
  population, chosen without weighted outcomes;
- the fixed discovery noise-contrast rule and all cases/amplitudes/counts;
- a scientifically justified real-data noise/response and uncertainty design,
  with its conditioning and limitations, and owner-approved loss margins;
- execution resources, outputs, stop rules and permission.

One real map's spatial RMS cannot acquire T1's independent-trial intervals.
Shared PTC fits, atmosphere, calibration and coefficient estimation remain
relevant. If no adequate uncertainty design is supplied, the correct T2
outcome is insufficient evidence for an acceptability decision. Do not invent
a NOI family, covariance, bootstrap independence or physical-flux guarantee.

129081's scientific values stay reserved pending the later replication access
decision. Independent-pointing replication precedes a pointing policy
recommendation. T3 feedback still requires a separate continuation decision.
This proposal selects neither U nor N and leaves Q02-C open.
