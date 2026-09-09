# Proposed bounded test of uniform occurrence weighting

Revision r0.2, 2026-09-09. Design for Q02-C owner review; no tests have run.
The owner authorized construction of tests, approved A and B's substance,
and left C open. The numerical candidate and case design here remain proposals.

## Program adherence and prior-work recovery

Use the [governing disposition and charter routing](README.md),
[frozen method definition](../../r0.4/METHOD_DEFINITION.md) and
[existing experiment/gate record](../../r0.4/EXPERIMENTS_AND_OPEN_DECISIONS.md).
Reuse frozen [PTC requirements](../../r0.4/inputs/manager/ptc/src/common/requirements.tex),
especially REQ-052–060 and REQ-099, and frozen
[MAP equations](../../r0.4/inputs/authority/map/src/common/equations.tex) and
[requirements](../../r0.4/inputs/authority/map/src/common/requirements.tex).
This is a manager experiment proposal, not a new scientific core or an admitted
coefficient/uncertainty family. No frozen source is changed.

## Question and comparison boundary

Within a declared detector-quality and noise regime, does uniform occurrence
weighting keep noise, source recovery and other required science losses within
acceptable margins relative to one practical noise-aware alternative?
Uniform weighting need not be optimal to be acceptable. Failure to find a
statistically clear difference is insufficient evidence of acceptability.

Compare **two candidate policies only**, U and N below. Hold flags, admitted
occurrences, coordinates, quantity/calibration, map placement and numerical
support policy fixed across them. A detector with more occurrences can have
more influence under U. Equal occurrence weighting does not equalize detectors
or observing time. The test must measure that imbalance.

Keep the historical Citlali control with its actual weighting and JINC behavior.
It remains the mandatory context for an eventual FRUIT comparison. U versus N
on the same ordinary-MAP base isolates the proposed weighting change. A change
relative to the historical control can involve several method choices and
cannot automatically be credited to weights or the selector. The recovered
[manager dossier](../../r0.4/inputs/manager/approved_scope/INTERNAL_DOSSIER.md)
reports a template weight label `validated`; that software label does not
identify a scientifically admitted coefficient estimator. It is not adopted
as the alternative here. No new UID 4460 explanation is needed for this design.

## Two fixed coefficient policies

**U — uniform occurrence reference.** Retain r0.1's proposed dimensionless
gamma_i = 1 on the exact bootstrap coefficient population I0. Keep that value
generation fixed while checking current compatibility, retention and QC at
each pass. This is a gridding coefficient, not noise variance or precision.

**N — inverse bootstrap residual scatter, proposed comparator.** Use one
coefficient per detector and declared PTC segment. From the common unmodified
bootstrap PTC result b, select a predeclared training subset J_dt for detector d
and segment t, using the same valid calibrated quantity and retained-use facts.
Training uses an exact source-exclusion region declared from prior context
and geometry before comparison, with predeclared evaluation occurrences
excluded. Neither injected truth nor the tested map's bright pixels may define
that region. Suspected source contamination is tested below, not assumed absent.

For n_dt training samples, propose the empirical centered second moment:

```text
mean_dt = sum_{j in J_dt} b_j / n_dt
v_dt    = sum_{j in J_dt} (b_j - mean_dt)^2 / n_dt
w_i     = 1 / v_{d(i),t(i)}
gamma_i = w_i / mean_{j in I0,a(i)} w_j
```

I0,a is the declared bootstrap coefficient population in the same array as i;
no cross-array normalization or fitting is introduced. v has the squared units
of the actual calibrated quantity; gamma is dimensionless and has occurrence
mean one on that population. Centering here defines a weight statistic and
does not introduce a second operation on the delivered PTC signal. This simple
scatter estimator is a test candidate; correlated samples, imperfect source
exclusion and nonstationarity can make it a poor noise proxy. It carries no
unbiased-variance, inverse-covariance, probability or precision claim.

Estimate once before the first weighted bootstrap map/model action. Freeze
values through all L passes; fresh current QC/retained-use checks remain
required. In each synthetic realization estimate from that realization's
available bootstrap, never from its known generating variance. The generating
variance may enter the diagnostic benchmark only. Repeated estimation at later
iterations would change the method and is outside this screen.

Bind actual segment/training populations, source guard, minimum sample count,
finite-value policy and exact coefficient/QC identity before running. Require
finite, strictly positive v and valid finite normalized coefficients for every
required occurrence. Insufficient training or failed current QC makes that
comparison unavailable. Do not silently substitute U, clip/cap values, or drop
detectors from only N. No new detector veto, flagging algorithm or rank change
is part of this comparison. Both arms obey the same existing hard invalidity
rules; weighting does not make an invalid detector scientifically admissible.

Disjoint training and evaluation occurrences do not imply independence: PTC
learning, atmosphere and calibration may be shared. Retain that dependence in
the uncertainty design. This also means fixed-state uncertainty conditions on
the estimated coefficients; a full-procedure claim must include their estimation.

## T1. Elementary mapping cases and falsifiable checks

Begin with controlled mapping inputs under separately admitted test-population
permissions. Use the same pixels, source and random realization in each paired
U/N comparison. A known-noise calculation is a diagnostic benchmark, never an
eligible causal coefficient policy. The following is a small set of cases,
not a Cartesian parameter sweep.

| Case | Proposed construction | What the result must resolve |
| --- | --- | --- |
| W00 — algebra and equal noise | Constant signal; equal known independent variances; same contributing occurrences. Check exact equal coefficient vectors, then separately estimate N from finite training samples. | Equal coefficient vectors give identical N/Q maps. Estimated N need not equal U even under equal generating variance; measure the cost of estimating weights. A constant model on complete matching supports rejoins as specified under either coefficient family. |
| W01 — different detector noise | Two equally represented detector populations with generating standard-deviation ratios 1:2 and 1:4; otherwise the same source and independent noise law. These ratios are proposed toy values, not measured TolTEC conditions. | Recover the analytic variance penalty and determine whether causal N approaches the diagnostic benchmark with the declared finite training set. |
| W02 — occurrence imbalance | Repeat the 1:2 case with the noisier population supplying three times as many occurrences; preserve identical membership across arms. | Measure per-detector influence and recovery/noise losses caused by sample-count imbalance. This is a designed population change between cases, not a new exclusion within an arm. |
| W03 — correlated or changing noise | Add one declared shared low-frequency component to W01; separately change the noisy group's scale between training and evaluation. | Determine when marginal bootstrap scatter ceases to rank useful information reliably. Bind correlation/time laws and amplitudes before generation. N is not presumed optimal here. |
| W04 — poor but eligible detectors | Separately add sparse bursts and a small calibration/gain error to a predeclared detector subset that remains admitted under the test's exact quality predicates. | Measure leakage, flux bias and morphology as well as variance. Lower scatter does not establish better calibration; any hard-invalid case must be unavailable in both arms. |
| W05 — imperfect source guard | A declared source lies partly outside the assumed training guard, at a fixed offset/width/strength, with no truth-based repair of J_dt. | Quantify signal-dependent downweighting and lost recovery. A cleaner-looking map cannot pass by attenuating the source. |

One useful conditional prediction is available without running these tests.
For n occurrences of the same scalar signal with independent, zero-mean errors
of known positive variances sigma_i^2:

```text
Var(U) = sum_i sigma_i^2 / n^2
Var(known-variance benchmark) = 1 / sum_i (1 / sigma_i^2)
Var(U) / Var(benchmark) >= 1
```

Equality holds for equal variances. With two equal-count populations whose
standard deviations differ by a factor of two, the predicted variance ratio
is 25/16. This is an analytic counterexample to universal noise optimality,
not an empirical result or a prediction for N with estimated weights.
With correlated errors use a^T C a for normalized map coefficients a and the
declared covariance C; scalar inverse variance is not generally optimal.
These predictions are conditional on common signal, domains and support.

## T2. Paired maps from identical PTC output

Recommend one representative observation and one observation with a wider
pre-action detector-noise distribution for the discovery screen, plus one
reserved independent observation for replication. POINT is the suggested
starting mode, as in E01; neither mode nor observation IDs are selected here.
Choose the cases from declared quality/noise metadata before inspecting U/N
recovery outcomes. Bind a reproducible selection rule and intended noise regime
before choosing IDs. A convenient low-noise observation cannot establish the
case for a noisier population.

Reuse one exact immutable PTC output, its retained-use population, AST joins
and state in both arms. Produce ordinary maps with U and N. No new PCA fit is
needed to answer this mapping-only question. Compare measured noise structure
on the predeclared evaluation region, per-detector contribution shares, and
fixed-estimator flux, centroid and morphology on the intended science region.
Real-map differences alone do not establish which map is closer to truth.

Propose a small known-signal set on the exact admitted PTC/MAP input domain:
one null, one nominal compact source, one offset/broader source, and one source
near the planned admission threshold. Bind actual profiles, calibrated response,
sign, amplitudes, placement and realization counts in Q05/Q06. These injections
measure mapping and coefficient-estimation effects only. They cannot establish
response through PTC learning. For the clean-guard cases, injected support is
outside J_dt and N stays the same; W05 deliberately tests contamination of the
predeclared training subset without redefining it from truth. Tests entering
PTC Learn require the distinct input permission described under T3.

If U already violates a predeclared mapping/science margin in the intended
regime, stop the uniform-reference recommendation for that regime. A feedback
run is not needed to excuse the loss. If both candidates fail required absolute
science criteria, or the evidence is inconclusive, report that result and keep
C open. If U passes the bounded mapping screen, present it for the separate
decision whether the feedback comparison is worth running. Do not expand the
weight family or search for a favorable observation after seeing results.

## Support and influence accounting in every comparison

Equal admitted occurrences do not guarantee equal usable map support: changing
gamma changes Q and can change the frozen MAP support gate. Keep each arm's
actual authorized native support, denominator and cause records. Never publish
a value outside that arm's valid domain to force agreement. Each feedback arm
may establish its own fixed bootstrap D_a under the controlled-reference rule;
later inferred admission/revocation within D_a remains a different fact.

Predeclare an evaluation region using geometry/prior context independently of
the weight outcomes. Report full native supports and the common available
subset, together with coverage lost from the intended region and from each
arm. A comparison only on the intersection is insufficient. Missing required
science pixels count against the support criterion or make a metric
unavailable; they cannot disappear from the score. Report sample counts,
detector shares, sum gamma and sum gamma squared. The concentration diagnostic
(sum gamma)^2 / sum gamma squared is not a count of independent measurements
when samples are correlated. Original ALIGN exposure remains physical exposure.

## T3. Conditional continuation through feedback

Only after an explicit continuation decision and exact route/input adoption,
compare U and N through the same ordinary-MAP FRUIT recipe. Use one fixed
baseline admission rule with its actual noise/response evidence; the recovered
empirical S/N baseline is proposed, subject to Q01/Q05. Do not also introduce
the new benefit-signal selector in this experiment. Keep original-parent
residual construction, grouping/rank recipe, containing-pixel projection,
unfiltered admitted total flux, unity application, no extra floor, numerical
support policy and finite extent L fixed across arms.

Each arm relearns centering/PCA on its own current residual. Changing map
weights can change the model and therefore later PTC state; that is part of
the coupled effect being tested. Share the common bootstrap, not later fitted
subspaces. U and N remain fixed coefficient generations with current QC.
For full-procedure recovery claims, inject at the expressly admitted measured/
CAL-stage parent before residual formation and PTC Learn; recompute the common
bootstrap and causal weights for each realization. Preserve actual beam,
quantity and calibration response, and retain simulation/shared-noise lineage.
REQ-099's remaining experimental-population exclusions require exact separate
permission; A's approval alone does not supply it.

Reuse the bounded case set, adding only the predeclared positive and negative
model-error perturbations needed to test entry, revocation and stability near
the admission boundary. Bind amplitudes/counts before execution. Downweighting
a detector at MAP cannot undo its influence on the preceding PTC fit; this
stage measures that interaction. Completion at L is not convergence, a stable
wrong answer is not recovery, and support loss is not successful termination.
Use the last required completed endpoint after a successful run, preserving
all intermediate products and failures; do not select the prettiest iteration.

## Measurements and decision rule

| Required output | Predeclared comparison |
| --- | --- |
| Noise and sensitivity | Paired variance or squared-RMS ratio V_U/V_N on a fixed source-free evaluation domain, plus spatial/correlation structure and coefficient-estimation uncertainty. Bind the estimator and its response/state conditioning. Q or gamma alone supplies no noise evidence. |
| Source recovery | Bias and error of the same declared calibrated flux estimator against permitted known signal, with attenuation and false-source behavior. Absolute science tolerances also apply if both arms are similarly bad. |
| Morphology | Centroid offset, width/shape and residual structure using the selected mode's declared estimands. Do not infer integrated or extended-source flux truth from a point-source peak calibration. |
| Leakage and admission | Source-free residual structure, false model promotion, legitimate signed recovery, and incorrect retention/revocation near thresholds. Keep the truth mask evaluation-only. |
| Support and detector quality | Native/common/evaluation-region coverage, all unavailable causes, detector/segment influence, noise dispersion, calibration/quality strata and their counts. Report required missing pixels. |
| Feedback stability and convergence | Per-pass changes, signed perturbation growth/decay, residual structure, inferred support changes and exact terminal cause; distinguish numerical completion from scientific convergence. Not applicable to T1/T2's mapping-only claims. |
| Cost | Runtime and peak aggregate process-tree memory, with coefficient estimation separated from MAP and PTC/feedback work. Compare matched resource settings and preserve logs. |

Before execution, approve practical margins for noise loss, flux bias/error,
centroid/shape, leakage/false promotion, support loss and instability, plus
absolute science limits and runtime/memory budgets. Numerical values must suit
the selected mode and use; none are inferred from this draft. Predeclare the
independent statistical units, paired contrasts, interval/confidence procedure,
required precision/realization count, and treatment of the several required
metrics and quality strata. Pixels, occurrences and iterations are not
automatically independent replicates. Shared state and noise remain disclosed.

For example, U passes the noise comparison only if the upper uncertainty bound
on V_U/V_N is at most 1 + delta_noise for the agreed domain. Apply the same
predeclared bounded-loss logic to the other required comparisons. Every
required criterion must pass; failure or unresolved uncertainty in a required
criterion cannot be averaged away by improved appearance or runtime. Report
both point estimates and uncertainty, including the diagnostic benchmark's
known assumptions. This is a bounded noninferiority test, not proof of equality.

Possible dispositions are: uniform acceptable for the tested regime; uniform
fails a required criterion in that regime; or evidence insufficient. A promising
discovery result requires the reserved independent-pointing replication before
a pointing policy recommendation. That observation must not set the estimator,
case values, thresholds or margins. A different mode needs matching independent
evidence. A restricted detector/noise domain may be proposed from the result,
but needs a causal pre-action eligibility rule, separate owner approval and
fresh evidence; post hoc exclusions cannot turn this screen into a pass.
Failure of U does not qualify N. If neither yields an adequate case, preserve
the negative result and return to the historical control and simpler contract.

## What must be bound before a run

| Existing decision | Exact remaining item for this screen |
| --- | --- |
| Q02 | Adopt applicable A/B successors where used; separately approve/adopt the experimental U/N families, current QC/profile/source records and actual MAP input/mode permissions. T1/T2 must bind their own admitted inputs even without a FRUIT loop. C remains a policy question. |
| Q03/Q04 | Exact PTC plan/segments where used, grid/AST association, coefficient population and numerical MAP support policy/value, with failure behavior. Reuse adopted values when applicable. |
| Q05 | Training guard/support and minimum count; actual noise, response and uncertainty evidence; synthetic/injection/null laws, stages and remaining population permissions. No covariance or precision is invented from scatter. |
| Q01/Q06 | T3's actual baseline rule if continued; selected mode/IDs/control, quality-regime selection, reserved replication, all case values/counts/seeds, margins/precision, L where used, software/resources/products and exact execution authorization. |

Prepare those bindings before requesting run authorization. The proposed
first execution unit is T1 plus T2; T3 is conditional follow-up. The test design
does not commission a replay, implementation, Unity action or qualification.
Routine bugs in a later authorized experiment may be repaired within its exact
method/gates/population/input/scope; changing any of those requires owner review.
