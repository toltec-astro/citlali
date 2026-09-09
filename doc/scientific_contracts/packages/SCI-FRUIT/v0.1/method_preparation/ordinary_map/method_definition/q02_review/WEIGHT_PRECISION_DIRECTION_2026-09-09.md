# Q02/Q05: approved weight-precision goal

2026-09-09. Scientific owner: Grant Wilson. Decision identity:
`SCI-FRUIT-Q02-WEIGHT-PRECISION-2026-09-09`.
**Approved as a provisional design objective; actual precision is unmeasured.**

## Program adherence and prior-work recovery

Continue under the [charter](../r0.4/inputs/program/README.md),
[pilot workflow](../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md)
and [frozen recovery](../r0.4/PRIOR_WORK.md). Adopt Q01 and Q02-A/B unchanged,
and retain the [approved weighting design](r0.2/WEIGHTING_TEST_PLAN.md).
Cite the [r0.4 preflight](r0.4/INPUT_PREFLIGHT.md) as implementation-informed
input evidence. This dated direction supersedes its recommendation to lead
with observation-wide pooling and a split execution decision. It preserves
every prior packet's bytes and findings. No new scientific-contract author,
allowed reference, upstream profile or numerical method is adopted.

## Exact owner decision

The owner asked what accuracy would be sufficient for estimating weights.
The manager proposed about 20% fractional uncertainty in relative detector
weights at one standard deviation; 10% would be desirable, with actual
acceptability judged by map noise, recovery and morphology. The owner replied:

> Sounds good to me.

This approves **20% as the initial precision goal**. It does not state that
20% has been achieved, establish a 95% interval, or turn 20% into a universal
hard veto on every detector. The weighting policy and its population-level
acceptability criteria remain Q02-C decisions. The earlier proposed 5% map-RMS
loss and other numerical science margins have not gained approval from this
answer.

The next preparation should justify a training window and any minimum sample
count from attainable precision and stability. The proposed count of 64 is
no longer the leading justification for T2 readiness. Its old feasibility
counts remain correct for that exact first-half rule. No replacement cutoff,
detector exclusion or exception has been selected. The unexecuted T1 fixture
and its existing numerical proposal remain unchanged.

## What the precision target means

The target concerns the normalized relative coefficients used in mapping,
gamma_i = w_i / mean(w on the declared evaluation population in the same array).
A common multiplier cancels. An uncertainty assessment must repeat this
normalization jointly; marginal error bars for individual unnormalized
inverse scatters do not by themselves establish uncertainty in gamma.

Report the fractional standard deviation under an explicitly declared
repeatability model, its conditioning and the uncertainty of that estimate.
This measures precision. Source contamination, changing noise, calibration
error and a poor relationship between residual scatter and useful map
information require separate bias/adequacy checks. No unbiased-variance,
optimal-weight, covariance or precision-matrix claim follows.

For independent Gaussian samples, the chi-square variance law gives useful
planning scales: individual inverse-scatter weights have about 21%, 18% and
10% relative dispersion for 50, 64 and 200 samples respectively. Those are
not measurements of normalized coefficients in these correlated PTC data.
The statistical references are [NIST's variance result](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm)
and [chi-square distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm),
consulted in the owner discussion on 2026-09-09. They remain manager planning
references, with no independent-author reference admission.

## Bounded assessment to prepare next

Use existing 123424 and 152389 discovery reductions. Preserve their PCA
chunks, signal, flags and calibrated quantities. Keep 129081's signal/noise
and weighting outcomes reserved. A longer noise-estimation window does not
require a longer PCA reduction chunk or a new export.

Prepare the following training-only assessment before any weighted-map
comparison:

1. Bind the recovered storage intervals and exact discovery input meanings.
   Preserve the existing separation of candidate training and evaluation
   occurrences. Count source-excluded, valid training support without using
   an evaluation map or injected truth to define it.
2. Start the window proposal with one, two and four consecutive original
   chunks, anchored at chunk zero and grouped without overlap. These correspond
   to approximately 5, 10 and 20 seconds. A longer window pools only the
   candidate training portions; it does not borrow evaluation samples.
   These are comparison proposals, not an adopted coefficient generation or
   a selection of a window per detector. Observation-wide pooling is deferred.
3. Assess temporal dependence relevant to the scatter statistic, including
   squared-residual dependence. Preserve time gaps and shared detector/PTC
   state. The final uncertainty method must account for joint normalization
   and justify its treatment of correlations; raw sample count and the
   effective count for a sample mean are insufficient substitutes.
4. Compare precision with changes in the underlying scatter over time.
   A longer window that mixes changing noise cannot be accepted solely for
   producing a smaller statistical error bar. Report source-contamination and
   finite-support limitations separately. No detector pruning or fallback is
   introduced to make a window pass.
5. Return per-array precision/availability distributions, contributions of
   poorly determined groups, and the precision-versus-time-resolution tradeoff.
   Any resampling, covariance or stationarity model, numerical settings and
   coverage criterion must be bound before execution. Unknown uncertainty
   remains unknown. This direction alone does not select a new NOI estimator
   or authorize the unbound training-scatter experiment.

The next deliverable is that bounded assessment's exact method and execution
proposal, with the smallest unresolved scientific choices made explicit.
There is no current request for another reduction, longer PCA chunks or
observation-wide pooling. Standalone T1 execution remains unapproved; it is
not inferred from approval of this precision goal. T2 mapping, replication
access and T3 feedback retain their separate decisions.

No new data analysis or experiment was performed to record this direction.
All reduction products, prior packets and both opaque archives are preserved.
`FRUIT-FEEDBACK-METHOD = unavailable_pending_separate_owner_approval`.
