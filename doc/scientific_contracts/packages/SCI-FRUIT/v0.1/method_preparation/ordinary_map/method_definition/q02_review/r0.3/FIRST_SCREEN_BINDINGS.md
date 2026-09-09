# First weighting screen: proposed numerical bindings

2026-09-09. This fills numerical choices left open in the approved r0.2 design.
The numbers are owner-review proposals, not approved defaults or test results.
The combined T1/T2 execution record remains incomplete until the real inputs
and their applicable scientific permissions are bound. T3 is not commissioned.

## Program adherence and prior-work recovery

Use the [current disposition](README.md) and [approved design](../r0.2/WEIGHTING_TEST_PLAN.md).
Retain its U/N coefficient law, causal training, same-population comparison,
support accounting and stop rules. Frozen MAP r0.7.1 supplies the conditional
N/Q and support formulas; actual numerical input permissions remain separate.
T1's abstract inputs are not CAL/PTC products, and T2's legacy products do not
become contract-conformant by being readable. No core or upstream source changes.

## T1 input and execution envelope

Propose a standalone local arithmetic experiment with one abstract array,
16 detectors d=0..15, one segment, and signal in arbitrary test units u. It
tests weighting and mapping, with no PTC fit, FRUIT pass or physical sky claim.
The required experimental-input authority would cover precisely these
synthetic rows and their diagnostic roles, without expanding frozen CAL/PTC
population permissions. The U and N laws are exactly those in r0.2.

| Binding | Proposed exact value |
| --- | --- |
| Geometry | 9 by 9 unit pixels centered at integer x,y from -4 through 4, with lower-inclusive/upper-exclusive edges. No WCS, interpolation or extrapolation. |
| Evaluation population I0 | Each detector visits each pixel at offsets (-1/4,-1/4), (-1/4,+1/4), (+1/4,-1/4), (+1/4,+1/4). Row-major pixel order: y then x, each increasing; offsets in the listed order. All occurrences finite/eligible by construction except explicit invalidity fixtures. |
| Training J_d | 256 prior samples per detector, excluded from I0. For d=0..3, positions x=3.25, y cycles -2,-1,0,1,2. For d=4..15, x=-4, with the same y cycle. The fixed assumed source guard is radius 3 about (0,0), so all J_d lie outside it. |
| N availability | Minimum training count 64; use all 256 J_d values. Require finite strictly positive centered second moment and finite positive normalized gamma for every required occurrence. No cap, replacement or detector drop. |
| Support | Propose coverage_cut=0.1 for this test alone, applying frozen MAP's exact order statistic and separate normalization/science predicates on all 81 pixels. Empty/failed required support is unavailable, not zero. This value is not inferred as a generally admitted MAP default. |
| Realizations | 1,024 independent synthetic trials per row below. Each trial has a paired null and signal input sharing the same noise and geometry. U and N consume the same input. No outcome-dependent extension. |
| Random identity | NumPy PCG64 with SeedSequence([20260909, row_index, trial_index, stream_index]); row indices 0..8 in table order, trial 0..1023. Streams 0/1: independent standard-normal training/evaluation errors in detector-major order; 2: common-component normals in time order; 3/4: burst indicators/signs. No random source placement. |
| Environment proposal | Local /Users/gwilson/tolteca/bin/python; recovered Python 3.13.2, NumPy 2.2.3, SciPy 1.15.2 on arm64 macOS. Pin the actual harness digest before execution. No implementation exists in this delivery. |
| Resource envelope | One worker, numerical-library threads fixed to one; at most 2 hours, 4 GiB peak aggregate process-tree memory and 2 GiB output. Exceedance stops with retained failure; no silent smaller population or omitted products. |

Use a compact source f(x,y)=A exp[-4 ln(2) r^2/w^2] for r<=3 and zero
outside, with r measured from its declared center. Nominal source: center
(0,0), width w=2 pixels and A=1.25 u. The weaker source has A=0.625 u; these
are 10 and 5 times the equal-noise uniform per-pixel standard deviation 0.125 u.
They are signal-strength labels, not an adopted inference threshold or S/N
product. The W05 source instead has center (1,0), w=2.8, A=1.25, with the same
radius-3 truncation; it intentionally overlaps some fixed training positions.
Truth enters generation and evaluation only, never changes J_d or gamma.

Base noise is independent Gaussian, standard deviation 1 u for d=0..7 and
r u for d=8..15. The source and null branches estimate N separately from
their available training values; in clean-guard cases those values coincide.
In W05 they need not. Freeze each estimated generation before mapping.

| Index / case | Exact change from this base | Signal |
| --- | --- | --- |
| 0 / W00 | r=1 | Nominal |
| 1 / W01a | r=2 | Nominal |
| 2 / W01b | r=4 | Nominal |
| 3 / W02 | r=2; noisy detectors have three visits at each listed offset, quiet detectors one. These are distinct independent-time occurrences; both arms use the same population. | Nominal |
| 4 / W03a | r=2; add the same stationary AR(1) process to every detector, coefficient 0.95 and marginal standard deviation 1 u. Start from N(0,1); innovations have scale sqrt(1-0.95^2). Continue from 256 training times into evaluation, without reset. | Nominal |
| 5 / W03b | Training r=2, evaluation r=4 | Weak |
| 6 / W04a | r=2; d=12..15 receive independent bursts with probability 0.01 per sample, sign equiprobable, amplitude 8 times that detector's base noise scale. Apply in training and evaluation. | Nominal |
| 7 / W04b | r=2; multiply source plus noise by 1.10 for d=12..15, in both training and evaluation; truth remains the intended calibrated source. | Nominal |
| 8 / W05 | r=2; fixed source guard remains unchanged despite broader offset source | W05 |

W03a's common component uses a common time index across detectors; its training
geometry need not be the same. W02 has no common component: repeat index is
innermost after each offset, so its unequal count has no ambiguous time pairing.
W04's poor detectors remain eligible under this synthetic finite-value policy;
this does not override any real detector veto.

Add deterministic W00 checks, outside the stochastic trial count: equal gamma
vectors produce identical maps; the projected constant correction returns on
complete matching support; lower/upper pixel edges follow the declared rule;
zero/nonfinite training scatter makes N unavailable; an explicitly invalid
occurrence fails the same gate in both arms. These verify arithmetic and failure
handling, not whether uniform weighting is scientifically acceptable.

The total stochastic work is nine rows times 1,024 trials times two input
states times two candidate policies: **36,864 map evaluations**. The
known-variance benchmark is an analytic diagnostic for independent Gaussian
rows, not a third deployable policy. U in W00/W01/W02 must reproduce its analytic
noise prediction within the predeclared Monte Carlo interval; failure blocks
interpretation. Estimated N is not required to equal the known-variance
benchmark. W03–W05 test limits, and may legitimately fail acceptability.
Do not require arbitrary stress cases to pass to claim a narrower tested regime.

## Estimands and proposed loss limits

Use the same fixed full-grid evaluation region in both arms, and retain native
and common support with unavailable causes. A required missing pixel blocks
that row's science decision. For synthetic null maps define M as the mean
of m_p^2 over the 81 pixels: it is squared RMS, including any bias, rather
than a recentered or automatically unbiased variance. Report the map mean too.

For signal tests, form the paired response map delta=m_signal-m_null and the
ideal noiseless binned source t on this exact geometry. Let A be the injected
peak amplitude. The amplitude diagnostic is A times dot(t,delta)/dot(t,t).
Compute centroid and x/y second central moments of delta on the full grid;
compare with the identical moments of t. Nonpositive total or variance makes
that metric unavailable. This is an evaluation of binned response; it is not
an integrated physical-flux or nonlinear source-fitting contract. Also report
signal-map error versus t to retain the effect of noise, and whole-grid
residual squared RMS. Do not infer full PTC response from these MAP-stage tests.

| Criterion | Proposed bound for a tested regime |
| --- | --- |
| Noise loss | Upper bound on mean M_U - 1.1025 mean M_N <=0: at most 5% excess RMS. |
| Response amplitude | Mean amplitude bias within +/-2% of A for each arm; also report the paired U-N difference. A similarly biased pair cannot pass by agreement. |
| Centroid / morphology | Mean response centroid bias in each axis within +/-0.02 w; mean x/y second-moment width bias within +/-2% of the corresponding ideal binned width, for each arm. |
| Leakage | Report source-response residual power and null spatial correlation. For the fixed gross-excursion diagnostic max_p abs(m_null,p)>5*0.125 u, the upper bound on probability for U minus that for N must be <=0.01. This is a diagnostic exceedance probability under the synthetic law, not a real-data false-source probability. |
| Support | Zero required science-region loss/unavailability; U's native coverage may be no more than one percentage point below N. Report both supports even when the criterion passes. |
| Cost | Remain within the resource envelope; report U/N elapsed time, weight-estimation cost and peak memory separately. No scientific loss is traded for speed. |

For continuous contrasts use trial-level means and sample standard errors,
with one-sided Student-t bounds at tail probability 0.05/256 (df=1023).
For two-sided criteria apply the same bound separately to each sign. Use exact
binomial bounds at that tail probability for each gross-excursion proportion;
the difference bound is upper(U)-lower(N). At most 256 scalar bounds may enter
this prespecified screen; report the actual number. Their Bonferroni allocation
does not assume independence across metrics or rows. The continuous bounds are
large-sample Monte Carlo approximations, not exact coverage guarantees. Report
that limitation and unresolved precision; do not convert it into qualification.
Support failures are recorded directly, never dropped from the denominator.

These are proposed screening tolerances. They need owner disposition before
execution and are not imported from structural-refactor bitwise comparisons.
Report pass, fail or inconclusive by case. Uniform is already known not to be
universally noise-optimal: W01a's ideal variance ratio is 25/16. The practical
question is whether the relevant real detector regime incurs an unacceptable
loss. A synthetic pass does not answer that population question.

## T2 bindings to finish from the requested bundles

Use the two discovery observations in the [collection request](DATA_COLLECTION_REQUEST.md),
with all available arrays reported separately. Keep each observation's actual
PTC output/recipe unchanged across U and N; no new PTC fit is part of T2.
Cross-observation differences are descriptive, with calibration and atmospheric
dependence retained. 129081's new weighting outcomes remain reserved.

Propose per-detector/per-existing-PTC-segment training from the first half of
that segment's original time indices, excluding the fixed source guard of
three times the largest declared nominal beam FWHM about the commanded source
position. Evaluation uses the second half, with the same fixed existing flags
in both arms. Minimum training count is 64. Normalize N over the exact
evaluation coefficient population, not a silently pruned subset. Original
segment boundaries, quality predicates, beam identity and geometry must be
verified before these rules produce an admitted population. A failed required
training group stops the paired screen instead of dropping a detector from N.

Before evaluation maps are opened, inventory sqrt(v) and its detector/segment
dispersion from these training-only samples. Proposed contrast criterion:
within an array, the ratio q90(sqrt(v))/q10(sqrt(v)) in one discovery observation
is at least 1.5 times the other, using sorted zero-based order statistic
floor(p*(n-1)). Treat arrays separately and list every group/count; no claim
that the pair covers a broader regime if this criterion fails. No replacement
case is chosen from weighted-map outcomes. Unknown prior-source/calibration
dependence remains unknown, not independent.

The grid/AST association, applicable ordinary MAP support policy, exact scalar
flag/quality meanings, signal/null injection authority and required statistical
evidence must be bound from the returned input record. Existing c=0.1 and
1-arcsec altaz/JINC metadata in 152389 are historical settings, not automatic
approval for a new ordinary MAP route. Per-detector coordinates cannot be
guessed from the boresight alone. Do not substitute raw APT sens or stored
weights for N's current training scatter.

T1's numerical limits are proposed starting criteria for a POINT screen, but
T2's actual flux/beam units, evidence conditioning and uncertainty procedure
still require review. A single real map's spatial RMS is descriptive; it cannot
be given the independent-trial intervals defined for T1. Disjoint halves still
share PTC fitted state. No new NOI estimator, full covariance, inference
probability or independent-noise claim is inferred here. These are explicit
remaining bindings, so this file is not yet a runnable T1/T2 authorization.

Retain per-trial inputs by deterministic regeneration identity, all small maps,
coefficients/scatter/support records, per-case metrics/failures, resource logs
and input/source hashes. The final record must account for every planned row.
No FRUIT convergence result exists in T1/T2; T3 supplies that separate question
only after its continuation decision. No scientific test or implementation was
performed while preparing this proposal.
