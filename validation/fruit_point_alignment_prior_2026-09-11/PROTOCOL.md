# POINT relative-position prior screen r0.1

2026-09-11. Identity SCI-FRUIT-POINT-ALIGNMENT-PRIOR@r0.1.

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md) and existing
[reviewed recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md).
Adopt the predecessor's [exact recurrence, input, PTC, map and measurement definitions](../fruit_point_coherent_feedback_2026-09-11/PROTOCOL.md).
Cite its [rejected candidate and controlled recovery](../fruit_point_coherent_feedback_2026-09-11/SCIENTIFIC_REPORT.md).
The owner now authorizes the proposed positional-prior test, correcting the
empirical premise: a1100/a1400 are very accurately aligned; a2000 has a small
systematic relative displacement of less than 2 arcsec. This supersedes the
earlier statement that all array centroids agree within 2 arcsec. It does not
reopen the rejected experiment, frozen contracts, weighting or production.
There is no independent author dispatch or new generic scientific derivation.

The owner has not supplied the a2000 displacement vector or a statistical
distribution. Treat its direction as unknown. Exact sharing of a1100/a1400's
position and a uniform admissible a2000 offset disk are explicit experimental
approximations, not measured zero error or a Gaussian uncertainty claim.

## Candidate and control

G is the unchanged r0.2 free per-array Gaussian feedback method. J changes
only inference of position and its necessary availability/admission conditions.
Fit the reconstructed total at every pass. Infer the common position from
a1100/a1400 jointly, with their own signed amplitudes, independent widths/angles
and planes. Minimize the sum of their unweighted pixel squared residuals on
their existing fixed domains. The common position is free within ±80 arcsec.
No nominal source location, flux, width or flux ratio is supplied.

Condition a2000's separate Gaussian-plus-plane fit on that common position,
allowing a free displacement of radius 0–2 arcsec and arbitrary direction.
a2000 cannot move the pair's position. This uses the pair as the positional
reference for this POINT policy; it does not recalibrate array geometry, WCS,
pointing corrections or any upstream product. Relative offsets may be inferred
again from each reconstructed total; no offset estimate becomes calibration.

Use the existing 4–60 arcsec width bounds, same 3R positive-peak admission and
same fixed feedback domain. Each array must pass its own peak and width check;
no source amplitude is borrowed from another array. Reject a2000 feedback when
neither reference array admits a source, or when its fitted displacement reaches
the 2 arcsec boundary (radius >=1.999 arcsec). Record this as prior tension,
not successful alignment. Zero radius is allowed; it is not a rejection.
Planes remain diagnostic only. Unavailable numerical fits fail the trajectory;
resolved admission rejection produces an explicit zero replacement correction.

Numerical solution: first compute the unchanged independent free fits for
initialization and evaluation. Solve the reference-pair fit from three starts:
the two independent centroids and their midpoint, retaining each independent
shape/amplitude/plane as its starting value. Select the successful solution
with minimum total squared residual. For a2000 use three width starts 6,18,42
arcsec, offset radius 1 arcsec and direction toward its independent fit.
Fit radius, direction and all seven shape/amplitude/plane parameters freely;
choose the smallest successful squared residual. Analytic Jacobians and the
predecessor's convergence tolerances/250-evaluation limit are used. A scalar
residual normalization aids numerical solution and does not weight detectors
or change relative array contributions. No start or boundary is tuned later.

Retain immutable-parent replacement, rank-5 network PTC relearning, ordinary
2-arcsec MAP, uniform occurrences, existing finite masks/support and seven
passes including bootstrap. There are no new reduction or weighting choices.
All inference uses maps available before its correction is applied.

## Fixed population and tests

Use the same 123424 pre-PTC parent and two nuisance seeds 20260911/20260912.
Retain both nulls, both aligned known Gaussians, the original two-component
mismatch and real 123424. Add five cases on seed 20260911 only:

1. Near-offset: the same Gaussian, with a1400 shifted (+0.2,−0.1) arcsec and
   a2000 shifted (+1.2,−1.2) arcsec; this also tests exact reference-pair sharing
   against a small genuine mismatch.
2. Outside-offset: a2000 shifted (+4,0) arcsec; other arrays unchanged.
3. Absent-a2000: the known Gaussian in a1100/a1400 and exactly no source in a2000.
4. Contaminant-only: a2000 alone contains a peak-180 Gaussian at (−24,+24)
   arcsec, widths (18,10) arcsec and angle 0.3 rad, added to the nuisance.
5. Source-plus-contaminant: the aligned known Gaussian in all arrays plus
   exactly that contaminant. Its paired response subtracts contaminant-only,
   not the uncontaminated null. This separates desired source response from
   the nuisance feature while retaining both absolute maps for inspection.

All quantities use the predecessor's legacy mJy/beam representation. A source
is peak 100, centroid (17,−11), widths (12,8) and angle 25 degrees unless the
case explicitly changes it. Truth never enters inference. Declare the added
feature as structured nuisance for this POINT association test, without a
claim that real astronomical contaminants can universally be discarded.
Every case/arm/pass relearns from its own parent minus its current model.
Total: 11 cases, two arms, seven passes = 154 passes. Preserve the previous
pixelwise arm as contextual evidence on its original cases; G/J is the matched
comparison here. Do not rerun or replace old products.

## Assessment and stopping

Use the same unconstrained Gaussian evaluator on resulting total/paired maps
in both arms. A prior-constrained position cannot prove recovered pointing.
Retain direct truth-map error, signed fixed-aperture recovery, amplitude,
centroid, both widths, residual/exterior structure, support, failure/rejection
causes, every model/learned state, cumulative wall time and peak memory.
Fit photometry in a fixed truth-centered 20-arcsec radius only for the added
contaminant response, because a full-domain fit can select the nuisance instead;
also retain the full-domain free fit and direct fixed-domain error. This
evaluation-only aperture is frozen now and never controls feedback.

For aligned and near-offset Gaussian cases require the inherited 5% peak and
width targets and 0.05-minor-width centroid target by pass seven, no >2 percentage
point increase in absolute amplitude error versus G, and no >10% increase in
direct map error/exterior leakage for the original mismatch. Require zero new
source admissions in either all-array null or absent a2000. The contaminant-only
case must not obtain a2000 source feedback without a reference-pair anchor.
With source-plus-contaminant, require desired-source response to meet the same
Gaussian targets by pass seven and keep a2000 feedback associated with the
reference pair rather than the remote nuisance. Compare to G's unconstrained
result. The outside-offset test is a falsification case: it must expose prior
tension/rejection or a failed unconstrained recovery target; it cannot be reported
as successful alignment merely because the model satisfies its own bound.

Real-data keep requires the same free evaluator to identify the common source
in a2000, or an explicit non-detection/zero-feedback result instead of displaced
promotion, with stable useful strong-array recovery and no new unexplained
flux/shape deterioration. Report OG published pointing as a separate benchmark,
not truth. Improvement must survive the synthetic gates; positional agreement
alone is insufficient. Report any cost increase honestly; a provisional keep
requires no more than twice G's cumulative method time at useful recovery.
That generous screening bound is a new development cost tolerance, not a
production latency requirement. Compare recovery against cumulative time, not
only equal pass count. No deployable stopping rule is inferred from truth.

Freeze candidate, gates, cases and code before execution. No scientific revision
or prior-strength/rank/threshold sweep is included. Repair only routine code
defects under the existing repair direction, preserving every attempt. If J
passes development, evaluate reserved 129081 only under this unchanged freeze
and a usable matching immutable pre-PTC input; an unavailable input is reported,
not replaced by a different parent. Otherwise leave 129081 reserved. This screen
must end with keep/revise/reject and numerical evidence, not open-ended design.

One process, four BLAS threads, two-hour campaign cap, 8 GiB peak RSS and 4 GiB
retained output. Timing charges all inference needed by each arm, excludes
evaluation-only work, and retains inclusive development wall time separately.
Common input setup is separate. No Unity or production change is authorized.
