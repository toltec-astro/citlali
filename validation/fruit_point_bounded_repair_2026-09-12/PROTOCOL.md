# One POINT evaluator repair and stopping-tolerance retest

2026-09-12 · r0.1 · Freeze before saved-map numerical execution.

## Program adherence and prior-work recovery

Follow the [scientific charter](../../doc/scientific_contracts/README.md) and
[reviewed recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md).
Adopt the accepted recurrence and POINT ownership boundaries. Cite and preserve
the [previous operational trial](../fruit_point_operational_gate_trial_2026-09-12/SCIENTIFIC_REPORT.md)
and its original registered outcomes. This is development evidence outside
frozen scientific authorship. No production method becomes available.

## Owner decision and limits

Decision `SCI-FRUIT-POINT-BOUNDED-REPAIR-RETEST-2026-09-12` records the owner's
authorization of **one bounded repair-and-retest**, with evaluator reassessment
and numerical qualification handled separately, followed only then by genuine
operational reruns. The owner requires source evidence, centroid usability,
peak-response usability, shape warnings and support limitations to be separate;
injection truth must never enter the usability rule. The owner requires a
consistent finite, feasible projected-gradient completion check and comparison
with tighter solutions of the same saved problems. A larger operational cap
or different solver family is not the remedy. Useful improvement, especially
in peak response, must follow; passing a repaired harness is insufficient.

The investigator binds those directions below. Preserve every prior map,
registered measurement and decision. No new observation, calibration, domain,
rank, grouping, coefficient selection, background, objective, normalization,
regularizer, prior, solver family or recurrence is authorized here. No 129081,
Unity, production, qualification campaign, or automatic second repair follows.
Routine coding defects can be fixed with both attempts recorded if none of
these scientific choices, gates, inputs or resource bounds changes.

## A. Reassess saved total maps

Read all 167 saved pass maps from the previous trial and retain its three
original measurement records alongside the new judgments. No pre-PTC input is
opened and no cleaning call occurs. Apply exactly the same functions to P/C.

Keep the original positive band-2-to-5 central source-evidence screen. It is
empirical evidence, not a calibrated probability. Associate that evidence with
the fitted source only if a positive coefficient center intersects the fitted
10%-of-peak Gaussian core. Keep the original free Gaussian-plus-plane fit and
all original fit parameters; the plane is a nuisance, never feedback.

The existing parameter-boundary, ten-core-pixel and support limits remain:
centroid radius <52 arcsec and at least 95% of the radius-1.5-major-FWHM circle
inside supported radius 60. A major width >24 arcsec remains outside this
compact precision estimator's domain; gross and boundary cases retain their
source and warning information. The original >25% core residual shape warning
is preserved but does not by itself veto a centroid or peak.

Add one data-only sensitivity check: the same free Gaussian-plus-plane fit on
the fixed supported radius-52 domain, without moving either domain to truth
or a known source. Keep the radius-60 fit as the reported measurement. The
probe must be finite, positive and clear of parameter bounds. Centroid usability
requires associated source evidence, the common fit/support limits and a
centroid difference <=0.25 arcsec between the two fits. Peak usability instead
requires the common limits, <=1% peak change between the fits and the retained
peak >5 current outer-MAD rule. Centroid usability does not inherit that peak
score requirement. These stability fractions are prospective development
choices, not calibrated uncertainty or operational telescope policy.

Truth is read only by a separate scoring/report function after judgments are
made. Score all retained states/passes, including withheld measurements. Report
terminal compact centroid errors against 1 arcsec, H/D and D/H peak-ratio errors
against 5%, unchanged-seed ratios, imposed widths and array-response loss.
An existing warning on both H and T does not demonstrate detection of a new
fault. Keep gross coma and near-boundary limitations explicit.

For the conditional rerun, A must preserve original records and P/C bootstrap
symmetry, report no usable N/B source measurements or reliable E corrections,
and expose gross-case limitations. At least the original ten terminal P compact
centroids must be available in count, and every newly admitted terminal compact
centroid must meet the externally scored 1-arcsec test. This evaluates the fixed
rule; it cannot relabel individual cases using their truth. Peak failures are
scientific performance results, not permission to retune the rule.

## B. Qualify the numerical criterion on saved problems

Use every retained C input map, all three arrays: 144 problems, including
60 with nonempty selected coefficients and 84 empty-support problems. Freeze
the original total maps, geometry, null calibration and prior source code.
Reconstruct and check selected support, background and normalization against
the retained records before solving. Empty support remains a valid zero model;
its repeated appearances do not create independent statistical evidence.

Keep the four-level transform, positive source image, weighted coefficient-error
objective, zero initialization and L-BFGS-B family. Let v be the normalized
nonnegative source variables, g their objective gradient, and
`s = max(norm_inf(g_at_zero), 1e-12)`. Use the original KKT residual convention:
`pg_i = 0` when `v_i <= 0 and g_i > 0`, otherwise `pg_i = g_i`.
Completion requires finite variables/objective/gradient, v >=0, fixed support
and `norm_inf(pg)/s <= 1e-4`. Record each part separately. Optimizer success
is diagnostic; it is neither necessary nor sufficient for this criterion.
Set internal `ftol=0`, `gtol=0` so a different internal accuracy test does not
silently replace it. Keep memory 10 and line-search limit 20 (previous defaults).

Capture the first accepted iterate satisfying 1e-4. It is operationally
available only within the unchanged 3000-iteration/30000-evaluation cap.
Continue that same uninterrupted optimizer history to its first 1e-6 iterate;
do not restart or choose among different reconstructed images. The diagnostic
continuation alone has a 12000-iteration/120000-evaluation ceiling and a
180-second per-problem check at accepted iterates. This extra work supplies
the tighter comparator, not a larger operational inference budget. Failure
to reach either required point is an unavailable comparison, never a pass.

For each nonzero pair require <=0.5% change in model sampled peak and <=0.1
arcsec movement in its positive-brightness centroid on the fixed domain.
These universal model-stability rulers are explicitly distinct from POINT
Gaussian measurements. Also run the same free Gaussian-plus-plane readout on
both models: where both are finite, positive and off parameter bounds, require
<=0.5% fitted-peak change and <=0.1 arcsec fitted-centroid change. Otherwise
retain an explicit readout limitation; such a pair cannot support precision
Gaussian claims. Record full-image relative L2, brightness, objectives, KKT
residuals, iteration counts and elapsed time. No truth is used in these checks.

The accuracy allocations are one tenth of the owner's 5% and 1-arcsec budgets.
All 60 nonempty problems must supply an operational stopping point, a tighter
point and passing applicable stability checks. Missing or unstable comparisons
block C; successful empty cases cannot compensate. A failed qualification does
not prove every starlet method unusable, but closes this bounded attempt at B.
It does not authorize a tolerance sweep, regularization or solver change.

## C. Conditional operational rerun and decision

Proceed only if A and B pass. Freeze the trajectory harness using this exact
criterion before execution. Retain all 34 original trajectories, their order,
seven passes, rank 5, immutable-parent residual relearning, full replacement
feedback and common reductions; all cases and operational questions from the
[original protocol](../fruit_point_operational_gate_trial_2026-09-12/PROTOCOL.md)
remain. The new evaluator supplies separate availability for the corresponding
use. Rerun both P/C; old candidate trajectories cannot be relabeled into a new
recurrence result. No earlier successful pass replaces an unavailable terminal.

Retain peak-ratio fidelity, unchanged-source repeatability, imposed broadening
and response loss, centroid errors, signed residuals, leakage/support, failures,
every development iterate, cumulative wall time, inference/cleaning/measurement
time and RSS. Preserve the named OG benchmark and its different timing scope.
Keep requires a useful operational improvement over matched P, especially peak
response or equivalent accuracy at lower cumulative time, with the original
common safeguards and <=2x matched full-wall cost. A harness-only improvement
does not qualify. Independent-pointing replication still precedes a policy.

A/B together are limited to one hour, 8 GiB RSS and 4 GiB new output, zero
cleaning calls; C, only if admitted, retains its separate one-hour, 8-GiB,
4-GiB and 238-call ceilings. A hard campaign alarm and between-problem resource
checks enforce the saved-map bound. No automatic extension. Preserve failures,
issue a compact keep/revise/reject disposition and stop this authorized attempt.
