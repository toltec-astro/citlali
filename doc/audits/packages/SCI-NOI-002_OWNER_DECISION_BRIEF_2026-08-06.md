# SCI-NOI-002 owner decision brief — 2026-08-06

Status: coordinator review record only. Exact-d501 audit
`4f1fec36f7802f3b5e8ac067377679946930983c` is `amend` with all eight P1
findings open, production `existing_use_only`, no count/default selection, and
no evidence authorization. Owner decisions D001/F001, D002/F002, D004/F004,
D005/F005, D006/F006, and D007/F007 are approved as recorded below. Owner
direction D003/F003 is a qualified package-level provenance contract, not
acceptance of the audit's broad per-product premise. D008/F008 is approved as
the held proportional-evidence contract below. No decision authorizes repair,
evidence execution, recipient dispatch, or production use.

## Owner-approved first decision: F001 / D001 — what does a realization stack estimate?

**Issue and current code.** Exact d501 calls centered pixel scatter with divisor
`R` “noise variance,” without naming the target random law, completed design,
support domain, or dependence correction; it degenerates at `R=1` and is low
under an iid common-law interpretation.

**Consequence.** The existing quantity can describe this finite,
source-imprinted stack, but it cannot by itself support physical-noise
covariance, inverse-variance, significance, or count-adequacy claims.

**Engineering versus science.** Engineering can record completed identities,
support, centering, and divisors. The scientific choice is the target law and
the conditions under which any calibrated uncertainty claim may be made.

**Approved policy — 2026-08-06.** Retain the existing `1/R` computation only
as empirically centered conditional finite-stack scatter for the completed
`source_imprinted_current` realization stack. Persist completed `R`, joint
assignment/design identity, source-imprint mode, common support/response/
validity, and exact normalization. `R=1` may have mathematically trivial zero
descriptive scatter, but is invalid for uncertainty or S/N use.

It is not repeated-physical-noise variance/covariance, inverse variance,
precision, significance, threshold authority, aperture uncertainty, or
production noise calibration. A future physical-noise estimator is separate:
prioritize balanced interleaved scan-split null maps and fixed-consumer
projections, especially aperture/background and source-amplitude operators.
Delete-group or residual block resampling is admissible only with preserved
detector/common-mode correlations and validation against genuine repeats or
blank apertures. Calibration, extinction, and persistent detector-calibration
terms remain separate systematics.

**Owner question.** Answered by this approval. No physical-noise estimator,
evidence request, default, count, Unity action, or repair is authorized.

**May defer.** A numerical formula, implementation SHA, count, and evidence
plan may wait until the target is chosen. This governs F002--F007 and all
consumer handoffs.

## Coupled scale and consumer decisions: F002, F004, F005

### F002 — global scale of the MAP pattern

**Issue and current code.** A reciprocal median of MAP nonprecision coefficient
times the same centered stack scatter globally rescales that coefficient and
silently uses scale one when the calibration region is empty.

**Consequence.** It is at most an in-sample engineering scale diagnostic, not
marginal precision or covariance calibration.

**Engineering versus science.** Engineering can make empty/invalid calibration
states explicit. The scientific choice is whether a bounded global-scale target
exists at all, and what data/validity region defines it.

**Approved policy — 2026-08-06.** Retain
`alpha = 1 / median_D(q_p V_p)` and `alpha q_p` only as a
`global_nonprecision_scale_diagnostic`. Never label or infer it as inverse
variance, precision, physical-noise weight, covariance correction,
significance calibration, or proof that `q_p` has the correct spatial form.
Persist original `q_p` identity, calibration-region definition and valid-pixel
count, `m`, `alpha`, realization/design/source-imprint identity, overlap/
same-stack fact, and explicit validity. An empty, nonfinite, or invalid region
is unavailable and must not silently substitute `alpha=1` as successful
calibration.

Do not expand this use or make it a new default. Preserve current live
application/overwrite behavior only as explicit `existing_use_only`
compatibility pending FLT and MODE consumer review; this approval does not
authorize removal or expansion. Long term, keep the scaled diagnostic separate
from the authoritative mapmaking coefficient unless independent validation
establishes proportionality to a named physical inverse-variance target.

**Owner question.** Answered by this approval. No implementation, evidence,
Unity action, default, count, or production change follows.

**May defer.** Any spatial covariance model, precision relabeling, and repair
are not selected here. Interacts with MAP’s nonprecision `weight_I` contract
and FLT-001/FLT-002.

### F004 — distinct S/N-like statistics and source claims

**Approved owner disposition — 2026-08-06.** Give every mathematically distinct
S/N-like quantity a distinct internal and product identity. Retain legacy
`sig2noise` names only as compatibility aliases with explicit identity,
deprecation, and not-significance metadata. Current values are descriptive or
engineering scores, not calibrated statistical significance.

Existing source-finder and pointing/OOF quicklook heuristic thresholds may
continue without adding expensive calculations. Invalid or zero denominators
produce invalid/unavailable status, never a numeric S/N=0 sentinel. No
false-positive probability, catalog completeness, universal N-sigma threshold,
or significance claim is allowed without separate SCI-SRC-001 validation of
response and null/search/selection/multiplicity behavior.

This is an identity-and-labeling policy only. F004 remains open pending an
authorized implementation of the distinct identities, compatibility metadata,
and invalid/unavailable behavior, followed by re-audit. It authorizes no
implementation, repair, evidence, Unity action, task launch, defaults/count
change, or production action.

### F005 — filtered/aperture uncertainty

**Approved owner disposition — 2026-08-06.** Preserve the current filtered
pixel product, but identify it truthfully as `filtered_pixel_stack_scatter`,
not point-source or aperture uncertainty. Apply the exact same realized
filter/operator and edge treatment to the signal map, every realization map,
and the applicable response/kernel map.

For aperture photometry, apply the complete fixed aperture-plus-background
operator separately to every realization and compute scatter of the resulting
scalar measurements. For fixed-template photometry, project or fit every
realization using the same fixed template and location. These outputs remain
conditional finite-stack scatter diagnostics unless separately validated as
physical uncertainty. Future validation may use blank apertures, split/null
maps, and repeated observations.

Do not require a dense covariance product or expensive per-pixel matrix
calculations. Package-level provenance remains authoritative under qualified
D003/F003. F005 remains open pending authorized implementation and re-audit.
This policy authorizes no implementation, repair, evidence, Unity action, task
launch, defaults/count change, or production action.

## Adaptive and count-policy decisions: F006 and F007

### F006 — FRUIT feedback

**Approved owner disposition with scientific requirement — 2026-08-06.** Keep
FRUIT’s existing numerical behavior `existing_use_only` pending SCI-FRUIT-001.
Identify its current median realization RMS only as a FRUIT scale gate / naive
baseline, not calibrated S/N or significance. Separate the internal
calculation from product persistence so the required scalar may be calculated
without forcing noise-map files to be written. If a nonzero gate is requested
and its scalar cannot be calculated validly, fail explicitly rather than
substituting a silent default. NOI-002 does not alter FRUIT’s algorithm, gate
threshold, or defaults.

Citlali needs a defensible map-uncertainty estimator for FRUIT/fruitloop
reductions of bright sources. MEDRMS was a reasonable naive first method, but
is not presumed adequate. SCI-FRUIT-001 must derive and audit the complete
iterative update, bright-source add-back, stopping law, and uncertainty
behavior; compare candidate estimators while retaining the median approach as
a baseline and account for adaptive bright-source reconstruction changing the
map/noise relationship. This does not select split-sample, cross-fit,
blank-aperture, or another successor method.

F006 remains open pending that SCI-FRUIT-001 contract, any separately
authorized implementation, and re-audit. This policy authorizes no
implementation, repair, evidence, Unity action, task launch, defaults/count
change, or production action.

### F007 — use-specific count adequacy

**Approved owner disposition with configuration clarification — 2026-08-06.**
Science=10 and Pointing=5 are current user-configured requested values, not
scientifically validated production values or universal defaults. Preserve
Science=10 only as current `existing_use_only` configuration behavior pending
use-specific convergence evidence. Citlali must not reinterpret a user’s
configuration value as a production or scientific authority.

Pointing and OOF quicklook products do not require noise maps, so their
standard noise work should remain/become disabled by default under the approved
minimal-calculation policy; optional explicit diagnostic use remains possible.
Standard Beammap noise work remains disabled/effective-zero/no-work, with
explicit opt-in only. `64` is optional validation/resource capacity, never a
target, minimum, default, or production recommendation.

The package records requested, effective, and completed counts. Incomplete
execution must bind products to actual completed count and validity, or fail
when the full request is required; it must never silently report requested as
completed. A future bounded Science convergence study must evaluate global
scale stability, aperture/template scatter, and later FRUIT uncertainty against
runtime and memory before any recommendation.

F007 remains open pending separately authorized implementation, use-specific
convergence evidence, and re-audit. This policy authorizes no implementation,
repair, evidence, Unity action, task launch, count/default change, or
production action.

## Established policy follow-through: F003 and F008

### F003 — provenance and labels

**Qualified owner direction — 2026-08-06.** The authoritative provenance unit
is the reduction package/bundle. Put complete semantic provenance once in an
authoritative compact package manifest/sidecar, rather than redundantly in
every FITS/HDU/file. Each individual product needs only a stable package/
provenance join, product identity/version and applicable digest/scope join,
plus the minimum validity/restriction label needed to prevent misuse if
detached. Audit and validation must verify package integrity and joins.

A detached product is unverified/out-of-contract; that does not prove that all
metadata must be duplicated. Record requested/effective/completed realization
count at package scope. Require per-pixel realization counts only when support
actually varies and a consumer needs them. Do not add dense covariance, sign
streams, per-sample IDs, or duplicated metadata.

This is an engineering/provenance policy, not acceptance of the audit’s initial
claim that every individual product is intrinsically under-specified. F003
remains open pending implementation, package-integrity/join validation, and
re-audit. It authorizes no implementation, evidence, defaults/count changes,
Unity, repair, or production action.

### F008 — proportional evidence

**Approved owner disposition — 2026-08-06.** Hold the following small,
deterministic local exact fixtures: R=1 invalidity; R=2 normalization;
duplicate, complementary, and simple-independent designs;
requested/effective/completed-count handling; empty calibration-region failure;
invalid-denominator behavior; and distinct S/N-like product identities.

Hold small deterministic local consumer fixtures for identical realized
filter/operator and edge treatment of signal, realizations, and
kernel-response; a two-pixel correlated case demonstrating realization-level
aperture projection; fixed aperture/template projection; and the FRUIT
configuration dependency without claiming validation of the full adaptive
procedure. Use analytic/exact expectations where possible.

Later astronomical validation is limited to an exact authorized repair SHA with
predeclared acceptance criteria: blank apertures, split/null maps, repeated
observations, and bright-source FRUIT tests routed through its separate audit.
Do not invent empirical tolerances, require a broad campaign, or treat mere
reduction success as statistical proof. F008 remains open pending separately
authorized implementation, exact fixtures, any admitted exact-repair-SHA
astronomical validation, and re-audit. It authorizes no evidence execution,
Unity action, task launch, repair, default/count change, or production action.

## Held recipient interfaces

The six audit proposals are held only: FLT-001 and MODE-001 for future
amendment/re-audit; FLT-002, SRC-001, and FRUIT-001 quarantined until their
independent cores; and BEAM-001 inactive unless the optional capability is
explicitly enabled. They do not dispatch recipients or decide their policies.
