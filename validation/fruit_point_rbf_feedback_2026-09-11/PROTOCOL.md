# Bounded positive RBF feedback screen r0.1

2026-09-11. Identity SCI-FRUIT-POINT-RBF@r0.1.

## Program adherence and prior-work recovery

The [owner directive](OWNER_DIRECTIVE.md) authorizes this isolated implementation
and run under the [program charter](../../doc/scientific_contracts/README.md).
Adopt the [reviewed method recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md)
and the prior coherent screen's exact input, immutable-parent replacement,
rank-5 network relearning, ordinary-map support/geometry, coefficients, masks,
projection, seven-pass bound and [OG control](../fruit_point_coherent_feedback_2026-09-11/OG_BENCHMARK.json).
P is its pixelwise control and G is its independent Gaussian r0.2 control.
The positional-prior candidate is historical evidence, not part of this method.
No production contract/profile/default, upstream geometry, weighting, OOF policy
or independent-author scope changes. The [exposure audit](EXPOSURE_AUDIT.md)
records 129081 as historically characterized and reserved for new comparisons.

## Fixed representation and fitting

R uses nonnegative coefficients of localized unit-integral Gaussian basis
functions on a fixed 3-arcsec lattice, FWHM 4 arcsec in each array. Centers cover
the circle of radius 94 arcsec about the existing map origin, sufficient for
the fixed feedback domain D (science support within 90 arcsec). No recentering,
peak seed, source-width fit or support shrinkage occurs. The chosen resolution
is checked against the 8-arcsec narrow compact structure and an independent
diffraction simulation; it is not an enforced PSF width.

Basis values are Gaussian cell averages on the existing 2-arcsec grid, from
error-function integrals. Each localized kernel is truncated beyond six sigma
with a finite cell margin and renormalized to unit integral on its full discrete
template before clipping to D. Record clipped integrals b_j. Models are zero
outside D and project through the unchanged containing-pixel route. The total
ordinary map is retained separately from fitted, applied and next feedback.

Uniform fit pixels F=D union O, where O is the existing supported 90–150 arcsec
annulus. Fit a plane (1,x/90,y/90) on F jointly with the RBF source; the source
columns are zero on O. This uses the outer domain to distinguish background
from a broad positive source. The plane is never feedback. Let h=3 arcsec and
c_j=a_j/h². Minimize, for c>=0 and free plane beta,

    mean_F (M - h² Phi c - B beta)²
      + lambda [mean_edges (c_i-c_j)² + epsilon mean_centers c_j²].

Edges join horizontal/vertical lattice neighbors. lambda=0.01 and epsilon=0.01.
This is one screened first-difference quadratic smoothness penalty, with no
sparsity or concentration objective. Both residual and penalty have units
(legacy mJy/beam)²; lambda/epsilon are dimensionless. a has integrated-brightness
units mJy/beam arcsec², not automatically physical flux density.

Eliminate the unpenalized plane by QR projection. Reuse sparse normal matrices
plus their rank-three background correction, diagonal variable scaling and
nonnegative L-BFGS-B solves. Maximum 2,000 iterations/6,000 function evaluations,
ftol=1e-13, gtol=1e-10, projected KKT residual <=1e-4 relative to the RHS.
Require solver success and finite results; otherwise fail explicitly. Record
conditioning upper bounds from the screened positive floor, sparse dimensions,
KKT, iterations and costs. Previous coefficients may initialize the same convex
fit; the result is always complete replacement, never accumulation.

## Separate coherent admission and separate concentration

Split F into the two fixed pixel-checkerboard parities. Independently fit the
same source/plane estimator on each parity. Predict its source on the other
parity. Project that prediction and the held-out measurement off the plane
space, and compute prediction dot measurement divided by R times prediction
norm. R is the predecessor's empirical annular MAD scale, not a noise-qualified
standard deviation. Require both cross-prediction scores >=5 and normalized
inner product of the two fitted source maps on D >=0.8. Missing/nonpositive
norm rejects admission. These are empirical spatial-coherence rules, not
independent folds, false-alarm probabilities or proof of astronomical origin.
If admitted, use the whole all-pixel fitted RBF model on D, including its faint
wings. Otherwise the replacement correction is explicitly zero. No Gaussian
fit or bright central seed is needed. No concentration value enters this rule.

Diagnostic aperture A is fixed D, with 4 arcsec² per cell. Precompute
b_j=4 sum_A Phi_j and H_ij=4 sum_A Phi_i Phi_j including all off-diagonal
overlaps. Compute F=b^T a, A_eff=F²/(a^T H a) and
N_eff=1/sum_j(a_j b_j/F)². Verify against direct reconstructed-grid integration.
Report A_eff in arcsec²; N_eff is basis-dependent. Both exclude the fitted plane.
Rejected/zero source has unavailable scientific concentration, with raw fit
diagnostics retained separately. Report clipping/support and basis identities.
Neither diagnostic controls fitting, lambda, support, admission, stopping or
reconstruction selection; smaller area is not a success criterion.

## Representation check before trajectory freeze

Before empirical execution, fit noiseless compact Gaussian (12x8 arcsec),
in-focus diffraction PSF and defocused/comatic PSF at translations (17,−11) and
(19,−10) arcsec. The optical truth is independent pupil Fourier propagation:
2048² FFT, radius-64-pixel pupil, central obscuration 0.30, 0.5-arcsec fine
sampling, defocus 0.60 waves times (2r²−1) and coma 0.30 waves times (3r²−2)u.
Cell-average the independent intensity using sixteen subpixels. This is a
bounded aberrated-PSF example, not a calibrated telescope-optics model.

Separate unregularized representation error (lambda=0) from the frozen penalized
fit. Raw relative image L2 <=5% is the representation adequacy check; report
brightness and concentration errors and grid-phase changes separately. If that
geometry fails, allow one recorded resolution adjustment before freeze, not a
kernel/regularization sweep. Concentration is not optimized. Also fit a pure
plane and verify that it yields no meaningful positive source component.

## Paired empirical cases and observables

Run nine cases and three arms, seven passes each (189 passes): real 123424;
the two existing nulls and two compact Gaussians with seeds 20260911/20260912;
the existing mismatch on seed 20260911; comatic PSF at brightness 1 and 0.5 on
that seed; and a background-only case on that seed. Bright coma has the same
unclipped integrated brightness as the peak-100 compact Gaussian. Half-bright
coma tests fainter organized structure. Background-only adds the full-grid plane
13+7x/90−4y/90 to nuisance before learning, with no astronomical source.
Retain its paired residual and absolute admitted model; positivity can absorb
background even when the explicit fitted plane is excluded. All injections
enter the immutable parent before independent learning in every arm/case/pass.

For compact/mismatch cases retain all old amplitude/width/centroid, aperture,
direct-error and exterior-leakage measurements. For coma define pointing as the
signed brightness centroid within the fixed radius-60-arcsec aperture about the
injected translation, compared with the independently generated noiseless
centroid on that aperture. It is not the injected translation, peak or fitted
Gaussian center. Retain the ordinary Gaussian POINT measurement separately.
For the primary integrated-brightness/centroid measurement subtract a plane fit
to the paired response on O; also retain raw signed measurements and direct
unadjusted map errors so this cannot hide leakage. No positivity clipping of
the measured map. Nonpositive integrated brightness makes its centroid unavailable.
Fixed structural regions are r<=15, 15<r<=35 and 35<r<=60 arcsec about the
truth translation; wings combine 15<r<=60. No region depends on recovered peaks.
Paired subtraction does not assert exact noise cancellation in nonlinear FRUIT.

## Decisions, costs and limits

Retain compact 5% amplitude/width and 0.05-minor-width centroid targets, no >2
percentage point absolute-amplitude degradation against G, and mismatch direct
error/exterior nondegradation allowance of 10%. For bright/half coma respectively,
require integrated-brightness error <=10%/15%, centroid error <=0.4/0.8 arcsec,
relative aperture image L2 <=0.15/0.25, and wing-brightness error <=20%/30%.
Require at least 20% less coma image error than both controls in at least two
arrays, no >10% degradation in the third, and no new coherent null/background
admission. Report failures per array. Concentration usefulness is assessed
separately: <=10%/20% A_eff error for bright/half admitted sources and <=2%
noiseless grid-phase change in relative concentration error; these do not select
the reconstruction. No universal focus or rare-failure-rate calibration follows.

Preserve the preceding two-times-control development cost ceiling, applied to
the cheaper matched control at comparable useful recovery; also report equal-pass
cost and cases where controls never qualify. A scientific gain that misses this
cost bound is reported as such, not a deployed POINT improvement. Charge full RBF
fit and both admission fits; separately time reusable setup, cleaning, mapmaking,
required inference, diagnostic integration, external evaluation and output.
The primary first-observation cost includes the full observation-specific RBF
basis setup; warm reuse is reported separately. Diagnostic cost remains charged
to elapsed method time but is itemized. Retain inclusive and method map-ready times; current unused next-model inference
does not delay the already available map. No unmatched OG speedup claim.

One initial candidate and at most one targeted, recorded and separately frozen
revision. No automatic sweep. One process/four BLAS threads, two hours per
campaign, 8 GiB peak RSS, 4 GiB output. Preserve all attempts and iteration
products. 129081 remains reserved for new comparison; use only after candidate
and decisions freeze and a development keep, with the historical-exposure
qualification. Close with separate keep/revise/reject conclusions for feedback
and concentration. No Unity, production or broader OOF/SCIENCE policy.
