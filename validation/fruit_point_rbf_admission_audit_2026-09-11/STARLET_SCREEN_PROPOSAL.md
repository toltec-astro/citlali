# One saved-map starlet screen — proposed owner decision

## Program adherence and prior-work recovery

The [owner reassessment](OWNER_REASSESSMENT.md) and completed
[retained-product audit](REPORT.md) motivate this proposal under the
[program charter](../../doc/scientific_contracts/README.md). Both RBF candidates
remain rejected. The next decision is permission to implement and screen one
map-to-feedback estimator, plus one new brighter-coma bootstrap. This document
is **proposed, not approved for numerical execution**. It does not authorize
full trajectories, production adoption, a new generic contract or independent
author dispatch. No RBF search or cross-array coupling is included.

The scientific question is whether scale-local evidence can support a useful
positive source estimate without requiring the whole faint morphology to be
reproducible. The preliminary screen must test selection and reconstruction,
not merely transform invertibility. Undecimated wavelet reconstruction needs
care when coefficients are modified; see [Starck, Fadili and Murtagh (2007)](https://doi.org/10.1109/TIP.2006.887733).
[MORESANE](https://arxiv.org/abs/1412.5387) is precedent for separating analysis
and synthesis in radio imaging. Its interferometric deconvolution is not part
of this proposal. The numerical choices below are experimental choices, not
requirements inherited from those papers.

## One exact candidate

1. **Geometry and background.** Retain the 2″ map grid, supported S, fixed
   source domain D and outer annulus O. Fit the plane (1,x/90,y/90) by uniform
   least squares on O and subtract it for feedback inference only. Retain its
   coefficients separately. The source estimate is nonnegative on D and zero
   elsewhere. The ordinary total map is unchanged. This outer-only plane fit
   is an explicit part of the new estimator identity, not an unnoticed change
   to the RBF controls' joint plane fit.

2. **Transform.** Four undecimated first-generation starlet detail bands plus
   the coarsest band, using the separable cubic B-spline filter
   h=[1,4,6,4,1]/16. At level j=1..4 insert 2^(j−1)−1 zeros between taps;
   c_j=H_j c_(j−1), w_j=c_(j−1)−c_j. Use zero extension for computation and
   propagate a complete-support validity mask through every filter footprint.
   A coefficient touching an unavailable input is ineligible; no filled value
   becomes evidence. Record eligible area per scale/array. The source domain
   is fixed; coarse-scale loss at boundaries is a limitation to test, not a
   reason to shrink D around a peak. Signed coefficients are retained.

3. **Noise and selection.** Calibrate only from saved null seed 20260911 map 1.
   Measure each band's empirical MAD scatter after the same plane removal, in
   two fixed Q strata divided at median Q on D. Use the strata's eligible
   source-domain pixels; require at least 64 per band/stratum and positive
   finite scatter or fail explicitly. Seed 20260912 is a separate preliminary
   null check, not used to set thresholds. Select eligible coefficients at
   locations in D with |coefficient| >=5 times their band's scatter. Apply the
   same rule to c_4: broad source brightness is neither automatically removed
   as background nor automatically retained. This is an empirical multiscale
   support, with no Gaussian requirement, bright seed, connected-component
   prerequisite, or whole-field cosine. The finite null check does not
   calibrate a false-alarm rate.

4. **Amplitude reconstruction.** Freeze that support Ω for the current map.
   Starting from u=0, solve the nonnegative image-space least-squares problem

       minimize 1/2 sum_(j,p in Ω) [W_j(u−Y)_p / sigma_(j,p)]²,
       u outside D = 0,

   where Y is the background-subtracted map and W includes the coarsest band.
   Use the exact adjoint of the redundant masked analysis operator, not an
   assumed orthogonal inverse. There is no soft-threshold subtraction from
   surviving amplitudes, concentration objective, sparsity penalty, or
   expected-flux/position/width constraint. This models the telescope-produced
   PSF; it does not deconvolve toward an ideal point. The declared numerical solution
   is zero-initialized L-BFGS-B: maxiter=300, maxfun=3000, ftol=1e−12,
   gtol=1e−8 after scaling Y by its outer-annulus MAD; require finite output,
   solver success and relative projected-gradient residual <=1e−4 (infinity
   norm divided by max(||A^T b||_infinity,1e−12), with A the selected,
   scatter-scaled analysis operator). Store
   objective, iterations and residual. The problem can be underdetermined;
   this procedural solution and support selection are not claimed unbiased
   or uniquely identified. Unconstrained directions starting at zero cannot
   be called evidence for missing sky.

5. **Admission and replacement.** Empty Ω returns zero. Nonempty Ω and an
   acceptable numerical solve return the reconstructed positive source; an
   exactly zero solution remains unavailable as an astronomical model.
   Admission is thus based on declared location/scale evidence, without a
   second demand for identical full morphologies. Report which bands supplied
   evidence. Every new map reselects Ω and reconstructs a complete replacement
   from zero, allowing revocation. A successful solve does not establish
   astronomical origin; null/background failures reject this candidate.
   Concentration may be recorded afterward but cannot affect any choice.

## Preliminary cases and stopping point

Use all existing saved first maps for diagnostics. The existing noise calibration
and all RBF-era cases are development data; the spare null is not a pristine
holdout. Only 129081 stays reserved for new comparisons with its historical
exposure disclosed.

Test the actual estimator on the original compact and independent comatic
noiseless maps at both existing sub-lattice translations and all three actual
coverage masks. Keep the calibrated thresholds active. Also test map-level
truth-plus-saved-null inputs for compact, mismatch and coma. Label these as
estimator tests: adding at map level does not reproduce signal-dependent PTC
learning and cannot replace the later FRUIT comparison. Retain both old coma
brightness levels as challenging cases.

Add exactly one **4× original bright-coma normalization**, frozen before the new
map is generated. This is motivated by the weak-array oracle result; it is not
an outcome-selected flux. Use the same independent PSF, location, seed 20260911,
immutable synthetic parent, rank-5 network grouping and current mapping choices.
Generate one common bootstrap map with learning rerun on that injected parent;
no model is applied. Check its processed-template diagnostic against the retained
paired null. A linear scaling forecast suggests stronger evidence, but changing
PTC learning can invalidate that forecast. If the actual new case does not
exceed the descriptive five-scale diagnostic in every array, report that the
intended regime is still unavailable and stop; do not increase brightness again.
This remains an empirical benchmark, not a calibrated significance.

Proposed permission covers implementation, focused algebra/adjoint/mask tests,
these map-level estimator checks, and **one new PTC cleaning pass total**.
Cap at 30 minutes, one process/four BLAS threads, 8 GiB RSS and 2 GiB outputs.
Preserve initial failures; routine defects may be fixed within these definitions.
No numerical-method or threshold revision is included in this screen.

Before full trajectories can be proposed, require:

- Noiseless compact fitted amplitude/width bias <=5% and centroid error <=0.4″;
  for noiseless 4× coma, integrated brightness <=10%, fixed-aperture image error
  <=15%, centroid <=0.4″ and wing brightness <=20% on each array/phase. These
  test the actual selected/reconstructed model, not transform identity.
- Zero admitted model in the two nulls, pure plane and saved background-only
  first maps. Calibration-null results are training checks; the other seed is
  reported separately. No relaxation if these fail.
- On actual processed first maps, nonempty admission for all six compact
  array/seed cases and all three arrays of the new brighter coma. Report model
  errors and brightness separately; first-pass feedback is not required to
  undo all prior cleaning attenuation before recurrence has acted.
- In map-level truth-plus-null tests, compact amplitude/width errors <=15%,
  centroid <=1″ and fixed-aperture image error <=25%; brighter-coma brightness
  <=20%, image error <=25%, centroid <=1″ and wings <=30%. These are preliminary
  estimator gates, not replacements for the tighter final output-map gates.
- No solver failure or unexplained boundary/phase discontinuity; report all
  eligible areas and clipping. Numerical unit checks include transform sum,
  adjoint identity, signed coefficients of a positive source, and background
  exclusion. Phase changes in relative noiseless brightness/image error must
  be <=2 percentage points. Median inference across arrays per map <=0.5 s
  and maximum <=2 s, with setup reported separately.

A failure closes this candidate at the preliminary stage. Passing supports a
separate review of a frozen matched P/G/starlet FRUIT run: at most ten cases,
three arms and seven passes (210), with current rank/recurrence/mapping unchanged
and the original tighter output-recovery and useful-runtime gates retained.
Reuse RBF results only for the original matching inputs. No full trajectory or
129081 run is authorized by this proposal. This is a new estimator–admission combination. A gain could not be attributed
to the wavelet representation alone; P/G admission remains unchanged and no
RBF arm is silently repaired. The next owner decision can therefore
be limited to the small preliminary screen above.
