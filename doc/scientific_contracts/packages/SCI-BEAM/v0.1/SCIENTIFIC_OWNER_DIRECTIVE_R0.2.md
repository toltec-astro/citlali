# SCI-BEAM v0.1 — Scientific-Owner Revision Directive r0.2

Status: governing scientific-owner direction; implementation comparison prohibited

Scientific owner: Grant Wilson

Directive date: `2026-08-17`

## Authority and task boundary

This directive supersedes conflicting scientific language in the still-
unfrozen `r0.1` draft. It retains contract version `v0.1`, advances the two
document representations to revision `r0.2`, and authorizes substantive
replacement rather than compatibility preservation. It does not alter the
content-bound `r0.1` author packet or claim implementation conformance,
response fidelity, observational performance, or production readiness.

The task produces a normative Formal Scientific/Engineering Contract and a
separate science-team rationale. It expressly excludes current Citlali code,
APT contracts or files, audits, repairs, tests, and production reductions.

## Settled scientific decisions

1. SCI-BEAM owns complete Beammap analysis and the complete Beammap APT,
   including effective-core fits, relative detector coordinates, source-APT
   `flxscale`, NEFD-like `sens`, independent quantity states, and empirical
   maps/diagnostics.
2. For detector `d`, `x_d(t)=Delta f_d(t)/f_d` is the uncalibrated detector
   observable; `xs` denotes its detector collection. The v0.1 fit input is the
   standardized per-detector Beammap `x^BM_{d,p}` in `Delta f/f`, not a
   calibrated map or a timestream-domain production fit.
3. Calibration uses the fixed, versioned nominal-beam source amplitude
   `F^TOA,nom_{s,a}` in top-of-atmosphere mJy per nominal beam at the same
   reference origin as the fitted amplitude. Finite-source dilution is already
   embodied in that quantity; no additional `H(0)` enters `flxscale`.
4. The fitted shape is the observation-local effective PSF core measured in
   the standardized Beammap. It is neither automatically intrinsic nor the
   complete finite-field PSF. Empirical maps, support, residuals, response
   lineage, and wing-completeness state remain required.
5. The fit realizes a complete positive-definite two-dimensional PSF tensor,
   including off-diagonal/rotation information. At circularity orientation is
   unavailable. FWHM is a compact representation; Gaussian solid angle is
   conditional and is never promoted to complete beam solid angle by algebra.
6. The fit plane is metrically orthonormal. Full WCS evaluation or a validated
   local Jacobian supplies angular scale, sign, handedness, rotation, shear,
   azimuth metric, and material spatial variation.
7. The complete local model Jacobian and joint covariance retain material
   cross terms. At circularity covariance is expressed in an invariant tensor
   basis.
8. When conventions are compatible, additional broadening is
   `Sigma_broad=Sigma_eff-Sigma_nom`. A Gaussian-broadening interpretation
   requires positive-semidefinite support within uncertainty; an indefinite
   result is model-inadequacy evidence, not a forced smoothing kernel.
9. Raw fitted detector offsets and horizon-derotated relative focal-plane
   coordinates are distinct. Derotation follows each detector's realized fit
   support and weighting, and the PSF tensor transforms with the coordinates.
10. APT origin is an arbitrary common-translation gauge. The physical
    field-rotation pivot is not established. A conventional pivot must not be
    labeled boresight or absolute array position. Bracketing pointing and
    science observations must use the same immutable APT realization and AST
    rotation convention unless a separately authorized transformation proves
    equivalence.
11. BEAM derives and publishes top-of-atmosphere source-APT `flxscale`:
    `A^TOA-eq=C^eff_s A^obs` and
    `flxscale=F^TOA,nom_s/A^TOA-eq`. Source atmosphere belongs to BEAM;
    target-observation atmosphere remains SCI-CAL's downstream operation.
12. BEAM derives and publishes `sens`, a top-of-atmosphere nominal-beam
    NEFD-like point-source sensitivity in `mJy beam_nom^-1 sqrt(s)`, from a
    robust off-source scan statistic multiplied by `flxscale`. Its exact
    scan-noise estimator and admission policies remain open.
13. `responsivity` has no canonical v0.1 scientific role and is deprecated
    compatibility metadata only.
14. Availability and validity are per quantity, not one all-or-nothing
    detector result. Every attempted detector and every unavailable quantity
    remain represented with causal reason codes.
15. Gaussian-core validity, complete-PSF/wing completeness, telescope-health
    use, calibration use, sensitivity use, and downstream kernel qualification
    are separate claims. Array/network stacks are optional diagnostics.
16. A science-impact program must determine required accuracy and qualify
    Gaussian, empirical, or unavailable downstream kernels. Authorship claims
    no validation result.
17. The first-principles Beammap APT schema is mandatory and independent of
    current storage. Per-detector quantities, APT-level calibration/geometry
    lineage, and immutable links to dense BEAM companions must be resolvable.

## Required document form

The science-team rationale targets 8–12 pages of main text and explains the
physical estimator, calibration, sensitivity, geometry, products, and needed
validation. Exact states, convergence machinery, the full requirement and
prediction inventories, and conformance conditions remain in the formal
contract. A crosswalk, open-decision ledger, r0.1 change log, cross-package
follow-up list, and consistency report accompany the two PDFs.

## Open decisions

Only genuinely unresolved scientific choices remain open: the exact `sens`
scan statistic, source exclusion/admission, exposure and bandwidth
normalization, model-adequacy diagnostics/thresholds, required PSF accuracy,
map depth/radius for wings, acceptable conventional-pivot residuals, future
physical boresight/pivot authority, and downstream kernel qualification.
