# POINT 152389 map-only JINC — numerical decisions needed

The scientific comparison has **not run**. The directive's section 3 requires
explicit numerical authorization and says historical settings/defaults do not
supply it. The accepted uniform coefficient family already has independent
JINC permission; that permission is not the blocker.

## Already resolved

- Analytic/discrete operator: SCI-JINC v0.1/r0.3, frozen
  a9f43877e01a661db13bd85b2e7f34ea5ac82fb7. Exact first positive J1 root,
  peak-normalized JINC × generalized exponential × second JINC;
  signed N/C, Q = sum(omega*kappa²), time = sum(kappa²/fs).
- Center floor(u+1/2), upper-bin phase midpoint, square half-width
  ceil(s*rmax/Delta), no radial crop, original full target grid.
- Dimensionless uniform gamma=1: SCI-PTC:uniform_constant@draft-0.1,
  registry SCI-PTC:analysis_gridding_coefficients@draft-0.1,
  profile SCI-PTC:uniform_coefficient_handoff@draft-0.1;
  PTC/JINC boundary v0.2-draft.1/r0.4 grants JINC under UFC-OD-009.
  This creates no inverse-variance meaning or default coefficient selection.
  This work order requires the already selected identical realized values.
- The current work order authorizes only the narrow saved rejoined-descendant
  adapter at T10. It does not relabel that descendant ordinary PTC output.

## Decision 1: explicitly select the numerical realization

These are recovered **historical candidates, not approved experiment values**:

| Array | s (arcsec), historical wavelength/45 m | a | b | c |
| --- | ---: | ---: | ---: | ---: |
| a1100 | 5.042028597151 | 1.1 | 1.67 | 2 |
| a1400 | 6.417127305465 | 1.1 | 2.17 | 2 |
| a2000 | 9.167324722093 | 1.1 | 3.17 | 2 |

The underlying candidate scales are exactly the declared formulas
(1.1/1000)/45, (1.4/1000)/45 and (2.0/1000)/45 radians; displayed arcseconds
are rounded. These are not the nominal calibration beam FWHM values 5/7.5/10.
Historical Point/OOF rmax is 1.5, while Science/Beammap uses 3.0.
Historical phase count nsub is 1. Do not choose a mode-dependent radius or
ambient phase count automatically. At the existing 2-arcsec grid, the accepted
ceil rule gives half-widths 4/5/7 pixels for rmax=1.5, or 8/10/14 for rmax=3.

Needed owner decision: select one exact tuple (s,a,b,c,rmax,nsub) per array,
with angular units and declared scale semantics; the historical 45 m
formula carries no inferred physical-aperture meaning. The table and rmax/nsub above may be
explicitly approved for this bounded comparison or replaced by supplied values;
no choice is implied by recovery. Scientific suitability remains an experiment
outcome, distinct from software numerical fidelity.

Provenance: quarantined legacy values in SCI-JINC PRIOR_WORK.md with source
fbfdc3479e7e61e2618bdf5ab81f6634df476e4c; current jinc_mm.h's wavelength/45 m
realization; data/config.yaml shape/phase defaults and rmax=3; the opt-in
Point kit config/tolteca/point/70_pipeline.yaml rmax=1.5. ODQ-102B expressly
withheld numerical authorization. Existing code and prior successful reductions
are evidence, not substitute approval under the current contract/directive.

## Decision 2: approve the numerical-adequacy profile

No approved exact profile or matching certificate was found in the frozen
package or later admitted authority. The historical rho >= 2*gamma_n rule,
rounded Bessel root and approximate 10^-3 guidance are not substitutes.

The owner-approved profile must specify these fields (or identify an existing
exact approved record containing them):

| Required field | Recovered status |
| --- | --- |
| Observable(s) and units | Exact profile absent; normalized m and its N/C error need an explicit choice |
| Absolute/relative inequality and reference norm | Not selected |
| Source/sample/pixel/parameter domain | Must bind the selected per-array realization and declared support |
| Numerator and signed-denominator error propagation | Not selected |
| Near-zero/cancellation rule | Not selected; nonzero C alone is algebraic support |
| Simultaneous/worst-case interpretation and constants | Not selected |
| Covered errors | Must state root/Bessel/kernel evaluation, phase/geometry representation and accumulation/normalization treatment |
| Certificate method, lifecycle and provenance | Not selected; a compatible certificate must then be generated and checked |

Passing the small independent oracle fixtures only validates those explicit
fixture cases. It cannot certify millions of observation contributions or turn
an arbitrary cancellation threshold into scientific authority. Selecting the
profile and subsequently demonstrating its certificate are separate steps.
The directive authorizes bounded implementation/validation, not fabrication of
these missing scientific decisions or a broad policy rewrite.

## Preserved stop

The fifteen saved inputs are the original adaptive C+D REF M0/T10 and the
three C T10 endpoints across three arrays. No REFSEQ result is the baseline.
Pending the two decisions, every JINC primary-map cell, injection-reference
comparison, nine-phase response, background/noise comparison and 256-trial
ensemble is not run. No JINC feedback, new basis/candidate, m10 application,
source-policy change or new default follows from this preparation.
