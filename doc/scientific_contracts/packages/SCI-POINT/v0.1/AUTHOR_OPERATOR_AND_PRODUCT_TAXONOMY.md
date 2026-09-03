# SCI-POINT v0.1 Author Operator And Product Taxonomy

Identity: `SCI-POINT_AUTHOR_TAXONOMY v0.1/r0.3`

Status: sanitized taxonomy candidate; exact profile identifiers author-assigned

## Operator

### `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`

Fits the approved known-source model on one exact observation-local per-array
map parent. Complete identity includes parent family/product/version/
generation, observation, array, WCS/grid/frame, terminal FRUIT ancestry when
applicable, source and expected-center relation, model, requested/effective/
realized search and support, weight/covariance use, constraints, response,
calibration, estimator version, and fit state.

Its numerical realization is unavailable until
`POINT-COMPATIBILITY-METHOD v0.1` is separately approved. Changing a
scientifically consequential element creates a different method or
an explicitly compatible successor. “Gaussian fit,” `raw`, or `filtered` is
not a complete identity.

## Required Per-Array Product Roles

| Product role | Meaning | Claim ceiling |
| --- | --- | --- |
| `POINT-FIT-RESULT` | numerical source-model fit, producer lifecycle, component states, support, method, and parent when compatibility authority is available | not an applied correction, intrinsic beam, source association, or source catalog |
| `POINT-DISPLACEMENT-MEASUREMENT` | fitted two-component AltAz tangent-plane source displacement | not absolute celestial position or correction vector |
| `POINT-AMPLITUDE-DIAGNOSTIC` | required fitted amplitude and QC metric in exact parent unit/calibration/response; candidate input to separately owner-governed CAL/TolProj amplitude admission | not universal source flux, standalone absolute calibration, or actual CAL/TolProj eligibility |
| `POINT-EFFECTIVE-SHAPE-DIAGNOSTIC` | required fitted widths/angle and QC metrics under exact parent response | not intrinsic telescope beam, detector PSF, SCI-BEAM product, or unique causal diagnosis |
| `POINT-FIT-QUALITY-CONTROL-METRICS` | centroid, amplitude, effective shape, uncertainty/constraint/support, and honest fit state for telescope/observing QC | not an automatic threshold/action or causal model |
| `POINT-LEGACY-DYNAMIC-RANGE` | canonical `fitted_amplitude_over_full_map_rms`, available only under the exact RMS method; legacy alias `sig2noise` | not statistical significance |
| `POINT-FORMAL-STANDARDIZATION` | amplitude/formal amplitude error | not empirical S/N or detection probability |
| `POINT-SOURCE-ASSOCIATION-STATE` | branch-independent source association: `established`, `unavailable`, or `failed` with method/domain/cause | not implied by central/global peak or fit quality |
| `POINT-FIXED-BRANCH-RESPONSE-STATE` | response conditional on one exact branch, support, weight state, and active constraints | not full-procedure response |
| `POINT-FULL-PROCEDURE-RESPONSE-STATE` | response with all data-dependent search/fallback/support/constraint decisions rerun | not observational pointing accuracy |
| `POINT-OBSERVATIONAL-BIAS-ACCURACY-STATE` | separately empirical centroid-bias/pointing-accuracy authority | unavailable bias is not zero bias |
| `POINT-UNCERTAINTY-STATE` | separately typed formal, coordinate, reference, response, empirical, NOI, and correction uncertainty availability | not an implied total uncertainty |
| unavailable role | typed unavailable quantity/claim/use and reason | not numerical zero or successful fit |

Every requested observation-array-parent-method fit has independent producer
lifecycle. Every component has an identifiability state. Every named use has
separate request, applicability, eligibility, and realization fields followed
by any prescribed consumer action. `diagnostic_display_only` is an action, not
an eligibility or producer state. Physical co-location in one table or file
does not merge atomicity.

## External Roles

- Cross-array aggregation and correction-record construction are external
  pointing-support-producer operations.
- Correction application is an AST operation.
- Photometric transfer is a CAL/TolProj operation.
- Joint-covariance, astrometric, empirical-repeatability, and NOI uncertainty
  companions are separately versioned external or successor products.

## Named-Use Policy Roles

The author shall define collision-free draft identities and mechanics for:

1. POINT per-array fit completeness;
2. pointing-support displacement admission;
3. telescope/observing QC parameter admission and actions; and
4. CAL/TolProj photometric-transfer amplitude admission.

The author must preserve the owner of each policy, use-specific evaluation
axes and actions, and VAL's registry/evaluator-only role. No universal
eligibility flag or POINT aggregate profile is permitted.
