# SCI-FLT-INF contradictions, ambiguities, and unavailable states

Record identity: `SCI-FLT-INF-GAPS v0.1/r0.1`

Status: Stage A owner-review record; absence is preserved rather than repaired

## Contradictions

### `wiener_filter` label versus apparent estimand

The active full path publishes a normalized numerator/denominator field built
from a template, parent weights, and a shaped inverse spectral model. That is
implementation-consistent with a local template-amplitude estimator. No exact
signal prior or posterior covariance was recovered. The historical/internal
mathematics also groups “matched or Wiener” approximations together.

Disposition: do not decide from naming. The scientific owner must choose
template amplitude, posterior reconstruction, both as separate methods, or a
different exact estimand.

### Requested full method versus realized lowpass substitution

The full path can substitute a constant spectral field when the required PSD
is missing or invalid while retaining the requested full-method name.

Disposition: current behavior is inventory evidence, not admissible science.
Future authority must fail closed or use an explicit selector whose output
retains the alternative method identity.

### Numerical zeros versus unavailable/null state

Small denominators are zeroed and output values at those pixels become zero.
No authority establishes that the estimand is scientifically zero there.

Disposition: treat interpretation as unavailable pending owner-approved
support/null/failure semantics. Never sanitize the numerical sentinel as
science.

### Historical edge-fill direction versus current fixed scope

Historical SCI-FLT-001 D001 approved median fill only as a numerical device
with a scientifically eroded region. Current SCI-FLT-FIXED Stage A selects
full-footprint-only and defers fill, taper, truncation, and support
renormalization.

Disposition: the historical decision is precedent for a distinct adaptive
edge package; it does not modify the protected fixed package or authorize a
current INF edge method.

### Historical empirical calibration versus frozen NOI typing

Historical FLT D003 retained one robust global empirical calibration of the
formal spatial pattern, while frozen NOI now sharply separates conditional
second moment, reciprocal scale, covariance, precision, and consumer weight.

Disposition: preserve the old decision as historical scope intent, but require
a new exact FLT/NOI derived-product boundary. No current coefficient promotion
is authorized.

### Registry commit identity versus recoverable Convolve object

The registry names commit `800e8ae433f87d3fb7521fcb1a7fdf1d32532949`,
which is unavailable in the local object database, but states SHA-256
`8d336242...` for the document. The document recovered at
`1bf77eadd7be1d12f285c272bb5f91511a3259f0` has that exact SHA-256.

Disposition: record the original commit as unavailable and the later exact
object/digest as recovered. Do not assert the two commit objects are identical
without the missing object.

## Ambiguities

### Parent coefficient/covariance identity

The full path takes a square root of the published map coefficient field and
uses it inside an inverse-spectral estimator. Frozen MAP authority does not
make that field precision by name. The exact covariance model, if any, is not
identified.

### Spectral state meaning

The spectral field is shaped, clamped, radially interpolated, and normalized.
It is unclear whether it is intended as noise covariance, a relative spectral
weight, an empirical preconditioner, a transfer mask, or a heuristic.

### Template meaning

Available templates include a parent kernel, analytic Gaussian/Airy profiles,
and a high-pass delta. Their normalization, parent-beam relationship, source
model, unit-source convention, and use as filter kernel versus estimator
template are not uniformly specified.

### Denominator meaning

The denominator may be an estimator normalization, a conditional Fisher-like
quantity, a response, a support diagnostic, or a nonprecision coefficient.
Its reciprocal variance interpretation is not authorized.

### Approximation adequacy

Several stopping paths and an internal denominator floor exist. No scientific
tolerance ties realized tail/update summaries to response, amplitude, or
uncertainty error.

### Edge parity

The learned window is shared with members, but the real signal uses an affine
background transformation and members are zero-centered/windowed. It is
unclear what exact transformed-member population corresponds to the published
science product.

### Kernel-response product

The parent kernel is processed under a uniform-weight full path while signal
uses spatially varying weights. It is unclear which response is intended and
whether the resulting kernel is adequate for spatially varying source
response.

### Observation/coadd selection

Runtime chooses observation filtering or coadd filtering from the coadd mode.
No science establishes equivalence, selection rationale, or a composition rule.

### Source/product boundary

Point-source-named planes and downstream Gaussian fits exist, but the boundary
between a map-domain amplitude field, response correction, selected source
amplitude, and catalog inference is not authoritative.

## Unavailable scientific states

| State | Reason unavailable | Consequence |
| --- | --- | --- |
| exact active full-path estimand | no owner selection or independent contract | no package name or product claim |
| genuine Wiener/posterior method | no complete prior/likelihood/operator/posterior specification recovered | no posterior reconstruction product |
| template-amplitude method | reusable GLS core exists, but parent covariance/template/support/response are unresolved | no scientific amplitude field or scalar |
| parent covariance/inverse-noise | MAP/JINC do not supply the inferred precision needed by implementation naming | denominator cannot be called Fisher information or inverse variance |
| template scientific identity | normalization, beam/source role, calibration, and learned/fixed status unresolved | response and units unavailable |
| approximation-qualified operator | no owner-approved truncation/convergence/floor error policy | exact method route unavailable |
| adaptive edge method | current behavior and old policy are not current scientific authority | edge-conditioned scientific support unavailable |
| data-thresholded mode selection | inactive implementation fragment and no method contract | no route or product |
| automatic fallback | no selector authority or realized-method product identity | requested-primary output fails closed |
| source-learned filter | no active route or frozen SRC authority recovered | deferred |
| observation/coadd equivalence | no commutation or population result | separate methods/parents only |
| fixed-state transformed NOI | exact INF owner authority and parity absent | transformed UNC unavailable |
| per-member-relearned NOI | complete learning graph and member method absent | separate method unavailable |
| NOI-informed successor | owner learning/update rule absent | no successor generation route |
| empirical coefficient promotion | frozen NOI boundary not satisfied | no precision/inverse-variance/consumer-weight claim |
| standardized significance | frozen NOI permits only exact conditional-scale standardization and SRC/MODE owns significance | no significance/detection claim |
| public product bundle/VAL profiles | estimand and named uses unresolved | publication unavailable |

## Confidence assessment

| Recovery conclusion | Confidence | Basis |
| --- | --- | --- |
| one combined `SCI-FLT-INF` contract is scientifically incoherent | high | roadmap split rule plus distinct estimand/state/response/lifecycle families |
| active full path is structurally template-amplitude-like | medium-high | direct algebra inspection; scientific intent absent |
| active full path is a complete posterior/Wiener reconstruction | low | no explicit signal prior or posterior covariance recovered |
| current NOI-member application is learned-once/fixed-state | high as implementation observation | state is resolved from real parent and reused for members |
| per-member relearning is currently active | low/negative recovery result | no active route found; absence limited to inspected base |
| adaptive edge operation is scientifically consequential | high | mask/background/window are parent-derived and alter signal/weight/kernel |
| `normalize_errors` is part of the base estimator | low | it occurs downstream through empirical NOI products/coefficient scaling |
| registry's missing commit equals the recovered later commit | unavailable | matching file digest is insufficient to identify missing commit contents |

## Stop conditions

Stage B must not be commissioned while:

- ODQ-001 has no owner answer;
- multiple selected estimands remain in one proposed package;
- required parent covariance, template, or prior authority is absent;
- method substitution can occur without explicit realized identity;
- fixed-state and relearned NOI graphs are not separated; or
- the proposed author inputs contain implementation-derived conclusions.
