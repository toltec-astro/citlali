# SCI-FLT-FIXED v0.1 Operator And Product Taxonomy

Status: sanitized Stage A author candidate awaiting exact-byte owner approval

## Package Identity

`SCI-FLT` is the tranche. `SCI-FLT-FIXED` is the first package. `SCI-FLT-INF`
is only a holding tranche and has no combined Stage B authority.

## Admitted Operator Family

### `FLT-FIXED-LINEAR`

One complete externally resolved fixed linear same-grid operator:

\[
  y = J_{\rm full}L_\Theta m.
\]

Every coefficient, parameter, grid/domain fact, normalization, support rule,
and lifecycle state is frozen before application. No additive term exists.

### `FLT-FIXED-CONV`

A structured `FLT-FIXED-LINEAR` method in which `L_Theta` is constructed from
one exact finite sampled convolution kernel. The complete finite operator and
full-footprint row selection remain the scientific transformation.

### `FLT-FIXED-CONV-LOWPASS`

A qualified subtype of `FLT-FIXED-CONV`. The low-pass claim exists only when
the exact frequency domain/metric, DC gain, passband, transition, stopband or
attenuation, phase, anisotropy, finite-grid/edge limitations, kernel,
normalization, and parameter provenance are complete. Without them, the
operator remains fixed convolution and low-pass is unavailable.

## Deferred Inference-Bearing Identities

The following are not SCI-FLT-FIXED methods and do not share one contract by
default:

- Wiener reconstruction/transformation;
- matched or generalized least-squares template-amplitude estimation;
- source-learned operation;
- data-derived spectral/mode selection or map-domain destriping;
- automatic method selection; and
- per-member relearning.

A frozen realized operator remains outside SCI-FLT-FIXED when its scientific
method/estimand depends on an inferred signal/noise model, prior,
regularization, learned state, or latent/template amplitude.

## Parent Roles

| Role | Parent |
| --- | --- |
| `FLT-PARENT-MAP-OBS` | One complete base/unfiltered MAP observation bundle |
| `FLT-PARENT-MAP-COADD` | One complete base/unfiltered MAP centered-integer coadd bundle |
| `FLT-PARENT-JINC-OBS` | One complete atomic JINC observation bundle |

Each method binds exactly one role. Shape or WCS equality cannot substitute
roles. SCI-FLT-FIXED does not coadd.

## Product Roles

| Role | Meaning |
| --- | --- |
| `FLT-PLAN` | Requested purpose, effective selection, and exact externally resolved fixed-linear plan |
| `FLT-OPERATOR` | Complete finite `J_full L_Theta`, including kernel/coefficients and generation |
| `FLT-SIG` | Transformed parent-map quantity on exact `S_out`; not automatic photometry or fitted amplitude |
| `FLT-UNIT-BEAM` | Output-unit derivation plus originating nominal-beam identity |
| `FLT-TRANSFER` | Local sampled transfer where scientifically defined, or unavailable |
| `FLT-RSP` | Exact transformed compatible parent response, or unavailable |
| `FLT-MODE` | Null, attenuated, invariant, and phase state |
| `FLT-INFLUENCE` | Parent-row influence; not physical exposure |
| `FLT-SUP` | Numerical and complete-footprint support facts |
| `FLT-VALID` | FLT-local validity/causes; not parent or downstream validity |
| `FLT-COV-FORMAL` | Complete, structured, partial, marginal, or unavailable deterministic covariance state |
| `NOI-UNC[FLT-SIG]` | Separately owned NOI empirical uncertainty attachment |
| `FLT-LINEAGE` | Immutable parent/plan/operator/output/companion/lifecycle/failure binding |

## Identity Rules

Changing the parent role/generation, requested or effective purpose, operator/
kernel, parameter, transfer qualification, normalization, WCS/grid/domain, row
selection, support/validity, response/covariance role, lifecycle, or failure
policy creates a different transformation and product generation.

Disabled, unavailable, failed, realized identity, and realized zero are
different states. Disabled produces no product. Identity and zero operators
produce separately parented realized FLT products.

## Forbidden Collisions

- `DET` is not used as the package abbreviation.
- Low-pass is not a synonym for smoothing.
- A source-shaped convolution is not a matched estimator.
- A kernel is not automatically a source response or PSF.
- A denominator/weight is not automatically precision.
- A marginal variance plane is not full covariance.
- FLT-local validity is not downstream eligibility.
- Influence is not exposure.
