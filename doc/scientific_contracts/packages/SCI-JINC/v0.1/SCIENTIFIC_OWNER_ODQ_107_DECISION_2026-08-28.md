# SCI-JINC-ODQ-107 — Scientific-Owner Disposition

Status: owner approved; bounded Stage A disposition

Scientific owner: Grant Wilson

Decision date: `2026-08-28`

## Approved Scientific Disposition

SCI-JINC v0.1 has one fixed, closed per-array observation-bundle schema. Every
produced bundle contains exactly the five scientific roles below. This schema
is derived from the recovered signed-estimator identity, its accepted formal-
support rule, and `SCI-JINC-ODQ-104`; it is not derived from implementation.

| Role | Mathematical quantity | Use in the JINC map | Base-v0.1 status |
| --- | --- | --- | --- |
| `jinc_signal_numerator` | `N_p=sum_i I_ip omega_i kappa_ip z_i` | Signed signal numerator. It combines with `C_p` to form the published map. | **Required** |
| `jinc_signed_normalization` | `C_p=sum_i I_ip omega_i kappa_ip` | Signed normalization and denominator in `m_p=N_p/C_p`; finite negative values are admissible and exact/unresolved cancellation is locally invalid. | **Required** |
| `jinc_quadratic_accumulator` | `Q_p=sum_i I_ip omega_i kappa_ip^2` | Required by the accepted local formal-support/conditioning rule (`Q_p>0`) and retains the distinct quadratic statistic. It is not by itself precision, variance, exposure or validity. | **Required** |
| `jinc_map` | `m_p=N_p/C_p` on accepted local JINC support | Published JINC map in signal unit `U`, carrying its ordinary pixel-level support/validity state. That local state is not whole-product unavailability and is not a separate role-availability object. | **Required** |
| `jinc_coefficient_squared_time` | `T_p^(kappa^2)=sum_i I_ip kappa_ip^2/f_s,i` | Authorized method-specific coefficient-squared temporal support in seconds; it does not determine map validity or physical exposure. | **Required** |

The accepted cancellation rule still requires evaluating

```text
rho_p = abs(C_p) / sum_i I_ip abs(omega_i kappa_ip)
```

against a floating-point error bound derived from the realized summation
method and contributor count. The absolute-term sum, contributor count, bound
and diagnostic value are required construction state where needed to apply
that rule; they are not persistent base-v0.1 bundle roles. Their exact
numerical policy remains `SCI-JINC-ODQ-109`. This follows the recovered owner
decision that no per-pixel diagnostic product is required.

Every required role spans the exact destination geometry even though some
pixels may have zero, insufficient, cancelled or otherwise invalid local
support under existing JINC rules. Those local states do not make the whole
role unavailable. If any one of the five required whole-product roles cannot
be formed, the affected array observation bundle fails closed: publish no
partial bundle and synthesize no placeholder role.

No base-v0.1 per-role availability object, generic optional or conditional-
required machinery, detailed missing-product cause vocabulary, per-pixel or
per-contribution provenance, operational-reason archive, placeholder role or
diagnostic product is authorized. The ODQ-106 observation/array/JINC-
realization/destination identity identifies the bundle; operational debugging
may be logged without becoming a required scientific product.

## Outside Or Deferred Roles

Standalone formal-support/availability planes, formal weight or variance,
full covariance, response companions, empirical noise/significance, physical
exposure, diagnostics and generalized provenance products are outside or
deferred from base v0.1 until a concrete scientific use is separately
authorized. The recovered response/covariance mathematics remains preserved
as future scientific reference, but creates no base-v0.1 product.

## Stage Consequence

`SCI-JINC-ODQ-107` is closed for base-v0.1 product schema and also defers
`SCI-JINC-ODQ-108` response/covariance products pending a concrete scientific
use. The decision changes sanitized Stage A author-control bytes and remains
subject to renewed exact-byte approval under `SCI-JINC-STAGE-A-Q002`. It does
not launch Stage B, create an availability/provenance framework, prescribe
implementation representation, modify implementation, or establish
conformity, validation, achieved performance, readiness or production status.
