# SCI-JINC v0.1 — Response And Covariance Product-Role Table r0.2

Status: implementation-blind Stage B scope-parity artifact; recovered
mathematics preserved, base-v0.1 products deferred by owner-approved ODQ-107

Prepared: `2026-08-29`

Scientific owner: Grant Wilson

## Controlling Scope Rule

`SCI-JINC-ODQ-107`, as incorporated into the exact Q002-approved packet,
fixes the complete base-v0.1 numerical bundle to five required roles:

1. `jinc_signal_numerator`;
2. `jinc_signed_normalization`;
3. `jinc_quadratic_accumulator`;
4. `jinc_map` with its local support/validity state; and
5. `jinc_coefficient_squared_time`.

Response, variance, formal weight, covariance, empirical noise/significance,
standalone companion availability, and limitation-record products are outside
or deferred from that bundle. The conditional mathematics remains available
for future scientific use and for avoiding false interpretations. It does not
create a present product.

The admitted exact PTC boundary nevertheless preserves producer-owned
upstream response and covariance/uncertainty facts when present. The
process-only SCI-VAL registry encodes both as `advisory`: their typed producer
states remain neutrally visible under the admitted JINC policy, but neither
gates this atomic sample-admission decision nor creates a JINC output role.
SCI-VAL authors no JINC scientific policy.

## Required Five-Role Bundle

| Role | Mathematical object | Base-v0.1 disposition | Response/covariance implication |
| --- | --- | --- | --- |
| `jinc_signal_numerator` | `N_p=sum_i I_ip omega_i kappa_ip z_i` | Required numerical plane | Not itself response, covariance, variance, precision, or significance. |
| `jinc_signed_normalization` | `C_p=sum_i I_ip omega_i kappa_ip` | Required numerical plane | Signed denominator only; finite negative values are admissible. |
| `jinc_quadratic_accumulator` | `Q_p=sum_i I_ip omega_i kappa_ip^2` | Required numerical plane | Not itself precision, covariance, variance, exposure, or validity. |
| `jinc_map` | `m_p=N_p/C_p` on accepted local support | Required derived numerical plane with local support/validity | No response/covariance companion follows automatically. |
| `jinc_coefficient_squared_time` | `sum_i I_ip kappa_ip^2/f_s,i` | Required numerical plane | Method-specific temporal accounting only; not weight, variance, covariance, or physical exposure. |

Failure to form any required whole-product role suppresses the array bundle.
Local invalid pixels remain ordinary `jinc_map` content; no per-role
availability plane is added.

## Response Families

| Family | Exact retained meaning | Required authority if a future product is requested | Base-v0.1 product state | Prohibitions and omitted claim |
| --- | --- | --- | --- | --- |
| Fixed-state JINC response | `R_JINC,fixed=A_JINC H_PTC,fixed`, holding exact signal membership, PTC coefficients, parameters, phase, square placement, edge crop, normalization, conditioning, and output rows fixed. A realized PTC-grid companion enters at JINC and receives the JINC operator exactly once. | Exact upstream response parent/domain plus the complete fixed JINC operator and row identity. | Outside/deferred; no product or companion-availability role. | Not a bare analytic JINC, generic beam, measured PSF, achieved response, hidden membership subset, double application, or inferred normalization. |
| PTC full-procedure finite difference with JINC fixed | Rerun the exact authorized PTC procedure under its declared perturbation while holding the exact JINC operator and membership fixed, then difference the JINC outputs under the declared convention. | Exact PTC procedure-response authority, perturbation, state-change record, and complete compatible JINC fixed state. | Outside/deferred. | Does not re-resolve JINC admission, coefficient, parameters, phase, support, cancellation, or destination state. |
| JINC re-resolved procedure response | Under a declared perturbation, re-resolve every authorized JINC-dependent admission, coefficient, parameter, phase, support, edge, cancellation, numerical-adequacy, destination, and product state. | Separate procedure authority defining perturbation, re-resolution, discontinuities, domain, lifecycle, and output. | Unavailable in base v0.1. | Must not be labeled fixed-state response or inferred from a single realized operator. |
| Whole-chain RTC-to-CAL-to-PTC-to-JINC response | A separately authorized complete rerun of the named chain with all learning, selection, calibration, PTC, and JINC states governed. | Separate whole-chain authority and complete rerun evidence. | Unavailable in base v0.1. | Partial composition or multiplication of incomparable companions is not this family. |

For every family, response membership is exactly the signal membership that
the family declares. An unavailable or locally invalid JINC pixel never
receives a zero response by substitution. Edge-truncated response, if a future
role authorizes it, uses the actual retained square membership and may be
asymmetric.

## Covariance And Formal-Uncertainty Families

Let `A_JINC` contain the fixed normalized rows
`A_pi=I_ip omega_i kappa_ip/C_p` on their exact available domains.

| Family or view | Exact retained relation and assumptions | Base-v0.1 product state | Required disclosure if later authorized |
| --- | --- | --- | --- |
| Exact transformed covariance | `C_JINC=A_JINC C_PTC A_JINC^T` on the exact declared available upstream block/domain. | Outside/deferred. | Exact upstream block/domain, axes, units, support, lifecycle, membership/operator rows, and every unavailable block. “Exact” applies only to the declared upstream covariance, not complete physical uncertainty. |
| Conditional diagonal formal variance | `Var_formal(m_p)=Q_p/C_p^2` only when the exact PTC family establishes `omega_i=Var(z_i)^-1` for mutually independent admitted occurrences. | Outside/deferred. | The exact inverse-variance and independence authority, population, units, and omitted correlated/other terms. |
| Conditional formal weight | `W_formal,p=C_p^2/Q_p` under the same exact assumptions. | Outside/deferred. | Same assumptions; inverse-square units alone are insufficient. With dimensionless `omega_i`, this is not formal inverse variance. |
| Conditional off-diagonal covariance | `Cov(m_p,m_p')=sum_i I_ip I_ip' omega_i kappa_ip kappa_ip'/(C_p C_p')` under the same diagonal upstream model. | Outside/deferred. | Shared-member identity, actual edge/support overlap, exact rows, assumptions, domain, and omitted correlations. |
| Partial, symbolic, summarized, or lineage-resolvable covariance state | A precisely typed non-complete state tied to the exact producer and lifecycle. | Producer fact may cross the boundary; no JINC product is created. | Exact meaning, known blocks/terms, omitted blocks/terms, resolution route, causes, and no-zero substitution. |
| Empirical noise, covariance, weight, or significance | Separately governed empirical science. | Outside SCI-JINC base v0.1; remains SCI-NOI or another explicit authority. | No inference from `C_p`, `Q_p`, time, hit count, coefficient units, or a formal shortcut. |

Every future covariance or formal-weight product must state whether it is
complete numerical, partial, symbolic, summarized, lineage-resolvable, or
unavailable. Missing covariance is never zero and never establishes
independence.

## Terms That A Future Product Must Not Hide

Any later authorized response/covariance role must state the disposition of,
at minimum:

- correlated detector and atmospheric terms;
- calibration and upstream-response uncertainty;
- PTC selection, nuisance, learned-model, and re-resolution uncertainty;
- coefficient and JINC-parameter uncertainty;
- phase, support, edge, WCS, and numerical-realization uncertainty;
- conditioning and unavailable-row effects; and
- exact lifecycle, parent, source, domain, and compatibility state.

Unknown or omitted terms are neither zero nor proof of irrelevance.

## Compact Replay Record Treatment

The bundle's compact generative record shall bind the immutable upstream
parent/source identities and the static ODQ-107 schema identity. It shall not
copy a producer response/covariance state or create a per-bundle JINC
response/covariance availability state. The contract-level table above, not a
bundle companion record, preserves the deferred mathematics. This bounded
replay metadata does not create:

- a response, covariance, variance, or formal-weight product;
- a generic optional/conditional role mechanism;
- a per-role availability product;
- a generalized provenance framework; or
- an achieved fidelity claim.

## Compatibility And Claim Boundary

A concrete future scientific use may authorize a versioned successor role.
That successor must bind the exact family, domain, parents, membership,
operator, assumptions, omissions, lifecycle, compatibility, and failure
semantics. No product is enabled by name similarity, available upstream data,
implementation convenience, or numerical agreement.

This table performs no response or covariance calculation and makes no
implementation-conformity, representation-fidelity, response-fidelity,
covariance-fidelity, numerical-validation, observational-performance,
readiness, or production claim.
