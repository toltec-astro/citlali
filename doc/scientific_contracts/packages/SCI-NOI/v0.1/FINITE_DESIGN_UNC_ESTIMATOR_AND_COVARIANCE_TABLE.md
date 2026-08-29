# SCI-NOI v0.1 — Finite-Design UNC Estimator And Covariance Table

Artifact identity: `SCI-NOI_FINITE_DESIGN_UNC_TABLE v0.1/r0.3`

Status: proposed sanitized Stage A scientific input; exact bytes await owner
approval

Every `NOI-UNC` method binds one complete row set below. No universal `1/B`,
`1/(B-1)`, independence, physical-noise, or asymptotic rule is authorized.

| Field | Required exact declaration | Unavailable or prohibited inference |
| --- | --- | --- |
| Target law | Conditional assignment-law scatter/covariance, repeated physical-noise uncertainty, calibrated empirical null, fixed consumer projection, or another named target | GEN mechanism or filename cannot select the target |
| Admitted ensemble | Exact route-specific GEN method/generation, earliest parent and operator graph, complete admitted-member set with every member successfully realized, design, source imprint, and member-admission evaluations | Rejected construction candidates are not members; requested or merely completed survivors from a failed GEN ensemble cannot be admitted |
| Center | Known target center, estimated empirical center, fitted center, or another exact reference | Centering cannot be inferred from a divisor |
| Estimator | Second moment versus covariance; exact design probabilities/weights; finite-design normalization/correction; missingness and dependence treatment; uncertainty of the estimate | No universal `1/B` or `1/(B-1)` |
| Design adequacy | `B_admitted_for_UNC`, `B_unique`, complement-pair count, exact design rank, null space, effective information, and method-specific minimum cardinality | Count, balance, complements, or uniqueness do not prove independence |
| Common domain | Exact member population for every covariance entry, pixel/row identities, WCS, support, response reference, unit/beam, and lifecycle | Pairwise numerical availability cannot silently create different entry populations |
| Missing-data method | When common membership is absent: exact estimator, symmetry rule, PSD property, rank/domain behavior, weighting, and unavailable entries | Missing covariance is not zero or independence |
| Representation | Retained ensemble, diagonal variance, stationary/kernel, block/spectral/low-rank/sparse structured covariance, fixed projection, full covariance, or unavailable | No representation is promoted by storage shape |
| Rank and inverse | Exact rank/null space, unresolved modes, regularization, inverse/generalized-inverse operator and subspace, inverse-bias treatment, and conditioning | A numerical inverse is not automatically statistical precision |
| Calibration and omissions | External calibration/overlap/independence state; response, nuisance, source leakage, learning/selection, and other omitted terms | “Empirical” does not mean calibrated or physical |
| Claim and use | Exact conditional/empirical meaning, allowed consumers/statistics, and use-specific adequacy | One-use adequacy does not authorize another covariance or tail claim |

## Approved Initial Estimand

ODQ-105B authorizes a zero-centered conditional detector-sign-randomization
second moment. For one exact all-members-successful ensemble, define

```text
D_common = {p : every admitted realization M_b supplies a valid finite value at p},
V_hat_cond(p) = sum_b omega_b M_b(p)^2,
sum_b omega_b = 1.
```

The finite design supplies the exact nonnegative `omega_b`; equal weights are
not inferred. The center is the known design target zero, so the finite
ensemble mean is not subtracted and no `B-1` correction applies. Residual source
imprint and structured nonzero content therefore remain in the second moment.
Outside `D_common`, the initial estimator is unavailable; it does not silently
use a smaller member subset.

The product records assignment dependence, complement structure, every member
count, exact design rank, use-specific effective information, and uncertainty
of `V_hat_cond` or an explicit unavailable state. It has squared signal units
and is not automatically physical-noise variance or MAP covariance. A square
root, projection, off-diagonal covariance, inverse, weight, or standardized
signal is a separately identified transformation/product.

ODQ-106 still governs additional covariance representations, domains, rank,
null space, and unavailable states.

Pseudo-realization count is not exposure, independent astronomical count, or
evidence that parent-map noise falls as `1/sqrt(B)`. Additional members may
improve estimation of the declared finite-design target; they do not create
new astronomical data or reduce the immutable parent's realized noise.

Every UNC product is an atomic versioned companion. It either publishes the
complete stated target/estimator/domain/representation/lifecycle or is typed
unavailable/failed. It never modifies the GEN ensemble or its MAP/JINC/PTC
parent. Missing-data estimators may address exact within-product domain
missingness only when authorized; they cannot rescue an admitted-member failure
or construct a survivor ensemble.
