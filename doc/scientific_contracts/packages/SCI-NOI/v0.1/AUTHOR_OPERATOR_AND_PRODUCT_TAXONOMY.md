# SCI-NOI v0.1 — Collision-Free Operator And Product Taxonomy

Status: proposed sanitized Stage B author input; exact bytes await owner approval

This taxonomy names roles without selecting an implementation, storage schema,
or hidden default. The semantic prefixes are normative within the proposed
packet:

- `NOI-GEN`: realization-ensemble generation;
- `NOI-UNC`: empirical uncertainty inference; and
- `NOI-STD`: derived signal standardization.

Bare `G`, `U`, and `Z` are prohibited as NOI family names. They collide with
MAP projection notation, the PTC removed component, and the PTC transformed
sample. `Z_i^PTC` remains reserved exclusively for the transformed PTC sample.

## Operator Families And Exact DAGs

| Proposed method family | Logical signature | Scientific role and present availability |
| --- | --- | --- |
| `NOI-GEN/FIXED-STATE-CONDITIONAL-SIGN` | immutable earliest parent + assignment design `S` + exact held-fixed operator state `Theta_0` -> realization ensemble | Estimates the declared sign-randomization law conditional on one realized reduction and learned operator state; proposed ordinary base method, pending owner approval |
| `NOI-GEN/RELEARNED` | immutable earliest parent + assignment design `S` + exact named learn/resolve replay -> realization ensemble | Attempts to include variation from the exact replayed learning procedure; typed unavailable until a complete rerun graph is owner-authorized |
| `NOI-GEN/RESIDUAL-FIXED` | immutable source-subtracted residual parent + fixed operator + assignment design `S` -> residual realization ensemble | Separate conditional residual method; unavailable until an exact source/FRUIT boundary exists |
| `NOI-UNC/EMPIRICAL-SCALE` | admitted ensemble + scalar/projection estimand -> empirical scale and uncertainty-of-scale state | One declared scalar or projected scale; no covariance or significance claim by existence |
| `NOI-UNC/DIAGONAL-VARIANCE` | admitted ensemble + exact estimator/domain -> marginal variance diagonal | Marginal variance only; no off-diagonal or precision claim |
| `NOI-UNC/STRUCTURED-COVARIANCE` | admitted ensemble + exact domain/model/regularization -> structured covariance | Stationary/kernel, block, spectral, low-rank, sparse, or another exact declared form |
| `NOI-UNC/PROJECTED-UNCERTAINTY` | admitted ensemble + fixed consumer operator/statistic -> uncertainty for that exact statistic | Captures correlations represented by the ensemble on the named projection without requiring dense covariance |
| `NOI-UNC/EMPIRICAL-INVERSE-OR-WEIGHT` | authorized positive UNC product + exact transform -> marginal inverse variance, precision, or consumer-effective weight | Every meaning is a separate product; none is a MAP-facing coefficient by analogy |
| `NOI-STD/EMPIRICAL-SCALE-STANDARDIZED-SIGNAL` | immutable signal parent + compatible authorized positive signal-unit scale -> dimensionless standardized signal | Ordinary claim is only “standardized by the stated empirical scale” |

For fixed-state generation, every member `b` has the exact graph

```text
realization_b = O_Theta0(R_b(parent))
```

where `parent` is the earliest immutable object to which assignment operator
`R_b` is applied, and `O_Theta0` is the complete externally owned held-fixed
operator graph. For relearned generation, every member has

```text
Theta_b       = LearnResolve_b(R_b(parent))
realization_b = O_Theta_b(R_b(parent)).
```

`Theta_b`, the exact stages rerun, their inputs, changed-state record, response,
and failure state are part of the member identity. A later UNC calculation,
filter choice, selection, or STD operation cannot retroactively modify an
earlier GEN member. Fixed-state and relearned members, and relearned members
with different replay graphs, cannot be pooled under one ensemble identity.

## Exact Ensemble-Design Object

Every GEN method binds the finite assignment design

```text
S = {s_bg},  b = 1,...,B,  g in the declared coherence-unit set,
```

plus its law or deterministic construction. `s_bg` is an assignment value,
not sample validity, a MAP/PTC coefficient, or a random physical-noise draw.
The design records:

- coherence-unit identity and partition;
- marginal assignment law and complete joint law;
- balance, complement pairing, replacement, duplicate, and cross-observation
  coupling rules;
- seed/key derivation, algorithm/version, stable parent IDs, method ID, member
  ID, and scheduling-independent regeneration rule;
- requested count `B_requested`, effective resolved design count `B_resolved`,
  terminal completed count `B_completed`, unique completed assignment count
  `B_unique`, count admitted by an exact UNC method `B_admitted_for_UNC`, and
  the applicable design rank; and
- the exact completed membership and each atomic member terminal state.

These counts are never aliases. Enabled GEN requires an owner-authorized
positive resolved design and a method-valid completed design. Disabled GEN is
an explicit zero-member/no-work state. The minimum positive cardinality for a
particular UNC estimator is part of that UNC method, not a global default.
Balance, complement pairing, and large `B` do not prove independence or
physical-noise sampling.

## Source-Imprint Specification

Every GEN method declares:

1. signal/source content in the earliest parent;
2. the precise cancellation or suppression target, if any;
3. assumptions under which cancellation is expected;
4. finite-design balance residuals;
5. variation of support, coefficient, projection, filter, or selection state;
6. scan-synchronous and other structured residuals;
7. source-model use and model error;
8. known or bounded leakage; and
9. the resulting permitted and prohibited claims.

Global assignment balance is not pixelwise source cancellation when support,
coefficients, projection, filtering, or membership varies. The ordinary
truthful GEN claim proposed for base v0.1 is
`source_imprinted_conditional_randomization_ensemble`. It is not a repeated
physical-noise ensemble, a calibrated null, or a source-free ensemble.
Residual/source-subtracted and full FRUIT-replayed methods remain separate.

## UNC Target, Estimator, And Covariance Table

Every UNC method binds one row completely; no generic `1/B` or `1/(B-1)` rule
is authorized.

| Field | Required exact declaration |
| --- | --- |
| Target law | Conditional assignment-law scatter/covariance, repeated physical-noise uncertainty, calibrated empirical null, fixed consumer projection, or another named target |
| Admitted ensemble | Exact GEN method/generation, parent graph, completed member set, design, source-imprint state, and member-QC profile evaluations |
| Center | Known target center, estimated empirical center, fitted center, or another exact reference |
| Estimator | Second moment versus covariance; finite-design normalization/correction; missingness and dependence treatment; uncertainty of the estimate |
| Design adequacy | `B_admitted_for_UNC`, unique assignment count, applicable design rank, complements/duplicates, effective information, and use-specific minimum cardinality |
| Domain | Exact pixel rows, projection/statistic, WCS, response reference, support intersection or missing-data domain, and units |
| Representation | Diagonal, retained ensemble, stationary/kernel, structured, projected, full covariance, or unavailable |
| Rank and inversion | Rank, null space, unresolved modes, regularization, inverse/generalized-inverse domain, and inverse-bias treatment |
| Calibration and omissions | External calibration/overlap/independence state; omitted correlations, nuisance, response, source-leakage, selection, and learning terms |
| Claim ceiling | Exact conditional/empirical meaning; physical-noise, calibrated-null, precision, or tail claims only when separately established |

A multivariate covariance uses a common completed member population over its
declared domain or an exact missing-data estimator that states symmetry,
positive-semidefinite, rank, and domain properties. Missing blocks are not
zero or independence. A numerical inverse is not automatically a precision
estimate. Pseudo-realization count is not exposure, independent astronomical
sample count, or evidence that parent-map noise falls as `1/sqrt(B)`.

## STD Numerator, Scale, And Claim Table

| Field | Required exact declaration |
| --- | --- |
| Numerator | Immutable MAP or JINC product, exact estimand, operation, generation, unit, response reference, WCS, support, validity, and lifecycle |
| Scale | Exact authorized UNC product and operation; whether standard deviation, standard error, projected uncertainty, calibrated scale, or another positive signal-unit quantity |
| Transformation | Any square root, projection, calibration, or other exact conversion needed to obtain the direct signal-unit denominator |
| Compatibility | Estimator, response reference, unit/beam, WCS, support, validity, lifecycle, and parent-generation compatibility |
| Dependence | Dependence between numerator and empirical scale and its consequence |
| Local behavior | Zero, negative, nonfinite, unavailable, incompatible, and outside-support handling |
| Product and claim | `empirical_scale_standardized_signal`; dimensionless; ordinarily only “standardized by the stated empirical scale” |

A variance, covariance matrix, inverse variance, precision, or consumer weight
is not itself a direct denominator. It requires an exact method-authorized
transformation to a positive scale in the numerator unit. Studentized
statistics, Gaussian z-scores, N-sigma claims, detection probability,
false-alarm rate, completeness, purity, and catalog decisions require separate
null, selection, search, multiplicity, and validation authority.

## Atomic Product And Lifecycle Roles

| Product role | Required identity and state | Explicit non-meaning |
| --- | --- | --- |
| GEN method/plan | requested/effective/observation-resolved/realized method, exact parent/operator graph, assignment design, persistence plan | Not a realization or adequacy result |
| GEN realization member | ensemble/member ID, exact parent and operator state, payload or reconstruction identity, unit/domain/support, terminal state, QC, causes, provenance | Atomic; not independently observed sky or physical-noise draw by default |
| GEN ensemble | exact completed member identities, joint design, counts, rank, source-imprint state, persistence/reconstruction, lifecycle | Not variance/covariance merely by existence |
| UNC center/second moment | exact target/reference and finite-design estimator | Not covariance unless its target and design conditions establish it |
| UNC variance/covariance | target law, estimator, representation, domain, rank/null, units, omitted terms, uncertainty/calibration, lifecycle | Not precision, MAP coefficient, or significance |
| UNC marginal inverse variance | inverse of one authorized marginal variance on its valid domain | Not a full precision matrix |
| UNC precision | exact inverse/generalized inverse on a declared subspace | Not `1/diag(C)` by default |
| UNC consumer-effective weight | inverse projected variance for one named operator/statistic | Not portable to another estimator or to PTC/MAP |
| STD standardized signal | exact numerator, transformed scale, compatibility, dependency, support, claim class | Not uncertainty or calibrated significance by itself |
| Persistence/reconstruction record | persisted/transient/streamed state, exact regeneration capability, sufficient statistics, audit limitation | Not statistical adequacy |

Each requested operation publishes its complete required atomic product or a
typed unavailable/failed state. A GEN ensemble with failed members may be
realized only when its exact method defines the remaining completed design as
valid; otherwise it is unavailable for UNC. No product automatically realizes
the next operator role.
