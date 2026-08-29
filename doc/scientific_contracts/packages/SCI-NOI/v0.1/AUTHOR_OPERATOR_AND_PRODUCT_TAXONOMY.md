# SCI-NOI v0.1 — Proposed Operator And Product Taxonomy

Status: proposed sanitized Stage B author input; not owner-approved

This taxonomy names roles without selecting an estimator, storage schema,
default, or implementation. Exact identifiers may be refined during Stage B,
but the separations may not be collapsed without owner approval.

## Operator Families

| Proposed family | Logical signature | Scientific role |
| --- | --- | --- |
| `NOI-G/FIXED-STATE` | immutable parent state + declared assignment design → realization ensemble | Estimates the declared randomization law conditional on one realized reduction and learned operator state |
| `NOI-G/RELEARNED` | immutable base inputs + perturbation/assignment design + declared rerun plan → realization ensemble | Attempts to include variation from exactly named learning/resolution steps |
| `NOI-G/RESIDUAL-FIXED` | immutable source-subtracted residual parent + fixed operator + assignment design → residual realization ensemble | Conditional residual-state method; does not prove source-free or physical-noise behavior |
| `NOI-U/EMPIRICAL-SCALE` | declared ensemble + scalar/projection estimand → empirical scale and uncertainty-of-scale state | One named scalar or projected scale; not covariance or significance unless separately established |
| `NOI-U/DIAGONAL-VARIANCE` | declared ensemble + pixel-domain estimator → variance diagonal | Marginal variance only; no off-diagonal or precision claim |
| `NOI-U/STRUCTURED-COVARIANCE` | declared ensemble + domain/model/regularization → structured covariance representation | Stationary kernel, block, spectral, low-rank, sparse, or other exact declared form |
| `NOI-U/PROJECTED-UNCERTAINTY` | declared ensemble + fixed consumer operator → uncertainty for that exact statistic | Captures correlations present in the ensemble on that projection without requiring dense covariance |
| `NOI-U/EMPIRICAL-WEIGHT` | authorized positive uncertainty product + declared inverse operation → marginal, precision, or consumer-effective weight | Separate role for each meaning; never a MAP-facing coefficient by analogy |
| `NOI-Z/STANDARDIZED-SIGNAL` | immutable signal parent + compatible authorized uncertainty scale → dimensionless standardized signal | “Standardized by the stated empirical scale” unless stronger null/selection authority exists |

`NOI-G/RELEARNED` is a family name, not one algorithm. Every distinct set of
rerun learning steps is a separately versioned method. Fixed and relearned
members cannot be pooled into one Family U ensemble.

## Product Roles

| Product role | Required identity | Explicit non-meaning |
| --- | --- | --- |
| Realization plan | requested/effective method, conditioning, assignment, count, persistence | Not realized data or adequacy |
| Assignment design | coherence units, balance/pairing, joint law, seed/key, cardinality | Not sample validity or uncertainty |
| Realization member | ensemble, member ID, parents, operator route, unit, support, availability/QC | Not independently observed sky or physical-noise draw by default |
| Realization ensemble | ordered completed member identities and joint design | Not variance/covariance merely by existence |
| Realization availability/QC | completion, support, reconstruction, duplicate/complement, failures | Not producer validity or consumer adequacy |
| Empirical center/second moment | exact centering/reference and finite normalization | Not covariance unless its mean/design conditions apply |
| Empirical variance | target law, estimator, squared unit, domain, limitations | Not precision, MAP coefficient, or significance |
| Empirical covariance | target law, representation, domain, omitted terms, regularization | Missing terms are not zero or independence |
| Empirical marginal inverse variance | inverse of one authorized marginal variance on valid domain | Not a full precision matrix |
| Empirical precision | exact inverse/generalized inverse on a declared subspace | Not `1/diag(C)` by default |
| Consumer-effective weight | inverse projected variance for one named operator | Not portable to another estimator |
| Standardized signal | exact numerator, empirical scale, compatibility, support, claim class | Not an uncertainty estimate or calibrated significance by itself |
| Persistence/reconstruction record | persisted/transient/streamed state and exact reconstruction capability | Not statistical adequacy |

## Covariance Representation States

The following are peer representation states, not an ordered requirement that
every product reach the last row:

1. unavailable, with cause and consequences;
2. retained empirical ensemble;
3. diagonal marginal variance;
4. stationary or kernel summary;
5. block, spectral, low-rank, sparse, or other structured covariance;
6. fixed consumer-projected uncertainty; and
7. full covariance or declared precision on an exact domain.

Every state records meaning, domain, support, units, parents, estimator,
limitations, and unavailable terms. A map or JINC parent remains valid within
its own claims when NOI covariance is unavailable.

## Standardized-Signal Claim Classes

| Claim class | Minimum meaning |
| --- | --- |
| `empirical-scale-standardized` | Signal divided by or multiplied by the exact stated empirical scale on a compatible domain |
| `null-standardized` | Additionally has an authorized fixed-statistic null distribution |
| `search-calibrated` | Additionally accounts for the exact search, selection, boundary, and multiplicity procedure |
| `detection-probability-qualified` | Additionally has an authorized decision model and validation for the stated probability claim |

Stage A proposes only the first class as the ordinary truthful ceiling unless
later authority justifies a stronger class.
