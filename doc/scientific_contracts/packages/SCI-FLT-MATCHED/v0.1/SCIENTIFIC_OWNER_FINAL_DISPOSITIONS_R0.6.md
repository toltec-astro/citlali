# SCI-FLT-MATCHED v0.1 r0.6 — Scientific-owner final dispositions

Date: `2026-09-01`

Owner: Grant Wilson

Status: final owner disposition; incorporated into the frozen v0.1/r0.6
scientific authority

## Human title and method identity

The selected human title is **Matched-template map amplitude estimation**
(`SODL-014`, option 1). The stable method ID remains `SCI-FLT-MATCHED`.
“Wiener filter” may survive only as historical or implementation terminology;
it does not identify a posterior-sky estimand. Every realization carries an
explicit `optimality_status`. Only `AO-001-A` with every exact local
constrained-GLS premise established may carry
`established_exact_local_GLS`; all weaker authorized realizations carry
`not_claimed`, and a realization lacking the required authority is
`unavailable`.

## AO-001 method authorization

`AO-001-A` and `AO-001-C` are separately package-authorized for ordinary-MAP
observation and coadd parent classes. They may serve the base filtered-
amplitude signal-product role and a named companion role only when that role
explicitly requests the same method identity and satisfies all of its
premises. Every request and realization binds exactly one authorized method.
No realization mixes methods, selects a method from target data, substitutes
one method for another, or falls back automatically.

- `AO-001-A` is the exact local constrained-GLS method only when every
  stochastic model, covariance, conditioning, local-domain, subspace,
  template, constraint, regularization, provenance, and generation premise in
  the contract is exactly bound. Missing authority makes A and its optimality
  and reference-variance claims unavailable; it never supplies a default.
- `AO-001-C` is the separately named
  `radially_symmetrized_field_power_spectral_weighting` method only when its
  exact field source, deterministic-residual imprint, population and state,
  chart and domain, transform, window, radial bins, multiplicity, averaging,
  regularization, support, generation, and diagnostic definitions are bound.
  Otherwise C is unavailable. C establishes no noise, covariance,
  stationarity, isotropy, minimum-variance, or optimality claim. Its required
  effective-sample, anisotropy/dispersion, and leakage diagnostics are evidence
  fields, not admission thresholds.
- `AO-001-B` and `AO-001-D` remain closed successor-authorship triggers and are
  not selectable v0.1 methods. Each requires its own concrete scientific
  contract before use.

The authorizations above define eligible parameterized scientific methods;
they do not assert that any concrete numerical realization, parent, or
implementation is available or conformant.

## Remaining SODL dispositions

| SODL | Final disposition |
| --- | --- |
| `001` | Close by the A/C package, parent-class, named-use, and exactly-one-method-per-realization authorization above. B/D remain successor triggers; mixing, target-data choice, substitution, and fallback are forbidden. |
| `002` | Close by parameterizing A over the complete exact premise set. A, its optimality status, and `v_GLS,reference=d_p^-1` are unavailable unless every premise is bound for the realization. |
| `003` | Close deferred: B is a nonselectable successor-authorship trigger. |
| `004` | Close by parameterizing C over the complete exact named field-power state and diagnostics. Missing authority makes C unavailable and implies no noise/covariance/isotropy/stationarity/optimality claim. |
| `005` | Close deferred: D is a nonselectable successor-authorship trigger. |
| `006` | Preserve the decided exact-operator versus engineering-numerics distinction. |
| `007` | Preserve the supersession of former approximation questions. |
| `008` | For observation and coadd parents, the base signal-product role requires no covariance companion and records the applicable named role as `AO-003-C` unavailable unless that role is explicitly requested. `CU` requires its named `AO-003-A` complete or `AO-003-B` projected covariance scope and exact authority; failure blocks `CU`, not the otherwise valid signal product. There is no fallback between scopes. |
| `009` | No joint U1+U2 total covariance is authorized in base v0.1. U2 CAL/BEAM/template random variables and cross-covariances are unavailable unless separately authorized by a successor or exact future amendment. |
| `010` | Freeze the finite `Q_FLT^0.1` state-query vocabulary defined in the contract. The queried state must remain exactly available or reconstructable through lineage. Exact representation is an engineering declaration; unsupported queries are unavailable and require successor authority. |
| `011` | Freeze the fixed, reference, and operational-realized response identities, domains, validity rules, and finite query vocabulary defined in the contract. Fixed-point response is unavailable without exact rerun authority. Exact representation is engineering-only; unsupported queries are unavailable. |
| `012` | Preserve the decided seven role meanings, dependency graph, and science/engineering separation. |
| `013` | The base `SP` role requires no optional covariance, NOI, response, or handoff companion beyond its required signal member. `CU` requires its covariance companion, `NU` its transformed-NOI companion, `RU` its response companion, and `FH` its handoff envelope. Failure blocks only the corresponding named use unless an exact later policy says otherwise; it does not change the estimand. |
| `014` | Select option 1, **Matched-template map amplitude estimation**, with optimality bounded exactly as stated above. |
| `015` | A point-source-flux use is authorized only when the exact unit-amplitude convention and CAL, BEAM, template, and calibration lineages are bound. Otherwise the output is only the declared template-shape amplitude, and literal flux interpretation is unavailable. |
| `016` | No additional intermediate is a public science product in v0.1. Any future intermediate requires an exact role, meaning, unit, validity, lifecycle, named-use policy, and failure contract. |
| `017` | Freeze immutable coexistence and explicit supersession for bundles, companions, generations, profiles, and successor FLT envelopes. Complete `Q_FLT^0.1` query state remains exactly available or reconstructable through lineage. Unsupported queries require successor authority. This scientific contract sets no wall-clock retention minimum; operational retention belongs to governance outside this contract. |

## Engineering declarations and typed unavailability

After the scientific decisions above, concrete AO-002 numerical status and
AO-003/004/005/006 representations remain engineering declarations. A
concrete realization that lacks its exact declaration is typed unavailable;
the freeze does not infer a representation, numerical profile, registration,
or implementation state.

## Freeze nonclaims

These dispositions do not establish an available MAP parent, numerical
weighting realization, registered SCI-VAL profile, implementation conformity,
numerical adequacy, response or covariance fidelity, detection performance,
observational validation, readiness, production suitability, production
authorization, or Unity activity. They do not authorize posterior/Wiener sky
reconstruction, convolution equivalence, source detection or catalog behavior,
FLT-owned coaddition, SRC ownership, or FRUIT science.
