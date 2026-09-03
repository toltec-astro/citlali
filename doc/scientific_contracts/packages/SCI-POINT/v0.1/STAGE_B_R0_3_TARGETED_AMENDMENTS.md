# SCI-POINT v0.1 r0.3 targeted amendments

This companion is a navigation record. The shared LaTeX common core remains
the sole normative scientific source. If this summary and the common core ever
differ, the common core governs.

## 1. SCI-VAL disposition and source binding

POINT authors no independent SCI-VAL function. Its crosswalk is subordinate to
the exact r0.3 owner directive. `null` below means no proposition, not a fourth
enumeration value.

| Condition | Request | Applicability | Eligibility | Realization |
|---|---|---|---|---|
| Not requested | `not_requested` | null | null | `not_produced` |
| Requested, known inapplicable, artifact written | `requested` | `inapplicable` | null | `realized` |
| Missing profile/source or unresolved scope, artifact written | `requested` | `applicability_unknown` | `decision_unavailable` | `realized` |
| Applicable, decisive exclusion, artifact written | `requested` | `applicable` | `ineligible` | `realized` |
| Applicable, required permission unresolved, artifact written | `requested` | `applicable` | `decision_unavailable` | `realized` |
| Applicable, all permissions present, artifact written | `requested` | `applicable` | `eligible` | `realized` |

An incomplete, failed, or absent decision artifact has realization
`incomplete`, `failed`, or `not_produced` and makes no authoritative eligibility
assertion. A structural conflict can be recorded in a realized artifact;
`failed` means artifact production failed.

## 2. Response-family and composition crosswalk

- Fixed differential POINT response: `R_theta,m^fixed`.
- Fixed parent response: `R_m,source^fixed`.
- Fixed source-domain composition:
  `R_theta,source^fixed = R_theta,m^fixed R_m,source^fixed`.
- POINT full procedure with parent held fixed:
  `R_theta,m^(POINT-FP | parent-fixed)`.
- Whole-chain full procedure: `R_theta,source^chain-FP`, available only with
  authority to rerun the entire upstream parent producer and POINT procedure.

The three scopes never alias. A finite-difference result is not multiplied as a
Jacobian without an exact composition theorem.

## 3. Full-procedure response comparability

Full-procedure response subtracts comparable numerical `theta_hat_P` outputs,
not lifecycle or record objects. Comparability requires the same intended
source, compatible association, identical roles, compatible gauge/order,
compatible units/basis, available target components, and an exact relation
between perturbation domains. Otherwise the response is unavailable or a typed
discontinuity/state transition. The record binds baseline/perturbed products
and states, perturbation source/domain/direction/unit/epsilon, finite or analytic
convention, branch changes, gauge comparison, component availability, cause,
and provenance.

## 4. Signed-versus-magnitude diagnostics

- `POINT-FORMAL-AMPLITUDE-STANDARDIZATION@1 = Ahat/sigma_A` is signed.
- `POINT-DYNAMIC-RANGE-DIAGNOSTIC/FITTED-AMPLITUDE-OVER-FULL-MAP-RMS@1`
  retains the sign of `Ahat` and is not universally nonnegative.
- `POINT-FORMAL-AMPLITUDE-MAGNITUDE-STANDARDIZATION@1 = abs(Ahat)/sigma_A`
  is distinct and not admitted without separate request and authority.
- A legacy alias requires proof of mathematical identity and explicit method
  approval.

## 5. Response status and product dependency

Every route binds an exact parent-response identity and status, which may be
unavailable. A numerical response is required only if the compatibility method
or declared claim requires it. Unavailable response can coexist with a fit,
amplitude, centroid, or displacement and blocks only dependent claims.

## 6. Base fit versus downstream decisions

Only the POINT completeness/publication decision participates in base fit
completion. Pointing-support, telescope-QC, and CAL/TolProj decisions are
separate requested child artifacts. Each references the immutable fit and owns
its profile, SCI-VAL tuple, lifecycle, cause, and provenance; it cannot mutate
the fit, create a new fit generation, or rescue another use.

## 7. Exact fit-weight mapping

`W_fit` is the exact object consumed by the fit. Uniform, reliability,
covariance, inverse covariance, or another approved source role is not an alias
for `W_fit`, but an explicit derivation is allowed. Its mapping binds identity,
source/target units, normalization, support, lifecycle state, claim ceiling,
and provenance.

## 8. Exposure and early-stop lifecycle

A published POINT product inherits parent observation/exposure lineage or uses
typed `not_applicable`; POINT creates no exposure. Per-array products from one
observation share exposure/reference terms, and aggregation requires dependence
authority. An early stop preserves only lifecycle records reached, records
terminal unavailable/failed/not-produced state with cause, never fabricates
`applied` or `fit_realized`, and never silently deletes a required row.
