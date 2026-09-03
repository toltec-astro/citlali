# SCI-FLT-FIXED v0.1 Product Role And Lifecycle Table

Status: sanitized Stage A author candidate awaiting exact-byte owner approval

## Atomic Product Bundle

| Role | Required content or honest state |
| --- | --- |
| `FLT-PARENT` | Exact immutable MAP observation, MAP coadd, or JINC observation parent identity and digest |
| `FLT-PLAN` | Requested method/purpose, effective selection, and complete externally resolved strict-linear plan |
| `FLT-OPERATOR` | Exact `J_full L_Theta`, kernel/parameters, grids/domains, coefficients, normalization, and generation |
| `FLT-SIG` | Transformed signal on exact scientific row domain `S_out` |
| `FLT-UNIT-BEAM` | Output unit derivation and originating nominal-beam convention |
| `FLT-TRANSFER` | Exact local transfer where defined, or typed unavailable |
| `FLT-RSP` | Exact transformed compatible parent response, or typed unavailable |
| `FLT-MODE` | Null, attenuated, invariant, and phase state where defined |
| `FLT-INFLUENCE` | Input-to-output causal/coefficient relation, distinct from physical exposure |
| `FLT-SUP` | Numerical computability and complete-footprint scientific support |
| `FLT-VALID` | FLT-local scientific validity and causes, distinct from parent validity and downstream eligibility |
| `FLT-COV-FORMAL` | Complete, structured, partial, marginal, or unavailable deterministic covariance state |
| `NOI-UNC[FLT-SIG]` | Optional separately owned NOI attachment to the exact transformed product/generation |
| `FLT-LINEAGE` | Parent, plan, operator, output, response/covariance, lifecycle, cause, failure, and provenance binding |

The atomic bundle contains all required role records even when an allowed
response/covariance role is explicitly unavailable. A missing required record
is not an atomic bundle.

## Lifecycle States

| State | Meaning | Product consequence |
| --- | --- | --- |
| `not_requested` | No accepted request for SCI-FLT-FIXED | No FLT product |
| `requested` | Method requested but not yet resolved | No realized product |
| `effective` | Accepted plan selects SCI-FLT-FIXED and does not disable it | Await exact operator resolution; no realized product |
| `disabled` | Effective plan disables the route | No FLT product; not identity |
| `unavailable` | Required parent/state/permission cannot be supplied | No FLT product; exact causes retained |
| `resolved` | Complete immutable plan/operator state exists | Eligible for input-admission evaluation, not yet a product |
| `applied` | Exact operator has been applied to exact parent | Intermediate until atomic bundle completion |
| `failed` | Required transformation or bundle completion failed | No complete product; failure propagates |
| `realized_identity` | Exact identity operator was requested/resolved/applied | Real separately parented FLT product |
| `realized_zero` | Exact zero operator was requested/resolved/applied | Real transformation with zero signal, not disabled or unavailable |
| `realized` | Complete atomic product and publication decision exist | Exact immutable successor product |
| `superseded` | A later plan/operator/product generation replaces it for a named use | Prior product remains immutable |

## Generation Rules

Changing parent, requested/effective purpose, operator/kernel, parameter,
transfer qualification, normalization, WCS/grid/domain, support/validity,
response/covariance role, lifecycle, or failure policy creates a new
transformation and product generation. A separately attached NOI product is a
new immutable companion and does not mutate FLT or parent claims.

No lifecycle state establishes implementation conformity, validation,
calibration, response/covariance fidelity, performance, readiness, freeze, or
production use.
