# SCI-MAP v0.2/r0.4 product-realization and attempt-outcome amendment

Status: **CANDIDATE scientific/ECS record model; no application attempt or
product is asserted**.

MAP product realization exists only when actual immutable accumulators,
scientific output rows, product values, role-qualified companion statuses,
product identity, and provenance were produced. The application-attempt
outcome is a separate field with exactly `succeeded`, `failed`, or
`not_produced`.

| Situation | Reached lifecycle | Attempt outcome | Product realization | Required retained evidence |
| --- | --- | --- | --- | --- |
| Requested product produced | Through `realized` | `succeeded` | Exact immutable product identity | All five stages, product contents/statuses, and provenance |
| Stop before application | Reached prefix through at most `observation_resolved` | No applied-attempt outcome | Absent | Exact stopping stage and cause; no future-stage placeholder |
| Post-application terminal failure | Through `applied` | `failed` | Absent | Applied-attempt identity, failure stage, cause, affected scope, and nonmutation evidence |
| Valid applied policy with no authorized output rows | Through `applied` | `not_produced` | Absent | Exact empty-support evidence specified by the owner disposition |

An unavailable response or covariance status inside a permitted realized base
bundle is a realized companion-status fact. It is neither a producer failure
nor a failed MAP attempt. An immutable failure/evidence artifact is evidence
about an attempt and does not itself realize a MAP product. ECS requirement
result and evidence-artifact realization remain two further independent axes.
