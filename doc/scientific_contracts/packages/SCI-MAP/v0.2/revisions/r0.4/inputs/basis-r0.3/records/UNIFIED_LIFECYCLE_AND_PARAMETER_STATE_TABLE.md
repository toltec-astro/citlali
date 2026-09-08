# SCI-MAP v0.2/r0.3 unified lifecycle, product, and attempt-state table

Status: **CANDIDATE for scientific-owner disposition**. The normative clauses
are in `../src/common/definitions.tex`, `equations.tex`, and
`requirements.tex`; this table is a review aid and creates no second
authority.

## Ordered MAP lifecycle

| Order | Exact state | Meaning | Record rule |
| ---: | --- | --- | --- |
| 1 | `requested` | Immutable accepted intent. | Record the request identity, owner, scope, value or role, cause, and provenance. Never reconstruct it from a later stage. |
| 2 | `effective` | Supported executable interpretation of the request. | Bind the selected supported, disabled, or typed-unavailable interpretation without mutating the request. |
| 3 | `observation_resolved` | Every observation-specific parent, coefficient, profile, geometry, mapping, parameter, support rule, output role, and persistence decision is concrete or typed unavailable. | This is the exact value/state used for pre-application validation. |
| 4 | `applied` | The exact resolved operation and parameter state was invoked or committed for one execution attempt. | Create only at the application event. Retain it with an exact terminal failure if output realization does not occur. |
| 5 | `realized` | Actual immutable accumulators, scientific output rows, product values, role-qualified companion statuses, product identity, and provenance were produced. | Create only for an actual MAP product. A failure/evidence record is not product realization and never authorizes the earlier application. |

The MAP plan, coefficient selection and application, projection, support
policy, output plan, coadd plan, response/covariance request, selected product
role, persistence decision, and `coverage_cut` use this order.

## Reached-state rule

| Situation | Allowed MAP records | Separate attempt outcome |
| --- | --- | --- |
| Completed product realization | All five states, in order, with exact values, product identity, and provenance. | `succeeded` |
| Stop before application | Only the reached prefix through at most `observation_resolved`; no `applied` or `realized` record and no future-stage placeholder. | No applied-attempt outcome |
| Applied attempt that fails before output realization | Reached prefix through `applied`, exact attempt identity, failure stage, cause, affected scope, and nonmutation evidence; no `realized` product or completion marker. | `failed` |
| Applied valid policy with no support-authorized output rows | Reached prefix through `applied`, exact empty-row proof and retained plan/support evidence; no `realized` product or completion marker. | `not_produced`; cause `no_support_authorized_output_rows` |

Every stage transition binds the exact predecessor. A later stage cannot
rewrite an earlier stage. The existence of a numerical value does not prove
that its stage was reached.

## `coverage_cut` binding

| State | Exact binding |
| --- | --- |
| `c_requested` | Accepted requested value and request identity. |
| `c_effective` | Supported interpretation of that request. |
| `c_observation_resolved` | Exact observation-specific value and typed authorization facts used by pre-application validation. |
| `c_applied` | Application-event binding, created only when the resolved support policy is invoked; it equals the resolved value for that attempt. |
| `c_realized` | Value recorded only in an actually realized immutable output; it equals the applied value for that realization. |

Pre-application admission evaluates `c_observation_resolved` and the resolved
typed override fact. It never requires `c_realized`. Inclusive support
thresholds use `c_applied` after the application event.

## Separate axes

The following are independent and must not be collapsed:

| Axis | Exact vocabulary |
| --- | --- |
| MAP operation/parameter lifecycle | `requested`, `effective`, `observation_resolved`, `applied`, `realized` |
| MAP application-attempt outcome | `succeeded`, `failed`, `not_produced` |
| MAP product realization | Exact immutable product identity and contents, or absent |
| PTC/VAL request | `requested`, `not_requested` |
| PTC/VAL applicability | `applicable`, `inapplicable`, `applicability_unknown` |
| PTC/VAL eligibility | `eligible`, `ineligible`, `decision_unavailable` |
| PTC/VAL decision-artifact realization | `realized`, `incomplete`, `failed`, `not_produced` |
| ECS requirement result | `pass`, `fail`, `blocked`, `not_applicable`, `not_assessed` |
| ECS evidence-artifact realization | `complete`, `incomplete`, `failed`, `not_produced` |

The identical spelling `requested`, `realized`, or `failed` on different axes
does not make the values interchangeable. An unavailable response or
covariance status in a permitted realized base bundle is a companion-status
fact, not a failed attempt.
