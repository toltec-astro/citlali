# SCI-POINT Producer Lifecycle, Identifiability, And Named Uses

Identity: `SCI-POINT_TYPED_STATE_AXES v0.1/r0.3`

## Producer/Lifecycle Axis

Allowed roles are `not_requested`, `requested`, `effective`, `unavailable`,
`resolved`, `fit_attempted`, `fit_realized`,
`complete_publication_candidate`, `publication_decided`, `published`,
`failed`, `not_produced`, and `superseded`. A failed attempt is distinct from
a scientifically unavailable method or route.

## Component-Identifiability Axis

Every required parameter role carries exactly one of
`available_identifiable`, `available_bound_censored`, `weakly_identified`,
`undefined_by_model_symmetry`, `unavailable`, or `failed`.

## Named-Use Evaluation And Action

Every named use preserves four separate evaluation fields:

| Field | Exact role |
| --- | --- |
| request | `requested` or `not_requested` |
| applicability | `applicable`, `inapplicable`, or `decision_unavailable` |
| eligibility | `eligible`, `ineligible`, or `decision_unavailable` |
| realization | `realized`, `not_realized`, `unavailable`, or `failed` |

After those axes are evaluated, the owning consumer may prescribe an action or
use mode such as `diagnostic_display_only`. That action is not an eligibility
value, cannot change eligibility, and cannot rescue an exclusion for another
named use. The token `diagnostic_only` is not used as a producer state or
SCI-VAL eligibility value.

One complete immutable fit may be ineligible or decision-unavailable for
pointing correction while separately eligible with prescribed action
`diagnostic_display_only` for a telescope-QC or diagnostic-display use, and
ineligible for photometric transfer. No field or action automatically
propagates across named uses.

Each array is scientifically atomic. Failure of one does not erase siblings;
POINT does not synthesize missing arrays or publish observation-wide success.
