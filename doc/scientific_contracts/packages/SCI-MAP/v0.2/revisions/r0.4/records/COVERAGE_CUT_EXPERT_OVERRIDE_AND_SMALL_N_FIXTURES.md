# SCI-MAP v0.2/r0.4 `coverage_cut`, small-N, and empty-support fixtures

Status: **CANDIDATE prospective specification**. Equations and requirements in
the shared authority are normative; no fixture has been executed against an
implementation.

## Typed above-one override

The exact plan fact is
`coverage_cut_above_one_override_requested`. It is independent of the
numerical cut and contains:

| Field | Exact obligation |
| --- | --- |
| `request_identity` | Immutable identity of the explicit override request. |
| `authority_or_profile` | Exact owner or authorized profile allowed to make the request. |
| `scope` | Observation, product role, plan, and execution scope to which the request applies. |
| reached lifecycle stages | Exact value and identity for each actually reached `requested`, `effective`, `observation_resolved`, `applied`, and `realized` stage. A completed realization has all five; an earlier stop has no future-stage placeholder. |
| `exact_value` | Same exact `coverage_cut` value bound by the stage record. |
| `cause` | Reason for the above-one request or for a nonpassing resolution. |
| `provenance` | Source identity, generation, owner decision/profile, and parent record. |

The resolved Boolean override is true only when the request identity,
authority/profile, scope, exact observation-resolved value, cause, provenance,
and reached-stage chain all agree. A number greater than one never authorizes
itself.

## Domain and application rules

| Resolved value | Pre-application behavior |
| --- | --- |
| `0` | Admit as no relative cut; finite positive `Q_p` remains mandatory. |
| `0 < c <= 1` | Admit as the ordinary recommended range. |
| `c > 1` with matching authorized override | Admit as explicit expert scope; support may be empty. If the applied authorized output-row set is empty, record `applied_no_support_authorized_output_rows`, outcome `not_produced`, and cause `no_support_authorized_output_rows`. |
| `c > 1` without matching authorized override | Fail before support construction, application, or required-product mutation. |
| Negative, non-finite, missing, or unrepresentable | Fail before support construction, application, or required-product mutation. |

There is no universal numerical default. Pre-application validation uses
`c_observation_resolved`. At the application event
`c_applied = c_observation_resolved`. A realized output records
`c_realized = c_applied`; that later value is never an application
prerequisite. Both thresholds are inclusive:

`T_norm = Q_star c_applied / 10`, `T_sci = Q_star c_applied`, with separate
finite-positive `Q_p` requirements.

## Exact zero-based order-statistic fixtures

For sorted positive finite values `P_(0) <= ... <= P_(N-1)`, use
`k = floor((floor(0.75 N) + N)/2)` and `Q_star = P_(k)` for `N>0`; for `N=0`,
`Q_star=0`. The concrete fixture population below uses `P_(r)=10(r+1)`.

| N | floor(0.75 N) | k | Expected Q_star | Transition note |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | n/a | 0 | Exact approved empty-population boundary convention; there is no selected population element and no zero-sky meaning. |
| 1 | 0 | 0 | 10 | First defined order statistic. |
| 2 | 1 | 1 | 20 | Index advances. |
| 3 | 2 | 2 | 30 | Index advances. |
| 4 | 3 | 3 | 40 | Index advances. |
| 5 | 3 | 4 | 50 | Inner floor holds; outer index advances. |
| 6 | 4 | 5 | 60 | Index advances. |
| 7 | 5 | 6 | 70 | Index advances. |
| 8 | 6 | 7 | 80 | Index advances. |
| 9 | 6 | 7 | 80 | First repeated index across adjacent N. |
| 10 | 7 | 8 | 90 | Index advances after repeat. |
| 11 | 8 | 9 | 100 | Index advances. |
| 12 | 9 | 10 | 110 | Index advances. |
| 13 | 9 | 11 | 120 | Inner floor holds; index advances. |
| 14 | 10 | 12 | 130 | Index advances. |
| 15 | 11 | 13 | 140 | Index advances. |
| 16 | 12 | 14 | 150 | Index advances. |
| 17 | 12 | 14 | 150 | Second repeated index across adjacent N. |
| 18 | 13 | 15 | 160 | Index advances after repeat. |

For each nonempty fixture and each strictly positive threshold, define the
entire declared population first, compute `N`, `k`, `Q_star`, and the threshold,
then include predeclared evaluation rows exactly at and immediately below that
threshold. The equality row establishes inclusive passage only when its
`Q_p` is finite and strictly positive; the lower positive row establishes the
relative-threshold failure. Evaluation rows must not be appended to the
population after `Q_star` is computed. For `N=0` and for `c=0`, the zero
threshold does not admit `Q_p=0`: the independent finite-positive predicate
still excludes zero, negative, non-finite, and absent normalization.

## Empty-support outcome fixture

For an exact valid above-one request whose applied thresholds authorize no
rows, retain every reached lifecycle stage through `applied`, the exact parent
and plan, coefficient/projection/support/profile/role identities, `Q_star`,
thresholds, support predicates, empty-row proof, attempt identity, cause, and
provenance. Expect attempt outcome `not_produced`, no product realization, no
completion marker, no dense zero map, no valid ordinary signal plane, no
no-row coadd member, and no fabricated response/covariance plane. No empty
bundle alternative is authorized.

This is a prospective fixture only. The r0.4 candidate route is presently
unevaluable, and this package asserts no applied attempt for a current object.
