# `SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE`, SCI-MAP v0.2/r0.4

Status: **CANDIDATE for scientific-owner disposition**. This record renders
the gate defined normatively by Equations 13--16 and REQ-008, REQ-016, and
REQ-017 in the shared authority.

## Scope

The gate applies separately to every proposed numerical
`Delta z_PTC-FP`, `Delta m_PTC-FP`, and `Delta m_PTC+MAP-RR`, and to any later
separately authorized coadd full-procedure or coadd re-resolved family.
Baseline and perturbed arrays are never subtractable merely because both exist
or have the same shape. The stable record identity is
`SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE`.

## Complete immutable authorization record

| Field | Required exact content |
| --- | --- |
| `comparison_record_id` | `SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE` plus immutable record generation. |
| `response_family` | Exactly one authorized family: PTC full-procedure, fixed-MAP projection of PTC full-procedure, PTC+MAP re-resolved, or a separately authorized coadd full-procedure/re-resolved family. |
| `baseline_product_id`, `perturbed_product_id` | Exact product identities and parents. |
| `scientific_quantity`, `unit` | Identical authorized quantity and unit, or an explicit exact conversion already authorized outside this subtraction. |
| `baseline_rows`, `perturbed_rows` | Occurrence identities for `Delta z`; MAP output-row identities for `Delta m`. |
| `source_frame_wcs`, `target_frame_wcs` | Exact frames, WCS identities, and generations for each operand and the common domain. |
| `additive_reference`, `gauge`, `null_space` | Baseline and perturbed state plus the exact authorized relation. |
| `baseline_support`, `perturbed_support` | Complete support identity and row set for each result. |
| `coefficient_basis`, `response_basis` | Exact family, generation, class, normalization, and relevant parents on each side. |
| `common_comparison_domain` | One exact finite scientific comparison domain authorized for this difference. |
| `baseline_map_to_common`, `perturbed_map_to_common` | Exact total maps from authorized operand rows into the common domain. |
| `unmatched_row_treatment` | Explicit treatment for every row present in only one output; it may make the numerical difference unavailable. |
| `perturbation` | Exact amplitude, sign, side, location, source state, template, and normalization. |
| `finite_difference_convention` | Exact numerator ordering and any divisor, including units. |
| `state_transition` | Complete baseline-to-perturbed transition record, including `Delta S_PTC-FP`. |
| `numerical_availability`, `cause` | `available` only if every required field authorizes subtraction; otherwise typed unavailable or a separately typed discontinuity/state transition with exact cause. |

For PTC+MAP re-resolved response, `Pi_0` and `Pi_+` may produce different
membership and output-row sets. The record must supply the common comparison
grid and both exact row maps before forming a difference. `Pi` is the immutable
MAP plan and must not be confused with the row selector `J_out`.

For a coadd procedure response, the record must additionally bind baseline and
perturbed observation admission, ordered membership, centered-integer
placement, support, coefficients, role, plan, and exact maps into the common
comparison domain. No such coadd-procedure family is currently authorized.
The coadd full-procedure response and the whole-chain response are separately
`unavailable`.

## Prohibited implicit operations

The authorization performs no implicit zero fill, union-domain sentinel,
support intersection, nearest-coordinate association, same-shape association,
row-position association, interpolation, reprojection, or gauge alignment.
Numerical equality of metadata or array extents is not an identity map.

`Delta S_PTC-FP` records changed state. It never by itself establishes that
the payloads inhabit a common vector space or may be subtracted. When no exact
authorized common comparison space exists, the numerical response is
unavailable or the outcome is only a separately typed state
transition/discontinuity.

This comparison-space authorization does not turn a finite difference into a
linear Jacobian. Propagation through `B_out` requires a separately exact
composition theorem and fixed coadd state.

## Prospective evidence

ECS fixtures must include: identical shapes with permuted row identities;
different supports with and without an explicitly authorized map; a missing
gauge relation; a changed `Pi` whose `J_out` domains differ; a complete exact
map that succeeds; and every prohibited implicit operation above. Each result
starts `not_assessed`; no execution or fidelity claim is made here.
