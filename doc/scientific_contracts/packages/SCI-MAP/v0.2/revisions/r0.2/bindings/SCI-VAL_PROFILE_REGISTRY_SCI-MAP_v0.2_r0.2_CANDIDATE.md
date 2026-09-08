# SCI-VAL candidate Profile Registry generation for SCI-MAP v0.2/r0.2

Registry identity:
`SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-v0.2-r0.2-candidate-2026-09-07`

Status: **CANDIDATE for scientific-owner disposition, VAL-owner review,
canonical activation, and freeze**. This exact record registers prospective
policy definitions and asserts no evaluation, implementation conformance,
validation, or realized route.

Scientific policy owner: Grant Wilson for SCI-MAP. SCI-VAL registers and may
evaluate these MAP-authored policies; it does not author MAP science,
aggregate, place, or execute MAP products.

Paired register identity:
`SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-v0.2-r0.2-candidate-2026-09-07`.

## Bound candidate source

| Field | Exact value |
| --- | --- |
| Shared authority identity | `SCI-MAP-v0.2-SHARED-AUTHORITY/r0.2` |
| Shared authority aggregate SHA-256 | `6a7e64377e31b8b0e4dac90a27a7662ac825bd59c5a87d8237eb533d7b8d37c0` |
| Aggregate construction | Ordered wrapper, then `common/notation.tex`, `definitions.tex`, `equations.tex`, `assumptions.tex`, `requirements.tex`, `edge_cases.tex`; for each source hash UTF-8 relative path, NUL, ASCII byte length, NUL, raw bytes, NUL. Paths are relative to `src/`. |
| Product-role Registry | `profiles/SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.2_CANDIDATE.md`; SHA-256 `2087ffc5a48d61888cbdf666be146522267bbf48b704b9562fa445baa6a7c305` |

## Profile records

| Exact profile identity | Source record and SHA-256 | Candidate registration |
| --- | --- | --- |
| `SCI-MAP:map_upstream_admission@2` | `profiles/SCI-MAP_OCCURRENCE_ADMISSION_PROFILE_v0.2_r0.2_CANDIDATE.md`; `bedd2a7b46b1a18c2fb3150a2f55982779f957e1b48a1c871057f9a925d17c63` | Scientific predicates and identifier are unchanged. This generation adds only the exact r0.2 compatible source, lifecycle, role, uniform-boundary, and candidate Registry/register binding. Earlier records and evaluations retain their own bindings. |
| `SCI-MAP:observation_coadd_admission@2` | `profiles/SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.2_CANDIDATE.md`; `e6372d1f72c78cab512987477ee1576f03ba63b79f23c88bb5fab772657adaf5` | New candidate successor profile for role-qualified companion rules. It permits base-coadd response/covariance unavailability without changing signal membership and requires exact named companions for stronger roles. Historical `@1` is immutable and no alias. |
| `SCI-MAP:one_hot_containing_pixel@1` | Shared authority aggregate above; exact unchanged rule inherited from the admitted basis. | Compatible r0.2 candidate source binding only; lower-inclusive, upper-exclusive unique containing pixel, with outer-boundary loss. |
| `SCI-MAP:uniform_observation_coadd_coefficient@1` | Shared authority aggregate above; exact unchanged rule inherited from the admitted basis. | Compatible r0.2 candidate source binding only; exact dimensionless `u_op=1` for every admitted observation row. |

## Registered product-role scope

This generation binds the six independently requestable roles:

- `SCI-MAP:base_observation_map@1`;
- `SCI-MAP:response_bearing_observation_map@1`;
- `SCI-MAP:covariance_qualified_observation_map@1`;
- `SCI-MAP:base_coadd@1`;
- `SCI-MAP:response_bearing_coadd@1`; and
- `SCI-MAP:covariance_qualified_coadd@1`.

Every evaluation names exactly one role, exact base parent and ordered
membership, every reached stage of the five-stage MAP lifecycle, and the four independent
VAL axes. Only requested/applicable/eligible/realized projects to pass. An ECS
requirement result and an ECS evidence-artifact realization are separate axes.

## Exact imported uniform composition

The occurrence profile may consume the owner-approved uniform slot only
through all of these exact records:

| Authority | Exact identity or SHA-256 |
| --- | --- |
| PTC-to-MAP boundary | `SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4`; `58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8` |
| Coefficient Registry | `SCI-PTC:analysis_gridding_coefficients@draft-0.1`; record SHA-256 `433309b21d17751ce3972d196f9ba73a5f910f36e88052046928a0163fb7ca76` |
| Uniform family | `SCI-PTC:uniform_constant@draft-0.1`; common-source SHA-256 `8bc42deb69e23c757883834f7cdb82b94e89210015333f483d15c32377a56685` |
| Uniform handoff profile | `SCI-PTC:uniform_coefficient_handoff@draft-0.1`; source-bound by the approved paired uniform VAL records |
| Uniform freeze manifest | `c6d23f7f20f35089eba9b5f6dd6e65aaf05561ab3da2d0f7d18e5b8256a2f016` |
| Approved uniform VAL Registry | `a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f` |
| Approved uniform VAL source register | `fb13be222a9a8bb928fa172253f65aaf6ede081a88e901cdf6eb3799fad35f6e` |
| Consumer composition | `PTC-UNIFORM-CONSUMER-COMPOSITION r0.4/2026-09-06` |

The literal `@draft-0.1` identifiers are retained exactly under the scientific
owner's explicit alternative. Their approved status is not revoked or
relabeled. Registration supplies no family selection, completed publication,
handoff realization, MAP admission, or numerical route.

## Supersession and activation

Any change in policy identity, owner, source digest, object, population,
predicate, role requirement, axis, exception, aggregation, failure scope, or
lifecycle requires a new immutable Registry generation. This candidate has
complete reviewable bytes. Its identity and digest do not by themselves make
it owner-accepted, canonically activated, frozen, or usable for retroactive
evaluation.
