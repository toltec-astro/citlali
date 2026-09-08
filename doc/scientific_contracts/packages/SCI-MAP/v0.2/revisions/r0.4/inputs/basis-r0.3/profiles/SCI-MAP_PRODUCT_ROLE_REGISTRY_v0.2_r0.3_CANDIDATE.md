# SCI-MAP v0.2/r0.3 candidate product-role Registry

Registry identity: `SCI-MAP_PRODUCT_ROLE_REGISTRY v0.2/r0.3-candidate-2026-09-08`

Status: **CANDIDATE for scientific-owner disposition**. This record defines
scientific roles; it asserts no realization, conformity, validation, or
canonical activation.

Scientific owner: Grant Wilson.

Normative source: `SCI-MAP-v0.2-SHARED-AUTHORITY/r0.3`; exact aggregate digest
is bound in `../bindings/SCI-MAP_SOURCE_BINDING_v0.2_r0.3_CANDIDATE.md` after
the final scientific-source bytes are sealed.

Every role below is independently requestable. Each request binds its exact
role, request identity, parent set, ordered MAP lifecycle
`requested/effective/observation_resolved/applied/realized`, output and
persistence plan, separate attempt outcome, product realization, cause, and
failure scope. A stronger role is a new
product over an exact base parent and never rewrites that parent's membership,
validity, response/covariance state, or provenance.

A role is realized only with actual immutable scientific rows and product
contents. A failed applied attempt retains failure evidence but no product. An
applied valid policy with no support-authorized rows has outcome
`not_produced`, cause `no_support_authorized_output_rows`, and no base or
stronger-role product.

| Exact role identity | Exact parent | Required scientific content | Missing or incompatible stronger companion |
| --- | --- | --- | --- |
| `SCI-MAP:base_observation_map@1` | Exact realized PTC/AST inputs and complete resolved MAP observation plan | Complete MAP-local realized operator identity; signal and base bundle; honest upstream-response and covariance identities/statuses | Base signal may remain valid when a stronger numerical upstream response or complete covariance is unavailable. |
| `SCI-MAP:response_bearing_observation_map@1` | Exact `SCI-MAP:base_observation_map@1` | Every numerical response object named by this selected role, each with exact source domain, basis, class, unit, normalization, parent, WCS/grid, and row identity; a numerical full-procedure object also requires the complete comparison-space record | This role is unavailable or fails before required-product mutation. The base parent is unchanged. Absence of a response family not named by this selected role does not block it. |
| `SCI-MAP:covariance_qualified_observation_map@1` | Exact `SCI-MAP:base_observation_map@1` | Every within-observation covariance block named by this selected role on exact rows | Missing blocks are unknown and block this role only; the base parent is unchanged. |
| `SCI-MAP:base_coadd@1` | Ordered set of exact complete base-observation bundles and one resolved coadd plan | Equal-observation signal arithmetic; complete base companions; honest response and covariance identities/statuses | Unavailable or basis-incompatible member response and partial/symbolic/summarized/lineage-resolvable/unavailable covariance are permitted with unchanged signal membership. Coadd response becomes unavailable with exact cause. |
| `SCI-MAP:response_bearing_coadd@1` | Exact `SCI-MAP:base_coadd@1` and its identical ordered signal-member set | One exact response family `R`; every member `R_o^(R)`; compatible source domain, perturbation definition, basis, class, unit, normalization, reference/gauge/null state, parent, WCS/grid, and row maps; fixed membership, placement, `B_out`, support, coefficients; exact composition theorem | For an existing base parent, a required incompatibility blocks this role and never selects a subset. A separately requested admission may reject before mutation only while constructing its own separately identified base/signal parent. No hidden subset, mixed family, zero companion, or finite-difference-as-Jacobian propagation. Coadd-procedure and whole-chain responses remain separately unavailable. |
| `SCI-MAP:covariance_qualified_coadd@1` | Exact `SCI-MAP:base_coadd@1` and its identical ordered signal-member set | Every named within- and cross-observation block on exact stacked row identities | For an existing base parent, a missing block blocks this role and never selects a subset. A separately requested admission may reject before constructing its own separately identified base/signal parent. Unknown is never zero or independence. |

After exact owner acceptance, canonical Registry/source binding, and
activation, SCI-VAL may evaluate the observation roles under
`SCI-MAP:map_upstream_admission@2` and the coadd roles under
`SCI-MAP:observation_coadd_admission@2` for requested objects. The historical
aggregate profile `SCI-MAP:observation_coadd_admission@1` remains immutable
and is no alias.

For this generation, both MAP profile semantics are candidate; scientific-
owner approval, Registry registration, and source binding are pending;
evaluability and object-specific decision realization are unavailable. Exact
historical approvals and evaluations are unchanged.
