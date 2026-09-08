# SCI-MAP v0.2/r0.4 product-role dependency matrix

Status: **CANDIDATE for scientific-owner disposition**. Exact role identities
are proposed in
`../profiles/SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.4_CANDIDATE.md`. Normative
authority remains the shared LaTeX core.

Each role is independently requestable and lifecycle-bound. A stronger role
is a new product over an exact base parent; it never rewrites that parent's
membership, validity, response/covariance state, or provenance.

| Exact role identity | Exact parent | Required content | Missing or incompatible companion |
| --- | --- | --- | --- |
| `SCI-MAP:base_observation_map@1` | Exact realized PTC/AST inputs and one complete observation-resolved MAP plan | Complete base bundle and MAP-local realized operator identity; honest upstream-response and covariance identities, states, limitations, and causes | A stronger numerical upstream response or complete covariance may be unavailable while the base signal remains valid for claims that do not require it. |
| `SCI-MAP:response_bearing_observation_map@1` | Exact `SCI-MAP:base_observation_map@1` | Every numerical response object named by this selected role, each with compatible source domain, basis, class, unit, normalization, parent, WCS/grid, and row identity; full-procedure response also requires a complete comparison-space record | Blocks or fails this role before its required-product mutation. The base parent is unchanged. Absence of a different response family not named by the role is irrelevant. |
| `SCI-MAP:covariance_qualified_observation_map@1` | Exact `SCI-MAP:base_observation_map@1` | Every within-observation covariance block named by this role on exact rows | Missing blocks are unknown and block this role only. The base parent is unchanged. |
| `SCI-MAP:base_coadd@1` | Ordered set of exact complete base-observation bundles plus one observation-resolved coadd plan | Equal-observation signal arithmetic; all base companions; honest response and covariance identities/statuses | Unavailable or basis-incompatible member response and partial, symbolic, summarized, lineage-resolvable, or unavailable covariance are permitted with unchanged signal membership. Coadd response is unavailable with exact cause. |
| `SCI-MAP:response_bearing_coadd@1` | Exact `SCI-MAP:base_coadd@1` and its identical ordered signal-member set | One exact response family `R`; every exact member `R_o^(R)`; compatible source domain, perturbation definition, basis, class, unit, normalization, reference/gauge/null state, parent, WCS/grid, and row maps; fixed membership, placement, `B_out`, support, and coefficient state; exact composition theorem | For an existing base parent, a required incompatibility blocks this role and never selects a subset. A separately requested admission may reject an observation before mutation only while constructing its own separately identified base/signal parent. No hidden subset, mixed family, zero companion, or finite-difference-as-Jacobian propagation is permitted. |
| `SCI-MAP:covariance_qualified_coadd@1` | Exact `SCI-MAP:base_coadd@1` and its identical ordered signal-member set | Every named within- and cross-observation covariance block on exact stacked rows | For an existing base parent, a missing block blocks this role and never selects a subset. A separately requested admission may reject before constructing its own separately identified base/signal parent. Unknown is never zero or independence. |

Observation roles use unchanged scientific predicates under
`SCI-MAP:map_upstream_admission@2`, with the r0.4 candidate source binding.
Coadd roles use candidate successor
`SCI-MAP:observation_coadd_admission@2`. Historical
`SCI-MAP:observation_coadd_admission@1` remains immutable and is no alias.

Both r0.4 profile semantics are candidate; owner approval, Registry
registration, and source binding are pending; evaluability and object-specific
decision realization are unavailable. These present-candidate states do not
alter any exact historical approval or evaluation.

Atomic rejection is role-qualified. An incompatible base-bundle identity,
unit, WCS/grid, shape, policy, parent, or other base-required fact rejects at
the declared role scope before mutation. A response or covariance
incompatibility has that effect only when the exact selected role and profile
name the companion as required.
