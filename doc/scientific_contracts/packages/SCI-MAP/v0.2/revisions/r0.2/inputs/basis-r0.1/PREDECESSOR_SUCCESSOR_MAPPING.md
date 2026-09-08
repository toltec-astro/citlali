# SCI-MAP v0.1 to v0.2 predecessor/successor mapping

Status: exact scientific mapping for the v0.2/r0.1 candidate. The frozen
v0.1/r0.7.1 files remain immutable and are not aliases for this successor.

## Authority generations

| Subject | Frozen predecessor | Proposed successor | Exact relation |
| --- | --- | --- | --- |
| Scientific contract | SCI-MAP v0.1/r0.7.1 | SCI-MAP v0.2/r0.1 candidate | New scientific version incorporates approved post-freeze decisions; no freeze yet. |
| Shared source | `SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.1.tex`; aggregate SHA-256 `649e1694b2a6353b0e5e8cb42ab73d3f03b1bc5cad671810361e802a87278e9b` | `SCI-MAP-v0.2_SHARED_AUTHORITY_r0.1.tex`; local aggregate recorded in `SOURCE_MANIFEST_R0.1.md` | Same ordered six-module architecture; changed bytes require a new immutable source identity. |
| PTC-to-MAP boundary | `SCI-PTC_TO_SCI-MAP v0.1/r0.1`; SHA-256 `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7` | `SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4`; SHA-256 `58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8` | Approved successor imports every predecessor rule and fills only the explicitly selected uniform coefficient slot under its complete handoff. |
| Original-footprint AST boundary | `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`; SHA-256 `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4` | Same exact boundary | Unchanged; no new astrometry or coordinate reconstruction. |
| MAP upstream admission | `SCI-MAP:map_upstream_admission@2`; SHA-256 `0717476c0a1d177074ee8702c18308f093d45a4913b22933f3fda3d33090a883` | Same policy identity, composed with `PTC-UNIFORM-CONSUMER-COMPOSITION r0.4/2026-09-06` and a future exact v0.2 source binding | Predicates do not change; producer QC does not replace MAP admission. Previous evaluations are not rebound. |
| MAP one-hot projection | `SCI-MAP:one_hot_containing_pixel@1` | Same exact policy identity under new v0.2 source generation | Scientific rule unchanged. |
| Coadd coefficient/admission | `SCI-MAP:uniform_observation_coadd_coefficient@1` and `SCI-MAP:observation_coadd_admission@1`; source file SHA-256 `d93c04488925931676b02dff433774ff2cda9846fdd1d3f34bff29d76efdd702` | Same scientific policy identities under a future exact v0.2-compatible source-binding generation | Arithmetic/admission predicates are unchanged. The successor clarifies equivalent logical in-memory/persisted routes and preconstruction grid identity without rewriting earlier evaluations. |
| PTC uniform Registry/family/profile | Absent from the frozen MAP source generation | Registry `SCI-PTC:analysis_gridding_coefficients@draft-0.1`; family `SCI-PTC:uniform_constant@draft-0.1`; profile `SCI-PTC:uniform_coefficient_handoff@draft-0.1` | Already owner-approved and source-bound; referenced, not reproduced. Registration alone is not selection or realization. |
| Owner-decision ledger | Nine IDs: eight open, OD-008 resolved | Same nine IDs: all decided for successor drafting | State transition is exact; predecessor ledger is not edited. |

## Shared-module mapping

| Successor module | Predecessor SHA-256 | Successor change |
| --- | --- | --- |
| `src/common/notation.tex` | `2b132704dd1ee8da7a56e5bafdc998df98422fe512736ac4f904fad8a693e569` | v0.2/r0.1 candidate metadata and status. |
| `src/common/definitions.tex` | `740f4a6f1ef0bbb12f721f192b7883144c247d039ab0c9dfa0ffae53cd711b65` | MAP-local response identity and exact admitted boundary/Registry/family/profile renderers. |
| `src/common/equations.tex` | `36329f4cd1a103c78fcdcc5ff247a850f40aba92a09156d1aa55a1411a430c04` | Four `coverage_cut` state/value stages and exact finite-nonnegative/expert admission predicate; estimator equations otherwise retained. |
| `src/common/assumptions.tex` | `bba33b92c4189fe5886ef849caebeeb5400bdc7f2572f58a74de53ca578881de` | Known MAP-local operator disclosure separated from upstream-conditioned and whole-chain response availability. |
| `src/common/requirements.tex` | `68acf81d6c27788495cc680a1819da5f269616493b9aba5a6e0a1d1058ba5fa7` | Stable clauses amended only as listed below. |
| `src/common/edge_cases.tex` | `47022012e79173a1778a4e5bdc6743b4691bfecb27faa3090bcf03458d87e123` | Stable predictions amended only as listed below. |

## Requirement identity preservation

All IDs `SCI-MAP-REQ-001` through `SCI-MAP-REQ-052` remain present exactly
once. These subjects incorporate approved successor content:

| Requirement | Incorporated change |
| --- | --- |
| REQ-001 | Contract metadata resolves through v0.2/r0.1 macros. |
| REQ-006 | Exact admitted uniform Registry/family/profile/boundary, explicit selection and complete handoff, no default/fallback, independent MAP classification. |
| REQ-008, REQ-016 | Known MAP-local realized-operator response disclosure and separate upstream-conditioned response. |
| REQ-011, REQ-018, REQ-039, REQ-048 | Version-scoped language updated from v0.1 to v0.2. |
| REQ-031, REQ-032 | Operational support purpose, full `coverage_cut` domain and four stages, exact zero, ordinary/expert behavior, failure-before-mutation, inclusive thresholds. |
| REQ-036, REQ-039, REQ-043, REQ-046, REQ-048 | Complete logical bundle, plan-controlled persistence, equivalent in-memory/persisted/coadd-only routes, required-publication failure. |
| REQ-038, REQ-043, REQ-046 | Preconstruction canonical-grid request, exact centered-integer embedding, new identity for later crop. |
| REQ-052 | Conditional Pointing/OOF operator reuse, mode-owner limits, free experimentation, and versioned official identity. |

Every other requirement retains the predecessor clause under the new contract
metadata and shared source generation.

## Prediction identity preservation

All IDs `SCI-MAP-PRED-001` through `SCI-MAP-PRED-025` remain present exactly
once. PRED-010 incorporates the admitted uniform handoff distinction;
PRED-012 the complete support-policy cases; PRED-013 the MAP-local versus
upstream response distinction; PRED-015 route-equivalent observation coadd;
PRED-018 preconstruction grid and separately identified crop; and PRED-024 the
v0.2 method boundary. Every other prediction retains its predecessor subject
and wording.

## Identity and claim limits

The exact local successor source hashes are determinable and recorded after
authoring. The future canonical Git commit/tree, immutable successor manifest
identity, and new VAL Registry/source-binding generation are absent from the
approved inputs and remain explicitly pending. No old evaluation, source
digest, boundary, profile result, application state, or validation result is
relabelled as v0.2.
