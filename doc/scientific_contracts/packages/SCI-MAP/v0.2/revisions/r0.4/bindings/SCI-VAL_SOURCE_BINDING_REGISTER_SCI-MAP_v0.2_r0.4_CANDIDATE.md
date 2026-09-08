# SCI-VAL candidate Source-Binding Register for SCI-MAP v0.2/r0.4

Register identity:
`SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-v0.2-r0.4-candidate-2026-09-08`

Status: **CANDIDATE proposed binding**. Profile semantics are `candidate`;
scientific-owner approval, Registry registration, and source binding are
`pending`; activation/evaluability and object-specific decision realization
are `unavailable`. Exact proposed bytes do not create a canonical binding,
evaluation result, route, implementation conformance, or validation.

Paired Profile Registry:
`SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-v0.2-r0.4-candidate-2026-09-08`, file
`bindings/SCI-VAL_PROFILE_REGISTRY_SCI-MAP_v0.2_r0.4_CANDIDATE.md`, SHA-256
`a50d078bb8f02815578bfc877870d7a3b0af61ab6b1c06b295e1170a5b83b1be`.

## Exact SCI-MAP source generation

Shared identity: `SCI-MAP-v0.2-SHARED-AUTHORITY/r0.4`.

Aggregate SHA-256:
`3b81ea8180dcc3f5f37964ac1177feb417fef6a1cdcc590b33f9431b54bc1e31`.
It hashes the sources in the order shown below. For each source, the aggregate
stream contains UTF-8 path relative to `src/`, NUL, ASCII byte length, NUL,
raw bytes, NUL.

| Order | Source path | Bytes | Raw SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `src/SCI-MAP-v0.2_SHARED_AUTHORITY_r0.4.tex` | 530 | `5e405136761255a8f00ff6ca92e943c63e0c3bb6939d56924e5b884206edb56e` |
| 2 | `src/common/notation.tex` | 656 | `5a5a004adbff226a0aa37a53f11fa6707f68a4ffac05d5469adffbd8926a0510` |
| 3 | `src/common/definitions.tex` | 20867 | `b06a5089766e6601ec8a08c952a6aafed5642da33e027fce7360dd6234b6534a` |
| 4 | `src/common/equations.tex` | 8482 | `40111a13a63331071219713bf4ee420c2ac71ee8f3d6331cb17c0d72fa4c28b6` |
| 5 | `src/common/assumptions.tex` | 2050 | `215e77c5a959511e4193ac961767631247d5e39eb2f626ec5a620d745f31d3a3` |
| 6 | `src/common/requirements.tex` | 56834 | `213e4e30dcb52ceaf30b46561948263e2ac823c86fbd12769af1fff7e251281d` |
| 7 | `src/common/edge_cases.tex` | 15432 | `95fe18adcd12dae0515f3c9db80d94becf51036ca629fc5f92c3f2657c76b615` |

The wrapper includes these six modules once in this exact order. The shared
core is normative; view prose cannot change it.

## View-entry and owner-register bytes

| Artifact | Raw SHA-256 |
| --- | --- |
| `src/formal-scientific-engineering-contract.tex` | `9c6a3d360736e96c295c01501e740c46c2944e895722e7bd965ff680d5416763` |
| `src/scientific-rationale.tex` | `83272b064257bd83c0bb4bca7ff12e11b2a550912ce22c42ade76cf18871e37a` |
| `src/engineering-conformance.tex` | `92a258832664d891443d1b616717978a41bc98c79dd4aa339b63eefbef7fd5f4` |
| `src/SCI-MAP-v0.2_OWNER_DECISION_REGISTER_r0.4.tex` | `0620311aee94a719a3974e53098ce5f50fc85193c4b71723f17195bd90524728` |

The three entries separately input `identity/formal.tex`,
`identity/rationale.tex`, or `identity/engineering.tex` on the physical cover.
Those mechanical generated inputs, exact tools, build record, and PDFs are
sealed by the manager-owned final packet; they are not scientific source.

## Exact MAP policy/profile bytes

| Object | File and SHA-256 |
| --- | --- |
| Product-role Registry | `profiles/SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.4_CANDIDATE.md`; `d42d93a9f46f4f22c62d010172a20bad4f43e400e8271cbdffa655d689bd9086` |
| Occurrence admission | `SCI-MAP:map_upstream_admission@2`; `profiles/SCI-MAP_OCCURRENCE_ADMISSION_PROFILE_v0.2_r0.4_CANDIDATE.md`; `5e3786f41286305677fb3cac7b791722f5935869d0a7f8229610f4e65cd89265` |
| Aggregate admission | `SCI-MAP:observation_coadd_admission@2`; `profiles/SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.4_CANDIDATE.md`; `290292888accabd4aab9afce1c2ceb471f3b3f275072724ffbe5caed4e3d1130` |

## Approved imported authorities

All locator paths below are portable beneath `inputs/` in the delivery packet.

| Role | Exact identity, locator, and SHA-256 |
| --- | --- |
| Author manifest | `SCI-MAP post-freeze successor proposed author references r0.1/2026-09-07`; `inputs/doc/scientific_contracts/studies/SCI_MAP_POST_FREEZE_RECONCILIATION_2026-09-07/AUTHOR_REFERENCES.json`; `83baf583f73784bebc6724025fd110e5545303f303c70af590b53c5696ee2c4f` |
| r0.4 owner directive | `SCI-MAP-OWNER-DIRECTIVE-v0.2-r0.4/2026-09-08`; `inputs/owner/SCI_MAP_R04_OWNER_DIRECTIVE.txt`; `34381dc7b1a19ddf6168e2259060fd808e97f8eac2b69074fbea983f02fb017b` |
| Author packet inventory | `SCI-MAP-R04-AUTHOR-PACKET/2026-09-08`; manager-copied exact `AUTHOR_PACKET_INVENTORY.json`; source packet SHA-256 `a3c78e5ca39ff9ca0c25f2acd48a0418478aa3be56d98da6fbed43fd49a2454b`; predecessor r0.3 packet SHA-256 `423b1e41575a019592bcc1e2b21292f77ab0e8873c205ba18cf808835048f6b1` |
| PTC-to-MAP boundary | `SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4`; `inputs/doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY_UNIFORM_R0.4_2026-09-06.md`; `58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8` |
| PTC coefficient Registry | `SCI-PTC:analysis_gridding_coefficients@draft-0.1`; `inputs/doc/scientific_contracts/packages/SCI-PTC/v0.1/COEFFICIENT_REGISTRY_UNIFORM_R0.4_2026-09-06.md`; `433309b21d17751ce3972d196f9ba73a5f910f36e88052046928a0163fb7ca76` |
| Uniform family source | `SCI-PTC:uniform_constant@draft-0.1`; common-source SHA-256 `8bc42deb69e23c757883834f7cdb82b94e89210015333f483d15c32377a56685`; freeze manifest locator `inputs/doc/scientific_contracts/packages/SCI-PTC-COEFFICIENT-UNIFORM/v0.1/FREEZE_MANIFEST_R0.4.json`, SHA-256 `c6d23f7f20f35089eba9b5f6dd6e65aaf05561ab3da2d0f7d18e5b8256a2f016` |
| Uniform handoff | `SCI-PTC:uniform_coefficient_handoff@draft-0.1`; bound by the admitted uniform VAL Registry SHA-256 `a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f` and source register SHA-256 `fb13be222a9a8bb928fa172253f65aaf6ede081a88e901cdf6eb3799fad35f6e` |
| Original-footprint coordinate boundary | `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`; `inputs/doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE_BOUNDARY.md`; `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4` |

The admitted uniform authority and literal `@draft-0.1` identifiers remain
approved as supplied. This candidate neither revokes nor relabels their prior
evaluations. The r0.4 dependent numerical route relies on this candidate
Registry/register generation. It remains unavailable, is not source-closed or
frozen, and has no object-specific decision. Later route closure requires a
separately authorized final immutable identity, exact binding, registration,
and activation; none is asserted here.

## Evaluation binding rule

A new evaluation may name a later final source generation only after the exact
paired Registry/register bytes have been accepted and canonically activated. It must
also bind the exact candidate/product, selected role and base parent, profile,
all reached MAP lifecycle stages, every PTC/VAL axis, fixtures, and evidence
artifacts. No prior evaluation is rebound. A changed source, role, predicate,
authority, boundary, or profile requires a successor register.
