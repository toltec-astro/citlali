# SCI-VAL candidate Source-Binding Register for SCI-MAP v0.2/r0.2

Register identity:
`SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-v0.2-r0.2-candidate-2026-09-07`

Status: **CANDIDATE for scientific-owner disposition, VAL-owner review,
canonical activation, and freeze**. This is an exact proposed binding over the
listed bytes. It records no evaluation result, realized route, implementation
conformance, or validation.

Paired Profile Registry:
`SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-v0.2-r0.2-candidate-2026-09-07`, file
`bindings/SCI-VAL_PROFILE_REGISTRY_SCI-MAP_v0.2_r0.2_CANDIDATE.md`, SHA-256
`a691a73e8a1a7bd73cb9e25ddcc6f8e281b473139c3b0564534929255dc298ee`.

## Exact SCI-MAP source generation

Shared identity: `SCI-MAP-v0.2-SHARED-AUTHORITY/r0.2`.

Aggregate SHA-256:
`6a7e64377e31b8b0e4dac90a27a7662ac825bd59c5a87d8237eb533d7b8d37c0`.
It hashes the sources in the order shown below. For each source, the aggregate
stream contains UTF-8 path relative to `src/`, NUL, ASCII byte length, NUL,
raw bytes, NUL.

| Order | Source path | Bytes | Raw SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `src/SCI-MAP-v0.2_SHARED_AUTHORITY_r0.2.tex` | 530 | `24839f9912e4bd9f7cae9a7be974e726a188d2e9fdadbdd0fcc5e913abd5c4e7` |
| 2 | `src/common/notation.tex` | 656 | `1975dfce4b50dda686744a495e3b6ae988fcf30812bfe577e61c2edc80f67cce` |
| 3 | `src/common/definitions.tex` | 15955 | `4762d0c3ccaf3c0cc4b2c8b3fc11fbc8e9e7a6d20414ede53172300dbd3886e5` |
| 4 | `src/common/equations.tex` | 8235 | `10a1f9829af3934d6fa71007fac4a7bfd0d9017431504d08ca9aeb82cfdc02f3` |
| 5 | `src/common/assumptions.tex` | 1787 | `87ef8e16a54e2cec76446d75b0896a18ed255278ab28822271bb182c7d922773` |
| 6 | `src/common/requirements.tex` | 51893 | `baef73170ce8638c451de66f2e224c2a5ca0c4a6957aba41bef60db0d8718094` |
| 7 | `src/common/edge_cases.tex` | 12679 | `addd77b0435868c1ddb381f995326f620e4cdf6a55f8d75085fcf26529d6f634` |

The wrapper includes these six modules once in this exact order. The shared
core is normative; view prose cannot change it.

## View-entry and owner-register bytes

| Artifact | Raw SHA-256 |
| --- | --- |
| `src/formal-scientific-engineering-contract.tex` | `53d0dbf7269ed826a96b96c56e216a53d2ded5d2e3ba2a9a42953bc33d671523` |
| `src/scientific-rationale.tex` | `fc6551fa650c23c925229e3dc1cc8ae0764c7688cf698c9adbbbf0dc60c7ba51` |
| `src/engineering-conformance.tex` | `ae745f2bd7ec288c1997b24b26720d7d7033dd05939638fac50bd65cf8f175a2` |
| `src/SCI-MAP-v0.2_OWNER_DECISION_REGISTER_r0.2.tex` | `ab3eb86737d4834a02032962a941b5db9d4c7925e231da56eb2042192e6d9d36` |

The three entries separately input `identity/formal.tex`,
`identity/rationale.tex`, or `identity/engineering.tex` on the physical cover.
Those mechanical generated inputs, exact tools, build record, and PDFs are
sealed by the manager-owned final packet; they are not scientific source.

## Exact MAP policy/profile bytes

| Object | File and SHA-256 |
| --- | --- |
| Product-role Registry | `profiles/SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.2_CANDIDATE.md`; `2087ffc5a48d61888cbdf666be146522267bbf48b704b9562fa445baa6a7c305` |
| Occurrence admission | `SCI-MAP:map_upstream_admission@2`; `profiles/SCI-MAP_OCCURRENCE_ADMISSION_PROFILE_v0.2_r0.2_CANDIDATE.md`; `bedd2a7b46b1a18c2fb3150a2f55982779f957e1b48a1c871057f9a925d17c63` |
| Aggregate admission | `SCI-MAP:observation_coadd_admission@2`; `profiles/SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.2_CANDIDATE.md`; `e6372d1f72c78cab512987477ee1576f03ba63b79f23c88bb5fab772657adaf5` |

## Approved imported authorities

All locator paths below are portable beneath `inputs/` in the delivery packet.

| Role | Exact identity, locator, and SHA-256 |
| --- | --- |
| Author manifest | `SCI-MAP post-freeze successor proposed author references r0.1/2026-09-07`; `inputs/doc/scientific_contracts/studies/SCI_MAP_POST_FREEZE_RECONCILIATION_2026-09-07/AUTHOR_REFERENCES.json`; `83baf583f73784bebc6724025fd110e5545303f303c70af590b53c5696ee2c4f` |
| r0.2 owner directive | `SCI-MAP-OWNER-DIRECTIVE-v0.2-r0.2/2026-09-07`; `inputs/owner/SCI_MAP_R02_OWNER_DIRECTIVE.txt`; `620d4ff8204b453e2e363e84b9fcfbc480c9f0f88f0f763e0c9be69b873ca2e7` |
| Author packet inventory | `SCI-MAP-R02-AUTHOR-PACKET/2026-09-07`; manager-copied exact `AUTHOR_PACKET_INVENTORY.json`; source packet SHA-256 `f13f29746b71e90e661ee8cdb7eaba781bec97be1baa11c06529a5c2150b4a29` |
| PTC-to-MAP boundary | `SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4`; `inputs/doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY_UNIFORM_R0.4_2026-09-06.md`; `58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8` |
| PTC coefficient Registry | `SCI-PTC:analysis_gridding_coefficients@draft-0.1`; `inputs/doc/scientific_contracts/packages/SCI-PTC/v0.1/COEFFICIENT_REGISTRY_UNIFORM_R0.4_2026-09-06.md`; `433309b21d17751ce3972d196f9ba73a5f910f36e88052046928a0163fb7ca76` |
| Uniform family source | `SCI-PTC:uniform_constant@draft-0.1`; common-source SHA-256 `8bc42deb69e23c757883834f7cdb82b94e89210015333f483d15c32377a56685`; freeze manifest locator `inputs/doc/scientific_contracts/packages/SCI-PTC-COEFFICIENT-UNIFORM/v0.1/FREEZE_MANIFEST_R0.4.json`, SHA-256 `c6d23f7f20f35089eba9b5f6dd6e65aaf05561ab3da2d0f7d18e5b8256a2f016` |
| Uniform handoff | `SCI-PTC:uniform_coefficient_handoff@draft-0.1`; bound by the admitted uniform VAL Registry SHA-256 `a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f` and source register SHA-256 `fb13be222a9a8bb928fa172253f65aaf6ede081a88e901cdf6eb3799fad35f6e` |
| Original-footprint coordinate boundary | `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`; `inputs/doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE_BOUNDARY.md`; `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4` |

The admitted uniform authority and literal `@draft-0.1` identifiers remain
approved as supplied. This candidate neither revokes nor relabels their prior
evaluations. The changed r0.2 numerical route depends on this candidate
Registry/register generation, so it is **not source-closed or frozen** until
scientific-owner disposition, canonical activation, final packet sealing, and
freeze are complete.

## Evaluation binding rule

A new evaluation may name this source generation only after the exact paired
Registry/register bytes have been accepted and canonically activated. It must
also bind the exact candidate/product, selected role and base parent, profile,
all reached MAP lifecycle stages, every PTC/VAL axis, fixtures, and evidence
artifacts. No prior evaluation is rebound. A changed source, role, predicate,
authority, boundary, or profile requires a successor register.
