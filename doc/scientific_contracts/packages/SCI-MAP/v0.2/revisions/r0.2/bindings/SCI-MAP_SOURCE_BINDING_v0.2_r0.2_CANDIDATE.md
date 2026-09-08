# SCI-MAP v0.2/r0.2 candidate source binding

Binding identity: `SCI-MAP_SOURCE_BINDING v0.2/r0.2-candidate-2026-09-07`.

Status: **complete candidate scientific-source bytes; scientific-owner
acceptance, canonical activation, independent exact-SHA review, and freeze
remain separate dispositions**. This
record establishes exact candidate byte relationships. It does not establish
a source-closed frozen numerical route, implementation conformity, fidelity,
validation, performance, readiness, or production authorization.

Scientific owner: Grant Wilson.

## Shared normative authority

Identity: `SCI-MAP-v0.2-SHARED-AUTHORITY/r0.2`.

Aggregate SHA-256:
`6a7e64377e31b8b0e4dac90a27a7662ac825bd59c5a87d8237eb533d7b8d37c0`.

The aggregate is the SHA-256 of one stream in this exact order:

1. `SCI-MAP-v0.2_SHARED_AUTHORITY_r0.2.tex`;
2. `common/notation.tex`;
3. `common/definitions.tex`;
4. `common/equations.tex`;
5. `common/assumptions.tex`;
6. `common/requirements.tex`; and
7. `common/edge_cases.tex`.

For each file, append UTF-8 path relative to `src/`, NUL, ASCII decimal byte
length, NUL, raw bytes, NUL. The module byte lengths and raw hashes are bound
in the paired candidate SCI-VAL source register and the scientific digest
report.

## Exact view and helper sources

| File | Raw SHA-256 | Role |
| --- | --- | --- |
| `src/formal-scientific-engineering-contract.tex` | `53d0dbf7269ed826a96b96c56e216a53d2ded5d2e3ba2a9a42953bc33d671523` | Normative formal view of the shared core. |
| `src/scientific-rationale.tex` | `0a16a682018a0df4464c3aca4d21ee2ea66336aa2c7fb9153b8f34b3848188d2` | Explanatory science-team view; no independent authority. |
| `src/engineering-conformance.tex` | `ae745f2bd7ec288c1997b24b26720d7d7033dd05939638fac50bd65cf8f175a2` | Prospective evidence procedure; no independent science. |
| `src/SCI-MAP-v0.2_OWNER_DECISION_REGISTER_r0.2.tex` | `ab3eb86737d4834a02032962a941b5db9d4c7925e231da56eb2042192e6d9d36` | Render helper for the unchanged nine stable MAP decision identities. |

Every entry source imports the same wrapper. Every view has an exact inline
cover input after its title: `identity/formal.tex`,
`identity/rationale.tex`, or `identity/engineering.tex`. Generated cover
fragments are manager-owned mechanical inputs and are separately sealed with
the build record to avoid a self-hash cycle.

## Profile and role source bytes

| Object | Exact file SHA-256 |
| --- | --- |
| Product-role Registry | `profiles/SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.2_CANDIDATE.md`: `2087ffc5a48d61888cbdf666be146522267bbf48b704b9562fa445baa6a7c305` |
| Occurrence profile binding | `profiles/SCI-MAP_OCCURRENCE_ADMISSION_PROFILE_v0.2_r0.2_CANDIDATE.md`: `bedd2a7b46b1a18c2fb3150a2f55982779f957e1b48a1c871057f9a925d17c63` |
| Aggregate successor profile | `profiles/SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.2_CANDIDATE.md`: `e6372d1f72c78cab512987477ee1576f03ba63b79f23c88bb5fab772657adaf5` |
| Candidate SCI-VAL Profile Registry | `bindings/SCI-VAL_PROFILE_REGISTRY_SCI-MAP_v0.2_r0.2_CANDIDATE.md`: `a691a73e8a1a7bd73cb9e25ddcc6f8e281b473139c3b0564534929255dc298ee` |
| Candidate SCI-VAL source register | `bindings/SCI-VAL_SOURCE_BINDING_REGISTER_SCI-MAP_v0.2_r0.2_CANDIDATE.md`: `1cecc8623a4ed0818a761203ff9178edad80308bdcbb0eebd3fa1a74ab8dfc61` |

The occurrence profile retains exact predicates and identity
`SCI-MAP:map_upstream_admission@2` under a new candidate source binding. The
aggregate profile is new `SCI-MAP:observation_coadd_admission@2`; historical
`@1` remains immutable and is no alias. All six product roles are independently
requestable and preserve exact base-parent membership.

## Original author authority and r0.2 instruction

| Artifact | Exact identity and SHA-256 |
| --- | --- |
| Original 56-reference manifest | `SCI-MAP post-freeze successor proposed author references r0.1/2026-09-07`; `83baf583f73784bebc6724025fd110e5545303f303c70af590b53c5696ee2c4f` |
| r0.1 accepted architecture | The 19 exact `inputs/basis-r0.1/` sources/PDFs/records listed by the author-packet inventory; preserved as basis, not mutated. |
| r0.2 owner directive | `SCI-MAP-OWNER-DIRECTIVE-v0.2-r0.2/2026-09-07`; `620d4ff8204b453e2e363e84b9fcfbc480c9f0f88f0f763e0c9be69b873ca2e7` |
| Author packet | `SCI-MAP-R02-AUTHOR-PACKET/2026-09-07`; inventory source SHA-256 `f13f29746b71e90e661ee8cdb7eaba781bec97be1baa11c06529a5c2150b4a29` |

The original 56-reference manifest remains byte-identical. Portable delivery
locators begin with `inputs/`; a locator alone is not authority unless the
target is included and hash-matched by the packet inventory.

## Imported producer and boundary closure

| Subject | Exact retained authority |
| --- | --- |
| PTC-to-MAP boundary | `SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4`; SHA-256 `58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8` |
| Uniform Registry | `SCI-PTC:analysis_gridding_coefficients@draft-0.1`; record SHA-256 `433309b21d17751ce3972d196f9ba73a5f910f36e88052046928a0163fb7ca76` |
| Uniform family | `SCI-PTC:uniform_constant@draft-0.1`; common-source SHA-256 `8bc42deb69e23c757883834f7cdb82b94e89210015333f483d15c32377a56685`; freeze-manifest SHA-256 `c6d23f7f20f35089eba9b5f6dd6e65aaf05561ab3da2d0f7d18e5b8256a2f016` |
| Uniform handoff | `SCI-PTC:uniform_coefficient_handoff@draft-0.1`; approved uniform VAL Registry SHA-256 `a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f`; source register SHA-256 `fb13be222a9a8bb928fa172253f65aaf6ede081a88e901cdf6eb3799fad35f6e` |
| Original-footprint AST boundary | `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`; SHA-256 `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4` |

The literal `@draft-0.1` and `v0.2-draft.1/r0.4` spellings are retained under
the owner's explicit alternative. Their admitted scientific status and old
evaluations are unchanged. Because this r0.2 route depends on the new
candidate profile/Registry/source-binding generation, this package makes
**no source-closed frozen numerical-route claim**.

## Manager-owned completion records

Exact generated-cover identities and hashes, build recipe/tools, clean-build
outcome, PDF visual/metadata QA, stable PDF hashes, package manifest, delivery
archive identity/hash, and archive sidecar are reported in the manager-owned
records delivered with the final package. This scientific source record does
not independently certify those mechanical outcomes. Stable PDF paths are:

- `pdf/SCI-MAP-FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT-v0.2.pdf`;
- `pdf/SCI-MAP-SCIENTIFIC-RATIONALE-v0.2.pdf`; and
- `pdf/SCI-MAP-ENGINEERING-CONFORMANCE-v0.2.pdf`.
