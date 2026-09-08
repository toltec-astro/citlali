# SCI-VAL candidate Source-Binding Register for SCI-MAP v0.2/r0.3

Register identity:
`SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-v0.2-r0.3-candidate-2026-09-08`

Status: **CANDIDATE proposed binding**. Profile semantics are `candidate`;
scientific-owner approval, Registry registration, and source binding are
`pending`; activation/evaluability and object-specific decision realization
are `unavailable`. Exact proposed bytes do not create a canonical binding,
evaluation result, route, implementation conformance, or validation.

Paired Profile Registry:
`SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-v0.2-r0.3-candidate-2026-09-08`, file
`bindings/SCI-VAL_PROFILE_REGISTRY_SCI-MAP_v0.2_r0.3_CANDIDATE.md`, SHA-256
`c360a9d4b091474ef51b414c78f3a599ffa8c7269a40587cd7ffa5995d5b3477`.

## Exact SCI-MAP source generation

Shared identity: `SCI-MAP-v0.2-SHARED-AUTHORITY/r0.3`.

Aggregate SHA-256:
`45b863cab2713075656e11f5d6008f377879079f0d6f52b08ac579bd04ada4c6`.
It hashes the sources in the order shown below. For each source, the aggregate
stream contains UTF-8 path relative to `src/`, NUL, ASCII byte length, NUL,
raw bytes, NUL.

| Order | Source path | Bytes | Raw SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `src/SCI-MAP-v0.2_SHARED_AUTHORITY_r0.3.tex` | 530 | `3a757501e3ff1461730bda94d359e7678460656d037c87761a651409864522ee` |
| 2 | `src/common/notation.tex` | 656 | `72cbcdc20c1f273b45bb76575677c8b0418b5add65d358de2b695c8f390181a5` |
| 3 | `src/common/definitions.tex` | 21203 | `48946f9b07af21c542c857baa93f3648a19e9060d0797260e6af59be834a0047` |
| 4 | `src/common/equations.tex` | 8464 | `96594c629131f5a6e6c79707ffcb0494ded5e394fda402ca572ce420427aa600` |
| 5 | `src/common/assumptions.tex` | 2013 | `3829874bd8051582d45ecbb74ecd4a797b16ce424bc3438416bb5b3370298bd2` |
| 6 | `src/common/requirements.tex` | 56527 | `395476b99c45d3cb59a48f9a4155196389342be0726d7f9dc1a82aaa0cb12411` |
| 7 | `src/common/edge_cases.tex` | 14581 | `29444fbce909b8528a416928fb5c59e5d6bd774d21ee604723096bbe3ebd8fde` |

The wrapper includes these six modules once in this exact order. The shared
core is normative; view prose cannot change it.

## View-entry and owner-register bytes

| Artifact | Raw SHA-256 |
| --- | --- |
| `src/formal-scientific-engineering-contract.tex` | `23fbfc367f2bdf244701b06cbcf62a80af958c2d1f8b389ea90df2a6b81b129c` |
| `src/scientific-rationale.tex` | `84dc40bafb4881b1419959acee2bef5c2d61e9c3d61b59d4ae96c580d74a6104` |
| `src/engineering-conformance.tex` | `aa2d5e5213f658b6c8d3b57c25dd3567fa4ea8de77b70b10431fd8ffe9ad05fc` |
| `src/SCI-MAP-v0.2_OWNER_DECISION_REGISTER_r0.3.tex` | `2e4f62cf01d85bbfd19dba247a3984a30b4cc7766947503214074e96b5e524bf` |

The three entries separately input `identity/formal.tex`,
`identity/rationale.tex`, or `identity/engineering.tex` on the physical cover.
Those mechanical generated inputs, exact tools, build record, and PDFs are
sealed by the manager-owned final packet; they are not scientific source.

## Exact MAP policy/profile bytes

| Object | File and SHA-256 |
| --- | --- |
| Product-role Registry | `profiles/SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.3_CANDIDATE.md`; `a8156c9f907c8f8081b26806d33c36b55e214651ced5897e1d4441e6c8886ccb` |
| Occurrence admission | `SCI-MAP:map_upstream_admission@2`; `profiles/SCI-MAP_OCCURRENCE_ADMISSION_PROFILE_v0.2_r0.3_CANDIDATE.md`; `3eff5e0698d66b33213d691cd447db32de05f97f05f7b232414d5867f58759b9` |
| Aggregate admission | `SCI-MAP:observation_coadd_admission@2`; `profiles/SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.3_CANDIDATE.md`; `ca42154c3ba25fa28d7fad50040e3498f5ae536fc77febccee7ce693d131ea61` |

## Approved imported authorities

All locator paths below are portable beneath `inputs/` in the delivery packet.

| Role | Exact identity, locator, and SHA-256 |
| --- | --- |
| Author manifest | `SCI-MAP post-freeze successor proposed author references r0.1/2026-09-07`; `inputs/doc/scientific_contracts/studies/SCI_MAP_POST_FREEZE_RECONCILIATION_2026-09-07/AUTHOR_REFERENCES.json`; `83baf583f73784bebc6724025fd110e5545303f303c70af590b53c5696ee2c4f` |
| r0.3 owner directive | `SCI-MAP-OWNER-DIRECTIVE-v0.2-r0.3/2026-09-08`; `inputs/owner/SCI_MAP_R03_OWNER_DIRECTIVE.txt`; `bc28893eb6e7d62e149e75aa1430531821557af0bb001ae3022d42c8ef74a7e1` |
| Author packet inventory | `SCI-MAP-R03-AUTHOR-PACKET/2026-09-08`; manager-copied exact `AUTHOR_PACKET_INVENTORY.json`; source packet SHA-256 `423b1e41575a019592bcc1e2b21292f77ab0e8873c205ba18cf808835048f6b1` |
| PTC-to-MAP boundary | `SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4`; `inputs/doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY_UNIFORM_R0.4_2026-09-06.md`; `58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8` |
| PTC coefficient Registry | `SCI-PTC:analysis_gridding_coefficients@draft-0.1`; `inputs/doc/scientific_contracts/packages/SCI-PTC/v0.1/COEFFICIENT_REGISTRY_UNIFORM_R0.4_2026-09-06.md`; `433309b21d17751ce3972d196f9ba73a5f910f36e88052046928a0163fb7ca76` |
| Uniform family source | `SCI-PTC:uniform_constant@draft-0.1`; common-source SHA-256 `8bc42deb69e23c757883834f7cdb82b94e89210015333f483d15c32377a56685`; freeze manifest locator `inputs/doc/scientific_contracts/packages/SCI-PTC-COEFFICIENT-UNIFORM/v0.1/FREEZE_MANIFEST_R0.4.json`, SHA-256 `c6d23f7f20f35089eba9b5f6dd6e65aaf05561ab3da2d0f7d18e5b8256a2f016` |
| Uniform handoff | `SCI-PTC:uniform_coefficient_handoff@draft-0.1`; bound by the admitted uniform VAL Registry SHA-256 `a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f` and source register SHA-256 `fb13be222a9a8bb928fa172253f65aaf6ede081a88e901cdf6eb3799fad35f6e` |
| Original-footprint coordinate boundary | `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`; `inputs/doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE_BOUNDARY.md`; `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4` |

The admitted uniform authority and literal `@draft-0.1` identifiers remain
approved as supplied. This candidate neither revokes nor relabels their prior
evaluations. The r0.3 dependent numerical route relies on this candidate
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
