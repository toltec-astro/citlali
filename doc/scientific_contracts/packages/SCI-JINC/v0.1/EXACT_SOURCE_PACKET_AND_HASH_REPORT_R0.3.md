# SCI-JINC v0.1 Exact Source Packet And Hash Report r0.3

Date: `2026-08-29`

Status: final implementation-blind freeze source and artifact report for the
conditional `SCI-JINC v0.1/r0.3` scientific authority

Required repository launch commit:
`2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c`

Targeted r0.3 repair base commit:
`334946689f39fb81af55a0ea39240490bd98f65f`

This report distinguishes normative scientific input, owner repair direction,
process-only dispatch/source closure, the exact owner-approved r0.2 repair
substrate, and r0.3 authored output. Repository instructions supplied process
controls only. No implementation candidate, source code, schema,
configuration, test, product, reduction, audit, validation material,
production behavior, or web source supplied scientific content.

This is not an implementation-conformity, representation-fidelity,
validation, achieved-performance, numerical-readiness, production-readiness,
or production record.

## Original Normative Scientific Packet

The original scientific packet is the exact
`AUTHOR_PACKET_MANIFEST.md` object at commit
`88dcce8b0f7b1d78053b25831b39cf370afd47cc`, SHA-256
`52a8e843456a8cb033b7593d9b9f67fb83b0ee565c91c141d8e16d46b906140e`,
together with only its 16 admitted exact objects and only the source portions
admitted through its covers.

The manifest digest and every admitted-object digest were independently
reproduced before r0.3 authorship: **16/16 passed; zero mismatched**. The
manifest and all 14 package-local admitted objects were byte-equal to the
named commit. The three external Git/PDF source objects were read only under
their covers and only in the admitted portions.

| # | Admitted exact object | Exact identity | Reproduced SHA-256 |
| --- | --- | --- | --- |
| 1 | Successor Scope Brief, including ODQ-107 authority | `SCOPE_BRIEF.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `5f2505c2760fc5cb07506249f33f449651aca67cccc9c444305b059674f0ddbd` |
| 2a | Frozen-core supersession cover | `AUTHOR_SUPERSESSION_COVER.md` at the manifest commit | `5be650ce4e25f161211955f60696033a18adb83d7ff0c9155a776b4e184d601e` |
| 2b | Frozen signed-estimator core, readable only with 2a | `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex` | `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24` |
| 3a | LMT-method reference cover | `AUTHOR_LMT_JINC_REFERENCE_COVER.md` at the manifest commit | `9b32095fc7e1773e13e70b4c21d4f402b0c7376aff0b776d41fa9f5a263b7c4f` |
| 3b | Schloerb method excerpt, original pages 15--19, readable only with 3a | `references/LMT_JINC_OTF_MAPMAKING_MEMO/Schloerb_JINC_memo_v1.1_METHOD_EXCERPT_pp15-19.pdf` at the manifest commit | `a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9` |
| 4 | Inherited decision and ownership table | `AUTHOR_DECISIONS_AND_OWNERSHIP.md` at the manifest commit | `8398f679bd487e07f80b0ac1db240cc639150d7b0328a8b79545d76d45a3cb9d` |
| 5 | Conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` at the manifest commit | `6819454e7fbbab9bfd25442f25c4ea920ef41245951a38d31bb1d28ca74d628e` |
| 6 | Generic analytic identity and TolTEC numerical-unavailability semantics | `ANALYTIC_JINC_IDENTITY.md` at the manifest commit | `5346085c2fdc677012217ca879ebd7cfb29e723656af337aa5694d0ef6909bed` |
| 7 | PTC-to-JINC boundary | `SCI-PTC_TO_SCI-JINC_BOUNDARY.md`, identity `SCI-PTC_TO_SCI-JINC v0.1/r0.3` | `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e` |
| 8 | AST-to-JINC boundary | `SCI-AST_TO_SCI-JINC_BOUNDARY.md`, identity `SCI-AST_TO_SCI-JINC v0.1/r0.2` | `efffa7059b59c89793fa1d523fb3bb48235f1ab55f7d55060af1600cbfd470a5` |
| 9 | JINC admission profile | `SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md`, immutable Registry identity `SCI-JINC:jinc_map_contribution@1` | `2db95da7e5d1b980df79993907d45ac0ababc3aa05c189bfb62dcf04ff2c2e8a` |
| 10 | Collision-free notation and units | `NOTATION_AND_UNITS.md` at the manifest commit | `2dd9d1e5e1414ea3bb9befd7ed28c25a2d140fe62ba506e2921181dce09d5ec0` |
| 11 | Geometry decision table | `GEOMETRY_DECISION_TABLE.md` at the manifest commit | `b811bb0ff53a4679a0a1f7538b64ffa4a3292c88445d065af348dfbaea1697cb` |
| 12 | Fixed grouping and product roles | `GROUPING_AND_PRODUCT_ROLES.md` at the manifest commit | `02c14c03821b5f00d0665f31b9f8bc6aed63781efb938dc6bfdfd38c98429bb9` |
| 14a | PTC coefficient-registry successor cover | `AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md` at the manifest commit | `941671d0f9113c94a15bf2de6b69bd9b21a528b41d745b6bbdb936e8e8d8646f` |
| 14b | Exact post-freeze PTC Registry predecessor, readable only with 14a and only in admitted sections | `54475956f6aefb839d43b2f0fb019a142cb64310:doc/scientific_contracts/packages/SCI-MAP/v0.1/POST_FREEZE_SCIENTIFIC_OWNER_DECISIONS_2026-08-28.md` | `4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c` |

The inherited-decision table and the grouping/product-role table are returned
unchanged through packet objects 4 and 12. They were not edited during r0.3
authorship.

## Targeted Owner Direction And Exact Repair Substrate

| Object | Exact binding | Authority state and use |
| --- | --- | --- |
| Targeted r0.2 repair directive | Session attachment `/Users/gwilson/.codex/attachments/22770832-8352-41d0-8db4-7a99b29bb0c0/pasted-text.txt`; SHA-256 `c07505861d91459f69e7d0989f11551e2a14265c916cd5772ea48a86bb186ed2` | Exact scientific-owner repair direction. The resulting r0.2 architecture is explicitly owner-approved as the basis for r0.3 by the final targeted directive. |
| Owner-approved r0.2 Stage B substrate | Commit `334946689f39fb81af55a0ea39240490bd98f65f`; tree `f870f36067dbc7a24bc86708b3ef06920ffb6138`; `STAGE_B_SOURCE_MANIFEST_R0.2.md` SHA-256 `371bc49826985903e4b9621724f509f0408fb1dd26055f37f5701349800422e1` | Exact bounded repair substrate; not an independent authority where it differs from the original packet, directives, or direct owner disposition. |
| Final targeted r0.3 owner-review directive | Session attachment `/Users/gwilson/.codex/attachments/35579593-a003-498f-ad21-07eae3428719/pasted-text.txt`; SHA-256 `4878e1745e085b4e33d2e71f1190299d72f2cfd7b2215e36a9e8405a977bd207` | Normative final repair scope, phase-lattice question, claim boundary, naming, source lifecycle, terminology, and deliverables. |
| Final owner-review/freeze-preflight directive | Session attachment `/Users/gwilson/.codex/attachments/8ee8dc6e-bece-42c0-aad1-3c294aab175a/pasted-text.txt`; SHA-256 `958cffeac67c11e916527c0f78e9c80d648f68d5eec38a0607fc4af1511dddec` | Accepts the r0.3 architecture, directs the bounded freeze, and requires separate exact center-tie authority rather than inference. |
| `SCI-JINC-DEC-PHASE-CENTER-001`, phase statement | Exact literal `I approve disposition A`; literal-text SHA-256 `9a70cbc63c0c79a7db70ad9796481fb0fe3f1f4c2d7524820b54cec68b8b1620` | Retains every positive `n_sub`, the half-open upper-bin convention, and its even-lattice exact-zero-phase offset. |
| `SCI-JINC-DEC-PHASE-CENTER-001`, center statement | Exact submitted literal `I approve the positive-axis half-pixel center-tie convention \\(c=\lfloor u+\tfrac12\rfloor\\) as part of SCI-JINC v0.1.`; literal-text SHA-256 `3b79351f7661e2432a5426fba3a16e9710c2fae0b34fd9b8f60dd45bca837ecb` | Separately approves `c=floor(u+1/2)` and the positive-axis half-pixel center tie; it is not inferred from the phase statement. |

No implementation behavior selected or modified either owner statement.

## Process-Only Dispatch And Exact-Source Closure

The following exact records were used only to establish dispatch
authorization, exact-byte identity, owner status, supersession, or
Registry/source closure. They did not add scientific content.

| Process-only record | Exact Git commit | Reproduced SHA-256 |
| --- | --- | --- |
| Owner ledger: `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `ebc0e907fe96163e48818fec99e42cc272b2cfb4` | `c46f49ff540f2bb9a2cdde79350bf4b24e457ddc9992744c5721ee52bb248fbb` |
| ODQ-107 decision record | `5835853bc0a6ffaa955dc4df05e18ad67243fe8b` | `b10768b0d264f936e1076353d15a0b1cfcee8409dfa0cef0d429851be2aa0e24` |
| ODQ-109 decision record | `ba02d2b2d1d90db1da4e25579629eaaaa841a6f7` | `a9e44ea09e76cbc68ac70ee3d1e9a862f1b6ab82eff62da5ac9bbac97d28034e` |
| Q002 approval record | `ebc0e907fe96163e48818fec99e42cc272b2cfb4` | `c70e8216e816a7f98486b4c61236acc49713a5ce1d6f5ba722ad6e015e0c7e9f` |
| `SCI_VAL_REGISTRY_BINDING_2026-08-28.md` | `2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c` | `ee8f20db5febdb51e39f7157449d6c2d03a0d17058605dd5531e9ab5ca439e30` |
| Exact JINC-specific `SOURCE_BINDING_REGISTER_JINC_STAGE_A_Q002_2026-08-28.md` snapshot | `2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c` | `0e7ca29ee2e9cd02fb1b76cf87cc64fce6164407a7801f9b9a105ca646317e88` |
| Exact JINC-specific `PROFILE_REGISTRY_JINC_STAGE_A_Q002_2026-08-28.md` snapshot | `2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c` | `4b9a1ebecfc847c83b59da772afd9b031ab1830e8febbb12d1a47f70ce5a1110` |

The complete exact JINC-specific snapshot files are locked sources. No
ambient current Registry or nearby successor substitutes for them. Any change
to a bound object requires a versioned SCI-JINC successor. The controlling
ODQ-107 and ODQ-109 scientific results were already present in the admitted
packet; the process records do not broaden or replace them.

The prior r0.2 author record discloses two accidental prohibited-output reads.
Their output was discarded and supplied no scientific content. No such new
read occurred during r0.3 repair.

## Final r0.3 Authored Outputs

| Authored output | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `fed76501dfc57540a4f383bf329d35118508f1e96a143b1cde8e09078d6dade1` |
| `src/common/definitions.tex` | `9e6f3ff015c753c879ff03be65fe981ac4f6ad2fc572d4a2f13f8a5240a12e1e` |
| `src/common/equations.tex` | `2b1870e92f9a0e6141fdda1a8865babae41208aafb73fc233f4befc0e1b665c1` |
| `src/common/assumptions.tex` | `15b811ab6ace92aa2d1713ae19b92454cb865e8862b82a599f94eca1003a1765` |
| `src/common/requirements.tex` | `207a85acb31a4f381b289781706c9f14058d330ff847e99023e9e5714c4d4dff` |
| `src/common/edge_cases.tex` | `815c70e925f103d989e4ec015a64d69ac0710c1a0c57789a4dfe754bdb81bd2d` |
| `src/scientific-rationale.tex` | `7cabea85eaa5ad9afbb0914c585d2fe7917806c9919964a465c0d9742fdb55e2` |
| `src/engineering-conformance.tex` | `a8cc9b66d22f1c4c0e9dc53c46724721f38fa2b2d267f74e7341b359874c19aa` |
| `CROSSWALK.md` | `df2bbb1f8eec53c91497d52b85591e66f86639f76c63686688367c96e309d2e5` |
| `AUTHOR_DRAFT_DECISION_RECORD.md` | `3245ff3bdf7ae2636a9c86b7fa24ff4ad8f1be147c6c1c30ab40df0abb6ded68` |
| `PHASE_LATTICE_OWNER_DISPOSITION_R0.3.md` | `0026111ff3c36bb5aea3ad1a1e8a2d0b99d09d288eb5c91488a5a1abf85b1bbd` |
| `NUMERICAL_CERTIFICATE_CLAIM_BOUNDARY_AMENDMENT_R0.3.md` | `6ea6eb30a9f9622255b3dc04f91d19a663e6ee5228a6a18bd90b7167e9f5577f` |
| `PROFILE_NAME_AND_SUPERSESSION_DISPOSITION_R0.3.md` | `be4985b8d190f6bb387afb279275b9a066b52d0df28c0ed5c53bbadbb25ffeff` |
| `SOURCE_REGISTRY_LIFECYCLE_DISPOSITION_R0.3.md` | `d8665e621a54478c58e9503c4d9bb42dd002ba750171cb9c2e3c8d380cc1aa28` |
| `REQUIREMENT_EQUATION_PREDICTION_SEMANTIC_CHANGE_MAP_R0.3.md` | `3769a03ff4008956be6d672e0f57781da4ec0f696b89a257f6d7813b42306d1f` |
| `RATIONALE_ECS_PARITY_REPORT_R0.3.md` | `7e47cfccb50e36b33960aba410b957940ff6e6ab6ca10960d36d77be39867947` |
| `PDF_VISUAL_QA_AND_METADATA_REPORT_R0.3.md` | `f5c361a76a862e31760c57d691430b43deedb9826c7e0ed93ed0c71a7fe62f4e` |
| `pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `53ed941658ae1205950a8bc533d569cc85b246a40bb6e448fbbc6d7f0509a7b8` |
| `pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `6b78e80bb485815292972c5de60c444954d7bb62902799d6fa4c3f421766114a` |

The SHA-256 of the six shared modules concatenated in the exact include order
is `ca6650743af30e34940b7360a92c66f6638e993e07648b329e05f107b3b9e657`.
Both views include that one core exactly once in the same order.

`EXACT_SOURCE_PACKET_AND_HASH_REPORT_R0.3.md` intentionally omits its own
digest. The final local Git commit binds this report and every authored output
as one coherent snapshot.

## Closure

The frozen authority surfaces contain 44 sequential requirements and 36
sequential predictions without gaps or renumbering. The numerical route
remains typed unavailable because no registered/selected/realized
JINC-permitted coefficient family and payload, authorized TolTEC numerical
parameter set, or exact numerical-adequacy profile/certificate is admitted or
supplied. No inherited 45 m denominator, shape values, or mode-dependent
`r_max` are authorized.

The frozen conditional implementation-independent r0.3 authority makes no
implementation-conformity, representation-fidelity, numerical or
observational validation, achieved-performance, response/covariance-fidelity,
parameter-adequacy, readiness, production, or production-authorization claim.
Any later scientific correction requires a versioned successor and shall not
modify these frozen bytes.
