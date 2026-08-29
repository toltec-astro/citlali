# SCI-JINC v0.1 Stage B Source Manifest r0.2

Date: 2026-08-29

Status: implementation-blind Stage B author-draft source and artifact record

Repair base commit: `d54cf2259b9bc0ca3c277ed75ba4e2641b8dd904`

Required repository base: `2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c`

This manifest distinguishes normative scientific input, repair direction,
process-only dispatch/source closure, and authored output. Repository
instructions supplied process controls only. Manager, owner-ledger, and
SCI-VAL registry records listed below supplied no new scientific content.

This is not an implementation-conformity, representation-fidelity,
validation, achieved-performance, numerical-readiness, production-readiness,
or production record.

## Normative Scientific Packet

The complete and exclusive original scientific packet is the exact
`AUTHOR_PACKET_MANIFEST.md` object at commit
`88dcce8b0f7b1d78053b25831b39cf370afd47cc`, SHA-256
`52a8e843456a8cb033b7593d9b9f67fb83b0ee565c91c141d8e16d46b906140e`,
together with only its 16 admitted exact objects and the admitted portions of
the covered sources. The manifest and all 16 object digests were reverified
before authorship: 16 passed and none mismatched. The manifest plus all 14
package-local admitted objects are byte-equal to that commit.

| # | Admitted exact object | Exact source identity | SHA-256 |
| --- | --- | --- | --- |
| 1 | Successor Scope Brief | `SCOPE_BRIEF.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `5f2505c2760fc5cb07506249f33f449651aca67cccc9c444305b059674f0ddbd` |
| 2a | Frozen-core supersession cover | `AUTHOR_SUPERSESSION_COVER.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `5be650ce4e25f161211955f60696033a18adb83d7ff0c9155a776b4e184d601e` |
| 2b | Frozen signed-estimator core, readable only with 2a | `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex` | `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24` |
| 3a | LMT-method reference cover | `AUTHOR_LMT_JINC_REFERENCE_COVER.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `9b32095fc7e1773e13e70b4c21d4f402b0c7376aff0b776d41fa9f5a263b7c4f` |
| 3b | Schloerb method excerpt, original pages 15--19, readable only with 3a | `references/LMT_JINC_OTF_MAPMAKING_MEMO/Schloerb_JINC_memo_v1.1_METHOD_EXCERPT_pp15-19.pdf` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9` |
| 4 | Recovered decisions and ownership | `AUTHOR_DECISIONS_AND_OWNERSHIP.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `8398f679bd487e07f80b0ac1db240cc639150d7b0328a8b79545d76d45a3cb9d` |
| 5 | Conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `6819454e7fbbab9bfd25442f25c4ea920ef41245951a38d31bb1d28ca74d628e` |
| 6 | Generic analytic identity and TolTEC numerical-unavailability semantics | `ANALYTIC_JINC_IDENTITY.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `5346085c2fdc677012217ca879ebd7cfb29e723656af337aa5694d0ef6909bed` |
| 7 | PTC-to-JINC r0.3 successor boundary | `SCI-PTC_TO_SCI-JINC_BOUNDARY.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e` |
| 8 | AST-to-JINC r0.2 successor boundary | `SCI-AST_TO_SCI-JINC_BOUNDARY.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `efffa7059b59c89793fa1d523fb3bb48235f1ab55f7d55060af1600cbfd470a5` |
| 9 | JINC map-contribution admission profile | `SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `2db95da7e5d1b980df79993907d45ac0ababc3aa05c189bfb62dcf04ff2c2e8a` |
| 10 | Collision-free notation and units | `NOTATION_AND_UNITS.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `2dd9d1e5e1414ea3bb9befd7ed28c25a2d140fe62ba506e2921181dce09d5ec0` |
| 11 | Geometry decision table | `GEOMETRY_DECISION_TABLE.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `b811bb0ff53a4679a0a1f7538b64ffa4a3292c88445d065af348dfbaea1697cb` |
| 12 | Fixed grouping and product roles | `GROUPING_AND_PRODUCT_ROLES.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `02c14c03821b5f00d0665f31b9f8bc6aed63781efb938dc6bfdfd38c98429bb9` |
| 14a | PTC coefficient-registry successor cover | `AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md` at `88dcce8b0f7b1d78053b25831b39cf370afd47cc` | `941671d0f9113c94a15bf2de6b69bd9b21a528b41d745b6bbdb936e8e8d8646f` |
| 14b | Exact post-freeze PTC registry predecessor, readable only with 14a and only in its admitted sections | `54475956f6aefb839d43b2f0fb019a142cb64310:doc/scientific_contracts/packages/SCI-MAP/v0.1/POST_FREEZE_SCIENTIFIC_OWNER_DECISIONS_2026-08-28.md` | `4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c` |

The covered core, PDF excerpt, and PTC registry predecessor were used only
under their covers. No full source beyond the admitted portions was used.

The exact PTC boundary, AST role binding, retained JINC admission-profile
record, recovered-decision table, and grouping/product-role table are returned
unchanged through the packet entries above. Their approved bytes were not
edited during r0.2 authorship.

## Targeted Repair Direction

The only added normative direction for r0.2 is the user-supplied
`SCI-JINC v0.1 r0.2 TARGETED STAGE B REPAIR DIRECTIVE`, session attachment
`/Users/gwilson/.codex/attachments/22770832-8352-41d0-8db4-7a99b29bb0c0/pasted-text.txt`,
SHA-256
`c07505861d91459f69e7d0989f11551e2a14265c916cd5772ea48a86bb186ed2`.

The r0.1 implementation-blind draft at repair-base commit
`d54cf2259b9bc0ca3c277ed75ba4e2641b8dd904` was used only as the targeted
repair substrate. It was not treated as an independent scientific source when
it differed from the exact packet or repair directive.

## Process-Only Dispatch And Exact-Source Closure

The following records were used only to establish dispatch authorization,
exact-byte identity, supersession, or registry/source closure. They did not
add scientific content.

| Process-only record | Exact Git commit | SHA-256 |
| --- | --- | --- |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `ebc0e907fe96163e48818fec99e42cc272b2cfb4` | `c46f49ff540f2bb9a2cdde79350bf4b24e457ddc9992744c5721ee52bb248fbb` |
| `SCIENTIFIC_OWNER_ODQ_107_DECISION_2026-08-28.md` | `5835853bc0a6ffaa955dc4df05e18ad67243fe8b` | `b10768b0d264f936e1076353d15a0b1cfcee8409dfa0cef0d429851be2aa0e24` |
| `SCIENTIFIC_OWNER_ODQ_109_DECISION_2026-08-28.md` | `ba02d2b2d1d90db1da4e25579629eaaaa841a6f7` | `a9e44ea09e76cbc68ac70ee3d1e9a862f1b6ab82eff62da5ac9bbac97d28034e` |
| `SCIENTIFIC_OWNER_STAGE_A_Q002_APPROVAL_2026-08-28.md` | `ebc0e907fe96163e48818fec99e42cc272b2cfb4` | `c70e8216e816a7f98486b4c61236acc49713a5ce1d6f5ba722ad6e015e0c7e9f` |
| `SCI_VAL_REGISTRY_BINDING_2026-08-28.md` | `2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c` | `ee8f20db5febdb51e39f7157449d6c2d03a0d17058605dd5531e9ab5ca439e30` |
| `../../SCI-VAL/v0.1/SOURCE_BINDING_REGISTER_JINC_STAGE_A_Q002_2026-08-28.md` | `2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c` | `0e7ca29ee2e9cd02fb1b76cf87cc64fce6164407a7801f9b9a105ca646317e88` |
| `../../SCI-VAL/v0.1/PROFILE_REGISTRY_JINC_STAGE_A_Q002_2026-08-28.md` | `2f49b7c2ce4508a02c25bb36b7dbe02602c5f59c` | `4b9a1ebecfc847c83b59da772afd9b031ab1830e8febbb12d1a47f70ce5a1110` |

The controlling scientific results attributed to ODQ-107 and ODQ-109 were
already present in the admitted packet. The process records above were not
used to broaden or replace those results. One incidental preflight read of
the prohibited `STAGE_A_CHANGE_LOG.md` was discarded immediately and supplied
no fact, inference, decision, or authored language; the author-draft decision
record preserves that disclosure. After the scientific draft and PDFs were
complete, one final phrase-audit command was also inadvertently scoped across
package Markdown and emitted snippets from prohibited manager, prior-work,
and internal records. That output was discarded immediately and produced no
change to an author surface, decision, inference, or scientific statement;
the check was rerun only on authorized outputs.

## Authored r0.2 Outputs

| Authored output | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `e1e71d382dc96bca4ff0e5a91914ff696375ca053fe2a19ca16f7c4548bc28cc` |
| `src/common/definitions.tex` | `38a0605081abb0eb1675868170ec58d4229692af295ac917d0628a7068b64ce2` |
| `src/common/equations.tex` | `89dca78be27612a294184fda3faf4bcdf5a53912df71f8bbc2e8c9ea333d0bd6` |
| `src/common/assumptions.tex` | `655771f4c942addbc7320eb917e1a1ef56ff2a6bab770b21b66fcffa1619fd18` |
| `src/common/requirements.tex` | `031ef555ccd227584270dbfe91a1430b868a8e52ccbb5b186eab976ccd8c4b4a` |
| `src/common/edge_cases.tex` | `b4ec8791c0ade201cab0394071e561d24714e9d262b61109d340a563387ebfd8` |
| `src/scientific-rationale.tex` | `201ae390281682b99dc197002f75968551b8867d05d8ed581e017e3e3c928a15` |
| `src/engineering-conformance.tex` | `61acaf4cb4352026a377edba2869e2f18f76ff624e087261a51aadf940c472dc` |
| `CROSSWALK.md` | `1650db426de15cbc8429c11381b1bfcec2ac4dd79600ea71954707a2b8954782` |
| `AUTHOR_DRAFT_DECISION_RECORD.md` | `e150ee47b9ba4758cbbb4e1b47893c1fdb1d0d74433c943a0d28156472753dbe` |
| `STAGE_A_SCOPE_PARITY_REPORT_R0.2.md` | `e2f61f4c3293c29b20ecdff2c5bec6e986789af5a0d09016d2f1c4e156f0553e` |
| `DISCRETE_GEOMETRY_AND_WCS_METRIC_DECISION_RECORD_R0.2.md` | `88fd3a7fb3bc9849a84e667cf41175db53a5de57e12e3f81d213acb79759612a` |
| `NUMERICAL_ADEQUACY_DISPOSITION_R0.2.md` | `b2440248306850aec0f5e6a2070e900fe916ddf1eb01b8b525c721b83d9047ec` |
| `RESPONSE_AND_COVARIANCE_PRODUCT_ROLE_TABLE_R0.2.md` | `29c8f4571d38af591ae05840b7698d5d5cd7a7aa11e3aa9836f52549adba786a` |
| `NOTATION_EQUATION_REQUIREMENT_PREDICTION_CHANGE_MAP_R0.2.md` | `a250838c3a2b1a783abf96caa03b54508653f220c0fc2ccfd53fb66dabbedf9d` |
| `RATIONALE_ECS_PARITY_REPORT_R0.2.md` | `10bce95c3137b2c923f85fde417646f218768f555ae6835423460e56863b014d` |
| `PDF_VISUAL_QA_REPORT_R0.2.md` | `9d98b4831e0ca6de19cace24d54d7721b21894bfe4ac91596ea54e6da932b43b` |
| `pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `3deab8fffb2af93375a187a5ba0e177921398f44e88963ef2d7a1b3e441331dc` |
| `pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `15fba087df7bff0560aca65854ce74e3a8de037614623b877fd6c885f3a9032a` |

`STAGE_B_SOURCE_MANIFEST_R0.2.md` intentionally does not contain its own
digest. The final local Git commit binds this manifest and every authored
output as one coherent snapshot.

## Closure

No unlisted scientific source, implementation candidate, configuration,
test, audit, validation material, reduction, production artifact, or web
source contributed to the r0.2 scientific draft. The draft remains pending
later manager, scientific-owner, consistency/freeze, and any separately
authorized assessment stages.
