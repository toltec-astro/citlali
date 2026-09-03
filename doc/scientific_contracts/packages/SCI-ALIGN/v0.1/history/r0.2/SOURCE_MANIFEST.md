# SCI-ALIGN Stage B Targeted r0.2 Source Manifest

Status: final targeted author-draft manifest; not scientific approval

Prepared: `2026-08-22`

This manifest records exact final bytes for the SCI-ALIGN Stage B targeted
r0.2 bundle. It does not hash itself. The shared boundary body also contains no
self-hash.

## Revision authority

| Authority | SHA-256 |
| --- | --- |
| `SCI-ALIGN STAGE B TARGETED REVISION DIRECTIVE` (`pasted-text.txt`) | `18f1a7a458b0e19cd545481fce3606c1e10da0eedfc5f5d588329116fb14e103` |

The remaining authorized inputs were the pre-revision Stage B files in this
SCI-ALIGN package as enumerated by the directive. No implementation, schema,
test, audit, repair, validation, production behavior, other package, web
source, or raw thread history was inspected as scientific authority.

## Final LaTeX sources

| Path | SHA-256 |
| --- | --- |
| `src/scientific-rationale.tex` | `10f7ebd5933b4c733f24a07a4ac386c7c0850a6c3a6a85eeffc298767f0439c9` |
| `src/engineering-conformance.tex` | `b1814caf663f467bf4ec060d54b63f80e60e639dc6f3580cff73b9e3f6e4e2e4` |
| `src/common/notation.tex` | `67677a5da5dbc3b40046df4b00d79e1e77576c06125f4c2638b89e8589c2bd31` |
| `src/common/definitions.tex` | `6c721f8f0bda9d10fd3a1f8bd1485d21aec9bddcd168cc2f71311312981ba845` |
| `src/common/assumptions.tex` | `326f698bbbe353669f7e84ca782bbe78f6c1464cf83407b74319d52009d47ea1` |
| `src/common/equations.tex` | `b08157dab4abef882083d0be1fb0f4e46b149469c30dccbef19ea1ac68312b9b` |
| `src/common/requirements.tex` | `3214fb209eee0f23129fe7eaa4373fc4ea0d4aae27cc937b50207ce380012110` |
| `src/common/edge_cases.tex` | `06c11677ee47d0512e98c0b1dc87dff0537b08f60993e2978dacb243e3aed47b` |

## Final registers, maps, and QA sources

| Path | SHA-256 |
| --- | --- |
| `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` | `359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf` |
| `CROSSWALK.md` | `1540ce9ed306db163bf41c3fd8b77973b0d74c34bfd5715db83e6906f6018435` |
| `OWNER_DECISION_REGISTER.md` | `74442deb67b65eca58adcd7f53656dc1c33dd916f1287826633d1484268e8582` |
| `AUTHOR_DRAFT_DECISIONS.md` | `a50adb108253d1164bc962d0d36cf6d9376a07d3430d16cc06772d3b5bcb4532` |
| `NOTATION_AND_SYMBOL_CHANGE_MAP.md` | `39d35b853bb4ac86d484c741187f669c8cf8a0317498b8ac529992493897d836` |
| `REQUIREMENT_EQUATION_PREDICTION_CHANGE_MAP.md` | `c217be359d9daca4e807366ffb3516ecee981a8fc193142f8c2f7c8ed36d9e8c` |
| `AVAILABILITY_REGISTER.md` | `c08ccdd114723cbdc0bc2345408c3a9047beeaddc18c4048c401a280dde2a8f2` |
| `PDF_VISUAL_QA_REPORT.md` | `70bb22df160d5f7396122ccb2a16c3c3e4f2a739ecd0bc7728335865723014c5` |
| `tools/verify_documents.py` | `80094f2cee9fd8d25a1fd32658a8e39055c2d5e86824a5fdb7683198ba6114d7` |

## Final PDFs

| Path | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/scientific-rationale.pdf` | 9 | `0f4f843c623897d2532d804f6e8aa480e1461768f509da793dcb23d68f2d5571` |
| `pdf/engineering-conformance.pdf` | 16 | `77363ab45288e3ab3219735c60cf65748a596565d724dbb105d1bd3fa86b971b` |

## Shared boundary equality proof

Profile identity is exactly `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; scientific owner
is Grant Wilson; the compatibility/supersession rule is preserved. The
coordinator reported installation of these final boundary bytes in the AST
packet: byte comparison passed and both ALIGN and AST copies have SHA-256
`359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf`.

## Build and claim boundary

The PDFs were built twice in deterministic mode and were byte-identical across
the repeated build. Poppler rendered all 25 final pages, all were inspected,
and final document QA found no visual defects. The horizontal-audit repair
reserves `x/r` exclusively for KID coordinates, uses `v` for generic formal
operands, and restores Figure 2's origin/method/refinement separation. These
checks establish document integrity only. Implementation conformity,
observational validation,
scientific approval, freeze, readiness, and production authorization remain
unassessed.
