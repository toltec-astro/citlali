# SCI-AST v0.1 Stage B r0.2 Source Manifest

Status: deterministic packet-control manifest for a targeted author draft;
not scientific approval, implementation conformity, empirical adequacy,
validation, freeze, or production readiness

Prepared: `2026-08-22`

## Authoritative Targeted Inputs

| Input | SHA-256 |
| --- | --- |
| supplied Stage B targeted revision directive (`pasted-text.txt`) | `15f751388d775920872228618c4bf073fcdb7fe5528fdc006789ae76a601062b` |
| installed `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` (`SCI-ALIGN_TO_SCI-AST v0.1/r0.1`) | `359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf` |

The subsequent horizontal-audit editorial feedback was supplied directly by
the coordinator, not as a separately hashed file. It requires the shared
ordinary-nonpolarimetric-coordinate-path wording and the explicit statement
that raw KID `x` is not Stokes I.

The ALIGN-led coordinator reported that the installed boundary is byte-identical
to the ALIGN authority copy. The AST author did not edit those boundary bytes.

## Canonical LaTeX Sources

| Path | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `f4281ca0639f8ec3a3b40abf8a1f318543479f0b39c9233423553dd45781f816` |
| `src/common/definitions.tex` | `45d287de975fd3b7294a7d406546354d3c8d8e8278846dc1f5b34b9cf2801f10` |
| `src/common/equations.tex` | `cb515e5d8070b3bd30bfe610d09501d1d72f36b5925ee354d9a232e4e5f8e70c` |
| `src/common/assumptions.tex` | `57615355fd299675a13d196ae5af4046395ffc2e03642575ee6a30f141731011` |
| `src/common/requirements.tex` | `8b72b9de79de24866b61222a206c29f0f3329f570a4dfbf432814d51ddb4e2dd` |
| `src/common/edge_cases.tex` | `0b9c4fa31e0c5565997f1f4282a85b17672030260a859c862f498fdefcf815f1` |
| `src/scientific-rationale.tex` | `546c711cc30e801c25334b18ac1e130336df6e7df3ed44896c9be3587b3ac712` |
| `src/engineering-conformance.tex` | `c5ac93f9abf8a8f83d5540b6e9ff67c9ad050e5afb3564deb1f120b4a2bde167` |

## Traceability, Ledger, And Proof Sources

| Path | SHA-256 |
| --- | --- |
| `CROSSWALK.md` | `1a38f27c18bc2291275f52420b99a57b645cf1fd554ca7072ca88fcf50813d14` |
| `OWNER_DECISION_REGISTER.md` | `ee90c9c87f4c50206ecc1f4bf8e8bc50c38939e4085bea61edfa902a74ba17d1` |
| `AUTHOR_DRAFT_DECISIONS.md` | `243f7e86686d85f5fea11f8d73ea211b59d29527ae1d02529da32db6abb6c49b` |
| `SLOT_DIRECTION_CHANGE_MAP.md` | `2a48d908b20974400e7968a1bac2875f7e2dd3b1b3b68299215960145f8f1614` |
| `ROLE_FACTORED_PARENTAGE_MAP.md` | `5cc9c6298ebf82a7be61825e637ca8b21d6af33f4d5dbebaab0899997bcddc80` |
| `REQUIREMENT_EQUATION_PREDICTION_CHANGE_MAP.md` | `7f3560714f66ad37d8af2030d36904d13fbdc893fc37324b719d6d7ff0779262` |
| `AVAILABILITY_REGISTER.md` | `c4489ce36686110652727ec1ed2f6e0a23e81dde470b0d9cbb2f4951bc0a9cff` |
| `BOUNDARY_IDENTITY_PROOF.md` | `3afd12492660166be8e4156e655ef478e8767f078389bebe1b7b26cfe79a0c87` |

## Deterministic Verification Tools

| Path | SHA-256 |
| --- | --- |
| `tools/verify_documents.sh` | `83843449e015a23d071036afe93a07d4cd6a51355132946a3e281dde54b2785e` |
| `tools/verify_pdf.py` | `5978468258e7bb9c9b4a416bd1b6ada749394bf5b5e474c4ed0d2a12b7502a43` |
| `tools/make_contact_sheets.py` | `0388b244ffc4bf35e2e79b000099a3a11631612049b01fcf2ff2a41623178f9f` |

## Built PDFs

| Path | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/scientific-rationale.pdf` | 9 | `78e339cf2d3d6b10e1444ca6dd46fcbf274561de69cb4f6a4b39349c6e2cd47c` |
| `pdf/engineering-conformance.pdf` | 21 | `376155de0296348055ffc4be565c1afc30c396a0b57f6a836fb1a6c079675a49` |

`SOURCE_MANIFEST.md` and `PDF_VISUAL_QA.md` are terminal control records and
are intentionally not self-hashed in this manifest.
