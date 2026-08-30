# PDF Visual QA and Metadata Report

Status: passed on 2026-08-30.

| PDF | SHA-256 | Pages rendered/declared | Metadata/text | Visual result |
| --- | --- | --- | --- | --- |
| `SCI-NOI_v0.1_STAGE_B_SCIENTIFIC_RATIONALE_DRAFT_r0.3.pdf` | `54b314e613a9a0da05b4cd2613e187bfc34976f6574fc7ff942197cbbc69d550` | 5/5 | Grant Wilson owner/author; fixed creator/date; identity/draft text extractable. | Passed; diagram and route table legible; no clipping, overlap, or missing glyph. |
| `SCI-NOI_v0.1_STAGE_B_NORMATIVE_CORE_DRAFT_r0.3.pdf` | `2adf601433e27c52da78a1eff95fa7caa9a0c03bb85b9078c955e36366f0b986` | 13/13 | Grant Wilson owner/author; fixed creator/date; identity/draft text extractable. | Passed; exact modules ordered and legible; dense requirements inspected full-size. |
| `SCI-NOI_v0.1_STAGE_B_ENGINEERING_CONFORMANCE_DRAFT_r0.3.pdf` | `eb003770808f51dccb9579fd55c342dc47a3c29d89e20c01c887f36b46be70e1` | 11/11 | Grant Wilson owner/author; fixed creator/date; identity/draft text extractable. | Passed; owner traceability complete and legible; dense appendix inspected full-size. |

Poppler rendered all 29 pages at 110 dpi. Contact-sheet inspection covered
every page; full-size inspection additionally covered all covers, the route
diagram, dense requirements, and dense owner-traceability content. `pypdf`
verified page counts, owner/creator metadata, extracted identity markers, and
hashes. A fresh temporary rebuild proved byte-for-byte equality for all PDFs.

This is rendering/build evidence only. It is not scientific validation,
implementation conformity, performance, readiness, or production evidence.
