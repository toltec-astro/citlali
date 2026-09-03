# PDF Visual QA and Metadata Report

Status: passed on 2026-08-30.

| PDF | SHA-256 | Pages rendered/declared | Metadata and extracted text | Visual inspection |
| --- | --- | --- | --- | --- |
| `SCI-NOI_v0.1_STAGE_B_SCIENTIFIC_RATIONALE_DRAFT_r0.2.pdf` | `f4f73fd1758a25c269ace8fdff88eaf9aae6813987e457bdf43907f1a4bffd4a` | 6/6 | Fixed creator/date metadata and draft/identity text passed. | All pages passed; no clipping, overlap, missing glyph, or unreadable region. |
| `SCI-NOI_v0.1_STAGE_B_NORMATIVE_CORE_DRAFT_r0.2.pdf` | `1cede3c209c69171ef0007d4da0312c1432fa9bdabdf9d10622c995169a5f466` | 15/15 | Fixed creator/date metadata and draft/identity text passed. | All pages passed; module order is correct; notation, equations, requirements, and predictions are legible with no clipping or overlap. |
| `SCI-NOI_v0.1_STAGE_B_ENGINEERING_CONFORMANCE_DRAFT_r0.2.pdf` | `3fcadc430cc1752c9a97d4d631dd8cba446e04d9cf151fbdd8ddaf64499a31d2` | 10/10 | Fixed creator/date metadata and draft/identity text passed. | All pages passed; traceability appendix is complete and legible with no clipping or overlap. |

Poppler rendered all 31 pages at 110 dpi. Contact-sheet inspection covered
every page; full-size inspection additionally covered dense notation,
requirements, and traceability pages. `pypdf` verified declared page counts,
creator metadata, extractable draft/identity markers, and PDF hashes.

The verifier rebuilt all three PDFs in a fresh temporary directory and proved
byte-for-byte equality. This QA is document-rendering evidence only; it is not
scientific validation, implementation conformity, readiness, or production
evidence.
