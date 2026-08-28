# SCI-MAP v0.1 r0.7 PDF Visual QA

Date: `2026-08-28`

Scope: rendered-document quality assurance only. This record does not establish
implementation conformity, scientific validation, achieved response,
observational performance, freeze, readiness, or production authorization.

## Stable artifacts

| Artifact | Pages | Page size | PDF version | SHA-256 |
| --- | ---: | --- | --- | --- |
| `pdf/SCI-MAP-FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT-v0.1.pdf` | 29 | US Letter, 612 x 792 pt | 1.5 | `a3de95602198019e743236925962f70d936e79c83ace0c97ef090904cd2597a8` |
| `pdf/SCI-MAP-SCIENTIFIC-RATIONALE-v0.1.pdf` | 14 | US Letter, 612 x 792 pt | 1.5 | `090122e0750d8be02298897fe0978c27d023994acc77db0ca3a74edcc299404f` |
| `pdf/SCI-MAP-ENGINEERING-CONFORMANCE-v0.1.pdf` | 22 | US Letter, 612 x 792 pt | 1.5 | `01daf915148878c50335fd8cc9da8e5adc63decedf7e54a78a3e38e499a91581` |

The stable files are byte-identical to their respective r0.7 draft aliases:

- `SCI-MAP-v0.1_FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT_r0.7-DRAFT.pdf`;
- `SCI-MAP-v0.1_SCIENCE-TEAM-RATIONALE_r0.7-DRAFT.pdf`; and
- `SCI-MAP-v0.1_ENGINEERING-CONFORMANCE_r0.7-DRAFT.pdf`.

## Inspection method and result

All 65 pages were rasterized independently with Poppler at 120 dpi. Every page
was then inspected in ordered contact sheets, with the title page, contents,
equations, tables, diagrams, appendices, headers, footers, and terminal pages
included.

Result: no clipped or overlapping text, unreadable table, broken equation,
missing figure, missing page, unintended blank page, or footer/header collision
was observed. Page numbering and document-revision labels are coherent. The
formal and engineering documents contain a few deliberately sparse continuation
pages produced by stable-ID inventories and long tables. Those pages remain
readable and are retained as cosmetic debt for the final layout-only pass; they
do not alter the r0.7 content closure.

The three LaTeX build logs were also checked before intermediate-file cleanup.
No unresolved reference, citation, overfull/underfull box, or other LaTeX
warning remained.
