# SCI-AST v0.1 Stage B r0.2 PDF Visual-QA Report

Status: document-production QA only; not scientific approval, implementation
conformity, empirical adequacy, validation, freeze, observational performance,
or production readiness

Prepared: `2026-08-22`

## Build And Structural Results

| PDF | Pages | Structural content | SHA-256 |
| --- | ---: | --- | --- |
| `scientific-rationale.pdf` | 9 | scientist narrative; exactly three explanatory figures; required profile/status phrases; all fonts embedded | `78e339cf2d3d6b10e1444ca6dd46fcbf274561de69cb4f6a4b39349c6e2cd47c` |
| `engineering-conformance.pdf` | 21 | six complete formal modules; 15 assumptions; exactly 90 unique requirements and 50 unique predictions; all fonts embedded | `376155de0296348055ffc4be565c1afc30c396a0b57f6a836fb1a6c079675a49` |

Both documents were built with fixed `SOURCE_DATE_EPOCH=1787270400`. The
verifier found no undefined references, LaTeX warnings, overfull boxes, blank
pages, invalid media boxes, rotations, or unembedded fonts.

## Poppler Rendering And Inspection

Every final page was rendered with Poppler `pdftoppm` at 130 dpi:

- scientist rationale: pages 1-9, 9 of 9 renders present;
- engineering conformance: pages 1-21, 21 of 21 renders present.

All 30 post-audit rendered pages were inspected, using labeled contact sheets plus the
individual page renders as the exact page inventory. Inspection covered title
and status panels, contents pages, equations, long tables, requirement and
prediction blocks, the three narrative figures, page numbers, margins, and
section transitions. It also covered the added narrative ordinary-path scope
paragraph, the revised frame/product-family definition, ASM-014, and PRED-047.

Result: zero observed clipping, overlap, broken tables, unreadable glyphs,
black boxes, missing content, incorrect rotation, inconsistent margins, or
footer/page-number defects. The rationale figures are legible and distinct:
ordered noncommuting chain; APT gauge/field-rotation and same-realization
transfer; ALIGN-grid/RTC-grid/MAP deposition stages.

## Determinism And Boundary Gate

The final verifier was run again after these records were written. It reproduced
both PDF hashes byte-for-byte and rechecked the exact installed ALIGN-boundary
digest
`359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf`.

This QA establishes only that the final document artifacts are structurally
complete and visually legible. It supplies no implementation or scientific
validation evidence.
