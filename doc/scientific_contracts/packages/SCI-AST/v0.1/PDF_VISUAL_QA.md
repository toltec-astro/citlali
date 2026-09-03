# SCI-AST v0.1 Stage B r0.3 PDF Metadata And Visual-QA Report

Status: document-production QA only; not scientific approval, implementation
conformity, empirical adequacy, validation, freeze, observational performance,
readiness, or production authorization

Prepared: `2026-08-22`

## Final PDFs

| PDF | Exact title | Author | Pages | SHA-256 |
| --- | --- | --- | ---: | --- |
| `pdf/scientific-rationale.pdf` | `SCI-AST v0.1 Scientific Rationale r0.3` | Grant Wilson | 9 | `40b1b1759715365722fbff778e75859d4095dbd227f79c6990866003015efeba` |
| `pdf/engineering-conformance.pdf` | `SCI-AST v0.1 Engineering Conformance Specification r0.3` | Grant Wilson | 21 | `08fb199054e2c79c226c78d76b7ecca5e00f70288db20ab435715baf83edc5c9` |

Both documents were built under fixed `SOURCE_DATE_EPOCH=1787400000`. Poppler
reports embedded creation time `2026-08-22 08:00:00 EDT`, consistent with the
visible date and the documented deterministic-build policy.

## Rendering And Inspection

- Renderer: Poppler `pdftoppm` at 130 dpi; metadata inspected with `pdfinfo`.
- Every one of the 30 final pages was rendered to PNG.
- All pages were inspected in ordered contact sheets, with the three rationale
  figures and dense formal pages also checked individually at rendered
  resolution.
- Inspection covered covers, status panels, contents, equations, tables,
  requirement/prediction blocks, figures, headers, footers, page numbers,
  margins, and section transitions.
- No clipped or overlapping text or graphics, broken table, blank page,
  unreadable glyph, black box, missing content, incorrect rotation, or
  footer/page-number defect was observed.
- The rationale figures preserve the ordered noncommuting chain, APT
  gauge/field-rotation and same-realization transfer, and the distinction
  between AST continuous coordinates and MAP deposition.

## Structural And Deterministic Checks

The durable verifier checks searchable and nonblank pages, complete inclusion
of all six formal modules, sequential stable identifiers, the complete
spherical oracle, the stable RTC-grid role, absence of a general pre-MAP
kernel-dependent stencil, no-double-derotation guards, the locked boundary
digest, metadata, and claim disclaimers. Repeated fixed-epoch builds are
required to remain byte-identical.

This report establishes document structure and visual legibility only. It
does not assess implementation, observations, scientific adequacy, or release
status.
