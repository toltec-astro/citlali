# SCI-ALIGN Stage B Targeted r0.2 PDF Visual-QA Report

Status: final document-layout QA; not scientific validation or implementation
conformity

Prepared: `2026-08-22`

## Final PDFs

| PDF | Pages | Nonblank pages | SHA-256 |
| --- | ---: | ---: | --- |
| `pdf/scientific-rationale.pdf` | 9 | 9 | `0f4f843c623897d2532d804f6e8aa480e1461768f509da793dcb23d68f2d5571` |
| `pdf/engineering-conformance.pdf` | 16 | 16 | `77363ab45288e3ab3219735c60cf65748a596565d724dbb105d1bd3fa86b971b` |

The scientist-facing rationale meets the requested 7-9 page range and contains
two explanatory vector figures. The Engineering Conformance Specification
contains the complete shared modules, all 20 canonical equations, all 55
stable requirements, and all 26 stable predictions.

## Rendering and inspection

- Renderer: Poppler `pdftoppm` 26.05.0 at 144 dpi.
- Metadata/page check: Poppler `pdfinfo` 26.05.0.
- Every one of the 25 final pages was rendered to PNG.
- All pages were inspected in contact-sheet sequence; the two explanatory
  figure pages were additionally inspected at original rendered resolution.
- A first-pass overlap in Figure 2 was corrected by separating the timeline
  from a single target-slot fact card; the PDF was rebuilt and all final pages
  were rerendered.
- Final inspection found no clipped text, overlapping text or graphics,
  broken tables, blank pages, black squares, unreadable glyphs, stray tool
  tokens, or inconsistent headers/footers/page numbers.
- Figure 1 clearly shows native detector/telescope occurrences, checked time
  relation, one detector-reference grid, paired `x/r`, and field mappings.
- Figure 2 clearly distinguishes native event time, nominal slot time,
  assignment residual, origin/method/source facts, nominal cell support, and
  acquired exposure.
- After the horizontal audit, all 25 pages were rendered and inspected again.
  The generic `v` operand family is clear in equations `004`, `005`, `007`,
  `011`, `013`, and `014`, while every displayed `x/r` remains a paired KID
  coordinate.
- Figure 2 now lists canonical origin as
  `original/synthesized/unavailable`, lists linear/circular interpolation on
  the method axis, and shows `original-invalid/guarded` as refinements. The
  widened fact card is unclipped, nonoverlapping, and legible.

## Deterministic and structural checks

The final verifier performed two complete deterministic Tectonic builds under
fixed `SOURCE_DATE_EPOCH`; both PDFs were byte-identical across the repeated
build. It also confirmed searchable text, every page nonblank, narrative page
range, two figure environments, exact shared-module inclusion in the
engineering view, sequential IDs, targeted notation, the locked boundary
digest, and explicit status disclaimers.

This report assesses PDF structure and appearance only. It does not establish
scientific approval, implementation behavior or conformity, observational
validation, freeze, readiness, or production authorization.
