# SCI-ALIGN Stage B r0.3 PDF Metadata And Visual-QA Report

Status: document-production QA only; not scientific approval, implementation
conformity, empirical adequacy, validation, freeze, observational performance,
readiness, or production authorization

Prepared: `2026-08-22`

## Final PDFs

| PDF | Exact title | Author | Pages | SHA-256 |
| --- | --- | --- | ---: | --- |
| `pdf/scientific-rationale.pdf` | `SCI-ALIGN v0.1 Scientific Rationale r0.3` | Grant Wilson | 9 | `3ff4de1c6a487e14285c7c4f37771c8106e78d94f4299cd3d92604ee0b0c4538` |
| `pdf/engineering-conformance.pdf` | `SCI-ALIGN v0.1 Engineering Conformance Specification r0.3` | Grant Wilson | 17 | `800f13a4133eac3e293533541f0e58fe90d0fd0a75afbe7f1068c0321de3b2a8` |

Both documents were built under fixed `SOURCE_DATE_EPOCH=1787400000`. Poppler
reports embedded creation time `2026-08-22 08:00:00 EDT`, consistent with the
visible date and the documented deterministic-build policy.

## Rendering And Inspection

- Renderer: Poppler `pdftoppm` at 144 dpi; metadata inspected with `pdfinfo`.
- Every one of the 26 final pages was rendered to PNG.
- All pages were inspected in ordered contact sheets, with figure and dense
  formal pages also checked individually at rendered resolution.
- Inspection covered covers, status panels, contents, equations, tables,
  requirement/prediction blocks, figures, headers, footers, page numbers,
  margins, and section transitions.
- No clipped or overlapping text or graphics, broken table, blank page,
  unreadable glyph, black box, missing content, incorrect rotation, or
  footer/page-number defect was observed.
- The rationale figure distinguishes physical acquired exposure from
  valid-original exposure and gives synthesized/missing support zero added
  acquired exposure. It also distinguishes exact occurrence time from mapped
  observing-state fields.

## Structural And Deterministic Checks

The durable verifier checks searchable and nonblank pages, exact shared-module
inclusion, sequential stable identifiers, r0.3 notation and exposure taxonomy,
the locked boundary digest, metadata, and claim disclaimers. Repeated fixed-
epoch builds are required to remain byte-identical.

This report establishes document structure and visual legibility only. It
does not assess implementation, observations, scientific adequacy, or release
status.
