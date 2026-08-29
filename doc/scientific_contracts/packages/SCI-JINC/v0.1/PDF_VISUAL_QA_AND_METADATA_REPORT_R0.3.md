# SCI-JINC v0.1 PDF Visual-QA And Metadata Report r0.3

Date: `2026-08-29`

Status: implementation-blind Stage B author-draft rendering report; visual QA
is not scientific validation, implementation conformity, achieved
performance, readiness, freeze, or production authorization

## Toolchain

- Tectonic `0.16.9`
- xdvipdfmx PDF producer `0.1`
- Poppler `pdfinfo` and `pdftoppm` `26.05.0`
- Pillow `11.1.0` for temporary contact-sheet assembly
- pypdf `6.7.2` for non-authoritative extraction checks

Both documents compiled from `src/` with Tectonic and the same six shared
modules. Final compile logs contain no unresolved reference, undefined
citation, multiply defined label, overfull box, emergency stop, or fatal
error. Remaining underfull-box diagnostics are line-spacing notices and do
not clip or overlap content.

## Metadata And File Results

| Field | Scientific Rationale | Engineering Conformance Specification |
| --- | --- | --- |
| Canonical PDF | `pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf` |
| SHA-256 | `b9b561ced6ce7a7e2cc5fe997937fcdf15563249c2f47d19590c41e27f45a0a3` | `12d3b3ae8265c86a4f2781302f7fdb6b0a353811db7020706fd837e257cca925` |
| Pages | 35 | 23 |
| Page size | US Letter, 612 x 792 pt | US Letter, 612 x 792 pt |
| Rotation | 0 degrees | 0 degrees |
| PDF version | 1.5 | 1.5 |
| Title | `SCI-JINC Scientific Rationale and Contract v0.1 r0.3` | `SCI-JINC Engineering Conformance Specification v0.1 r0.3` |
| Author | `Implementation-blind Stage B scientific author` | same |
| Subject | `Signed-coefficient JINC observation mapmaker scientific contract` | `Engineering conformance view of the SCI-JINC scientific contract` |
| Keywords | `SCI-JINC, JINC, scientific contract, Stage B draft` | `SCI-JINC, JINC, engineering conformance, Stage B draft` |
| Encryption/forms/JavaScript | none/none/no | none/none/no |

## Complete Visual Inspection

Every canonical PDF page was rasterized at 120 dpi. The 35 rationale pages
were inspected in contact sheets covering pages `1--4`, `5--8`, `9--12`,
`13--16`, `17--20`, `21--24`, `25--28`, `29--32`, and `33--35`. The 23 ECS
pages were inspected in contact sheets covering pages `1--4`, `5--8`,
`9--12`, `13--16`, `17--20`, and `21--23`.

Results:

- all expected pages rendered and page counters agree with `pdfinfo`;
- title pages, revision headers, footers, tables of contents, section starts,
  equations, long tables, requirements, predictions, cross-reference table,
  source-closure appendix, references, and blank-result template are visible;
- table continuations are coherent across page boundaries;
- the rationale narrative ends before its appendix begins on page 13;
- Equation 24 is visibly the unchanged coefficient-squared temporal-accounting
  formula in both documents;
- `SCI-JINC-PRED-017` visibly states the exact even-`n_sub` center radius and
  coefficient behavior;
- no clipped text, cropped rule, overlapping text, missing glyph, broken
  equation, unintended blank page, or unreadable metadata/header defect was
  found.

No implementation candidate, configuration, product, test, audit, validation
result, reduction, or production behavior was inspected. The visual result is
only a document-rendering check.
