# SCI-JINC v0.1 PDF Visual QA Report r0.2

Date: 2026-08-29

Status: implementation-blind Stage B author-draft presentation record

Scope: both canonical r0.2 PDFs

This report records compilation and visual presentation checks only. It is
not an implementation-conformity, representation-fidelity, scientific
validation, achieved-performance, numerical-readiness, production-readiness,
or production claim.

## Toolchain And Build

- Tectonic: `0.16.9`
- PDF producer reported by `pdfinfo`: `xdvipdfmx (0.1)`
- Poppler `pdfinfo` and `pdftoppm`: `26.05.0`
- render resolution: 120 dpi PNG

From `doc/scientific_contracts/packages/SCI-JINC/v0.1/src/`, the two sources
were compiled independently with the equivalent commands:

```text
/opt/homebrew/bin/tectonic --keep-logs --outdir <temporary>/rationale scientific-rationale.tex
/opt/homebrew/bin/tectonic --keep-logs --outdir <temporary>/engineering engineering-conformance.tex
```

Both commands completed successfully. The final logs contained no LaTeX
errors and no overfull boxes; only underfull-box warnings remained. The
resulting PDFs were copied to the canonical package paths before rendering.

## Canonical PDF Identity

| PDF | SHA-256 | Pages | Page size | Bytes |
| --- | --- | ---: | --- | ---: |
| `pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `3deab8fffb2af93375a187a5ba0e177921398f44e88963ef2d7a1b3e441331dc` | 33 | 612 x 792 pt (US Letter) | 180964 |
| `pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `15fba087df7bff0560aca65854ce74e3a8de037614623b877fd6c885f3a9032a` | 22 | 612 x 792 pt (US Letter) | 155703 |

The embedded titles are respectively:

- `SCI-JINC Scientific Rationale and Contract v0.1 r0.2`
- `SCI-JINC Engineering Conformance Specification v0.1 r0.2`

Neither PDF is encrypted, rotated, tagged as suspicious, or contains a form
or JavaScript according to `pdfinfo`.

## Render And Inspection Coverage

Every page of each canonical PDF was rendered with the equivalent command:

```text
pdftoppm -png -r 120 <canonical-pdf> <temporary-render-prefix>
```

All rendered pages were assembled into four-page contact sheets and visually
inspected. Coverage was:

- scientific rationale: pages 1--4, 5--8, 9--12, 13--16, 17--20, 21--24,
  25--28, 29--32, and 33;
- engineering conformance specification: pages 1--4, 5--8, 9--12, 13--16,
  17--20, and 21--22.

Inspection checked page boundaries, headers, footers, page numbering,
paragraph flow, equations, code-like identifiers, tables, long-table
continuations, appendix transitions, and obvious missing-glyph or unreadable
text failures.

## Findings

- all 33 rationale pages and all 22 ECS pages were present and legible;
- no clipping, overlap, truncated table, broken header/footer, missing page
  number, or unreadable text was observed;
- the rationale main narrative ends before the appendices, which begin on
  page 13;
- the whitespace before the shared canonical modules on ECS page 7 is an
  intentional section transition;
- the sparse final ECS template page is intentional and contains no visual
  defect.

Visual QA disposition: **pass for the r0.2 Stage B author-draft presentation
snapshot identified by the hashes above**.
