# SCI-MAP v0.2/r0.1 PDF compilation and visual QA

Status: local candidate artifact check, 2026-09-07. This record establishes
document production and visual readability only. It is not scientific-owner
disposition, implementation conformance, validation, Registry activation,
freeze, readiness, or application evidence.

## Compilation

All three entry points were compiled from `src/` with Tectonic 0.16.9 in
cached-only, untrusted mode. The compiler ran to completion and resolved all
references. Final logs contain no overfull box, undefined-control-sequence,
undefined-reference, multiply-defined-label, misplaced-alignment, or rerun
warning. The engineering and formal logs retain only underfull prose/table
notices; rendered inspection found no corresponding clipping, overlap, or
unreadable content.

| View | Stable PDF | Pages | SHA-256 |
| --- | --- | ---: | --- |
| Scientist-facing rationale | `pdf/SCI-MAP-SCIENTIFIC-RATIONALE-v0.2.pdf` | 14 | `0fa060f251e1e2bfb222f6b9d87e74261fdef6d3f915ba33306231ce563d07b1` |
| Engineering conformance | `pdf/SCI-MAP-ENGINEERING-CONFORMANCE-v0.2.pdf` | 21 | `ca3e056a508b4a56f3b98db5162a7d3e78def54225b8c4dcae0963284fb7b271` |
| Formal scientific/engineering contract | `pdf/SCI-MAP-FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT-v0.2.pdf` | 30 | `e89e962487e16d1b6ee09d8cad8c18dd472bb3aa6d53f196172523c42f44f6ff` |

`pdfinfo` 26.05.0 reopened every artifact and reported letter-size,
unrotated, unencrypted PDF 1.5 with no suspect flag. Titles identify SCI-MAP
v0.2 and r0.1 candidate status.

## Render inspection

Poppler `pdftoppm` 26.05.0 rendered all 65 pages to PNG at 110 dpi. Every page
was inspected in ordered contact sheets and dense tables/equations were checked
at rendered resolution. The inspection found:

- no blank page, isolated continuation line, clipped glyph, overlap, broken
  table, or material outside the page boundary;
- complete and readable title/status blocks, running headers, footers, page
  numbers, equations, decision tables, 52 requirement records, and 25
  prediction records;
- the engineering prediction records kept intact after removing inherited hard
  group breaks, with no stranded preregistration line; and
- readable formal threshold equations and engineering scope-discrimination
  cases, including the separate inclusive normalization/science predicates.

The PNG pages, contact sheets, compiler logs, and intermediate files are
preserved under `output/qa/map-v0.2/` outside the deliverable package. They are
supporting production evidence and do not enter the scientific source binding.
