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
| Engineering conformance | `pdf/SCI-MAP-ENGINEERING-CONFORMANCE-v0.2.pdf` | 21 | `f5912c7be6f8683a843171388800882041a55605b79830f228d2137a122acace` |
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

## Exact-SHA review repair

Independent review of candidate
`075ca6068c0329824af5b779dc185ea64b93b784` identified two engineering-view
exposition discrepancies. The repaired view now distinguishes an explicitly
requested preconstruction canonical grid from selection of a compatible common
coadd target for otherwise-compatible existing bundles, without permitting
relabeling or resampling to manufacture compatibility. It also identifies
\(\Pi\) as the immutable realized plan and \(J_{\rm out}\) as the exact
output-row selector.

The shared scientific source, rationale source/PDF, and formal source/PDF were
unchanged. Only the engineering source was recompiled. Its final log again has
no overfull box, undefined control/reference, label, alignment, or rerun
warning; the same three harmless underfull prose notices remain. Poppler
reopened and rendered all 21 repaired engineering pages at 110 dpi, and every
page was reinspected without blank pages, isolated continuations, clipping,
overlap, or broken tables. The affected execution-order and discrimination
table text is readable on rendered pages 4--5.

The final PNG pages, contact sheets, compiler logs, and intermediate files are
preserved under the isolated repair tree's `qa/map-v0.2/` outside the
deliverable package. Pre-repair engineering render and compiler evidence is
retained under `qa/map-v0.2/pre-repair/`. These are supporting document-production
records and do not enter the scientific source binding.
