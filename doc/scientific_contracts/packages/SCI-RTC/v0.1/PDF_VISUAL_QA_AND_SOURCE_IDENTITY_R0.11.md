# SCI-RTC v0.1/r0.11 PDF visual QA and source identity

Date: 2026-08-21

Status: Complete artifact QA for the owner-review candidate; not scientific
freeze, implementation conformity, validation, or production evidence.

## Canonical artifacts

| Artifact | Pages | SHA-256 | Metadata |
| --- | ---: | --- | --- |
| `pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | 15 | `f92cefdd064a250466d75be7b1aafb9725c22ff2930a8fecef5a9e1db7315dbd` | US Letter; unencrypted; no form; no JavaScript; title ends `v0.1/r0.11` |
| `pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | 60 | `b11dbf3bfc835f7bf144d4f6088960b3b3a7ff0409a3d93ddcd5514ff8bc24d5` | US Letter; unencrypted; no form; no JavaScript; title ends `v0.1/r0.11` |

## Construction and source identity

- Tectonic compiled each TeX view from `src/` without warnings.
- The engineering view imports, in order and exactly once, notation,
  definitions, equations, assumptions, requirements, and predictions.
- The rationale imports none of the normative core and contains no independent
  displayed normative mathematics.
- Poppler text extraction confirms r0.11 in both artifacts and confirms
  `SCI-RTC-EQ-040`, `SCI-RTC-REQ-138`, `SCI-RTC-PRED-103`, and the exact r0.11
  end-of-core marker in the engineering artifact.

## All-page visual inspection

All 75 final pages were rasterized with Poppler. Contact sheets were inspected
for every page. Full-page inspection included both title pages, rationale
diagram and closing page, the r0.11 equation block, early and late requirement
tables, early and late prediction tables, end-of-core marker, conformance
routing pages, and final checklist.

Disposition: pass. No clipped content, overlap, blank spill page, footer
collision, table truncation, malformed glyph, equation collision, broken rule,
unexpected rotation, or inconsistent page geometry remains.
