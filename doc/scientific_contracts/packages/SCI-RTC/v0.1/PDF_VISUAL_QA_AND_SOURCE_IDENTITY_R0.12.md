# SCI-RTC v0.1/r0.12 PDF visual QA and source identity

Date: 2026-08-21

Status: Complete artifact QA for the owner-review candidate. This record is
not scientific freeze, implementation conformity, validation, or production
evidence.

## Canonical artifacts

| Artifact | Pages | SHA-256 | Metadata |
| --- | ---: | --- | --- |
| `pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | 12 | `3c3a0d6f0b592f4c28d8a337f230a8d521ade5b874708e2605388e608d3f52c6` | US Letter; unencrypted; no form; no JavaScript; title ends `v0.1/r0.12` |
| `pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | 62 | `bd9acf1fc84fcf6f65a5b82f16fe43fbfa4181a41b8d820b7bbcee67fe4ffc61` | US Letter; unencrypted; no form; no JavaScript; title ends `v0.1/r0.12` |

## Construction and source identity

- Tectonic compiled both TeX views without warnings.
- The engineering view imports notation, definitions, equations, assumptions,
  requirements, and predictions in order and exactly once.
- The rationale imports none of the normative core and contains no independent
  displayed normative equation.
- Poppler extraction confirms r0.12 in both artifacts and finds
  `SCI-RTC-EQ-042`, `SCI-RTC-REQ-143`, `SCI-RTC-PRED-108`, and the exact r0.12
  end-of-core marker in the engineering artifact.

## All-page visual inspection

All 74 final pages were rasterized with Poppler. Contact sheets were inspected
for every page. Full-page inspection included both title pages, the rationale
pair diagram and closing page, the pair-support equation block, REQ-139--143,
PRED-104--108, the end-of-core marker, conformance routing, and the final
checklist.

Disposition: pass. No clipped content, overlap, blank or sparse spill page,
footer collision, table truncation, malformed glyph, equation collision,
broken rule, unexpected rotation, or inconsistent page geometry remains.
