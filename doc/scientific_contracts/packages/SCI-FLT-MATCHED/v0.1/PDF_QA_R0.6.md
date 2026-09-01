# SCI-FLT-MATCHED v0.1 r0.6 PDF QA

Date: `2026-09-01`

Status: PASS for final-build PDF mechanics and all-page visual inspection; not
scientific approval, implementation conformity, response/covariance fidelity,
observational validation, performance, readiness, production, or freeze

## Final canonical artifacts

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 45 | `a931c946ab83aab36278056106b525638d9f017ce461a1359b5ce95efc3deb3d` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 41 | `e14b5dce8559311e57217f927496bf809042d492fd76a42ac3839ed18f69d861` |

## Mechanical inspection

- Both final PDFs reopen successfully with Poppler and pypdf.
- Both are unencrypted, form-free, JavaScript-free, unrotated US Letter
  documents with one consistent 612 by 792 point page box.
- Both Tectonic logs contain zero compile errors, undefined controls or
  references, missing characters, and overfull boxes.
- The canonical PDFs are byte-identical to the final build outputs used for
  rendering and verification.

## All-page visual inspection

The final canonical bytes were rendered at 120 dpi after the last source
change: 45 scientific pages and 41 engineering pages, 86 pages total. Four
contact sheets per view cover every page. Every contact sheet was inspected,
with full-resolution inspection of title/status, stochastic-domain/operator,
lifecycle, GLS/response, requirements/consequences, option/disposition, and
closing-firewall pages.

No clipping, overlap, malformed equations, table overflow, unreadable text,
blank required content, broken page transition, or incorrect page geometry was
observed. Deliberate underfull-box warnings from narrow longtable cells do not
cause visible defects.
