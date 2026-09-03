# SCI-CAL v0.1 Rationale r0.4 / Engineering r0.3 Build Review

Date: `2026-08-20`

Scope: artifact, mechanical, and visual review of the bounded
science/engineering consistency repair. This review does not replace the
fresh implementation-blind consistency review and does not establish
implementation conformity.

## Result

- Canonical science PDF: 14 nonempty letter-size pages; SHA-256
  `0ff32bbd63b42fca3cd8273ba9f8297213402ec6fa920ccd63afdddb9aaa09b7`.
- Canonical engineering PDF: 25 nonempty letter-size pages; SHA-256
  `3795bcf13c18f707b4935d07eab465d1c65ed241d7227a4f036f80c8e9a3af70`.
- Tectonic completed all reruns with no LaTeX warning, overfull/underfull box,
  undefined-reference, or multiply-defined-reference diagnostic in either
  final log.
- Poppler rendered all 39 pages. Whole-document contact-sheet inspection and
  full-size inspection of both title pages, the revised delivery-boundary
  figure, the four-source reference boundary, the compact crosswalk, and the
  corrected disposition table found no clipping, overlap, or unreadable
  content.
- Extracted PDF text contains every stable assumption, requirement, and edge
  ID in the engineering view; every `SCI-CAL-OWNER-Q01--Q09` ID; the r0.4/r0.3
  document relationship; and the immutable-delivery and Q06-only limitations.
- The package mechanical verifier passes 11 sequential assumptions, 50
  sequential requirements, 30 sequential edge predictions, 50 sequential
  crosswalk requirement rows, both reviewed source/PDF hashes, and the
  corrected terminology checks.

## Remaining Gate

The pair required a fresh implementation-blind consistency review and explicit
scientific-owner freeze disposition. The subsequent review passed with no
blockers; the owner-freeze disposition remains. Q01--Q09 remain open. No
implementation, scientific-validation, or production-readiness claim follows
from this build review.
