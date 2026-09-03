# SCI-RTC v0.1/r0.6 Manager Review

Date: `2026-08-18`

Disposition: the bounded r0.6 scientific correction is complete and accepted
for scientific-owner review and freeze disposition. This is not scientific
approval, implementation conformity, representation fidelity, observational
validation, science-impact qualification, or production readiness.

## Independence And Authority

The revision remains implementation-blind. It applies the supplied r0.6
scientific-owner review and the owner's three explicit decisions without
inspecting source behavior, tests, audits, reductions, or production evidence.
The six shared files under `src/common/` remain the sole normative authority and
are imported exactly once by each audience view. The engineering wrapper
contains no independent displayed mathematics.

## Decision And Scientific-Consistency Review

- Atmospheric templates are diagnostic evidence only in RTC v0.1. They are not
  numerically subtracted from science $x$; common-mode removal requires PTC or
  separately approved successor authority.
- Shift learning uses the original paired data with spike candidates masked,
  excluded, or robustly downweighted. Donor replacement begins only after
  segmentation, remains within stable segments, and never crosses an unresolved
  or accepted boundary.
- Conditioned $x$ is the sole required numerical RTC output. Immutable raw $r$
  and every causal diagnostic and selector remain in the atomic bundle; a
  conditioned $r$ requires its own explicit channel authority.
- The IQ-to-$x/r$ mapping identity and leakage estimators now retain complete
  coordinate, support, response, uncertainty, and validity authority.
- Plateau and pre/post-shift semantics, replacement causality, cross-channel
  contamination protections, and the new scientific falsifiers are mutually
  consistent across rationale, requirements, predictions, and crosswalk.

## Mechanical, Build, And Visual Review

- Shared inventory: 38 definitions, 37 displayed equation tags, 12 assumptions,
  108 sequential requirements, and 71 sequential predictions.
- Records: 24 author decisions and 74 owner-ledger entries: 64 open, five
  resolved, and five deferred.
- Exact crosswalk coverage, stable identifiers, approved packet hashes, and
  once-per-view shared-core imports pass the package verifier.
- Both PDFs compile without warnings or errors. The rationale is 49 pages with
  exactly 12 substantive pre-appendix narrative pages; the engineering view is
  41 pages.
- All 90 final Poppler-rendered pages were inspected. No clipping, overlap,
  broken table, bad glyph, header/footer defect, orphaned heading, or unreadable
  page remains.
- Both canonical PDFs are US Letter, unencrypted, and contain no forms or
  JavaScript.
- Canonical rationale SHA-256:
  `f337edafcf552087e674530255dc6f1d178b7d27d4a6718281129132e609c024`.
- Canonical engineering SHA-256:
  `32c5555a1ca9e4ddb1cbc3190a142f28fcdc44f8256d901a79bc8c91850542a9`.

The remaining 64 open owner decisions retain named unavailable effects. No
numerical default, implementation result, conformity result, validation result,
or readiness claim was inferred during this review.
