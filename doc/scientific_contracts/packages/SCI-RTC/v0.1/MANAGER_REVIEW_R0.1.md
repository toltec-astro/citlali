# SCI-RTC v0.1/r0.1 — Manager Review

Status: accepted for scientific-owner review; not scientifically approved or
frozen

Reviewed: `2026-08-17`

## Independence And Write-Boundary Review

The fresh implementation-blind author received only the exact content-bound
packet in [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). The author
did not receive implementation, audit, repair, validation, prior-work,
internal-dossier, sibling-package, status, or production material. The Git
diff contains author changes only in the permitted `src/`, `pdf/`,
`CROSSWALK.md`, `AUTHOR_DRAFT_DECISIONS.md`, and
`SCIENTIFIC_OWNER_DECISION_LEDGER.md` locations. Manager status and review
files were changed only after authorship ended.

The package verifier rechecked all four approved input hashes, including the
exact retained-core Git object. All hashes match the approved manifest.

## Scientific And Structural Review

The r0.1 draft uses one six-file shared normative core imported exactly once
by each view. It contains:

- 20 definitions;
- 24 displayed equations;
- 12 bounded assumptions;
- 54 sequential `SCI-RTC-REQ-NNN` requirements; and
- 26 sequential `SCI-RTC-PRED-NNN` falsifiable predictions.

The crosswalk covers every definition, equation, assumption, requirement, and
prediction exactly. The engineering wrapper contains no independent displayed
mathematics or normative science.

The owner modification to `RTC-SCOPE-D004` is represented literally. Under
`z_i = flxscale_i x_i`, raw donor `q` to raw target `d` uses
`flxscale_q / flxscale_d` only when both factors are valid for the exact
detector occurrences under one compatible calibration convention/domain and
the target factor is nonzero. Legacy `responsivity` is neither required nor
restored. Prediction `SCI-RTC-PRED-003` fixes the direction numerically, and
`SCI-RTC-PRED-004` fixes the unavailable cases.

The retained eight broad question families are decomposed into 28 exact owner
entries: 23 `OPEN`, one `CONDITIONAL`, and four `DEFERRED`. Each entry states
the operation or claim unavailable without authority. This decomposition does
not answer an owner choice or broaden v0.1.

## Build And PDF Review

Both sources compile with Tectonic 0.16.9 using cached resources without TeX
errors, unresolved references, box warnings, or other document warnings.
Independent manager rebuilds reproduce the same page counts and identical
extracted text page by page; PDF bytes differ only in creation timestamps.

| Output | Pages | Main narrative | SHA-256 |
| --- | ---: | ---: | --- |
| `SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | 25 | 10 substantive pre-appendix pages | `8df6816260025ed18d1f58302a7917967f5c9d194b643d02cc5a74636385a5a5` |
| `SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | 17 | — | `f92d0264901cd351a05202ff65d0fd141701357a2ff253d769b0bf6077cc0b82` |

Both outputs are US Letter, unencrypted, and contain no forms or JavaScript.
All 42 final pages were rendered through Poppler at 144 dpi and independently
inspected. No clipping, overlap, bad glyph, broken table, header/footer defect,
unreadable content, or sparse/orphan pagination remains.

## Disposition

The r0.1 pair is coherent and ready for scientific-owner review. This manager
disposition establishes neither scientific approval nor implementation
conformity, representation fidelity, observational performance,
science-impact qualification, validation, or production readiness. No
implementation or observational validation was run.
