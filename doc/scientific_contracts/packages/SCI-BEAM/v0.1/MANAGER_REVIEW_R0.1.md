# SCI-BEAM v0.1 — Contract-Manager Review of r0.1

Status: ready for scientific-owner review; not accepted or frozen

Review date: `2026-08-16`

## Review outcome

The implementation-blind author produced one shared normative core, a
scientific rationale, an engineering conformance view, and an exact crosswalk.
The draft contains 46 sequential requirements and 24 sequential falsifiable
predictions. Both rendered views import the same six shared modules and the
engineering view contains no independent normative science.

The first manager review identified three bounded defects and returned only
those defects to the same implementation-blind author:

1. reference-origin normalization had been described as unit-peak without a
   condition establishing that the convolved profile peaks at the origin;
2. convergence used an ordinary parameter difference even though ellipse
   orientation is modulo pi and can be unavailable at circularity; and
3. the scientific rationale did not yet meet the house-standard substantive
   narrative length.

The author corrected all three without receiving implementation, audit,
validation, production, or active ALIGN/AST information. The final `r0.1`
draft defines exact reference-origin normalization, uses declared
per-parameter convergence metrics with modulo-pi angle handling and separate
availability/candidate/support/valid-detector stability, and contains nine
substantive pages before its appendices.

## Firewall and traceability disposition

The exact three input artifacts retained their approved SHA-256 values during
authorship. Only the authorized Stage B output paths changed. The author
correctly recorded `SCI-BEAM-ODQ-001` because the packet named
`BEAM-SCOPE-D001--D012` but withheld the decision ledger. After the author
froze `r0.1`, the manager added the exact already-approved ID-to-disposition
mapping to [`CROSSWALK.md`](CROSSWALK.md). This closes the traceability-only
question without changing scientific substance or exposing new material to
the author.

## Questions retained for the scientific owner

The following seven questions remain open and are not draft blockers. They
control later effective policies or stronger interpretations, not the
algebraic completeness of `r0.1`:

- `SCI-BEAM-ODQ-002`: numerical support, convergence, QC, and S/N policies for
  a production profile;
- `SCI-BEAM-ODQ-003`: required consecutive stable-transition count;
- `SCI-BEAM-ODQ-004`: authorized singular-covariance retained-subspace or
  regularization procedures;
- `SCI-BEAM-ODQ-005`: approved model-inadequacy diagnostics and dispositions;
- `SCI-BEAM-ODQ-006`: source/amplitude convention combinations that permit a
  calibration candidate;
- `SCI-BEAM-ODQ-007`: response-completeness statement required for an
  intrinsic detector-plus-telescope beam interpretation; and
- `SCI-BEAM-ODQ-008`: whether and how a successor may authorize richer
  beam/background families.

No numerical default or scientific-owner disposition is inferred for these
questions.

## QA record

- Scientific rationale: 13 pages; Appendix A begins on page 10.
- Engineering conformance specification: 9 pages.
- Stable normative counts: 46 requirements and 24 predictions.
- Crosswalk: 70 exact ordered rows.
- Both PDFs contain every normative identifier and `v0.1/r0.1` identity.
- Both views import each of the six shared modules exactly once.
- Compilation completed without warnings or unresolved references.
- Poppler rendering and page-by-page visual inspection found no clipping,
  overlap, broken tables, unreadable glyphs, or header/footer defects.
- PDFs are letter-sized, unencrypted, and contain no forms or JavaScript.

These checks establish document integrity only. They do not establish
implementation conformance, representation or response fidelity,
observational performance, or production readiness.

## Next gate

Grant Wilson reviews the scientific rationale and the seven retained owner
questions. After any owner-directed revision is complete, a fresh
implementation-blind consistency reviewer checks the rationale, shared core,
engineering view, crosswalk, and decision records before any freeze decision.
