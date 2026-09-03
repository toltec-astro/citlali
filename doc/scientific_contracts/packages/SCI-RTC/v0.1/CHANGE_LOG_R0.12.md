# SCI-RTC v0.1/r0.12 change log

Date: 2026-08-21

Comparison baseline: approved r0.11 architecture at commit
`85e1e6c6865f74f1a97e99fab465714f43877c3d`.

- Added the exact owner correction directive and OWNER-097--103.
- Corrected native mapping/ALIGN terminology in the rationale and formal core.
- Made iterative learning, evaluation, and replay explicitly pair-based on the
  original admitted pair, with conditional output projection.
- Added distinct common-grid, pair-action/operator-support,
  coordinate-availability, and covariance-support authority.
- Propagated unavailable coordinate-specific affine correction through full
  downstream operator influence, including FIR support, IIR state to reset,
  and sampling support.
- Replaced remaining $x$-only composition/prefilter shorthand with paired
  operator language governed by $x$-domain science budgets.
- Enumerated OWNER-090--096 in the rationale, clarified raw-$r$ parentage and
  event evidence, and removed the forced break responsible for a sparse spill
  page.
- Added DEF-052, EQ-042, REQ-139--143, and PRED-104--108; synchronized both
  views, crosswalks, verifier, candidate records, and canonical PDFs.

No implementation, validation, performance, science-qualification, or
production claim is made. R0.12 remains a candidate pending owner freeze.
