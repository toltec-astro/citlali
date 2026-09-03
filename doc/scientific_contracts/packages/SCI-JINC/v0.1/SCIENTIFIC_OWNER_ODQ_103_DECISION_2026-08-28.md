# SCI-JINC-ODQ-103 — Scientific-Owner Disposition

Status: owner approved; bounded Stage A disposition

Scientific owner: Grant Wilson

Decision date: `2026-08-28`

Exact owner-directive attachment SHA-256:
`39bd1cb9d0bf45ecf5a6c336dc3da04f5e978cc0228e4ba73e5f01926fd2c75e`

## Approved Scientific Disposition

AST owns the coordinate realization associated with each detector sample,
including frame, units, coordinate validity/support facts, and the identity of
the parent detector sample. JINC consumes the AST coordinate corresponding to
the same processed sample realization that enters the JINC estimator. That
association remains exact across alignment, filtering, decimation, or any
other change of sample realization.

Row order, nearest-time or tolerance matching, detector ordering, numerical
coordinate equality, and other inferred fallbacks are prohibited. Missing,
duplicate, or ambiguous sample-coordinate association makes the coordinate
unavailable for that JINC contribution. The exact association is scientific
authority because a different coordinate changes the estimator; the key,
index, table join, object relationship, or other data-model mechanism used to
realize it is engineering choice, not scientific authority.

AST supplies coordinate facts. JINC owns their use relative to destination
pixels: local offset geometry, radial coordinate, dimensionless radius, finite
JINC support, signed coefficient, and admission for JINC map contribution.
AST does not decide JINC support, calculate or authorize a JINC coefficient,
manufacture a general JINC-valid flag, or encode JINC kernel semantics.

The single canonical JINC-owned admission profile for this scientific use is
`SCI-JINC:jinc_map_contribution@1`. It may consume shared upstream facts and
policy components, but it does not inherit an ordinary SCI-MAP admission
result, validity mask, or producer-owned JINC-usability conclusion. Additional
profiles require a genuinely distinct scientific use.

Sample admission and sample-pixel support are separate decisions. An admitted
sample may contribute to some destination pixels and not others. Outside
finite support and a contract-defined zero coefficient are ordinary
no-contribution results, not upstream invalidity or defect causes. An
unavailable or ambiguous AST coordinate prevents geometry evaluation and is
not equivalent to outside support. A negative JINC coefficient is normal and
is not an admission failure, invalidity, or defect.

Every JINC-owned accumulator derived from a contribution uses the same
admitted sample-pixel pair and the same signed-coefficient identity. Different
accumulators may apply different contract-defined algebraic functions of that
coefficient, but may not use inconsistent admission, coordinate, or
coefficient realizations.

Producer-owned facts and causes cross the boundary; a producer-owned JINC-
usability decision does not. JINC may add local causes for genuine JINC
failures, including missing/duplicate/ambiguous association, unavailable AST
coordinate, unavailable authorized parameters, non-finite local geometry,
inadmissible coefficient evaluation, or another explicit JINC precondition.
It creates no cause merely for a negative coefficient, outside-support pair,
contract-defined zero, or disagreement with ordinary MAP. Existing contract
cause/support mechanisms and appropriate product granularity are used; no new
per-contribution provenance system is authorized.

## Stage Consequence

`SCI-JINC-ODQ-103` is closed for the AST coordinate boundary, exact scientific
sample association, JINC-owned map-contribution admission, support/admission
separation, accumulator coupling, and cause policy. This decision changes
sanitized Stage A author-control bytes and remains subject to renewed exact-
byte approval under `SCI-JINC-STAGE-A-Q002`. It does not launch Stage B,
specify a data-model join architecture, alter implementation, import ordinary
MAP validity, or authorize per-contribution provenance machinery.
