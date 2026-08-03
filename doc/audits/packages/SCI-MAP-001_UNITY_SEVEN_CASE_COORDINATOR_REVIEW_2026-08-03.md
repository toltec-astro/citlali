# SCI-MAP-001 Unity seven-case coordinator review — 2026-08-03

Status: accepted as bounded map/product evidence; no finding, dependency, or
package status is closed

Package: `SCI-MAP-001`

Evidence request: `SCI-MAP-001-UNITY-001`

Reviewed handback: seven completed Point/Science cases from the exact repaired
candidate `ed28dafb37f9113c0d3c95297148157129a90886`

## Accepted evidence boundary

The returned seven-case bundle is accepted only for the following
map/product-scope statements:

- every case used the same frozen Unity executable, SHA-256
  `693c14898faa1d41a854030b86cdde2729bf121442eb8427feffb4d4e57686c5`,
  reported as `v4.0.0-3628-ged28dafb`;
- the fixed Point/Science observations, arrays, runtime policies, and saved
  configuration identities match the supplied case matrix;
- all returned launcher and Citlali logs are clean of error, critical, and
  `config.invalid` records;
- Point sequential/OpenMP products are bitwise identical; the two Science
  sequential/OpenMP comparisons have matching finite masks and only the
  reported round-off-scale differences;
- the shared C/E/X image extensions are bitwise identical where their product
  forms overlap; and
- `MAP-UNITY-PR1` correctly explains S-X's coadd-only persisted ensemble and
  its compact per-observation empirical planes.

The minimal returned transfer-manifest SHA-256 is
`d56aa24133352d0b33d3db2986b77c7d4c20b3b4ecb8d22a21ef8e15c8fdf066`.

## JINC consequence

All seven reductions selected `mapmaking.method: naive`. The presence of
`mapmaking.jinc_filter` configuration parameters does not select JINC
mapmaking and does not yield a JINC map or a naive/JINC comparison. This
evidence therefore supplies no implementation, response, normalization,
parameter-selection, edge, parallel, or product-conformity conclusion about
the JINC mapmaker.

SCI-MAP-002 must treat the returned evidence as a naive-only control and must
derive/audit the JINC operator independently. In particular, it must not reuse
the ordinary-naive positive-coefficient contribution predicate for JINC
without a JINC-specific derivation and explicit disposition.

## Unchanged limits and next path

The review does not establish estimator correctness, calibrated uncertainty or
significance semantics, primitive-level reconstruction, effective-rate
authority, F010/F011 closure, full MAP conformance, or production eligibility.
`CAP-SCIENCE` remains on Unity as retained contingency evidence and is neither
required for this review nor authorized for wholesale local transfer.

MAP remains `implementation_status: nonconformant`,
`validation_status: in_progress`, and `production_status: existing_use_only`.
No application change, repair integration, re-audit, Unity action, or product
policy beyond `MAP-UNITY-PR1` is authorized by this review.
