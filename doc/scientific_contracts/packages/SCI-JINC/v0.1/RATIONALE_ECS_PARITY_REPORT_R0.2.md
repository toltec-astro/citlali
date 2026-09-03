# SCI-JINC v0.1 Rationale/ECS Parity Report r0.2

Date: 2026-08-29

Status: implementation-blind Stage B author-draft parity record

Scope: the r0.2 scientist-facing rationale, engineering conformance
specification (ECS), six shared canonical modules, crosswalk, and canonical
PDFs only

This report records source and presentation parity. It is not an
implementation-conformity, representation-fidelity, validation,
achieved-performance, numerical-readiness, production-readiness, or
production claim.

## Shared Scientific Authority

Both views include the same six modules exactly once and in this order:

1. `src/common/notation.tex`
2. `src/common/definitions.tex`
3. `src/common/equations.tex`
4. `src/common/assumptions.tex`
5. `src/common/requirements.tex`
6. `src/common/edge_cases.tex`

The shared-module source hashes at the audited snapshot are:

| Shared module | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `e1e71d382dc96bca4ff0e5a91914ff696375ca053fe2a19ca16f7c4548bc28cc` |
| `src/common/definitions.tex` | `38a0605081abb0eb1675868170ec58d4229692af295ac917d0628a7068b64ce2` |
| `src/common/equations.tex` | `89dca78be27612a294184fda3faf4bcdf5a53912df71f8bbc2e8c9ea333d0bd6` |
| `src/common/assumptions.tex` | `655771f4c942addbc7320eb917e1a1ef56ff2a6bab770b21b66fcffa1619fd18` |
| `src/common/requirements.tex` | `031ef555ccd227584270dbfe91a1430b868a8e52ccbb5b186eab976ccd8c4b4a` |
| `src/common/edge_cases.tex` | `b4ec8791c0ade201cab0394071e561d24714e9d262b61109d340a563387ebfd8` |

No view-local definition supersedes or forks that shared authority. The
scientist-facing view explains the rationale and boundaries; the ECS presents
the same authority as engineering obligations and testable predictions.

## Identifier And Crosswalk Audit

Mechanical inspection gave the following results:

- requirements are exactly `SCI-JINC-REQ-001` through
  `SCI-JINC-REQ-044`, with 44 unique sequential definitions and no gaps or
  duplicates;
- predictions are exactly `SCI-JINC-PRED-001` through
  `SCI-JINC-PRED-036`, with 36 unique sequential definitions and no gaps or
  duplicates;
- the canonical prediction-to-requirement trace covers every prediction
  exactly once;
- `CROSSWALK.md` contains exactly 44 three-column data rows, one for each
  requirement, with no extra prose row, gap, or duplicate;
- the scientist-facing source has 66 labels, 50 references, and 41 equation
  references; the ECS source has 59 labels, 44 references, and 38 equation
  references; neither source has a duplicate label or unresolved reference.

The stable r0.1 identifiers were preserved. Only
`SCI-JINC-REQ-043`--`044` and `SCI-JINC-PRED-033`--`036` were appended for
the r0.2 repair.

## Scientific-State Parity

The two views consistently state all of the following:

- the package is `SCI-JINC v0.1/r0.2`, dated 2026-08-29, and remains an
  implementation-blind Stage B author draft;
- the numerical bundle has exactly the five ODQ-107 roles
  `jinc_signal_numerator`, `jinc_signed_normalization`,
  `jinc_quadratic_accumulator`, `jinc_map`, and
  `jinc_coefficient_squared_time`;
- a compact bundle-level generative record is required for replay, but it is
  not a sixth numerical role, a dense per-contribution archive, or a broader
  provenance framework;
- response, covariance, formal-weight, availability, diagnostic, and
  generalized-provenance products remain outside the approved v0.1 output
  scope; their conditional mathematics does not create those products;
- exact PTC r0.3, AST r0.2, the retained
  `SCI-JINC:jinc_map_contribution@1` admission-profile identity, and paired
  SCI-VAL source/profile records close interfaces without supplying new
  scientific content;
- the exact discrete operator uses one-based integer FITS centers,
  componentwise `floor(x+1/2)` center selection, half-open phase bins,
  midpoint representatives, the exact logical cache key, the admitted square
  affine-WCS metric, full-square support enumeration, and point evaluation at
  destination centers;
- every positive integer phase subdivision is admitted, including the
  disclosed even-subdivision zero-phase representative on the positive side;
- the JINC root is the exact first positive root of `J_1`; a decimal
  rendering is nonnormative;
- algebraic support and certified numerical adequacy are separate states;
  the packet supplies no exact numerical-adequacy owner profile and matching
  certificate, so numerical and near-cancellation support remain typed
  unavailable;
- the TolTEC parameter sets, inherited 45 m denominator, inherited shape
  values, and mode-dependent `r_max` remain unauthorized and unavailable;
- neither view invents a PTC family, hidden default,
  response/covariance/exposure product, MAP inheritance, implementation
  behavior, or broader provenance/diagnostic framework.

The views make no implementation-conformity, representation-fidelity,
validation, achieved-performance, numerical-readiness, production-readiness,
or production claim. Those later assessment layers remain unassessed.

## Audited View And PDF Snapshot

| Artifact | SHA-256 | Pages |
| --- | --- | ---: |
| `src/scientific-rationale.tex` | `201ae390281682b99dc197002f75968551b8867d05d8ed581e017e3e3c928a15` | n/a |
| `src/engineering-conformance.tex` | `61acaf4cb4352026a377edba2869e2f18f76ff624e087261a51aadf940c472dc` | n/a |
| `CROSSWALK.md` | `1650db426de15cbc8429c11381b1bfcec2ac4dd79600ea71954707a2b8954782` | n/a |
| `pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `3deab8fffb2af93375a187a5ba0e177921398f44e88963ef2d7a1b3e441331dc` | 33 |
| `pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `15fba087df7bff0560aca65854ce74e3a8de037614623b877fd6c885f3a9032a` | 22 |

The PDF titles identify r0.2 and both PDFs use US Letter pages. Compilation,
rendering, and page-by-page visual inspection are recorded separately in
`PDF_VISUAL_QA_REPORT_R0.2.md`.

## Disposition

Rationale/ECS parity: **pass for the implementation-blind r0.2 Stage B author
draft snapshot recorded above**. This disposition says only that the two
draft views share and present one coherent scientific authority.
