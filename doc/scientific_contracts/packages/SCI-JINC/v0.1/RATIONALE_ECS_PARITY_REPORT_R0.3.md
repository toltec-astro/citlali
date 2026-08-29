# SCI-JINC v0.1 Rationale/ECS Parity Report r0.3

Date: `2026-08-29`

Status: final implementation-blind freeze-parity evidence; not implementation
conformity, representation fidelity, validation, achieved performance,
readiness, production, or production authorization

## Shared Scientific Authority

Both `src/scientific-rationale.tex` and
`src/engineering-conformance.tex` include the following six files exactly
once and in this identical order:

1. `src/common/notation.tex`
2. `src/common/definitions.tex`
3. `src/common/equations.tex`
4. `src/common/assumptions.tex`
5. `src/common/requirements.tex`
6. `src/common/edge_cases.tex`

The SHA-256 of the byte concatenation in that order is
`ca6650743af30e34940b7360a92c66f6638e993e07648b329e05f107b3b9e657`.
The individual source hashes are:

| Shared module | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `fed76501dfc57540a4f383bf329d35118508f1e96a143b1cde8e09078d6dade1` |
| `src/common/definitions.tex` | `9e6f3ff015c753c879ff03be65fe981ac4f6ad2fc572d4a2f13f8a5240a12e1e` |
| `src/common/equations.tex` | `2b1870e92f9a0e6141fdda1a8865babae41208aafb73fc233f4befc0e1b665c1` |
| `src/common/assumptions.tex` | `15b811ab6ace92aa2d1713ae19b92454cb865e8862b82a599f94eca1003a1765` |
| `src/common/requirements.tex` | `207a85acb31a4f381b289781706c9f14058d330ff847e99023e9e5714c4d4dff` |
| `src/common/edge_cases.tex` | `815c70e925f103d989e4ec015a64d69ac0710c1a0c57789a4dfe754bdb81bd2d` |

The scientist-facing and engineering-facing prose introduce no competing
normative module. If their prose is narrower or less detailed, the shared
module controls.

## Mechanical Parity Results

| Check | Result |
| --- | --- |
| Requirement definitions | PASS: 44 occurrences, sequential `SCI-JINC-REQ-001`--`044`, no gap or duplicate |
| Prediction definitions | PASS: 36 occurrences, sequential `SCI-JINC-PRED-001`--`036`, no gap or duplicate |
| Crosswalk | PASS: 44 requirement rows |
| Assumptions | PASS: shared `SCI-JINC-ASM-001`--`009` in both views |
| Numbered equations | PASS: 29 shared labels in one sequence in both views |
| Equation 24 | PASS: `kappa-time` remains Equation 24 and its formula is unchanged; the `SCI-JINC-DEC-PHASE-CENTER-001` even-lattice center consequence is unnumbered |
| Labels and references | PASS: no undefined, multiply defined, or duplicate-label diagnostic in either final compile log |
| Layout diagnostics | PASS: no overfull box in either final compile log |
| Revision and owner identity | PASS: both source views and PDF metadata report `v0.1/r0.3`; both covers state `Scientific owner: Grant Wilson`, and PDF Author is `Grant Wilson` |
| Phase/center decision | PASS: `SCI-JINC-DEC-PHASE-CENTER-001` binds the separately approved phase-lattice and center-tie statements in notation, definitions, equations, assumptions, requirements, and predictions |
| Numerical claim boundary | PASS: a certificate pass is limited to finite-precision fidelity to the exact selected discrete oracle; other claim layers remain separate |
| Admission semantics | PASS: canonical human-readable term is JINC upstream-occurrence admission; immutable Registry identity `SCI-JINC:jinc_map_contribution@1` is retained |
| Source lifecycle | PASS: exact-source-lock model A is stated; ambient current-Registry substitution is explicitly prohibited |
| Terminology | PASS: superseded temporal-product and sample-admission wording is absent from r0.3 author surfaces |
| Stale identities | PASS: no stale predecessor-revision document/boundary label and no invented replacement Registry identity occurs in r0.3 author surfaces |

The phrase "current Registry" appears only inside explicit prohibitions; it is
not a moving source reference.

## View And PDF Bindings

| Artifact | SHA-256 |
| --- | --- |
| `src/scientific-rationale.tex` | `7cabea85eaa5ad9afbb0914c585d2fe7917806c9919964a465c0d9742fdb55e2` |
| `src/engineering-conformance.tex` | `a8cc9b66d22f1c4c0e9dc53c46724721f38fa2b2d267f74e7341b359874c19aa` |
| `CROSSWALK.md` | `df2bbb1f8eec53c91497d52b85591e66f86639f76c63686688367c96e309d2e5` |
| `pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `53ed941658ae1205950a8bc533d569cc85b246a40bb6e448fbbc6d7f0509a7b8` |
| `pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `6b78e80bb485815292972c5de60c444954d7bb62902799d6fa4c3f421766114a` |

These results establish internal source/view parity only. They do not assess
an implementation candidate, representation, numerical realization,
response/covariance fidelity, observation, performance, readiness, or
production state.
