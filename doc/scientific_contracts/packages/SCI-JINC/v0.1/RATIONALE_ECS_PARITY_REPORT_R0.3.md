# SCI-JINC v0.1 Rationale/ECS Parity Report r0.3

Date: `2026-08-29`

Status: implementation-blind Stage B author-draft mechanical report; not
scientific-owner acceptance, implementation conformity, validation, freeze,
readiness, or production authorization

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
`f26803a2ff5edee4f6de4d5cdfcd6b5314622d68ca67cc0faa6caae1fb781348`.
The individual source hashes are:

| Shared module | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `1881cd9f5c77997eb70d22525653013ae53beea51f53679a5e8668baaffb3751` |
| `src/common/definitions.tex` | `aa1fc097b7821b6d98ed0f8cc81968061259a60ec2ca5bb50a8ff0b45c884bdc` |
| `src/common/equations.tex` | `d618e578c331da801c8a73ea2e22b932ab70fabddd2a4df26e0cd01a952b2c33` |
| `src/common/assumptions.tex` | `5a8c8e9e17d93954ba9c3e6ffce828e2ca312609948c5636bc45d7cd1c373c25` |
| `src/common/requirements.tex` | `4b36a33b9f3ef26204c7ca5abdd5b6c713c504e8f25db2e1331685d86c41da91` |
| `src/common/edge_cases.tex` | `a1b0cfca3e61e60d82a8a5b2cc8d34fefd018213f425f2add6a62cfa0dc74be8` |

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
| Equation 24 | PASS: `kappa-time` remains Equation 24 and its formula is unchanged; the Disposition-A even-lattice center consequence is unnumbered |
| Labels and references | PASS: no undefined, multiply defined, or duplicate-label diagnostic in either final compile log |
| Layout diagnostics | PASS: no overfull box in either final compile log |
| Revision identity | PASS: both source views and PDF metadata report `v0.1/r0.3` |
| Phase decision | PASS: one owner-approved Disposition-A state and paired positive-axis center-tie state in notation, definitions, equations, assumptions, requirements, and `SCI-JINC-PRED-017` |
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
| `src/scientific-rationale.tex` | `280d2553224d7ab39d06b1b8aaefe5c662811e0f899640358e4229e3adb4271b` |
| `src/engineering-conformance.tex` | `9462d4048970c691c6bfa5a4ce2c78a7040bf3a5cab39feba4577703d5e99234` |
| `CROSSWALK.md` | `18cb475388859fa5dafe7bed3d7c13aa5894fee5da14afb20e89ba539acd6ae4` |
| `pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `b9b561ced6ce7a7e2cc5fe997937fcdf15563249c2f47d19590c41e27f45a0a3` |
| `pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `12d3b3ae8265c86a4f2781302f7fdb6b0a353811db7020706fd837e257cca925` |

These results establish internal source/view parity only. They do not assess
an implementation candidate, representation, numerical realization,
response/covariance fidelity, observation, performance, readiness, or
production state.
