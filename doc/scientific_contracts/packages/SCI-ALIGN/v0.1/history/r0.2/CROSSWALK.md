# SCI-ALIGN v0.1 - Stage B Targeted Rationale-to-Contract Crosswalk r0.2

Status: targeted author-draft traceability; no scientific approval,
implementation conformity, validation, freeze, readiness, or production claim

Prepared: `2026-08-22`

## Revision authority and view architecture

This targeted revision used only the SCI-ALIGN Stage B Targeted Revision
Directive (SHA-256
`18f1a7a458b0e19cd545481fce3606c1e10da0eedfc5f5d588329116fb14e103`)
and the existing Stage B SCI-ALIGN package items authorized by that directive.
No implementation, schema, test, audit, repair, validation, production
behavior, other package, web source, or raw thread history was inspected as
scientific authority.

The two PDFs now have intentionally different audience structures:

| View | Content rule |
| --- | --- |
| `src/scientific-rationale.tex` | Scientist-facing 9-section narrative, two explanatory figures, a small set of motivating equations, compact crosswalk, and owner-question summary; 7-9 pages required. |
| `src/engineering-conformance.tex` | Complete shared formal authority through all six `src/common/*.tex` modules, all 20 equations, 55 requirements, and 26 predictions. |

The complete formal modules are `src/common/notation.tex`,
`src/common/definitions.tex`, `src/common/assumptions.tex`,
`src/common/equations.tex`, `src/common/requirements.tex`, and
`src/common/edge_cases.tex`.

## Narrative to formal contract

| Rationale section | Canonical modules / equations | Stable requirements | Stable predictions |
| --- | --- | --- | --- |
| 1. What ALIGN establishes | notation; definitions purpose and ownership | `SCI-ALIGN-REQ-001`-`006`, `034`-`036` | `SCI-ALIGN-PRED-016`, `018`, `021` |
| 2. Native event time versus nominal slot | notation; `SCI-ALIGN-EQ-001`-`003` | `007`-`015` | `001`, `002`, `009` |
| 3. Tune/readout before paired `x/r` | definitions paired relation; `EQ-006`-`007` | `001`, `004`-`006` | `019`-`021` |
| 4. Field-specific mapping | field definitions; `EQ-004`-`005` | `016`-`019` | `002`-`006`, `025` |
| 5. Observing state versus correction | separated telescope families | `020`-`023`, `048` | `015`, `022`, `025` |
| 6. Origin, gaps, synthesis, exposure | gaps; `EQ-008`, `018`-`020` | `024`-`030`, `034`-`036` | `007`, `008`, `014`, `021` |
| 7. Scans and windows | five interval identities; `EQ-009`-`010` | `031`-`033`, `053` | `010`, `011`, `026` |
| 8. AST and RTC transfer | exact boundary and downstream ownership | `045`-`055` | `023`-`026` |
| 9. Response, covariance, questions | assumptions; `EQ-011`-`017`; owner register | `037`-`044` | `004`, `012`, `013`, `017`, `018`, `024` |

## Targeted semantic repair mapping

| Directive repair | Canonical locations | Stable IDs amended without renumbering |
| --- | --- | --- |
| Stable ALIGN slot `s`; local row `j`; RTC sample `n` | notation, grid/assignment/window/source/exposure equations, definitions transfer, shared boundary, provenance | `REQ-003`; `PRED-019`; equations `002`, `003`, `004`, `005`, `008`-`010`, `018`-`020` |
| Reserve `x/r` for raw KID coordinates; use `i_ref`, `t^ref`, `delta_(i->ref)` | notation, clock equation, rationale figures/prose, shared boundary | `REQ-008`-`010`; `PRED-001`; `EQ-001` |
| Remove generic non-KID `x` operands; use neutral `v` input/value family | notation, field equations, block operator, response/covariance equations, rationale | equations `004`, `005`, `007`, `011`, `013`, `014`; requirement/prediction IDs unchanged |
| Rename product-state `Q` to `mathcal S_ALIGN` | notation, local-state equation, author decision record | equation `019`; no requirement or prediction ID changed |
| Restore Figure 2 origin/method axes | rationale Figure 2 and QA report | origin is original/synthesized/unavailable; interpolation classes remain methods; formal IDs unchanged |
| Circular difference exactly `[-P/2,P/2)`; antipodal unavailable absent unwrap authority | definitions, circular equation, field table, shared boundary | `REQ-018`; `PRED-005`; `EQ-005` |
| Nonpolarimetric scope without calling raw `x` Stokes I | assumptions, telescope-family and scope prose | `REQ-023`, `051`; prediction fail-closed prose |
| Scientist-facing narrative separated from complete formal view | wrappers and verifier | IDs unchanged; Engineering view remains complete |

## Shared ALIGN-to-AST authority

The locked boundary is `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`, scientific owner
Grant Wilson, with the preserved compatibility/supersession rule. Its final
SHA-256 is
`359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf`.
The boundary body contains no self-hash. The coordinator reported byte-for-byte
installation in the AST packet: `cmp` passed and both packet copies have this
same digest.

## Stable identity completeness

- Requirements remain exactly `SCI-ALIGN-REQ-001` through
  `SCI-ALIGN-REQ-055`.
- Predictions remain exactly `SCI-ALIGN-PRED-001` through
  `SCI-ALIGN-PRED-026`.
- Equations remain exactly `SCI-ALIGN-EQ-001` through `020`.
- Assumptions remain exactly `SCI-ALIGN-ASM-001` through `012`.
- Owner decisions and open-question identities remain unchanged.

All open/deferred claims are enumerated in `OWNER_DECISION_REGISTER.md` and
`AVAILABILITY_REGISTER.md`. This revision records candidate scientific
semantics only. Implementation conformity, observational validation,
scientific approval, freeze, readiness, and production authorization remain
unassessed.
