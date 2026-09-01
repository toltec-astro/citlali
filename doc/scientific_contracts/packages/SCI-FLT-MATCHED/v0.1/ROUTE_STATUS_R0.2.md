# SCI-FLT-MATCHED v0.1 r0.2 Route-Status Table

Status date: `2026-08-31`

| Route/fact | r0.2 status | Exact consequence |
| --- | --- | --- |
| Generic exact estimator | defined in draft | Coordinate-basis `n_p/d_p`, unit response, general-sky relation, local constrained-GLS theorem, and exclusions are mathematically closed for owner review. |
| Base output-anchor lattice | owner decided | One exact parent-pixel-center anchor with identical parent map structure; no interpolation. |
| Ordinary-MAP parent realization | known but not supplied | Boundary is drafted; no specific observation or coadd parent is admitted by this package draft. |
| Template realization | known but not supplied | Boundary is drafted; no specific template object, amplitude convention, or CAL/BEAM lineage is admitted. |
| `AO-001` weighting | unresolved decision | Observation and coadd weighting realizations are unavailable. |
| Numerical-conformance policy | unavailable | Exact science is decided, but no engineering numerical profile is preregistered and no route is assessed. |
| Fixed-state reference response definition | defined in draft | `R_fixed` and `R_reference` are defined generically; no selected weighting/template makes a realized response-qualified product available. |
| Full-procedure response | unavailable | No Learn--Resolve rerun authority or finite-difference profile is selected. |
| Response-qualified publication/use | unresolved decision | Response domain/query/consumer scope and exact representation remain open; RU verdict is unregistered. |
| Covariance-qualified route | unresolved decision | No `C_parent`, covariance scope, calibration cross-terms, or exact representation is selected. |
| NOI fixed-state route | boundary only | Exact parity boundary is drafted; no NOI population/companion or covariance equivalence is authorized. |
| FLT-to-FRUIT producer envelope | boundary only | Minimum one-way facts are defined; no FRUIT query vocabulary or FRUIT science is authorized. |
| SCI-VAL named-use profile | unavailable | Seven role semantics are decided, but no profile facts/policies/actions/versions/representation are registered or evaluated. |
| Implementation assessment | unassessed | No implementation, configuration, schema, tests, audits, products, reductions, or historical behavior were inspected. |
| Response/covariance fidelity | not assessed | No realized operator or product exists under this draft. |
| Observational validation/performance | not performed | No observational evidence, source-detection behavior, readiness, production, or Unity activity is claimed. |

No AO selection can change a parent/template fact from unavailable to available
or promote a boundary draft into a realized route.
