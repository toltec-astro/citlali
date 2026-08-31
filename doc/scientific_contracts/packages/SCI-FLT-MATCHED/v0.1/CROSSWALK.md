# SCI-FLT-MATCHED v0.1 Stage B Crosswalk

Status: Stage B draft; all `AO-001` through `AO-006` alternatives are
unselected. This crosswalk makes no implementation, conformity, validation,
achieved-performance, readiness, production, scientific-freeze, or Unity
claim.

The two PDFs import the same six files under `src/common/`. Consequently every
normative definition, equation, assumption, requirement, prediction, option,
edge case, consequence, failure rule, and validation consequence below is
byte-identical at source in both views. Scientist-view narrative outside the
shared core is explanatory. Engineering conformance material outside the
shared core is normative for the structure and failure semantics of any future
conformity claim, but it is not scientific authority, implementation evidence,
or a reported conformity or validation result.

## Input key

| Key | Exact admitted object |
| --- | --- |
| MAN | `AUTHOR_PACKET_MANIFEST.md` |
| SB | `SCOPE_BRIEF.md` |
| ASC | `AUTHOR_SUPERSESSION_COVER.md` |
| ACO | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` |
| SOD | `SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md` |
| OST | `AUTHOR_OPERATOR_STATE_AND_PRODUCT_TAXONOMY.md` |
| AB | `AUTHOR_BOUNDARIES.md` |
| RAO | `REQUIRED_AUTHORED_OPTION_SETS.md` |

Links appearing in admitted objects were not opened and are not inputs.

## Stable requirement crosswalk

Both view columns refer to the imported shared-core section; the engineering
test family adds future evidence routing without reporting a result.

| Requirement | Packet basis | Shared-core locus | Scientist view | Engineering view/test family |
| --- | --- | --- | --- | --- |
| `SCI-FLT-MATCHED-REQ-001` | SB assignment; SOD ODQ-001, package identity; ASC supersession | Definitions; Requirements | Shared core, Method and estimand | Shared core; CT-001, CT-004 |
| `SCI-FLT-MATCHED-REQ-002` | SB assignment; SOD ODQ-001/002 | Definitions; Requirements | Scientific question; shared core | Shared core; CT-003 |
| `SCI-FLT-MATCHED-REQ-003` | SOD ODQ-003; AB MAP parent | Parent classes; Requirements | Frozen state/parent rationale; shared core | Shared core; CT-002 |
| `SCI-FLT-MATCHED-REQ-004` | SB fixed boundaries; SOD ODQ-003/010 | Parent classes; Frozen state; Requirements | Frozen state rationale; shared core | Shared core; CT-002 |
| `SCI-FLT-MATCHED-REQ-005` | SB estimator/template; SOD ODQ-005/008; OST taxonomy | Template-response product; Requirements | Scientific question; shared core | Shared core; CT-003, CT-008 |
| `SCI-FLT-MATCHED-REQ-006` | SOD ODQ-005; ASC deferred families | Template-response product; Requirements | Scientific question; shared core | Shared core; CT-003, CT-014 |
| `SCI-FLT-MATCHED-REQ-007` | SB normative estimator; RAO AO-001 | Notation; Assumptions; Requirements | Shared core | Shared core; CT-001, CT-003, CT-005 |
| `SCI-FLT-MATCHED-REQ-008` | SB normative estimator; SOD ODQ-006; OST reference operator | Equations; Requirements | Normalization rationale; shared core | Shared core; CT-004, CT-007 |
| `SCI-FLT-MATCHED-REQ-009` | SOD ODQ-006; OST reference operator | Equations; Validity table; Requirements | Normalization rationale; shared core | Shared core; CT-004, CT-005 |
| `SCI-FLT-MATCHED-REQ-010` | SB fixed boundaries; SOD ODQ-007 | Influence support; Validity; Requirements | Edge rationale; shared core | Shared core; CT-005, CT-014 |
| `SCI-FLT-MATCHED-REQ-011` | SOD ODQ-008; OST response; RAO AO-005 | Fixed-state response; Requirements | Response rationale; shared core | Shared core; CT-008 |
| `SCI-FLT-MATCHED-REQ-012` | SB estimator; SOD ODQ-004/009; RAO AO-001 | Optimality criterion; Requirements | Normalization rationale; shared core | Shared core; CT-004, CT-006 |
| `SCI-FLT-MATCHED-REQ-013` | SOD ODQ-004; RAO AO-001 | Weighting; AO-001; Requirements | Owner-disposition guide; shared core | Shared core; CT-001, CT-006 |
| `SCI-FLT-MATCHED-REQ-014` | ACO identities; AB MAP parent; RAO AO-001 | Weighting; AO-001; Requirements | Shared core | Shared core; CT-006 |
| `SCI-FLT-MATCHED-REQ-015` | SB fixed boundaries; SOD ODQ-010; OST state generations | Frozen state; Requirements | Frozen-state rationale; shared core | Shared core; CT-009 |
| `SCI-FLT-MATCHED-REQ-016` | SOD ODQ-003/010 | Parent classes; Frozen state; Requirements | Frozen-state rationale; shared core | Shared core; CT-002, CT-009 |
| `SCI-FLT-MATCHED-REQ-017` | SOD ODQ-010; ACO NOI ownership; AB NOI | Frozen state; Requirements | Frozen-state rationale; shared core | Shared core; CT-009 |
| `SCI-FLT-MATCHED-REQ-018` | SOD ODQ-010; OST state generations; AB NOI | Frozen state; Requirements | Frozen-state rationale; shared core | Shared core; CT-009 |
| `SCI-FLT-MATCHED-REQ-019` | SB fixed boundaries; SOD ODQ-011 | Requirements; Edge cases | Owner-disposition guide; shared core | Shared core; CT-006, CT-012, CT-014 |
| `SCI-FLT-MATCHED-REQ-020` | SOD ODQ-009; AB NOI | Conditional covariance; Requirements | Uncertainty rationale; shared core | Shared core; CT-010 |
| `SCI-FLT-MATCHED-REQ-021` | SB estimator; SOD ODQ-009 | Optimality/covariance; Requirements | Normalization rationale; shared core | Shared core; CT-006, CT-010 |
| `SCI-FLT-MATCHED-REQ-022` | SB fixed boundaries; SOD ODQ-009; ACO identities | Uncertainty budget; Requirements | Uncertainty rationale; shared core | Shared core; CT-010 |
| `SCI-FLT-MATCHED-REQ-023` | SB product/lifecycle; SOD ODQ-012/013; OST product roles | Product roles; Requirements | Product rationale; shared core | Shared core; CT-011 |
| `SCI-FLT-MATCHED-REQ-024` | SB product/lifecycle; SOD ODQ-013; OST product roles | Product roles; Requirements | Product rationale; shared core | Shared core; CT-011 |
| `SCI-FLT-MATCHED-REQ-025` | SB product/lifecycle; SOD ODQ-009/013; OST product roles | Product roles; Requirements | Product rationale; shared core | Shared core; CT-010, CT-011 |
| `SCI-FLT-MATCHED-REQ-026` | SB lifecycle; SOD ODQ-013 | Lifecycle states; Requirements | Product rationale; shared core | Shared core; CT-011, CT-012 |
| `SCI-FLT-MATCHED-REQ-027` | OST product roles | Product roles; Requirements | Shared core | Shared core; CT-011, CT-014 |
| `SCI-FLT-MATCHED-REQ-028` | SB lifecycle; SOD ODQ-012; AB FLT to FRUIT | FLT to FRUIT; AO-005; Requirements | Product rationale; owner guide; shared core | Shared core; CT-008, CT-013 |
| `SCI-FLT-MATCHED-REQ-029` | SOD ODQ-012; ACO FRUIT ownership; AB FLT to FRUIT | FLT to FRUIT; Requirements | Product rationale; shared core | Shared core; CT-013, CT-014 |
| `SCI-FLT-MATCHED-REQ-030` | SB assignment/exclusions; SOD ODQ-002; AB downstream interpretation | Method identity; Requirements; Exclusions | Scientific question; shared core | Shared core; CT-014 |
| `SCI-FLT-MATCHED-REQ-031` | SOD ODQ-001/011; ASC supersession/deferred families | Method identity; Requirements; Exclusions | Scientific question; shared core | Shared core; CT-014 |
| `SCI-FLT-MATCHED-REQ-032` | SB required options; MAN assignment/nonclaims; RAO global rule | Requirements; Option closure | Owner-disposition guide; shared core | Shared core; CT-001, CT-012, CT-014 |
| `SCI-FLT-MATCHED-REQ-033` | SOD ODQ-006; RAO AO-002 | Approximation metrics; AO-002; Requirements | Shared core | Shared core; CT-005, CT-007 |
| `SCI-FLT-MATCHED-REQ-034` | MAN release/nonclaims; SOD closing nonclaims | Authority layers; Requirements; Nonclaim closure | Governance rationale; shared core | Claim discipline; shared core; CT-001, CT-015 |
| `SCI-FLT-MATCHED-REQ-035` | SOD ODQ-013; ACO VAL ownership; AB VAL | Authority layers; AO-006; Requirements | Product/governance rationale; shared core | Shared core; CT-012 |
| `SCI-FLT-MATCHED-REQ-036` | SOD ODQ-008/009; ACO CAL/BEAM ownership; AB CAL/BEAM | Units; Uncertainty budget; Requirements | Scientific question; response/uncertainty rationale; shared core | Shared core; CT-003, CT-008, CT-010 |
| `SCI-FLT-MATCHED-REQ-037` | SOD ODQ-003; ACO MAP ownership; AB MAP parent | Parent classes; Product roles; Requirements | Frozen-state rationale; shared core | Shared core; CT-002 |
| `SCI-FLT-MATCHED-REQ-038` | ACO identities; OST method identity/product roles; SOD ODQ-012/013 | Method identity; Product roles; Requirements | Product rationale; shared core | Claim record; shared core; CT-001, CT-009, CT-011 |
| `SCI-FLT-MATCHED-REQ-039` | MAN release/nonclaims; SOD closing nonclaims | Requirements; Nonclaim closure | Status boxes; shared core | Status boxes; shared core; CT-001, CT-015 |

## Falsifiable prediction crosswalk

| Prediction | Scientific basis | Required evidence pattern | Engineering route |
| --- | --- | --- | --- |
| `SCI-FLT-MATCHED-PRED-001` | Exact `N/D`; OST unit self-response | Unit matching-template response over validity cover | CT-004, CT-008 |
| `SCI-FLT-MATCHED-PRED-002` | Linearity of fixed `L`; supplied-template amplitude | Noiseless amplitude scaling over state/location strata | CT-003, CT-004 |
| `SCI-FLT-MATCHED-PRED-003` | SB unbiasedness; zero conditional mean | Independent repeated fixed-state ensemble plus analytic expectation | CT-003, CT-004 |
| `SCI-FLT-MATCHED-PRED-004` | SOD ODQ-007 complete support | Missing-support injections produce no numerical amplitude | CT-005, CT-014 |
| `SCI-FLT-MATCHED-PRED-005` | SOD ODQ-006 invalid/nonpositive normalization | Each invalid-`D` class yields typed unavailable/failed and no zero | CT-004, CT-005, CT-014 |
| `SCI-FLT-MATCHED-PRED-006` | SOD ODQ-003/010 parent/state separation | Observation/coadd identity and state-generation comparison | CT-002, CT-009 |
| `SCI-FLT-MATCHED-PRED-007` | SOD ODQ-010; AB NOI parity | State/operator/support/normalization/failure identity for every compatible NOI member | CT-009 |
| `SCI-FLT-MATCHED-PRED-008` | SOD ODQ-009 covariance propagation | Empirical fixed-state covariance versus exact propagation with sampling uncertainty | CT-010 |
| `SCI-FLT-MATCHED-PRED-009` | SOD ODQ-009 `D^-1` restriction | Label/consumer audit under non-GLS weight with invariant amplitude | CT-006, CT-010 |
| `SCI-FLT-MATCHED-PRED-010` | SOD ODQ-006; RAO AO-002 | Full selected-envelope metrics, coverage, and support/null equality | CT-005, CT-007 |
| `SCI-FLT-MATCHED-PRED-011` | ACO immutable identities; OST realization tuple | Mutation/successor attempts preserve old identity and create new generation | CT-007, CT-009, CT-011 |
| `SCI-FLT-MATCHED-PRED-012` | SOD ODQ-013 atomic signal bundle | Failure injection into every required member | CT-011 |
| `SCI-FLT-MATCHED-PRED-013` | SB/SOD optional companion rule | Optional companion failure under each named-use policy | CT-010, CT-011 |
| `SCI-FLT-MATCHED-PRED-014` | SOD ODQ-005/008 amplitude scaling/units | Template rescaling law and modeled-signal invariance | CT-003 |
| `SCI-FLT-MATCHED-PRED-015` | OST full response; SOD ODQ-008 | Noninvariance probe rejects insufficient single-kernel reduction | CT-008 |
| `SCI-FLT-MATCHED-PRED-016` | SOD ODQ-011 no selector/fallback | Request every unavailable route and observe exact rejection | CT-012, CT-014 |
| `SCI-FLT-MATCHED-PRED-017` | SOD ODQ-012; AB FLT to FRUIT | Complete documented handoff-query suite without hidden state | CT-008, CT-013 |
| `SCI-FLT-MATCHED-PRED-018` | SB/SOD/AB exclusions and nonclaims | Public-field, profile, and consumer interpretation audit | CT-013, CT-014, CT-015 |

## Assumption and uncertainty crosswalk

| Shared IDs | Packet basis | Audit closure |
| --- | --- | --- |
| `ASM-001`--`ASM-002` | SOD ODQ-003; ACO/AB MAP ownership | CT-001--CT-003, CT-005 |
| `ASM-003`--`ASM-004` | SOD ODQ-005/007; OST template/response | CT-003--CT-005 |
| `ASM-005`--`ASM-006` | SOD ODQ-004/006; RAO AO-001 | CT-004--CT-007 |
| `ASM-007`--`ASM-009` | SB unbiasedness; SOD ODQ-009 | CT-004, CT-010 |
| `ASM-010`--`ASM-011` | SOD ODQ-010; AB NOI | CT-009 |
| `ASM-012`--`ASM-013` | RAO AO-002/003/005 | CT-007, CT-008, CT-010 |
| `ASM-014`--`ASM-015` | SOD ODQ-008/011; AB CAL/BEAM | CT-003, CT-006, CT-012, CT-014 |
| `U1`--`U2` | SOD ODQ-008/009; AB CAL/BEAM | CT-010 |
| `U3`--`U4` | SOD ODQ-006/010; RAO AO-002 | CT-007, CT-009, CT-010 |
| `U5`--`U6` | RAO AO-003; AB NOI | CT-009, CT-010 |
| `U7` | SOD ODQ-005/008; excluded interpretation | CT-003, CT-014 |

## Authored-option alternative crosswalk

Every row below is unselected. The alternatives, assumptions, consequences,
observables/bounds, failure rules, and validation consequences are fully stated
in `src/common/requirements.tex`, imported by both views.

| Alternative | Packet assignment | Scientific consequence | Principal validation route | Route unavailable until selected/parameterized |
| --- | --- | --- | --- | --- |
| `SCI-FLT-MATCHED-AO-001-A` | SOD ODQ-004/009; RAO AO-001 | Exact inverse-covariance GLS; only route eligible for GLS optimality and `D^-1` variance | CT-006, CT-010 | Exact-GLS weight and associated optimality/variance claim |
| `SCI-FLT-MATCHED-AO-001-B` | SOD ODQ-004; RAO AO-001 | Structured covariance-derived approximation; normalization retained, optimality withheld unless exact | CT-006, CT-007, CT-010 | Structured weighting realization |
| `SCI-FLT-MATCHED-AO-001-C` | SOD ODQ-004; ASC historical-candidate limit; RAO AO-001 | Radially symmetrized average map-noise PSD as weaker spectral candidate absent separate covariance authority | CT-006, CT-007 | PSD weighting, conventions, and all associated bounds |
| `SCI-FLT-MATCHED-AO-001-D` | SOD ODQ-004; RAO AO-001 | Declared positive-semidefinite weaker weighting with no covariance/optimality claim | CT-006, CT-007 | Weaker weight realization |
| `SCI-FLT-MATCHED-AO-002-A` | SOD ODQ-006; RAO AO-002 | Exact numerical-identity envelope; no scientific approximation | CT-007 | Exact-identity conformity route |
| `SCI-FLT-MATCHED-AO-002-B` | SOD ODQ-006; RAO AO-002 | Strict uniform `10^-3` operator/response bounds, `2x10^-3` covariance bound, exact support/null | CT-007 | Strict approximate realization claim |
| `SCI-FLT-MATCHED-AO-002-C` | SOD ODQ-006; RAO AO-002 | Named-use core/tail ceilings with exact support/null and bounded covariance projections | CT-007, CT-012 | Profile-specific approximate realization claim |
| `SCI-FLT-MATCHED-AO-003-A` | SOD ODQ-009; RAO AO-003 | Full exact explicit conditional covariance | CT-010, CT-011 | Explicit covariance product/use |
| `SCI-FLT-MATCHED-AO-003-B` | SOD ODQ-009; RAO AO-003 | Exact structured conditional covariance with declared query vocabulary | CT-010, CT-011 | Structured covariance product/use |
| `SCI-FLT-MATCHED-AO-003-C` | SOD ODQ-009; RAO AO-003 | Authorized projected covariance; all omitted correlations remain unknown | CT-010, CT-012 | Named projected consumer |
| `SCI-FLT-MATCHED-AO-003-D` | SOD ODQ-009; RAO AO-003 | Exact lineage-resolvable covariance on demand | CT-010, CT-011 | On-demand covariance query/use |
| `SCI-FLT-MATCHED-AO-003-E` | SOD ODQ-009; RAO AO-003 | Typed covariance unavailability while signal may remain complete | CT-010, CT-014 | Every covariance-dependent use |
| `SCI-FLT-MATCHED-AO-004-A` | SOD ODQ-010/013; RAO AO-004 | Full frozen-state materialization | CT-009, CT-011 | Full-state audit/reanalysis route |
| `SCI-FLT-MATCHED-AO-004-B` | SOD ODQ-010/013; RAO AO-004 | Structured compact state with exact declared queries | CT-009, CT-011 | Compact-state audit/reanalysis route |
| `SCI-FLT-MATCHED-AO-004-C` | SOD ODQ-010/013; RAO AO-004 | Exact lineage reconstruction of learn/declare-once state | CT-009, CT-011 | Lineage-only state audit/reanalysis route |
| `SCI-FLT-MATCHED-AO-005-A` | SOD ODQ-012; RAO AO-005 | Full response materialization | CT-008, CT-013 | Full response and response-dependent handoff |
| `SCI-FLT-MATCHED-AO-005-B` | SOD ODQ-012; RAO AO-005 | Exact structured response with documented query contract | CT-008, CT-013 | Structured response queries and handoff |
| `SCI-FLT-MATCHED-AO-005-C` | SOD ODQ-012; RAO AO-005 | Exact lineage-resolvable response on demand | CT-008, CT-009, CT-013 | Lineage response queries and handoff |
| `SCI-FLT-MATCHED-AO-006-A` | SOD ODQ-013; RAO AO-006 | Six role-separated immutable profiles | CT-012 | Six-profile VAL evaluation/publication policy |
| `SCI-FLT-MATCHED-AO-006-B` | SOD ODQ-013; RAO AO-006 | Three layers retaining six mandatory subverdicts | CT-012 | Layered VAL evaluation/publication policy |
| `SCI-FLT-MATCHED-AO-006-C` | SOD ODQ-013; RAO AO-006 | One composite with six addressable verdicts; scalar action forbidden | CT-012 | Composite-vector VAL evaluation/publication policy |

## Product, lifecycle, and claim-layer closure

| Concern | Normative locus | Scientist explanation | Engineering evidence |
| --- | --- | --- | --- |
| Atomic signal bundle | Definitions: Product roles; REQ-023/024 | Product and governance rationale | CT-011 |
| Qualified companions | Definitions: Product roles; REQ-025 | Response and uncertainty; product rationale | CT-010/011 |
| Five lifecycle states | Definitions: Lifecycle; REQ-026 | Product rationale | CT-011/012 |
| FLT to FRUIT interface | Definitions; AO-005; REQ-028/029 | Product rationale; owner guide | CT-008/013 |
| No source or FRUIT science | Definitions; Edge cases; REQ-029--031 | Scientific question; product rationale | CT-013/014 |
| Scientific authority vs conformity vs validation vs performance vs readiness vs production | Definitions: Authority layers; REQ-034/039 | Governance rationale | CT-001/015 |

## Source/PDF consistency obligation

`build/verify_consistency.py` checks that both LaTeX views import all six shared
modules in the same order, that the shared source contains the complete stable
ID sets, that both rendered PDFs contain those IDs and all 21 exact option
identities, and that draft/nonclaim language survives rendering. Its report is
a build-consistency artifact only, not scientific validation or implementation
conformity evidence.
