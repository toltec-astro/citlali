# SCI-FLT-MATCHED v0.1 Stage B Crosswalk

Status: Stage B r0.5 final targeted type/lifecycle/covariance-role closure
draft; all weighting,
covariance-role/scope, representation,
named-use, and numerical routes remain unselected. This crosswalk makes no
implementation, conformity, response/covariance-fidelity, validation,
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
| R02 | `SCIENTIFIC_OWNER_R0.2_DIRECTIVE_2026-08-31.md` |
| R03 | `CHATGPT_PRO_INDEPENDENT_REVIEW_R0.2_2026-08-31.md` plus the owner's instruction to apply its directed repairs |
| R04 | `CHATGPT_PRO_INDEPENDENT_REVIEW_R0.3_2026-09-01.md` plus `SCIENTIFIC_OWNER_R0.4_DIRECTIVE_2026-09-01.md` |

| R05 | `SCIENTIFIC_OWNER_R0.5_DIRECTIVE_2026-09-01.md` plus its resulting closure amendments and owner-disposition packets |

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
| `SCI-FLT-MATCHED-REQ-020` | SOD ODQ-009; AB NOI; R04 C01--C03 | Conditional covariance; Requirements | Uncertainty rationale; shared core | Shared core; CT-010 |
| `SCI-FLT-MATCHED-REQ-021` | SB estimator; SOD ODQ-009 | Optimality/covariance; Requirements | Normalization rationale; shared core | Shared core; CT-006, CT-010 |
| `SCI-FLT-MATCHED-REQ-022` | SB fixed boundaries; SOD ODQ-009; ACO identities; R04 C01--C03 | Uncertainty budget; Requirements | Uncertainty rationale; shared core | Shared core; CT-010 |
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
| `SCI-FLT-MATCHED-REQ-040` | R02 anchor disposition | Parent classes and anchor lattice; Requirements | Scientific question; shared core | Shared core; CT-002, CT-003 |
| `SCI-FLT-MATCHED-REQ-041` | R02 local restriction/inversion | Weighting/local covariance; GLS theorem; Requirements | Normalization rationale; shared core | Shared core; CT-004, CT-006 |
| `SCI-FLT-MATCHED-REQ-042` | R02 support taxonomy | Five support roles; Requirements | Edge rationale; shared core | Shared core; CT-005, CT-009 |
| `SCI-FLT-MATCHED-REQ-043` | R02 general-sky amendment | General-sky relation; Requirements | Response rationale; shared core | Shared core; CT-004, CT-008 |
| `SCI-FLT-MATCHED-REQ-044` | R02 response-family amendment | Learn--Resolve--Apply; full-procedure response; Requirements | State/response rationale; shared core | Shared core; CT-008, CT-009 |
| `SCI-FLT-MATCHED-REQ-045` | R02 realized/reference separation; R04 C02/C08 | Realized/reference response and covariance; Requirements | Response rationale; shared core | Shared core; CT-007, CT-008, CT-010 |
| `SCI-FLT-MATCHED-REQ-046` | R02 AO-003 refactor; R04 C03 owner disposition | Covariance scope/representation; Requirements | Owner guide; shared core | Shared core; CT-010 |
| `SCI-FLT-MATCHED-REQ-047` | R02 AO-004 refactor | Immutable-state representation; Requirements | State rationale; shared core | Shared core; CT-009 |
| `SCI-FLT-MATCHED-REQ-048` | R02 AO-005/boundary directive | Response science and exact representation; Requirements | Product rationale; owner guide; shared core | Shared core; CT-001, CT-008, CT-013 |
| `SCI-FLT-MATCHED-REQ-049` | R02 AO-006/SCI-VAL directive | SCI-VAL evaluation; AO-006; Requirements | Governance rationale; shared core | Shared core; CT-012 |
| `SCI-FLT-MATCHED-REQ-050` | R02 singular-mode taxonomy | GLS theorem; Edge cases; Requirements | Normalization rationale; shared core | Shared core; CT-004--CT-006 |

## Falsifiable prediction crosswalk

| Prediction | Scientific basis | Required evidence pattern | Engineering route |
| --- | --- | --- | --- |
| `SCI-FLT-MATCHED-PRED-001` | Exact `N/D`; OST unit self-response | Unit matching-template response over validity cover | CT-004, CT-008 |
| `SCI-FLT-MATCHED-PRED-002` | Linearity of fixed `L`; supplied-template amplitude | Noiseless amplitude scaling over state/location strata | CT-003, CT-004 |
| `SCI-FLT-MATCHED-PRED-003` | SB unbiasedness; zero conditional mean | Independent repeated fixed-state ensemble plus analytic expectation | CT-003, CT-004 |
| `SCI-FLT-MATCHED-PRED-004` | SOD ODQ-007 complete support | Missing-support injections produce no numerical amplitude | CT-005, CT-014 |
| `SCI-FLT-MATCHED-PRED-005` | SOD ODQ-006 invalid/nonpositive normalization | Each invalid-`D` class yields typed unavailable/failed and no zero | CT-004, CT-005, CT-014 |
| `SCI-FLT-MATCHED-PRED-006` | SOD ODQ-003/010 parent/state separation | Observation/coadd identity and state-generation comparison | CT-002, CT-009 |
| `SCI-FLT-MATCHED-PRED-007` | SOD ODQ-010; AB NOI parity | State/operator/support/normalization/failure identity for every `K_NOI`-admitted NOI member | CT-009 |
| `SCI-FLT-MATCHED-PRED-008` | SOD ODQ-009 covariance propagation; R03 F05; R04 C01/C02/C07 | Exact fixed-state covariance identity on one fixed codomain; empirical estimators use a separate preregistered protocol | CT-010 |
| `SCI-FLT-MATCHED-PRED-009` | SOD ODQ-009 `D^-1` restriction | Label/consumer audit under non-GLS weight with invariant amplitude | CT-006, CT-010 |
| `SCI-FLT-MATCHED-PRED-010` | SOD ODQ-006; RAO AO-002 | Full selected-envelope metrics, coverage, and support/null equality | CT-005, CT-007 |
| `SCI-FLT-MATCHED-PRED-011` | ACO immutable identities; OST realization tuple | Mutation/successor attempts preserve old identity and create new generation | CT-007, CT-009, CT-011 |
| `SCI-FLT-MATCHED-PRED-012` | SOD ODQ-013 atomic signal bundle | Failure injection into every required member | CT-011 |
| `SCI-FLT-MATCHED-PRED-013` | SB/SOD optional companion rule | Optional companion failure under each named-use policy | CT-010, CT-011 |
| `SCI-FLT-MATCHED-PRED-014` | SOD ODQ-005/008 amplitude scaling/units; R04 C06 | Pure fixed-state amplitude-coordinate rescaling and separate rerun case | CT-003 |
| `SCI-FLT-MATCHED-PRED-015` | OST full response; SOD ODQ-008 | Noninvariance probe rejects insufficient single-kernel reduction | CT-008 |
| `SCI-FLT-MATCHED-PRED-016` | SOD ODQ-011 no selector/fallback | Request every unavailable route and observe exact rejection | CT-012, CT-014 |
| `SCI-FLT-MATCHED-PRED-017` | SOD ODQ-012; AB FLT to FRUIT | Complete documented handoff-query suite without hidden state | CT-008, CT-013 |
| `SCI-FLT-MATCHED-PRED-018` | SB/SOD/AB exclusions and nonclaims | Public-field, profile, and consumer interpretation audit | CT-013, CT-014, CT-015 |
| `SCI-FLT-MATCHED-PRED-019` | R02 general-sky amendment; R04 C07 | Two overlapping templates compared with exact response sum | CT-004, CT-008 |
| `SCI-FLT-MATCHED-PRED-020` | R02 nuisance-response amendment; R04 C07 | Constant/gradient background and exact `L b` comparison | CT-004, CT-008 |
| `SCI-FLT-MATCHED-PRED-021` | R02 off-diagonal response; R04 C07 | Unit template at a distinct anchor with nonzero response | CT-004, CT-008 |
| `SCI-FLT-MATCHED-PRED-022` | R02 mismatch amendment; R04 C07 | Mismatched-shape injection and response-weighted amplitude | CT-003, CT-004, CT-008 |
| `SCI-FLT-MATCHED-PRED-023` | R02 response-family amendment | Authorized Learn--Resolve perturbation rerun and state-change record | CT-008, CT-009 |
| `SCI-FLT-MATCHED-PRED-024` | R02 realized covariance; R03 F02/F05; R04 C01/C02/C07 | Operational `P_C F_g` covariance on one fixed finite codomain; matrix identity only after fixed-state linearity | CT-007, CT-010 |
| `SCI-FLT-MATCHED-PRED-025` | R05 numerical application-domain closure | Missing construction-only payload with exact-zero final coefficient remains defined; nonzero coefficient activates dependency and unavailability | CT-004, CT-005 |

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
| `SCI-FLT-MATCHED-AO-001-A` | SOD ODQ-004/009; RAO AO-001; R02 | Exact constrained local inverse-covariance GLS; sole route eligible for optimality and `d_p^-1` variance | CT-004, CT-006, CT-010 | Exact-GLS weight and associated optimality/variance claim |
| `SCI-FLT-MATCHED-AO-001-B` | SOD ODQ-004; RAO AO-001; R02; R03 F08 | Nonselectable successor-authorship envelope until one concrete structured covariance-derived `W_p` is supplied | CT-006, CT-010 | Concrete structured weighting authorship and review |
| `SCI-FLT-MATCHED-AO-001-C` | SOD ODQ-004; ASC; RAO AO-001; R02; R03 F03; R04 C04 owner disposition | Exact `A_p`, `D_p`, and radially symmetrized field-power `W_p=A_p^dagger D_p A_p`, with truthful source, mandatory diagnostics, no validity thresholds, and no implied noise/covariance/isotropy/optimality | CT-006, CT-007 | Field-power weighting and exact conventions |
| `SCI-FLT-MATCHED-AO-001-D` | SOD ODQ-004; RAO AO-001; R03 F08 | Nonselectable successor-authorship trigger for one concrete weaker PSD `W_p` | CT-006, CT-007 | Concrete weaker-weight authorship and review |
| `SCI-FLT-MATCHED-AO-002-A` | SOD ODQ-006; RAO AO-002; R02 | One preregistered engineering profile for a numerical realization of the exact operator; no privileged scientific threshold | CT-007 | Numerical-conformance statement |
| `SCI-FLT-MATCHED-AO-002-B` | R02; R03 F08 | Nonselectable trigger to author an intentionally distinct scientific operator with separate identity and owner-approved error budget | CT-007 | Concrete scientifically distinct successor authorship |
| `SCI-FLT-MATCHED-AO-002-C` | R02 | Typed numerical-route unavailability when comparator/profile/coverage is absent | CT-007, CT-014 | Every numerical-conformance statement |
| `SCI-FLT-MATCHED-AO-003-A` | SOD ODQ-009; RAO AO-003; R02; R03 F04 | Complete covariance scope within one explicitly named role | CT-010, CT-011 | Complete named-role covariance use |
| `SCI-FLT-MATCHED-AO-003-B` | SOD ODQ-009; RAO AO-003; R02 | Named projected scientific covariance scope; outside correlations unknown | CT-010, CT-012 | Named projected consumer |
| `SCI-FLT-MATCHED-AO-003-C` | SOD ODQ-009; RAO AO-003; R02; R04 C03 owner disposition | Typed covariance role/status unavailability while signal remains complete unless a named-use policy requires the companion | CT-010, CT-014 | Every covariance-dependent use |
| `SCI-FLT-MATCHED-AO-003-D` | R02; R03 F08 | Engineering exact resident explicit or structured representation of the selected named-role covariance scope | CT-010, CT-011 | Resident covariance queries |
| `SCI-FLT-MATCHED-AO-003-E` | R02; R03 F08 | Engineering exact lineage/on-demand representation of the selected named-role covariance scope | CT-010, CT-011 | On-demand covariance queries |
| `SCI-FLT-MATCHED-AO-004-A` | SOD ODQ-010/013; RAO AO-004; R02; R03 F08 | Engineering full materialization preserving exact immutable state/query identity | CT-009, CT-011 | Full-state query/reanalysis route |
| `SCI-FLT-MATCHED-AO-004-B` | SOD ODQ-010/013; RAO AO-004; R02; R03 F08 | Engineering compact exact state preserving the same query identity | CT-009, CT-011 | Compact-state query/reanalysis route |
| `SCI-FLT-MATCHED-AO-004-C` | SOD ODQ-010/013; RAO AO-004; R02; R03 F08 | Engineering exact lineage reconstruction preserving the same query identity | CT-009, CT-011 | Lineage state query/reanalysis route |
| `SCI-FLT-MATCHED-AO-005-A` | SOD ODQ-012; RAO AO-005; R02; R03 F08 | Engineering full response representation after domain/query/validity/consumer scope is fixed | CT-008, CT-013 | Full response queries and handoff |
| `SCI-FLT-MATCHED-AO-005-B` | SOD ODQ-012; RAO AO-005; R02; R03 F08 | Engineering exact structured response preserving the fixed scientific query object | CT-008, CT-013 | Structured response queries and handoff |
| `SCI-FLT-MATCHED-AO-005-C` | SOD ODQ-012; RAO AO-005; R02; R03 F08 | Engineering exact lineage/on-demand response preserving the fixed scientific query object | CT-008, CT-009, CT-013 | Lineage response queries and handoff |
| `SCI-FLT-MATCHED-AO-006-A` | SOD ODQ-013; RAO AO-006; R02; R03 F08 | Engineering seven-record layout for one owner-disposed dependency graph | CT-012 | SCI-VAL named-use evaluation |
| `SCI-FLT-MATCHED-AO-006-B` | SOD ODQ-013; RAO AO-006; R02; R03 F08 | Engineering grouped lossless layout retaining seven mandatory subverdicts | CT-012 | SCI-VAL named-use evaluation |
| `SCI-FLT-MATCHED-AO-006-C` | SOD ODQ-013; RAO AO-006; R02; R03 F08 | Engineering seven-role vector layout; scalar action forbidden | CT-012 | SCI-VAL named-use evaluation |

## Product, lifecycle, and claim-layer closure

| Concern | Normative locus | Scientist explanation | Engineering evidence |
| --- | --- | --- | --- |
| Atomic signal bundle | Definitions: Product roles; REQ-023/024 | Product and governance rationale | CT-011 |
| Qualified companions | Definitions: Product roles; REQ-025 | Response and uncertainty; product rationale | CT-010/011 |
| Complete lifecycle vocabulary | Definitions: Lifecycle; REQ-026 | Product rationale | CT-011/012 |
| FLT to FRUIT interface | Definitions; AO-005; REQ-028/029 | Product rationale; owner guide | CT-008/013 |
| No source or FRUIT science | Definitions; Edge cases; REQ-029--031 | Scientific question; product rationale | CT-013/014 |
| Scientific authority vs conformity vs representation fidelity vs SCI-VAL vs observational validation vs performance/readiness/production | Definitions: claim layers; REQ-034/039 | Governance rationale | CT-001/015 |

## Source/PDF consistency obligation

`build/verify_consistency.py` checks that both LaTeX views import all shared
modules in the same order, that the shared source contains all 50 stable
requirement IDs, all 25 prediction IDs, and all 21 exact option
identities, and that draft/nonclaim language survives rendering. Its report is
a build-consistency artifact only, not scientific validation or implementation
conformity evidence.
