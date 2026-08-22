# SCI-AST v0.1 — Stage B r0.3 Targeted Draft Crosswalk

Status: author-draft traceability; not scientific approval, implementation
conformity, validation, freeze, or readiness

Prepared: `2026-08-22`

This crosswalk traces the admitted author authority, the targeted revision
directive, stable owner decisions, typed open questions, six canonical formal
modules, 90 requirements, and 50 falsifiable predictions. Requirement and
prediction mappings are many-to-many; the tables identify primary coverage
rather than transferring ownership. The exact shared interface is
`SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; its installed source digest is recorded in
`SOURCE_MANIFEST.md`.

## Canonical Artifacts And Views

| Artifact | Role | Included in scientist PDF | Included in engineering PDF |
| --- | --- | --- | --- |
| `src/common/notation.tex` | Symbols, units, indices, role-factored parents, availability and five cause types | narrative summary | yes |
| `src/common/definitions.tex` | Scientific object, ownership, boundaries, ordered composition, lifecycle | narrative summary | yes |
| `src/common/equations.tex` | Exact identities/oracle, conditional production relation, WCS, Jacobian, RTC/MAP relations | selected explanatory equations | yes |
| `src/common/assumptions.tex` | Fifteen bounded assumptions and explicit owner-question gates | compact summary | yes |
| `src/common/requirements.tex` | `SCI-AST-REQ-001`–`SCI-AST-REQ-090` | compact range crosswalk | yes |
| `src/common/edge_cases.tex` | `SCI-AST-PRED-001`–`SCI-AST-PRED-050` | compact range crosswalk | yes |
| `src/scientific-rationale.tex` | 8–10 page scientist-facing narrative with three explanatory figures | source | companion view |
| `src/engineering-conformance.tex` | Complete formal Engineering Conformance Specification | companion view | source |
| `OWNER_DECISION_REGISTER.md` | Exact stable owner decisions and open questions | supporting register | supporting register |
| `AUTHOR_DRAFT_DECISIONS.md` | Every author choice, inconsistency disposition, question and unavailable claim | supporting record | supporting record |

## Approved Packet Input To Canonical Authority

| Approved input | Primary canonical destination | Stable output coverage |
| --- | --- | --- |
| `SCOPE_BRIEF.md` | all six modules | Full v0.1 boundary; `REQ-001`–`090`; `PRED-001`–`050` |
| `AUTHOR_DECISIONS_AND_OPEN_QUESTIONS.md` | `OWNER_DECISION_REGISTER.md`, assumptions, requirements | `SCI-AST-001-D001`–`D008`; `AST-SCOPE-D001`–`D027`; `AST-OWNER-Q001`–`Q008` |
| `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` (`SCI-ALIGN_TO_SCI-AST v0.1/r0.1`) | notation, definitions, requirements, edge cases | `REQ-006`–`012` (exact profile/transfer, stable slot `s`, paired-signal provenance and five causes); `REQ-013`–`022` (producer-selected pointing support, exact occurrence-time/current-state separation, interpolation within support and no extrapolation); `REQ-056`–`060` (role validity, dependency-limited failure and layered atomicity); `REQ-073`–`079` (ALIGN/RTC role parents and no angular filtering); `REQ-080`–`083` (AST/MAP projection ownership, base pre-MAP facts, exact MAP-owned `G_pi`); `REQ-084`–`088` (parent/version and four-stage provenance); `PRED-004`–`010`, `015`, `018`, `025`, `030`, `035`–`040`, `049` |
| `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md` | notation, definitions, equations, requirements, edge cases | `REQ-073`–`079`, `082`; `PRED-035`–`038` |
| `DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md` | definitions, equations, requirements, edge cases | `REQ-023`–`035`, `065`–`072`, `080`–`083`; `PRED-011`–`017`, `032`–`040` |
| `AUTHOR_SUPERSESSION_COVER.md` plus frozen independent core | equations, requirements, edge cases, author dispositions | Exact spherical and WCS identities; adopted limiting cases; all explicit supersessions recorded in `AUTHOR_DRAFT_DECISIONS.md` |
| `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | notation, definitions, requirements | Identity/indexing; frames/topology; producer/consumer boundaries; four-stage provenance; claim separation |

## Previously Decided Constraint Crosswalk

| Owner decision | Canonical realization | Primary requirements | Primary predictions |
| --- | --- | --- | --- |
| `SCI-AST-001-D001` | Layered detector binding; explicit sign/basis/handedness/order | `REQ-003`–`005`, `014`–`015`, `024`–`026`, `033`–`038` | `PRED-002`–`003`, `011`–`012` |
| `SCI-AST-001-D002` | TAN domain finite `D > 0`; no universal `D_min` | `REQ-042`–`043` | `PRED-021`–`022` |
| `SCI-AST-001-D003` | AltAz/science frame split and circular topology | `REQ-036`–`044` | `PRED-018`–`022`, `047` |
| `SCI-AST-001-D004` | Constant/MJD/span support and adequacy-governed time | `REQ-013`–`022`, `089` | `PRED-004`–`008`, `046` |
| `SCI-AST-001-D005` | Exact six-zero legacy adapter and explicit WCS request states | `REQ-045`–`048` | `PRED-023`–`025` |
| `SCI-AST-001-D006` | Typed map-center-only Jacobian/covariance | `REQ-065`–`072` | `PRED-032`–`034` |
| `SCI-AST-001-D007` | Bounded production small-angle relation; spherical oracle | `REQ-035`, `089` | `PRED-045`–`046` |
| `SCI-AST-001-D008` | Precision gates, factorized validity, truthful products, atomic provenance | `REQ-001`, `010`–`012`, `056`–`064`, `084`–`090` | `PRED-009`–`010`, `030`–`031`, `041`–`046`, `049`–`050` |

## Approved Scope-Decision Crosswalk

| Owner decision | Canonical realization | Primary requirements | Primary predictions |
| --- | --- | --- | --- |
| `AST-SCOPE-D001` | Complete boundary and atomic output | `REQ-001`–`090` | `PRED-001`–`050` |
| `AST-SCOPE-D002` | Core admitted only through supersession cover | `REQ-035`, `042`–`043`, `065`–`072`, `090` | `PRED-021`, `032`–`034`, `045`, `050` |
| `AST-SCOPE-D003` | Exact ALIGN import; no AST policy selection | `REQ-006`–`011` | `PRED-009`–`010` |
| `AST-SCOPE-D004` | Five independent causes; no rescue/precedence | `REQ-012`, `056`–`057` | `PRED-010`, `048` |
| `AST-SCOPE-D005` | Producer record selection, ALIGN time, AST application | `REQ-013`–`022` | `PRED-004`–`008`, `015` |
| `AST-SCOPE-D006` | Upstream center selection, deterministic AST realization | `REQ-045`–`046`, `049`–`051` | `PRED-023`, `025` |
| `AST-SCOPE-D007` | Map-center uncertainty only | `REQ-065`–`072` | `PRED-032`–`034` |
| `AST-SCOPE-D008` | Preregistered approximation/time/precision gates | `REQ-035`, `089` | `PRED-045`–`046` |
| `AST-SCOPE-D009` | Only the ordinary nonpolarimetric coordinate path is in scope; optional HWPR timing remains only an ALIGN-parent fact; no AST demodulation, polarization calibration/response, or Stokes reconstruction; raw KID `x` is not Stokes I; other named non-goals remain outside AST | `REQ-039`–`041`, `080`, `090`; `ASM-014` | `PRED-040`, `047`–`048`, `050` |
| `AST-SCOPE-D010` | Exact content-bound input packet only | author verification record | document-control check, not a scientific prediction |
| `AST-SCOPE-D011` | Stable decisions and precise owner questions | `OWNER_DECISION_REGISTER.md`; `SCI-AST-ASM-001`–`015` | typed unavailable outcomes throughout |
| `AST-SCOPE-D012` | Contract/status claim separation | `REQ-090` | `PRED-050` |
| `AST-SCOPE-D013` | One immutable ALIGN boundary; no substitutes | `REQ-006`–`009` | `PRED-009`–`010`, `049` |
| `AST-SCOPE-D014` | Complete geometry/rotation artifact and no inferred pivot/origin | `REQ-023`–`029` | `PRED-012`, `015`–`017` |
| `AST-SCOPE-D015` | Same APT/rotation convention or exact uncertain transform | `REQ-030`–`032` | `PRED-013`–`014` |
| `AST-SCOPE-D016` | Eight-stage noncommuting AST order | `REQ-021`, `027`, `033`–`034`, `040`, `042`, `045`, `049`–`053` | `PRED-001`, `011`, `015`, `018`, `021`, `023`, `026`–`028`, `048` |
| `AST-SCOPE-D017` | Producer support; AST bracket-only interpolation | `REQ-013`–`021` | `PRED-004`–`007` |
| `AST-SCOPE-D018` | Legacy adapter only; zero otherwise numeric | `REQ-047`–`048` | `PRED-024` |
| `AST-SCOPE-D019` | Automatic WCS uses exact upstream center only | `REQ-045`–`046` | `PRED-023`, `025` |
| `AST-SCOPE-D020` | AST TAN/WCS versus MAP deposition ownership | `REQ-050`–`055`, `080`–`083` | `PRED-026`–`030`, `038`–`040` |
| `AST-SCOPE-D021` | Typed `astrometric_map_center_jacobian` | `REQ-065`–`072` | `PRED-032`–`034` |
| `AST-SCOPE-D022` | BEAM/TolAPT ownership and bounded small-angle gate | `REQ-023`–`035`, `089` | `PRED-012`–`017`, `045`–`046` |
| `AST-SCOPE-D023` | Coordinate independence from finite `x/r`; distinct observing state/causes | `REQ-009`–`012`, `021`, `056`–`057` | `PRED-009`–`010`, `015` |
| `AST-SCOPE-D024` | Separate ALIGN/RTC coordinate parents; no angular filtering | `REQ-073`–`079` | `PRED-035`–`037` |
| `AST-SCOPE-D025` | RTC parent on applicable delegated `G_pi` | `REQ-081`–`083` | `PRED-038`–`039` |
| `AST-SCOPE-D026` | Geometry authorities, compatibility, typed-open/fail-closed state | `REQ-023`–`032` | `PRED-012`–`017` |
| `AST-SCOPE-D027` | Sanitized decisions/questions admitted; raw process excluded | `OWNER_DECISION_REGISTER.md`; `AUTHOR_DRAFT_DECISIONS.md` | document-control check, not a scientific prediction |

## Requirement Inventory And Primary Authority

Every requirement identifier appears exactly once in the canonical
`requirements.tex`; these ranges give the primary authority route.

| Requirement range | Subject | Primary packet authority |
| --- | --- | --- |
| `SCI-AST-REQ-001`–`012` | lifecycle, identity, exact ALIGN import, `x/r` independence, five causes | scope, ALIGN boundary, conventions, `D001`, `D008`, `D013`, `D023` |
| `SCI-AST-REQ-013`–`022` | pointing selection, modes, support, current-state separation, covariance | core under cover, ALIGN boundary, conventions, `D004`, `D005`, `D017` |
| `SCI-AST-REQ-023`–`035` | geometry authority/artifact, same realization, noncommuting order, adequacy | geometry boundary, scope, `D001`, `D007`, `D014`–`D016`, `D022`, `D026` |
| `SCI-AST-REQ-036`–`044` | vector direction, topology, frame split, TAN domain | core under cover, conventions, `D002`, `D003` |
| `SCI-AST-REQ-045`–`055` | center, WCS request state/identity, continuous/discrete pixels | scope, core under cover, conventions, `D005`, `D006`, `D018`–`D020` |
| `SCI-AST-REQ-056`–`064` | validity, atomic outputs, full/mini, lifecycle/parity | scope, ALIGN boundary, core under cover, `D004`, `D008`, `D023` |
| `SCI-AST-REQ-065`–`072` | typed center Jacobian and covariance | scope, geometry boundary, core under cover, `D006`, `D007`, `D021` |
| `SCI-AST-REQ-073`–`079` | ALIGN-grid and RTC-grid coordinate roles and response | RTC boundary, scope, `D024` |
| `SCI-AST-REQ-080`–`083` | AST/MAP ownership and delegated `G_pi` | scope, ALIGN/RTC/geometry boundaries, `D020`, `D025` |
| `SCI-AST-REQ-084`–`090` | four-stage provenance, empirical gates, claim separation | scope, conventions, core under cover, `D008`, `D012` |

## Prediction Inventory And Primary Requirement Coverage

| Prediction range | Primary requirement coverage |
| --- | --- |
| `SCI-AST-PRED-001`–`008` | `REQ-013`–`022`, `033`–`038` |
| `SCI-AST-PRED-009`–`010` | `REQ-006`–`012`, `056`–`057` |
| `SCI-AST-PRED-011`–`017` | `REQ-023`–`035` |
| `SCI-AST-PRED-018`–`022` | `REQ-036`–`044` |
| `SCI-AST-PRED-023`–`031` | `REQ-045`–`061` |
| `SCI-AST-PRED-032`–`034` | `REQ-065`–`072` |
| `SCI-AST-PRED-035`–`037` | `REQ-073`–`079` |
| `SCI-AST-PRED-038`–`040` | `REQ-080`–`083` |
| `SCI-AST-PRED-041`–`044` | `REQ-058`–`064`, `084`–`088` |
| `SCI-AST-PRED-045`–`050` | `REQ-035`, `039`–`041`, `089`–`090` |

## Typed Open-Question Crosswalk

| Open question | Canonical conditional gate | Requirements kept conditional | Unavailable claim |
| --- | --- | --- | --- |
| `AST-OWNER-Q001` | `SCI-AST-ASM-002` | `REQ-009`, `021`, `086` | Unconditional field admission/equivalence |
| `AST-OWNER-Q002` | `SCI-AST-ASM-001` | `REQ-045`–`046`, `086` | Unconditional physical-center provenance |
| `AST-OWNER-Q003` | `SCI-AST-ASM-010` | `REQ-069`–`072` | Complete family uncertainty statement |
| `AST-OWNER-Q004` | `SCI-AST-ASM-005`–`007` | `REQ-035`, `089` | Quantitative achieved approximation/time/precision adequacy |
| `AST-OWNER-Q005` | `SCI-AST-ASM-013` | `REQ-080`–`083` remain generic | MAP-003-specific retained-grid interface |
| `AST-OWNER-Q006` | `SCI-AST-ASM-003` | `REQ-023`–`032` | Unconditional geometry and pointing transfer |
| `AST-OWNER-Q007` | `SCI-AST-ASM-011` | `REQ-065`–`072` | Available family-specific Jacobian |
| `AST-OWNER-Q008` | closed by r0.3 owner directive; `SCI-AST-ASM-012` records disposition | `REQ-074`–`079`, `082` | Ordinary science chain no longer blocked; missing exact RTC parent makes only the RTC role unavailable |

No open question supplies a default or weakens an unrelated exact relation.

## r0.3 bounded-repair parity

| Scientific concept | Rationale location | Formal authority | Stable falsifier |
| --- | --- | --- | --- |
| `s/j/n/d/p` and `x/r` meanings | reading guide; ALIGN parent; RTC section | notation and role parents; REQ-006-012, 073-079 | PRED-009-010, 035-037 |
| Complete spherical oracle | Section 3 | exact-displacement equation; REQ-035-036 | PRED-032, 045 |
| RTC-grid ownership | Section 7 | `SCI-AST:rtc_output_grid_coordinates@1`; REQ-074-079 | PRED-035-038 |
| No angular signal filtering | Section 7 | `not-angular-filter`; REQ-078-079 | PRED-037 |
| No double field rotation | Sections 3-4 | geometry declaration/application count; REQ-024, 027-028, 034 | PRED-014-016 |
| `G_pi` and MAP ownership | Section 8 | deposition parent; REQ-080-083 | PRED-038-040 |
| Exact occurrence time and typed unavailability | ALIGN parent and Sections 3, 9 | definitions; REQ-006-012, 021, 056-060 | PRED-009-010, 036, 049 |

No stable requirement or prediction identifier is renumbered.
