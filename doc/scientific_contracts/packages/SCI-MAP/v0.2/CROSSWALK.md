# SCI-MAP v0.2/r0.1 requirement crosswalk

Status: candidate traceability aid; no implementation, validation, freeze, or
production claim. Canonical wording lives in the six common LaTeX modules.
Section names are navigation aids and do not create a second authority.

| Requirement | Scientist-facing source | Engineering interpretation |
| --- | --- | --- |
| SCI-MAP-REQ-001 | `notation.tex`; rationale cover and Document authority | Record contract v0.2 and document r0.1 separately in every view and artifact. |
| SCI-MAP-REQ-002 | Rationale: What SCI-MAP estimates; `definitions.tex` quantity | Accept only the exact realized positive-rank PTC transformed product and beam/unit parent; reject CAL fallback, disabled/no-product routes, and Stokes promotion. |
| SCI-MAP-REQ-003 | Rationale: From samples to pixels; `definitions.tex` occurrence identity | Join by complete stable occurrence and generation, never container, row, time, or coordinate equality. |
| SCI-MAP-REQ-004 | Rationale: From samples to pixels; producer/consumer flow | Preserve PTC producer facts, PTC QC, MAP admission, VAL evaluation, causes, and generations as distinct records. |
| SCI-MAP-REQ-005 | Rationale: From samples to pixels; Projection, response, and removed modes | Bind same-`n` AST coordinates and apply only the one-hot half-open MAP projection. |
| SCI-MAP-REQ-006 | Rationale: What SCI-MAP estimates; exact uniform binding block | Require explicit admitted-family selection and complete exact handoff; apply independent MAP numerical classification with no default or fallback. |
| SCI-MAP-REQ-007 | Rationale: Coverage, support, and validity; exposure equations | Deduplicate stable originals and place each once by its own AST original-footprint coordinate and target WCS. |
| SCI-MAP-REQ-008 | Rationale: Projection, response, and removed modes; response equations | Disclose the known MAP-local operator and separately classify upstream-conditioned fixed, PTC-procedure, re-resolved, and whole-chain states. |
| SCI-MAP-REQ-009 | Rationale: Units, WCS, products, and downstream use | Keep requested, effective, observation-resolved, applied, and realized state one-way and observation-local. |
| SCI-MAP-REQ-010 | Rationale: From samples to pixels; gate equations | Evaluate typed gates in the ordered DAG, project only exact passing states, and classify before arithmetic. |
| SCI-MAP-REQ-011 | Rationale: From samples to pixels; ordinary accumulator equations | Accumulate numerator, normalization, and matching response over exactly the admitted positive contribution set. |
| SCI-MAP-REQ-012 | Rationale: From samples to pixels; support-row selection equations | Publish normalized scientific values only on exact authorized finite-positive rows; unsupported rows are absent, not zero. |
| SCI-MAP-REQ-013 | Rationale: Weights and uncertainty | Keep numerator and normalization roles distinct from sky estimate, precision, exposure, coverage, and significance. |
| SCI-MAP-REQ-014 | Rationale: What SCI-MAP estimates; input-class predictions | Verify constant preservation only on the exact fixed support-row domain and do not promote it to stronger response claims. |
| SCI-MAP-REQ-015 | Rationale: Projection, response, and removed modes | Keep one-hot containing-pixel projection as the exact estimator family; reject fractional/WLS substitution. |
| SCI-MAP-REQ-016 | Rationale: Projection, response, and removed modes; operator equations | Bind and disclose one exact row-selected MAP operator for signal and every named linear companion. |
| SCI-MAP-REQ-017 | Rationale: Projection, response, and removed modes | Call a kernel realized response only with exact template, sample parent, family, state, support rows, and lineage. |
| SCI-MAP-REQ-018 | Rationale: Projection, response, and removed modes | Apply no MAP/coadd signal centering and preserve upstream removed-mode state. |
| SCI-MAP-REQ-019 | Rationale: Weights and uncertainty; covariance equation | Propagate only the declared exact-domain covariance and preserve every incomplete or unavailable term honestly. |
| SCI-MAP-REQ-020 | Rationale: Weights and uncertainty; formal-weight equations | Call normalization inverse variance only when coefficient and independence premises and projection identity all hold. |
| SCI-MAP-REQ-021 | Rationale: Weights and uncertainty; formal-weight equations | Compute and label formal diagonal weight separately from normalization, covariance precision, exposure, hits, and empirical weight. |
| SCI-MAP-REQ-022 | Rationale: Weights and uncertainty; approved OD-004 | Persist actual uncertainty/covariance meaning, domain, assumptions, omissions, limits, state, and lineage without zero fill. |
| SCI-MAP-REQ-023 | Rationale: Weights and uncertainty; conditionality statement | Label response/covariance conditional when state is data-derived; require a separate joint model for unconditional claims. |
| SCI-MAP-REQ-024 | Rationale: Weights and uncertainty | Prevent merely formal standardized signal from being represented as empirical significance. |
| SCI-MAP-REQ-025 | Rationale: Coverage, support, and validity | Store geometric, route, contribution, count, two exposure, two support, and MAP-validity facts separately. |
| SCI-MAP-REQ-026 | Rationale: Coverage, support, and validity | Evaluate geometric incidence, route candidacy, and estimator contribution over their distinct populations. |
| SCI-MAP-REQ-027 | Rationale: Coverage, support, and validity | Count only exact contribution-set members and retain the distinct cause for every exclusion. |
| SCI-MAP-REQ-028 | Rationale: Observation coaddition | Count complete admitted observation bundles per row, not sample hits or attempts. |
| SCI-MAP-REQ-029 | Rationale: Coverage, support, and validity; exposure equations | Form upstream-eligible exposure from exact eligible ancestors using each original's own coordinate, before coefficient/MAP numerical rejection. |
| SCI-MAP-REQ-030 | Rationale: Coverage, support, and validity; exposure/coadd equations | Form retained and coadd exposure from admitted ancestor unions without duplicating originals or interpreting seconds as precision. |
| SCI-MAP-REQ-031 | Rationale: Coverage, support, and validity; threshold equations | Preserve the exact population/order statistic and record four exact `coverage_cut` stages; enforce zero, ordinary, expert, and invalid states before support mutation. |
| SCI-MAP-REQ-032 | Rationale: Coverage, support, and validity; threshold equations | Apply the two inclusive thresholds only after exact cut admission; finite-positive normalization remains mandatory and support has no stronger physical meaning. |
| SCI-MAP-REQ-033 | Rationale: Coverage, support, and validity | Compute MAP-local validity only from its exact required facts; limited response/covariance alone does not invalidate or promote it. |
| SCI-MAP-REQ-034 | Rationale: From samples to pixels; ordered gate flow | Retrieve only structurally admitted payloads, classify them safely, and complete every gate before arithmetic. |
| SCI-MAP-REQ-035 | Rationale: Coverage, support, and validity | Keep zero, low-positive, negative, non-finite, and out-of-grid causes distinct with their exact contribution behavior. |
| SCI-MAP-REQ-036 | Rationale: Observation coaddition; complete-bundle block | Construct one complete immutable logical bundle, with MAP-local response and honest companion states, before coaddition or persistence. |
| SCI-MAP-REQ-037 | Rationale: Observation coaddition | Admit a complete observation atomically before changing any coadd state. |
| SCI-MAP-REQ-038 | Rationale: Observation coaddition; Units, WCS, products | Honor a preconstruction target grid and exact centered-integer embedding; give any later crop a new row-selected identity and reject other grid transforms. |
| SCI-MAP-REQ-039 | Rationale: Observation coaddition; coadd equations | Apply equal-observation arithmetic on exact rows identically for conforming in-memory and persisted bundles. |
| SCI-MAP-REQ-040 | Rationale: Observation coaddition; response companion equation | Keep base-signal membership when upstream response is honestly unavailable; require complete compatible response for response-bearing roles. |
| SCI-MAP-REQ-041 | Rationale: Observation coaddition; covariance companion equation | Publish complete coadd covariance only with every required located block; otherwise publish the exact incomplete state without assuming independence. |
| SCI-MAP-REQ-042 | Rationale: Observation coaddition | Reject correlated GLS as a substitution for the ordinary positive-coefficient coadd. |
| SCI-MAP-REQ-043 | Rationale: Units, WCS, products, and downstream use | Persist complete product identity, including operator/response, coefficient, grid, logical/persistence, lifecycle, parent, and cause fields. |
| SCI-MAP-REQ-044 | Rationale: Units, WCS, products, and downstream use | Bind full AST frame/WCS and exact signal/beam units; reject name-based frame or beam equivalence. |
| SCI-MAP-REQ-045 | Rationale: Units, WCS, products, and downstream use | Keep full-precision WCS authoritative and check FITS/index round trips, orientation, and 0.1-arcsec bound. |
| SCI-MAP-REQ-046 | Rationale: Units, WCS, products, and downstream use | Record enough exact state to reconstruct cut transitions, coefficient handoff, operator, exposure, response, coadd route, and failures. |
| SCI-MAP-REQ-047 | Rationale: Units, WCS, products, and downstream use; failure table | Propagate each failure at its declared scope before live state mutation or completion. |
| SCI-MAP-REQ-048 | Rationale: Observation coaddition; publication block | Require the logical bundle, enforce plan-required publication, permit equivalent in-memory/coadd-only routes, and retain PTC completion duties. |
| SCI-MAP-REQ-049 | Rationale: Units, WCS, products, and downstream use | Preserve immutable base/unfiltered MAP identity and validity through all derivative products. |
| SCI-MAP-REQ-050 | Rationale: Units, WCS, products, and downstream use | Supply the complete scientific bundle to consumers and let each consumer declare stronger claim requirements without rewriting MAP. |
| SCI-MAP-REQ-051 | Rationale: Validation; fixed-operator discussion | Claim the same noise operator only when every operator-defining fact and row is fixed or identically prescribed. |
| SCI-MAP-REQ-052 | Rationale: Units, WCS, products, and downstream use; Owner decisions | Bound Pointing/OOF reuse to the exact operator and mode authority; reject method/claim promotion while preserving independent experimentation and versioned identity. |

Completeness: 52 unique contiguous requirement IDs appear once in this table
and once in `src/common/requirements.tex`.
