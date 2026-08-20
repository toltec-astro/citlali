# SCI-PTC v0.1 -- Requirement And Prediction Crosswalk

Status: frozen scientific authority `v0.1/r0.4`; generated from the shared normative macro metadata

Coverage: 89 requirements and 50 predictions.

| Identifier | Shared canonical source | Scientist-facing source | Engineering-facing source | Owner decision | Dependency |
| --- | --- | --- | --- | --- | --- |
| `SCI-PTC-REQ-001` | `src/common/requirements.tex` (Admitted calibrated signal) | Rationale 1.1; compact crosswalk | Engineering normative requirements | D001,D002 | SCI-CAL |
| `SCI-PTC-REQ-002` | `src/common/requirements.tex` (Complete RTC lineage) | Rationale 1.1; compact crosswalk | Engineering normative requirements | D001,D008 | SCI-RTC |
| `SCI-PTC-REQ-003` | `src/common/requirements.tex` (Upstream availability) | Rationale 1.1; compact crosswalk | Engineering normative requirements | D003 | SCI-RTC,SCI-CAL |
| `SCI-PTC-REQ-004` | `src/common/requirements.tex` (Matrix shape) | Rationale 1.1; compact crosswalk | Engineering normative requirements | D001 | SCI-CAL |
| `SCI-PTC-REQ-005` | `src/common/requirements.tex` (Time identity) | Rationale 1.1; compact crosswalk | Engineering normative requirements | D001 | SCI-RTC |
| `SCI-PTC-REQ-006` | `src/common/requirements.tex` (Detector identity) | Rationale 1.1; compact crosswalk | Engineering normative requirements | D011 | APT |
| `SCI-PTC-REQ-007` | `src/common/requirements.tex` (Within-array domain) | Rationale 3.2; compact crosswalk | Engineering normative requirements | D011 | APT,SCI-CAL |
| `SCI-PTC-REQ-008` | `src/common/requirements.tex` (Requested operation) | Rationale 4.1; compact crosswalk | Engineering normative requirements | D010,D012 | Owner policy |
| `SCI-PTC-REQ-009` | `src/common/requirements.tex` (One-way state lifecycle) | Rationale 4.1 and 8.1; compact crosswalk | Engineering normative requirements | D004,D009 | None |
| `SCI-PTC-REQ-010` | `src/common/requirements.tex` (Unit preservation limit) | Rationale 1.2; compact crosswalk | Engineering normative requirements | D002,D014 | SCI-CAL |
| `SCI-PTC-REQ-011` | `src/common/requirements.tex` (Cause is not action) | Rationale 2.1; compact crosswalk | Engineering normative requirements | D007 | Cause producers, shared VAL types |
| `SCI-PTC-REQ-012` | `src/common/requirements.tex` (Composite named-use support) | Rationale 2.1; compact crosswalk | Engineering normative requirements | D007 | PTC and named-use owners |
| `SCI-PTC-REQ-013` | `src/common/requirements.tex` (Finite eligible arithmetic) | Rationale 2.1; compact crosswalk | Engineering normative requirements | D007 | PTC named-use policy |
| `SCI-PTC-REQ-014` | `src/common/requirements.tex` (No silent zero filling) | Rationale 2.1; compact crosswalk | Engineering normative requirements | D007,D010 | None |
| `SCI-PTC-REQ-015` | `src/common/requirements.tex` (Fit-excluded apply-allowed state) | Rationale 2.1; compact crosswalk | Engineering normative requirements | D007 | PTC named-use policy |
| `SCI-PTC-REQ-016` | `src/common/requirements.tex` (Direct synthesized/replaced occurrences) | Rationale 2.1; compact crosswalk | Engineering normative requirements | D007 | SCI-RTC,ALIGN |
| `SCI-PTC-REQ-017` | `src/common/requirements.tex` (Transitive influence) | Rationale 2.2; compact crosswalk | Engineering normative requirements | D007 | SCI-RTC,VAL |
| `SCI-PTC-REQ-018` | `src/common/requirements.tex` (Mask identity) | Rationale 2.2; compact crosswalk | Engineering normative requirements | D004,D007 | AST,ALIGN |
| `SCI-PTC-REQ-019` | `src/common/requirements.tex` (Declared estimand) | Rationale 1.2; compact crosswalk | Engineering normative requirements | D004,D014 | None |
| `SCI-PTC-REQ-020` | `src/common/requirements.tex` (No physical-origin inference) | Rationale 1.2; compact crosswalk | Engineering normative requirements | D004,D014 | None |
| `SCI-PTC-REQ-021` | `src/common/requirements.tex` (Removed subspace publication) | Rationale 1.2; compact crosswalk | Engineering normative requirements | D014 | None |
| `SCI-PTC-REQ-022` | `src/common/requirements.tex` (Additive reference and null space) | Rationale 1.3; compact crosswalk | Engineering normative requirements | D014,D017 | None |
| `SCI-PTC-REQ-023` | `src/common/requirements.tex` (Centering contract) | Rationale 3.1; compact crosswalk | Engineering normative requirements | D017 | None |
| `SCI-PTC-REQ-024` | `src/common/requirements.tex` (Scaling contract) | Rationale 3.1; compact crosswalk | Engineering normative requirements | D017 | None |
| `SCI-PTC-REQ-025` | `src/common/requirements.tex` (Gauge-invariant claim) | Rationale 3.1; compact crosswalk | Engineering normative requirements | D014,D015 | None |
| `SCI-PTC-REQ-026` | `src/common/requirements.tex` (Estimator-family identity) | Rationale 4.1; compact crosswalk | Engineering normative requirements | D010 | None |
| `SCI-PTC-REQ-027` | `src/common/requirements.tex` (Robust common-mode family) | Rationale 4.1; compact crosswalk | Engineering normative requirements | D010 | None |
| `SCI-PTC-REQ-028` | `src/common/requirements.tex` (Fixed-template family) | Rationale 4.1; compact crosswalk | Engineering normative requirements | D010 | None |
| `SCI-PTC-REQ-029` | `src/common/requirements.tex` (Masked/weighted PCA family) | Rationale 4.1--4.2; compact crosswalk | Engineering normative requirements | D007,D010,Q002 | None |
| `SCI-PTC-REQ-030` | `src/common/requirements.tex` (Diagnostic-only conditioned $r$) | Rationale 4.2; compact crosswalk | Engineering normative requirements | D008,Q001 | SCI-RTC |
| `SCI-PTC-REQ-031` | `src/common/requirements.tex` (Hierarchical groups) | Rationale 3.2; compact crosswalk | Engineering normative requirements | D011 | APT |
| `SCI-PTC-REQ-032` | `src/common/requirements.tex` (Joint or sequential order) | Rationale 3.2; compact crosswalk | Engineering normative requirements | D011 | None |
| `SCI-PTC-REQ-033` | `src/common/requirements.tex` (Data-derived grouping) | Rationale 3.2; compact crosswalk | Engineering normative requirements | D011 | None |
| `SCI-PTC-REQ-034` | `src/common/requirements.tex` (Finite candidate set) | Rationale 4.3; compact crosswalk | Engineering normative requirements | D012 | Owner policy |
| `SCI-PTC-REQ-035` | `src/common/requirements.tex` (Conjunctive admission) | Rationale 4.3; compact crosswalk | Engineering normative requirements | D012 | Owner policy |
| `SCI-PTC-REQ-036` | `src/common/requirements.tex` (No compensating score) | Rationale 4.3; compact crosswalk | Engineering normative requirements | D012 | Owner policy |
| `SCI-PTC-REQ-037` | `src/common/requirements.tex` (Ordering and ties) | Rationale 4.3; compact crosswalk | Engineering normative requirements | D012 | Owner policy |
| `SCI-PTC-REQ-038` | `src/common/requirements.tex` (Nonnested candidates) | Rationale 4.3; compact crosswalk | Engineering normative requirements | D012 | Owner policy |
| `SCI-PTC-REQ-039` | `src/common/requirements.tex` (Empty admission set) | Rationale 4.3; compact crosswalk | Engineering normative requirements | D012 | Owner policy |
| `SCI-PTC-REQ-040` | `src/common/requirements.tex` (No universal threshold) | Rationale 4.3; compact crosswalk | Engineering normative requirements | D012 | None |
| `SCI-PTC-REQ-041` | `src/common/requirements.tex` (Source-mask limit) | Rationale 3.3; compact crosswalk | Engineering normative requirements | D004,D014 | AST,ALIGN |
| `SCI-PTC-REQ-042` | `src/common/requirements.tex` (Diagnostic definition) | Rationale 5.1; compact crosswalk | Engineering normative requirements | D009 | Owner policy |
| `SCI-PTC-REQ-043` | `src/common/requirements.tex` (Pathology discrimination) | Rationale 5.1; compact crosswalk | Engineering normative requirements | D009 | Owner policy |
| `SCI-PTC-REQ-044` | `src/common/requirements.tex` (Owner-controlled thresholds) | Rationale 5.1; compact crosswalk | Engineering normative requirements | D009 | Owner policy |
| `SCI-PTC-REQ-045` | `src/common/requirements.tex` (Finite refinement) | Rationale 5.2; compact crosswalk | Engineering normative requirements | D009 | None |
| `SCI-PTC-REQ-046` | `src/common/requirements.tex` (Immutable-parent refit) | Rationale 5.2; compact crosswalk | Engineering normative requirements | D009 | SCI-CAL |
| `SCI-PTC-REQ-047` | `src/common/requirements.tex` (Non-fit dispositions) | Rationale 5.2; compact crosswalk | Engineering normative requirements | D007,D009 | VAL |
| `SCI-PTC-REQ-048` | `src/common/requirements.tex` (Refinement stopping state) | Rationale 5.2; compact crosswalk | Engineering normative requirements | D009 | Owner policy |
| `SCI-PTC-REQ-049` | `src/common/requirements.tex` (Internal iteration identity) | Rationale 5.3; compact crosswalk | Engineering normative requirements | D009 | None |
| `SCI-PTC-REQ-050` | `src/common/requirements.tex` (PTC pass identity) | Rationale 5.3; compact crosswalk | Engineering normative requirements | D005,D009 | FRUIT |
| `SCI-PTC-REQ-051` | `src/common/requirements.tex` (External recurrence boundary) | Rationale 5.3; compact crosswalk | Engineering normative requirements | D004,D005 | FRUIT |
| `SCI-PTC-REQ-052` | `src/common/requirements.tex` (Coefficient taxonomy) | Rationale 6.1; compact crosswalk | Engineering normative requirements | D015 | SCI-MAP |
| `SCI-PTC-REQ-053` | `src/common/requirements.tex` (Fitted-loading semantics) | Rationale 6.1; compact crosswalk | Engineering normative requirements | D015 | None |
| `SCI-PTC-REQ-054` | `src/common/requirements.tex` (Analysis/gridding coefficient) | Rationale 6.1; compact crosswalk | Engineering normative requirements | D015 | SCI-MAP |
| `SCI-PTC-REQ-055` | `src/common/requirements.tex` (Coefficient re-estimation) | Rationale 6.1; compact crosswalk | Engineering normative requirements | D015 | SCI-MAP |
| `SCI-PTC-REQ-056` | `src/common/requirements.tex` (Empirical statistic limit) | Rationale 6.2; compact crosswalk | Engineering normative requirements | D015 | NOI |
| `SCI-PTC-REQ-057` | `src/common/requirements.tex` (Uncertainty taxonomy) | Rationale 6.2; compact crosswalk | Engineering normative requirements | D014,D015 | SCI-CAL,NOI |
| `SCI-PTC-REQ-058` | `src/common/requirements.tex` (Conditional covariance) | Rationale 6.2; compact crosswalk | Engineering normative requirements | D014 | NOI |
| `SCI-PTC-REQ-059` | `src/common/requirements.tex` (Selection uncertainty) | Rationale 6.2; compact crosswalk | Engineering normative requirements | D012,D014 | NOI |
| `SCI-PTC-REQ-060` | `src/common/requirements.tex` (Cross-observation covariance) | Rationale 6.2; compact crosswalk | Engineering normative requirements | D014 | SCI-CAL,NOI |
| `SCI-PTC-REQ-061` | `src/common/requirements.tex` (Admitted upstream response) | Rationale 7.1; compact crosswalk | Engineering normative requirements | D003,D013 | SCI-RTC,SCI-CAL |
| `SCI-PTC-REQ-062` | `src/common/requirements.tex` (Fixed-state companion) | Rationale 7.1; compact crosswalk | Engineering normative requirements | D013,Q002 | SCI-RTC,SCI-CAL |
| `SCI-PTC-REQ-063` | `src/common/requirements.tex` (Full-procedure response) | Rationale 7.2; compact crosswalk | Engineering normative requirements | D013 | None |
| `SCI-PTC-REQ-064` | `src/common/requirements.tex` (Response family) | Rationale 7.2; compact crosswalk | Engineering normative requirements | D013 | None |
| `SCI-PTC-REQ-065` | `src/common/requirements.tex` (Package response boundary) | Rationale 7.2; compact crosswalk | Engineering normative requirements | D013 | SCI-RTC,SCI-CAL |
| `SCI-PTC-REQ-066` | `src/common/requirements.tex` (Response status) | Rationale 7.1--7.3 and 8.3; compact crosswalk | Engineering normative requirements | D013,D016 | None |
| `SCI-PTC-REQ-067` | `src/common/requirements.tex` (Map-center diagnostic) | Rationale 7.3; compact crosswalk | Engineering normative requirements | D016 | SCI-MAP,BEAM |
| `SCI-PTC-REQ-068` | `src/common/requirements.tex` (No response overclaim) | Rationale 7.3; compact crosswalk | Engineering normative requirements | D002,D016 | SCI-MAP,BEAM |
| `SCI-PTC-REQ-069` | `src/common/requirements.tex` (Transformed intermediate) | Rationale 8.1; compact crosswalk | Engineering normative requirements | D005 | SCI-MAP |
| `SCI-PTC-REQ-070` | `src/common/requirements.tex` (Persisted role) | Rationale 8.1; compact crosswalk | Engineering normative requirements | D005 | None |
| `SCI-PTC-REQ-071` | `src/common/requirements.tex` (Material provenance) | Rationale 8.1; compact crosswalk | Engineering normative requirements | D005 | None |
| `SCI-PTC-REQ-072` | `src/common/requirements.tex` (Atomic required outputs) | Rationale 8.1; compact crosswalk | Engineering normative requirements | D005 | None |
| `SCI-PTC-REQ-073` | `src/common/requirements.tex` (Typed fallback) | Rationale 8.2; compact crosswalk | Engineering normative requirements | D003,D007 | None |
| `SCI-PTC-REQ-074` | `src/common/requirements.tex` (Insufficient support) | Rationale 8.2; compact crosswalk | Engineering normative requirements | D007,D012 | None |
| `SCI-PTC-REQ-075` | `src/common/requirements.tex` (Random state) | Rationale 8.2; compact crosswalk | Engineering normative requirements | D006 | None |
| `SCI-PTC-REQ-076` | `src/common/requirements.tex` (Disabled PTC route) | Rationale 8.3; compact crosswalk | Engineering normative requirements | D005 | SCI-MAP |
| `SCI-PTC-REQ-077` | `src/common/requirements.tex` (Direct CAL-to-MAP separation) | Rationale 8.3; compact crosswalk | Engineering normative requirements | D005 | SCI-MAP |
| `SCI-PTC-REQ-078` | `src/common/requirements.tex` (Excluded signal roles) | Rationale 8.3; compact crosswalk | Engineering normative requirements | D002,D008 | SCI-RTC,BEAM |
| `SCI-PTC-REQ-079` | `src/common/requirements.tex` (Ownership preservation) | Rationale 8.3; compact crosswalk | Engineering normative requirements | D004,D005 | Adjacent packages |
| `SCI-PTC-REQ-080` | `src/common/requirements.tex` (Evidence-layer separation) | Rationale 9.2; compact crosswalk | Engineering normative requirements | D006 | None |
| `SCI-PTC-REQ-081` | `src/common/requirements.tex` (External source/residual parent) | Rationale 3.3; compact crosswalk | Engineering normative requirements | D004,D014 | AST,FRUIT |
| `SCI-PTC-REQ-082` | `src/common/requirements.tex` (Shifted/null surrogate) | Rationale 8.2; compact crosswalk | Engineering normative requirements | D006,D007 | VAL shared semantics,NOI |
| `SCI-PTC-REQ-083` | `src/common/requirements.tex` (Exact frozen application map) | Rationale 4.2; compact crosswalk | Engineering normative requirements | D010,Q002 | None |
| `SCI-PTC-REQ-084` | `src/common/requirements.tex` (Candidate evidence retention) | Rationale 4.1 and 4.3; compact crosswalk | Engineering normative requirements | D012 | Owner policy |
| `SCI-PTC-REQ-085` | `src/common/requirements.tex` (Typed astronomical-transfer predicate) | Rationale 4.3 and 7.2; compact crosswalk | Engineering normative requirements | D012,D013 | Owner policy |
| `SCI-PTC-REQ-086` | `src/common/requirements.tex` (Expectation and bias statement) | Rationale 6.2; compact crosswalk | Engineering normative requirements | D010,D014 | NOI |
| `SCI-PTC-REQ-087` | `src/common/requirements.tex` (Response-domain chain) | Rationale 7.1; compact crosswalk | Engineering normative requirements | D013 | SCI-CAL |
| `SCI-PTC-REQ-088` | `src/common/requirements.tex` (Independent state axes) | Rationale 8.3; compact crosswalk | Engineering normative requirements | D005,D013 | None |
| `SCI-PTC-REQ-089` | `src/common/requirements.tex` (Fit-excluded application availability) | Rationale 4.2; compact crosswalk | Engineering normative requirements | D007,D010 | None |
| `SCI-PTC-PRED-001` | `src/common/edge_cases.tex` (Disabled route identity) | Rationale 8.3; compact crosswalk | Engineering falsifiable predictions | D005 | SCI-MAP |
| `SCI-PTC-PRED-002` | `src/common/edge_cases.tex` (Same-unit response loss) | Rationale 1.3; compact crosswalk | Engineering falsifiable predictions | D002,D014 | SCI-CAL |
| `SCI-PTC-PRED-003` | `src/common/edge_cases.tex` (Correlation non-identification) | Rationale 1.2; compact crosswalk | Engineering falsifiable predictions | D004,D014 | None |
| `SCI-PTC-PRED-004` | `src/common/edge_cases.tex` (Gauge invariance) | Rationale 3.1; compact crosswalk | Engineering falsifiable predictions | D014,D015 | None |
| `SCI-PTC-PRED-005` | `src/common/edge_cases.tex` (Centering null mode) | Rationale 3.1 and 7.2; compact crosswalk | Engineering falsifiable predictions | D017 | None |
| `SCI-PTC-PRED-006` | `src/common/edge_cases.tex` (Scaling restoration) | Rationale 3.1; compact crosswalk | Engineering falsifiable predictions | D017 | None |
| `SCI-PTC-PRED-007` | `src/common/edge_cases.tex` (Zero-fill is not exclusion) | Rationale 2.1; compact crosswalk | Engineering falsifiable predictions | D007,D010 | None |
| `SCI-PTC-PRED-008` | `src/common/edge_cases.tex` (Fit-excluded apply-allowed) | Rationale 2.1 and 4.2; compact crosswalk | Engineering falsifiable predictions | D007 | None |
| `SCI-PTC-PRED-009` | `src/common/edge_cases.tex` (Support-stage independence) | Rationale 2.1; compact crosswalk | Engineering falsifiable predictions | D007 | VAL |
| `SCI-PTC-PRED-010` | `src/common/edge_cases.tex` (Influence through fitting) | Rationale 2.2; compact crosswalk | Engineering falsifiable predictions | D007 | VAL |
| `SCI-PTC-PRED-011` | `src/common/edge_cases.tex` (Source-mask boundary) | Rationale 3.3; compact crosswalk | Engineering falsifiable predictions | D004,D014 | AST,ALIGN |
| `SCI-PTC-PRED-012` | `src/common/edge_cases.tex` (Within-array isolation) | Rationale 3.2; compact crosswalk | Engineering falsifiable predictions | D011 | None |
| `SCI-PTC-PRED-013` | `src/common/edge_cases.tex` (Hierarchy order materiality) | Rationale 3.2; compact crosswalk | Engineering falsifiable predictions | D011 | None |
| `SCI-PTC-PRED-014` | `src/common/edge_cases.tex` (Common-mode denominator failure) | Rationale 4.1; compact crosswalk | Engineering falsifiable predictions | D010 | None |
| `SCI-PTC-PRED-015` | `src/common/edge_cases.tex` (Fixed-template orthogonality) | Rationale 4.1; compact crosswalk | Engineering falsifiable predictions | D010 | None |
| `SCI-PTC-PRED-016` | `src/common/edge_cases.tex` (Rank-deficient gauge) | Rationale 4.1; compact crosswalk | Engineering falsifiable predictions | D010,D012 | None |
| `SCI-PTC-PRED-017` | `src/common/edge_cases.tex` (Least-aggressive selection) | Rationale 4.3; compact crosswalk | Engineering falsifiable predictions | D012 | Owner policy |
| `SCI-PTC-PRED-018` | `src/common/edge_cases.tex` (No compensating metric) | Rationale 4.3; compact crosswalk | Engineering falsifiable predictions | D012 | Owner policy |
| `SCI-PTC-PRED-019` | `src/common/edge_cases.tex` (Deterministic tie) | Rationale 4.3; compact crosswalk | Engineering falsifiable predictions | D012 | Owner policy |
| `SCI-PTC-PRED-020` | `src/common/edge_cases.tex` (Empty candidate set) | Rationale 4.3; compact crosswalk | Engineering falsifiable predictions | D012 | Owner policy |
| `SCI-PTC-PRED-021` | `src/common/edge_cases.tex` (Immutable-parent refinement) | Rationale 5.2; compact crosswalk | Engineering falsifiable predictions | D009 | SCI-CAL |
| `SCI-PTC-PRED-022` | `src/common/edge_cases.tex` (Output-only classification) | Rationale 5.2; compact crosswalk | Engineering falsifiable predictions | D009 | VAL |
| `SCI-PTC-PRED-023` | `src/common/edge_cases.tex` (Refinement oscillation) | Rationale 5.2; compact crosswalk | Engineering falsifiable predictions | D009 | Owner policy |
| `SCI-PTC-PRED-024` | `src/common/edge_cases.tex` (Diagnostic population dependence) | Rationale 5.1; compact crosswalk | Engineering falsifiable predictions | D009 | Owner policy |
| `SCI-PTC-PRED-025` | `src/common/edge_cases.tex` (Diagnostic-only $r$ inertia) | Rationale 4.2; compact crosswalk | Engineering falsifiable predictions | D008,Q001 | SCI-RTC |
| `SCI-PTC-PRED-026` | `src/common/edge_cases.tex` (Raw-$r$ incompatibility) | Rationale 4.2; compact crosswalk | Engineering falsifiable predictions | D008 | SCI-RTC |
| `SCI-PTC-PRED-027` | `src/common/edge_cases.tex` (Coefficient gauge) | Rationale 6.1; compact crosswalk | Engineering falsifiable predictions | D015 | None |
| `SCI-PTC-PRED-028` | `src/common/edge_cases.tex` (Coefficient normalization) | Rationale 6.1; compact crosswalk | Engineering falsifiable predictions | D015 | SCI-MAP |
| `SCI-PTC-PRED-029` | `src/common/edge_cases.tex` (Conditional covariance) | Rationale 6.2; compact crosswalk | Engineering falsifiable predictions | D014 | NOI |
| `SCI-PTC-PRED-030` | `src/common/edge_cases.tex` (Selection variance term) | Rationale 6.2; compact crosswalk | Engineering falsifiable predictions | D012,D014 | NOI |
| `SCI-PTC-PRED-031` | `src/common/edge_cases.tex` (Fixed companion noninterference) | Rationale 7.1; compact crosswalk | Engineering falsifiable predictions | D013 | None |
| `SCI-PTC-PRED-032` | `src/common/edge_cases.tex` (Full-procedure nonlinearity) | Rationale 7.2; compact crosswalk | Engineering falsifiable predictions | D013 | None |
| `SCI-PTC-PRED-033` | `src/common/edge_cases.tex` (Package response boundary) | Rationale 7.2; compact crosswalk | Engineering falsifiable predictions | D013 | SCI-RTC,SCI-CAL |
| `SCI-PTC-PRED-034` | `src/common/edge_cases.tex` (Map-center factor availability) | Rationale 7.3; compact crosswalk | Engineering falsifiable predictions | D016 | SCI-MAP,BEAM |
| `SCI-PTC-PRED-035` | `src/common/edge_cases.tex` (Compact versus extended response) | Rationale 7.3; compact crosswalk | Engineering falsifiable predictions | D016 | BEAM |
| `SCI-PTC-PRED-036` | `src/common/edge_cases.tex` (Random-state reproducibility) | Rationale 8.2; compact crosswalk | Engineering falsifiable predictions | D006 | None |
| `SCI-PTC-PRED-037` | `src/common/edge_cases.tex` (Required-output atomicity) | Rationale 8.1; compact crosswalk | Engineering falsifiable predictions | D005 | None |
| `SCI-PTC-PRED-038` | `src/common/edge_cases.tex` (Evidence-layer separation) | Rationale 9.2; compact crosswalk | Engineering falsifiable predictions | D006 | None |
| `SCI-PTC-PRED-039` | `src/common/edge_cases.tex` (Frozen component versus projection) | Rationale 4.2; compact crosswalk | Engineering falsifiable predictions | D010,Q002 | None |
| `SCI-PTC-PRED-040` | `src/common/edge_cases.tex` (Temporal versus detector action) | Rationale 4.2; compact crosswalk | Engineering falsifiable predictions | D010,Q002 | None |
| `SCI-PTC-PRED-041` | `src/common/edge_cases.tex` (No double upstream response) | Rationale 7.1; compact crosswalk | Engineering falsifiable predictions | D013 | SCI-CAL |
| `SCI-PTC-PRED-042` | `src/common/edge_cases.tex` (Typed procedure state change) | Rationale 7.2; compact crosswalk | Engineering falsifiable predictions | D013 | None |
| `SCI-PTC-PRED-043` | `src/common/edge_cases.tex` (Candidate-evidence reconstruction) | Rationale 4.1 and 4.3; compact crosswalk | Engineering falsifiable predictions | D012 | Owner policy |
| `SCI-PTC-PRED-044` | `src/common/edge_cases.tex` (External-parent fail-closed) | Rationale 3.3; compact crosswalk | Engineering falsifiable predictions | D004,D014 | FRUIT |
| `SCI-PTC-PRED-045` | `src/common/edge_cases.tex` (Surrogate support transformation) | Rationale 8.2; compact crosswalk | Engineering falsifiable predictions | D006,D007 | VAL shared semantics,NOI |
| `SCI-PTC-PRED-046` | `src/common/edge_cases.tex` (Astronomical-predicate typing) | Rationale 4.3; compact crosswalk | Engineering falsifiable predictions | D012,D013 | Owner policy |
| `SCI-PTC-PRED-047` | `src/common/edge_cases.tex` (Unavailable bias claim) | Rationale 6.2; compact crosswalk | Engineering falsifiable predictions | D010,D014 | NOI |
| `SCI-PTC-PRED-048` | `src/common/edge_cases.tex` (Independent product and response states) | Rationale 8.3; compact crosswalk | Engineering falsifiable predictions | D005,D013 | None |
| `SCI-PTC-PRED-049` | `src/common/edge_cases.tex` (Undefined fit-excluded extension) | Rationale 4.2; compact crosswalk | Engineering falsifiable predictions | D007,D010 | None |
| `SCI-PTC-PRED-050` | `src/common/edge_cases.tex` (Conservative support composition) | Rationale 2.1; compact crosswalk | Engineering falsifiable predictions | D007 | PTC and named-use owners |
