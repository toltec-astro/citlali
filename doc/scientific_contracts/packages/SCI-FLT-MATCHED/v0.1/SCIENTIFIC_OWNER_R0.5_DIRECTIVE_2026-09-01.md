SCI-FLT-MATCHED v0.1 r0.5 — FINAL TARGETED
TYPE, LIFECYCLE, COVARIANCE-ROLE, AND OWNER-DISPOSITION DIRECTIVE

The SCI-FLT-MATCHED r0.4 scientific architecture is accepted as the basis for
the next targeted revision.

Do not perform another broad matched-template derivation.

Preserve:

- SCI-FLT-MATCHED as distinct from SCI-FLT-FIXED;
- the normalized fixed-template amplitude estimator;
- n_p = t_p^T W_p m_p, d_p = t_p^T W_p t_p, and a_hat_p = n_p/d_p;
- exact matching-template unit response;
- exact local restriction before covariance inversion;
- the constrained local GLS theorem and its bounded competitor class;
- self-adjoint positive-semidefinite scientific weighting;
- parent-pixel-center output anchors and no subpixel search;
- exact materialized, structured, or lineage/query template representations;
- separate application, template, state, response, and storage supports;
- complete-support-only estimation;
- Learn–Resolve–Apply and immutable fixed state;
- observation and coadd applications as distinct methods/generations;
- fixed-state, full-procedure, operational-realized, and exact-reference
  response types;
- operational-realized covariance distinct from exact-reference covariance;
- the U1–U7 uncertainty separation;
- AO-001 as the principal scientific weighting family;
- AO-002 numerical conformance rather than package-wide scientific tolerance;
- AO-003 scope separate from covariance representation;
- AO-004/AO-005 query science separate from representation;
- seven role-separated PA/SA/SP/CU/NU/RU/FH semantics;
- the finite FLT-to-FRUIT producer envelope without FRUIT science;
- no candidate selection, deblending, source existence, catalog, significance,
  posterior/Wiener reconstruction, generic-convolution, or production claim;
- all stable requirement and prediction IDs.

Do not inspect implementation, configuration, schemas, tests, audits, repairs,
validation products, reductions, generated products, defaults, historical
behavior, or production status.

Do not renumber SCI-FLT-MATCHED-REQ-001 through REQ-050 or
SCI-FLT-MATCHED-PRED-001 through PRED-024. Amend existing draft wording where
the same obligation is being corrected. Append IDs only for genuinely new
scientific obligations.

======================================================================
1. TYPE THE PARENT FACT DOMAIN AND NUMERICAL APPLICATION DOMAIN
======================================================================

The current equations form m_p = E_p m over D_loc(p), while exact-zero final
application coefficients are said not to require or dereference their parent
payloads.

Define separately:

    S_parent_fact
      = exact parent row-identity and fact domain;

    D_m
      = {q in S_parent_fact:
           an admitted finite real parent signal payload exists at q};

    D_loc(p)
      = predeclared template/covariance/state-construction domain;

    ell_p^star
      = W_p t_p / d_p;

    c_p
      = E_p^T ell_p^star;

    S_apply(p)
      = {q : c_pq != 0}.

Evaluate the scientific amplitude only as:

    a_hat_p
      = sum_{q in S_apply(p)} c_pq m_q.

Every q in S_apply(p) shall belong to D_m and pass the exact parent-row
admission and support predicates.

Coordinates in D_loc(p) outside S_apply(p) may influence template, covariance,
rank, normalization, or learned-state construction but shall not require or
dereference a signal payload merely because they occur in a dense
representation.

Exact zero shall use the canonical scientific coefficient representation, not
a floating threshold.

The shorthand m_p = E_p m may be used only when the complete numerical vector
exists, or as an explicitly defined algebraic completion whose values outside
S_apply(p) provably cannot affect any scientific or numerical result.

Update:

- notation;
- Equations 1–4;
- support definitions;
- REQ-007, REQ-008, REQ-010, REQ-042;
- PRED-004;
- validity and edge tables;
- ECS operator and support tests.

Add a prediction in which a D_loc coordinate has no numerical parent payload
but has an exact-zero final application coefficient. The amplitude shall remain
defined and unchanged. Making that coefficient any exact nonzero value shall
activate the dependency and make the anchor unavailable.

======================================================================
2. SEPARATE SIGNAL AND COMPANION-QUALIFIED VALIDITY DOMAINS
======================================================================

Do not use one undifferentiated V whose wording can make optional response or
covariance availability a base-signal predicate.

Define:

    V_signal
      = anchors satisfying the exact parent, template, application-support,
        operator, normalization, unit/calibration, signal-publication, and
        selected weighting-route predicates;

    V_response(r)
      = subset of V_signal satisfying the exact named response-role r
        availability and policy;

    V_covariance(r)
      = subset of V_signal satisfying the exact named covariance-role r
        authority, codomain, representation, and policy;

    V_NOI
      = subset of V_signal satisfying the exact K_NOI and NOI-use policy;

    V_FRUIT_handoff
      = subset or bundle role satisfying the finite Q_FLT handoff policy.

A base signal bundle shall carry the exact identity, lifecycle status, cause,
and provenance of every response and covariance role, but those role states may
be unavailable unless the requested signal named use requires them.

An unavailable optional companion shall not remove an anchor from V_signal or
make the signal bundle partial.

A response-qualified or covariance-qualified named use may be narrower and
fail closed.

Update:

- the definition of V;
- product roles;
- validity table;
- REQ-020, REQ-023–025, REQ-045–049;
- the seven-role policy dependency graph;
- route-status and ECS evidence records.

======================================================================
3. COMPLETE THE LIFECYCLE VOCABULARY
======================================================================

The text currently says learned-candidate and published identities are distinct,
but the lifecycle table does not define those states.

Use one complete lifecycle such as:

    not_requested
    requested
    effective
    disabled
    unavailable
    learned_candidate
    resolved
    applied
    complete_publication_candidate
    publication_decided
    realized
    published
    failed
    not_produced
    superseded.

Define:

- learned_candidate:
    one immutable Learn output with diagnostics and uncertainty, not yet
    selected for application;

- resolved:
    one owner-authorized frozen state selected for Apply;

- realized:
    the producing map generated immutable numerical values and complete
    producer facts;

- published:
    the authorized publication action exposed the complete product under its
    exact public role;

- publication_decided:
    the exact named-use decision exists even when the outcome is not_produced.

If realized and published are intentionally one state, remove every claim that
they are distinct. Do not leave an undefined published generation.

Update the state graph, REQ-015, REQ-026, product atomicity, SCI-VAL axes,
failure table, and ECS lifecycle tests.

======================================================================
4. KEEP STOCHASTIC CONDITIONING PRE-OUTCOME
======================================================================

The present g includes applied and realized identities while h = (g, theta)
conditions every fixed-state expectation and covariance statement.

Define a pre-outcome frozen condition:

    h_pre = (g_resolved, theta),

where g_resolved binds every fact fixed before evaluation of a parent draw:

- parent and parent class;
- template;
- D_loc and extraction;
- selected weighting state;
- support and missingness rule;
- subspace/rank/null;
- regularization;
- numerical profile;
- fixed covariance selector P_C;
- fixed response-query domain;
- failure semantics.

Execution-attempt identity, realized-product identity, publication identity,
success/failure outcome, and observed values remain immutable provenance but
shall not be conditioning variables in the U1 law.

Use h_pre consistently in:

- zero-mean assumptions;
- C_parent|h;
- reference covariance;
- operational covariance;
- GLS variance;
- fixed-state response.

If a later method intentionally conditions on a realized outcome, it requires a
separately named population, selection model, and covariance role.

Retain the existing prohibition on successful-production conditioning,
censoring, pairwise deletion, and draw-dependent domains.

======================================================================
5. SEPARATE GLS REFERENCE MARGINAL VARIANCE FROM OPERATIONAL COVARIANCE
======================================================================

Define separately:

    v_GLS,reference(p)
      = d_p^{-1}

only under every AO-001-A constrained local GLS premise;

    C_U1,reference
      = P_C L C_parent|h_pre L^T P_C^T;

    C_U1,realized
      = Cov[P_C F_g(m) | h_pre].

AO-003 selects the scientific scope and representation of a named operational
covariance role. It shall not erase a theorem-supported reference marginal
variance.

Add an exact AO-001/AO-003 compatibility table:

- AO-001-A plus AO-003-C:
    v_GLS,reference may remain available;
    operational covariance is unavailable;

- AO-001-A plus projected or complete AO-003 scope:
    retain reference marginal variance and the separately named operational
    covariance object;

- AO-001-C or another non-GLS weight:
    d_p is normalization only under every AO-003 state;

- missing or mismatched parent covariance authority:
    no GLS reference variance and no operational covariance.

Revise AO-003-C. It may state that no operational covariance, covariance-based
standardization, draw, independence, or covariance-dependent use follows. It
shall not say that no variance of any kind exists when AO-001-A separately
establishes v_GLS,reference.

Update REQ-012, REQ-020–022, REQ-045–046, PRED-008–009, the uncertainty table,
and ECS covariance evidence.

======================================================================
6. CLARIFY THE FIXED-TEMPLATE SOURCE BOUNDARY
======================================================================

Replace the broad prohibition that the template may not be learned from a
“source.”

Use:

    The template may be supplied by an exact owner-authorized source-class,
    morphology, beam, or physical-response authority.

    It shall not be learned, selected, centered, tuned, updated, or
    reparameterized from:

    - the numerical payload of the target MAP parent;
    - the candidate or peak being evaluated;
    - the resulting amplitude field;
    - target-derived residuals;
    - a NOI realization member;
    - or another post-selection product.

A predeclared bank of source-class templates is allowed only when the exact
template-selection rule is independent of target-map outcomes. Selecting among
templates after inspecting the target is a different inference method.

Update REQ-005–006, ASM-003, template validity, and template-learning failure
fixtures.

======================================================================
7. OWNER-DISPOSITION THE WORD “OPTIMAL”
======================================================================

Preferred disposition:

    Retain package identity SCI-FLT-MATCHED.

    Change the human-readable title to:

        Matched-template map amplitude estimation

    or:

        Normalized matched-template map filtering.

    Reserve “optimal” for a realization satisfying AO-001-A and every theorem
    premise.

If the owner retains the historical title “Optimal matched-template map
filtering,” require every public product and boundary to carry:

    optimality_status =
        established_exact_local_GLS
        | not_claimed
        | unavailable.

The package title, familiar filename, or SCI-FLT-MATCHED identity shall never
be evidence of optimality.

AO-001-C and every weaker weighting route shall always carry
optimality_status = not_claimed.

Update REQ-001, titles/covers, route status, product vocabulary, and metadata
accordingly.

======================================================================
8. MAKE DOWNSTREAM BOUNDARY RECORDS REQUEST-QUALIFIED
======================================================================

Every signal product shall carry exact MAP and template parent boundaries.

NOI and FLT-to-FRUIT records shall carry exact named-use state, which may be:

    not_requested
    inapplicable
    unavailable
    compatible
    requested_pending
    realized_child_exists

under exact lifecycle and ownership.

A base signal product shall not require an active NOI or FRUIT child.

No later child may mutate the FLT parent.

REQ-038 shall require exact boundary identity/status records, not successful
realization of every downstream use.

Apply the same rule to PA, SA, SP, CU, NU, RU, and FH: each role remains
separately addressable, but a not-requested optional role shall not block an
unrelated base signal named use.

======================================================================
9. CONSOLIDATE SOURCE AND AUTHORITY CLOSURE
======================================================================

Create one r0.5 authority manifest binding:

- the original eight-object author packet;
- the r0.2 closure directive;
- the r0.3 review-repair owner authority;
- the r0.4 covariance-scope and AO-001-C diagnostic-policy dispositions;
- this r0.5 directive and every resulting owner disposition;
- exact shared normative modules;
- scientific rationale and ECS sources;
- MAP-to-FLT-MATCHED boundary;
- template-to-FLT-MATCHED boundary;
- FLT-MATCHED-to-NOI boundary;
- FLT-MATCHED-to-FRUIT producer-envelope boundary;
- all seven role policy/profile drafts;
- AO decision ledger;
- route-status record;
- source/view parity and build records;
- rendered PDFs.

Record path, version, byte count, SHA-256, authority role, approval state,
compatibility/supersession relation, and generated-view relation.

The independent review may remain process provenance. Scientific authority
shall be the exact owner disposition/directive, not a model-product name.

Return the actual source bytes and reproduce every quoted digest.

======================================================================
10. OWNER-DISPOSITION GUIDE
======================================================================

Present the remaining scientific choices separately.

Recommended dispositions:

AO-001-A:
    authorize as the sole route allowed to claim exact local-GLS optimality and
    d_p^{-1} reference marginal conditional variance, when every theorem premise
    exists.

AO-001-B:
    retain as successor-authorship trigger until one concrete structured
    covariance-derived W_p is authored.

AO-001-C:
    permit as a separately named nonoptimal
    radially_symmetrized_field_power_spectral_weighting route;
    mandatory diagnostics remain evidence and not v0.1 validity gates;
    no noise, covariance, stationarity, isotropy, optimality, or d_p^{-1}
    variance claim.

AO-001-D:
    retain as successor trigger rather than a selectable catch-all.

AO-002:
    exact-operator numerical realization or typed unavailable is an engineering
    preregistration result;
    an intentionally different operator is a separately authored method.

AO-003:
    select complete, named projected, or unavailable scope independently for
    each covariance role;
    select resident versus lineage representation afterward.

AO-004:
    scientific owner fixes the finite required state-query vocabulary;
    engineering chooses full, compact-exact, or lineage representation.

AO-005:
    scientific owner fixes response type, domain, query vocabulary, validity,
    and consumer scope;
    engineering chooses full, structured, or lineage representation.

AO-006:
    seven verdict meanings and dependency graph remain normative;
    separate/grouped/vector storage is engineering representation only.

Observation and coadd weighting choices remain separate decisions. No route is
an automatic fallback for another.

======================================================================
11. DELIVERABLES
======================================================================

Return:

1. revised Scientific Rationale and Contract r0.5;
2. revised Engineering Conformance Specification r0.5;
3. exact shared normative modules;
4. numerical parent-domain and application-row amendment;
5. signal/response/covariance/NOI validity-domain crosswalk;
6. complete lifecycle amendment;
7. pre-outcome conditioning amendment;
8. GLS-reference versus operational-covariance compatibility table;
9. fixed-template source-boundary amendment;
10. optimality-title owner disposition;
11. downstream boundary-status amendment;
12. AO owner-disposition packet;
13. complete authority manifest and source bytes;
14. equation/requirement/prediction semantic-change map;
15. rationale/ECS parity report;
16. PDF visual-QA and metadata report.

Preserve SCI-FLT-MATCHED-REQ-001 through REQ-050 and
SCI-FLT-MATCHED-PRED-001 through PRED-024. Use REQ-051/PRED-025 and later only
for genuinely new obligations.

Do not claim a numerical route, implementation conformity, response or
covariance fidelity, observational validation, detection performance,
readiness, scientific freeze, production suitability, production
authorization, or Unity activity.