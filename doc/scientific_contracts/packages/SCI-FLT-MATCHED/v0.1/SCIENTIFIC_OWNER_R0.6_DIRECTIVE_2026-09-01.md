SCI-FLT-MATCHED v0.1 r0.6 — FINAL MICRO-REPAIR,
OWNER-DISPOSITION, AND CONDITIONAL-FREEZE PREFLIGHT

The SCI-FLT-MATCHED r0.5 scientific architecture is accepted as the basis for
the freeze candidate.

Do not perform another broad matched-template derivation.

Preserve:

- SCI-FLT-MATCHED as distinct from SCI-FLT-FIXED;
- the fixed-template, fixed-anchor, one-parameter amplitude estimand;
- the parent-pixel-center anchor lattice;
- exact coordinate-basis weighting;
- t_p = E_p t_p_full;
- d_p = t_p^T W_p t_p;
- ell_p^star = W_p t_p / d_p;
- c_p = E_p^T ell_p^star;
- S_apply(p) = {q : c_pq != 0};
- exact sparse application over S_apply(p);
- exact-zero coefficients as nondependencies;
- self-adjoint positive-semidefinite W_p and exact positive d_p;
- local restriction before covariance inversion;
- the constrained local-GLS theorem and its bounded competitor class;
- observation and coadd separation;
- no filter/coadd commutation claim;
- fixed-state, full-procedure, operational-realized, and reference response;
- GLS reference variance separated from operational covariance;
- U1 through U7 uncertainty separation;
- separate signal, response, covariance, NOI, and FRUIT-handoff validity roles;
- Learn–Resolve–Apply with immutable resolved state;
- fixed-state NOI parity and no per-member relearning;
- request-qualified NOI and FRUIT boundary records;
- scientific scope separated from engineering representation;
- seven PA/SA/SP/CU/NU/RU/FH role meanings;
- the finite FLT-to-FRUIT producer envelope without FRUIT science;
- every existing exclusion and nonclaim.

Do not inspect implementation, configuration, schemas, tests, audits, repairs,
validation products, reductions, generated products, defaults, historical
behavior, or production state.

Preserve SCI-FLT-MATCHED-REQ-001 through REQ-050 and
SCI-FLT-MATCHED-PRED-001 through PRED-025. Amend an existing identity when its
existing obligation is merely being made type-correct. Use REQ-051/PRED-026
and later only for genuinely new normative obligations.

======================================================================
1. DISTINGUISH THE STOCHASTIC MODEL PARENT FROM THE OBSERVED PAYLOAD
======================================================================

The r0.5 contract defines observed m_q only on D_m but defines local covariance
and the GLS competitor vector on the complete D_loc(p), which may contain
coordinates outside D_m.

Introduce exact distinct objects.

Define:

    S_parent_fact
      = exact parent row-identity and fact domain;

    D_model
      = exact coordinate domain of the authorized parent stochastic model;

    M : D_model -> R
      = parent random vector under one exact declared stochastic law;

    C_parent|h_pre
      = Cov[M | h_pre];

    D_m
      = {q in S_parent_fact:
           an admitted finite real observed parent payload exists};

    m_obs : D_m -> R
      = immutable observed MAP payload.

Require for any AO-001-A local GLS route:

    D_loc(p) subset D_model;

    M_p = E_p M;

    C_p = Cov[M_p | h_pre]
        = E_p C_parent|h_pre E_p^T.

Require for actual amplitude application:

    S_apply(p) subset D_m;

    a_hat_p(m_obs)
      = sum_{q in S_apply(p)} c_pq m_obs_q.

A coordinate may belong to D_loc(p) and D_model, and may influence covariance,
rank, weighting, normalization, or operator construction, while lacking an
observed payload, only when its exact final application coefficient is zero.

This introduces no imputation and no partial-support estimator.

If the stochastic model or covariance authority does not cover any coordinate
required by D_loc(p), AO-001-A, its optimality claim, and
v_GLS,reference=d_p^-1 are unavailable, even when a separately authorized
non-GLS sparse amplitude might remain computable.

Use the stochastic random vector M for:

- zero-mean assumptions;
- parent-law covariance;
- local GLS theorem;
- reference covariance;
- stochastic expectations.

Use m_obs for:

- one exact numerical parent realization;
- exact sparse application;
- implementation-facing payload availability.

Where L_p,g u is used on a partially available numerical object, define it
through the exact sparse contraction:

    L_p,g u
      = sum_{q in S_apply(p)} c_pq u_q.

A dense c_p^T u shorthand is authorized only when u is defined on the complete
parent-coordinate domain or an exact algebraic completion is proved
irrelevant.

Update:

- notation and symbol table;
- coordinate-basis covariance convention;
- Equations 1 through 11 as applicable;
- general-sky relations;
- constrained GLS theorem;
- fixed/full-procedure response;
- REQ-007, REQ-008, REQ-010, REQ-020 through REQ-022, REQ-041,
  REQ-042, and REQ-045;
- ASM-002, ASM-004, ASM-007 through ASM-009;
- PRED-003, PRED-008, PRED-024, and PRED-025;
- validity and edge-case tables;
- ECS CT-004, CT-005, CT-008, and CT-010.

Add a prediction, if not already fully represented, that:

- an exact-zero application coordinate with no observed m_obs payload can
  remain valid when all template/model/covariance construction authority
  exists;
- missing covariance/model authority on a required D_loc coordinate makes
  AO-001-A unavailable;
- changing that exact-zero coefficient to nonzero activates the observed
  payload dependency and makes the anchor unavailable when the payload is
  absent.

======================================================================
2. KEEP h_pre FREE OF REALIZED PAYLOAD VALUES
======================================================================

Retain:

    h_pre = (g_resolved, theta).

Clarify that g_resolved may bind:

- parent product role, identity, and class;
- the exact stochastic-law/population identity;
- template;
- D_loc and extraction;
- weighting state;
- support and missingness rules;
- subspaces, rank, null, and regularization;
- numerical profile;
- fixed P_C;
- response-query domain;
- failure semantics.

It shall not condition U1 statements on:

- the numerical values of m_obs;
- an observed parent-payload digest that uniquely fixes the random outcome;
- execution success;
- a realized product identity whose existence depends on the draw;
- publication;
- censoring;
- pairwise deletion;
- or a draw-dependent domain.

Parent identity in h_pre means the predeclared product role, population, and
experimental conditioning identity, not conditioning on the realized random
payload itself.

A method that conditions on observed target payload requires a separately
named conditional population and covariance role.

======================================================================
3. REPAIR THE PRODUCT LIFECYCLE ORDER
======================================================================

The r0.5 vocabulary is complete but its order places a complete publication
candidate and publication decision before numerical realization, even though
the candidate contains the amplitude field and “realized” means that the
producing map generated immutable values.

Use this order for a successful product route:

    not_requested
      or
    requested
      -> effective
      -> learned_candidate, when Learn applies
      -> resolved
      -> applied
      -> realized
      -> complete_publication_candidate
      -> publication_decided
      -> published | not_produced.

Disabled and unavailable are pre-application dispositions where applicable.

Failed may branch from:

- Learn;
- Resolve;
- Apply;
- numerical realization;
- product closure;
- publication evaluation;
- publication action.

Define:

    applied
      = the resolved producing operation was invoked;

    realized
      = immutable numerical values and execution-outcome provenance were
        generated;

    complete_publication_candidate
      = every required realized value, identity, status, validity fact,
        companion identity/status record, lifecycle member, and provenance
        member exists and is ready for publication-policy evaluation;

    publication_decided
      = the exact owner-policy/SCI-VAL decision artifact exists;

    published
      = the accepted immutable candidate was exposed under its declared
        science-product role;

    not_produced
      = a complete disposition intentionally created no public product.

Do not define complete_publication_candidate as already publication-eligible
before the publication policy is evaluated. It is complete and eligible for
policy evaluation.

SCI-VAL decision-artifact realization remains separate from FLT product
realization.

Update:

- lifecycle amendment;
- state graph;
- product atomicity;
- REQ-015, REQ-024, and REQ-026;
- SP role semantics;
- four-axis examples;
- edge/failure tables;
- ECS CT-011 and CT-012;
- route status and semantic-change map.

======================================================================
4. CLARIFY AO-001 AUTHORIZATION MULTIPLICITY
======================================================================

The package may authorize more than one separately named AO-001 method route.

In particular, the owner may authorize both:

    AO-001-A
      exact constrained local inverse-covariance GLS;

and:

    AO-001-C
      radially_symmetrized_field_power_spectral_weighting.

This does not allow one realization to mix or choose between them.

Require:

- every request and realization binds exactly one AO-001 method identity;
- observation and coadd authorizations and generations remain separate;
- there is no automatic selector;
- there is no target-data-driven route choice;
- there is no fallback from A to C or C to A;
- a failed or unavailable requested route remains failed or unavailable;
- comparing or authorizing two routes does not imply their estimands,
  covariance claims, or optimality meanings are identical.

Revise SODL-001 so that it asks which route identities are package-authorized
for each parent class and named use, followed by which one exact route is
requested for a realization. Do not frame package authorization as necessarily
one global A-versus-C choice.

Retain:

- AO-001-A as the sole route eligible for
  established_exact_local_GLS and v_GLS,reference=d_p^-1;
- AO-001-C as nonoptimal and carrying no implied noise, covariance,
  stationarity, isotropy, or d_p^-1 variance claim;
- AO-001-B and AO-001-D as successor-authorship triggers.

======================================================================
5. PRESENT THE TITLE FOR EXPLICIT OWNER DISPOSITION
======================================================================

Do not silently infer a title choice.

Retain the recommendation:

    Matched-template map amplitude estimation.

Reserve the word “optimal” for a realization satisfying AO-001-A and every
local constrained-GLS theorem premise.

Every product and boundary retains exactly one:

    optimality_status =
        established_exact_local_GLS
        | not_claimed
        | unavailable.

If the scientific owner selects the recommended title, update all covers,
metadata, navigation, manifest records, and human-readable references while
retaining method identity SCI-FLT-MATCHED.

If the historical title is retained, continue to state prominently that it is
not realization-level optimality evidence.

======================================================================
6. LABEL ROLE-PROFILE DRAFTS HONESTLY
======================================================================

The current files under role_profiles/ describe role semantics but are not
complete owner-approved or Registry-bound SCI-VAL profiles.

Either rename the directory and records to role_semantics/, or place this
header in every file:

    Status: role-semantics draft only
    Scientific owner approval: pending
    SCI-VAL Registry state: unregistered
    SCI-VAL evaluability: unavailable
    Authority effect: none
    Missing policy fields are not inferred

Do not register or call these records executable profiles until each exact
record binds:

- profile ID and revision;
- scientific-policy owner;
- source authority and version;
- object/domain;
- request, applicability, eligibility, and realization semantics;
- restrictions;
- exceptions;
- missing/conflict behavior;
- aggregation/propagation compatibility;
- prescribed consumer action;
- failure scope;
- lifecycle and provenance.

Do not expand missing policies merely to make this draft look complete.

======================================================================
7. REPAIR PACKAGE STATUS AND STANDALONE-BUNDLE CLOSURE
======================================================================

Replace the README statement:

    Prior-work recovery and owner decisions: complete

with:

    Stage A recovery and author-input decisions: complete.
    Stage B scientific-owner dispositions remain open.

The standalone bundle currently contains package-local links to files that are
not present, including:

- PRIOR_WORK.md;
- INTERNAL_DOSSIER.md;
- SEMANTIC_CHANGE_MAP_R0.3.md;
- SEMANTIC_CHANGE_MAP_R0.4.md;
- NOTATION_CROSSWALK_R0.4.md;
- REPRESENTATION_CROSSWALK_R0.4.md;
- ROUTE_STATUS_R0.4.md;
- OWNER_DECISION_PARITY_R0.4.md;
- STAGE_B_MANAGER_REVIEW_2026-08-31.md;
- SOURCE_BYTE_REPORT_R0.4.md.

Choose one exact bundle policy:

A. include those files with exact digest and explicit historical/non-normative
   authority classification; or

B. remove the package-local links from the standalone authority bundle and
   retain only current r0.5/freeze-candidate records.

Preferred disposition: B.

The included verify_stage_a.py is not runnable from the current standalone
bundle because its required Stage A package-history files are absent.

Either:

- include every required input and classify it; or
- remove verify_stage_a.py from the authority-only standalone bundle; or
- rename and label it as a repository-context historical verifier that is
  intentionally not runnable from this archive.

The standalone bundle shall contain no unresolved package-local Markdown link.

Repository-relative program links may remain only with an explicit statement
that they require the complete repository context and are not bundle-local
objects.

======================================================================
8. SUPPLY THE COMPRESSED-ARCHIVE SIDECAR
======================================================================

The bundle README states that a separately distributed sidecar binds the
compressed archive. Supply that sidecar with the returned archive, or remove
the claim.

For the archive reviewed in this pass, the independently observed values were:

    size = 575058 bytes

    SHA-256 =
    ebfa5c653a6c989c7250ed2191898022b9f6382aeb8613e3bc120d3ed4561682

A rebuilt or changed archive shall receive its own newly computed sidecar.
Do not reuse the digest above after any byte change.

======================================================================
9. SOURCE, BUILD, AND VISUAL PREFLIGHT
======================================================================

After repair, rerun and return:

- author-packet SHA-256 verification;
- active authority-manifest verification;
- byte-count and digest verification for every active object;
- internal-link audit with zero unresolved bundle-local links;
- standalone verifier audit;
- shared-module import-order and byte-identity checks;
- requirement/prediction/AO count and gap checks;
- clean document builds;
- PDF metadata checks;
- all-page render QA;
- compressed-archive digest and sidecar verification.

Preserve unless a new obligation is deliberately added:

    50 requirements
    25 predictions
    6 AO families
    21 AO alternatives
    17 retained SODL identities.

If a new stochastic-model-domain requirement is appended, use REQ-051.
If a new falsifiable model-domain consequence is appended, use PRED-026.
Do not renumber existing identities.

======================================================================
10. OWNER-DISPOSITION AND CONDITIONAL-FREEZE OUTPUT
======================================================================

Return an owner-disposition packet that clearly separates:

A. package-level method authorization;
B. per-parent-class authorization;
C. per-named-use authorization;
D. per-realization method selection;
E. engineering representation declarations;
F. unavailable routes.

Recommended title:

    Matched-template map amplitude estimation.

Recommended AO architecture:

- authorize AO-001-A as the sole exact local-GLS
  optimal/reference-variance method when every theorem premise holds;
- authorize AO-001-C as a separately named nonoptimal field-power-spectral
  weighting method;
- retain AO-001-B and AO-001-D as successor-authorship triggers;
- retain AO-002 exact numerical realization or typed unavailability as an
  engineering declaration;
- select AO-003 scope separately for each named covariance role;
- owner-bind finite AO-004 state-query vocabulary before representation;
- owner-bind AO-005 response type/domain/query/validity before representation;
- retain seven AO-006 role meanings and dependency graph as normative, with
  layout engineering-only.

No authorized method is an automatic fallback for another.

After exact owner dispositions, return a proposed conditional scientific freeze
for SCI-FLT-MATCHED v0.1.

The freeze may establish the generic estimator family, method subtypes,
mathematical response/covariance/support/product meanings, and typed unavailable
routes.

It shall not establish:

- an available numerical MAP parent;
- an available numerical weighting realization;
- a registered SCI-VAL profile;
- implementation conformity;
- numerical adequacy;
- response or covariance fidelity;
- detection performance;
- observational validation;
- readiness;
- production suitability;
- production authorization;
- Unity activity.

======================================================================
11. DELIVERABLES
======================================================================

Return:

1. final micro-repaired Scientific Rationale and Contract;
2. final micro-repaired Engineering Conformance Specification;
3. exact shared normative modules;
4. stochastic-model versus observed-payload domain amendment;
5. corrected lifecycle state graph;
6. AO-001 authorization/realization multiplicity amendment;
7. title owner-disposition record;
8. honestly labeled role-semantics/profile drafts;
9. revised decision ledger;
10. revised route-status record;
11. semantic-change map;
12. complete active authority manifest;
13. source-byte and link-closure report;
14. build/consistency report;
15. PDF visual-QA and metadata report;
16. compressed output archive and matching archive sidecar;
17. proposed conditional scientific-owner freeze record.

Do not claim a numerical route, implementation conformity, response or
covariance fidelity, observational validation, source-detection performance,
readiness, production suitability, production authorization, or Unity status.