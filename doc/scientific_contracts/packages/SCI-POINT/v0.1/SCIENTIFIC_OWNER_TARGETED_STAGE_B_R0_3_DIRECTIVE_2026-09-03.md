SCI-POINT v0.1 r0.3 — FINAL TARGETED STAGE B
VAL-SEMANTICS, RESPONSE-TYPE, DIAGNOSTIC,
AND CONDITIONAL-FREEZE DIRECTIVE

The SCI-POINT Stage B r0.2 scientific architecture is accepted as the basis
for the final targeted revision.

Do not perform another broad POINT derivation.

Preserve:

- the conditional generic SCI-POINT contract;
- one observation-local, per-array, value-or-status fit atom;
- no POINT-owned cross-array aggregate, telescope correction, correction
  selection, or correction application;
- MAP, JINC, FLT-FIXED, and FLT-MATCHED as distinct parent families;
- terminal FRUIT as ancestry and not a fifth parent family;
- no coadd or intermediate-FRUIT parent;
- the invariant zero-background six-role elliptical-Gaussian family;
- unavailable numerical width, angle, amplitude-domain, objective, weighting,
  search, fallback, solver, and formal-error conventions;
- expected source position, parent-map origin, search center, seed, fitted
  centroid, and source association as distinct objects;
- Delta_POINT = fitted centroid minus expected source position in the exact
  AltAz tangent basis;
- source association independently required for every search branch;
- POINT-FIT-AMPLITUDE-COMPONENT distinct from
  POINT-SOURCE-ASSOCIATED-AMPLITUDE-DIAGNOSTIC;
- the two independently requested scalar diagnostic product roles;
- role-state atomicity;
- separate compatibility, formal-error, and full-map-RMS method gates;
- fixed-branch response, full-procedure response, observational bias/accuracy,
  parent response, and source association as separate roles;
- Equation 13 covariance with all cross terms;
- separate use ownership;
- diagnostic display as consumer action only;
- every present claim ceiling and nonclaim.

Do not inspect implementation, configuration, schemas, defaults, tests, audits,
repairs, validation, reductions, generated products, logs, accepted runs,
performance records, adjacent-package implementation, repository history,
external literature, or web sources.

Preserve SCI-POINT-REQ-001 through REQ-038 and
SCI-POINT-PRED-001 through PRED-032. Amend existing IDs where correcting the
same obligation. Use REQ-039/PRED-033 and later only for genuinely new
independent obligations.

======================================================================
1. REPLACE THE POINT-AUTHORED SCI-VAL TABLE WITH THE EXACT BOUND SEMANTICS
======================================================================

The r0.2 axis vocabularies are correct, but the disposition table is not.

Do not independently author a new SCI-VAL disposition function inside POINT.
Either:

A. import and content-bind the exact governing SCI-VAL function; or

B. provide a subordinate crosswalk that cannot differ from the bound SCI-VAL
   authority.

Use:

    request:
        requested | not_requested

    applicability:
        applicable | inapplicable | applicability_unknown

    eligibility:
        eligible | ineligible | decision_unavailable

    realization:
        realized | incomplete | failed | not_produced.

Preserve the absence of an eligibility proposition as a distinct condition.
Do not replace it with decision_unavailable or ineligible.

Required semantics:

A. use not requested

    request       = not_requested
    applicability = no proposition
    eligibility   = no proposition
    realization   = not_produced

B. requested use known to be inapplicable

    request       = requested
    applicability = inapplicable
    eligibility   = no proposition
    realization   = realized

when the decision artifact is successfully written.

C. missing profile, source binding, or unresolved structural scope

    request       = requested
    applicability = applicability_unknown
    eligibility   = decision_unavailable
    realization   = realized

when the accounting decision artifact is successfully written.

D. applicable use with a decisive owner-defined exclusion

    request       = requested
    applicability = applicable
    eligibility   = ineligible
    realization   = realized

E. applicable use with no exclusion but one unresolved required permission

    request       = requested
    applicability = applicable
    eligibility   = decision_unavailable
    realization   = realized

F. applicable use with every required permission established

    request       = requested
    applicability = applicable
    eligibility   = eligible
    realization   = realized

G. incomplete, failed, or not-produced decision artifact

The realization axis is respectively incomplete, failed, or not_produced.
A defective or absent decision artifact carries no authoritative eligibility
assertion merely because its realization failed.

A structural scientific conflict may be faithfully recorded in a realized
decision artifact. Use realization=failed only when production of the decision
artifact itself fails.

Retain producer facts regardless of the named-use outcome.

Update:

- Section 3.2 and ECS Section 1.2;
- every edge-case row using that table;
- all four draft profile templates;
- REQ-027, REQ-030, REQ-034, and REQ-037;
- PRED-023, PRED-024, and PRED-031;
- owner/view parity and traceability.

======================================================================
2. TYPE FULL-PROCEDURE RESPONSE AVAILABILITY
======================================================================

Equation 10 shall not subtract two complete-procedure outputs unless both
outputs are numerically comparable.

Define the numerical procedure output used by the response, for example:

    theta_hat_P(m)

rather than subtracting an untyped complete product/lifecycle object.

A baseline and perturbed output are comparable only when they have:

- the same intended source identity;
- compatible source-association state;
- the same declared parameter roles;
- one compatible width/angle gauge and axis ordering;
- compatible units and tangent basis;
- numerical availability of every response target component;
- an exact perturbation-domain relation.

If one invocation:

- fails;
- produces no fit;
- associates a different source;
- loses source association;
- changes to an incompatible parameter gauge;
- crosses exact circularity or another nondifferentiable state;
- produces an unavailable target component;

then the affected numerical full-procedure response is unavailable or a
separately typed discontinuity/state transition. Do not subtract incompatible
values.

Every differential or finite-difference response record shall bind:

- baseline product and state;
- perturbed product and state;
- perturbation source, domain, direction, and unit;
- epsilon magnitude;
- one-sided, two-sided, or analytic convention;
- search/fallback/retry/support/weight/constraint/association changes;
- parameter-gauge comparison;
- component availability;
- cause and provenance.

======================================================================
3. SEPARATE POINT-ONLY AND WHOLE-CHAIN RESPONSE
======================================================================

Do not use one unsuperscripted equation:

    R_theta,source = R_theta,m R_parent

for every response family.

Define the fixed-state chain rule only where valid:

    R_theta,source^fixed
      =
      R_theta,m^fixed
      R_m,source^fixed,

with exact compatible:

- domains and codomains;
- source convention;
- parent and POINT state generations;
- units;
- WCS/tangent basis;
- support;
- derivative convention.

Define separately:

    R_theta,m^(POINT-FP | parent-fixed)

for a response that reruns the complete POINT procedure while holding the
realized parent product fixed.

Define:

    R_theta,source^chain-FP

only when separate authority reruns every data-dependent upstream parent
operation included in that claim as well as POINT.

For MAP, JINC, FLT-FIXED, FLT-MATCHED, and terminal-FRUIT parents, bind which
parent response family is being consumed. A fixed-state parent companion,
parent full-procedure response, and whole-chain response are never aliases.

A finite-difference response shall not be multiplied by another response as
though it were a linear Jacobian unless an exact composition theorem applies.

If the required upstream rerun or compatibility authority is absent, the
whole-chain response remains unavailable.

Update Equations 8-11, response roles, UNAV-018/022, response records, REQ-006,
REQ-025, REQ-026, REQ-036, and PRED-016/017/025/030.

======================================================================
4. SETTLE SIGNED VERSUS MAGNITUDE STANDARDIZATION
======================================================================

Equation 7 currently fixes:

    Z_A = A_hat / sigma_A

while the prose says the signed-versus-absolute convention remains open.

These are different products.

Preferred disposition:

    POINT-FORMAL-AMPLITUDE-STANDARDIZATION@1
      = A_hat / sigma_A

is the signed canonical quantity.

If an absolute-magnitude quantity is desired, assign a separate identity, for
example:

    POINT-FORMAL-AMPLITUDE-MAGNITUDE-STANDARDIZATION@1
      = abs(A_hat) / sigma_A.

POINT-FORMAL-ERROR-METHOD owns the construction and validity of sigma_A. It
does not silently choose whether the numerator is signed or absolute.

Similarly, retain:

    FITTED-AMPLITUDE-OVER-FULL-MAP-RMS@1

as the precise identity for Equation 6. If A_hat remains signed, describe the
quantity as a signed fitted-amplitude/RMS diagnostic rather than implying a
universally nonnegative dynamic range.

Bind the same exact convention to any approved sig2noise or fit_sig2noise
alias. A legacy alias is admitted only when it is mathematically identical.

Update Equations 6-7, product names/ceilings, method-record acceptance,
REQ-023/024, and PRED-021/022.

======================================================================
5. CLARIFY RESPONSE STATUS VERSUS AVAILABLE RESPONSE
======================================================================

A base numerical fit or fit-amplitude component shall require an exact parent
response identity and status record. It shall not automatically require an
available complete numerical response unless:

- the selected compatibility method uses that numerical response; or
- the requested claim/named use explicitly requires it.

Use:

    exact response identity/status, possibly unavailable

rather than an ambiguous requirement for “exact parent response.”

An unavailable response may coexist with:

- a numerical fit;
- a fit-amplitude component;
- a processed-map centroid;
- a source-associated processed-map displacement,

when every dependency of those narrower quantities is available.

It blocks only the response-, bias-, pointing-, correction-, calibration-, or
photometric claims whose exact dependency rows require it.

Likewise, REQ-006 shall require binding of the route response-chain,
processed-profile, Gaussian-compatibility, response-center, model-mismatch,
and centroid-bias state records. It shall state which numerical objects may be
unavailable for the base fit.

Update:

- notation for A;
- route table;
- product/claim dependency matrix;
- REQ-006, REQ-019, and REQ-026;
- PRED-016, PRED-017, and PRED-030.

======================================================================
6. KEEP DOWNSTREAM NAMED-USE DECISIONS OUT OF BASE FIT COMPLETION
======================================================================

The base POINT fit atom may require its POINT-owned fit-completeness/publication
decision.

The following remain separate immutable child decision records:

- POINTING-SUPPORT displacement admission;
- TELESCOPE-QC parameter admission/action;
- CAL-TOLPROJ source-associated amplitude transfer admission.

An unrequested, unavailable, incomplete, failed, or not-produced downstream
decision shall not block publication of an otherwise complete POINT fit atom.

A later named-use decision:

- references the immutable POINT fit;
- has its own request, profile, SCI-VAL realization, lifecycle, and provenance;
- does not mutate the fit;
- does not create a new POINT fit generation;
- does not rescue another use.

Replace wording that the complete fit “has use-specific evaluation facts” with
wording that it may have separately parented decision records for requested
uses.

The publication candidate shall contain:

- the POINT-owned fit-publication/completeness state; and
- identities/statuses of requested linked decisions where required;

but shall not require successful realization of every possible downstream use.

======================================================================
7. CLARIFY W_fit ROLE MAPPING
======================================================================

W_fit is the exact weighting object consumed by the compatibility fit.

Its scientific source may be:

- uniform weighting;
- reliability-derived weighting;
- covariance-derived weighting;
- inverse covariance;
- another owner-approved construction.

Those source roles are not aliases.

Require an exact mapping from the source object to W_fit, including identity,
units, normalization, support, state, and claim ceiling.

Numerical equality or inverse-square units do not make a reliability weight an
inverse covariance. Conversely, saying W_fit is “distinct from” a reliability
or inverse-covariance source shall not prohibit an approved method from
explicitly deriving W_fit from that source.

======================================================================
8. REPAIR EXPOSURE AND EARLY-STOP RECORDS
======================================================================

Replace “exposure identity” in the POINT published-product record with:

    inherited parent observation/exposure lineage or typed not_applicable.

State:

- POINT creates no acquired exposure;
- one fit product is not an additional independent observation;
- per-array fits from one observation generally share parent exposure and
  source-reference terms;
- downstream aggregation requires explicit dependence authority.

Repair PRED-013.

Missing or non-finite support shall:

- follow the future compatibility method’s exact stop rule;
- preserve every lifecycle record actually reached;
- record the exact terminal unavailable, failed, or not-produced state;
- never fabricate applied or fit_realized records when the route stopped
  earlier;
- never silently delete required rows.

======================================================================
9. SOURCE AND AUTHORITY CLOSURE
======================================================================

Return the exact bytes of:

- the owner-approved Stage A r0.3 author packet;
- its manifest and archive sidecar;
- SCI-POINT-COMMON-CORE/v0.1-r0.3;
- rationale source;
- ECS source;
- exact build recipe and tool identities;
- all parent-boundary requirement records;
- all four named-use profile drafts;
- compatibility, formal-error, and full-map-RMS method templates;
- requirement/prediction traceability;
- owner-decision parity report;
- common-core/view parity report;
- semantic-change report;
- clean-build report;
- PDF render/metadata report;
- source and PDF hashes;
- compressed delivery archive and matching archive sidecar.

Both PDF covers shall bind the same:

- Stage A packet;
- common core;
- owner directive;
- build record.

The common core remains the sole normative scientific source. The rationale
explains it. The ECS authors prospective evidence procedure only.

Use one PDF metadata convention that names both the scientific program and
Grant Wilson’s scientific-owner role unambiguously.

======================================================================
10. IDENTIFIERS AND PREFLIGHT
======================================================================

Preserve:

    SCI-POINT-REQ-001 through REQ-038
    SCI-POINT-PRED-001 through PRED-032.

If a genuinely new obligation is unavoidable, begin with:

    SCI-POINT-REQ-039
    SCI-POINT-PRED-033.

Verify:

- one byte-identical common core in both views;
- exact VAL semantics and absence-of-proposition behavior;
- no stale decision_unavailable/ineligible assignments for not-requested or
  inapplicable uses;
- exact response-type and composition parity;
- exact signed/magnitude diagnostic identity;
- no required downstream child inside base fit completion;
- canonical lifecycle and approximately_centered spelling;
- exact source and manifest hashes;
- all PDFs reopen and render without clipping, overlap, broken glyphs,
  malformed equations, or unreadable tables;
- title, owner, revision, date, status, and metadata agree.

======================================================================
11. DELIVERABLES
======================================================================

Return:

1. revised Scientific Rationale and Contract r0.3;
2. revised Engineering Conformance Specification r0.3;
3. exact shared normative common core;
4. SCI-VAL disposition/source-binding amendment;
5. response-family and composition crosswalk;
6. full-procedure response comparability rule;
7. signed-versus-magnitude diagnostic disposition;
8. response-status/product-dependency amendment;
9. base-fit versus downstream-decision boundary amendment;
10. W_fit source-role mapping amendment;
11. exposure and early-stop lifecycle amendment;
12. revised unavailable-state and prediction registers;
13. complete source/build/authority packet;
14. semantic-change and owner-parity reports;
15. PDF visual-QA and metadata report;
16. proposed conditional scientific-owner freeze disposition.

Do not claim a numerical route, implementation conformity, response or
covariance fidelity, uncertainty coverage, observational pointing accuracy,
validation, performance, readiness, production suitability, production
authorization, or Unity activity.