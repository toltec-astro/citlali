WORK ORDER: MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001

Purpose
=======

Perform a fresh-context, implementation-blind horizontal scientific audit of
the frozen Citlali map-space contract family.

The central question is:

    When all in-scope contracts are believed simultaneously, do they define
    a scientifically coherent set of map-space products, transformations,
    uncertainty products, and downstream measurements?

This is not a new local review of any individual package. Do not reopen settled
package-local science merely because another formulation might be preferable.
Audit the interfaces, shared concepts, product identities, authorized routes,
and cross-package claims.

Audit Base and Repository Safety
================================

Bind this audit to exact commit:

    5f0fc20042b88fb6cd883c92d1b59b7f22832901

Do not use a moving branch name as the scientific audit identity.

Before beginning:

1. Verify that the exact commit exists.
2. Record:
   - commit SHA;
   - tree SHA;
   - parent SHA;
   - current branch or detached-HEAD state;
   - clean/dirty worktree status.
3. Create a fresh dedicated worktree or audit task based exactly on that commit.
4. Do not move `codex/refactor-mainline` or any other canonical ref.
5. Do not push, integrate, activate, clean up, or alter another worktree.
6. Do not inspect or disturb the active FRUIT or historical ALIGN worktrees.
7. Do not rely on conversation memory as scientific authority.

This work order authorizes read-only inspection of the frozen contract sources
and creation of audit-only artifacts in a dedicated audit directory. It does
not authorize changes to frozen package content, application code, validation
products, canonical branches, or remotes. Do not commit the audit artifacts
unless separately authorized.

Scientific Authority Hierarchy
==============================

Resolve the authoritative source set from repository records at the exact audit
commit.

Use the following precedence:

1. Exact frozen package authority manifests, freeze records, approved manifests,
   and their bound artifacts.
2. Current package-level authoritative Stage B products and package READMEs,
   where those READMEs accurately describe the frozen authority.
3. The accepted predecessor boundary material needed to interpret the
   PTC-to-map-space ingress.
4. Manager-facing indices, roadmaps, and status records for inventory and
   current-status routing only.
5. Historical or superseded records only as history; they may not override a
   later frozen authority.

Do not use:

- application source code;
- implementation tests;
- runtime behavior;
- validation outcomes as substitutes for contract authority;
- active development branches;
- recovery dossiers or prior drafts to fill an authoritative gap;
- informal descriptions from conversations;
- assumptions about what the current implementation probably does.

Package verification scripts may be run to confirm identity, completeness, and
internal consistency. Their success does not by itself prove cross-package
scientific coherence.

If two purportedly authoritative sources conflict and the repository does not
unambiguously establish precedence, record an authority collision and invoke
the stop-and-escalate rule. Do not choose the interpretation that seems most
reasonable.

In-Scope Scientific Contracts
=============================

The first-pass horizontal audit includes:

- SCI-MAP
- SCI-JINC
- SCI-FLT-FIXED
- SCI-FLT-MATCHED
- SCI-NOI
- SCI-POINT

The audit is anchored at the frozen PTC-to-map-space handoff.

PTC is present only as an upstream boundary provider. Do not reopen PTC
internals, algorithms, or local scientific decisions. Determine whether MAP
and JINC require only what the accepted PTC boundary promises.

If a downstream package requires a stronger guarantee than the PTC boundary
provides, report a downstream boundary mismatch. Do not silently strengthen
PTC.

Explicitly Out of Scope
=======================

The following are outside this audit:

- active SCI-FRUIT development;
- the scientific design or validation of FRUIT;
- FRUIT attachment policy beyond identifying its deferred boundary;
- PTC internals;
- RTC, CAL, AST, ALIGN, or VAL package-local science;
- application-code conformance;
- runtime correctness or performance;
- implementation architecture;
- source detection or catalog production;
- invention of new algorithms, thresholds, estimators, or uncertainty methods;
- repair of frozen artifacts during the audit.

A frozen contract may refer to FRUIT as a possible parent, ancestry envelope,
or future attachment. Such references may be recorded as external/deferred
boundaries, but FRUIT must not be imported into the audit’s pass/fail reasoning.

VAL may be mentioned where the frozen contracts assign it a registry or
evaluation role. VAL must not be treated as the author of package-specific
scientific admission, eligibility, threshold, or named-use policies.

Required Architectural Interpretation
=====================================

Do not assume that the six packages form one serial pipeline.

In particular:

- JINC is not a stage that necessarily follows MAP.
- MAP and JINC are distinct mapmaking routes or producers whose exact
  relationship must be derived from their frozen contracts.
- FLT-FIXED and FLT-MATCHED have distinct scientific contracts and must not be
  collapsed into a generic “FLT” operation.
- NOI uncertainty products must not be equated with PTC weights, mapmaking
  accumulation coefficients, coverage, inverse variance, covariance, or any
  other quantity unless the frozen contracts explicitly authorize that
  equivalence.
- POINT is a downstream measurement package for eligible parent products. It
  is not a general source-detection or catalog-production package.
- Some authorized routes may bypass one or more packages.
- Some products may be companions rather than serial transformations.
- A product may be suitable for one named use and unsuitable for another.

The auditor must derive the scientific product graph from the authority rather
than imposing a linear sequence such as:

    PTC -> MAP -> JINC -> FLT -> NOI -> POINT

That formulation is prohibited unless the contracts themselves establish a
particular edge.

Core Horizontal-Audit Principle
===============================

For every authorized producer-to-consumer boundary, ask both questions:

1. Does the producer guarantee everything the consumer requires?
2. Does the consumer claim no more than the producer has established?

A consumer may narrow a producer’s claims. It may not silently strengthen them.

A producer’s silence, typed unavailability, qualified state, or route-specific
limitation may not be converted downstream into:

- a default value;
- a zero value;
- diagonal covariance;
- statistical independence;
- universal calibration;
- unrestricted spatial support;
- a known transfer function;
- successful lifecycle completion;
- scientific eligibility;
- aggregate availability.

Phase 1 — Exact Source and Authority Manifest
=============================================

Produce an exact source manifest before drawing scientific conclusions.

For every in-scope package, record:

- package identity and version;
- active frozen revision;
- exact package authority record;
- exact manifest or binding record;
- manifest digest;
- artifact digests where available;
- freeze or approval state;
- supersession relationship;
- verifier used and its result;
- whether any package status language conflicts with its freeze record.

Also identify the exact PTC boundary sources admitted for this audit.

Classify each source as:

- authoritative scientific source;
- authoritative status/identity source;
- accepted boundary source;
- manager-facing inventory;
- superseded historical material;
- excluded material.

The manifest must make it possible for an independent reviewer to reconstruct
the exact authority set without using the conversation history.

If the authoritative revision of any in-scope package cannot be uniquely
identified, stop the scientific interpretation of that package and escalate.

Phase 2 — Product and Boundary Graph
====================================

Construct a product-and-boundary graph from the frozen contracts.

The graph must show, where defined:

- each scientific product;
- its authoritative producer;
- eligible consumers;
- parent product identity;
- alternate mapmaking routes;
- direct and filtered routes;
- companion uncertainty or randomization products;
- required response, kernel, support, covariance, or metadata companions;
- bypasses;
- route-specific gates;
- unavailable or deferred edges;
- external boundaries, including the deferred FRUIT attachment.

Do not create nodes merely because an implementation might contain them.

For each edge, give the exact contract evidence establishing:

- that the route is authorized;
- which parent product is consumed;
- which guarantees cross the boundary;
- which guarantees do not cross;
- which route-specific conditions apply.

At minimum, determine whether the frozen authority permits and consistently
defines:

- PTC to MAP;
- PTC to JINC;
- direct eligible map-product consumption by POINT;
- eligible filtered-map consumption by POINT;
- NOI attachment or propagation relationships;
- required response and uncertainty companions.

This list is a set of questions, not a presumption that every route exists.

Phase 3 — Cross-Package Scientific Conformance Matrix
=====================================================

Create one cross-package matrix with rows organized primarily by product or
producer-to-consumer boundary, not merely by document.

For every applicable route, audit the following dimensions.

A. Product identity and parentage
---------------------------------

Check:

- stable product identity;
- package and route identity;
- exact parent or ancestry;
- mapmaker identity;
- filter identity;
- array identity where applicable;
- observation identity where applicable;
- distinction between direct, fixed-filtered, and matched-filtered products;
- whether downstream products retain enough ancestry to interpret their claims;
- whether identifiers can collide across routes.

A JINC product must not silently become a MAP product, or vice versa. A filtered
product must not lose the identity of its parent and transformation.

B. Units, calibration, and reference state
------------------------------------------

Check:

- numerical units;
- calibration state;
- surface-brightness versus flux-like interpretations;
- per-pixel, per-beam, integrated, or other normalization;
- reference plane or response convention;
- whether filtering changes the interpretation or normalization;
- whether downstream quantities remain in the exact parent product’s units.

No package may promote a route-dependent amplitude into universal source flux
without explicit authority.

C. Coordinates, geometry, and support
-------------------------------------

Check:

- coordinate basis;
- projection and pixel geometry;
- reference position;
- map support;
- valid support;
- coverage;
- masks and flags;
- edge behavior;
- interpolation assumptions;
- footprint and finite-support semantics;
- whether a consumer distinguishes no coverage from measured zero;
- whether filtered support differs from parent support.

D. Signal and residual semantics
---------------------------------

Check:

- what physical or statistical quantity the map represents;
- which background or baseline has been removed, retained, or left
  unidentifiable;
- whether zero has a scientific interpretation;
- whether signal, residual, model, diagnostic, and randomization products remain
  distinct;
- whether any consumer assumes a background convention that the producer does
  not guarantee.

E. Coefficients, weights, coverage, variance, and covariance
------------------------------------------------------------

Audit every use of terms such as:

- weight;
- coefficient;
- coverage;
- hit count;
- inverse variance;
- variance;
- RMS;
- uncertainty;
- covariance;
- randomization;
- significance;
- normalization.

Determine whether the same term is being used for different quantities, or
different terms for the same quantity.

Do not infer equivalence from similar dimensions or numerical use.

In particular, verify that NOI quantities are not silently substituted for
mapmaking coefficients or PTC weights, and that mapmaking weights are not
automatically interpreted as complete statistical uncertainty.

F. Response, transfer function, kernel, and effective source shape
------------------------------------------------------------------

Check:

- whether every transformed product has a defined or explicitly unavailable
  response;
- kernel parentage;
- normalization;
- spatially variable versus invariant response;
- filter-induced response changes;
- support changes;
- whether POINT receives the information needed to interpret centroid,
  amplitude, widths, and angle;
- whether fitted widths and angles are correctly limited to the effective
  processed-map source shape;
- whether any contract mislabels processed-map response as intrinsic telescope
  beam or SCI-BEAM authority.

A filtering operation may not preserve the parent response merely by silence.

G. Uncertainty and statistical claims
-------------------------------------

Check:

- which uncertainties are published;
- their estimation authority;
- route dependence;
- marginal versus joint uncertainty;
- availability of covariance;
- propagation through filtering;
- relationship between NOI products and POINT parameter uncertainties;
- distinction among measurement uncertainty, fit uncertainty, repeatability,
  systematic uncertainty, astrometric uncertainty, and QC diagnostics;
- typed unavailable states.

Absence of joint covariance must never be interpreted as:

- zero covariance;
- diagonal covariance;
- independent fitted parameters;
- permission for a covariance-dependent use.

Honor all frozen POINT method gates and typed unavailable method authorities.
Do not recover or invent an unavailable compatibility, search, weighting,
formal-error, full-map-RMS, covariance, or uncertainty method from implementation
or historical evidence.

H. Lifecycle, completeness, partial success, and failure
--------------------------------------------------------

Check:

- producer lifecycle states;
- per-array atomicity;
- sibling-array survival after one array fails;
- partial observation success;
- qualified results;
- unavailable results;
- diagnostic products;
- failed products;
- aggregate admission;
- downstream aggregation ownership;
- fallback behavior;
- constraint-limited fits.

Do not convert a per-array result into whole-observation success.

Verify that POINT’s per-array results and downstream aggregation responsibilities
remain distinct wherever the frozen contracts require that distinction.

I. Eligibility, named use, and policy ownership
-----------------------------------------------

Distinguish:

- existence of a product;
- lifecycle completeness;
- scientific validity;
- eligibility for a particular named use;
- the action prescribed to a consumer;
- aggregate admission.

A result may be eligible for one use and ineligible for another.

Verify that `diagnostic_only`, where used by the frozen authority, is treated as
a named-use or consumer-action disposition rather than as a global producer
lifecycle state or universal “bad fit” flag.

Confirm the separate ownership of, where applicable:

- fit-result completeness;
- pointing-correction eligibility;
- telescope or observing-condition QC admission;
- photometric-transfer eligibility;
- uncertainty-use eligibility.

VAL may register or evaluate such policies. It may not author them.

J. Deterministic, inferential, and diagnostic products
------------------------------------------------------

Check whether each product is:

- a deterministic transformation;
- an estimator;
- an inference-bearing product;
- a diagnostic;
- a randomization or uncertainty companion;
- a named-use evaluation.

Do not merge these categories for documentary convenience.

FLT-FIXED and FLT-MATCHED must retain their exact frozen scientific distinctions.
Neither FLT package may acquire source detection merely because POINT may
subsequently fit an eligible map product.

K. Unavailable and not-applicable states
----------------------------------------

For every absent field, method, route, or companion, classify it as one of:

- available;
- conditionally available;
- explicitly unavailable;
- scientifically inapplicable;
- not required for that route;
- ambiguous due to a contract defect.

Do not use “missing” as a substitute for this classification.

Known Owner-Bound Constraints
=============================

The following are owner-bound constraints to test for consistent expression
across package boundaries. They are not invitations to reopen the decisions:

1. MAP and JINC are not to be modeled as a mandatory serial pair.
2. FLT-FIXED and FLT-MATCHED retain distinct product identities and claims.
3. Filter packages output filtered forms of admitted map products; they do not
   own source detection.
4. POINT is not a general detection or catalog package.
5. POINT’s centroid displacement is its primary pointing measurement.
6. POINT fitted amplitude remains in the exact parent product’s units,
   calibration, and response unless another owner authorizes a further use.
7. POINT fitted widths and angle describe the effective fitted source shape
   after the exact mapmaking and filtering route.
8. POINT amplitude, widths, and angle may serve as telescope and
   observing-condition QC metrics without becoming universal flux or intrinsic
   beam measurements.
9. Published marginal formal errors do not establish joint covariance.
10. Unavailable joint covariance is not zero, diagonal, or independent.
11. Per-array success is atomic; failure of one array need not erase valid
    sibling-array results.
12. Whole-observation or cross-array aggregation remains downstream unless
    explicitly assigned.
13. Named-use eligibility is distinct from result existence and lifecycle.
14. `diagnostic_only` is not a universal producer state.
15. NOI uncertainty semantics remain distinct from mapmaking accumulation and
    PTC weighting semantics unless exact authority says otherwise.
16. VAL registers and evaluates policies; it does not author package-local
    scientific policies.
17. Active FRUIT is excluded from this audit.

If an in-scope frozen contract appears to contradict one of these constraints,
do not “correct” the contract from this list. Record the exact contradiction
and escalate it as an authority or scientific-meaning conflict.

Representative Contract Traces
==============================

In addition to the matrix, perform several explicit end-to-end contract traces.

At minimum include:

1. One MAP-origin direct route to an eligible downstream consumer.
2. One JINC-origin direct route.
3. One fixed-filtered route, if authorized.
4. One matched-filtered route, if authorized.
5. One route involving NOI uncertainty or randomization products.
6. One POINT result that is complete but not eligible for every named use.
7. One partial per-array success case.
8. One case involving an explicitly unavailable response, covariance, method,
   or uncertainty.
9. One failure or fallback case.
10. One edge/support-limited case.

For each trace, show:

- exact parent product;
- transformations;
- units and calibration;
- support and masks;
- response or kernel state;
- uncertainty companions;
- lifecycle state;
- named-use eligibility;
- unavailable information;
- final claims that are and are not permitted.

These are documentary traces. Do not run observations or infer code behavior.

Finding Classification
======================

Classify every finding as one of:

CRITICAL
    A contradiction or authority collision that changes scientific meaning,
    makes a product uninterpretable, permits incompatible numerical
    interpretations, or causes a consumer to require guarantees the producer
    does not provide.

MAJOR
    A producer-consumer mismatch involving product identity, units, coordinates,
    support, response, uncertainty, lifecycle, eligibility, or failure
    semantics that can plausibly produce an incorrect scientific use.

MODERATE
    A route, policy, identifier, or unavailable-state ambiguity that does not
    immediately change the numerical quantity but could cause inconsistent
    admission, propagation, or downstream use.

MINOR
    A terminology, cross-reference, inventory, or status defect whose intended
    scientific meaning is otherwise unambiguous.

Do not inflate editorial differences into scientific defects. Conversely, do
not downgrade a scientific contradiction merely because the implementation may
currently avoid the problematic route.

Every finding must contain:

- stable finding ID;
- severity;
- exact package or boundary;
- exact source paths and line references;
- authoritative revision and digest;
- conflicting statements or missing guarantee;
- scientific consequence;
- affected authorized routes;
- whether the issue is:
  - contradiction;
  - ambiguity;
  - undocumented dependency;
  - authority collision;
  - typed unavailability violation;
  - stale manager record;
- smallest plausible repair locus;
- whether frozen package content would need a successor revision;
- whether a scientific-owner decision is genuinely required.

Proposed Repairs
================

Do not implement repairs during this audit.

For each finding, propose the smallest scientifically honest repair class:

A. Manager-only reconciliation
    Appropriate only when frozen package authority is already coherent and a
    status, roadmap, index, or explanatory record is stale.

B. Cross-reference or terminology repair
    Appropriate only when scientific meaning is already uniquely established
    and no claim changes.

C. Successor package revision
    Required when a frozen contract itself must change. Do not patch the frozen
    artifact in place.

D. New boundary companion
    Appropriate when two locally valid packages need an explicit shared
    boundary artifact without changing their local science.

E. Scientific-owner decision
    Required when the authority does not determine a unique scientifically
    meaningful resolution.

F. Explicit typed unavailability
    Appropriate when no authorized method or guarantee exists and no owner
    decision has supplied one.

Do not “harmonize” contracts by deleting scientifically meaningful differences.
Do not repair a missing method by inventing a generic industry-standard method.
Do not use implementation behavior as the proposed contract.

Owner Decision Queue
====================

Create an owner-decision item only when the contracts do not already determine
the answer.

Each owner decision must:

- state one narrow question;
- identify the affected products and routes;
- quote or cite the conflicting authorities;
- explain the scientific consequence;
- present the smallest viable options;
- identify what remains unavailable under each option;
- give a recommendation where the evidence supports one;
- avoid bundling unrelated issues.

Do not ask the owner to decide implementation mechanics that belong to a later
engineering contract.

Stop-and-Escalate Rule
======================

Immediately stop all repair drafting and do not resolve the issue yourself if
you find a contradiction involving any of the following:

- product identity or parentage;
- mapmaker identity;
- units or normalization;
- coordinate basis;
- calibration state;
- spatial support or zero-versus-no-data meaning;
- response, transfer function, or kernel;
- variance, covariance, or statistical independence;
- availability versus unavailability;
- lifecycle or partial-success semantics;
- named-use policy ownership;
- source-detection ownership;
- PTC boundary guarantees;
- freeze or supersession authority.

You may continue read-only checks that do not depend on the disputed
interpretation, but the final audit disposition must remain blocked.

Return an interim escalation containing:

- exact conflict;
- affected packages and routes;
- authoritative evidence;
- consequences of each interpretation;
- checks that can continue safely;
- the narrow owner decision or authority repair needed.

Required Deliverables
=====================

Create one dedicated audit directory, preferably:

    doc/scientific_contracts/audits/
      MAP_SPACE_HORIZONTAL_AUDIT_001/

Keep the deliverables concise and non-duplicative. Use matrices and references
instead of repeating the same prose in several files.

Required files:

1. SOURCE_AUTHORITY_MANIFEST.md
   Exact commit/tree identity, admitted package authorities, manifests,
   digests, freeze records, source classification, verifier results, and
   explicit exclusions.

2. PRODUCT_AND_BOUNDARY_GRAPH.md
   Scientist-readable graph and route catalog showing alternatives, companions,
   bypasses, unavailable edges, and the deferred FRUIT boundary.

3. CROSS_PACKAGE_CONFORMANCE_MATRIX.md
   Product- and boundary-centered matrix covering all audit dimensions. Use
   controlled statuses such as:

       PASS
       CONDITIONAL
       AMBIGUOUS
       CONTRADICTION
       UNAVAILABLE
       NOT_APPLICABLE

4. FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md
   Severity-ranked findings, bounded repair classes, and only genuine owner
   questions.

5. HORIZONTAL_AUDIT_REPORT.md
   Concise synthesis containing:
   - exact audited SHA;
   - purpose and scope;
   - package inventory;
   - product topology;
   - principal invariants;
   - representative route traces;
   - findings summary;
   - unresolved scientific questions;
   - explicit FRUIT exclusion;
   - final disposition and recommendation.

6. verify_horizontal_audit.py
   A bounded audit verifier that confirms:
   - the audit is bound to the exact intended commit;
   - all six in-scope packages are represented;
   - the PTC boundary is represented but not reopened;
   - FRUIT is marked excluded;
   - every admitted source has an identity or digest;
   - all graph edges appear in the conformance matrix;
   - all non-PASS matrix entries map to a finding or explicit unavailable state;
   - no frozen package artifact was modified;
   - no application code was modified;
   - no package freeze record was modified;
   - no canonical ref movement is implied.

Do not produce PDFs during this audit unless separately authorized.

Final Disposition
=================

End with exactly one recommended disposition:

ACCEPT
    The in-scope frozen contracts are horizontally coherent as written.

ACCEPT WITH MANAGER-ONLY CORRECTIONS
    Scientific authority is coherent; only non-authoritative inventory, status,
    or explanatory records require correction.

ACCEPT WITH BOUNDED CONTRACT REPAIR
    The architecture is scientifically coherent, but one or more frozen
    packages or shared boundaries require a separately authorized successor
    revision.

BLOCKED FOR SCIENTIFIC-OWNER DECISION
    A scientifically consequential ambiguity or contradiction cannot be
    resolved from current authority.

REJECT
    The in-scope contracts cannot simultaneously define a coherent map-space
    scientific system without substantial redesign.

The disposition must distinguish:

- local package correctness;
- cross-package boundary correctness;
- documentary/status correctness;
- implementation conformance, which is not assessed here.

Completion Conditions
=====================

The audit is complete only when:

- the exact authority set is reproducibly bound;
- all six in-scope packages have been examined;
- every authorized product route has been enumerated;
- every producer-consumer boundary has been checked in both directions;
- product identity, units, support, response, uncertainty, lifecycle,
  eligibility, failure, and unavailable-state semantics have been audited;
- MAP and JINC have not been treated as a serial pair without authority;
- NOI quantities have not been conflated with mapmaking or PTC quantities;
- POINT has not been expanded into source detection;
- FRUIT has remained out of scope;
- frozen artifacts remain byte-identical;
- every substantive finding is evidence-backed;
- every owner question is genuinely necessary;
- no repair, integration, mainline advancement, or push has occurred.

At completion, return:

1. the exact audited commit and tree;
2. the source-manifest digest;
3. verifier results;
4. finding counts by severity;
5. the recommended disposition;
6. the paths to all audit artifacts;
7. a clear statement that frozen package bytes, application code, canonical
   refs, FRUIT, and ALIGN were not changed.

Owner-Approved Refinements Incorporated 2026-09-03
===================================================

These refinements are part of this consolidated work order and have the same
owner-approved force as the complete directive above:

1. Preserve the complete approved directive as `WORK_ORDER.md` and bind its
   SHA-256 in `SOURCE_AUTHORITY_MANIFEST.md` so the audit instructions are
   independently reconstructable.
2. Admit, as boundary-only scientific sources where referenced by the in-scope
   frozen authorities, the exact PTC-to-MAP and PTC-to-JINC boundaries,
   AST-to-MAP and AST-to-JINC coordinate boundaries, exact SCI-VAL
   Registry/profile records, and `doc/SCIENTIFIC_CONVENTIONS.md`. These
   sources may establish only their boundary/shared-convention roles; do not
   reopen PTC, AST, VAL, ALIGN, RTC, or CAL package-local science.
3. Include each in-scope package's shared formal core, scientist-facing
   rationale, and engineering-conformance view as contract representations.
   Audit the engineering-conformance views for fidelity to the frozen
   scientific authority and cross-package boundary—not for application
   implementation conformance.
4. Assign stable product IDs and stable route/edge IDs, and reuse them
   consistently across `PRODUCT_AND_BOUNDARY_GRAPH.md`,
   `CROSS_PACKAGE_CONFORMANCE_MATRIX.md`,
   `FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md`,
   `HORIZONTAL_AUDIT_REPORT.md`, representative traces, and
   `verify_horizontal_audit.py`.
5. Every representative trace must be supported by authority. When a requested
   route is not authorized, record a negative `NOT_AUTHORIZED`,
   `UNAVAILABLE`, or `NOT_APPLICABLE` trace with exact evidence instead of
   constructing a hypothetical route.

Owner Disposition Incorporated 2026-09-03
==========================================

**Owner disposition — MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001**

I accept the audit’s stop-and-escalate finding and the classification of the
four conflicts as MAJOR. I resolve the disputed scientific meanings in favor
of the frozen SCI-MAP r0.7.1 and SCI-JINC r0.3 authorities.

The four conflicts are to be resolved as follows:

1. **MAP physical identity:** Preserve the frozen nonpolarimetric
   total-intensity-equivalent MAP identity. The conflicting generic
   Stokes-\(I\) language in `doc/SCIENTIFIC_CONVENTIONS.md` is superseded to the
   extent that it assigns formal Stokes-\(I\) identity to this MAP product.
2. **MAP observation coaddition:** Preserve the frozen dimensionless
   \(u_{\mathrm{op}}=1\) coefficient for each admitted MAP observation row. Make
   explicit that this is an observation-level coaddition rule and does not
   replace or flatten authorized sample-, pixel-, numerator-, denominator-,
   validity-, or coverage-level information. Do not generalize this into a
   JINC coaddition rule: base SCI-JINC v0.1 authorizes no cross-observation
   coaddition.
3. **Exposure:** Preserve MAP exposure as a geometric quantity based on unique
   original AST-coordinate occurrences. Do not redefine physical exposure
   through processed signal membership, filtering footprint, interpolation,
   operator support, response support, or statistical weight. Do not infer a
   physical-exposure or standalone-support product for base JINC.
4. **JINC product closure:** Preserve the exact frozen five-role SCI-JINC base
   bundle. Additional weight, support, response, covariance, exposure,
   diagnostic, or generalized-provenance numerical roles are not implicit,
   optional, or downstream-inferable base-v0.1 products. Where a representative
   route would require such a role, record the appropriate negative
   `NOT_AUTHORIZED`, `UNAVAILABLE`, or `NOT_APPLICABLE` trace with exact
   authority evidence rather than constructing a hypothetical product or
   route.

These decisions resolve the interpretation conflict; they do not erase the
repository inconsistency. Resume the remaining implementation-blind,
read-only horizontal audit at exact commit
`5f0fc20042b88fb6cd883c92d1b59b7f22832901`, tree
`97a4d908061e51418f93afc1d97d27433af441b8`.

Within the already authorized audit artifacts:

- record each of the four findings as **OWNER-RESOLVED /
  SHARED-SOURCE-REPAIR-REQUIRED**;
- identify the exact conflicting shared-conventions clauses and the exact
  frozen authority supporting the disposition;
- specify the necessary clause-level repair without editing the
  shared-conventions file;
- ensure the product graph, conformance matrix, findings record, report,
  representative traces, and verifier use the disposition consistently;
- retain explicit negative traces for unauthorized JINC roles and routes;
- complete any remaining independent audit work and rerun
  `verify_horizontal_audit.py`; and
- recommend a separate, narrowly scoped follow-on work order to repair
  `doc/SCIENTIFIC_CONVENTIONS.md`.

Do not alter SCI-MAP, SCI-JINC, any other frozen package,
`doc/SCIENTIFIC_CONVENTIONS.md`, application code, validation products,
canonical refs, FRUIT, or ALIGN under this work order. Do not install
dependencies or broaden the task to address the missing `reportlab`
environment. Keep all changes confined to the existing audit directory and
uncommitted.

Do not downgrade the four findings merely because an owner interpretation has
now been supplied. The final disposition should distinguish between:

- scientific/package coherence under the recorded owner disposition; and
- the outstanding repository documentation repair.

Do not issue an unqualified PASS while the conflicting shared-conventions
clauses remain unrepaired. If any additional consequential authority conflict
appears while completing the audit, apply the stop-and-escalate rule again
rather than selecting another interpretation.

Return the completed audit status, revised finding states and counts, verifier
result, exact affected shared-conventions clauses, recommended follow-on repair
scope, artifact paths, and the required nonmutation statement. Make no
implementation, validation, performance, readiness, production, activation,
or Unity claim.
