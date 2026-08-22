# Six-Package Owner Decision Ledger

## Scope

This ledger contains only unresolved decisions whose consequences cross at least one boundary among SCI-ALIGN, SCI-AST, SCI-RTC, SCI-CAL, SCI-PTC, SCI-VAL, or the downstream SCI-MAP reference. It does not resolve a question, author a missing policy, or convert an external dependency into package authority. Existing package-local decision IDs are retained where one already exists.

## Finding-to-decision trace

| Finding | Cross-package decision | Existing package decision / authority |
|---|---|---|
| F-001 | XOD-001 | PTC-AUTH-D027 and frozen PTC clauses conflict |
| F-002 | XOD-002 | PTC scientific owner |
| F-003 | XOD-003 | VAL Source Binding Register authority |
| F-004 | XOD-004; XOD-005 | PTC and MAP use owners; VAL Registry binds only |
| F-005 | XOD-006 | Canonical profile owner and VAL Registry |
| F-006 | XOD-008 | RTC and AST scientific owners |
| F-007 | XOD-009 | AST plus geometry/APT owner |
| F-008; F-009; F-010 | — package-local source corrections, not collapsed into a cross-package policy | RTC, PTC, and AST scientific owners respectively |
| F-011 | XOD-017 | PTC scientific owner |
| F-012; F-021 | XOD-011 | SCI-MAP-OD-003 plus every affected response/use owner |
| F-013 | XOD-012 | SCI-MAP-OD-004 plus upstream uncertainty owners |
| F-014 | XOD-010 | SCI-MAP-OD-008 |
| F-015 | XOD-007 | Aggregate/coadd use owner and VAL Registry |
| F-016; F-022 | XOD-016 | CAL, VAL, MAP, ALIGN, and AST source authorities as applicable |
| F-017 | XOD-015 | Named external producers |
| F-018 | XOD-013 | MAP plus route/coefficient/policy owners |
| F-019 | XOD-018 | Cross-package exposure owner assignment |
| F-020 | XOD-004; XOD-005 | CAL fact owner plus each PTC/MAP use owner |
| F-023 | XOD-014 | PTC-OD-010 and MAP estimator-use owner |
| F-024 | XOD-019 | SCI-MAP-OD-007 |
| F-025 | XOD-020 | SCI-MAP-OD-009 |

## Priority 0 — internally consistent frozen authority

### XOD-001 — Exact PTC transformed-signal identity

- **Exact question:** In frozen SCI-PTC v0.1/r0.4, is `U_hat` in `Z=Y-U_hat` intended to include the separately learned nonrestored location `lambda`, or must the transformed-signal equation be replaced by an explicit operation on `Y-lambda`?
- **Alternatives:** (a) redefine `U_hat` and every dependent response/null-space statement to include `lambda`; (b) amend the subtraction identity to compose nonrestoring centering and the frozen subspace explicitly; (c) select a different complete identity in a versioned successor.
- **Scientific consequences:** Determines transformed values, additive null space, fixed-state derivative, full-procedure response, companion propagation, and replay.
- **Conservative state while open:** PTC-dependent transformed signal and exact PTC response are scientifically unavailable for cross-package handoff.
- **Affected packages:** SCI-PTC, SCI-VAL, SCI-MAP; response consumers.
- **Affected profile invariants:** PTC operator application, output retention, response companion, MAP upstream admission.
- **MAP/downstream consequence:** No exact PTC-parented signal/response bundle.
- **Authority required:** SCI-PTC scientific owner through an explicit reopening or versioned successor; no audit-side textual harmonization.

### XOD-002 — Complete PTC named-use truth rule

- **Exact question:** How shall `b_U=false` be dispositioned when every listed permission predicate is true, and what exact knowledge algebra governs true, false, unknown, and conflict in the PTC composite?
- **Alternatives:** (a) add `NOT b_U` to exclusion and adopt a declared four-state evaluation; (b) replace the pair of Boolean equations with an owner-approved decision table; (c) delegate the full policy proposition to registered use-specific profiles and retain only producer facts in PTC.
- **Scientific consequences:** Determines fit, application, output, coefficient, response, and population admission without manufactured eligibility.
- **Conservative state while open:** Every decision using `eq:cause-support` is decision unavailable.
- **Affected packages:** SCI-PTC, SCI-VAL, SCI-MAP.
- **Affected profile invariants:** all PTC named-use profiles and MAP upstream admission.
- **MAP/downstream consequence:** No usable PTC policy decision.
- **Authority required:** SCI-PTC scientific owner; VAL may execute but not author the rule.

## Priority 1 — source and profile integrity

### XOD-003 — Current VAL source bindings

- **Exact question:** What exact version/digest and compatibility statement shall replace each stale ALIGN, AST, RTC, CAL, and MAP row in the VAL Source Binding Register?
- **Alternatives:** (a) bind the exact consolidated versions/digests and declare compatibility; (b) bind a versioned adapter; (c) declare the affected profile incompatible/unavailable.
- **Scientific consequences:** Establishes whether VAL can pass its structural gate and replay a decision.
- **Conservative state while open:** applicability unknown and decision unavailable for every affected import.
- **Affected packages:** SCI-ALIGN, SCI-AST, SCI-RTC, SCI-CAL, SCI-VAL, SCI-MAP.
- **Affected profile invariants:** independent exposure, all PTC profiles, MAP admission, aggregate profiles.
- **MAP/downstream consequence:** no source-current VAL decision enters the MAP bundle.
- **Authority required:** SCI-VAL Registry/source-binding authority with each imported scientific owner supplying compatibility authority.

### XOD-004 — PTC use-profile registrations

- **Exact question:** What are the actual owner-approved propositions for basis-fit admission, loading-fit admission, frozen-operator application, output retention, coefficient/QC population, response-companion admission, and empirical/simulation population?
- **Alternatives:** for each use, register a complete distinct profile; declare the use unsupported; or replace several names only if the owner proves exact proposition/domain equivalence.
- **Scientific consequences:** Determines which occurrences may affect learning, application, output, coefficients, response, or empirical populations while preserving cause and lifecycle distinctions.
- **Conservative state while open:** reserved names are unusable and requested decisions are unavailable.
- **Affected packages:** SCI-PTC and SCI-VAL; producers supplying facts; MAP for output/coefficient uses.
- **Affected profile invariants:** the seven reserved PTC profile families.
- **MAP/downstream consequence:** no policy-authorized PTC output or coefficient can contribute.
- **Authority required:** SCI-PTC scientific owner for PTC uses; Registry binds but does not invent them.

### XOD-005 — MAP upstream-admission profile

- **Exact question:** What exact upstream proposition, fact set, response/uncertainty roles, cause handling, exception authority, domain, and missing behavior define `SCI-MAP:map_upstream_admission`?
- **Alternatives:** register one complete MAP-owned atomic profile; declare ordinary MAP admission unsupported; or define multiple explicitly different MAP use profiles rather than aliases.
- **Scientific consequences:** Separates package-local validity from MAP-use eligibility while leaving numerical contribution and final validity MAP-owned.
- **Conservative state while open:** no MAP admission eligibility proposition exists.
- **Affected packages:** all six producers/VAL plus SCI-MAP.
- **Affected profile invariants:** MAP upstream admission and every compatible aggregate/coadd profile.
- **MAP/downstream consequence:** hard stop before MAP numerical action.
- **Authority required:** SCI-MAP scientific owner; VAL Registry binding/evaluation only.

### XOD-006 — Complete independent-exposure registry record

- **Exact question:** Is `SCI-VAL:independent_exposure@1` atomic-only, and what explicit aggregation/propagation compatibility or `not_applicable` statement completes its Registry record at the current source versions?
- **Alternatives:** complete and supersede the row; create a new version with exact atomic-source compatibility; or withdraw usability while retaining the underlying invariant in package facts.
- **Scientific consequences:** Determines registry integrity and prevents an atomic origin decision from being reused as an aggregate policy.
- **Conservative state while open:** the nominal profile is not treated as a registry-complete usable input.
- **Affected packages:** SCI-ALIGN, SCI-RTC, SCI-VAL and downstream consumers.
- **Affected profile invariants:** direct representative synthesized/replaced nonexceptionable restriction.
- **MAP/downstream consequence:** independent-exposure status cannot be treated as a current reusable MAP-policy input.
- **Authority required:** policy owner named in the profile plus SCI-VAL Registry authority and current source owners.

### XOD-007 — Aggregate and observation-coadd policy

- **Exact question:** What exact aggregate/coadd profile owns population, time support, atomic source-profile compatibility, denominator, missing handling, operator, threshold/polarity, uncertainty, binding state, and reverse-propagation generation?
- **Alternatives:** register a homogeneous atomic-source aggregate profile; register a separate owner-approved transformation into a common proposition; or leave coadd admission unavailable.
- **Scientific consequences:** Prevents heterogeneous-profile fractions and retroactive changes to atomic decisions.
- **Conservative state while open:** no VAL aggregate or coadd eligibility; MAP still atomically rejects incompatible bundles.
- **Affected packages:** SCI-VAL, SCI-MAP and any observation-bundle producers.
- **Affected profile invariants:** aggregate decision, observation-coadd admission, lifecycle generation.
- **MAP/downstream consequence:** no policy-authorized coaddition.
- **Authority required:** actual aggregate/coadd use owner; SCI-VAL Registry/evaluator; SCI-MAP retains estimator compatibility and atomicity.

## Priority 2 — exact coordinate and numerical MAP boundary

### XOD-008 — RTC-to-AST sample-grid boundary artifact

- **Exact question:** Which approved versioned boundary body composes the RTC `n`-grid facts with AST's `SCI-AST:rtc_output_grid_coordinates@1` parent requirements?
- **Alternatives:** approve a new exact boundary artifact; identify an already committed equivalent and prove identity; or keep the RTC-grid AST role unavailable.
- **Scientific consequences:** Binds representative ALIGN slot, output time, phase/delay, segment, full support, response, correction state, status, and provenance without reconstructing RTC.
- **Conservative state while open:** no source-closed AST RTC-grid role for a MAP handoff.
- **Affected packages:** SCI-RTC, SCI-AST, SCI-MAP.
- **Affected profile invariants:** coordinate identity, full RTC response support, MAP coordinate parent.
- **MAP/downstream consequence:** exact coordinate parent blocked.
- **Authority required:** joint RTC/AST boundary owners; package mathematics remains unchanged.

### XOD-009 — Detector geometry/field-rotation boundary

- **Exact question:** What exact measured-geometry/APT realization, occurrence association, representation, rotation law, pivot/gauge, application count, support, and covariance artifact is authoritative for AST?
- **Alternatives:** bind an owner-approved external boundary; bind a proven versioned equivalence transform; or leave dependent coordinates unavailable.
- **Scientific consequences:** Determines detector direction, field rotation, WCS, uncertainty, and nonduplicate composition.
- **Conservative state while open:** no inference from APT row, design identity, field name, or plausible numeric values.
- **Affected packages:** external SCI-BEAM/APT/TolProj authorities, SCI-AST, SCI-PTC, SCI-MAP.
- **Affected profile invariants:** AST coordinate validity; PTC masks/grouping; MAP coordinate admission.
- **MAP/downstream consequence:** geometry-dependent MAP coordinates blocked.
- **Authority required:** measured-geometry and association owners plus AST boundary approval.

### XOD-010 — MAP projection classes and `G_pi` (`SCI-MAP-OD-008`)

- **Exact question:** Which projection classes, normalization/conservation rules, boundary-loss semantics, and upstream materializer are authorized for ordinary MAP v0.1?
- **Alternatives:** one-hot only; specified fractional classes; another explicitly defined class; each with exact support and edge rules—or no numerical route.
- **Scientific consequences:** Defines deposition, estimand, normalization, response, covariance, hits, exposure placement, and boundary behavior.
- **Conservative state while open:** AST supplies only continuous pixel/optional nominal containing pixel; no `G_pi` or numerical contribution.
- **Affected packages:** SCI-AST and SCI-MAP; PTC coefficient and VAL admission remain separate.
- **Affected profile invariants:** MAP projection parent, contribution set, support, response/covariance.
- **MAP/downstream consequence:** hard numerical-route stop.
- **Authority required:** SCI-MAP scientific owner; AST may materialize only the resolved request.

### XOD-014 — Exact MAP-facing analysis coefficient (`PTC-OD-010`)

- **Exact question:** Which PTC or other upstream analysis/gridding coefficient family supplies MAP's positive scalar `omega_i`, with what index, statistic, factors, unit, normalization, support, lifecycle, and prohibited interpretations?
- **Alternatives:** define one PTC-owned family; identify another explicit producer-owned family; or leave ordinary contribution unavailable.
- **Scientific consequences:** Establishes estimator membership and normalization without calling the coefficient precision, exposure, support, or significance.
- **Conservative state while open:** no coefficient; no MAP contribution even with signal and coordinate.
- **Affected packages:** SCI-PTC, SCI-VAL, SCI-MAP.
- **Affected profile invariants:** coefficient/QC profile, MAP admission, contribution, precision separation.
- **MAP/downstream consequence:** hard numerical-route stop independent of `G_pi`.
- **Authority required:** SCI-PTC owner for a PTC family or the explicitly named alternative producer; MAP owns its estimator use.

### XOD-018 — Exposure carrier into the MAP bundle

- **Exact question:** Which package transparently carries or transforms ALIGN physical/valid-original exposure through RTC, CAL, and PTC into upstream-eligible and retained MAP exposure, and what exact parent/support/accounting rule applies?
- **Alternatives:** designate transparent preservation through intermediate packages; define an explicit retained-exposure producer at one stage; or leave MAP exposure unavailable.
- **Scientific consequences:** Keeps physical acquisition, valid-original exposure, synthesized support, retained exposure, weight, and hit count distinct.
- **Conservative state while open:** MAP exposure members and exposure-dependent decisions are unavailable; they are not reconstructed from duration or weight.
- **Affected packages:** SCI-ALIGN, SCI-RTC, SCI-CAL, SCI-PTC, SCI-VAL, SCI-MAP.
- **Affected profile invariants:** PTC output retention, MAP admission, exposure projection, coadd aggregation.
- **MAP/downstream consequence:** exposure products/claims blocked even if a signal-only record exists.
- **Authority required:** cross-package scientific owner assignment, with each transforming stage declaring its exact relation.

### XOD-019 — MAP support-policy numerical domain (`SCI-MAP-OD-007`)

- **Exact question:** What numerical domain, boundary-case disposition,
  recommended range, effective-policy authority, and failure behavior govern
  dimensionless `coverage_cut`?
- **Alternatives:** approve one exact closed domain and boundary rule; approve
  distinct versioned domains for distinct profiles; or leave every numerical
  value unauthorized.
- **Scientific consequences:** Changes support-authorized output membership and
  therefore signal, response, covariance, exposure, validity, and publication.
- **Conservative state while open:** only a value explicitly admitted by an
  owner-authorized effective policy may be used; otherwise fail before support
  rows or required-product mutation.
- **Affected packages:** SCI-MAP, SCI-VAL profile binding, and every upstream
  handoff whose samples would enter MAP support.
- **Affected profile invariants:** MAP admission, support-policy population,
  required products, transition/replay.
- **MAP/downstream consequence:** no numerical ordinary route under an
  unadmitted `coverage_cut`; no default range is inferred.
- **Authority required:** SCI-MAP scientific owner; profile owner binds the
  resolved policy but does not invent it.

### XOD-020 — Canonical-grid preparation and future reprojection (`SCI-MAP-OD-009`)

- **Exact question:** Is crop/pad to a canonical common grid authorized for
  observation coadd, who owns it, and who owns future reprojection or mosaic
  operators beyond centered-integer placement?
- **Alternatives:** authorize an exact preparation operator/profile; retain
  strict rejection and define a separate future transform package; or leave
  grid-changing products unavailable.
- **Scientific consequences:** Determines identity, WCS, response, covariance,
  support, validity, provenance, and atomic coadd compatibility after any grid
  change.
- **Conservative state while open:** reject odd-difference or otherwise
  incompatible bundles; MAP performs no crop, pad, fractional shift,
  reprojection, interpolation, or mosaic.
- **Affected packages:** SCI-MAP, SCI-VAL aggregate/coadd policy, and any future
  grid-transform producer.
- **Affected profile invariants:** common-grid identity, coadd admission,
  response/covariance propagation, lifecycle.
- **MAP/downstream consequence:** incompatible-grid coadd and future
  reprojection/mosaic are blocked; an otherwise compatible single-observation
  map is not blocked by this decision alone.
- **Authority required:** SCI-MAP scientific owner and the future named
  grid-transform scientific owner.

## Priority 3 — response, uncertainty, alternate route, and source closure

### XOD-011 — Response-unavailable MAP use (`SCI-MAP-OD-003`)

- **Exact question:** Must every scientifically usable map have a realized response, or which exact restricted consumers and claims may accept a typed response-unavailable bundle?
- **Alternatives:** response required universally; an owner-defined restricted consumer envelope; or product-role-specific requirements.
- **Scientific consequences:** Determines bundle completeness and whether a finite signal can support any response-independent claim.
- **Conservative state while open:** all response-dependent consumers reject; no general usability claim is made.
- **Affected packages:** upstream response producers, SCI-VAL profiles, SCI-MAP and generic consumers.
- **Affected profile invariants:** PTC response companion, MAP admission, consumer envelope.
- **MAP/downstream consequence:** complete-response handoff remains blocked.
- **Authority required:** SCI-MAP scientific owner and each downstream use owner.

### XOD-012 — Minimum covariance/uncertainty representation (`SCI-MAP-OD-004`)

- **Exact question:** What conditional covariance representation must persist or remain lineage-resolvable, and which calibration, astrometric, response, selection, model, atmosphere, beam, and cross-observation terms are required for each claim?
- **Alternatives:** persisted matrix/operator; structured lossless factorization; owner-approved summary plus exact lineage; or typed unavailable for claims needing more.
- **Scientific consequences:** Determines which covariance, precision, standardized-signal, and total-uncertainty statements are allowed.
- **Conservative state while open:** component-limited conditional claims only; normalization is not precision; omitted terms are unavailable, not zero.
- **Affected packages:** all six and SCI-MAP; NOI/generic uncertainty consumers.
- **Affected profile invariants:** response/uncertainty structural gates, MAP admission, coadd compatibility.
- **MAP/downstream consequence:** no total uncertainty, significance, or universal precision claim.
- **Authority required:** SCI-MAP owner for bundle representation; upstream owners for their terms; downstream use owners for stronger needs.

### XOD-013 — Separately governed direct CAL-to-MAP route

- **Exact question:** Is a PTC-bypassing CAL-to-MAP route authorized, and if so what signal parent, coefficient, VAL profile, response, uncertainty, coordinate, support, and provenance contract governs it?
- **Alternatives:** authorize a distinct complete route; explicitly prohibit it in this generation; or leave it outside scope/unavailable.
- **Scientific consequences:** Determines whether PTC disabled is terminal or merely selects a different fully named estimator route.
- **Conservative state while open:** PTC disabled produces no PTC-route MAP product and does not enable a CAL-only product.
- **Affected packages:** SCI-CAL, SCI-PTC, SCI-VAL, SCI-AST, SCI-MAP.
- **Affected profile invariants:** route identity, map admission, coefficient, response, disabled state.
- **MAP/downstream consequence:** no fallback numerical route.
- **Authority required:** route scientific owner(s), including MAP and the upstream coefficient/policy owners.

### XOD-015 — External producer-boundary identities

- **Exact question:** Which immutable owner-approved sources bind Tune/readout mapping, telescope clock/event semantics, observing fields, pointing support, center, geometry/APT, beam/flxscale/nominal beam, atmosphere observational inputs, frame/EOP/refraction, and source/calibrator models for each dependent role?
- **Alternatives:** supply exact named boundary artifacts; declare a role not applicable; or keep the dependent fact unavailable.
- **Scientific consequences:** Determines input identity, timing, coordinates, calibration, response, masks, and uncertainty without reconstruction.
- **Conservative state while open:** each affected role fails closed at its declared scope; unrelated signal or coordinate facts are preserved.
- **Affected packages:** all six plus external owners and SCI-MAP.
- **Affected profile invariants:** all profiles whose structural gates import these facts.
- **MAP/downstream consequence:** one or more logical handoff members may be unavailable.
- **Authority required:** each named external scientific owner; this audit cannot validate an unsupplied external contract.

### XOD-016 — Freeze and final source-manifest closure

- **Exact question:** What exact canonical text/PDF versions and digests constitute frozen SCI-CAL, SCI-VAL, and SCI-MAP scientific authority after their remaining reviews and decisions?
- **Alternatives:** freeze the current active revisions after resolving blockers; issue successor revisions; or retain active/nonfrozen status.
- **Scientific consequences:** Establishes exact replay and caps whether the cross-package handoff can be authoritative rather than a draft.
- **Conservative state while open:** only a non-authoritative audit/draft handoff may be produced.
- **Affected packages:** SCI-CAL, SCI-VAL, SCI-MAP and all consumers binding them.
- **Affected profile invariants:** source binding, registry, MAP bundle version.
- **MAP/downstream consequence:** no frozen MAP handoff profile.
- **Authority required:** each package scientific owner and final source-manifest authority.

### XOD-017 — Complete PTC disabled product-state mapping

- **Exact question:** Which exact PTC learned, fitted, centering, removed-component, applied, coefficient, transformed, response, covariance, and publication states exist or do not exist when PTC is disabled?
- **Alternatives:** expand `eq:disabled-route` to every named role while retaining orthogonal response/covariance axes; replace it with a role-indexed table; or issue a successor definition.
- **Scientific consequences:** Makes disabled replay exhaustive without conflating no product with unavailable response.
- **Conservative state while open:** no PTC-dependent numerical product or MAP route; omitted roles are not inferred as realized.
- **Affected packages:** SCI-PTC, SCI-VAL, SCI-MAP.
- **Affected profile invariants:** disabled route, product realization, response/covariance axes.
- **MAP/downstream consequence:** no false fallback or completion marker.
- **Authority required:** SCI-PTC scientific owner.

## Dependency order

The shortest defensible dependency sequence is: resolve XOD-001/XOD-002 and
the package-local source conflicts; bind coordinate/external authorities under
XOD-008/XOD-009/XOD-015; update source bindings under XOD-003; complete registry
records under XOD-004--007; resolve coefficient/exposure/projection/support
under XOD-014/XOD-018/XOD-010/XOD-019; resolve the alternate route, response,
uncertainty, disabled, and coadd/grid scopes under
XOD-013/XOD-011/XOD-012/XOD-017/XOD-020; then complete the affected package
reviews and final freezes under XOD-016. Freeze may proceed earlier only for a
package whose own prerequisites are complete; final cross-package freeze cannot
precede the authorities it binds. Parallel work is possible, but no later
decision supplies authority for an earlier missing fact.
