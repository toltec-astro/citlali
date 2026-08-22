# SCI-VAL Profile Coverage And Binding Report

Audit identity: `SIX_PACKAGE_WIDE_SCALE_55EFD8A`

Immutable library commit: `55efd8a54464636a24e621f6d1b60486d235b20e`

Prepared: `2026-08-22`

Scope: contract congruence only. This report does not inspect or validate an
implementation, representation, schema, test, reduction, observation,
production product, or operational deployment.

## 1. Executive Disposition

SCI-VAL Core is internally coherent on the authority split, four independent
decision axes, open-world knowledge, structural-conflict handling,
false-dominant restriction composition, cause preservation, nonretroactive
lifecycle, and homogeneous aggregation mechanics.

The current Profile Registry and source bindings are **not congruent for
wide-scale cross-package evaluation** at the audited snapshot:

1. the sole nominally registered profile,
   `SCI-VAL:independent_exposure@1`, omits an explicit aggregation/propagation
   compatibility field or an explicit `not_applicable`, although the Registry
   Rule requires that field for every usable record;
2. that profile's RTC-origin source binding still names SCI-RTC v0.1/r0.9,
   while the supplied current cross-package fact is frozen SCI-RTC v0.1/r0.12;
3. the CAL binding remains v0.1/r0.3 while the supplied current CAL sources are
   rationale r0.5 and ECS r0.4;
4. ALIGN and AST remain a combined meaning-only, unversioned source row even
   though the supplied current facts say each is frozen at v0.1/r0.3;
5. all package-qualified PTC profiles, MAP upstream admission, diagnostic
   display, and every aggregate/coaddition proposition are unbound or absent.

Under SCI-VAL's own fail-closed rules, these omissions and source mismatches
must not be repaired by inference. They produce unavailable evaluation at the
affected scope until new immutable bindings are supplied and compatibility is
reviewed.

## 2. Authority And Evidence Boundary

All SCI-VAL evidence anchors below are exact current package paths and clause
identities at the immutable commit. The current adjacent-package revision facts
were supplied by the wide-scale audit coordinator and were not independently
reopened in this pass.

The governing split is:

| Layer | Contract authority | Exact evidence |
| --- | --- | --- |
| Producer | Owns atomic fact truth, causes, explicit negatives, local composites/supports, influence, response/uncertainty availability, parentage, and lifecycle. | `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/requirements.tex::SCI-VAL-REQ-001--007`; `SCOPE_OWNERSHIP_DECISION_R0.2.md::Decision items 1--2` |
| Named-use owner | Owns applicability, restrictions, exceptions, thresholds, aggregation/propagation, failure scope, compatibility, and admission for its exact use. | `src/common/definitions.tex::Core, registry, and scientific ownership`; `SCI-VAL-REQ-002` |
| Profile Registry | Binds immutable profile identity to actual owner, source, domain, restrictions, exceptions, compatibility, roles, and missing behavior; supplies no predicate. | `PROFILE_REGISTRY.md::Registry Rule`; `SCI-VAL-REQ-043--044` |
| VAL Core | Owns shared types, knowledge/four-axis logic, immutable provenance, cause preservation, deterministic supplied-profile evaluation, aggregation mechanics, and replay. | `src/common/definitions.tex::Two layers, one authority boundary`; `SCI-VAL-REQ-001--002,043` |
| Consumer | Owns estimator-specific numerical action, contribution, response/covariance, normalization, coaddition, recurrence, and final product validity. | `src/common/definitions.tex::Downstream authority remains downstream`; `SCI-VAL-REQ-035--036` |

Registration does not transfer policy ownership. A namespace records where a
profile is registered, not who owns its scientific policy.

## 3. Core Decision Semantics

### 3.1 Four axes

The exact axes are:

- request: `requested | not_requested`;
- applicability: `applicable | inapplicable | applicability_unknown`;
- eligibility: `eligible | ineligible | decision_unavailable`;
- realization: `realized | incomplete | failed | not_produced`.

Not-requested and known-inapplicable cases instantiate no eligibility
proposition (`emptyset_E`). A failed/unproduced artifact does not manufacture
an eligibility disposition. Exact authority:
`src/common/notation.tex::Four independent decision axes`,
`src/common/equations.tex::Disposition function`, and
`SCI-VAL-REQ-011--014`.

### 3.2 Knowledge, conflict, and exception behavior

The open-world knowledge states are `T`, `F`, `U`, and `C`: authoritative
positive, authoritative explicit negative, unknown/missing, and
contradictory/ambiguous/out-of-domain. Silence is `U`; only the owning producer
under a declared complete cause family may assert explicit absence. Exact
authority: `src/common/notation.tex::Knowledge is not Boolean silence` and
`SCI-VAL-REQ-004--005`.

Conflict precedence is scope-sensitive:

- structural identity, parentage, owner/source, registry, compatibility, or
  applicability conflict gives `applicability_unknown` and
  `decision_unavailable` before consumer mutation;
- after the domain is established, a known decisive exclusion remains
  `ineligible` despite unrelated non-gating unknown/conflict;
- without a decisive exclusion, a required unknown/conflict gives
  `decision_unavailable`;
- all conflict evidence remains in the immutable reason record.

Exact authority: `VAL-R03-D005`, `SCI-VAL-REQ-014--015,037`, and
`SCI-VAL-PRED-006--008`.

Only a resolved, permitted, same-profile exception can transform its named
exceptionable restriction. Exception uncertainty/conflict cannot neutralize
the restriction. Changed exception resolution changes resolved-profile lineage
and decision identity. The direct-origin invariant in
`SCI-VAL:independent_exposure@1` is non-exceptionable. Exact authority:
`SCI-VAL-REQ-017,045` and `SCI-VAL-PRED-010,019`.

### 3.3 Cause and influence preservation

Causes and influence accumulate as a commutative, associative, idempotent
set/graph union without erasure or global ranking. Exact/conservative coverage
and confirmed/possible epistemic status remain distinct. A conservative edge
cannot be promoted to exact. The use owner, not VAL, maps possible influence to
permit, exclude, scientific unavailability, or review. Review is action
metadata, not a fourth eligibility disposition. Exact authority:
`src/common/equations.tex::No rescue, cause preservation, and determinism`,
`Exact and conservative causal influence`, and `SCI-VAL-REQ-007,018,021--022`.

### 3.4 Response and uncertainty roles

The closed owner-supplied role set is:

- `structural_gate`;
- `required_permission`;
- `decisive_exclusion`;
- `advisory`.

VAL applies the deterministic role semantics but does not select or reinterpret
the role. Unavailable response/uncertainty is never identity response or zero
uncertainty. Exact authority: `VAL-R03-D006`,
`src/common/equations.tex::Response, uncertainty, and payload ordering`, and
`SCI-VAL-REQ-025`.

## 4. Profile Coverage

### 4.1 Nominally registered canonical profile

| Field | Current contract state | Congruence finding |
| --- | --- | --- |
| Registry key | `SCI-VAL:independent_exposure@1` | Current canonical key; former `VAL.core.independent_exposure@1` has no alias. |
| Use | `independent_exposure` | Bound. |
| Owner | Grant Wilson as owner of the contract-level invariant | Bound; namespace does not transfer ownership. |
| Source | r0.3/r0.2 owner directives and formal clauses | Bound at the VAL package level. |
| Domain | Exact sample-detector occurrence with authoritative representative-source identity and origin | Bound. |
| Decisive invariant | Synthesized/replaced exact representative source is not an original independent astronomical exposure | Bound and non-exceptionable. |
| Exceptions | None for the decisive invariant | Bound. |
| Compatibility | Direct-origin invariant, domain identity, and no-exception rule must remain exact | Bound in general terms. |
| Response/uncertainty roles | No role imposed by the direct-origin invariant | Bound. |
| Missing/conflict behavior | Structural/origin uncertainty gives unavailable; authoritative synthesized/replaced origin gives ineligible | Bound. |
| Aggregation/propagation compatibility | **No explicit row and no explicit `not_applicable`** | **Internal registry-completeness conflict.** `PROFILE_REGISTRY.md::Registry Rule` requires this field for every usable profile, and missing binding makes a profile unavailable. |
| RTC origin-source binding | SCI-RTC v0.1/r0.9 | **Stale** relative to supplied frozen SCI-RTC v0.1/r0.12. `SOURCE_BINDING_REGISTER.md::SCI-RTC row` requires new binding and compatibility review when origin semantics/source change. |

Disposition: the profile remains the sole nominally registered scientific
proposition and its direct-origin invariant remains normative, but its current
registry record is **not demonstrably usable/congruent** under the Registry
Rule until the missing aggregation/propagation field is made explicit and the
RTC source binding is reviewed against r0.12. This report does not infer either
answer.

### 4.2 Reserved or absent profiles

| Profile | Registry state | Consequence |
| --- | --- | --- |
| `SCI-PTC:basis_fit_admission` | Reserved, unbound | Cannot evaluate basis-fit admission. |
| `SCI-PTC:loading_fit_admission` | Reserved, unbound | Cannot infer from basis-fit policy. |
| `SCI-PTC:operator_application` | Reserved, unbound | Fit exclusion cannot define application by inference. |
| `SCI-PTC:output_retention` | Reserved, unbound | Output action remains stage-specific. |
| `SCI-PTC:coefficient_qc_population` | Reserved, unbound | Not an estimator-fit alias. |
| `SCI-PTC:response_companion` | Reserved, unbound | Response role/predicates remain owner-supplied. |
| `SCI-PTC:empirical_or_simulation_population` | Reserved, unbound | No population policy is inferred. |
| `SCI-MAP:map_upstream_admission` | Reserved, unbound | MAP boundary meaning is known, but admission is unavailable. |
| `<PACKAGE>:diagnostic_display` | Namespace template only | Display permission supplies no stronger use. |
| Any aggregate/coaddition profile | Absent | No detector aggregate, coaddition-admission, or reverse-propagation proposition is evaluable. |

Exact authority: `PROFILE_REGISTRY.md::Package-Qualified Names Reserved But
Not Bound Here`, `Aggregate Profile Rule`, `SCI-VAL-REQ-009,030,044,046--047`,
and `SCI-VAL-PRED-003,021,023`.

The former broad label `analysis_or_gridding_contribution` is not a v0.1 key
and has no automatic alias. The current MAP-facing upstream question is
`SCI-MAP:map_upstream_admission`; numerical contribution remains MAP-owned.

## 5. Source-Binding Congruence

The comparison below preserves the current VAL row and the supplied current
wide-scale state separately.

| Producer/source | VAL source-binding register | Supplied current wide-scale fact | Congruence | Contract consequence |
| --- | --- | --- | --- | --- |
| SCI-RTC | v0.1/r0.9 frozen authority | v0.1/r0.12 frozen authority | **Stale / non-congruent** | New binding plus compatibility review of `SCI-VAL:independent_exposure@1`; no reinterpretation of representative-source/origin semantics. |
| SCI-CAL | v0.1/r0.3 architecture-frozen rationale; scientific authority not frozen | current rationale r0.5 and ECS r0.4 | **Stale / non-congruent** | CAL-dependent profiles remain unavailable until exact factor/domain/binding/atmosphere/response/uncertainty semantics are rebound. |
| SCI-PTC | v0.1/r0.4 frozen authority | v0.1/r0.4 current/frozen | **Source revision congruent** | Stage meanings may remain bound; every PTC policy profile is nevertheless unbound. |
| SCI-MAP | v0.1/r0.3 house rationale; scientific authority not frozen | v0.1/r0.3 current; scientific authority not frozen | **Version/status congruent, authority provisional** | Boundary separation may be retained; exact MAP upstream-admission profile remains unavailable. |
| SCI-ALIGN | Combined ALIGN/AST/TEL meaning-only row; no standalone frozen version | SCI-ALIGN v0.1/r0.3 frozen | **Missing exact binding** | New standalone owner/source/version binding and compatibility consequence required before an exact profile may depend on it. |
| SCI-AST | Combined ALIGN/AST/TEL meaning-only row; no standalone frozen version | SCI-AST v0.1/r0.3 frozen | **Missing exact binding** | New standalone owner/source/version binding and compatibility consequence required before an exact profile may depend on it. |
| TEL | Combined meaning-only row; no standalone frozen version | No new current fact supplied in this audit lane | **Unresolved** | Remains meaning-only; no exact compatibility claim. |

Exact VAL authority: `SOURCE_BINDING_REGISTER.md::Purpose`, producer rows, and
`Binding Rule`; `SCI-VAL-REQ-008,029,049`; `SCI-VAL-PRED-020,024`.

## 6. Aggregation And Coaddition Coverage

SCI-VAL Core requires every aggregate to bind its own registered profile,
actual owner/source, aggregate object/domain, exact homogeneous atomic source
profile, population/time support, four-axis counts, denominator, missing
treatment, operator, threshold/polarity, response/uncertainty treatment,
propagation authority, failure scope, and lifecycle generation.

Base v0.1 aggregation is homogeneous in exact profile identity/version,
lifecycle stage, object type, and applicability domain. Empty/unknown
denominators are unavailable or inapplicable, never a valid zero fraction.
Partition interchange requires an owner-declared sufficient summary,
associative combine rule, and exact equivalence. Reverse propagation creates
generation `k+1` and cannot rewrite or feed its generation-`k` denominator.

No aggregate or coaddition profile is registered. Therefore individually
eligible occurrences or observations cannot be promoted into an aggregate or
coaddition decision by VAL Core alone. MAP retains coaddition and final map
validity. Exact authority: `src/common/equations.tex::Homogeneous aggregation
and anti-circular propagation`, `SCI-VAL-REQ-030--035,047--048`, and
`SCI-VAL-PRED-013--014,021--022`.

## 7. Package Producer-Fact Coverage

| Producer | Facts admitted by current VAL boundary | Coverage finding |
| --- | --- | --- |
| RTC | Conditioned-x identity/grid/representative source; origin; direct causes; operator controls; support; exact/conservative influence; response/uncertainty availability; lifecycle. | Semantics represented, but source binding stale at r0.9. |
| CAL | Detector/sample and RTC parent; factor/domain, detector binding, atmosphere, response, calibration availability/validity, conditional uncertainty/correlation scope. | Semantics represented, but source binding stale and scientific authority conditional. |
| PTC | CAL parent; basis/loading fit, application, output, coefficient/QC, response, empirical/simulation and downstream-support facts; staged lifecycle. | Source binding matches r0.4; no exact PTC-owned policy profile registered. |
| MAP | Boundary between upstream admission, projection, contribution, retained exposure, supports, response/covariance, coaddition, and final validity. | Source version/status matches supplied state; exact admission profile absent. |
| ALIGN | Time, association, origin/synthesis and local validity roles only. | Exact frozen r0.3 source not bound. |
| AST | Coordinate/frame/association and local validity roles only. | Exact frozen r0.3 source not bound. |

Producer facts remain producer-owned. VAL cannot reconstruct a local support
from causes, promote missing facts, or invent policy predicates. Exact
authority: `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`,
`SCI-VAL-REQ-001--007,038,049`.

## 8. Downstream Consumer Action Ownership

- PTC retains fit/application/output/coefficient/QC support and action scope.
  Only a PTC-owned fit-support change requires refit or fitted-state
  invalidation (`SCI-VAL-REQ-026--027`).
- MAP retains projection, numerical contribution, retained exposure,
  normalization/science-policy support, response/covariance, coaddition,
  companion identity, and final raw-map validity. A VAL admission is necessary
  but not sufficient (`SCI-VAL-REQ-035`).
- Every consumer must bind the exact immutable VAL decision used by its action.
  It may not relabel `decision_unavailable` as eligible, erase causes, or
  promote an invalid parent from a finite/local descendant
  (`SCI-VAL-REQ-027--029,036`).
- Review remains metadata. Exact serialization is engineering-deferred and
  does not create an eligibility state (`SCI-VAL-OWNER-QB003`).

## 9. Preserved Contradictions And Unavailable Claims

The local congruence IDs below trace to the material finding and cross-package
decision ledgers as follows. A split row is intentional where one local
observation contains two independently blocking propositions.

| Local congruence ID | Material finding | Owner decision | Matrix rows |
|---|---|---|---|
| VAL-CONG-C001 | F-005 | XOD-006 | AUTH-003; EXC-002; PROF-001 |
| VAL-CONG-C002 | F-003 | XOD-003 | SRC-RTC; PROF-001 |
| VAL-CONG-C003 | F-003 | XOD-003 | SRC-CAL |
| VAL-CONG-C004 | F-003 | XOD-003 | SRC-ALIGN; SRC-AST |
| VAL-CONG-C005 — named-use portion | F-004 | XOD-004; XOD-005 | PROF-PTC-*; PROF-MAP; ACT-PTC; ACT-MAP |
| VAL-CONG-C005 — aggregate/coadd portion | F-015 | XOD-007 | AGG-001--006; PROF-AGG |
| VAL-CONG-C006 | — engineering-deferred/profile-local representation, not a missing Core science finding | — | AGG-004; ACT-GENERIC |

### VAL-CONG-C001 — Canonical profile completeness

`PROFILE_REGISTRY.md` calls `SCI-VAL:independent_exposure@1` registered, but its
row omits aggregation/propagation compatibility even though the same file's
Registry Rule requires that field for every usable record and says missing
binding makes the profile unavailable. This audit does not infer `not_applicable`.

### VAL-CONG-C002 — Stale RTC binding in the sole canonical profile

The profile depends on RTC representative-origin semantics through an r0.9
binding while the supplied current frozen authority is r0.12. The source
register itself requires a new binding and compatibility review when those
semantics/source change. No compatibility conclusion is inferred.

### VAL-CONG-C003 — Stale CAL binding

The continuing register names CAL r0.3, while the supplied current CAL
rationale/ECS revisions are r0.5/r0.4. CAL-dependent response, uncertainty,
factor/domain, atmosphere, and detector-binding claims remain unavailable
until rebound.

### VAL-CONG-C004 — ALIGN/AST current authority absent from register

The register says no standalone frozen source/version exists; the supplied
current facts say both packages are frozen at v0.1/r0.3. The combined
meaning-only row is not silently upgraded to exact compatibility.

### VAL-CONG-C005 — Policy coverage is intentionally sparse

PTC source meanings match and MAP boundary meanings match their supplied
revisions, but every PTC profile and MAP upstream-admission profile remains
unbound. No aggregate/coadd profile exists. A source match is not a policy.

### VAL-CONG-C006 — Representation and profile-local details remain deferred

The exact serialization carriers for an uninstantiated eligibility slot and
review metadata remain engineering-deferred. Sufficient summaries, associative
combine rules, and partition equivalence remain aggregate-profile-local. These
are not implementation choices VAL Core may invent.

## 10. Required Next Actions

1. **RTC owner/VAL registry review:** add an immutable r0.12 source binding and
   perform the compatibility review required by
   `SOURCE_BINDING_REGISTER.md::SCI-RTC row` for
   `SCI-VAL:independent_exposure@1`.
2. **Canonical profile owner:** amend or supersede the canonical registry record
   with explicit aggregation/propagation compatibility or explicit
   `not_applicable`; do not fill the omission editorially without owner
   authority.
3. **CAL owner/VAL registry review:** bind current rationale r0.5/ECS r0.4 and
   record exact compatibility/unavailable consequences for every dependent
   meaning.
4. **ALIGN and AST owners/VAL registry review:** replace or supersede the
   meaning-only combined row with exact independent v0.1/r0.3 owner/source
   bindings and compatibility consequences.
5. **PTC owner:** register each actually required package-qualified policy as a
   distinct immutable record; do not infer one stage from another.
6. **MAP owner:** register an exact `SCI-MAP:map_upstream_admission` policy if
   evaluation is required; retain projection, contribution, support,
   response/covariance, coaddition, and final validity as MAP-owned stages.
7. **Aggregate/coaddition owner:** register every aggregate proposition with
   exact atomic-source binding, domain, population/support, denominator,
   missing behavior, operator, threshold/polarity, uncertainty, propagation,
   and generation rule.
8. **Engineering representation owner:** define non-aliasing carriers for
   `emptyset_E` and review metadata, then assess representation fidelity in a
   separate evidence layer.

No action above establishes implementation conformity, scientific validation,
freeze, production readiness, or adjacent-package readiness.
