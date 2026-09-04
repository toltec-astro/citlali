# MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001 Completed Horizontal Audit Report

Status: completed under owner disposition `MSP-OD-001`; scientific/package
coherence is established, and shared-source repair remains required.

## Exact audit binding

| Field | Exact value |
| --- | --- |
| Commit | `5f0fc20042b88fb6cd883c92d1b59b7f22832901` |
| Tree | `97a4d908061e51418f93afc1d97d27433af441b8` |
| Parent | `9a2780aa3bd8343fea87ac0b28b390384118c883` |
| Checkout | detached HEAD; initially clean |
| Consolidated work-order SHA-256 | `b5cfdc0d2e9b72984b48bbe46e6d5750699828e47370e36996f72fc0b7196d4f` |
| Original work-order attachment SHA-256 | `400388f1172bd155866f770debbd5754c0cf86ee364e31b5a6d2bdadc2c82713` |
| Source-authority-manifest SHA-256 | `d21d1446ebcdda8597cf08a4568be91906e3cc22e97f9e7f5544a5fa590b2cd5` |

The local `refs/heads/codex/refactor-mainline` resolved to the exact audit
commit at preflight and was not moved.  Commit/tree identity, not the branch
name, controls this audit.

## Purpose, scope, and authority boundary

This fresh-context, implementation-blind audit tested whether the frozen
map-space family can be believed simultaneously as one coherent set of
products, transformations, uncertainty companions and measurements.  It
examined frozen SCI-MAP, SCI-JINC, SCI-FLT-FIXED, SCI-FLT-MATCHED, SCI-NOI and
SCI-POINT, anchored at the exact PTC-to-map-space handoff.

PTC-to-MAP/JINC and AST coordinate boundaries, exact SCI-VAL records and
`doc/SCIENTIFIC_CONVENTIONS.md` were admitted only for their boundary/shared-
convention roles.  PTC, AST, VAL, RTC, CAL and ALIGN package-local science was
not reopened.  Engineering-conformance views were checked only for fidelity to
frozen science and boundaries.

Active FRUIT and historical ALIGN worktrees/branches were not inspected.
Application source/tests, runtime behavior, validation outcomes, prior-work or
recovery dossiers, active branches, Unity and remotes were excluded.

## Package inventory and independent local results

| Package | Frozen identity | Package-local audit result | Present numerical route |
| --- | --- | --- | --- |
| SCI-MAP | v0.1/r0.7.1 | manifest-bound; identity/internal verifier PASS; rationale/ECS faithful | MSP-U-001 |
| SCI-JINC | v0.1/r0.3 | manifest/tag/source locks; identity/internal verifier PASS; rationale/ECS faithful | MSP-U-003 |
| SCI-FLT-FIXED | v0.1 conditionally frozen | 61-object authority bound; recorded verifier PASS; rationale/ECS faithful | MSP-U-005 |
| SCI-FLT-MATCHED | v0.1/r0.6 | 46-object identity/internal verifier PASS; rationale/ECS faithful | MSP-U-006 |
| SCI-NOI | v0.1/r0.5 | manifest plus post-snapshot approval bound; identity/internal verifier PASS; rationale/ECS faithful | MSP-U-007 |
| SCI-POINT | v0.1/r0.4 over unchanged r0.3 science | identity/internal verifier PASS; 38 REQs/32 PREDs/23 UNAV states; rationale/ECS faithful | MSP-U-008 and MSP-U-009 |

These are source identity, completeness, internal-consistency and
representation-fidelity results only.  The FLT-FIXED rerun could not start
because the mandated local environment lacks `reportlab`; the audit relied on
its immutable recorded PASS plus independently checked manifest identities.
No dependency was installed, and no result in this table is an implementation-
conformance, validation, or realized numerical-route conclusion.

## Product topology

The stable registry and exact route catalog are in
`PRODUCT_AND_BOUNDARY_GRAPH.md`.  In compact form:

- MSP-P004 ordinary MAP and MSP-P006 JINC are sibling mapmaking products from
  PTC/AST boundaries, never a mandatory serial pair (MSP-E001/002/003 versus
  MSP-E005/006; negative MSP-E007/008).
- MSP-P005 is MAP's exact centered-integer equal-observation coadd under
  MSP-E004.  Its dimensionless `u_op=1` applies once per admitted observation
  row and does not flatten sample-, pixel-, numerator-, denominator-, validity-
  or coverage-level information.
- Base SCI-JINC v0.1 authorizes neither a cross-observation coadd nor numerical
  roles outside its exact five-role bundle.
- MSP-P007 FLT-FIXED and MSP-P009 FLT-MATCHED are distinct direct transforms of
  admitted parents; no implicit cascade or identity substitution is authorized
  (MSP-E009--MSP-E016).
- MSP-P010 NOI-GEN, MSP-P011 NOI-UNC and MSP-P012 NOI-STD are distinct
  randomization/uncertainty/standardization products.  They never rewrite
  mapmaking or PTC quantities (MSP-E017--MSP-E022 and negative MSP-E031).
- MSP-P013 and MSP-P014 are per-array POINT fit/measurement products from one
  exact route.  Named-use decisions and VAL evaluations remain separate
  MSP-P015/MSP-P016 objects (MSP-E023--MSP-E029).  POINT is not source
  detection or a catalog (negative MSP-E032).
- MSP-E030 records only the excluded deferred MSP-PX01 FRUIT envelope.

## Principal frozen invariants under MSP-OD-001

1. A MAP or filtered-map consumer does not receive a scientifically complete
   bare signal plane.  Parent, route, units, calibration, frame/WCS, support,
   response/covariance state, lifecycle, ancestry and causes remain explicit.
2. MAP's ordinary quantity is the frozen nonpolarimetric total-intensity-
   equivalent signal; generic `I` labeling does not establish formal Stokes I.
3. MAP coaddition uses dimensionless `u_op=1` for each admitted observation
   row.  This observation-level rule does not erase or replace other authorized
   levels of information and does not extend to JINC.
4. MAP exposure is geometric unique-original occurrence accounting at each
   original's own AST ALIGN-grid coordinate.  It is independent of processed
   membership, filtering, interpolation, operator/response support and
   statistical weight.
5. JINC has exactly five numerical base roles.  Its compact generative record
   is information state, not a sixth numerical role.  Weight, standalone
   support, response, covariance, exposure, diagnostic and generalized-
   provenance numerical products are not base-v0.1 roles.
6. FIXED is the full-footprint same-grid deterministic operator
   `J_full L_Theta`; MATCHED is a fixed-anchor normalized template-amplitude
   estimator.  Numerical resemblance does not merge them.
7. NOI's marginal second moment is not covariance, precision, significance,
   mapmaking coefficient or PTC weight.  No `@1` profile substitutes for the
   changed, unregistered r0.5 `@2` meanings.
8. POINT amplitude keeps the exact parent unit/calibration/response; width and
   angle are effective processed-source-shape quantities.  Displacement is a
   measurement, not a correction.  Missing joint covariance is never zero,
   diagonal or independence.
9. Per-array atomicity and sibling survival are preserved.  A per-array success
   is not whole-observation success, and a complete result does not imply
   eligibility for every named use.

## Authority-supported representative traces

No trace invents an unavailable route or an excluded JINC role.  `CONDITIONAL`
describes a frozen type-level state transition, not an observed instance.

| Trace ID | Stable products/edges | Result | Exact authority basis |
| --- | --- | --- | --- |
| MSP-T001 | MSP-P001 --MSP-E001--> MSP-P004 | scientific identity owner-resolved under `MSP-OD-001`; numerical route `UNAVAILABLE`; shared-source repair required | MAP PTC boundary:45-75; MAP requirements:4,8; MSP-F-001/MSP-U-001 |
| MSP-T002 | MSP-P001 --MSP-E005--> MSP-P006 | exact five-role closure owner-resolved; numerical route `UNAVAILABLE`; every excluded base role `NOT_AUTHORIZED` | JINC PTC boundary:58-92,124-148; requirements:193-231,290-309; MSP-F-004/MSP-U-003 |
| MSP-T003 | MSP-P004 --MSP-E007--> MSP-P006 | `NOT_AUTHORIZED` | JINC definitions:5-10 and requirements:17-21 make MAP and JINC siblings |
| MSP-T004 | MSP-P004 --MSP-E004--> MSP-P005 | `u_op=1` meaning owner-resolved; type-level `CONDITIONAL`; current source-closed realization `UNAVAILABLE`; shared-source repair required; not a JINC coadd | MAP coadd profiles:9-26; requirements:39-44; MSP-F-002/MSP-U-002/MSP-U-011 |
| MSP-T005 | MSP-P004 --MSP-E023--> MSP-P013 | `UNAVAILABLE` direct POINT route | POINT MAP boundary:3-27 and assumptions:82-102; MSP-U-008 |
| MSP-T006 | MSP-P004 --MSP-E009--> MSP-P007 --MSP-E025--> MSP-P013 | `UNAVAILABLE` fixed-filtered POINT route | FIXED MAP boundary:21-85; POINT FIXED boundary:3-29; MSP-U-005/MSP-U-008 |
| MSP-T007 | MSP-P004 + MSP-P008 --MSP-E012/014--> MSP-P009 --MSP-E026--> MSP-P013 | `UNAVAILABLE` matched-filtered POINT route | MATCHED MAP/template boundaries and POINT MATCHED boundary:3-28; MSP-U-006/MSP-U-008 |
| MSP-T008 | MSP-P004 --MSP-E009--> MSP-P007 --MSP-E021--> NOI family | `UNAVAILABLE` | identical fixed operator is required for every NOI member; no variance/weight-plane filtering; MSP-U-005/MSP-U-007 |
| MSP-T009 | MSP-P004 + MSP-P008 --MSP-E012/014--> MSP-P009 --MSP-E022--> NOI family | `UNAVAILABLE` | predeclared compatibility only; no outcome adaptation; MSP-U-006/MSP-U-007 |
| MSP-T010 | MSP-P013 --MSP-E027--> complete MSP-P014 --MSP-E028/029--> MSP-P015/MSP-P016 | fit-to-measurement is `CONDITIONAL`; every current named-use evaluation is `UNAVAILABLE` | POINT definitions:98-120,138-166; exact Registry has no POINT profiles; MSP-U-008/MSP-U-009 |
| MSP-T011 | one per-array MSP-P013 fails while a sibling reaches MSP-P014 | `CONDITIONAL` authorized partial-success transition; no aggregate success | POINT assumptions:4-80,196-211 and requirements:14-18,44-50 |
| MSP-T012 | MSP-P006 without a base response or joint covariance --MSP-E024--> MSP-P013 | `UNAVAILABLE`; no response/covariance product is inferred, and absence is not zero or independence | JINC requirements:290-309; POINT equations:240-253; MSP-F-004/MSP-U-003/MSP-U-008/MSP-U-011 |
| MSP-T013 | MSP-P007 has no nonzero full-footprint rows | `NOT_PRODUCED`, never a zero-valued successful map | FIXED shared core:362-403,609-658 |
| MSP-T014 | MSP-P011/MSP-P012 --MSP-E031--> PTC/MAP coefficient | `NOT_AUTHORIZED` | NOI requirements:206-217; MSP-U-004 |
| MSP-T015 | MSP-P009 --MSP-E030--> MSP-PX01 | `NOT_APPLICABLE` | exact deferred envelope only; active FRUIT excluded; MSP-U-010 |
| MSP-T016 | multiple MSP-P006 observation bundles --MSP-E008--> ordinary coadd | `NOT_AUTHORIZED`; no JINC coadd product is constructed | JINC requirements:257-260; exact base bundle requirements:218-231; MSP-U-004 |

## Findings, states, and exact affected shared clauses

The frozen package cores, exact package boundaries and audience
representations showed no additional consequential authority conflict.  The
four original contradictions remain MAJOR because their shared-source clauses
remain unrepaired:

| Finding | Severity | Current state | Exact `doc/SCIENTIFIC_CONVENTIONS.md` clauses |
| --- | --- | --- | --- |
| MSP-F-001 | MAJOR | `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | 330-332; 413-416 |
| MSP-F-002 | MAJOR | `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | 446-454; 698 |
| MSP-F-003 | MAJOR | `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | 470-475; 491-492; 503-505; 700 |
| MSP-F-004 | MAJOR | `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | 497-501; 510-515; 533-548; 697-702 only insofar as applied to JINC |
| MSP-F-005 | MINOR | `OPEN / MANAGER-ONLY-CORRECTION-DEFERRED` | not a shared-conventions conflict |
| MSP-F-006 | MINOR | `OPEN / MANAGER-ONLY-CORRECTION-DEFERRED` | not a shared-conventions conflict |

Counts are 0 CRITICAL, 4 MAJOR, 0 MODERATE and 2 MINOR.  Exact conflicting and
frozen-authority evidence, scientific consequences, owner dispositions and
clause-level repairs are recorded in
`FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md`.

## Two-axis final conclusion

Scientific/package coherence is established under `MSP-OD-001`: the frozen
SCI-MAP v0.1/r0.7.1 and SCI-JINC v0.1/r0.3 meanings control the four disputed
areas, and the rest of the six-package topology remains coherent.  Every
current numerical, response/covariance, named-use, and unauthorized route state
continues to fail closed exactly as recorded.

Repository-documentation coherence is not established.  The conflicting
shared-conventions clauses remain byte-for-byte present, so all four MAJOR
findings require shared-source repair and this report does not issue an
unqualified `PASS`.

## Recommended follow-on repair scope

Commission a separate work order against an exact owner-specified base,
authorizing edits only to `doc/SCIENTIFIC_CONVENTIONS.md` at clauses 330-332,
413-416, 446-454, 470-475, 491-492, 497-501, 503-505, 510-515, 533-548, and
697-702 only to the extent applicable to the affected MAP/JINC meanings.  It
should implement the four product-scoped repairs in the findings record, retain
unrelated clauses, cite the exact frozen authorities, check cross-references,
and receive independent review.

That follow-on must exclude all frozen packages, application code, validation
products, canonical refs, FRUIT, ALIGN, dependency installation, Unity, and the
two manager-only minor findings unless separately authorized.

## Audit verifier result

`verify_horizontal_audit.py` passes the exact commit/tree/parent and detached
state, the consolidated work-order and original-attachment identities, all 71
admitted source digests, six-package representation, 17 products, 32 graph and
matrix edges, all 16 representative traces, four owner-resolved MAJOR finding
states, explicit negative-route closure, the seven-file audit-only mutation
scope, and the required qualified disposition.

## Claim boundary and nonmutation statement

This audit makes no implementation, implementation-conformance, application-
validation, observational-validation, achieved-response, achieved-covariance,
uncertainty-coverage, performance, readiness, production, activation,
deployment, or Unity claim.  SCI-MAP, SCI-JINC, every other frozen package,
`doc/SCIENTIFIC_CONVENTIONS.md`, application code, validation products,
canonical refs, FRUIT and ALIGN were not changed.  No dependency was installed.
No commit, push, merge, rebase, cleanup, ref movement or PDF audit deliverable
occurred.  All changes are uncommitted and confined to the existing audit
directory.

Recommended disposition: **ACCEPT WITH BOUNDED CONTRACT REPAIR**
