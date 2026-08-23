# WP-1 SCI-PTC Scientific-Owner Decision Packet

Status: **proposal history; superseded where different by the approved
scientific-owner disposition linked below**

Date: `2026-08-23`

Scientific owner: Grant Wilson

Audited branch: `codex/scientific-contract-library`

Immutable scientific-library source commit:
`55efd8a54464636a24e621f6d1b60486d235b20e`

Immutable baseline-audit commit:
`88c5df87277c22ab807e4a9ba74b7596c9586dc8`

Timestream closure-program commit preceding this packet:
`fbad8179bfc971790ed2510e5a7d4ad4386263dc`

Governing scope:
[`TIMESTREAM_AUDIT_DISPOSITION_ADDENDUM.md`](TIMESTREAM_AUDIT_DISPOSITION_ADDENDUM.md)

Approved scientific-owner disposition:
[`WP1_PTC_SCIENTIFIC_OWNER_DISPOSITION_2026-08-23.md`](WP1_PTC_SCIENTIFIC_OWNER_DISPOSITION_2026-08-23.md)

The disposition record governs wherever its approved language differs from
the recommendations below. In particular, it replaces the proposal's
disabled-role table with an RTC-terminal export route, admits configured
network- or array-level PCA with support at the selected grouping level,
requires an explicitly configured strictly positive rank, and removes
automatic candidate admission from the first ordinary route.

Tracked records:
[`F-001`](../SIX_PACKAGE_WIDE_SCALE_55EFD8A/SIX_PACKAGE_HORIZONTAL_COHERENCE_FINDINGS.md),
`F-002`, the PTC/timestream facet of `F-004`, `F-009`, `F-011`, the
PTC/timestream facet of `F-020`, `XOD-001`, `XOD-002`, `XOD-004`, `XOD-017`,
and `TS-CLAR-001`. The MAP facets of the two mixed findings remain open and
deferred.

## 1. Purpose and authority boundary

This packet turns WP-1 into a finite set of owner choices. It does not amend
SCI-PTC r0.4, bind a VAL profile, authorize a numerical threshold, close an
audit finding, or assert implementation behavior. The owner may approve,
modify, defer, or reject each proposed decision independently.

No SCI-PTC or SCI-VAL package source was edited in preparing this packet. The
frozen package bytes remain those admitted at the immutable source commit.
The most relevant frozen-source digests are:

| Source | SHA-256 |
| --- | --- |
| SCI-PTC normative equations | `4d56ab506f88d26a7af061dcc3b7a8a1e852255999dd4a975cc5cb3517ed3d14` |
| SCI-PTC normative requirements | `74a077b631bdbcfbdf72306d5dff1693ba93f66d86b9dc5384d63997e3268d62` |
| SCI-PTC author decisions and open-decision ledger | `ea42dea6c88d22458fef85ec7d46e92bb8d487e9901eeb13e3b1ff4804d7c54c` |
| SCI-VAL Profile Registry | `d552499fe04309213e05cef11006755ab301e5186399e9515db94b8a81e79d3f` |
| SCI-VAL deterministic evaluation equations | `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e` |

The recommendations were derived from contract sources only. Citlali
implementation behavior was not inspected and supplies no decision here.

## 2. Decision summary

| Proposed decision | Owner action requested | Recommendation | Effect if approved |
| --- | --- | --- | --- |
| `WP1-OWNER-D001` | Select the successor treatment of the transformed-signal identity | Preserve nonrestoring centering; define the exact application first and define total removed signal by identity | Selects the narrow candidate F-001 mathematical repair for successor drafting |
| `WP1-OWNER-D002` | Select fitted-coordinate units and null-space types for the successor | Keep correlated and total removed components distinct; bind the exact `D` mapping from fitted to physical coordinates; publish three distinct null/loss roles | Selects the details needed to avoid a second F-001 ambiguity |
| `WP1-OWNER-D003` | Select the successor treatment of the incomplete named-use truth rule | Make every base permission an ordinary T/F/U/C restriction in a complete PTC-owned profile evaluated by VAL Core | Selects a total rule covering the formerly omitted `b_U=false` state without redesigning VAL Core |
| `WP1-OWNER-D004` | Disposition all seven reserved PTC uses and CAL quality class | Six distinct intended-supported profiles for the ordinary route; empirical/simulation capability explicitly unsupported; science-qualification class required and engineering-only class excluded | Selects the PTC policy sources that WP-5 may bind only after all completion gates pass |
| `WP1-OWNER-D005` | Select the reused-candidate-symbol correction | Preserve `\mathcal C` for causes and rename the finite candidate set `\mathcal Q_{\rm cand}` | Selects a notation-only F-009 correction for successor drafting |
| `WP1-OWNER-D006` | Select the complete disabled-route role map | Adopt the exhaustive role table in Section 7 | Selects the formal F-011 correction without authorizing a MAP route |
| `WP1-OWNER-D007` | Select the proposed minimum ordinary route structure | Explicit-request, single-array, strict-support masked PCA/SVD; detector-wise mean centering; identity scaling; detector-right frozen-subspace action | Selects the structural proposal for `TS-CLAR-001` and PTC-OD-002/003 |
| `WP1-OWNER-D008` | Select route refinement and optional-role dispositions | Advisory diagnostics, zero support-changing refits, CAL-grid fixed-state response companion, no optional `r`, source protection, simulation, persistence, stronger covariance, or MAP role | Keeps the first numerical route bounded |
| `WP1-OWNER-D009` | Select successor change control | Prepare a bounded SCI-PTC v0.1/r0.5 successor after approval; preserve r0.4 and all stable audit IDs | Authorizes drafting, not freezing, the successor |

Approval of all recommendations still does **not** supply the numerical
candidate predicates and thresholds required by PTC-OD-001. Section 8.6
isolates that final owner parameter set rather than guessing it.

## 3. `WP1-OWNER-D001`: transformed-signal identity

### 3.1 Frozen-source conflict

The fitted-parent equation currently says

\[
\widehat U=\widehat M\widehat A^{\mathsf T},
\qquad
Z=\mathcal A_\Theta(Y^{\rm CAL})=Y^{\rm CAL}-\widehat U.
\]

The exact application equation separately says

\[
\mathcal A_\Theta(Y')=
D\,\widetilde{\mathcal A}_\Theta
\!\left[D^{-1}(Y'-\lambda)\right],
\]

with scaling restored and `\lambda` explicitly not restored. The source calls
`\widehat U` the correlated component and does not make `\lambda` part of it.
The two equations therefore do not define the same fitted-parent result in
general.

The dominant frozen decisions are:

- `PTC-AUTH-D027`: output is `P(x-lambda)`, not
  `lambda + P(x-lambda)`; and
- `PTC-OWNER-Q002`: base PCA/SVD uses the frozen realized subspace and metric;
  subtraction of one frozen numerical reconstruction is a separate family.

### 3.2 Recommended identity

Preserve the general frozen family map and define the physical-unit affine
application first:

\[
\boxed{
\mathcal A_\Theta(Y')
=D\,\widetilde{\mathcal A}_\Theta
\!\left[D^{-1}(Y'-\lambda)\right]
}
\]

and then define the total removed component by identity:

\[
\boxed{
U_{{\rm total},\Theta}(Y')
=Y'-\mathcal A_\Theta(Y')
}
\]

For the linear PCA/SVD specialization, let
`\widetilde{\mathcal A}_\Theta(\widetilde Y')=L_\Theta\widetilde Y'`.
Only for that specialization,

\[
\mathcal A_\Theta(Y')=D L_\Theta D^{-1}(Y'-\lambda),
\]

which gives

\[
U_{{\rm corr},\Theta}(Y')
=D(I-L_\Theta)D^{-1}(Y'-\lambda),
\]

\[
U_{{\rm total},\Theta}(Y')
=\lambda+U_{{\rm corr},\Theta}(Y').
\]

On the immutable fitted parent,

\[
Z=\mathcal A_\Theta(Y^{\rm CAL})
=Y^{\rm CAL}-U_{{\rm total},\Theta}(Y^{\rm CAL}).
\]

This is the narrowest repair because it preserves the already frozen physical
choice and estimator family. It corrects the weaker fitted-parent shorthand
instead of reversing nonrestoring centering.

### 3.3 Alternatives not recommended

1. Redefine `\widehat U` as the total removed component. This can be coherent,
   but it conflates additive-reference removal with the fitted correlated
   component and requires wider semantic changes.
2. Insert `\lambda` as a fitted basis/template mode. This can be coherent only
   with new rank, gauge, axis, and decomposition authority and would broaden
   the estimator science.
3. Restore `\lambda` after projection. This reverses `PTC-AUTH-D027` and
   multiple frozen definitions, assumptions, requirements, and predictions.
   It would be an intentional new-science successor rather than a bounded
   repair.

Setting `\lambda=0`, assuming it lies in the learned subspace, or allowing an
implementation default to choose among these alternatives is prohibited.

### 3.4 Owner disposition

Recommended: approve the boxed identities and preserve nonrestoring
centering.

## 4. `WP1-OWNER-D002`: fitted coordinates, response, and null roles

### 4.1 Fitted-coordinate bridge

The factorization is fitted against `\widetilde Y`, while existing notation
can be read as assigning physical CAL units directly to
`\widehat M\widehat A^{\mathsf T}`. The successor should instead declare:

\[
\widehat{\widetilde U}_{\rm corr}
=\widehat M\widehat A^{\mathsf T},
\qquad
\widehat U_{\rm corr}
=D\widehat{\widetilde U}_{\rm corr}.
\]

Thus `\widehat{\widetilde U}_{\rm corr}` is in the exact scaled fitting
coordinate, `\widehat U_{\rm corr}` is in the admitted CAL physical unit, and
`U_{\rm total}` additionally contains the nonrestored `\lambda`.

For the linear specialization, the fixed-state derivative is then

\[
\boxed{J_\Theta=D L_\Theta D^{-1}.}
\]

The separately named frozen-component subtraction family may remain, but its
operand must be explicitly typed. Define

\[
\widehat U_{\rm total}
:=U_{{\rm total},\Theta}(Y^{\rm CAL}).
\]

To reproduce the nonrestoring result on that immutable fitted parent, use

\[
\mathcal A^{\rm sub}(Y')=Y'-\widehat U_{\rm total},
\qquad
\mathrm D\mathcal A^{\rm sub}(Y)[H]=H.
\]

It remains a different affine family on new inputs and is not relabeled as
base PCA/SVD.

### 4.2 Three distinct loss/null roles

The successor should not call all information loss simply “the null space.”
It should publish separately:

1. the fixed-state linear null `ker(J_Theta)`;
2. full-procedure centering-invariant or unidentifiable modes when `lambda` is
   re-estimated; and
3. the affine additive-reference loss caused by discarding `lambda`.

With frozen `lambda`, an additive perturbation is propagated by `J_Theta` and
is not necessarily zero. With re-estimated translation-equivariant centering,
that same declared mode may be absorbed into the new `lambda`. This distinction
must propagate through fixed-state response, full-procedure response,
companions, covariance, replay, and provenance.

### 4.3 Minimum successor clause map

The bounded repair must review, at minimum:

- notation for `\widehat U`, `D`, `\lambda`, the removed subspace,
  `\mathcal N_{\rm PTC}`, `\mathcal A_\Theta`, and `J_\Theta`;
- definitions 002--005, 007, 014, 017--018, and 030--032;
- assumptions 006--007, 014, and 023--025;
- equations `ptc-subtraction`, `center-scale`, `masked-factor`,
  `conditional-cleaner`, `frozen-component`, `full-procedure`,
  `local-companion`, `conditional-response`, `e2e-response`, `e2e-state`, and
  `conditional-covariance`;
- requirements 019, 021--026, 029, 058, 061--063, 069, 071, 083, and
  086--087; and
- predictions 005, 015, 029, 031--033, and 039--042.

Recommended: approve the coordinate/unit bridge, the boxed fixed-state
derivative, the typed frozen-subtraction operand, and the three loss/null
roles.

## 5. `WP1-OWNER-D003`: complete PTC/VAL decision rule

### 5.1 Preserved authority split

The successor should preserve these owners exactly:

- CAL owns one enumerated observation-level quality class `Q`, including the
  distinct values `science-qualification-eligible` and `engineering-only`,
  plus its identity and lineage. No class value is a universal action.
- PTC owns its atomic facts, producer-local composites, the exact seven
  PTC-use policies, and the action taken from each use decision.
- the VAL Profile Registry binds a profile to its real owner, exact source,
  domain, restrictions, exceptions, roles, compatibility, lifecycle, and
  missing/conflict behavior. It supplies no PTC predicate.
- VAL Core owns structural gating, T/F/U/C knowledge semantics, deterministic
  evaluation, cause preservation, and immutable decision lifecycle. It does
  not author or approve PTC science.

No VAL Core redesign is needed.

### 5.2 Recommended total rule

Remove the incomplete local Boolean pair as independent decision authority.
For each named use, either place `b_U` in the same complete owner-bound profile
as an ordinary required restriction or replace it with the profile's complete
restriction set.

For a requested, structurally bound, applicable profile:

| Knowledge after exact authority checking | Decision |
| --- | --- |
| Any applicable required permission is authoritative false | `ineligible` |
| No false; one or more required permissions is unknown or conflicting | `decision_unavailable` |
| Every required permission is authoritative true | `eligible` |

Therefore `b_U=F` with all other permissions true is `ineligible`. Because
`b_U` is a required restriction, `b_U=U` or `b_U=C`, absent a decisive false,
yields `decision_unavailable`. An advisory conflict is retained and neutral.
A non-gating required conflict is normalized to unresolved and yields
`decision_unavailable` when no decisive false exists; a decisive false still
yields `ineligible` while preserving that conflict.

Additional rules that remain mandatory:

- an unbound profile, source, owner, domain, or structural compatibility makes
  applicability unknown and the decision unavailable before policy
  evaluation;
- a known decisive false dominates unrelated non-gating unknown/conflict,
  while every reason and conflict is retained;
- only a resolved, permitted, same-profile exception may yield a traceable
  exceptional permission; unknown/conflicting exceptions never rescue false;
- not requested or known inapplicable yields no eligibility proposition, not
  `ineligible`;
- realization, response, covariance, and validation/evidence remain separate
  axes; and
- a later application, output, QC, or advisory decision never rewrites the
  immutable fitted state.

If PTC retains a producer-local composite, it must state its owner, exact
inputs, four-state domain, missing/conflict rule, scope, use, version, and
action. VAL consumes it as an owner-supplied fact; VAL does not reconstruct an
undocumented Boolean shortcut.

Recommended: approve this VAL-congruent rule and retire the incomplete
`eligible_U`/`ineligible_U` pair as standalone decision authority.

## 6. `WP1-OWNER-D004`: seven PTC use dispositions

Each reserved name remains a distinct scientific proposition. A common
restriction module may be referenced, but a fit decision cannot silently
stand in for application, output, coefficient/QC, response, or simulation.

### 6.1 Exact propositions and recommended first-route disposition

| Reserved profile | Exact proposition | Recommendation for the first ordinary route |
| --- | --- | --- |
| `SCI-PTC:basis_fit_admission` | May this exact occurrence influence construction or selection of the temporal basis/template for the declared CAL parent, segment, array, group, candidate, and lifecycle? | **Supported:** author a complete distinct profile |
| `SCI-PTC:loading_fit_admission` | May this exact occurrence influence the fitted detector loading for the declared basis/template, detector, group, and lifecycle? | **Supported:** author a complete distinct profile; it may reference the same immutable restriction module as basis fit but remains a different action |
| `SCI-PTC:operator_application` | May the resolved frozen `A_Theta` act at this exact occurrence and generation? | **Supported:** author a complete distinct profile; fit exclusion is not application exclusion, and every REQ-089 input is required |
| `SCI-PTC:output_retention` | May this exact transformed occurrence remain in and be published as the requested PTC output role? | **Supported:** author a complete distinct profile; it cannot rewrite fit or application support |
| `SCI-PTC:coefficient_qc_population` | May this exact occurrence/detector enter the named internal PTC statistic, selection-QC population, or diagnostic coefficient family? | **Supported, internal only:** author a complete profile for exact named statistics; no MAP coefficient, weight, precision, or fit alias |
| `SCI-PTC:response_companion` | May this exact companion, parent domain, response role, support, and lifecycle enter the declared response calculation? | **Supported narrowly:** author a CAL-grid, fixed-state companion profile used by the astronomical-transfer selection predicate; it never enters learning |
| `SCI-PTC:empirical_or_simulation_population` | May this exact residual, surrogate, or simulation realization enter its exact empirical/simulation population and lifecycle? | **Capability explicitly unsupported** by this route version; runtime `not_requested` remains a separate request-axis state |

If empirical statistics and simulation populations prove to have different
domains or actions, their owner must use distinct versioned records rather
than force equivalence under the combined reserved label.

“Supported” in this table selects the owner's intended capability and the
profile that must be authored. It does not make a reserved key usable or a
profile complete. Each profile remains unavailable until its full immutable
record, final source binding, and every route-specific gate are present.

### 6.2 CAL quality facts for the first ordinary route

Recommended first-route policy:

- exact admitted CAL product identity and enumerated class `Q` are structural;
- `Q = science-qualification-eligible` is the required class;
- `Q = engineering-only` is explicitly excluded;
- no first-route exception is permitted; and
- missing/conflicting classification yields `decision_unavailable`.

This is a proposed PTC use policy, not a reinterpretation of CAL. CAL's
`science-qualification-eligible` class remains operational guidance and is
not an achieved science-qualified, calibrated-science, atmosphere-fidelity,
or performance claim. A future engineering processed-timestream route would
need separately versioned PTC profiles that preserve the engineering label.

Every active signal-occurrence profile should also retain the exact REQ-016
exclusion for direct ALIGN-synthesized or RTC-replaced occurrences. The
response companion is a separately typed perturbation object, not an acquired
independent exposure; its profile instead binds its exact parent domain and
prohibits entry into learning. Transitive influence must be dispositioned by
exact use; unresolved affected occurrences fail closed, while unaffected
occurrences need not be erased.

The profiles are atomic-only for this first route. A detector aggregate or
propagated QC decision requires a separately named aggregate profile; it
cannot masquerade as the atomic coefficient/QC record.

### 6.3 Allowed disposition types

For each of the seven identities, the owner must select exactly one:

1. a complete immutable profile for a supported/requested use;
2. a versioned explicit unsupported-capability declaration with exact use,
   domain, reason, and lifecycle; runtime request state remains separate; or
3. an owner-proved exact equivalence followed by an explicit Registry
   alias/replacement covering proposition, object/domain, owner/source,
   restrictions, missing/conflict behavior, exceptions, roles, aggregation,
   lifecycle, and action.

VAL never infers equivalence. The recommendation above uses no equivalence
aliases.

Recommended: approve the six supported dispositions, the one explicit
unsupported disposition, and the CAL-fact mapping.

## 7. Formal repairs

### 7.1 `WP1-OWNER-D005`: F-009 symbol map

Preserve `\mathcal C` exclusively for the accumulated cause/fact graph.
Rename the finite candidate set to `\mathcal Q_{\rm cand}`:

\[
c_\star=\min_{\preceq}\left\{
c\in\mathcal Q_{\rm cand}:
P_{\rm corr}(c)\land P_{\rm sky}(c)\land P_{\rm cond}(c)
\land P_{\rm supp}(c)\land P_{\rm stab}(c)\land P_{\rm QC}(c)
\right\}.
\]

Required semantic map:

| Frozen occurrence | Successor occurrence |
| --- | --- |
| `\mathcal C` in cause/fact accumulation | unchanged `\mathcal C` |
| `\mathcal C` in candidate selection | `\mathcal Q_{\rm cand}` |

This is a notation-only repair. It authorizes no global text substitution and
no numerical or scientific change.

Recommended: approve.

### 7.2 `WP1-OWNER-D006`: F-011 disabled-role table

The successor should distinguish existence of the disabled-disposition record
from realization of a PTC scientific role.

| Role | Required disabled disposition |
| --- | --- |
| Requested/effective control | Preserve the requested state separately; realize an immutable effective-control record explicitly stating PTC disabled |
| Observation-resolved state | Records the disabled route and any supplied upstream terminal identity; no numerical PTC admission is inferred |
| Learned/fitted evidence `\mathfrak L` | Disabled; no candidate is fitted or evaluated |
| Resolved model `\Theta` | Disabled; no estimator, rank, subspace, metric, or application map exists |
| Centering/scaling | Disabled; no learned `\lambda`, scale, or PTC null-space claim exists |
| Basis, loadings, fitted/removed component | Disabled; no `\widehat M`, `\widehat A`, `\widehat U`, or removed subspace exists |
| PTC named-use decisions | Preserve each named-use request state separately. Unrequested gives no eligibility proposition. Requested plus authoritatively known inapplicability to the disabled route also gives no eligibility proposition. Requested plus unresolved applicability or structural material gives `applicability_unknown` and `decision_unavailable` |
| Applied state `\Lambda` | Disabled; no subtraction or transformed application occurs |
| Transformed signal `Z` | Product realization `disabled`; no numerical sentinel |
| Diagnostics, coefficients, refinement, simulation, randomness | Preserve each request state separately; product realization is disabled and no fitted loading, QC or analysis coefficient, refit, iteration, draw, surrogate, or fallback state exists |
| Response | Independent axis: unrequested gives `not_computed_or_not_requested_for_this_product`; a separately requested PTC-dependent response is `unavailable` with disabled cause; never `invalid` solely because PTC is disabled |
| Covariance | Same independent-axis rule as response; the upstream CAL covariance state remains unchanged |
| Persisted PTC products | Disabled; no partial TOD or sibling artifact is complete |
| Publication/provenance | Publish only the immutable disabled-disposition/provenance record, not a PTC scientific product |
| Outward PTC-dependent MAP role | Explicitly disabled/non-realized solely to preserve the existing role map; this authorizes no MAP work |
| Direct CAL-to-MAP | Outside PTC authority; neither authorized nor prohibited |
| Upstream RTC/CAL products | Preserved unchanged and independently queryable |
| Failure | Product-realization `disabled` is not failure, rejection, invalidity, or unavailability; a separately requested response or covariance may still be `unavailable` with cause on its own axis |

The exact disabled reason token is an engineering serialization choice under
the semantic state above. The observation-resolved record should include the
CAL-parent identity when it was supplied; disabling PTC must not require a CAL
instance that was never requested or supplied.

Recommended: approve the table.

## 8. Minimum ordinary processed-timestream route

Proposed route identity:

```text
SCI-PTC:ordinary_calibrated_x_array_pca@1
```

The identity is provisional until owner approval and successor publication.
It is deliberately quantity-neutral: no Stokes, independent sky-estimator,
map, pixel, beam-realization, significance, or production role is attached.

### 8.1 `WP1-OWNER-D007`: route structure

Recommended first route:

| Field | Proposed binding |
| --- | --- |
| Request behavior | Explicit request required; no estimator, coordinate, rank, threshold, support, or fallback default |
| Numerical parent | One exact admitted ordinary SCI-CAL `x` product with complete required RTC lineage |
| Domain | One declared array and one coherent declared PTC segment; no cross-array fit |
| Estimator family | Masked-support PCA/SVD: use profiles first select a strict finite rectangular common support, then the numerical fit uses unit weights and the uniform Euclidean metric on that rectangle |
| Grouping | One fixed array-wide group; no network/local hierarchy and no data-derived grouping in this route |
| Basis/loading/application/output supports | Four distinct use decisions; first route requires the same strict common rectangular occurrence support for fit and application and therefore provides no fit-excluded extension |
| Centering | Detector-wise time-axis arithmetic mean over the exact declared fit support, independently for each detector in the one declared segment |
| Center restoration | Never restore `\lambda`; publish it and its support/weights/boundaries |
| Scaling | Identity `D=I`; no learned scale estimator |
| Acting space | Detector-right frozen-subspace projection, with exact linear temporal-coefficient recomputation under the frozen metric |
| Candidate family | Exact explicitly requested finite nested ranks `{0,...,k_max}`, ordered least to most aggressive; `k_max` is supplied in the request and never inferred |
| Output | In-memory transformed calibrated detector TOD on the exact admitted occurrence identity |
| Failure | No passing candidate or insufficient common support gives rejected/unavailable with cause; no aggressive fallback, interpolation, zero fill, or partial complete product |

Why detector-right is recommended: this route declares the fitted detector
loading subspace as the removed subspace and recomputes its temporal
coefficients linearly at each admitted time under the frozen metric. A
temporal-left route is scientifically defensible, but it is a different
application family and should receive a separate explicit owner choice rather
than be inferred from matrix orientation.

For full-column-rank fitted loading matrix `\widehat A` on the declared
detector domain, the proposed exact right-action operator is

\[
P_{\widehat A}
=\widehat A(\widehat A^{\mathsf T}\widehat A)^+
 \widehat A^{\mathsf T},
\qquad
L^d_\Theta=I_D-P_{\widehat A},
\]

\[
\mathcal A_\Theta(Y')=(Y'-\lambda)L^d_\Theta,
\qquad
J_\Theta[H]=H L^d_\Theta,
\]

where the exact generalized inverse, tolerance, detector ordering, fitted
subspace, and gauge/degeneracy state are frozen in `Theta`. The operator is
defined by the subspace projector, not by an arbitrary basis orientation.

The route does not make ordinary PTC a universal default. An absent or
incomplete request remains not requested or unavailable at the exact missing
scope. If owner-approved and incorporated in the successor, this resolves
PTC-OD-002 by rejecting default instantiation and the first-route portion of
PTC-OD-003 by selecting the exact centering/scaling contract above.

Recommended: approve the route structure.

### 8.2 Candidate construction and selection

The request must bind:

- the exact finite integer `k_max` and any rank constrained by actual support;
- ascending rank order, deterministic ties, and boundary polarity;
- the exact degeneracy tolerance and a rule that a selected boundary does not
  silently split an unresolved degenerate block;
- the exact fitted state retained for every candidate; and
- every predicate, threshold, uncertainty treatment, and cause in Section
  8.6.

Rank zero is the least aggressive candidate, not an automatic fallback. If it
does not pass every required predicate, the transformed product is unavailable
or rejected.

### 8.3 `WP1-OWNER-D008`: refinement and optional roles

Recommended first-route dispositions:

| Role | Proposed disposition |
| --- | --- |
| Post-fit detector assessment | Diagnostics may be computed and reported only under exact definitions; they are advisory |
| Support-changing refinement | None; maximum support-changing refits is zero |
| Policy-changing detector classification | Not requested |
| Conditioned `r` | Not requested; frozen diagnostic-only `r` authority remains unchanged |
| Source protection/external source parent | Not requested; no source-mask survival claim |
| Response companion | Requested narrowly: exact CAL-grid fixed-state companion for the declared astronomical-transfer predicate; companion never enters learning |
| Complete source-to-PTC response | Not established by the local companion alone; remains a separate TS-R claim |
| Full-procedure response | Not requested |
| Conditional covariance | Preserve exact available/unavailable state; no total-covariance, precision, or significance claim |
| Stronger covariance/selection uncertainty | Capability deferred for this route; any requested stronger covariance or selection-uncertainty claim is `unavailable` until its required terms exist |
| Internal coefficient/QC population | Supported only for exact selection/diagnostic statistics; no MAP-facing analysis coefficient |
| Empirical/simulation population | Capability explicitly unsupported by this route version; runtime request state is separately `not_requested` for the first route instance |
| Persisted PTC TOD | Not requested in the first route; the in-memory PTC intermediate remains the requested product |
| MAP center, MAP coefficient, MAP output | Excluded from this program |

Zero refinement is an owner policy choice, not an inferred implementation
limit. Any later nonzero refinement needs exact diagnostic thresholds, a finite
bound, stopping states, immutable-parent refit, and a new route/profile
generation.

The local response companion is required because the selected rank must pass
an exact astronomical-transfer predicate. Its intended profile, once complete
and registered, does not by itself establish the complete upstream
source/beam-to-PTC response claim.

Recommended: approve these bounded dispositions.

### 8.4 Explicit exclusions

This route does not authorize:

- cross-array, network/local hierarchical, data-derived, two-sided, or
  vectorized PCA;
- a fit-excluded application extension;
- a default rank, mode count, variance fraction, eigengap, singular-value
  threshold, diagnostic threshold, or fallback;
- raw `r`, `r`-derived `x` subtraction, polarimetry, source restoration, or
  recurrence;
- persistence, response, uncertainty, precision, significance, science
  qualification, performance, conformance, or production claims beyond the
  exact requested role; or
- any MAP role.

### 8.5 Runtime instance versus contract authority

PTC-OD-007 remains a runtime binding requirement. The contract must define the
exact SCI-CAL producer interface and required response/uncertainty states, but
the contract repository need not contain a particular observation's CAL
payload. Each execution separately binds its admitted CAL instance or fails at
the exact affected scope.

### 8.6 Numerical parameter hold: PTC-OD-001 remains open

Frozen sources provide no numerical admission predicates or thresholds. A
numerically executable route still requires one owner-approved parameter
record containing all of the following:

| Required predicate family | Owner must supply |
| --- | --- |
| Residual contamination | Exact residual metric, population/support, unit, uncertainty treatment, threshold, and pass polarity |
| Astronomical transfer | Exact CAL-grid companion and source class, fixed-state response metric, amplitude/location/support domain, uncertainty treatment, threshold, and pass polarity |
| Conditioning | Exact numerical condition metric, unit, support, threshold, generalized-inverse/tolerance relation, and pass polarity |
| Support | Minimum time samples, detectors, degrees of freedom/rank margin, finite/common-support rules, threshold, and pass polarity |
| Stability | Exact comparison population or segment relation, subspace/response stability metric, uncertainty treatment, threshold, and pass polarity |
| QC | Exact named diagnostic/statistic, population, normalization, uncertainty treatment, threshold, and pass polarity |

No scalar score may compensate for a failed predicate. If the owner is not
ready to select these values, the correct disposition is to approve the route
structure while leaving automatic candidate admission unavailable. The next
owner packet should then be a dedicated numerical-predicate and response-
companion parameter packet, supported by explicitly authorized scientific
evidence rather than current code behavior.

## 9. `WP1-OWNER-D009`: successor authorization and closure meaning

Recommended change-control decision:

1. Preserve frozen SCI-PTC v0.1/r0.4 unchanged.
2. After explicit owner approval of this packet, draft one bounded
   SCI-PTC v0.1/r0.5 successor containing only:
   - the D001/D002 transformed-signal, unit, response, and null-role repair;
   - the D003 complete PTC/VAL fact-policy boundary;
   - the D004 immutable PTC policy sources and seven dispositions;
   - the D005 symbol repair;
   - the D006 disabled-role table; and
   - the owner-selected structural route choices from D007/D008.
3. Do not claim a numerically executable ordinary route until the Section 8.6
   parameter record is owner-approved and incorporated.
4. Produce a clause/equation/prediction change map and exact source digests.
5. Obtain explicit scientific-owner approval and freeze of the successor.
6. Only after SCI-PTC and SCI-CAL stabilize, let WP-5 bind/register the exact
   supported profiles and current source digests in SCI-VAL.
7. Run a fresh six-package, timestream-only clean-room re-audit under the
   addendum. Preserve `F-001`--`F-025` and `XOD-001`--`XOD-020`; use only the
   addendum's `TS_CLOSED`, `TS_PARTIAL_MAP_OPEN`, `TS_NOT_CLOSED`, or
   `TS_REGRESSED` vocabulary. Never declare a mixed or MAP-deferred baseline
   finding globally closed from a timestream-only result.

If r0.5 freezes before the Section 8.6 parameter record is approved, it is a
structural, non-executable successor. It may preserve the intended profile
dispositions, but it cannot enter final WP-5 registration or the WP-7 `TS-S`
re-audit as the source-closed ordinary numerical route. A later immutable
owner-approved parameter authority and exact successor/profile binding are
required first.

Approval of this packet authorizes drafting and review. It is not the
successor freeze, owner approval of final successor text, finding closure,
implementation conformance, validation, performance, or readiness evidence.

## 10. Explicit non-decisions

This packet does not decide or imply:

- a numerical predicate or threshold listed in Section 8.6;
- a complete source/beam-to-PTC TS-R route;
- total covariance, precision, significance, or achieved science quality;
- conditioned-`r`, source-protection, empirical, simulation, persistence, or
  recurrence authority;
- a Stokes identity or independent sky estimator; or
- a MAP admission profile, projection, coefficient, exposure, support,
  response policy, direct CAL-to-MAP route, coadd, reprojection, or MAP
  product.

## 11. Copyable scientific-owner response

The owner may respond with one line per decision. `APPROVE` means “approve the
recommendation exactly as written”; `MODIFY` should include replacement text;
`DEFER` leaves the corresponding source state open.

```text
WP1-OWNER-D001: APPROVE | MODIFY | DEFER
WP1-OWNER-D002: APPROVE | MODIFY | DEFER
WP1-OWNER-D003: APPROVE | MODIFY | DEFER
WP1-OWNER-D004: APPROVE | MODIFY | DEFER
WP1-OWNER-D005: APPROVE | MODIFY | DEFER
WP1-OWNER-D006: APPROVE | MODIFY | DEFER
WP1-OWNER-D007: APPROVE | MODIFY | DEFER
WP1-OWNER-D008: APPROVE | MODIFY | DEFER
WP1-OWNER-D009: APPROVE | MODIFY | DEFER

PTC-OD-001 NUMERICAL PARAMETERS:
  DEFER AND PREPARE A DEDICATED OWNER PACKET
  | SUPPLY/ATTACH OWNER-AUTHORIZED PARAMETER RECORD

Owner: Grant Wilson
Decision date: YYYY-MM-DD
Additional constraints:
```

## 12. Recommended owner response

The audit coordinator recommends approving `WP1-OWNER-D001` through
`WP1-OWNER-D009` exactly as written and selecting:

```text
PTC-OD-001 NUMERICAL PARAMETERS:
  DEFER AND PREPARE A DEDICATED OWNER PACKET
```

That disposition authorizes drafting the formal corrections and selects the
proposed first-route architecture without inventing the only values the frozen
source genuinely does not contain. It leaves the numerical route
non-executable until the Section 8.6 parameter authority exists.
