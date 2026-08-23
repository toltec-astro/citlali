# WP-1 SCI-PTC v0.1/r0.5 Successor Author Packet

Status: **authorized drafting packet; non-authoritative until the completed
successor receives separate scientific-owner approval and freeze**

Prepared date: `2026-08-23`

Scientific owner: Grant Wilson

Target branch: `codex/scientific-contract-library`

Immutable scientific-library source commit:
`55efd8a54464636a24e621f6d1b60486d235b20e`

Frozen SCI-PTC predecessor: `v0.1/r0.4`, frozen at source commit
`55efd8a54464636a24e621f6d1b60486d235b20e`

Approved owner-disposition commit: `f407b527e`

Approved owner authority:
[`WP1_PTC_SCIENTIFIC_OWNER_DISPOSITION_2026-08-23.md`](WP1_PTC_SCIENTIFIC_OWNER_DISPOSITION_2026-08-23.md)

## 1. Purpose and firewall

This packet supplies a bounded, implementation-blind authoring plan for one
SCI-PTC v0.1/r0.5 candidate. It does not itself amend SCI-PTC r0.4 or approve
the resulting r0.5 text.

The author may use only:

1. the exact frozen SCI-PTC v0.1/r0.4 package at source commit `55efd8a54`;
2. the approved owner-disposition record at commit `f407b527e`;
3. the existing package's implementation-blind author inputs named by
   `AUTHOR_PACKET_MANIFEST.md`; and
4. this mechanical change blueprint.

The author shall not inspect Citlali or adjacent-package implementation,
configuration behavior, tests, reductions, validation results, performance
evidence, or Unity state. It shall not use implementation defaults to complete
an equation, estimator, threshold, or failure rule.

The baseline audit and proposal packet may be used only to preserve stable
finding and decision identities and to confirm that the listed defects are
addressed. They do not supply scientific repair content where the owner
disposition is silent.

## 2. Exact r0.4 pre-change digests

| Frozen r0.4 source | SHA-256 |
| --- | --- |
| `src/common/notation.tex` | `108022499a5179bc8bbf44060bdc00680ec89c56486c3f833564e63d2e700df7` |
| `src/common/definitions.tex` | `38770c599c8e7b56357577114e799462368f745b703c6690c43d601b5ab4fe6f` |
| `src/common/equations.tex` | `4d56ab506f88d26a7af061dcc3b7a8a1e852255999dd4a975cc5cb3517ed3d14` |
| `src/common/assumptions.tex` | `f4f4ec3593419917071714a9586f22d31019487fb37c75f0dad836578d63e80e` |
| `src/common/requirements.tex` | `74a077b631bdbcfbdf72306d5dff1693ba93f66d86b9dc5384d63997e3268d62` |
| `src/common/edge_cases.tex` | `d6df04f82a219f1804e41e638095197c1d139d428c0bd693c6a794233d29c493` |
| `src/scientific-rationale.tex` | `a2c7301448bcbf5402abce61182930a7f65475b5f7e206d4fded2f76b189d272` |
| `src/engineering-conformance.tex` | `ef219eed260722a503592c022d6e07e5789fc09d4f132f1bf491eaa9af0c6fac` |
| `CROSSWALK.md` | `2211e823f045d78bc2afa491996667e7d931fc59c1f0e51deb896af2bf734125` |
| `AUTHOR_DRAFT_DECISIONS.md` | `ea42dea6c88d22458fef85ec7d46e92bb8d487e9901eeb13e3b1ff4804d7c54c` |
| `src/generate_crosswalk.py` | `5843f948fa60f4937761f1903620d31a4be9e3c687f67578a77b3abd56cdcb42` |
| `src/verify_contract.py` | `4e8c129259714e7f0940b58b46edf070d08e0ebb924504691fd1813c9bdcffbe` |

The predecessor remains recoverable byte-for-byte from commit `55efd8a54`.
No file shall be represented as unchanged if its r0.5 digest differs.

## 3. Successor scope

The r0.5 candidate shall contain only:

- the D001 application/removed-signal repair;
- the D002 complete use-specific T/F/U/C rule;
- the D003 use-specific permission and classification boundary;
- the D004 cause/candidate notation repair;
- the D005 RTC-terminal disabled-route correction;
- the D006 configured network- or array-level PCA route;
- the D007 explicit strictly positive rank and fail-closed rule;
- the D008 zero-refinement ancillary dispositions; and
- the D009 change-control, verification, and review state.

No MAP estimator, projection, coefficient, support threshold, coadd,
reprojection, or product semantics may be added. The statement that ordinary
Citlali map production requires PTC is a route exclusion, not MAP authorship.

## 4. Normative notation change map

The r0.5 notation shall:

1. distinguish the scaled-coordinate fitted correlated component
   `\widehat{\widetilde U}_{\rm corr}` from its physical-unit realization
   `\widehat U_{\rm corr}=D\widehat{\widetilde U}_{\rm corr}`;
2. add `U_{{\rm total},\Theta}(Y')=Y'-\mathcal A_\Theta(Y')`;
3. reserve `\mathcal C` for accumulated causes/facts;
4. introduce `\mathcal M_{\rm cand}` for candidate model specifications and
   retain `c` and `c_\star` for an individual candidate and selected
   candidate;
5. define a configured group `g` as either one full array `D_a` or one
   readout network `D_{a,n}`;
6. distinguish group-local fit and application influence masks or weights;
7. introduce the explicitly requested group rank `k_{{\rm req},g}` with
   strictly positive domain;
8. retain `\Theta_g` and its exact group-local support, mask, metric,
   subspace, rank, degeneracy, generalized-inverse, and failure state; and
9. represent the RTC-terminal route without assigning product-realization
   states to nonexistent CAL/PTC/MAP products.

## 5. Definition change map

Preserve all stable definition IDs. Amend only the meanings affected below,
and append new IDs rather than renumbering unaffected definitions.

| Existing definition | Required r0.5 treatment |
| --- | --- |
| `DEF-002` correlated-mode estimand | Distinguish latent estimand, scaled-coordinate fit, physical-unit correlated component, and operator-defined total removed signal. Preserve physical-origin neutrality. |
| `DEF-003` transformed detector signal | Define the result first as `\mathcal A_\Theta(Y^{\rm CAL})`; do not define it by subtracting only `\widehat U_{\rm corr}`. |
| `DEF-004` removed subspace | Make group identity and configured network/array level mandatory. |
| `DEF-005` null space | Separate fixed-state linear null, re-estimated-centering invariance/unidentifiability, and affine additive-reference loss. |
| `DEF-007` centering state | Bind detector-wise time-axis centering within exact group and segment; retain nonrestoration and deferred baseline treatment. |
| `DEF-013` named-use policy | Replace the incomplete local Boolean description with an owner-supplied proposition evaluated under the complete T/F/U/C rule. |
| `DEF-014` fixed-state map | Bind the exact group, support, mask, rank, metric, and group-local coefficient recomputation. |
| `DEF-015` full procedure | Retain as a separately named unavailable/deferred stronger role for the first ordinary route; do not imply that first-route diagnostics refit. |
| `DEF-019`/`DEF-020` assessment/refinement | Make ordinary-route diagnostics advisory and its maximum support-changing refinements zero. Retain refinement only as a separately authorized future capability. |
| `DEF-026` least-aggressive candidate | Limit to a future automatic-selection family; it is not the first ordinary route's configured-rank rule. |
| `DEF-029` learned evidence | Permit the first route to retain its one explicitly requested candidate and feasibility evidence without fabricating automatic admission predicates. |
| `DEF-030` resolved model | Require exact configured grouping level and strictly positive requested rank. |
| `DEF-031` projection family | Identify detector-right, group-local, mask-aware coefficient recomputation as the first ordinary route specialization. |
| `DEF-035` product realization | Do not use a `disabled` PTC product record as a substitute for the D005 RTC-terminal route. |

Append explicit definitions for:

- the ordinary configured-rank PCA route;
- group-local support and operator state;
- an RTC-terminal export route selected by PTC-disabled configuration; and
- fail-closed rank resolution distinct from PTC-disabled routing.

## 6. Equation change map

### 6.1 Transformed signal

Replace the contradictory fitted-parent shorthand with:

\[
\widehat{\widetilde U}_{\rm corr}
=\widehat M\widehat A^{\mathsf T},
\qquad
\widehat U_{\rm corr}
=D\widehat{\widetilde U}_{\rm corr},
\]

\[
\boxed{Z=\mathcal A_\Theta(Y^{\rm CAL})},
\qquad
\boxed{
U_{{\rm total},\Theta}(Y^{\rm CAL})
=Y^{\rm CAL}-Z
}.
\]

For the nonrestoring linear specialization:

\[
\mathcal A_\Theta(Y')
=D L_\Theta D^{-1}(Y'-\lambda),
\]

\[
U_{{\rm corr},\Theta}(Y')
=D(I-L_\Theta)D^{-1}(Y'-\lambda),
\qquad
U_{{\rm total},\Theta}(Y')
=\lambda+U_{{\rm corr},\Theta}(Y').
\]

The exact operator is primary. Equality on the fitted parent does not turn
`U_{\rm total}` into one physically identified contaminant.

### 6.2 Knowledge evaluation

Replace the uncovered `eligible_U`/`ineligible_U` pair with a total evaluation
over the exact required restriction set `\mathcal R_U`:

\[
\operatorname{Decision}_U=
\begin{cases}
\texttt{ineligible},
  &\exists r\in\mathcal R_U:\;K(r)=\mathsf F,\\
\texttt{eligible},
  &\forall r\in\mathcal R_U:\;K(r)=\mathsf T,\\
\texttt{decision\_unavailable},
  &\text{otherwise},
\end{cases}
\]

after exact request, binding, and applicability checks. Every cause and
knowledge state is retained. Known inapplicability yields no eligibility
proposition, not `ineligible`.

### 6.3 Configured grouping

Replace the implication that the first route is hierarchical with:

\[
\mathcal G=
\begin{cases}
\{D_a\}, & \text{array-level PCA},\\
\{D_{a,n}\}_n, & \text{network-level PCA}.
\end{cases}
\]

Only one grouping level is selected for a route instance. The group-local
model is fitted and applied independently for each `g\in\mathcal G`.

### 6.4 Mask-aware detector-space application

For the ordinary route, use identity internal scaling and detector-wise
time-axis centering. The exact group-local fit remains the declared masked
least-squares PCA/SVD objective with excluded entries carrying zero influence,
not numerical zero.

For centered application row `x_{g,t}=Y^{\rm CAL}_{g,t}-\lambda_g`, loading
matrix `\widehat A_g`, and exact diagonal application-influence matrix
`W^{\rm app}_{g,t}`, the author shall review the following complete-operator
specialization for scientific-owner acceptance:

\[
\widehat m^{\rm app}_{g,t}
=x_{g,t}W^{\rm app}_{g,t}\widehat A_g
 \left(\widehat A_g^{\mathsf T}W^{\rm app}_{g,t}
 \widehat A_g\right)^+,
\]

\[
z_{g,t}
=x_{g,t}-\widehat m^{\rm app}_{g,t}\widehat A_g^{\mathsf T}.
\]

This equation is an author-level exact realization of the approved mask-aware
coefficient-recomputation rule and therefore remains subject to the separate
final r0.5 owner review. Its generalized inverse, tolerance, output support,
detector binding, and insufficient-support behavior are material state. On
complete uniform support it reduces to

\[
Z_g=(Y_g^{\rm CAL}-\lambda_g)
\left[I-\widehat A_g
(\widehat A_g^{\mathsf T}\widehat A_g)^+
\widehat A_g^{\mathsf T}\right].
\]

The exact fixed-state linear operator acts on the compatible kernel with the
same group, support, mask, metric, rank, and subspace. `\lambda_g` is frozen
and is not subtracted from the perturbation.

### 6.5 Explicit positive rank

Replace automatic candidate selection for the first route with:

\[
\mathcal M_{{\rm cand},g}=\{c_g(k_{{\rm req},g})\},
\qquad
k_{{\rm req},g}\in\mathbb Z,
\qquad
1\le k_{{\rm req},g}\le k_{{\rm admissible},g}.
\]

The singleton representation records the requested model specification; it
does not run the future automatic conjunctive-admission policy. Rank zero or
an unrealizable positive rank yields a failed/unavailable ordinary PTC route
with exact group and cause. It does not trigger centering-only output,
rank clipping, rank-one substitution, PTC-disabled state, RTC export, or a
map.

Retain the conjunctive least-aggressive equation only as a separately named
future automatic-selection family whose numerical predicates remain
unavailable.

### 6.6 Zero-refinement lifecycle

The first route realizes exactly one immutable-parent fit for each group and
has maximum support-changing refinements equal to zero. Advisory diagnostics
do not alter `\mathfrak L`, `\Theta_g`, support, or rank. Retain the general
immutable-parent refit equation only for a separately authorized future
refinement family.

### 6.7 Disabled route

Replace the PTC-product disabled-role equation with the route transition:

\[
\boxed{
\mathsf{PTC\ request}=\mathsf{disabled}
\Longrightarrow
\mathsf{route}=\mathsf{RTC\_terminal\_export}
\Longrightarrow
\mathsf{terminal}=\mathsf{successful\ after\ complete\ RTC\ publication}
}.
\]

CAL, PTC, and MAP are not entered and no products for those stages are
realized. Failure of the required RTC export is a route failure. Rank zero is
not an alias for this transition.

## 7. Assumption change map

Preserve existing IDs and make the following bounded changes:

- qualify `ASM-009` as a future automatic-selection-family assumption rather
  than the first configured-rank route;
- qualify `ASM-010`/`ASM-011` as future refinement/hierarchical-family rules;
- strengthen `ASM-013` so conditioned `r` cannot influence rank choice;
- specialize `ASM-023`/`ASM-024` for the ordinary detector-right group-local
  route while preserving distinct future family identities;
- strengthen `ASM-029` with D002's total T/F/U/C evaluation; and
- append assumptions for mutually exclusive configured network/array
  grouping, group-local mask/support, identity scaling, positive configured
  rank, zero refinement, and RTC-terminal disabled routing.

## 8. Requirement change map

Preserve every existing requirement ID. Revise the following requirements and
append route-specific requirements after `SCI-PTC-REQ-089`.

| Requirement | Required r0.5 treatment |
| --- | --- |
| `REQ-008` | The ordinary request must identify configured grouping level and strictly positive rank. Automatic candidates/predicates are required only for a separately requested automatic-selection family. |
| `REQ-012` | Bind each exact use to the complete D002 T/F/U/C rule; remove the incomplete Boolean authority. |
| `REQ-019`--`REQ-024` | Carry exact group identity; make transformed output operator-primary; distinguish correlated and total removed components and three null/loss roles. |
| `REQ-029` | Bind the ordinary specialization to group-local detector-right mask-aware coefficient recomputation and fixed-state kernel action. |
| `REQ-030` | Conditioned `r` is advisory only and cannot influence ordinary rank selection. |
| `REQ-031`--`REQ-033` | Make the first route's configured network/array choice nonhierarchical and mutually exclusive. Retain hierarchy/data-derived grouping only as separately authorized families. |
| `REQ-034`--`REQ-040` | Limit conjunctive candidate selection to a separately requested future automatic-selection family. The first route uses explicit positive rank and mathematical feasibility only. |
| `REQ-041` | Preserve the source-mask limit, while stating source protection is not requested by the first route. |
| `REQ-042`--`REQ-048` | Diagnostics are advisory and maximum support-changing refinements are zero for the first route. General refinement obligations remain conditional on a future requested family. |
| `REQ-058`--`REQ-060` | Preserve conditional/unknown uncertainty semantics and group-local conditioning. Do not add total-covariance claims. |
| `REQ-062` | Require the kernel to use the exact group-local fixed-state linear operator, without refitting or subtracting `\lambda` from the perturbation. |
| `REQ-063`--`REQ-068` | Preserve stronger response roles as unavailable/deferred for the first route; no map-center diagnostic is required. |
| `REQ-069`--`REQ-072` | Preserve the transformed in-memory PTC intermediate and optional persistence; do not add a standalone persistence requirement. |
| `REQ-073`--`REQ-074` | Prohibit rank fallback, silent clipping, route conversion, zero filling, and centering-only output. |
| `REQ-076` | Replace fabricated disabled PTC-role states with the successful RTC-terminal export route and required RTC publication/failure semantics. |
| `REQ-077` | State that ordinary Citlali map production requires PTC and that no direct CAL-to-MAP fallback is selected by this route. Do not add MAP estimator semantics. |
| `REQ-083` | Incorporate the corrected total-removed identity and exact group-local application map. |
| `REQ-084`--`REQ-085` | Permit singleton configured-rank evidence; automatic predicate evidence is required only for the future automatic-selection family. |
| `REQ-088` | Keep independent axes for realized PTC products; clarify that the RTC-terminal route never enters those PTC axes. |
| `REQ-089` | Bind fit-excluded application availability at the exact network/array group level. |

Append requirements covering at minimum:

1. exact ordinary route identity;
2. mutually exclusive configured network/array grouping;
3. group-local support, masks, centering, subspace, rank, application, and
   kernel state;
4. detector-wise time-axis arithmetic centering and identity scaling;
5. explicit integer `k\ge1` and per-group feasibility;
6. rank-zero and unrealizable-rank fail-closed behavior;
7. one immutable-parent fit and zero support-changing refinements;
8. local fixed-state kernel propagation;
9. classification preservation and use-owned admission; and
10. no automatic selector, source protection, simulation, recurrence, or
    stronger response/covariance claim in the first route.

## 9. Prediction change map

Preserve stable prediction IDs. Amend:

- `PRED-001` to expect complete RTC publication followed by successful
  termination with no CAL/PTC/MAP execution;
- `PRED-005` to distinguish fixed-state kernel action from re-estimated
  centering behavior;
- `PRED-007`--`PRED-010` to bind mask/support effects to the configured group;
- `PRED-012` to test both cross-array isolation and cross-network isolation in
  network mode;
- `PRED-013` to make network and array routes alternative, not ordered
  hierarchical stages;
- `PRED-017`--`PRED-020`, `PRED-043`, and `PRED-046` to apply only to a future
  automatic-selection family;
- `PRED-021`--`PRED-024` to state that the first route's advisory diagnostics
  never initiate refit;
- `PRED-025` to include rank-choice inertia under conditioned-`r` changes;
- `PRED-031`, `PRED-039`, and `PRED-040` to test the exact group-local kernel
  projection; and
- `PRED-048` to distinguish an unentered PTC stage from an unavailable
  response on a realized PTC product.

Append predictions for:

1. array-mode cross-network coupling versus network-mode isolation;
2. network-level support changes affecting only their exact group;
3. rank zero failing without RTC-route conversion;
4. unrealizable positive rank failing without clipping;
5. configured rank reproducibility;
6. full-support reduction of the mask-aware operator to the ordinary
   detector-space projector;
7. masked samples having zero coefficient influence without becoming zeros;
8. identical data/kernel operator state; and
9. advisory diagnostics leaving `\Theta_g` unchanged.

## 10. Rationale and engineering-view obligations

The scientist-facing rationale shall explain, in plain language:

- why the complete operator is primary;
- why `\lambda`, correlated removal, and total removal are distinct;
- why neither removed quantity has unique physical origin;
- why network-level and array-level PCA have different scientific support and
  coupling domains;
- why the kernel uses the frozen linear operator;
- why configured positive rank is different from automatic scientific rank
  selection;
- why `k=0` is invalid rather than a harmless no-op; and
- why PTC-disabled means an RTC-export workflow, not a partial full
  reduction.

The engineering view shall express the exact same authority, including input
and output identities, units, group and support scope, mask influence,
strictly positive rank, generalized-inverse state, failure behavior, kernel
domain, lifecycle, and route termination. It shall not map these obligations
to current implementation fields or claim implementation conformance.

## 11. Author-decision and crosswalk obligations

`AUTHOR_DRAFT_DECISIONS.md` shall:

- record `WP1-OWNER-D001`--`D009` as approved successor inputs;
- preserve prior owner decisions and open identities rather than renumbering
  them;
- mark automatic candidate selection, source protection, refinement,
  simulation, full-procedure response, stronger covariance, and the deferred
  `\lambda` treatment accurately;
- record the mask-aware coefficient-recomputation equation as candidate r0.5
  author text pending final owner approval; and
- state that no implementation or numerical validation evidence was used.

Regenerate `CROSSWALK.md` deterministically. Every changed or appended
requirement and prediction must map to the approved WP-1 decision, exact
rationale location, engineering location, and dependency. The generator and
verifier may be updated mechanically but shall not encode new scientific
defaults.

## 12. Verification and handoff gates

Before returning the candidate for owner review:

1. compile both LaTeX views without errors or unresolved references;
2. run the package identifier, crosswalk, audience-separation, digest, and PDF
   coverage checks;
3. confirm no stable requirement or prediction ID was silently removed or
   reused;
4. run targeted text checks for the superseded conflicting equations and
   disabled-role language;
5. render both PDFs and perform page-by-page visual inspection;
6. produce exact r0.5 source and PDF digests;
7. produce an r0.4-to-r0.5 clause/equation/prediction change map;
8. state which findings the candidate is intended to repair, without marking
   any finding closed; and
9. return the candidate for a separate scientific-owner approval/freeze
   decision.

No final VAL binding or horizontal re-audit begins from an unfrozen r0.5
candidate.

