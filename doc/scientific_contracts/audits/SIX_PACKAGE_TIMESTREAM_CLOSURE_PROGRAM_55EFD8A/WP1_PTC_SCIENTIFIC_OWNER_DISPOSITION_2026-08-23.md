# WP-1 SCI-PTC Scientific-Owner Disposition

Status: **scientific-owner decisions approved; successor drafting authorized;
no package successor frozen by this record**

Decision date: `2026-08-23`

Scientific owner: Grant Wilson

Audited branch: `codex/scientific-contract-library`

Immutable scientific-library source commit:
`55efd8a54464636a24e621f6d1b60486d235b20e`

Immutable baseline-audit commit:
`88c5df87277c22ab807e4a9ba74b7596c9586dc8`

Proposal packet:
[`WP1_PTC_SCIENTIFIC_OWNER_DECISION_PACKET.md`](WP1_PTC_SCIENTIFIC_OWNER_DECISION_PACKET.md)

Governing scope:
[`TIMESTREAM_AUDIT_DISPOSITION_ADDENDUM.md`](TIMESTREAM_AUDIT_DISPOSITION_ADDENDUM.md)

## 1. Authority and precedence

This record captures the scientific owner's completed, one-at-a-time
disposition of `WP1-OWNER-D001` through `WP1-OWNER-D009`. It supersedes the
recommendations in the proposal packet wherever the language differs. The
proposal packet remains preserved as the decision history and is not silently
rewritten into an approval record.

The decisions below authorize preparation of a bounded SCI-PTC v0.1/r0.5
successor. They do not amend the frozen SCI-PTC v0.1/r0.4 bytes, approve final
r0.5 text, freeze a successor, bind a VAL profile, close an audit finding, or
establish implementation conformity, validation, performance, or production
readiness.

SCI-MAP estimator, projection, coefficient, support, response, coaddition,
and reprojection work remains deferred. The only downstream statement made
here is the route fact that ordinary Citlali map production requires PTC.

## 2. Decision summary

| Decision | Owner disposition | Binding result |
| --- | --- | --- |
| `WP1-OWNER-D001` | **APPROVE WITH CLARIFICATIONS** | The complete application operator is primary; total removed signal is the input-output difference; nonrestoring centering is preserved for the bounded successor; physical origin remains neutral; later fast approximate baseline treatment is deferred. |
| `WP1-OWNER-D002` | **APPROVE** | Use the common complete T/F/U/C knowledge rule for each exact named use; preserve causes and keep request, applicability, eligibility, realization, response, covariance, and evidence distinct. |
| `WP1-OWNER-D003` | **APPROVE WITH CLARIFICATIONS** | Use-specific permissions remain distinct propositions but may share common policy machinery; upstream classification is preserved; engineering-only classification does not prohibit PTC mathematics by itself. |
| `WP1-OWNER-D004` | **APPROVE WITH CLARIFICATION** | Preserve `\mathcal C` for cause/fact accumulation and use `\mathcal M_{\rm cand}` for candidate model specifications; retain candidate symbol `c`. |
| `WP1-OWNER-D005` | **APPROVE WITH OWNER CORRECTION** | PTC-disabled selects successful RTC-terminal export; CAL, PTC, and MAP are not run; no map exists without PTC. |
| `WP1-OWNER-D006` | **APPROVE WITH OWNER CORRECTION** | The ordinary route supports configured network-level or array-level PCA; grouping, support, fit, application, resolved state, and kernel response all use the selected scope. |
| `WP1-OWNER-D007` | **APPROVE WITH OWNER CORRECTION** | Rank is explicitly configured and strictly positive; `k=0` and unrealizable ranks fail closed without fallback or route conversion. |
| `WP1-OWNER-D008` | **APPROVE** | The first route performs one immutable-parent fit per group, keeps diagnostics advisory, and supports exact fixed-state kernel propagation without hidden refit, `r` influence, source protection, simulation, or recurrence. |
| `WP1-OWNER-D009` | **APPROVE** | Prepare and review one bounded SCI-PTC v0.1/r0.5 successor; final text requires separate owner approval and freeze. |

## 3. `WP1-OWNER-D001`: operator and removed-signal identity

The PTC application operator is scientifically primary:

\[
\boxed{Z=\mathcal A_\Theta(Y^{\rm CAL})}.
\]

The total removed component is defined from the exact input-output difference:

\[
\boxed{
U_{{\rm total},\Theta}(Y^{\rm CAL})
=Y^{\rm CAL}-Z
}.
\]

For the present nonrestoring-centering operation,

\[
U_{\rm total}=\lambda+U_{\rm corr}.
\]

The successor shall distinguish:

1. `\lambda`, the learned additive centering or reference term;
2. `U_{\rm corr}`, the component rejected by the realized correlated-mode
   operator; and
3. `U_{\rm total}`, the operator-defined difference between input and output.

Neither `U_{\rm corr}` nor `U_{\rm total}` is assigned a unique atmospheric,
electronic, detector-noise, or astronomical physical origin merely because it
is correlated or removed.

Nonrestoring centering remains the exact operation described by the bounded
successor. A later owner question shall consider a fast, memory-bounded,
weighted-binning-compatible treatment of the baseline state for FRUITLOOPS.
That future work shall not turn Citlali into a maximum-likelihood mapmaker,
shall not require dense timestream covariance or exact `N^{-1}`, and shall
restart every recurrence from the immutable admitted parent or an exactly
equivalent immutable reference-centered representation. No future estimator,
segmentation, threshold, or recurrence is selected here.

For fixed resolved state, the exact derivative or linear part of
`\mathcal A_\Theta` acts on a compatible astronomical kernel. The kernel is
not treated by subtracting one frozen numerical reconstruction.

## 4. `WP1-OWNER-D002`: complete use-specific knowledge rule

For each requested, structurally bound, applicable PTC use:

| Required knowledge state | Decision |
| --- | --- |
| Every applicable required restriction is authoritatively true | `eligible` |
| Any decisive applicable restriction is authoritatively false | `ineligible` |
| No decisive false exists and at least one required restriction is unknown, conflicting, ambiguous, or out of domain | `decision_unavailable` |

All contributing facts, causes, false restrictions, unknowns, conflicts, and
applicable exceptions remain visible. A decisive false is not rescued by an
unrelated unknown or conflict.

`not_requested`, `inapplicable`, `applicability_unknown`, eligibility, and
realization states such as `realized`, `failed`, `incomplete`, and
`not_produced` remain distinct. Response, covariance, validation/evidence, and
product realization remain independent axes.

PTC owns the scientific proposition for each PTC use. VAL owns the shared
knowledge types, structural checks, deterministic evaluation, cause
preservation, provenance, and resulting decision artifact. VAL does not
invent, repair, strengthen, or generalize a missing PTC policy. No VAL Core
redesign is authorized.

## 5. `WP1-OWNER-D003`: use-specific validity and classification

The scientific permissions for basis learning, loading estimation, frozen
operator application, output retention, coefficient/QC population, response
use, and empirical/simulation use remain distinct propositions. They may
reference a common set of PTC facts and restrictions and may share nearly all
implementation machinery. Permission for one use never implies permission
for another.

PTC preserves upstream classification. It does not upgrade a CAL
`engineering-only` input into a science-qualified input.

The classification does not, by itself, prohibit PTC mathematics. Admission
belongs to the exact named use: `engineering-only` prohibits an ordinary
science product only where that product's owner requires science-qualified
input. No separately named engineering route is required unless later
differing scientific policy makes one necessary.

## 6. `WP1-OWNER-D004`: candidate and cause symbols

`\mathcal C` is reserved for accumulated causes and facts. The finite set of
candidate model specifications is

\[
\boxed{\mathcal M_{\rm cand}}.
\]

An individual candidate `c` is a candidate model specification within a
fixed estimator context. It may determine rank, subspace, or other declared
model state, but is not itself the final resolved `\Theta`. Evaluation yields
fitted state and evidence; the selected `c_\star` is then resolved into
`\Theta`.

The individual symbol `c` is retained because `m` is already used for
temporal-mode entries. This is a notation-only repair and authorizes no
numerical change or global textual substitution.

## 7. `WP1-OWNER-D005`: PTC-disabled route

PTC-disabled is an RTC-terminal export route, not a set of fabricated absent
PTC products:

\[
\boxed{
\mathrm{PTC\ disabled}
\Longrightarrow
\mathrm{complete\ RTC\ export}
\Longrightarrow
\mathrm{successful\ Citlali\ termination}
}.
\]

The route shall:

1. complete and publish the required consumer-neutral RTC timestream bundle;
2. terminate the Citlali reduction successfully after RTC;
3. not run CAL, PTC, or MAP;
4. produce no CAL, PTC, or MAP product; and
5. select no direct CAL-to-MAP fallback.

Its intended use is export of RTC-conditioned timestreams for a companion ML
mapmaker without a full Citlali reduction. Failure to produce or publish the
required RTC bundle is a route failure. Successful RTC termination is not a
PTC failure.

The orchestration or pipeline route authority owns the CLI termination
behavior. Library components return the typed successful RTC-terminal result;
they do not call process-level `exit()` themselves.

The exact companion handoff schema, including any required RTC-grid
coordinate artifact, belongs to the later RTC/AST boundary work. PTC-disabled
does not authorize a Citlali map, and ordinary Citlali map production has no
PTC-free route.

## 8. `WP1-OWNER-D006`: network- or array-level PCA

The ordinary enabled PTC request selects exactly one grouping level:

\[
\mathcal G=
\begin{cases}
\{D_a\}, & \text{array-level PCA},\\[2mm]
\{D_{a,n}\}_n, & \text{network-level PCA}.
\end{cases}
\]

For every selected group `g`, the exact group identity scopes centering,
masks, fit support, application support, loading subspace, coefficient solve,
rank state, resolved model, transformed output, and kernel projection:

\[
Z_g=\mathcal A_{\Theta_g}(Y_g^{\rm CAL}),
\qquad
H_{{\rm PTC},g}=J_{\Theta_g}[H_g].
\]

For the fixed-state linear specialization,

\[
Z_g=(Y_g^{\rm CAL}-\lambda_g)L_{\Theta_g},
\qquad
H_{{\rm PTC},g}=H_gL_{\Theta_g}.
\]

Network-level PCA realizes independent network-level supports and operators.
Array-level PCA realizes one array-level support and operator and may model
correlations across networks in that array. No support or model silently
borrows detectors outside its declared group. Network- and array-level PCA
are alternative configured routes, not automatically sequential stages.

Masked occurrences have zero statistical influence under the declared
mask-aware fit; they are not numerical zeros. The exact metric, support,
mask, subspace, degeneracy treatment, and generalized-inverse state are part
of `\Theta_g`. The successor shall not require one strict common rectangular
support merely to avoid representing valid group-local masks.

The bounded ordinary route uses detector-wise time-axis centering and identity
internal scaling. The learned additive reference is not restored, consistent
with D001.

## 9. `WP1-OWNER-D007`: explicit positive rank

The ordinary route uses an explicitly configured rank at the selected
network or array grouping level:

\[
\boxed{
k_{{\rm req},g}\in\mathbb Z,
\qquad
1\le k_{{\rm req},g}\le k_{{\rm admissible},g}
}.
\]

If configuration supplies one common requested rank for multiple network
groups, that common request remains explicit while each network retains its
independent fitted support, subspace, resolved state, and feasibility result.
A per-group rank mapping is permitted only when explicitly supplied by the
request.

Rank zero is not a valid PTC operation. It shall fail closed and shall not:

- produce a centering-only PTC timestream;
- be interpreted as PTC-disabled;
- select the RTC-export route;
- be silently promoted to rank one; or
- produce a map.

An unrealizable positive rank likewise fails with exact group and cause. PTC
shall not silently clip, increase, or replace it. The first route performs no
automatic scientific rank selection. Diagnostics may characterize the
requested result but do not change its rank. A later automatic-selection
family requires separate owner authority for its candidate construction,
predicates, thresholds, boundary polarities, uncertainty treatments, and
failure behavior.

## 10. `WP1-OWNER-D008`: bounded ancillary behavior

For the first ordinary route:

1. PTC performs one fit and application per configured group from the
   immutable admitted CAL parent.
2. Post-fit diagnostics may be computed and retained but are advisory; they
   do not change group membership, support, rank, or the immutable fitted
   state.
3. No support-changing refit or hidden iterative refinement occurs.
4. Conditioned `r` does not enter PTC learning, rank choice, subtraction, or
   output. Its RTC diagnostic authority remains unchanged.
5. A compatible astronomical kernel may be propagated through the exact
   frozen group-specific operator. This establishes only the declared local
   fixed-state PTC response, not a complete source-to-PTC or end-to-end
   response.
6. Source masking or source protection is not part of this initial operator.
7. Available conditional uncertainty state is preserved without promotion to
   total covariance, precision, or significance.
8. Empirical/simulation populations are outside this first route.
9. The transformed timestream may proceed directly to the downstream
   Citlali map stage; standalone PTC-timestream persistence is not required
   by this decision.
10. FRUITLOOPS recurrence and baseline re-estimation remain under the
    deferred D001 `\lambda` follow-up.

Any later nonzero refinement or recurrence must restart from the immutable
CAL parent or an exactly equivalent immutable reference-centered parent and
requires separately approved lifecycle, stopping, and response authority.

## 11. `WP1-OWNER-D009`: bounded successor authorization

Prepare one bounded SCI-PTC v0.1/r0.5 candidate containing only the normative
changes required by D001--D008. The work shall:

1. preserve the frozen v0.1/r0.4 source through immutable Git history and
   exact pre-change digests;
2. produce an r0.4-to-r0.5 clause, equation, requirement, prediction, and
   owner-decision change map;
3. preserve stable normative and audit identities wherever their meanings are
   unchanged;
4. update exact source digests and mechanical verification artifacts;
5. perform a package-local clean-room consistency review without consulting
   Citlali implementation;
6. return the complete r0.5 candidate for separate scientific-owner review;
7. freeze it only after explicit approval of the final successor text;
8. defer final VAL source binding and profile registration until SCI-PTC and
   SCI-CAL versions are stable; and
9. rerun the timestream-only horizontal audit against one new immutable
   consolidation commit after all required producer, boundary, and profile
   work is complete.

## 12. Preserved open questions and claim boundary

The following remain deliberately unresolved or outside the first route:

- the future `\lambda`/FRUITLOOPS baseline treatment;
- automatic data-driven rank selection and numerical admission thresholds;
- source protection, support-changing refit, and iterative refinement;
- full-procedure and complete source-to-PTC response;
- stronger covariance, total uncertainty, precision, and significance;
- empirical/simulation validation and achieved performance;
- MAP admission, projection, coefficients, support, response policy,
  coaddition, reprojection, and product authority; and
- implementation conformity and production readiness.

The fixed-state kernel rule remains valid within its exact PTC-local domain:
the kernel uses the derivative or linear part of the same realized operator,
group, support, mask, metric, subspace, and rank that acted on the data. This
does not close the stronger response claim tiers.

