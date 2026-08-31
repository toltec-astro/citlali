# SCI-FRUIT v0.1 — Additive Reformulation Equivalence Analysis

Status: **Stage A analysis; equivalence not established; no algorithm change**

## Question

Can the recovered historical transition be reformulated as

\[
  F_{k+1}=F_k+\Delta F_{k+1}
\]

without changing its numerical or scientific meaning?

The answer at this Stage A revision is **not proven**. Defining a diagnostic
difference `Delta F = F_historical_next - F_k` after both maps exist is an
algebraic identity on a common grid. It does not prove that mapping a processed
residual and accumulating it is the same algorithm, nor that the difference is
an independently calibrated or scientifically interpretable sky product.

## Historical Form To Be Matched

Suppressing observation indices, the recovered complete-product transition is

\[
  F^{\mathrm{hist}}_{k+1}
  = B_{k+1}M_{k+1}
    \left[
      C_{k+1}(X_{k+1}-P^-_{k+1}S_k(F_k);A_k)
      +P^+_{k+1}S_k(F_k);
      w_{k+1},A_k
    \right].
\]

`F_k` in this compact equation stands for the versioned accepted state whose
numerical map predecessor is the selected complete route product. `S` constructs
the feedback model, `P^-` and `P^+` project at subtraction and restoration,
`C` processes only the residual, `M` constructs and normalizes the complete
map/coadd, and `B` selects the raw/filtered observation/coadd route.

A common proposed increment implementation would instead form

\[
  \Delta F_{k+1}
  \stackrel{?}{=}
  B_{k+1}M_{k+1}
  \left[C_{k+1}(X_{k+1}-P_{k+1}S_k(F_k);A_k)ight]
\]

and add it to `F_k`. That equality does not follow from the historical code.

## Necessary Equivalence Conditions

Every row is necessary for the corresponding route and validity domain. A
failure or unresolved row means the proposed formulation cannot be called
representational only.

| Condition | Why additive accumulation needs it | Stage A evidence state |
| --- | --- | --- |
| Same mathematical state space | `F_k` and `Delta F` must have identical estimand, units, calibration, grouping, component order, WCS/grid/frame, and response convention before addition is defined | **Unavailable/unproven.** Four candidate route families have different estimands and no route is numerically admitted |
| Same accepted-model construction | The additive path must use exactly the historical `S_k`, including sign, thresholds, RMS/weight use, support, source center, kernel handling, and invalid/missing rules | **Recoverable as implementation behavior; not scientifically approved and not proven route-complete** |
| Projection/remapping consistency | Subtraction, restoration, and any map-domain equivalent must use compatible geometry, interpolation, pixel center, detector grouping, boundary, and out-of-grid behavior | **Partially evidenced, not proven.** Historical calls reuse map/policy/geometry, but add-back may see fewer unflagged samples |
| Restored support equality or explicit compensation | `P^+S(F_k)` must equal the model contribution represented by adding `F_k`, on exactly the samples/pixels entering the next map | **Known not to be a general literal equality.** Historical residual operations can change flags before add-back; one study measured a small nonzero support difference |
| Conditional mapmaker linearity | At fixed weights, masks, coefficients, grouping, and normalization, mapping `r+q` must equal mapping `r` plus mapping `q` | **Unproven across admitted methods.** Sparse accumulation can be additive before normalization, but the complete estimator/normalization/route chain has not been proved linear |
| Projection-mapmaker left identity | The complete historical chain applied to the restored model must satisfy `B M(P^+S(F_k)) = F_k` on the accepted support, or the increment law must carry the exact non-identity response term | **Unavailable/unproven.** A subtract/add TOD round trip is not a proof that mapmaking after projection returns the predecessor map |
| Identical weight law | Residual weights, resets, post-add-back recomputation, weight feedback, and normalization denominators must match the historical complete-map path | **Unproven and policy-dependent.** Historical behavior can retain residual weights or recompute after restoration |
| Identical mask/flag/penalty law | Sample/detector eligibility must be the same in residual mapping, restoration, final mapping, and any additive decomposition | **Unproven and iteration-dependent.** Learned and residual-created flags can change support |
| Identical response and kernel treatment | Signal and kernel/model response must be projected, restored, normalized, and filtered consistently; lost modes cannot be silently supplied by `F_k` | **Unavailable.** The scientific feedback response and null space have not been authored |
| Identical filtering | For a filtered route, filtering after complete map construction must commute with the proposed decomposition, including weight/profile/edge/normalization behavior | **Unavailable/unproven.** No commutation or linearity proof exists, and the relevant numerical parent routes remain unavailable |
| Identical observation/coadd composition | Observation-specific predecessors and common coadd predecessors must preserve ordered membership, per-observation geometry/weights, coadd normalization, and selected output identity | **Unproven.** The historical route distinction is known; a common additive state law is not established |
| Identical iteration-dependent learning | RTC/PTC re-estimation, learned masks, detector penalties, weight-validation state, selection, and any stop history must receive exactly the same residual/restored inputs and evolve identically | **Unavailable/unproven.** Focused tests explicitly do not validate the nonlinear cleaner or iteration-dependent learning equivalence |
| Identical normalization and finite/missing policy | Zero/invalid weights, coverage cuts, non-finite values, partial support, and normalization order must yield the same complete product and failure state | **Unavailable/unproven** |
| Numerical equality criterion | Exact versus bounded-tolerance equality, floating reduction order, and accepted comparison domains must be specified | **Unavailable.** No equivalence acceptance profile has been approved |

## Strong Form Of The Required Argument

For a proposed residual-map increment, the proponent must establish, for the
exact admitted route and fixed realized state,

\[
\begin{aligned}
 &B M\left(C(X-P^-S(F_k))+P^+S(F_k);w,A_k\right)\\
 &\quad = F_k + \Delta F_{k+1}
\end{aligned}
\]

on the declared common domain. If `Delta F` is defined as `left side - F_k`,
the equation is tautological but supplies no cheaper or structurally different
update. If `Delta F` is instead obtained by mapping only the processed residual,
then the conditions above must derive the equality. In particular, the proof
must expose any non-identity remapping/response term rather than assume
`M P = I`.

The same proof must cover kernel/response components and all state that changes
weights or support. A proof for a fixed linear toy cleaner establishes only that
toy domain. It does not establish equivalence for nonlinear, relearned, masked,
filtered, or coadded production routes.

## Validation Obligations For Choice 2

An owner choice to pursue a mathematically equivalent reformulation would still
require, before implementation replacement or compatibility claims:

1. a route-specific derivation naming the exact state space and every operator;
2. projector/remapper and mapmaker identity tests on valid, edge, partial, and
   missing support for signal and kernel/response components;
3. fixed-weight and both historical weight-policy comparisons;
4. mask/flag/penalty and learning-phase comparisons, including state evolution;
5. raw/filtered observation and raw/filtered coadd comparisons for every
   actually admitted route;
6. uninterrupted and exact-restart trajectory equality, including checkpoint
   sufficiency and lineage;
7. response/transfer and injected-signal comparisons over the authorized
   scientific validity domain; and
8. an owner-approved numerical equality/tolerance and fail-closed discrepancy
   rule.

Until those obligations are met, the additive formulation is a candidate new
recurrence, not a proven representation of the historical one.

## Increment Status And Retention

An update contribution may be assigned a stable identity for lineage,
diagnostics, equivalence testing, or restart. That identity does not make it an
independently calibrated sky estimate, a terminal science map, or a permanent
archive requirement. Exact restart requires the current accepted feedback
state, all other causal state, and enough transition identity/content to
reproduce later results. Earlier contributions may be discarded, compacted, or
reconstructible when the approved transition law and retention policy prove
that doing so cannot change a later required result.
