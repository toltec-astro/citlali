# JINC versus ordinary MAP: disposition of the sidebar

Date: 2026-09-08. Status: manager recommendation and conditional explanatory
example; no new normative core clause or numerical-method approval.
The owner supplied the [sidebar](inputs/manager/SIDEBAR_JINC_VS_NAIVE.txt) and
asked: “Please see this sidebar discussion of JINC vs naive mapmaking for this
study.” Its exact bytes are in the source inventory. The original discussion
remains manager-only; it is not admitted as independent scientific authority.

## Decision proposed

Use ordinary MAP to define the first new FRUIT method. Keep the historical
control faithful to its own mapmaker and complete configuration. These serve
different purposes. A method's parent route, forward mapmaker and feedback
projection are part of its identity; changing them must never rewrite the
historical baseline. The JINC-first proposal is superseded as a recommendation,
not as an empirical result or an approved numerical method.

The frozen ordinary MAP contract already selects the forward one-hot
containing-pixel rule, with half-open cell boundaries, no outer-upper-boundary
contribution and normalized positive coefficients. That does not select the
reverse feedback projection. Both MAP and JINC remain conditionally frozen
with their distinct numerical prerequisites.

## What the illustrative identity establishes

Use (T_k) for the complete normalized, realized forward MAP operator, including
its exact admitted sample population, coefficients and output-row selection;
this avoids confusing it with MAP's unnormalized placement matrix (G).
Let (P_k^+) be a proposed model-to-rejoin-sample projection and
(m_k^{\rm applied}) the actual model operand. An accepted model may replace
that symbol only if exact acceptance-to-application identity has been declared.
For a fixed realized (T_k), linearity gives the conditional decomposition

\[
\widehat s_{k+1}
 =T_k r_k^{\rm clean}+T_k P_k^+m_k^{\rm applied}.
\]

If (P_k^+) assigns every final contributing sample the value of its containing
model pixel under the *same* grid, association and cell rule, then for each
permitted output pixel (p),

\[
\widehat s_{k+1,p}
 = \frac{\sum_{i\in\mathcal C_{k,p}}\gamma_{k,i}
       (r^{\rm clean}_{k,i}+m^{\rm applied}_{k,p})}
       {Q_{k,p}}
 =m^{\rm applied}_{k,p}
  +\frac{\sum_{i\in\mathcal C_{k,p}}\gamma_{k,i}
       r^{\rm clean}_{k,i}}{Q_{k,p}},\qquad
 Q_{k,p}=\sum_{i\in\mathcal C_{k,p}}\gamma_{k,i}>0.
\]

This is a specialization of frozen MAP normalization and FRUIT composition,
not a newly selected recurrence. It requires compatible quantities, numerical
scales, references/gauges and model values on every required row; finite
arithmetic and normalization; the same final contributing population and
coefficients in numerator and denominator; complete consistent model restoration
on that population; and exact support-authorized output rows. An unsupported
model pixel cannot be supplied by zero-filling. Additional post-rejoin signal
operations must be included in the declared operator and can invalidate this
simple specialization.

Thus (T_k P_k^+) is identity only on the declared supported model/output
subspace (or the corresponding row restriction), not on every possible sky or
full stored map. The identity concerns the *rejoined applied-model contribution*.
It establishes neither equality of removal/rejoin support nor cancellation of
the removed path. It does not identify (P_k^+) with the removal projector.

The coefficients and flags need not be identical across iterations for this
per-realization algebra to hold. Their data dependence still changes the
procedure and its response conditioning. A fixed realized binning operator
is not a claim that selection, cleaning, learning or the full recurrence is
linear. The original measured parent may be rerun at every iteration while a
new complete map replaces its predecessor; no residual-accumulation rule follows.

## What remains open

The explicit feedback projection, support and restoration rules are owner
choices. The candidate/accepted/applied model relation, selector companions,
residual operator and bypass graph, learning, coefficient generations,
continuation, stopping, terminal selection and failure behavior still require
exact definition. Ordinary MAP removes neither its PTC coefficient gate nor
its support-policy gate and does not infer uncertainty from positive weights.

No statement above establishes unit response to true sky, a recovered missing
mode, independent noise, convergence or a useful intervention. JINC is not
shown wrong or unstable: a later JINC method must independently establish its
chosen normalized gridding/projection round trip and transfer only conclusions
whose assumptions remain valid. No numerical example, replay or qualification
was run to reach this recommendation.
