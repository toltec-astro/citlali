# Two-sided PTC boundary closure proposal

Status: residual-relearning direction approved; exact amended crossings
proposed and **not admitted under current frozen authority**.
Scope owner: Grant Wilson; each producer retains its scientific ownership.
Paper-local labels below identify proposed binding records, not registered
product IDs. The current binding remains frozen MAP v0.1/r0.7.1 and PTC
v0.1/r0.5; no successor is adopted by this packet.

Authority locations: [CAL parent requirements](inputs/manager/cal/src/common/requirements.tex),
[PTC requirements](inputs/manager/ptc/src/common/requirements.tex),
[PTC definitions](inputs/manager/ptc/src/common/definitions.tex),
[MAP requirements](inputs/authority/map/src/common/requirements.tex) and
[exact PTC-to-MAP boundary](inputs/authority/map/SCI-PTC_TO_SCI-MAP_BOUNDARY.md).
These recovery links do not expand future author-reference permissions.

## Exact product chain

    admitted CAL-Y[k,a] -> FRUIT removal -> FRUIT-R[k,a]
      -> authorized PTC Learn/Consider/Apply on current residual -> PTC-Zres[k,a]
      -> FRUIT rejoin -> FRUIT-Zjoin[k,a] -> authorized ordinary MAP

At bootstrap k=0, CAL-Y[0,a] goes directly to ordinary PTC and its actual
PTC-Z[0,a] to ordinary MAP; removal/residual-parent/rejoin crossings are
not_applicable under the proposed bootstrap rule. The missing coefficient and
MAP support permissions still block numerical bootstrap today.

Use the same exact sample-row/detector-column occurrence coordinate domain,
array and immutable segment identities throughout a compatible crossing.
Times and row positions are attributes/locators, not scientific joins. Every
new product has its own generation with immutable parent references.

| Crossing | Producer -> consumer; exact roles | Quantity, reference, support and occurrence identity | Authority / exact missing permission | Response, uncertainty, validity and provenance consequence |
| --- | --- | --- | --- | --- |
| BC01 | SCI-CAL admitted CAL-Y[k,a] -> FRUIT pre-removal input | Calibrated-x-derived detector-time quantity, top-of-atmosphere point-source-equivalent mJy per fixed nominal beam; exact CAL numerical scale/reference, sample n/UID, segment/array and RTC/CAL/AST ancestry. Required CAL support is preserved. | Frozen CAL and PTC identify the legitimate calibrated parent; frozen FRUIT allows a declared iteration-input role. Actual producer realization and method approval remain required. No CAL scientific change proposed. | Carry complete source-to-CAL response state and uncertainty/causes; do not reconstruct them from a multiplier or raw RTC product. FRUIT adds a consuming record without mutating CAL. |
| BC02 | FRUIT applied MAP-model plus CAL-Y[k,a] -> FRUIT-R[k,a] | New residual quantity on the CAL-compatible difference domain; exact subtraction sign, same array/WCS-derived occurrence association, model support S_a and numerical scale-one/no-offset projection. Outside S_a use the explicit no-model residual branch. | FRUIT owns subtraction. Q02 must approve the proposed correction-domain/reference embedding of the PTC-conditioned MAP model into the CAL domain. Equal unit/shape is not authority. If the embedding needs another conversion, this exact proposed route is unavailable until separately revised. | Residual retains both CAL and model parent/influence, model-supported modes and all compatibility/support causes. It is not unchanged CAL, true residual sky or independent of its model. Any residual response/uncertainty needs the joint input/model dependence. |
| BC03 | FRUIT-R[k,a] plus fixed recipe C -> PTC Learn/Consider/resolve/Apply | The immutable current residual is both fit and application parent; distinct basis/loading/application supports, group/segment/detector domains and original CAL/model ancestry are bound. Learn current centering/subspace, resolve Theta_k, then apply that same state to that residual. | **BC-IN required.** PTC REQ-001 and D001/REQ-090/096 admit a CAL primary parent. REQ-081/D036 do not independently admit a FRUIT residual. Propose a controlled residual-parent permission for one configured-rank fit and application per group/segment, preserving ordinary centering/scaling, zero refinements and failure rules. | New learned/resolved/applied generations bind current residual, model influence and CAL ancestry. Full-procedure response includes learned-state changes; conditional fixed-state response names Theta_k. Missing BC-IN prevents learning as well as application; no CAL relabeling or Theta_0 fallback. |
| BC04 | Current PTC application -> PTC-Zres[k,a] -> FRUIT rejoin consumer | PTC-owned processed-residual output with current discarded lambda_k/reference/null-space state, calibrated unit and exact output support/retention. Separate from ordinary bootstrap PTC-Z[0,a]. | BC-IN must name output, retention/QC applicability, current learned state, fit/application parent and source bindings. PTC equations alone do not admit this residual-parent product. | Publish total removed signal relative to the actual residual and separate correlated subspace; no atmospheric-origin claim. Response/uncertainty distinguish current fixed Theta_k from the full relearning procedure and carry all model/parent dependence. |
| BC05 | PTC-Zres[k,a] and projected applied model -> FRUIT-Zjoin[k,a] | New rejoined sample product, not PTC-Zres with edited values; exact addition compatibility, same required sample association and supported model value. No additional centering restoration or post-rejoin signal operation. Model-only final samples prohibited. | FRUIT owns rejoin; Q02 requires its explicit additive embedding/reference compatibility. Preserve the original representative-source facts and add the model/transformed-parent influence; do not assign a new admissible origin class by inference. | Model and residual share parents. Preserve both uncertainty paths/cross dependence, output support, missing-mode origin, exact model generation and causes. No unit sky response or acquired exposure is manufactured. |
| BC06 | FRUIT-Zjoin[k,a], retained gamma_0 and current QC -> ordinary MAP input | Same compatible calibrated quantity/nominal beam, exact RTC-grid AST sample association, original lineage and new product generation. Fixed bootstrap I_MAP,a and S_a; gamma's value generation differs from current Theta_k and compatibility/QC generations. | **BC-OUT required.** MAP REQ-002, exact PTC boundary and map_upstream_admission@2 admit the realized PTC transformed product, not this rejoined child by numerical resemblance. Propose a controlled MAP input/boundary/profile/source-binding successor for this exact FRUIT descendant and coefficient use. Frozen MAP normalization/support science is reused, not replaced. | All original producer retention/classification and cause facts remain; no automatic synthesized-origin rescue or inherited eligibility. MAP still owns admission, positivity/finite classification, one-hot contribution and validity. Missing BC-OUT blocks MAP use. |
| BC07 | Authorized MAP operation -> unfiltered normalized ordinary MAP bundle -> next FRUIT model | Science-policy-supported S_a, complete normalized signal/numerator/normalization and distinct response/uncertainty/support/exposure records. Signal uses RTC-grid coordinates; unique-original footprint exposure uses original ALIGN-grid coordinates. | Frozen MAP v0.1/r0.7.1 supplies estimator and bundle meanings. Exact coefficient family and admitted numerical coverage_cut/effective policy are still needed. FRUIT's M02 then selects the previous complete bundle, not an increment. | Keep MAP validity separate from FRUIT acceptance/terminal completion and downstream pointing fitness. No added exposure, precision or independence across arrays/iterations. |

## Precise proposed upstream actions

**BC-IN: residual learning and application permission.** Permit the exact
FRUIT residual as the immutable input of one complete configured-rank PTC pass
per group/segment. Bind calibrated ancestry, applied-model subtraction and
influence, difference-domain embedding, reference, each use's support and
generation. At k=0 the parent remains CAL; at k>=1 it is the newly constructed
residual, never a previous cleaned residual. Learn the ordinary centering and
subspace from that current input, resolve Theta_k and apply it to that same
input. Keep the recipe/rank/support rules fixed, use identity internal scaling,
subtract and publish but do not restore lambda_k, and allow no support-changing
refinement or previous-state fallback. Existing ordinary PTC decisions are
reused to the extent explicitly admitted by this bounded successor.

Emit separately identified learning evidence, resolved state, processed residual,
retention and current QC records. Fixed-state queries name Theta_k; complete
procedure queries carry its learning dependence and state transitions. This is
a proposed controlled PTC scientific parent/boundary successor. The learning
direction is accepted; the exact upstream amendment is not yet approved or
adopted. It does not authorize a different PCA estimator or adaptive rank.

**BC-OUT: rejoined-input permission.** Permit only the exact FRUIT rejoined
child of BC-IN plus its bound applied model to enter the otherwise unchanged
ordinary MAP estimator. Define the new input class and admission predicates,
coordinate/original-lineage association, producer retention/causes, retained
coefficient compatibility and current QC. The existing MAP clause requiring a
PTC product cannot be evaded by renaming the FRUIT child. This therefore needs
a controlled scientific input-boundary/profile/source-binding successor;
it is not merely an adapter or an implementation fix. Its exact version and
bytes must be separately approved before adoption. Retain r0.7.1 as the current
binding; do not import a candidate MAP revision to fill the gap.

**Coefficient family and QC.** Separately approve the M05 unit-analysis family,
its exact bootstrap population and constant statistic, PTC-owned family/QC
profile and explicit retained use with each BC-IN/BC-OUT generation. Current learned Theta_k and product generations
must be bound without treating them as bootstrap identities. This
resolves a producer choice withheld by frozen MAP; it does not make unity a
fallback, a residual-noise weight or a precision estimate. Residual-derived,
post-rejoin-recomputed and retained coefficients keep different rule/population/
generation identities. All numerical payload/admission checks remain active.

These proposed permissions must also establish the stated correction-domain
reference/gauge compatibility. No automatic offset, zero-mode restoration,
model extrapolation or unit conversion is supplied if that compatibility fails.

## Exact remaining status

The learning direction has been decided for the successor paper method. The
PTC learning/application and MAP rejoined-input amendments have not yet been
approved or adopted. The stable gate state remains
`unavailable_under_current_frozen_parent_permissions`.

This states the limit of the approved contracts; it is not evidence that the
calculation is mathematically impossible or that eigenvectors must be known
beforehand. The first feedback pass is blocked at BC03; its downstream handoff
would also be blocked at BC06. Required configuration bindings are separate:
the applicable PTC effective plan, gridding family/QC and admitted numerical
MAP support policy. Missing coefficient/support bindings also prevent bootstrap.
Actual observation, executable, extent value and resource bindings belong to
a separately authorized execution request; this packet makes no implementation
readiness or execution claim.

Return this specific disposition before broadening science if the exact
amendments cannot close these interfaces. Historical JINC control remains
unchanged; no experiment is designed here.
