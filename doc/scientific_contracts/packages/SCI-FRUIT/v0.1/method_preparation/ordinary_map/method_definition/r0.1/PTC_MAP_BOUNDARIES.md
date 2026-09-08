# Two-sided PTC boundary closure proposal

Status: exact proposed crossings; **not closed under current frozen authority**.
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
      -> authorized fixed-bootstrap PTC application -> PTC-Zres[k,a]
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
| BC03 | FRUIT-R[k,a] plus Theta_0 -> PTC fixed application | Residual operand and bootstrap-fitted complete PTC state have exact same group/segment/detector/application domain, metric, rank, affine centering and support. The fit parent is CAL-Y[0,a]; the application parent is FRUIT-R[k,a]. | **BC-IN required.** PTC REQ-001 and D001/REQ-090/096 bind the base primary parent/fit to admitted CAL. REQ-081/D036 require external-residual identity but do not override that route. Propose a bounded permission for fixed application to this FRUIT residual; no residual fit, recentering or rank/support change. | New application generation and exact input-versus-fit-parent distinction. Fixed-map internal coefficients are evaluated on the residual; full-rank/finite guards and PTC failure scopes persist. Missing BC-IN prevents invocation; do not pretend the residual is CAL. |
| BC04 | PTC application -> PTC-Zres[k,a] -> FRUIT rejoin consumer | PTC-owned processed-residual output, preserved calibrated unit with its published discarded lambda/reference/null-space state and exact output support/retention. Separate from ordinary CAL-parent PTC-Z[0,a]. | BC-IN must name this output, its retention/profile applicability and parent/state binding. Frozen PTC equations provide operator mathematics only, not automatic admission of the new parent/output role. | Publish total removed signal relative to the actual residual operand and separate fitted-subspace state; do not label correlation as atmosphere. Response/uncertainty bind the residual parent and frozen Theta, not a fictitious direct CAL-only pass. |
| BC05 | PTC-Zres[k,a] and projected applied model -> FRUIT-Zjoin[k,a] | New rejoined sample product, not PTC-Zres with edited values; exact addition compatibility, same required sample association and supported model value. No additional centering restoration or post-rejoin signal operation. Model-only final samples prohibited. | FRUIT owns rejoin; Q02 requires its explicit additive embedding/reference compatibility. Preserve the original representative-source facts and add the model/transformed-parent influence; do not assign a new admissible origin class by inference. | Model and residual share parents. Preserve both uncertainty paths/cross dependence, output support, missing-mode origin, exact model generation and causes. No unit sky response or acquired exposure is manufactured. |
| BC06 | FRUIT-Zjoin[k,a], retained gamma_0 and current QC -> ordinary MAP input | Same compatible calibrated quantity/nominal beam, exact RTC-grid AST sample association, original lineage and new product generation. Fixed bootstrap I_MAP,a and S_a; gamma's value generation differs from current compatibility/QC generations. | **BC-OUT required.** MAP REQ-002, exact PTC boundary and map_upstream_admission@2 admit the realized PTC transformed product, not this rejoined child by numerical resemblance. Propose a controlled MAP input/boundary/profile/source-binding successor for this exact FRUIT descendant and coefficient use. Frozen MAP normalization/support science is reused, not replaced. | All original producer retention/classification and cause facts remain; no automatic synthesized-origin rescue or inherited eligibility. MAP still owns admission, positivity/finite classification, one-hot contribution and validity. Missing BC-OUT blocks MAP use. |
| BC07 | Authorized MAP operation -> unfiltered normalized ordinary MAP bundle -> next FRUIT model | Science-policy-supported S_a, complete normalized signal/numerator/normalization and distinct response/uncertainty/support/exposure records. Signal uses RTC-grid coordinates; unique-original footprint exposure uses original ALIGN-grid coordinates. | Frozen MAP v0.1/r0.7.1 supplies estimator and bundle meanings. Exact coefficient family and admitted numerical coverage_cut/effective policy are still needed. FRUIT's M02 then selects the previous complete bundle, not an increment. | Keep MAP validity separate from FRUIT acceptance/terminal completion and downstream pointing fitness. No added exposure, precision or independence across arrays/iterations. |

## Precise proposed upstream actions

**BC-IN: fixed application permission.** Permit only the exact ordinary
bootstrap-resolved PTC map to consume a FRUIT residual whose complete calibrated
ancestry, model subtraction, difference-domain embedding, support and reference
are bound. The bootstrap fit remains on CAL. Reuse its centering/subspace/
metric/rank/support; reevaluate internal application coordinates on the residual.
Emit a separately identified processed-residual product with explicit original
fit parent, actual application parent, output retention, response/uncertainty
state and failure. This is a proposed controlled PTC boundary/route successor,
not an interpretation that existing external-residual vocabulary already grants
all these permissions. It permits no residual learning or new PCA estimator.

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
profile and explicit retained use with each BC-IN/BC-OUT generation. This
resolves a producer choice withheld by frozen MAP; it does not make unity a
fallback, a residual-noise weight or a precision estimate. Residual-derived,
post-rejoin-recomputed and retained coefficients keep different rule/population/
generation identities. All numerical payload/admission checks remain active.

These proposed permissions must also establish the stated correction-domain
reference/gauge compatibility. No automatic offset, zero-mode restoration,
model extrapolation or unit conversion is supplied if that compatibility fails.

## Specific unavailable-route disposition

Under the currently bound authorities, the first feedback iteration k=1 is
blocked at BC03; even an assumed successful residual application would encounter
another block at BC06. Missing coefficient/support bindings also prevent
numerical bootstrap. The disposition is
`unavailable_under_current_frozen_parent_permissions`, with causes BC-IN,
BC-OUT, coefficient-family/QC and numerical support-policy binding.

Paper rules are concrete, but this route is not closed or executable. If these
bounded successor permissions cannot be approved without broader science,
return this same unavailable-route disposition to the owner before any wider
fit, selector, producer, population or method is introduced. The historical
JINC control remains unchanged; no experiment is designed here.
