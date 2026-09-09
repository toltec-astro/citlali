# Q02: three decisions for the ordinary-MAP FRUIT boundary

Date: 2026-09-09. Scientific owner: Grant Wilson. Review r0.1.
Status: recommendations prepared; all three decisions below are **pending**.
The owner's “okay, let's Q02” authorizes this review preparation, not adoption.
Q01's scientific definition remains accepted and frozen.

## Program adherence and prior-work recovery

This bounded review follows the [charter](../../r0.4/inputs/program/README.md),
[pilot workflow](../../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md) and
[r0.4 recovery](../../r0.4/PRIOR_WORK.md). It adopts the
[Q01 freeze and its Q02 separation](../../r0.4/SCIENTIFIC_OWNER_FREEZE_R0.4.md),
reviews the existing B01–B12 proposals, and adds the necessary profile-version
clarification S1 below. No generic derivation, new author packet or experiment
is commissioned. Exact source and review-file hashes are in
[the review manifest](REVIEW_MANIFEST.json).

## Recommendation

Approve these three choices for the **controlled ordinary-MAP reference**.
Approval would settle their scientific substance for controlled upstream
amendment preparation. The exact successor sources and registry bindings must
still be completed, reviewed and adopted before these permissions are active.

| Decision under Q02 | Recommended choice | Scientific consequence |
| --- | --- | --- |
| Q02-A — residual PTC pass | Admit the exact FRUIT-created calibrated residual to one PTC Learn–resolve–Apply pass, using versioned permissions | PTC can learn on sky-subtracted data. The existing ordinary CAL route and its restrictions remain available unchanged. |
| Q02-B — correction and rejoin | Use the matching containing-pixel projection and scale-one/no-added-offset correction convention, subject to exact quantity/reference/support compatibility | The current numerical model can be removed and rejoined consistently. This does not restore lost sky modes or establish unit astronomical response. |
| Q02-C — MAP handoff and coefficients | Admit the explicitly identified FRUIT-rejoined descendant, with uniform occurrence coefficients for this reference and fresh current QC | Gridding is fixed and transparent. Equal sample weights are a deliberate scientific choice, with no precision or optimal-noise claim. |

These are three dispositions of existing `SCI-FRUIT-OM-BOUNDARY-002`, not three
new contract packages. They can be answered separately. Approval of A or B
alone does not make the complete chain available.

## Q02-A. One PTC pass on the current residual

**Recommend approve B01–B07, with S1's explicit profile-version correction.**
The CAL ancestor remains immutable. FRUIT constructs each residual from that
parent and the current applied model. PTC learns current centering and its
configured-rank subspace once, resolves the current state and holds it fixed
for that pass's application and conditional response queries. FRUIT owns the
outer recurrence; PTC does not acquire a source selector or internal FRUIT loop.

The existing centering, identity scaling, positive-rank and solve rules remain.
Total removed signal is measured relative to the actual residual. The
discarded centering location is retained as a fact and never restored by
rejoin. There is no rank fallback, support refinement or prior-state rescue.
Ordinary CAL processing keeps its own permissions. Bootstrap uses that route;
later passes require the new residual route.

This is permission to use the already defined processing in a new input role,
not proof that the fitted contamination is correct or that FRUIT improves
recovery. Numerical rank, grouping and effective-plan values remain Q03.
REQ-099's other exclusions, including simulation/empirical populations,
full-procedure numerical response and stronger covariance, remain in force.

### S1. Required clarification to B05 and downstream references

The [registered PTC profiles](../../../../../../../SCI-VAL/v0.1/PROFILE_REGISTRY.md)
bind the ordinary route and exact source identities. Their supersession rules
require new immutable versions for changed domain or source bindings. B05's
reference to preserving `SCI-PTC:output_retention@1` semantics must therefore
not be read as expanding that registered key to residual-derived products.

Proposed controlling clarification for Q02-A and Q02-C:

> Preserve the existing ordinary-route profile records and evaluations.
> Residual-route basis fitting, loading fitting where used, operator
> application, output retention and requested conditional-companion use shall
> bind new immutable profile versions wherever their domain or authoritative
> source changes. Preserve the existing restrictions, causes and separate
> named-use meanings; add only the explicitly approved residual-parent scope.
> Downstream coefficient/QC and MAP records shall reference the exact current
> residual-route retention/application decisions, not an ordinary-route @1
> result or an alias. No profile name alone makes the route available.

This is an explicit proposed correction to B05 and any B09–B11 reference that
would otherwise reuse an ordinary-route key for a changed domain. It is not
an edit to the frozen annex or permission to relax other predicates. The
successor versions and exact source hashes will be bound during controlled
adoption; no new registry entry is created here.

## Q02-B. Remove and rejoin the declared numerical correction

**Recommend approve B08 and M04's matching projection for this reference.**
Use the same exact AST/WCS containing pixel on both model paths, with
lower-inclusive/upper-exclusive boundaries and no interpolation, extrapolation
or clamping. Project the declared calibrated map correction with scale one
and no added offset on matching applicable removal/rejoin supports. Outside
the fixed reference domain, use its explicit residual-only branch. Known
inference rejection is intentional zero correction; an unknown value is not.

Compatibility must include the actual calibrated quantity, nominal beam,
spectral/calibration lineage, coordinate association, support and additive
reference. A common `mJy/beam` label is insufficient. CAL's point-source-peak
convention does not grant surface-brightness, integrated-flux or true-sky
interpretation. If the operands cannot satisfy those exact conditions, the
route stays unavailable rather than silently converting or rebasing them.

With complete finite restoration on the same contributing occurrences,
mapping the projected model reproduces its declared map values on the stated
domain. That conditional numerical identity does not prove unit sky response,
recover absolute sky/optical DC, reverse PTC centering, or make model and data
independent. A different interpolation or response-corrected physical model
would need a separately declared method. No such extension is selected here.

## Q02-C. Rejoined MAP input and uniform gridding

**Recommend approve B09–B11, subject to S1, for the controlled reference only.**
MAP receives an explicitly identified FRUIT-rejoined descendant with its CAL,
residual, PTC, learned-state, model and rejoin parents. It is not mislabeled as
unchanged PTC output. Current PTC retention, producer coefficient/QC and MAP
admission remain separate decisions. Exact same-sample AST joins, original
ALIGN-footprint exposure, origin/influence causes, response/uncertainty states,
complete-bundle publication and failure behavior are carried through unchanged.

Approve proposed family `PTC-ANALYSIS/UNIFORM-OCCURRENCE@proposal-r0.1`:
dimensionless gamma_i=1 on the declared bootstrap coefficient population,
retained by value generation with fresh compatibility/QC on every pass.
Its QC proposal remains `SCI-PTC:uniform_occurrence_gridding_qc@proposal-r0.1`;
actual registered identities/source bindings are adoption work. New PCA
coefficients are distinct from these fixed gridding values. Required current
failure blocks this reference, without silent reweighting or support loss.

For a supported pixel, after every frozen MAP contribution gate passes, this
specializes MAP's N/Q estimator to the arithmetic mean of its contributing
occurrences. It gives equal weight to occurrences, not equal weight per detector,
equal exposure, independent observations or inverse noise variance. More
eligible samples can give a detector more influence. Normalization Q supplies
no S/N or covariance evidence, and `coverage_cut` still needs Q04's exact policy.

The benefit is an easily interpreted, fixed gridding choice while studying
feedback. The tradeoff is that it does not downweight noisier eligible samples
and can be less statistically efficient when noise differs substantially.
Ordinary/naive MAP can use an approved nonuniform coefficient family; “naive”
does not force this choice. If uniform gridding is not wanted, defer that part
of C and select an exact alternative family and QC before MAP execution.
Do not infer inverse variance from a convenient stored weight.

Keep the historical control's actual weights and JINC behavior. The uniform
naive reference is a distinct method combination; a change relative to that
control cannot automatically be credited to the selector. Within a selector
comparison, both arms use the same chosen gridding, while each relearns PTC
on its own residual. No experiment is selected or authorized by this choice.

## Exact clause disposition and adoption boundary

| Existing source | Decision / disposition |
| --- | --- |
| B01–B07 | A, including explicit S1 successor-profile handling. Preserve ordinary CAL permissions and REQ-099's remaining exclusions. |
| B08; M04 projection | B, conditional calibrated correction and matching containing-pixel model paths. |
| B09 | C, explicit uniform family, value generation and fresh current QC/retained use, with S1 current-profile references. |
| B10–B11 | C, explicit rejoined product and proposed MAP boundary/profile successor; preserve ordinary MAP branch and arithmetic. |
| B12 | Retain limits. Exact POINT/OOF mode/WCS registration remains an open Q02 item once a mode is selected. No raw-frequency BEAM, OOF transfer or downstream inference authority is granted. |

Target [annex](../../r0.4/BOUNDARY_AMENDMENTS.md) SHA-256:
`528ca373cf70c480f97280de4a47a160d93992fc3f550f74bfd49b62f21daf28`.
The r0.4 scientific definition and archived annex remain unchanged. S1 is the
only new proposed clause clarification in this review. The ordinary MAP
arithmetic continues to bind exactly v0.1/r0.7.1.

After owner disposition, prepare the exact PTC/MAP successor clauses and
changed profile/source-binding records as one coherent adoption set. Preserve
old versions and bind the retained coefficient generation to each current
product. Review that exact set before adoption; this sheet is not itself a
registry update. Existing predicates are not weakened while changing their
explicitly approved parent domain.

Q01 remains frozen. Its numerical inference policy, Q03 PTC settings, Q04
support policy, Q05 required evidence and Q06 case/order/execution bindings
remain separate. Approving A–C does not by itself close every Q02 mode binding
or make a full numerical method available. No author dispatch, implementation,
replay, injection, qualification, production change or Unity action follows.
The remaining owner task here is to approve, revise or defer A, B and C with
these limits; no answer has been inferred.
