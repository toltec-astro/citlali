# Initial POINT reference within mode-aware FRUIT

2026-09-10 · r0.2 · **Owner-review proposal; preparation authorized, choices pending.**
Repository base: `fdeb0e2ad8b3682ee38982af6a25082ba9830317`.

## Program adherence and prior-work recovery

Follow the [charter](r0.4/inputs/program/README.md),
[pilot workflow](r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md) and
[reviewed prior work](r0.4/PRIOR_WORK.md). Adopt the frozen generic Stage A/B
core, [ordinary-MAP definition](r0.4/METHOD_DEFINITION.md), and later
[Q02-A/B substance approval](q02_review/r0.2/README.md). Adopt the
[weighting closure](q02_review/WEIGHTING_EXPLORATION_CLOSURE_2026-09-10.md).
Cite historical recovery and the completed weighting results as manager evidence;
exclude them from independent scientific authorship. Defer new selectors and
further experiments. Nothing frozen is superseded.

The owner supplied the [Reference Method Brief discussion](MINIMUM_REFERENCE_METHOD_REVIEW_2026-09-10.json)
for consideration. Cite it as review input, not approval or scientific authority;
its recommendations are checked against frozen M02/M03/M05 and the
[September 9 direction](r0.4/inputs/owner/LATER_MODE_AWARE_DIRECTION_2026-09-09.txt).
This successor clarifies the pending [r0.1 brief](MINIMUM_REFERENCE_METHOD_BRIEF_R0.1.md):
POINT purpose, runtime policy boundaries and evaluator limits. Preserve r0.1's
bytes. No new package, author packet, architecture split or Stage B dispatch
is commissioned; the numerical choices under Q01–Q06 remain proposals.

## Recommendation

**Make the intent boundary explicit now; propose POINT as the first bounded
reference to complete.** Retain historical Citlali as the mandatory control.
Propose uniform occurrence weights only for this unqualified POINT reference.
Keep the recovered empirical S/N admission policy where its exact evidence is
compatible. Defer a new numerical selector; the existing runtime mode-policy
interface remains required now. POINT, weighting and the recovery targets are
still pending. No observation, test or campaign order is selected.

This makes the method documentable without claiming scientific improvement.
The [weighting screen](../../../../../../../../validation/fruit_q02_mapping_screen_2026-09-10/SCIENTIFIC_REPORT.md)
showed seven synthetic cases with a uniform-weight failure and substantial
native-map changes under N4U. It established neither physical source attenuation
nor a preferred policy. Choosing uniform for a limited reference would accept
its known limitations; it would not resolve the original noise-efficiency concern.
N4U stays recorded evidence, with no further weighting search proposed.

The distinction is already in the frozen definition:

| Level | What stays shared / what belongs to this POINT reference |
| --- | --- |
| Scientific structure | Reuse the generic conditional roles and this ordinary-MAP method's original-parent residual, current-residual PTC relearning, total reconstruction and replacement/revocation recurrence. These reference choices are not universal requirements for every possible FRUIT method. |
| Runtime intent policy | M02's Learn–Consider–Apply must bind the actual priors, evidence, contamination alternatives, admission/sign, flux construction and application. POINT localization/scale assumptions belong here, not in shared recurrence logic or only in an external scoring script. The S/N baseline still needs a complete named policy. |
| Evaluation and acceptance | The evaluator, target quantities, test domain and tolerances below are POINT-specific. They grant no OOF, BEAM or SCIENCE adequacy. No one-to-one mapping between program labels and algorithms is imposed; mixed intents require explicit policy composition. |

**Paper check using OOF:** M02/M03/M05 allow a policy to admit faint, distributed,
disconnected structure without a bright core or Gaussian shape, using the same
declared recurrence and a different evaluator. Localization and Gaussian fitting
are not inputs required by that recurrence. No missing scientific interface role
was identified in this check. OOF numerical policy/input/response bindings remain
unavailable; no OOF estimator or data study is added. Fixed support and unity
application remain reference restrictions, not universal mode requirements.

## 1. POINT purpose and evaluation

The primary purpose is **recovering the pointing offset without bias toward the
nominal position**, while preserving width, distortion and photometric information.
Propose one localized source in the admitted calibrated, nominal-beam quantity:
centroid offset in arcsec, apparent major/minor FWHM and orientation, and peak
amplitude in mJy/beam. No tight position or nominal-width prior may hide the
pointing/defocus information being measured. Exact ranges and softness remain
policy bindings. These apparent quantities do not establish absolute sky level,
deconvolved shape or integrated flux in Jy. A unit label alone is insufficient.

Propose an **evaluation-only** elliptical Gaussian plus planar-background fit
for the subset of POINT responses it can adequately describe. It does not define
all legitimate POINT morphology. Fit amplitude, centroid and shape with declared
unweighted least squares, without a noise-likelihood claim or fixed origin/width.
Retain signed fitted amplitude and the full signed map. The fitted nuisance plane
does not restore lost DC. The fit neither selects feedback pixels nor replaces
their calibrated flux; Gaussian feedback replacement would be a separate method.

Declare the fit region, parameter range, solve, adequacy rule and support before
any test. Missing support, degeneracy or an inadequate Gaussian description makes
that evaluator's recovery result unavailable. **Evaluator inadequacy is not by
itself FRUIT failure.** Keep the case in the record; do not drop it, call it a pass,
or change the evaluator after seeing its outcome. A required unavailable result
blocks the claimed test conclusion. Distorted responses need a separately bound
suitable evaluator and full residual-shape evidence, without changing the feedback
rule. Native fits remain descriptive without an independently justified target.

For controlled cases, truth must supply the injected apparent source and its
sample-domain/calibration relation before residual formation and PTC learning.
Compare fitted parameters with that truth while accounting explicitly for
sampling and map response. A fixed-template projection, a fitted peak, and an
integrated flux remain different observables. No automatic response correction
or integrated-flux conversion is proposed.

## 2. Proposed success targets and evidence limits

These are **new owner-review targets for a later bounded recovery test**, not
accepted tolerances, measured performance, or reinterpretations of the weighting
screen's 2% alerts. The numbers are proposed starting requirements, not derived
from its outcomes. Let A* be nonzero injected peak amplitude and w* the injected
minor-axis FWHM. These are proposed POINT-domain goals, not generic FRUIT accuracy
requirements. Evaluate each predeclared case and required array separately.

| Quantity | Proposed target / required evidence |
| --- | --- |
| Position | Norm of ensemble centroid bias at most 0.05 w*. Report individual errors and scatter, so cancellation cannot conceal poor precision. |
| Peak amplitude | Absolute ensemble bias at most 5% of \|A*\|. Report amplitude RMSE separately; target no more than 10% above the exact historical control. |
| Width/shape | Each fitted FWHM's absolute ensemble bias at most 5% of its truth value; retain orientation and full signed residual morphology. Circular-source orientation is not applicable. An inadequate Gaussian evaluator leaves its shape claim unavailable; it does not diagnose FRUIT failure. |
| Leakage/contamination | On predeclared source and exterior domains, report signed source residuals, exterior residual error and false model admission. Proposed exterior error-RMS limit: no more than 10% above control. Null cases need their own false-admission bound before any probability or acceptability claim. |
| Support | No loss of the controlled reference's required domain, contributors or PTC use population. No selecting successful arrays or silently reducing the evaluation region. |
| Iteration and cost | Retain total/model/support evolution, signed perturbation behavior, terminal causes, runtime and peak aggregate memory. Fixed extent is not convergence; numerical stability guards and resource limits require a declared plan. |

Bias, RMSE and acceptance uncertainty require a declared truth/noise ensemble
and simultaneous interval procedure. Shared pixels, windows and iterations are
not independent repetitions. An unresolved interval is inconclusive; an unavailable
required metric cannot pass. Final domains, error estimators, null bounds, sample
counts and decision rules belong in a separately reviewed Q05/Q06 test protocol.
No ensemble or test has been selected by this brief.

Keep two separate judgments: whether an eventual implementation realizes the
specified recurrence, and whether this POINT policy recovers the intended
information over its declared domain. Passing either does not establish the other.
Existing signed mapping checks support fixed-state N/Q arithmetic only. Existing
spatial RMS supplies no real-data variance or physical recovery. A cleaning or
feedback claim needs response through the named residual construction, PTC
relearning, inference, recurrence and terminal selection. That stronger experiment
needs separate input/response permission. No universal covariance project is a
prerequisite to documenting a conditional method.

## 3. The reference recipe we can preserve now

| Part | Exact recovered definition or proposed disposition |
| --- | --- |
| Historical control | Preserve its original-parent recurrence, actual PCA learning, S/N and any flux gate, JINC projection/mapmaking, weighting, support, stopping and products. Source anchor: `SCI-FRUIT-HISTORICAL-RECURRENCE@f70701ad`, commit `f70701ad488444f3e2528c6bbe3e798863c9e301`. Source identity is not an executed control binding. |
| Ordinary-MAP reference | Retain `FRUIT-FEEDBACK-METHOD/ordinary-map-mode-aware-relearning@r0.3`, as clarified/frozen in paper r0.4, with exact MAP v0.1/r0.7.1. No new scientific-definition vote. |
| Bootstrap and parent | Unseeded k=0: ordinary PTC and complete MAP bundle; model/removal/rejoin are not applicable. At k≥1, subtract the current model from that pass's original-parent CAL realization, never from a previous cleaned result. |
| PTC | Relearn centering and the configured-rank subspace once on each current residual; resolve and hold that state for Apply. Keep the approved centering, identity scaling and one-fit rules. Array grouping remains proposed; exact positive ranks/segments/fit/metric/guard settings remain Q03. |
| Model and composition | Re-infer from the complete reconstructed total; allow entry and revocation. Leading reference: binary admission × unfiltered total-map flux, full replacement, unity application, no new flux floor. Q02-B's matching containing-pixel, scale-one/no-added-offset removal/rejoin remains conditional on exact compatibility and source adoption. |
| Admission | Recover the empirical S/N statistic, sign, threshold/ties, companions and missing behavior exactly. Transfer only where quantity/response/noise evidence permits. Do not use MAP normalization as S/N. Missing evidence keeps this reference unavailable; all-support/both-sign remains a separately named baseline, not a fallback. |
| Weighting and support | Propose Q02-C's dimensionless gamma_i=1 only for this POINT reference per admitted occurrence, fixed from bootstrap with fresh current QC; no inverse-variance claim. Retain fixed D_a/contributors and separate changing model-admission support. Exact coverage_cut policy/value remains Q04. |
| Endpoint and failure | Retain finite predeclared L≥2, including bootstrap; select the final required completed bundle only after required decisions pass. Missing dependencies, instability or resource exhaustion cannot select an earlier successful-looking map. Preserve sibling/partial products with their actual status. |

The [historical dossier](../../historical_control/r0.1/INTERNAL_DOSSIER.md)
and [control-feasibility record](../../historical_control/r0.1/inputs/manager/empirical_controls/EL_G0_HISTORICAL_CONTROL_FEASIBILITY_R0.1.md)
leave the executable, dependency/runtime environment, executed ordered/effective
configuration and full input/product binding unavailable. Prefer a preserved
artifact; a pinned reconstruction needs its own comparison/approval. The recent
FRUIT-disabled PTC exports and template rank/iteration values do not fill these
fields. Do not rebuild or request fresh reductions just to complete this paper.
Without an exact control, a baseline-relative numerical comparison stays unavailable.

Historical JINC and this ordinary-MAP reference may differ in several consequential
choices. Treat them as method combinations unless those differences can be held
fixed under admitted permissions. Any later selector comparison must share one
ordinary-MAP recipe, with each arm relearning on its own residual.

## 4. Decisions for this brief and the bounded next deliverable

The following recommendations are **pending**, using existing decision identities.
Approval can be scoped by row; it would not silently complete Q01–Q06.

| Existing decision | Proposed owner disposition |
| --- | --- |
| Q05/Q06: intended use and recovery | Accept the initial POINT reference for paper completion: pointing-offset recovery with preserved shape/photometric information, the Gaussian evaluator only on its declared adequate subset, and §2's provisional POINT targets. Exact estimator/test bindings still return for review; no observation, run order or population is selected. |
| Q02-C: limited reference | Accept uniform occurrence weighting and the explicitly identified rejoined MAP handoff for this unqualified POINT reference only, under B09–B11 and Q02-A's S1 versioned-profile rule. Acknowledge demonstrated noise-efficiency limits. This is a scientific choice for subsequent controlled amendment preparation, not evidence of detector-noise adequacy or adoption of unspecified sources. If not accepted, leave the coefficient slot unavailable; do not restart a weighting search. |
| Q01–Q06: finish the definition | Prepare one exact reference record: recover the historical binding, fill approved fields, and return the actual unresolved numerical/source choices together. Retain a named POINT S/N policy where compatible; bind its runtime roles now. Defer new selector development, other intents' numerical policies, optional priors, damping and alternative PTC families. |

That next record must supply the exact S/N policy/evidence (Q01/Q05), controlled
PTC/MAP successor clauses and immutable VAL/source bindings including POINT
applicability (Q02), PTC effective plan (Q03), support policy (Q04), and exact
reference inputs, extent, terminal/stability/resource settings (Q06). Q02-A/B
need no repeat substance vote. Historical values are evidence, not automatic
numerical approvals. State any remaining unavailable field rather than filling
it from the weighting test.

**Completion is a precise conditional reference contract with honest missing
evidence.** If a requested recovery claim truly needs new evidence, return one
specific bounded test proposal with its input permissions and acceptance rules.
No test is a prerequisite to restating the already frozen conditional core.
A numerical recommendation still requires independent-pointing replication;
129081 remains unopened for scientific values and is not assigned to a new test.

No upstream source, registry, reduction or archived result is changed. This brief
authorizes no implementation, replay, Unity use, qualification, production or
independent-author dispatch. `FRUIT-FEEDBACK-METHOD =
unavailable_pending_separate_owner_approval`; the route remains
`unavailable_under_current_frozen_parent_permissions`.
