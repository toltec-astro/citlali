# Ordinary-MAP mode-aware feedback: scientific definition and rationale

Paper revision r0.3, 2026-09-09. Proposed specialization identity:
`FRUIT-FEEDBACK-METHOD/ordinary-map-mode-aware-relearning@r0.3`.
The [owner direction](OWNER_DIRECTION_AND_CROSSWALK.md) is binding within its
scope. The complete instantiated method, numeric policies and exact upstream
amendments still need approval. This specializes the frozen generic core;
it neither edits it nor selects a universally best FRUIT policy.

## M01. Iteration parent, domains and products

FRUIT alternates between PTC-owned estimation of correlated contamination from
model-subtracted data and mode-aware construction of a replacement feedback
model from the reconstructed total. Both estimations have assumptions; neither
identifies physical sky or atmosphere solely by producing a correlated feature.

The reference request retains one observation, separately identified required
array bundles, original-parent RTC/CAL reruns with fixed requested controls,
one immutable segment plan, proposed array-wide configured-rank PCA groups and
one fixed ordinary-MAP grid per array. New upstream products get their own
realized generations linked to the same immutable measured ancestry. A frozen
CAL realization is the pre-removal operand for that pass; no previously cleaned
residual becomes its replacement. Repeated upstream realization is not new
independent data. Actual calibration, beam, quantity, coordinates, producer
support and occurrence compatibility remain required.

At k=0 the unseeded bootstrap fits/applies ordinary PTC on CAL and forms the
complete ordinary MAP bundle Q_(0,a). Model candidate/accepted/applied, removal,
feedback residual and rejoin roles are explicitly not_applicable with reason
`bootstrap_no_predecessor`; no missing model is silently made zero. Missing
predecessor at k>=1 fails rather than restarting bootstrap.

Define D_a as the nonempty bootstrap MAP-valid, science-policy-supported output
domain formerly called S_a. Keep D_a, the normalization/contribution population,
PTC use-support populations and WCS/occurrence association fixed for this
reference. Model admission A_(k,a) is a separate, variable subset of D_a.
Missing required data or evidence is not permission to shrink D_a or relabel
unavailable cells as rejected sky. Keep measured, candidate/accepted/applied,
removal, residual-operation, rejoin, final-MAP, learning-influence and response
supports in the core's eight distinct support roles; record exact equalities.

Existing Q_(k,a) remains the complete reconstructed-total MAP bundle. In this
exposition T_(k,a) means its signal with the complete bundle referenced; it is
not a new registry product. Selected feedback is a distinct derived model and
never replaces the unthresholded science map. Original exposure and map validity
remain MAP-owned and do not change when a feedback decision rejects a feature.

## M02. Total-map evidence and replacement inference

After completing Q_(k,a), FRUIT.Learn gathers the selected observing policy's
allowed evidence from T_(k,a), named companions, prior context and exact earlier
model/state references. FRUIT.Consider resolves admission, sign, flux estimator,
floor, gain and transition decisions. FRUIT.Apply constructs the exact accepted
and applied operands for iteration k+1. These responsibilities are runtime
contracts as well as a development workflow, not a required class hierarchy.

Completion of Q_k means the MAP bundle is complete. It does not by itself mean
the larger FRUIT iteration/request is complete: a required next-model decision
can still fail. Preserve Q_k in that case, and block the next action. At the
terminal endpoint, no unused next model is implicitly required; the effective
method must identify its required terminal evidence and decisions.

Each policy record binds: mode/use and version; pixel/feature decision domain;
allowed priors and their softness; competing contamination explanations;
evidence product/population/generation, units, response and noise meaning;
statistic and numerical rule including equality/ties; sign; missing/conflict
behavior; feature-to-pixel support mapping; flux estimator; optional floor;
application transformation and gain; prior-state access; and dependence/claims.
No opaque “confidence” field fills these slots. A probability or posterior
requires its law, hypotheses, prior and conditioning; an empirical score or
tail probability has only its declared meaning.

Re-infer on the reconstructed total, not only a cleaned-residual map. Every
previously admitted feature may be retained, change amplitude or be revoked;
previously excluded structure may enter within D_a. Store reasons and evidence
for both transitions. Do not union new support with old support or retain an
old flux merely because it was previously accepted. Rejoining the current model
is bookkeeping for total reconstruction; it is not permanent acceptance.

The current model is already in T_k. Its reappearance or repeated acceptance
is not independent confirmation. Re-inference permits correction but does not
guarantee that an erroneous feature will be rejected or evidence will improve.
Spatial coherence needs a competing coherent-contamination model, not just an
attractive map. Required missing evidence yields the existing unavailable
decision and blocks that required model construction; a declared limited policy
may operate only under its own explicit evidence requirements.

The intentionally unthresholded baseline accepts all finite supported values
of either sign on D_a. It protects nuisance structure too and remains a distinct
comparison policy. The historical empirical S/N policy is also legitimate;
its actual statistic, optional flux-cut logic and missing-companion behavior
must be recovered exactly. Neither is silently promoted to the new mode policy.

## M03. Flux construction, sign, floor and gain

Leading reference hypothesis: the candidate numerical model is the unfiltered
total-map flux on D_a; the accepted estimate applies the mode policy's binary
admission mask a_(k,a,p). A structure decision may admit weak individual pixels.
On finite, valid, decision-resolved rows,

    mhat_(k+1,a,p) = a_(k,a,p) T_(k,a,p),    a in {0,1}.

Admitted rows retain their original calibrated total-map values. A known
rejection produces a deliberately policy-defined zero correction or the exact
equivalent residual-only branch, tagged as inferred exclusion, never a measured
zero. Unknown decisions, absent values and nonfinite values are not multiplied
by zero to invent a model. The model record separates its valid definition
domain D_a from its admitted support A_(k+1,a). An admitted numerical zero is
still admitted; numerical nonzero support is a separate fact. Complete rejection is a
valid explicitly empty admitted support only when every required decision is
resolved; it is not the bootstrap or a missing-evidence fallback.

Candidate, accepted and actually applied models have distinct identities,
spaces, supports, parentage and recorded transformations. The leading gain
hypothesis is identity application: m_(k+1)=mhat_(k+1). Thresholded flux is not
claimed unbiased. Filtered or matched-filter evidence may determine admission
under a named policy, but filter amplitudes are not substituted for calibrated
total-map values. Fitted-source replacement, denoising, shrinkage, extrapolation
or probability-weighted flux need separate estimator identities and response
treatment; belief that a feature exists is not a flux fraction.

No FRUIT-wide sign prohibition exists. The mode policy declares the sign of
the astronomical quantity separately from signed processing response and map
noise. A physical positivity prior must say how it acts on the chosen model
domain; it does not justify clipping a signed response image or final map.
Legitimately negative signals, including decrements, must be permitted by their
declared policy. Both positive and negative artifacts can be falsely promoted.

An additional flux floor is not required for the leading comparison. If later
requested, bind its target observable (pixel, fitted amplitude, surface
brightness or integrated feature), unit, sign, support, equality behavior and
whether it selects or changes values. No silent clipping/rounding. Inference-
only and inference-plus-floor remain distinct; a per-pixel floor may destroy
structure-level admission and must earn that loss.

Non-unity gain is experimental. Direct attenuation m_next=alpha*mhat_next
changes applied amplitude. Relaxation m_next=(1-g)*m_current+g*mhat_next changes
the transition and can retain newly rejected content. They are different
methods; g is unrelated to MAP's gridding gamma_i. Any relaxation proposal must
state whether rejected support is cleared immediately or decays as separately
marked legacy content, its termination, and its influence. Neither rule is
selected here. Smaller gain is not assumed safer, and unity is not proven stable.

## M04. Current-residual PTC and total reconstruction

For k>=1, m_(k,a) is the exact applied model obtained from Q_(k-1,a). In the
authorized difference spaces and on declared support,

    rho_(k,a) = CAL-Y[k,a] - Pi_(k,a) m_(k,a)
    Theta_(k,a,g) = PTC.LearnConsider(rho_(k,a); fixed recipe/context)
    c_(k,a,g) = PTC.Apply(rho_(k,a); Theta_(k,a,g))
    Zjoin_(k,a) = c_(k,a) + B_(k,a) m_(k,a)
    Q_(k,a) = ordinary_MAP_complete_bundle(Zjoin_(k,a), gamma_0, current QC).

PTC owns the estimator and its evidence/resolution/application. FRUIT owns
residual construction, placement of that operation, sky inference and outer
recurrence. The PCA reference learns current centering and detector-loading
subspace once per immutable residual/group/segment under the same configured
positive rank, grouping, support/metric/fit rules and tolerances. It uses the
ordinary binary-support arithmetic centering and identity internal scaling.
It resolves Theta_k then holds that state fixed during its Apply and conditional
queries. Internal modal coordinates are evaluated under that resolved map;
that evaluation is not subspace relearning. Equal successive subspaces are
allowed. No automatic rank selection, support-changing refit, forced change,
centering restoration or earlier-Theta fallback is introduced.

The model value path bypasses direct residual processing, but its subtraction
changes the learning input and therefore can change Theta_k. Retain this causal
influence and the current discarded lambda_k. Applying PCA's resolved subspace,
subtracting a frozen numerical component and rerunning full learning are
different contracted operations. The full resolved PTC map is affine when it
carries a nonzero discarded centering location; linear illustrations do not
erase that offset or restore optical DC.

FRUIT's general PTC boundary requires estimator/method identity, actual parent,
learned-state identity, resolution, application semantics, valid output/support,
response/noise states and failure. It does not require eigenvectors universally.
Robust correlated-component methods, weighted/noise-aware PCA/factor models and
common-mode plus low-order focal-plane structure are candidate PTC families.
Each must have its own exact approved contract and be re-estimated as that
contract prescribes on the current residual. No alternative is activated or
allowed to weaken PTC obligations in this revision.

The proposed reference Pi/B use MAP's exact same AST/WCS half-open containing
pixel, no interpolation/extrapolation, and scale-one/no-added-offset correction
embedding B08. Required removal/rejoin supports agree per applied model. Outside
D_a use the declared residual-only branch. MAP normalization and final D_a rows
stay separate from the feedback projection. Let H_a denote the frozen-state
normalized MAP operator on its exact contributing occurrences and D_a rows.
Only with finite complete restoration and compatible quantity/reference,

    H_a B_(k,a) m_(k,a) = m_(k,a),
    T_(k,a) = H_a c_(k,a) + m_(k,a).

Otherwise retain T=MAP(c+B m); do not simplify it or invent a missing operand.
This conditional applied-model round trip is not unit astronomical response,
true-sky recovery or inverse projection. The complete bundle is required for
next inference, not a bare or thresholded signal plane.

## M05. Mode-policy interface and scientific consequences

The policy slots in M02 are required now; incomplete numerical policies remain
unavailable for application. No mode automatically inherits another's statistic,
threshold, flux observable, shape prior or downstream fitness.

| Regime | Permitted direction and required evidence meaning | What must stay measurable / experimental |
| --- | --- | --- |
| POINT | Dominant localized source; soft location/scale priors and declared response, noise and contamination alternatives | Centroid may be substantially offset; width/shape may reveal defocus or distortion. No tight nominal-position or in-focus-beam prior. Exact score/prior bounds remain Q01. |
| OOF | Source presence comes from observing design; policy must accommodate faint, distributed, distorted or disconnected response without a central anchor | No bright core, high-S/N seed, ring, contiguity or narrow morphology requirement. “Fourteen or more” deformation terms is not a cap/model. Coherence and cross-focus/observation evidence remain experimental and may not impose the surface being measured. Human-visible structure is not quantitative truth. |
| BEAM | Detector-specific location and calibration-relevant flux, with soft location/compactness and an exact S/N convention | Centroid, shape, width and amplitude remain free. Discussion-scale S/N of a few tens is contextual, not a threshold or interchangeable peak/integrated/per-pixel statistic. Ensemble borrowing is experimental; no forced agreement with strong detectors. |
| SCIENCE compact blank field | Unknown positions, compact-source response and declared search/noise population; historical S/N and matched-filter-style evidence are candidate comparisons | Feedback admission is not a catalogue detection. Filter outputs do not automatically supply projected calibrated flux. Catalogue and filtering ownership remain separate. |
| SCIENCE extended / mixed | Explicit scale/shape/sign priors and alternatives for bright extended emission, faint outskirts and faint diffuse/decrement targets | Coherence is not origin. Allow legitimate negative signal. A mixed field can contain all regimes; policy composition/switching must be explicit, with no hidden universal SCIENCE selector. |

The first numerical reference remains one-observation ordinary calibrated MAP.
OOF/BEAM/science interface definitions do not supply missing calibrated inputs,
mode registration, raw-frequency map authority or downstream fitting contracts.
These precise limitations are B12/Q02, not a reason to postpone the interface.

## M06. State, QC and failure

Use the core categories at the declared scope: upstream controls and policy
versions externally supplied; PTC state and sky evidence relearned per pass;
resolved PTC maps fixed within Apply/RF-01/02; current decision/application
values and transformations explicitly recorded; direct model path bypassed;
unrequested floor/alternative/gain operations not_applicable; required missing
facts unavailable. Any fixed structure with recomputed operator-defining
coefficients is declared by its own method; internal modal coordinates are
not an eighth category. Changed fitted values are new state generations;
changed estimator/selector/sign/gain/support/transition rules change method.

Retain the proposed constant dimensionless analysis/gridding gamma_0 family
and its bootstrap value generation, independent of learned PCA coefficients.
B09 specifies its family/population and fresh current compatibility/QC. It
is not precision, covariance, S/N or NOI evidence. Relearning does not silently
reweight; dynamic inference support does not alter MAP contributor membership.

Every required current occurrence, model decision, path operand, centering,
rank solve, coefficient/QC and output publication must satisfy its own contract.
No zero-fill of missing values, borrowed solve, lower rank, support shrinkage,
renormalization or prior fitted-state substitution rescues failure. Intentional
zeros after known inference rejection have the distinct M03 meaning.

The all-required-array barrier is a request-completion policy. Advance k only
after all required array iterations complete. A required failure makes the
larger request failed and blocks its next k; it does not invalidate a completed
sibling's immutable product or establish statistical independence. Preserve
completed products and failed/partial attempts under their actual states.
An explicitly chosen historical fallback is a separately bound method/run,
never a relabeling of this failed run or replacement of its retained evidence.

## M07. Extent, continuation and terminal product

Retain finite integer L>=2 including bootstrap k=0; no L value is selected.
Successful required iterations are 0..L-1. The reference terminal rule selects
Q_(L-1,a) only after all required products complete and any requested required
stability/termination decisions pass. This is a predeclared endpoint, not a
claim that the last or highest-flux map is best. No earlier-product rescue
after a required failure, instability stop or missing terminal dependency.

Record all nine frozen terminal fields, including candidate population,
rule/causes, convergence/resource/failure facts, RF-05 and uncertainty states,
continuation/terminality and requested named-use decisions. Compare on fixed
D_a with model admission changes reported separately. Same reference convention
does not require equal learned lambda or subspace; any needed conversion is
explicit and cannot restore unknown modes.

Retain every complete Q, evidence/decision, candidate/accepted/applied model,
PTC fit/resolution/application generation, support, coefficient/QC generation,
transformation, failure and terminal cause, with exact reconstruction where
permitted. No new registry or file format is required. No restart/continuation
sufficiency is claimed from persistence alone. Runtime and peak aggregate
process-tree memory include per-pass learning, inference and retained products;
limits/counters remain Q06. Resource exhaustion is a failed limited attempt,
not convergence or successful completion of the required extent.

## M08. Convergence, stability, recovery and uncertainty

Convergence requires a declared quantity/domain, norm/denominator, tolerance,
direction/persistence, missing behavior and stopping consequence. Examine model,
reconstructed total, admitted-support changes and the mode's relevant scientific
observable on fixed comparison support. No tolerances or data-driven convergence
stop are selected here. L completion records convergence unassessed unless an
exact separately requested criterion was evaluated.

Stability concerns response to positive and negative model perturbations,
near-threshold decisions, support transitions and learned-state changes. Report
decay, persistence, oscillation or growth on declared measured and unconstrained
subspaces. Compare PTC cleaning action/subspace, not arbitrary eigenvector
sign/order. A requested instability guard that triggers stops with an explicit
failed/unstable cause; its metric/threshold/action must be bound before use.
No silent damping, ignored instability or last-map success label follows.
The local Jacobian of a declared update can be studied where differentiable;
discrete transitions require direct tests. No universal contraction proof or
global stability follows from bounded empirical tests.

Scientific correctness separately concerns recovery, bias and contamination.
A stable wrong fixed point can preserve artifacts or suppress genuine flux.
In the fixed-linear compatible illustrative limit, C(y-m)+m=Cy+(I-C)m:
an erroneous model component removed by C can persist through the model path.
This is not the relearned estimator or a sign-based stability theorem. Neutral
behavior in an unidentifiable mode differs from amplification in a measured
mode. Prior/model-supported content remains labeled as such, never independent
measured recovery; iteration creates no information about missing modes.

RF-01/02 condition on current complete Theta_k and reevaluate fixed-map internal
coordinates. RF-03 includes this pass's prescribed relearning and every
inference/update operation inside its declared query boundary. Bind whether
incoming m_k construction from Q_(k-1) or outgoing m_(k+1) construction is in
that boundary; earlier state held external to it remains explicit conditioning.
This indexing convention cannot silently omit a prescribed learned operation.
RF-04 propagates dependence through fixed extent; RF-05 includes consequential
termination/selection. RF-06 and RF-07 retain their distinct upstream-rerun and
fixed-upstream-generation parent-domain query scopes. Missing query authority
or inputs are unavailable. B07 preserves PTC REQ-099's stronger-tier exclusions.

Keep all thirteen uncertainty roles and seven response roles with exact
conditioning. Model, residual, total, selector and PTC fit share data and state.
Repeated acceptance is not independent detection, and more iterations do not
shrink errors by themselves. Fixed-state noise and fully relearned/selected
uncertainty are different targets. Formal weights establish no significance.
NOI owns its law and uncertainty methods; VAL evaluates owner-supplied rules.
No covariance independence, unbiased flux, source recovery, pointing accuracy,
OOF solution, beam calibration, catalogue claim or production fitness is inferred.

## M09. Twenty-one-class record

| Class | Current disposition |
| --- | --- |
| 01 Authority/lineage | Owner direction and proposed method identity above; full packet/amendment/configuration/execution states separate. |
| 02 Route/grouping | Original CAL→FRUIT residual→PTC pass→FRUIT rejoin→ordinary MAP; proposed array grouping, mode interfaces and B01–B12 limits. |
| 03 Input rule | M01/M04 original measured ancestry and immutable current CAL/residual; never cumulative cleaned input. |
| 04 Target/model | M02/M03 reconstructed total and separate candidate/accepted/applied replacement model, feedback suitability only. |
| 05 Construction/selection/application | Required mode-policy evidence/statistic; leading binary support times total flux, explicit sign/floor/gain, revocation. |
| 06 Accumulation | Full replacement; algebraic rejoin is not permanent acceptance or increment accumulation. |
| 07 Typed composition | M04 and B08 exact removal/process/rejoin domains; affine PTC and conditional MAP round trip. |
| 08 Additive/unavailable modes | Reference convention explicit; policy-zero versus missing; no lost-mode or true-sky restoration. |
| 09 Operation graph | Runtime FRUIT Learn/Consider/Apply and PTC-owned Learn/Resolve/Apply, with evidence/decisions at named boundaries. |
| 10 State by operation | M04/M06 fresh learned states, fixed application scope, actual model influence and method/state distinction. |
| 11 Support/missing | Fixed D_a and required measured populations, dynamic admitted A_k, exact path supports and unavailable handling. |
| 12 Update/transition | Re-infer from Q_k for k+1; unity transformation reference, revocation; non-unity gain remains experimental. |
| 13 Response | M08 exact seven roles; current-Theta conditional versus relearned/recursive/terminal scopes. |
| 14 Uncertainty | M08 dependent data/model/selection/fit roles; NOI authority and missing-law limits. |
| 15 Continuation | M07 causal state retained; no continuation sufficiency or restart claim. |
| 16 Stochastic state | Reference policy deterministic once bound; any stochastic PTC/inference/experiment needs exact generator/state/population or justified not_applicable. |
| 17 Convergence | M08 separately declared criterion, no tolerance selected; finite completion is not convergence. |
| 18 Resources | Whole request/stage time and peak owned-process-tree memory, including learning/model storage; Q06 exact bindings. |
| 19 Terminal selection | M07 final required completed endpoint only after required success; no highest-flux or failure rescue. |
| 20 Persistence | Complete totals, model stages, evidence/decisions, learned states, attempts and existing lineage structures. |
| 21 Completion/failure/publication | Unseeded bootstrap, per-array product truth, all-required-array barrier, explicit instability/resource/failure causes. |
