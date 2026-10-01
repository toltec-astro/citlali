# POINT to OOF — 2026-10-01

POINT has an honest mixed checkpoint: more source response through feedback, no
established R/S improvement in photometry, pointing accuracy or sensitivity, and
substantially less supported area in the operational successor maps. Calibration
lineage explains the factor difference; absolute calibration preference remains
open. Three corrected152420S roles are the only pending POINT batch. See
[assessment](../validation/point_quality_2026-10-01/ASSESSMENT.md). OOF is next;
no additional POINT tuning campaign and **no OOF observations launched here**.

## Authority and earliest unsupported boundary

The adopted October1 directive is explicit: OOF feedback must not require a bright
core, highS/Nseed, ring, connected footprint, Gaussian shape or narrow morphology
family. Cross-focus coherence remains a candidate, not adopted inference. The
September9`SCI-FRUIT-owner-directive-2026-09-09.md` is still unavailable after
bounded local/worktree/history searches; this recovery gap was already recorded
in `handoff/MAP_SUCCESSOR_FIRST_FORWARD_001_2026-09-21.md`. Do not reconstruct its
missing content from the filename.

Recovered exact September8 conditional freeze in sibling346d:
`doc/scientific_contracts/packages/SCI-FRUIT/v0.1/scientific_core/SCI_FRUIT_STAGE_B_R0.4_OWNER_CONDITIONAL_FREEZE_2026-09-08.md`.
Corev0.1/r0.4, reviewed5ce01b62ddfb6bb57120e872a02da0fd8cfae5ed,
inventorySHA3dc912d9f6a58d9f92561ad10b9442485c5b47232e5327389d2fb3b89c296e92.
It freezes generic recurrence/boundaries, leaving numerical methods unavailable
pending separate authority. September11RBF directive explicitly authorizes a bounded
POINT experiment, not universalOOFpolicy. October1supplies the current mode-aware
constraints. No missing historical instruction is treated as an implicit policy.

Existing MAP OOF attachment envelope is at
`doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/OOF_ATTACHMENT_ENVELOPE.md`:
require parent/frame/beam/covariance/support ownership; pointing displacement is
not an automatically applied correction, and apparent width is not intrinsicOOF.

**First execution blocker is mode/acquisition routing.**
`tools/timestream_successor/point_pipeline_inputs.py:16–17` acceptsPOINT only,
subobservation0/scan2; this OOF sequence is scan1. Its downstream source gate and
feedback contract are also POINT-specific. Merely changing the input label would
misrepresent the native acquisition and import a bright-source guard. Shared
RTC/CAL/PTC/MAP owners are reusable after explicit mode-aware request/identity
fixtures; the POINT entry point is not an OOF executable. No second pipeline or
silentPOINT fallback is prepared.

## One complete local sequence

Nominate acquisition-definedOOF **152385/0/1,152386/0/1,152387/0/1**,
source1146+399,2026-02-19UTC. Inventory is
[exact local metadata and dependencies](../validation/point_quality_2026-10-01/results/oof-inventory.json).
All three have the same eleven recorded networks0,1,2,3,4,5,7,8,9,11,12,
covering a1100(0–5),a1400(7–9),a2000(11,12). Networks6and10 are not present in
these acquisitions; they are not labeled missing downloads. Each points to exact
Tune152384/0/1. Raw headers/bytes and KMPtextSHA256 were checked against canonical
APT-v2 source records. FullrawSHA values are bound from the manifests, not falsely
reported as rehashed by this metadata inventory.

| Observation | RequestedZ(mm) | Zoffset from middle(mm) | Actual actuatorZ(mm) | Raw bytes |
|---|---:|---:|---:|---:|
|152385|−3.5665967173|−2|−20.9586143494|351642632|
|152386|−1.5665967173|0|−19.0036315918|352751216|
|152387|+0.4334032827|+2|−17.0463314056|352010712|

The sign here is increasing requested physicalM2Z, not an inferred optical
wavefront convention. ActualZincludes elevation-dependent corrections; offsets
are not derived by subtracting rounded actuator readings. No assertion that the
middle acquisition is best focus. TelescopeAcquisitionTime spans and elevation
ranges are in the inventory, about65–67s each around06:18–06:22UTC; requestedfocus
headers and allM2corrections are preserved. Exact UTCs appear in the metadata
summary appended below. Focus states stay separate throughout.

Raw/telescope/tune source paths are listed per network/member under
`/Users/gwilson/work_toltec/local_data/citlali-validation/v1/oof/data/`.
Canonical matchedAPT-v2 components and reports are under
`/Users/gwilson/work_toltec/local_data/citlali-validation/v2/oof/apts/v2/`,
with148670beammap parent. ComponentSHA256, manifest and matched-report identities
are retained. Total localraw1,056,404,560bytes; missingpaths[]; missingrawbytes0.
No archive download or Unity action is required for this nominated sequence.
Input readiness still requires fullcontent verification at future execution freeze,
actual detector-row selection/calibration admission, and native mode support.

## Actual reference and smallest first comparison

Historical control:
`.../citlali-validation/v2/oof/refactor/1146+399/reduced/redu00/`;
configuration`citlali_o152385_0_1_c3.yaml`,SHA256
26298ac72682ff0a99d70cdc36beb85ea8b601cf18826c0e2f021ac5402e2d43.
It actually sets `runtime.reduction_type:oof`,1arcsecJINC/AltAz,
coveragecut0.5, standardPCA3 pernetwork,validatedweights, lowpass16Hz/32terms,
downsample2,extinctionenabled. Preserve exact config for remaining settings.
`source_strategy:psf_preserve`,Gaussianmeasurementenabled,map-center strategy.
Crucially **FRUITenabled:false**: `sig2noise_limit:100,max_iters:1` are inactive
configuration fields. This archive does not demonstrate active S/N-based feedback;
do not invent such a historical control. Its30arcsecsource-mask/20arcsecprotection
and Gaussian diagnostic are disclosed historical assumptions, not currentOOFtruth.

First discriminator after mode routing/fixtures: **a1100networks1–4, all three
focus states, cold M0 only, no FRUIT or source admission prerequisite**. This is
an explicitly changed geometry/PTC control, not full-instrument reproduction.
Reuse historical full-array maps as context; for a matched operational comparison
prepare3R+3Sroles only when the existing R implementation can select the same
population without a parallel pipeline. Rreproduction preserves the effective
historical config except that named population restriction. Suses existing paired
RTC/CAL/observed-entryALS/nativeordinaryMAP contracts with2arcsecFK5grid, cold
rank3 inherited as the **declared historical-rank control**, not POINT rank10;
cap500/tolerance1e-5 are numerical solver settings, no tuning. A future freeze
must verify rank3 is supported in the mode-aware route; failure is a route blocker,
not license to substitute rank10. No feedback applications, no highS/Ncore guard,
no default40arcsecsupport. Whole acquisition-derived footprint/support retained.

Those six are a **next-milestone proposed maximum**, not starts authorized/charged
against this POINT batch. Reuse any genuinely compatible products; do not run six
because six slots were described. A matched historical launcher and nativeOOF
mode are currently unavailable, so there is **no valid OOF launch command yet**.
Smallest immediate work is one mode/scan1 identity + full-footprint fixture and
saved-map unit/frame/solver-adapter check, with zero observational passes. Then
freeze the first three-stateM0 comparison and actual footprint/noise support.

Resource envelope: serialfocusstates and roles, at most2networkworkers and
1numericalthread,1active trajectory, shared30GBceiling/2GBfreereserve. Preliminary
reserve7GBactive+2.7GBpercompletedrole+1.1GBreport/metadata, six-role peak≤21.6GB
before prior retained evidence. This is an admission ceiling, not a measuredOOF
prediction. Exact schedule/cellcount reservation must be computed from the mode-aware
prepare step and combined with retainedPOINT bytes; compact prior owned scratch
first if required. Do not assume these additional sixroles fit alongside thisPOINT
batch. NoOOFrawdownload is budgeted because needed acquisition inputs are local.

**A real scientific decision remains before feedback:** the distributed-signal
admission/uncertainty policy and its endpoint/work budget. No numericalOOFmethod
has been recovered as accepted. Cross-focus coherence, RBF inference or another
approach requires an explicit evidence-backed method decision; it is not settled
by extending POINT's radius. The M0boundary experiment can characterize PTC/support
without presuming that feedback policy. It cannot validate absent low-brightness
feedback recovery. POINT threshold3,rank10,N5,Gaussianguard and40arcsecprior are
not adopted as OOF defaults.

## OOF measurements and established solver

Use free distributed morphology diagnostics over one acquisition-supported domain
perfocusstate, declared before successor outcomes. Preserve additivebackground,
normalization and shape as separate observables. Evaluate fixed spatial scales,
positive/negative bowls, asymmetric wings, lowbrightness distributed structure,
correlated background, falsepromotion/nullmaps, acceptedexposure/sourcecrossings,
common support and original-domainloss, time/memory. If the scan does not provide
source-free background, say uncertainty is unavailable rather than fitting noise
to the distorted source. No ring/core/contiguous-footprint requirement.

For later response evaluation, use preRTCoriginalx perturbations, unchangedr and
nativevalidity, independentnormallearning, known broad/asymmetric inputs defined
independently of the inferencebasis. State scale-dependent transfer and amplitude
response separately; compare conditionalnoise only with a compatible measuredgain.
Do not optimize the PSF to look narrower or suppress its actual wings. Correlated
full-sequence dependence must remain explicit; crossfocuspooling is not authorized.

An existing inversion path is local`/Users/gwilson/GitHub/LMTOOF/python/OOFProgram.py`
(`OOFProgram`/`OOFLMProgram`,model/solverinvocations,Zernikeparameters) with
`OOFDriverFrontend.py` and`TolTECBeamMap3.py`. Reuse it only through a checked
adapter. Verify wavelength,angularcoordinates,beamnormalization,focusunit/sign,
pointing-offset convention and support/uncertainty first: the reader supplies
fallbacks for absentOOF_W,OOF_M2Z and otherheaders, which cannot establish provenance.
Do not use defaults as measured facts. Compare each arm's per-focus forward-model
residual maps and weightedresidualstructure, fitted Zernike/wavefront changes and
surfaceRMS under identicalsolveroptions. Report modeldependence and condition/
uncertainty; agreement with its own fitted model is not independenttruth. No new
inversionsolver, actuator-command output, telescope-control change or automatic
surface correction is part of this handoff.

Actual telescope-stream timing/elevation (AcquisitionTime, TelElAct):

- 152385: 2026-02-19T06:18:02.890798+00:00 to 2026-02-19T06:19:08.283875+00:00, 65.393s; elevation 56.4638–56.6479deg.
- 152386: 2026-02-19T06:19:41.253038+00:00 to 2026-02-19T06:20:48.283409+00:00, 67.030s; elevation 56.7381–56.9309deg.
- 152387: 2026-02-19T06:21:10.800409+00:00 to 2026-02-19T06:22:16.089969+00:00, 65.290s; elevation 56.9870–57.1648deg.
