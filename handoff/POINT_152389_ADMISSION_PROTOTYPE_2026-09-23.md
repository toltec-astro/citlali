# Bright POINT connected-support admission prototype

## Work order / preflight

Owner directive: attachment4def4dc7-4111-4a16-939c-c2309faa533b/Pasted text.txt,
"bounded bright-POINT feedback-admission prototype". Tier2, one scientific-module
probe continuing the completed MAP+ diagnostics on existing branch
codex/map-successor-first-forward-001, clean worktree
/private/tmp/citlali-point-additive-20260923, exact base
`ecf9aaf3c7911da588b26828c418f5193c02aee7`. No additional branch, no integration.
Main checkout unrelated dirt untouched. Read all three normative governance
files and verified their digests against accepted06a3ade51... and canonical
775078363... effectiveness record. Read current status, MAP/POINT/additive
handoffs, saved fitter and companion boundaries. Existing authority remains
unchanged; current owner directive defines this explicitly provisional experiment.

Inputs: original rescued obs152389 maps and source descriptions; completed
Sep23 additive companion-r2, exact manifest
 d650deddab4886d9ad386183a6425e5309643cab8effb82fd645106dde3f38c8,
source d70233e1c22b96065f700d43ed65f952da3cf32a. Verify all parents with existing
boundary, source parameters, noise model/masks, generator and method code hashes.
No uncertainty-model relearning or completed uncertainty campaign rerun.

New companion owns candidate admission/diagnostic products only. Direct immutable
producer inputs MAP/source-fit/noise-model -> explicit threshold controls ->
candidate brightness/support -> diagnostic fit/table. No Engine or pipeline
orchestration growth. Small41x41 source stamps and32 trials per array, sequential
small fits with math threading1; no heavy per-sample objects or upstream work.
New files point_admission.py, run_point_admission.py, point_admission_report.py,
test_point_admission.py, POINT_ADMISSION.md, this handoff/status.

Method: fixed40arcsec-radius disk about AST center; anchor nearest measured
centroid pixel, zero-based coordinate rounded floor(pixel+0.5), exact half ties
toward increasing index. Eight-connectivity. Joint finite MAP/noise validity,
strictly positive scale, no extrapolation. D=M-b0, cases(t,scale multiplier)
(3,1),(2,1),(5,1),(3,2), last equivalent original-scale threshold6. Retain D on
anchor component, zero on valid deliberate exclusions, NaN on invalid/outside
domain. No fallback, smoothing, dilation, hole fill, taper, rescaling, Gaussian
replacement, negative-bowl admission or threshold tuning. Boundary means an
admitted pixel has an eight-neighbor outside the analysis disk. Unsupported
contact means an eight-neighbor inside the disk lacks joint validity; record all
unsupported disk pixels separately. Missing fit footprint withholds full diagnostic
fit rather than silently shrinking it. Fixed-template q and footprint stay frozen
to original source and are applied to D/candidate/excluded with consistent b0.
No brightness sum is converted to integrated flux.

32 existing deterministic identities per array (seed23152389,array,0..31), same
noise across cases, clean saved Gaussian+constant baseline. Remeasure each raw
trial to obtain its own centroid/background; compare against saved additive trial
parameters, then measure its unmasked D and candidates with unchanged fitter.
All constraint/failure states retained. Figures must show real wings, excluded
structure and support changes, not rely on unchanged maximum pixels.

Gates: focused anchoring/connectivity/invalid/background/scaling/nonmutation and
failure tests; successor suite, baseline suite with supplemental current-source
cached native CLI, required config preflight; exact parents/artifacts;32 trials;
visual QA; independent fresh-context exact-SHA review (required by governance).
Unchanged CTest/Spack/Unity/upstream behavior retains earlier evidence; no new
representative campaign triggered. Scientific suitability and implementation
review remain separate. Stop after one recommendation and reproducible products;
no FRUIT loop, timestream subtraction, PTC relearning, rejoin, integration, push,
activation or cleanup. A2000 caveat is carried forward, not a prerequisite to
this authorized bright-source comparison. Consequential method/ownership/scope
conflict triggers reassessment; routine numerical/reporting choices remain here.


## Bounded input-binding repair

The first committed run stopped before candidate generation: the full saved
amplitude crop contains NaNs outside noise support, whereas additive make_context
stores zeros there solely for numeric noise generation, with separate validity.
The initial equality check compared these representations directly. Match the
already accepted finite-or-zero generator representation for that binding only;
retain the original scale/validity for admission so unsupported pixels remain
invalid and cannot receive zero uncertainty. No inference or candidate method
changes. Preserve the failed attempt; rerun exactly the same32 identities. Focused
invalid-support tests cover zero/NaN scale exclusion. This is an input-adapter
repair within the owner-authorized boundary, not a new scientific choice.

## Completed run and recommendation

Numerical runtime source `b9d455af113d46d42b83ba2ac7087de017e121ab`, tree
`f4c98067483fee5b105823a9d182ad42306b67a8`. Durable root:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-admission-prototype-20260923`.
Use root ASSESSMENT.md and companion-r2. Initial companion/log remain preserved
as input-binding failure evidence. No candidate or trial was generated there.
The owner directive is retained verbatim as OWNER_WORK_ORDER.md.

Recommend retaining the reference threshold3/existing pixel-scale candidate in
all arrays for a first separately authorized bounded FRUIT experiment. No
production rule, FRUIT execution or promotion follows. Actual fitted amplitude
changes+0.360/+0.953/+0.247%, maximum absolute width changes0.710/1.226/0.418%,
centroid changes0.0060/0.0015/0.0227arcsec for a1100/a1400/a2000. Frozen-template
projection excluded0.116/0.225/0.149%; this emphasizes the original core and is
not an integrated-flux loss fraction. Actual reference areas112/152/228arcsec²,
28/38/57 pixels, well inside the analysis domain. Each1257-pixel fit disk has
complete MAP/noise support; no boundary or unsupported contact in any case.
Actual anchor ratios56.70/50.20/31.44 are descriptive, not significance.

All four actual cases retain source association. Actual permissive/reference/
restrictive/stress areas are124/112/88/80,184/152/116/112,232/228/180/164arcsec².
Threshold-dependent changes lie at the periphery in the inspected figures; no
obvious unrelated structure dominates reference. This does not prove that all
excluded brightness is noise. At doubled scale, actual fit amplitudes increase
2.66–3.44% and widths contract up to5.19%; threshold6 equivalence is explicit.
That increased peak is a refit response to clipping, not candidate rescaling.

All96 raw-source fits and96 unmasked source-relative fits succeed; all384
candidate fits succeed with no constrained states, no boundary/unsupported
contacts and32/32 source admission in every case/array. Original measured
centroids/backgrounds are recomputed per trial and reproduce the prior additive
trial vectors. Reference median paired amplitude changes+0.88/+1.30/+2.20% show
clipping-related diagnostic bias. Candidate reference trial area ranges100–124,
136–164,224–256arcsec² versus permissive108–200,152–220,264–312arcsec².
A2000 reference median major/minor narrowing2.55/2.44%, maximum3.63%; stress
narrowing reaches9.76/9.07% with fitted peak change up to7.42%. Preserve this
material conditional sensitivity despite certain admission. Actual a2000
threshold2→3 changes only4arcsec² versus synthetic median48arcsec²: the Gaussian
trial source does not establish actual non-Gaussian morphology. A2000 spatial
representativeness remains a caveat, not a prerequisite waived or repaired by
this task. No independent spatial-model test is claimed from scale×2.

Candidate validity, NaN/zero distinctions, signed exclusions, frozen-template
projections and every trial fit/constraint/failure are persisted. Original
maps/source/additive products are unchanged. Companion36.891s wall,11.038s CPU,
453.3MB peak RSS, one worker and math threads1. First-figure font-cache startup
accounts for15.406s plotting; inference campaign not rerun. Per-array verification,
measurements, trials and plotting times are separately recorded.

## Gates and independent implementation review

Seven focused tests pass; final successor suite229 tests/111 subtests passes;
required config preflight passes without drift. Supplemental local cached
AppleClang/Homebrew CLI rebuilt at exact b9d455af1 for baseline protocol tests;
all207 baseline tests/137 subtests pass. Rebuild uses the prior separately
recorded local realization; no package download, build adaptation or upstream
reduction. Duplicate-library link warning remains; no unexpected error-level
output in successful runs. Native CTest/Spack/Unity are not newly rerun or
claimed for this Python-only saved-product companion. Temporary baseline-test
build symlink removed after tests; unrelated main-checkout dirt untouched.

Fresh-context independent reviewer bound ff73cdd0 then repaired b9d455af1.
Scientific/behavioral pass with limitations, architecture/ownership pass, and
repository/numerical-evidence pass; no open implementation findings. Independently
verified35 manifest entries,78 FITS HDUs, unchanged parent-map/measurement hashes,
96 exact seed/raw-fit reproductions and all384 trial plus12 actual candidate
masks/images using separate flood-fill reconstruction. Reproduced7 focused tests
and inspected all3 figures. Final companion manifest SHA256:
`42e656d6fedefc1a47599333983bc5a7db31acb42e78be0ec3449a80fe577f71`.
Final owner assessment, broader gate evidence and completion-document candidate
require their own exact-SHA closure review.

Stop at this recommendation. Subsequent authorized FRUIT work must start from
immutable parent data, relearn input-dependent PTC state after sky subtraction,
reconstruct the total, and infer a replacement model that remains revisable or
removable. This task implements none of that loop, and performs no integration,
push, activation, production qualification, cleanup or final calibration claim.
