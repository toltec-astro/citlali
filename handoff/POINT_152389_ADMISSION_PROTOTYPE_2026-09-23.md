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
