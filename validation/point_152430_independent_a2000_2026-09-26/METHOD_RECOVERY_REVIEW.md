# Independent frozen-method review — 2026-09-26

Scope: read-only method recovery before exact-candidate preflight for obs152430,
a2000. Initial source HEAD43844bfe07b81c3f70bd01f370f46820982f5bba, worktree
/private/tmp/citlali-point-additive-20260923. No observation processing, fit,
covariance-learning, calibration, injection, feedback, or repository edits by
this reviewer. This is preliminary method advice, not candidate approval.

Read AGENTS.md and all three doc/governance documents; governance effectiveness
and accepted hashes recovered from doc/INTEGRATION_LEDGER.md (accepted
06a3ade51c1b3f38887295433d913811bf25cd14, incorporation
77507836325eff9f469062d5884481ea37599594). Read owner directive
/Users/gwilson/Downloads/CODEX_INDEPENDENT_POINT_A2000_FROZEN_METHOD_VALIDATION.md.
Applied toltec-context routing and read repository/authority maps. Recovered
completed guarded, expected-source, correlation, post-FRUIT, C+D, and importance
records and actual Python scientific owners. Input selection is separately reviewed.

## Method binding and potential blockers

1. The BASE admission amplitude is not the C0 amplitude. This is explicit in
   run_point_corr_admission.py:46 and its inputs.npz construction: `base_scale`
   comes from original additive-r2, three-major-FWHM source exclusion;
   `scale`/PSD come from post-FRUIT four-major-FWHM union exclusion. Replacing
   base_scale with C0 scale changes the accepted selection method.

   Original amplitude construction: full observed support with finite signal,
   finite positive NORMALIZATION; source-centered circular radius3*major;
   global ordinary background mean; five coverage quantile bins, unique edges,
   last edge advanced with nextafter; per-bin sum residual-squared/(n-1),
   minimum100 pixels; endpoint continuation plus explicit amplitude_valid flag.
   Finally multiply every finite bin scale by sqrt(c0), where c0 is zero-lag
   normalized background second product. It is exactly mean(z[background]^2)
   mathematically (FFT realization differs only in roundoff). This factor must
   not be omitted merely because an amplitude-only extraction skips PSD work.
   An extracted amplitude-only stage may preserve these semantics without a
   second covariance-family learn. Verify against retained compact 152389
   amplitude fixtures under the existing numerical tolerances.

2. Full C0 must be learned once from the new unperturbed M0 and explicit
   four-FWHM exclusion. Freeze request: mask_fwhm4, lag_pixels32,
   taper_power1, bins5, minimum_pixels100, minimum_pairs200, region=all,
   domain_factor1, stamp_size49 (padded100 in retained implementation).
   Historical seed923152389 is generator identity only; no random draws are
   authorized here. The four-FWHM union historically used M0/T1/T3/T5/T10 fits;
   those future fits are not available prospectively in this experiment.
   Apply the same union operation to the one available unperturbed M0 fit;
   freeze this singleton union before feedback and record the adaptation.
   Never transplant old geometry/noise or add future endpoints to C0.

3. One guarded M0 fit attempt must bridge geometry initialization and C0.
   Current measure_expected validates C0 before optimizer, whereas source
   exclusion needs fitted widths/centroid. A split of the unchanged numerical
   attempt from completion of its covariance/SNR acceptance resolves this:
   one optimizer call using the exact expected-source seed, bounds, W=I and
   limits; provisional finite geometry for noise initialization; then final
   acceptance of that same attempt using new C0. Do not run historical
   fit_diagnostic/measure (sigma floor0.5) then a guarded refit (floor1.5).
   `make_context` is unsuitable as a convenience initializer because it calls
   historical measure again on a Gaussian baseline. Build fixed linear weights
   directly from the accepted M0 params. No candidate/accepted POINT products
   exist before final guard passes; rejection preserves map and clears outputs.

4. Authoritative guarded thresholds are present and full precision:
   CORR3.689982959208037, DIAG4.686076304689278. Artifact:
   /Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-expected-source-guard-20260925-r2/calibration/thresholds.json
   SHA256e068bda489308aa726540d4e1f353fd134a753fddc17d85d1a112385f882c143.
   The checked-in expected-source COMPLETION names this -r2 root; threshold
   digest equals retained root-manifest entry. Do not accidentally use the
   earlier failed-gate unguarded correlation calibration's values.

5. Geometry: physical tangent east is positive first axis/decreasing FITS
   column (CDELT1<0), north is positive second/increasing row; grid2arcsec;
   guard/feedback40arcsec AST-centered disk. M0 accepted centroid/major defines
   fixed template (unit-peak Gaussian inside3*major, q=t/sum(t²)), apertures
   (1,1.5,2 times M0 major), background/profile geometry. K0 is original
   unperturbed M0 BASE admitted connected mask, not the noise source mask.
   Guard is one seven-free-parameter fit, strict amplitude S/N>10 and both
   sigmas>1.5, centroid radius<=20, complete finite support, existing solver
   limits and rejection semantics. Full C0 is shared for every guard; DIAG
   changes only supplement evidence. BASE ratio>=3; supplemental score>tau;
   full footprint checked before deliberate center/K0/currentBASE exclusions;
   signed neighbor evidence, neff>=3, and one noniterative connectivity step.

6. The retained point_response_cd.processing_guard includes the 152389
   network3/scan5 failure exception. The new obs must not call it unchanged
   unless a narrow explicit argument disables inherited exceptions. Require
   every new required segment to converge and preserve other native checks.

7. Fixed template/aperture values subtract the current fitted constant. An
   uncertainty from bare fixed weights alone omits that nuisance response.
   Existing importance closeout explicitly marks matching endpoint scales
   unresolved, rather than using historical A+B ensemble. New endpoint
   conditional linear propagation can use the existing seven-parameter
   sensitivity/Jacobian semantics, including the constant response, or label
   unsupported scales unresolved. Neither prior152389 errors nor independent
   pixel RMS are matching errors. No new ensemble/refit is authorized.

## Execution/cleanup review criteria for forthcoming candidate

Exactly15 logical trajectories / at most150 feedback calls. BASE REF first;
profile memory/runtime, freeze its accepted T10-derived shapes before other
policy endpoints or branches. Four physical-input M0s; policy reuse and REF
reuse for seed. Only first seed candidate receives shoulder. T10 consumes m9;
m10 retained unapplied. At most9 ideal additions and3 endpoint covariances.
Fresh residual-specific PTC each cycle, no frozen bases; immutable shared
postRTC/CAL parent; native injection includes eligible non-MAP entries, keeps
invalidity and paired-r unchanged. Old failed-segment exemptions disallowed.

Review prospective budget against actual allocations and sizes; per-worker
transient products bounded and promptly compacted. A sampled disk counter alone
is not a peak ceiling if one unchecked allocation can exceed it. Enumerate
inactive run-owned temporary products and remove only after compact-output
verification; retain raw/tune/APT and reproduction records in permanent repo.
Threshold portability must not be called false-admission calibration. Effects
need own-policy REF and matched physical/source/noise scales, CORR versus DIAG,
and signed paired-noise qualification.

Current three-category disposition: scientific-method recovery complete with
above implementation requirements; architecture read-only/pass; evidence
hygiene read-only/pass. Await exact candidate SHA for implementation preflight.

## Dirty implementation preflight review

Reviewed the bounded new point_independent.py, run_independent_point.py,
prepare_independent_point.py; diffs to point_additive/point_expected/
point_response_cd; private native RTC/CAL/MAP/FRUIT152430 adapters; and four
new focused tests. Producer reports42 focused Python tests passed and CLI build
passed. No scientific execution performed by this reviewer.

Findings sent promptly to writer (repairs underway; exact-SHA verification pending):

- Major/scientific: endpoint learner used each endpoint fit's widths/centroid
  for exclusion, rather than frozen M0 source mask/support. Repair to use saved
  M0 mask and support intersect actual support, with full WCS identity check.
- Major/evidence: prospective lacked generated input/config hashes and run
  checks authenticated originals but not the reviewed generated requests.
  Realized C0/K0/geometry context also reloaded without a sealed identity check.
  Bind input manifest/environment/binary and realized operand manifest, verify
  at each dependent stage. Binary may bind through exact review JSON to avoid
  self-reference in source commit.
- Major/ordering: references stage did not require frozen shapes, allowing
  CORR/DIAG endpoints before mandated BASE-derived test-shape freeze.
- Runtime: size() raced parallel temporary deletion between is_file/stat;
  tolerate vanished files or use consistent synchronization.

No further numerical-owner changes or obvious signature/runtime blockers found.
Single M0 attempt initializer uses exact unchanged guard solver; amplitude-only
branch retains zero-lag normalization; injected parents authenticate against
fresh native projection; residual PTC is relearned each application; own-policy
controls preserve native membership; endpoint linear operators account for the
refitted constant. Unsupported source transfer must have null resolved sigma.
Final closeout must still handle complete15-cell accounting, M0 rejection empty
products, failed-stage dependent-cell disposition, and enumerated temporary
cleanup; these are execution/closure requirements, not approvals to expand scope.

Disposition remains repair-required pending exact candidate and prospective
identity; preliminary findings are not approval of a moving worktree.
