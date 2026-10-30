# Bounded source-recovery test for weighted POINT

2026-10-30 — proposal only. No implementation, new source fit, trial preparation
or reduction is authorized or performed by this document. Noise weighting and the
2″ grid remain fixed. Grant continues to launch any later reductions himself.

## Decision this test serves

Does the current weighted POINT procedure recover known compact and broader
source perturbations, and does an unsupported feature placed only in its starting
feedback model disappear? Answer this before choosing a new wing policy.

The first test uses **one observation (152418), the current policy only, and five
all-array cases**. It does not implement Gaussian-tail completion. This
baseline-first design is smaller and more informative than immediately comparing
several policies with unknown true source brightness. The a1100/a1400 question
motivates the test; a2000 remains included because the current interface and first
integration milestone require all three arrays.

This is finite response at the **post-RTC/CAL, pre-PTC boundary**, conditional on
shared frozen health. It does not validate raw-data processing, detector rejection,
absolute flux calibration, continuous-beam sampling or performance on faint SCIENCE.
A known perturbation on a real source measures local full-procedure response; it
does not reveal the unknown true brightness of that original source.

## Program adherence and prior-work recovery

See [PRIOR_WORK.md](PRIOR_WORK.md) for the charter link and explicit adopt/cite/
abstract/defer/exclude dispositions. September25 establishes a useful diagnostic
design, not current weighted-route validity or new execution authority. Its wing
recovery/model-memory tradeoff is the reason for retaining both data and seed
controls. The current proposal defines an adaptation, not a new scientific-contract
package. No unresolved model choice is silently made production authority.

## The five cases

All rows run all three arrays with the same fixed numerical recipe. Percentages
are **test perturbation strengths**, not performance tolerances.

| Case | Original calibrated data used by PTC | First feedback model | Purpose |
|---|---|---|---|
| REF | Unchanged | Ordinary model from its own M0 | Matched reference and no-intervention replay |
| CORE+ | Add +10% compact perturbation | Ordinary model from its own perturbed M0 | Positive compact response |
| CORE− | Add −10% compact perturbation | Ordinary model from its own perturbed M0 | Signed response/asymmetry; not negative-source recognition |
| SHOULDER | Add +5% broader, offset perturbation | Ordinary model from its own perturbed M0 | Recovery of a shape different from one centered narrow Gaussian |
| SEED | Exactly REF data and M0 | REF's accepted m0 plus the same shoulder, **once** | Dependence on unsupported initial feedback |

Each case has M0 plus at most ten FRUIT applications. M10 consumes m9; no m10
candidate is applied. Compare replacement candidates m0–m9 and total maps M0–M10.
No added policy, target, signal amplitude, source shape, random realization,
iteration, smoothing, pixel size or automatic numerical retry is in this matrix.

**Hard ceiling:** five all-array trajectories = fifteen array trajectories;
fifty all-array feedback applications =150 array feedback applications. There
are four distinct data parents/M0 calculations because SEED shares REF M0 and its
selector context. Thus at most54 all-array native cycle invocations (4 M0 +50
feedback), excluding focused tiny fixtures in a later implementation phase.
Logical map endpoints include the shared M0. Existing saved REF is evidence for
replay, not a reason to omit the matched REF under new diagnostic instrumentation.

## Frozen signals and frame

Use the saved current noise-weighted **M10 fit** of each152418 array to define
its new compact/shoulder truth rasters, and that array's saved **M0 major FWHM**
for the shoulder displacement. Existing anchors are verified in [anchors.json](anchors.json).
Do not refit to choose convenient injection shapes or tune after seeing results.
These are new152418 operands using the historical recipe, not the old152389 rasters.
At implementation/preparation, publish the exact parameter vectors, complete
signal rasters, signed peak/brightness-area, taper loss and hashes before any run.

Let G be the saved elliptical Gaussian with its fitted constant set to zero.
CORE+ is0.10G; CORE− is its exact negative. SHOULDER uses0.05 of the saved peak,
both Gaussian sigmas doubled, the same angle and a center displaced by+0.5 M0
major FWHM along **positive azimuth offset**, with unchanged elevation offset.
Pixel columns decrease along positive azimuth (CDELT1<0); do not copy the old
FK5 east/north labels or degree-valued header arithmetic into this AltAz route.
All brightness is mJy/nominal beam; positions and widths are arcseconds.

For all three data signals and the seed, multiply by the historical origin-centered
radial window: one for r≤30″; (1+cos(pi(r−30)/6))/2 for30″<r<36″; zero thereafter.
Do not renormalize after tapering. Retain the current40″ computational disk and
2″ grid. Every nonzero injection pixel and its native containing-pixel projection
must be valid. Missing support blocks realization; it is not filled or silently
removed from the truth. Original invalid detector/time entries remain unchanged.

This preserves the retained discrete test and deliberately does **not** test
whether2″ cells adequately represent a continuous optical beam. That separate
sampling question stays parked.

## Intervention placement and learning

Bind shared original CAL, exact detector/slot/scan identities and coordinates,
frozen optical health (including10% response cut and current unresolved decisions),
original eligibility and source-free masks. Apply the known raster through the
same native coordinate association to **all eligible original samples**, including
samples later excluded from MAP. Form y_I=y+P(I) once; preserve immutable y_I
and its binding. Every pass starts again from that same original or injected
parent, subtracts only the latest source model, freshly learns/applies PTC, restores
exactly the subtracted model and maps the result. No accumulated previous output,
additional injection restoration term or reuse of a cleaner learned on a different
residual is permitted.

Keep ranks3/4/6, coverage cut0.15, current source guard, connected threshold3,
unity feedback gain, complete replacement and ten-pass budget. Keep the existing
**noise-weighting rule**, not frozen numerical coefficients: residual noise,
cleaning coefficients and map coefficients may change as consequences of the
perturbation. That adaptation is part of the response being measured.

Preserve current selector initialization: REF, CORE+, CORE− and SHOULDER each
learn C0/BASE from their own M0, then freeze those scales within that trajectory.
SEED shares REF's exact M0 context. September25's cross-parent frozen selector
is not silently imported. A later candidate comparison would share each physical
parent's M0/context across policies, but no such candidate is included here.

**Known overlap:** the original source-free radii are35″/42.5″/50″. The retained
36″ raster taper, including containing-pixel footprints, can place nonzero input
on a1100 samples labeled source-free. We explicitly retain this as part of the
current full-procedure response; source-free masks are unchanged but their values
need not be. Record per array the projected nonzero count and squared-signal sum
on those samples. Do not claim isolated feedback-only attenuation or unchanged
noise-learning inputs for the data-injected cases. No silent tighter taper or
larger exclusion radius is introduced.

## Feedback-only control: exact meaning

First require REF's ordinary source guard and model to pass. Preserve that
unmodified m0. Make a separately identified diagnostic seeded model by adding
SHOULDER on its valid domain and updating admitted support accordingly. The
injected component has a one-use admission exception; all finite-value,
coordinate, validity and boundary checks still apply. It must not masquerade as
an ordinary threshold-selected product, and no source guard is bypassed.

SEED's data and M0 are byte-identical to REF. Apply the seeded model only on M1.
At M1 and every later map, generate the next source model through the ordinary
policy with no additional seed, forced connection or permanent mask. Compare
SEED−REF alongside SHOULDER−REF using the same shoulder reference. A seed imprint
that survives is initialization sensitivity; it does not prove that the real
observation lacks similar sky structure or establish an infinite-iteration result.

Conversely, seed decay does not prove a permanently imposed Gaussian tail harmless:
such a forced component would be present in both seeded and unseeded candidate
runs. Any future persistent-tail proposal still needs its own known-response
assessment, and should not be accepted solely on this one-time control.

## Weighted reference and fixed measurement contract

Each case is differenced against this policy's own matched REF at the same cycle.
Construct the injection-only **ordinary weighted MAP** reference on that cycle's
REF admitted inputs, coefficients, normalization and support, bypassing PTC.
The old routine's unit-coefficient and HITS==NORMALIZATION assumptions are invalid.

For the retained grid-aligned raster, every admitted sample in a pixel receives
the same I[p]. Positive-weight normalization therefore gives I[p] directly on
supported pixels. A compact native verification of that identity, checked against
an independent coordinate evaluation, may avoid writing detector-time reference
arrays. It must still verify actual projection/support and normalization; do not
assert this equivalence for a future continuous/interpolated injection.

Freeze the prospective **40″ origin-centered benchmark disk** on the2″ grid,
including pixels where the truth injection is zero. Require it fully supported
in REF M0 before launch and in each compared stage. Record all native output-mask
and detector/sample-support changes; do not demand identical accumulated weights.
A lost benchmark pixel makes that fixed-domain measurement unavailable. Never
shrink to a convenient intersection or source mask to improve a metric. Other
native-domain diagnostics are separately labeled, not replacements for the fixed
benchmark. Preserve original input-support hashes; downstream support changes
are reported outcomes unless they violate existing processing/validity contracts.

Measurements:

- At M0–M10, gain g=<Delta,I>/<I,I>, distortion eta=||Delta−I||/||I||, and
  orthogonal distortion rho=||Delta−gI||/||I|| on the fixed benchmark. Preserve
  eta²=(g−1)²+rho² as a numerical check. Use signed negative I for CORE−.
- Compact sign asymmetry and the even map (CORE+ +CORE− −2REF)/2; signed
  response/residual maps and profiles, including bowls and pedestals.
- Existing full-map source fits: peak, centroid, major/minor widths and
  gauge-invariant ellipse shape, fitted constant and guard status. Never fit a
  signed difference with the positive-source recognizer.
- Freeze template, centroid, profile bins and apertures from REF M0. Aperture
  radii are1,1.5,2 times that original major FWHM. Report raw brightness-area,
  fitted-constant contribution and background-subtracted result separately.
  Use the current/retained measurement owner's exact definitions; bind their
  raster footprints before launch rather than mixing quadrature conventions.
- At M10, at most **nine ideal map-addition measurement controls**: three data
  perturbations ×three arrays, measured on REF+known weighted reference. Compare
  actual full-map measurement increments with ideal increments; absolute errors
  precede ratios. Undefined/negligible denominators stay unavailable under the
  retained numerical convention, never converted to large invented gains.
- For SEED, total-map and candidate shoulder projections, residual norm,
  extra support, aperture/constant changes and final drift. Compare data-supported
  shoulder and feedback-only response side by side. No scientific seed threshold
  is inferred from tiny numerical drift.
- Freeze a descriptive background region45–60″ from the analysis origin on
  REF M0, require its support, and report STD/MAD and signed residual structure.
  This is an explicitly declared current-map region, not September's covariance
  benchmark or a source-error estimate. Lost support makes its statistic unavailable.

Existing conditional C0 sigma_A remains a source-guard scale, not a calibrated
endpoint flux uncertainty. No new covariance family fit, Monte Carlo ensemble,
admission threshold calibration, uncertainty rescaling or per-cycle false-feedback
claim is part of this test.

## Stop rules and decision table

1. **Prelaunch:** no observation processing until exact inputs/operands, source,
   executable, environment, diagnostic interfaces, numerical checks, resource
   ceiling, output preservation and owner shell procedure are prepared/reviewed.
   This document does not supply that implementation or authorize a launch.
2. **REF first:** its no-intervention scientific planes/support must reproduce the
   saved current reference under frozen execution settings. Expect exact scientific
   array equality; provenance fields may differ. A discrepancy stops the matrix for
   diagnosis—no post-hoc tolerance relaxation. REF also establishes resource use.
   If REF cannot complete, do not launch dependent cases.
3. **Branch failure:** source rejection, unsupported projection, missing/corrupt
   binding, nonfinite required values, new failed cleaning segment, failed
   subtraction/restoration accounting or existing boundary violation stops that
   all-array case before another feedback pass. Preserve its state. Do not use a
   stale model, relax the guard or retry automatically. Unaffected predeclared
   cases may complete under the owner-run process.
4. **Budget:** always stop at M10 or earlier failure. No cycle11, automatic new
   amplitude/policy/observation, or threshold/rank/pixel retuning. Poor gain,
   distortion, seed memory and unresolved terminal drift are completed findings.
5. **Assessment once:** produce the full five-case status table and one recommendation.
   Numerical settling and higher brightness cannot certify success.

| Finding at the cap | Decision; no automatic further run |
|---|---|
| Baseline meets predeclared POINT response requirements and seed-memory limits | Retain it; stop feedback tuning and return to POINT product/calibration integration. This is not absolute calibration certification. |
| Clear compact/shoulder distortion or model memory | Identify one specific mechanism and propose one bounded remedy/comparison. No default Gaussian completion. |
| Tradeoff, failed comparison or insufficient scientific acceptance criteria | Declare inconclusive; retain unqualified status and stop. Do not tune until a preferred peak appears. |

**Scientific acceptance is separate from execution completion.** No accepted
current limits for gain error, shape/aperture/centroid changes or seed memory are
established by the recovered record. Grant must set those release-use criteria
before results can be called adequate; this proposal invents none. The historical
roughly5% calibration/PSF goals are context, not automatic thresholds for this
partial-boundary experiment. An exploratory execution can end with descriptive
findings, but cannot promote the pipeline without the criteria. Criteria must not
be chosen retrospectively to make a result pass.

There is **no automatic second-observation validation** in this allowance. If a
candidate is later justified,152389 is the proposed independent POINT check under
a separately reviewed bounded matrix. SCIENCE preparation and POINT-to-calibration
interface work need not wait for an open-ended feedback research program.

## Implementation gap, resources and next deliverable

Current strict weighted requests have no injection/seed diagnostic interface.
A future bounded change must add explicit native intervention bindings, preserve
ordinary provenance and default behavior, and provide weighted reference and
seed-expiry verification. Coordinator code must remain sequencing-only: native
owners handle projections, immutable parents and subtract/restore; the POINT
owner handles source candidate formation. Do not invoke the old runner against
this route or put scientific matrix arithmetic into Engine/orchestration.

Focused future gates: no-op parity, exact injected-parent identity on valid entries,
unchanged invalid entries, signed/no-source fixtures, single-use seed expiry,
replacement removing old support, weighted normalization/projection, frame/handedness,
source-rejection behavior and required diagnostic failures. Select applicable
native/config/baseline gates at implementation, then freeze the actual executable.
No build environment is changed by this proposal; local evidence is not Unity/Spack
qualification.

Quiet mode remains the default. Retain compact signals, per-pass maps/models,
measurements, support hashes/counts, intervention/accounting summaries and exact
identities. Do not create full detector-time tracing archives. Share original
CAL/health inputs; preserve old reductions. Resource bytes/wall ceiling must be
measured and declared during preparation; five-case arithmetic is not a storage
or time estimate. Plan sequential all-array cases initially, with existing within-
run thread controls. No unapproved parallel workload increase or scratch cleanup.

Next deliverable, only after owner approval of this definition: bounded diagnostic
implementation and a concrete owner-run trial with exact commands and readiness.
No such commands, new runtime, rasters or trial have been prepared in this turn.
