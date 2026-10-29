# Optical response, inverse noise weighting and configured FRUIT

Owner: Grant Wilson. On 2026-10-29 the owner requested implementation of the
full discussed suite and tests on the two POINT observations. The targets are
152389 and152418, all three arrays. Grant alone launches observational reductions
from the shell; this work prepares them and performs non-observational engineering
checks. Existing inputs, APTs, frozen runtimes and reductions are preserved.

## Preflight and authority

Engineering Governance, Timestream Successor Governance and Review and
Conformance were read. Current sequencing is the configurable POINT increment
and the accepted optical policy in `doc/OPTICAL_RESPONSE_POLICY_2026-10-29.md`.
The accepted governance identities are recorded in the integration ledger;
the standing2026-09-30 owner amendment permits in-scope development and local
commits. This extends the existing single spine increment, not a second lane.

Exact clean local base: `c6ea76247213cdb8ce51c9ed935a9744b15beb9a`.
Branch: `codex/configurable-point-20261028`.
Worktree: `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
Canonical application base and earlier numerical recovery remain as recorded
in `validation/configured_point_2026-10-28/WORK_ORDER.md`; the local candidate is
not canonical integration or a production release. Initially staged, unstaged
and untracked inventories were empty.

This is Tier2 scientific, interface and application work. The source-independent
owner decisions are the accepted optical-response/default policy and the combined
noise/FRUIT proposal. On2026-10-29 Grant additionally chose: “Use the resampled
interval and report its limitations.” The uncertainty method resamples existing
science time blocks and separate peer-detector groups for a conditional99%
interval, with unresolved detectors withheld. It does not claim a validated99%
coverage rate or observation-wide false-rejection guarantee.

## Scope and scientific ownership

Implement a new explicitly selected configured POINT profile. Preserve the
existing uniform profile and its scientific identities. YAML stays at the
application boundary; requested, effective and realized state remain separate.
No scientific calculations or mutable cross-stage state enter `Engine`.

- RTC and CAL remain their existing owners. Stop initial network processing at
  completed CAL before learning an observation-wide optical decision. Saved
  calibrated arrays avoid retaining all RTC/raw parent chains in memory.
- AST supplies corrected delta-source AltAz coordinates and source-protection
  membership. The initial POINT exclusion radii are35/42.5/50arcsec from the
  approved20arcsec pointing allowance plus three nominal FWHM. These masks
  protect optical/noise learning; they do not themselves remove source samples
  from maps.
- A new typed optical-response owner learns independent-network references and
  response evidence, considers the configurable default0.10 threshold with the
  approved conditional interval/stability checks, and freezes observation-local
  membership. Prior unavailable, weak, reversed and unresolved states stay
  distinct. Source CAL/APT facts are not rewritten.
- A typed inverse-noise PTC owner learns full residual variance after a
  leave-detector-out common-signal fit, whitens for existing observed-entry ALS,
  restores calibrated units and computes map coefficients from
  `1/max(v_ref,v_remaining)`. Source masks, validity and noise availability are
  distinct. No arbitrary duration threshold, weight cap or unity fallback is
  imported. A finite zero post-clean variance is permitted with positive
  supported reference variance; an absent variance estimate is unavailable.
- A separate empirical coefficient/admission family binds exact CAL, health,
  residual, segment and detector identities. It does not relabel the accepted
  uniform family or assert complete covariance/optimal precision.
- Native FRUIT owns subtraction and exact source-operand restoration; each pass
  relearns noise and cleaning from the immutable CAL parent. Corrected-frame
  model projection and existing POINT/BASE model selection retain their owners.
  The40arcsec feedback disk is separate from the noise/source masks. The
  configurable default ten feedback applications are a budget, not convergence.
- The Python application layer sequences CAL, health, weighted M0, source-model
  selection and fresh cycles, then publishes all-array results. Quiet mode emits
  compact required diagnostics; detailed transports are optional.

All quantities preserve calibrated mJy/nominal-beam units, exact network-local
scheduled-slot/native-row relations, selected APT occurrence identities,
non-finite/missing causes and corrected frame. Noise variance has squared signal
units; empirical coefficient has reciprocal variance units. Normalization is
not silently labeled full map precision.

## Prior work recovery

Adopt the current immutable-parent FRUIT lifecycle and POINT/BASE policies in
`doc/BRIGHT_POINT_PIPELINE.md`, current AST corrected-frame implementation, and
existing masked ALS/forward-map kernels. Cite the saved152418 response assessment
as motivation and validation targets, not mathematical authority.

The later `codex/fruit-adaptive-cleaning-20261007` history at `3a74a5c53` supplies
reusable typed coefficient-binding and explicit offset-frame projector patterns.
Recover those patterns with current ownership/tests. Exclude its OOF-specific
MAD estimator,5s/1s cuts, relative-flat0.01, tail thresholds, cap4, absolute source
thresholds, adaptive-rank and quadratic-feedback policies. They are not this
owner-approved POINT design. Historical weight/gain failures motivate explicit
eligibility and uncertainty, not a new arbitrary cap.

## Gates and exclusions

Focused synthetic tests cover optical response/uncertainty/source contamination,
parameter-sensitive rejection, high-noise versus weak response, sign/missingness,
weighted ALS/SVD and equal-weight behavior, calibrated units, coefficient support
and identity, source-mask separation, exact FRUIT restoration and fresh learning,
AltAz model geometry, real merged-config propagation and required diagnostics.
Run applicable CTest, baseline-tool tests and the full configuration preflight.
Review an exact coherent candidate independently, repair supported findings, and
verify source/executable identity before freezing trial runtimes.

Build evidence is local Mac supplemental evidence, not a reproduction of the
accepted Unity/Spack V2 campaign. No build-environment migration is included.
The owner-run observations assess maps, fits, exclusions, effective coverage,
weight trajectories, timing and comparison with preserved references. No new
scientific acceptance threshold is invented or relaxed to obtain completion.

No agent reductions, Unity access, pushes, canonical integration, OOF/beammap
activation, SCIENCE/coadd qualification, permanent APT changes or reference
cleanup. Missing scientific prerequisites or unsupported requested products
remain explicit. A newly substantive policy choice returns to Grant while
independent authorized implementation continues.
