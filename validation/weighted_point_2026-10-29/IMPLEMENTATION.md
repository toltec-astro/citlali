# Configured weighted POINT candidate

This implements the owner-approved optical-response, inverse-noise and FRUIT
suite as the explicit `citlali-point-v2` profile. The old v1 profile and frozen
reductions remain available. This is a local candidate, not canonical integration,
production qualification or the completed collaborator release.

## Requested and effective behavior

`data/configured_point_v2.yaml` selects weighted observed-entry ALS, ranks 3/4/6,
the existing disturbance exclusion, the existing RTC filters, a default optical
threshold of 0.10, and ten feedback applications. YAML controls ranks, pixel
size, coverage cut, worker count, threshold, feedback enable/budget, verbosity
and intermediate retention. Threshold domain is 0 < f <= 1; omitted f is 0.10.
The ordinary TolTECA merge remains the boundary. Existing legacy algorithm
controls are explicitly retired by the selected v2 migration, with dispositions
in the effective configuration. Legacy feedback stays disabled: the new typed
native/Python composition owns these passes.

The prepared two-observation trial retains the preceding comparison's 2 arcsec
pixels, coverage cut 0.15, ranks 3/4/6 and three upstream workers. None of these
values is retuned from new map appearance. The threshold remains an owner tuning
parameter, not an empirically validated universal limit.

## Scientific owners and lifecycle

1. Run the existing RAW/KIDs/RTC/CAL path once, ending at completed CAL for every
   supplied network. Required producer receipts bind the exact processing
   intervals. This barrier permits references across networks without retaining
   every original raw/RTC object graph.
2. AST evaluates corrected delta-source AltAz on the exact scheduled RTC slots.
   Masks at 35/42.5/50 arcsec protect health and noise learning, not map samples.
   CAL values are joined to their original slot, cause, selected APT occurrence,
   assigned time, geometry and calibration identity. APTs are never edited.
3. `PointOpticalEvidence` constructs half-second learning bins, robust reference
   populations with the candidate's entire network held out and equal network
   influence, and an array-normalized signed response. Two disjoint peer panels
   and early/late science-block portions test stability. A sole other network
   additionally requires shared coupling with available other-array references.
   Missing reference excitation, source support or interval information is an
   unresolved state, not a dark-detector claim.
4. Grant authorized a conditional resampled interval. There are 2,000 deterministic
   science-block bootstrap draws per panel/window, frozen robust-fit weights and
   frozen normal-response scale. The conservative envelope of the six intervals
   must establish positive response above f to retain, response wholly within
   (-f,+f) to call weak, or a wholly negative response below -f to flag reversed.
   Boundary-overlapping and unavailable decisions are withheld as unresolved.
   At least two informative blocks in each tested portion are necessary to
   estimate variation; this is not a minimum uninterrupted-duration cut.
5. The statistical limitations are required output: the intervals condition on
   the constructed references, robust weights, normal-response scale, and assumed
   exchangeability of processing blocks. Peer panels provide reference sensitivity,
   not a demonstrated sampling law for all possible reference populations. True
   99% repeated-sampling coverage and observation-wide false-rejection rates are
   unvalidated. The slope is not applied as a new flux calibration.
6. `PointOpticalPlan` freezes observation-local membership before PTC. Every later
   cycle authenticates the same complete health publication and CAL/mask/order
   bindings; it cannot silently relearn health from a shrinking residual.
7. `inverse_noise_ptc` estimates full untrimmed residual variance after an offset
   and slope fit against a leave-detector-out median network signal, on valid
   source-free samples. It uses SSE/(N-2), requiring supported positive finite
   reference variance. This median is a noise-estimation reference and is separate
   from the cross-network optical-health reference. It uses shared order statistics
   rather than rebuilding a full donor list for every detector.
8. The existing ALS owner learns/applies to detector-whitened data. Results return
   to physical calibrated units. Post-clean source-free variance uses N-1 before
   source restoration. The empirical map coefficient is
   `1 / max(reference_variance, remaining_variance)`. Finite zero remaining
   variance is allowed with positive reference variance. Unsupported measurements
   remain unavailable; no clipping, cap, fixed duration or unity fallback is added.
9. `inverse_noise_fruit` owns exact subtraction and restoration. M0 uses a zero
   operand. Each later pass starts from original CAL minus the current projected
   model, learns fresh noise/basis, applies it, restores that exact operand and
   maps the total with the same coefficients. There is no cascade of already
   cleaned samples and no RTC rerun.
10. Existing POINT-EXPECTED/BASE selects a replacement model in the corrected
    AltAz frame and 40 arcsec feedback disk. Initial C0/BASE selection context is
    frozen; source fits update. This selection context is not detector noise or
    endpoint uncertainty. Outside the disk means no feedback; missing model
    support remains unavailable. A failed source guard stops feedback explicitly
    and preserves the last complete all-array maps. Ten passes are a budget,
    never a convergence claim.

## Products and honest failure states

The empirical coefficient family is separate from the unchanged uniform family:
`OWNER-PTC:empirical-inverse-noise@2026-10-29-v1`. Signal stays mJy/nominal beam.
FITS NUMERATOR is beam/mJy, NORMALIZATION is beam2/mJy2, and normalization is an
empirical analysis coefficient sum, not measured map precision or significance.
Weights and their causes are bound to exact segment/health/residual identities.
The fixed map grid encloses original science-supported CAL coordinates before
health/noise exclusions, so changing weights cannot silently change feedback WCS.

Required outputs include all three maps and qualified fit diagnostics, the existing
disturbance-drop summary, optical-health counts and exception identities, interval
limitations, actual final contributors, per-segment unavailable-weight reasons and
weight ranges, and an explicit feedback trajectory/stop cause. Quiet mode omits
per-detector normal-weight tables and per-sample cycle transports; verbose enables
the extra detector and selection diagnostics. Immutable health evidence is saved
once. Generated CAL transports can be released after successful publication;
inputs, APTs, source models, receipts and maps remain preserved. Prior reductions
are outside all cleanup paths.

Missing data can remove a detector or segment while supported observations continue.
All three arrays must produce valid maps; required I/O or association errors fail
the process. A stopped feedback selector is reported separately from completed
map publication. Source/health/prior-cycle associations are verified natively and
in Python before use. The current diagnostic fits still do not claim the existing
PPT calibration-consumer contract; that collaborator-release integration remains
an explicit later gate.

## Engineering and observational validation

Local synthetic tests cover signed/weak/noisy/source-contaminated response,
conditional intervals and parameter changes; exact restored health; weighted SVD,
uniform-equivalence and tail-power cases; omitted/unavailable variance; frozen
support; exact source add-back; AltAz projection; observation/cycle/health/processing
tampering; empirical FITS units and all-array publication; fresh model replacement,
source guard stops and normal v1 behavior. Full CTest, baseline-tool tests and
configuration preflight are recorded in the final review/handoff.

These are supplemental Mac AppleClang/Homebrew checks. They do not reproduce the
accepted Unity Spack V2 campaign. Exact candidate review and a matching committed
source/executable precede the frozen owner trials. No observational processing,
Unity access, push, integration or production activation is performed by Codex.

After Grant runs 152389 and 152418, inspect actual health exclusions and unresolved
rates on all arrays, noise-weight ranges and contributor loss, structured background,
source amplitude/shape/position and qualified errors through the feedback trajectory,
source-model stop causes, and elapsed time. Compare preserved maps to understand
changes and improvement, not to force old values. Existing scientific criteria
remain unchanged; newly needed decisions return to Grant with observed evidence.
