# POINT FRUIT gates by operational use

2026-09-12 · r0.1 · **Prospective design for owner review; no new run authorized.**
Purposes U1–U5 and peak response as the primary OOF gain measure are established
by the [owner record](POINT_OPERATIONAL_USE_CASES_2026-09-12.md).
The gate applicability changes and new numerical targets below are proposals.

## Program adherence and prior-work recovery

Use the [charter](r0.4/inputs/program/README.md),
[roadmap](r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md)
and [reviewed recovery](r0.4/PRIOR_WORK.md). Adopt the frozen scientific roles,
current-residual PTC relearning and total-map replacement recurrence.
Preserve the [accepted brief](POINT_REFERENCE_BRIEF_ACCEPTANCE_2026-09-10.md),
[ordinary-MAP definition](r0.4/METHOD_DEFINITION.md), and all completed results.
This is an implementation-informed development proposal, not a successor
normative core or an independent-author packet.

## 1. Decision to make

**Target a useful improvement in routine pointing and relative peak gain
(U3 and U2), while requiring honest startup and degradation/health information
(U1, U4 and U5).** A method may be useful for a declared subset of POINT uses.
Do not require one first candidate to become a qualified solution for all five,
or generalize it to OOF, BEAM or SCIENCE.

Declare uses, source/condition envelope, required arrays, evaluators and limits
before the comparison. For each use report **adequate**, **degraded/OOF indicated**
where applicable, **no source established**, or **unassessable**, with reasons.
Separately score whether that response was correct on the controlled test.
“Unassessable” prevents false confidence but fails an availability requirement
when the case was predeclared measurable. No method may pass by rejecting all
difficult cases or selecting only successful arrays after the run.

## 2. Gates that protect every claimed use

| Gate | Required behavior in a later authorized test |
| --- | --- |
| G0 — valid method and output | Verify original-parent subtraction, relearning on each current residual, inference from reconstructed total, replacement/revocation, and distinct total/model/background products. Retain signed maps, input quality and terminal/failure causes. Required failed solves and missing products cannot pass. |
| G1 — genuine source evidence | On the small predeclared null/background challenge set, propose zero accepted astronomical feedback models and zero positive POINT source reports. Score these separately. Noise-scale and eligibility rules must be available before action. Zero events in a small set is a screen result, not a false-alarm probability. |
| G2 — no false reassurance | Gross distortion, a declared response-loss case, and insufficient support must not become a confident healthy-source or valid-offset report through imposed position, width, amplitude, model shape or array agreement. Retain the relevant upstream quality facts even if nuisance structure is legitimately removed by PTC. |
| G3 — honest spatial domain | Declare separately source-evidence eligibility, model extent, diagnostic/source apertures, total-map support and background/computational domain. Report clipping and lost contributors; never shrink the evaluation domain to remove a failure. Source evidence inside a useful region does not imply zero astronomical brightness outside it. |
| G4 — stable, affordable behavior | Retain every development iteration, source/model/support changes, perturbation/failure behavior, cleaning passes, cumulative wall time and peak memory. Freeze failure/resource and terminal rules. A stable scalar or a visually good earlier map does not rescue a failed run. |

No false-health probability is inferred from a finite challenge set. All
probabilistic claims require a separately specified ensemble and calibrated
uncertainty. Comprehensive detector-noise qualification is not a prerequisite
to this bounded comparison; limited evidence must instead carry limited claims.

## 3. Primary measurements and use-specific acceptance

Measurements are evaluation-only and act on the reconstructed total. The
feedback policy may use its own declared estimator. Neither estimator receives
injected truth, expected flux, true centroid or true width. Truth is an external
ruler. A free elliptical Gaussian plus nuisance plane remains an option only
where its adequacy is established; distorted cases need a predeclared suitable
measurement, not a forced Gaussian or a post-result choice of evaluator.

For array a and observation t, let P(a,t) be measured source **peak response**
after separately handling the fitted background. Bind its quantity, units,
map response/normalization, estimator and uncertainty. It is not automatically
the brightest map pixel, integrated flux, or the feedback model's maximum.
For a compatible same-source pair define

    R_peak(a) = P(a,after) / P(a,before).

Carry the stated calibration and observing-condition corrections and their
uncertainties, including relevant correlation between the pair. An unstable or
unavailable denominator makes the ratio unavailable. Do not normalize each map
to an expected source flux or width. A measured ratio alone is an association;
attributing its change to OOF requires adequate condition comparability.
This observational gain is distinct from FRUIT's feedback application gain.

| Use | Primary gate proposed for the first bounded test | Retained diagnostic evidence |
| --- | --- | --- |
| U1 — startup | For predeclared detectable sources, including grossly distorted ones, establish source presence without a forced compact shape and expose gross degradation or explicitly inadequate shape information. Correctly withhold precise offsets when unreliable. Null/background controls must satisfy G1. | Per-array source evidence, broad location if measurable, apparent extent/asymmetry, peak, residual structure and support. Full-wing fidelity and 5% photometry are not universal U1 gates. An unusable map that cannot answer either startup question is not a successful acquisition result. |
| U2 — OOF gain/recentering | Recover the known same-source peak ratio with proposed relative error ≤5% in each finite controlled pair. Recover the pointing offset in cases designated measurable under U3. Include an unchanged pair and a mildly deformed pair whose peak genuinely changes. | Individual peaks and uncertainties, integrated brightness within a declared aperture, widths/asymmetry, calibration/condition differences and residuals. Do not demand unchanged peak or ideal shape across OOF. |
| U3 — routine pointing | Retain the accepted provisional centroid-bias target ≤0.05 of the injected minor FWHM in the adequate compact/mild-deformation envelope. Report individual offset and displacement errors, scatter and uncertainty. A small screen does not establish ensemble bias. Bind an operational per-observation error/precision allowance in arcsec before execution; do not silently turn a bias target into a stricter limit on every noisy realization. | Peak, widths, distortion, signed residuals, support and uncertainty; matched-reference errors and the identified OG offset benchmark. Declare the position reference and axis convention for a distorted source before testing; do not let a larger recovered width enlarge the error tolerance. |
| U4 — OOF need | Preserve the sign and useful size of a known peak-response loss and accompanying shape change; an unchanged comparison must not acquire a spurious degradation report. Use a proposed 10% peak-loss challenge and an unchanged pair as a first sensitivity probe, with U2's 5% ratio-error limit. | Width/shape change, residuals, comparison epoch and conditions. The 10% probe is not an adopted OOF scheduling threshold. Gross deformation must remain visible even where a scalar width is inadequate. A final action threshold and its missed-change/false-alert tolerance remain an operations decision. |
| U5 — tune/condition health | Preserve a predeclared array-specific response-loss or input-quality abnormality in the POINT/QA evidence. The affected array must show the loss or become explicitly unreliable, rather than being restored by a source/array prior. An unaffected comparison must not be falsely labeled faulty. | Per-array peak ratios, available detector/network quality, loading/condition facts, noise/residual changes and coverage. A synthetic response fault tests observability, not a physical model or diagnosis of clouds or retuning. No automatic all-arrays-healthy conclusion. |

Where a use requires a numerical measurement, “unavailable” blocks that claim.
Where its correct operational answer is a warning, a trustworthy warning may
be success. It must be derived by a declared POINT/QA rule from retained
evidence, not assigned from injected truth. A reviewer-only observation is
descriptive evidence, not an automated operational gate pass.

## 4. What changes relative to earlier tests

These changes apply only to a separately approved successor protocol. The
[central-domain starlet result](../../../../../../../../validation/fruit_point_starlet_central_domain_2026-09-12/SCIENTIFIC_REPORT.md)
and earlier rejections keep their original rules and dispositions.

| Earlier requirement or metric | Proposed future disposition |
| --- | --- |
| Accurate whole-image reconstruction, including extended coma wings | Keep image error, aperture loss and signed residual structure as diagnostics for every case. Make a numerical full-image limit blocking only where needed for the claimed centroid, gain, degradation or morphology result. A poor whole-image score does not alone defeat U1 or U3. Conversely, a good centroid cannot establish U2/U4/U5 adequacy. |
| ≤5% absolute peak and each FWHM bias | Retain as accepted provisional targets where absolute peak/width recovery is claimed and the evaluator is adequate. For U2, propose the explicitly new ≤5% relative peak-ratio requirement as primary. For U1 and gross-distortion warnings, propose no universal precise peak/FWHM requirement. |
| Amplitude RMSE and exterior error RMS ≤1.10 historical control | Retain their evidence and applicability to the earlier accepted photometric reference. For the successor, propose exterior contamination as a common guard, with fixed domains and matched-control comparison; amplitude RMSE is blocking where amplitude recovery is claimed. OG and matched-policy ratios must be labeled separately and never treated as equivalent controls. Exact finite-screen rules and uncertainty still need freezing. |
| 60-arcsec model boundary | Retain as the completed experiment's identity, not an operational POINT limit. A future candidate may distinguish evidence region from model extent, with independent off-center/support tests. This document selects neither a new radius nor a taper. |
| Solver success and valid feedback | Keep as hard requirements for the method actually declared. Failed last iterates remain unavailable. A different completion rule or a declared zero-feedback fallback would be a method change requiring its own review and freeze. A preserved valid bootstrap may inform startup QA without being relabeled a successful terminal FRUIT result. |
| 0.5 s median array inference screen | Keep the recorded timing failure. Propose judging the successor's benefit against total cumulative wall time and memory, with inference cost still reported. The old microbenchmark limit need not be a universal operational ceiling; the next protocol must declare an actual resource budget. |

All losses remain visible even when a metric is diagnostic. No source-support
change, causal claim or unqualified uncertainty is hidden by the use-specific
scope. The accepted brief's exact bytes are preserved; its revised applicability
above awaits a successor owner decision.

## 5. Small future comparison, not another broad search

Propose the following eight scene states, using two predeclared nuisance
realizations and all required arrays. Reuse states across questions: for
example, the mildly broadened/healthy pair serves both U2 and U4. This is a
maximum of 16 synthetic scenes per policy, not a rank/threshold/shape grid.
Exact strengths, coordinates, shapes, quality perturbation, seeds and pass/
resource budget must be recorded in one case file before execution.

| State | Main question tested |
| --- | --- |
| N — no source | G1: no invented source or astronomical feedback. |
| B — background/contamination only | G1/G2: nuisance structure is not astronomical feedback or a reassuring source. |
| H — healthy compact source at a free nonzero offset | U2/U3 anchor; compare its two nuisance realizations for an unchanged-response check. |
| H-shift — same source with a known small additional offset | U3: recover the displacement without pulling toward the nominal origin. |
| D — mildly broadened/deformed version with a declared 10% lower peak | U2/U4 paired with H: preserve relative response and measurable shape change. Hold the declared source normalization consistently through the known transformation; do not impose it on inference. |
| C — grossly distorted startup source | U1/G2: establish a detectable source and expose degradation without requiring full coma-wing recovery. The retained bright coma is an available development stress template, not the entire POINT population. |
| T — H with one declared array/detector-response abnormality | U5/G2: preserve its observable effect or the reason the affected measurement is unreliable; do not borrow an amplitude from the other arrays. |
| E — an off-center source encountering the declared support boundary | G3: expose lost support and unreliable measurements; no quiet recentering of the domain from the fitted source or truth. |

Use fixed rank 5 and otherwise common reduction choices for the first
matched pixelwise-reference/candidate comparison on 123424. Preserve and
identify the OG reduction as the operational benchmark for offsets and
latency; record its different processing/timing boundaries. Do not reconstruct
an unavailable historical environment or reopen weighting as prerequisites.

Map-level checks can screen an estimator. Any new claim about FRUIT recovery,
gain, false feedback or pass savings must rerun the relevant learning on each
arm's model-subtracted parent. Two nuisance realizations and reused states do
not establish ensemble bias, operational false-alarm rates, or independent
pointing replication. Freeze any uncertainty procedure and distinguish measured
errors from its claims. More samples are a separate decision, not an automatic
extension when a small screen is inconclusive.

Keep the complete development iteration sequence. Plot centroid, peak ratio,
width/shape, residual metrics and failures against cumulative wall time, and
report cleaning-pass counts and peak memory. Use a predeclared result/terminal
rule for the primary comparison, not the best-looking truth-scored iteration.
Retrospective time-to-accuracy curves do not authorize an early-stopping policy.

**Keep** a candidate for the demonstrated subset if it satisfies its required
gates and improves peak/pointing recovery or time to equivalent accuracy at
acceptable cost relative to the matched reference. **Revise** only for one
identified weakness under a separately bounded allowance. **Reject** if a
required safeguard fails or no useful benefit is demonstrated within the
budget; distinguish inconclusive evidence from demonstrated harm. OG is a
strong operational benchmark, not the ceiling. Replication must precede a
numerical policy recommendation. Freeze the candidate before any newly
authorized 129081 comparison; its historical exposure remains disclosed.

## 6. Approval boundary and next concrete step

| Item | Present state |
| --- | --- |
| Five operational purposes and peak response primary for OOF gain | Owner-established in the linked 2026-09-12 record. No further purpose/primary-observable vote is needed. |
| Use-specific gate applicability, finite-case checks, 5% peak-ratio target and 10% sensitivity probe | Proposed here. They do not replace the old screen or establish an OOF trigger. |
| One candidate, exact evaluators/domain/case file, operational offset precision in arcsec, valid-solve rules, and execution budget | Not selected by this note. Bind only the fields needed for a bounded comparison; do not commission comprehensive noise qualification or an OOF method. |
| New numerical execution or a change to the starlet solver/domain | Requires a corresponding separate decision and prospective freeze. This note grants no new PTC passes, replay, 129081 use, Unity, production or qualification. |

Recommended next step: review this applicability change, then prepare one
short executable comparison proposal around U2/U3 and the U1/U4/U5 safeguards.
The central-evidence/model-extent distinction is motivated by existing evidence;
it is not itself a selected estimator or a reason to reopen every rejected
candidate. Preserve the bounded keep/revise/reject objective.
