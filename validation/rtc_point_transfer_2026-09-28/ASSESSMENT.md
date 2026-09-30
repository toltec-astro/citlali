# POINT152420 completion assessment — 2026-09-30

All eight roles completed at runnable source `518e05891cf8eb2e84621cdcfcea83525f3d1c49`
(tree `26c77a9890beb3a9087ada142d71b6f56d0a9ab0`). **Park Q in this tested
a1100 networks 1–4 development profile; retain B as comparator/rollback.** The
completed experiment does not demonstrate a consequential common-background benefit from Q,
while its realized broad-source response is lower. Independent scientific review
supports this conservative disposition with the estimator limitations below.
This is not unrestricted fit validation or production acceptance.

## Source evolution and historical comparison

The B reference becomes 30.69% brighter after five applications. Its axis ratio
falls from 2.234 to 2.115, but major FWHM grows from 10.814 to 10.980 arcsec and
geometric-mean width grows from 7.236 to 7.551 arcsec. It is slightly rounder,
not narrower. Brighter output alone does not establish correct astronomy.

**Fitter limitation:** starred B endpoints reach the artificial angular boundary
at −pi. The periodic-angle optimizer can therefore terminate on a coordinate
boundary; these intermediate peaks and widths do not establish unrestricted-fit
accuracy, physical oscillation, or convergence. M0 and both final reference fits
have no active bounds. The raw source-evolution plot/export must be read with
this qualification. Fixed-template and aperture measurements subtract the fitted
constant, and feedback admission uses that constant and centroid: they are not
fully independent of this limitation. No corrected-solver trajectory is claimed.

Peaks and constants are mJy/nominal beam; centroid and widths are arcsec. Successor
centroids are FK5 J2000 tangent offsets about RA 176.7429079166667 deg,
Dec 39.97619555555556 deg on a 2 arcsec grid. Historical centroids are native
AltAz offsets on 1 arcsec grids, not registered to those successor axes.

| Map | Peak above constant | Constant | Centroid x, y | Major × minor FWHM | Axis ratio | Geometric-mean FWHM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 223.28 | -2.04 | 4.722, 7.464 | 10.814 × 4.842 | 2.234 | 7.236 |
| T1 * | 265.42 | -1.22 | 4.822, 7.445 | 10.317 × 5.425 | 1.902 | 7.481 |
| T2 * | 278.79 | -0.96 | 4.835, 7.410 | 10.326 × 5.541 | 1.864 | 7.564 |
| T3 | 290.74 | -0.92 | 4.774, 7.407 | 10.864 × 5.176 | 2.099 | 7.498 |
| T4 * | 283.12 | -0.86 | 4.800, 7.404 | 10.434 × 5.577 | 1.871 | 7.628 |
| T5 | 291.81 | -0.87 | 4.729, 7.409 | 10.980 × 5.193 | 2.115 | 7.551 |
| Earlier V2 (no FRUIT, 1 arcsec) | 251.70 | -1.61 | 8.461, -2.786 | 8.156 × 5.936 | 1.374 | 6.958 |
| redu09 (FRUIT index 4, 1 arcsec) | 574.59 | 32.59 | 7.353, -4.943 | 12.567 × 7.312 | 1.719 | 9.586 |
| Q T5 | 291.87 | −0.87 | 4.718, 7.405 | 10.970 × 5.177 | 2.119 | 7.536 |

Q's M0→T5 peaks are 223.37, 274.21, 287.28, 290.84, 291.51, 291.87;
its reference fits have no active bounds. B and Q final peaks differ by only
0.058 mJy/beam. Their model-conditional endpoint peak standard deviations are
14.11 and 14.24 mJy/beam. Those are single-map conditional quantities, not
paired B−Q or injection-difference errors. Intermediate exported guard errors
use frozen M0 covariance, not cycle-specific noise; historical comparable errors
are unavailable. No significance or convergence claim follows from these errors.

The final B peak is 50.8% of redu09's 574.59 mJy/beam. redu09 is broader and has
an elevated fitted constant (+32.59 versus −0.87). Its 2 arcsec averaging check
still gives 559.89 mJy/beam. Earlier V2 averaging gives 242.78. Sampling alone
does not remove the difference. Historical standard PCA rank 5, weighting,
mapmaking and detector/calibration context differ from this rank-10 observed-entry
ALS/uniform ordinary MAP experiment. redu09 records FRUIT index 4, not proof of
exactly these five applications. No pixelwise subtraction, position-angle
comparison, unique FRUIT attribution, absolute-flux or historical-ground-truth
claim is made. See the [saved historical assessment](../point152420_map_comparison_2026-09-29/redu09/ASSESSMENT.md).

## Genuine residuals, contamination and support

B's background standard deviation changes only 29.3403→29.3152 mJy/beam from
M0 to the genuine T5 post-PTC residual, on 7,882 common pixels outside the frozen
four-major-FWHM source mask. Robust scale changes 24.2426→24.1003. The brighter
source has not established a broad removal of surrounding contamination or made
surrounding structure demonstrably more credible.

Q selects the same network 2 / detector 175 / physical run 0 in all four Q
inputs. Its off-source triggering event has onset 4258 and window [4136,4502],
but exclusion covers the whole native run [0,7691). B retained 7,190 contributions,
including 1,303 inside the population source region and 1,585 inside the 40 arcsec
fit disk. Selected-run B T5 background std is 1,747.19 mJy/beam. Its Q trace is
absent by exclusion, not a zero residual or cleaned recovered exposure.

| T5 residual region, common support | Pixels | B std | Q std | Change |
| --- | ---: | ---: | ---: | ---: |
| Background outside frozen source mask | 7,875 | 29.3098 | 29.2947 | −0.052% |
| Selected-event footprint | 29 | 57.7685 | 56.5916 | −2.04% |
| Quiet-control footprint | 68 | 26.6585 | 26.3296 | −1.23% |
| 40 arcsec fit disk | 1,257 | 23.4660 | 23.3932 | −0.31% |

Background tails beyond the fixed B six-MAD threshold remain 9→9; background
robust scale rises 24.0911→24.2349. There are 1,152 common physical detector/runs
with no retained-count changes; median per-run Q/B background-std and coherent
half-second-contrast ratios are essentially 1. The images and selected/quiet
excerpts show localized changes without a broad common-support improvement.
These are descriptive, spatially correlated comparisons, not independent-pixel tests.

B has 9,351 supported pixels, Q 9,370, common 9,344: **7 B pixels lost and 26
Q-only supported pixels**, with none lost in the fit disk or selected footprint.
The 26 arise from support/normalization behavior, not recovered exposure. Native
MAP contributions fall 8,163,715→8,156,525, exactly the 7,190 removed cells.
Lost B-pixel residuals range −59.35 to +38.97 mJy/beam; their absence is not counted
as contamination cleaning. All common-support and removed-data results remain separate.

## Matched source response and disposition

Each injected role is compared to its own uninjected policy reference. Both
policies use identical B/M0-bound raster geometry, while Q selection is rerun on
each perturbed original-derived input and learned states remain independent.
Pixel gain is projection of actual injected-minus-reference onto ideal response
on common 40 arcsec support; eta is relative norm error to ideal.

| Injection | B pixel gain | Q pixel gain | B eta | Q eta |
| --- | ---: | ---: | ---: | ---: |
| Compact +10% | 1.0281 | 1.0262 | 0.1708 | 0.1698 |
| Compact −10% | 1.0232 | 1.0236 | 0.1784 | 0.1777 |
| Broad +5% shoulder | 0.8608 | 0.7797 | 0.3140 | 0.3364 |

Compact response is similar, with nonzero shape error. Broad response is partial
in both and 8.1 percentage points lower in realized Q. Broad fixed-template
ratios are 0.9947 B / 0.9400 Q; two-FWHM aperture ratios 0.7854 / 0.7106.
Those fitted-constant-dependent diagnostics do not override pixel-space findings.
B compact-plus T5 ideal fit and Q compact-minus T5 actual fit hit the angle bound;
their peak ratios (1.693 and 1.350) are not reliable unrestricted-fit response
validation. Conditional RTC gain near 1 does not establish full pipeline response.

No paired uncertainty establishes statistical significance or isolates quarantine
from downstream learning and numerical-path effects. Nevertheless, the tiny observed
common-background change plus the broad-response tradeoff and estimator ambiguity
provide no evidence for provisional Q adoption. **Park Q, preserve B and all
frozen evidence.** This does not establish that Q is universally ineffective,
that B is scientifically correct, or that corrected fitting would reproduce these
trajectories. The angle-bound limitation must be addressed before stronger
quantitative peak-response or unrestricted-fit validation (retained debt D20).

## Implementation, accounting and runtime

The completed repair separates deterministic auxiliary template support from the
unchanged 40 arcsec fit/admission/feedback region; no template clipping or missing
pixel zero-padding is used. Latest cap-500/rank-10/tolerance-1e-5 and approved
short-stretch RTC diagnostic-padding policies remain frozen. Completed-child
scratch compaction and report-size accounting were repaired before this run.
Standing owner authority and two-replacement budget are preserved by
COMPLETION_AUTHORITY.md; this assessment does not change scientific policy.

Eight parent starts, eight trajectories, **40 feedback applications**, eight
endpoints, 32 native RTC Apply receipts: all complete, exits 0, no unexpected
error-level run-log messages, no unknown accounting. Every T5 consumes m4;
m5 is measured/published but not applied. No new observational role was run for
this assessment. Zero replacement starts were used after the directive; two
remain available only for diagnosed in-scope replacements. Two historical failed
trajectories remain charged: aggregate known history is ten trajectories and
40 feedback applications, plus the initial build-only failure with no science.

The campaign ran 08:49:03–09:36:05 EDT (47 min 1.9 s). Raw-to-final B-REF is
174.51 s; Q-REF 182.11 s. B initial RTC/PTC/MAP/guard is 90.11 s, followed by
five feedback applications and endpoint in 82.54 s. Q equivalents are 92.00 and
83.67 s. Eight child-role times total 1,507.68 s; the remaining campaign time
includes gates, preparation, selection and report work. Nested spans are not
additive and this single campaign is not a repeated timing benchmark. Sampled
aggregate role RSS peaks range 2.86–3.43 GB.

Retained campaign size is 18.378 GB and prior counted roots 2.623 GB, within the
shared 30 GB ceiling. Per-role completed-child scratch removal manifests remain
preserved and removed paths are absent. No additional scratch was deleted in
this closeout; original data, historical maps, frozen products and prior evidence
remain intact.

## Verification and reproducibility limits

[Completion audit](completion-results/execution-verification.json) verifies the 36 compact
report bindings, RTC receipts, five-application histories and candidate linkage,
fixed geometry/injection rasters, exits/logs and accounting. [Independent review](completion-results/scientific-review.json)
binds the exact runnable source/tree and independently reproduces common-support
scatter and pixel gains; its pass is explicitly limited to truthful completion
and conservative parking, with separate scientific, architectural and hygiene
dispositions. [Background comparison](completion-results/B-background.json)
is a read-only NumPy/Astropy check with bound inputs. The audit scripts accept
explicit output destinations and never launch a fit or reduction.

A supplemental equivalent-angle saved-map refit was attempted but **did not run**:
local venv SciPy `_spropack` fails macOS `dlopen` with invalid `__thread_bss`
section metadata, both inside and outside the sandbox. The [error log](completion-results/supplemental-refit-import-error.log)
is preserved. No package was changed, no substitute fitter used, and no refit
numbers are inferred. This separate current-environment limitation does not
undo the completed native campaign or successful saved-array audit; repair it
before any new SciPy-dependent validation. The fitter diagnosis is based on the
recorded active bounds and code, not a claimed successful diagnostic refit.

Campaign root: `/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930`.
Its original terminal STATUS remains `COMPLETE_PENDING_OWNER_REVIEW`; this dated
assessment supplies the reviewed Q parking disposition without rewriting frozen
run evidence. Numerical source remains 518e05891; a later report/document commit
must not be represented as the source that executed. No push, integration,
production change, expanded fit policy, new target, or further campaign launch.
