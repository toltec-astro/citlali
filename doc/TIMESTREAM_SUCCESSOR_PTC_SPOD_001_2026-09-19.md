# Bounded PTC frequency/time diagnostic — 2026-09-19

## Work order / preflight

Owner: Grant Wilson, directive `4dad2fdb-901a-4483-b70e-66bca9346dde`.
Purpose: measure frequency-dependent detector correlations and their time
stability on immutable CAL inputs, to select one useful next PTC experiment.
Tier 2; module-probe slot. The pending multi-network spine is unchanged.

Read: AGENTS.md, toltec-context, ENGINEERING_GOVERNANCE,
TIMESTREAM_SUCCESSOR_GOVERNANCE and REVIEW_AND_CONFORMANCE. Their effective
incorporation is the ledger record for `06a3ade51c1b3f38887295433d913811bf25cd14`.
Read current status, SCIENTIFIC_GOALS, architecture, scientific conventions,
SCI-PTC v0.1/r0.5 and September 18 observed-entry and partial-completion owner
bindings. SCI-PTC requirements 001–009, 026–029, 084–088, 090–099 preserve
immutable CAL identity, group-local masks, explicit fitting policy and limited
response claims. The current directive authorizes diagnostic Welch SPOD, not
a new production estimator or subtraction policy.

Verified live canonical: `e32c4584dec07617d6caa857d39bf13047b6c4b0`.
Literal probe base: `0ef1c6e626630962a41ebab0cdd3b5833ee6b039` (reviewed
short-array low-pass closure; remote feature ref absent at preflight).
Branch: `codex/timestream-successor-ptc-spod-001`.
Worktree: `/private/tmp/citlali-timestream-successor-ptc-spod-001`.
Initial staged/unstaged/untracked state: clean. No other worktree repurposed.

PTC diagnostic tooling owns preparation, spectral evidence and advisory
comparisons. It reads published CAL/RTC/PTC receipts and exact binary products;
it emits separate diagnostic artifacts. Detector identity, mJy/nominal-beam
units, actual native time, masks, processing history and VAL generations stay
bound to their parents. No Engine or application-route change. Observation
residency does not define Fourier, pooling or production-fit intervals.

Included: full supported network12 and existing small network0/7 CAL cohorts;
complex Welch CSD, local/pooled subspaces, separated-realization checks,
controlled tests, existing-solver comparisons only where meaningful, cost
accounting. Excluded: new ingestion, new scientific thresholds, automatic
rank/notch selection, spectral subtraction, PTC replacement, MAP and FRUIT.
Missing other-network saved CAL inputs are reported, not manufactured.

Expected paths: `tools/timestream_successor/*ptc_spod*`, this record, status,
and existing PTC handoff. Gates: estimator/reader/phase/subspace/mask tests,
real-data exact-input checks, existing Python/config/baseline gates and
independent fresh-context exact-SHA review on all three governance axes.
Production source/defaults must remain byte-identical to base. No C++ changes
are planned, so no new compilation/Unity gate is triggered. Local Python
uses the tolteca venv, single-thread numerical libraries and headless plots.

Reassess on inadequate common support, incompatible calibration/filter/identity,
insufficient spectral realizations or a required policy change. Stop at an
honest uncertainty/coverage limitation instead of inventing support. No push,
canonical integration, cleanup or production activation is authorized.

## Scientific result

**Retain the current PTC baseline. The most promising next experiment is
longer shared learning of the specific ~11 Hz detector pattern, tested on
held-out data with source injections.** This diagnostic does not establish a
reason to lengthen every broadband PTC fit or replace the cleaner with SPOD.

The observation is SCIENCE/Lissajous NGC4449, 152390/0/2, approximately
1,241 seconds. Primary inputs are the immutable calibrated x values actually
entering PTC, in mJy/nominal-beam. No upstream stage was rerun. The runtime
Learn/Consider/Apply path and all admission rules remain unchanged; this is
separate advisory learning evidence, not an Apply plan or production flag.

| Saved network | Fixed diagnostic population | CAL output sampling / Nyquist | 2-second / 8-second Fourier realizations | Distinct time used, 2s / 8s |
|---|---:|---:|---:|---:|
| 12, 2 mm | 363 of 414 requested | 61.035 / 30.518 Hz | 581 / 70 | 661.62 / 415.76 s |
| 0, 1.1 mm | 11 of 12 saved | 122.070 / 61.035 Hz | 1184 / 272 | 1203.31 / 1159.48 s |
| 7, 1.4 mm | 6 of 12 saved | 122.070 / 61.035 Hz | 1191 / 273 | 1212.30 / 1167.48 s |

The network12 population includes **every detector with any CAL support**;
selection never uses a spectral ranking. The original 51 zero-support
identities remain unavailable. Common support is 738.56 seconds; short
stretches unable to hold a full window explain the smaller Fourier coverage.
The smaller network0/7 cohorts are prior development selections, not an
array-wide census. Networks1–5,8,9,11 have raw inputs but no saved connected
CAL products in this checkpoint; networks6/10 have no supplied raw inputs.
Those are the exact barriers to further coverage. The multi-network ingress
spine remains the next implementation responsibility, independent of this probe.

The 307/67/33-tap low-pass profiles and their exact hashes are retained.
Network12 is factor2; networks0/7 retain native rate. All three low-pass gains
at 11 Hz are within 0.001% of unity. Eight network12 channels
(269,296,300,307,366,402,406,438) additionally have the accepted explicit
1-Hz/3-second notch: its amplitude gain at 11 Hz is 0.02250. Their treatment
history remains visible in the fixed population. No new notch was applied.
Existing original-native D2 spectra are reported separately in original-x
units and on their own support. They show the original feature, but cannot
supply a matched attenuation or calibrated power comparison. A missing
post-treatment feature would not establish original absence. No r subtraction
or mixed-unit x/r decomposition is performed.

### What the frequency dependence says

At 11.006 Hz on network12, the 8-second estimate puts **52.15%** of total
spectral power in its leading detector pattern, versus **8.69–9.12%** in eight
phase-scrambled controls retaining the exact individual detector powers.
Its loading participation is equivalent to **147 detectors**
(`1/sum(|u|^4)`, descriptive rather than a membership count). This is strong
shared structure, not just one loud detector. On the small network0/7 cohorts,
the corresponding fractions are 94.65%/98.77%, well above their unequal-power
controls of 41.14–41.46%/42.15–44.36%.

Low-frequency structure is also present. At 0.250 Hz on network12, the first
one/three modes carry 36.0%/73.9%, and at 0.500 Hz 25.1%/52.5%; the leading
mode exceeds its diagonal-power control. The leading 0.5-Hz loading spans an
effective 45 detectors. Network0 likewise shows shared low-frequency power.
Network7's large low-frequency leading fractions are substantially explained
by power concentrated in one of its six detectors: those fractions alone
must not be called a common atmospheric mode.

The very strong network12 15–17 Hz peaks chiefly occupy channels74 and121.
For example, at 15.759 Hz the leading fraction is 85.1%, but the randomized
control is already 84.7% and participation is 1.02 detectors. Much of the
broadband 20-Hz leading power is likewise concentrated in channel181. These
are power-concentration findings, **not new health or rejection decisions**.
SPOD eigenvalue concentration alone is insufficient evidence for correlated
contamination.

Nearby frequencies do not necessarily describe the same disturbance. In
network12 the pooled 10.756- and 11.006-Hz leading patterns have overlap
`cos²(angle)=0.00283` (87 degrees); the former is dominated by one detector.
Thus changes in which bin is largest in 10–12.5 Hz cannot simply be called
frequency drift of the coherent 11-Hz feature. The small network0/7 cohorts
have much more similar adjacent-bin patterns, with overlaps 0.9991/0.9981.
No central-frequency drift below the estimator resolution is established.

### Stability, scan context and uncertainty

The 8-second network12 estimate has bin spacing 0.12507 Hz and Hann equivalent
noise bandwidth 0.18761 Hz. For the native-rate networks they are
0.12494/0.18742 Hz. The 2-second estimator gives 0.50029/0.75043 Hz. Neither
resolves atmospheric fluctuations below approximately 0.125 Hz. Fourier
segments, **120-second local pooling**, and the existing approximately
**10-second PTC fits** are separate choices. Windows may cross engineering
fit boundaries in the uncleaned CAL parent, but never physical runs or invalid
support. Comparisons to cleaned PTC output require windows contained in one
actual saved fit.

Network12's two separated pooled halves give 0.919 overlap for the leading
11-Hz pattern (17 realizations each), while local spectral power changes by
a factor of 2.22. The two smaller networks give 0.998/0.999 overlap and power
changes of about 2.9. This supports a comparatively stable pattern with
changing strength, especially in the small cohorts. It does not prove exact
observation-wide stationarity.

Local network12 8-second estimates contain only **3–11 realizations** and
therefore have rank at most 3–11, despite containing 363 detectors. Their
median between-interval 11-Hz overlap is 0.631, compared with 0.498 for the
available separated local split controls. At 0.5 Hz both are low
(0.165/0.156). This does **not** distinguish real atmospheric pattern evolution
from estimation uncertainty. The sharper spectral estimate trades precision
in time-local subspaces for frequency resolution. Three- and ten-dimensional
principal-angle comparisons and leading eigenvalues are also retained; a
nearly tied individual eigenvector is not given a unique physical identity.

TEL actual RA/Dec, elevation, speed and direction summaries are bound to the
same admitted Fourier intervals. These descriptive tangent-plane derivatives
do not replace AST's motion authority. For network12 the selected-pool median
speed spans 21.6–33.5 arcsec/s and elevation 51.70–54.54 degrees; all four
scan-direction quadrants contribute. Across only eleven pools, 0.5/11-Hz
power has Spearman association with median speed of -0.036/+0.055 and with
elevation -0.055/-0.082. These establish no simple scan-state explanation.
Some small-cohort pointing associations are larger, but confounding by time,
masks and population prevents causal attribution. None of this diagnoses
gain, tune, atmosphere evolution or detector health.

### Existing cleaner comparison and one next experiment

On 471 identical supported 2-second network12 windows contained within saved
PTC fits, the current rank10 cleaner leaves **52.9% of the input 11-Hz power**.
For 14 eligible 8-second windows, the fraction is 53.8%. These are total
frequency-bin powers including uncorrelated noise, not isolated line power or
a sensitivity metric. Median leading-SPOD containment in the actual local
rank10 PTC span is 0.63/0.58 for the two settings. Its broadband modes capture
some of this pattern, while strongly representing the individual-detector
15–17-Hz disturbances.

Nine existing-solver replays use the first, middle and last saved CAL segments:
ALS10 baseline, ALS5, and explicit pairwise-covariance10 comparator. All preserve
membership/masks and converge; all three ALS10 outputs exactly reproduce the
saved baseline. In the first/middle segments, ALS5 leaves 77.6%/83.9% of the
11-Hz bin power versus ALS10's 51.9%/68.0%. The last segment has insufficient
complete Fourier windows and reports that measurement unavailable. Mean
absolute residual pair correlation also rises at rank5. Fitting costs differ
only by milliseconds (approximately 0.027–0.030 seconds). Pairwise10 closely
tracks ALS10 here; this does not replace the accepted mask-aware solver.

The existing staggered width-two-sample Gaussian CAL-grid probe, evaluated
under each exact frozen basis without subtracting the learned location,
retains matched-template amplitudes 0.740–0.846 at ALS10 versus 0.821–0.935 at
ALS5. This illustrates the cost/contamination/response tradeoff; it is **not**
a complete astronomical response or a data-dependent injection test. No
operator was relearned after adding a source, and no scientific rank winner
is selected.

**One recommended next experiment:** compare the current ALS10 baseline with
one explicitly bounded treatment using a longer-learned ~11-Hz detector
pattern, trained and evaluated on separated windows. Include paired source
injections both with frozen state and with relearning. Keep broadband
low-frequency cleaning local, all masks/physical boundaries unchanged, and
no automatic band/rank selection. This tests whether the demonstrated stable
shared feature can be reduced with less source loss than indiscriminately
raising broadband rank. It needs a separate treatment work order; this probe
does not implement it. Nothing here justifies a universal interval/rank
change for POINT, OOF, BEAM or SCIENCE.

### Method references and boundaries

The diagnostic uses complex Fourier snapshots and averaged cross-spectral
matrices, with identity detector metric, arithmetic window-mean removal,
periodic Hann, one-sided density normalization and no padding. Subspaces are
compared with principal angles, invariant to phase and internal basis rotation.
This is the conventional Welch SPOD construction, with actual realization
counts/rank limits retained. See [Towne, Schmidt & Colonius (2018), section 3](https://arxiv.org/html/1708.04393v2)
and [Schmidt & Colonius (2020)](https://authors.library.caltech.edu/records/gcvbg-5ks79).

Separated controls share no samples and leave at least one full Fourier-window
gap; they are not asserted statistically independent of slow atmosphere.
Pooled-vs-local agreement shares data and is explicitly not an independent
check. Eight phase-randomized controls preserve diagonal spectral power and
remove interdetector phase coherence; they are descriptive controls, not new
production significance thresholds. No multitaper/adaptive estimator is needed
to reach this bounded conclusion.

The frozen/relearned response distinction follows the contract and the
measured data-dependence discussed by [Downes et al. (2012)](https://academic.oup.com/mnras/article/423/1/529/1747027).
That work motivates the limitation; its scientific selection rules are not
imported into Citlali.

## Cost, reproduction and conformance

The completed single-process runs explicitly cap OpenMP/OpenBLAS/Accelerate
at one thread. This Python realization does not expose its BLAS thread count
through threadpoolctl; the environment caps and measured process CPU are
recorded rather than claiming a directly observed backend count.

| Network | Input preparation | Fourier preparation/FFT | Eigensolves | Comparisons and controls | Publication | Input recheck | Total wall / CPU | Peak process memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 6.84 s | 0.35 s | 1.60 s | 2.41 s | 1.01 s | 1.24 s | 13.49 / 12.34 s | 1.90 GB |
| 0 | 0.17 s | 0.07 s | 0.29 s | 0.15 s | 0.74 s | 0.03 s | 1.50 / 1.41 s | 0.27 GB |
| 7 | 0.15 s | 0.04 s | 0.19 s | 0.10 s | 0.69 s | 0.02 s | 1.25 / 1.15 s | 0.24 GB |

These are local warm-file-cache measurements, not a Unity/scaling benchmark.
Network12 input preparation verifies the actual CAL/PTC relation and loads
both parent and baseline outputs; it is paid once for both Fourier settings.
The nine solver trials each take about 0.2 s internally, including loading,
preparation, fitting, application, the existing frozen response and output;
they never repeat RTC/CAL. Their exact costs and artifacts are separate.
The first exploratory plotting run incurred a font-cache build and cold input
cost; it is not substituted for the retained steady-state accounting.

Evidence root:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-ptc-spod-20260919`.
Each network's `result.json` records every contributing window, cohort
occurrence, filter and parent/VAL identity, all numerical summaries, exact
input hashes, source-file hashes, thread limits and costs. Compact mode files
and three figures per network accompany it; no CSD-by-time cube is retained.
All 2,400 network12 input artifacts and the 56 artifacts for each smaller run
are rehashed after use and remain unchanged. Saved parent data are preserved
in the accepted `successor-array-lowpass-530743ec5-20260919` evidence campaign.

Reproduce one network, using the retained exact producing input configuration
and a **new** output directory:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/ptc-spod-mpl XDG_CACHE_HOME=/private/tmp/ptc-spod-cache \
  /Users/gwilson/tolteca/bin/python \
  /private/tmp/citlali-timestream-successor-ptc-spod-001/tools/timestream_successor/run_ptc_spod.py \
  --input /private/tmp/citlali-multinetwork-001-20260919/a2000-explicit/output \
  --config /private/tmp/citlali-multinetwork-001-20260919/network12-explicit-input.json \
  --output /private/tmp/ptc-spod-network12-new
```

Use the same entry for `a1100-smoke`/`network0-input.json` and
`a1400-smoke`/`network7-input.json`. The existing-solver comparison is
`compare_ptc_spod_rank.py --ptc <saved donor-continuity/ptc> --binary
<exact citlali_ptc_cost> --output <new directory>`. It chooses only the first,
middle and last existing segments; it cannot join or redefine their intervals.
The report's 120-second pools assign windows by midpoint; neighboring primary
pools can share the edge of overlapping Fourier windows and are not independent
replicates. Exact window bounds are retained. TEL context is restricted to
contributing support inside each nominal pool.

Focused validation: 10 controlled tests pass, including complex CSD agreement
with SciPy, phase/rotation invariance, changing subspace, independent masked
noise, diagonal-power concentration, nonfinite/insufficient support, input
integrity and frozen response on zero-support rows. All 99 successor Python
and 207 baseline Python tests pass; config preflight passes 130 tests and all
audits. Baseline CLI-dependent checks use the preserved, explicitly identified
`530743ec5` executable, whose production sources/configs are unchanged; this
is not a new candidate C++ build. An initial baseline-harness run lacked its
expected build-path link and failed eleven assertions; supplying that same
preserved CLI resolved those harness failures. The temporary link was removed.

No production source, CMake, data configuration, coefficient artifact, mask,
filter, scientific contract, default route or prior output changed. Three
exact saved ALS10 replays demonstrate baseline numerical equivalence. The
new diagnostics have zero unexpected warnings/errors in their completed runs.
A preliminary response diagnostic correctly stopped on an all-excluded row;
its bookkeeping was repaired to preserve unavailable rows, with a focused
test. Initial NumPy complex-multiply status warnings were eliminated by using
SciPy's explicit complex BLAS operation and validating against SciPy CSD;
no warning suppression or input dropping was used.

No C++ build changes are made, so a new representative Spack/Unity gate is not
triggered. The earlier Unity result still applies only to its exact source.
Independent fresh-context exact-SHA review is required before closure; it
must cover scientific meaning, diagnostic ownership and evidence hygiene.
Canonical integration, pushes and any later treatment remain owner-controlled.
