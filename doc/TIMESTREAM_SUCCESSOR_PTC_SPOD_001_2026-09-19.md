# Bounded PTC frequency/time diagnostic — 2026-09-19

## Full available-network result — 2026-09-21

The bounded observation 152390/0/2 campaign is complete with preserved partial
results. The unchanged observed-entry ALS10 baseline runs RTC → CAL → PTC on
all eleven supplied networks; all 44 worker-count comparisons pass. The returned
parallel diagnostic is PASS for all eleven networks. This establishes development
coverage and reproducibility, not qualification of every observing mode or
permission to change treatment. The original September 19 small-cohort result
below remains historical evidence; this section supplies the broader census.

**Decision: retain ALS10. One network-12-only source-preservation experiment is
justified by remaining identifiable 11 Hz structure. No general atmospheric
interval change or blanket extra spectral cleaning is justified by this run.**

### Coverage and support

The approximately 1,241.4-second SCIENCE observation requests 5,518 detectors;
3,962 have nonzero CAL support and some PTC support. Networks 6 and 10 were not
supplied. The accounting JSON groups those missing identities under `unbound`;
its empty per-array missing lists do not mean complete thirteen-network coverage.
Shorter-array output cadences remain 122.0703125 Hz, versus 61.03515625 Hz for
networks 11/12. CAL/PTC values retain their mJy/nominal-beam convention.

| Array | Supplied networks | Requested | CAL/PTC-supported | CAL detector-s | PTC detector-s |
|---|---|---:|---:|---:|---:|
|1.1 mm|0–5|3,173|2,296|2,831,134.253|2,827,960.492|
|1.4 mm|7–9|1,368|932|1,151,820.644|1,151,820.644|
|2.0 mm|11–12|977|734|899,181.953|899,181.953|

Network 4 scan 121 retains its one failed ALS fit: native interval
`[147621,148841)`, 319 fit detectors, 100 iterations, not converged. Its 387,422
eligible CAL samples explain all CAL-to-PTC loss: 3,173.761 detector-seconds,
0.0650% of all CAL detector-time (0.806% within network 4). Other intervals remain
available. There are 1,364 fitted intervals and one failure; failure is not
converted into success by the diagnostic PASS.

| Network | Supported / requested | Full common s | Full distinct Fourier s, 2s / 8s | Full 8s windows | Supplement detectors | Supplement 8s distinct s / windows |
|---:|---:|---:|---:|---:|---:|---:|
|0|498 / 630|1098.2|1026.4 / 776.0|139|449|863.9 / 165|
|1|406 / 494|1084.3|975.4 / 644.0|119|366|1059.7 / 228|
|2|372 / 510|941.5|837.5 / 528.0|95|335|851.9 / 162|
|3|382 / 533|1083.7|1031.4 / 855.9|166|344|979.9 / 194|
|4|320 / 532|650.7|412.8 / 92.0|13|288|520.0 / 91|
|5|318 / 474|1121.1|1064.4 / 859.9|168|287|955.8 / 193|
|7|184 / 420|1168.6|1116.4 / 919.9|180|166|1003.8 / 203|
|8|384 / 460|1162.4|1058.4 / 739.9|140|346|1099.7 / 240|
|9|364 / 488|1139.7|1119.4 / 1031.7|225|328|1163.5 / 265|
|11|371 / 486|843.2|786.5 / 583.7|103|334|779.6 / 150|
|12|363 / 491|738.6|661.6 / 415.8|70|327|675.6 / 130|

Every primary population contains all CAL-supported detectors. Every supplement
uses the fixed 90% cohort selected solely by temporal availability under the
existing full/90%/75% rule. This favors continuous support; it is not an unbiased
replacement census. Diagnostic intersection/window losses are not production
exclusions. Eight-second full-population coverage spans 7.4–83.1% of the
observation; supplementation raises it to 41.9–93.7%, without changing any
production mask. Window counts include overlap and are not independent counts.
Per-pool realization/rank limits and exact memberships remain in the result JSON.

### What the spectra establish

The fixed near-11-Hz CAL feature generalizes across all three arrays. At the
eight-second bin (10.9951 Hz on shorter arrays, 11.0063 Hz on 2 mm), most networks
have a broadly distributed leading pattern and strong agreement between
separated pooled halves. Network 4 is an exception: effective participation is
only 2.6 detectors, with 13 windows and overlap 0.052. Its supplement improves
time support but participation remains 2.4. Network 9 is more concentrated
(14.6) than the other shared examples. Participation is a loading statistic,
not a detector membership count. No inter-network phase coherence or single
physical origin is inferred.

| Network | Input leading power fraction | Phase-control range | Effective participation | Separated-half pattern overlap |
|---:|---:|---:|---:|---:|
|0|92.3%|7.3–8.3%|116.2|0.995|
|1|51.3%|20.2–20.5%|123.5|0.857|
|2|58.2%|16.8–17.4%|30.3|0.881|
|3|92.8%|9.3–9.7%|93.1|0.996|
|4|71.9%|52.4–55.0%|2.6|0.052|
|5|86.4%|10.0–10.3%|127.2|0.987|
|7|98.7%|7.2–7.9%|75.2|0.999|
|8|91.5%|12.7–13.3%|67.4|0.990|
|9|92.4%|20.7–21.2%|14.6|0.996|
|11|95.3%|7.2–8.1%|133.9|0.995|
|12|52.1%|8.7–9.1%|147.1|0.919|

Remaining structure after PTC is network-specific. On full two-second support,
projection of PTC output onto a separately learned CAL pattern exceeds all eight
frozen-operator phase controls in both folds for networks 1, 2, 8 and 12. The
90% supplement repeats that comparison for 1, 2 and 12, but not 8. Network 11
is mixed; 0, 3, 4, 5, 7 and 9 do not exceed those controls in either full
two-second fold. This is a descriptive comparison, not a detector rejection
threshold or proof of zero contamination elsewhere. Absolute pattern PSD also
matters: network 8's two-second full residual is only 823–839, versus
104,300–220,730 (1), 58,032–67,741 (2), and 51,104–53,852 (12), with different
populations and support. These are collective unit-norm-pattern PSDs, not
per-detector sensitivity estimates.

Network 12 gives the clearest consistent follow-up case. Exact matched-subset
PSD in `(mJy/nominal-beam)^2/Hz` along the separately learned unit-norm pattern:

| Profile / fold | CAL pattern PSD | PTC pattern PSD | Eight conditional controls | Controlled evaluation windows |
|---|---:|---:|---:|---:|
|Full 2s A|395,835|53,852|11,281–14,973|75|
|Full 2s B|434,059|51,104|12,110–15,710|75|
|Full 8s A|610,758|60,321|3,724–8,100|6|
|90% 2s A|326,839|43,288|6,620–9,053|69|
|90% 2s B|367,838|37,608|8,390–10,437|69|

Full 2s uses 471 matched/controlled windows overall, with 90 separated training
windows per fold. Full 8s has 14 matched/controlled windows; the available fold
trains on 17 windows. Its opposite fold has no evaluation support. The supplement
has 645 matched but 460 controlled 2s windows; the table uses only each fold's
69 controlled evaluations. Supplement 8s has no fold with two controlled
evaluations, so it supplies no confirming pattern-control result. These are
important limits on repeatability, not reasons to substitute excluded samples.

The controls preserve observed individual spectra, destroy input coherence and
apply the actual frozen PTC operator, including its induced correlations. They
are eight conditional realizations, not a physical noise estimate, confidence
interval or astronomical transfer test. Separated pattern training/evaluation
windows share no samples, but the production operator was learned from the
saved data and slow variations can remain correlated. The former roughly 53%
total-bin-power ratio is **not line-removal efficiency**. Two-second Hann
resolution also blends shoulders; the eight-second 10.7562/11.0063-Hz leading
patterns have overlap 0.00283 and must not be treated as one moving peak.
Existing notches and all CAL/PTC preprocessing histories remain bound.

Large individual-channel peaks recur on multiple networks: examples include
network 0 channel 348, network 1 channel 492, network 5 channel 76, and network
12 channels 74/121. At inspected 15.75/16.75/20-Hz bins, some leading fractions
closely match the diagonal-preserving phase controls and participation is near
one. They are not evidence that every detector shares a disturbance. Other
features, including network 3 and 16.75-Hz examples on 7/8, are more distributed;
this census does not assign all peaks the same mechanism or create new flags.

Low-frequency uncertainty still limits the atmospheric conclusion. Full 8s
local pools have at most 27 realizations (network 4 has only 0–7); their rank
cannot be the full detector count. Supplemental support does not resolve
fluctuations below approximately 0.125 Hz. Network 12's 0.5-Hz median
between-pool/within-pool leading overlaps are 0.165/0.156 full and 0.258/0.132
supplement: differences remain compatible with estimation uncertainty.
Network 5 has a narrower indication of leading-direction change at 0.5 Hz in
2s profiles (0.482/0.830 full; 0.470/0.862 supplement), while its three-mode span
is more stable (0.827/0.855; 0.837/0.834). The 8s comparison does not reinforce
a broad evolution claim. These descriptive overlaps cannot separate changing
sky structure, atmosphere, gains or detector health. Failure to establish
evolution does not establish stationarity.

### Execution costs and next decision

The eleven-network reduction timings remain 167.4, 116.5, 66.6, 41.1 and
25.1 minutes for 1/2/4/8/12 requested workers. Twelve requests start eleven
network workers, with measured peak process RSS 81.31 GB and 6.66x single-run
speedup over one worker. All 44 exact worker comparisons passed separately in
job 64656301. No optional timing repeats were performed. These are development
processing costs with normal evidence/publication, not diagnostic costs.

The parallel diagnostic continuation took approximately 36 minutes from
submission, including its 7m34s finalizer. Networks 0/1 reused already completed
serial results; nine new computations were concurrent. Their individual
scientific-process times range 6.27–26.01 minutes. This is not a clean
eleven-network serial/parallel speedup measurement.

| Network | Diagnostic process min | Process peak RSS GB | Origin |
|---:|---:|---:|---|
|0|25.61|6.46|prior serial, reused|
|1|19.03|5.52|prior serial, reused|
|2|15.06|4.92|parallel|
|3|22.15|5.23|parallel|
|4|6.27|3.89|parallel|
|5|21.08|4.42|parallel|
|7|13.66|2.97|parallel|
|8|23.29|5.32|parallel|
|9|26.01|5.25|parallel|
|11|9.89|2.62|parallel|
|12|8.21|2.47|parallel|

Across the eleven completed diagnostic computations, 85.2% of their summed
190.26 process-wall minutes falls under `comparisons`: repeated phase controls,
held-out patterns, subspace comparisons and matched residual decompositions.
This includes numerical work, not merely comparing files. Individual input
preparation took 59–93 seconds; publication 8–12 seconds, rounded. Summed
process times are not campaign elapsed time. NumPy/SciPy OpenBLAS each report
one thread on Unity. The finalizer's summary subprocess used 115.5 seconds and
0.248 GB sampled process-family RSS; its whole Slurm batch MaxRSS was
15,391,260 KiB. Those are distinct scopes/measurements and not interchangeable.

**Proposed next experiment, requiring owner authorization:** compare ALS10
against one fixed, longer-learned treatment of network 12's specific 11.006-Hz
pattern, with separate learning/evaluation and paired astronomical injections.
Measure residual structure and source amplitude/waveform/shape together using
both frozen-state and relearned-state response. Preserve masks, existing
notches, local broadband fits and rank; do not chase the 10.756-Hz feature,
sweep intervals/ranks, or add other networks to this experiment. This can decide
whether remaining contamination offers useful source-preserving recovery.
It does not authorize treatment now. ALS10 stays the default development
baseline; MAP, FRUIT and automatic selection remain outside this work order.

### Evidence binding

Owner-returned archive `results-parallel-20260921T165411130692.tar.gz` has SHA256
`228e9da61b343c1fd795014b2eda44c547828aa92aba66f7c5219d55ac353c96`.
All 162 compact manifest entries, 12 sealed records, 13 current task FINALs,
eleven Slurm task outcomes and 77 returned result/plot hashes verify locally.
Forty-four numerical NPZ products and original science arrays remain on Unity;
their job-recorded hashes were not independently recomputed on the Mac.
Fourteen scheduling and seventeen diagnostic tests passed in preparation.
Before/after scientific source is clean at
`5cd3e28396a08917986f3ec73314a69d3605bd0d`; scheduling/reporting source is
`f5d589c84fd75d2d4af95d06ed41b2c8af80ab45`. All diagnostic code hashes match the
runtime; CAL plan 1, RTC attempt 1001 and VAL generation 2 remain explicit.

Durable intake, compact derivation, independently reviewed interpretation and
summary figure are under
`/Users/gwilson/work_toltec/local_data/2026-refactor/citlali-successor-full-array-result-20260921`.
Original returned archive/marker remain in the sibling
`citlali-successor-all-networks-20260920-return` directory. Exact per-network
JSON metric paths and limitations are in `SCIENCE_REVIEW.md`; `INTAKE.json`
records coverage/cost cross-checks. The existing
[performance handoff](../handoff/TIMESTREAM_SUCCESSOR_RTC_PERFORMANCE_001_2026-09-18.md)
records repository and review closure. No scientific executable changed here.

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
Canonical integration, pushes and any later treatment remain owner-controlled.

## Exact-source review and closure record

The completed diagnostic implementation is
`d4f13c13c1915665abf786cc074597e6360a969d`, tree
`be18ab1455104a40d35302c4085200f1e60a4b32`, with literal base and parent
`0ef1c6e626630962a41ebab0cdd3b5833ee6b039`.
Independent fresh-context, read-only review is **PASS WITH RECORDED LIMITATIONS**:
scientific/behavioral conformance passes with the limitations above;
architecture/ownership and repository/evidence hygiene pass. No finding
requires repair. The reviewer independently reproduced all six spectral
profiles and retained complex mode arrays exactly, reproduced 10 focused and
99 successor tests, verified all bound input hashes, and checked the nine
solver artifacts, three exact baseline comparisons and frozen-response records.

The original 61-file evidence packet remains immutable. Its `SHA256SUMS`
digest is `6f71931d248cda59ad2bd823d3a873190d6c30e1d58e58d21b65be8f1d3bbb8f`.
The same evidence root now also retains `SOURCE_BINDING.json` and
`INDEPENDENT_REVIEW_d4f13c13c.md`; the latter's SHA-256 is
`703bd3cafed864d75607d8bfd040d5f8ef48da84dd34b943a7937dbd5d033e8b`.
These are supplemental records, not edits to the sealed evidence or its
original report. A later documentation-only closure records this review;
its exact identity and independent review are retained separately in
`CLOSURE.json` and a supplemental manifest.

Final live-ref verification finds canonical still at
`e32c4584dec07617d6caa857d39bf13047b6c4b0`, and the prerequisite
`codex/timestream-successor-multinetwork-001` now pushed at
`0ef1c6e626630962a41ebab0cdd3b5833ee6b039` (it was absent at preflight).
The diagnostic branch is unpushed and unintegrated. Its clean worktree is
`/private/tmp/citlali-timestream-successor-ptc-spod-001`. The completed probe
is retained for owner disposition; the existing multi-network spine remains
the pending implementation responsibility. No treatment, rank or default
change is implied by closure.
