# TIMESTREAM-SUCCESSOR-RTC-NOTCH-RECOVERY-001

## Coherent-removal and science-error continuation — 2026-09-15

The owner explicitly authorizes a bounded coherence assessment, one limited
network-local coherent candidate if supported, and four-arm science-error
comparison on 152390/n12. This directive permits a small map diagnostic through
existing downstream components. It supersedes the previous timestream-only
diagnostic restriction for this experiment, without authorizing CAL/PTC/MAP or
FRUIT redesign. It prohibits an arbitrary residual-line epsilon, further notch
optimization, a general framework, automatic selection and production activation.
The exact directive is preserved at
`/private/tmp/citlali-rtc-coherent-removal-2026-09-15/OWNER_DIRECTIVE.md`, SHA256
`d98308b092355cf8f544b7cc25467263dcddf5d7bedd3d0788d768579b3e12c2`.

Same branch/worktree: `codex/timestream-successor-rtc-notch-recovery-001`,
`/private/tmp/citlali-timestream-successor-rtc-notch-recovery-001`.
Literal continuation base: `5b39eb9e4eabd2389e16745c7b8c26d93969bd11`, tree
`ddf09aa3770fb15b3b929daa157ebb0dfcd9feaa`.
Exact implementation: `10ac50f4537d11cde77a0e7c44cce0a2155471da`, tree
`77865c5917c6d38cfe357ab2a2cd712cd13d12a4`, direct parent the literal base.
The source changes twelve CMake/tool/test files, with no production-library
changes. `source-state.json` below records every content digest. This separate
documentation-only commit is the closure; its exact SHA/tree and independent
review are recorded externally to avoid circular identity.

Effective governance remains accepted `06a3ade51c1b3f38887295433d913811bf25cd14`,
incorporated `77507836325eff9f469062d5884481ea37599594`, with all three accepted
digests in `preflight.json`. Scientific authority remains the accepted WP-7.1
baseline/router, SCI-RTC lifecycle/support requirements and ADR 0017–0023,
including the retained numerical half-support and optical/motion/cadence rules.
The user's directive permits an explicitly ineligible experiment, not a waiver
of those rules or entry in an accepted-plan index.

### Runtime boundaries and fixed experiment

The engineering learn/consider/apply workflow remains distinct from execution.
Offline `CoherentEvidence` learns exact original pair/time/VAL/support-bound
donor trajectories. `CoherentPlan` freezes donor bases, target coefficients,
inverse Gram matrices, training support and an explicit failed admission.
Diagnostic Apply checks exact identities and replays original paired x/r.
The same coordinate-diagonal operator/metric/support is fitted separately to
x and r; no x waveform is copied into r and no x-only fallback exists. The
existing C++ runtime witness separately repeats actual transient/spectral Learn
with a named sky overlay present before Learn, then frozen original Apply.
Its coherent negative overlay and additional support remain identified
diagnostics; this does not implement an admitted native coherent RTC stage.

Affected channels are 269,402,296,300,366,438,406,307; descriptor-quiet controls
are 193,231,226,330. The 310 disjoint four-second matched windows use 102
training windows and two guarded sets of 101 later evaluation windows, with
reciprocal disjoint donor/target groups. x rank-one fraction is 0.9724 and
held-out complex power-error ratios 0.0212–0.0486 (14.6–22.1% RMS). r's fraction
is 0.252, without comparable prediction. Shared structure and small descriptive
motion/pointing correlations do not establish physical origin or no astronomy.

Only one candidate is tested: a measured common amplitude-and-phase trajectory,
fixed coordinate-specific target quadratures and no moving detector coupling,
extra modes or center/width search. Its declared centered 489-point demodulator
is separate from the unchanged 488-sample descriptive estimator. Training is
native interval [244,50024), 1.998848–409.788416 s. Although direct half-support
with LPF is 3.252 s, full projection reach is 1234.747 s because fitted
coefficients depend on early training. **The candidate fails the five-second
support rule.** Thirty-two target coefficients plus learned donor weights and
the correlated trajectory are model costs; conditional fit rank two is not
total effective degrees of freedom. Full nonlinear effective degrees of freedom
and calibrated parameter covariance remain unavailable.

Four arms replay original data: unchanged LPF/F2, prior six-second finite notch
plus LPF, coherent subtraction plus LPF, and rejection of the eight affected
occurrences. All other originally eligible detectors retain LPF/F2. Existing
paired exclusions, native time, beam, actual motion and cadence bounds remain.
Known sky is injected through all 419 originally APT-good detectors' own
pointing; 72 already flagged columns remain excluded, including explicitly
unavailable geometry/calibration fields. The provisional readout average and
r native leakage ratio 0.2 are declared, not measured calibration.

Existing Cleaner uses ten modes learned anew in each arm/injection. Existing
NaiveMapmaker uses fixed diagnostic APT scales, weights, two-arcsec pixels and
four ten-second chunks. This is a conditional partition/projection, not admitted
PTC scans or production CAL/JINC/FRUIT. Eight shared-frequency-phase background
realizations preserve measured cross-detector and x/r spectra; unknown original
sky/line magnitudes remain in that conditional background. Known added lines
and sky provide numerical truth without treating cleaned output or sidebands
as clean-noise authority. Four phase controls are response probes, not extra
independent noise draws. Original-template, original-frozen-projection and
source-before-coherent-Learn comparisons remain distinct. All nine actual C++
original/sky-before-Learn pairs retain identical transient causes/exclusions;
the stochastic population trials fix those exclusions while relearning coherent
and PCA state. They do not estimate full-pipeline transient-policy variance.

### Scientific result and retained limits

Coherent removal does **not consistently beat the finite notch or rejection**
in compact-source science error. No extra RTC line treatment demonstrates a
material benefit over the unchanged LPF plus downstream reference here.
Coherent-minus-LPF paired peak-MSE changes are -11.6,-7.0,+50.0,+143.3 with
descriptive standard errors 41.6,94.7,40.5,38.8 in (mJy/beam)^2 across three
crossings and the boundary challenge. Centroid, extended signal, artifacts and
covariance use separate units and both common/actual support in the full report.
Only 42.2% of the extended aperture is covered; that result is poorly constrained.

Actual 9.75–12.25 Hz x power on six complete common windows after PCA is
420.0/668.2/709.8 for LPF/finite/coherent on detector269 and
216.3/428.9/409.1 on detector402, in matched-APT (mJy/beam)^2. Extra suppression
before PCA does not imply a better result after relearning PCA and changing
support. Eight-target retained seconds are 6457.0/2624.2/3525.8/0 for
LPF/finite/coherent/rejection. Coherent's approximately 902 s advantage over
the finite notch is not qualified recovery. The finite boundary challenge
retains 14/46 central-lobe samples versus 46/46 for LPF/coherent and zero for
rejection; surviving tails or another detector's map do not recover that crossing.

Local model Learn+Consider including I/O and hashing takes 0.131–0.186 s for
the 1241 s selected record; Apply takes 0.046–0.067 s before LPF/downstream.
These are bounded local measurements, not whole-array or Unity throughput.
Runtime cost is small here, but science advantage and support compliance remain
absent. Retain LPF as the reference for continued comparison. Do not choose
epsilon or activate anything. Broader use first needs a support-compliant
model and measured benefit with a bound downstream/source/noise/covariance
reference; this result does not authorize that development or a model search.

### Exact gates, evidence and independent review

Evidence root: `/private/tmp/citlali-rtc-coherent-removal-2026-09-15`.
Owner report: `REPORT.md`; detailed bias/uncertainty/MSE, coverage, artifact and
covariance tables: `analysis-02`, `paired-error-comparison.json`; actual band
power: `residuals-01`; figures: `figures-02`. `source-completion.json` explicitly
names authoritative and superseded directories. Exact native exports reuse
the previously sealed producer; `geometry-final`, `runtime-witness-final` and
`exact-downstream` supply final compiled-source qualification for earlier pilots.

Final local AppleClang/arm64/Homebrew supplemental gates pass: 1,132 runnable
CTests, one unchanged disabled; 64 RTC Python tests; 207 baseline tests plus
137 subtests; require-all config; exact CLI. Exact replay reproduces 495
geometry, 160 runtime and 4,464 downstream artifacts. Conditional Python FIR
versus actual frozen C++ Apply differs by at most 4.88e-19 native units.
All 3,623 files in five prior seals remain unchanged. No Unity or Spack-backed
V2/production gate was performed. Three exact-source executables are saved in
`bin` and bound in `binary-binding.json`; build realization is preserved.

Preserved resolved pilots include nullable-APT handling, serialization/NumPy
warnings, premature analysis, and the original source-only control misbinding.
`controls-02/control-4` correctly retains the original frozen projection and
does not claim learning from an absent line. The first baseline run failed
because the conventional build link was absent; the exact-CLI rerun passed
and the temporary link was removed. Failed pilots are not acceptance evidence.

Independent fresh-context exact-source reviewer `/root/rtc_coherent_exact_review`
passes all three axes with recorded limitations and no actionable findings.
Report: `review-source.md`, SHA256
`2f70fa3ca0f1b2a0707087ff410f8ced7d812c6e117f37f48adc7993cd82b96b`.
Manifest `EVIDENCE_SHA256SUMS` covers 14,008 files and 23,345,961,668 bytes,
SHA256 `221ed79ddff18ea1c89b0232410c8e87902401d02b240dca5fc363e393a5dd22`.
`source-seal.json` lists excluded caches and separate review/closure receipts.
Fresh independent exact-SHA documentation review and final identities are bound
in `review-closure.md`, `closure-state.json` and `closure-binding.json`.

Live read-only GitHub verification before closure reports canonical
`0f52e1a421416eadaf7d8d99d64f537132395c0c` and experiment
`1870f99c90218dc7af83d8478eae146a09643bc2`. Canonical is unchanged and remains
an ancestor; no moving-base reconciliation is required for this same-branch
continuation. Population/coherent commits remain local pending owner push.
Unrelated owner dirt, all prior evidence and production-inactive state remain
preserved. No integration or cleanup is authorized by this closure.

Owner publication, from the Mac, using the same branch:

```sh
git -C /private/tmp/citlali-timestream-successor-rtc-notch-recovery-001 push git@github.com:toltec-astro/citlali.git refs/heads/codex/timestream-successor-rtc-notch-recovery-001:refs/heads/codex/timestream-successor-rtc-notch-recovery-001
```

## Treatment-aware population audit continuation — 2026-09-15

The owner directive supersedes further isolated-notch optimization with a
bounded detector/network/array contribution audit. Live refs verify previous
short-notch closure `1870f99c90218dc7af83d8478eae146a09643bc2` and canonical
`0f52e1a421416eadaf7d8d99d64f537132395c0c` are pushed. This continuation stays
on the existing `codex/timestream-successor-rtc-notch-recovery-001` module
branch; no new branch, canonical integration, push or production activation.
Earlier publication-pending notes below are historical.

Final implementation source `225931039d33a71b079f06942da7a912c0f39615`, tree
`d725902a9aa6c3b2a4c26bfa08c668eb6391c392`, preserves the original native
spectral/transient estimators and frozen paired Apply. RTC Learn now retains
complete observation-local population/exclusion context, and Consider
requires a complete exact-bound baseline before reporting optional recovery
contribution. Offline corpus knowledge is separately identified and never
silently imported into runtime Learn. Unavailable recovery remains distinct
from rejection's zero contribution. No automatic selector or new framework.

The prior 143-file/13-observation audit is reused: 71,734 occurrences, 70,024
producer-eligible, and 286 independent inspections. Only 152390/a2000 n11/n12
has the prior exact motion/beam/filter bindings for physical post-filter
population accounting. That denominator is 769,224.876 detector-seconds and
4,895.119 fixed APT reference-weight seconds. Rejecting occurrences selected
by the exploratory approximately 11 Hz descriptor costs 19.18% of time and
26.44% of this proxy; rejecting their measurement-window intervals costs
13.00% and 17.81%. The existing six-second footprint offers ceilings of
7.54%/10.44% for whole observations or 4.99%/6.87% on active-window support.
These are conditional support ceilings, not qualified recovery or map
sensitivity. Cross-network simultaneity and missing physical domains remain
unavailable; arrays and science products are not pooled.

A single new contribution-ranked replay, 152390/n12/ch269 at 11.0063 Hz,
retains 328.016 s versus 807.182 s for LPF/F2 and reduces actual common-support
x target-band power 90.10% to 5.1556e-15. Paired source tests reach 0.691% peak
change and 2.500% waveform error over the retained 8.5–113 arcsec/s fixed
fixtures. The original 221 arcsec/s and boundary crossings remain excluded.
Its potential reference contribution is 44.2 times the prior bright, low-value
case. A targeted pair 269/402 has 0.943 original x coherence near 11 Hz; this
supports shared structure but no physical-origin or interference inference.
The prior 54 Hz/quiet cases continue to favor no added notch.

Recommendation: targeted qualification of the recurring, high-contribution
11 Hz population is worthwhile; retain rejection as the competing treatment
for low-contribution or independently established unhealthy occurrences.
No covered recovery is claimed for broad, drifting, crowded, or unqualified
multiple-line features. The missing narrow-line residual allowance and bound
cleaned-noise/source reference remain explicit, alongside source-domain and
covariance qualification. No scientific threshold was invented.

Final supplemental local gates pass: 1,132 runnable CTests (1,133 registered,
one unchanged disabled), 19 focused native controls, 58 RTC Python tests,
207 baseline tests + 137 subtests, config require-all, exact CLI and saved driver
bindings. The new 491-detector runtime Learn witness reproduces old spectra
and original-pair identities; population construction takes 5.68 s and owns
0.51 MB of logical context. One-pair Apply adds 37–38 ms over LPF, not a
whole-array throughput claim. 18 uninjected repeat artifacts are identical.
All 3,385 files across four prior evidence seals are unchanged.

Independent exact-source review passes with recorded limitations and no
remaining findings. The 238-file source evidence seal is
`066e5a494e6d1056461c9f2394cb6c401dfb168f99f34ab1841b7f08df5f7d83`.
The owner report is
`/private/tmp/citlali-rtc-treatment-aware-audit-2026-09-15/REPORT.md`.
Complete native-cell accounting and machine-readable network/array tables
remain beside it. The runtime witness is `native-final`; the one new replay
is `campaign/valuable`; all 10 original source overlays remain exact and each
trial runs afresh on original x/r. The targeted coherence assessment uses 310
matched native-disjoint windows on the exact n12 time axis, not a common grid.

Source review: `/private/tmp/citlali-rtc-treatment-aware-audit-2026-09-15/review-source.md`.
Reviewer: `/root/rtc_treatment_population_exact_review`.
Source-review SHA256: `b880b9ced45b4634e023d24d77b0b00da951588be95dd99e0083431c61b4ec1a`.
Evidence manifest: `/private/tmp/citlali-rtc-treatment-aware-audit-2026-09-15/EVIDENCE_SHA256SUMS`;
238 files, 586,672,804 bytes. The separately written source-seal and review/closure
receipts avoid circular hash bindings. Source state, all changed path digests,
effective governance, three conformance dispositions, recorded in-progress
repairs and final exact gates are in the same evidence root.

The only reviewer-requested source repair added seal checks for reused APT
weights, prior summary/invocations and native-cell endpoints. Exact accepted
audit/short-notch/census seal digests are checked before reading those inputs;
corruption controls pass and final accounting is byte-identical to the prior
development pass. An earlier wrong build-target name and a test-helper defect
are preserved as resolved failures. No failed gate is counted as acceptance.

Documentation closure receives its own fresh independent exact-SHA review.
The candidate remains local and unintegrated. Production filtering is
inactive; no Unity/Spack V2 qualification, cleanup, CAL/PTC/MAP work or new
filter search was performed.

## Shorter finite-duration continuation — 2026-09-15

The owner requests: “Let's try some shorter duration notch filters.” This
continues the same bounded module branch from reviewed/pushed closure
`a4f3234cf4a4cb77b64a6aa5716c3632baafe4c5`; no second branch or WIP slot is
opened. Live remote verification confirms that closure and canonical
`0f52e1a421416eadaf7d8d99d64f537132395c0c` are published. No new canonical
integration, push, policy waiver or production activation is authorized by
this continuation. Earlier publication-pending notes below are historical.

Exact source: `dcf40780e582d24f7a09fc387f9e8598b63b0a5c`.
Tree: `85985c65a5820099f8247bff1762b9703cc750f3`.
Parent: `a4f3234cf4a4cb77b64a6aa5716c3632baafe4c5`.
Worktree/branch remain the existing notch-recovery worktree/branch below.
The documentation-only closure has its own external exact-SHA review receipt.

The effective governance and SCI-RTC/ADR authority map are unchanged; their
exact identities and the new preflight are under
`/private/tmp/citlali-rtc-short-notch-2026-09-15/preflight.json`.
Runtime Learn retains accepted original spectra and transient evidence.
Runtime Consider's existing trial specification gains one optional identified
finite centered notch vector; its response composes with the existing LPF.
Frozen Apply executes the two centered stages using ordered binary64 FMA,
with full summed footprints, exact native timing/row phase and paired
exclusions. Finite and IIR notches cannot be mixed in this bounded operator.
The old IIR path and default low-pass behavior are preserved; no Engine or
production route is changed. The engineering development workflow does not
substitute for these implemented runtime responsibilities.

Three explicit Hann band-stop designs use a fixed requested 0.5 Hz band and
unity DC, with no depth optimization or width search. End-to-end spans are
approximately 1, 3 and 6 s (123, 367 and 733 taps). The same 307-tap FIR/F2
follows each notch. Complete half-supports are 1.7531, 2.7525 and 4.2517 s,
meeting the retained five-second limit. These are exact finite dependencies,
not truncated IIR guards. Every valid output has complete real input support;
no padding, renormalization, gap joining or new donor values are used.

The same 152390/a2000 in-band, higher-frequency and independently selected
quiet spectral-control pairs and ten exact previous full-sky source overlays
are reused. There are 165 Apply calls across five plans/case, plus a separate
uninjected timing/determinism repeat. Original x/r, initial VAL, producer
validity, Tune/APT/array association, actual AST/telescope relation, provisional
centered readout averaging and the preserved motion/sampling bounds remain
exactly bound. The F2 raw-speed ceiling remains 123.277762 arcsec/s. Source
fixtures extend to about 113.1 arcsec/s on retained support; the original
221.4 arcsec/s maximum and boundary challenges lose their principal crossings.
Tiny remaining Airy-tail ratios are not source recovery. The fixed fixture
set does not qualify all newly retained crossings or phases near the ceiling.

| In-band treatment | Retained seconds | x target-band power removed | Largest tested peak change | Largest waveform error |
|---|---:|---:|---:|---:|
| FIR alone | 795.394 | approximately 0% | 0.041% | 0.067% |
| 1 s finite + FIR | 683.729 | 43.23% | 0.833% | 1.588% |
| 3 s finite + FIR | 495.190 | 88.82% | 0.784% | 2.401% |
| 6 s finite + FIR | 310.657 | 99.794% | 0.780% | 2.908% |
| Reject | 0 | unavailable | unavailable | unavailable |

Power measurements use identical support across the new non-rejection
trials: 123 complete native/output diagnostic windows for contaminated cases,
132 for the quiet control. Exact windows and averaging conventions are
recorded. Six-second native target-band powers are 6.586e-14 x and 5.980e-14 r,
including noise and sky. Negative sideband-subtracted descriptors indicate a
spectral hole, not negative physical contamination or validated classification.
Compared with the prior IIR, retained duration increases about 10.5-fold and
peak distortion decreases, but the old/new spectral window sets differ.

The 54 Hz case still receives about 114 dB native-band suppression from FIR
alone; extra finite notches change the folded output band by only about one
part per million. The quiet control supplies no notch rationale. Retained
native-cell duration is not qualified recovery or sensitivity: existing
screening excludes 1.384448 s per pair, prior conditional direct support has
no retained overlap, and full source-bound transient/PCA-scan admission is
still unavailable. No scan relation is invented. Mapped 1% response budgets
and a wider phase/speed domain remain unqualified.

Recommendation: retain the six-second trial as the provisional in-band
recovery candidate from this fixed comparison, keep the selected high-frequency
case on the FIR-only path, and leave the quiet control unnotched. The
six-second chain still loses 60.9% of FIR-retained support and changes the
waveform by up to 2.9%. The new scientific decision before broader line use
remains the residual narrow-line allowance and bound cleaned-noise reference;
the existing broadband 1% rule does not answer it. Existing source-preservation
policy still needs conformance evidence, not a waiver or a new threshold.
Production filtering remains inactive.

Local supplemental AppleClang 21 arm64 Release/C++20 gates pass: 1,128 runnable
CTests out of 1,129 registered (same disabled lifecycle test), 34 focused native
checks, four Python controls, 211 baseline/Python tests plus 137 subtests,
required config and exact CLI source binding. Preserved dependency revisions
and all nine patches are unchanged. No unexpected error output occurred in
completed final gates/campaign. No Spack/Unity V2 reproduction is claimed.
The temporary gate `build` symlink was removed; the source worktree was clean.

Independent fresh-context exact-source review: PASS WITH RECORDED LIMITATIONS,
no findings on any axis. Report
`/private/tmp/rtc-short-notch-review-dcf40780e.md`, SHA256
`25c3cbcc89cb6430f7473bdc2c9f2c575e2e7bbb4be325a79867ffb462b6fb48`.
Independent witnesses check 87 exact input/injection bindings, all 15 plans'
run/support/phase relations, 150 injected masks, 360 ordered-FMA values and
7,350 response-power cells. All 99 original/LPF/rejection baseline artifacts
match prior evidence, and all 84 repeated base artifacts are deterministic.
Prior 870-file evidence re-verifies intact.

Per-pair Apply costs are approximately 33–34 ms for FIR alone, 43–45 ms for
1 s, 60–61 ms for 3 s and 68–71 ms for 6 s. Plan construction is 1.7–2.0 ms.
These runs process different retained support and do not establish whole-array
or complete-pipeline throughput.

Full report/plots: `/private/tmp/citlali-rtc-short-notch-2026-09-15/REPORT.md`.
New evidence seal: 458 files, 575,244,280 bytes; `EVIDENCE_SHA256SUMS` SHA256
`c14769f45b56ed4eb1b65bee86b3cbdddb0454d686fcb82bb65518ab50abf8a5`.
Source, environment, inputs, coefficient manifests, commands, output data,
window support, source responses, timings, reviews and witness code are bound
there. Closure review/completion is sealed separately to avoid circular SHA
claims. No automatic selection, new bank, map qualification, subsequent
RTC/PTC work, new canonical admission, push or cleanup occurs.

The owner's eventual continuation push is:

```sh
git -C /private/tmp/citlali-timestream-successor-rtc-notch-recovery-001 push git@github.com:toltec-astro/citlali.git refs/heads/codex/timestream-successor-rtc-notch-recovery-001:refs/heads/codex/timestream-successor-rtc-notch-recovery-001
```


## Completed canonical admission

The owner explicitly authorized completed-assessment canonical integration,
followed by this separate bounded experiment, with production filtering inactive.
Local `codex/refactor-mainline` is now
`0f52e1a421416eadaf7d8d99d64f537132395c0c`, tree
`a95619bd7b24b032b562b6dfb4498aeba79d3d43`. Parents are canonical
`86c20b31f7300ba4063be044380b61cd0baf25eb` and accepted assessment closure
`0bcb132df284257418c6d5196b50c549b9775673`; merge base is
`b675bb64a7054f7b24403c79898965e8765cfd02`. Both histories and all topic
executable bytes are preserved; only status/ledger narratives were reconciled.

Exact integration gates passed: local CLI/safety, 1,113 runnable CTests,
required config preflight, 207 baseline tests plus 137 subtests. Fresh independent
three-axis review passed with recorded limitations and no findings. Report
`/private/tmp/rtc-assessment-integration-review-0f52e1a42.md` has SHA256
`c834a97d8751338e8a0bf3d65ba8b3d9a694d0ed9b6d575493c32f1b4dbe95aa`.
The guarded canonical update checked the expected old ref and fresh remote
authority. Completion evidence is
`/private/tmp/citlali-rtc-assessment-integration-2026-09-14/completion.json`.
The old prunable worktree registration and unrelated work remain untouched.

Fresh GitHub verification at experiment close still reports remote canonical
`86c20b31f7300ba4063be044380b61cd0baf25eb` and assessment
`0bcb132df284257418c6d5196b50c549b9775673`. The experiment branch is not remote.
Canonical publication remains owner-performed; no push or activation occurred.

## Exact experiment and ownership

Branch: `codex/timestream-successor-rtc-notch-recovery-001`.
Worktree: `/private/tmp/citlali-timestream-successor-rtc-notch-recovery-001`.
Base: integrated canonical `0f52e1a421416eadaf7d8d99d64f537132395c0c`.
Final source: `c9765cd5017ea74da791f759144bd73dbf979f48`.
Source tree: `969420246c398dfa03245e34f5b051db4feab32e`.
Parent/initial failed candidate: `09f43fe2c657274da8bfd42f4502b682602f02f3`.
This documentation-only closure has its own exact-SHA review recorded externally.

Effective governance is the accepted `06a3ade51c1b3f38887295433d913811bf25cd14`
package incorporated at `77507836325eff9f469062d5884481ea37599594`. Exact
governance digests, environment/dependencies and authority map are in the sealed
preflight. Scientific authority remains the successor router, SCI-RTC requirements
021, 030, 060–070, 072–077, 129–130 and 140, and the accepted ADR 0020–0023
optical, motion and sampling authorities. No new scientific threshold is selected.

Runtime Learn reuses accepted original spectral/transient evidence. Runtime
Consider combines the existing line-transfer assessment, exclusion plan and exact
native AST/array domain into a frozen explicit experimental plan. Runtime Apply
executes that complete plan afresh on original x/r, with exact initial VAL and
partition bindings. Paired source injections are identified diagnostic overlays
under the same plan, never new original evidence or a reason to change support.
Original x/r and the accepted D2, VAL and transient implementations are preserved.
No Engine state, production configuration route or generic filter-bank framework
is added. This implements runtime boundaries separately from following the
engineering learn/consider/apply development workflow.

## Experiment and result

The sealed existing audit supplies three 152390/a2000 pairs: network 11/channel
455 at 10.743655 Hz, network 11/channel 105 at 53.710827 Hz, and independently
selected quiet spectral control network 12/channel 193. Roughly 29 Hz adds no
distinct question here and is not tested. The quiet control is not certified
transient-free. No map product is used.

Each case compares one supplied 307-tap symmetric FIR and F2 decimation, one
explicit 0.5 Hz notch followed by that same FIR/F2, and detector-observation
rejection. Actual telescope time/pointing, accepted AST source, Tune/APT/array
association and native cadence are bound. Full-sky 50 m Airy/150 GHz injections
use the provisional centered uniform readout average and checked quadrature.
There are 99 Apply calls, including ten injections per case/plan, plus an
uninjected determinism/timing repeat. Every original, mask, row relation and
paired-plan binding is verified.

The retained v>=1 arcsec/s floor and >=4 output samples/Airy FWHM, with accepted
margins, restrict this explicit F2 trial to raw speed <=123.277762 arcsec/s.
All exclusions split input support before filtering. The mature forward/reverse
notch has a whole-run dependency footprint and approximately 10.052 s endpoint
guard including the FIR. It is explicitly **ineligible under the retained
five-second finite-half-support rule**; no guard truncation or policy exception
is introduced. The FIR alone has approximately 1.253 s half-support.

| Case | FIR/F2 retained seconds | Notch/FIR/F2 retained seconds | Rejection |
|---|---:|---:|---:|
| 11 Hz | 795.394 | 29.639 | 0 |
| 54 Hz | 795.394 | 29.639 | 0 |
| Quiet | 807.182 | 40.722 | 0 |

The 11 Hz notch suppresses measured native target-band power by about 99.9%,
but discards 96.27% of FIR-retained support and attenuates the fastest retained
injected source peak by about 1.25%, with about 3.04% waveform error. The 54 Hz
case receives approximately 114 dB native target-band suppression from the FIR
alone; adding the notch changes the measured folded output band negligibly.
The quiet control gains no justification for treatment and incurs similar
in-band source distortion. These are finite timestream measurements on exact
common support, not mapped 1% response-budget or filter-bank qualification.

Existing screening Apply excludes 1.384448 s of each 1,241.391104 s input.
Retained durations above are candidate support beyond that exclusion, not
qualified recovery or sensitivity. Prior conditional direct-transient intervals
are 0 s, 0 s and 0.008192 s, with no retained overlap. Source-bound complete
transient admission and actual PCA-scan relations are unavailable; no scan
binding or final production recovery total is fabricated. The excluded-fast and
boundary injection challenges lose their principal crossing; low-pass metrics
on tiny remaining Airy tails do not establish recovery.

Recommendation: do not activate this simple notch. The valuable in-band
contamination class warrants a separately authorized shorter-footprint recovery
attempt; the selected high-frequency case favors FIR-only suppression and the
quiet control should remain unnotched. Before broader line admission, the specific
scientific gap is the permitted residual narrow-line contribution after the
complete treatment chain and its bound cleaned-noise reference. The broadband
1% allowance does not answer it. Existing beam/motion, sampling, response and
five-second support rules remain settled. A conforming operator is also still
an engineering prerequisite. No automatic selection, parameter search, later
RTC/PTC work, map qualification or production activation follows this result.

## Gates, review, failures and evidence

Final local AppleClang 21 arm64 Release/C++20 supplemental gates pass:
1,124 runnable CTests, 1,125 registered with the existing disabled
MapFitterLifecycle.ExactProductSequence; 11 focused C++ controls; four Python
controls; 211 baseline/Python tests and 137 subtests; required config preflight;
CLI source binding. Prior dependency revisions and nine local patches remain
unchanged. No Spack/Unity V2 reproduction or operational reduction is claimed.
The repeated nine base plans reproduce 54 artifacts byte-for-byte. Measured
per-pair FIR Apply is about 33 ms; the notch's 3–3.5 ms cost reflects extensive
run rejection and is not a whole-array performance comparison.

Initial independent review required repair of missing lower-speed and sampling
support and nonconforming FIR accumulation. The repaired exact source enforces
those bounds and ordered binary64 FMA; all final data were replayed. Prior
campaigns are preserved as superseded evidence. Initial CLI-prerequisite and
mistyped safety-target command failures are retained and explicitly dispositioned;
successful final runs contain no unexpected error-level output.

Fresh independent source review passes all three axes with recorded limitations
for negative experiment evidence, without remaining repair findings. Report
`/private/tmp/rtc-notch-recovery-review-c9765cd50.md` has SHA256
`7527702a4f9499b20eefb388b68b31ca46a019a7ad10cac135f9fcf2bb3da4cd`.
The independent witness verifies input/injection hashes, actual run erosion,
motion/sampling and row phase, 90 injected-output masks and 102 selected FMA
results. Exact documentation closure review does not replace this source review.

Full report and reproducible measurements:
`/private/tmp/citlali-rtc-notch-recovery-2026-09-14/REPORT.md`.
Evidence seal: 870 files, 1,188,128,812 bytes, including preserved failed attempts
and the exact driver binary, excluding mutable build/cache trees.
`EVIDENCE_SHA256SUMS` SHA256:
`e1984208e1241e9e7ff97dafd3dc950dffa8d6b6ec5b6e1b10ef10a446464e73`.
External closure review/completion is sealed separately to avoid circular identity.
The experiment remains separate from canonical; source acceptance, any later
canonical integration and production qualification remain distinct decisions.

## Owner publication commands

After reviewing this result, publish the already authorized assessment integration:

```sh
git -C /private/tmp/citlali-integrate-timestream-rtc-assessment-001 push git@github.com:toltec-astro/citlali.git refs/heads/codex/refactor-mainline:refs/heads/codex/refactor-mainline
```

Publish the experiment branch separately; this does not activate filtering:

```sh
git -C /private/tmp/citlali-timestream-successor-rtc-notch-recovery-001 push git@github.com:toltec-astro/citlali.git refs/heads/codex/timestream-successor-rtc-notch-recovery-001:refs/heads/codex/timestream-successor-rtc-notch-recovery-001
```
