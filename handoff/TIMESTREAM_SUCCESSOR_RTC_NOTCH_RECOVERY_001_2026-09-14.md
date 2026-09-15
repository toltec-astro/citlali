# TIMESTREAM-SUCCESSOR-RTC-NOTCH-RECOVERY-001

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
