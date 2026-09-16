# Bounded early RTC common-mode health diagnostic

Owner: Grant Wilson. Tier 2. Implementation/replay authorized by the complete
2026-09-16 common-mode directive, preserved in
`/private/tmp/citlali-rtc-common-mode-health-2026-09-16/OWNER_DIRECTIVE.md`.
Directive SHA-256: `c2e492f13db15c9193b9a26019c98376399d739af72325cef27a859da153fc8a`.
This is the immediate task; residual-line outcome policy is deferred.

## Preflight

Clean literal base `46c3de155bdf36c9bf01e59a4db0cccdf3626acc`, tree
`e5e4b7e74cc39961d024d3c6d368ff051b367b2a`, existing worktree
`/private/tmp/citlali-timestream-successor-rtc-notch-recovery-001` and branch
`codex/timestream-successor-rtc-notch-recovery-001`. Live refs verified read-only:
canonical `0f52e1a421416eadaf7d8d99d64f537132395c0c`, published topic
`becccce85816379e578ac1a325e8c9fe156defbe`. No ancestry reconciliation in this task.
Effective governance acceptance `06a3ade51c1b3f38887295433d913811bf25cd14`,
incorporation `77507836325eff9f469062d5884481ea37599594`; the three effective
engineering/successor/review texts, AGENTS, toltec-context, current status,
SCI-RTC r0.12, VAL initial use and September 16 finite-filter exception were
checked. The latter does not widen adaptive diagnostic support.

## Fixed experiment choices

The initial development replay used 152390 / network 12 / a2000 / SCIENCE
NGC4449 as a 491-detector proxy. The owner clarification below replaces it
with the final full 630-detector network 0 case. Reference eligibility
additionally requires Tune and APT flag=flag2=0. Missing calibration does not
bar raw fits. The twelve-column network12 control is only an Apply-parity
fixture, never a substitute for the full-network reference.

Runtime Learn follows the existing initial spike screen, before detailed event
fitting. Its exact initial VAL, originals, AST motion and existing processing
generation are immutable parents. Later VAL generations require a separate
explicit use binding; this increment never silently interprets them.
For each existing processing interval / physical run, contributors are fixed:
exclude a contributor with an unresolved candidate edge, failed paired noise
screen or missing original pair on otherwise speed-eligible support. These are
diagnostic population reasons, not new scientific flags. Center each retained
contributor by its median on that support. Compute one across-detector median
at each native timestamp. No interpolation across missing speed/gap support.
Target fits split at speed/validity boundaries and every candidate edge.
Minimum three contributors and 64 contiguous paired samples; inadequate support
or constant reference is inconclusive. These are numerical availability guards,
not detector-health thresholds. Reference membership changes at interval edges
are recorded; models do not fit across them.

Free signed affine Huber fits use tuning 1.345, MAD scale 1.4826 and bounded
IRLS, reusing existing robust-fit conventions. No polynomial or gain evolution
model. Formal uncertainty unavailable; empirical segment variation is reported.
Compare flxscale*gain to its signed interval peer median only after fitting.
Bounded self-excluded reference checks cover conspicuous and ordinary targets.
Reference/residual products remain diagnostics; shared structure stays visible.

Consider joins the new evidence to original spike/noise, native spectral and
calibration/validity evidence, reporting ranked inspection candidates without
rejection thresholds. Plan/Apply consume no new diagnostic output. Full-network
replay stops before expensive treatment selection; controlled and twelve-column
parity checks demonstrate the unchanged science path. Report timing, support,
membership, plots and prospective costs only. No mask/weight/calibration change,
donors, PCA, filtering activation, FRUIT, new branch, integration or push.

## Validation and stop

Controlled sign/gain/offset, noise/instability, shared disturbance, inadequate
and changing support, self-exclusion, identity and unchanged science tests;
public header isolation, focused and repository gates. Local AppleClang/Homebrew
environment is supplemental, not Unity/Spack qualification. Commit a coherent
candidate, bind exact source/tree and replay evidence, obtain fresh independent
scientific, architecture and repository/evidence review. Stop after this bounded
deliverable. Owner subsequently decides whether any exclusion rule is justified.

## Owner clarification and bounded reassessment

After the development proxy replay, the owner identifies Joey's negative-gain
example as **the same observation (152390), network 0**. The final real-data
experiment therefore uses that full 630-detector a1100 network. Network 12's
preliminary result remains explicitly a development proxy; it cannot confirm
or refute network 0. The existing twelve-channel network 12 cohort remains
only a science-output parity check. No cross-network reference or pooled band
is introduced. Network 0 can enter the existing caller only in diagnostic-only
mode; its raw/Tune/APT/AST and own band/cadence support are checked anew.
This is a within-scope input correction under the original directive's stated
preference, not an added multi-observation campaign or changed fit policy.

## Final implementation and observational evidence

Runtime source `9a47dd8d30c844ac2838827c438486f3a53a58d7`, tree
`e784b9e437a70071755383f028eea9470f9d61fe`, parent
`987f9cc3d415271968c10ec681b89b79a9cf5bee`. The parent contains the bounded
implementation; the final repair also selects conspicuous signed raw response
when calibration is unavailable and breaks reference plots at processing
boundaries. Reference/fits/population/processing products are byte-identical
between these candidates. This handoff's closure changes documentation only;
its own exact SHA and source-equivalence proof are bound externally.

The final 152390/network0 replay has 630 diagnostic targets, 548 initially
eligible reference contributors, 622 targets with fits, and 124 existing
processing intervals. Contributors per interval are min/median/max 251/534/548.
The original adaptive speed support remains 1--67.9841 arcsec/s, with 124,011
available reference positions of 124,207 speed-eligible positions. Missing
reference support is inconclusive; it is not a population rejection. Missing
fit rows require consulting the exported interval support.

Self-excluded inspections include channels 221, 445, 236 and 426 plus ordinary
controls 301 and 492. Channel426 has sustained negative-response episodes
(75/124 interval median gains negative), positive episodes, prior APT flags
and zero flxscale; it is not a newly discovered stable inversion among fully
qualified detectors. The exact channel Joey inspected remains unidentified.
All eight negative median raw-response targets are outside the good-flag
reference population: three explicitly flagged, five missing APT/calibration
authority. No calibrated good-population target has negative median calibrated
relative response. Channel221 adds weak/unstable response context; channel445
adds a calibration-consistency question (relative response 6.21). Channel236's
large oscillatory residual is already explained by the original 58.53 Hz
spectral evidence. These are descriptive findings, not new scientific flags.

The good reference population contains 556,712.165376 adaptive-eligible
detector-seconds before detailed transient treatment. Prospective whole-record
exclusion of one selected good-population target would affect 1,015.898112
detector-seconds, 0.18248%; this is neither sensitivity-weighted nor a net loss
beyond transient/filter exclusions. No exclusion is made. Recommend retaining
diagnostic-only status; the evidence does not establish a new automatic cut.

Learn takes 8.817 s, Consider 0.303 s, six self-exclusion checks 2.793 s, and
export 1.193 s locally. Main owned logical evidence is 43,978,862 bytes plus
7,273,776 bytes of self-check reference planes and small fit/scratch storage;
shared original parents and allocator overhead are excluded. Cumulative child
peak RSS is not incremental diagnostic memory. No production throughput claim.

## Completion gates and preservation

- Ten focused controls pass (nine common-mode, one frozen-plan parity), with
  public-header isolation. At exact final source, 1,178/1,178 runnable CTests
  pass; the existing MapFitterLifecycle.ExactProductSequence remains disabled.
- Baseline Python 207, RTC Python 67, full required config preflight, helper
  lint/compilation and whitespace checks pass. The final selector repair does
  not change the Python/config gate surfaces. Environment is local AppleClang
  21/Homebrew/C++20 supplemental, not Unity/Spack V2 qualification.
- Exact-source enabled/disabled real-data controls preserve 370 files bitwise:
  numerical data/spectra, validity/causes/support, scheduled native rows, and
  event/support/processing decisions. Both complete frozen-plan arm receipts
  agree after omitting only three measured performance fields. Original and
  admitted x/r parents remain unchanged.
- Final replay receipts bind the exact candidate. The native fixture exporter
  is unchanged existing source with an earlier revision stamp; its outputs
  are individually hash-bound and are not attributed to the final executable.
- No unexpected error-level output in successful final gates. Existing raw
  exporter kind-var warning is retained; no authority is inferred from it.

External evidence root:
`/private/tmp/citlali-rtc-common-mode-health-2026-09-16`.
`REPORT.md` contains the compact table, inspection plots and bounded conclusion;
`SOURCE.json` contains exact source/tree, changed-path digests and environment;
`parity-9a47dd8d3.json` binds the unchanged science path. The 848-file final
`EVIDENCE_SHA256SUMS` digest is
`2618755f63a6a588f0eb2902f7d19e1dc36f71ed745e1756dc31501311d854b4`.
Review/closure records are sealed separately. Earlier proxies are retained as
development evidence and do not substitute for this final network0 result.

Fresh independent exact-SHA review of runtime candidate
`9a47dd8d30c844ac2838827c438486f3a53a58d7`: **pass with recorded limitations**;
no remaining material findings. Scientific/behavioral conformance passes with
the reported reference/uncertainty limits; architecture/ownership passes;
repository/evidence hygiene passes with local-environment and later-closure
limits. The reviewer independently verifies ten focused tests, all 370 parity
files, frozen-plan receipts, 848 sealed artifacts and 630 native payload hashes.
`INDEPENDENT_REVIEW.md` SHA-256:
`dc355810da0a9fd1e843b50466b381b9b335884bfb4ef10ee2644f1d5563ae67`.
This verdict excludes the subsequent documentation closure, whose exact SHA,
clean state, unchanged-runtime proof and independent disposition are recorded
in external `CLOSURE.json` / `CLOSURE_REVIEW.md`.

Runtime Learn owns original/VAL/reference/support/fit/residual evidence;
Consider owns descriptive joins and inspection selection. Plan/Apply have no
new consumer, rejection, calibration or subtraction authority. No canonical
integration, push, cleanup, production activation, FRUIT or bright-source
qualification is performed. The next owner decision is whether to retain the
diagnostic only or separately authorize a scoped health-policy proposal.
