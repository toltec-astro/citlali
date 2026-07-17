# Phase 4.2 Technique And Performance Evaluation - 2026-07-16

## Executive Conclusion

The active Citlali pipeline is not in need of a wholesale numerical rewrite.
Its production techniques are recognizable, defensible approaches for this
instrument class, and the accepted point, OOF, Beammap, and science reductions
provide unusually strong whole-pipeline regression evidence. The review found
three different kinds of work:

1. two correctness/capability defects that can be repaired without changing
   accepted scientific behavior;
2. three measured Beammap costs and one science timing blind spot that justify
   focused performance work; and
3. several plausible implementation costs that must be measured before anyone
   optimizes them.

Maximum-likelihood mapmaking was the only P0 finding. It was selectable even
though Beammap silently skipped map population and the science implementation
was an experimental per-chunk least-squares solver rather than a validated
global noise-aware maximum-likelihood mapmaker. The current review tranche
keeps the implementation for future research but rejects the method at typed
preflight. It therefore cannot produce an invalid successful production run.

Required pointing and Beammap FITS metadata also contained catch-and-substitute
fallbacks. Those fallbacks are removed in this tranche so metadata write
failures propagate, consistent with the project-owner decision that required
write failures fail the reduction.

The Phase 4.2 **component census is complete**: every active component is
assigned to one of the 13 review units in
`validation/phase4_2_component_review.json`. Phase 4.2 itself is not closed.
It still needs the bounded measurements named below, a scientific validation
decision for source finding, and the deferred build-infrastructure review.

## Scope And Method

This review used:

- the active CMake compile graph and all 721 C++ headers/sources under
  `include/` and `src/`;
- the typed-config authority, lifecycle, output, and failure boundaries;
- 466 discovered C++ tests, 113 config-preflight tests, 106 baseline-tool tests,
  and 3 refactor-tool tests;
- accepted point `redu67`, OOF `redu02`, Beammap `redu06`, and science
  `redu28` stage profiles;
- the accepted product-comparison ledger in `doc/REFACTOR_STATUS.md`;
- the May code and performance audits, checked against current code rather
  than copied forward; and
- focused inspection of complexity, allocations, copies, synchronization,
  filesystem behavior, and failure handling in active paths.

Evidence labels and dispositions have the meanings fixed in
`PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md`. Inclusive or
concurrent profile scopes overlap and must not be added together as wall time.

This was not a new Unity campaign. The user will continue operational Unity
validation. Build-system modernization is intentionally deferred pending the
TolTECA developer's current compilation design.

## Verification Snapshot

| Gate | Result |
| --- | --- |
| `citlali_cli` local build before review | pass; no-op 1.06 s |
| C++ test-target local build before review | pass; no-op 1.62 s |
| CTest before review | 460/460 pass in 2.19 s |
| Config preflight | 113/113 pass in 7.57 s; all four mode kits exact; 576-leaf contract; zero boundary-audit findings |
| Baseline-tool tests | 106/106 pass |
| Refactor-tool tests | 3/3 pass |
| CTest after focused review changes | 466/466 pass in 2.25 s |
| `citlali_cli` after focused review changes | pass; one-header-change rebuild of CLI translation unit and link 60.02 s |

The test count increased by six: one maximum-likelihood capability rejection,
one analytic flux-conversion test, one detector-specific calibration test, one
detector-pointing test, and two source-finder safety regressions. The new direct
include also exposed and repaired a missing `fits_io.h` dependency in the
public `timestream.h` header.

## Real-Workload Profile Evidence

### Point `redu67`

| Stage | Seconds | Interpretation |
| --- | ---: | --- |
| reduction total | 110.154 | accepted exact point run |
| observation TOD pipeline | 62.180 | dominant processing scope |
| outputs and accumulation | 43.776 | includes overlapping ordered writers |
| Wiener map filter | 35.498 | known optional post-processing cost |
| RTC TOD writer scopes | 39.414 | overlapping writer durations, not additive wall time |
| PTC diagnostic writer scopes | 32.711 | overlapping writer durations |
| map output | 7.409 | bounded |
| all map diagnostics | 0.946 | not a material point bottleneck |

The performance wrapper measured 131.08 external wall seconds, 110.477 Citlali
log seconds, and 908,316 KiB peak RSS. Point products are exact against the
accepted predecessor.

### OOF `redu02`

| Stage | Seconds | Interpretation |
| --- | ---: | --- |
| reduction total | 40.514 | accepted OOF run |
| observation TOD pipelines | 32.334 | dominant |
| RTC diagnostic writer scopes | 11.143 | overlapping across observations |
| PTC diagnostic writer scopes | 6.393 | overlapping across observations |
| map output | 2.157 | bounded |
| all map diagnostics | 0.316 | negligible |

OOF shares pointing orchestration and Gaussian fitting and is exact against its
accepted predecessor, with tight accepted tolerance against OG.

### Beammap `redu06`

| Stage | Seconds | Fraction of TOD scope | Interpretation |
| --- | ---: | ---: | --- |
| observation TOD pipeline | 4,013.946 | 100% | dominant run scope |
| PTC cleaning, three passes | 1,565.923 | 39.0% | largest measured compute target |
| map population | 1,250.498 | 31.2% | second measured compute target |
| PTC diagnostic sidecar | 344.554 | 8.6% | measured compute/I/O target |
| raw map output | 160.046 | 4.0% | required output cost |
| map fitting | 108.567 | 2.7% | expected 5,234-map fitting cost |
| source-aware RTC rerun | 87.424 | 2.2% | intentional second pass |
| split FITS writing | 82.615 | 2.1% | required product cost |
| normalization | 52.382 | 1.3% | bounded |
| map-buffer reset | 21.395 | 0.5% | copy/reset suspicion not established here |

The sidecar implementation opens and closes one shared NetCDF file per scan
and holds the global NetCDF mutex while it computes detector statistics,
window summaries, and detector pointing as well as while it writes. That makes
the 344.554 s an actionable implementation finding, not merely a storage
observation. Beammap products remain exact across the accepted checkpoint.

### Science `redu28`

| Stage | Seconds | Interpretation |
| --- | ---: | --- |
| cumulative four-iteration reduction | 2,881.526 | saved fruit-loop sequence |
| observation TOD pipelines | 2,799.461 | 97.2% of cumulative reduction scope |
| observation input preparation | 24.021 | bounded |
| observation setup | 23.230 | bounded |
| all map output | 18.175 | bounded |
| all coadd outputs | 16.144 | bounded |
| all map diagnostics | 8.023 | negligible relative to TOD |
| all map filtering | 4.266 | negligible in this configuration |

The current science profile does not split the TOD scope into RTC, PTC, and
mapmaking. Optimizing any one of them from this evidence would be guesswork.
Science `redu28` through `redu31` has accepted exact low-level config and
near-machine-precision scientific-product equivalence.

## Review Records

### 1. CLI, Session, Selection, And Failure Reporting

- **Purpose and boundary:** select one reduction mode, establish a run-owned
  session, classify exceptions, and return one structured result to CLI policy.
- **Technique:** explicit `ReductionSession`, canonical error categories,
  output-root lease, and CLI-only process exits.
- **Assumptions/failure policy:** one active session per process; sequential
  reuse is supported; nested use is rejected; required failures reach the CLI.
- **Performance:** setup is negligible. The best-effort Linux crash handler is
  diagnostic only.
- **Evidence:** sequential success/failure tests, session failure-boundary
  tests, six direct exits all confined to CLI or inactive legacy executables.
- **Disposition:** **Retain**. Improve the fatal signal handler only if this
  diagnostic code is next touched; it is not fully async-signal-safe.

### 2. Config, Plans, Adapters, Validation, And Provenance

- **Purpose and identity:** preserve requested policy, derive effective policy,
  and record realized state without allowing YAML to become runtime state.
- **Technique:** typed domain objects, one-way legacy adapters, immutable
  requested/effective separation, lifecycle cardinality, atomic provenance.
- **Assumptions:** TolTECA owns ordered YAML merging; Citlali owns validation of
  the resulting low-level config; units and sentinels are domain-specific.
- **Performance:** config work is outside hot loops and is negligible.
- **Evidence:** 15 authority domains, 576 classified leaves, 131 audited config
  read sites, exact four-mode kit expansion, zero remaining boundary findings.
- **Disposition:** **Retain**. Phase 4.1 human authoring remains a separate
  operator-interface trial.

### 3. Raw Input, KIDs, Calibration, Telescope, And Astrometry

- **Purpose and identity:** convert detector data through the external KIDs
  library, align telescope/calibration streams, establish detector identity,
  units, coordinate frames, and per-observation pointing state.
- **Technique:** checked KIDs input boundary; explicit detector/network/array
  inventories; per-detector flux factors; Rayleigh-Jeans `uK`; tangent-plane
  pointing in radians with configured offsets in arcseconds.
- **Scientific assessment:** these are established transformations and current
  behavior matches OG. The owner has confirmed `xs`, `rs`, `is`, and `qs` as
  supported KIDs representations and the pointing-offset policy.
- **Numerical/failure behavior:** non-finite detector inputs and mismatched
  lengths fail; missing optional telescope fields retain explicit fallbacks.
- **Performance:** input preparation is small in all profiles. Detector
  pointing is recomputed in several hot mapmaking paths and needs measurement.
- **Evidence:** mode-level exact validation plus new analytic unit conversion,
  detector-specific calibration, and pointing-rotation tests.
- **Disposition:** **Retain**; **Measure** repeated pointing only inside the
  mapmaking hotspot; add non-`xs` fixture coverage when representative inputs
  are available.

### 4. RTC Processing

- **Purpose:** flag and despike raw samples, filter, downsample, calibrate,
  correct extinction, and produce RTC diagnostics/TOD.
- **Technique:** established deterministic despiking/filtering pipeline with
  explicit source protection, zero-phase filtering policy, edge guards, and
  one-way typed configuration.
- **Scientific assessment:** retained OG behavior is appropriate. Filter and
  despike choices have explicit sample-rate and Nyquist validation.
- **Numerical/failure behavior:** invalid rates, non-finite input, impossible
  filters, and output failures are fatal; flags preserve missing-data policy.
- **Performance:** not isolated in science; source-aware Beammap RTC rerun is
  87.424 s. Current point diagnostics overlap with ordered writers.
- **Evidence:** exact active-mode TOD products, deterministic runs, direct
  notch-filter tests, extensive config/edge-guard tests.
- **Disposition:** **Retain**; add RTC/PTC/mapmaking nested science profiles
  before considering implementation changes.

### 5. PTC Cleaning, Weighting, Diagnostics, And Outputs

- **Purpose:** remove correlated atmospheric/instrumental modes, establish
  detector weights, apply second-pass rejection, and publish diagnostics.
- **Technique:** scan/group covariance and PCA with Eigen/Spectra solvers;
  configured standard PCA is the production baseline, while adaptive selectors
  are explicitly guarded expert choices.
- **Scientific assessment:** PCA removal of correlated bolometer modes is
  established practice, but removed-mode depth changes the transfer function.
  Production behavior should remain fixed unless injection/transfer-function
  evidence supports a scientific change.
- **Numerical behavior:** active-detector covariance avoids flagged rows;
  eigensolver failure propagates or uses an explicitly logged configured
  fallback in guarded selector paths.
- **Cost:** dense covariance is roughly quadratic in detector count and
  eigendecomposition cubic in group size. Beammap PTC cleaning is 1,565.923 s.
- **I/O:** Beammap diagnostic sidecar is 344.554 s and serializes substantial
  CPU work under the NetCDF mutex.
- **Disposition:** **Retain** the scientific algorithm; **Measure** covariance,
  eigensolve, reconstruction, and rejection separately; **Improve** sidecar
  lock scope and file lifetime with exact-product gates.

### 6. Map Geometry, Naive/JINC Mapmaking, Kernels, And Weights

- **Purpose:** project flagged weighted detector samples onto observation and
  coadd maps with explicit coordinate frame, grouping, kernel, weight, coverage,
  and noise identities.
- **Technique:** naive weighted binning for ordinary science/pointing and JINC
  footprint accumulation for detector Beammap products. These are appropriate
  production techniques for the validated use cases.
- **Complexity:** naive work is linear in valid samples but uses per-call sparse
  triplets and a serialized dense merge. JINC adds kernel-footprint work per
  sample and subpixel variant. Both recompute detector pointing.
- **Evidence:** exact point/science/Beammap products; Beammap population is
  1,250.498 s. No faithful isolated kernel benchmark yet exists.
- **Maximum likelihood:** the retained implementation solves independent
  per-chunk least-squares systems, ignores the noise model and several output
  contracts, and is not the global maximum-likelihood estimator. It is now
  rejected at production preflight.
- **Disposition:** **Retain** naive/JINC; **Measure** JINC footprint and pointing
  costs first; **Clarify/Retire** the current ML implementation unless a future
  project supplies an approved algorithm, product contract, and reference
  validation.

### 7. Fruit-Loop Feedback And Iteration Control

- **Purpose:** iteratively estimate source structure, feed it back into TOD
  processing, and retain requested iteration products.
- **Technique:** explicit iteration lifecycle, prior-map identity validation,
  convergence/limit policy, and typed learning provenance.
- **Scientific assessment:** iterative recovery around PCA filtering is an
  established strategy. Exact cut levels and stopping policy remain owner
  choices because they alter recovered spatial scales and flux.
- **Performance:** cost intentionally multiplies observation processing; the
  accepted science four-iteration profile reflects that.
- **Failure behavior:** missing/wrong map inputs fail; concurrent runs sharing
  one output root are rejected. Flat `reduNN` saved-iteration identity remains
  recorded post-refactor debt.
- **Disposition:** **Retain**; **Clarify** convergence rationale per science
  profile; defer nested run/iteration identity redesign as already decided.

### 8. Pointing And OOF Orchestration And Fitting

- **Purpose:** derive pointing offsets and Gaussian source parameters; reuse the
  pointing path for OOF products.
- **Technique:** deterministic Ceres nonlinear least-squares Gaussian fitting,
  explicit fit cardinality, and owner-confirmed astrometry application:
  interpolate two bracketing pointings, hold one pointing constant, or use
  config offsets when none exist.
- **Numerical behavior:** invalid/duplicate fit results fail; fits remain
  sequential where determinism and solver stability matter.
- **Performance:** map fitting is not material in point/OOF profiles.
- **Evidence:** exact point and OOF products, direct pointing test, lifecycle
  and fit-cardinality tests.
- **Disposition:** **Retain**.

### 9. Beammap Iteration, Priors, Fitting, Flagging, And Products

- **Purpose:** produce per-detector beam maps, fit beam parameters, update APT
  state, reject bad fits, and publish split products and diagnostics.
- **Technique:** phased iterative JINC mapmaking, deterministic Gaussian fitting,
  typed prior/flagging policy, source-aware RTC rerun, and explicit product
  cardinality. TolPROJ, not Citlali, owns calibrator identity and flux estimate.
- **Scientific assessment:** accepted exact Beammap tables, TOD, diagnostic
  NetCDF, and split FITS validate the current technique. The code must not
  broaden priors or flagging without a named scientific change.
- **Performance:** PTC cleaning, map population, and sidecar I/O account for the
  actionable majority. Full `ptcs0`/`calib_scans0` copies remain a derived
  memory/bandwidth concern, not a demonstrated wall-time bottleneck.
- **Disposition:** **Retain** science; **Measure/Improve** the three observed
  hotspots in that order; measure copy/RSS cost before redesigning state.

### 10. Source Finding, Coadd, Filtering, Wiener, And Noise Products

- **Purpose:** combine observations, derive empirical noise products, filter
  maps, fit/detect sources, and publish final products.
- **Technique:** weighted coaddition, jackknife noise realizations, optimized
  FFTW/OpenMP Wiener filtering, local-extremum source detection, and Gaussian
  source fitting.
- **Scientific assessment:** coadd/noise/Wiener behavior is validated and
  appropriate for current profiles. Source finding has safety coverage but not
  an adequate injection-recovery characterization; the owner has warned that it
  has not been extensively exercised.
- **Performance:** Wiener costs 35.498 s in the full point case; diagnostics are
  negligible in all reviewed profiles. Full noise cubes scale as rows x columns
  x maps x realizations and need peak-RSS evidence on a naturally noise-heavy
  run.
- **Disposition:** **Retain** coadd, Wiener, and diagnostics; **Measure** noise
  memory; treat source-finder science results as experimental until synthetic
  positive, negative, edge, crowded, completeness, and false-positive tests
  establish an accepted contract.

### 11. FITS, NetCDF, ECSV, Manifests, And Publication

- **Purpose:** publish required scientific arrays, tables, metadata, config
  identity, provenance, and optional diagnostics.
- **Technique:** checked FITS/NetCDF/ECSV wrappers, atomic small sidecars,
  required-output cardinality, ordered writers, compression/chunking helpers.
- **Failure policy:** required file/image/table/metadata failures propagate.
  Optional profiling remains deliberately non-fatal.
- **Performance:** ordinary output is bounded. Beammap sidecar and split FITS
  are measured costs; repeated NetCDF open/close and over-broad lock scope are
  the first I/O improvements to evaluate.
- **Evidence:** required-output failure tests, product contracts, strict mode
  comparisons, and removal of the remaining metadata fallback sites.
- **Disposition:** **Retain** contract; **Improve** Beammap sidecar with exact
  byte/schema/value comparison and before/after timing.

### 12. Concurrency, Memory, Logging, Profiling, And Libraries

- **Purpose:** use cluster cores deterministically, bound shared-library access,
  observe stage cost, and preserve diagnosable failures.
- **Technique:** GRPPI scan/detector boundaries, OpenMP/FFTW where established,
  mutex-protected map merges and NetCDF, ordered output writers, run-owned stage
  profiler, Eigen/Spectra/Ceres/FFTW/netCDF/CCfits libraries.
- **Determinism:** repeated runs and exact products support the current parallel
  boundaries. No new virtual dispatch or allocation abstraction has been added
  to hot loops.
- **Memory:** point peak RSS is measured; Beammap peak RSS is not. Noise cubes
  and iterative state copies are the main derived risks.
- **Profiling limitation:** nested and concurrent stage times are inclusive and
  overlapping. Stage totals are useful for ranking but not additive wall time.
- **Build footprint:** 700 public headers and 96,740 C++ lines remain, with most
  runtime implementation header-defined. Recompiling the CLI translation unit
  after a header change took 60.02 s locally. Build-boundary work remains
  explicitly deferred.
- **Disposition:** **Retain** current concurrency; **Measure** scaling/RSS only
  on representative runs; **Clarify** profile semantics; defer build work.

### 13. Tests, Validation Contracts, And Diagnostics

- **Purpose:** detect behavioral regressions at the smallest useful level and
  establish mode-level scientific/output equivalence.
- **Technique:** fast C++ contract tests, Python config/tool tests, strict
  product comparison, accepted-run ledger, stage profiles, and operational
  diagnostics.
- **Strength:** mode-level evidence is excellent and CTest is fast enough for
  every local edit cycle.
- **Gap:** test count previously overstated numerical coverage because most
  tests exercise config/lifecycle orchestration. Direct calibration, pointing,
  and source-finder tests are added now; naive/JINC accumulation, PCA cleaning,
  despike reproducibility, noise statistics, and positive source-recovery
  accuracy still lack focused fixtures.
- **Disposition:** **Improve** selectively. Add a direct test or faithful
  benchmark only when it protects an accepted technique or an admitted
  performance project; do not build a generic framework without a use case.

## Finding Register

| ID | Priority | Evidence | Finding | Disposition / owner / gate |
| --- | --- | --- | --- | --- |
| 42-001 | P0 | Observed + Derived | `maximum_likelihood` was selectable; Beammap skipped population and science used an incomplete per-chunk least-squares implementation. | **Resolved locally:** reject at typed preflight. Engineering owns any future reintroduction; require global noise-model algorithm, product contract, tests, and reference validation. |
| 42-002 | P1 | Owner decision + Observed | Source finding is selectable but lacks adequate scientific injection/recovery characterization. | **Owned block:** scientific owner + engineering. Keep experimental; no production-accuracy claim until synthetic recovery/false-positive suite and one representative mode validation pass. |
| 42-003 | P1 | Observed code path | Required pointing/Beammap FITS keywords could be silently omitted or replaced with zero after a write exception. | **Resolved locally:** exceptions now propagate. Validate next natural point and Beammap runs. |
| 42-004 | P2 | Observed | Beammap PTC cleaning consumes 1,565.923 s, 39.0% of TOD scope. | **Measure:** add nested covariance/eigensolve/reconstruction/rejection scopes; optimize only the measured dominant operation; preserve exact products. |
| 42-005 | P2 | Observed | Beammap map population consumes 1,250.498 s, 31.2% of TOD scope. | **Measure:** faithful JINC benchmark and nested pointing/kernel/merge scopes; compare thread scaling before code change. |
| 42-006 | P2 | Observed + Derived | Beammap PTC diagnostic sidecar consumes 344.554 s and holds global NetCDF lock during CPU diagnostics plus per-scan open/write/close. | **Improve:** compute records before lock, then assess ordered persistent writer/batching. Gate on exact NetCDF schema/values and wall time. |
| 42-007 | P2 | Observed | Science spends 2,799.461 s in TOD but cannot attribute RTC vs PTC vs mapmaking. | **Measure:** add three nested production scopes on next naturally required science run. No optimization before attribution. |
| 42-008 | P2 | Derived | Noise products allocate full realization cubes and repeatedly traverse/copy them. | **Measure:** wrapper peak RSS and output volume at two `n_noise` values when a noise-heavy validation is naturally required. Trigger improvement if memory approaches cluster/operator limits or scales unexpectedly. |
| 42-009 | P2 | Derived | Naive mapmaking builds sparse triplet vectors and serially merges dense maps. | **Measure only if 42-007 identifies mapmaking:** benchmark representative scan against a direct/tiled accumulator; require exact accumulation within accepted floating-order policy. |
| 42-010 | P2 | Derived | Detector pointing is recomputed for geometry and mapmaking. | **Measure with 42-005/007:** cache only if pointing is material and bounded cache memory is demonstrated. |
| 42-011 | P2 | Derived | Beammap copies full PTC/calibration scan state between iterations. | **Measure:** add copy scope and Beammap peak RSS before changing ownership. |
| 42-012 | P2 | Observed | Direct numerical fixtures remain sparse for PCA, naive/JINC, despike, and noise statistics. | **Improve incrementally:** add focused tests alongside accepted changes; new calibration/pointing/source safety tests partially close this gap. |
| 42-013 | P2 | Observed | Header-heavy architecture causes broad recompilation; CLI TU rebuild took 60.02 s locally. | **Deferred by owner:** revisit only after TolTECA build approach is known. Preserve the current working local/Unity flow. |
| 42-014 | P3 | Derived | Best-effort fatal-signal backtrace uses APIs that are not all async-signal-safe. | **Clarify/Improve when touched:** use `sigaction` and a minimal write-only crash path; never move this into library policy. |
| 42-015 | P3 | Observed | Stage totals can be misread as additive despite nested/concurrent scopes. | **Clarify:** document inclusive/overlap semantics in profiler output/tooling; use external wall time as total. |
| 42-016 | P3 | Derived | Rejected experimental ML implementation still adds header/maintenance weight. | **Retire or isolate** after confirming no research consumer; not urgent while the capability gate is in place. |

## Old Audit Dispositions

| May performance item | Current decision |
| --- | --- |
| P-001 naive triplets/merge | Still plausible; demoted from presumed bottleneck to **Measure after science attribution**. |
| P-002 noise cubes | Still plausible memory risk; **Measure peak RSS**, no rewrite yet. |
| P-003 unconditional diagnostics | Current profiles reject the bottleneck hypothesis; **Retain**. |
| P-004 JINC footprint | Now an **Observed Beammap hotspot** at the enclosing population scope; instrument internally before optimizing. |
| P-005 repeated pointing | Still derived; measure inside JINC/science scopes. |
| P-006 global NetCDF lock | **Observed** for Beammap sidecar; lock scope is broader than I/O and is a bounded improvement project. |
| P-007 PCA cleaning | **Observed largest Beammap compute stage**; retain technique, instrument internals. |
| P-008 Beammap state copies | Still derived; measure copy time/RSS. |
| P-009 Wiener allocation/plans | Imported optimized implementation is validated; current full point cost is known and acceptable. **Retain** until new evidence. |
| P-010 no benchmarks | Still true for hot kernels; add only the two evidence-driven JINC/PCA cases, not a broad benchmark program. |

All high-confidence May correctness findings F-001 through F-009 are repaired
in current code. This tranche adds direct tests for F-001/F-002, F-003/F-004,
and detector-pointing behavior. Mode validation remains the acceptance evidence
for the broader pipeline interactions.

## Scientific And Library Context

- PCA removal of correlated bolometer modes is established, but its transfer
  function and removed-mode depth are science choices: [Downes et al. 2012](https://arxiv.org/abs/1103.3072)
  and [Scott et al. 2008](https://academic.oup.com/mnras/article/385/4/2225/1036374).
- A genuine maximum-likelihood mapmaker solves a global TOD/noise-weighted map
  problem rather than independent chunk fits: [Stompor et al. 2002](https://doi.org/10.1103/PhysRevD.65.022003).
- Eigen documents `LeastSquaresConjugateGradient` as a solver for `Ax=b`; using
  that library correctly does not by itself make a system an astronomical
  maximum-likelihood mapmaker: [Eigen reference](https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1LeastSquaresConjugateGradient.html).
- Ceres is an appropriate mature nonlinear least-squares library for Gaussian
  beam/point-source fitting: [Ceres Solver](https://ceres-solver.org/).
- Wiener filtering is an established optimal linear method under its assumed
  signal/noise model: [Bunn et al. 1994](https://arxiv.org/abs/astro-ph/9404007).

These references support technique classification, not proof that Citlali's
specific parameters are scientifically optimal. Those parameters remain tied
to injection tests, calibrator data, and project-owner decisions.

## Bounded Execution Queue

1. Unity-compile the two correctness repairs and run one ordinary point
   reduction. This validates required pointing keywords and the unchanged
   supported mapmaking path.
2. On the next naturally required Beammap, validate required detector metadata
   and collect a wrapper record with peak RSS. Do not schedule a Beammap solely
   for this census.
3. Add PTC-cleaning substage scopes and move Beammap sidecar CPU preparation
   outside the NetCDF lock. Keep these as separate commits so performance and
   product gates are attributable.
4. Add map-population substage scopes and a faithful JINC microbenchmark. Decide
   whether pointing, kernel updates, or merge synchronization is actually the
   target.
5. Add RTC/PTC/mapmaking scopes to science and use the next required science run
   for attribution.
6. Define the source-finder experimental capability statement and synthetic
   injection-recovery matrix with the scientific owner.
7. Revisit build/compiled-boundary work only after the TolTECA developer's
   current build model is available.

This queue is intentionally finite. It does not authorize open-ended header
splitting, a new mapmaker, a cleaner rewrite, or a generic benchmark framework.

## Phase 4.2 Exit Assessment

| Exit condition | Status |
| --- | --- |
| Every active component assigned | **Pass**; machine-readable census added |
| No unowned P0/P1 | **Pass locally**; P0 and metadata P1 repaired, source-finder P1 has named owners and explicit capability block |
| Suspicions measured or trigger-deferred | **Pass for census**; each derived concern has a trigger |
| Dominant runtime contributors have dispositions | **Pass** for observed point/OOF/Beammap/science profiles |
| Dominant memory contributors have evidence | **Open**; Beammap/noise-heavy peak RSS still required |
| Intentional science changes have successor evidence | **Not applicable in this tranche**; no accepted scientific algorithm changed |
| Finite P2/P3 backlog | **Pass**; register contains 13 P2/P3 items with gates |
| External reviewer can reproduce decisions | **Pass for current evidence**, subject to the named future measurements |

The review census is therefore complete, while Phase 4.2 is approximately
**80% complete**. The remaining work is measurement and bounded remediation,
not another broad code-reading pass.
