# Configured POINT parallel execution — plan and work order, 2026-11-04

Grant authorized a holistic parallel-processing plan followed by implementation,
explicitly favoring additional memory where it reduces pipeline time. Grant alone
executes every build, test, syntax check, probe, benchmark, reduction and validator.
No push, runtime replacement, integration or Unity access is included.

Grant clarified that OG Citlali supplies the standard of design care, not an
implementation to duplicate. Modern mechanisms may be used where appropriate.

## Preflight and recovery

Tier 2 successor spine increment. Base: clean accepted mainline
09794feddde610e4332b0924fabf235e2d10005e. Branch codex/point-parallel-20261104,
isolated worktree under citlali-point-parallel-20261104/source. The hashing
candidate e3f6921687237d4c395500483fb9d5373ec1b03a is parked intact as unaccepted
evidence, not a dependency. Its M0 outlier remains unexplained. No concurrent
implementation of that module is selected.

Read AGENTS, Engineering, Successor and Review/Conformance governance; authority
digests/acceptance are recorded in canonical INTEGRATION_LEDGER. CURRENT_STATE,
REFACTOR_STATUS, ARCHITECTURE, SCIENTIFIC_CONVENTIONS, APPLICATION_BASELINES,
RETAINED_DEBT and the WP7/scientific authority records retain their boundaries.
Existing optical/weighted PTC, CAL, AST, MAP and FRUIT contracts are unchanged.
The accepted seed repair is essential: task scheduling must not change seeds.

OG reference (read-only): /Users/gwilson/GitHub/citlali at ffc6b9070,
include/citlali/core/engine/pointing.h: sequential input generator feeds a
configured grppi scan farm; workers own scan buffers, and output/accumulation
have explicit synchronization. Adopt independent work units, resource budgeting,
owned scratch and ordered publication. Do not import Engine state, process
statics, new dependencies or old scientific algorithms. The current native
observation executor already shares prepared immutable inputs and runs RTC/CAL
networks concurrently with serialized NetCDF I/O.

Measured trigger: accepted seed baseline 152389 M0--M5 375--377s; saved receipts
show about 39s health and 34--37s per map with downstream worker count 1 despite
runtime.n_threads=3. Native profiles identify substantial saved-input verification
and loading as well as numerical work. Current configured calls pass workers
only upstream. No promised speedup follows from this evidence.

## One budget, explicit phases

runtime.n_threads remains the requested CPU-worker budget (existing 1--12 domain).
Resolve available CPU/affinity/Slurm limits through the established resource
discovery helper. Record requested, available, effective and started workers;
use fewer workers when there are fewer jobs. Keep numerical-library internal
threads at 1 for outer-worker phases, including newly created worker threads.
Never multiply outer workers by a separate BLAS/OpenMP pool. A stage boundary
joins all work before the next phase; M0 -> feedback -> M1 remains ordered.

| Phase | Work units and ownership | Implementation/disposition |
| --- | --- | --- |
| Input/config/output setup | configuration, shared APT/TEL/native-clock preparation | Keep authority and output claims sequential; preserve serialized NetCDF and shared preparation. No unsafe parallel file-handle access. |
| RTC/CAL (v1 PTC upstream) | independent network, shared immutable preparation | Retain existing parallel owner; apply available-CPU cap and report resource realization. |
| Saved CAL loading | per-network verification, YAML, binary reads and AST joins | Initially preserve sequential loader because its shared library/NetCDF preparation requires a separate isolation change; explicitly time/report it. No integrity-check removal or unreviewed cache. |
| Health | network aggregation/reference work; independent detector bootstrap checks | Parallelize independent jobs with barriers around cohort identity and array normalization. Preserve each detector's sequential resampling arithmetic and exact seed. |
| Weighted PTC/map pass | independent network/physical-segment cleaning and map-member construction | Parallel segment jobs with private matrices, masks, evidence and contribution vectors; merge in original network/segment/sample/detector order. Existing MAP resolve/apply and output writing remain ordered. |
| Feedback and final fits | independent arrays | Reduction-owned spawn processes inherit the native effective budget, capped at three arrays, and numerical-library limits before imports. Reuse across passes; join before the next pass or publication. Frozen contexts cross by value and regain read-only array flags after transport. |
| Publication/provenance | ordered product identity, required writes and no-replace directory publication | Serial commit after every worker succeeds. Failures join outstanding jobs and propagate; no completed-success publication. Existing native partial/unavailable and failed receipts remain preserved. |
| Configured v1 / historical saved MAP and JINC | existing map owners | Retain explicit serial admission; parallel saved-MAP admission is limited to configured weighted POINT. |

No nested pools, speculative DAG, generic persistent executor, global cache or
persistent cross-reduction state. Small reusable index-work scheduling serves
the actual health and segment owners. Scientific request objects do not acquire
worker counts: execution options are separate parameters.

## Memory and ordering

Additional worker-owned matrices and contribution buffers are authorized here.
Keep immutable observation inputs shared; never copy the whole observation per
worker. Bound outstanding segment results to a configured-worker-sized batch,
merge deterministically, and release before the next batch. This spends memory
on simultaneous numerical work without retaining every temporary from all passes.
Keep snapshots/receipts and failure semantics intact. Report wall time, CPU time
and process peak RSS where available; sampled/aggregate measures are labeled.
Further persistent saved-CAL reuse across subprocesses is a separate architectural
increment after measured results, not an unannounced integrity-policy change.

## Gates and completion

Prepare native scheduler concurrency/cap/exception tests; health serial/parallel
seed/evidence identity; segment ordering and failure/publication coverage;
Python worker propagation/resource-report and array-failure coverage. Existing
focused optical/weighted, CTest, baseline tools and required config preflight
remain owner-run gates. Public headers and Spack CLI/version must build.

Fresh independent plan/design and exact-candidate read-only review are required.
First owner packet builds/tests only. After successful reviewed software gates,
prepare matched 152389 M0--M5 serial and configured-parallel runs on the same
candidate, same settings/inputs/environment, with scientific equivalence and
stage wall/CPU/RSS evidence. No invented scientific tolerance or speed target.
A scheduling-dependent product/seed, oversubscription, shared mutable scientific
state, missing required failure, unbounded memory or absent authority stops work.

Scope: connected configured POINT/native services. Inventory retained standard
and native other-mode ownership but do not change frozen 4.x/5.x or implement
SCIENCE/Beammap/OOF algorithms in this work. No wholesale rewrite, threshold,
calibration, rank, sample/coordinate/weight or feedback-policy changes.

## Design review and static disposition

Independent read-only design review requested Python library caps before import,
pre-extracted YAML controls/digests, weighted-only saved-MAP worker admission,
and explicit preservation of partial native receipts. The implementation adopts
these points. Fresh-context exact-SHA milestone review remains required; the
design review is not candidate approval.

Scientific/behavioral: no intentional numerical change. Existing seed identity,
per-job arithmetic, contribution order, thresholds, source models and fit owners
are preserved by inspection. Focused exact-value tests are prepared; representative
same-candidate 1/3-worker results and predecessor comparison remain pending.
No numerical tolerance has been added or widened.

Architecture/ownership: standard-library indexed native jobs serve three actual
owners (RTC/CAL, optical learning, weighted segment execution). Native setup owns
CPU availability; Python consumes its effective budget. No Engine growth or new
external dependency. Spawn isolates already-initialized parent numerical libraries;
worker-local contexts and outputs prevent shared fit state. Serial accumulation,
input integrity checks and staged publication retain existing ownership.

Repository/evidence: isolated candidate based on the accepted seed-repair
mainline, without the parked hashing change. Mainline, frozen runtimes, pointing
work, retained standard and other-mode algorithms are untouched. Git whitespace
inspection is clean; no compiler, interpreter check, test or scientific runner
has been executed by Codex. Owner packet is outside the source tree and will bind
the exact reviewed candidate, tree, runner and retained Spack environment.

Review triggers/dispositions: new execution-only sidecars and process lifecycle
are within this authorized resource increment; require packaging, lifecycle,
failure and requested/effective-resource tests. Native saved-input reload and
MAP sum remain measured serial candidates, not hidden omissions. Python pool
startup and bounded scratch may offset gains; only owner measurements decide
performance acceptance. Retained-standard scan/map/Wiener policies and other-mode
successor scheduling are inventoried compatibility boundaries, not new targets.

