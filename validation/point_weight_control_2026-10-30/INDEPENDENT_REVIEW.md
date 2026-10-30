# Independent exact-SHA review — 2026-10-30

Reviewer: fresh-context read-only subagent `weight_control_review`. No implementation edits, rebuild, observational reduction, real-data fit, Unity access, remote operation, push, or integration was performed.

Verdict: **pass with recorded limitations**. No blocking, major, or minor findings requiring a candidate change were identified.

## Exact identity and scope

- Candidate: `d042083be71f76a406fb698707f7bde94c928d62`.
- Tree: `888d6e33904ed4d2fef190542181f47d9643cf49`.
- Direct parent/base: `3def84c209846c3c9f58459de51d107726e8dd1b` (ancestry independently verified).
- Branch: `codex/configurable-point-20261028`.
- Worktree: `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
- Initial and final tracked/untracked source state: clean during review. `git diff --check` passes.
- Scope: one prepared owner-run 152418 comparison, sharing upstream CAL and frozen optical health, varying detector coefficients in cleaning and mapmaking, retaining the same FRUIT recipe with separate evolving source models, and requiring exact support equality at every corresponding cycle.

Authority/preflight read: repository AGENTS; Engineering Governance; Timestream Successor Governance including its standing 2026-09-30 amendment; Review and Conformance; canonical governance-effectiveness section of the integration ledger; current REFACTOR_STATUS; applicable architecture/dependency/Engine/failure rules; scientific-convention and bright-POINT routing; standing owner COMPLETION_AUTHORITY; weighted-POINT implementation record; this candidate's WORK_ORDER, HANDOFF and CHECKS. TolTEC context skill and repository/authority maps were read, together with the relevant TolProj refactor-preparation documentation. The candidate is not canonical integration or production acceptance.

Changed-path inventory reviewed: `data/configured_point_v2.yaml`; `doc/REFACTOR_STATUS.md`; the two PTC inverse-noise/empirical-handoff public headers and their implementations; native `point_weighted_execution.h`; native inverse-noise tests; configured POINT CLI, comparison coordinator, config, products, weighted orchestration and package helper; three associated Python test files; and the four work-order/check/handoff/preparation files under `validation/point_weight_control_2026-10-30/` (21 changed paths total).

## Scientific and behavioral conformance — PASS with limitations

No actionable findings. Equal control retains the existing reference/remaining-noise availability requirements and does not restore unavailable detectors. Its cleaning operates in unwhitened calibrated space, map coefficients are unit constants, and exact FRUIT add-back remains owned by the existing native implementation. Default inverse-noise formulas and configuration behavior remain intact. Unit-scale interpretation is expressly arbitrary rather than measured precision, with a separate coefficient family checked in both publication and feedback ingestion.

The coordinator reuses the same prepared network inputs, CAL publications and health binding. Native support evidence covers ordered detector/slot domains, original eligibility, source-free masks, residual eligibility, cleaned support and actual admitted map samples. The comparison authenticates cycle publications/plans, array/cycle/health association, matching observation/WCS, network/scan/rank, all six support keys, cleaner convergence, complete cycle indices and equal completed budgets. Incomplete or mismatched pairs cannot acquire `matched_support_complete`; failure propagates and generated evidence is retained. Support is measured before the fixed map coverage cut; coefficient-dependent pixel coverage is explicitly documented as a weighting consequence.

This is the combined cleaning-plus-mapmaking weighting effect. It does not separately estimate those two contributions. M0 is the pre-feedback contrast; later models evolve independently, so later differences include the mediated FRUIT effect. No actual support match, successful ten-pass observational trajectory, amplitude conclusion, calibration qualification, or detector-health truth is established before the owner run.

## Architectural conformity and ownership — PASS

No actionable findings. Numerical coefficient choice and noise qualification remain in native PTC. Existing VAL/MAP admission owns exact coefficient/support association. Python handles shared-publication orchestration, lifecycle, identity checks and comparison without implementing timestream science. No Engine growth, reverse scientific dependency, mutable global learned state, new calibration rule, source-selection threshold or scientific fallback was introduced. Compact per-segment hashes avoid publishing per-sample histories. Existing shared transport cleanup runs only after the two consumers and required publication finish successfully.

## Repository, branch and evidence hygiene — PASS with limitations

No actionable findings. Candidate/parent/tree match the commissioned review, the change inventory stays within the bounded work order, and source remains clean. Fresh preparation refuses existing destination/evidence paths; check-only is the default. Original numbered configuration digests match the pre-preparation inventory. The old `redu00` inventory remains and its three published maps still match their saved RUN.json SHA-256 values. New trial has no `reduced` directory before or after independent check-only execution.

Prepared project root: `/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`.
Trial: `pointings-weight-control-152418-20261030`.
READY: `evidence/weight-control-152418-20261030/READY.json`.
READY SHA-256: `7364d1e8449a7de2a4957bbc3e60299bb81ec1371e9e45455e33f5174e6354a0`.
Runtime: `software/configured-point-control-d042083be`.
Manifest SHA-256: `604ed0ae0b0aed6d46f8b46d485e0e4003cceef3ad7e50dbb454ffc8759e3ce6`.
Native SHA-256: `436c8ff369be4d3ea0a190567d8b0890e6680852e15b21f1b4d24ed659ceadb6`.

Independently verified every bundled member digest and size, every bundled source byte against `git show` at the candidate, frozen driver against candidate preparation source, READY/support/config/shared-APT bindings, original numbered-config bindings, and native `--version` (`d042083be`). `bash 02_redu.sh` exited 0 with explicit CHECK ONLY and no reduction launched. Its existing preparation interpreter and merged TolTECA/resource checks confirmed all arrays, ranks 3/4/6, 2 arcsec pixels, coverage 0.15, response cut 0.10, ten feedback applications, three workers and comparison opt-in.

The sandbox denied the prepared project cache location during this independent check, causing Matplotlib temporary-cache fallback and a Fontconfig cache warning/error on stderr. The check completed successfully; this was an environment/cache diagnostic, not scientific execution or a reduction failure. No GUI backend was used. The preparation is local-host-specific; external Python package contents and native shared libraries are not fully pinned by the bundle, as its manifest already records. Raw/low-level observation preparation remains deferred to the owner's explicit launch.

## Gate reproduction and environment

Independently reproduced:

- `build/bin/citlali_ptc_inverse_noise_test`: 14/14 pass, including independent SVD equivalence, unequal-noise control, unsupported-noise protection, exact FRUIT restoration and family/evidence mismatch rejection. Test executable SHA-256 `097160f828e82425280112119cf1d84ef7747e696ac3a1ec171ea5dc4d17c565`.
- `/Users/gwilson/tolteca/bin/python -B -m pytest -p no:cacheprovider` on `test_configured_point_comparison.py`, `test_configured_point_v2_config.py`, `test_configured_point_weighted.py`, and `test_package_configured_point.py`: 102 passed in 15.64 seconds. Agg backend and task-specific `/tmp` caches were set.
- `git diff --check`, exact identities/ancestry/clean state, runtime/source/digest verification and the prepared check-only launch: pass.

Broader evidence was inspected in CHECKS, not rerun by this reviewer: 268 configured/runtime Python tests, 1,368 enabled CTest tests (one inherited disabled test explicitly excluded), 207 baseline tests plus 137 subtests, and full config preflight. All are supplemental local Mac AppleClang/Homebrew evidence, retaining KIDs `04088da-dirty`; none is representative Unity/Spack evidence.

Owner retains observational launch, scientific assessment, integration, push and production decisions. No permission expansion follows from this review.
