# Independent exact-SHA Tier 2 native review

Verdict: **repair required**.

- Candidate: `4f7d31957d2135601145c63387e310622fe16d64`.
- Tree: `f5d22e465327a0f77101fbfc4bc1400bf88862ac`.
- Parent / local work-order base: `c6ea76247213cdb8ce51c9ed935a9744b15beb9a`.
- Branch: `codex/configurable-point-20261028`.
- Worktree: `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
- Initial and final tracked/untracked inventory was clean. Review made no repository edits and performed no build, observation processing, Unity access, push, integration, or cleanup.

## Scope and authority

Fresh-context read-only review of native optical learning/decisions, inverse-noise PTC, empirical coefficient/admission, FRUIT offset projection and restoration, and saved CAL/health/feedback adapter bindings. The candidate changes 46 files; the main Python application is separately reviewed.

Read `AGENTS.md`, all three governance documents, the governance effectiveness record in `doc/INTEGRATION_LEDGER.md`, current `doc/REFACTOR_STATUS.md`, relevant architecture/scientific conventions, `doc/OPTICAL_RESPONSE_POLICY_2026-10-29.md`, and `validation/weighted_point_2026-10-29/{WORK_ORDER,IMPLEMENTATION,LOCAL_CHECKS}.md`. Applied the TolTEC context skill and its authority routing. The owner-approved conditional resampled interval is assessed as authorized; lack of demonstrated population coverage is a recorded limitation, not a finding requiring an alternative scientific method.

## Scientific and behavioral conformance: major finding, blocks freeze

### R1: Preserve every fitted science block in the resampled interval

Location: `src/citlali/core/pipeline/point_optical_response.cpp:104-107`, especially line 107.

`interval()` fits the full paired sample domain at line 103 and groups all those fitted rows at lines 104-105. It then includes a block in resampling only if that block independently supports an intercept-plus-slope fit. A one- or two-bin block, or a block with constant reference level, is silently deleted even though its rows contributed to the reported/full response and can carry valid between-block excitation. The bootstrap therefore estimates a different restricted-domain response. The full/early/late envelope does not repair this because each portion applies the same deletion.

A synthetic counterexample reproduces this mechanism with 12 science blocks. Blocks 0, 1, 6 and 7 each have 10 paired bins with `x=linspace(-1,1,10)` and `y=(0.03+0.01*(block%6))*x+0.0001*sin(arange(10))`. The other blocks each have one valid paired bin with `x=y`, respectively -100, -70, 70, 100, -90, -60, 60 and 90. The all-row robust fit is 0.9997211557; early and late all-row fits are 0.9997510910 and 0.9996830357. Nevertheless, all three current block-bootstrap 99% intervals are approximately [0.03000457, 0.04000464], because only blocks 0/1/6/7 survive line 107. With unit normal-response scale these bounds produce `weak_response` at the default 0.10 although the full-domain response is approximately normal. Sparse source-free support is permitted by the interface and is not inherently invalid.

Consequence: a misleadingly confident weak decision can exclude a normally responding detector from PTC basis learning and both map numerator and normalization. The accepted optical policy requires adequate uncertainty over the response being evaluated, and the work order explicitly preserves missingness/support rather than introducing an unrecorded sample selection.

Smallest bounded repair: retain the moment record for every block represented in the fitted rows when resampling; preserve explicit block/rank adequacy checks and return an unavailable interval when required draws cannot support the fit. Do not silently delete fitted rows/clusters to obtain a finite interval. Add regression coverage for sparse and constant-within-block reference support showing that the retained bootstrap domain equals the fit domain and cannot establish weak response solely by discarding normal-response blocks. This repairs the implementation of the authorized conditional bootstrap without changing the configured threshold.

Evidence: `/tmp/weighted-point-optical-counterexample.py` and `/tmp/weighted-point-optical-counterexample.txt`. This independent synthetic calculation mirrors the native moment solver, 20-step IRLS, quantile and SplitMix64/rejection-index functions, using fixed synthetic seed 42 rather than claiming an actual detector seed. It does not claim an end-to-end native application run. Script SHA-256: `9f6e38e1bb6631772d2e089f2d90bd2acbe43b84cb12feaadd3810613e9a9cac`; output SHA-256: `786cd8fef17548036be060463a664869bf537d1fd3aa69e418d437cb9573e685`. The domain deletion is directly visible in the candidate source.

No other blocking mathematical, unit, missingness, or failure-propagation issue was identified in the reviewed scope. The native threshold boundary admits values above 1 although the configured Python parser enforces the documented upper bound; this does not affect the configured application path and is not a freeze blocker.

## Architectural conformity and ownership: pass within reviewed scope

No `Engine` growth or orchestration science algorithm was introduced. AST owns corrected source geometry; the optical owner publishes immutable evidence/decisions; PTC owns noise learning/whitening and physical-unit restoration; FRUIT owns the exact operand and fresh residual; the separate empirical coefficient family and MAP admission bind exact publication and occurrence identities. Native adapters parse, authenticate, join, and invoke those owners. No second authority or mutable synchronization back into requested configuration was found.

## Repository, branch and evidence hygiene: pass with stated limitations

The exact candidate and tree match the assigned review; its parent matches the recorded local base, branch/worktree ownership is unchanged, and the tree remained clean. `git diff --check HEAD^ HEAD` passed. The candidate is explicitly a local development candidate, not canonical integration or production qualification. The required corrected-SHA review must follow repair; this verdict does not transfer to a later commit.

## Gate reproduction and environment

Ran existing local native test executables without rebuilding:

- `citlali_point_optical_response_test`: 8/8 passed.
- `citlali_ptc_inverse_noise_test`: 12/12 passed.
- `citlali_fruit_cycle_test`: 10/10 passed.
- `citlali_point_weighted_bindings_test`: 8/8 passed.
- `citlali_forward_map_test`: 17/17 passed.

Total: 55/55 passed, no unexpected error-level output. These tests include independent leave-detector-out variance and whitened-SVD checks, unit scaling, frozen exclusion/source support, unavailable weights, exact FRUIT add-back, AltAz projection, native producer binding and map admission. None exercises the R1 sparse-block bootstrap-domain mismatch.

Test binary SHA-256 values:

- optical: `706e0e3a7e36092c19ba8b9b22de8c37e26d71b7b9d9c38a9c8eb27742ef5dde`.
- inverse noise: `e760f7f33f3207e122f3fa6a92919ab61ed33a1d864664a87e2e72df0e631e67`.
- FRUIT: `3adb5ae9e40699dc905270813e5a6632f0037d97d532a384ee210b4eacf61aab`.
- bindings: `027c076226992d469eed347fe3bd3e0a31893d2293a0dc961ef90e0723682881`.
- forward MAP: `344cb18c80c2c339828c96ff4c84d634c5ab27240ee3fb9207fae1fa31e5b932`.

The independent numerical counterexample used `/Users/gwilson/tolteca/bin/python` with `MPLBACKEND=Agg`, synthetic arrays only, and no plot or observed inputs. The test executables are the existing Mac AppleClang/Homebrew fallback realization, supplemental evidence rather than accepted Unity/Spack reproduction. Full suite/preflight results in `LOCAL_CHECKS.md` were inspected but not independently rerun by this reviewer.

Owner-run 152389/152418 validation remains pending. Conditional interval coverage, population false rejection, and end-to-end performance remain explicitly unvalidated. No observation trial should be frozen from this candidate until R1 is repaired and the new exact SHA is reviewed.
