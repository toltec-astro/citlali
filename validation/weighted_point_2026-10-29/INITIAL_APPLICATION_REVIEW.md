# Independent exact-SHA application review

Candidate: `4f7d31957d2135601145c63387e310622fe16d64`  
Tree: `f5d22e465327a0f77101fbfc4bc1400bf88862ac`  
Parent/base: `c6ea76247213cdb8ce51c9ed935a9744b15beb9a`  
Worktree: `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`  
Review date: 2026-10-29

Verdict: **pass with recorded limitations for the reviewed application scope**. No actionable findings were identified. This is not a numerical-method review or observational acceptance.

## Preflight and authority

This fresh-context review remained read-only in the repository. HEAD, parent and tree were verified at the start/end; staged, unstaged and untracked inventories were empty. The review read AGENTS.md, Engineering Governance, Timestream Successor Governance, and Review and Conformance, including the standing 2026-09-30 development amendment. Governance effectiveness was checked against the integration ledger's accepted exact-commit/digest record. It also read the current status section, the optical-response owner decision, and `validation/weighted_point_2026-10-29/{WORK_ORDER,IMPLEMENTATION,LOCAL_CHECKS}.md`.

The TolTEC context skill, repository map and authority map routed the operational TolTECA comparison to the read-only `origin/main:tolteca/reduce/engines/citlali.py` implementation. No sibling repository was changed.

## Scope and inspected boundary

Inspected the changed Python configuration, CLI, weighted coordinator, feedback selector, product publisher and runtime packaging; the new v2 YAML; their synthetic tests; and the native saved-CAL, processing-binding, feedback-provenance/FITS-reader, dispatcher and map-FITS boundary interactions. Consulted existing `configured_point_inputs.py`, POINT-EXPECTED, source patch, candidate guard and FITS writer implementations as direct dependencies.

Verified by code inspection:

- The new profile requires explicit v2 schema/migration/method, preserves v1 as the dump-config default, and records the actual configurable response fraction, ranks, pixels, coverage, worker count and feedback budget. Requested input dictionaries are copied before adaptation. Legacy FRUIT remains disabled.
- The ordinary TolTECA wrapper command shape, `--version`, `--dump_config`, merged low-level configuration and output-directory injection remain supported. Upstream v2 processing terminates at CAL; v1 retains its PTC route.
- Every weighted cycle binds the same original CAL receipts, producer-bound processing intervals and complete frozen-health bytes. M0 uses no source model; later models replace their predecessors and bind the previous cycle's array, observation, health, map and bundle digests.
- Python feedback FITS and native reader agree on primary candidate values, `ADMITTED`/`VALID` planes, calibrated `mJy/beam`, explicit array/nominal beam, corrected AltAz arcsecond axes and cropped CRPIX values. The degree conversion stays inside the historical selector adapter. Explicit high-precision WCS cards preserve the native equality checks.
- Empirical numerator/normalization units remain `beam/mJy` and `beam2/mJy2`; final publication requires a homogeneous all-array cycle/health association and verifies the native plan/map digests. The uniform family remains separate.
- Source-selector unavailability stops feedback and retains the last complete all-array maps; malformed identity, required I/O, native failure and error-level output propagate. Quiet products retain health states/limitations, per-segment unavailable noise reasons and ranges, contributor counts, and the explicit feedback trajectory/stop outcome.
- Packaging includes the new modules and overlay, verifies tracked cleanliness and native/source revision agreement, authenticates frozen members, and retains explicit external-Python/shared-library limitations.

## Reproduced gates

Used `/Users/gwilson/tolteca/bin/python`, `MPLBACKEND=Agg`, task-specific `/tmp` Matplotlib/cache paths, `PYTHONDONTWRITEBYTECODE=1`, and pytest without cache provider.

Focused reproduction: **105 passed in 15.36 seconds** across:

- `test_configured_point_feedback.py`
- `test_configured_point_v2_config.py`
- `test_configured_point_weighted.py`
- `test_package_configured_point.py`

Log: `/tmp/weighted-point-app-review-pytest.log`.

These use synthetic FITS and mocked native stage calls. The real native feedback reader and upstream/saved-CAL joins were inspected, not executed against observations. The reported full 256 Python, 1,364 CTest, baseline-tool and config-preflight results were read as candidate evidence, not independently reproduced in this review. No concurrent native build was launched.

## Three dispositions

1. **Scientific and behavioral conformance: pass with recorded limitations in the reviewed application scope.** No blocker/major/minor finding. Configuration propagation, provenance, units, all-array publication, failure semantics and feedback lifecycle conform to the bounded recorded request. The response estimator's statistics, PTC numerical method, real rejected populations, real runtime/memory, map quality and actual trial behavior remain outside this application-only verdict. Synthetic/mocked tests do not establish repeated-sampling interval coverage or a working observational reduction.
2. **Architectural conformity and ownership: pass in the reviewed scope.** No blocker/major/minor finding. Python coordinates calls, persistence and publication; numerical CAL/optical/PTC/FRUIT/MAP owners remain distinct. The feedback selector adapter reuses the existing POINT/BASE owners and freezes observation-local context. No Engine growth was introduced in the inspected paths.
3. **Repository, branch and evidence hygiene: pass with recorded limitations.** No blocker/major/minor finding. The exact candidate is the direct child of the recorded work-order base and the worktree stayed clean. Local fallback/synthetic evidence is correctly distinguished from Spack/Unity validation, owner trials, canonical integration and production acceptance. No raw/saved observational CAL processing, Unity access, reduction, push, integration, production activation or cleanup occurred.

Any source repair requires review binding to its new exact SHA; this verdict does not transfer automatically.
