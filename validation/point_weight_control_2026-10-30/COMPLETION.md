# Prepared comparison completion — 2026-10-30

Disposition: reviewed local candidate and prepared owner trial. No observational
reduction or real-data fit was run. The first integration/release milestone is
not claimed complete. Temporal studies remain paused.

## Exact source and runtime

- Base: `3def84c209846c3c9f58459de51d107726e8dd1b`.
- Reviewed implementation/preparation candidate:
  `d042083be71f76a406fb698707f7bde94c928d62`.
- Candidate tree: `888d6e33904ed4d2fef190542181f47d9643cf49`.
- Branch/worktree: `codex/configurable-point-20261028` at
  `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
- The CLI was rebuilt after committing; native `--version` reports `d042083be`,
  matching that exact source. Runtime freezing occurred from a clean source tree.
- Frozen runtime:
  `/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/software/configured-point-control-d042083be`.
- 243 runtime members; native SHA-256:
  `436c8ff369be4d3ea0a190567d8b0890e6680852e15b21f1b4d24ed659ceadb6`.
- Runtime manifest SHA-256:
  `604ed0ae0b0aed6d46f8b46d485e0e4003cceef3ad7e50dbb454ffc8759e3ce6`.
- Launcher SHA-256:
  `9d9ea0e3546b149f2391dda2f867c253e805834660588a2c7374ebfad148ebd6`.
- TolProj read-only source during preparation:
  `6309a5a852a0412cee867992aff01341a56b0ebf`, clean.

## Prepared project and checks

Trial:
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weight-control-152418-20261030`.

READY:
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/evidence/weight-control-152418-20261030/READY.json`.
SHA-256 `7364d1e8449a7de2a4957bbc3e60299bb81ec1371e9e45455e33f5174e6354a0`.

All nine numbered configurations, preparation helper/driver/support files,
runtime and selected shared APT bindings are frozen in READY. The actual TolTECA
merge checks 152418 only, all three arrays, ranks 3/4/6, 2 arcsec pixels, coverage
0.15, optical threshold 0.10, ten feedback applications, three workers, quiet
mode and explicit `weight_comparison: true`. TolProj executable snapshot and
local/node-local launcher guards were reused. The check-only shell driver passed;
the new trial has no `reduced` directory. Exact low-level observational inputs
are generated and validated only at Grant's explicit launch.

Original source numbered configurations match their saved inventory. The writer
also rehashed all 18 old uniform/M0/M10 maps from the saved-product audit; every
map still matches. The independent reviewer separately checked the old final
152418 maps against their saved RUN digests. No reference, APT or input changed.

Local gates are in [CHECKS.md](CHECKS.md). The last new coordinator tests pass
11/11 after formatting, and full config preflight passed again. Compiler/linker
output contains only the existing duplicate-libcitlali warning. The reviewer
observed sandbox-specific cache/Fontconfig stderr messages in its check-only
import; those did not prevent the check and are explicitly distinguished from
reduction/scientific errors. Build and Python environment remain supplemental
Mac evidence, not Spack/Unity validation.

## Independent disposition and preserved limits

[Independent review](INDEPENDENT_REVIEW.md), SHA-256
`5a130e10adae1a621b462e6acf25f9488c84962605f4eaf3153a11c77fe32388`, binds
the exact candidate above and verifies the prepared runtime/trial.
Scientific/behavioral: pass with recorded limitations. Architecture/ownership:
pass. Repository/evidence: pass with documented external-environment limits.
No actionable findings or repair required. Reviewer reproduced 14 native and
102 focused Python tests, all runtime/source bindings and check-only launch.

Intentional scientific change is confined to the explicit equal-coefficient
experimental arm. Default weighting, RTC/CAL, health threshold/membership policy,
source recipe and calibration ineligibility remain unchanged. Native PTC owns
the control; Python only orchestrates and verifies. Extra memory/evidence is
bounded per segment; no observational speedup or scientific result is asserted.

No integration, push, production activation, Unity access, reduction, or additional
experiment is performed. The frozen runtime remains bound to the reviewed source
even though a later documentation-only closure commit records this completion.
The closure adds only this file and the unchanged reviewer report.

**Next action belongs to Grant:** use the exact commands in [HANDOFF.md](HANDOFF.md).
After completion, first inspect `weight-comparison.json` for full budget and exact
support agreement, then assess M0/M10 map and fit changes across all arrays.
This tests combined cleaning/map weighting and its mediated FRUIT effect, not
absolute flux accuracy or a generic detector-health claim.
