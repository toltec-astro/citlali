# Build-sequence repair — 2026-09-29

Owner: Grant Wilson. New directive: “Repair and test that the process builds and
runs but then stop and issue a new launch command for me to run.” This authorizes
local preflight execution in this turn and a new owner launch, with no observation
trajectory in this turn. Original no-retry/no-extra-science budget remains intact.

Base: 0b96d1c9ca35cace9a106585c0ed8be8cd5293fe, clean initial worktree;
branch codex/rtc-point-transfer-20260928, existing ptc-rank-response worktree.
This remains the same bounded probe, not another scientific work order.
Tier 2: ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE and
REVIEW_AND_CONFORMANCE read; accepted source, canonical incorporation, digests
and subject authorities remain those recorded in PREFLIGHT.md. No scientific,
AST/RTC, interface, Engine, native numerical or build-environment policy change.

First attempt STATUS=PROCESS_FAILED, build exit 2, 59.68 seconds. The main
executable compiled; the same Make invocation did not recognize the newly
regenerated citlali_rtc_point_select target. The resulting on-disk Makefiles do
contain it. Repair: explicitly configure before starting the multi-target build.
No failed-gate bypass; required gate failures remain terminal.

Scope: runner sequence, a mutually exclusive preflight-only mode, failure-preserving
new root binding, fixtures and handoff documentation. Preflight-only executes all
normal preflight gates then returns before prepare_role/launch. No review is needed
to run tests; scientific launch still requires a passing exact-source independent
review. Both modes require clean source and the local venv. Each mode refuses reuse
of its exclusive marker. No automatic retry is added.

New root:
/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-build-repair-20260929

The failed root remains untouched and is added to PRIOR, retaining all 284278
bytes measured at repair start. The prior scientific retention/caps stay unchanged;
capacity is recomputed at execution including failed evidence. Preflight output
lives in the new root/preflight-only/<source-SHA>; logs/evidence count toward shared bytes.

Gates: short runner fixtures; actual configure and both native executables;
CTest/check; baseline tools; POINT tools; config require-all; native all-five-cycle
request checks; historical saved-only checks. These operate on fixtures/saved
historical evidence, not a new observed trajectory. No detector I/Q solve or POINT
152420 reduction is authorized in this turn. Final exact-source independent review
is required. Evidence is external under preflight-only and PREFLIGHT_REVIEW.json,
so its generation does not change the tested/reviewed source commit. Local
AppleClang/Homebrew cached dependencies remain supplemental, not Unity Spack.

Stop on any unresolved required gate; preserve all logs. Repair bounded defects
within this authorized execution-preflight scope, never change scientific policy
to make tests pass. No Unity, push, merge, production activation or deletion.
Completion is READY_FOR_OWNER_LAUNCH with zero science and a new exact command;
actual observation and final scientific assessment remain pending.

## Required POINT native fixture

The first repair preflight at b317e7041f118fce075320d595948d15d4da3c21
exited 0 with zero science: 1318 CTests passed (one previously disabled lifecycle
test), 207 baseline tests passed, config/native request/history/runner gates passed.
POINT reported 203 passed and one skipped: the frozen saved-apply fixture requires
POINT_FROZEN_APPLY_BINARY. This is a gate-wiring omission, not accepted coverage.
The bounded follow-up adds its existing target to the build and supplies that
explicit binary to the POINT suite. No test/scientific semantics change. Preflight
records are now keyed by exact source SHA so each corrected source is tested once
without overwriting earlier evidence or allowing same-source replay. The first
record remains at preflight-only/STATUS.json; final evidence is in the SHA child.
