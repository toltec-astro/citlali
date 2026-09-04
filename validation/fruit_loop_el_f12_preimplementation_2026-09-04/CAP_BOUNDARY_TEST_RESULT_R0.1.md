# EL-F12 — Existing exclusion-cap boundary evidence

Date: `2026-09-04`

Status: **focused synthetic tests pass; experimental action remains undefined
pending owner decision CAP-001**

Repository: `/Users/gwilson/.codex/worktrees/4c31/citlali-refactor`

Branch: `codex/sci-fruit-v0.1-empirical-lane`

Baseline: `d78d4d94a3ab67746c46e7519fc5e32ddbcb1844`, descended from the required
`d39d4685baa0aeda036305031491121ce97008ec`. At test time the only C++ change was
the added synthetic cases in `tests/test_learning_target_application.cpp`.
The production handler and its stage routing were unchanged. The amendment
manifest binds the exact tested file and source references.

## Check and result

The test invokes the real Engine stage entry points using synthetic flags,
APT identities and learned records; it does not simulate or invoke the
intervening RTC processing. Its observation name is `synthetic-cap-boundary`
and detector identities are 1000 onward. No UID 4460 data is used in the new
cases. This establishes handler behavior for supplied boundary states, not
their frequency in the accepted pointing or an intervention's effectiveness.

- 3/100 proposed unflagged detectors: before-RTC cap rejects, flags unchanged.
- Same population at PTC: cap also rejects, flags and APT flags unchanged.
- Two proposed detectors absent at PTC: 1/98 accepted; the remaining proposed
  detector is flagged, other detectors remain unflagged. The unchanged handler
  reports the two absent records as unmatched/invalid and still considers the
  matched record. This receipt term is not a new EL-F12 input-validity rule.
- 2/100 at RTC: exact 2% boundary accepted.

Commands, run from the repository above:

```sh
cmake --build build --target citlali_learning_target_application_test -j 8
ctest --test-dir build -R '^citlali::learning_target_application::' --output-on-failure
```

Build exit status 0; CTest exit status 0. Eight of eight tests passed, including
six existing target/exclusion tests and the two new boundary tests. The build
and CTest logs are retained as
[build log](cap_boundary_build_r0.1.log) and
[test log](cap_boundary_ctest_r0.1.log). They contain no unexpected error-level
output. Expected cap rejection uses the fixture's null logger.

Focused test executable SHA-256: `c132c23e2ea0382c874925cf577ad0b6301dcd2bbc9ff45bc73f046b84e1ab43`.

This is a focused verification of unchanged handler semantics. The complete
EL-F12 build, full CTest, baseline/FRUIT Python suites and configuration
preflight have not been run for a prototype; no prototype exists yet. This
test executable is not the prospective scientific replay executable.

## Prerequisite and preservation check

The [input identity report](INPUT_IDENTITY_PREFLIGHT_R0.1.json) records 53
successful checks: all 26 approved proposal members and 27 external input
files (15 science/configuration files and 12 text fit reports). The same
identities were rechecked before sealing this evidence. The retained EL-F2
alpha-one map files for all three arrays and iterations 0–6 are present; no
new map comparison or claim of bitwise equality has been made.

Both untracked review archives retain their intake hashes:

| Archive | SHA-256 |
| --- | --- |
| `SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz` | `5f11836908aa6aeb4f51690209a32dc6e8d4cee4e6b9c223903c6c57033b9b22` |
| `SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz` | `761ba278a53e32ad1d5d3977230cc4f1f90f257056b547508342797f584167ed` |

No reduction products were written, removed or replaced. No EL-F12 external
output root was staged. Primary trajectories: 0/8; conditional restarts: 0/4;
diagnostic-defect replacement trajectories: 0/1. These synthetic unit tests
consume no reduction or replay allowance. The unrelated SCI-ALIGN checkout
under `/Users/gwilson/GitHub/citlali-refactor` was left untouched.
