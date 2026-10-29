# Local engineering checks before exact-candidate review

Environment: existing Mac AppleClang/Homebrew fallback build; Python
`/Users/gwilson/tolteca/bin/python`. These are supplemental checks, not accepted
Spack/Unity application evidence. Existing native dependency version retains its
reported `kids 04088da-dirty` qualification. No dependency installation/change.

- CLI target built with `cmake --build build --target citlali_cli -j 8`.
  No compiler errors; the existing duplicate-libcitlali linker warning remains.
- CTest: 1,364 enabled tests passed, zero failures, 1,365 discovered, 7.51 seconds.
  One pre-existing disabled `MapFitterLifecycle.ExactProductSequence` test.
  Includes 12 inverse-noise/MAP, eight optical, ten FRUIT, 17 forward MAP/AST
  and eight new persisted-binding tests, plus isolated public/private headers.
- Configured POINT Python tests and runtime packaging: 256 passed in 24.05 seconds.
  Includes real synthetic FITS/selection with mocked native stages; no observation.
- Baseline tools: 207 passed in 24.45 seconds.
- Full config preflight `--require-all`: passed. A first attempt stopped because
  sandbox permissions prevented py_compile writing the worktree bytecode cache;
  rerunning with `PYTHONPYCACHEPREFIX=/tmp/weighted-point-pycache` passed all gates.
- Intended scientific-change ledger validation passed; `git diff --check` passed.

Logs are currently `/tmp/weighted-point-cli-final-build.log`,
`/tmp/weighted-point-ctest-final.log`, `/tmp/weighted-point-python-tests.log`,
`/tmp/weighted-point-baseline-tests.log`, `/tmp/weighted-point-config-preflight.log`.
Final handoff records exact committed source, executable and frozen configuration
identities after independent review. The test counts do not constitute evidence
that 152389/152418 improve, or that any detector should actually be dropped.

Review-driven repairs before freezing: producer-bound processing intervals;
native feedback observation/health/prior-cycle/model/map associations; immutable
health across Python cycles; full empirical map units/plan validation; explicit
feedback stops; qualified FITS geometry and exact WCS representation; persisted
health reasons checked against the decision owner. No threshold was changed.

Practical performance choices: CAL once; health once; cached per-block bootstrap
moments; shared leave-one-out median order statistics; no raw/RTC parent graph
retention across all networks; no per-sample cycle exports; compact normal-weight
ranges; contributor bookkeeping once per detector. Actual end-to-end runtime
remains for Grant's two trials; no unmeasured speedup factor is asserted.
