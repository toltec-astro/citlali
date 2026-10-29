# Local engineering checks before exact-candidate review

The first exact candidate `4f7d31957d2135601145c63387e310622fe16d64`
passed the checks below but failed independent numerical review. It was never
frozen into a trial runtime. [Initial native review](INITIAL_NATIVE_REVIEW.md)
identified one major issue: a response fit used sparse/constant blocks that its
bootstrap silently omitted, permitting a false confidently weak decision. The
[independent reproducer](review_counterexample.py) and [initial output](review_counterexample.txt)
are synthetic, not observation processing. Application review passed its scope
in [the separate report](INITIAL_APPLICATION_REVIEW.md).

The bounded repair retains every contributing fitted block. If a bootstrap draw
cannot support the fit, its interval is explicitly unavailable and the detector
remains unresolved. The method identity is now bootstrap-v2, preventing restored
v1 evidence from bypassing the repair. Two new public Learn/Consider regressions
cover both formerly omitted singleton/constant blocks and explicit resample rank
failure. All ten optical tests pass; no threshold or other scientific policy was
changed. Corrected exact-source review and matching executable are required before
freezing either owner trial. The repaired full CTest suite passes 1,366 enabled
tests with zero failures in 7.36 seconds (one inherited disabled test). Restoring
only the old filtering line in an isolated temporary test binary makes both new
tests fail, confirming that they catch the actual false-weak decision. Logs:
`/tmp/weighted-point-repaired-ctest.log` and
`/tmp/citlali-point-optical-old-block-filter-regression.log`.

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
