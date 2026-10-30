# Local checks before freezing

Supplemental Mac AppleClang/Homebrew build; `/Users/gwilson/tolteca/bin/python`.
No dependency installation/change, Unity/Spack validation or observational
reduction. The existing KIDs dependency retains its `04088da-dirty` qualification.

- CLI and inverse-noise test targets build successfully. Existing duplicate
  `libcitlali.a` linker warning only; no compiler error.
- 14 focused native inverse-noise/control/MAP tests pass. New tests compare
  equal-control cleaning against independent unweighted SVD on unequal-noise
  detectors, preserve unavailable-noise exclusion and exact FRUIT restoration,
  and reject control/ordinary coefficient-family evidence substitution.
- 268 configured POINT and runtime-packaging Python tests pass. Includes the
  paired coordinator, all six exact-support comparisons, shared CAL/health,
  separate arm feedback, opt-in parsing and control product units.
- Full CTest: 1,368 enabled tests passed; zero failures; 1,369 discovered.
  One inherited disabled `MapFitterLifecycle.ExactProductSequence` test remains
  disabled and is not claimed as tested. Duration 51.09 seconds while other
  independent checks also ran; not a performance benchmark.
- Baseline tools: 207 passed, 137 subtests passed, 25.82 seconds.
- Configuration preflight `--require-all`: passed, no missing required data.
- New comparison/coordinator test/preparation Python files: Ruff format/check
  passes. `git diff --check` passes.

Logs: `/tmp/point-weight-control-build.log`,
`/tmp/point-weight-control-python-final.log`,
`/tmp/point-weight-control-ctest.log`,
`/tmp/point-weight-control-baseline.log`,
`/tmp/point-weight-control-config.log`.

The branch implementation adds only the explicit paired control. Native PTC owns
the numerical coefficient choice; VAL/MAP still binds exact support and receives
the coefficients. The Python boundary shares upstream and health, sequences
independent same-recipe trajectories, authenticates publications and compares
support. No Engine state, scientific threshold, calibration contract or default
weighting change. The extra runtime work is one additional PTC/MAP/FRUIT arm;
RAW/RTC/CAL and health are not repeated. Additional quiet evidence is a few
per-segment hashes rather than detector-sample files. Actual runtime and scientific
effect are intentionally left for the owner-run trial.

Source must be committed, rebuilt for matching `--version`, independently reviewed,
then bound into the fresh runtime/trial before handoff. Exact identities and
preparation-only results are recorded in COMPLETION.md after those checks.
