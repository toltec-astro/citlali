# Local validation

Validation was run with `$HOME/tolteca/bin/python` and headless Matplotlib.

- `test_analyze_sci_align_001_lissajous_timestream.py` together with
  `test_visualize_sci_align_001_lissajous_fit.py`: 15 tests passed in 47.768
  seconds. The suite includes injected zero, negative, positive,
  static-offset, pure-hysteresis, and joint regimes; irregular scan sampling,
  flags, detector gains, baselines, correlated noise, wrap-safe interpolation,
  invariant common support, derivative agreement, convergence gates, and the
  abnormal inherited-start multistart fallback regression. It also covers the
  runtime deadline/event contract, deterministic fixed-support identities,
  contiguous fit-unit construction, and exact model-component objective
  reconstruction.
- `py_compile`: estimator, generalized visualization renderer, and both focused
  test modules passed.
- `tools/config/run_config_preflight.py --require-all`: 123 unit tests passed;
  8 compact compatibility cases passed; no skips, warnings, gaps, review
  requirements, or boundary drift.
- All nine opened external observation packages verify against their current
  `SHA256SUMS` manifests.
- `summarize_partial_results.py` regenerated all four compact machine-readable
  artifacts byte-for-byte in a fresh temporary directory.
- Both representative two-page PDFs were rendered completely. The anchor
  pages are legible and show a single objective minimum and well-behaved
  paired distribution. The stop pages are legible and show a single smooth
  full-data profile plus the bootstrap optimizer-start pile-up and broad
  secondary support described in the evidence note.
- The three-page read-only ObsNum 136280 scan audit was rendered completely.
  It confirms that the pile-up is not concentrated in low-scan-diversity
  draws and visually exposes the opposing scan-6/scan-7 and residual-MSE
  associations. Its checksum manifests and the optimizer-probe manifest both
  verify.
- The generalized visualization renderer was exercised end-to-end against an
  authenticated ObsNum 150818 software-QA fixture adapted only to the current
  protocol identity. It produced 49 pages across six PDFs: 16 detailed
  detector/crossing pages, 12 all-crossing residual-atlas pages, 14 weighted
  residual-footprint pages, five objective-profile pages, one model-adequacy
  page, and one standard-map context page. Every page was rendered and visually
  inspected; the output checksum manifest verified. This fixture validates the
  software and layout only and is not retained as new scientific evidence.
- The final validation-package `SHA256SUMS` manifest verifies every retained
  file.
- The final staged diagnostic diff passes `git diff --cached --check`.

The 32 pre-existing owner bundle files remain untracked and untouched. They
are why no global clean-worktree claim is made.
