# Local validation

Validation was run with `$HOME/tolteca/bin/python` and headless Matplotlib.

- `test_analyze_sci_align_001_lissajous_timestream.py`: 10 tests passed in
  45.411 seconds. The suite includes injected zero, negative, positive,
  static-offset, pure-hysteresis, and joint regimes; irregular scan sampling,
  flags, detector gains, baselines, correlated noise, wrap-safe interpolation,
  invariant common support, derivative agreement, and convergence gates.
- `py_compile`: estimator, test, and compact-result projection passed.
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
- The final validation-package `SHA256SUMS` manifest verifies every retained
  file.
- The final staged diagnostic diff passes `git diff --cached --check`.

The 32 pre-existing owner bundle files remain untracked and untouched. They
are why no global clean-worktree claim is made.
