# Reproduction of the fixed failed-gate experiment

This records the executed sequence, not authorization to repeat it now. Use a
clean Git checkout of the exact stage source below, the existing Python
environment, and the preserved `provenance/bin/citlali`. Source archives retain
the contents but do not contain Git metadata required by these commands.
All consumed original paths and expected hashes are in freeze/prospective.json.
Execution verifies these originals before consumption. Never regenerate their
expected identities from current file contents.

Use /Users/gwilson/tolteca/bin/python, with OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, MPLBACKEND=Agg and writable
task-specific MPLCONFIGDIR/XDG_CACHE_HOME. Run from the checkout root. OUT below
denotes a fresh absolute destination; BIN denotes the preserved executable.
REVIEW denotes the independent review record for the exact stage source and
new prospective hash. Reviews in this bundle document the original execution;
they must not be edited or represented as approval of a different freeze.

1. Preparation and original calibration source:
   `ee54efb6d32759003d4249f3a4ad90dbeb9c10d9`.
   Native SHA256:
   `5823281906ab4aed249469d387f4d433bcbac7a4837036b9c371ac6ac2b8ad06`.

   ```sh
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_corr_admission.py prepare --output "$OUT" --binary "$BIN"
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_corr_admission.py calibration --output "$OUT" --review "$REVIEW"
   ```

   The preparation requires a matched source/version and a new independent
   input/method review before calibration. This original calibration source
   omitted failed-scene maxima when forming intermediate thresholds. Those
   values are invalid diagnostics and may never launch a trajectory. Preserve
   raw trials; apply step 2 before validation. Each phase has exactly 1024
   fields per array with the frozen SHA256-derived PCG64 seeds.

2. Reviewed failure-accounting disposition and validation source:
   `0a651e88a07cccd3431d54bc76d7b457df640e84`.

   ```sh
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_corr_admission.py calibration-disposition --output "$OUT"
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_corr_admission.py validation --output "$OUT" --review "$REVIEW"
   ```

   This marks all six thresholds unavailable, without replacement draws,
   retuning or recalibration. Validation repeats the source procedure on its
   independent 1024 fields per array, with the first 256 reused at 1.25 noise
   amplitude. Failed maxima, threshold-dependent rates and confidence bounds
   remain unknown. The new-policy launch gates must remain blocked.

3. Saved evidence and three BASE REF controls, source:
   `bdd3d5645dc294fcaf84dbcc84d84e8c360f2aa2`.

   ```sh
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_corr_feedback.py screen --output "$OUT"
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_corr_feedback.py BASE --output "$OUT" --review "$REVIEW"
   ```

   This reuses the unchanged frozen native executable and original M0 products.
   Thirty BASE native applications must match same-stage original maps,
   measurements, candidates, applied operands, means, bases, cleaned samples
   and causes. The m10 candidates remain unapplied. Do not run CORR/DIAG REF
   or dependent signal/seed branches under these failed gates.

4. Reporting and closeout: exact revisions are recorded under
   provenance/SOURCE_AND_BINARY.json. The first report used
   `9a2c2e8418c99d54d8453bdc29408c3e6a196e57`; the final report corrects only the
   availability-panel label. First-render artifacts are preserved separately.

   ```sh
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/report_point_corr_admission.py --output "$OUT"
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/close_point_corr_admission.py inventory --output "$OUT"
   /Users/gwilson/tolteca/bin/python tools/timestream_successor/close_point_corr_admission.py seal --output "$OUT" --review "$FINAL_REVIEW"
   ```

   Reporting regenerates existing validation seeds for fixed-row covariance
   checks and reuses recorded fit parameters; it adds no independent draw or
   fit. The report directory must be fresh. Inventory requires the successful
   gate logs named in the closeout command; final sealing follows independent
   realized numerical/figure/provenance review and rechecks consumed originals.

Validation commands are `python -m pytest tools/timestream_successor -q`,
`python -m pytest tools/baseline -q`, the 34 native/CLI CTests, and
`python tools/config/run_config_preflight.py --require-all`, using the required
Python executable above. Successor tests use POINT_FROZEN_APPLY_BINARY and
CITLALI_JINC_DIAGNOSTIC pointing to the matching build helpers. The baseline
suite expects build/bin/citlali; during this execution a temporary build link
was installed and removed in try/finally. Native source has not changed since
ee54efb6d. Existing build evidence is local AppleClang/cached-dependency
validation, not a Unity or Spack V2 replication.

No dependency installation, new science selections, additional independent
noise draws, new-policy feedback, m10 application, cycle 11, production/default
change, Unity access, merge or push is part of this record.
