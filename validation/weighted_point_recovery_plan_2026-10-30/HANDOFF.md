# Five-case 152418 recovery test — prepared for Grant

The diagnostic interface and local trial are prepared. No restoration, reduction,
new source fit on saved observation data, or Unity operation was performed by
Codex. Existing reductions remain intact. Read-only readiness reports:
`Ready; no reductions launched.` The `run-001` output directory does not yet exist.

## Run from your Mac shell

```bash
bash /Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-source-recovery-152418-20261030/02_redu.sh --run
```

Omit `--run` for a read-only readiness check. Keep the terminal open. The launcher
uses a subshell, so its strict Bash settings do not leak into your interactive
shell. It freezes numerical library threads to one and defaults to compact output.
It checks runtime and input hashes before creating a fresh output directory.
There is no automatic retry, overwrite, resume, additional case or cycle11.

The previous run deliberately removed its intermediate calibrated timestreams.
Your command first uses the original executable/configuration to recreate those
inputs in the new trial, then checks **16,565 payload files across11 networks**
against their original hashes. A mismatch stops before REF. Original receipts are
copied into a new parent directory, and verified payloads are hard linked there;
the old reduction is never changed. This is one prerequisite CAL invocation,
followed by at most54 native map-cycle invocations. Optical health is reused,
not relearned. The recorded calibration and telescope/APT associations are fixed.

## Cases and fixed settings

| Case | Intervention | Initial context |
|---|---|---|
| REF | No input/model intervention; exact saved-map parity required at every cycle | Own M0 |
| CORE+ | +10% compact saved-M10 Gaussian, tapered30–36″ | Own M0 |
| CORE− | Exact negative of CORE+ | Own M0 |
| SHOULDER | +5% peak, doubled sigmas, +0.5 M0 major FWHM in azimuth, same taper | Own M0 |
| SEED | REF data and M0; shoulder added only to first feedback operand | Shared REF M0/context |

Every case includes a1100, a1400 and a2000. Noise-weighting rules,2″ containing-pixel
grid, ranks3/4/6, coverage0.15,10% optical-response cut, ordinary source guard and
connected BASE feedback remain fixed. Numerical weights relearn from each current
residual; they are not frozen coefficients. Each trajectory ends at M10 or failure.
SEED has no separate M0 computation and its extra component is not forced later.

The campaign has a four-hour wall deadline, a20 GiB stage-boundary output ceiling,
and a50 GiB free-space reserve. Native calls receive the remaining deadline and
Python work is also bounded. One in-flight stage can cross the storage boundary;
the post-stage/terminal checks then stop the run and prevent a completed claim.
The saved upstream stage took75.26s and the earlier paired run1130.62s; neither is
a measured runtime for this test. No observation run was used to estimate cost.

## Results to inspect afterward

All new output is under:
`.../pointings-source-recovery-152418-20261030/run-001/`.

- `RUN.json`: complete/stopped status, invocation count, exact cycle receipts,
  ordinary source fits/guard decisions and measured-product bindings.
- `RESTORATION.json`: exact original-CAL payload verification and new parent roots.
- `<case>/cycle-00` through `cycle-10`: three native maps and plans; feedback
  candidates only for M0–M9. SEED's shared M0 is described in `SEED/M0-SHARED.json`.
- `<case>/cycle-*/measurements.json`, `differences.npz`: fixed40″ response gains,
  distortion, lost/gained support, source fits, ellipse, frozen template/apertures,
  signed profiles,45–60″ background statistics and candidate changes.
- `SEED/seed/`: separately labelled, one-use feedback operand and provenance.
- `IDEAL_CONTROLS.json` and the three data cases' `IDEAL_COMPARISON.json`: at most
  nine ideal M10 map additions, absolute measurement increments/errors. Ratios
  are deliberately not invented for small or undefined increments.
- `SUMMARY.json` and `*-compact-even-*.npz`: all-cycle compact sign asymmetry/even
  maps and final-cycle drift. A stopped branch retains its available evidence.

Per-array native plans verify the actual REF weighted MAP operation on compact
and shoulder inputs, including actual coefficients/normalization and independent
coordinate association. They retain original support and parent hashes plus the
projected source-free overlap counts/power. No heavy detector-time tracing is added.

Frozen source definitions and full pixel footprints are under `inputs/` before
launch. Required geometry is fully supported in the saved REF M0 on all arrays;
subsequent support loss makes that measurement unavailable rather than shrinking
its comparison domain. Source/background and aperture-constant terms remain
separate. No signed-difference source fits are performed.

## Software and readiness

Reviewed implementation: `d0d3eeb7981975a2f43f1a55878668410aa6c20b`.
Runtime: `.../software/configured-point-recovery-d0d3eeb79/`.
Native SHA256: `47d31683e06ac8af3cd70a7bcfbd50c315577c01218d660cba54aa937ca77229`.
Runtime-manifest SHA256: `fd619c0d42ecaa613e82f07731d5b859b995235fbde424ef0bba9a38d67d17f9`.
Trial-manifest SHA256: `229ec59da325db0a7949e08a4fec75d150a3f834ddd2f30756f40fbc403abead`.
The preparation/readiness record authenticates761 bindings. The launcher binds
the exact manifest digest; changing a case/configuration requires new preparation.
The recorded review file is also a bound dependency in this worktree and must
remain available. Original project inputs/archives and the local Python environment
are deliberate external dependencies; this is not a portable installation.

Verification:280 focused Python tests passed before the two review repairs;
then all five runner tests, including both new terminal-limit regressions, passed.
Native recovery tests exercise the actual M1/M2 switch. Latest full CTest:
1372 passed, one pre-existing disabled lifecycle test. Baseline207 tests plus137
subtests and full required configuration preflight passed. Independent exact-source
review: pass with recorded limitations, both findings closed. Clean native rebuild
reports `gd0d3eeb79`; the final snapshot and read-only readiness check passed afterward.

This is supplemental local evidence. The existing build uses external dependencies
(including the previously recorded dirty KIDs dependency); it is not Unity Spack
qualification. Runtime metadata records external Python/library limitations.

## Stop and scientific interpretation

REF mismatch stops the matrix. Input mismatch, unsupported projection, failed
cleaning, unavailable source feedback or an existing validity violation stops the
case before another feedback application. Resource violations stop the entire run.
Unaffected declared cases may finish after an ordinary branch failure.

There are **no newly invented scientific pass thresholds**. Completion measures
response and model memory; it does not certify flux recovery, absolute calibration,
M1 optical quality, continuous-beam sampling, faint SCIENCE, infinite-iteration
behavior or permanently imposed source wings. All products remain diagnostic and
calibration-ineligible. After Grant reports completion, assess the bounded matrix
once and decide whether to retain the current policy or propose one specific remedy;
do not automatically tune or add runs.
