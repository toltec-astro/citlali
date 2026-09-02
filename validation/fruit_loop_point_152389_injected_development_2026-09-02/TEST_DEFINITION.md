# FRUIT centered-source injection development test: pointing 152389

Status: **completed exploratory-development test; not qualification, candidate
ranking, or a stopping-rule decision**

Test ID: `SCI-FRUIT-POINT-152389-INJECT-CENTER-100MJY-R0.1`

## Question

Starting from the completed iteration-0 checkpoint for the local pointing
observation, how much of a known compact source does the current FRUIT path
recover in iterations 1 and 2?

This is a paired test. The control and injected reductions use the same raw
data, APT, effective science configuration, iteration-0 restart state, local
Citlali executable, and sequential execution policy. They differ only in the
output directory and whether the diagnostic source is enabled. The injected
map minus the control map is the measured synthetic-source response.

## Frozen inputs and settings

- Observation: `152389`, development-only copy under
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389`.
- Source configuration SHA-256:
  `dc0df89b706f1af9f32d747861f8c23975ded7cb0cf5c706110e7a96126d5909`.
- Restart: fresh single-thread reference
  `attempt-04/reference/reduced/redu00/citlali_restart_checkpoint.nc`, SHA-256
  `85419a82e050ae5d3685313abf413e239ffab272d75d336bd43633fc845dfdf8`.
- Executable: `build/bin/citlali`, reported version
  `sci-noi-v0.1-stage-a-22-g92d174630`, SHA-256
  `5a6f6741ee81c0b78ff718d2d4b6674f3b9a27a476a6d4a29105d3b68b319c38`.
- Execution: local `--grppiex seq` with `runtime.n_threads: 1`. A fresh
  uninterrupted three-iteration reference and both restarted branches use
  this same execution setting.
- Optional full RTC/PTC timestream outputs are disabled in the fresh reference
  and both branches by `LOCAL_IO_SUPPRESSION.yaml`. Diagnostic sidecars remain
  enabled. The exact-control check below tests restart identity under this
  common setup; it is not an output-enabled/output-disabled comparison.
- Injection start: absolute zero-based FRUIT iteration 1.
- Saved paired iterations: 1 and 2.
- Injected amplitude: `100 mJy/beam` in each of `a1100`, `a1400`, and `a2000`.
- Injected shape and position: the pristine per-detector kernel already
  produced by this pointing reduction. This is the central pointing-source
  position; it is not an independently placed off-center source.
- New run root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-r0.1`.

The common 100 mJy/beam amplitude is a deliberately simple first probe. It was
selected before opening injected results. If it proves too weak or too strong,
that outcome remains part of this record and a differently identified test is
required; this test is not silently retuned.

## Predeclared checks

1. The iteration-1 control signal, kernel, and weight arrays must be exactly
   equal to the preserved uninterrupted iteration-1 reference. Failure makes
   the pair uninterpretable until explained.
2. For each array and iteration, form
   `transfer = injected signal map - control signal map`.
3. At the known central injection position, report the fitted transfer
   amplitude divided by 100 mJy/beam.
4. Report a full-map least-squares projection of the transfer onto the
   injected run's `kernel_I`, divided by 100 mJy/beam.
5. Report transfer FWHM relative to the fitted kernel FWHM and transfer/kernel
   centroid separation.
6. Report transfer-map change from iteration 1 to iteration 2, and any
   control/injected kernel or weight differences.

No pass/fail recovery band is assigned. The purpose is to establish a working
truth-referenced measurement and expose the present recurrence's behavior,
not to qualify it or choose an iteration.

## Attempt record

Attempt 1 retained the source configuration's optional RTC/PTC NetCDF output.
It loaded the exact restart successfully, then terminated with `SIGSEGV` while
HDF5/NetCDF output was active during scan processing. No iteration product was
completed and no injected result was opened. Its incomplete output remains at
the frozen run root under `control/reduced`.

Attempt 2 uses a distinct subdirectory and the common I/O-suppression overlay
described above. It completed, but its iteration-1 control differed from the
earlier uninterrupted reference at relative RMS `1.9e-12` to `1.1e-11` in
signal, `6.9e-14` to `3.8e-13` in kernel, and `1.0e-12` to `8.6e-12` in
weight. Those are minute numerical-ordering differences, but they fail the
predeclared bitwise test. No injected branch was run.

Attempt 3 therefore creates a fresh uninterrupted reference and its restarted
pair with one thread throughout. This is a local reproducibility repair, not a
science-setting change. Its exact-control comparison remains mandatory and is
not replaced by a numerical tolerance.

Attempt 3 did not create that intended reference: its base YAML still named a
restart, and a later YAML `null` did not erase the earlier scalar during config
merging. The run consequently continued iterations 1 and 2 and was not used as
a reference. Attempt 4 uses the original no-restart source YAML as its base,
then applies the reference and single-thread overlays. No injected branch was
run in attempts 1--3.

Attempt 4 completed. Its fresh reference products carry absolute iterations
0, 1, and 2. The restarted control is bitwise equal to uninterrupted iteration
1 for `signal_I`, `kernel_I`, and `weight_I` in all three arrays. Only after
that check passed was the injected branch run.

The first generated report used a global map-maximum search to initialize the
Gaussian diagnostic. In two iteration/array rows it selected a brighter
off-source subtraction artifact rather than the source whose position was
known by construction. Those Gaussian entries were invalid and were not
interpreted. The diagnostic now restricts both the peak search and fitted
centroid to 25 arcsec around the frozen central injection position. A
regression test includes a brighter distant artifact. This post-run repair
makes the stated source identity explicit; it does not change the pair, truth
amplitude, comparison, thresholds, or scientific classification.

The completed measurements and interpretation are in [`README.md`](README.md).
The exact rows are in
[`injected_source_iteration_metrics.csv`](injected_source_iteration_metrics.csv),
and [`manifest.json`](manifest.json) hashes the executable, configurations,
restart, every map used, and the compact outputs.

## Scientific boundary

The diagnostic source is added to the calibrated PTC signal after RTC
processing, learned RTC/PTC masks, bad-detector removal, and duplicate-tone
removal. It is added immediately before subtraction of the previous FRUIT map,
then passes through the PTC cleaner, the FRUIT noise/add-back path, weighting,
and mapmaking. On the next iteration, the accepted injected response can enter
the carried FRUIT model.

Therefore this test measures the compact-source response of the
PTC-cleaning/FRUIT/mapmaking recurrence in the real observation context. It
does **not** measure attenuation or distortion introduced earlier by RTC
filtering, RTC despiking, calibration, or detector-selection operations, and
it does not test an off-center or extended source. The positive source is
co-located with the real pointing source, so the pair preserves any nonlinear
interaction with that source rather than claiming a blank-field response.
