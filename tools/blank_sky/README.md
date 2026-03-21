# Blank-Sky Residual Tools

This directory is for reusable analysis utilities that treat cleaned Citlali
timestreams as a blank-sky null experiment.

For these audits, the working assumption is:

- astrophysical signal in the field is negligible at the per-sample level
- cleaned residuals should look like independent Gaussian noise
- statistically significant non-Gaussianity, inter-detector coherence, scan
  coupling, or spectral lines are contamination until proven otherwise

## Current Tool

`blank_sky_null_audit.py`

Audit a cleaned timestream netCDF file, usually a `*_ptc_timestream.nc`, and
write:

- a detailed per-scan, per-network CSV
- a network summary CSV
- a short markdown report ranking the most suspicious networks/scans

Metrics include:

- Gaussian tail excess at `|z| > 3, 4, 5`
- skewness and excess kurtosis of standardized residuals
- pairwise detector-correlation and top-mode metrics
- circular-shift surrogate null tests for coherence
- low/mid/high band common-mode power ratios
- strongest common-mode spectral line and frequency
- coupling of the common mode to `TelElAct`, `TelAzAct`, derivatives, and time
- simple scan-stationarity checks across quartiles

## Example

```bash
$HOME/tolteca/bin/python tools/blank_sky/blank_sky_null_audit.py \
  --nc-file /path/to/toltec_commissioning_science_151930_ptc_timestream.nc \
  --array a1100 \
  --networks 0,1,2,3,4,5 \
  --scans all \
  --utils-root ~/GitHub/toltec-data-product-utilities \
  --outdir /path/to/output_dir
```

For a fast triage pass, keep the defaults. For more stable surrogate-null
numbers, increase `--n-surrogates`.

`localize_detector_clusters.py`

Read one detector-cluster CSV from `analyze_timestream_correlations.py` plus the
matching PTC timestream file, then summarize and plot where the multi-detector
clusters live:

- focal-plane position from `apt_x_t/apt_y_t`
- sky-track centroid from `det_ra/det_dec`

Example:

```bash
$HOME/tolteca/bin/python tools/blank_sky/localize_detector_clusters.py \
  --ptc /path/to/toltec_commissioning_science_151930_ptc_timestream.nc \
  --detector-csv /path/to/obs151930_scan000_nw02_detectors.csv \
  --outdir /path/to/localize_scan000_nw02
```

`mp_mode_estimator.py`

Estimate an adaptive PCA cut depth from a fitted Marchenko-Pastur bulk model.
This is aimed at coherent/common-mode cleaning rather than impulsive glitches:

- robustly whiten detectors within each scan/network
- optionally band-limit the timestream before covariance estimation
- fit the bulk eigenspectrum with a scaled MP model
- count modes above the fitted upper edge `lambda_plus`

Example:

```bash
$HOME/tolteca/bin/python tools/blank_sky/mp_mode_estimator.py \
  --nc-file /path/to/toltec_commissioning_science_151930_rtc_timestream.nc \
  --array a1100 \
  --networks 0,1,2,3,4,5 \
  --scans all \
  --band-low-hz 0.05 \
  --band-high-hz 0.5 \
  --configured-k 18 \
  --outdir /path/to/mp_mode_estimate
```

Outputs:

- `mp_mode_estimate_detailed.csv`
- `mp_mode_estimate_summary_by_network.csv`
- `MP_MODE_ESTIMATE.md`

`non_gaussian_classifier.py`

Classify suspicious scan/network rows by likely failure family. This is aimed at
the practical question of what kind of contamination is left:

- impulsive / missed-spike-like
- step-like / level-shift-like
- narrowband / line-like
- coherent / common-mode-like

The tool works on RTC or PTC timestreams and writes:

- `non_gaussian_classifier_detailed.csv`
- `non_gaussian_classifier_summary_by_network.csv`
- `NON_GAUSSIAN_CLASSIFIER.md`

Example:

```bash
$HOME/tolteca/bin/python tools/blank_sky/non_gaussian_classifier.py \
  --nc-file /path/to/toltec_commissioning_science_151930_rtc_timestream.nc \
  --array a1100 \
  --networks 0,1,2,3,4,5 \
  --scans all \
  --utils-root ~/GitHub/toltec-data-product-utilities \
  --outdir /path/to/non_gaussian_classifier
```

`rtc_impulsive_slot_report.py`

Summarize the compact RTC impulsive-event capture products written by the new
RTC instrumentation. This is meant to answer the next concrete question after
the summary metrics: what do the top captured events actually look like?

For local-residual despike development, the report also tracks whether the
compact morphology gates saw each captured event as:

- a local raw candidate
- a local raw rejection because the event looked too broad or too step-like
- a local delta candidate
- an accepted local delta trigger
- a rejected candidate that looked too broad or too step-like

The tool reads the `rtc_impulsive_slot_*` variables from an `*_rtcdiag.nc`
sidecar when available, or from an RTC timestream file as fallback, and writes:

- `rtc_impulsive_slot_report_detailed.csv`
- `rtc_impulsive_slot_report_summary_by_network.csv`
- `RTC_IMPULSIVE_SLOT_REPORT.md`
- `rtc_impulsive_slot_gallery.png`

Example:

```bash
$HOME/tolteca/bin/python tools/blank_sky/rtc_impulsive_slot_report.py \
  --nc-file /path/to/toltec_commissioning_science_152524_rtcdiag.nc \
  --array a1100 \
  --networks 0,1,2,3,4,5 \
  --scans all \
  --outdir /path/to/rtc_impulsive_slot_report
```

`rtcdiag_survey_report.py`

Summarize RTC diagnostics across a whole `reduXX` tree. This is the lightweight
survey entry point and prefers `*_rtcdiag.nc` products automatically, falling
back to `*_rtc_timestream.nc` when needed.

It writes:

- `rtcdiag_survey_by_obsnum.csv`
- `rtcdiag_survey_by_obsnum_network.csv`
- `rtcdiag_survey_by_network.csv`
- `rtcdiag_survey_top_scan_network_rows.csv`
- `rtcdiag_survey_top_impulsive_slots.csv`
- `RTCDIAG_SURVEY_REPORT.md`

Example:

```bash
$HOME/tolteca/bin/python tools/blank_sky/rtcdiag_survey_report.py \
  --redu-dir /path/to/reduced/redu37 \
  --array a1100 \
  --networks 0,1,2,3,4,5 \
  --outdir /path/to/rtcdiag_survey_report
```

`rtcdiag_dash_app.py`

Interactive engineering dashboard for `rtcdiag` survey products. This is meant
for quick drilldown by obsnum, network, and scan while keeping `rtcdiag` as the
authoritative low-weight source.

Before first use, install Dash into the shared local venv:

```bash
~/toltec/bin/pip install --upgrade dash
```

Example:

```bash
~/toltec/bin/python tools/blank_sky/rtcdiag_dash_app.py \
  --redu-dir /path/to/reduced/redu40 \
  --array a1100 \
  --networks 0,1,2,3,4,5 \
  --host 127.0.0.1 \
  --port 8050
```

The app provides:

- obsnum severity table
- network summary table
- scan-by-network severity heatmap for the selected obsnum
- per-network scan trends for step and impulsive metrics
- top scan/network and top slot tables

`despike_diagnostic_report.py`

Build a readable reduction-local report that explains what the current RTC and
PTC despiking paths did on one `reduXX`. This is meant to bridge the gap
between survey tables and raw timestream debugging.

It writes:

- `ptc_second_pass_by_scan_network.csv`
- `ptc_second_pass_by_obsnum.csv`
- `DESPIKE_DIAGNOSTIC_REPORT.md`
- a small set of representative case PNGs

The report combines:

- RTC survey severity and mask activity
- PTC second-pass cluster counts and added-flag fractions
- representative accepted and busy-vetoed PTC cases with RTC/PTC plots

Example:

```bash
~/toltec/bin/python tools/blank_sky/despike_diagnostic_report.py \
  --redu-dir /path/to/reduced/redu65 \
  --array a1100
```

`rtc_line_audit.py`

Audit RTC timestreams for persistent narrowband contamination using masked
Welch PSDs computed only from contiguous good samples (`flags == 0`). This is
meant for RTC outputs taken after despike/flagging but before later
filter/downsample stages.

It produces two separate outcomes:

- scan/network line clusters that may justify a global or network-level notch
- recurrent detector-local lines that may justify bad-detector flagging instead

It writes:

- `rtc_line_audit_scan_network.csv`
- `rtc_line_audit_detector_peaks.csv`
- `rtc_line_audit_bad_detectors.csv`
- `RTC_LINE_AUDIT.md`

Useful notes:

- use `--output-scans` when you want the real `output_scan_index` values from
  Citlali rather than the internal zero-based scan index
- the default `--max-det 128` is intentional for interactive work; use
  `--max-det 0` only when you want all detectors and can afford the runtime
- broad shared lines should stay in the notch table, while detector-local
  recurrent lines should rise to the bad-detector table

Example:

```bash
~/toltec/bin/python tools/blank_sky/rtc_line_audit.py \
  --redu-dir /path/to/reduced/redu01 \
  --array a1100 \
  --networks all \
  --output-scans 73,77,82 \
  --max-det 128 \
  --outdir /path/to/rtc_line_audit
```

`rtc_line_family_report.py`

Build a reduction-level summary from an existing `rtc_line_audit` directory and
plot representative PSDs for:

- broad shared line families that may justify notch filtering
- recurrent detector-local line candidates that may justify bad-detector flagging

It writes:

- `rtc_line_family_summary.csv`
- `rtc_line_bad_detector_representatives.csv`
- `rtc_shared_line_psd_gallery.png`
- `rtc_bad_detector_psd_gallery.png`
- `RTC_LINE_FAMILY_REPORT.md`

Example:

```bash
~/toltec/bin/python tools/blank_sky/rtc_line_family_report.py \
  --redu-dir /path/to/reduced/redu01
```
