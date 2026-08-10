# SCI-ALIGN-001 Lissajous pointing diagnostic

This package freezes a read-only test of whether retained bright Lissajous
pointings favor a scalar map-time lag, empirical map-axis direction-sign
terms, or their joint form.  It does not alter Citlali, rerun a reduction, or
prescribe a timing/pointing correction.

## Frozen observation selection

The selection contains both immediately bracketing 3C273 pointings for every
analyzed 3C273 beammap with a complete retained standard-trial PTC timestream
and PPT.  No velocity-sector centroid was measured before this selection and
protocol were written.

| Beammap | Pointings before, after |
|---:|:---|
| 131925 | 131920, 131926 |
| 133543 | 133542, 133544 |
| 135397 | 135396, 135398 |
| 136279 | 136278, 136280 |
| 148670 | 148669, 148671 |
| 150819 | 150818, 150820 |
| 151126 | 151125, 151127 |
| 151600 | 151599, 151601 |
| 151950 | 151949, 151951 |
| 152451 | 152450, 152452 |
| 152882 | 152881, 152883 |

All 22 pointings are 3C273 observations and populate all eight fixed
velocity-angle sectors.  Sixteen have a1100 PPT S/N at least 50.  The cohort
spans mean telescope elevation 22.109 to 71.870 degrees.  The two pointings
around beammap 148670 and the two around 150819 are labeled anchors, but no
row is omitted from the primary run on the basis of S/N or a directional
result.

`selected_pointings.json` binds every exact PTC and PPT path and SHA-256.
`selected_pointings.csv` is the human-readable mirror.
`frozen_protocol.json` fixes the sector geometry, mapping, fit, and model
definitions.  `SHA256SUMS` protects those three files and this README.

## Diagnostic

The tool is
`tools/diagnostics/analyze_sci_align_001_lissajous_pointing.py`.

For a1100 detectors retained with `apt_flag == 0`, it:

1. computes telescope map-plane velocity independently within every retained
   scan, excluding inter-scan gaps;
2. assigns samples to eight fixed velocity-angle sectors above 5 arcsec/s;
3. reproduces Citlali's elevation-rotated detector pointing relation and
   detector-grouped nearest-pixel naive accumulation on a 2-arcsec grid;
4. fits a common Gaussian-core centroid to each sector map; and
5. compares equal-weight descriptive fits for:
   - a constant centroid;
   - `centroid = intercept + tau * map_velocity`;
   - independent `sign(vx)` and `sign(vy)` map-axis terms; and
   - the joint lag plus sign model.

The direction-sign terms are empirical map-space parameters.  They do not by
themselves identify encoder backlash, secondary motion, or another physical
mechanism.  Sector-centroid covariance and scan-block bootstrap uncertainty
are not yet available, so BIC and residual-RMS comparisons are descriptive,
not formal model selection.  The analysis does not establish a universal
correction or the upstream origin of any fitted lag.

## Reproduction

Inventory/freeze:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sci_align_lissajous_mpl \
XDG_CACHE_HOME=/tmp/sci_align_lissajous_cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/analyze_sci_align_001_lissajous_pointing.py inventory \
  --standard-trial-root \
    /Users/gwilson/work_toltec/local_data/2026-ENG-hero-multiyear-pointings-v1/diagnostics/standard_trial \
  --output validation/sci_align_001_lissajous_pointing_2026-08-10
```

Run the frozen cohort:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sci_align_lissajous_run_mpl \
XDG_CACHE_HOME=/tmp/sci_align_lissajous_run_cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/analyze_sci_align_001_lissajous_pointing.py run \
  --selection-dir \
    validation/sci_align_001_lissajous_pointing_2026-08-10 \
  --output /path/to/new/output
```

`--obsnum` may be repeated for an identity-bound subset.  `aggregate` can
rebuild the corpus table and summary PDF from a complete set of per-pointing
results without re-reading PTC data.

Each observation receives checksum-protected sector centroids, model results,
maps, a result manifest, and a two-page PDF.  The corpus root receives
`corpus_model_results.ecsv`, `corpus_model_summary.pdf`, `manifest.json`, and
`SHA256SUMS`.

## Local validation

On 2026-08-10:

- four synthetic tests recovered exact injected scalar-lag, axis-sign, and
  joint parameters;
- all 22 frozen observations completed with eight successful sector fits;
- the reconstructed full centroid for anchor pointing 150818 agreed with its
  retained a1100 PPT centroid to 0.021 arcsec in two dimensions;
- two complete executions produced identical numeric per-observation results
  and identical checksum-protected corpus tables and summary PDFs; and
- the complete corpus summary and representative sector-map PDFs were
  rendered to PNG and visually checked for clipping, overlap, and map-fit
  pathologies.

The first scientific interpretation should retain the 22-map scope and the
uncertainty limits above.  The smallest strengthening step is a whole-scan
block bootstrap of sector centroids and fitted model parameters; it should
precede a causal claim or correction proposal.
