# Citlali Baseline Harness

This directory contains lightweight utilities for recording and comparing
Citlali reduction outputs during the structural refactor.

These tools do not run Citlali and do not require Unity access from Codex. The
intended workflow is:

1. Run a reduction in the real validation environment.
2. Run `summarize_outputs.py` on the completed reduction output directory.
3. Save the generated JSON manifest with the validation notes.
4. After a refactor change, run the same reduction again and compare manifests
   with `compare_manifests.py`.

Use the local Python environment requested by the repo instructions:

```bash
$HOME/tolteca/bin/python tools/baseline/summarize_outputs.py \
  --case science_naive_noise_off \
  --output-dir /path/to/redu00 \
  --git-sha 376e0022 \
  --branch codex/structural-refactor \
  --config-file /path/to/70_reduce.yaml \
  --command "citlali /path/to/70_reduce.yaml" \
  --manifest-out /path/to/science_naive_noise_off.baseline.json
```

Then compare a later run:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_manifests.py \
  /path/to/science_naive_noise_off.baseline.json \
  /path/to/science_naive_noise_off.refactor.json \
  --ignore-sha256
```

`--ignore-sha256` is useful when files contain creation timestamps or other
volatile metadata. Without it, the comparator reports full-file checksum
changes in addition to structured FITS/netCDF/table summaries.

## Deterministic Refactor Gate

For structural refactor work, use a deterministic reduction mode as the first
behavior-preservation gate:

- `parallel_policy: seq`
- `n_threads: 1`
- same input data, APTs, config, and Citlali command for the protected baseline
  checkout and the refactor checkout

OpenMP runs are still useful for performance and stress testing, but current
OMP reductions have run-to-run drift. Until that is fixed, the one-thread `seq`
case is the reliable functional gate for the refactor branch.

Generate manifests for each completed reduction:

```bash
$HOME/tolteca/bin/python tools/baseline/summarize_outputs.py \
  --case point_152389_citlali_seq_redu02 \
  --output-dir /path/to/2026-refactor/point/citlali/reduced/redu02 \
  --manifest-out /tmp/point_152389_citlali_seq_redu02_manifest.json

$HOME/tolteca/bin/python tools/baseline/summarize_outputs.py \
  --case point_152389_refactor_seq_redu02 \
  --output-dir /path/to/2026-refactor/point/refactor/reduced/redu02 \
  --manifest-out /tmp/point_152389_refactor_seq_redu02_manifest.json
```

Then compare them with the deterministic policy wrapper:

```bash
tools/baseline/compare_deterministic_manifests.sh \
  /tmp/point_152389_citlali_seq_redu02_manifest.json \
  /tmp/point_152389_refactor_seq_redu02_manifest.json
```

The wrapper sets `--ignore-sha256`, `--atol 2e-8`, and `--rtol 1e-10`, and it
ignores volatile paths, mtimes, byte sizes, run labels, and log line counts. It
still compares structured FITS, netCDF, CSV, and ECSV summaries. Override the
tolerances with `CITLALI_BASELINE_ATOL` and `CITLALI_BASELINE_RTOL`, or pass
additional `compare_manifests.py` arguments after the two manifest paths.

## Product Triage Reports

Use `compare_reduction_products.py` when you want a fast, reduction-aware
summary of the latest output products rather than a strict deterministic
manifest gate. It resolves `latest` `reduNN` directories, compares matching
FITS/netCDF/table products, reports missing or extra products, and ranks numeric
array/column differences by absolute and fractional size.

For the Phase 1 complete-point acceptance gate, compare every TOD array and
fail on missing, changed, or skipped required items. Profiling elapsed time is
volatile and must be explicitly excluded:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  /path/to/baseline/reduNN \
  /path/to/candidate/reduNN \
  --mode point \
  --include-timestream \
  --max-array-elements 0 \
  --exclude citlali_profile.ecsv \
  --strict
```

Strict exit codes are `2` for product-set differences, `3` for skipped items,
and `4` for changed items. The ordinary triage mode continues to report all
findings while returning success.

Pointing validation against the latest downloaded refactor run:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  --base-root /Users/gwilson/work_toltec/local_data/2026-refactor \
  --mode point \
  --baseline-redu latest \
  --candidate-redu latest \
  --report-out /tmp/point_product_compare.md
```

Beammap validation once the long run has landed:

```bash
$HOME/tolteca/bin/python tools/baseline/audit_reduction_run.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/beammap/refactor/reduced \
  --expected-mode beammap \
  --expected-label refactor

$HOME/tolteca/bin/python tools/baseline/compare_reduction_audits.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/beammap/citlali/reduced \
  /Users/gwilson/work_toltec/local_data/2026-refactor/beammap/refactor/reduced \
  --expected-mode beammap \
  --report-out /tmp/beammap_audit_compare.md

$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  --base-root /Users/gwilson/work_toltec/local_data/2026-refactor \
  --mode beammap \
  --baseline-redu latest \
  --candidate-redu latest \
  --report-out /tmp/beammap_product_compare.md
```

After processed-timestream provenance is enabled, require a valid versioned
sidecar on a single run with `--require-processed-provenance`. When comparing
against an older accepted baseline that predates the sidecar, require it only
on the candidate:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_audits.py \
  /path/to/accepted/reduNN \
  /path/to/candidate/reduNN \
  --expected-mode point \
  --baseline-label refactor \
  --candidate-label refactor \
  --require-candidate-processed-provenance
```

Raw-timestream provenance is required per observation after its production
publication boundary is enabled. Require it on a single candidate with
`--require-raw-provenance`, or against an older accepted baseline with:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_audits.py \
  /path/to/accepted/reduNN \
  /path/to/candidate/reduNN \
  --expected-mode point \
  --baseline-label refactor \
  --candidate-label refactor \
  --require-candidate-raw-provenance
```

The audit verifies every matching provenance file. This matters for science
reductions, which contain one timestream-output and one raw-timestream
provenance sidecar per observation. Raw semantic checks require an initialized
and completed observation plus available, nonnegative scan and required-write
counts. Flagged-sample and dynamic-notch counts remain explicitly unavailable
until their lifecycle owners are migrated.

For processed provenance, structural validation is followed by semantic
checks: resolution records must agree with requested/effective cleaner,
source-mask, weighting, and fruit-loop state; realized source protection and
iteration counts must agree with the effective plan. A disagreement fails the
audit even when the YAML schema is complete.

Science coadd triage, with an explicit baseline/candidate pair when the latest
directories are not the intended comparison:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/science/citlali/reduced/redu13 \
  /Users/gwilson/work_toltec/local_data/2026-refactor/science/refactor/reduced/redu09 \
  --mode science \
  --top 30 \
  --json-out /tmp/science_product_compare.json \
  --report-out /tmp/science_product_compare.md
```

By default, the product triage skips logs, configs, `learning_iter_*.csv`, and
`*_timestream.nc` files so the report stays focused on map/table/diagnostic
products. Pass `--include-timestream` or explicit `--include`/`--exclude` globs
for deeper diagnostics.

Before running a deep product comparison, use `audit_reduction_run.py` as a
cheap identity and completion check. It reads only the low-level config, log,
and file inventory, so it can quickly confirm that a reduction wrote under the
intended `citlali` or `refactor` tree and summarize coarse timing such as
mapmaking, PTC diagnostics sidecar writing, APT table writing, fit-QC writing,
and split map output.

Use `compare_reduction_audits.py` before the heavier product comparison when
both OG and refactor runs are present. It checks both run identities, completion
markers, blocking log records, stable product inventory counts, and timing
deltas without opening large FITS/netCDF arrays. Any error-level or more severe
log record blocks a successful audit; warnings remain informational. The profile
sidecar `citlali_profile.ecsv` is reported separately and does not count as a
stable product mismatch.

## Files

- `run_manifest_template.yaml`: human-fillable run record template for Unity or
  other validation notes.
- `validation_record_template.md`: Markdown validation note template for one
  baseline or comparison run.
- `summarize_outputs.py`: walks an output directory and writes a JSON manifest
  with file metadata, SHA-256 checksums, and optional structured summaries for
  FITS, netCDF, CSV, ECSV, and logs.
- `compare_manifests.py`: compares two JSON manifests and reports missing
  files, changed metadata, changed checksums, and changed structured summaries.
- `compare_deterministic_manifests.sh`: wrapper around `compare_manifests.py`
  with the standard deterministic refactor gate policy.
- `audit_reduction_run.py`: fast preflight audit for one completed `reduNN`
  directory or reduced root; reports path identity, completion markers, product
  inventory, and coarse timing without reading large array payloads.
- `compare_reduction_audits.py`: compares two audit summaries, including
  expected labels, completion status, stable product inventory counts, profile
  sidecars, and coarse timing deltas.
- `compare_reduction_products.py`: reduction-aware product triage report for
  latest/direct `reduNN` pairs, with FITS/netCDF/table numeric differences.
- `validate_validation_ledger.py`: validates required identity, config hash,
  completion, comparison, and accepted-difference fields in the checked-in
  `validation/accepted_runs.json` ledger.
- `examples/tiny_reduction/`: a fake tiny output directory for checking the
  tools without a Citlali reduction.
- `examples/tiny_manifest.json`: an illustrative manifest shape for the tiny
  example.

## Optional Dependencies

The summarizer always records file size and SHA-256 checksums using the Python
standard library. Additional structured summaries are enabled when these
packages are available:

- `astropy` for FITS and ECSV summaries
- `netCDF4` for netCDF summaries
- `numpy` for array statistics

If an optional dependency is unavailable, the manifest records a warning and
continues.

## Tiny Example

Generate and compare a manifest for the fake tiny output directory:

```bash
$HOME/tolteca/bin/python tools/baseline/summarize_outputs.py \
  --case tiny_example \
  --output-dir tools/baseline/examples/tiny_reduction \
  --manifest-out /tmp/tiny_manifest.generated.json \
  --skip-content-hash

$HOME/tolteca/bin/python tools/baseline/compare_manifests.py \
  /tmp/tiny_manifest.generated.json \
  /tmp/tiny_manifest.generated.json
```

The comparator should report `manifests match`.

The product triage tool can also be checked against the tiny fixture:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  tools/baseline/examples/tiny_reduction \
  tools/baseline/examples/tiny_reduction
```

# Beammap Scientific Equivalence

`compare_beammap_scientific_equivalence.py` applies the scientific-owner
accepted, scale-aware Beammap gate in
`validation/profiles/beammap_scientific_equivalence_v1.json`. It requires exact
detector identities, flags, and split-FITS product sets, then checks APT
quantities and per-detector signal/weight/kernel RMS differences against the
versioned profile.

```bash
$HOME/tolteca/bin/python \
  tools/baseline/compare_beammap_scientific_equivalence.py \
  /path/to/beammap/citlali/reduced/reduNN \
  /path/to/beammap/refactor/reduced/reduNN \
  --json-out /tmp/beammap_equivalence.json \
  --report-out /tmp/beammap_equivalence.md
```

The command exits nonzero when any bound is exceeded. Use the generic
`compare_reduction_products.py` for exact run-to-run determinism and product
inventory checks; this profile is specifically for accepted OG/refactor
scientific equivalence and must not be relaxed without scientific-owner review.

# Science Scientific Equivalence

`compare_science_scientific_equivalence.py` applies the accepted science gate
in `validation/profiles/science_scientific_equivalence_v2.json`. It requires
exact FITS/netCDF product sets and integer diagnostic state, keeps a strict
raw-map bound, and separately checks the owner-approved Wiener-filtered map,
PTC weight, detector-median, and remaining floating diagnostic bounds. The v1
profile remains available for historical ledger records.

```bash
$HOME/tolteca/bin/python \
  tools/baseline/compare_science_scientific_equivalence.py \
  /path/to/science/citlali/reduced/reduNN \
  /path/to/science/refactor/reduced/reduNN \
  --json-out /tmp/science_equivalence.json \
  --report-out /tmp/science_equivalence.md
```

The profile is an OG/refactor scientific-equivalence gate. Continue to use the
generic comparator for exact same-build determinism checks.
