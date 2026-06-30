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
