# Citlali Baseline Harness

This directory contains lightweight utilities for recording and comparing
Citlali reduction outputs during the structural refactor.

## Profile-Driven Validation

The normal acceptance entry point is `validate_reduction.py`. It resolves an
immutable baseline through `validation/accepted_runs.json` and the named
profile in `validation/validation_profiles.json`, then runs:

1. the completed-run and required-provenance audit;
2. an exact comparison of the merged low-level Citlali YAML;
3. the profile-pinned scientific product contract; and
4. the profile-pinned numerical product comparator and tolerances.

List the active profiles:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_reduction.py --list-profiles
```

List the prepared Phase 5 successor profiles:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_reduction.py \
  --list-preparing-profiles
```

Validate a downloaded reduction and retain the delegated JSON results:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_reduction.py \
  /path/to/candidate/reduNN \
  --profile phase4-science-152390-152392-v1 \
  --output-dir /tmp/citlali-science-validation \
  --json-out /tmp/citlali-science-validation.json \
  --report-out /tmp/citlali-science-validation.md
```

Use `--baseline` when the ledger's accepted local path is unavailable on the
current host. A zero exit status means all four gates passed. This command
delegates scientific comparison to the existing mode-specific tools; it does
not introduce a second comparison implementation.

Preparing profiles have no accepted ledger baseline and therefore always
require an explicit `--baseline`. A successful check is reported as
`prepared gates pass (not accepted)`. The prepared Phase 5 profiles compare
scientific config exactly while applying the versioned
`tolteca-native-project-bindings-v1` policy to machine/project path prefixes.
The bound file or directory identity remains exact, and no path is generally
ignored.

Validate and render the successor-epoch readiness record with:

```bash
$HOME/tolteca/bin/python tools/baseline/phase5_readiness.py
```

Add `--require-ready` only at promotion time. The command returns nonzero while
any fixture or global blocker remains.

Add `--verify-fixtures` to rerun all four declared profile checks in one
command. Delegated JSON and Markdown reports are retained under
`--fixture-output-dir`; the command fails if an actual gate result differs from
the checked readiness record.

The product-contract gate reads the candidate's generated `citlali_o*.yaml`.
Products controlled by explicit output switches are required when requested
and forbidden when disabled. Unconditional mode products remain required and
genuinely optional diagnostics may be absent. The versioned registry in
`validation/product_contracts.json` also records family identity, coordinate
frame, axes, units policy, indexing, missing-value policy, and write-failure
policy. See
`doc/PHASE4_SCIENTIFIC_PRODUCT_CONTRACT_2026-07-16.md` for the contract scope
and known metadata debt.

Canonical baseline APT v1 is a separate, standalone artifact contract. It is
not referenced by an accepted validation profile and does not change the
historical generic Beammap APT checks. Validate one producer-owned ECSV and its
exact adjacent completion receipt manually with:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_product_contract.py \
  /path/to/beammap_apt.ecsv \
  --artifact-contract apt-prod-001-canonical-baseline-apt-v1
```

The artifact mode reads the ECSV and `<apt>.ecsv.sha256` receipt as one bound
pair, requires exact canonical serialization, independently recomputes the
semantic and occurrence-bound envelope SHA-256 values, and verifies the exact
byte SHA-256/count. A zero status means the pair is conformant to the
unactivated contract; it is not production-profile acceptance. Static
validation confirms the visible completion marker and its binding, but the
producer tests remain the authority for receipt-last publication ordering,
failure cleanup, raw-source equality, and unchanged science. Historical APTs
remain historical/test-only and are neither migrated nor admitted through this
mode.

## Observation-Specific Canonical APT v1

APT-PROD-002 adds a separate, still-unactivated successor contract for making
one observation-specific canonical APT from a verified immutable Beammap
baseline. The persisted product is exactly one `*.apt.ecsv` plus its adjacent
`*.apt.ecsv.sha256` completion receipt. The complete observation target and
match relation remain independently canonicalized logical records, but they
are embedded in the final APT metadata: v1 does not publish `.target.ecsv`,
`.relation.ecsv`, a bundle manifest, or JSON APT data.

The supported machine boundary is the Citlali CLI's strict JSON-line protocol:

```bash
printf '%s\n' \
  '{"protocol":"citlali-canonical-apt-protocol-v1","request_id":"describe-1","operation":"describe-baseline-v1","payload":{"baseline_ecsv":"/path/to/baseline.ecsv"}}' \
  | build/bin/citlali --canonical-apt-contract-v1
```

The option must be the only CLI argument. It consumes exactly one
LF-terminated JSON request and returns exactly one JSON response line. Its
three operations are:

- `describe-baseline-v1`: reread and verify a canonical baseline ECSV/receipt
  pair and return the complete typed descriptor and immutable reference;
- `issue-observation-apt-v1`: verify the baseline and bound raw/KMP sources,
  materialize caller-supplied occurrence-scoped target and relation facts,
  construct and reread the final ECSV, then publish it no-replace with the
  receipt visible last; and
- `validate-observation-apt-v1`: reread an already published final
  ECSV/receipt pair against its verified baseline and return the complete
  typed target, relation, output, identity, and transport result.

The request supplies legitimate observation values, selected match facts, and
provenance. It does not supply Citlali's schemas, field catalog, digests,
output-local keys, or final occurrence. Exact request and response examples,
including generalized per-field source selection, live in
`test_validate_product_contract.py`; the normative boundary is documented in
[`../../doc/CANONICAL_APT_OBSERVATION_V1.md`](../../doc/CANONICAL_APT_OBSERVATION_V1.md).

The Python artifact selector intentionally rejects all APT-PROD-002 contract
IDs. Target and relation are not standalone artifacts, and final issuance and
validation are available only through the versioned Citlali protocol. A
successful protocol validation proves conformance to an unactivated contract;
it is not validation-profile admission, accepted-run evidence, downstream
ingestion authority, or production activation.

Accepted profiles are versioned snapshots. Future intentional product changes
create successor validation epochs with a predecessor comparison and recorded
scientific rationale; they do not rewrite or loosen an old profile.

Post-baseline scientific changes are tracked separately in
`validation/intended_science_changes.json`. The ledger distinguishes inherited
baseline behavior from later imports and links each import to its source and
integration commits, expected numerical or schema effect, affected modes and
product families, and accepted validation evidence. Validate Git ancestry,
claimed patch identity, evidence references, and product-family references
with:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_science_change_ledger.py
```

A later OG revision used as a comparator is not automatically an imported
change. Future intentional algorithm, default, or product changes must add a
ledger entry and, where applicable, a successor validation epoch.

## Controlled Performance Campaigns

Use `run_performance_case.py` on Unity to wrap one baseline or candidate
reduction with GNU Time. It attaches portable command, host/CPU, config, input,
runtime-policy, log, stage-profile, I/O, wall-time, and peak-RSS evidence to the
new `reduNN` directory. Use `analyze_performance_campaign.py` after the paired
runs are downloaded. The analyzer enforces warmups, alternating order, one
host/CPU-affinity signature, matched config/input policy, binary identity,
complete measurements, and the campaign budgets.

The checked-in Beammap template is
`validation/performance/beammap_campaign_template.json`. The complete protocol
and Unity command sequence are in
`doc/PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md`. Do not use historical or
concurrent runs to fill a controlled campaign retrospectively.

## Historical Beammap Corpus

The historical Beammap corpus is a population census, not a controlled paired
campaign. Use the same `run_performance_case.py` wrapper for each observation,
with one frozen release binary. The wrapper's campaign fields provide unique
run bookkeeping; the corpus manifest supplies the authoritative observation
identity.

For example, from one TolPROJ Beammap project directory on Unity:

```bash
python "$HOME/work_toltec/citlali_dev/citlali_refactor/tools/baseline/run_performance_case.py" \
  --campaign-id beammap-historical-release-census-v1 \
  --case-id beammap-148670 \
  --role candidate \
  --phase measured \
  --pair-index 0 \
  --build-type Release \
  --citlali-executable /work/toltec/citlali_dev/citlali_refactor/build/bin/citlali \
  --reduced-root reduced \
  --output performance/beammap-148670.json \
  -- tolteca reduce
```

After downloading the reduction, add its attached metadata to a copy of
`validation/performance/beammap_corpus_template.json`:

```json
{
  "observation_id": 148670,
  "metadata": "3c273/reduced/redu00/performance_run.json",
  "comparisons": [
    {
      "label": "previous-release",
      "metadata": "historical/148670/performance_run.json"
    }
  ]
}
```

Comparisons are optional and must describe the same observation. Analyze the
current population with:

```bash
$HOME/tolteca/bin/python tools/baseline/analyze_beammap_corpus.py \
  /path/to/beammap_corpus.json \
  --json-out /tmp/beammap_corpus_result.json \
  --report-out /tmp/beammap_corpus_result.md
```

The analyzer verifies the observation number from Beammap provenance, requires
one current record for every expected observation, and reports runtime/RSS/I/O
distributions, workload-normalized rankings, workload relationships, identity
groupings, stage summaries, and explicit same-observation ratios. Unlike
observations are never turned into implicit pairs, and ranked observations are
not discarded as outliers. See
`doc/BEAMMAP_CORPUS_PERFORMANCE_CENSUS_PLAN_2026-07-23.md` for the governing
interpretation and completion policy.

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

OOF uses Citlali's `pointing` execution type but remains a distinct validation
intent and directory. Compare paired OOF products with:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  --base-root /Users/gwilson/work_toltec/local_data/2026-refactor \
  --mode oof \
  --baseline-redu latest \
  --candidate-redu latest \
  --report-out /tmp/oof_product_compare.md
```

Beammap validation once the long run has landed:

```bash
$HOME/tolteca/bin/python tools/baseline/audit_reduction_run.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/beammap/refactor/reduced \
  --expected-mode beammap \
  --expected-label refactor \
  --require-mapmaking-provenance \
  --require-post-processing-provenance \
  --require-beammap-provenance

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
counts. The audit pairs raw and timestream-output sidecars by observation,
rejects missing completion sidecars, cross-checks scan counts, and validates
resolved sample-rate/downsample relationships. Flagged-sample and dynamic-notch
counts remain explicitly unavailable until their lifecycle owners are migrated.

For processed provenance, structural validation is followed by semantic
checks: resolution records must agree with requested/effective cleaner,
source-mask, weighting, and fruit-loop state; realized source protection and
iteration counts must agree with the effective plan. A disagreement fails the
audit even when the YAML schema is complete.

Mapmaking provenance is required after its production publication boundary is
enabled. Require it on a single run with `--require-mapmaking-provenance`, or
only on a newer comparison candidate with:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_audits.py \
  /path/to/accepted/reduNN \
  /path/to/candidate/reduNN \
  --expected-mode beammap \
  --baseline-label refactor \
  --candidate-label refactor \
  --require-candidate-mapmaking-provenance
```

The mapmaking audit checks that requested and effective grouping and units
agree with their resolution records, that automatic/fallback decisions are
internally consistent, and that the run completed with the expected
mapmaking-executed state. Version-2 sidecars additionally require a contiguous
sequence of identified, completed observations; finite positive pixel sizes;
consistent logical map-product counts; and matching observation/coadd
completion cardinality. Historical version-1 sidecars remain readable.

Coadd and noise-product provenance can be required on a newer candidate while
retaining an accepted baseline that predates those sidecars:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_audits.py \
  /path/to/accepted/reduNN \
  /path/to/candidate/reduNN \
  --expected-mode science \
  --baseline-label refactor \
  --candidate-label refactor \
  --require-candidate-coadd-provenance \
  --require-candidate-noise-products-provenance
```

The coadd audit cross-checks requested, effective, and realized activation,
map cardinality, and required-write completion. The noise audit additionally
checks the fixed random-number-generator identity, realization counts for
observations and coadds, optional product and realization-write cardinality,
and consistency with mapmaking and coadd provenance.

Pointing provenance records the five-key source policy, its resolved fit and
header-radius behavior, and per-observation fit cardinality. Require it only on
a newer pointing candidate when comparing with a pre-sidecar baseline:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_audits.py \
  /path/to/accepted/reduNN \
  /path/to/candidate/reduNN \
  --expected-mode point \
  --baseline-label refactor \
  --candidate-label refactor \
  --require-candidate-mapmaking-provenance \
  --require-candidate-pointing-provenance
```

The pointing audit verifies requested/effective policy resolution, contiguous
observation identities, map and fit counts, required output completion, and
agreement with mapmaking provenance. Raw pointing fits require mapmaking but
remain independent of optional filtered-observation and coadd outputs.

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

By default, the product triage skips logs, configs, `learning_iter_*.csv`,
`learning_housekeeping_iter_*.csv`, and
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

For candidates that include the observation-resolved astrometry contract, add
`--require-astrometry-provenance` to `audit_reduction_run.py`, or
`--require-candidate-astrometry-provenance` to
`compare_reduction_audits.py`. These gates validate authority, configured
offsets, effective application mode, observation coverage, and realized
installation/interpolation counts without reading large products.

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
- `run_performance_case.py`: Unity-side GNU Time wrapper and portable reduction
  evidence capture for one warmup or measured campaign run.
- `analyze_performance_campaign.py`: validates a paired performance protocol
  and reports median/IQR runtime, peak RSS, I/O, and stage timing ratios.
- `analyze_beammap_corpus.py`: validates a heterogeneous Beammap release census
  and reports population, workload, stage, and explicit same-observation
  evidence without treating unlike observations as repeats.
- `validate_validation_ledger.py`: validates required identity, config hash,
  completion, comparison, and accepted-difference fields in the checked-in
  `validation/accepted_runs.json` ledger.
- `validate_product_contract.py`: validates historical profile-bound reduction
  contracts and, through its separate `--artifact-contract` selector, an
  unactivated canonical baseline APT v1 ECSV/receipt pair.
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
artifact-local detector row/UID membership, flags, and split-FITS product sets,
then checks APT
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

# Fruit-Loop Convergence Evidence

`analyze_fruit_loop_convergence.py` measures consecutive saved fruit-loop
iterations offline. It verifies low-level configuration and raw-coadd product
identity, evaluates every configured array independently, and includes
learning-state stability from the Citlali logs. Candidate stopping rules remain
explicitly non-production evidence.

```bash
export NGC4449_ROOT=/path/to/NGC4449
$HOME/tolteca/bin/python \
  tools/baseline/analyze_fruit_loop_convergence.py \
  validation/fruit_loops/ngc4449_full_spatial_learning_study.json \
  --json-out /tmp/ngc4449-convergence.json \
  --report-out /tmp/ngc4449-convergence.md
```

The checked NGC4449 study and its interpretation are documented in
`doc/FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md`. The tool does not authorize
production early stopping or replace a scientific validation profile. The
checked result under `validation/fruit_loops/` records the observed sequence
without making it a portable input fixture.
