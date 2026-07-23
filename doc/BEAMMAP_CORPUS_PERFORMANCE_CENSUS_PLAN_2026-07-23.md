# Beammap Corpus Performance Census Plan - 2026-07-23

## Purpose

The planned re-reduction of approximately 50 historical Beammap observations
will span the real operational variation in scan count, detector population,
map count, iteration behavior, observing conditions, output volume, and shared
storage load. This project turns those runs into a versioned performance
census for future Citlali releases.

The census is not a controlled repeated-trial campaign and is not a Phase 5
integration gate. Unlike observations must not be treated as interchangeable
timing samples. The census describes a population, relates cost to recorded
workload, preserves same-observation comparisons where they exist, and
identifies observations that warrant focused investigation.

This project changes no Citlali runtime, numerical algorithm, build behavior,
or product contract.

## Existing Foundation

`tools/baseline/run_performance_case.py` already captures one reduction as
`citlali-performance-run-v1`, including:

- executable, dependency, host, storage, config, input, and runtime identity;
- GNU Time wall time, CPU use, peak RSS, filesystem counters, faults, and
  context switches;
- Citlali log time, serious log counts, and profile-stage totals; and
- the produced reduction directory and attached evidence.

The corpus reuses that record. It does not add census instrumentation to
Citlali's hot path.

`tools/baseline/analyze_performance_campaign.py` remains the tool for a
controlled same-observation baseline/candidate campaign. The corpus analyzer
has different statistical semantics and must not weaken that protocol.

## Manifest Contract

The checked corpus manifest records:

- a corpus and release identity;
- the complete expected Beammap observation list;
- one current-release performance record per observation;
- explicit historical or future comparison records nested beneath that same
  observation;
- the expected build type, binary version token, dependency revisions, and
  runtime-policy requirements; and
- any notes needed to explain invalid or operationally unusual runs.

Observation identity is not inferred from filenames. The analyzer verifies the
manifest observation number against `beammap_provenance.yaml`. A comparison is
accepted only beneath the same verified observation.

Paths are relative to the manifest when possible. Large reduction products
remain outside Git.

## Workload Evidence

The analyzer reads existing Beammap provenance to recover, when available:

- detector count;
- map count;
- scan count;
- completed iteration count;
- summed active-map iterations;
- summed mapmaking passes;
- optional detector-TOD sample extent; and
- local output file count and byte volume.

Manifest-provided workload overrides are allowed only for evidence unavailable
from current provenance and remain visible in the result. They do not silently
replace conflicting provenance.

## Analysis Rules

For the current-release population, report:

- completeness and rejected-record reasons;
- distributions of Citlali time, external wall time, peak RSS, filesystem
  counters, output volume, and important profile stages;
- runtime normalized by scan count and by the recorded detector/map/iteration
  workload;
- simple correlation and linear-fit diagnostics between workload descriptors
  and runtime/RSS when at least three varied observations are available;
- highest absolute and normalized-cost observations; and
- host, runtime-policy, config-digest, executable, and dependency groupings.

For each explicit same-observation comparison, report current/comparison ratios
for runtime, RSS, I/O, and shared profile stages. Do not pool these ratios with
unpaired observations.

Outliers remain in the census. Rankings and fit residuals are investigation
signals, not automatic exclusions or proof of a code regression. Shared VAST
traffic and node conditions remain possible explanations.

## Failure And Completeness Policy

The analyzer reports an incomplete census when:

- an expected observation has no current record;
- current observation identities are duplicated or disagree with provenance;
- a current run failed, lacks GNU Time evidence, contains serious log records,
  or lacks required runtime/RSS metrics;
- current records do not use the declared binary/dependency/build identity;
- a required single runtime policy is not preserved; or
- an explicit comparison refers to another observation.

A failed observation stays recorded with its reason and is rerun or explicitly
disposed. It is not deleted after the result is seen.

The corpus is complete only when every expected observation has one accepted
current-release record. Comparisons are additive evidence and are not required
for observations that have no genuinely comparable historical run.

## Deliverables

1. `validation/performance/beammap_corpus_template.json`
2. `tools/baseline/analyze_beammap_corpus.py`
3. focused analyzer tests
4. operator instructions in `tools/baseline/README.md`
5. a filled external manifest and reports when the historical re-reduction
   begins

## Exit Gate

The tooling phase is complete when:

- synthetic tests prove completeness, identity, pairing, grouping,
  workload-normalization, and failure behavior;
- the baseline-tool suite passes;
- an example partial manifest produces an explicit incomplete result rather
  than an accidental success; and
- the living status and retained-debt register identify the census as ready
  for data collection, not already complete.

Retained-debt item D14 closes only after the real corpus has been run and its
report accepted.

## Tooling Verification - 2026-07-23

- Eight focused tests pass for complete heterogeneous populations, explicit
  same-observation comparison, cross-observation rejection, missing expected
  observations, binary identity, campaign identity, conflicting workload
  overrides, serious-log failure, grouping, and workload relationships.
- All 117 baseline-tool tests pass.
- Ruff check and formatting gates pass for the new analyzer and tests.
- The empty checked template produces an intentional incomplete verdict with
  only `expected observation list is empty`.
- A read-only extraction against the downloaded accepted Beammap recovered
  5,234 detectors, 5,234 maps, 198 scans, three iterations, 15,702 active-map
  passes, and the local 34-file/12.88-GB output footprint from existing
  provenance and filesystem metadata.
