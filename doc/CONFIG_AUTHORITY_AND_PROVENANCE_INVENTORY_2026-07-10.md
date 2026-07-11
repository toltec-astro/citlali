# Config Authority And Provenance Inventory - 2026-07-10

This document starts Phase 2 of the adopted refactor roadmap without changing
runtime behavior. The machine-readable source of truth is
`tools/config/config_authority_inventory.json`; its structural checks live in
`tools/config/validate_config_authority_inventory.py`.

The earlier exposure policy answers **which controls users should normally
see**. This inventory answers a different question: **which representation is
allowed to control execution**.

## Authority Contract

The target data flow is strictly one way:

```text
merged requested YAML -> typed requested config -> effective execution plan
                                            |                 |
                                            v                 v
                                  temporary legacy adapter   realized metadata
```

- Raw YAML may be read only at a declared loading boundary.
- Typed requested config preserves user intent after parsing and validation.
- An effective plan records defaults, normalization, activation, and policy
  resolution without mutating the requested values.
- Execution consumes the effective typed plan.
- Legacy processor objects may temporarily receive typed values. They must
  never populate or override typed config.
- Realized metadata records what actually ran and what products were emitted.

This contract prevents the common failure mode where requested values,
normalized values, and runtime side effects are all stored in one mutable
object and become impossible to distinguish after a reduction.

## Current Census

The inventory groups the current surface into 13 authority domains. This is a
subsystem census, not yet a leaf-by-leaf schema claim.

| Current state | Meaning | Domains |
| --- | --- | ---: |
| `typed-authoritative` | Execution policy is already typed and has no declared legacy adapter. | 4 |
| `typed-authoritative-with-adapter` | Typed policy is authoritative but one compatibility target remains. | 1 |
| `mixed-adapter` | Typed state exists, but execution still depends materially on legacy processor/map objects. | 7 |
| `external-boundary` | Configuration belongs to an external subsystem whose schema boundary must be recorded. | 1 |

Every domain currently has incomplete provenance. Runtime and product logs
already provide pieces of effective/realized state, but there is no uniform
requested/effective/realized record yet.

### Runtime Authority Checkpoint - 2026-07-11

Runtime now has separate requested, effective, and realized typed state.
Execution-time thread setup, worker-farm sizing, runtime control flow, and
output reduction-type routing consume the effective plan. Direct access to the
mutable runtime mirror remains only at config construction boundaries. Focused
tests prove that requested values remain unchanged when effective policy
differs and that realized OMP, Eigen, and FFTW state is recorded.

The versioned `citlali-runtime-provenance-v1` schema is emitted atomically as
`runtime_provenance.yaml` in each reduction directory. It records every
requested runtime value, effective runtime values and thread plan, and realized
OMP, Eigen, FFTW, parallel-policy, and reduction-type state. Publication is a
required output: write failure propagates and removes the temporary artifact.

Unity point reduction `redu27`, built from `0dc08555`, closes the runtime gate.
It has zero serious log issues, exact pre-existing products relative to
`redu26`, and the intended additional `runtime_provenance.yaml`. The requested
and effective values match the unchanged merged YAML; realized state records
six OMP threads, one Eigen thread, initialized FFTW with one plan thread,
parallel policy `omp`, and reduction type `pointing`. Runtime provenance is now
`complete`.

### Timestream Output Authority Checkpoint - 2026-07-11

The first `timestream-core-output` slice moves TOD output shape to typed
authority. RTC outer-buffer allocation, RTC/PTC mini/full layout, outer-context
padding, summaries, and NetCDF metadata now derive from
`TimestreamOutputConfig`. Divergence tests established that typed modes win
over the former processor mirrors before those mirrors were removed.

Typed selection already produces realized scan-to-row mappings and output
cardinalities in `TodOutputState`. Scan-index construction now also receives
typed chunk mode, value, and force policy directly instead of reading the
telescope mirror.

The early telescope-file chunk-validity check also receives typed chunking
policy directly. The former processor output-mode/context and telescope
chunking fields have been removed; parser and NetCDF append boundaries now
receive typed state explicitly.

The versioned `citlali-timestream-output-provenance-v1` schema is now written
atomically as `timestream_output_provenance.yaml` in each observation
directory. It records requested stream and chunking controls, effective output
type and selected chunks, and realized scan-to-output mappings, cardinalities,
and registered TOD files. Per-observation placement preserves correct state for
multi-observation science reductions.

Unity point reduction `redu28`, built from `5411a82e`, validates this schema.
The unchanged merged config produces 12 effective RTC and PTC chunks, realized
as contiguous output rows `0..11` with both registered TOD files present. The
run has zero serious log issues and exact pre-existing products relative to
`redu27`. After validation, the write-only adapters were removed. The local CLI
and test build, all 229 tests, and the complete eight-case config preflight pass.
The domain is now `typed-authoritative` with complete provenance.

### Raw Timestream Authority Checkpoint - 2026-07-11

The first `raw-timestream` slice moves downsample enablement, requested factor
or frequency, and anti-alias filter validation to `RawTimeChunkConfig`.
Effective sample-rate preflight now follows typed policy even when the legacy
RTC mirror disagrees. A factor derived from requested frequency is written to
both typed config and the RTC downsampler because the numerical downsampler
still consumes that execution adapter. The domain remains `mixed-adapter` until
the remaining RTC policy reads are migrated. Filter and kernel enablement now
also drive observation setup, map allocation, TOD/FITS product shape,
coaddition, and beammap orchestration from typed config. Flux-unit and
extinction setup decisions use the same authority. Numerical filter, kernel,
and calibration objects remain processor-owned execution state.
Source-protection activation is now derived from typed raw/processed policy and
reduction type, recorded as typed realized state, then copied to processor
objects only for numerical execution. Learned raw masks consume that typed
state. FITS event-mask metadata and RTC diagnostic impulsive-product shape also
derive from typed flagging config.
Model-protected PTC line-audit enablement, model requirements, selected notch
families, iteration count, frequency overrides, and edge-guard policy now use
typed raw config. The low-level RTC notch methods still accept their existing
processor options type, which remains a numerical adapter. RTC diagnostic, TOD,
and summary provenance serialize typed despike, event-mask, filter, and
line-audit settings; processor edge-context values are retained only where they
represent realized sample counts.
Duplicate-tone rejection now reads the typed minimum frequency separation, and
RTC diagnostic source-bandwidth ratios read typed FIR enablement and cutoff.
FITS and TOD tau metadata use typed extinction enablement while retaining the
processor calibration object for the numerical atmospheric calculation.
RTC diagnostic and RTC TOD diagnostic schemas now receive typed downsample and
impulsive-capture policy explicitly. This removes the remaining external raw
product-shape decision from processor mirrors; internal RTC algorithms still
consume numerical adapter structures.

### Processed Timestream Authority Checkpoint - 2026-07-11

The first `processed-timestream` slice moves fruit-loop lifecycle and map-path
policy to `TimestreamFruitLoopsConfig`: enablement, effective iteration count,
retained-iteration layout, initial model path/type, previous-iteration path,
and learning source-model availability. Beammap and disabled-loop policy
normalization updates typed effective state, then synchronizes iteration count
and retention into `PTCProc` for the existing loop implementation. Divergence
fixtures now establish that typed policy controls orchestration.

The next slice makes typed fruit-loop enablement and weight-recomputation
policy authoritative for processed-timestream model subtraction/add-back,
noise-weight retention, final noise-map population, and beammap adaptive-gate
setup. Runtime model buffers and numerical operations remain processor-owned;
typed configuration decides whether those operations run. A divergence test
keeps the legacy processor mirrors intentionally contradictory and verifies
that they cannot override typed policy.

Fruit-loop interpolation override selection and runtime-policy logging also
read the typed policy directly. The selected interpolation mode is copied to
`PTCProc` as realized numerical state; processor mirrors no longer participate
in choosing or reporting that policy.

Fruit-loop configuration metadata in TOD NetCDF headers, PTC diagnostic
NetCDF files, and map FITS headers now serializes the typed effective policy.
Pointing policy warnings also use typed fruit-loop enablement and iteration
count. Detector-specific fitted vectors remain processor-owned runtime results;
only configuration values and array flux limits moved to typed authority.

The compact PTC-diagnostic metadata block now follows the same rule for
cleaner selection, adaptive cleaning, correlation penalties, busy-row
suppression, and second-pass event policy. Diagnostic arrays continue to
describe realized processor state; `CONFIG.*` entries describe typed policy.

Full TOD NetCDF and map FITS cleaning metadata now follows that boundary too:
cleaner mode, Marchenko-Pastur selection, adaptive selection, correlation
penalties, busy-row suppression, and second-pass policy are typed. Per-array
removed-eigenmode counts remain realized processor results by design.

Weight-selection, cutoff, validation, and raw/processed inverse-variance
metadata in TOD, PTC-diagnostic, and FITS products now uses typed policy. The
PTC diagnostic sampling-window duration remains processor-owned because it is
not yet represented in typed configuration; this exception is explicit at the
serializer boundary rather than an implicit policy fallback.

Optional PTC TOD diagnostic schema presence now uses typed processed policy
for second-pass, correlation-grouping, correlation-penalty, busy-row, and
adaptive-cleaner blocks. The schema boundary no longer asks `PTCProc` whether
configured diagnostic families should exist.

Learning diagnostics, learned PTC sample-mask application, and learned
mapmaking detector exclusion now use typed second-pass source-protection state
and thresholds. The synchronized `PTCProc` fields remain internal numerical
adapter inputs, not external orchestration authority.

A focused one-way adapter now copies typed fruit-loop effective policy into
`PTCProc`. The current legacy parser still seeds the typed model, but numerical
code receives a canonical typed-to-processor synchronization step. This is the
replacement seam for moving fruit-loop parsing directly into the typed loader
without simultaneously rewriting PTC algorithms.

Core fruit-loop fields now have a direct typed reader: enablement, retained
iterations, path/type, mode, S/N threshold, array flux limits, and iteration
limit are read from YAML into `TimestreamFruitLoopsConfig` before the one-way
adapter runs. Expert numerical fields remain mirrored during the staged parser
extraction; the combined legacy parser is still invoked for those domains.

The direct reader now also covers the expert fruit-loop surface: local-noise
geometry, adaptive support, weight feedback, center retention, interpolation,
legacy centering, and post-addback weight policy. The legacy combined parser
still executes as a compatibility parser, but typed values are reread directly
and then applied through the one-way adapter.

The companion lexical census currently finds 611 direct config-access
expressions across 30 files:

| Boundary | Files | Accesses |
| --- | ---: | ---: |
| Legacy processor/parser | 8 | 488 |
| Typed loader | 10 | 32 |
| External schema | 5 | 29 |
| Legacy entrypoint | 4 | 55 |
| CLI | 3 | 7 |

No access is currently unclassified. The high legacy-parser count is a useful
migration denominator, but it does not mean 488 execution-time fallbacks:
these reads are concentrated in processor `get_config` implementations called
during loading. The next operational work should move one coherent parser
domain at a time, then prove that execution uses its typed effective plan.

## Migration Order

1. **Freeze the inventory and loading boundaries.** Add a domain whenever a
   new config family appears; do not introduce execution-time raw-YAML reads.
2. **Introduce immutable effective plans at policy boundaries.** Start with
   runtime, output selection, and mapmaking activation because they already
   have focused policy helpers and broad mode coverage.
3. **Convert legacy processor reads by domain.** Raw and processed timestream
   processing require explicit read inventories before mirrored fields can be
   removed. Preserve the one-way typed-to-legacy adapter during migration.
4. **Persist provenance.** Emit requested config identity, effective policy,
   and realized product/cardinality summaries with stable schema versions.
5. **Remove adapters only after validation.** Pointing, beammap, science, and
   OOF coverage must show equivalent products and expected metadata before a
   legacy field is deleted.

## Migration Gates

A domain is complete only when all of the following are true:

1. Its raw-YAML reads are confined to the declared loader.
2. Its typed requested type covers every supported setting in that domain.
3. Normalization and cross-setting policy produce a separate effective plan.
4. Execution code reads the effective plan, not raw YAML or legacy policy
   fields.
5. Requested, effective, and realized states have stable machine-readable
   provenance.
6. Mode-appropriate product validation passes before the compatibility adapter
   is removed.

## Near-Term Scope

Phase 1 Beammap and science validation has landed. Runtime authority migration
is now operational behind focused tests and a rollback boundary; persistence
of its provenance record is the next runtime-domain step. Other domains remain
subject to the same one-domain-at-a-time migration and validation gates.

Validate the inventory with:

```bash
$HOME/tolteca/bin/python tools/config/validate_config_authority_inventory.py
$HOME/tolteca/bin/python tools/config/audit_config_authority_reads.py \
  --fail-on-review
```
