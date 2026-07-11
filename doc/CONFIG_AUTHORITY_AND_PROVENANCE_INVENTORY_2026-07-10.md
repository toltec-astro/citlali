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
the remaining RTC policy reads are migrated.

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
