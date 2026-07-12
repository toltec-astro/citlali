# Processed Timestream Config Transition

This note defines the temporary `PTCProc` configuration boundary and its
finite removal gate. It applies to:

- `timestream.fruit_loops`
- `timestream.processed_time_chunk.clean`
- `timestream.processed_time_chunk.weighting`
- `timestream.processed_time_chunk.flagging`

## Current authority

The authoritative sequence in `Engine::get_ptc_config` is now:

1. Begin from deterministic typed defaults.
2. Read the low-level YAML directly into typed request fields.
3. Resolve context-free effective decisions in typed helpers.
4. Populate `PTCProc` through one-way typed-to-legacy execution adapters.

No processor value written in step 4 may subsequently overwrite the typed
request or effective plan.

`ProcessedTimestreamExecutionPlan` is the transitional in-memory shape for
this contract. It provides independent requested and effective snapshots,
typed effective-resolution records, and a separate realized-state record.
Following acceptance of the matched beammap checkpoint, it is now owned and
initialized by `Engine`; processed runtime accessors select the effective
snapshot once initialized. The root typed config is synchronized for
compatibility consumers that have not moved to the processed accessors.
Successful reductions publish this plan as versioned processed-timestream
provenance. Production no longer invokes the legacy parser or any PTC-to-typed
mirror.

For repeated reductions in one process, reset the complete plan from a fresh
request. Disabled sections retain the values supplied in the requested and
effective snapshots, while their activation flag prevents execution; stale
effective-resolution and realized state from a prior run is always cleared.
Do not reset individual subfields piecemeal.

Pure snapshot serializers cover all 171 frozen legacy fields and are enforced
by the boundary audit. Effective-resolution and realized-state components use
explicit availability markers. The versioned provenance schema composes these
components and was added only after the Engine wiring passed Unity validation.

The matched beammap prerequisite was accepted on 2026-07-12: refactor
`4b0126e7` `redu14` is exact against accepted refactor `redu11` and passes
`beammap-scientific-equivalence-v1` against OG `b83c8750` `redu01`. Engine
wiring was then implemented as one bounded authority change. Unity point
`86c47fa7` `redu34` is exact against accepted `redu33`, including all RTC/PTC
timestream arrays, so the authority change is accepted. This does not by itself
satisfy the provenance or legacy-parser-removal gates below.

## Versioned provenance

`processed_timestream_provenance.yaml` uses schema
`citlali-processed-timestream-provenance-v1` and is written atomically in the
reduction directory after successful completion. It contains:

- `requested`: the accepted canonical typed request after parsing,
  compatibility defaults, and canonical enum/group representation;
- `effective.config`: the processed configuration used by runtime accessors;
- `effective.resolutions`: explicit availability and decision records for
  cleaner mode, source-mask inheritance, weighting dependencies, fruit-loop
  interpolation, and iteration policy;
- `realized`: source-protection activation, completed iteration count, and
  convergence state.

The merged low-level YAML remains the source-level record. In particular,
absence versus an explicit value is represented by the corresponding
resolution record where the typed snapshot cannot itself retain YAML
presence. An uninitialized plan or any create/write/rename failure fails the
reduction; no partial provenance document is accepted.

Point provenance was accepted on 2026-07-12. `81020d46` `redu35` contains all
required v1 sections and availability records, and remains scientifically exact
against accepted `redu34`, including every RTC/PTC timestream array. Beammap and
science also exercise their mode-specific effective and realized records.

Beammap provenance was accepted on 2026-07-12. `50235fd6` `redu15` is exact
against accepted `redu14` across all comparable scientific products and records
the expected forced single fruit-loop iteration, inherited 15-arcsec source
mask, JINC interpolation, and one completed iteration. The final matched science
pair, OG `ffc6b907` `redu27` and refactor `50235fd6` `redu24`, passes science-
equivalence profile v2. Raw maps retain the strict `1e-8` bound and measure
`2.33e-11`; owner-accepted Wiener-filtered maps measure 0.986% against their
separate 1.5% bound; product sets and integer diagnostics are exact; PTC weight,
detector-median, and other diagnostic bounds pass. The science sidecar records
four effective and completed fruit-loop iterations, inherited source-mask
policy, JINC interpolation, and source-protection intent without falsely marking
ordinary science as source-aware. All cross-mode removal prerequisites are now
satisfied.

## State classification

### Requested

Canonical accepted values derived from the merged low-level Citlali YAML,
including compatibility defaults and parse-time canonical representation, for
cleaner selection and settings, weighting settings, validation policy,
correlation penalties, busy-row policy, second-pass flagging, source
protection intent, and fruit-loop policy.

### Effective

Context-free decisions derived before processing:

- cleaner subgroup names are lowercased, `network` becomes `nw`, unsupported
  groups are removed, and duplicates are removed while preserving order;
- absent `weighting.source_mask_radius_arcsec` inherits the effective cleaner
  mask radius;
- `weighting.type: validated` forces weight validation on;
- busy-row suppression is disabled unless second-pass local flagging is on;
- cleaner mode selection and legacy aliases resolve to one canonical typed
  representation.

### Realized

Observation- or iteration-dependent state must not be written back into the
request. Examples include source-protection activation, selected source
centers, fruit-loop interpolation and JINC execution settings, convergence,
accumulated validation factors, generated flags, learned groups, and output
diagnostic summaries.

## Legacy parser removal gate

The production calls to `read_processor_config(ptcproc, ...)` and
`seed_processed_timestream_config_from_legacy(...)` were removed only after all
of the following became true:

- every one of the 171 unique YAML paths currently read by
  `PTCProc::get_config` has a direct typed reader or an explicitly documented
  compatibility alias (currently enforced at 171/171 by
  `audit_processed_timestream_boundary.py`);
- focused tests cover typed-to-processor parity for fruit loops, cleaning,
  weighting, validation, correlation penalties, busy-row suppression, and
  second-pass flagging (currently exhaustive for every adapter assignment);
- tests cover every effective decision listed above, including absent-key and
  repeated-configuration behavior (a real value-initialized `PTCProc` snapshot
  now also matches the complete typed processed default snapshot);
- the complete config preflight passes for pointing, OOF, beammap, and science;
- current matched pointing, beammap, and science reductions pass strict
  product comparison, with complete TOD comparison where applicable;
- accepted runs contain no unexpected errors and are recorded in
  `validation/accepted_runs.json`;
- requested, effective, and realized processed-timestream provenance is
  versioned and labeled without treating processor runtime state as the
  immutable request.

The boundary audit now requires zero legacy parser, compatibility-seed, or
direct processed-mirror calls in `Engine::get_ptc_config`. The unused legacy
parser body remains temporarily as the frozen 171-path audit source. Moving
that path set to a standalone manifest and deleting the dead body is a later
mechanical cleanup, not a config-authority change.

OOF may reuse the pointing execution gate while that relationship remains an
explicit supported contract. Polarimetry requires its own gate before any
polarimetry-specific PTC behavior becomes typed-authoritative.

## Stop conditions

Do not:

- synchronize typed and processor state bidirectionally;
- add new YAML reads to `PTCProc`;
- classify observation-dependent values as requested config;
- remove the compatibility parser based only on compilation, unit tests, or a
  point reduction;
- redesign cleaner, weighting, or fruit-loop algorithms as part of this
  boundary migration.
