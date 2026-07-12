# Processed Timestream Config Transition

This note defines the temporary `PTCProc` configuration boundary and its
finite removal gate. It applies to:

- `timestream.fruit_loops`
- `timestream.processed_time_chunk.clean`
- `timestream.processed_time_chunk.weighting`
- `timestream.processed_time_chunk.flagging`

## Current authority

The intended transitional sequence in `Engine::get_ptc_config` is:

1. Run the legacy `PTCProc::get_config` parser as a compatibility seed.
2. Mirror legacy values into the typed model only to retain compatibility
   defaults and behavior not yet independently resolved.
3. Read the low-level YAML directly into typed request fields.
4. Resolve context-free effective decisions in typed helpers.
5. Populate `PTCProc` through one-way typed-to-legacy adapters.

No processor value written in step 5 may subsequently overwrite the typed
request or effective plan.

`ProcessedTimestreamExecutionPlan` is the transitional in-memory shape for
this contract. It currently provides independent requested and effective
snapshots, typed effective-resolution records, and a separate realized-state
record. It is deliberately not wired into `Engine` or output yet: the existing
typed config remains execution authority until the matched beammap checkpoint
is reviewed and the plan can replace it in one bounded change. A partial plan
must not be serialized as complete processed-timestream provenance.

## State classification

### Requested

Values explicitly supplied by the merged low-level Citlali YAML, including
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

Remove `read_processor_config(ptcproc, ...)` and the PTC-to-typed mirrors only
after all of the following are true:

- every one of the 171 unique YAML paths currently read by
  `PTCProc::get_config` has a direct typed reader or an explicitly documented
  compatibility alias (currently enforced at 171/171 by
  `audit_processed_timestream_boundary.py`);
- focused tests cover typed-to-processor parity for fruit loops, cleaning,
  weighting, validation, correlation penalties, busy-row suppression, and
  second-pass flagging (currently exhaustive for every adapter assignment);
- tests cover every effective decision listed above, including absent-key and
  repeated-configuration behavior;
- the complete config preflight passes for pointing, OOF, beammap, and science;
- current matched pointing, beammap, and science reductions pass strict
  product comparison, with complete TOD comparison where applicable;
- accepted runs contain no unexpected errors and are recorded in
  `validation/accepted_runs.json`;
- requested, effective, and realized processed-timestream provenance is
  versioned and labeled without treating processor runtime state as the
  immutable request.

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
