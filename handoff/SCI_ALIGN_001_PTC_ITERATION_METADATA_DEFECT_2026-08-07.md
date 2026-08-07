# SCI-ALIGN-001 confirmed engineering defect: Beammap PTC iteration metadata

## Disposition

This is a distinct confirmed Citlali engineering defect. It is not evidence
for or against the SCI-ALIGN-001 approximately 12 ms left/right timing offset,
and its repair does not change detector samples, timestamps, pointing,
filtering, cleaning, weights, or map accumulation. Retain this note for
SCI-ALIGN-001 closeout and route the lifecycle issue to the future
SCI-BEAM-001 inbox.

## Exact Unity trigger

The owner ran ObsNum 150819 as Unity Slurm job 62679409 with the locally built
binary whose SHA256 was
`b497122bdc02021229a31be4a46d9fc9657be4a1d2eec3538a87d4a4568b2237`.
The input configuration was
`sci_align_001_naive_full_ptc_singlepass_150819_2026-08-07/preparation/citlali_o150819_naive_full_ptc_singlepass.yaml`
with SHA256
`1a78fde8228386abdc89f6ec8a4fb0229fdbda6110183b4be637a50ac4a71c2a`.
Its relevant realized controls were:

- `beammap.iter_max: 1`;
- `beammap.direction_mode: standard`;
- `mapmaking.method: naive` and `mapmaking.grouping: detector`;
- fruit loops disabled; and
- processed TOD output enabled in full/all-detector mode.

The reduction completed iteration 0 in 627.105 s, then exited after 11m03.437s
with:

```text
session.unhandled_exception: failed to update required PTC TOD FRUITLOOPS_ITER in /work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_naive_full_ptc_singlepass_150819_2026-08-07/o150819/redu00/150819/raw/full_ptc/toltec_commissioning_beammap_150819_ptc_timestream.nc: required PTC TOD file has no FRUITLOOPS_ITER variable
```

The failed root contained a 4,278,775-byte PTC TOD and a 106,983,396-byte
PTC-diagnostic file. The log and partial-file state are engineering defect
evidence; the incomplete PTC TOD is not an accepted downstream data product.

## Root cause and affected lifecycle

`Engine::create_tod_files<ptc_timestream>()` created the PTC NetCDF schema at
observation setup but did not create `FRUITLOOPS_ITER`. During the Beammap run
loop, `write_beammap_processed_ptc_tod()` first called the fail-closed
`update_ptc_tod_fruitloops_iter()` helper and only then would append retained
processed chunks. The general `add_tod_header()` path that normally created
`FRUITLOOPS_ITER` ran later, after the Beammap loop. The required metadata
update therefore failed before any full-PTC scan append.

The affected path is Beammap with processed TOD output enabled, independent
of `direction_mode`, mapmaker choice, and the SCI-ALIGN timing diagnostic. The
observed `standard`/naive trigger proves that split-direction accumulation is
not required to expose it. No broader Beammap schema defect is inferred.

## Bounded repair

The selected repair creates `FRUITLOOPS_ITER` with the initial PTC file schema
and leaves the iteration-time updater fail-closed. The existing scalar NetCDF
writer already updates an existing variable idempotently, so the later general
TOD header writes the final value without creating a duplicate.

Two broader alternatives were deliberately rejected:

- allowing the iteration-time updater to create a missing variable would
  weaken the required-schema contract; and
- moving the complete final TOD header ahead of Beammap processing would
  publish metadata whose final values are not all available and would broaden
  lifecycle behavior.

This repair is not a general Beammap schema redesign.

## Local regression evidence

The focused
`ptc_tod_schema.iteration_field_exists_before_final_header_and_updates` test
reproduces the required ordering: initial schema creation, Beammap's
iteration-time update, then the final auxiliary-metadata pass. Evidence at the
final commit is:

- Citlali CLI build: pass;
- all 15 `citlali::safety` tests: pass, including the new lifecycle test; and
- complete configuration preflight: pass, including all 123 Python tests,
  four mode kits, eight compact-compatibility cases, and zero authority or
  surface-coverage gaps.

## Remaining acceptance evidence

One owner-run Unity validation remains: rebuild the final repair commit and
repeat the exact single-pass full-PTC ObsNum 150819 configuration at a fresh
output root. Acceptance requires a zero job exit, the normal `citlali is done`
terminal record, a present nonempty full PTC TOD, and a readable
`FRUITLOOPS_ITER` value matching the realized output iteration. No additional
Beammap change or timing interpretation is authorized by that validation.
