# SCI-ALIGN-001 3C273 corpus diagnostic kit

This package is the bounded, read-only owner-execution contract for the 3C273
Beammap corpus. It inventories retained reductions, runs the preregistered
left/right and raw-counter diagnostics, freezes independent validation groups,
and aggregates only compact outputs. It does not launch Citlali, contact
Unity, alter an application/configuration/reduction product, reassociate raw
rows, or authorize a physical timing correction.

## Owner entry points

- `UNITY_RUNBOOK.md` gives the exact owner-run inventory, freeze, serial,
  Slurm, resume, aggregate, checksum, and compact-transfer commands.
- `frozen_analysis_protocol.json` is the before-corpus-results scientific and
  statistical authority.
- `candidate_manifest.schema.json` and `OUTPUT_SCHEMA.md` define the portable
  interfaces.
- `OWNER_DECISION_BRIEF.md` states the producer-informed interpretation and
  stopping point.
- `RETURN_BUNDLE_PROMPT.md` is the exact prompt for the inventory-only or
  completed-bundle review task.
- `authoritative_obsnums_2026-08-03.json` and its schema are the checksum-bound
  40-ObsNum corpus authority. They are copied into inventory and selected-
  manifest packages and verified by serial and Slurm scripts before execution.

## Local evidence

- `local_148670_regression.json` records the authenticated Beammap 148670
  regression, exact T0 vector, raw-counter diagnostics, and byte-identical
  fresh replay.
- `synthetic_aggregation_test_evidence.json` records the covariance/group-aware
  A--G synthetic decision coverage and deterministic aggregate replay.
- `validation_gates.json` records the repository and focused gates.
- `changed_paths.tsv` proves the bounded changed-path surface relative to the
  frozen predecessor commit.

## Interpretation boundary

NTP supplies an integer-second T0 at ROACH initialization; the ROACH clocks
and PPS share the Octo-distributed reference, while PPS does not restart the
detector cadence. Arbitrary millisecond NTP error and differential oscillator
drift are therefore strongly disfavored. Distinct stable network integration
phase, detector-frame quantization of PPS observation, adjacent/non-atomic
metadata association, and start/end/centroid timestamp semantics remain
separate hypotheses. The prior Stage-A proof begins at delivered `D[n]/Ts[n]`
and cannot rule out an upstream FPGA metadata-to-integration association error.

The stopping point is a checksum-bound diagnostic kit and owner-returned
compact evidence bundle. No 3C273 result alone authorizes a production change.

The owner run directory is deliberately below the 3C273 analysis directory but
is an explicit inventory exclusion. It may contain generated serial/Slurm
scripts, execution identity, compact outputs, and archive metadata; it is not
a retained reduction and is never a candidate-discovery root.
