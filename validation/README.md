# Validation Ledger

`accepted_runs.json` is the machine-readable index of accepted Citlali
refactor validation checkpoints. Large reduction products remain outside Git;
each record stores enough identity, policy, and location information to find
and interpret them.

## Rules

- Add a record only after checking run identity, completion, serious log
  records, product inventory, and the mode-appropriate numeric comparison.
- Use a full Git SHA where it is available. Historical records may retain the
  short SHA embedded by the binary.
- Record unavailable evidence as `null` plus a limitation. Do not infer hashes,
  commands, dependency versions, or resource measurements that were not
  retained by the run.
- List every volatile exclusion and every accepted difference explicitly.
- A strict comparator failure is not an accepted structural-equivalence result
  unless every changed item is an approved intended change recorded in
  `accepted_differences`.
- Keep local and validation-host paths when known, but do not commit large
  scientific products.
- Preserve old entries. Correct factual mistakes with a normal reviewed commit.

## Required Record Areas

Each record contains:

- candidate and baseline source/dependency identity;
- build and runtime policy;
- ordered config inputs and canonical/effective/realized hashes where retained;
- dataset and invocation identity;
- product and log audit results;
- comparator policy, tolerances, exclusions, and verdict;
- timing and resource evidence;
- artifact locations, accepted differences, limitations, and disposition.

The schema is versioned by the top-level `schema_version`. Additive fields may
be introduced within version 1. A semantic change to required meanings needs a
new schema version and migration note.

Validate the ledger before committing an entry:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_validation_ledger.py
```

## Mode-Gate Overlays

The files under `validation/configs` are high-precedence TolTECA overlays for
specific acceptance gates. Copy the same overlay into the corresponding OG and
refactor workdirs; TolTECA merges it after the lower-numbered `NN*.yaml` files.

- `science_post_processing_84_reduce.yaml` keeps science mapmaking,
  coaddition, and Wiener filtering active while disabling fruit loops, noise
  realizations, and source finding. It isolates coadd filtering and should
  produce one reduction directory rather than a retained fruit-loop series.
- `beammap_post_processing_80_reduce.yaml` fixes the Beammap gate at three
  fitting iterations while disabling unrelated map filtering, source finding,
  and noise realizations. Beammap source fitting remains required by reduction
  mode.

Before accepting a pair, confirm that the merged low-level configs differ only
in expected executable/output paths and run the mode-appropriate product and
provenance audits.
