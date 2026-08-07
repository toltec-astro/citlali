# Validation Ledger

This directory contains executable contracts and evidence. The documentation
layer and its separation between user guidance, reusable scientific methods,
and validation are described in [`doc/README.md`](../doc/README.md).

`accepted_runs.json` is the machine-readable index of accepted Citlali
refactor validation checkpoints. Large reduction products remain outside Git;
each record stores enough identity, policy, and location information to find
and interpret them.

`validation_profiles.json` names the currently active validation epoch and the
accepted point, OOF, science, and Beammap snapshots for that epoch. A profile
pins the provenance requirements, config policy, product comparator, and
scientific tolerance file. Validate both files with:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_validation_ledger.py
$HOME/tolteca/bin/python tools/baseline/validation_profiles.py --list
```

## Validation Epochs

An accepted snapshot is evidence about a named version of the pipeline, not a
claim that Citlali's products can never evolve. Post-refactor development may
intentionally make non-incremental changes to algorithms, defaults, schemas,
or final products.

When an intentional change reaches validation:

1. Preserve the existing ledger record, epoch, profile, and tolerance file.
2. Compare the new result against the predecessor snapshot and retain the
   result.
3. Add a new accepted ledger record that states the affected products,
   configuration, scientific meaning, and rationale.
4. Create a successor epoch and profile. Obtain scientific-owner approval when
   algorithms, defaults, or scientific products change.
5. Make the successor epoch active without deleting the historical one.

Do not weaken an existing profile or replace its baseline merely to make a
changed run pass. A bug fix or intentional scientific change can be accepted;
it must remain distinguishable from behavior-preserving refactor work.

## One Validation Command

Run all three required gates for a downloaded candidate with its named
profile:

```bash
$HOME/tolteca/bin/python tools/baseline/validate_reduction.py \
  /path/to/candidate/reduNN \
  --profile phase4-point-152389-v1 \
  --output-dir /tmp/citlali-point-validation \
  --report-out /tmp/citlali-point-validation.md
```

The command audits completion and required provenance, requires an exact
low-level config match, and runs the profile's strict or scientific product
comparator. It uses the accepted record's local artifact path as the baseline.
Pass `--baseline /path/to/accepted/reduNN` on a host where that path differs.
Any failed gate rejects the candidate.

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
- Link current records to their validation epoch and profile. Historical
  records may predate those additive fields.

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

## Human-run audit campaigns

`campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb/` is the frozen owner package
for exact repair candidate `ed28dafb37f9113c0d3c95297148157129a90886`.
It contains the seven-case manifest, explicit deployment-value and result
schemas, raw-source/scan/detector/sample authority schema, deterministic
expert-overlay generator, input/hash preflight, allocation wrappers, and
request-specific analyzer. The human owner has already executed the seven
cases; do not rerun the owner runbook. The returned products remain in the
owner-supplied external corpus and are reconciled in
`SCI-MAP-001_EXISTING_CORPUS_CLOSEOUT_2026-08-05.md`. That closeout is an
evidence handoff, not an accepted validation record or finding disposition.
Missing independent processed-term ledgers and same-case `S-X-SEQ`
observation-realization bytes remain explicit evidence gaps, not values to
infer from final FITS or sibling products.

The later project-owner amendment
[`SCI-MAP-001_OWNER_SCOPE_EVIDENCE_AMENDMENT_2026-08-05.md`](../handoff/SCI-MAP-001_OWNER_SCOPE_EVIDENCE_AMENDMENT_2026-08-05.md)
accepts F012 only for the bounded external product/execution/SEQ-OMP claims and
retains those missing lanes as limitations. It authorizes local repair and
production-path tests for the `S-X-SEQ` observation-realization persistence
defect; it does not authorize or require a repeat Unity reduction.
