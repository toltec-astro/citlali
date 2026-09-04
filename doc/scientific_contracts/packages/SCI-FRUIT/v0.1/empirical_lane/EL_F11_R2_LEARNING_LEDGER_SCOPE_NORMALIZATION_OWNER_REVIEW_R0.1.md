# SCI-FRUIT EL-F11-R2 — learning-ledger scope normalization owner review r0.1

Decision identity:
`SCI-FRUIT-EL-F11-R2-LEARNING-LEDGER-SCOPE-NORMALIZATION-R0.1`

Status: **owner decision required; prospective accounting remains unopened**

## Plain-language situation

The approved EL-F11 replay ran successfully. Its ordinary maps match the
historical iteration-4 maps bit for bit. The analysis nevertheless stopped
before reading the new JINC accounting values because we had required the two
learning CSV files to be identical as complete files.

They are not complete-file peers. The historical file was written during an
uninterrupted run and contains the learning records from iterations 0 through
4. The checkpoint restart performed only iteration 4, so its file contains
only iteration 4. When the historical file is restricted to iteration 4, its
439 rows match the restart file's 439 rows exactly, in the same order and in
every text field.

This is strong evidence that the iteration-4 learning decisions reproduced.
It is not a pass under the rule we registered, because that rule said to
compare the complete files byte for byte.

## Proposed no-replay repair

Do not rerun Citlali. Preserve the failed r0.4 analysis and every exact output
already bound by `REGISTRATION_R0.3.yaml`. Authorize a narrowly revised
learning-ledger compatibility gate that must:

1. require the reference and replay CSV headers to be identical, including
   field names and order;
2. require every replay row's literal `iter` field to identify the registered
   completed absolute iteration, `4`;
3. select from the cumulative historical file only rows whose literal `iter`
   field identifies iteration `4`;
4. require the selected historical rows and all replay rows to have equal
   counts and to match in order and in every raw CSV string field, without
   numeric parsing, tolerance, sorting, or normalization; and
5. report the complete historical and replay iteration counts, while treating
   only the absence of historical iterations 0--3 from the restart-produced
   file as an allowed retention-scope difference.

The retained historical file remains protected by its registered full-file
size and SHA-256 identity. No earlier row is changed, regenerated, or inferred,
and the replay is not required to duplicate prior-iteration diagnostics that
already exist in the preserved checkpoint lineage.

After implementing and testing exactly this comparison, a successor analysis
registration must bind the repaired analyzer and the same retained replay
outputs before the JINC receipt is opened. The analysis may then continue once
through the unchanged gate sequence. If this exact row comparison or any later
gate fails, it stops again.

## Why this is the right scientific comparison

The purpose of this compatibility gate is to determine whether restarting at
iteration 4 reproduced iteration 4's learned decisions. Exact ordered equality
of every iteration-4 record tests that purpose directly. Requiring a one-
iteration restart to re-emit diagnostic history from iterations it did not
execute tests file-retention behavior instead.

The proposed rule is intentionally stricter than comparing summaries: row
counts, row order, and every serialized field must match exactly. It does not
declare the complete files equivalent or conceal the difference in their
iteration coverage.

## What does not change

R2 changes only the learning-ledger compatibility comparison. It does not
change or authorize:

- any input, checkpoint, executable, configuration, replay, algorithm,
  recurrence, target, penalty, threshold, support rule, region, numerical
  bound, persistence metric, or claim limit;
- another Citlali replay or replacement of any retained file;
- a detector judgment, candidate selector, intervention, or safeguard;
- production use, FRUIT or JINC qualification, Gate D, Stage B, or Unity
  activity.

All twelve ordinary map-plane gates, the map-diagnostic gate, checkpoint gate,
receipt and ledger gates, exact closure and forward-error rules, persistence
population, descriptive metrics, and interpretation limits remain exactly as
registered for EL-F11.

## Owner choices

### Choice A — Approve exact iteration-4 row comparison (recommended)

Approve
`SCI-FRUIT-EL-F11-R2-LEARNING-LEDGER-SCOPE-NORMALIZATION-R0.1` against the
exact `EL_F11_R2_BUNDLE_MANIFEST_R0.1.md`. This authorizes the exact no-replay
analyzer repair, focused tests, a new output-bound analysis registration, and
one continuation of analysis against the retained EL-F11 products.

### Choice B — Keep the whole-file failure

Retain EL-F11 as a compatibility failure, leave the prospective accounting
values unopened, and end this test.

### Choice C — Request a different compatibility rule

Revise the ledger comparison or analysis scope before any prospective
accounting value is opened.

The exact affirmative statement for Choice A is:

> I approve `SCI-FRUIT-EL-F11-R2-LEARNING-LEDGER-SCOPE-NORMALIZATION-R0.1` against the exact `EL_F11_R2_BUNDLE_MANIFEST_R0.1.md`.
