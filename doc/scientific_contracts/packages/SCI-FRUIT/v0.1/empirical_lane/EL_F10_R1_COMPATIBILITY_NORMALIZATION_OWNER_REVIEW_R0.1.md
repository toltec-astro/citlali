# SCI-FRUIT EL-F10-R1 — compatibility-normalization owner review r0.1

Decision identity:
`SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1`

Status: **owner decision required; no accounting analysis is authorized**

## Plain-language situation

The one approved EL-F10 replay ran successfully and produced the intended
diagnostic files. Its ordinary maps match the historical EL-F6 N5 maps bit
for bit. We nevertheless stopped before reading the new accounting values
because one checkpoint text field failed the exact rule we had registered.

The old checkpoint omits a setting that did not exist when it was written.
The new checkpoint spells out that setting's historical default:

```yaml
map_pixel_outlier_detector_exclusion_application: pre_cleaning
```

Nothing else in the learning policy differs. The checkpoint has the same
structure, and its only other changed value is the already allowed executable
version. This is the same old-versus-explicit-default issue that EL-F8 found
and repaired before its successful runs.

## Proposed repair

Do not rerun Citlali. Preserve the failed r0.1 result and the exact retained
files. Before opening either accounting file, create a new analysis
registration that:

1. binds the existing receipt, target ledger, three science maps, checkpoint,
   and replay log by their recorded sizes and SHA-256 identities;
2. permits `learning_policy_yaml` to differ only when parsing both policies,
   inserting `pre_cleaning` for this one missing key, and comparing every
   normalized key and value exactly;
3. still requires `creator_version` to be the only other checkpoint value
   difference and requires checkpoint structure to remain identical; and
4. leaves every ordinary-map bitwise gate, total-accumulator exact-closure
   gate, 305/34/271 sample ledger, binary64 error formula and safety factor,
   support rule, region, trigger pixel, descriptive summary, and claim limit
   unchanged from EL-F10 r0.1.

The frozen analyzer already implements the proposed narrow policy
normalization: a changed `learning_policy_yaml` is accepted only if the two
parsed policies become exactly equal after absence of this single key is
normalized to `pre_cleaning`. The current production restart reader uses the
same bounded rule. The R1 registration will additionally require the observed
allowed-difference set to be exactly `creator_version` plus
`learning_policy_yaml` before opening accounting values.

## Why this is scientifically defensible

The proposed change is not inferred from the UID accounting output; those
values have not been opened. It is based on evidence available independently
of them:

- EL-F8 documented this exact historical missing-field case before its
  successful compatibility runs;
- the current enum default is `pre_cleaning`;
- the current restart reader treats historical absence as exactly
  `pre_cleaning` and rejects a different explicit value; and
- EL-F10's twelve registered ordinary map planes already reproduce EL-F6 N5
  bitwise.

The important procedural defect was that EL-F10 r0.1 did not name this known
serialization normalization in advance. R1 does not erase that defect or call
r0.1 a pass. It asks permission for a new, explicit, no-replay analysis rule
against frozen files.

## What stays blocked

R1 does not authorize another replay, a change to Citlali, a change to the
diagnostic files, a penalty or threshold change, a detector judgment, a
safeguard choice, a recurrence change, production use, JINC or FRUIT
qualification, Gate D, Stage B, or Unity activity. If the normalized policy
comparison or any later unchanged gate fails, analysis stops again.

## Owner choices

### Choice A — Approve the no-replay compatibility repair (recommended)

Approve
`SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1` against the exact
`EL_F10_R1_BUNDLE_MANIFEST_R0.1.md`. This authorizes an owner-authorization
record, the exact output-bound R1 registration described here, and one run of
the already frozen analysis against the retained EL-F10 products. It
authorizes no Citlali replay.

### Choice B — Keep the EL-F10 r0.1 stop

Leave the accounting values unopened and retain the existing compatibility
failure as the end of this test.

### Choice C — Request a different repair

Revise the allowed compatibility rule or the analysis scope before opening
the accounting values.

The exact affirmative statement for Choice A is:

> I approve `SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1` against the exact `EL_F10_R1_BUNDLE_MANIFEST_R0.1.md`.
